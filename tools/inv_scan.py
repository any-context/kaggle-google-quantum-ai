#!/usr/bin/env python3
"""Full-input, bit-parallel value simulation of an invsqrt kickmix circuit, with peephole scans.

Implements the techniques of the technical guide (ch. 3): every Toffoli-like gate (CCX, CCZ, CSWAP)
is checked, over ALL 3,133,452 inputs and several measurement-outcome assignments, for

  * CCX a b t : a&b == 0           -> delete            (never fires)
                a&b == a / == b    -> CX a t / CX b t   (one control implies the other)
                a&b == XOR(subset of live wires) ^ const  -> CX chain (affine product)
  * CCZ a b c : a&b&c == 0         -> delete
                a&b&c affine       -> CZ / Z chain
  * CSWAP c a b : c&(a^b) == 0     -> delete            (domain-redundant swap)
                  c == 1 wherever a!=b -> SWAP (free)

Only the value part is simulated (phase gates are permutation-neutral), one uint64 word = 64
inputs, so the whole domain is 48,961 words per wire.  Measurement outcomes (HMR) are random per
seed; a candidate must hold for every seed tried.  The official verifier remains the ground truth:
apply the rewrite and run it.

Usage:
    tools/inv_scan.py CIRCUIT.kmx [--seeds 4] [--apply OUT.kmx] [--max-x 3133452]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

MAX_X = 3 * (2 ** 10 - 2) ** 2  # 3,133,452
WIDTH = 22
PHASE_OPS = {"Z", "CZ", "Z_POW", "NEG", "INFO_TAG", "REGISTER", "APPEND_TO_REGISTER"}


def parse(line: str):
    parts = line.split()
    if not parts:
        return None, [], None
    cond = None
    if len(parts) >= 3 and parts[-2] == "if":
        cond = parts[-1]
        parts = parts[:-2]
    return parts[0], parts[1:], cond


class Sim:
    def __init__(self, n_inputs: int, seed: int):
        self.n = n_inputs
        self.words = (n_inputs + 63) // 64
        self.rng = np.random.default_rng(seed)
        self.q: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}
        self.stack: list[np.ndarray] = []
        ones = np.full(self.words, ~np.uint64(0), dtype=np.uint64)
        if n_inputs % 64:
            ones[-1] = np.uint64((1 << (n_inputs % 64)) - 1)
        self.all = ones
        self.mask = ones

    def qv(self, name):
        a = self.q.get(name)
        if a is None:
            a = np.zeros(self.words, dtype=np.uint64)
            self.q[name] = a
        return a

    def bv(self, name):
        a = self.b.get(name)
        if a is None:
            a = np.zeros(self.words, dtype=np.uint64)
            self.b[name] = a
        return a

    def set_x(self, xs: np.ndarray, qubits: list[str]):
        for j, qn in enumerate(qubits):
            bits = ((xs >> np.uint64(j)) & np.uint64(1)).astype(np.uint8)
            bits = np.pad(bits, (0, self.words * 64 - len(bits)))
            self.q[qn] = np.packbits(bits, bitorder="little").view(np.uint64).copy()

    def step(self, op, args, cond):
        M = self.mask if cond is None else (self.mask & self.bv(cond))
        if op in PHASE_OPS or op == "CCZ":
            return
        if op == "X":
            self.qv(args[0])[:] ^= M
        elif op == "CX":
            self.qv(args[1])[:] ^= self.qv(args[0]) & M
        elif op == "CCX":
            self.qv(args[2])[:] ^= self.qv(args[0]) & self.qv(args[1]) & M
        elif op == "SWAP":
            a, b = self.qv(args[0]), self.qv(args[1])
            d = (a ^ b) & M
            a ^= d
            b ^= d
        elif op == "CSWAP":
            c, a, b = self.qv(args[0]), self.qv(args[1]), self.qv(args[2])
            d = (a ^ b) & c & M
            a ^= d
            b ^= d
        elif op == "HMR":
            r = self.rng.integers(0, 2 ** 64, size=self.words, dtype=np.uint64, endpoint=False)
            bb = self.bv(args[1])
            bb[:] = (bb & ~M) | (r & M)
            self.qv(args[0])[:] &= ~M
        elif op == "R":
            self.qv(args[0])[:] &= ~M
        elif op == "BIT_STORE0":
            self.bv(args[0])[:] &= ~M
        elif op == "BIT_STORE1":
            self.bv(args[0])[:] |= M
        elif op == "BIT_INVERT":
            self.bv(args[0])[:] ^= M
        elif op == "PUSH_CONDITION":
            self.stack.append(self.mask)
            self.mask = M
        elif op == "POP_CONDITION":
            self.mask = self.stack.pop()
        else:
            raise ValueError(f"unknown op {op}")


def popcount(a: np.ndarray) -> int:
    return int(np.unpackbits(a.view(np.uint8)).sum())


def gf2_solve_affine(cols: list[np.ndarray], target: np.ndarray, names: list[str], allmask: np.ndarray, sample_words: int = 64):
    """Find a subset S of cols and c in {0,1} with XOR(S) ^ c == target on all words.

    Solve on the first `sample_words` words by Gaussian elimination over GF(2) (bits as rows), then
    confirm on everything.  Returns (names_in_S, const) or None.
    """
    k = len(cols)
    if k == 0:
        return None
    # rows = shots (sample), columns = wires + const ; build as uint8 matrix (sample_words*64 x k+1)
    rows = sample_words * 64
    # sample words spread over the whole domain (the low inputs all share large z, so a prefix
    # sample would make the threshold columns collinear with the constant)
    # include the low words too: inputs with large z live only in the first few hundred words
    # log-spaced words so that every leading-zero count z is represented in the sample
    idx = np.unique(np.concatenate([np.arange(min(160, len(target))),
                                    np.geomspace(1, len(target) - 1, 4 * sample_words).astype(int),
                                    np.linspace(0, len(target) - 1, sample_words).astype(int)]))
    rows = len(idx) * 64
    A = np.zeros((rows, k + 2), dtype=np.uint8)
    for j, c in enumerate(cols):
        A[:, j] = np.unpackbits(c[idx].view(np.uint8), bitorder="little")
    A[:, k] = 1
    A[:, k + 1] = np.unpackbits(target[idx].view(np.uint8), bitorder="little")
    # eliminate
    piv_cols = []
    r = 0
    for cidx in range(k + 1):
        pr = np.flatnonzero(A[r:, cidx])
        if len(pr) == 0:
            continue
        p = r + pr[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        mask = A[:, cidx].astype(bool)
        mask[r] = False
        A[mask] ^= A[r]
        piv_cols.append(cidx)
        r += 1
        if r == rows:
            break
    # inconsistency: a zero row with rhs 1
    if np.any((A[:, :k + 1].sum(axis=1) == 0) & (A[:, k + 1] == 1)):
        return None
    sol = np.zeros(k + 1, dtype=np.uint8)
    for i, cidx in enumerate(piv_cols):
        sol[cidx] = A[i, k + 1]
    # confirm on full domain
    acc = np.zeros_like(target)
    sel = []
    for j in range(k):
        if sol[j]:
            acc ^= cols[j]
            sel.append(names[j])
    if sol[k]:
        acc = ~acc
    if np.array_equal(acc & allmask, target):
        return sel, int(sol[k])
    return None


def scan(lines: list[str], n_inputs: int, seed: int, x_qubits: list[str], log):
    sim = Sim(n_inputs, seed)
    xs = np.arange(1, n_inputs + 1, dtype=np.uint64)
    sim.set_x(xs, x_qubits)
    findings = {}
    t0 = time.time()
    for i, line in enumerate(lines):
        op, args, cond = parse(line)
        if op is None:
            continue
        if op in ("CCX", "CCZ", "CSWAP"):
            # Conservative: ignore the (outcome-dependent) condition mask and require the property on
            # every shot.  A property that holds only on the shots where the block fires cannot be
            # trusted, because the firing set is a random, possibly rare, measurement branch.
            unconditional = cond is None and not sim.stack
            if op == "CCX":
                a, b = sim.qv(args[0]), sim.qv(args[1])
                p = a & b
                if not p.any():
                    findings[i] = ("delete", "a&b==0")
                elif np.array_equal(p, a):
                    findings[i] = ("CX", args[0])
                elif np.array_equal(p, b):
                    findings[i] = ("CX", args[1])
                elif unconditional:
                    live = [(n, v) for n, v in sim.q.items() if n != args[2] and v.any()]
                    res = gf2_solve_affine([v for _, v in live], p, [n for n, _ in live], sim.all)
                    if res is not None:
                        findings[i] = ("affine", res)
            elif op == "CCZ":
                p = sim.qv(args[0]) & sim.qv(args[1]) & sim.qv(args[2])
                if not p.any():
                    findings[i] = ("delete", "abc==0")
                else:
                    for k in range(3):
                        others = [args[j] for j in range(3) if j != k]
                        if np.array_equal(p, sim.qv(others[0]) & sim.qv(others[1])):
                            findings[i] = ("CZ", others)
                            break
            elif op == "CSWAP":
                c, a, b = sim.qv(args[0]), sim.qv(args[1]), sim.qv(args[2])
                d = a ^ b
                if not (d & c).any():
                    findings[i] = ("delete", "swap never effective")
                elif np.array_equal(d & c, d):
                    findings[i] = ("SWAP", None)
        sim.step(op, args, cond)
        if i % 10000 == 0:
            log(f"    seed {seed}: line {i}/{len(lines)}  {time.time() - t0:.1f}s")
    # final checks: x restored, others zero
    exp = Sim(n_inputs, 0)
    exp.set_x(xs, x_qubits)
    bad = sum(1 for qn in x_qubits if not np.array_equal(sim.q[qn], exp.q[qn]))
    dirty = [n for n, v in sim.q.items() if n not in x_qubits and v.any()]
    log(f"  seed {seed}: x restored={'yes' if bad == 0 else f'NO ({bad} wires)'}  dirty scratch={dirty[:5]}  "
        f"{time.time() - t0:.1f}s  findings={len(findings)}")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuit", type=Path)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-x", type=int, default=MAX_X)
    ap.add_argument("--apply", type=Path, default=None, help="write the rewritten kmx here")
    a = ap.parse_args()
    lines = a.circuit.read_text().splitlines()
    x_qubits = [parse(l)[1][0] for l in lines if l.startswith("APPEND_TO_REGISTER")]
    assert len(x_qubits) == WIDTH, x_qubits
    log = lambda s: print(s, flush=True)
    log(f"{a.circuit}: {len(lines)} lines, x register {x_qubits[0]}..{x_qubits[-1]}, inputs 1..{a.max_x}")
    common = None
    for s in range(a.seeds):
        f = scan(lines, a.max_x, 1000 + s, x_qubits, log)
        if common is None:
            common = f
        else:
            common = {i: v for i, v in common.items() if i in f and f[i] == v}
    log(f"\n{len(common)} findings stable over {a.seeds} seeds:")
    saved = 0
    for i, (kind, info) in sorted(common.items()):
        log(f"  line {i}: {lines[i]:40s} -> {kind} {info}")
        if kind in ("delete", "CX", "CZ", "SWAP", "affine"):
            saved += 1
    log(f"Toffoli-like gates removable (static count): {saved}")
    if a.apply and saved:
        out = []
        for i, line in enumerate(lines):
            if i in common:
                kind, info = common[i]
                op, args, cond = parse(line)
                suffix = f" if {cond}" if cond else ""
                if kind == "delete" or kind == "never-executed" and False:
                    continue
                if kind == "CX":
                    out.append(f"CX {info} {args[2]}{suffix}")
                    continue
                if kind == "CZ":
                    out.append(f"CZ {info[0]} {info[1]}{suffix}")
                    continue
                if kind == "SWAP":
                    out.append(f"SWAP {args[1]} {args[2]}{suffix}")
                    continue
                if kind == "affine":
                    sel, const = info
                    for n in sel:
                        out.append(f"CX {n} {args[2]}{suffix}")
                    if const:
                        out.append(f"X {args[2]}{suffix}")
                    continue
            out.append(line)
        a.apply.write_text("\n".join(out) + "\n")
        log(f"wrote {a.apply} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
