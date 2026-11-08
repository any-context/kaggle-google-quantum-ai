#!/usr/bin/env python3
"""Redundancy / affine-Toffoli scan of a dfthcsos ANGLE_k family on its reachable domain.

The verifier applies circuit k with the word registers holding batch k-1 (forward pass) and batch k
(reverse pass), for every address.  Those 2 x num_vectors states are the whole reachable domain
(hosts' ruling: ANGLE_k only has to be correct on them).  We simulate the value part of each circuit
bit-parallel over that domain and test every unconditional Toffoli-like gate exactly as
inv_scan.py does for invsqrt.  Conditional blocks (measurement branches) are never rewritten.

Usage:
    tools/angle_scan.py DFTHCSOS_DIR [--seeds 3] [--apply OUT_DIR]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inv_scan import Sim, gf2_solve_affine, parse  # noqa: E402


def find_repo() -> Path:
    import subprocess
    here = Path(__file__).resolve().parents[1]
    for c in (here, (here / subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=here, capture_output=True,
                                           text=True).stdout.strip()).resolve().parent):
        if (c / "data" / "verifiers" / "dfthcsos").exists():
            return c
    raise SystemExit("no data/verifiers/dfthcsos")


def set_bits(sim: Sim, qubits: list[str], values: np.ndarray):
    for j, qn in enumerate(qubits):
        bits = ((values >> j) & 1).astype(np.uint8)
        bits = np.pad(bits, (0, sim.words * 64 - len(bits)))
        sim.q[qn] = np.packbits(bits, bitorder="little").view(np.uint64).copy()


def read_int(sim: Sim, qubits: list[str], n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int64)
    for j, qn in enumerate(qubits):
        out |= np.unpackbits(sim.qv(qn).view(np.uint8), bitorder="little")[:n].astype(np.int64) << j
    return out


def scan_circuit(lines, sel_q, word_q, addresses, words_in, words_out, seed):
    n = len(addresses)
    sim = Sim(n, seed)
    set_bits(sim, sel_q, addresses.astype(np.uint64))
    for j, wq in enumerate(word_q):
        set_bits(sim, wq, words_in[:, j].astype(np.uint64))
    findings = {}
    for i, line in enumerate(lines):
        op, args, cond = parse(line)
        if op is None:
            continue
        if op in ("CCX", "CCZ", "CSWAP"):
            uncond = cond is None and not sim.stack
            if op == "CCX":
                a, b = sim.qv(args[0]), sim.qv(args[1])
                p = a & b
                if not p.any():
                    findings[i] = ("delete", "a&b==0")
                elif np.array_equal(p, a):
                    findings[i] = ("CX", args[0])
                elif np.array_equal(p, b):
                    findings[i] = ("CX", args[1])
                elif uncond:
                    live = [(nm, v) for nm, v in sim.q.items() if nm != args[2] and v.any()]
                    res = gf2_solve_affine([v for _, v in live], p, [nm for nm, _ in live], sim.all, sample_words=min(64, sim.words))
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
    # contract check
    ok_sel = np.array_equal(read_int(sim, sel_q, n), addresses)
    got = np.stack([read_int(sim, wq, n) for wq in word_q], axis=1)
    ok_words = np.array_equal(got, words_out)
    public = set(sel_q) | {q for wq in word_q for q in wq}
    dirty = [nm for nm, v in sim.q.items() if nm not in public and v.any()]
    return findings, ok_sel, ok_words, dirty


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dfthcsos", type=Path)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--apply", type=Path, default=None)
    a = ap.parse_args()
    repo = find_repo()
    sys.path.insert(0, str(repo / "data" / "verifiers"))
    import kickmix as km
    from dfthcsos.scoreinfra.circuits import read_circuit_family, recover_words_from_circuits, register_layout

    circuits = read_circuit_family(a.dfthcsos)
    layout = register_layout(circuits)
    sel_q = [f"q{q.id}" for q in layout.selection]
    word_q = [[f"q{q.id}" for q in w] for w in layout.words]
    npz = np.load(a.dfthcsos / "submission.npz")
    R, B, C = npz["W"].shape
    num_vectors = 76 + R * B
    J = len(word_q)
    print(f"{len(circuits)} circuits, selection {len(sel_q)} qubits, J={J}, brot={layout.brot}, vectors {num_vectors}")
    words, _, toff = recover_words_from_circuits(circuits, num_vectors=num_vectors, num_angles=75, shots_per_address=1, seed=0)
    print(f"official traversal OK: words {words.shape}, mean Toffoli (fwd+rev) {toff:.1f}")
    K = len(circuits)
    padded = np.zeros((num_vectors, K * J), dtype=np.int64)
    padded[:, :75] = words
    batches = [padded[:, k * J:(k + 1) * J] for k in range(K)]
    zero = np.zeros_like(batches[0])
    addresses = np.concatenate([np.arange(num_vectors)] * 2)
    total_static = 0
    total_found = 0
    rewritten = {}
    for k, circ in enumerate(circuits):
        import io
        buf = io.StringIO()
        circ.to_file(buf, format="kmx")
        lines = buf.getvalue().splitlines()
        win = np.concatenate([batches[k - 1] if k else zero, batches[k]])           # forward: batch k-1 ; reverse: batch k
        wout = np.concatenate([batches[k], batches[k - 1] if k else zero])
        common = None
        for s in range(a.seeds):
            f, ok_sel, ok_words, dirty = scan_circuit(lines, sel_q, word_q, addresses, win, wout, 500 + s)
            if not (ok_sel and ok_words and not dirty):
                print(f"  ANGLE_{k} seed {s}: MODEL MISMATCH sel={ok_sel} words={ok_words} dirty={dirty[:4]}")
            common = f if common is None else {i: v for i, v in common.items() if i in f and f[i] == v}
        ccx = sum(1 for l in lines if l.split()[:1] in (["CCX"], ["CCZ"], ["CSWAP"]))
        total_static += ccx
        total_found += len(common)
        print(f"  ANGLE_{k}: {ccx} Toffoli-like, {len(common)} rewritable: {[(i, lines[i], v[0]) for i, v in sorted(common.items())][:6]}")
        rewritten[k] = (lines, common)
    print(f"total Toffoli-like {total_static}, rewritable {total_found}")
    if a.apply and total_found:
        a.apply.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.dfthcsos / "submission.npz", a.apply / "submission.npz")
        for k, (lines, common) in rewritten.items():
            out = []
            for i, line in enumerate(lines):
                if i in common:
                    kind, info = common[i]
                    op, args, cond = parse(line)
                    suffix = f" if {cond}" if cond else ""
                    if kind == "delete":
                        continue
                    if kind == "CX":
                        out.append(f"CX {info} {args[2]}{suffix}"); continue
                    if kind == "CZ":
                        out.append(f"CZ {info[0]} {info[1]}{suffix}"); continue
                    if kind == "SWAP":
                        out.append(f"SWAP {args[1]} {args[2]}{suffix}"); continue
                    if kind == "affine":
                        sel, const = info
                        out += [f"CX {nm} {args[2]}{suffix}" for nm in sel]
                        if const:
                            out.append(f"X {args[2]}{suffix}")
                        continue
                out.append(line)
            km.Circuit("\n".join(out) + "\n").to_file(a.apply / f"angle_k_{k}.kmb", format="kmb")
        print(f"wrote {a.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
