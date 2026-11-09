#!/usr/bin/env python3
"""Replace the invsqrt normalize shifter (lines 130..293 of 1c5b1ee2.kmx) by a CSWAP network.

Contract at the block entry (verified with inv_scan's full-domain simulator):
    q0..q21  : x (bit j on qj), z = number of leading zeros (x != 0, so z <= 21)
    q22      : [z>=1]            q23+k : [z>=k+2]  (k = 0..18, i.e. [z>=2]..[z>=20])
    q47      : [z>=21]           q42..q45 : binary floor(z/2)   q46 : z odd
    q48      : scratch |0>
Block exit: q0..q21 hold x << z (leading one on q21), everything else unchanged, q48 = 0.
(The original block also erased its own scratch q49..q52; we never create them.)

A gate list is a JSON {"n": 22, "free_leading": bool, "gates": [{"pair": [a,b], "active_z": [...]}]}.
Controls are realised for free: q48 ^= XOR of thresholds [z>=k] over the set K such that
[z in active] = XOR_{k in K} [z>=k]  (K = {k : f(k) != f(k-1)}, f(0) = 0).  Thresholds: k=1 -> q22,
k=2..20 -> q(21+k), k=21 -> q47.  With free_leading the leading one is cleared before the network
(CX [z==k] into q(21-k)) and re-created by X q21 afterwards.

Usage:
    tools/shifter_splice.py --gates build/shifter/fl/n22_N37.json --out build/inv/new.kmx [--verify]
    tools/shifter_splice.py --barrel --out build/inv/barrel.kmx --verify     # 79-gate reference network
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inv_scan import MAX_X, Sim, parse  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
SRC = HERE / "build" / "inv" / "1c5b1ee2.kmx"
A, B = 129, 292          # 0-based inclusive range replaced ('CX q29 q49' .. 'CSWAP q37 q17 q19'); 293 = HMR q22 kept
N = 22
SCRATCH = "q48"


def threshold_wire(k: int) -> str:
    if k == 1:
        return "q22"
    if 2 <= k <= 20:
        return f"q{21 + k}"
    if k == 21:
        return "q47"
    raise ValueError(k)


def active_to_K(active: list[int]) -> list[int]:
    f = [1 if z in set(active) else 0 for z in range(N)]
    assert f[0] == 0, "gates must be inactive for z=0"
    return [k for k in range(1, N) if f[k] != f[k - 1]]


def barrel_gates():
    """Descending binary barrel shifter with zero-knowledge pruning (79 gates), for pipeline tests."""
    gates = []
    for m in (16, 8, 4, 2, 1):
        active = [z for z in range(1, N) if (z // m) % 2 == 1]
        # data occupies 0..21-(z mod 2m) before this stage... use the generic bound 0..21-m
        for i in range(N - m - 1, -1, -1):
            gates.append({"pair": [i, i + m], "active_z": active})
    return gates


def build_block(gates, free_leading: bool) -> list[str]:
    out = []
    if free_leading:
        # clear the leading one: for z=k the leading one sits on q(21-k); toggle it iff z==k
        # z=0: bit 21 is 1 -> X q21 ; CX q22 q21  (q22=[z>=1]) toggles back for z>=1
        out += ["X q21", "CX q22 q21"]
        for k in range(1, N):
            tgt = f"q{21 - k}"
            out.append(f"CX {threshold_wire(k)} {tgt}")
            if k + 1 <= 21:
                out.append(f"CX {threshold_wire(k + 1)} {tgt}")
    cur: set[int] = set()
    for gt in gates:
        K = set(active_to_K(gt["active_z"]))
        for k in sorted(cur ^ K):
            out.append(f"CX {threshold_wire(k)} {SCRATCH}")
        cur = K
        a, b = gt["pair"]
        out.append(f"CSWAP {SCRATCH} q{a} q{b}")
    for k in sorted(cur):
        out.append(f"CX {threshold_wire(k)} {SCRATCH}")
    if free_leading:
        out.append("X q21")
    return out


def simulate_compare(lines_old, lines_new, seed=1000):
    """Run both circuits up to the end of the block and compare every wire on the full domain."""
    xq = [parse(l)[1][0] for l in lines_old if l.startswith("APPEND_TO_REGISTER")]
    xs = np.arange(1, MAX_X + 1, dtype=np.uint64)
    states = []
    for lines, end in ((lines_old, B + 1), (lines_new, None)):
        s = Sim(MAX_X, seed)
        s.set_x(xs, xq)
        stop = end if end is not None else (len(lines) - (len(lines_old) - (B + 1)))
        for i in range(stop):
            op, args, cond = parse(lines[i])
            if op:
                s.step(op, args, cond)
        states.append({k: v for k, v in s.q.items()})
    o, n_ = states
    zero = np.zeros((MAX_X + 63) // 64, dtype=np.uint64)
    bad = [k for k in sorted(set(o) | set(n_), key=lambda t: int(t[1:]))
           if not np.array_equal(o.get(k, zero), n_.get(k, zero))]
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", type=Path)
    ap.add_argument("--barrel", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify", action="store_true", help="run the official verifier (full) on the result")
    ap.add_argument("--src", type=Path, default=SRC)
    a = ap.parse_args()
    lines = a.src.read_text().splitlines()
    assert lines[A].startswith("CX q29 q49") and lines[B].startswith("CSWAP q37 q17 q19") and lines[B + 1].startswith("HMR q22"), \
        (lines[A], lines[B], lines[B + 1])
    if a.barrel:
        gates, fl = barrel_gates(), False
    else:
        spec = json.loads(a.gates.read_text())
        assert spec["n"] == N
        gates, fl = spec["gates"], spec.get("free_leading", False)
    block = build_block(gates, fl)
    new_lines = lines[:A] + block + lines[B + 1:]
    ncs = sum(1 for l in block if l.startswith("CSWAP"))
    print(f"block: {len(block)} lines, {ncs} CSWAP (original 47); free_leading={fl}")
    bad = simulate_compare(lines, new_lines)
    print("full-domain state after block differs on wires:", bad if bad else "none (exact match)")
    a.out.write_text("\n".join(new_lines) + "\n")
    import kickmix as km
    c = km.Circuit.from_file(str(a.out))
    kmb = a.out.with_suffix(".kmb")
    c.to_file(str(kmb), format="kmb")
    print(f"wrote {a.out} and {kmb}; qubits={c.num_qubits} toffoli_like(static)={c.operation_statistics().num_toffoli_like}")
    if a.verify and not bad:
        cmd = [sys.executable, str(HERE / "tools" / "salt_cv.py"), str(kmb), "--tasks", "invsqrt", "--tag", a.out.stem]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
