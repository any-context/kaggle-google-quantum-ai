#!/usr/bin/env python3
"""Second-tier invsqrt scans from the technical guide: dirty 5->4 CCX pattern (3.3.2), history cache
(3.3.3) with occupancy check, and rotation merging (3.5.2 / 3.5.5) on value-equal or constant wires.

All value questions are answered on the full input domain with the bit-parallel simulator of
inv_scan.py (one outcome assignment per seed; conditional blocks are NOT trusted: only unconditional
gates are considered for rewrites).

Usage: tools/inv_scan2.py CIRCUIT.kmx [--seed 1000]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inv_scan import MAX_X, Sim, parse, popcount  # noqa: E402


def dirty_pattern(lines):
    """CCX a b d; CCX d c t; CCX a b d; CCX d c t; CCX a b c  (all unconditional, consecutive)."""
    hits = []
    ops = [parse(l) for l in lines]
    for i in range(len(lines) - 4):
        w = ops[i:i + 5]
        if any(o[0] != "CCX" or o[2] is not None for o in w):
            continue
        a, b, d = w[0][1]
        if w[2][1] != [a, b, d]:
            continue
        if w[1][1][2] != w[3][1][2] or w[1][1][0] != d or w[3][1][0] != d:
            continue
        c, t = w[1][1][1], w[1][1][2]
        if w[3][1][1] != c or w[4][1] != [a, b, c]:
            continue
        hits.append((i, a, b, c, d, t))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("circuit", type=Path)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-x", type=int, default=MAX_X)
    a = ap.parse_args()
    lines = a.circuit.read_text().splitlines()
    xq = [parse(l)[1][0] for l in lines if l.startswith("APPEND_TO_REGISTER")]

    print("== (a) dirty 5->4 CCX pattern")
    hits = dirty_pattern(lines)
    print(f"   {len(hits)} occurrences", hits[:5])

    sim = Sim(a.max_x, a.seed)
    sim.set_x(np.arange(1, a.max_x + 1, dtype=np.uint64), xq)
    products = {}      # hash(product array) -> list of (line, a, b, t)
    rot = []           # (line, wire, angle, value hash, const)
    occ = np.zeros(len(lines), dtype=np.int32)
    erased_at = {}     # wire -> last line where it was reset (HMR/R)
    live_set = set()
    for i, line in enumerate(lines):
        op, args, cond = parse(line)
        if op is None:
            continue
        uncond = cond is None and not sim.stack
        if op == "CCX" and uncond:
            p = sim.qv(args[0]) & sim.qv(args[1])
            h = hashlib.blake2b(p.tobytes(), digest_size=16).hexdigest()
            products.setdefault(h, []).append((i, args[0], args[1], args[2], popcount(p)))
        if op == "Z_POW" and uncond:
            v = sim.qv(args[0])
            pc = popcount(v)
            rot.append((i, args[0], args[1], hashlib.blake2b(v.tobytes(), digest_size=16).hexdigest(),
                        hashlib.blake2b((~v & sim.all).tobytes(), digest_size=16).hexdigest(), pc))
        sim.step(op, args, cond)
        if op in ("HMR", "R"):
            erased_at[args[0]] = i
        # occupancy: wires with any nonzero value (cheap incremental: only re-check touched wires)
        for w in [x for x in args if x.startswith("q")]:
            if sim.q.get(w) is not None and sim.q[w].any():
                live_set.add(w)
            else:
                live_set.discard(w)
        occ[i] = len(live_set)
    print(f"   occupancy: max {occ.max()} at line {int(occ.argmax())}, lines at max: {(occ == occ.max()).sum()}")

    print("== (b) history-cache candidates (same a&b product computed by two unconditional CCX)")
    cands = 0
    feasible = 0
    for h, lst in products.items():
        if len(lst) < 2 or lst[0][4] == 0:
            continue
        for (i1, a1, b1, t1, _), (i2, a2, b2, t2, _) in zip(lst, lst[1:]):
            cands += 1
            peak = int(occ[i1:i2 + 1].max())
            if peak < 62:
                feasible += 1
                print(f"   lines {i1}->{i2} product {a1}&{b1} (then {a2}&{b2}) peak occupancy {peak} -> cache possible")
    print(f"   {cands} recompute pairs, {feasible} with a free wire over the whole interval")

    print("== (c) rotations on value-equal / constant wires")
    byval = {}
    for (i, w, ang, hv, hnv, pc) in rot:
        if pc == 0:
            print(f"   line {i}: Z_POW {w} {ang[:12]} on an always-0 wire -> delete")
        elif pc == a.max_x:
            print(f"   line {i}: Z_POW {w} {ang[:12]} on an always-1 wire -> global phase")
        byval.setdefault(hv, []).append((i, w, ang))
    merges = 0
    for hv, lst in byval.items():
        if len(lst) > 1:
            merges += len(lst) - 1
            print(f"   value-equal rotation wires: {[(i, w, ang[:10]) for i, w, ang in lst]}")
    comp = 0
    for (i, w, ang, hv, hnv, pc) in rot:
        if hnv in byval:
            comp += 1
            print(f"   line {i}: {w} is the complement of rotation wire(s) {[x[1] for x in byval[hnv]]}")
    print(f"   {len(rot)} unconditional Z_POW, {merges} mergeable by equality, {comp} complements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
