#!/usr/bin/env python3
"""Structured search for the n=22 normalizing shifter: layers of segment reversals, controls by SAT.

A layer = reversal of wires [a, b] (floor((b-a+1)/2) CSWAPs); every CSWAP in a layer gets its own
free control function of z (the SAT decides them), so a layer is just a convenient way to pick
swap pairs.  Pairs fixed -> the SAT only chooses controls and is fast.  Random restarts + greedy
local moves on the segment list, keeping total cost below --budget.

Usage: tools/shifter_layers.py --n 22 --budget 46 --layers 6 --trials 200 [--free-leading] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shifter_sat import check, solve  # noqa: E402


def layer_pairs(a: int, b: int):
    return [(a + i, b - i) for i in range((b - a + 1) // 2)]


def cost(segs):
    return sum((b - a + 1) // 2 for a, b in segs)


def pairs_of(segs):
    out = []
    for a, b in segs:
        out += layer_pairs(a, b)
    return out


def random_segs(n, L, rng, budget):
    while True:
        segs = []
        for _ in range(L):
            a = rng.randrange(0, n - 1)
            b = rng.randrange(a + 1, n)
            segs.append((a, b))
        if cost(segs) <= budget:
            return segs


def feasible(n, segs, free_leading, timeout=60):
    pairs = pairs_of(segs)
    ok, gates = solve(n, len(pairs), timeout, verbose=False, free_leading=free_leading, solver="glucose", fixed_pairs=pairs)
    return ok, gates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--budget", type=int, default=46)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--free-leading", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--segs", type=str, default=None, help="JSON list of [a,b] to test directly")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if a.segs:
        segs = [tuple(x) for x in json.loads(a.segs)]
        t0 = time.time()
        ok, gates = feasible(a.n, segs, a.free_leading, a.timeout)
        print(f"segs {segs} cost {cost(segs)} -> {ok} ({time.time() - t0:.1f}s)")
        if ok:
            for gt in gates:
                print("  ", gt)
            if a.save:
                json.dump({"n": a.n, "N": len(gates), "free_leading": a.free_leading, "gates": gates}, open(a.save, "w"), indent=1)
        return 0
    best = None
    for trial in range(a.trials):
        segs = random_segs(a.n, a.layers, rng, a.budget)
        t0 = time.time()
        ok, gates = feasible(a.n, segs, a.free_leading, a.timeout)
        c = cost(segs)
        print(f"trial {trial}: {segs} cost {c} -> {ok} ({time.time() - t0:.1f}s)", flush=True)
        if ok and (best is None or c < best[0]):
            best = (c, segs, gates)
            if a.save:
                json.dump({"n": a.n, "N": len(gates), "free_leading": a.free_leading, "segs": segs, "gates": gates}, open(a.save, "w"), indent=1)
            print(f"  ** new best cost {c}", flush=True)
    print("best:", best[:2] if best else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
