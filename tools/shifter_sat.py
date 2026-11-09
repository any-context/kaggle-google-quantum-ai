#!/usr/bin/env python3
"""Minimal controlled-swap network for a normalizing barrel shifter (discussion 757030, problem 1).

Problem: n wires hold x != 0 with z leading zeros (z = 0..n-1).  Shift left by z so the leading one
lands on wire n-1.  Each gate is CSWAP(c(z), a, b) where the control c(z) may be ANY Boolean function
of z (the thermometer of z is available and XORs are free).  Zero bits may end anywhere.  Find a
network with N gates, or prove none exists.

Encoding (per shift amount z, data bits i = 0..n-1-z are tracked as one-hot locations):
    g[t][p]        gate t swaps pair p = (a,b)            (exactly one pair per gate)
    c[t][z]        gate t is active for shift z
    loc[z][t][i][w]  data bit i sits on wire w before gate t (exactly one w)
    goal: loc[z][N][i][i+z]

Usage:
    tools/shifter_sat.py --n 6 --gates 7              # one instance
    tools/shifter_sat.py --n 9 --min                   # decrease N until UNSAT (prints the optimum)
    tools/shifter_sat.py --n 22 --gates 46 --timeout 7200 --save out.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Cadical195, Glucose4, MapleChrono


def build(n: int, N: int, zs=None, free_leading: bool = False, symmetry: bool = True, fixed_pairs=None, max_used=None):
    # free_leading: the leading one (bit n-1-z) is a known constant; X gates are free, so it is not tracked
    ndata = (lambda z: n - z - 1) if free_leading else (lambda z: n - z)
    pool = IDPool()
    clauses = []
    pairs = list(itertools.combinations(range(n), 2))
    g = {(t, p): pool.id(("g", t, p)) for t in range(N) for p in pairs}
    zs = list(range(n)) if zs is None else zs
    c = {(t, z): pool.id(("c", t, z)) for t in range(N) for z in zs}
    loc = {}
    for z in zs:
        for t in range(N + 1):
            for i in range(ndata(z)):
                for w in range(n):
                    loc[z, t, i, w] = pool.id(("l", z, t, i, w))

    def eo(lits):
        enc = CardEnc.equals(lits=lits, bound=1, vpool=pool, encoding=EncType.seqcounter)
        clauses.extend(enc.clauses)

    for t in range(N):
        eo([g[t, p] for p in pairs])
    if fixed_pairs is not None:
        assert len(fixed_pairs) == N
        for t, p in enumerate(fixed_pairs):
            clauses.append([g[t, tuple(sorted(p))]])
    # m[t][w]: wire w is touched by gate t
    m = {}
    for t in range(N):
        for w in range(n):
            m[t, w] = pool.id(("m", t, w))
            ps = [g[t, p] for p in pairs if w in p]
            clauses.append([-m[t, w]] + ps)
            for lit in ps:
                clauses.append([-lit, m[t, w]])
    for z in zs:
        for i in range(ndata(z)):
            # initial and final positions
            for w in range(n):
                clauses.append([loc[z, 0, i, w] if w == i else -loc[z, 0, i, w]])
            clauses.append([loc[z, N, i, i + z]])
            for t in range(N):
                eo([loc[z, t + 1, i, w] for w in range(n)])
                for w in range(n):
                    L0 = loc[z, t, i, w]
                    # stays if gate inactive or wire untouched
                    clauses.append([-L0, c[t, z], loc[z, t + 1, i, w]])
                    clauses.append([-L0, m[t, w], loc[z, t + 1, i, w]])
                    # moves to partner if active and pair contains w
                    for v in range(n):
                        if v == w:
                            continue
                        p = (w, v) if w < v else (v, w)
                        clauses.append([-L0, -c[t, z], -g[t, p], loc[z, t + 1, i, v]])
    # symmetry breaking: consecutive gates on disjoint wire pairs commute, so demand increasing pair index
    if symmetry:
        pidx = {p: i for i, p in enumerate(pairs)}
        for t in range(N - 1):
            for p in pairs:
                for q in pairs:
                    if pidx[q] < pidx[p] and not (set(p) & set(q)):
                        clauses.append([-g[t, p], -g[t + 1, q]])
    # z = 0 needs no movement: force every gate inactive for z=0 (harmless, removes freedom)
    if 0 in zs:
        for t in range(N):
            clauses.append([-c[t, 0]])
    if max_used is None:
        # symmetry breaking: a gate that is active for no z is useless -> require at least one active z
        for t in range(N):
            clauses.append([c[t, z] for z in zs if z != 0])
    else:
        # pool mode: used[t] <-> gate t active for some z ; at most max_used gates used
        used = []
        for t in range(N):
            u = pool.id(("u", t))
            used.append(u)
            acts = [c[t, z] for z in zs if z != 0]
            clauses.append([-u] + acts)
            for lit in acts:
                clauses.append([-lit, u])
        clauses.extend(CardEnc.atmost(lits=used, bound=max_used, vpool=pool, encoding=EncType.seqcounter).clauses)
    return pool, clauses, pairs, g, c, loc


def solve(n: int, N: int, timeout: float | None, verbose=True, free_leading=False, solver="auto", symmetry=True, fixed_pairs=None, max_used=None):
    t0 = time.time()
    pool, clauses, pairs, g, c, loc = build(n, N, free_leading=free_leading, symmetry=symmetry and fixed_pairs is None, fixed_pairs=fixed_pairs, max_used=max_used)
    if verbose:
        print(f"  n={n} N={N}: {pool.top} vars, {len(clauses)} clauses (built in {time.time() - t0:.1f}s)", flush=True)
    use_glucose = (solver == "glucose") or (solver == "auto" and timeout)
    s = (Glucose4 if use_glucose else Cadical195)(bootstrap_with=clauses)
    if timeout and use_glucose:
        # CaDiCaL cannot be interrupted from pysat; Glucose can.
        import threading
        timer = threading.Timer(timeout, lambda: s.interrupt())
        timer.start()
        res = s.solve_limited(expect_interrupt=True)
        timer.cancel()
    else:
        res = s.solve()
    dt = time.time() - t0
    if res is None:
        if verbose:
            print(f"  -> TIMEOUT after {dt:.0f}s", flush=True)
        return None, None
    if not res:
        if verbose:
            print(f"  -> UNSAT ({dt:.1f}s)", flush=True)
        return False, None
    model = set(l for l in s.get_model() if l > 0)
    gates = []
    for t in range(N):
        p = next(p for p in pairs if g[t, p] in model)
        active = [z for z in range(n) if c[t, z] in model]
        if max_used is not None and not active:
            continue  # unused pool slot
        gates.append({"pair": p, "active_z": active})
    if verbose:
        print(f"  -> SAT ({dt:.1f}s)" + (f", {len(gates)} gates used" if max_used is not None else ""), flush=True)
    return True, gates


def check(n: int, gates, free_leading=False) -> bool:
    for z in range(n):
        nd = n - z - 1 if free_leading else n - z
        state = [i if i < nd else -1 for i in range(n)]
        for gt in gates:
            if z in gt["active_z"]:
                a, b = gt["pair"]
                state[a], state[b] = state[b], state[a]
        for w in range(n):
            if state[w] >= 0 and w != state[w] + z:
                return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--gates", type=int, default=None)
    ap.add_argument("--min", action="store_true", help="decrease N from --gates (default 2n) until UNSAT")
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--free-leading", action="store_true", help="leading one is a known constant (X is free): do not track it")
    ap.add_argument("--solver", choices=["auto", "cadical", "glucose"], default="auto", help="cadical ignores --timeout (use an external timeout)")
    ap.add_argument("--no-symmetry", action="store_true")
    ap.add_argument("--pool", type=str, default=None, help="JSON list of ordered candidate pairs; SAT picks a subset")
    ap.add_argument("--max-used", type=int, default=None, help="pool mode: at most this many gates")
    a = ap.parse_args()
    n = a.n
    if a.min:
        N = a.gates or 2 * n
        best = None
        while N > 0:
            ok, gates = solve(n, N, a.timeout, free_leading=a.free_leading, solver=a.solver, symmetry=not a.no_symmetry)
            if ok:
                assert check(n, gates, a.free_leading)
                best = (N, gates)
                N -= 1
            else:
                break
        if best:
            print(f"OPTIMUM n={n}: {best[0]} gates" + ("" if ok is False else " (next N timed out)"))
            for t, gt in enumerate(best[1]):
                print(f"   {t:2d}: CSWAP {gt['pair']}  active z={gt['active_z']}")
            if a.save:
                json.dump({"n": n, "N": best[0], "free_leading": a.free_leading, "gates": best[1]}, open(a.save, "w"), indent=1)
        return 0
    pool_pairs = None
    if a.pool:
        pool_pairs = [tuple(x) for x in json.loads(Path(a.pool).read_text())] if Path(a.pool).exists() else [tuple(x) for x in json.loads(a.pool)]
        a.gates = len(pool_pairs)
    ok, gates = solve(n, a.gates, a.timeout, free_leading=a.free_leading, solver=a.solver, symmetry=not a.no_symmetry,
                      fixed_pairs=pool_pairs, max_used=a.max_used)
    if ok:
        assert check(n, gates, a.free_leading)
        for t, gt in enumerate(gates):
            print(f"   {t:2d}: CSWAP {gt['pair']}  active z={gt['active_z']}")
        if a.save:
            json.dump({"n": n, "N": a.gates, "free_leading": a.free_leading, "gates": gates}, open(a.save, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
