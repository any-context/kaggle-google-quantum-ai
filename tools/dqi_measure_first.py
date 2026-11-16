#!/usr/bin/env python3
"""Measured-target Toffoli peephole for kickmix circuits (DQI), our own implementation.

Identity (Kirill Tymchishin, Pi-Gauge & Weight Refinement v21; hosen42's measure-first idea):
    CCX a b t ; ... ; HMR t m        ==>        HMR t m ; CZ a b if m ; ...
provided that nothing touches `t` between the CCX and the HMR, and that nothing touches the
classical bit `m` in that window either (the HMR is moved earlier, so `m` is overwritten earlier).

Why it is exact: HMR measures `t` in the X basis, records the outcome `m` and resets `t`.
On a basis state the CCX leaves `t ^ (a & b)` on the wire, so the original phase is
(-1)^(m * (t ^ ab)) = (-1)^(m*t) * (-1)^(m*ab).  Measuring first contributes (-1)^(m*t), and the
classically conditioned CZ contributes (-1)^(m*ab), evaluated on the *same* a, b the CCX saw
because the CZ is inserted at the CCX's position.  Both leave `t` reset.  The conditioned CZ is
Clifford, so one Toffoli disappears per site.

Usage:
    tools/dqi_measure_first.py scan  IN.kmx                 # list candidate sites
    tools/dqi_measure_first.py apply IN.kmx OUT.kmx [--only-when-control-measured]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse(line: str):
    """Return (op, args, cond) for an instruction line; cond is the `if bN` bit or None."""
    parts = line.split()
    if not parts:
        return None, [], None
    cond = None
    if len(parts) >= 3 and parts[-2] == "if":
        cond = parts[-1]
        parts = parts[:-2]
    return parts[0], parts[1:], cond


def touched(op: str, args: list[str]) -> list[str]:
    """Wires (q*/b*) an instruction reads or writes, excluding the angle of Z_POW."""
    if op in ("APPEND_TO_REGISTER", "REGISTER", "INFO_TAG"):
        return []
    return [a for a in args if a and a[0] in "qb"]


def find_sites(lines: list[str], require_control_measured: bool, log=print, generalized: bool = False):
    """Return rewrite sites.

    Basic site:        CCX a b t ; <nothing touches t or m> ; HMR t m
    Generalized site:  CCX a b t ; <only *writes* to t: CX c t | CCX x y t | X t> ; HMR t m
      Every write becomes a phase conditioned on m at its own position:
        CCX x y t -> CZ x y if m,   CX c t -> Z c if m,   X t -> NEG if m
      and the HMR moves to the first CCX's position.  Each CCX in the window saves one Toffoli.
    Each site is (ccx_index, hmr_index, m, writes) with writes = [(index, kind, args...)].
    """
    last_touch: dict[str, int] = {}
    pending: dict[str, dict] = {}          # t -> {"start": i, "depth": d, "writes": [...]}
    depth = 0
    sites = []
    t0 = time.time()
    for i, line in enumerate(lines):
        op, args, cond = parse(line)
        if op is None:
            continue
        if op == "PUSH_CONDITION":
            depth += 1
        elif op == "POP_CONDITION":
            depth -= 1
        w = touched(op, args)
        if cond:
            w = w + [cond]

        handled = False
        if op == "HMR" and cond is None and args[0] in pending:
            t, m = args[0], args[1]
            st = pending.pop(t)
            ok = st["depth"] == depth and last_touch.get(m, -1) < st["start"]
            if ok:
                sites.append((st["start"], i, m, st["writes"]))
            handled = True
        elif generalized and cond is None and op in ("CX", "CCX", "X") and args[-1] in pending \
                and all(x not in pending or x == args[-1] for x in args[:-1]):
            # an extra write to a pending wire: absorb it (controls must not be pending wires themselves)
            t = args[-1]
            st = pending[t]
            if st["depth"] == depth:
                st["writes"].append((i, op, *args[:-1]))
                for x in w:
                    if x != t and x in pending:
                        del pending[x]
                handled = True
        if not handled:
            for x in w:
                if x in pending:
                    del pending[x]
        if op == "CCX" and cond is None and (not handled or args[2] not in pending):
            a, b, t = args
            pending[t] = {"start": i, "depth": depth, "writes": [(i, "CCX", a, b)]}
        for x in w:
            last_touch[x] = i
        if i % 10_000_000 == 0 and i:
            log(f"  scanned {i:,} lines, {len(sites)} sites, {time.time() - t0:.0f}s", file=sys.stderr)
    return sites


def apply(lines: list[str], sites) -> list[str]:
    drop = set()
    insert: dict[int, list[str]] = {}
    for start, hmr_i, m, writes in sites:
        drop.add(hmr_i)
        for (idx, kind, *ctrl) in writes:
            drop.add(idx)
            if kind == "CCX":
                repl = [f"CZ {ctrl[0]} {ctrl[1]} if {m}"]
            elif kind == "CX":
                repl = [f"Z {ctrl[0]} if {m}"]
            else:  # X t
                repl = [f"NEG if {m}"]
            insert.setdefault(idx, []).extend(repl)
        t = lines[hmr_i].split()[1]
        insert[start] = [f"HMR {t} {m}"] + insert.get(start, [])
    out = []
    for i, line in enumerate(lines):
        if i in insert:
            out.extend(insert[i])
        if i in drop:
            continue
        out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("scan", "apply"))
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path, nargs="?")
    ap.add_argument("--only-when-control-measured", action="store_true",
                    help="(reserved) restrict to Kirill's stricter pattern")
    ap.add_argument("--max-sites", type=int, default=None)
    ap.add_argument("--generalized", action="store_true", help="also absorb CX/CCX/X writes before the HMR")
    a = ap.parse_args()

    t0 = time.time()
    lines = a.src.read_text().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    print(f"read {len(lines):,} lines in {time.time() - t0:.1f}s", file=sys.stderr)
    sites = find_sites(lines, a.only_when_control_measured, generalized=a.generalized)
    print(f"{len(sites)} candidate sites", file=sys.stderr)
    n_ccx = sum(1 for s in sites for wr in s[3] if wr[1] == "CCX")
    print(f"Toffolis removable: {n_ccx}", file=sys.stderr)
    for s in sites[:40]:
        print("  start@%d HMR@%d m=%s writes=%s" % (s[0], s[1], s[2], [(k, *c) for (_, k, *c) in s[3]]))
    if a.mode == "apply":
        if a.max_sites:
            sites = sites[:a.max_sites]
        out = apply(lines, sites)
        a.dst.write_text("\n".join(out) + "\n")
        print(f"wrote {a.dst} ({len(out):,} lines, {len(sites)} sites rewritten)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
