#!/usr/bin/env python3
"""Summarise the DQI salt sweep: expected cost, failure rate and draw risk per file.

Reads build/cv/salts/dqi_<tag>_<salt>.json plus build/cv/anchor/dqi_<tag>_default.json and prints,
per file: number of draws, failed shots per draw, mean cost, min/max cost, and the implied overall
score contribution (reference / mean cost).  One failed shot in 102,400 costs about 0.46 % of the
DQI cost (R jumps from 1.19126 to 1.19673), i.e. about 0.15 % of the overall score.
"""
from __future__ import annotations

import glob
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = 24439463583672.93

rows = defaultdict(list)
for f in glob.glob(str(ROOT / "build/cv/salts/dqi_*_*.json")) + glob.glob(str(ROOT / "build/cv/anchor/dqi_*_default.json")):
    m = re.match(r"dqi_([0-9a-f]+)_(\w+)\.json", Path(f).name)
    if not m:
        continue
    try:
        r = json.loads(Path(f).read_text())
    except json.JSONDecodeError:
        continue
    if r.get("status") != "PASS":
        continue
    rows[m.group(1)].append((m.group(2), r["score"], r.get("value_errors", 0) + r.get("phase_errors", 0), r.get("toffoli_count")))

print(f"{'file':10} {'draws':>5} {'fails/draw':<22} {'mean cost':>16} {'min':>16} {'max':>16} {'ref/mean':>9} {'T(mean)':>10}")
for tag, lst in sorted(rows.items(), key=lambda kv: statistics.mean(x[1] for x in kv[1])):
    costs = [x[1] for x in lst]
    fails = [x[2] for x in lst]
    T = statistics.mean(x[3] for x in lst if x[3])
    print(f"{tag:10} {len(lst):5d} {''.join(str(k) for _, k in sorted(zip([x[0] for x in lst], fails))):<22} "
          f"{statistics.mean(costs):16.0f} {min(costs):16.0f} {max(costs):16.0f} {REF / statistics.mean(costs):9.4f} {T:10.1f}")
print("\nfails/draw lists the failed-shot count per draw (sorted by salt name; 'default' first when present).")
