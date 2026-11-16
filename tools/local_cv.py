#!/usr/bin/env python3
"""Local CV for the Google Quantum AI competition: score a portfolio exactly like the hosted metric.

A *portfolio* is a directory (or a submission.zip) holding `invsqrt.kmb`, `dqi.kmb` and `dfthcsos/`.
For each benchmark the unchanged official verifier is run, the cost is read from its JSON report and
the overall score is the geometric mean of `reference_cost / cost` (the metric's formula).  The
reference costs are the hosts' baseline costs that the metric uses.

Why salts matter: the hosted metric seeds the random inputs and measurement outcomes with a salted
hash of each file, so the *draw* differs per salt.  DQI is the only benchmark whose cost depends on
the draw (one failed shot in 102,400 costs about 0.15% of the overall score).  Use `--salts` to
estimate the expected DQI cost over several draws and `--default-salt` to reproduce the single
"default" draw that public notebooks report.

Every run appends one JSON line per (portfolio, benchmark, salt) to the ledger
`build/cv/ledger.jsonl` so that nothing measured is lost.

Usage:
    tools/local_cv.py PORTFOLIO [--salts s1 s2 ...] [--default-salt] [--tasks invsqrt dqi dfthcsos]
                      [--dqi-workers 12] [--tag NAME]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VERIFIERS = DATA / "verifiers"
LEDGER = ROOT / "build" / "cv" / "ledger.jsonl"
PY = sys.executable

# The metric's baseline costs (also printed in the public circuits datasets' QCC_RELEASE_MANIFEST.json).
REFERENCE_COST = {
    "invsqrt": 2076774.233715282,
    "dqi": 24439463583672.93,
    "dfthcsos": 3355599588.3155375,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    """Hash of a dfthcsos directory: sorted (name, sha256) pairs."""
    h = hashlib.sha256()
    for p in sorted(path.iterdir()):
        if p.is_file():
            h.update(p.name.encode())
            h.update(sha256(p).encode())
    return h.hexdigest()


def verifier_cmd(task: str, target: Path, salt: str | None, dqi_workers: int) -> list[str]:
    if task == "invsqrt":
        cmd = [PY, str(VERIFIERS / "invsqrt" / "verify_submission_invsqrt.py"), str(target)]
    elif task == "dqi":
        cmd = [PY, str(VERIFIERS / "dqi" / "verify_submission_dqi.py"), str(target),
               "--max-workers", str(dqi_workers)]
    elif task == "dfthcsos":
        cmd = [PY, str(VERIFIERS / "dfthcsos" / "verify.py"), str(target)]
    else:
        raise ValueError(task)
    if salt is not None:
        cmd += ["--salt", salt]
    return cmd


def run_one(task: str, target: Path, salt: str | None, dqi_workers: int) -> dict:
    cmd = verifier_cmd(task, target, salt, dqi_workers)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = {"status": "UNPARSEABLE", "stdout_head": proc.stdout[:500], "stderr_tail": proc.stderr[-1500:]}
    cost = report.get("score") if report.get("status") == "PASS" else None
    return {
        "task": task,
        "salt": salt if salt is not None else "<default>",
        "status": report.get("status"),
        "cost": cost,
        "ratio": (REFERENCE_COST[task] / cost) if cost else None,
        "wall_s": round(wall, 2),
        "exit_code": proc.returncode,
        "report": {k: report.get(k) for k in (
            "num_qubits", "toffoli_count", "toffoli_equivalent", "rotation_count", "value_errors",
            "phase_errors", "empirical_error_rate", "effective_error_rate", "repetition_factor",
            "lambda_eff", "circuit_sha256", "submission_sha256", "error")},
    }


def resolve_portfolio(arg: Path, tmp: Path) -> Path:
    if arg.is_dir():
        return arg
    if arg.suffix == ".zip":
        with zipfile.ZipFile(arg) as z:
            z.extractall(tmp)
        return tmp
    raise SystemExit(f"{arg}: not a directory or .zip")


def geometric_mean(ratios: list[float]) -> float:
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("portfolio", type=Path)
    ap.add_argument("--tasks", nargs="+", default=["invsqrt", "dqi", "dfthcsos"])
    ap.add_argument("--salts", nargs="*", default=[], help="extra salts (each adds one DQI/… run)")
    ap.add_argument("--default-salt", action="store_true", default=True,
                    help="also run with no --salt (the public-notebook 'default draw'); on by default")
    ap.add_argument("--no-default-salt", dest="default_salt", action="store_false")
    ap.add_argument("--salted-tasks", nargs="+", default=["dqi"],
                    help="tasks whose cost depends on the draw; only these are repeated per salt")
    ap.add_argument("--dqi-workers", type=int, default=12)
    ap.add_argument("--tag", default=None, help="label stored in the ledger")
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    a = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="cv_") as tmpdir:
        pf = resolve_portfolio(a.portfolio, Path(tmpdir))
        targets = {"invsqrt": pf / "invsqrt.kmb", "dqi": pf / "dqi.kmb", "dfthcsos": pf / "dfthcsos"}
        for t in a.tasks:
            if not targets[t].exists():
                raise SystemExit(f"missing {targets[t]}")
        ids = {t: (sha256(targets[t]) if targets[t].is_file() else sha256_dir(targets[t])) for t in a.tasks}
        tag = a.tag or a.portfolio.name
        print(f"portfolio {tag}")
        for t in a.tasks:
            print(f"  {t:9s} {ids[t][:12]}")

        salts: list[str | None] = ([None] if a.default_salt else []) + list(a.salts)
        results: dict[str, list[dict]] = {t: [] for t in a.tasks}
        a.ledger.parent.mkdir(parents=True, exist_ok=True)
        for t in a.tasks:
            for s in (salts if t in a.salted_tasks else salts[:1]):
                r = run_one(t, targets[t], s, a.dqi_workers)
                r |= {"tag": tag, "file_sha256": ids[t], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                results[t].append(r)
                with open(a.ledger, "a") as fh:
                    fh.write(json.dumps(r, default=str) + "\n")
                rep = r["report"]
                extra = ""
                if t == "dqi":
                    extra = f" value_err={rep.get('value_errors')} phase_err={rep.get('phase_errors')}"
                elif t == "dfthcsos":
                    extra = f" lambda_eff={rep.get('lambda_eff')}"
                print(f"  {t:9s} salt={r['salt']:<18} {r['status']:5} cost={r['cost']!s:<22} "
                      f"ratio={r['ratio']!s:<20} Q={rep.get('num_qubits')} wall={r['wall_s']}s{extra}")

    # Summary: default-draw score and expected score over all draws (mean cost per task).
    def score_from(costs: dict[str, float]) -> float | None:
        if any(costs.get(t) is None for t in a.tasks):
            return None
        return geometric_mean([REFERENCE_COST[t] / costs[t] for t in a.tasks])

    default_costs = {t: results[t][0]["cost"] for t in a.tasks}
    mean_costs = {}
    for t in a.tasks:
        ok = [r["cost"] for r in results[t] if r["cost"]]
        mean_costs[t] = (sum(ok) / len(ok)) if ok and len(ok) == len(results[t]) else None
    print()
    print(f"score (default draw) = {score_from(default_costs)}")
    if len(salts) > 1:
        fails = sum(1 for r in results.get("dqi", []) if r["status"] != "PASS")
        print(f"score (mean over {len(salts)} draws) = {score_from(mean_costs)}   dqi failed runs: {fails}")
    summary = {"tag": tag, "ids": ids, "score_default": score_from(default_costs),
               "score_mean": score_from(mean_costs) if len(salts) > 1 else None,
               "salts": [s or "<default>" for s in salts],
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kind": "summary"}
    with open(a.ledger, "a") as fh:
        fh.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
