#!/usr/bin/env python3
"""Multi-salt local CV with a worst-case summary, for the Google Quantum AI competition.

The hosted metric seeds every verifier with sha256(file) merged with a hidden per-row salt, and the
public and private leaderboards use different salts.  invsqrt and dfthcsos costs are static (they
verify the whole input domain / closed-form lambda_eff) but the DQI cost carries a repetition factor
R(eps+) that jumps about +0.46 % per failed shot in 102,400.  A single-salt local score is therefore
a single draw, and the right local CV is a *distribution* over draws.

This script runs the unchanged official verifiers over several salts and reports, per task and for
the overall geometric-mean score:

    default   the salt='' draw (what public notebooks report)
    mean      mean cost over all draws               -> expected private score
    worst     max cost over all draws                -> the CV number to quote and gate on
    fails     draws on which the verifier FAILed (eps+ >= 1/16 or wrong output) -> submission = 0

Portfolio = directory or submission.zip holding invsqrt.kmb, dqi.kmb, dfthcsos/.  A single .kmb (or a
single dfthcsos directory) can be priced too with --tasks <task>.

Usage:
    tools/salt_cv.py PORTFOLIO --salt-set gate                # default + g1..g8 (fixed set, CRN across candidates)
    tools/salt_cv.py PORTFOLIO --salts a b c                  # explicit salts
    tools/salt_cv.py PORTFOLIO --random-salts 5 --seed 7      # reproducible pseudo-random salts
    tools/salt_cv.py dqi.kmb --tasks dqi --salt-set gate      # price one DQI file
    tools/salt_cv.py PORTFOLIO --all-salted                   # also repeat invsqrt/dfthcsos per salt (parity check)
    tools/salt_cv.py --summary                                # re-print the worst-case table from the ledger

Repo layout: the verifiers and venv live in the main checkout (data/, .venv/); when run from a git
worktree they are found through `git rev-parse --git-common-dir`.  Override with --repo / --python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# The hosted metric's baseline costs (reference implementation under the hosts' salt).
REFERENCE_COST = {
    "invsqrt": 2076774.233715282,
    "dqi": 24439463583672.93,
    "dfthcsos": 3355599588.3155375,
}
TASKS = ("invsqrt", "dqi", "dfthcsos")
SALT_SETS = {
    "explore": ["s1", "s2", "s3"],
    "gate": [f"g{i}" for i in range(1, 9)],
    "final": [f"h{i}" for i in range(1, 9)],
}


def find_repo(explicit: Path | None) -> Path:
    """Main checkout holding data/verifiers (worktree-aware)."""
    cands = []
    if explicit:
        cands.append(explicit)
    cands.append(ROOT)
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        cands.append((ROOT / common).resolve().parent)
    except Exception:
        pass
    for c in cands:
        if (c / "data" / "verifiers" / "dqi" / "verify_submission_dqi.py").exists():
            return c
    raise SystemExit("cannot find data/verifiers; pass --repo /path/to/checkout")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.iterdir()):
        if p.is_file():
            h.update(p.name.encode())
            h.update(sha256(p).encode())
    return h.hexdigest()


def verifier_cmd(repo: Path, py: str, task: str, target: Path, salt: str | None, dqi_workers: int) -> list[str]:
    v = repo / "data" / "verifiers"
    if task == "invsqrt":
        cmd = [py, str(v / "invsqrt" / "verify_submission_invsqrt.py"), str(target)]
    elif task == "dqi":
        cmd = [py, str(v / "dqi" / "verify_submission_dqi.py"), str(target), "--max-workers", str(dqi_workers)]
    elif task == "dfthcsos":
        cmd = [py, str(v / "dfthcsos" / "verify.py"), str(target)]
    else:
        raise ValueError(task)
    if salt is not None:
        cmd += ["--salt", salt]
    return cmd


KEEP = ("num_qubits", "toffoli_count", "toffoli_equivalent", "value_errors", "phase_errors",
        "empirical_error_rate", "effective_error_rate", "repetition_factor", "lambda_eff",
        "circuit_sha256", "submission_sha256", "error")


def run_one(repo: Path, py: str, task: str, target: Path, salt: str | None, dqi_workers: int, raw_dir: Path | None) -> dict:
    cmd = verifier_cmd(repo, py, task, target, salt, dqi_workers)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        report = {"status": "UNPARSEABLE", "error": (proc.stderr or proc.stdout)[-1200:]}
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{task}_{salt or 'default'}.json").write_text(proc.stdout or json.dumps(report))
        if proc.stderr:
            (raw_dir / f"{task}_{salt or 'default'}.err").write_text(proc.stderr)
    passed = report.get("status") == "PASS"
    cost = report.get("score") if passed else None
    return {
        "task": task, "salt": salt if salt is not None else "<default>", "status": report.get("status"),
        "cost": cost, "ratio": (REFERENCE_COST[task] / cost) if cost else None,
        "k": (report.get("value_errors", 0) or 0) + (report.get("phase_errors", 0) or 0) if task == "dqi" and passed else None,
        "wall_s": round(wall, 1), "exit_code": proc.returncode,
        "report": {k: report.get(k) for k in KEEP if k in report},
    }


def resolve_targets(arg: Path, tasks: list[str], tmp: Path) -> dict[str, Path]:
    if arg.is_file() and arg.suffix in (".kmb", ".kmx"):
        if len(tasks) != 1:
            raise SystemExit("a single circuit file needs exactly one --tasks entry")
        return {tasks[0]: arg}
    if arg.is_dir() and tasks == ["dfthcsos"] and not (arg / "dfthcsos").exists():
        return {"dfthcsos": arg}
    if arg.suffix == ".zip":
        with zipfile.ZipFile(arg) as z:
            z.extractall(tmp)
        arg = tmp
        # tolerate one wrapping directory inside the zip
        if not (arg / "invsqrt.kmb").exists():
            inner = [p for p in arg.iterdir() if p.is_dir()]
            if len(inner) == 1 and (inner[0] / "invsqrt.kmb").exists():
                arg = inner[0]
    if not arg.is_dir():
        raise SystemExit(f"{arg}: not a portfolio directory, zip or circuit file")
    t = {"invsqrt": arg / "invsqrt.kmb", "dqi": arg / "dqi.kmb", "dfthcsos": arg / "dfthcsos"}
    for k in tasks:
        if not t[k].exists():
            alt = list(arg.rglob(t[k].name))
            if len(alt) == 1:
                t[k] = alt[0]
            else:
                raise SystemExit(f"missing {t[k]}")
    return {k: t[k] for k in tasks}


def gmean(ratios: list[float]) -> float:
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def summarize(tag: str, ids: dict[str, str], results: dict[str, list[dict]], tasks: list[str]) -> dict:
    """Per-task default / mean / worst cost, failures, and the three overall scores."""
    per_task = {}
    for t in tasks:
        rs = results[t]
        costs = [r["cost"] for r in rs if r["cost"]]
        fails = [r["salt"] for r in rs if r["status"] != "PASS"]
        default = next((r["cost"] for r in rs if r["salt"] == "<default>"), None)
        per_task[t] = {
            "draws": len(rs), "fails": fails, "n_fail": len(fails),
            "cost_default": default,
            "cost_mean": (sum(costs) / len(costs)) if costs else None,
            "cost_worst": max(costs) if costs else None,
            "cost_best": min(costs) if costs else None,
            "k_per_draw": {r["salt"]: r["k"] for r in rs} if t == "dqi" else None,
            "num_qubits": next((r["report"].get("num_qubits") for r in rs if r["report"].get("num_qubits")), None),
        }

    def score(key: str) -> float | None:
        costs = [per_task[t][key] for t in tasks]
        if any(c is None for c in costs):
            return None
        return gmean([REFERENCE_COST[t] / c for t, c in zip(tasks, costs)])

    any_fail = any(per_task[t]["n_fail"] for t in tasks)
    return {
        "kind": "summary", "tag": tag, "ids": ids, "tasks": tasks,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_task": per_task,
        "score_default": score("cost_default"),
        "score_mean": score("cost_mean"),
        # worst-case CV: the worst passing draw per task; a failed draw anywhere makes it 0 (metric raises).
        "score_worst": 0.0 if any_fail else score("cost_worst"),
        "score_worst_passing": score("cost_worst"),
    }


def print_summary(s: dict) -> None:
    print()
    print(f"== {s['tag']}  ({s['timestamp']})")
    print(f"{'task':9} {'Q':>5} {'draws':>5} {'fails':>5} {'default':>18} {'mean':>18} {'worst':>18} {'ratio(worst)':>13} {'k per draw'}")
    for t in s["tasks"]:
        p = s["per_task"][t]
        ratio = (REFERENCE_COST[t] / p["cost_worst"]) if p["cost_worst"] else float("nan")
        ks = "" if not p["k_per_draw"] else " ".join(f"{k}:{v}" for k, v in p["k_per_draw"].items())
        print(f"{t:9} {str(p['num_qubits']):>5} {p['draws']:5d} {p['n_fail']:5d} {p['cost_default']!s:>18.18} "
              f"{p['cost_mean']!s:>18.18} {p['cost_worst']!s:>18.18} {ratio:13.5f} {ks}")
    fmt = lambda x: "n/a" if x is None else f"{x:.5f}"
    print(f"score  default={fmt(s['score_default'])}  mean(expected)={fmt(s['score_mean'])}  "
          f"WORST(CV)={fmt(s['score_worst'])}" + ("" if s["score_worst"] else f"  (worst passing draw {fmt(s['score_worst_passing'])})"))


def ledger_summaries(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    out = []
    for line in ledger.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("kind") == "summary":
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("portfolio", type=Path, nargs="?")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--salts", nargs="*", default=[], help="explicit extra salts")
    ap.add_argument("--salt-set", choices=sorted(SALT_SETS), help="a fixed named salt set (CRN across candidates)")
    ap.add_argument("--random-salts", type=int, default=0, help="add N pseudo-random salts (see --seed)")
    ap.add_argument("--seed", type=int, default=None, help="seed for --random-salts (default: time-based)")
    ap.add_argument("--no-default-salt", action="store_true", help="skip the salt='' draw")
    ap.add_argument("--all-salted", action="store_true", help="repeat invsqrt/dfthcsos per salt too (they should be bit-stable)")
    ap.add_argument("--dqi-workers", type=int, default=12)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--repo", type=Path, default=None, help="main checkout with data/ and .venv/")
    ap.add_argument("--python", default=None, help="interpreter with kickmix (default: <repo>/.venv/bin/python)")
    ap.add_argument("--ledger", type=Path, default=ROOT / "build" / "cv" / "salt_ledger.jsonl")
    ap.add_argument("--raw-dir", type=Path, default=None, help="keep raw verifier JSON here (default build/cv/salt_raw/<tag>)")
    ap.add_argument("--summary", action="store_true", help="print the worst-case table of every summary in the ledger")
    a = ap.parse_args()

    if a.summary:
        rows = ledger_summaries(a.ledger)
        if not rows:
            print("no summaries yet")
            return 0
        rows.sort(key=lambda s: -(s.get("score_worst") or 0))
        print(f"{'tag':40} {'default':>10} {'mean':>10} {'WORST':>10} {'dqi fails':>9} {'dqi worst k':>11}")
        for s in rows:
            d = s["per_task"].get("dqi", {})
            ks = d.get("k_per_draw") or {}
            fmt = lambda x: "n/a" if x is None else f"{x:.5f}"
            print(f"{s['tag'][:40]:40} {fmt(s['score_default']):>10} {fmt(s['score_mean']):>10} {fmt(s['score_worst']):>10} "
                  f"{d.get('n_fail', '-')!s:>9} {max([v for v in ks.values() if v is not None], default='-')!s:>11}")
        return 0

    if a.portfolio is None:
        ap.error("portfolio required (or --summary)")
    repo = find_repo(a.repo)
    py = a.python or (str(repo / ".venv" / "bin" / "python") if (repo / ".venv" / "bin" / "python").exists() else sys.executable)

    salts: list[str | None] = [] if a.no_default_salt else [None]
    if a.salt_set:
        salts += SALT_SETS[a.salt_set]
    salts += a.salts
    if a.random_salts:
        rng = random.Random(a.seed if a.seed is not None else time.time_ns())
        salts += [f"r{rng.getrandbits(48):012x}" for _ in range(a.random_salts)]
    if not salts:
        ap.error("no salts: use --salt-set / --salts / --random-salts (or drop --no-default-salt)")

    with tempfile.TemporaryDirectory(prefix="saltcv_") as tmpdir:
        targets = resolve_targets(a.portfolio, a.tasks, Path(tmpdir))
        ids = {t: (sha256(p) if p.is_file() else sha256_dir(p)) for t, p in targets.items()}
        tag = a.tag or a.portfolio.stem
        raw_dir = a.raw_dir or (ROOT / "build" / "cv" / "salt_raw" / tag)
        a.ledger.parent.mkdir(parents=True, exist_ok=True)
        print(f"portfolio {tag}   repo={repo}   python={py}")
        for t in a.tasks:
            print(f"  {t:9} {ids[t][:12]}  {targets[t]}")
        print(f"  salts: {[s or '<default>' for s in salts]}")

        results: dict[str, list[dict]] = {t: [] for t in a.tasks}
        for t in a.tasks:
            draws = salts if (t == "dqi" or a.all_salted) else salts[:1]
            for s in draws:
                r = run_one(repo, py, t, targets[t], s, a.dqi_workers, raw_dir)
                r |= {"tag": tag, "file_sha256": ids[t], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                results[t].append(r)
                with open(a.ledger, "a") as fh:
                    fh.write(json.dumps(r, default=str) + "\n")
                extra = f" k={r['k']}" if t == "dqi" else ""
                err = f"  {r['report'].get('error', '')[:160]}" if r["status"] != "PASS" else ""
                print(f"  {t:9} salt={r['salt']:<16} {str(r['status']):6} cost={r['cost']!s:<20.20} "
                      f"ratio={r['ratio']!s:<12.12} wall={r['wall_s']}s{extra}{err}", flush=True)

    s = summarize(tag, ids, results, a.tasks)
    with open(a.ledger, "a") as fh:
        fh.write(json.dumps(s) + "\n")
    print_summary(s)
    return 0 if s["score_worst"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
