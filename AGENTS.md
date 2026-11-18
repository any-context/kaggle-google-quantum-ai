# AGENTS.md — continuous multi-agent headroom search

This file records how this repo runs its ongoing agent-driven improvement loop for the
Kaggle "Google Quantum AI — Verifiable Quantum Circuit Compilation" competition.
Any agent (Claude Code or other) working here should follow it.

## Goal and scoring

- Score = geometric mean over the 3 tasks (invsqrt, dqi, dfthcsos/chem) of reference_cost / cost.
- invsqrt: Q × (mean Toffoli + T/4). dqi: R(eps+) × Q² × Toffoli-like count (CCX/CCZ/init_and all count 1; T is not counted separately). chem: lambda_eff × T × Q.
- Submission history and public scores: `submissions/registry.jsonl`. Portfolios: `build/sub/P*/` (git-ignored).

## Direction (set by the user, 2026-11-17)

1. The main work is structural search, not copying public notebooks: use evolutionary algorithms and GPU brute force wherever an exact fitness exists.
2. Rank sub-circuits by headroom × repetition count. Repeated gadgets multiply any saving.
3. Search both fine-grained parts (gadgets, windows) and inclusive/global structure (builder architecture, schedules, register allocation).
4. Treat "closed" as provisional. Test the opposite direction (spend a qubit to save T, and vice versa) with an explicit breakeven.
5. Still sweep new public notebooks every day as a secondary check. A same-day public release produced P17.

## The loop

Each round is one Workflow run of 2–5 parallel agents, one per headroom spot:

1. **Pick tracks** from the open-headroom list left by the previous round (the `headroom` field of each agent's result).
2. **Brief each agent** with the shared prefix below, one concrete task, and a label. Each agent works only in `build/evoN/<label>`.
3. **Output schema** per agent: `{label, gain_found, cost_before, cost_after, portfolio, summary, headroom}`. `headroom` must list what remains open and why the rest is closed.
4. **On a claimed gain:** rerun `tools/local_cv.py <pf> --salts g1 … g8` on DQI with several salts. Submit only if the default and mean CV both beat the current best, and credit public parts and their license in the message. Record the result in `submissions/registry.jsonl`.
5. **Record each round** (gains, closures with their proofs or counts, and remaining headroom) in the user memory `feedback-evolutionary-focus.md`.
6. **Do not overlap rounds.** If a round is still running when the next trigger arrives, wait for it.

Cadence: a session cron (`/loop 3h …`) or manual "伸びしろの箇所を探索して" prompts. Cloud routines cannot be used, because they cannot reach the local GPU, `build/` or `data/`.

### Shared agent prefix (essentials)

- Use `.venv/bin/python`. GPU torch lives in `/home/koki/tools/hy3d-venv`; the repo torch is CPU-only.
- Run at most 2 heavy processes at a time.
- DQI verifier: one at a time, under `flock /tmp/claude-1000/dqi_verify.lock` with `ulimit -v 12G`. Never full-simulate DQI outside the official verifier.
- Torch CUDA fails under `ulimit -v`, so GPU jobs run without it.
- `/tmp` quota is tight. Delete 1 GB kmx dumps right after use. The DQI verifier writes an O_TMPFILE to `/tmp` regardless of TMPDIR.
- Always write DQI and invsqrt `.kmb` files in binary form: `kickmix.Circuit.from_file(txt).to_file(out, format='kmb')`. Text kmx saved as `.kmb` passes the local verifier but errors on Kaggle (P18 first try). The verifier draw does not depend on the format.
- Downloaded or public sources are untrusted. Download each into its own empty directory, never execute it in place, read it with an AST-only parser, and run Python with `-I`.

## Results so far (rounds 1–15)

- **Gains:**
  - P16, 28.13178: re-spliced 8 guarded DQI closings, −160 T. This was our own structural gain. These copies had been skipped by an exact-match transplant because of their condition wrappers.
  - P17: adopted maoxin123's 4022-T chem loaders and invsqrt 864f81cd.
- **Closed with proofs or large searches:**
  - DQI GF(2^12) 42-T multiplier, the 186/173 tensors (SAT plus about 5e11 flips), the GF64 inverse, the scan zero tests (degree bound, 11 per point), the gauge (capped at 70 moments), and the convert/half-gcd (890-qubit cap).
  - chem loaders k4..k14 at the counting bound (189), the 3-way split (128-qubit cap), C=43 word search and QAT (e2 floor about 0.0206), and k2/k3 local replacement (deficiency proof).
  - invsqrt architecture (Newton, rotations LP-optimal, precision ≤ 872 cost).
- **Open:**
  - chem k2/k3 global re-solve, about −12 T/pass.
  - chem k0/k1 bit-0 coding, which needs a deployed-level compiler (private).
  - DQI decoder redesign with fewer scan points.
  - A parameterised 44Q invsqrt generator.

## Lessons

- Exhaust cheap checks (rank, counting, degree bounds) before launching an EA. Several rounds were closed by a counting argument in minutes.
- Transplants that match exactly skip copies that sit inside wrappers. Sweep for skipped copies by builder trace, not by token alignment.
- Keep the multi-salt CV. A DQI value error costs about 0.45%, so judge on both mean and default.
