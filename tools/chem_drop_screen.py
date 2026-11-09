#!/usr/bin/env python3
"""Screen which ensemble copy c of the chemistry factorization could be dropped (C -> C-1).

The dfthcsos heuristic cost falls by ~78 Toffolis (-0.52 % of T_total) per copy removed from the
(R, B, C) weight tensor while the angle words -- hence the 15 loaders -- stay byte-identical.  The
question is whether the error_2 gate (20 mHa) is still reachable with one copy fewer.  For every c
this script deletes copy c, minimises error_2 alone over the remaining weights (torch L-BFGS on the
closed-form kernels of tools/chem_refit_gpu.py) and reports the reachable error_2 floor.  error_2
depends on w only, so this is a pure (quartic) least-squares problem; the lambda_eff side is then
settled by tools/chem_refit_gpu.py --drop-c on the survivors.

Usage:
    tools/chem_drop_screen.py --init DIR/submission.npz [--iters 400] [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chem_refit_gpu as C  # noqa: E402

torch.set_default_dtype(torch.float64)


def e2_floor(model, w_init, iters, lr=1.0):
    w = torch.as_tensor(w_init, device=model.device).clone().requires_grad_(True)
    if model.assignment is None:
        model.set_assignment(w.detach(), torch.zeros((w.shape[0], w.shape[2]), device=model.device))
    opt = torch.optim.LBFGS([w], lr=lr, max_iter=iters, history_size=100, tolerance_grad=1e-14,
                            tolerance_change=1e-16, line_search_fn="strong_wolfe")
    hist = []

    def closure():
        opt.zero_grad()
        m = model.metrics(w[None], torch.zeros((1,) + tuple(w.shape[:1]) + (w.shape[2],), device=model.device))
        loss = m["error_2"][0] ** 2
        loss.backward()
        hist.append(float(m["error_2"][0]))
        return loss

    opt.step(closure)
    with torch.no_grad():
        m = model.metrics(w[None], torch.zeros((1, w.shape[0], w.shape[2]), device=model.device))
    return float(m["error_2"][0]), len(hist), w.detach().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="screen only these copies")
    args = ap.parse_args()
    device = torch.device(args.device)
    cache_dir = C.ROOT / "build" / "chem_gpu"
    npz = np.load(args.init, allow_pickle=False)
    params = {k: np.asarray(npz[k]) for k in npz.files}
    w0 = params["W"].astype(np.float64)
    words, edges = params["angle_words"], params["tree_edges"]
    brot = int(np.ceil(np.log2(int(words.max()) + 1)))
    ref = C.load_ref(cache_dir)
    S, beta_1 = C.fixed_shift()
    R, B, Cc = w0.shape
    t0 = time.time()
    model = C.Model(ref, S, beta_1, words, edges, brot, (R, B, Cc), device, cache_dir)
    full, n, _ = e2_floor(model, w0, args.iters)
    print(f"C={Cc}: error_2 floor {full:.6f} after {n} evals ({time.time() - t0:.0f}s)", flush=True)
    # contribution of each copy: sum_r ||X_rc||_F^2 = sum_r w_rc^T P_rr w_rc
    P = model.P
    wt = torch.as_tensor(w0, device=device)
    contrib = torch.einsum("rbc,rbrd,rdc->c", wt, P, wt).cpu().numpy()
    order = np.argsort(contrib)
    cands = args.only if args.only is not None else order.tolist()
    rows = []
    for c in cands:
        t1 = time.time()
        w_drop = np.delete(w0, c, axis=2)
        m_drop = C.Model(ref, S, beta_1, words, edges, brot, (R, B, Cc - 1), device, cache_dir)
        floor, n, _ = e2_floor(m_drop, w_drop, args.iters)
        rows.append((c, float(contrib[c]), floor))
        print(f"drop c={c:2d} (|X|^2={contrib[c]:.4f}): error_2 floor {floor:.6f} "
              f"[{n} evals, {time.time() - t1:.0f}s]", flush=True)
    rows.sort(key=lambda r: r[2])
    print("best drops:", [(c, round(f, 6)) for c, _, f in rows[:8]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
