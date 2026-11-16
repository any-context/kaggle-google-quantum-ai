#!/usr/bin/env python3
"""Fixed-word chemistry weight refit (our implementation of the public "refit W / H2_I" method).

The dfthcsos cost is lambda_eff * T * Q.  With the 15 ANGLE_k loaders fixed, T and Q are fixed and
the recovered angle words (hence the vectors u) are fixed; only the factorization weights
w (R,B,C) [stored as `W`] and W (R,C) [stored through `H2_I = W + sum_b w`] are free.  The verifier's
closed-form metrics (dfthcsos/scoreinfra/dfthc_math.py) are re-expressed here in torch so that
lambda_eff can be minimised by gradient descent under the verifier's gates:

    error_1 <= 0.020,  error_2 <= 0.020,  E_gap > 3.914898 Ha,  Lambda > 49.

lambda_eff = sqrt(E_gap * (2 Lambda - E_gap)) grows with both E_gap and Lambda in the feasible
region, so the optimum sits on the E_gap floor with the error gates active; we therefore minimise
Lambda with a penalty that keeps the gates satisfied, then rescale/round and re-check with the
*official* numpy evaluation before writing the npz.

Usage:
    tools/chem_refit.py --init DIR_WITH_dfthcsos --words build/chem/syed_words.npz --out build/chem/refit.npz
                        [--steps 2000] [--lr 1e-4]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "verifiers"))
from dfthcsos.verify import load_inputs, evaluate_words, E_GAP_FLOOR, LAMBDA_FLOOR  # noqa: E402
from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors  # noqa: E402

torch.set_default_dtype(torch.float64)
ERR_GATE = 0.020


class Model:
    def __init__(self, inputs, words, edges, brot):
        ref = inputs.ref
        N = inputs.num_orbitals
        self.N, self.eta = N, int(ref["electrons"])
        self.ecore, self.energy_upper = float(ref["ecore"]), float(inputs.energy_upper)
        self.beta_1 = float(inputs.beta_1)
        u_all = reconstruct_u_vectors(np.asarray(words) / (1 << (brot - 1)), edges)
        u_all = u_all / np.linalg.norm(u_all, axis=1, keepdims=True)
        R, B, C = inputs.w.shape
        self.one_body = torch.tensor(u_all[:N])                     # (N, N) rows
        u_b = u_all[N:].reshape(R, B, N)
        u_b = u_b / np.linalg.norm(u_b, axis=2, keepdims=True)
        self.u_b = torch.tensor(u_b)                                  # (R, B, N)
        oei, tei = np.asarray(ref["oei"], float), np.asarray(ref["tei"], float)
        h1_thc = oei - 0.5 * np.einsum("prrq->pq", tei)
        self.h1_thc = torch.tensor(h1_thc)
        S = np.asarray(inputs.S, float)
        self.S = torch.tensor(S)
        # error_2 target: tei + the two S shift terms on the diagonal pairs (fixed tensor).
        target = tei.copy()
        d = np.arange(N)
        target[:, :, d, d] += 0.5 * S[:, :, None]
        target[d, d, :, :] += 0.5 * S[None, :, :]
        self.h2_target = torch.tensor(target).reshape(N * N, N * N)
        # Optimise in the *stored* parametrisation (float32 `W` and `H2_I` in the npz), so that the
        # float32 rounding at save time cannot move the metrics off the gates.
        # Keep the npz's own dtypes (float64 for the public refit files): no re-rounding at save time.
        self.w_dtype = np.asarray(inputs.params["W"]).dtype
        self.h_dtype = np.asarray(inputs.params["H2_I"]).dtype
        self.w0 = torch.tensor(np.asarray(inputs.params["W"]).astype(np.float64))
        self.h0 = torch.tensor(np.asarray(inputs.params["H2_I"]).astype(np.float64))
        self.eye = torch.eye(N)

    def metrics(self, w, h, assignment=None):
        N = self.N
        W = h - w.sum(dim=1)                                                     # native W from stored H2_I
        X = torch.einsum("tjp,tjq,tjl->tlpq", self.u_b, self.u_b, w)            # (R, C, N, N)
        h1p = self.h1_thc - 0.5 * self.eta * self.S + self.beta_1 * self.eye - torch.einsum("rc,rcpq->pq", W, X)
        eig, vec = torch.linalg.eigh(h1p)
        lam_h1 = eig.abs().sum()
        # error_1: pair submitted one-body rows with eigenvectors (assignment fixed per call)
        if assignment is None:
            overlaps = (self.one_body @ vec).abs().detach().numpy()
            qi, ii = linear_sum_assignment(-overlaps)
            assignment = (torch.as_tensor(qi), torch.as_tensor(ii))
        qi, ii = assignment
        matched = torch.empty_like(eig)
        matched[qi] = eig[ii]
        h1p_qrom = torch.einsum("j,jp,jq->pq", matched, self.one_body, self.one_body)
        recon = h1p_qrom + 0.5 * self.eta * self.S - self.beta_1 * self.eye + torch.einsum("rc,rcpq->pq", W, X)
        error_1 = torch.linalg.norm(recon - self.h1_thc)
        Xf = X.reshape(-1, N * N)
        error_2 = torch.linalg.norm(Xf.T @ Xf - self.h2_target)
        w_B = W + w.sum(dim=1)
        lam_h2 = 0.25 * ((w.abs().sum(dim=1) + w_B.abs()) ** 2).sum()
        Lambda = lam_h1 + lam_h2
        E_SOS = self.ecore - self.eta * self.beta_1 - (lam_h1 - eig.sum()) - 0.5 * (W ** 2).sum()
        E_gap = self.energy_upper - E_SOS
        lam_eff = torch.sqrt(torch.clamp(E_gap * (2 * Lambda - E_gap), min=1e-12))
        return dict(error_1=error_1, error_2=error_2, Lambda=Lambda, E_gap=E_gap, lambda_eff=lam_eff), assignment


def official(inputs, words, edges, brot, w, h, w_dtype=np.float64, h_dtype=np.float64):
    """Evaluate with the verifier's own numpy code on the arrays exactly as they will be stored."""
    params = dict(inputs.params)
    params = {k: np.asarray(v) for k, v in params.items()}
    params["W"] = np.asarray(w, dtype=w_dtype)
    params["H2_I"] = np.asarray(h, dtype=h_dtype)
    inp2 = load_inputs(params)
    m = evaluate_words(inp2, words, edges, brot)
    return m, params



def flat_grad(out, params):
    gs = torch.autograd.grad(out, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


def unflat(vec, params):
    out, k = [], 0
    for p in params:
        n = p.numel()
        out.append(vec[k:k + n].reshape(p.shape))
        k += n
    return out


def projected_descent(model, w, h, *, steps, alpha, err_target, gap_target, log_every):
    """Minimise lambda_eff on the manifold {err1 = err_target, err2 = err_target, E_gap = gap_target}.

    Each iteration: gradients of the objective and of the active constraints; move along the
    objective gradient projected onto the constraints' null space; then restore the constraints
    with two Gauss-Newton corrections (same Jacobian).  Accept the step only if lambda_eff
    decreased and the constraints are within margin; otherwise halve alpha.
    """
    params = [w, h]
    assign = None
    best = None
    t0 = time.time()
    f1 = None
    for step in range(1, steps + 1):
        m, assign = model.metrics(w, h, assignment=None if step % 20 == 1 else assign)
        f0 = float(m["lambda_eff"].detach())
        feas0 = (float(m["error_1"]) <= err_target + 1e-11 and float(m["error_2"]) <= err_target + 1e-11
                 and float(m["E_gap"]) >= gap_target - 1e-9)
        cons = [m["error_1"] - err_target, m["error_2"] - err_target, gap_target - m["E_gap"]]
        g_f = flat_grad(m["lambda_eff"], params)
        J = torch.stack([flat_grad(c, params) for c in cons])            # (3, n)
        JJt = J @ J.T + 1e-18 * torch.eye(3)
        lam = torch.linalg.solve(JJt, J @ g_f)
        p = -(g_f - J.T @ lam)
        p = p / (p.norm() + 1e-30)
        theta0 = torch.cat([q.detach().reshape(-1) for q in params])
        accepted = False
        a = alpha
        mm = m
        for _ in range(6):
            theta = theta0 + a * p
            for _r in range(2):
                with torch.no_grad():
                    for q, v in zip(params, unflat(theta, params)):
                        q.copy_(v)
                    mm, _ = model.metrics(w, h, assignment=assign)
                rr = torch.stack([mm["error_1"] - err_target, mm["error_2"] - err_target, gap_target - mm["E_gap"]])
                theta = theta - J.T @ torch.linalg.solve(JJt, rr)
            with torch.no_grad():
                for q, v in zip(params, unflat(theta, params)):
                    q.copy_(v)
                mm, _ = model.metrics(w, h, assignment=assign)
            f1 = float(mm["lambda_eff"])
            feas = (float(mm["error_1"]) <= err_target + 1e-11 and float(mm["error_2"]) <= err_target + 1e-11
                    and float(mm["E_gap"]) >= gap_target - 1e-9)
            if feas and (f1 < f0 or not feas0):
                accepted = True
                break
            a *= 0.5
        if not accepted:
            with torch.no_grad():
                for q, v in zip(params, unflat(theta0, params)):
                    q.copy_(v)
            alpha *= 0.5
        else:
            alpha = min(alpha * 1.2, a * 2)
            best = (f1, w.detach().clone(), h.detach().clone())
        if step % log_every == 0 or step == 1:
            print(f"step {step:4d} lam_eff={(f1 if accepted else f0):.9f} accepted={accepted} alpha={alpha:.3e} "
                  f"err1={float(mm['error_1']):.11f} err2={float(mm['error_2']):.11f} E_gap={float(mm['E_gap']):.10f} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if alpha < 1e-12:
            print("alpha collapsed; stopping", flush=True)
            break
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=Path, required=True, help="dfthcsos dir whose submission.npz gives the start")
    ap.add_argument("--words", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-7)
    ap.add_argument("--gap-margin", type=float, default=5e-9, help="keep E_gap >= floor + margin")
    ap.add_argument("--err-margin", type=float, default=5e-10, help="keep errors <= gate - margin")
    ap.add_argument("--penalty", type=float, default=1e9)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--method", choices=("adam", "projected"), default="projected")
    ap.add_argument("--alpha", type=float, default=1e-3, help="projected step length (parameter-space norm)")
    a = ap.parse_args()

    inputs = load_inputs(a.init / "submission.npz")
    z = np.load(a.words)
    words, edges, brot = z["angle_words"], z["tree_edges"], int(z["brot"])
    m0 = evaluate_words(inputs, words, edges, brot)
    print("start (official):", {k: float(m0[k]) for k in ("error_1", "error_2", "Lambda", "E_gap", "lambda_eff")})

    model = Model(inputs, words, edges, brot)
    w = model.w0.clone().requires_grad_(True)
    W = model.h0.clone().requires_grad_(True)   # stored H2_I
    with torch.no_grad():
        m, assign = model.metrics(w, W)
    print("start (torch):   ", {k: float(v) for k, v in m.items()})
    gap_target = E_GAP_FLOOR + a.gap_margin
    err_target = ERR_GATE - a.err_margin
    best = (float(m["lambda_eff"]), w.detach().clone(), W.detach().clone())
    if a.method == "projected":
        pb = projected_descent(model, w, W, steps=a.steps, alpha=a.alpha, err_target=err_target,
                               gap_target=gap_target, log_every=a.log_every)
        if pb is not None:
            best = pb
    opt = torch.optim.Adam([w, W], lr=a.lr)
    t0 = time.time()
    for step in range(1, (a.steps if a.method == "adam" else 0) + 1):
        opt.zero_grad()
        m, assign = model.metrics(w, W, assignment=None if step % 50 == 1 else assign)
        # Objective: lambda_eff itself, plus quadratic penalties outside the (slightly shrunk) gates.
        viol = (torch.relu(m["error_1"] - err_target) ** 2 + torch.relu(m["error_2"] - err_target) ** 2
                + torch.relu(gap_target - m["E_gap"]) ** 2 + torch.relu(LAMBDA_FLOOR + 1 - m["Lambda"]) ** 2)
        loss = m["lambda_eff"] + a.penalty * viol
        loss.backward()
        opt.step()
        feasible = (m["error_1"] <= err_target and m["error_2"] <= err_target and m["E_gap"] >= gap_target)
        if feasible and float(m["lambda_eff"].detach()) < best[0]:
            best = (float(m["lambda_eff"]), w.detach().clone(), W.detach().clone())
        if step % a.log_every == 0 or step == 1:
            print(f"step {step:5d} lam_eff={float(m['lambda_eff']):.8f} err1={float(m['error_1']):.10f} "
                  f"err2={float(m['error_2']):.10f} E_gap={float(m['E_gap']):.10f} Lambda={float(m['Lambda']):.6f} "
                  f"viol={float(viol):.3e} best={best[0]:.8f} {time.time() - t0:.0f}s", flush=True)

    # Official re-check on the best feasible iterate after float32 storage rounding.
    lam_best, wb, Wb = best
    mo, params = official(inputs, words, edges, brot, wb.numpy(), Wb.numpy(), model.w_dtype, model.h_dtype)
    ok = (mo["error_1"] <= ERR_GATE and mo["error_2"] <= ERR_GATE and mo["E_gap"] > E_GAP_FLOOR
          and mo["Lambda"] > LAMBDA_FLOOR and mo["gap_in_domain"])
    print("best (torch) lambda_eff:", lam_best)
    print("best (official, f32-rounded):", {k: float(mo[k]) for k in ("error_1", "error_2", "Lambda", "E_gap", "lambda_eff")}, "gates ok:", ok)
    print("improvement vs start: %.6e (relative %.3e)" % (float(m0["lambda_eff"]) - float(mo["lambda_eff"]),
                                                          (float(m0["lambda_eff"]) - float(mo["lambda_eff"])) / float(m0["lambda_eff"])))
    if ok:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        orig = np.load(a.init / "submission.npz")
        save = {k: orig[k] for k in orig.files}
        save["W"] = params["W"]
        save["H2_I"] = params["H2_I"]
        np.savez(a.out, **save)
        print("wrote", a.out, {k: (v.shape, v.dtype) for k, v in save.items()})
    else:
        print("NOT written: best iterate fails a gate after rounding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
