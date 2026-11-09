#!/usr/bin/env python3
"""GPU fixed-word chemistry weight refit (population augmented-Lagrangian Adam in torch).

The dfthcsos cost is lambda_eff * T * Q.  With the 15 ANGLE_k loaders fixed, T, Q and the recovered
angle words (hence every vector u) are fixed; only the factorization weights w (R,B,C) [npz key `W`]
and the native offsets W (R,C) [npz key `H2_I` = W + sum_b w] are free.  The verifier gates are

    error_1 <= 0.020,  error_2 <= 0.020,  E_gap > 3.914898 Ha,  Lambda > 49.

Compared with tools/chem_refit.py (projected gradient on the N^4 tensors, ~2 s/step on CPU) this
version rewrites the metrics in closed form on precomputed kernels so that one evaluation is a few
hundred MFLOP:

    <X_rc, X_r'c'> = w_rc^T P_rr' w_r'c'      P[(r,b),(r',b')] = (u_rb . u_r'b')^2
    <G, Tt>        = sum_rc w_rc^T K_r w_rc    K_r[b,b'] = (u_rb u_rb | u_rb' u_rb')_Tt
    error_2^2      = sum (w P w)^2 - 2 sum w K w + ||Tt||^2
    error_1        = || sum_j matched_j q_j q_j^T - h1' ||_F         (q_j = fixed one-body rows)

and runs a *population* of perturbed copies in one batched graph on the GPU, each under an
augmented-Lagrangian penalty with its own learning rate / noise.  The best feasible member is
re-evaluated with the verifier's own numpy code (evaluate_dfthc_angle_words) before the npz is written.

Usage:
    tools/chem_refit_gpu.py --init DIR/submission.npz --out build/chem_gpu/refit.npz \
        [--pop 32] [--steps 20000] [--lr 1e-4] [--device cuda]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT
# A worktree keeps no data/; fall back to the main checkout.
for cand in (ROOT, *[p for p in ROOT.parents if (p / "data" / "verifiers").is_dir()]):
    if (cand / "data" / "verifiers").is_dir():
        MAIN = cand
        break
sys.path.insert(0, str(MAIN / "data" / "verifiers"))
from dfthcsos.scoreinfra.reference import (  # noqa: E402
    ORBITALS, UPPER_ENERGY, check_fcidump, load_fixed_shifts, load_reference,
)
from dfthcsos.scoreinfra.dfthc_math import (  # noqa: E402
    evaluate_dfthc_angle_words, reconstruct_u_vectors, symmetric_from_lower,
)

torch.set_default_dtype(torch.float64)
ERR_GATE = 0.020
E_GAP_FLOOR = 3.914898
LAMBDA_FLOOR = 49.0


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Reference data (cached: the FCIDUMP parse takes ~10 s)
# ---------------------------------------------------------------------------

def load_ref(cache_dir: Path) -> dict:
    cache = cache_dir / "ref_cache.npz"
    if cache.is_file():
        d = np.load(cache)
        return {"oei": d["oei"], "tei": d["tei"].astype(np.float64), "ecore": float(d["ecore"]),
                "electrons": int(d["electrons"]), "orbitals": int(d["orbitals"])}
    ref = load_reference(check_fcidump())
    cache_dir.mkdir(parents=True, exist_ok=True)
    # tei holds float32-rounded values: storing it as float32 is lossless.
    np.savez(cache, oei=ref["oei"], tei=np.asarray(ref["tei"], dtype=np.float32), ecore=ref["ecore"],
             electrons=ref["electrons"], orbitals=ref["orbitals"])
    return {"oei": ref["oei"], "tei": np.asarray(ref["tei"], dtype=np.float64), "ecore": float(ref["ecore"]),
            "electrons": int(ref["electrons"]), "orbitals": int(ref["orbitals"])}


def fixed_shift() -> tuple[np.ndarray, float]:
    fs = load_fixed_shifts()
    S = np.asarray(symmetric_from_lower(fs["Hsym_lower"], ORBITALS), dtype=np.float64)
    beta_1 = float(fs["H1_I"]) - 0.5 * float(np.trace(S))
    return S, beta_1


def official_metrics(words, edges, brot, w, W, S, beta_1, ref) -> dict:
    return evaluate_dfthc_angle_words(words, w, W, S, beta_1, ref, brot=brot, tree_edges=edges,
                                      energy_upper=UPPER_ENERGY)


# ---------------------------------------------------------------------------
# Closed-form model
# ---------------------------------------------------------------------------

class Model:
    def __init__(self, ref, S, beta_1, words, edges, brot, shape, device, cache_dir: Path):
        N = ref["orbitals"]
        R, B, C = shape
        self.N, self.R, self.B, self.C = N, R, B, C
        self.eta = ref["electrons"]
        self.ecore, self.energy_upper, self.beta_1 = ref["ecore"], UPPER_ENERGY, beta_1
        self.device = device
        u_all = reconstruct_u_vectors(np.asarray(words) / (1 << (brot - 1)), edges)
        u_all = u_all / np.linalg.norm(u_all, axis=1, keepdims=True)
        one_body = u_all[:N]
        u_b = u_all[N:].reshape(R, B, N)
        u_b = u_b / np.linalg.norm(u_b, axis=2, keepdims=True)
        oei, tei = np.asarray(ref["oei"], float), ref["tei"]
        h1_thc = oei - 0.5 * np.einsum("prrq->pq", tei)
        key = hashlib.sha256(np.ascontiguousarray(u_all).tobytes() + S.tobytes()).hexdigest()[:16]
        cache = cache_dir / f"kernels_{key}.npz"
        if cache.is_file():
            d = np.load(cache)
            P, K, c0 = d["P"], d["K"], float(d["c0"])
        else:
            t0 = time.time()
            P, K, c0 = self._kernels(u_b, tei, S, device)
            np.savez(cache, P=P, K=K, c0=c0)
            log(f"kernels computed in {time.time() - t0:.1f}s -> {cache}")
        T = lambda a: torch.as_tensor(np.asarray(a), device=device)
        self.P = T(P).reshape(R, B, R, B)
        self.K = T(K)                       # (R, B, B)
        self.c0 = c0
        self.u_b = T(u_b)
        self.one_body = T(one_body)
        self.h1_base = T(h1_thc - 0.5 * self.eta * S + beta_1 * np.eye(N))
        self.S, self.h1_thc = S, h1_thc
        self.assignment = None

    @staticmethod
    def _kernels(u_b, tei, S, device):
        R, B, N = u_b.shape
        U = torch.as_tensor(u_b.reshape(R * B, N), device=device)
        P = ((U @ U.T) ** 2).cpu().numpy()
        target = np.array(tei, dtype=np.float64, copy=True)
        d = np.arange(N)
        target[:, :, d, d] += 0.5 * S[:, :, None]
        target[d, d, :, :] += 0.5 * S[None, :, :]
        Tt = torch.as_tensor(target.reshape(N * N, N * N), device=device)
        c0 = float((Tt * Tt).sum())
        K = np.empty((R, B, B))
        ub = torch.as_tensor(u_b, device=device)
        for r in range(R):
            V = torch.einsum("bp,bq->bpq", ub[r], ub[r]).reshape(B, N * N)
            K[r] = (V @ Tt @ V.T).cpu().numpy()
        del Tt
        return P, K, c0

    def set_assignment(self, w, W):
        """Pair the fixed one-body rows with eigenvectors of h1' (as the verifier does, once)."""
        with torch.no_grad():
            h1p = self._h1p(w[None], W[None])[0]
            _, vec = torch.linalg.eigh(h1p)
            overlaps = (self.one_body @ vec).abs().cpu().numpy()
            qi, ii = linear_sum_assignment(-overlaps)
        # matched[qi] = eig[ii]  ->  for row j (=qi sorted), eigen index perm[j]
        perm = np.empty(self.N, dtype=np.int64)
        perm[qi] = ii
        self.assignment = torch.as_tensor(perm, device=self.device)

    def _h1p(self, w, W):
        # sum_rc W_rc X_rc = sum_rb (sum_c W_rc w_rcb) u_rb u_rb^T
        coef = torch.einsum("krc,krbc->krb", W, w)
        return self.h1_base[None] - torch.einsum("krb,rbp,rbq->kpq", coef, self.u_b, self.u_b)

    def metrics(self, w, W):
        """Batched: w (k,R,B,C), W (k,R,C) -> dict of (k,) tensors."""
        R, B, C = self.R, self.B, self.C
        h1p = self._h1p(w, W)
        eig, vec = torch.linalg.eigh(h1p)                                   # (k,N), (k,N,N)
        lam_h1 = eig.abs().sum(-1)
        matched = eig[:, self.assignment]                                   # (k,N) eigenvalue of row j
        h1p_qrom = torch.einsum("kj,jp,jq->kpq", matched, self.one_body, self.one_body)
        error_1 = torch.linalg.norm((h1p_qrom - h1p).reshape(w.shape[0], -1), dim=-1)
        # error_2 from the kernels
        Z = torch.einsum("krbc,rbsd->krcsd", w, self.P)                     # (k,R,C,R,B)
        Q = torch.einsum("krcsd,ksde->krcse", Z, w)                         # <X_rc, X_se>
        gram = (Q * Q).sum(dim=(1, 2, 3, 4))
        cross = torch.einsum("krbc,rbd,krdc->k", w, self.K, w)
        e2sq = gram - 2.0 * cross + self.c0
        error_2 = torch.sqrt(torch.clamp(e2sq, min=0.0))
        w_B = W + w.sum(dim=2)
        lam_h2 = 0.25 * ((w.abs().sum(dim=2) + w_B.abs()) ** 2).sum(dim=(1, 2))
        Lambda = lam_h1 + lam_h2
        E_SOS = self.ecore - self.eta * self.beta_1 - (lam_h1 - eig.sum(-1)) - 0.5 * (W ** 2).sum(dim=(1, 2))
        E_gap = self.energy_upper - E_SOS
        lam_eff = torch.sqrt(torch.clamp(E_gap * (2 * Lambda - E_gap), min=1e-12))
        return dict(error_1=error_1, error_2=error_2, Lambda=Lambda, E_gap=E_gap, lambda_eff=lam_eff)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

def al_loss(m, mu, rho, err_t, gap_t):
    """Augmented Lagrangian for g = (e1 - t, e2 - t, gap_t - E_gap) <= 0 (per member)."""
    g = torch.stack([m["error_1"] - err_t, m["error_2"] - err_t, gap_t - m["E_gap"]], dim=-1)   # (k,3)
    pen = (torch.clamp(mu + rho * g, min=0.0) ** 2 - mu ** 2) / (2 * rho)
    return m["lambda_eff"] + pen.sum(-1), g


def feasible(m, err_t, gap_t, tol=0.0):
    return (m["error_1"] <= err_t + tol) & (m["error_2"] <= err_t + tol) & (m["E_gap"] >= gap_t - tol) \
        & (m["Lambda"] > LAMBDA_FLOOR)


def run_alm_lbfgs(model, wt0, Wt0, off0, args, err_t, gap_t):
    """Augmented Lagrangian, inner problem solved by L-BFGS-B (scipy) on the split variables.

    theta = [a, b, W] with a, b >= 0 and w = a - b.  Function and gradient come from the batched
    torch model (k = 1) on the device; scipy only sees float64 numpy vectors.
    """
    from scipy.optimize import minimize
    device = wt0.device
    n_w, n_W = wt0.numel(), Wt0.numel()
    a0 = torch.clamp(wt0, min=0).reshape(-1).cpu().numpy()
    b0 = torch.clamp(-wt0, min=0).reshape(-1).cpu().numpy()
    theta = np.concatenate([a0, b0, Wt0.reshape(-1).cpu().numpy()])
    if args.perturb > 0:
        rng = np.random.default_rng(args.seed)
        wflat = wt0.reshape(-1).cpu().numpy()
        d = args.perturb * rng.standard_normal(n_w) * np.abs(wflat)
        wp = wflat + d
        theta[:n_w] = np.clip(wp, 0, None)
        theta[n_w:2 * n_w] = np.clip(-wp, 0, None)
    bounds = [(0.0, None)] * (2 * n_w) + [(None, None)] * n_W
    mu = np.zeros(3)
    rho = np.array([args.rho_err, args.rho_err, args.rho_gap])
    state = {"n": 0}

    def unpack(th):
        t = torch.as_tensor(th, device=device)
        a = t[:n_w].reshape(wt0.shape)
        b = t[n_w:2 * n_w].reshape(wt0.shape)
        W = t[2 * n_w:].reshape(Wt0.shape)
        return a, b, W

    def metrics_np(th):
        with torch.no_grad():
            a, b, W = unpack(th)
            m = model.metrics((a - b)[None], W[None])
        return {k: float(v[0]) for k, v in m.items()}

    def fg(th):
        state["n"] += 1
        a, b, W = unpack(th)
        a.requires_grad_(True); b.requires_grad_(True); W.requires_grad_(True)
        m = model.metrics((a - b)[None], W[None])
        mu_t = torch.as_tensor(mu, device=device)[None]
        rho_t = torch.as_tensor(rho, device=device)[None]
        loss, _ = al_loss(m, mu_t, rho_t, err_t, gap_t)
        loss = loss[0]
        ga, gb, gW = torch.autograd.grad(loss, (a, b, W))
        g = torch.cat([ga.reshape(-1), gb.reshape(-1), gW.reshape(-1)]).cpu().numpy()
        return float(loss.detach()), g

    def viol(m):
        return np.array([m["error_1"] - err_t, m["error_2"] - err_t, gap_t - m["E_gap"]])

    # Multiplier warm start: least-squares KKT estimate at theta0 (mu >= 0).
    m = metrics_np(theta)
    log(f"alm start: lam={m['lambda_eff']:.9f} viol={viol(m)}")
    t0 = time.time()
    best = (float("inf"), theta.copy())
    prev_v = None
    for outer in range(1, args.outer + 1):
        res = minimize(fg, theta, jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=args.inner, maxcor=args.maxcor, ftol=1e-16, gtol=1e-12, maxls=50))
        theta = res.x
        m = metrics_np(theta)
        v = viol(m)
        vmax = float(np.max(v))
        log(f"outer {outer:3d} inner_it={res.nit} nfev={state['n']} lam={m['lambda_eff']:.9f} Lambda={m['Lambda']:.6f} "
            f"viol=[{v[0]:.2e},{v[1]:.2e},{v[2]:.2e}] mu={np.round(mu, 3)} rho={rho} {time.time() - t0:.0f}s")
        mu = np.maximum(0.0, mu + rho * v)
        if prev_v is not None:
            worse = np.maximum(v, 0) > 0.25 * np.maximum(prev_v, 0)
            rho = np.where(worse & (np.maximum(v, 0) > 1e-10), rho * 4, rho)
        prev_v = v
        # restoration: Gauss-Newton onto the targets, then record if feasible
        th_r = restore(theta, unpack, model, err_t, gap_t, device, wt0, Wt0, n_w)
        mr = metrics_np(th_r)
        vr = viol(mr)
        # err_t / gap_t already sit args.*_margin inside the official gates; accept half that slack here
        tol = 0.5 * min(args.err_margin, args.gap_margin)
        log(f"   restored: lam={mr['lambda_eff']:.9f} viol=[{vr[0]:.2e},{vr[1]:.2e},{vr[2]:.2e}]")
        if np.all(vr <= tol) and mr["lambda_eff"] < best[0]:
            best = (mr["lambda_eff"], th_r.copy())
            log(f"   new best feasible {mr['lambda_eff']:.9f}")
        if vmax <= 1e-10 and res.nit < 5:
            break
    a, b, W = unpack(best[1])
    return best[0], (a - b).cpu().numpy(), W.cpu().numpy()


def restore(theta, unpack, model, err_t, gap_t, device, wt0, Wt0, n_w, iters=6):
    """A few Gauss-Newton steps onto {e1 = err_t - d, e2 = err_t - d, E_gap = gap_t + d} (d: tiny slack)."""
    th = torch.as_tensor(theta, device=device).clone()
    slack = 0.0
    for _ in range(iters):
        th.requires_grad_(True)
        a = th[:n_w].reshape(wt0.shape); b = th[n_w:2 * n_w].reshape(wt0.shape); W = th[2 * n_w:].reshape(Wt0.shape)
        m = model.metrics((a - b)[None], W[None])
        r = torch.stack([m["error_1"][0] - (err_t - slack), m["error_2"][0] - (err_t - slack), (gap_t + slack) - m["E_gap"][0]])
        # only move constraints that are violated or within the slack band
        J = torch.stack([torch.autograd.grad(r[i], th, retain_graph=True)[0] for i in range(3)])
        with torch.no_grad():
            # respect the bounds: zero columns of a/b that sit at 0 and would go negative is overkill; clamp after
            step = J.T @ torch.linalg.solve(J @ J.T + 1e-20 * torch.eye(3, device=device), r)
            th = th.detach() - step
            th[:2 * n_w].clamp_(min=0)
    return th.detach().cpu().numpy()


def run(args) -> int:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cache_dir = ROOT / "build" / "chem_gpu"
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = np.load(args.init, allow_pickle=False)
    params = {k: np.asarray(npz[k]) for k in npz.files}
    w0 = params["W"].astype(np.float64)
    h0 = params["H2_I"].astype(np.float64)
    if args.drop_c:
        # Remove ensemble copies (C -> C - len): the words and loaders stay, only w / H2_I shrink.
        w0 = np.delete(w0, args.drop_c, axis=2)
        h0 = np.delete(h0, args.drop_c, axis=1)
        params["W"], params["H2_I"] = w0.astype(params["W"].dtype), h0.astype(params["H2_I"].dtype)
    W0 = h0 - w0.sum(axis=1)
    words = params["angle_words"] if args.words is None else np.load(args.words)["angle_words"]
    edges = params["tree_edges"]
    brot = args.brot or int(math.ceil(math.log2(int(words.max()) + 1)))
    ref = load_ref(cache_dir)
    S, beta_1 = fixed_shift()
    R, B, C = w0.shape
    log(f"device={device} shape R,B,C={R},{B},{C} brot={brot} words={words.shape}")

    model = Model(ref, S, beta_1, words, edges, brot, (R, B, C), device, cache_dir)
    wt0 = torch.as_tensor(w0, device=device)
    Wt0 = torch.as_tensor(W0, device=device)
    model.set_assignment(wt0, Wt0)
    with torch.no_grad():
        m0 = {k: float(v[0]) for k, v in model.metrics(wt0[None], Wt0[None]).items()}
    off0 = official_metrics(words, edges, brot, w0, W0, S, beta_1, ref)
    log("start model   :", json.dumps({k: round(v, 12) for k, v in m0.items()}))
    log("start official:", json.dumps({k: (round(float(v), 12) if not isinstance(v, bool) else v) for k, v in off0.items()}))
    for k in ("error_1", "error_2", "Lambda", "E_gap", "lambda_eff"):
        if abs(m0[k] - float(off0[k])) > 1e-7 * max(1.0, abs(float(off0[k]))):
            log(f"MODEL MISMATCH on {k}: {m0[k]} vs {off0[k]}")
            return 2
    if args.check_only:
        return 0

    err_t = ERR_GATE - args.err_margin
    gap_t = E_GAP_FLOOR + args.gap_margin
    if args.method == "lbfgs":
        val, w_i, W_i = run_alm_lbfgs(model, wt0, Wt0, off0, args, err_t, gap_t)
        return finish(args, params, words, edges, brot, S, beta_1, ref, off0, [(val, w_i, W_i)])
    k = args.pop
    g = torch.Generator(device=device).manual_seed(args.seed)
    # Split parametrisation keeps |w| exact at the kinks: w = a - b, a,b >= 0.
    a = torch.clamp(wt0, min=0)[None].repeat(k, 1, 1, 1)
    b = torch.clamp(-wt0, min=0)[None].repeat(k, 1, 1, 1)
    W = Wt0[None].repeat(k, 1, 1).clone()
    # member 0 is the unperturbed start; the others get relative noise of increasing size
    if k > 1:
        scale = torch.logspace(math.log10(args.noise_min), math.log10(args.noise_max), k - 1, device=device)
        nz = torch.randn((k - 1,) + tuple(wt0.shape), generator=g, device=device)
        a[1:] = torch.clamp(a[1:] + scale[:, None, None, None] * nz * wt0.abs()[None], min=0)
        b[1:] = torch.clamp(b[1:] - scale[:, None, None, None] * nz * wt0.abs()[None], min=0)
    a.requires_grad_(True); b.requires_grad_(True); W.requires_grad_(True)
    lrs = torch.logspace(math.log10(args.lr_min), math.log10(args.lr_max), k, device=device) if k > 1 \
        else torch.full((1,), args.lr, device=device)
    # Adam with per-member learning rate: implement by scaling the parameter groups via a hook-free trick:
    # use a single optimiser at lr=1 and multiply the update by lrs through grad scaling is wrong for Adam
    # (scale invariant), so keep separate state tensors manually.
    opt = AdamPerMember([a, b, W], lrs, betas=(0.9, 0.999), eps=1e-12)
    mu = torch.zeros(k, 3, device=device)
    rho = torch.tensor([args.rho_err, args.rho_err, args.rho_gap], device=device)[None].repeat(k, 1)
    best_val = torch.full((k,), float("inf"), device=device)
    best_a, best_b, best_W = a.detach().clone(), b.detach().clone(), W.detach().clone()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        w = a - b
        m = model.metrics(w, W)
        loss, gcon = al_loss(m, mu, rho, err_t, gap_t)
        with torch.no_grad():
            feas = feasible(m, err_t, gap_t)
            val = torch.where(feas, m["lambda_eff"].detach(), torch.full_like(best_val, float("inf")))
            improved = val < best_val
            if improved.any():
                best_val = torch.where(improved, val, best_val)
                best_a[improved] = a.detach()[improved]
                best_b[improved] = b.detach()[improved]
                best_W[improved] = W.detach()[improved]
        opt.zero_grad()
        loss.sum().backward()
        opt.step(step)
        with torch.no_grad():
            a.clamp_(min=0); b.clamp_(min=0)
            ov = torch.minimum(a, b)          # both parts positive is never optimal
            a.sub_(ov); b.sub_(ov)
            if step % args.mu_every == 0:
                mu.copy_(torch.clamp(mu + rho * gcon.detach(), min=0.0))
            if step % args.log_every == 0 or step == 1:
                i = int(torch.argmin(val))
                log(f"step {step:6d} best_feas={float(best_val.min()):.9f} now[{i}] lam={float(m['lambda_eff'][i]):.9f} "
                    f"e1={float(m['error_1'][i]):.10f} e2={float(m['error_2'][i]):.10f} gap={float(m['E_gap'][i]):.9f} "
                    f"feas={int(feas.sum())}/{k} mu={[round(float(x), 3) for x in mu[i]]} {time.time() - t0:.0f}s")

    order = torch.argsort(best_val).tolist()
    cands = [(float(best_val[i]), (best_a[i] - best_b[i]).cpu().numpy(), best_W[i].cpu().numpy())
             for i in order[: args.n_official] if math.isfinite(float(best_val[i]))]
    return finish(args, params, words, edges, brot, S, beta_1, ref, off0, cands)


def finish(args, params, words, edges, brot, S, beta_1, ref, off0, cands) -> int:
    """Official numpy re-evaluation of candidate (model_value, w, W); write the first that passes."""
    for i, (val, w_i, W_i) in enumerate(cands):
        off = official_metrics(words, edges, brot, w_i, W_i, S, beta_1, ref)
        ok = (off["error_1"] <= ERR_GATE and off["error_2"] <= ERR_GATE and off["E_gap"] > E_GAP_FLOOR
              and off["Lambda"] > LAMBDA_FLOOR and off["gap_in_domain"])
        log(f"candidate {i}: model {val:.9f} official {off['lambda_eff']:.9f} gates_ok={ok} "
            f"e1={off['error_1']:.12f} e2={off['error_2']:.12f} gap={off['E_gap']:.12f}")
        if ok and off["lambda_eff"] < float(off0["lambda_eff"]):
            out = dict(params)
            out["W"] = w_i.astype(params["W"].dtype)
            out["H2_I"] = (W_i + w_i.sum(axis=1)).astype(params["H2_I"].dtype)
            off2 = official_metrics(words, edges, brot, out["W"].astype(np.float64),
                                    out["H2_I"].astype(np.float64) - out["W"].astype(np.float64).sum(axis=1),
                                    S, beta_1, ref)
            ok2 = (off2["error_1"] <= ERR_GATE and off2["error_2"] <= ERR_GATE and off2["E_gap"] > E_GAP_FLOOR
                   and off2["Lambda"] > LAMBDA_FLOOR and off2["gap_in_domain"])
            if not ok2:
                log("  stored-dtype re-check failed; skipping")
                continue
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.out, **out)
            rel = off2["lambda_eff"] / float(off0["lambda_eff"]) - 1
            log(f"wrote {args.out}: lambda_eff {off0['lambda_eff']:.9f} -> {off2['lambda_eff']:.9f} ({rel:+.3e})")
            return 0
    log("no candidate beat the start under the official gates")
    return 1


class AdamPerMember:
    """Adam whose learning rate differs per leading (population) index."""

    def __init__(self, params, lrs, betas=(0.9, 0.999), eps=1e-12):
        self.params = params
        self.lrs = lrs
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, t):
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            mhat = m / (1 - self.b1 ** t)
            vhat = v / (1 - self.b2 ** t)
            lr = self.lrs.reshape((-1,) + (1,) * (p.dim() - 1))
            p.sub_(lr * mhat / (vhat.sqrt() + self.eps))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", type=Path, required=True, help="submission.npz to start from")
    ap.add_argument("--words", type=Path, default=None, help="npz with angle_words (default: from --init)")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "chem_gpu" / "refit.npz")
    ap.add_argument("--brot", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--lr-max", type=float, default=1e-4)
    ap.add_argument("--noise-min", type=float, default=1e-4)
    ap.add_argument("--noise-max", type=float, default=1e-1)
    ap.add_argument("--rho-err", type=float, default=1e6)
    ap.add_argument("--rho-gap", type=float, default=1e4)
    ap.add_argument("--mu-every", type=int, default=50)
    ap.add_argument("--err-margin", type=float, default=2e-9)
    ap.add_argument("--gap-margin", type=float, default=2e-9)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--n-official", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=("adam", "lbfgs"), default="lbfgs")
    ap.add_argument("--outer", type=int, default=30)
    ap.add_argument("--inner", type=int, default=500)
    ap.add_argument("--maxcor", type=int, default=50)
    ap.add_argument("--perturb", type=float, default=0.0, help="relative gaussian perturbation of w at start")
    ap.add_argument("--drop-c", type=int, nargs="*", default=None, help="ensemble copies to delete before refitting")
    return run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
