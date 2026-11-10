"""Bridges between a seed residual stream and a donor residual stream.

Row-vector convention throughout: y ~ x @ W (+ b), W [d_in, d_out].

  fit_ridge          closed-form ridge (optionally without intercept: seed MoE slots have no biases),
                     lambda = ridge * trace(X'X) / d (the scaling used by titan.transplant / the seed
                     team's stitch.py), R^2 on the fit set and on a held-out set
  anchor bridges     data-free initialisations from the token tables of both models on the seed
                     vocabulary (vocabulary-transplanted donor rows):
                       input anchor   seed embedding row  -> donor embedding row      (P_in  [d_s, d_d])
                       output anchor  donor logit effect  -> seed logit effect        (P_out [d_d, d_s]),
                                      min || P_out @ E_seed' - H_donor' ||, i.e. a donor residual update keeps
                                      its effect on the seed-vocabulary logits as far as the seed's
                                      unembedding can express it
                     An anchor bridge is an INITIALISATION at depth 0 / at the output; whether it holds at an
                     inner layer must be measured with paired activations of both models (needs the full
                     seed model).
"""
from __future__ import annotations

import hashlib

import numpy as np


def r2(pred: np.ndarray, target: np.ndarray, center: np.ndarray | None = None) -> float:
    c = target.mean(0) if center is None else center
    return float(1.0 - ((target - pred) ** 2).sum() / max(((target - c) ** 2).sum(), 1e-30))


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3, bias: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """y ~ x @ W + b. Returns (W float32 [d_in, d_out], b float32 [d_out] (zeros without bias), info)."""
    x = np.ascontiguousarray(x, np.float64)
    y = np.ascontiguousarray(y, np.float64)
    A = np.concatenate([x, np.ones((x.shape[0], 1))], 1) if bias else x
    ata = A.T @ A
    lam = float(ridge) * float(np.trace(ata)) / A.shape[1]
    Wb = np.linalg.solve(ata + lam * np.eye(A.shape[1]), A.T @ y)
    W, b = (Wb[:-1], Wb[-1]) if bias else (Wb, np.zeros(y.shape[1]))
    info = {"r2_fit": r2(x @ W + b, y), "ridge": float(ridge), "lambda": lam, "n": int(x.shape[0]),
            "d_in": int(x.shape[1]), "d_out": int(y.shape[1]), "bias": bool(bias)}
    return W.astype(np.float32), b.astype(np.float32), info


def evaluate(W, b, x, y, center=None) -> float:
    return r2(np.asarray(x, np.float64) @ np.asarray(W, np.float64) + np.asarray(b, np.float64), np.asarray(y, np.float64), center)


def split_tokens(n: int, holdout: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(n * holdout))
    return np.sort(perm[k:]), np.sort(perm[:k])


def input_anchor(seed_emb: np.ndarray, donor_rows: np.ndarray, rows_ok: np.ndarray, ridge: float = 1e-3,
                 bias: bool = False, holdout: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """P_in: seed embedding -> donor embedding, fitted on tokens in ``rows_ok`` (e.g. exact matches)."""
    idx = np.where(rows_ok)[0]
    fit, hold = split_tokens(len(idx), holdout, seed)
    W, b, info = fit_ridge(seed_emb[idx[fit]], donor_rows[idx[fit]], ridge, bias)
    info["r2_holdout"] = evaluate(W, b, seed_emb[idx[hold]], donor_rows[idx[hold]], donor_rows[idx[fit]].mean(0))
    info.update(kind="input-anchor (token embeddings, depth 0)", n_fit=int(len(fit)), n_holdout=int(len(hold)))
    return W, b, info


def output_anchor(seed_emb: np.ndarray, donor_head_rows: np.ndarray, rows_ok: np.ndarray, ridge: float = 1e-3,
                  holdout: float = 0.2, seed: int = 0) -> tuple[np.ndarray, dict]:
    """P_out [d_d, d_s] with P_out @ E_seed' ~ H_donor'  <=>  E_seed @ P_out' ~ H_donor  (per token row):
    a ridge fit from seed unembedding rows to donor head rows, transposed. No intercept (a map of residual
    updates). R^2 = share of the donor's seed-vocabulary logit directions the seed unembedding can express."""
    idx = np.where(rows_ok)[0]
    fit, hold = split_tokens(len(idx), holdout, seed)
    Wt, _, info = fit_ridge(seed_emb[idx[fit]], donor_head_rows[idx[fit]], ridge, bias=False)
    info["r2_holdout"] = evaluate(Wt, 0.0, seed_emb[idx[hold]], donor_head_rows[idx[hold]], donor_head_rows[idx[fit]].mean(0))
    info.update(kind="output-anchor (logit effect on the seed vocabulary)", n_fit=int(len(fit)), n_holdout=int(len(hold)))
    return np.ascontiguousarray(Wt.T), info


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def logit_effect_agreement(P_out: np.ndarray, seed_emb: np.ndarray, donor_head_rows: np.ndarray, n_dirs: int = 256,
                           seed: int = 0) -> dict:
    """For random donor residual updates d: compare donor logits d @ H' with seed logits (d @ P_out) @ E'
    (cosine over the vocabulary, and overlap of the 10 most-promoted tokens)."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n_dirs, donor_head_rows.shape[1])).astype(np.float64)
    a = d @ donor_head_rows.astype(np.float64).T
    b = (d @ P_out.astype(np.float64)) @ seed_emb.astype(np.float64).T
    a -= a.mean(1, keepdims=True)
    b -= b.mean(1, keepdims=True)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
    ta, tb = np.argsort(-a, 1)[:, :10], np.argsort(-b, 1)[:, :10]
    top = np.mean([len(set(x) & set(y)) / 10 for x, y in zip(ta, tb)])
    return {"logit_cosine_mean": float(cos.mean()), "logit_cosine_min": float(cos.min()), "top10_overlap_mean": float(top),
            "directions": n_dirs}
