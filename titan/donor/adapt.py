"""Donor MoE expert -> seed MoE slot.

Seed slot format (the seed team's accel/moe.py): per layer
    moe.experts.w1 / w3  [E, fe, d_s]      moe.experts.w2  [E, d_s, fe]
    moe.router.weight    [E, d_s]          moe.router.bias [E]  (buffer: steers top-k selection only)
    y = w2 (silu(w1 x) * (w3 x)),  x = rms_norm(residual) * ffn_norm.weight  (no bias terms anywhere)

Donor expert: y_d = down (silu(gate u) * (up u)), u = donor expert input (donor normalised residual).

The three kinds of operations are kept apart and recorded per tensor:

  COPY        donor payload bytes, unchanged (SHA-256 identical to the hub file range); BF16/FP8 values are
              widened to float32 exactly.
  PROJECTION  deterministic arithmetic with bridges P_in [d_s, d_d] (x @ P_in ~ u, no intercept) and
              P_out [d_d, d_s] (donor residual update @ P_out ~ seed residual update):
                  w1' = gate @ P_in'     w3' = up @ P_in'     w2' = P_out' @ down
              so that slot(x) == P_out-mapped donor expert applied to x @ P_in  (exact identity, tested);
              zero-padding of hidden units up to the slot width is exact; removing hidden units when the
              donor expert is wider than the slot is lossy and reported (kept importance share).
  HEAL        nothing here trains. Inserted slots are switched off through the selection bias, so the
              grown model equals the seed at step 0 (tested); making them useful is training.
"""
from __future__ import annotations

import numpy as np


def fold_expert(gate: np.ndarray, up: np.ndarray, down: np.ndarray, P_in: np.ndarray, P_out: np.ndarray):
    """gate/up [f, d_d], down [d_d, f]; P_in [d_s, d_d]; P_out [d_d, d_s] -> (w1 [f, d_s], w3 [f, d_s], w2 [d_s, f])."""
    Pi = P_in.astype(np.float64)
    w1 = gate.astype(np.float64) @ Pi.T
    w3 = up.astype(np.float64) @ Pi.T
    w2 = P_out.astype(np.float64).T @ down.astype(np.float64)
    return w1.astype(np.float32), w3.astype(np.float32), w2.astype(np.float32)


def unit_importance(w1: np.ndarray, w3: np.ndarray, w2: np.ndarray, act_abs_mean: np.ndarray | None = None) -> np.ndarray:
    """Importance of each hidden unit j for the expert's output: ||w2[:, j]|| times the unit's typical
    activation size (measured mean |silu(g_j) u_j| when available, else the product of its input-row norms)."""
    out_norm = np.linalg.norm(w2.astype(np.float64), axis=0)
    act = act_abs_mean.astype(np.float64) if act_abs_mean is not None else \
        np.linalg.norm(w1.astype(np.float64), axis=1) * np.linalg.norm(w3.astype(np.float64), axis=1)
    return out_norm * act


def fit_slot(w1, w3, w2, fe: int, importance: np.ndarray | None = None):
    """Bring an expert with f hidden units to a slot with fe units. f <= fe: zero padding (exact).
    f > fe: keep the fe most important units (lossy). Returns (w1, w3, w2, info)."""
    f = w1.shape[0]
    if f <= fe:
        z = np.zeros((fe - f, w1.shape[1]), np.float32)
        info = {"mode": "zero-pad (exact)" if f < fe else "same width (exact)", "donor_units": f, "slot_units": fe}
        return np.concatenate([w1, z]), np.concatenate([w3, z]), np.concatenate([w2, np.zeros((w2.shape[0], fe - f), np.float32)], 1), info
    imp = unit_importance(w1, w3, w2) if importance is None else np.asarray(importance, np.float64)
    keep = np.sort(np.argsort(-imp, kind="stable")[:fe])
    info = {"mode": "unit selection (lossy)", "donor_units": f, "slot_units": fe,
            "kept_importance_share": float(imp[keep].sum() / max(imp.sum(), 1e-30)), "kept_units": keep.tolist()}
    return w1[keep], w3[keep], w2[:, keep], info


def anchor_pair(seed_emb, donor_head_rows, rows_ok, ridge: float = 1e-2):
    """Data-free bridges between the two pre-head spaces, from the token tables on the seed vocabulary:
        Q [d_s, d_d]  seed pre-head vector -> donor pre-head vector   (H @ Q' ~ E)
        P [d_d, d_s]  donor pre-head update -> seed pre-head update   (P @ E' ~ H')
    fitted over the whole seed vocabulary; R^2 on held-out tokens (fit on `rows_ok` tokens only) is
    reported next to the fit R^2. Returns (Q, P, meta)."""
    from . import bridge
    Es, HB = np.asarray(seed_emb, np.float64), np.asarray(donor_head_rows, np.float64)
    Q_T, _, qi = bridge.fit_ridge(HB, Es, ridge, bias=False)
    W, _, pi = bridge.fit_ridge(Es, HB, ridge, bias=False)
    Q, P = Q_T.T.astype(np.float64), W.T.astype(np.float64)
    _, _, qh = bridge.input_anchor(HB, Es, rows_ok, ridge)
    _, ph = bridge.output_anchor(Es, HB, rows_ok, ridge)
    meta = {"Q_seed_to_donor_prehead": {**qi, "r2_holdout_tokens": qh["r2_holdout"], "sha256": bridge.sha256_array(Q.astype(np.float32))},
            "P_donor_to_seed_prehead": {**pi, "r2_holdout_tokens": ph["r2_holdout"], "sha256": bridge.sha256_array(P.astype(np.float32))}}
    return Q, P, meta


def slot_bridges(Q, P, seed_final_gain, donor_final_gain, donor_post_gain, donor_residual_rms: float,
                 slot_gain=1.0, seed_residual_rms: float = 1.0):
    """Move the pre-head anchors to an expert at donor layer L and a seed slot:
        P_in  = diag(g_seed / g_slot) @ Q @ diag(w_post(L) / w_final)
        P_out = diag(w_final) @ P @ diag(1 / g_seed) * (rms_seed / rms_donor(L))
    g_slot (gain of the seed layer's ffn_norm) and rms_seed are properties of the seed model at the slot's
    layer; without them the defaults 1.0 are plain scalars the caller has to report as missing."""
    gs, wd = np.asarray(seed_final_gain, np.float64), np.asarray(donor_final_gain, np.float64)
    P_in = (gs / slot_gain)[:, None] * Q * (np.asarray(donor_post_gain, np.float64) / wd)[None, :]
    P_out = wd[:, None] * P / gs[None, :] * seed_residual_rms / donor_residual_rms
    return P_in, P_out


def selection_error(act: np.ndarray, w2: np.ndarray, keep: np.ndarray) -> float:
    """Relative squared error of an expert's output when only the hidden units `keep` remain, on measured
    hidden activations act [n, f] (act = silu(w1 x) * (w3 x) on real routed inputs): ||Y - Y_keep||^2 / ||Y||^2."""
    a, w = act.astype(np.float64), w2.astype(np.float64)
    drop = np.setdiff1d(np.arange(a.shape[1]), keep)
    full = a @ w.T
    return float(((a[:, drop] @ w[:, drop].T) ** 2).sum() / max((full ** 2).sum(), 1e-30))


def project_router_row(gate_row: np.ndarray, P_in: np.ndarray) -> np.ndarray:
    """Donor router row g [d_d] -> seed-space row r [d_s] with x @ r ~ u @ g. An initialisation only: the
    donor routes with a softmax over its own experts, the seed with sigmoid scores and a selection bias."""
    return (P_in.astype(np.float64) @ gate_row.astype(np.float64)).astype(np.float32)


def swiglu(x, w1, w3, w2):
    g = x @ w1.T
    return ((g / (1.0 + np.exp(-g))) * (x @ w3.T)) @ w2.T


def insert_slots(experts: dict, new: list[dict], off_bias: float = -4.0) -> dict:
    """Append adapted experts to a layer's slot tensors. experts: {"w1" [E, fe, d], "w3", "w2" [E, d, fe],
    "router" [E, d], "bias" [E]}; new: [{"w1", "w3", "w2", "router"}]. New slots get selection bias
    ``off_bias``: sigmoid scores lie in (0, 1), so with off_bias <= -1 a new slot is never among the top-k
    while at least top_k old slots exist; the layer's output is unchanged (identity at insertion)."""
    if off_bias > -1.0:
        raise ValueError("off_bias must be <= -1 to keep new slots out of the top-k")
    out = {k: np.concatenate([experts[k]] + [n[k][None] for n in new]) for k in ("w1", "w3", "w2", "router")}
    out["bias"] = np.concatenate([experts["bias"], np.full(len(new), off_bias, np.float32)])
    return out


def seed_route(x, router, bias, top_k: int, route_scale: float):
    """The seed's routing rule (accel/moe.py::route) in NumPy."""
    s = 1.0 / (1.0 + np.exp(-np.clip(x @ router.T, -60.0, 60.0)))
    idx = np.argsort(-(s + bias[None, :]), axis=1, kind="stable")[:, :top_k]
    sel = np.take_along_axis(s, idx, 1)
    return idx, sel / sel.sum(1, keepdims=True) * route_scale


def seed_moe(x, experts: dict, top_k: int, route_scale: float):
    idx, gate = seed_route(x, experts["router"], experts["bias"], top_k, route_scale)
    y = np.zeros_like(x)
    for e in np.unique(idx):
        tok, slot = np.where(idx == e)
        y[tok] += swiglu(x[tok], experts["w1"][e], experts["w3"][e], experts["w2"][e]) * gate[tok, slot, None]
    return y, idx
