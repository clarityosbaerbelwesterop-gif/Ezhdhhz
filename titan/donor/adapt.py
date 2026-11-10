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
