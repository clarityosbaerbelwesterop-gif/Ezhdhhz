"""Qwen3-MoE donor forward, one layer at a time (weights streamed by HTTP range, never a whole model in
memory). Semantics follow transformers' Qwen3MoeDecoderLayer exactly (tested against it on random
weights): pre-norm; attention with RMSNorm on the q/k head vectors before RoPE (rotate-half convention),
GQA; MoE router = softmax over all experts, top-k, renormalised; experts = down(silu(gate x) * up x).

Tokens are kept flat ([N, d] with a sequence id and a position); attention pads per batch, the MoE runs
on valid tokens only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QwenMoeConfig:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    experts: int
    top_k: int
    moe_inter: int
    eps: float
    rope_theta: float
    vocab: int
    norm_topk_prob: bool = True

    @classmethod
    def from_hf(cls, c: dict) -> "QwenMoeConfig":
        return cls(c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"],
                   c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"], c["num_experts"],
                   c["num_experts_per_tok"], c["moe_intermediate_size"], c["rms_norm_eps"], float(c["rope_theta"]),
                   c["vocab_size"], bool(c.get("norm_topk_prob", True)))


LAYER_KEYS = {"in_ln": "input_layernorm.weight", "post_ln": "post_attention_layernorm.weight",
              "q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight",
              "o": "self_attn.o_proj.weight", "qn": "self_attn.q_norm.weight", "kn": "self_attn.k_norm.weight",
              "gate": "mlp.gate.weight"}
EXPERT_KEYS = {"e_gate": "gate_proj.weight", "e_up": "up_proj.weight", "e_down": "down_proj.weight"}


def layer_tensor_names(layer: int, experts: int) -> dict:
    pre = f"model.layers.{layer}."
    out = {k: pre + v for k, v in LAYER_KEYS.items()}
    for e in range(experts):
        for k, v in EXPERT_KEYS.items():
            out[f"{k}.{e}"] = f"{pre}mlp.experts.{e}.{v}"
    return out


def rms_norm(x, w, eps: float):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_tables(T: int, head_dim: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    fr = torch.arange(T, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([fr, fr], -1)
    return emb.cos(), emb.sin()


def _rot(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin


def attention(h, W: dict, cfg: QwenMoeConfig, cos, sin):
    """h [B, T, d] (padded at the end; causal, so padding never reaches valid positions)."""
    B, T, d = h.shape
    H, KV, hd = cfg.heads, cfg.kv_heads, cfg.head_dim
    xn = rms_norm(h, W["in_ln"], cfg.eps)
    q = rms_norm(F.linear(xn, W["q"]).view(B, T, H, hd), W["qn"], cfg.eps).transpose(1, 2)
    k = rms_norm(F.linear(xn, W["k"]).view(B, T, KV, hd), W["kn"], cfg.eps).transpose(1, 2)
    v = F.linear(xn, W["v"]).view(B, T, KV, hd).transpose(1, 2)
    c, s = cos[None, None, :T], sin[None, None, :T]
    q, k = _rot(q, c, s), _rot(k, c, s)
    rep = H // KV
    k = k[:, :, None].expand(B, KV, rep, T, hd).reshape(B, H, T, hd)
    v = v[:, :, None].expand(B, KV, rep, T, hd).reshape(B, H, T, hd)
    sc = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(hd)
    sc = sc + torch.triu(torch.full((T, T), float("-inf")), 1)
    o = torch.matmul(torch.softmax(sc, -1), v).transpose(1, 2).reshape(B, T, H * hd)
    return h + F.linear(o, W["o"])


def route(xn, W: dict, cfg: QwenMoeConfig):
    p = torch.softmax(F.linear(xn, W["gate"]).float(), -1)
    topv, topi = torch.topk(p, cfg.top_k, -1)
    if cfg.norm_topk_prob:
        topv = topv / topv.sum(-1, keepdim=True)
    return topi, topv


def moe(h, W: dict, cfg: QwenMoeConfig, groups=None, n_groups: int = 0):
    """h [N, d] (valid tokens only). Returns (h_out, info) where info holds the routing table and, when
    ``groups`` [N] (int group id per token, -1 = ignore) is given, per-expert statistics per group:
    count, gate mass, and energy sum ||gate * y_e||^2."""
    N, d = h.shape
    xn = rms_norm(h, W["post_ln"], cfg.eps)
    topi, topw = route(xn, W, cfg)
    out = torch.zeros_like(h)
    E = cfg.experts
    stats = None
    if groups is not None:
        stats = {"count": torch.zeros(n_groups, E, dtype=torch.float64), "mass": torch.zeros(n_groups, E, dtype=torch.float64),
                 "energy": torch.zeros(n_groups, E, dtype=torch.float64)}
    flat_e = topi.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    tok_of = torch.arange(N).repeat_interleave(cfg.top_k)[order]
    w_of = topw.reshape(-1)[order]
    bounds = torch.searchsorted(flat_e[order], torch.arange(E + 1))
    for e in range(E):
        a, b = int(bounds[e]), int(bounds[e + 1])
        if a == b:
            continue
        tok = tok_of[a:b]
        x = xn[tok]
        y = F.linear(F.silu(F.linear(x, W["e_gate"][e])) * F.linear(x, W["e_up"][e]), W["e_down"][e]) * w_of[a:b, None]
        out.index_add_(0, tok, y)
        if stats is not None:
            g = groups[tok]
            ok = g >= 0
            if ok.any():
                gi = g[ok]
                stats["count"][:, e] += torch.bincount(gi, minlength=n_groups).double()
                stats["mass"][:, e] += torch.bincount(gi, weights=w_of[a:b][ok].double(), minlength=n_groups)
                stats["energy"][:, e] += torch.bincount(gi, weights=y[ok].pow(2).sum(-1).double(), minlength=n_groups)
    info = {"topi": topi, "topw": topw, "stats": stats, "xn": xn,
            "out_energy": out.pow(2).sum(-1), "in_energy": h.pow(2).sum(-1)}
    return h + out, info


def unpack_layer(blobs: dict, names: dict, cfg: QwenMoeConfig, dtype=torch.float32) -> dict:
    """{logical key: BF16 payload bytes} -> tensors ([E, f, d] / [E, d, f] for the experts)."""
    def t(key, shape):
        return torch.frombuffer(bytearray(blobs[names[key]]), dtype=torch.bfloat16).reshape(shape).to(dtype)
    d, hd, f, E = cfg.hidden, cfg.head_dim, cfg.moe_inter, cfg.experts
    W = {"in_ln": t("in_ln", (d,)), "post_ln": t("post_ln", (d,)), "q": t("q", (cfg.heads * hd, d)),
         "k": t("k", (cfg.kv_heads * hd, d)), "v": t("v", (cfg.kv_heads * hd, d)), "o": t("o", (d, cfg.heads * hd)),
         "qn": t("qn", (hd,)), "kn": t("kn", (hd,)), "gate": t("gate", (E, d))}
    W["e_gate"] = torch.stack([t(f"e_gate.{e}", (f, d)) for e in range(E)])
    W["e_up"] = torch.stack([t(f"e_up.{e}", (f, d)) for e in range(E)])
    W["e_down"] = torch.stack([t(f"e_down.{e}", (d, f)) for e in range(E)])
    return W


class FlatBatch:
    """Variable-length sequences stored flat: hidden [N, d]; seq[n], pos[n]; attention pads per batch."""

    def __init__(self, lengths: list[int]):
        self.lengths = list(lengths)
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)]).astype(np.int64)
        self.N = int(self.starts[-1])
        self.seq = torch.from_numpy(np.repeat(np.arange(len(lengths)), lengths))
        self.pos = torch.from_numpy(np.concatenate([np.arange(n) for n in lengths]) if lengths else np.zeros(0, np.int64))
        self.order = sorted(range(len(lengths)), key=lambda i: -self.lengths[i])

    def attend(self, h, W, cfg, cos, sin, batch: int = 8):
        out = torch.empty_like(h)
        for i0 in range(0, len(self.order), batch):
            ids = self.order[i0:i0 + batch]
            T = max(self.lengths[i] for i in ids)
            x = torch.zeros(len(ids), T, h.shape[1], dtype=h.dtype)
            for r, i in enumerate(ids):
                x[r, :self.lengths[i]] = h[self.starts[i]:self.starts[i + 1]]
            y = attention(x, W, cfg, cos, sin)
            for r, i in enumerate(ids):
                out[self.starts[i]:self.starts[i + 1]] = y[r, :self.lengths[i]]
        return out
