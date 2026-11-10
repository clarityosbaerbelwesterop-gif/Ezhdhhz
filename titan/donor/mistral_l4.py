"""Mistral Large 4 text stack, one layer at a time, with experts streamed one by one.

Semantics follow transformers' MistralLarge4TextModel (tested against it on random weights):
  attention  multi-head latent attention: q = wq(x) split into [nope | rope] parts; keys/values come from a
             rank-`kv_rank` latent (wkv_a -> RMSNorm -> wkv_b) plus one shared rotary key part; interleaved
             RoPE with learned frequency scaling (LeRoPE); NoPE layers (no rotary part); layers listed in
             `kv_source` reuse the latent keys of an earlier layer; queries are scaled per head by
             s*log(pos+1)+b (scalable softmax). The sparse-attention indexer selects min(top-k, T) tokens;
             for sequences up to top-k (2048) tokens it selects all of them and is therefore omitted here.
  mlp        layer 0 dense SwiGLU; other layers: softmax router (fp32), top-k, renormalised, routed
             experts w2(silu(w1 x) * w3 x) plus one always-on shared expert.
  head       final RMSNorm, output matrix, logits soft-capped: cap * tanh(logits / cap).
Checkpoint weights are FP8 (E4M3) with one float32 scale per 128x128 block: W = fp8 * qscale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ML4Config:
    hidden: int = 6144
    layers: int = 72
    heads: int = 64
    qk_rope: int = 64
    qk_nope: int = 64
    v_head: int = 128
    kv_rank: int = 512
    experts: int = 128
    top_k: int = 4
    moe_inter: int = 6144
    eps: float = 1e-5
    routed_scale: float = 1.0
    softcap: float = 30.0
    vocab: int = 204800
    use_rope: list = field(default_factory=list)        # per layer: 1 = rotary layer, 0 = NoPE layer
    kv_source: list = field(default_factory=list)
    mlp_types: list = field(default_factory=list)       # "dense" | "sparse"
    index_topk: int = 2048

    @classmethod
    def from_hf_text(cls, t: dict) -> "ML4Config":
        return cls(t["hidden_size"], t["num_hidden_layers"], t["num_attention_heads"], t["qk_rope_head_dim"],
                   t["qk_nope_head_dim"], t["v_head_dim"], t["kv_lora_rank"], t["n_routed_experts"],
                   t["num_experts_per_tok"], t["moe_intermediate_size"], t["rms_norm_eps"],
                   t.get("routed_scaling_factor", 1.0), t.get("final_logit_softcapping", 30.0), t["vocab_size"],
                   list(t["no_rope_layers"]), list(t["kv_source_layers"]), list(t["mlp_layer_types"]),
                   t.get("index_topk", 2048))

    def stores_kv(self, layer: int) -> bool:
        return any(s == layer for s in self.kv_source[layer + 1:])


ATTN_KEYS = {"in_ln": "attention_norm.weight", "post_ln": "ffn_norm.weight", "wq": "attention.wq.weight",
             "wo": "attention.wo.weight", "wkv_b": "attention.wkv_b.weight", "ss_s": "attention.ssmax_scaling.s",
             "ss_b": "attention.ssmax_scaling.b"}
OWN_KV_KEYS = {"wkv_a": "attention.wkv_a_with_mqa.weight", "kv_a_norm": "attention.kv_a_norm.weight"}


def rms_norm(x, w, eps: float):
    return w * (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps))


def rope_interleave(x, cos, sin):
    """x [..., T, 2n]; cos/sin [T, n] (one angle per interleaved pair). Returns cat(even', odd')."""
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


def rope_tables(T: int, inv_freq: torch.Tensor, alpha: torch.Tensor, attention_scaling: float = 1.0):
    fr = torch.arange(T, dtype=torch.float32)[:, None] * (inv_freq.float() * alpha.float().exp())[None, :]
    return fr.cos() * attention_scaling, fr.sin() * attention_scaling


def rope_frequencies(text_config: dict):
    """(inv_freq, attention_scaling) of the checkpoint's rotary scheme (YaRN for the released model), from
    the transformers rope utilities with the checkpoint's text config. The learned LeRoPE factor is applied
    in rope_tables."""
    from transformers import MistralLarge4TextConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    tc = MistralLarge4TextConfig(**{k: v for k, v in text_config.items() if k != "model_type"})
    kind = tc.rope_parameters["rope_type"]
    if kind == "default":
        dim = getattr(tc, "head_dim", None) or tc.hidden_size // tc.num_attention_heads
        return 1.0 / (tc.rope_parameters["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)), 1.0
    return ROPE_INIT_FUNCTIONS[kind](tc)


def attention(h, W: dict, cfg: ML4Config, layer: int, cos, sin, shared_k=None):
    """h [B, T, d] padded at the end. Returns (h + attn, latent keys [B, T, kv_rank] of this layer or None)."""
    B, T, d = h.shape
    H = cfg.heads
    use_rope = bool(cfg.use_rope[layer])
    rope_d = cfg.qk_rope if use_rope else 0
    nope_d = cfg.qk_nope + (0 if use_rope else cfg.qk_rope)
    qk = cfg.qk_nope + cfg.qk_rope
    xn = rms_norm(h, W["in_ln"], cfg.eps)
    q = F.linear(xn, W["wq"]).view(B, T, H, qk).transpose(1, 2)
    q_pass, q_rot = q[..., :nope_d], q[..., nope_d:]
    own = cfg.kv_source[layer] == layer
    if own:
        ckv = F.linear(xn, W["wkv_a"])
        k_pass = rms_norm(ckv[..., :cfg.kv_rank], W["kv_a_norm"], cfg.eps)
        k_rot = ckv[..., cfg.kv_rank:].view(B, 1, T, rope_d)
    else:
        if use_rope:
            raise NotImplementedError("key sharing between rotary layers")
        k_pass, k_rot = shared_k, h.new_zeros(B, 1, T, 0)
    if use_rope:
        c, s = cos[None, None, :T], sin[None, None, :T]
        q_rot = rope_interleave(q_rot, c, s)
        if own:
            k_rot = rope_interleave(k_rot, c, s)
    q = torch.cat([q_pass, q_rot], -1)
    log_pos = torch.arange(1, T + 1, dtype=torch.float32).log()
    q = q * (W["ss_s"][None, :, None] * log_pos[None, None, :] + W["ss_b"][None, :, None]).unsqueeze(-1)
    kv = F.linear(k_pass, W["wkv_b"]).view(B, T, H, nope_d + cfg.v_head).transpose(1, 2)
    k = torch.cat([kv[..., :nope_d], k_rot.expand(B, H, T, rope_d)], -1)
    v = kv[..., nope_d:]
    sc = torch.matmul(q, k.transpose(2, 3)) * (qk ** -0.5)
    sc = sc + torch.triu(torch.full((T, T), float("-inf")), 1)
    o = torch.matmul(torch.softmax(sc.float(), -1).to(q.dtype), v).transpose(1, 2).reshape(B, T, H * cfg.v_head)
    return h + F.linear(o, W["wo"]), (k_pass if own else None)


def route(xn, gate_w, cfg: ML4Config):
    p = F.linear(xn.float(), gate_w.float()).softmax(-1)
    topw, topi = torch.topk(p, cfg.top_k, -1)
    topw = topw / (topw.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scale
    return topi, topw


def swiglu(x, w1, w3, w2):
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


def dequant(payload: bytes, shape, scale: bytes | None, block=(128, 128)) -> torch.Tensor:
    """FP8-E4M3 payload (+ float32 block scales) -> float32 weight [out, in]; W = fp8 * scale per block."""
    w = torch.frombuffer(bytearray(payload), dtype=torch.float8_e4m3fn).reshape(shape).to(torch.float32)
    if scale is not None:
        o, i = shape
        so, si = -(-o // block[0]), -(-i // block[1])
        s = torch.frombuffer(bytearray(scale), dtype=torch.float32).reshape(so, si)
        if o % block[0] == 0 and i % block[1] == 0:
            w = (w.view(so, block[0], si, block[1]) * s[:, None, :, None]).view(o, i)
        else:
            w = w * s.repeat_interleave(block[0], 0)[:o].repeat_interleave(block[1], 1)[:, :i]
    return w


class FlatBatch:
    """Variable-length sequences stored flat ([N, d]); attention pads per batch and can carry a second
    per-token array (the shared latent keys) through the same padding."""

    def __init__(self, lengths: list[int]):
        self.lengths = list(lengths)
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)]).astype(np.int64)
        self.N = int(self.starts[-1])
        self.seq = torch.from_numpy(np.repeat(np.arange(len(lengths)), lengths))
        self.pos = torch.from_numpy(np.concatenate([np.arange(n) for n in lengths]))
        self.order = sorted(range(len(lengths)), key=lambda i: -self.lengths[i])

    def _pad(self, flat, ids, T):
        x = flat.new_zeros(len(ids), T, flat.shape[1])
        for r, i in enumerate(ids):
            x[r, :self.lengths[i]] = flat[self.starts[i]:self.starts[i + 1]]
        return x

    def attend(self, h, W, cfg, layer, cos, sin, shared_k=None, batch: int = 8):
        out = torch.empty_like(h)
        keys = h.new_zeros(h.shape[0], cfg.kv_rank) if cfg.kv_source[layer] == layer and cfg.stores_kv(layer) else None
        for i0 in range(0, len(self.order), batch):
            ids = self.order[i0:i0 + batch]
            T = max(self.lengths[i] for i in ids)
            sk = self._pad(shared_k, ids, T) if shared_k is not None else None
            y, kp = attention(self._pad(h, ids, T), W, cfg, layer, cos, sin, sk)
            for r, i in enumerate(ids):
                out[self.starts[i]:self.starts[i + 1]] = y[r, :self.lengths[i]]
                if keys is not None:
                    keys[self.starts[i]:self.starts[i + 1]] = kp[r, :self.lengths[i]]
        return out, keys


def softcap_logits(lg, cap: float):
    return torch.tanh(lg / cap) * cap


def layer_tensor_names(layer: int, cfg: ML4Config) -> dict:
    """Checkpoint names of a layer's non-expert tensors and of its FP8 scale tensors."""
    pre = f"layers.{layer}."
    out = {k: pre + v for k, v in ATTN_KEYS.items()}
    if cfg.kv_source[layer] == layer:
        out.update({k: pre + v for k, v in OWN_KV_KEYS.items()})
    if cfg.mlp_types[layer] == "sparse":
        out["gate"] = pre + "gate.weight"
        for w in ("w1", "w2", "w3"):
            out[f"shared.{w}"] = f"{pre}shared_experts.{w}.weight"
    else:
        for w in ("w1", "w2", "w3"):
            out[f"dense.{w}"] = f"{pre}feed_forward.{w}.weight"
    return out


def expert_tensor_names(layer: int, e: int) -> dict:
    return {w: f"layers.{layer}.experts.{e}.{w}.weight" for w in ("w1", "w2", "w3")}


def scale_name(weight_name: str) -> str:
    return weight_name[:-len("weight")] + "qscale_weight"


def flops_per_token(cfg: ML4Config) -> float:
    """Matmul FLOPs of one token through one sparse layer (top-k routed experts + shared expert + attention)."""
    d, f = cfg.hidden, cfg.moe_inter
    qk = cfg.qk_nope + cfg.qk_rope
    attn = d * cfg.heads * qk + cfg.heads * cfg.v_head * d + d * (cfg.kv_rank + cfg.qk_rope) + cfg.kv_rank * cfg.heads * (cfg.qk_nope + cfg.v_head)
    return 2.0 * ((cfg.top_k + 1) * 3 * d * f + attn)


__all__ = ["ML4Config", "attention", "route", "swiglu", "dequant", "FlatBatch", "rope_tables", "rms_norm",
           "softcap_logits", "rope_frequencies", "layer_tensor_names", "expert_tensor_names", "scale_name", "flops_per_token", "math"]
