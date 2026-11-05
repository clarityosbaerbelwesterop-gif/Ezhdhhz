"""Configurable decoder-only transformer for TITAN (bring your own architecture).

Everything is expressed with TITAN graph ops, so the same model trains on every backend
(ISA-level emulation, cycle model, fast/accel IR executors, storage offload, LoRA):

    norm         layernorm | rmsnorm                      prenorm   True (pre-LN) | False (post-LN)
    activation   gelu | relu | silu | swiglu | geglu       bias      biases on output projections / MLP / head
    position     learned | rope | sinusoidal | none       qkv_bias  biases on q, k, v (GPT-2 style)
    heads / kv_heads (grouped-query attention)            tie_embeddings  LM head = token embedding^T

Attention is fused across heads (split_heads -> batched products -> merge_heads), token and
position embeddings are row gathers. A config is a dataclass, a JSON file or a TOML file.
"""
from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from ..compiler.graph import Graph, Tensor
from ..dtypes import DType
from .layers import ParamStore, linear, mm

NORMS = ("layernorm", "rmsnorm")
ACTIVATIONS = ("gelu", "relu", "silu", "swiglu", "geglu")
POSITIONS = ("learned", "rope", "sinusoidal", "none")


@dataclass
class TransformerConfig:
    vocab: int
    seq: int = 16
    d_model: int = 32
    layers: int = 1
    heads: int = 2
    kv_heads: int | None = None
    d_ff: int | None = None
    norm: str = "layernorm"
    norm_eps: float = 1e-5
    activation: str = "gelu"
    position: str = "learned"
    rope_base: float = 10000.0
    tie_embeddings: bool = False
    bias: bool = True
    qkv_bias: bool = False
    prenorm: bool = True
    causal: bool = True
    name: str = "tlm"

    def __post_init__(self):
        if self.kv_heads is None:
            self.kv_heads = self.heads
        if self.d_ff is None:
            self.d_ff = (-(-int(8 * self.d_model / 3) // 32) * 32 if self.activation in ("swiglu", "geglu")
                         else 4 * self.d_model)
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads

    @property
    def gated(self) -> bool:
        return self.activation in ("swiglu", "geglu")

    def validate(self) -> "TransformerConfig":
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.heads % self.kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if self.norm not in NORMS or self.activation not in ACTIVATIONS or self.position not in POSITIONS:
            raise ValueError(f"norm in {NORMS}, activation in {ACTIVATIONS}, position in {POSITIONS}")
        if self.position == "rope" and self.head_dim % 2:
            raise ValueError("rope needs an even head dimension")
        for k in ("vocab", "seq", "d_model", "layers", "heads", "d_ff"):
            if getattr(self, k) <= 0:
                raise ValueError(f"{k} must be > 0")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown transformer config keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path, **overrides) -> "TransformerConfig":
        """JSON or TOML file; ``vocab`` may come from the tokenizer via overrides."""
        raw = read_model_config(path)
        raw.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(raw)

    def to_dict(self) -> dict:
        return asdict(self)


def read_model_config(path) -> dict:
    """Architecture settings from a JSON or TOML file (a ``[model]`` / ``"model"`` section is accepted)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"model config {p} not found")
    raw = tomllib.loads(p.read_text()) if p.suffix == ".toml" else json.loads(p.read_text())
    raw = raw.get("model", raw) if isinstance(raw.get("model"), dict) else raw
    return dict(raw)


class TransformerLM:
    kind = "lm"

    def __init__(self, config: TransformerConfig):
        self.config = config.validate()

    # compatibility attributes used by the runtime
    @property
    def name(self):
        return self.config.name

    @property
    def vocab(self):
        return self.config.vocab

    @property
    def seq(self):
        return self.config.seq

    @property
    def layers(self):
        return self.config.layers

    def describe(self) -> dict:
        return {"kind": self.kind, "architecture": "decoder-only transformer", **self.config.to_dict()}

    # ------------------------------------------------------------------ building blocks
    def _norm(self, g: Graph, ps: ParamStore, x: Tensor, name: str) -> Tensor:
        c = self.config
        if c.norm == "rmsnorm":
            return g.mul(g.rmsnorm(x, c.norm_eps), ps.get(f"{name}.gamma", (1, x.cols), "ones"))
        y = g.layernorm(x, c.norm_eps)
        return g.add(g.mul(y, ps.get(f"{name}.gamma", (1, x.cols), "ones")), ps.get(f"{name}.beta", (1, x.cols), "zeros"))

    def _rope(self, g: Graph, x: Tensor, batch: int, heads: int, tag: str) -> Tensor:
        c = self.config
        T, dh = c.seq, c.head_dim
        inv = c.rope_base ** (-np.arange(0, dh, 2) / dh)
        ang = np.arange(T)[:, None] * inv[None, :]
        cos = np.tile(np.concatenate([np.cos(ang), np.cos(ang)], 1), (batch * heads, 1))
        sin = np.tile(np.concatenate([np.sin(ang), np.sin(ang)], 1), (batch * heads, 1))
        h = dh // 2
        R = np.zeros((dh, dh))
        R[np.arange(h) + h, np.arange(h)] = -1.0          # rotate_half(x) = [-x2, x1] = x @ R
        R[np.arange(h), np.arange(h) + h] = 1.0
        k = len(g.tensors)
        rot = g.matmul(x, g.const(R, name=f"{c.name}.rope_R_{tag}{k}"))
        return g.add(g.mul(x, g.const(cos, name=f"{c.name}.rope_cos_{tag}{k}")),
                     g.mul(rot, g.const(sin, name=f"{c.name}.rope_sin_{tag}{k}")))

    def _attention(self, g, ps, x, batch, name, compute, mask):
        c = self.config
        H, KV, dh = c.heads, c.kv_heads, c.head_dim
        q = linear(g, ps, x, H * dh, f"{name}.q", bias=c.qkv_bias, compute=compute)
        k = linear(g, ps, x, KV * dh, f"{name}.k", bias=c.qkv_bias, compute=compute)
        v = linear(g, ps, x, KV * dh, f"{name}.v", bias=c.qkv_bias, compute=compute)
        qh, kh, vh = g.split_heads(q, batch, H), g.split_heads(k, batch, KV), g.split_heads(v, batch, KV)
        if c.position == "rope":
            qh, kh = self._rope(g, qh, batch, H, "q"), self._rope(g, kh, batch, KV, "k")
        kh = g.repeat_heads(kh, batch, KV, H // KV)
        vh = g.repeat_heads(vh, batch, KV, H // KV)
        s = g.scale(mm(g, qh, kh, compute, tb=True, batch=batch * H), 1.0 / math.sqrt(dh))
        if mask is not None:
            s = g.add(s, mask)
        o = g.merge_heads(mm(g, g.softmax(s), vh, compute, batch=batch * H), batch, H)
        return linear(g, ps, o, c.d_model, f"{name}.o", bias=c.bias, compute=compute)

    def _mlp(self, g, ps, x, name, compute):
        c = self.config
        h = linear(g, ps, x, c.d_ff, f"{name}.ff1", bias=c.bias, compute=compute)
        if c.activation in ("gelu", "relu"):
            h = getattr(g, c.activation)(h)
        elif c.activation == "silu":
            h = g.mul(h, g.sigmoid(h))
        else:
            up = linear(g, ps, x, c.d_ff, f"{name}.ff3", bias=c.bias, compute=compute)
            gate = g.mul(h, g.sigmoid(h)) if c.activation == "swiglu" else g.gelu(h)
            h = g.mul(gate, up)
        return linear(g, ps, h, c.d_model, f"{name}.ff2", bias=c.bias, compute=compute)

    def _block(self, g, ps, x, batch, name, compute, mask):
        c = self.config
        if c.prenorm:
            x = g.add(x, self._attention(g, ps, self._norm(g, ps, x, f"{name}.ln1"), batch, f"{name}.attn", compute,
                                         mask))
            return g.add(x, self._mlp(g, ps, self._norm(g, ps, x, f"{name}.ln2"), name, compute))
        x = self._norm(g, ps, g.add(x, self._attention(g, ps, x, batch, f"{name}.attn", compute, mask)), f"{name}.ln1")
        return self._norm(g, ps, g.add(x, self._mlp(g, ps, x, name, compute)), f"{name}.ln2")

    # ------------------------------------------------------------------ model
    def build(self, g: Graph, ps: ParamStore, batch: int, compute: DType = DType.FP32) -> tuple[Tensor, Tensor]:
        """Input ``x``: token ids [batch*seq, 1] (INT32). Returns (ids input, logits [batch*seq, vocab])."""
        c = self.config
        ps.layers = c.layers
        T, d = c.seq, c.d_model
        ids = g.input("x", (batch * T, 1), DType.INT32)
        tok = ps.get(f"{c.name}.tok_emb", (c.vocab, d), "embed")
        h = g.embedding(ids, tok)
        if c.position == "learned":
            pos_ids = g.const(np.tile(np.arange(T), batch)[:, None], DType.INT32, name=f"{c.name}.pos_ids")
            h = g.add(h, g.embedding(pos_ids, ps.get(f"{c.name}.pos_emb", (T, d), "embed")))
        elif c.position == "sinusoidal":
            pos = np.arange(T)[:, None] / (10000.0 ** (np.arange(0, d, 2)[None, :] / d))
            table = np.zeros((T, d))
            table[:, 0::2], table[:, 1::2] = np.sin(pos), np.cos(pos)
            h = g.add(h, g.const(np.tile(table, (batch, 1)), name=f"{c.name}.sinusoid"))
        mask = None
        if c.causal:
            m = np.triu(np.full((T, T), -1e9), 1)
            mask = g.const(np.tile(m, (batch * c.heads, 1)), name=f"{c.name}.mask")
        for i in range(c.layers):
            h = self._block(g, ps, h, batch, f"{c.name}.b{i}", compute, mask)
        h = self._norm(g, ps, h, f"{c.name}.lnf")
        if c.tie_embeddings:
            logits = mm(g, h, tok, compute, tb=True)
            if c.bias:
                logits = g.add(logits, ps.get(f"{c.name}.head.b", (1, c.vocab), "zeros"))
        else:
            logits = linear(g, ps, h, c.vocab, f"{c.name}.head", bias=c.bias, compute=compute)
        return ids, logits


def load_model_plugin(spec: str, vocab: int, seq: int, **kwargs):
    """``path/to/file.py:factory`` or ``package.module:factory``; ``factory(vocab=..., seq=..., **kw)`` returns
    an object with ``build(g, ps, batch, compute) -> (ids_input, logits)``, ``name``, ``seq``, ``kind == "lm"``."""
    import importlib
    import importlib.util
    mod_name, _, attr = spec.rpartition(":")
    if not mod_name or not attr:
        raise ValueError("model plugin must look like 'file.py:factory' or 'package.module:factory'")
    if mod_name.endswith(".py"):
        sp = importlib.util.spec_from_file_location(Path(mod_name).stem, mod_name)
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_name)
    factory = getattr(mod, attr)
    import inspect
    try:
        sig = inspect.signature(factory)
        if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}   # pass what the factory accepts
    except (TypeError, ValueError):
        pass
    model = factory(vocab=vocab, seq=seq, **kwargs)
    for need in ("build", "name", "seq"):
        if not hasattr(model, need):
            raise TypeError(f"model plugin {spec} lacks '{need}'")
    return model
