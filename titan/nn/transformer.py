"""Configurable decoder-only transformer for TITAN (bring your own architecture).

Everything is expressed with TITAN graph ops, so the same model trains on every backend
(ISA-level emulation, cycle model, fast/accel IR executors, storage offload, LoRA):

    norm         layernorm | rmsnorm                      prenorm   True (pre-LN) | False (post-LN)
    activation   gelu | relu | silu | swiglu | geglu       bias      biases on output projections / MLP / head
    position     learned | rope | sinusoidal | none       qkv_bias  biases on q, k, v (GPT-2 style)
    heads / kv_heads (grouped-query attention)            tie_embeddings  LM head = token embedding^T

    experts / top_k / capacity_factor / moe_every / moe_aux / shared_expert   mixture-of-experts MLPs

Attention is fused across heads (split_heads -> batched products -> merge_heads), token and
position embeddings are row gathers. A config is a dataclass, a JSON file or a TOML file.

Mixture of experts: a router picks the top-k experts per token (softmax renormalized over them);
tokens are dispatched sparsely (gather -> expert-batched GEMMs -> gate -> scatter-add) with a
per-expert capacity of ceil(capacity_factor * tokens * k / experts) (overflow tokens skip the
expert, as in Switch/GShard), plus the Switch load-balancing loss. The TITAN ISA path uses the
mathematically identical dense dispatch (every expert sees every token, gate 0 when unselected).
``build_decode`` builds a one-token step with a key/value cache for fast generation.
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
    experts: int = 0                 # > 0: mixture-of-experts MLP with this many experts
    top_k: int = 2                   # experts per token
    capacity_factor: float = 1.25    # expert capacity = ceil(cf * tokens * top_k / experts)
    moe_every: int = 1               # MoE in every n-th block (1 = all blocks)
    moe_aux: float = 0.01            # load-balancing loss coefficient
    shared_expert: bool = False      # dense MLP added to the experts' output (always active)

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
        if self.experts and not (1 <= self.top_k <= self.experts and self.moe_every >= 1):
            raise ValueError("MoE needs 1 <= top_k <= experts and moe_every >= 1")
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

    def rope_table(self, positions) -> tuple[np.ndarray, np.ndarray]:
        """cos / sin rows [len(positions), head_dim] (rotate-half convention)."""
        c = self.config
        dh = c.head_dim
        inv = c.rope_base ** (-np.arange(0, dh, 2) / dh)
        ang = np.asarray(positions)[:, None] * inv[None, :]
        return np.concatenate([np.cos(ang), np.cos(ang)], 1), np.concatenate([np.sin(ang), np.sin(ang)], 1)

    def sinusoid_table(self, positions) -> np.ndarray:
        d = self.config.d_model
        pos = np.asarray(positions)[:, None] / (10000.0 ** (np.arange(0, d, 2)[None, :] / d))
        table = np.zeros((len(pos), d))
        table[:, 0::2], table[:, 1::2] = np.sin(pos), np.cos(pos)
        return table

    def _rotate(self, g: Graph, x: Tensor, cos: Tensor, sin: Tensor, tag: str) -> Tensor:
        dh = self.config.head_dim
        h = dh // 2
        R = np.zeros((dh, dh))
        R[np.arange(h) + h, np.arange(h)] = -1.0          # rotate_half(x) = [-x2, x1] = x @ R
        R[np.arange(h), np.arange(h) + h] = 1.0
        rot = g.matmul(x, g.const(R, name=f"{self.config.name}.rope_R_{tag}{len(g.tensors)}"))
        return g.add(g.mul(x, cos), g.mul(rot, sin))

    def _rope(self, g: Graph, x: Tensor, batch: int, heads: int, tag: str) -> Tensor:
        c = self.config
        cos, sin = self.rope_table(np.arange(c.seq))
        k = len(g.tensors)
        return self._rotate(g, x, g.const(np.tile(cos, (batch * heads, 1)), name=f"{c.name}.rope_cos_{tag}{k}"),
                            g.const(np.tile(sin, (batch * heads, 1)), name=f"{c.name}.rope_sin_{tag}{k}"), tag)

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
        if c.experts and self._layer is not None and (self._layer + 1) % c.moe_every == 0:
            y = self._moe(g, ps, x, f"{name}.moe", compute)
            return g.add(y, self._dense_mlp(g, ps, x, f"{name}.shared", compute)) if c.shared_expert else y
        return self._dense_mlp(g, ps, x, name, compute)

    def _act(self, g, h, up=None):
        c = self.config
        if c.activation in ("gelu", "relu"):
            return getattr(g, c.activation)(h)
        if c.activation == "silu":
            return g.mul(h, g.sigmoid(h))
        gate = g.mul(h, g.sigmoid(h)) if c.activation == "swiglu" else g.gelu(h)
        return g.mul(gate, up)

    def _moe(self, g, ps, x, name, compute):
        """Top-k mixture of experts; sparse dispatch on TITAN IR, dense dispatch on the ISA path."""
        c = self.config
        E, k, N, d, f = c.experts, c.top_k, x.rows, x.cols, c.d_ff
        s = linear(g, ps, x, E, f"{name}.router", bias=False)                       # router logits (FP32)
        t, m = s, None
        for i in range(k):                                                          # top-k mask, no gradient
            sel = g.scale(g.step(g.scale(g.sub(t, g.max(t, 1)), -1.0)), -1.0, 1.0)   # 1 where t == row max
            m = sel if m is None else g.add(m, sel)
            if i + 1 < k:
                t = g.sub(t, g.scale(sel, 1e9))
        gates = g.softmax(g.add(s, g.scale(m, 1e9, -1e9)))                          # renormalized over top-k
        if getattr(ps, "dense_moe", False):
            C, ids = N, g.const(np.tile(np.arange(N), E)[:, None], DType.INT32, name=f"{name}.dense_ids_c{N}")
        else:
            C = N if self._no_drop else min(N, max(1, math.ceil(c.capacity_factor * N * k / E)))
            ids = g.moe_slots(m, C)
        sel_c = g.const(np.repeat(np.eye(E), C, axis=0), name=f"{name}.slot_expert_c{C}")
        ge = g.sum(g.mul(g.embedding(ids, gates), sel_c), 1)                        # gate of every slot [E*C,1]
        xe = g.embedding(ids, x)                                                    # dispatch [E*C, d]
        gpt2 = getattr(ps, "scheme", "xavier") == "gpt2"
        std_in = 0.02 if gpt2 else math.sqrt(2.0 / (d + f))
        std_out = 0.02 / math.sqrt(2 * c.layers) if gpt2 else math.sqrt(2.0 / (d + f))
        e_of_slot = None
        if c.bias:
            e_of_slot = g.const(np.repeat(np.arange(E), C)[:, None], DType.INT32, name=f"{name}.slot_e_c{C}")

        def proj(inp, rows, cols, wname, std):
            y = mm(g, inp, ps.get(f"{name}.{wname}", (E * rows, cols), ("normal", std)), compute, batch=E)
            if c.bias:
                y = g.add(y, g.embedding(e_of_slot, ps.get(f"{name}.{wname}_b", (E, cols), "zeros")))
            return y
        h = proj(xe, d, f, "w1", std_in)
        h = self._act(g, h, proj(xe, d, f, "w3", std_in) if c.gated else None)
        ye = g.mul(proj(h, f, d, "w2", std_out), ge)
        y = g.embedding_grad(ids, ye, N)                                            # combine (scatter-add)
        if c.moe_aux and not self._no_drop:                                         # Switch load-balancing loss
            frac = g.mean(m, 0)
            prob = g.mean(g.softmax(s), 0)
            aux = g.scale(g.sum(g.mul(frac, prob), 2), E / k)
            if not hasattr(g, "aux_losses"):
                g.aux_losses = []
            g.aux_losses.append((c.moe_aux, aux))
        return y

    def _dense_mlp(self, g, ps, x, name, compute):
        c = self.config
        h = linear(g, ps, x, c.d_ff, f"{name}.ff1", bias=c.bias, compute=compute)
        up = linear(g, ps, x, c.d_ff, f"{name}.ff3", bias=c.bias, compute=compute) if c.gated else None
        return linear(g, ps, self._act(g, h, up), c.d_model, f"{name}.ff2", bias=c.bias, compute=compute)

    _layer = None
    _no_drop = False

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
            h = g.add(h, g.const(np.tile(self.sinusoid_table(np.arange(T)), (batch, 1)), name=f"{c.name}.sinusoid"))
        mask = None
        if c.causal:
            m = np.triu(np.full((T, T), -1e9), 1)
            mask = g.const(np.tile(m, (batch * c.heads, 1)), name=f"{c.name}.mask")
        for i in range(c.layers):
            self._layer = i
            h = self._block(g, ps, h, batch, f"{c.name}.b{i}", compute, mask)
        self._layer = None
        return ids, self._head(g, ps, h, tok, compute)

    def _head(self, g, ps, h, tok, compute):
        c = self.config
        h = self._norm(g, ps, h, f"{c.name}.lnf")
        if c.tie_embeddings:
            logits = mm(g, h, tok, compute, tb=True)
            if c.bias:
                logits = g.add(logits, ps.get(f"{c.name}.head.b", (1, c.vocab), "zeros"))
        else:
            logits = linear(g, ps, h, c.vocab, f"{c.name}.head", bias=c.bias, compute=compute)
        return logits

    # ------------------------------------------------------------------ generation (key/value cache)
    def build_decode(self, g: Graph, ps: ParamStore, batch: int, tmax: int, compute: DType = DType.FP32) -> dict:
        """One generation step for ``batch`` sequences with a key/value cache of ``tmax`` positions.

        Inputs: tok [B,1] ids, pos [B,1] positions (learned), pvec [B,d] (sinusoidal), rope_cq/sq [B*H,dh] and
        rope_ck/sk [B*KV,dh] (rope), slot [B*KV,1] cache rows of the current position, amask [B*H,tmax]
        (0 = visible, -1e9 = not yet written), kc{i}/vc{i} [B*KV*tmax, dh] caches (current rows zero).
        Outputs: logits [B, vocab]; k{i}, v{i} [B*KV, dh] = this position's keys/values (to store)."""
        c = self.config
        if not c.causal:
            raise ValueError("generation needs a causal model")
        ps.layers = c.layers
        B, H, KV, dh, d = batch, c.heads, c.kv_heads, c.head_dim, c.d_model
        tok = ps.get(f"{c.name}.tok_emb", (c.vocab, d), "embed")
        h = g.embedding(g.input("tok", (B, 1), DType.INT32), tok)
        if c.position == "learned":
            h = g.add(h, g.embedding(g.input("pos", (B, 1), DType.INT32), ps.get(f"{c.name}.pos_emb", (c.seq, d), "embed")))
        elif c.position == "sinusoidal":
            h = g.add(h, g.input("pvec", (B, d)))
        rope = None
        if c.position == "rope":
            rope = [g.input(n, (B * hh, dh)) for n, hh in (("rope_cq", H), ("rope_sq", H), ("rope_ck", KV), ("rope_sk", KV))]
        slot = g.input("slot", (B * KV, 1), DType.INT32)
        amask = g.input("amask", (B * H, tmax))
        self._no_drop = True
        for i in range(c.layers):
            self._layer = i
            name = f"{c.name}.b{i}"
            kc, vc = g.input(f"kc{i}", (B * KV * tmax, dh)), g.input(f"vc{i}", (B * KV * tmax, dh))

            def attn(x, i=i, name=name, kc=kc, vc=vc):
                q = linear(g, ps, x, H * dh, f"{name}.attn.q", bias=c.qkv_bias, compute=compute)
                k = linear(g, ps, x, KV * dh, f"{name}.attn.k", bias=c.qkv_bias, compute=compute)
                v = linear(g, ps, x, KV * dh, f"{name}.attn.v", bias=c.qkv_bias, compute=compute)
                qh, kh, vh = g.split_heads(q, B, H), g.split_heads(k, B, KV), g.split_heads(v, B, KV)
                if rope:
                    qh = self._rotate(g, qh, rope[0], rope[1], "dq")
                    kh = self._rotate(g, kh, rope[2], rope[3], "dk")
                g.output(f"k{i}", kh)
                g.output(f"v{i}", vh)
                K = g.add(kc, g.embedding_grad(slot, kh, B * KV * tmax))        # cache + this position
                V = g.add(vc, g.embedding_grad(slot, vh, B * KV * tmax))
                K, V = g.repeat_heads(K, B, KV, H // KV), g.repeat_heads(V, B, KV, H // KV)
                sc = g.add(g.scale(mm(g, qh, K, compute, tb=True, batch=B * H), 1.0 / math.sqrt(dh)), amask)
                o = g.merge_heads(mm(g, g.softmax(sc), V, compute, batch=B * H), B, H)
                return linear(g, ps, o, d, f"{name}.attn.o", bias=c.bias, compute=compute)
            if c.prenorm:
                h = g.add(h, attn(self._norm(g, ps, h, f"{name}.ln1")))
                h = g.add(h, self._mlp(g, ps, self._norm(g, ps, h, f"{name}.ln2"), name, compute))
            else:
                h = self._norm(g, ps, g.add(h, attn(h)), f"{name}.ln1")
                h = self._norm(g, ps, g.add(h, self._mlp(g, ps, h, name, compute)), f"{name}.ln2")
        self._layer, self._no_drop = None, False
        g.output("logits", self._head(g, ps, h, tok, compute))
        return {"batch": B, "tmax": tmax}


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
