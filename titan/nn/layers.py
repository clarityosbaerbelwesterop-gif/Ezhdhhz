"""Neural-network building blocks expressed as TITAN graphs.

Every block only emits graph ops that lower to TITAN ISA. ``compute`` selects the
matrix-engine input precision (FP32, or BF16/FP16 mixed precision with FP32
accumulation); element-wise math stays in FP32.
"""
from __future__ import annotations

import numpy as np

from ..compiler.graph import Graph, Tensor
from ..dtypes import DType


class ParamStore:
    """Creates named, deterministically initialized parameters (and reuses them by name)."""

    def __init__(self, g: Graph, seed: int = 0, values: dict | None = None, meta: bool = False,
                 lazy: bool = False):
        self.g, self.rng, self.values = g, np.random.default_rng(seed), values or {}
        self.params: dict[str, Tensor] = {}
        self.meta = meta          # shapes only: define models larger than host memory
        self.lazy = lazy          # values generated on the device, one parameter at a time
        self.seed = seed
        self.scheme = "xavier"    # xavier | gpt2 (std 0.02; residual projections / sqrt(2 * layers))
        self.layers = 1

    def get(self, name: str, shape, init: str = "xavier") -> Tensor:
        if name in self.params:
            return self.params[name]
        if self.meta:
            t = self.g.param(name, tuple(shape), meta=True)
            self.params[name] = t
            return t
        if self.scheme == "gpt2" and init in ("xavier", "embed"):
            resid = name.endswith((".ff2.w",)) or ".attn.o" in name
            init = ("normal", 0.02 / np.sqrt(2 * self.layers) if resid else 0.02)
        if self.lazy:
            t = self.g.param_lazy(name, tuple(shape), _initializer(self.seed, name, tuple(shape), init))
            self.params[name] = t
            return t
        if isinstance(init, tuple):
            v = self.rng.standard_normal(shape) * init[1]
        elif init == "xavier":
            v = self.rng.standard_normal(shape) * np.sqrt(2.0 / (shape[0] + shape[1]))
        elif init == "zeros":
            v = np.zeros(shape)
        elif init == "ones":
            v = np.ones(shape)
        elif init == "embed":
            v = self.rng.standard_normal(shape) * 0.1
        else:
            raise ValueError(init)
        v = self.values.get(name, v)
        t = self.g.param(name, np.asarray(v, dtype=np.float32))   # device precision; halves host memory
        self.params[name] = t
        return t


def _initializer(seed: int, name: str, shape: tuple, kind: str):
    """Deterministic per-parameter initializer (independent of creation order), FP32 throughout."""
    import zlib

    def make():
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        if isinstance(kind, tuple):
            return rng.standard_normal(shape, dtype=np.float32) * np.float32(kind[1])
        if kind == "xavier":
            return rng.standard_normal(shape, dtype=np.float32) * np.float32(np.sqrt(2.0 / (shape[0] + shape[1])))
        if kind == "embed":
            return rng.standard_normal(shape, dtype=np.float32) * np.float32(0.1)
        if kind == "zeros":
            return np.zeros(shape, dtype=np.float32)
        if kind == "ones":
            return np.ones(shape, dtype=np.float32)
        raise ValueError(kind)
    return make


def mm(g: Graph, a: Tensor, b: Tensor, compute: DType = DType.FP32, ta=False, tb=False, batch=1) -> Tensor:
    if compute != DType.FP32:
        a, b = g.cast(a, compute), g.cast(b, compute)
    return g.bmm(a, b, batch, ta, tb)


def linear(g: Graph, ps: ParamStore, x: Tensor, dout: int, name: str, bias: bool = True,
           compute: DType = DType.FP32) -> Tensor:
    w = ps.get(f"{name}.w", (x.cols, dout))
    y = mm(g, x, w, compute)
    if bias:
        y = g.add(y, ps.get(f"{name}.b", (1, dout), "zeros"))
    return y


def activation(g: Graph, x: Tensor, act: str) -> Tensor:
    return getattr(g, act)(x)


def mlp(g: Graph, ps: ParamStore, x: Tensor, sizes: list[int], act: str = "relu", name: str = "mlp",
        compute: DType = DType.FP32) -> Tensor:
    for i, d in enumerate(sizes):
        x = linear(g, ps, x, d, f"{name}.{i}", compute=compute)
        if i < len(sizes) - 1:
            x = activation(g, x, act)
    return x


def layernorm(g: Graph, ps: ParamStore, x: Tensor, name: str) -> Tensor:
    y = g.layernorm(x)
    return g.add(g.mul(y, ps.get(f"{name}.gamma", (1, x.cols), "ones")), ps.get(f"{name}.beta", (1, x.cols), "zeros"))


def causal_mask(batch: int, seq: int) -> np.ndarray:
    m = np.triu(np.full((seq, seq), -1e9), 1)
    return np.tile(m, (batch, 1))


def attention(g: Graph, ps: ParamStore, x: Tensor, batch: int, seq: int, heads: int, name: str,
              causal: bool = True, compute: DType = DType.FP32) -> Tensor:
    """Multi-head self-attention on row-stacked sequences x = [batch*seq, d].

    Head h uses projections W{q,k,v}_h [d, d/heads] and W{o}_h [d/heads, d]; this is
    mathematically identical to slicing fused Wq/Wk/Wv/Wo matrices, without needing
    strided slice ops in the graph.
    """
    d = x.cols
    if d % heads:
        raise ValueError("d_model must be divisible by heads")
    dh = d // heads
    mask = g.const(causal_mask(batch, seq), name=f"{name}.mask") if causal else None
    out = None
    for h in range(heads):
        q = mm(g, x, ps.get(f"{name}.q{h}", (d, dh)), compute)
        k = mm(g, x, ps.get(f"{name}.k{h}", (d, dh)), compute)
        v = mm(g, x, ps.get(f"{name}.v{h}", (d, dh)), compute)
        s = g.scale(mm(g, q, k, compute, tb=True, batch=batch), 1.0 / np.sqrt(dh))
        if mask is not None:
            s = g.add(s, mask)
        p = g.softmax(s)
        o = mm(g, p, v, compute, batch=batch)
        o = mm(g, o, ps.get(f"{name}.o{h}", (dh, d)), compute)
        out = o if out is None else g.add(out, o)
    return g.add(out, ps.get(f"{name}.ob", (1, d), "zeros"))


def transformer_block(g: Graph, ps: ParamStore, x: Tensor, batch: int, seq: int, heads: int, d_ff: int,
                      name: str, causal: bool = True, compute: DType = DType.FP32) -> Tensor:
    """Pre-LayerNorm transformer block: x + Attn(LN(x)); h + MLP(LN(h))."""
    h = g.add(x, attention(g, ps, layernorm(g, ps, x, f"{name}.ln1"), batch, seq, heads, f"{name}.attn",
                           causal, compute))
    f = linear(g, ps, layernorm(g, ps, h, f"{name}.ln2"), d_ff, f"{name}.ff1", compute=compute)
    f = linear(g, ps, g.gelu(f), x.cols, f"{name}.ff2", compute=compute)
    return g.add(h, f)
