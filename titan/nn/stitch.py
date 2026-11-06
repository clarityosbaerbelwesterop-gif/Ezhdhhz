"""Stitched transformer: segments taken from different donor models, joined by linear bridges.

Each segment keeps its donor's architecture exactly (width, heads / kv_heads, norm, activation,
RoPE, MoE, ...). Bridges map one segment's residual stream into the next segment's width
(``{name}.bridge{s}``). Embeddings follow segment 0, the final norm and LM head the last segment.
Built from TITAN graph ops only, so it trains, fine-tunes and runs on every backend.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..compiler.graph import Graph
from ..dtypes import DType
from .layers import ParamStore, linear
from .transformer import TransformerConfig, TransformerLM

SEGMENT_KEYS = {"d_model", "layers", "heads", "kv_heads", "d_ff", "norm", "norm_eps", "activation", "position",
                "rope_base", "bias", "qkv_bias", "prenorm", "experts", "top_k", "capacity_factor", "moe_every",
                "moe_aux", "shared_expert"}


@dataclass
class StitchConfig:
    vocab: int
    seq: int
    segments: list = field(default_factory=list)     # [{d_model, layers, heads, ..., "donor": "..."}]
    head_bias: bool = True
    name: str = "frank"

    def segment_config(self, s: int) -> TransformerConfig:
        seg = {k: v for k, v in self.segments[s].items() if k in SEGMENT_KEYS}
        return TransformerConfig(vocab=self.vocab, seq=self.seq, name=f"{self.name}.s{s}", **seg)


class StitchedLM:
    kind = "lm"
    _no_drop = False                    # evaluation / inference: MoE segments without capacity drops

    def __init__(self, config: StitchConfig):
        if not config.segments:
            raise ValueError("a stitched model needs at least one segment")
        self.config = config
        self.segs = [TransformerLM(config.segment_config(s)) for s in range(len(config.segments))]

    @classmethod
    def from_dict(cls, d: dict) -> "StitchedLM":
        keys = {"vocab", "seq", "segments", "head_bias", "name"}
        return cls(StitchConfig(**{k: v for k, v in d.items() if k in keys}))

    name = property(lambda self: self.config.name)
    seq = property(lambda self: self.config.seq)
    vocab = property(lambda self: self.config.vocab)
    layers = property(lambda self: sum(s.config.layers for s in self.segs))

    def describe(self) -> dict:
        return {"kind": "lm", "architecture": "stitched", **asdict(self.config)}

    def build(self, g: Graph, ps: ParamStore, batch: int, compute: DType = DType.FP32):
        c, s0 = self.config, self.segs[0].config
        ps.layers = self.layers
        T, V = c.seq, c.vocab
        ids = g.input("x", (batch * T, 1), DType.INT32)
        h = g.embedding(ids, ps.get(f"{c.name}.tok_emb", (V, s0.d_model), "embed"))
        if s0.position == "learned":
            pos_ids = g.const(np.tile(np.arange(T), batch)[:, None], DType.INT32, name=f"{c.name}.pos_ids")
            h = g.add(h, g.embedding(pos_ids, ps.get(f"{c.name}.pos_emb", (T, s0.d_model), "embed")))
        elif s0.position == "sinusoidal":
            h = g.add(h, g.const(np.tile(self.segs[0].sinusoid_table(np.arange(T)), (batch, 1)), name=f"{c.name}.sinusoid"))
        for s, seg in enumerate(self.segs):
            sc = seg.config
            seg._no_drop = self._no_drop
            if s:
                h = linear(g, ps, h, sc.d_model, f"{c.name}.bridge{s}", bias=True)
            mask = None
            if sc.causal:
                m = np.triu(np.full((T, T), -1e9), 1)
                mask = g.const(np.tile(m, (batch * sc.heads, 1)), name=f"{sc.name}.mask")
            for i in range(sc.layers):
                seg._layer = i
                h = seg._block(g, ps, h, batch, f"{sc.name}.b{i}", compute, mask)
            seg._layer = None
        last = self.segs[-1]
        h = last._norm(g, ps, h, f"{c.name}.lnf")
        return ids, linear(g, ps, h, V, f"{c.name}.head", bias=c.head_bias, compute=compute)
