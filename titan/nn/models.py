"""Model definitions used by the training runtime and QuickStart."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..compiler.graph import Graph, Tensor
from ..dtypes import DType
from .layers import ParamStore, layernorm, linear, mlp, transformer_block


@dataclass
class MLPClassifier:
    in_dim: int
    classes: int
    hidden: list[int] = field(default_factory=lambda: [64, 64])
    act: str = "relu"
    name: str = "mlp"
    kind: str = "classifier"

    def build(self, g: Graph, ps: ParamStore, batch: int, compute: DType) -> tuple[Tensor, Tensor]:
        x = g.input("x", (batch, self.in_dim))
        return x, mlp(g, ps, x, self.hidden + [self.classes], self.act, self.name, compute)

    def describe(self) -> dict:
        return asdict(self)


@dataclass
class TinyTransformerLM:
    """Decoder-only character LM: one-hot tokens -> embedding -> blocks -> LN -> vocabulary logits."""
    vocab: int
    seq: int = 16
    d_model: int = 32
    heads: int = 2
    d_ff: int = 64
    layers: int = 1
    name: str = "tlm"
    kind: str = "lm"

    def build(self, g: Graph, ps: ParamStore, batch: int, compute: DType) -> tuple[Tensor, Tensor]:
        tok = g.input("x", (batch * self.seq, self.vocab))                     # one-hot tokens (host prepared)
        pos = g.const(np.tile(np.eye(self.seq), (batch, 1)), name="pos_onehot")
        emb = g.matmul(tok, ps.get(f"{self.name}.tok_emb", (self.vocab, self.d_model), "embed"))
        h = g.add(emb, g.matmul(pos, ps.get(f"{self.name}.pos_emb", (self.seq, self.d_model), "embed")))
        for i in range(self.layers):
            h = transformer_block(g, ps, h, batch, self.seq, self.heads, self.d_ff, f"{self.name}.b{i}",
                                  True, compute)
        h = layernorm(g, ps, h, f"{self.name}.lnf")
        return tok, linear(g, ps, h, self.vocab, f"{self.name}.head", compute=compute)

    def describe(self) -> dict:
        return asdict(self)


def param_count(ps: ParamStore) -> int:
    return int(sum(np.prod(t.shape) for t in ps.params.values()))
