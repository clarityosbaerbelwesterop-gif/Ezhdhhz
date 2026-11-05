"""Model definitions used by the training runtime and QuickStart."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..compiler.graph import Graph, Tensor
from ..dtypes import DType
from .layers import ParamStore, mlp
from .transformer import TransformerConfig, TransformerLM


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


class TinyTransformerLM(TransformerLM):
    """Backward-compatible constructor for the configurable transformer (GPT-style defaults)."""

    def __init__(self, vocab: int, seq: int = 16, d_model: int = 32, heads: int = 2, d_ff: int = 64,
                 layers: int = 1, name: str = "tlm", **config):
        super().__init__(TransformerConfig(vocab=vocab, seq=seq, d_model=d_model, heads=heads, d_ff=d_ff,
                                           layers=layers, name=name, **config))


def param_count(ps: ParamStore) -> int:
    return int(sum(np.prod(t.shape) for t in ps.params.values()))
