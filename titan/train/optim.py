"""Optimizers expressed as graph operations, so parameter updates execute on TITAN.

Host-side work per step: Adam's bias-corrected learning rate lr_t (one scalar).
"""
from __future__ import annotations

import math

import numpy as np

from ..compiler.graph import Graph, Tensor


class SGD:
    host_inputs: tuple = ()

    def __init__(self, lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 0.0):
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay

    def apply(self, g: Graph, params: list[Tensor], grads: list[Tensor]) -> None:
        for p, gr in zip(params, grads):
            if self.wd:
                gr = g.add(gr, g.scale(p, self.wd))
            if self.momentum:
                v = g.param(f"{p.name}.momentum", np.zeros(p.shape))
                gr = g.add(g.scale(v, self.momentum), gr)
                g.assign(v, gr)
            g.assign(p, g.sub(p, g.scale(gr, self.lr)))

    def host_feeds(self, step: int) -> dict:
        return {}

    def describe(self) -> dict:
        return {"name": "sgd", "lr": self.lr, "momentum": self.momentum, "weight_decay": self.wd}


class Adam:
    host_inputs = ("opt.lr_t",)

    def __init__(self, lr: float = 3e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps

    def apply(self, g: Graph, params: list[Tensor], grads: list[Tensor]) -> None:
        lr_t = g.input("opt.lr_t", (1, 1))
        for p, gr in zip(params, grads):
            m = g.param(f"{p.name}.adam_m", np.zeros(p.shape))
            v = g.param(f"{p.name}.adam_v", np.zeros(p.shape))
            m_new = g.add(g.scale(m, self.b1), g.scale(gr, 1 - self.b1))
            v_new = g.add(g.scale(v, self.b2), g.scale(g.square(gr), 1 - self.b2))
            g.assign(m, m_new)
            g.assign(v, v_new)
            upd = g.mul(g.div(m_new, g.scale(g.sqrt(v_new), 1.0, self.eps)), lr_t)
            g.assign(p, g.sub(p, upd))

    def host_feeds(self, step: int) -> dict:
        t = step + 1
        return {"opt.lr_t": np.array([[self.lr * math.sqrt(1 - self.b2 ** t) / (1 - self.b1 ** t)]])}

    def describe(self) -> dict:
        return {"name": "adam", "lr": self.lr, "beta1": self.b1, "beta2": self.b2, "eps": self.eps}


def make_optimizer(name: str, lr: float):
    if name == "sgd":
        return SGD(lr)
    if name == "adam":
        return Adam(lr)
    raise KeyError(f"unknown optimizer {name!r}")
