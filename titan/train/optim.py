"""Optimizers expressed as graph operations, so parameter updates execute on TITAN.

Host-side work per step: Adam's bias-corrected learning rate lr_t (one scalar).
"""
from __future__ import annotations

import math

import numpy as np

from ..compiler.graph import Graph, Tensor


def _state(g: Graph, name: str, like: Tensor) -> Tensor:
    """Zero-initialized optimizer state; lazy (device-side zeros) when the parameter itself is lazy."""
    if like.value is None:
        shape = like.shape
        return g.param_lazy(name, shape, lambda: np.zeros(shape, dtype=np.float32))
    return g.param(name, np.zeros(like.shape, dtype=np.float32))


class SGD:
    host_inputs: tuple = ()

    def __init__(self, lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 0.0):
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay

    def apply(self, g: Graph, params: list[Tensor], grads: list[Tensor]) -> None:
        for p, gr in zip(params, grads):
            if self.wd:
                gr = g.add(gr, g.scale(p, self.wd))
            if self.momentum:
                v = _state(g, f"{p.name}.momentum", p)
                gr = g.add(g.scale(v, self.momentum), gr)
                g.assign(v, gr)
            g.assign(p, g.sub(p, g.scale(gr, self.lr)))

    def host_feeds(self, step: int) -> dict:
        return {}

    def describe(self) -> dict:
        return {"name": "sgd", "lr": self.lr, "momentum": self.momentum, "weight_decay": self.wd}


class Adam:
    host_inputs = ("opt.lr_t",)

    def __init__(self, lr: float = 3e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 warmup: int = 0, schedule: str = "constant", total_steps: int = 0, min_lr_ratio: float = 0.1,
                 weight_decay: float = 0.0):
        self.lr, self.b1, self.b2, self.eps, self.warmup = lr, beta1, beta2, eps, warmup
        self.schedule, self.total, self.min_ratio, self.wd = schedule, total_steps, min_lr_ratio, weight_decay
        if weight_decay:
            self.host_inputs = ("opt.lr_t", "opt.wd_t")

    def apply(self, g: Graph, params: list[Tensor], grads: list[Tensor]) -> None:
        lr_t = g.input("opt.lr_t", (1, 1))
        wd_t = g.input("opt.wd_t", (1, 1)) if self.wd else None
        for p, gr in zip(params, grads):
            m = _state(g, f"{p.name}.adam_m", p)
            v = _state(g, f"{p.name}.adam_v", p)
            m_new = g.add(g.scale(m, self.b1), g.scale(gr, 1 - self.b1))
            v_new = g.add(g.scale(v, self.b2), g.scale(g.square(gr), 1 - self.b2))
            g.assign(m, m_new)
            g.assign(v, v_new)
            upd = g.mul(g.div(m_new, g.scale(g.sqrt(v_new), 1.0, self.eps)), lr_t)
            if self.wd and p.rows > 1 and p.cols > 1:       # decoupled weight decay on matrices (AdamW)
                upd = g.add(upd, g.mul(p, wd_t))
            g.assign(p, g.sub(p, upd))

    def lr_at(self, step: int) -> float:
        """Scheduled learning rate: linear warmup, then constant or cosine decay to min_lr_ratio * lr."""
        t = step + 1
        if self.warmup and t <= self.warmup:
            return self.lr * t / self.warmup
        if self.schedule == "cosine" and self.total:
            prog = min(1.0, (t - self.warmup) / max(1, self.total - self.warmup))
            return self.lr * (self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))
        return self.lr

    def host_feeds(self, step: int) -> dict:
        t = step + 1
        lr = self.lr_at(step)
        feeds = {"opt.lr_t": np.array([[lr * math.sqrt(1 - self.b2 ** t) / (1 - self.b1 ** t)]])}
        if self.wd:
            feeds["opt.wd_t"] = np.array([[lr * self.wd]])
        return feeds

    def describe(self) -> dict:
        return {"name": "adamw" if self.wd else "adam", "lr": self.lr, "beta1": self.b1, "beta2": self.b2,
                "eps": self.eps, "warmup_steps": self.warmup, "schedule": self.schedule,
                "min_lr_ratio": self.min_ratio, "weight_decay": self.wd}


def make_optimizer(name: str, lr: float, warmup: int = 0, schedule: str = "constant", total_steps: int = 0,
                   min_lr_ratio: float = 0.1, weight_decay: float = 0.0):
    if name == "sgd":
        return SGD(lr, weight_decay=weight_decay)
    if name == "adam":
        return Adam(lr, warmup=warmup, schedule=schedule, total_steps=total_steps, min_lr_ratio=min_lr_ratio,
                    weight_decay=weight_decay)
    raise KeyError(f"unknown optimizer {name!r}")
