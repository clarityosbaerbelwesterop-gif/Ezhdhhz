"""Trusted host reference: evaluates a Graph with NumPy in float64 (int64 for integers).

Casts round to the target precision so that mixed-precision graphs are compared
against the mathematically intended computation, not against FP32 everywhere.
"""
from __future__ import annotations

import numpy as np

from ..dtypes import DType, convert, round_to, wrap_int32
from .graph import Graph, Tensor


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _gelu_grad(x):
    u = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)


_UN = {"relu": lambda x: np.maximum(x, 0), "gelu": _gelu, "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
       "tanh": np.tanh, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "rsqrt": lambda x: 1 / np.sqrt(x),
       "recip": lambda x: 1 / x, "step": lambda x: (x > 0).astype(np.float64), "gelu_grad": _gelu_grad,
       "square": np.square}


def _bmm(a, b, batch, ta, tb):
    ra, rb = a.shape[0] // batch, b.shape[0] // batch
    outs = []
    for i in range(batch):
        A = a[i * ra:(i + 1) * ra]
        Bm = b[i * rb:(i + 1) * rb]
        outs.append((A.T if ta else A) @ (Bm.T if tb else Bm))
    return np.vstack(outs)


def _row(fn, x, eps):
    if fn == "softmax":
        e = np.exp(x - x.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    if fn == "layernorm":
        xc = x - x.mean(1, keepdims=True)
        return xc / np.sqrt((xc * xc).mean(1, keepdims=True) + eps)
    return x / np.sqrt((x * x).mean(1, keepdims=True) + eps)


def evaluate(g: Graph, feeds: dict, params: dict | None = None) -> dict[int, np.ndarray]:
    """Return {tensor id: value} for every tensor of the graph."""
    env: dict[int, np.ndarray] = {}
    params = params or {}
    for t in g.tensors:
        if t.kind == "input":
            if t.name not in feeds:
                raise KeyError(f"missing feed {t.name!r}")
            v = np.asarray(feeds[t.name])
        elif t.kind == "param":
            v = np.asarray(params.get(t.name, t.value))
        elif t.kind == "const":
            v = t.value
        else:
            continue
        v = v.astype(np.int64 if t.dtype.is_int else np.float64).reshape(t.shape)
        if t.dtype.is_float:
            v = round_to(t.dtype, v.astype(np.float32)).astype(np.float64) if t.dtype != DType.FP32 else v
        env[t.id] = v
    for n in g.nodes:
        x = [env[i.id] for i in n.inputs]
        op, a = n.op, n.attrs
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if op in ("matmul", "bmm"):
                r = _bmm(x[0], x[1], a["batch"], a["ta"], a["tb"])
                if n.output.dtype.is_int:
                    r = wrap_int32(r).astype(np.int64)
            elif op == "add":
                r = x[0] + x[1]
            elif op == "sub":
                r = x[0] - x[1]
            elif op == "mul":
                r = x[0] * x[1]
            elif op == "div":
                r = x[0] / x[1]
            elif op == "maximum":
                r = np.maximum(x[0], x[1])
            elif op == "scale":
                r = a["alpha"] * x[0] + a["beta"]
            elif op in _UN:
                r = _UN[op](x[0])
            elif op == "cast":
                src, dst = n.inputs[0].dtype, n.output.dtype
                if dst.is_int:
                    r = convert(x[0].astype(np.float32 if src.is_float else np.int64), src if src.is_int else
                                DType.FP32, dst, a["scale"]).astype(np.int64)
                else:
                    r = round_to(dst, (x[0] * a["scale"]).astype(np.float32)).astype(np.float64) \
                        if dst != DType.FP32 else x[0] * a["scale"]
            elif op == "broadcast":
                r = np.broadcast_to(x[0], n.output.shape)
            elif op == "reduce":
                ax = {0: 0, 1: 1, 2: None}[a["axis"]]
                fn = {0: np.sum, 1: np.max, 2: np.mean}[int(a["rop"])]
                r = fn(x[0], axis=ax, keepdims=True)
            elif op in ("softmax", "layernorm", "rmsnorm"):
                r = _row(op, x[0], a["eps"])
            elif op == "transpose":
                r = x[0].T
            elif op == "xent":
                z, y = x[0], x[1]
                zc = z - z.max(1, keepdims=True)
                logp = zc - np.log(np.exp(zc).sum(1, keepdims=True))
                r = np.array([[-(y * logp).sum() / z.shape[0]]])
            else:
                raise NotImplementedError(op)
        env[n.output.id] = np.broadcast_to(np.asarray(r), n.output.shape).astype(
            np.int64 if n.output.dtype.is_int else np.float64)
    return env


def run_reference(g: Graph, feeds: dict, params: dict | None = None) -> tuple[dict, dict]:
    """(outputs by name, assigned param values by name)."""
    env = evaluate(g, feeds, params)
    outs = {k: env[t.id] for k, t in g.outputs.items()}
    assigned = {t.name: env[v.id] for t, v in g.assigns}
    return outs, assigned


def value_of(env: dict, t: Tensor) -> np.ndarray:
    return env[t.id]
