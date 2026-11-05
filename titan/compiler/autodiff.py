"""Reverse-mode automatic differentiation on TITAN graphs.

Gradients are themselves graph nodes, so the backward pass is compiled to TITAN
ISA like the forward pass. Gradient tensors of float values are FP32; matmul
rules cast the upstream gradient to the operand precision (mixed precision:
BF16/FP16 operands, FP32 accumulation).
"""
from __future__ import annotations

from ..dtypes import DType
from .graph import DUAL, Graph, GraphError, Tensor


def sum_to(g: Graph, t: Tensor, shape) -> Tensor:
    if t.shape == tuple(shape):
        return t
    r, c = shape
    if (r, c) == (1, 1):
        return g.sum(t, 2)
    if r == 1:
        return g.sum(t, 0)
    if c == 1:
        return g.sum(t, 1)
    raise GraphError(f"cannot reduce {t.shape} to {shape}")


def _mm_rules(g: Graph, n, gy: Tensor, need: set):
    a, b = n.inputs
    ta, tb, bt = n.attrs["ta"], n.attrs["tb"], n.attrs["batch"]
    gyc = g.cast(gy, a.dtype)
    out = []
    if a.id in need:              # only build the products that are needed (e.g. not for one-hot inputs)
        out.append((a, g.bmm(gyc, b, bt, False, not tb) if not ta else g.bmm(b, gyc, bt, tb, True)))
    if b.id in need:
        out.append((b, g.bmm(a, gyc, bt, not ta, False) if not tb else g.bmm(gyc, a, bt, True, ta)))
    return out


def _rules(g: Graph, n, gy: Tensor, need: set = frozenset()):
    op, x, y = n.op, n.inputs, n.output
    if op in ("matmul", "bmm"):
        return _mm_rules(g, n, gy, need)
    if op == "add":
        return [(x[0], sum_to(g, gy, x[0].shape)), (x[1], sum_to(g, gy, x[1].shape))]
    if op == "sub":
        return [(x[0], sum_to(g, gy, x[0].shape)), (x[1], sum_to(g, g.scale(gy, -1.0), x[1].shape))]
    if op == "mul":
        return [(x[0], sum_to(g, g.mul(gy, x[1]), x[0].shape)), (x[1], sum_to(g, g.mul(gy, x[0]), x[1].shape))]
    if op == "div":
        ga = g.div(gy, x[1])
        gb = g.scale(g.div(g.mul(gy, x[0]), g.square(x[1])), -1.0)
        return [(x[0], sum_to(g, ga, x[0].shape)), (x[1], sum_to(g, gb, x[1].shape))]
    if op == "maximum":
        s = g.step(g.sub(x[0], x[1]))
        return [(x[0], sum_to(g, g.mul(gy, s), x[0].shape)),
                (x[1], sum_to(g, g.mul(gy, g.scale(s, -1.0, 1.0)), x[1].shape))]
    if op == "scale":
        return [(x[0], g.scale(gy, n.attrs["alpha"]))]
    if op == "relu":
        return [(x[0], g.mul(gy, g.step(x[0])))]
    if op == "gelu":
        return [(x[0], g.mul(gy, g.gelu_grad(x[0])))]
    if op == "sigmoid":
        return [(x[0], g.mul(gy, g.mul(y, g.scale(y, -1.0, 1.0))))]
    if op == "tanh":
        return [(x[0], g.mul(gy, g.scale(g.square(y), -1.0, 1.0)))]
    if op == "exp":
        return [(x[0], g.mul(gy, y))]
    if op == "log":
        return [(x[0], g.mul(gy, g.recip(x[0])))]
    if op == "sqrt":
        return [(x[0], g.mul(gy, g.scale(g.recip(y), 0.5)))]
    if op == "rsqrt":
        return [(x[0], g.mul(gy, g.scale(g.mul(g.square(y), y), -0.5)))]
    if op == "recip":
        return [(x[0], g.scale(g.mul(gy, g.square(y)), -1.0))]
    if op == "square":
        return [(x[0], g.mul(gy, g.scale(x[0], 2.0)))]
    if op in ("step", "gelu_grad"):
        return []
    if op == "cast":
        if x[0].dtype.is_int:
            return []
        return [(x[0], g.cast(gy, x[0].dtype) if x[0].dtype != DType.FP32 else gy)]
    if op == "broadcast":
        return [(x[0], sum_to(g, gy, x[0].shape))]
    if op == "reduce":
        rop, axis = int(n.attrs["rop"]), n.attrs["axis"]
        if rop == 1:
            raise GraphError("gradient of reduce-max is not supported")
        cnt = {0: x[0].rows, 1: x[0].cols, 2: x[0].rows * x[0].cols}[axis]
        gg = g.scale(gy, 1.0 / cnt) if rop == 2 else gy
        return [(x[0], g.broadcast(gg, x[0].shape))]
    if op == "softmax":
        dot = g.sum(g.mul(gy, y), 1)
        return [(x[0], g.mul(y, g.sub(gy, dot)))]
    if op == "layernorm":
        eps = n.attrs["eps"]
        mu = g.mean(x[0], 1)
        var = g.mean(g.square(g.sub(x[0], mu)), 1)
        rstd = g.rsqrt(g.scale(var, 1.0, eps))
        inner = g.sub(g.sub(gy, g.mean(gy, 1)), g.mul(y, g.mean(g.mul(gy, y), 1)))
        return [(x[0], g.mul(inner, rstd))]
    if op == "rmsnorm":
        eps = n.attrs["eps"]
        r = g.rsqrt(g.scale(g.mean(g.square(x[0]), 1), 1.0, eps))
        return [(x[0], g.mul(g.sub(gy, g.mul(y, g.mean(g.mul(gy, y), 1))), r))]
    if op == "transpose":
        return [(x[0], g.transpose(gy))]
    if op == "embedding":
        return [(x[1], g.embedding_grad(x[0], gy, x[1].rows))]
    if op == "embedding_grad":                       # scatter-add: gradient w.r.t. dy is a gather
        return [(x[1], g.embedding(x[0], gy))]
    if op in ("onehot", "moe_slots"):
        return []
    a = n.attrs
    if op == "split_heads":
        return [(x[0], g.merge_heads(gy, a["batch"], a["heads"]))]
    if op == "merge_heads":
        return [(x[0], g.split_heads(gy, a["batch"], a["heads"]))]
    if op == "repeat_heads":
        return [(x[0], g.sum_head_groups(gy, a["batch"], a["heads"], a["groups"]))]
    if op == "sum_head_groups":
        return [(x[0], g.repeat_heads(gy, a["batch"], a["heads"], a["groups"]))]
    if op == "xent":
        z, onehot = x[0], x[1]
        sm = g.softmax(z)
        if n.attrs.get("weighted"):
            sm = g.mul(sm, g.sum(onehot, 1))
        d = g.scale(g.sub(sm, onehot), 1.0 / z.rows)
        return [(z, g.mul(d, gy))]
    raise GraphError(f"no gradient rule for {op}")


def gradients(g: Graph, loss: Tensor, wrt: list[Tensor]) -> list[Tensor]:
    """Append backward nodes to ``g``; return d loss / d t for every t in ``wrt``."""
    if loss.shape != (1, 1):
        raise GraphError("loss must be a [1,1] tensor")
    need = {t.id for t in wrt}
    for nd in g.nodes:
        prim = nd.inputs[:2] if nd.op == "xent" else nd.inputs[:-1] if nd.op in DUAL else nd.inputs
        if any(i.id in need for i in prim):
            need.add(nd.output.id)
    if loss.id not in need:
        raise GraphError("loss does not depend on any requested tensor")
    grads: dict[int, Tensor] = {loss.id: g.const(1.0, name=f"dloss{len(g.tensors)}")}
    forward = [nd for nd in g.nodes if nd.output.id <= loss.id]
    for nd in reversed(forward):
        gy = grads.get(nd.output.id)
        if gy is None:
            continue
        for inp, gi in _rules(g, nd, gy, need):
            if inp.id not in need or inp.dtype.is_int:
                continue
            prev = grads.get(inp.id)
            grads[inp.id] = gi if prev is None else g.add(prev, gi)
    out = []
    for t in wrt:
        gt = grads.get(t.id)
        out.append(gt if gt is not None else g.scale(t, 0.0))
    return out
