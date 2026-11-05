"""Computation graph (front end of the TITAN compiler).

All tensors are 2-D ``[rows, cols]``. Batched matrix products use ``bmm`` on
row-stacked operands. The graph is built in topological order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dtypes import DType, accum_dtype
from ..isa.spec import Act, Norm, ReduceOp

UNARY = {"relu": Act.RELU, "gelu": Act.GELU, "sigmoid": Act.SIGMOID, "tanh": Act.TANH, "exp": Act.EXP,
         "log": Act.LOG, "sqrt": Act.SQRT, "rsqrt": Act.RSQRT, "recip": Act.RECIP, "step": Act.STEP,
         "gelu_grad": Act.GELU_GRAD, "square": Act.SQUARE}
BINARY = ("add", "sub", "mul", "div", "maximum")
EWISE = set(UNARY) | set(BINARY) | {"scale", "cast", "broadcast"}
ROWNORM = {"softmax": Norm.SOFTMAX, "layernorm": Norm.LAYERNORM, "rmsnorm": Norm.RMSNORM}
# ops with two lowerings: the last input is an exact decomposition into primitives (used for TITAN ISA);
# IR-level executors run them as dedicated kernels (gather / scatter-add / one-hot).
DUAL = {"onehot", "embedding", "embedding_grad"}
HEADS = {"split_heads", "merge_heads", "repeat_heads", "sum_head_groups"}
# data-dependent routing without an ISA lowering (TITAN IR executors only; models fall back to dense dispatch)
IR_ONLY = {"moe_slots"}


class GraphError(ValueError):
    pass


@dataclass(eq=False)
class Tensor:
    graph: "Graph"
    id: int
    shape: tuple[int, int]
    dtype: DType
    kind: str                       # input | param | const | value
    name: str = ""
    node: "Node | None" = None
    value: np.ndarray | None = None  # initial value for param / const

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    def __repr__(self) -> str:
        return f"%{self.id}<{self.name or self.kind}:{self.shape[0]}x{self.shape[1]}:{self.dtype.short}>"

    # sugar
    def __add__(self, o): return self.graph.add(self, o)
    def __sub__(self, o): return self.graph.sub(self, o)
    def __mul__(self, o): return self.graph.mul(self, o)
    def __truediv__(self, o): return self.graph.div(self, o)
    def __matmul__(self, o): return self.graph.matmul(self, o)
    def __neg__(self): return self.graph.scale(self, -1.0)

    @property
    def T(self): return self.graph.transpose(self)


@dataclass(eq=False)
class Node:
    id: int
    op: str
    inputs: list[Tensor]
    attrs: dict
    output: Tensor | None = None

    def __repr__(self) -> str:
        a = ", ".join(f"{k}={v}" for k, v in self.attrs.items())
        return f"{self.output} = {self.op}({', '.join(map(repr, self.inputs))}{'; ' + a if a else ''})"


def _bshape(a, b) -> tuple[int, int]:
    out = []
    for x, y in zip(a, b):
        if x != y and 1 not in (x, y):
            raise GraphError(f"shapes {a} and {b} do not broadcast")
        out.append(max(x, y))
    return tuple(out)


@dataclass
class Graph:
    name: str = "graph"
    tensors: list[Tensor] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    outputs: dict[str, Tensor] = field(default_factory=dict)
    assigns: list[tuple[Tensor, Tensor]] = field(default_factory=list)

    # ------------------------------------------------------------ leaves
    def _tensor(self, shape, dtype, kind, name="", value=None) -> Tensor:
        shape = tuple(int(s) for s in shape)
        if len(shape) != 2 or min(shape) <= 0:
            raise GraphError(f"tensors must be 2-D and non-empty, got {shape}")
        t = Tensor(self, len(self.tensors), shape, DType.parse(dtype), kind, name, value=value)
        self.tensors.append(t)
        return t

    def _unique(self, name: str) -> None:
        names = self.__dict__.setdefault("_names", set())
        if name in names:
            raise GraphError(f"duplicate tensor name {name!r}")
        names.add(name)

    def input(self, name: str, shape, dtype=DType.FP32) -> Tensor:
        self._unique(name)
        return self._tensor(shape, dtype, "input", name)

    def param(self, name: str, init, dtype=DType.FP32, meta: bool = False) -> Tensor:
        """Persistent device state. ``init`` is an array or a shape (zero init).

        ``meta=True`` records only the shape (no host memory is allocated); used to define and
        analyse models far larger than the host can hold (e.g. 100B parameters).
        """
        self._unique(name)
        if meta:
            shape = tuple(int(s) for s in (init if isinstance(init, tuple) else np.shape(init)))
            return self._tensor(shape, dtype, "param", name, value=None)
        arr = np.zeros(init) if isinstance(init, tuple) else np.asarray(init)
        if arr.ndim == 1:
            arr = arr[None, :]
        return self._tensor(arr.shape, dtype, "param", name, value=arr)

    def param_lazy(self, name: str, shape, initializer, dtype=DType.FP32) -> Tensor:
        """Parameter whose values are produced on the device by ``initializer()`` (never held by the graph).

        Used for models whose weights do not fit host memory: the runtime materializes one
        parameter at a time directly into its storage (RAM or a memory-mapped file).
        """
        t = self.param(name, tuple(shape), dtype, meta=True)
        t.initializer = initializer
        return t

    def matmul_flops(self) -> int:
        """Exact 2*M*N*K operation count of all (batched) matrix products in the graph."""
        total = 0
        for nd in self.nodes:
            if nd.op in ("matmul", "bmm") and not nd.attrs.get("impl"):
                a, b = nd.inputs[0], nd.inputs[1]
                bt = nd.attrs["batch"]
                k = a.rows // bt if nd.attrs["ta"] else a.cols
                total += 2 * nd.output.rows * nd.output.cols * k
        return total

    def param_count(self) -> int:
        return sum(t.rows * t.cols for t in self.params())

    def const(self, value, dtype=DType.FP32, name: str = "") -> Tensor:
        arr = np.asarray(value, dtype=np.float64 if DType.parse(dtype).is_float else np.int64)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr[None, :]
        name = name or f"const{len(self.tensors)}"
        self._unique(name)
        return self._tensor(arr.shape, dtype, "const", name, value=arr)

    def _op(self, op: str, inputs, shape, dtype, **attrs) -> Tensor:
        for t in inputs:
            if t.graph is not self:
                raise GraphError("tensor from another graph")
        node = Node(len(self.nodes), op, list(inputs), attrs)
        out = self._tensor(shape, dtype, "value")
        out.node = node
        node.output = out
        self.nodes.append(node)
        return out

    # ------------------------------------------------------------ matrix
    def matmul(self, a: Tensor, b: Tensor, ta: bool = False, tb: bool = False) -> Tensor:
        return self.bmm(a, b, 1, ta, tb)

    def bmm(self, a: Tensor, b: Tensor, batch: int, ta: bool = False, tb: bool = False) -> Tensor:
        """Batched product over row-stacked operands: a=[batch*Ra, Ca], b=[batch*Rb, Cb]."""
        if a.dtype != b.dtype:
            raise GraphError(f"matmul dtypes differ: {a.dtype.name} vs {b.dtype.name}")
        if a.rows % batch or b.rows % batch:
            raise GraphError("bmm: rows must be divisible by batch")
        ra, ca = a.rows // batch, a.cols
        rb, cb = b.rows // batch, b.cols
        m, k = (ca, ra) if ta else (ra, ca)
        k2, n = (cb, rb) if tb else (rb, cb)
        if k != k2:
            raise GraphError(f"matmul inner dims differ: {k} vs {k2}")
        op = "matmul" if batch == 1 else "bmm"
        return self._op(op, [a, b], (batch * m, n), accum_dtype(a.dtype), ta=ta, tb=tb, batch=batch)

    # ------------------------------------------------------------ element-wise
    def _binary(self, op, a, b):
        if not isinstance(b, Tensor):
            b = self.const(b, a.dtype)
        if not isinstance(a, Tensor):
            a = self.const(a, b.dtype)
        if a.dtype != b.dtype:
            raise GraphError(f"{op}: dtypes differ ({a.dtype.name}, {b.dtype.name})")
        return self._op(op, [a, b], _bshape(a.shape, b.shape), a.dtype)

    def add(self, a, b): return self._binary("add", a, b)
    def sub(self, a, b): return self._binary("sub", a, b)
    def mul(self, a, b): return self._binary("mul", a, b)
    def div(self, a, b): return self._binary("div", a, b)
    def maximum(self, a, b): return self._binary("maximum", a, b)

    def scale(self, x: Tensor, alpha: float, beta: float = 0.0) -> Tensor:
        """alpha * x + beta."""
        if x.dtype.is_int:
            raise GraphError("scale requires a float tensor")
        return self._op("scale", [x], x.shape, x.dtype, alpha=float(alpha), beta=float(beta))

    def _unary(self, op, x):
        if x.dtype.is_int:
            raise GraphError(f"{op} requires a float tensor")
        return self._op(op, [x], x.shape, x.dtype)

    def relu(self, x): return self._unary("relu", x)
    def gelu(self, x): return self._unary("gelu", x)
    def sigmoid(self, x): return self._unary("sigmoid", x)
    def tanh(self, x): return self._unary("tanh", x)
    def exp(self, x): return self._unary("exp", x)
    def log(self, x): return self._unary("log", x)
    def sqrt(self, x): return self._unary("sqrt", x)
    def rsqrt(self, x): return self._unary("rsqrt", x)
    def recip(self, x): return self._unary("recip", x)
    def step(self, x): return self._unary("step", x)
    def gelu_grad(self, x): return self._unary("gelu_grad", x)
    def square(self, x): return self._unary("square", x)

    def cast(self, x: Tensor, dtype, scale: float = 1.0) -> Tensor:
        dtype = DType.parse(dtype)
        if dtype == x.dtype and scale == 1.0:
            return x
        return self._op("cast", [x], x.shape, dtype, scale=float(scale))

    def broadcast(self, x: Tensor, shape) -> Tensor:
        shape = tuple(shape)
        if _bshape(x.shape, shape) != shape:
            raise GraphError(f"cannot broadcast {x.shape} to {shape}")
        if x.shape == shape:
            return x
        return self._op("broadcast", [x], shape, x.dtype)

    # ------------------------------------------------------------ reductions / rows
    def reduce(self, x: Tensor, op: str | ReduceOp, axis: int) -> Tensor:
        op = ReduceOp[op.upper()] if isinstance(op, str) else ReduceOp(op)
        if x.dtype not in (DType.FP32, DType.INT32):
            raise GraphError("reduce requires FP32 or INT32 input (cast first)")
        if op == ReduceOp.MEAN and x.dtype.is_int:
            raise GraphError("mean requires a float tensor")
        shape = {0: (1, x.cols), 1: (x.rows, 1), 2: (1, 1)}[axis]
        return self._op("reduce", [x], shape, x.dtype, rop=op, axis=axis)

    def sum(self, x, axis=2): return self.reduce(x, ReduceOp.SUM, axis)
    def mean(self, x, axis=2): return self.reduce(x, ReduceOp.MEAN, axis)
    def max(self, x, axis=2): return self.reduce(x, ReduceOp.MAX, axis)

    def _rownorm(self, op, x, eps):
        if x.dtype.is_int:
            raise GraphError(f"{op} requires a float tensor")
        return self._op(op, [x], x.shape, x.dtype, eps=float(eps))

    def softmax(self, x): return self._rownorm("softmax", x, 0.0)
    def layernorm(self, x, eps=1e-5): return self._rownorm("layernorm", x, eps)
    def rmsnorm(self, x, eps=1e-5): return self._rownorm("rmsnorm", x, eps)

    def transpose(self, x: Tensor) -> Tensor:
        return self._op("transpose", [x], (x.cols, x.rows), x.dtype)

    # ------------------------------------------------------------ tokens and attention heads
    def onehot(self, ids: Tensor, n: int) -> Tensor:
        """ids [N,1] INT32 -> one-hot [N,n] FP32 (no gradient)."""
        if ids.cols != 1 or not ids.dtype.is_int:
            raise GraphError("onehot expects integer ids of shape [N,1]")
        iota = self.const(np.arange(n)[None, :], name=f"iota{n}_{len(self.tensors)}")
        d = self.sub(self.cast(ids, DType.FP32), iota)
        impl = self.step(self.scale(self.square(d), -1.0, 0.25))          # 1 exactly where id == column
        return self._op("onehot", [ids, impl], (ids.rows, n), DType.FP32, n=n)

    def embedding(self, ids: Tensor, table: Tensor) -> Tensor:
        """Row lookup table[ids] -> [N, d] (gather on IR executors; one-hot product on TITAN ISA)."""
        oh = self.onehot(ids, table.rows)
        impl = self.matmul(oh, table)
        impl.node.attrs["impl"] = True
        return self._op("embedding", [ids, table, impl], (ids.rows, table.cols), table.dtype)

    def embedding_grad(self, ids: Tensor, dy: Tensor, vocab: int) -> Tensor:
        """Scatter-add of dy rows into a [vocab, d] gradient."""
        oh = self.onehot(ids, vocab)
        impl = self.matmul(oh, dy, ta=True)
        impl.node.attrs["impl"] = True
        return self._op("embedding_grad", [ids, dy, impl], (vocab, dy.cols), DType.FP32, vocab=vocab)

    def moe_slots(self, mask: Tensor, capacity: int) -> Tensor:
        """Expert routing table: mask [N, E] (1 = token n routed to expert e) -> token ids [E*capacity, 1]
        (INT32), expert-major, tokens in order; slots beyond an expert's load hold -1 (empty), tokens beyond
        its capacity are dropped. Gathers of id -1 give zero rows and scatter-adds skip them. No gradient."""
        if mask.dtype != DType.FP32 or capacity <= 0:
            raise GraphError("moe_slots expects an FP32 mask and capacity > 0")
        return self._op("moe_slots", [mask], (mask.cols * capacity, 1), DType.INT32, capacity=capacity,
                        experts=mask.cols)

    def split_heads(self, x: Tensor, batch: int, heads: int) -> Tensor:
        """[B*T, H*dh] -> [B*H*T, dh] (head-major rows, for batched attention)."""
        if x.rows % batch or x.cols % heads:
            raise GraphError("split_heads: shape not divisible")
        return self._op("split_heads", [x], (x.rows * heads, x.cols // heads), x.dtype, batch=batch, heads=heads)

    def merge_heads(self, x: Tensor, batch: int, heads: int) -> Tensor:
        """[B*H*T, dh] -> [B*T, H*dh]."""
        if x.rows % (batch * heads):
            raise GraphError("merge_heads: shape not divisible")
        return self._op("merge_heads", [x], (x.rows // heads, x.cols * heads), x.dtype, batch=batch, heads=heads)

    def repeat_heads(self, x: Tensor, batch: int, kv_heads: int, groups: int) -> Tensor:
        """[B*KV*T, dh] -> [B*KV*G*T, dh]: each key/value head serves G query heads (grouped-query attention)."""
        if groups == 1:
            return x
        return self._op("repeat_heads", [x], (x.rows * groups, x.cols), x.dtype, batch=batch, heads=kv_heads,
                        groups=groups)

    def sum_head_groups(self, x: Tensor, batch: int, kv_heads: int, groups: int) -> Tensor:
        """Inverse reduction of repeat_heads (its gradient): [B*KV*G*T, dh] -> [B*KV*T, dh]."""
        if groups == 1:
            return x
        return self._op("sum_head_groups", [x], (x.rows // groups, x.cols), x.dtype, batch=batch, heads=kv_heads,
                        groups=groups)

    def xent(self, logits: Tensor, onehot: Tensor, weighted: bool = False) -> Tensor:
        """Mean softmax cross-entropy over rows -> [1,1].

        Composite op: the forward pass is built from primitives (numerically stable
        log-softmax); autodiff uses the fused rule d/dz = (softmax(z) - y) / rows.
        ``weighted``: target rows may be scaled or zero (loss masking, e.g. prompt tokens in SFT);
        the rule becomes d/dz = (softmax(z) * rowsum(y) - y) / rows.
        """
        if logits.shape != onehot.shape or logits.dtype != DType.FP32 or onehot.dtype != DType.FP32:
            raise GraphError("xent: logits and onehot must be FP32 with equal shapes")
        zc = self.sub(logits, self.max(logits, 1))
        logp = self.sub(zc, self.log(self.sum(self.exp(zc), 1)))
        impl = self.scale(self.sum(self.mul(onehot, logp), 2), -1.0 / logits.rows)
        return self._op("xent", [logits, onehot, impl], (1, 1), DType.FP32, weighted=weighted)

    # ------------------------------------------------------------ graph io
    def output(self, name: str, t: Tensor) -> Tensor:
        if name in self.outputs:
            raise GraphError(f"duplicate output {name!r}")
        self.outputs[name] = t
        return t

    def assign(self, target: Tensor, value: Tensor) -> None:
        """Write ``value`` into param ``target`` at the end of the program (optimizer updates)."""
        if target.kind != "param" or target.shape != value.shape or target.dtype != value.dtype:
            raise GraphError("assign target must be a param with identical shape and dtype")
        self.assigns.append((target, value))

    def params(self) -> list[Tensor]:
        return [t for t in self.tensors if t.kind == "param"]

    def inputs(self) -> list[Tensor]:
        return [t for t in self.tensors if t.kind == "input"]

    def dump(self) -> str:
        lines = [f"graph {self.name}"]
        lines += [f"  {t!r}" for t in self.tensors if t.kind != "value"]
        lines += [f"  {n!r}" for n in self.nodes]
        lines += [f"  output {k} = {v!r}" for k, v in self.outputs.items()]
        lines += [f"  assign {t!r} <- {v!r}" for t, v in self.assigns]
        return "\n".join(lines)
