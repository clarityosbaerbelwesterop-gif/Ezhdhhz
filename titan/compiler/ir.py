"""TITAN IR: kernel-level intermediate representation between the graph and the ISA.

A Module is an ordered list of Kernels. Each kernel reads/writes global Buffers
and carries enough information (tiling-independent) to be lowered to ISA:
    gemm       C = op(A) @ op(B) [batched] followed by a fused element-wise epilogue
    ewise      fused chain of element-wise steps (with broadcast inputs)
    reduce     SUM / MAX / MEAN along an axis
    rownorm    softmax / layernorm / rmsnorm along rows
    transpose  2-D transpose
    copy       buffer-to-buffer copy (parameter updates)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..dtypes import DType, nbytes
from ..isa.spec import Space, TensorRef


@dataclass(eq=False)
class Buffer:
    name: str
    shape: tuple[int, int]
    dtype: DType
    kind: str = "temp"        # input | param | const | output | temp
    addr: int = -1

    @property
    def nbytes(self) -> int:
        return nbytes(self.dtype, self.shape[0] * self.shape[1])

    @property
    def ref(self) -> TensorRef:
        if self.addr < 0:
            raise RuntimeError(f"buffer {self.name} not allocated")
        return TensorRef(Space.GLOBAL, self.addr, self.shape[0], self.shape[1], self.dtype)

    @property
    def persistent(self) -> bool:
        return self.kind != "temp"

    def __repr__(self) -> str:
        loc = f"@{self.addr:#x}" if self.addr >= 0 else ""
        return f"{self.name}[{self.shape[0]}x{self.shape[1]}:{self.dtype.short}]{loc}"


@dataclass
class Step:
    op: str
    args: list            # ('in', i) external input, ('s', j) earlier step, ('acc',) gemm accumulator
    dtype: DType
    attrs: dict = field(default_factory=dict)


@dataclass(eq=False)
class Kernel:
    kind: str
    shape: tuple[int, int]
    inputs: list[Buffer] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)     # full | row | col | scalar (per input)
    steps: list[Step] = field(default_factory=list)
    output: Buffer | None = None
    attrs: dict = field(default_factory=dict)
    nodes: list[int] = field(default_factory=list)
    internal: dict = field(default_factory=dict)       # tensor id -> step ref
    head: object = None                                 # graph Tensor produced by the last step
    in_tensors: list = field(default_factory=list)

    def describe(self) -> str:
        ins = ", ".join(f"{b!r}" + (f"<{m}>" if m != "full" else "") for b, m in zip(self.inputs, self.modes))
        s = f"{self.kind:9s} {self.output!r} <- ({ins})"
        if self.attrs:
            s += " {" + ", ".join(f"{k}={v}" for k, v in self.attrs.items() if k != "a_t") + "}"
        if self.steps:
            s += " steps=[" + "; ".join(f"{st.op}{tuple(st.args)}" for st in self.steps) + "]"
        if len(self.nodes) > 1:
            s += f" fused_nodes={len(self.nodes)}"
        return s


@dataclass
class Module:
    name: str
    kernels: list[Kernel]
    buffers: list[Buffer]

    def dump(self) -> str:
        out = [f"module {self.name}"]
        out += [f"  buffer {b!r} kind={b.kind}" for b in self.buffers]
        out += [f"  k{i:03d} {k.describe()}" for i, k in enumerate(self.kernels)]
        return "\n".join(out)
