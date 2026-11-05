"""Convenience constructors for TITAN instructions (used by compiler, sequencer and tests)."""
from __future__ import annotations

from .spec import Act, Flag, Instruction, Norm, Opcode, ReduceOp, SyncOp, TensorRef, f2imm

_BA = {"row": Flag.BCAST_A_ROW, "col": Flag.BCAST_A_COL, "scalar": Flag.BCAST_A_SCALAR, None: Flag.NONE}
_BB = {"row": Flag.BCAST_B_ROW, "col": Flag.BCAST_B_COL, "scalar": Flag.BCAST_B_SCALAR, None: Flag.NONE}


def _n(notify: bool) -> int:
    return int(Flag.NOTIFY) if notify else 0


def nop(core: int = 0) -> Instruction:
    return Instruction(Opcode.NOP, core)


def load(core: int, dst: TensorRef, src: TensorRef, notify: bool = False) -> Instruction:
    return Instruction(Opcode.LOAD, core, (dst, src), flags=_n(notify))


def store(core: int, dst: TensorRef, src: TensorRef, notify: bool = False) -> Instruction:
    return Instruction(Opcode.STORE, core, (dst, src), flags=_n(notify))


def copy(core: int, dst: TensorRef, src: TensorRef, transpose: bool = False) -> Instruction:
    return Instruction(Opcode.COPY, core, (dst, src), flags=int(Flag.TRANSPOSE) if transpose else 0)


def matmul(core: int, c: TensorRef, a: TensorRef, b: TensorRef, acc: bool = False, ta: bool = False,
           tb: bool = False) -> Instruction:
    f = (Flag.ACC if acc else 0) | (Flag.TRANS_A if ta else 0) | (Flag.TRANS_B if tb else 0)
    return Instruction(Opcode.MATMUL, core, (c, a, b), flags=int(f))


def gemm(core: int, c: TensorRef, a: TensorRef, b: TensorRef, alpha: float = 1.0, beta: float = 0.0,
         ta: bool = False, tb: bool = False, notify: bool = False) -> Instruction:
    f = (Flag.TRANS_A if ta else 0) | (Flag.TRANS_B if tb else 0) | _n(notify)
    return Instruction(Opcode.GEMM, core, (c, a, b), (f2imm(alpha), f2imm(beta)), flags=int(f))


def vec(op: Opcode, core: int, dst: TensorRef, a: TensorRef, b: TensorRef, bcast_a=None, bcast_b=None) -> Instruction:
    return Instruction(op, core, (dst, a, b), flags=int(_BA[bcast_a] | _BB[bcast_b]))


def scale_add(core: int, dst: TensorRef, a: TensorRef, alpha: float, beta: float, b: TensorRef | None = None,
              bcast_a=None, bcast_b=None) -> Instruction:
    ops = (dst, a) if b is None else (dst, a, b)
    return Instruction(Opcode.VECTOR_SCALE_ADD, core, ops, (f2imm(alpha), f2imm(beta)),
                       flags=int(_BA[bcast_a] | _BB[bcast_b]))


def reduce(core: int, dst: TensorRef, src: TensorRef, op: ReduceOp, axis: int) -> Instruction:
    return Instruction(Opcode.REDUCE, core, (dst, src), (int(axis),), func=int(op))


def accumulate(core: int, dst: TensorRef, src: TensorRef) -> Instruction:
    return Instruction(Opcode.ACCUMULATE, core, (dst, src))


def activation(core: int, dst: TensorRef, src: TensorRef, fn: Act) -> Instruction:
    return Instruction(Opcode.ACTIVATION, core, (dst, src), func=int(fn))


def normalization(core: int, dst: TensorRef, src: TensorRef, fn: Norm, eps: float = 1e-5) -> Instruction:
    imm = () if fn == Norm.SOFTMAX else (f2imm(eps),)
    return Instruction(Opcode.NORMALIZATION, core, (dst, src), imm, func=int(fn))


def convert(core: int, dst: TensorRef, src: TensorRef, scale: float = 1.0) -> Instruction:
    return Instruction(Opcode.CONVERT, core, (dst, src), (f2imm(scale),))


def signal(core: int, sem: int, amount: int = 1) -> Instruction:
    return Instruction(Opcode.SYNCHRONIZE, core, imm=(sem, amount), func=int(SyncOp.SIGNAL))


def wait(core: int, sem: int, amount: int = 1) -> Instruction:
    return Instruction(Opcode.SYNCHRONIZE, core, imm=(sem, amount), func=int(SyncOp.WAIT))


def barrier(core: int = 0, first: int | None = None, count: int | None = None) -> Instruction:
    imm = () if first is None else (first, count)
    return Instruction(Opcode.BARRIER, core, imm=imm)


def execute_kernel(core: int, kernel_id: int, global_offset: int = 0) -> Instruction:
    imm = (kernel_id,) if not global_offset else (kernel_id, global_offset)
    return Instruction(Opcode.EXECUTE_KERNEL, core, imm=imm)


ACT = Act
NORM = Norm
REDUCE_OP = ReduceOp
