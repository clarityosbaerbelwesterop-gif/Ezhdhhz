"""TITAN ISA v1.0 — opcodes, operand model, instruction record and validation rules.

The normative prose specification is docs/ISA.md; this module is the executable
form of it. ``tests/test_isa.py`` checks that both stay in sync.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag

from ..dtypes import DType, nbytes

ISA_VERSION = (1, 0)


class Space(IntEnum):
    GLOBAL = 0   # global memory (HBM abstraction), all cores
    SHARED = 1   # on-chip shared SRAM, all cores
    LOCAL = 2    # scratchpad SRAM private to the executing core

    @property
    def tag(self) -> str:
        return "GSL"[self]


class Unit(IntEnum):
    CTRL = 0
    DMA = 1
    MXU = 2
    VPU = 3
    SFU = 4


class Opcode(IntEnum):
    NOP = 0x00
    LOAD = 0x01
    STORE = 0x02
    COPY = 0x03
    MATMUL = 0x10
    GEMM = 0x11
    VECTOR_ADD = 0x20
    VECTOR_MULTIPLY = 0x21
    VECTOR_SUB = 0x22
    VECTOR_MAX = 0x23
    VECTOR_DIV = 0x24
    VECTOR_SCALE_ADD = 0x25
    REDUCE = 0x30
    ACCUMULATE = 0x31
    ACTIVATION = 0x40
    NORMALIZATION = 0x41
    CONVERT = 0x42
    SYNCHRONIZE = 0x50
    BARRIER = 0x51
    EXECUTE_KERNEL = 0x60


class Flag(IntFlag):
    NONE = 0
    ACC = 1 << 0             # MATMUL: C += A@B
    TRANS_A = 1 << 1
    TRANS_B = 1 << 2
    TRANSPOSE = 1 << 3       # COPY: dst = src^T
    BCAST_A_ROW = 1 << 4     # operand a is [1,N], broadcast over rows
    BCAST_A_COL = 1 << 5     # operand a is [M,1], broadcast over columns
    BCAST_A_SCALAR = 1 << 6  # operand a is [1,1]
    BCAST_B_ROW = 1 << 7
    BCAST_B_COL = 1 << 8
    BCAST_B_SCALAR = 1 << 9
    NOTIFY = 1 << 15         # post a completion record when the instruction retires


class Act(IntEnum):
    RELU = 0
    GELU = 1        # tanh approximation
    SIGMOID = 2
    TANH = 3
    EXP = 4
    LOG = 5
    SQRT = 6
    RSQRT = 7
    RECIP = 8
    STEP = 9        # 1 if x > 0 else 0 (ReLU derivative)
    GELU_GRAD = 10  # derivative of the tanh-approximated GELU
    SQUARE = 11
    IDENTITY = 12


class Norm(IntEnum):
    SOFTMAX = 0     # row-wise, max-subtracted
    LAYERNORM = 1   # row-wise, no affine; imm0 = eps
    RMSNORM = 2     # row-wise, no affine; imm0 = eps


class ReduceOp(IntEnum):
    SUM = 0
    MAX = 1
    MEAN = 2


class Axis(IntEnum):
    ROWS = 0   # reduce over rows    -> [1, N]
    COLS = 1   # reduce over columns -> [M, 1]
    ALL = 2    #                      -> [1, 1]


class SyncOp(IntEnum):
    SIGNAL = 0
    WAIT = 1


class ErrorCode(IntEnum):
    ILLEGAL_OPCODE = 1
    BAD_ENCODING = 2
    OPERAND_COUNT = 3
    ILLEGAL_SPACE = 4
    OUT_OF_BOUNDS = 5
    MISALIGNED = 6
    DTYPE_MISMATCH = 7
    SHAPE_MISMATCH = 8
    BAD_FUNC = 9
    BAD_FLAGS = 10
    BAD_CORE = 11
    BAD_SEMAPHORE = 12
    KERNEL_NOT_FOUND = 13
    KERNEL_DEPTH = 14
    DEADLOCK = 15
    NUMERIC = 16
    BAD_IMMEDIATE = 17


class ISAError(Exception):
    def __init__(self, code: ErrorCode, msg: str):
        super().__init__(f"{code.name}: {msg}")
        self.code = code


# ------------------------------------------------------------------ operands

@dataclass(frozen=True)
class TensorRef:
    """A 2-D strided tensor in one address space. ``stride`` is in elements."""
    space: Space
    addr: int
    rows: int
    cols: int
    dtype: DType
    stride: int = 0

    def __post_init__(self):
        object.__setattr__(self, "space", Space(self.space))
        object.__setattr__(self, "dtype", DType(self.dtype))
        if self.stride == 0:
            object.__setattr__(self, "stride", self.cols)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def numel(self) -> int:
        return self.rows * self.cols

    @property
    def nbytes(self) -> int:
        """Logical payload bytes (what a DMA moves)."""
        return nbytes(self.dtype, self.numel)

    @property
    def span(self) -> int:
        """Bytes between the first and one-past-last byte touched."""
        return nbytes(self.dtype, (self.rows - 1) * self.stride + self.cols)

    @property
    def end(self) -> int:
        return self.addr + self.span

    @property
    def contiguous(self) -> bool:
        return self.stride == self.cols or self.rows == 1

    def sub(self, r0: int, c0: int, rows: int, cols: int) -> "TensorRef":
        """View of a sub-block (no data movement)."""
        if r0 < 0 or c0 < 0 or r0 + rows > self.rows or c0 + cols > self.cols:
            raise ValueError(f"sub-block ({r0},{c0},{rows},{cols}) outside {self.shape}")
        eo = r0 * self.stride + c0
        if eo * self.dtype.bits % 8:
            raise ValueError("INT4 sub-block must start on a byte boundary")
        off = eo * self.dtype.bits // 8
        return TensorRef(self.space, self.addr + off, rows, cols, self.dtype, self.stride)

    def with_addr(self, addr: int) -> "TensorRef":
        return TensorRef(self.space, addr, self.rows, self.cols, self.dtype, self.stride)

    def __str__(self) -> str:
        st = "" if self.stride == self.cols else f"/{self.stride}"
        return f"{self.space.tag}:{self.addr:#x}:{self.rows}x{self.cols}{st}:{self.dtype.short}"


# ------------------------------------------------------------------ instruction

def f2imm(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def imm2f(i: int) -> float:
    return struct.unpack("<f", struct.pack("<I", i & 0xFFFFFFFF))[0]


@dataclass(frozen=True)
class Instruction:
    opcode: Opcode
    core: int = 0
    operands: tuple[TensorRef, ...] = ()
    imm: tuple[int, ...] = ()
    func: int = 0
    flags: int = 0

    def __post_init__(self):
        object.__setattr__(self, "opcode", Opcode(self.opcode))
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "imm", tuple(int(i) for i in self.imm))
        object.__setattr__(self, "flags", int(self.flags))

    @property
    def spec(self) -> "OpSpec":
        return SPEC[self.opcode]

    def fimm(self, i: int, default: float = 0.0) -> float:
        return imm2f(self.imm[i]) if i < len(self.imm) else default

    def iimm(self, i: int, default: int = 0) -> int:
        return self.imm[i] if i < len(self.imm) else default

    def has(self, f: Flag) -> bool:
        return bool(self.flags & f)

    def with_core(self, core: int) -> "Instruction":
        return Instruction(self.opcode, core, self.operands, self.imm, self.func, self.flags)


# ------------------------------------------------------------------ spec table

@dataclass(frozen=True)
class OpSpec:
    name: str
    unit: Unit
    operands: tuple[int, int]          # (min, max) tensor operands
    imm: tuple[int, int]               # (min, max) immediates
    imm_kind: str = ""                 # per-immediate kind: 'f' float32, 'i' integer
    flags: int = Flag.NOTIFY
    func: type | None = None
    roles: tuple[str, ...] = ()
    summary: str = ""


_BCAST = (Flag.BCAST_A_ROW | Flag.BCAST_A_COL | Flag.BCAST_A_SCALAR |
          Flag.BCAST_B_ROW | Flag.BCAST_B_COL | Flag.BCAST_B_SCALAR)
_N = Flag.NOTIFY

SPEC: dict[Opcode, OpSpec] = {
    Opcode.NOP: OpSpec("NOP", Unit.CTRL, (0, 0), (0, 0), summary="No operation."),
    Opcode.LOAD: OpSpec("LOAD", Unit.DMA, (2, 2), (0, 0), roles=("dst", "src"),
                        summary="DMA GLOBAL|SHARED -> LOCAL|SHARED, same shape and dtype."),
    Opcode.STORE: OpSpec("STORE", Unit.DMA, (2, 2), (0, 0), roles=("dst", "src"),
                         summary="DMA LOCAL|SHARED -> GLOBAL|SHARED, same shape and dtype."),
    Opcode.COPY: OpSpec("COPY", Unit.DMA, (2, 2), (0, 0), flags=_N | Flag.TRANSPOSE, roles=("dst", "src"),
                        summary="DMA any space -> any space; optional 2-D transpose."),
    Opcode.MATMUL: OpSpec("MATMUL", Unit.MXU, (3, 3), (0, 0), flags=_N | Flag.ACC | Flag.TRANS_A | Flag.TRANS_B,
                          roles=("c", "a", "b"), summary="C (+)= op(A) @ op(B) on the matrix engine, LOCAL operands."),
    Opcode.GEMM: OpSpec("GEMM", Unit.MXU, (3, 3), (2, 2), "ff", flags=_N | Flag.TRANS_A | Flag.TRANS_B,
                        roles=("c", "a", "b"),
                        summary="C = alpha*op(A)@op(B) + beta*C on GLOBAL operands (sequencer macro-op)."),
    Opcode.VECTOR_ADD: OpSpec("VECTOR_ADD", Unit.VPU, (3, 3), (0, 0), flags=_N | _BCAST, roles=("dst", "a", "b"),
                              summary="dst = a + b (broadcasting)."),
    Opcode.VECTOR_MULTIPLY: OpSpec("VECTOR_MULTIPLY", Unit.VPU, (3, 3), (0, 0), flags=_N | _BCAST,
                                   roles=("dst", "a", "b"), summary="dst = a * b (broadcasting)."),
    Opcode.VECTOR_SUB: OpSpec("VECTOR_SUB", Unit.VPU, (3, 3), (0, 0), flags=_N | _BCAST, roles=("dst", "a", "b"),
                              summary="dst = a - b (broadcasting)."),
    Opcode.VECTOR_MAX: OpSpec("VECTOR_MAX", Unit.VPU, (3, 3), (0, 0), flags=_N | _BCAST, roles=("dst", "a", "b"),
                              summary="dst = max(a, b) (broadcasting)."),
    Opcode.VECTOR_DIV: OpSpec("VECTOR_DIV", Unit.VPU, (3, 3), (0, 0), flags=_N | _BCAST, roles=("dst", "a", "b"),
                              summary="dst = a / b (float only, broadcasting)."),
    Opcode.VECTOR_SCALE_ADD: OpSpec("VECTOR_SCALE_ADD", Unit.VPU, (2, 3), (2, 2), "ff", flags=_N | _BCAST,
                                    roles=("dst", "a", "b"),
                                    summary="dst = alpha*a + beta*b, or alpha*a + beta without b (float only)."),
    Opcode.REDUCE: OpSpec("REDUCE", Unit.VPU, (2, 2), (1, 1), "i", func=ReduceOp, roles=("dst", "src"),
                          summary="dst = reduce_op(src) along imm0 axis (0 rows, 1 cols, 2 all)."),
    Opcode.ACCUMULATE: OpSpec("ACCUMULATE", Unit.VPU, (2, 2), (0, 0), roles=("dst", "src"),
                              summary="dst += src in accumulator precision (INT32 wrap / FP32)."),
    Opcode.ACTIVATION: OpSpec("ACTIVATION", Unit.SFU, (2, 2), (0, 0), func=Act, roles=("dst", "src"),
                              summary="dst = f(src), element-wise (float only)."),
    Opcode.NORMALIZATION: OpSpec("NORMALIZATION", Unit.SFU, (2, 2), (0, 1), "f", func=Norm, roles=("dst", "src"),
                                 summary="Row-wise softmax / layernorm / rmsnorm (float only); imm0 = eps."),
    Opcode.CONVERT: OpSpec("CONVERT", Unit.VPU, (2, 2), (0, 1), "f", roles=("dst", "src"),
                           summary="dst = round_dst(src * scale); integer destinations saturate."),
    Opcode.SYNCHRONIZE: OpSpec("SYNCHRONIZE", Unit.CTRL, (0, 0), (2, 2), "ii", func=SyncOp,
                               summary="SIGNAL: sem[imm0] += imm1 after all prior work of the core retires; "
                                       "WAIT: block until sem[imm0] >= imm1, then subtract imm1."),
    Opcode.BARRIER: OpSpec("BARRIER", Unit.CTRL, (0, 0), (0, 2), "ii",
                           summary="All cores in [imm0, imm0+imm1) (default: all) wait until every one arrives."),
    Opcode.EXECUTE_KERNEL: OpSpec("EXECUTE_KERNEL", Unit.CTRL, (0, 0), (1, 2), "ii",
                                  summary="Run kernel imm0 on this core; imm1 = byte offset added to GLOBAL addresses."),
}

FLOAT_TYPES = (DType.FP16, DType.BF16, DType.FP32)
VECTOR_TYPES = (DType.INT32, DType.FP16, DType.BF16, DType.FP32)
MXU_INPUT_TYPES = (DType.INT4, DType.INT8, DType.FP16, DType.BF16, DType.FP32)
BINARY_VECTOR = (Opcode.VECTOR_ADD, Opcode.VECTOR_MULTIPLY, Opcode.VECTOR_SUB, Opcode.VECTOR_MAX,
                 Opcode.VECTOR_DIV, Opcode.VECTOR_SCALE_ADD)


# ------------------------------------------------------------------ validation

def _err(code: ErrorCode, ins: Instruction, msg: str):
    raise ISAError(code, f"{ins.opcode.name}@{ins.core}: {msg}")


def check_ref(ref: TensorRef, cfg, ins: Instruction) -> None:
    if ref.rows <= 0 or ref.cols <= 0:
        _err(ErrorCode.SHAPE_MISMATCH, ins, f"empty tensor {ref}")
    if ref.rows > 1 and ref.stride < ref.cols:
        _err(ErrorCode.SHAPE_MISMATCH, ins, f"stride < cols in {ref}")
    if ref.dtype == DType.INT4:
        if ref.cols % 2 or (ref.rows > 1 and ref.stride % 2):
            _err(ErrorCode.MISALIGNED, ins, f"INT4 tensors need even cols/stride: {ref}")
    elif ref.addr % (ref.dtype.bits // 8):
        _err(ErrorCode.MISALIGNED, ins, f"address not aligned to element size: {ref}")
    if cfg is not None:
        cap = {Space.GLOBAL: cfg.global_bytes, Space.SHARED: cfg.shared_bytes, Space.LOCAL: cfg.local_bytes}[ref.space]
        if ref.addr < 0 or ref.end > cap:
            _err(ErrorCode.OUT_OF_BOUNDS, ins, f"{ref} exceeds {ref.space.name} capacity {cap}")


def _bshape(ref: TensorRef, ins: Instruction, which: str, full: tuple[int, int]) -> None:
    row = Flag.BCAST_A_ROW if which == "a" else Flag.BCAST_B_ROW
    col = Flag.BCAST_A_COL if which == "a" else Flag.BCAST_B_COL
    sca = Flag.BCAST_A_SCALAR if which == "a" else Flag.BCAST_B_SCALAR
    sel = [f for f in (row, col, sca) if ins.flags & f]
    if len(sel) > 1:
        _err(ErrorCode.BAD_FLAGS, ins, f"operand {which}: more than one broadcast mode")
    want = full
    if sel:
        want = {row: (1, full[1]), col: (full[0], 1), sca: (1, 1)}[sel[0]]
    if ref.shape != want:
        _err(ErrorCode.SHAPE_MISMATCH, ins, f"operand {which} shape {ref.shape} != {want}")


def validate(ins: Instruction, cfg=None, kernel_ids=None) -> None:
    """Static validation of one instruction. Raises ISAError on the first violation."""
    if ins.opcode not in SPEC:
        _err(ErrorCode.ILLEGAL_OPCODE, ins, "unknown opcode")
    sp = SPEC[ins.opcode]
    n, ni = len(ins.operands), len(ins.imm)
    if not sp.operands[0] <= n <= sp.operands[1]:
        _err(ErrorCode.OPERAND_COUNT, ins, f"{n} operands, expected {sp.operands}")
    if not sp.imm[0] <= ni <= sp.imm[1]:
        _err(ErrorCode.OPERAND_COUNT, ins, f"{ni} immediates, expected {sp.imm}")
    if ins.flags & ~int(sp.flags):
        _err(ErrorCode.BAD_FLAGS, ins, f"flags {ins.flags:#x} not allowed (mask {int(sp.flags):#x})")
    if sp.func is not None:
        if ins.func not in {int(v) for v in sp.func}:
            _err(ErrorCode.BAD_FUNC, ins, f"func {ins.func} not in {sp.func.__name__}")
    elif ins.func != 0:
        _err(ErrorCode.BAD_FUNC, ins, "func must be 0")
    if cfg is not None and not 0 <= ins.core < cfg.cores:
        _err(ErrorCode.BAD_CORE, ins, f"core {ins.core} outside 0..{cfg.cores - 1}")
    for r in ins.operands:
        check_ref(r, cfg, ins)

    op, o = ins.opcode, ins.operands

    def spaces(ref, allowed, role):
        if ref.space not in allowed:
            _err(ErrorCode.ILLEGAL_SPACE, ins, f"{role} in {ref.space.name}, allowed {[s.name for s in allowed]}")

    def same(a, b, what="shape"):
        if a.shape != b.shape:
            _err(ErrorCode.SHAPE_MISMATCH, ins, f"{what} {a.shape} != {b.shape}")

    def dtypes(ref, allowed, role):
        if ref.dtype not in allowed:
            _err(ErrorCode.DTYPE_MISMATCH, ins, f"{role} dtype {ref.dtype.name} not allowed")

    L, G, S = Space.LOCAL, Space.GLOBAL, Space.SHARED
    if op in (Opcode.LOAD, Opcode.STORE, Opcode.COPY):
        dst, src = o
        if op == Opcode.LOAD:
            spaces(dst, (L, S), "dst"); spaces(src, (G, S), "src")
        elif op == Opcode.STORE:
            spaces(dst, (G, S), "dst"); spaces(src, (L, S), "src")
        if dst.dtype != src.dtype:
            _err(ErrorCode.DTYPE_MISMATCH, ins, "DMA does not convert; use CONVERT")
        if ins.has(Flag.TRANSPOSE):
            if dst.shape != (src.cols, src.rows):
                _err(ErrorCode.SHAPE_MISMATCH, ins, f"transpose dst {dst.shape} != {(src.cols, src.rows)}")
            if src.dtype == DType.INT4:
                _err(ErrorCode.DTYPE_MISMATCH, ins, "INT4 transpose unsupported")
        else:
            same(dst, src)
    elif op == Opcode.MATMUL or op == Opcode.GEMM:
        c, a, b = o
        want = L if op == Opcode.MATMUL else G
        for r, role in ((c, "c"), (a, "a"), (b, "b")):
            spaces(r, (want,), role)
        dtypes(a, MXU_INPUT_TYPES, "a")
        if a.dtype != b.dtype:
            _err(ErrorCode.DTYPE_MISMATCH, ins, "a and b must share a dtype")
        acc = DType.INT32 if a.dtype.is_int else DType.FP32
        if c.dtype != acc:
            _err(ErrorCode.DTYPE_MISMATCH, ins, f"c must be {acc.name} for {a.dtype.name} inputs")
        m, k = (a.cols, a.rows) if ins.has(Flag.TRANS_A) else a.shape
        k2, nn = (b.cols, b.rows) if ins.has(Flag.TRANS_B) else b.shape
        if k != k2 or c.shape != (m, nn):
            _err(ErrorCode.SHAPE_MISMATCH, ins, f"[{m}x{k}] @ [{k2}x{nn}] -> {c.shape}")
        if op == Opcode.GEMM and a.dtype.is_int:
            alpha, beta = ins.fimm(0), ins.fimm(1)
            if alpha != 1.0 or beta not in (0.0, 1.0):
                _err(ErrorCode.BAD_IMMEDIATE, ins, "integer GEMM requires alpha=1, beta in {0,1}")
    elif op in BINARY_VECTOR:
        for r in o:
            spaces(r, (L,), "operand")
        dst = o[0]
        dtypes(dst, FLOAT_TYPES if op in (Opcode.VECTOR_DIV, Opcode.VECTOR_SCALE_ADD) else VECTOR_TYPES, "dst")
        for r in o[1:]:
            if r.dtype != dst.dtype:
                _err(ErrorCode.DTYPE_MISMATCH, ins, "all operands must share the dst dtype")
        _bshape(o[1], ins, "a", dst.shape)
        if len(o) == 3:
            _bshape(o[2], ins, "b", dst.shape)
        elif op != Opcode.VECTOR_SCALE_ADD:
            _err(ErrorCode.OPERAND_COUNT, ins, "binary op needs 3 operands")
        elif ins.flags & (Flag.BCAST_B_ROW | Flag.BCAST_B_COL | Flag.BCAST_B_SCALAR):
            _err(ErrorCode.BAD_FLAGS, ins, "no operand b to broadcast")
    elif op == Opcode.REDUCE:
        dst, src = o
        spaces(dst, (L,), "dst"); spaces(src, (L,), "src")
        dtypes(src, VECTOR_TYPES, "src")
        if dst.dtype != src.dtype:
            _err(ErrorCode.DTYPE_MISMATCH, ins, "dst dtype must equal src dtype")
        if ins.func == ReduceOp.MEAN and src.dtype.is_int:
            _err(ErrorCode.DTYPE_MISMATCH, ins, "MEAN requires a float dtype")
        ax = ins.imm[0]
        if ax not in (0, 1, 2):
            _err(ErrorCode.BAD_IMMEDIATE, ins, f"axis {ax}")
        want = {0: (1, src.cols), 1: (src.rows, 1), 2: (1, 1)}[ax]
        if dst.shape != want:
            _err(ErrorCode.SHAPE_MISMATCH, ins, f"dst {dst.shape} != {want}")
    elif op == Opcode.ACCUMULATE:
        dst, src = o
        spaces(dst, (L,), "dst"); spaces(src, (L,), "src")
        dtypes(dst, (DType.INT32, DType.FP32), "dst")
        if src.dtype != dst.dtype:
            _err(ErrorCode.DTYPE_MISMATCH, ins, "src dtype must equal dst dtype")
        same(dst, src)
    elif op in (Opcode.ACTIVATION, Opcode.NORMALIZATION):
        dst, src = o
        spaces(dst, (L,), "dst"); spaces(src, (L,), "src")
        dtypes(src, FLOAT_TYPES, "src"); dtypes(dst, FLOAT_TYPES, "dst")
        same(dst, src)
    elif op == Opcode.CONVERT:
        dst, src = o
        spaces(dst, (L,), "dst"); spaces(src, (L,), "src")
        same(dst, src)
    elif op == Opcode.SYNCHRONIZE:
        sem = ins.imm[0]
        if cfg is not None and not 0 <= sem < cfg.semaphores:
            _err(ErrorCode.BAD_SEMAPHORE, ins, f"semaphore {sem}")
        if ins.imm[1] <= 0:
            _err(ErrorCode.BAD_IMMEDIATE, ins, "semaphore amount must be > 0")
    elif op == Opcode.BARRIER:
        if ni == 2:
            first, count = ins.imm
            if count <= 0 or first < 0 or (cfg is not None and first + count > cfg.cores):
                _err(ErrorCode.BAD_CORE, ins, f"barrier core range [{first},{first + count})")
            if not first <= ins.core < first + count:
                _err(ErrorCode.BAD_CORE, ins, "issuing core must be inside the barrier range")
        elif ni == 1:
            _err(ErrorCode.OPERAND_COUNT, ins, "barrier takes 0 or 2 immediates")
    elif op == Opcode.EXECUTE_KERNEL:
        if kernel_ids is not None and ins.imm[0] not in kernel_ids:
            _err(ErrorCode.KERNEL_NOT_FOUND, ins, f"kernel {ins.imm[0]}")


def barrier_range(ins: Instruction, n_cores: int) -> range:
    if len(ins.imm) == 2:
        return range(ins.imm[0], ins.imm[0] + ins.imm[1])
    return range(n_cores)


def mm_dims(ins: Instruction) -> tuple[int, int, int]:
    """(M, N, K) of a MATMUL/GEMM instruction."""
    _, a, b = ins.operands
    m, k = (a.cols, a.rows) if ins.has(Flag.TRANS_A) else a.shape
    n = b.rows if ins.has(Flag.TRANS_B) else b.cols
    return m, n, k


@dataclass
class Program:
    """An executable TITAN program: one command stream plus a kernel table."""
    instructions: list[Instruction] = field(default_factory=list)
    kernels: dict[int, list[Instruction]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def validate(self, cfg=None) -> "Program":
        ids = set(self.kernels)
        for kid, body in self.kernels.items():
            for ins in body:
                if ins.opcode == Opcode.BARRIER:
                    raise ISAError(ErrorCode.ILLEGAL_OPCODE, f"kernel {kid}: BARRIER not allowed inside kernels")
                validate(ins, cfg, ids)
        for ins in self.instructions:
            validate(ins, cfg, ids)
        return self

    def __len__(self) -> int:
        return len(self.instructions)
