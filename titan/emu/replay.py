"""Replay executor: fast, bit-exact functional execution of a TITAN ISA program.

A TITAN program is static. Its functional instruction order (including the
effect of SIGNAL/WAIT/BARRIER) does not depend on data, so it is computed once
by the reference engine. Every data instruction is then pre-bound to a closure
that works on persistent NumPy views of the machine memory. Running the
program executes exactly the same instructions, in the same order, with the
same arithmetic as the reference engine (`tests/test_replay.py` checks that
the resulting memory images are bit-identical). It only removes the per-
instruction Python overhead (decoding, view construction, dispatch).
"""
from __future__ import annotations

import time

import numpy as np

from ..dtypes import DType
from ..isa.spec import Axis, Flag, Instruction, Norm, Opcode, Program, ReduceOp
from .engine import FUNCTIONAL, Engine, Machine, RunResult
from .semantics import ACTIVATIONS, dma_copy, execute, normalize

F32 = DType.FP32
_BIN = {Opcode.VECTOR_ADD: np.add, Opcode.VECTOR_MULTIPLY: np.multiply, Opcode.VECTOR_SUB: np.subtract,
        Opcode.VECTOR_MAX: np.maximum, Opcode.VECTOR_DIV: np.divide}


def _overlap(a, b) -> bool:
    return a.space == b.space and a.addr < b.end and b.addr < a.end


def bind(m: Machine, core: int, ins: Instruction):
    """Return a zero-argument callable executing ``ins`` on ``core`` (None for no-ops)."""
    op, o = ins.opcode, ins.operands
    if op == Opcode.NOP:
        return None
    mem = lambda r: m.mem(core, r.space)  # noqa: E731
    if op in (Opcode.LOAD, Opcode.STORE, Opcode.COPY):
        dst, src = o
        sm, dm = mem(src), mem(dst)
        if dst.dtype == DType.INT4 or (sm is dm and _overlap(dst, src)):
            return lambda: dma_copy(sm, dm, ins)
        d, s = dm.raw_view(dst), sm.raw_view(src)
        if ins.has(Flag.TRANSPOSE):
            s = s.T
        return lambda: np.copyto(d, s)

    def generic():
        read, write = m.reader(core), m.writer(core)
        return lambda: execute(read, write, ins)

    if any(r.dtype != F32 for r in o):
        return generic()
    v = [mem(r).raw_view(r) for r in o]
    if op == Opcode.MATMUL:
        c, a, b = v
        ta, tb, acc = ins.has(Flag.TRANS_A), ins.has(Flag.TRANS_B), ins.has(Flag.ACC)

        def mm():
            A = np.ascontiguousarray(a)
            B = np.ascontiguousarray(b)
            r = (A.T if ta else A) @ (B.T if tb else B)
            if acc:
                r = r + c
            c[...] = r
        return mm
    if op in _BIN:
        fn = _BIN[op]
        d, a, b = v
        return lambda: fn(a, b, out=d)
    if op == Opcode.VECTOR_SCALE_ADD:
        al, be = np.float32(ins.fimm(0)), np.float32(ins.fimm(1))
        if len(v) == 3:
            d, a, b = v
            return lambda: np.add(al * a, be * b, out=d)
        d, a = v
        return lambda: np.add(al * a, be, out=d)
    if op == Opcode.ACCUMULATE:
        d, s = v
        return lambda: np.add(d, s, out=d)
    if op == Opcode.REDUCE:
        d, s = v
        ax = {Axis.ROWS: 0, Axis.COLS: 1, Axis.ALL: None}[Axis(ins.imm[0])]
        fn = ReduceOp(ins.func)
        shape = d.shape

        def red():
            x = np.ascontiguousarray(s)
            if fn == ReduceOp.SUM:
                r = x.sum(axis=ax, keepdims=True, dtype=np.float32)
            elif fn == ReduceOp.MAX:
                r = x.max(axis=ax, keepdims=True)
            else:
                r = x.mean(axis=ax, keepdims=True, dtype=np.float32)
            d[...] = np.asarray(r).reshape(shape)
        return red
    if op == Opcode.ACTIVATION:
        d, s = v
        f = ACTIVATIONS[ins.func]
        return lambda: np.copyto(d, f(np.ascontiguousarray(s)).astype(np.float32))
    if op == Opcode.NORMALIZATION:
        d, s = v
        fn, eps = Norm(ins.func), ins.fimm(0, 1e-5)
        return lambda: np.copyto(d, normalize(fn, np.ascontiguousarray(s), eps))
    return generic()


class ReplayProgram:
    """A program pre-bound to one machine. ``run()`` == Engine(machine, FUNCTIONAL).run(program)."""

    def __init__(self, machine: Machine, prog: Program):
        self.m = machine
        eng = Engine(machine, FUNCTIONAL)
        eng.record, eng.execute_data = [], False
        self.template = eng.run(prog)            # schedule + static work counts, no data touched
        self.steps = [f for f in (bind(machine, c, ins) for c, ins in eng.record) if f is not None]

    def run(self) -> RunResult:
        t0 = time.perf_counter()
        with np.errstate(all="ignore"):
            for f in self.steps:
                f()
        t = self.template
        return RunResult(FUNCTIONAL, t.instructions, dict(t.counts), t.ops, t.mxu_macs, t.dma_bytes,
                         clock_ghz=t.clock_ghz, completions=list(t.completions),
                         host_wall_s=time.perf_counter() - t0)
