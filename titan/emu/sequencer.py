"""GEMM macro-op sequencer of the command processor.

A GEMM instruction (GLOBAL operands) is expanded into LOAD / MATMUL / VECTOR /
STORE micro-ops that run on the issuing core, using the reserved GEMM staging
region at the top of that core's local SRAM. A and B tiles and the accumulator
are double-buffered so DMA overlaps matrix-engine work in the cycle model.
"""
from __future__ import annotations

import math

from ..dtypes import DType, accum_dtype, nbytes
from ..isa import build as B
from ..isa.spec import ErrorCode, Flag, Instruction, ISAError, Space, TensorRef, mm_dims


def _align(x: int, a: int = 64) -> int:
    return -(-x // a) * a


def gemm_tiles(cfg, m: int, n: int, k: int, dt: DType, budget: int) -> tuple[int, int, int]:
    """Largest MXU-aligned tiles whose double-buffered working set fits in `budget` bytes."""
    even = dt == DType.INT4
    tm = min(m, 4 * cfg.mxu_rows)
    tn = min(n, 4 * cfg.mxu_cols)
    while True:
        acc = 3 * _align(tm * tn * 4) + 512
        per_k = 2 * (nbytes(dt, tm) + nbytes(dt, tn)) + 256
        tk = min(k, (budget - acc) // per_k) if budget > acc else 0
        if even:
            tm, tn, tk = [max(2, v - v % 2) if v > 1 else v for v in (tm, tn, tk)]
        if tk >= 1 and 2 * (_align(nbytes(dt, tm * tk)) + _align(nbytes(dt, tk * tn))) + acc <= budget:
            return tm, tn, tk
        if tm == 1 and tn == 1:
            raise ISAError(ErrorCode.OUT_OF_BOUNDS, "GEMM staging region too small")
        tm, tn = max(1, tm // 2), max(1, tn // 2)


def expand_gemm(ins: Instruction, cfg) -> list[Instruction]:
    c, a, b = ins.operands
    core, ta, tb = ins.core, ins.has(Flag.TRANS_A), ins.has(Flag.TRANS_B)
    m, n, k = mm_dims(ins)
    dt, acc_dt = a.dtype, accum_dtype(a.dtype)
    alpha, beta = ins.fimm(0, 1.0), ins.fimm(1, 0.0)
    base, budget = cfg.staging_base, cfg.local_bytes - cfg.staging_base
    tm, tn, tk = gemm_tiles(cfg, m, n, k, dt, budget)

    off = base
    def region(size):
        nonlocal off
        r = off
        off += _align(size)
        return r
    a_buf = [region(nbytes(dt, tm * tk)) for _ in range(2)]
    b_buf = [region(nbytes(dt, tk * tn)) for _ in range(2)]
    acc_buf = [region(tm * tn * 4) for _ in range(2)]
    cin_buf = region(tm * tn * 4)
    if off > cfg.local_bytes:
        raise ISAError(ErrorCode.OUT_OF_BOUNDS, "GEMM staging overflow")

    L = lambda addr, r, cc, d: TensorRef(Space.LOCAL, addr, r, cc, d)  # noqa: E731
    out: list[Instruction] = []
    t = 0
    for i0 in range(0, m, tm):
        mt = min(tm, m - i0)
        for j0 in range(0, n, tn):
            nt = min(tn, n - j0)
            acc = L(acc_buf[t % 2], mt, nt, acc_dt)
            for kk, k0 in enumerate(range(0, k, tk)):
                kt = min(tk, k - k0)
                ga = a.sub(k0, i0, kt, mt) if ta else a.sub(i0, k0, mt, kt)
                gb = b.sub(j0, k0, nt, kt) if tb else b.sub(k0, j0, kt, nt)
                la = L(a_buf[kk % 2], ga.rows, ga.cols, dt)
                lb = L(b_buf[kk % 2], gb.rows, gb.cols, dt)
                out += [B.load(core, la, ga), B.load(core, lb, gb), B.matmul(core, acc, la, lb, kk > 0, ta, tb)]
            if beta != 0.0:
                cin = L(cin_buf, mt, nt, acc_dt)
                out.append(B.load(core, cin, c.sub(i0, j0, mt, nt)))
                if acc_dt.is_int:
                    out.append(B.accumulate(core, acc, cin))
                else:
                    out.append(B.scale_add(core, acc, acc, alpha, beta, cin))
            elif alpha != 1.0:
                out.append(B.scale_add(core, acc, acc, alpha, 0.0))
            st = B.store(core, c.sub(i0, j0, mt, nt), acc)
            out.append(st)
            t += 1
    if ins.has(Flag.NOTIFY) and out:
        last = out[-1]
        out[-1] = Instruction(last.opcode, last.core, last.operands, last.imm, last.func, last.flags | Flag.NOTIFY)
    return out


def tile_count(cfg, m, n, k, dt) -> int:
    tm, tn, tk = gemm_tiles(cfg, m, n, k, dt, cfg.local_bytes - cfg.staging_base)
    return math.ceil(m / tm) * math.ceil(n / tn) * math.ceil(k / tk)
