"""Mode C — analytical performance model (roofline). Every number here is an ESTIMATE."""
from __future__ import annotations

import math
from collections import defaultdict

from ..dtypes import DType, accum_dtype, nbytes
from ..emu.engine import expand_program
from ..isa.spec import SPEC, Opcode, Space, Unit, mm_dims

LABEL = "analytical estimate (roofline; ideal overlap; not a measurement)"


def gemm_roofline(cfg, m: int, n: int, k: int, dt: DType) -> dict:
    ops = 2 * m * n * k
    traffic = nbytes(dt, m * k) + nbytes(dt, k * n) + nbytes(accum_dtype(dt), m * n)
    intensity = ops / traffic
    peak = cfg.peak_ops(dt)
    bw = cfg.hbm_gbps * 1e9
    # MXU quantization: partial array tiles and uneven per-core work
    R, C = cfg.mxu_rows, cfg.mxu_cols
    tiles = math.ceil(m / R) * math.ceil(n / C)
    per_core = math.ceil(tiles / cfg.cores)
    tile_eff = (m * n) / (per_core * cfg.cores * R * C)
    t_compute = ops / (peak * tile_eff)
    t_mem = traffic / bw
    t = max(t_compute, t_mem)
    return {"ops": ops, "compulsory_bytes": traffic, "arithmetic_intensity": intensity,
            "ridge_point": cfg.ridge_point(dt), "bound": "compute" if t_compute >= t_mem else "memory",
            "mxu_tile_efficiency": tile_eff, "estimated_seconds": t, "attainable_ops_per_s": ops / t,
            "peak_ops_per_s": peak, "label": LABEL}


def program_analysis(prog, cfg) -> dict:
    """Static analysis of an ISA program: work, traffic and roofline lower bound."""
    macs = defaultdict(int)
    mxu_cycles = defaultdict(float)
    vpu = sfu = 0
    traffic = defaultdict(int)
    count = 0
    for e in expand_program(prog, cfg):
        ins = e.ins
        count += 1
        unit = SPEC[ins.opcode].unit
        if ins.opcode == Opcode.MATMUL:
            m, n, k = mm_dims(ins)
            dt = ins.operands[1].dtype
            macs[dt.name] += m * n * k
            mxu_cycles[e.core] += math.ceil(m / cfg.mxu_rows) * math.ceil(n / cfg.mxu_cols) * \
                math.ceil(k / cfg.mxu_rate(dt))
        elif unit == Unit.VPU:
            vpu += max(r.numel for r in ins.operands)
        elif unit == Unit.SFU:
            sfu += ins.operands[0].numel
        elif unit == Unit.DMA:
            dst, src = ins.operands
            traffic[f"{src.space.name}->{dst.space.name}"] += src.nbytes
    hbm = sum(v for k_, v in traffic.items() if "GLOBAL" in k_)
    hz = cfg.clock_ghz * 1e9
    t_mxu = max(mxu_cycles.values(), default=0) / hz
    t_hbm = hbm / (cfg.hbm_gbps * 1e9)
    t_vpu = vpu / (cfg.vpu_lanes * cfg.cores * hz)
    t_sfu = sfu / (cfg.sfu_lanes * cfg.cores * hz)
    terms = {"mxu_critical_core": t_mxu, "global_memory": t_hbm, "vpu": t_vpu, "sfu": t_sfu}
    bound = max(terms, key=terms.get)
    total_ops = 2 * sum(macs.values())
    return {"instructions_after_expansion": count, "mxu_macs_by_dtype": dict(macs), "dma_bytes_by_path": dict(traffic),
            "global_bytes": hbm, "lower_bound_seconds": max(terms.values()), "bound_by": bound,
            "time_terms_s": terms, "arithmetic_intensity": total_ops / hbm if hbm else float("inf"),
            "label": LABEL}


def roofline_points(cfg, dt: DType, intensities) -> list[tuple[float, float]]:
    peak, bw = cfg.peak_ops(dt), cfg.hbm_gbps * 1e9
    return [(i, min(peak, i * bw)) for i in intensities]


def scale_requirements(target_tflops: float, dt: DType, clock_ghz: float, mxu: int = 128,
                       rate: float = 1.0, ops_per_byte: float = 281.0) -> dict:
    """What a design must provide to reach ``target_tflops`` at a given balance point (estimate)."""
    macs_per_cycle = target_tflops * 1e12 / (2 * clock_ghz * 1e9)
    cores = math.ceil(macs_per_cycle / (mxu * mxu * rate))
    return {"target_tflops": target_tflops, "precision": dt.name, "clock_ghz": clock_ghz,
            "macs_per_cycle": macs_per_cycle, "cores_of_%dx%d" % (mxu, mxu): cores,
            "memory_tbps_for_balance": target_tflops / ops_per_byte, "balance_ops_per_byte": ops_per_byte,
            "label": LABEL}


__all__ = ["gemm_roofline", "program_analysis", "roofline_points", "scale_requirements", "LABEL", "Space"]
