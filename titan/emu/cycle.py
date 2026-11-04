"""Mode B — transaction-level cycle model (cycle-approximate, NOT cycle-accurate RTL timing).

Models per-core functional units (CTRL, DMA engines, MXU, VPU, SFU), a memory
scoreboard (RAW/WAR/WAW hazards on byte ranges), in-flight limits, shared
bandwidth resources (HBM, L2, shared SRAM) with FIFO reservation, an L2 cache
and the fill/drain behaviour of the systolic matrix engine.
"""
from __future__ import annotations

import math
from collections import defaultdict

from ..hw.memory import L2Cache
from ..isa.spec import SPEC, Flag, Instruction, Norm, Opcode, Space, TensorRef, Unit, mm_dims


def _key(core: int, ref: TensorRef):
    return ("L", core) if ref.space == Space.LOCAL else ("G",) if ref.space == Space.GLOBAL else ("S",)


def access_sets(core: int, ins: Instruction):
    """(reads, writes) as lists of (space_key, lo, hi)."""
    o = ins.operands
    if not o:
        return [], []
    rng = lambda r: (_key(core, r), r.addr, r.end)  # noqa: E731
    dst = o[0]
    reads = [rng(r) for r in o[1:]]
    if (ins.opcode == Opcode.MATMUL and ins.has(Flag.ACC)) or ins.opcode == Opcode.ACCUMULATE:
        reads.append(rng(dst))
    return reads, [rng(dst)]


def _overlap(a, b) -> bool:
    for ka, lo, hi in a:
        for kb, lo2, hi2 in b:
            if ka == kb and lo < hi2 and lo2 < hi:
                return True
    return False


class CycleModel:
    def __init__(self, cfg):
        self.cfg = cfg
        n = cfg.cores
        self.unit_free = [defaultdict(float) for _ in range(n)]
        self.dma_free = [[0.0] * cfg.dma_engines for _ in range(n)]
        self.hbm_free = self.l2_free = self.shared_free = 0.0
        self.l2 = L2Cache(cfg.l2_kib * 1024, cfg.l2_line, cfg.l2_ways)
        self.inflight: list[list] = [[] for _ in range(n)]
        self.last_end = [0.0] * n
        self.busy = [defaultdict(float) for _ in range(n)]
        self.stall = [defaultdict(float) for _ in range(n)]
        self.bytes = defaultdict(int)
        self.hbm_bytes = 0
        self.mxu_useful_macs = 0.0
        self.mxu_capacity_macs = 0.0
        bpc = cfg.bytes_per_cycle
        self.noc_bpc, self.hbm_bpc = bpc(cfg.noc_link_gbps), bpc(cfg.hbm_gbps)
        self.l2_bpc, self.shared_bpc, self.local_bpc = bpc(cfg.l2_gbps), bpc(cfg.shared_gbps), bpc(cfg.local_gbps)

    # ------------------------------------------------------------- costs
    def _dma(self, start: float, ins: Instruction) -> tuple[float, float]:
        cfg = self.cfg
        dst, src = ins.operands
        nbytes = src.nbytes
        path = f"{src.space.name}->{dst.space.name}"
        self.bytes[path] += nbytes
        done = start
        lat = 0
        if Space.LOCAL in (src.space, dst.space) and (src.space != Space.LOCAL or dst.space != Space.LOCAL):
            done = max(done, start + nbytes / self.noc_bpc)
        if src.space == Space.LOCAL and dst.space == Space.LOCAL:
            done = max(done, start + nbytes / self.local_bpc)
            lat = cfg.local_latency
        if src.space == Space.GLOBAL:
            hits, misses = self.l2.access(src, write=False)
            t = max(start, self.l2_free)
            self.l2_free = t + nbytes / self.l2_bpc
            done = max(done, self.l2_free)
            if misses:
                mb = misses * cfg.l2_line
                self.hbm_bytes += mb
                t = max(start, self.hbm_free)
                self.hbm_free = t + mb / self.hbm_bpc
                done = max(done, self.hbm_free)
                lat = max(lat, cfg.hbm_latency)
            else:
                lat = max(lat, cfg.l2_latency)
        if dst.space == Space.GLOBAL:
            self.l2.access(dst, write=True)
            self.hbm_bytes += nbytes
            t = max(start, self.hbm_free)
            self.hbm_free = t + nbytes / self.hbm_bpc
            done = max(done, self.hbm_free)
            lat = max(lat, cfg.l2_latency)
        if Space.SHARED in (src.space, dst.space):
            mult = 2 if src.space == dst.space == Space.SHARED else 1
            t = max(start, self.shared_free)
            self.shared_free = t + mult * nbytes / self.shared_bpc
            done = max(done, self.shared_free)
            lat = max(lat, cfg.shared_latency)
        return done - start, lat

    def _compute(self, ins: Instruction) -> tuple[float, float]:
        cfg, op = self.cfg, ins.opcode
        unit = SPEC[op].unit
        if unit == Unit.MXU:
            m, n, k = mm_dims(ins)
            dt = ins.operands[1].dtype
            rate = cfg.mxu_rate(dt)
            tiles = math.ceil(m / cfg.mxu_rows) * math.ceil(n / cfg.mxu_cols)
            occ = tiles * math.ceil(k / rate)
            self.mxu_useful_macs += m * n * k
            self.mxu_capacity_macs += occ * cfg.mxu_rows * cfg.mxu_cols * rate
            return occ, cfg.mxu_rows + cfg.mxu_cols + cfg.mxu_latency
        if unit == Unit.VPU:
            elems = max(r.numel for r in ins.operands)
            lat = cfg.vpu_latency + (math.ceil(math.log2(cfg.vpu_lanes)) if op == Opcode.REDUCE else 0)
            return math.ceil(elems / cfg.vpu_lanes), lat
        if unit == Unit.SFU:
            elems = ins.operands[0].numel
            passes = 3 if op == Opcode.NORMALIZATION else 1
            if op == Opcode.NORMALIZATION and ins.func == Norm.SOFTMAX:
                passes = 3
            return passes * math.ceil(elems / cfg.sfu_lanes), cfg.sfu_latency
        return 1.0, 0.0

    # ------------------------------------------------------------- schedule
    def schedule(self, core: int, ins: Instruction, t_issue: float) -> tuple[float, float]:
        reads, writes = access_sets(core, ins)
        live = [r for r in self.inflight[core] if r[2] > t_issue]
        self.inflight[core] = live
        t_dep = t_issue
        for r_reads, r_writes, r_end in live:
            if r_end > t_dep and (_overlap(r_writes, reads) or _overlap(r_reads, writes) or
                                  _overlap(r_writes, writes)):
                t_dep = r_end
        unit = SPEC[ins.opcode].unit
        if unit == Unit.DMA:
            eng = min(range(len(self.dma_free[core])), key=lambda i: self.dma_free[core][i])
            t_unit = self.dma_free[core][eng]
        else:
            t_unit = self.unit_free[core][unit]
        start = max(t_issue, t_dep, t_unit)
        st = self.stall[core]
        st["dependency"] += t_dep - t_issue
        st["unit_busy"] += max(0.0, start - max(t_issue, t_dep))
        if unit == Unit.DMA:
            occ, lat = self._dma(start, ins)
            self.dma_free[core][eng] = start + occ
        else:
            occ, lat = self._compute(ins)
            self.unit_free[core][unit] = start + occ
        self.busy[core][unit.name] += occ
        end = start + occ + lat
        if ins.operands:
            self.inflight[core].append((reads, writes, end))
        self.last_end[core] = max(self.last_end[core], end)
        return start, end

    def counters(self, total_cycles: float) -> dict:
        cfg = self.cfg
        n = cfg.cores
        units = sorted({u for b in self.busy for u in b})
        busy = {u: sum(b[u] for b in self.busy) for u in units}
        util = {u: (busy[u] / (total_cycles * n * (cfg.dma_engines if u == "DMA" else 1)) if total_cycles else 0.0)
                for u in units}
        stalls = defaultdict(float)
        for s in self.stall:
            for k, v in s.items():
                stalls[k] += v
        return {
            "unit_busy_cycles": busy,
            "unit_utilization": util,
            "stall_cycles": dict(stalls),
            "dma_bytes_by_path": dict(self.bytes),
            "hbm_bytes": self.hbm_bytes,
            "l2": self.l2.stats(),
            "mxu_useful_macs": self.mxu_useful_macs,
            "mxu_tile_efficiency": (self.mxu_useful_macs / self.mxu_capacity_macs) if self.mxu_capacity_macs else 0.0,
        }
