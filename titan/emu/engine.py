"""TITAN virtual accelerator: machine state, command processor and execution engine.

One engine implements both
  * Mode A (functional): architectural semantics, unbounded queues, no time, and
  * Mode B (cycle model): the same instruction order plus the CycleModel timing.
Results of the two modes are reported separately and never merged.
"""
from __future__ import annotations

import heapq
import json
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..isa.spec import (SPEC, ErrorCode, Flag, Instruction, ISAError, Opcode, Program, Space, SyncOp, TensorRef,
                        barrier_range, validate)
from ..hw.memory import Memory
from .cycle import CycleModel
from .semantics import OpStats, dma_copy, execute, stats_of
from .sequencer import expand_gemm

FUNCTIONAL, CYCLE = "functional", "cycle"


class TitanFault(RuntimeError):
    """Precise fault raised by the virtual accelerator."""

    def __init__(self, code: ErrorCode, msg: str, core: int = -1, origin: int = -1, ins: Instruction | None = None):
        where = f" [core {core}, stream index {origin}{', ' + ins.opcode.name if ins else ''}]" if core >= 0 else ""
        super().__init__(f"{code.name}: {msg}{where}")
        self.code, self.core, self.origin, self.ins = code, core, origin, ins


class Machine:
    """Architectural state of one TITAN-X1 device."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.glob = Memory("GLOBAL", cfg.global_bytes)
        self.shared = Memory("SHARED", cfg.shared_bytes)
        self.local = [Memory(f"LOCAL{c}", cfg.local_bytes) for c in range(cfg.cores)]

    def mem(self, core: int, space: Space) -> Memory:
        return self.glob if space == Space.GLOBAL else self.shared if space == Space.SHARED else self.local[core]

    def reader(self, core: int):
        return lambda ref: self.mem(core, ref.space).read(ref)

    def writer(self, core: int):
        return lambda ref, v: self.mem(core, ref.space).write(ref, v)

    # host access (models host <-> device transfers; host-side, not TITAN execution)
    def host_write(self, ref: TensorRef, values) -> None:
        if ref.space == Space.LOCAL:
            raise ValueError("host can only write GLOBAL or SHARED memory")
        self.mem(0, ref.space).write(ref, np.asarray(values))

    def host_read(self, ref: TensorRef) -> np.ndarray:
        if ref.space == Space.LOCAL:
            raise ValueError("host can only read GLOBAL or SHARED memory")
        return self.mem(0, ref.space).read(ref)


@dataclass
class Entry:
    seq: int
    origin: int
    core: int
    ins: Instruction
    barrier: int = -1
    need: tuple = ()
    dispatch: float = 0.0


@dataclass
class RunResult:
    mode: str
    instructions: int
    counts: dict
    ops: int
    mxu_macs: int
    dma_bytes: int
    cycles: float | None = None
    clock_ghz: float = 0.0
    counters: dict = field(default_factory=dict)
    completions: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    host_wall_s: float = 0.0

    @property
    def simulated_seconds(self) -> float | None:
        return None if self.cycles is None else self.cycles / (self.clock_ghz * 1e9)

    def summary(self) -> dict:
        d = {"mode": self.mode, "instructions_retired": self.instructions, "ops": self.ops,
             "mxu_macs": self.mxu_macs, "dma_bytes": self.dma_bytes,
             "host_emulation_wall_s": round(self.host_wall_s, 6)}
        if self.cycles is not None:
            d.update({"simulated_cycles": self.cycles, "simulated_seconds_at_clock": self.simulated_seconds,
                      "simulated_ops_per_s": self.ops / self.simulated_seconds if self.simulated_seconds else 0.0})
        return d

    def chrome_trace(self, path: str) -> None:
        ev = [{"name": t["op"], "ph": "X", "pid": t["core"], "tid": t["unit"], "ts": t["start"],
               "dur": max(t["end"] - t["start"], 0.01), "args": {"origin": t["origin"]}} for t in self.trace]
        with open(path, "w") as f:
            json.dump({"traceEvents": ev, "displayTimeUnit": "ns"}, f)


def expand_program(prog: Program, cfg):
    """Command-processor front end: kernel inlining, GEMM sequencing, barrier fan-out (no execution)."""
    seq = 0
    bid = 0

    def relocate(ins: Instruction, off: int) -> Instruction:
        if not off:
            return ins
        ops = tuple(r.with_addr(r.addr + off) if r.space == Space.GLOBAL else r for r in ins.operands)
        return Instruction(ins.opcode, ins.core, ops, ins.imm, ins.func, ins.flags)

    def one(ins: Instruction, origin: int, depth: int, off: int):
        nonlocal seq, bid
        ins = relocate(ins, off)
        if off:
            try:
                validate(ins, cfg, set(prog.kernels))
            except ISAError as e:
                raise TitanFault(e.code, str(e), ins.core, origin, ins) from None
        if ins.opcode == Opcode.EXECUTE_KERNEL:
            if depth >= cfg.kernel_max_depth:
                raise TitanFault(ErrorCode.KERNEL_DEPTH, f"kernel nesting > {cfg.kernel_max_depth}",
                                 ins.core, origin, ins)
            body = prog.kernels.get(ins.imm[0])
            if body is None:
                raise TitanFault(ErrorCode.KERNEL_NOT_FOUND, f"kernel {ins.imm[0]}", ins.core, origin, ins)
            for b in body:
                yield from one(b.with_core(ins.core), origin, depth + 1, off + ins.iimm(1))
        elif ins.opcode == Opcode.GEMM:
            try:
                micro = expand_gemm(ins, cfg)
            except ISAError as e:
                raise TitanFault(e.code, str(e), ins.core, origin, ins) from None
            for u in micro:
                seq += 1
                yield Entry(seq, origin, ins.core, u)
        elif ins.opcode == Opcode.BARRIER:
            rng = tuple(barrier_range(ins, cfg.cores))
            bid += 1
            for c in rng:
                seq += 1
                yield Entry(seq, origin, c, ins.with_core(c), bid, rng)
        else:
            seq += 1
            yield Entry(seq, origin, ins.core, ins)

    for origin, ins in enumerate(prog.instructions):
        yield from one(ins, origin, 0, 0)


class Engine:
    def __init__(self, machine: Machine, mode: str = FUNCTIONAL, trace: bool = False, matmul_hook=None):
        if mode not in (FUNCTIONAL, CYCLE):
            raise ValueError(f"unknown mode {mode!r}")
        self.m, self.cfg, self.mode, self.trace = machine, machine.cfg, mode, trace
        self.matmul_hook = matmul_hook   # optional callable(core, ins, read, write) -> bool (RTL offload)
        self.record: list | None = None  # if a list: (core, instruction) of every data instruction in order
        self.execute_data = True         # False: schedule only (used to build replay programs)

    # ---------------------------------------------------------- command processor
    def expand(self, prog: Program):
        return expand_program(prog, self.cfg)

    # ---------------------------------------------------------- execution
    def run(self, prog: Program, validate_program: bool = True) -> RunResult:
        t0 = time.perf_counter()
        cfg, n = self.cfg, self.cfg.cores
        if validate_program:
            try:
                prog.validate(cfg)
            except ISAError as e:
                raise TitanFault(e.code, str(e)) from None
        entries = list(self.expand(prog))
        cyc = CycleModel(cfg) if self.mode == CYCLE else None
        depth = cfg.queue_depth if cyc else float("inf")
        window = max(cfg.queue_depth // 4, 4)
        queues = [deque() for _ in range(n)]
        dispatched = [0] * n
        issue_times: list[list[float]] = [[] for _ in range(n)]
        inflight_ends: list[list[float]] = [[] for _ in range(n)]
        core_ready = [0.0] * n
        barrier_wait: list[int | None] = [None] * n
        barriers: dict[int, dict] = {}
        sems = [deque() for _ in range(cfg.semaphores)]
        counts: dict[str, int] = {}
        ops = macs = dbytes = 0
        retired = 0
        end_time = 0.0
        queue_delay = cp_stall = sync_stall = barrier_stall = 0.0
        completions, trace = [], []
        cp_i, cp_time = 0, 0.0
        cp_step = cfg.cp_dispatch_cycles if cyc else 0
        last = (lambda c: cyc.last_end[c]) if cyc else (lambda c: 0.0)

        def sem_available(sem: int) -> int:
            return sum(a for _, a in sems[sem])

        while True:
            best = None
            if cp_i < len(entries):
                e = entries[cp_i]
                k = dispatched[e.core]
                t = None
                if k < depth:
                    t = cp_time
                elif len(issue_times[e.core]) > k - depth:
                    t = max(cp_time, issue_times[e.core][int(k - depth)])
                if t is not None:
                    best = (t, 1, -1)
            for c in range(n):
                if barrier_wait[c] is not None or not queues[c]:
                    continue
                e = queues[c][0]
                if e.ins.opcode == Opcode.SYNCHRONIZE and e.ins.func == SyncOp.WAIT:
                    if sem_available(e.ins.imm[0]) < e.ins.imm[1]:
                        continue
                t = max(core_ready[c], e.dispatch)
                cand = (t, 0, c)
                if best is None or cand < best:
                    best = cand
            if best is None:
                if cp_i >= len(entries) and not any(queues) and all(b is None for b in barrier_wait):
                    break
                blocked = [f"core {c}: " + ("barrier" if barrier_wait[c] is not None else
                                            queues[c][0].ins.opcode.name if queues[c] else "idle")
                           for c in range(n)]
                raise TitanFault(ErrorCode.DEADLOCK, "no instruction can make progress (" + "; ".join(blocked) + ")")
            t, kind, c = best
            if kind == 1:
                e = entries[cp_i]
                if cyc and t > cp_time:
                    cp_stall += t - cp_time
                e.dispatch = t
                queues[e.core].append(e)
                dispatched[e.core] += 1
                cp_i += 1
                cp_time = t + cp_step
                continue

            e = queues[c].popleft()
            ins = e.ins
            issue_times[c].append(t)
            queue_delay += t - e.dispatch
            name = ins.opcode.name
            counts[name] = counts.get(name, 0) + 1
            retired += 1
            start = end = t
            try:
                if ins.opcode == Opcode.BARRIER:
                    st = barriers.setdefault(e.barrier, {"need": set(e.need), "arrived": {}})
                    st["arrived"][c] = max(t, last(c))
                    barrier_wait[c] = e.barrier
                    if set(st["arrived"]) == st["need"]:
                        release = max(st["arrived"].values()) + (cfg.barrier_latency if cyc else 0)
                        for cc, arr in st["arrived"].items():
                            barrier_stall += release - arr
                            core_ready[cc] = release
                            barrier_wait[cc] = None
                        del barriers[e.barrier]
                        end = release
                    else:
                        end = st["arrived"][c]
                elif ins.opcode == Opcode.SYNCHRONIZE:
                    sem, amount = ins.imm
                    if ins.func == SyncOp.SIGNAL:
                        end = max(t, last(c))
                        sems[sem].append((end, amount))
                        core_ready[c] = t + (1 if cyc else 0)
                    else:
                        need, tsat = amount, 0.0
                        while need:
                            ts, a = sems[sem][0]
                            take = min(a, need)
                            need -= take
                            tsat = max(tsat, ts)
                            if take == a:
                                sems[sem].popleft()
                            else:
                                sems[sem][0] = (ts, a - take)
                        start = end = max(t, tsat)
                        sync_stall += end - t
                        core_ready[c] = end + (1 if cyc else 0)
                else:
                    if self.record is not None:
                        self.record.append((c, ins))
                    handled = False
                    if not self.execute_data:
                        stt = stats_of(ins)
                    else:
                        read, write = self.m.reader(c), self.m.writer(c)
                        if self.matmul_hook is not None and ins.opcode == Opcode.MATMUL:
                            handled = self.matmul_hook(c, ins, read, write)
                        if handled:
                            stt = stats_of(ins)
                        elif ins.opcode in (Opcode.LOAD, Opcode.STORE, Opcode.COPY):
                            src, dst = ins.operands[1], ins.operands[0]
                            stt = dma_copy(self.m.mem(c, src.space), self.m.mem(c, dst.space), ins)
                        else:
                            stt = execute(read, write, ins)
                    ops += stt.ops
                    macs += stt.mxu_macs
                    dbytes += stt.bytes
                    if cyc:
                        fl = inflight_ends[c]
                        while fl and fl[0] <= t:
                            heapq.heappop(fl)
                        if len(fl) >= window:
                            t2 = heapq.heappop(fl)
                            cyc.stall[c]["inflight_limit"] += t2 - t
                            t = t2
                        cyc.advance(min(t, cp_time))
                        start, end = cyc.schedule(c, ins, t)
                        heapq.heappush(fl, end)
                        core_ready[c] = t + 1
            except ISAError as err:
                raise TitanFault(err.code, str(err), c, e.origin, ins) from None
            except (FloatingPointError, OverflowError, ValueError) as err:
                raise TitanFault(ErrorCode.NUMERIC, str(err), c, e.origin, ins) from None
            end_time = max(end_time, end)
            if ins.flags & Flag.NOTIFY:
                completions.append({"origin": e.origin, "core": c, "opcode": name, "time": end})
            if self.trace:
                trace.append({"core": c, "origin": e.origin, "op": name, "unit": SPEC[ins.opcode].unit.name,
                              "issue": t, "start": start, "end": end})

        res = RunResult(self.mode, retired, counts, ops, macs, dbytes, clock_ghz=cfg.clock_ghz,
                        completions=completions, trace=trace)
        if cyc:
            total = max([end_time] + cyc.last_end)
            res.cycles = total
            res.counters = cyc.counters(total)
            res.counters["stall_cycles"].update({"sync_wait": sync_stall, "barrier": barrier_stall})
            res.counters["queue_delay_cycles"] = queue_delay
            res.counters["cp_dispatch_stall_cycles"] = cp_stall
        res.host_wall_s = time.perf_counter() - t0
        return res


def run(machine: Machine, prog: Program, mode: str = FUNCTIONAL, **kw) -> RunResult:
    return Engine(machine, mode, **kw).run(prog)
