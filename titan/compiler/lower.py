"""TITAN compiler: Graph -> TITAN IR -> (fusion, memory planning, tiling, scheduling) -> TITAN ISA.

Pipeline
  1. canonicalize composites (xent -> its primitive decomposition), dead-code elimination
  2. kernel formation with operator fusion (element-wise chains, GEMM epilogues)
  3. parameter-update placement (in-place element-wise updates or copies)
  4. global memory planning (persistent buffers + liveness-based reuse of temporaries)
  5. per-kernel tiling into LOCAL SRAM (double buffering, resident weights)
  6. multi-core work distribution (row blocks / round-robin tiles)
  7. barrier insertion with partition-aware elimination
  8. ISA emission + static validation of the generated program
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from ..dtypes import DType, accum_dtype, nbytes
from ..hw.memory import Allocator
from ..isa import build as B
from ..isa.spec import ISA_VERSION, Act, Axis, Norm, Opcode, Program, ReduceOp, Space, TensorRef
from .graph import BINARY, EWISE, ROWNORM, UNARY, Graph, Tensor
from .ir import Buffer, Kernel, Module, Step

_VEC = {"add": Opcode.VECTOR_ADD, "sub": Opcode.VECTOR_SUB, "mul": Opcode.VECTOR_MULTIPLY,
        "div": Opcode.VECTOR_DIV, "maximum": Opcode.VECTOR_MAX}
_BC = {"full": None, "row": "row", "col": "col", "scalar": "scalar"}


class CompileError(Exception):
    pass


@dataclass
class CompileOptions:
    fuse: bool = True
    barrier_elim: bool = True
    double_buffer: bool = True
    gemm_macro: bool = False          # lower plain GEMMs to the GEMM macro instruction
    gemm_tiles: tuple | None = None   # (tm, tn, tk) override, used by the autotuner
    gemm_distribution: str = "auto"   # auto | rows | tiles
    resident_b: bool = True           # keep a small B operand resident in LOCAL SRAM
    cores: int | None = None          # restrict the number of cores used


@dataclass
class Executable:
    program: Program
    module: Module
    cfg: object
    buffers: dict[str, Buffer]        # named buffers: inputs, params, consts, outputs
    init: dict[str, np.ndarray]       # initial values of params and consts
    inputs: list[str]
    outputs: list[str]
    params: list[str]
    stats: dict = field(default_factory=dict)
    options: CompileOptions = field(default_factory=CompileOptions)

    def ref(self, name: str) -> TensorRef:
        return self.buffers[name].ref


def _mode(shape, full) -> str:
    if shape == full:
        return "full"
    if shape == (1, 1):
        return "scalar"
    if shape[0] == 1 and shape[1] == full[1]:
        return "row"
    if shape[1] == 1 and shape[0] == full[0]:
        return "col"
    raise CompileError(f"operand shape {shape} does not broadcast to {full}")


def _al(x: int, a: int = 64) -> int:
    return -(-int(x) // a) * a


def _row_blocks(m: int, n: int):
    b = math.ceil(m / n)
    return [(c, c * b, min(m, (c + 1) * b)) for c in range(n) if c * b < m]


def _eb(dt: DType) -> float:
    return dt.bits / 8


# ===================================================================== emitter

class Emitter:
    """Collects instructions and inserts barriers only where cross-core hazards exist."""

    def __init__(self, ncores: int, elim: bool):
        self.ins: list = []
        self.pending: list = []
        self.ncores, self.elim = ncores, elim
        self.barriers = 0
        self.barriers_avoided = 0

    @staticmethod
    def _conflict(a, b, use_sig: bool) -> bool:
        if not (a[0] < b[1] and b[0] < a[1]) or not (a[2] or b[2]):
            return False
        return not (use_sig and a[3] is not None and a[3] == b[3])

    def sync(self, accesses) -> None:
        if self.ncores > 1 and self.pending:
            need = any(self._conflict(a, p, self.elim) for a in accesses for p in self.pending)
            if need:
                self.barrier()
            elif any(self._conflict(a, p, False) for a in accesses for p in self.pending):
                self.barriers_avoided += 1
        self.pending.extend(accesses)

    def barrier(self) -> None:
        if self.ncores > 1:
            self.ins.append(B.barrier(0, 0, self.ncores))
            self.barriers += 1
        self.pending = []

    def emit(self, *ins) -> None:
        self.ins.extend(ins)


def _acc(buf: Buffer, write: bool, sig):
    return (buf.addr, buf.addr + buf.nbytes, write, sig)


class _Local:
    """Bump allocator over the compiler-visible part of a core's LOCAL SRAM."""

    def __init__(self, cfg):
        self.limit = cfg.staging_base
        self.off = 0

    def take(self, nbytes_: float) -> int:
        a = self.off
        self.off += _al(max(1, math.ceil(nbytes_)))
        if self.off > self.limit:
            raise CompileError(f"LOCAL SRAM plan exceeds {self.limit} bytes")
        return a


def _L(addr, rows, cols, dt) -> TensorRef:
    return TensorRef(Space.LOCAL, addr, rows, cols, dt)


# ===================================================================== compiler

class Compiler:
    def __init__(self, g: Graph, cfg, opts: CompileOptions | None = None, share_with: "Executable | None" = None):
        self.g, self.cfg = g, cfg
        self.share = share_with
        self.o = opts or CompileOptions()
        self.n = max(1, min(self.o.cores or cfg.cores, cfg.cores))
        self.budget = cfg.staging_base

    # ------------------------------------------------------------ helpers
    @staticmethod
    def canon(t: Tensor) -> Tensor:
        while t.node is not None and t.node.op == "xent":
            t = t.node.inputs[2]
        return t

    def _sig(self, m: int):
        return ("rows", m, self.n)

    # ------------------------------------------------------------ passes
    def form_kernels(self):
        g, C = self.g, self.canon
        roots = [C(t) for t in g.outputs.values()] + [C(v) for _, v in g.assigns]
        live, stack = set(), list(roots)
        while stack:
            t = stack.pop()
            if t.node is not None and t.node.id not in live:
                live.add(t.node.id)
                stack.extend(C(i) for i in t.node.inputs)
        uses = Counter()
        for nd in g.nodes:
            if nd.id in live:
                for i in nd.inputs:
                    uses[C(i).id] += 1
        for r in roots:
            uses[r.id] += 1
        self.uses = uses
        kernels: list[Kernel] = []
        producer: dict[int, Kernel] = {}

        def add_input(K: Kernel, t: Tensor):
            for i, x in enumerate(K.in_tensors):
                if x is t:
                    return ("in", i)
            K.in_tensors.append(t)
            K.modes.append(_mode(t.shape, K.shape))
            return ("in", len(K.in_tensors) - 1)

        def add_step(K: Kernel, nd, ins):
            args = [K.internal[t.id] if t.id in K.internal else add_input(K, t) for t in ins]
            K.steps.append(Step(nd.op, args, nd.output.dtype, dict(nd.attrs)))
            K.internal[nd.output.id] = ("s", len(K.steps) - 1)
            K.head = nd.output
            K.nodes.append(nd.id)

        for nd in g.nodes:
            if nd.id not in live or nd.op == "xent":
                continue
            ins = [C(i) for i in nd.inputs]
            if nd.op in EWISE:
                K = None
                if self.o.fuse:
                    for t in ins:
                        P = producer.get(t.id)
                        if (P is not None and P.kind in ("ewise", "gemm") and P.head is t and uses[t.id] == 1
                                and nd.output.shape == P.shape and P.attrs.get("batch", 1) == 1):
                            K = P
                            break
                if K is not None:
                    kernels.remove(K)
                else:
                    K = Kernel("ewise", nd.output.shape)
                add_step(K, nd, ins)
                kernels.append(K)
                producer[nd.output.id] = K
            elif nd.op in ("matmul", "bmm"):
                K = Kernel("gemm", nd.output.shape, attrs={"ta": nd.attrs["ta"], "tb": nd.attrs["tb"],
                                                            "batch": nd.attrs["batch"]})
                K.in_tensors = [ins[0], ins[1]]
                K.modes = ["full", "full"]
                K.internal[nd.output.id] = ("acc",)
                K.head = nd.output
                K.nodes.append(nd.id)
                kernels.append(K)
                producer[nd.output.id] = K
            elif nd.op in ("reduce", "transpose") or nd.op in ROWNORM:
                kind = "rownorm" if nd.op in ROWNORM else nd.op
                attrs = {"fn": nd.op} if kind == "rownorm" else {}
                if kind == "rownorm":
                    attrs["eps"] = nd.attrs["eps"]
                if kind == "reduce":
                    attrs = {"rop": ReduceOp(nd.attrs["rop"]), "axis": nd.attrs["axis"]}
                K = Kernel(kind, nd.output.shape, attrs=attrs)
                K.in_tensors, K.modes = [ins[0]], ["full"]
                K.head = nd.output
                K.nodes.append(nd.id)
                kernels.append(K)
                producer[nd.output.id] = K
            else:
                raise CompileError(f"no lowering for op {nd.op!r}")
        self.producer = producer
        return kernels

    def place_assigns(self, kernels):
        """Parameter updates: write element-wise results in place when provably safe, else copy at the end."""
        out_ids = {self.canon(t).id for t in self.g.outputs.values()}
        inplace, copies = [], []

        def copy_of(value, target):
            ck = Kernel("copy", value.shape, attrs={"target": target})
            ck.in_tensors, ck.modes, ck.head = [value], ["full"], value
            return ck

        for target, value in self.g.assigns:
            v = self.canon(value)
            K = self.producer.get(v.id)
            if (K is not None and K.kind == "ewise" and K.head is v and self.uses[v.id] == 1
                    and v.id not in out_ids and K not in inplace):
                K.attrs["target"] = target
                inplace.append(K)
            else:
                copies.append(copy_of(v, target))
        copy_inputs = {id(x) for ck in copies for x in ck.in_tensors}
        for i, K in enumerate(inplace):
            t = K.attrs["target"]
            later_reads = any(x is t for P in inplace[i + 1:] for x in P.in_tensors)
            if later_reads or id(t) in copy_inputs:
                del K.attrs["target"]
                copies.append(copy_of(K.head, t))
        for K in inplace:
            kernels.remove(K)
        return kernels + inplace + copies

    def plan_memory(self, kernels):
        g, C = self.g, self.canon
        bufs: dict[int, Buffer] = {}
        named: dict[str, Buffer] = {}
        gal = Allocator(self.cfg.global_bytes, 0, 256)

        def leaf(t: Tensor) -> Buffer:
            if t.id not in bufs:
                bufs[t.id] = Buffer(t.name, t.shape, t.dtype, t.kind)
                named[t.name] = bufs[t.id]
            return bufs[t.id]

        for t in g.tensors:
            if t.kind in ("input", "param"):
                leaf(t)
        for K in kernels:
            for t in K.in_tensors:
                if t.kind != "value":
                    leaf(t)
        for name, t in g.outputs.items():
            t = C(t)
            if t.kind != "value":
                named[name] = leaf(t)
            else:
                if t.id not in bufs:
                    bufs[t.id] = Buffer(f"out:{name}", t.shape, t.dtype, "output")
                named[name] = bufs[t.id]
        for K in kernels:
            if K.kind == "copy":
                K.output = leaf(K.attrs["target"])
            elif "target" in K.attrs:
                K.output = leaf(K.attrs["target"])
                bufs[K.head.id] = K.output
            elif K.head.id in bufs:
                K.output = bufs[K.head.id]
            else:
                K.output = bufs[K.head.id] = Buffer(f"t{K.head.id}", K.head.shape, K.head.dtype, "temp")
        if self.share is not None:
            # alias same-named persistent state; keep every other persistent range of the peer intact
            peer = {id(b): b for b in self.share.buffers.values()}
            for b in peer.values():
                gal.reserve(b.addr, b.nbytes)
            for b in {id(b): b for b in bufs.values()}.values():
                pb = self.share.buffers.get(b.name)
                if b.kind in ("param", "const") and pb is not None:
                    if pb.shape != b.shape or pb.dtype != b.dtype:
                        raise CompileError(f"shared buffer {b.name}: {pb.shape}/{pb.dtype.name} vs "
                                           f"{b.shape}/{b.dtype.name}")
                    b.addr = pb.addr
        for b in {id(b): b for b in bufs.values()}.values():
            if b.persistent and b.addr < 0:
                b.addr = gal.alloc(b.nbytes)
        last: dict[int, int] = {}
        for k, K in enumerate(kernels):
            K.inputs = [bufs[t.id] for t in K.in_tensors]
            for b in K.inputs:
                last[id(b)] = k
        scratch_peak = 0
        for k, K in enumerate(kernels):
            if K.output.addr < 0:
                K.output.addr = gal.alloc(K.output.nbytes)
            if K.kind == "reduce" and K.attrs["axis"] == 2:
                sb = Buffer(f"scratch{k}", (self.n, 1), K.output.dtype, "temp")
                sb.addr = gal.alloc(sb.nbytes)
                K.attrs["scratch"] = sb
            for b in {id(x): x for x in K.inputs}.values():
                if not b.persistent and last.get(id(b)) == k and b.addr >= 0 and b is not K.output:
                    gal.release(b.addr)
            if "scratch" in K.attrs:
                gal.release(K.attrs["scratch"].addr)
            scratch_peak = max(scratch_peak, gal.peak)
        self.global_peak = gal.peak
        allb = list({id(b): b for b in list(bufs.values()) + [K.output for K in kernels]}.values())
        return named, allb

    # ------------------------------------------------------------ codegen helpers
    def _fit(self, rows_max, cols, b_full, b_col, b_row, fixed, budget, slots, even=False):
        cc = cols
        while True:
            avail = budget - 64 * (slots + 1) - b_row * cc - fixed
            denom = b_full * cc + b_col
            rc = min(rows_max, int(avail // denom)) if avail > 0 and denom > 0 else (rows_max if avail > 0 else 0)
            if even and cc > 1:
                cc -= cc % 2
            if rc >= 1:
                return rc, cc
            if cc <= 1 or (even and cc <= 2):
                raise CompileError("kernel working set does not fit into LOCAL SRAM")
            cc = (cc + 1) // 2

    def _steps(self, em, core, steps, arg_ref, slots, rc, cc):
        out = None
        for j, st in enumerate(steps):
            dst = _L(slots[j], rc, cc, st.dtype)
            refs = [arg_ref(a, rc, cc) for a in st.args]
            op = st.op
            if op in BINARY:
                (a, ma), (b, mb) = refs
                em.emit(B.vec(_VEC[op], core, dst, a, b, bcast_a=_BC[ma], bcast_b=_BC[mb]))
            elif op in ("scale", "broadcast"):
                (a, ma), = refs
                al, be = (st.attrs["alpha"], st.attrs["beta"]) if op == "scale" else (1.0, 0.0)
                em.emit(B.scale_add(core, dst, a, al, be, bcast_a=_BC[ma]))
            elif op in UNARY:
                (a, ma), = refs
                if ma != "full":
                    em.emit(B.scale_add(core, dst, a, 1.0, 0.0, bcast_a=_BC[ma]))
                    a = dst
                em.emit(B.activation(core, dst, a, UNARY[op]))
            elif op == "cast":
                (a, ma), = refs
                if ma != "full":
                    raise CompileError("cast of a broadcast operand is not supported")
                em.emit(B.convert(core, dst, a, st.attrs["scale"]))
            else:
                raise CompileError(f"step op {op}")
            out = dst
        return out

    @staticmethod
    def _gsub(buf: Buffer, mode, r0, c0, rc, cc):
        if mode == "full":
            return buf.ref.sub(r0, c0, rc, cc)
        if mode == "row":
            return buf.ref.sub(0, c0, 1, cc)
        if mode == "col":
            return buf.ref.sub(r0, 0, rc, 1)
        return buf.ref.sub(0, 0, 1, 1)

    @staticmethod
    def _lshape(mode, rc, cc):
        return {"full": (rc, cc), "row": (1, cc), "col": (rc, 1), "scalar": (1, 1)}[mode]

    # ------------------------------------------------------------ kernels
    def gen_ewise(self, K: Kernel, em: Emitter):
        M, N = K.shape
        sets = 2 if self.o.double_buffer else 1
        ins = list(zip(K.inputs, K.modes))
        even = any(b.dtype == DType.INT4 for b, _ in ins) or any(s.dtype == DType.INT4 for s in K.steps)
        b_full = sum(_eb(b.dtype) for b, m in ins if m == "full") + sum(_eb(s.dtype) for s in K.steps)
        b_col = sum(_eb(b.dtype) for b, m in ins if m == "col")
        b_row = sum(_eb(b.dtype) for b, m in ins if m == "row")
        blocks = _row_blocks(M, self.n)
        rows_max = max(r1 - r0 for _, r0, r1 in blocks)
        rc, cc = self._fit(rows_max, N, b_full * sets, b_col * sets, b_row * sets, 0,
                           self.budget, (len(ins) + len(K.steps)) * sets, even)
        sig = self._sig(M)
        acc = [_acc(b, False, sig if m in ("full", "col") else None) for b, m in ins]
        acc.append(_acc(K.output, True, sig))
        em.sync(acc)
        loc = _Local(self.cfg)
        resident = cc == N
        pslot = {i: loc.take(_eb(b.dtype) * (N if m == "row" else 1)) for i, (b, m) in enumerate(ins)
                 if resident and m in ("row", "scalar")}
        islot = [[loc.take(_eb(b.dtype) * math.prod(self._lshape(m, rc, cc))) if i not in pslot else -1
                  for i, (b, m) in enumerate(ins)] for _ in range(sets)]
        sslot = [[loc.take(_eb(s.dtype) * rc * cc) for s in K.steps] for _ in range(sets)]
        for core, r0, r1 in blocks:
            for i in pslot:
                b, m = ins[i]
                shp = self._lshape(m, 1, N)
                em.emit(B.load(core, _L(pslot[i], *shp, b.dtype), self._gsub(b, m, 0, 0, 1, N)))
            j = 0
            for rr in range(r0, r1, rc):
                rcur = min(rc, r1 - rr)
                for c0 in range(0, N, cc):
                    ccur = min(cc, N - c0)
                    s = j % sets
                    j += 1
                    lrefs = {}
                    for i, (b, m) in enumerate(ins):
                        shp = self._lshape(m, rcur, ccur)
                        if i in pslot:
                            lrefs[i] = (_L(pslot[i], *self._lshape(m, 1, N), b.dtype), m)
                            continue
                        lr = _L(islot[s][i], *shp, b.dtype)
                        em.emit(B.load(core, lr, self._gsub(b, m, rr, c0, rcur, ccur)))
                        lrefs[i] = (lr, m)

                    def arg_ref(a, rc_, cc_, s=s):
                        if a[0] == "in":
                            return lrefs[a[1]]
                        st = K.steps[a[1]]
                        return _L(sslot[s][a[1]], rc_, cc_, st.dtype), "full"

                    last = self._steps(em, core, K.steps, arg_ref, sslot[s], rcur, ccur)
                    em.emit(B.store(core, K.output.ref.sub(rr, c0, rcur, ccur), last))

    def _gemm_tiles(self, M, N, Kd, dt, n_epi_full, b_epi_row, resident, sets):
        R, Cc = self.cfg.mxu_rows, self.cfg.mxu_cols
        even = dt == DType.INT4
        if self.o.gemm_tiles:
            tm, tn, tk = self.o.gemm_tiles
            tm, tn, tk = min(tm, M), min(tn, N), min(tk, Kd)
        else:
            tm, tn = min(M, 4 * R), min(N, 4 * Cc)
            if resident and N <= 8 * Cc:
                tn = N
            tk = Kd
        e = _eb(dt)

        def need(tm, tn, tk):
            a = sets * _al(e * tm * tk)
            b = _al(e * Kd * N) if resident else sets * _al(e * tk * tn)
            return a + b + 2 * _al(4 * tm * tn) + n_epi_full * _al(4 * tm * tn) + _al(b_epi_row * tn) + 1024

        while need(tm, tn, tk) > self.budget:
            if tk > 32 and not resident:
                tk = max(32, tk // 2)
            elif tn > Cc:
                tn = max(Cc, tn // 2)
            elif tm > R:
                tm = max(R, tm // 2)
            elif tk > 1 and not resident:
                tk = max(1, tk // 2)
            elif tn > 1:
                tn = max(1, tn // 2)
            elif tm > 1:
                tm = max(1, tm // 2)
            else:
                raise CompileError("GEMM tile does not fit into LOCAL SRAM")
        if even:
            tm, tn, tk = [max(2, v - v % 2) if v > 1 else v for v in (tm, tn, tk)]
        return tm, tn, tk

    def gen_gemm(self, K: Kernel, em: Emitter):
        a_buf, b_buf = K.inputs[0], K.inputs[1]
        ta, tb, bt = K.attrs["ta"], K.attrs["tb"], K.attrs["batch"]
        Mt, N = K.shape
        M = Mt // bt
        Kd = (a_buf.shape[0] // bt) if ta else a_buf.shape[1]
        dt, acc_dt = a_buf.dtype, accum_dtype(a_buf.dtype)
        epi = list(zip(K.inputs[2:], K.modes[2:]))
        dist = self.o.gemm_distribution
        if dist == "auto":
            dist = "rows" if bt == 1 and M >= self.n * self.cfg.mxu_rows else "tiles"
        if bt != 1:
            dist = "tiles"
        sig = self._sig(M) if dist == "rows" else None
        accs = [_acc(a_buf, False, sig if not ta else None), _acc(b_buf, False, None)]
        accs += [_acc(b, False, sig if m in ("full", "col") else None) for b, m in epi]
        accs.append(_acc(K.output, True, sig))
        if self.o.gemm_macro and not K.steps:
            em.sync([(a[0], a[1], a[2], None) for a in accs])
            for bi in range(bt):
                for core, r0, r1 in _row_blocks(M, self.n):
                    ra = (a_buf.ref.sub(bi * Kd, r0, Kd, r1 - r0) if ta else
                          a_buf.ref.sub(bi * M + r0, 0, r1 - r0, Kd))
                    rows_b = b_buf.shape[0] // bt
                    rb = b_buf.ref.sub(bi * rows_b, 0, rows_b, b_buf.shape[1])
                    em.emit(B.gemm(core, K.output.ref.sub(bi * M + r0, 0, r1 - r0, N), ra, rb, 1.0, 0.0, ta, tb))
            return
        em.sync(accs)
        sets = 2 if self.o.double_buffer else 1
        resident = self.o.resident_b and bt == 1 and nbytes(dt, Kd * N) <= self.budget // 3
        n_epi_full = sum(1 for _, m in epi if m in ("full", "col")) + len(K.steps)
        b_epi_row = sum(4 for _, m in epi if m in ("row", "scalar"))
        tm, tn, tk = self._gemm_tiles(M, N, Kd, dt, n_epi_full, b_epi_row, resident, sets)
        K.attrs["tiles"] = (tm, tn, tk)
        K.attrs["resident_b"] = resident
        K.attrs["distribution"] = dist
        loc = _Local(self.cfg)
        res = loc.take(nbytes(dt, Kd * N)) if resident else -1
        abuf = [loc.take(nbytes(dt, tm * tk)) for _ in range(sets)]
        bbuf = [] if resident else [loc.take(nbytes(dt, tk * tn)) for _ in range(sets)]
        accb = [loc.take(4 * tm * tn) for _ in range(2)]
        eslot = [loc.take(4 * tm * tn if m in ("full", "col") else 4 * tn) for _, m in epi]
        sslot = [loc.take(_eb(s.dtype) * tm * tn) for s in K.steps]
        tiles = []
        if dist == "rows":
            for core, r0, r1 in _row_blocks(M, self.n):
                for i0 in range(r0, r1, tm):
                    for j0 in range(0, N, tn):
                        tiles.append((core, 0, i0, min(tm, r1 - i0), j0))
        else:
            t = 0
            for bi in range(bt):
                for i0 in range(0, M, tm):
                    for j0 in range(0, N, tn):
                        tiles.append((t % self.n, bi, i0, min(tm, M - i0), j0))
                        t += 1
        per_core: dict[int, list] = {}
        for tl in tiles:
            per_core.setdefault(tl[0], []).append(tl)
        bres = None
        if resident:
            bres = _L(res, *b_buf.shape, dt)
        for core, tl in per_core.items():
            if resident:
                em.emit(B.load(core, bres, b_buf.ref))
            u = 0
            for t, (_, bi, i0, mt, j0) in enumerate(tl):
                nt = min(tn, N - j0)
                acc = _L(accb[t % 2], mt, nt, acc_dt)
                for kk, k0 in enumerate(range(0, Kd, tk)):
                    kt = min(tk, Kd - k0)
                    ga = a_buf.ref.sub(bi * Kd + k0, i0, kt, mt) if ta else a_buf.ref.sub(bi * M + i0, k0, mt, kt)
                    la = _L(abuf[u % sets], ga.rows, ga.cols, dt)
                    em.emit(B.load(core, la, ga))
                    if resident:
                        lb = bres.sub(j0, k0, nt, kt) if tb else bres.sub(k0, j0, kt, nt)
                    else:
                        gb = (b_buf.ref.sub(bi * N + j0, k0, nt, kt) if tb else
                              b_buf.ref.sub(bi * Kd + k0, j0, kt, nt))
                        lb = _L(bbuf[u % sets], gb.rows, gb.cols, dt)
                        em.emit(B.load(core, lb, gb))
                    em.emit(B.matmul(core, acc, la, lb, acc=kk > 0, ta=ta, tb=tb))
                    u += 1
                lrefs = {}
                for i, (b, m) in enumerate(epi):
                    lr = _L(eslot[i], *self._lshape(m, mt, nt), b.dtype)
                    em.emit(B.load(core, lr, self._gsub(b, m, i0, j0, mt, nt)))
                    lrefs[i + 2] = (lr, m)

                def arg_ref(a, rc_, cc_, acc=acc):
                    if a[0] == "acc":
                        return acc, "full"
                    if a[0] == "in":
                        return lrefs[a[1]]
                    return _L(sslot[a[1]], rc_, cc_, K.steps[a[1]].dtype), "full"

                last = self._steps(em, core, K.steps, arg_ref, sslot, mt, nt) if K.steps else acc
                em.emit(B.store(core, K.output.ref.sub(bi * M + i0, j0, mt, nt), last))

    def gen_rownorm(self, K: Kernel, em: Emitter):
        x, out = K.inputs[0], K.output
        M, N = K.shape
        sets = 2 if self.o.double_buffer else 1
        blocks = _row_blocks(M, self.n)
        rows_max = max(r1 - r0 for _, r0, r1 in blocks)
        b = _eb(x.dtype) + _eb(out.dtype)
        avail = self.budget // sets - 256
        rc = min(rows_max, int(avail // (b * N)))
        if rc < 1:
            raise CompileError(f"row of {N} elements does not fit into LOCAL SRAM for {K.attrs['fn']}")
        sig = self._sig(M)
        em.sync([_acc(x, False, sig), _acc(out, True, sig)])
        loc = _Local(self.cfg)
        xs = [loc.take(_eb(x.dtype) * rc * N) for _ in range(sets)]
        ys = [loc.take(_eb(out.dtype) * rc * N) for _ in range(sets)]
        fn = {"softmax": Norm.SOFTMAX, "layernorm": Norm.LAYERNORM, "rmsnorm": Norm.RMSNORM}[K.attrs["fn"]]
        for core, r0, r1 in blocks:
            for j, rr in enumerate(range(r0, r1, rc)):
                rcur, s = min(rc, r1 - rr), j % sets
                lx, ly = _L(xs[s], rcur, N, x.dtype), _L(ys[s], rcur, N, out.dtype)
                em.emit(B.load(core, lx, x.ref.sub(rr, 0, rcur, N)),
                        B.normalization(core, ly, lx, fn, K.attrs["eps"] or 1e-5),
                        B.store(core, out.ref.sub(rr, 0, rcur, N), ly))

    def gen_reduce(self, K: Kernel, em: Emitter):
        x, out = K.inputs[0], K.output
        M, N = x.shape
        rop, axis = K.attrs["rop"], K.attrs["axis"]
        dt = x.dtype
        e = _eb(dt)
        part = ReduceOp.MAX if rop == ReduceOp.MAX else ReduceOp.SUM
        loc = _Local(self.cfg)
        sets = 2 if self.o.double_buffer else 1

        def combine(core, acc_ref, tmp_ref):
            if part == ReduceOp.MAX:
                em.emit(B.vec(Opcode.VECTOR_MAX, core, acc_ref, acc_ref, tmp_ref))
            else:
                em.emit(B.accumulate(core, acc_ref, tmp_ref))

        if axis == Axis.COLS:
            blocks = _row_blocks(M, self.n)
            rows_max = max(r1 - r0 for _, r0, r1 in blocks)
            rc, cc = self._fit(rows_max, N, e * sets, 3 * e, 0, 0, self.budget, 2 * sets + 3)
            sig = self._sig(M)
            em.sync([_acc(x, False, sig), _acc(out, True, sig)])
            xs = [loc.take(e * rc * cc) for _ in range(sets)]
            accr, tmpr = loc.take(e * rc), loc.take(e * rc)
            for core, r0, r1 in blocks:
                j = 0
                for rr in range(r0, r1, rc):
                    rcur = min(rc, r1 - rr)
                    acc = _L(accr, rcur, 1, dt)
                    for c0 in range(0, N, cc):
                        ccur = min(cc, N - c0)
                        lx = _L(xs[j % sets], rcur, ccur, dt)
                        j += 1
                        em.emit(B.load(core, lx, x.ref.sub(rr, c0, rcur, ccur)))
                        if cc == N:
                            em.emit(B.reduce(core, acc, lx, rop, Axis.COLS))
                        elif c0 == 0:
                            em.emit(B.reduce(core, acc, lx, part, Axis.COLS))
                        else:
                            tmp = _L(tmpr, rcur, 1, dt)
                            em.emit(B.reduce(core, tmp, lx, part, Axis.COLS))
                            combine(core, acc, tmp)
                    if rop == ReduceOp.MEAN and cc != N:
                        em.emit(B.scale_add(core, acc, acc, 1.0 / N, 0.0))
                    em.emit(B.store(core, out.ref.sub(rr, 0, rcur, 1), acc))
            return
        if axis == Axis.ROWS:
            cb = math.ceil(N / self.n)
            cblocks = [(c, c * cb, min(N, (c + 1) * cb)) for c in range(self.n) if c * cb < N]
            rc, cc = self._fit(M, cb, e * sets, 0, 3 * e, 0, self.budget, 2 * sets + 3)
            em.sync([_acc(x, False, None), _acc(out, True, None)])
            xs = [loc.take(e * rc * cc) for _ in range(sets)]
            accr, tmpr = loc.take(e * cc), loc.take(e * cc)
            for core, c0b, c1b in cblocks:
                j = 0
                for c0 in range(c0b, c1b, cc):
                    ccur = min(cc, c1b - c0)
                    acc, tmp = _L(accr, 1, ccur, dt), _L(tmpr, 1, ccur, dt)
                    for rr in range(0, M, rc):
                        rcur = min(rc, M - rr)
                        lx = _L(xs[j % sets], rcur, ccur, dt)
                        j += 1
                        em.emit(B.load(core, lx, x.ref.sub(rr, c0, rcur, ccur)))
                        if rr == 0:
                            em.emit(B.reduce(core, acc, lx, part, Axis.ROWS))
                        else:
                            em.emit(B.reduce(core, tmp, lx, part, Axis.ROWS))
                            combine(core, acc, tmp)
                    if rop == ReduceOp.MEAN:
                        em.emit(B.scale_add(core, acc, acc, 1.0 / M, 0.0))
                    em.emit(B.store(core, out.ref.sub(0, c0, 1, ccur), acc))
            return
        # Axis.ALL: per-core partials -> scratch -> barrier -> final reduction on core 0
        sb: Buffer = K.attrs["scratch"]
        blocks = _row_blocks(M, self.n)
        rows_max = max(r1 - r0 for _, r0, r1 in blocks)
        rc, cc = self._fit(rows_max, N, e * sets, 0, 0, 4 * 64, self.budget, 2 * sets + 3)
        sig = self._sig(M)
        em.sync([_acc(x, False, sig), _acc(sb, True, None), _acc(out, True, None)])
        xs = [loc.take(e * rc * cc) for _ in range(sets)]
        accr, tmpr = loc.take(e), loc.take(e)
        fin = loc.take(e * self.n)
        for core, r0, r1 in blocks:
            acc, tmp = _L(accr, 1, 1, dt), _L(tmpr, 1, 1, dt)
            j = 0
            for rr in range(r0, r1, rc):
                rcur = min(rc, r1 - rr)
                for c0 in range(0, N, cc):
                    ccur = min(cc, N - c0)
                    lx = _L(xs[j % sets], rcur, ccur, dt)
                    em.emit(B.load(core, lx, x.ref.sub(rr, c0, rcur, ccur)))
                    em.emit(B.reduce(core, acc if j == 0 else tmp, lx, part, Axis.ALL))
                    if j:
                        combine(core, acc, tmp)
                    j += 1
            em.emit(B.store(core, sb.ref.sub(core, 0, 1, 1), acc))
        nb = len(blocks)
        if nb > 1:
            em.barrier()
            em.sync([_acc(sb, False, None), _acc(out, True, None)])
        lf, res = _L(fin, nb, 1, dt), _L(accr, 1, 1, dt)
        if nb > 1:
            em.emit(B.load(0, lf, sb.ref.sub(0, 0, nb, 1)), B.reduce(0, res, lf, part, Axis.ALL))
        if rop == ReduceOp.MEAN:
            em.emit(B.scale_add(0, res, res, 1.0 / (M * N), 0.0))
        em.emit(B.store(0, out.ref, res))

    def gen_transpose(self, K: Kernel, em: Emitter):
        x, out = K.inputs[0], K.output
        M, N = x.shape
        e = _eb(x.dtype)
        sets = 2 if self.o.double_buffer else 1
        t = 64
        while 2 * sets * _al(e * t * t) + 512 > self.budget and t > 1:
            t //= 2
        em.sync([_acc(x, False, None), _acc(out, True, None)])
        loc = _Local(self.cfg)
        ins_ = [loc.take(e * t * t) for _ in range(sets)]
        outs = [loc.take(e * t * t) for _ in range(sets)]
        k = 0
        cnt: Counter = Counter()
        for r0 in range(0, M, t):
            for c0 in range(0, N, t):
                rc, cc = min(t, M - r0), min(t, N - c0)
                core = k % self.n
                s = cnt[core] % sets
                cnt[core] += 1
                k += 1
                li, lo = _L(ins_[s], rc, cc, x.dtype), _L(outs[s], cc, rc, x.dtype)
                em.emit(B.load(core, li, x.ref.sub(r0, c0, rc, cc)), B.copy(core, lo, li, transpose=True),
                        B.store(core, out.ref.sub(c0, r0, cc, rc), lo))

    def gen_copy(self, K: Kernel, em: Emitter):
        x, out = K.inputs[0], K.output
        M, N = x.shape
        sig = self._sig(M)
        em.sync([_acc(x, False, sig), _acc(out, True, sig)])
        for core, r0, r1 in _row_blocks(M, self.n):
            em.emit(B.copy(core, out.ref.sub(r0, 0, r1 - r0, N), x.ref.sub(r0, 0, r1 - r0, N)))

    # ------------------------------------------------------------ driver
    def compile(self) -> Executable:
        g = self.g
        kernels = self.form_kernels()
        kernels = self.place_assigns(kernels)
        named, allb = self.plan_memory(kernels)
        em = Emitter(self.n, self.o.barrier_elim)
        gen = {"ewise": self.gen_ewise, "gemm": self.gen_gemm, "rownorm": self.gen_rownorm,
               "reduce": self.gen_reduce, "transpose": self.gen_transpose, "copy": self.gen_copy}
        for K in kernels:
            gen[K.kind](K, em)
        meta = {"graph": g.name, "config": self.cfg.name, "config_hash": self.cfg.config_hash(),
                "isa": "%d.%d" % ISA_VERSION, "options": asdict(self.o)}
        prog = Program(em.ins, {}, meta).validate(self.cfg)
        module = Module(g.name, kernels, allb)
        init = {t.name: np.asarray(t.value) for t in g.tensors if t.kind in ("param", "const") and t.name in named}
        kinds = Counter(K.kind for K in kernels)
        stats = {
            "graph_nodes": len(g.nodes), "kernels": len(kernels), "kernels_by_kind": dict(kinds),
            "fused_nodes": sum(len(K.nodes) for K in kernels if len(K.nodes) > 1),
            "instructions": len(em.ins), "barriers": em.barriers, "barriers_avoided": em.barriers_avoided,
            "global_bytes_peak": self.global_peak, "cores_used": self.n,
            "inplace_updates": sum(1 for K in kernels if K.kind != "copy" and "target" in K.attrs),
        }
        return Executable(prog, module, self.cfg, named, init,
                          [t.name for t in g.inputs()], list(g.outputs), [t.name for t in g.params()],
                          stats, self.o)


def compile_graph(g: Graph, cfg, opts: CompileOptions | None = None,
                  share_with: Executable | None = None) -> Executable:
    """Compile ``g`` for ``cfg``. With ``share_with``, same-named params/consts alias that executable's buffers."""
    return Compiler(g, cfg, opts, share_with).compile()
