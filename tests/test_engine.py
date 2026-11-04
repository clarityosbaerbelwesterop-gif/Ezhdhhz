"""Command processor, scheduling, synchronization, memory-system timing and fault handling."""
import numpy as np
import pytest

from titan.dtypes import DType
from titan.emu import CYCLE, FUNCTIONAL, Machine, TitanFault, run
from titan.isa import ErrorCode, Opcode, Program, Space, TensorRef
from titan.isa import build as B
from tests.conftest import Placer


def _mm_program(cfg, m_, n_, k_, gp, lp, core=0):
    a, b, c = gp(m_, k_, DType.FP32), gp(k_, n_, DType.FP32), gp(m_, n_, DType.FP32)
    la, lb, lc = lp(m_, k_, DType.FP32), lp(k_, n_, DType.FP32), lp(m_, n_, DType.FP32)
    return (a, b, c), [B.load(core, la, a), B.load(core, lb, b), B.matmul(core, lc, la, lb), B.store(core, c, lc)]


def test_functional_and_cycle_modes_agree_on_data(tiny, rng):
    gp, lp = Placer(Space.GLOBAL), Placer(Space.LOCAL)
    (a, b, c), prog = _mm_program(tiny, 8, 8, 8, gp, lp)
    A, Bm = rng.standard_normal((8, 8)).astype(np.float32), rng.standard_normal((8, 8)).astype(np.float32)
    outs = {}
    for mode in (FUNCTIONAL, CYCLE):
        m = Machine(tiny)
        m.host_write(a, A)
        m.host_write(b, Bm)
        r = run(m, Program(prog), mode)
        outs[mode] = m.host_read(c)
        assert (r.cycles is None) == (mode == FUNCTIONAL)
    np.testing.assert_array_equal(outs[FUNCTIONAL], outs[CYCLE])
    np.testing.assert_allclose(outs[CYCLE], A @ Bm, rtol=1e-5)


def test_signal_wait_orders_cores(tiny, rng):
    gp = Placer(Space.GLOBAL)
    src, mid, dst = gp(4, 8, DType.FP32), gp(4, 8, DType.FP32), gp(4, 8, DType.FP32)
    l0, l1 = TensorRef(Space.LOCAL, 0, 4, 8, DType.FP32), TensorRef(Space.LOCAL, 0, 4, 8, DType.FP32)
    from titan.isa import Act
    prog = Program([
        B.wait(1, 0, 1),                        # consumer first in the stream: must block
        B.load(1, l1, mid), B.activation(1, l1, l1, Act.EXP), B.store(1, dst, l1, notify=True),
        B.load(0, l0, src), B.scale_add(0, l0, l0, 2.0, 0.0), B.store(0, mid, l0, notify=True), B.signal(0, 0, 1),
    ])
    x = rng.standard_normal((4, 8)).astype(np.float32)
    for mode in (FUNCTIONAL, CYCLE):
        m = Machine(tiny)
        m.host_write(src, x)
        r = run(m, prog, mode)
        np.testing.assert_allclose(m.host_read(dst), np.exp(2 * x), rtol=1e-5)
        if mode == CYCLE:
            t = {c["opcode"] + str(c["core"]): c["time"] for c in r.completions}
            assert t["STORE1"] > t["STORE0"]
            assert r.counters["stall_cycles"]["sync_wait"] > 0


def test_deadlock_detected(tiny):
    with pytest.raises(TitanFault) as e:
        run(Machine(tiny), Program([B.wait(0, 1, 1)]))
    assert e.value.code == ErrorCode.DEADLOCK


def test_barrier_synchronizes_all_cores(tiny, rng):
    gp = Placer(Space.GLOBAL)
    g = [gp(2, 4, DType.FP32) for _ in range(4)]
    loc = TensorRef(Space.LOCAL, 0, 2, 4, DType.FP32)
    prog = Program([
        B.load(0, loc, g[0]), B.scale_add(0, loc, loc, 1.0, 1.0), B.store(0, g[1], loc),
        B.barrier(0),
        B.load(1, loc, g[1]), B.scale_add(1, loc, loc, 3.0, 0.0), B.store(1, g[2], loc),
    ])
    x = rng.standard_normal((2, 4)).astype(np.float32)
    for mode in (FUNCTIONAL, CYCLE):
        m = Machine(tiny)
        m.host_write(g[0], x)
        r = run(m, prog, mode)
        np.testing.assert_allclose(m.host_read(g[2]), 3 * (x + 1), rtol=1e-6)
    assert r.counts["BARRIER"] == tiny.cores


def test_kernels_with_relocation_and_errors(tiny, rng):
    gp = Placer(Space.GLOBAL)
    a, b = gp(2, 4, DType.FP32), gp(2, 4, DType.FP32)
    loc = TensorRef(Space.LOCAL, 0, 2, 4, DType.FP32)
    kern = [B.load(0, loc, a), B.scale_add(0, loc, loc, -1.0, 0.0), B.store(0, b, loc)]
    off = 4096
    prog = Program([B.execute_kernel(0, 1), B.execute_kernel(1, 1, off)], {1: kern})
    x, y = rng.standard_normal((2, 4)).astype(np.float32), rng.standard_normal((2, 4)).astype(np.float32)
    m = Machine(tiny)
    m.host_write(a, x)
    m.host_write(a.with_addr(a.addr + off), y)
    run(m, prog)
    np.testing.assert_array_equal(m.host_read(b), -x)
    np.testing.assert_array_equal(m.host_read(b.with_addr(b.addr + off)), -y)
    deep = Program([B.execute_kernel(0, 1)], {1: [B.execute_kernel(0, 1)]})
    with pytest.raises(TitanFault) as e:
        run(Machine(tiny), deep)
    assert e.value.code == ErrorCode.KERNEL_DEPTH
    with pytest.raises(TitanFault) as e:
        run(Machine(tiny), Program([B.execute_kernel(0, 1, 1 << 30)], {1: kern}))
    assert e.value.code == ErrorCode.OUT_OF_BOUNDS


def test_invalid_program_is_rejected_before_execution(tiny):
    bad = Program([B.load(0, TensorRef(Space.LOCAL, 0, 2, 2, DType.FP32), TensorRef(Space.GLOBAL, 0, 2, 3, DType.FP32))])
    with pytest.raises(TitanFault) as e:
        run(Machine(tiny), bad)
    assert e.value.code == ErrorCode.SHAPE_MISMATCH


def _tiled_loads(cfg, double: bool, n_tiles=8):
    """Stream n_tiles BF16 tiles from global; each feeds a matmul of comparable duration."""
    gp, lp = Placer(Space.GLOBAL), Placer(Space.LOCAL)
    src = [gp(64, 64, DType.BF16) for _ in range(n_tiles)]
    w = lp(4, 64, DType.BF16)
    bufs = [lp(64, 64, DType.BF16), lp(64, 64, DType.BF16)]
    acc = lp(4, 64, DType.FP32)
    prog = [B.load(0, w, gp(4, 64, DType.BF16))]
    for i in range(n_tiles):
        buf = bufs[i % 2] if double else bufs[0]
        prog += [B.load(0, buf, src[i]), B.matmul(0, acc, w, buf, acc=i > 0)]
    return Program(prog)


def test_double_buffering_overlaps_dma_and_compute(tiny):
    single = run(Machine(tiny), _tiled_loads(tiny, False), CYCLE)
    double = run(Machine(tiny), _tiled_loads(tiny, True), CYCLE)
    assert double.cycles < 0.85 * single.cycles
    assert single.ops == double.ops


def _stream(cores):
    gp = Placer(Space.GLOBAL)
    out = []
    for c in cores:
        for _ in range(4):
            out.append(B.load(c, TensorRef(Space.LOCAL, 0, 64, 64, DType.FP32), gp(64, 64, DType.FP32)))
    return Program(out)


def test_global_bandwidth_contention(tiny):
    # HBM slower than one core's NoC link: two cores must share HBM and take ~2x as long.
    cfg = tiny.replace(hbm_gbps=16.0)
    one = run(Machine(cfg), _stream([0]), CYCLE)
    two = run(Machine(cfg), _stream([0, 1]), CYCLE)
    assert two.cycles > 1.8 * one.cycles
    assert two.counters["hbm_bytes"] == 2 * one.counters["hbm_bytes"]


def test_no_contention_when_link_bound(tiny):
    # Default tiny: per-core link (32 B/cycle) is the bottleneck, HBM (64 B/cycle) serves two cores.
    one = run(Machine(tiny), _stream([0]), CYCLE)
    two = run(Machine(tiny), _stream([0, 1]), CYCLE)
    assert two.cycles < 1.05 * one.cycles


def test_l2_cache_hits_on_reuse(tiny):
    g = TensorRef(Space.GLOBAL, 0, 16, 16, DType.FP32)
    loc = TensorRef(Space.LOCAL, 0, 16, 16, DType.FP32)
    r = run(Machine(tiny), Program([B.load(0, loc, g), B.load(1, loc, g)]), CYCLE)
    l2 = r.counters["l2"]
    assert l2["read_misses"] == 16 * 16 * 4 // tiny.l2_line
    assert l2["read_hits"] == l2["read_misses"]


def test_queue_depth_backpressure(tiny):
    loc = TensorRef(Space.LOCAL, 0, 64, 64, DType.FP32)
    g = TensorRef(Space.GLOBAL, 0, 64, 64, DType.FP32)
    r = run(Machine(tiny), Program([B.load(0, loc, g)] * 40 + [B.nop(1)]), CYCLE)
    assert r.counters["cp_dispatch_stall_cycles"] > 0
    assert r.counters["queue_delay_cycles"] > 0


@pytest.mark.parametrize("dt", [DType.FP32, DType.BF16, DType.INT8, DType.INT4])
def test_gemm_macro(tiny, rng, dt):
    m_, n_, k_ = 22, 18, 30
    gp = Placer(Space.GLOBAL)
    a, b = gp(m_, k_, dt), gp(n_, k_, dt)
    acc_dt = DType.INT32 if dt.is_int else DType.FP32
    c = gp(m_, n_, acc_dt)
    if dt.is_int:
        lo = -8 if dt == DType.INT4 else -128
        A, Bm = rng.integers(lo, -lo, (m_, k_)), rng.integers(lo, -lo, (n_, k_))
        C0 = rng.integers(-50, 50, (m_, n_))
        alpha, beta = 1.0, 1.0
        want = A @ Bm.T + C0
    else:
        from titan.dtypes import round_to
        A = round_to(dt, rng.standard_normal((m_, k_)).astype(np.float32))
        Bm = round_to(dt, rng.standard_normal((n_, k_)).astype(np.float32))
        C0 = rng.standard_normal((m_, n_)).astype(np.float32)
        alpha, beta = 0.5, -1.0
        want = alpha * A.astype(np.float64) @ Bm.T + beta * C0
    for mode in (FUNCTIONAL, CYCLE):
        m = Machine(tiny)
        m.host_write(a, A)
        m.host_write(b, Bm)
        m.host_write(c, C0)
        r = run(m, Program([B.gemm(1, c, a, b, alpha, beta, tb=True, notify=True)]), mode)
        got = m.host_read(c)
        if dt.is_int:
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
        assert r.counts["MATMUL"] >= 1 and len(r.completions) == 1


def test_trace_export(tiny, tmp_path):
    gp, lp = Placer(Space.GLOBAL), Placer(Space.LOCAL)
    _, prog = _mm_program(tiny, 4, 4, 4, gp, lp)
    from titan.emu import Engine
    r = Engine(Machine(tiny), CYCLE, trace=True).run(Program(prog))
    assert [t["op"] for t in r.trace] == ["LOAD", "LOAD", "MATMUL", "STORE"]
    assert all(t["end"] >= t["start"] >= t["issue"] for t in r.trace)
    r.chrome_trace(str(tmp_path / "t.json"))
    assert (tmp_path / "t.json").stat().st_size > 0
    assert Opcode.MATMUL.name in r.counts
