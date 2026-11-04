"""`titan` command-line interface.

    titan info [--arch dev]             architecture summary (analytical figures labelled as such)
    titan backends                      execution backends and their real availability
    titan quickstart [...]              one-command training on the virtual accelerator
    titan matmul --m --n --k            vertical slice: graph -> IR -> ISA -> execution -> verification
    titan bench [--suite quick]         reproducible benchmark report (verified / simulated / analytical / measured)
    titan tune --m --n --k              verified tiling search against the compiler baseline
    titan scale / dp-train              multi-device scaling study / data-parallel training
    titan rtl-verify                    RTL (Verilator) vs emulator bit-exact correspondence
    titan synth                         Yosys resource estimates (pre-place-and-route)
    titan asm FILE [--run]              assemble (and run) a TITAN assembly program
    titan disasm FILE                   disassemble a TITAN binary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import list_presets, load_config
from .dtypes import DType


def _print_kv(d: dict, indent: int = 2) -> None:
    w = max((len(k) for k in d), default=0)
    for k, v in d.items():
        print(" " * indent + f"{k:<{w}}  {v}")


# ------------------------------------------------------------------ info / backends

def cmd_info(a) -> int:
    cfg = load_config(a.arch)
    print(f"{cfg.name}  ({cfg.description})  config hash {cfg.config_hash()}")
    _print_kv({"cores": cfg.cores, "MXU per core": f"{cfg.mxu_rows}x{cfg.mxu_cols} PEs",
               "VPU / SFU lanes": f"{cfg.vpu_lanes} / {cfg.sfu_lanes}", "clock (assumed)": f"{cfg.clock_ghz} GHz",
               "LOCAL SRAM / core": f"{cfg.local_sram_kib} KiB ({cfg.gemm_staging_kib} KiB GEMM staging)",
               "SHARED SRAM": f"{cfg.shared_sram_kib} KiB", "L2": f"{cfg.l2_kib} KiB, {cfg.l2_ways}-way",
               "global memory": f"{cfg.global_mem_gib} GiB @ {cfg.hbm_gbps} GB/s (emulated: "
                                f"{cfg.emulated_global_mib} MiB)"})
    print("\nAnalytical dense peaks of this configuration (estimates, not measurements):")
    for dt in (DType.INT4, DType.INT8, DType.FP16, DType.BF16, DType.FP32):
        print(f"  {dt.name:5s} {cfg.peak_ops(dt) / 1e12:12.3f} TOPS   ridge point {cfg.ridge_point(dt):8.1f} ops/B")
    try:
        from .perf.reference import compare_peak
        print("\nPaper-peak comparison with NVIDIA B300 (same precision, dense, vendor/third-party figures):")
        for dt in (DType.BF16, DType.FP16):
            c = compare_peak(cfg, dt)
            print(f"  {dt.name:5s} TITAN {c['titan_peak_tflops']:.1f} TFLOPS vs B300 {c['b300_peak_tflops']:.1f} "
                  f"TFLOPS -> compute ratio {c['ratio']:.4g}x, memory-bandwidth ratio {c['bandwidth_ratio']:.3g}x")
        print(f"  ({c['label']}; memory-bound workloads scale with the bandwidth ratio, not the compute ratio)")
    except ImportError:
        pass
    return 0


def cmd_backends(a) -> int:
    from .runtime import backend_status
    print("TITAN execution backends:")
    for b, ok, why in backend_status():
        print(f"  [{'x' if ok else ' '}] {b.label:20s} {why}")
        print(f"      results are: {b.kind}")
    return 0


# ------------------------------------------------------------------ quickstart

def _ask(prompt: str, options: list[str], default: str) -> str:
    opts = "/".join(options)
    while True:
        r = input(f"{prompt} [{opts}] (default {default}): ").strip() or default
        if r in options:
            return r
        print(f"  please choose one of: {opts}")


def cmd_quickstart(a) -> int:
    from .runtime import BACKENDS, backend_status
    from .train import PROFILES, TrainConfig, Trainer
    from .train.datasets import DATASETS
    if a.interactive:
        print("ROUGE TITAN QuickStart\n")
        a.model = _ask("1. model", sorted(PROFILES), a.model)
        ds_opts = ["text"] if a.model == "transformer" else [d for d in sorted(DATASETS) if d != "text"]
        a.dataset = _ask("2. dataset", ds_opts, ds_opts[0] if a.dataset is None else a.dataset)
        a.profile = _ask("3. training profile", sorted(PROFILES[a.model]), a.profile)
        b = input(f"4. batch size (default {PROFILES[a.model][a.profile]['batch']}): ").strip()
        a.batch = int(b) if b else None
        a.precision = _ask("   precision", ["fp32", "bf16", "fp16"], a.precision)
        print("5. execution backend:")
        for bk, ok, why in backend_status():
            print(f"     {bk.key:10s} {'OK ' if ok else 'N/A'} {why}")
        a.backend = _ask("   backend", list(BACKENDS), a.backend)
        if input("6. QuickStart now? [Y/n]: ").strip().lower() in ("n", "no"):
            return 0
    tc = TrainConfig(model=a.model, dataset=a.dataset, profile=a.profile, batch=a.batch, precision=a.precision,
                     backend=a.backend, steps=a.steps, lr=a.lr, optimizer=a.optimizer, seed=a.seed,
                     eval_every=a.eval_every, ckpt_every=a.ckpt_every, arch=a.arch, out_dir=a.out)
    try:
        tr = Trainer(tc)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.resume:
        tr.load_checkpoint(a.resume)
    s = tr.run()
    tr.close()
    print("\nSummary:")
    _print_kv({k: v for k, v in s.items() if k not in ("placement", "provenance")})
    print(f"\nexecuted on TITAN: {', '.join(s['placement']['titan'])}")
    print(f"executed on host : {', '.join(s['placement']['host'])}")
    print(f"run directory    : {tr.out}  (config.json, metrics.jsonl, log.txt, checkpoints/, summary.json)")
    return 0


# ------------------------------------------------------------------ vertical slice

def cmd_matmul(a) -> int:
    from .compiler import CompileOptions, Graph, compile_graph, run_reference
    from .isa import disassemble
    from .perf.analytical import gemm_roofline
    from .runtime import Device
    cfg = load_config(a.arch)
    dt = DType.parse(a.dtype)
    g = Graph("matmul")
    x, w = g.input("a", (a.m, a.k), dt), g.input("b", (a.k, a.n), dt)
    g.output("c", g.matmul(x, w))
    exe = compile_graph(g, cfg, CompileOptions(gemm_macro=a.macro))
    print("== TITAN IR ==")
    print(exe.module.dump())
    print(f"\n== TITAN ISA ({len(exe.program)} instructions; first {min(a.show, len(exe.program))}) ==")
    for ins in exe.program.instructions[:a.show]:
        print("  " + disassemble(ins))
    rng = np.random.default_rng(a.seed)
    if dt.is_int:
        lim = 8 if dt == DType.INT4 else 128
        feeds = {"a": rng.integers(-lim, lim, (a.m, a.k)), "b": rng.integers(-lim, lim, (a.k, a.n))}
    else:
        feeds = {"a": rng.standard_normal((a.m, a.k)), "b": rng.standard_normal((a.k, a.n))}
    ref, _ = run_reference(g, feeds)
    report = {"workload": f"C[{a.m}x{a.n}] = A[{a.m}x{a.k}] @ B[{a.k}x{a.n}]", "precision": dt.name,
              "config": cfg.name}
    for mode in ("functional", "cycle"):
        out = Device(cfg).run(exe, feeds, mode)
        err = float(np.abs(out.outputs["c"].astype(np.float64) - ref["c"]).max())
        r = out.result.summary()
        r["max_abs_error_vs_reference"] = err
        report[mode] = r
    if a.rtl:
        from .rtl.cosim import MXUTileSim, make_matmul_hook
        if not dt.is_int:
            print("error: the RTL backend implements the INT8/INT4 matrix engine only", file=sys.stderr)
            return 2
        hook = make_matmul_hook(MXUTileSim(8, 8, 8))
        out = Device(cfg).run(exe, feeds, "functional", matmul_hook=hook)
        report["rtl_cosim"] = {"bit_exact_vs_reference": bool(np.array_equal(out.outputs["c"], ref["c"])),
                               "matmuls_on_rtl": hook.stats.get("rtl_matmuls", 0),
                               "method": "MATMUL executed by the Verilator model of rtl/titan_mxu_tile.sv; "
                                         "DMA/control in the functional model"}
    report["analytical"] = gemm_roofline(cfg, a.m, a.n, a.k, dt)
    print("\n== results (modes are reported separately) ==")
    print(json.dumps(report, indent=2, default=float))
    ok = report["functional"]["max_abs_error_vs_reference"] <= (0 if dt.is_int else 1e-3 * max(1, a.k ** 0.5))
    ok = ok and report.get("rtl_cosim", {}).get("bit_exact_vs_reference", True)
    print("\nVERIFICATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------ performance

def cmd_bench(a) -> int:
    from .perf.benchmark import run_suite, write_report
    res = run_suite(a.suite, a.arch, a.seed)
    jp, mp = write_report(res, a.out)
    print(f"\nreport: {mp}\nraw data: {jp}")
    return 0 if all(r["verification"]["passed"] for r in res["records"]) else 1


def cmd_tune(a) -> int:
    from .tune import tune_gemm
    cfg = load_config(a.arch)
    r = tune_gemm(cfg, a.m, a.n, a.k, DType.parse(a.dtype), a.seed, a.candidates, log=print if a.verbose else None)
    best = r["best"]
    print(f"{r['workload']} on {cfg.name}: baseline {r['baseline']['cycles']:.0f} simulated cycles")
    if best:
        print(f"best verified candidate: {best['cycles']:.0f} cycles, options {best['options']}")
    print(f"speed-up vs baseline: {r['speedup_vs_baseline']:.3f}x  ({r['candidates_evaluated']} evaluated, "
          f"{r['candidates_rejected']} rejected)  method: {r['method']}")
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2, default=str))
    return 0


# ------------------------------------------------------------------ multi-chip

def cmd_scale(a) -> int:
    from .multichip import scaling_study
    r = scaling_study(tuple(a.devices), a.topology, a.arch, a.global_batch, tuple(a.hidden),
                      chiplets_per_package=a.chiplets, link_gbps=a.link_gbps, latency_us=a.latency_us)
    print(r["workload"], "on", r["arch"])
    print(f"{'devices':>7} {'system':>22} {'compute us':>11} {'comm us':>9} {'step us':>9} {'speedup':>8} "
          f"{'efficiency':>10} {'comm %':>7}")
    for x in r["rows"]:
        print(f"{x['devices']:>7} {x['system']:>22} {x['sim_compute_s'] * 1e6:>11.2f} {x['model_comm_s'] * 1e6:>9.2f} "
              f"{x['step_s'] * 1e6:>9.2f} {x['speedup']:>8.2f} {100 * x['efficiency']:>9.1f}% "
              f"{100 * x['comm_fraction']:>6.1f}%")
    print("method:", r["method"])
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2, default=str))
    return 0


def cmd_dp_train(a) -> int:
    from .multichip import DataParallelTrainer
    tr = DataParallelTrainer(a.devices, arch=a.arch, dataset=a.dataset, global_batch=a.global_batch, lr=a.lr,
                             seed=a.seed, mode=a.mode)
    for i in range(a.steps):
        r = tr.step()
        if not r["replicas_identical"]:
            print(f"error: replicas diverged at step {i}", file=sys.stderr)
            return 1
        if (i + 1) % max(1, a.steps // 5) == 0:
            msg = f"step {i + 1:4d} loss {r['loss']:.4f} replicas identical"
            if "step_s" in r:
                msg += (f" | simulated step {r['step_s'] * 1e6:.1f} us (compute {r['sim_compute_s'] * 1e6:.1f}, "
                        f"all-reduce {r['model_comm_s'] * 1e6:.1f} modelled)")
            print(msg)
    print(f"eval accuracy (device 0): {tr.evaluate():.3f}   devices={a.devices} system={tr.sys.name}")
    return 0


# ------------------------------------------------------------------ RTL

def cmd_rtl_verify(a) -> int:
    from .rtl.cosim import correspondence_test
    rc = 0
    for dw in a.data_w:
        r = correspondence_test(a.cases, a.seed, a.rows, a.cols, dw)
        print(f"INT{dw} {r['tile']} tile: {r['cases']} random MATMUL cases, {r['mismatches']} mismatches, "
              f"{r['rtl_jobs']} RTL jobs, {r['rtl_cycles']} RTL cycles  [{r['simulator']}]")
        rc |= r["mismatches"] != 0
    print("method: RTL co-simulation (Verilator) vs TITAN functional golden model, bit-exact comparison")
    return int(rc)


def cmd_synth(a) -> int:
    from .rtl.synth import synthesize, yosys_version
    rows = []
    for top, params in (("rtl_mac_array", {}), ("titan_mxu_tile", {"ROWS": 4, "COLS": 4, "DATA_W": 8}),
                        ("titan_mxu_tile", {"ROWS": 8, "COLS": 8, "DATA_W": 8}),
                        ("titan_mxu_tile", {"ROWS": 8, "COLS": 8, "DATA_W": 4})):
        for target in a.targets:
            r = synthesize(top, params, target, a.out)
            rows.append(r)
            print(f"{top:15s} {str(params):40s} {target:8s} cells={r['total_cells']:6d} {r.get('summary', '')}")
    pnr = []
    if a.pnr:
        from .rtl.synth import place_and_route_ecp5
        for params, dev in (({"ROWS": 4, "COLS": 4, "DATA_W": 8}, "25k"), ({"ROWS": 8, "COLS": 8, "DATA_W": 8}, "85k"),
                            ({"ROWS": 8, "COLS": 8, "DATA_W": 4}, "85k")):
            r = place_and_route_ecp5("titan_mxu_tile", params, dev, out_dir=a.out)
            pnr.append(r)
            print(f"P&R titan_mxu_tile {params} on {r['device']}: {r['status']} "
                  f"Fmax(STA) {r.get('fmax_mhz')} util {r.get('utilization', r.get('reason'))}")
    if a.out:
        out = Path(a.out)
        (out / "synth_summary.json").write_text(json.dumps({"tool": yosys_version(), "results": rows,
                                                            "place_and_route": pnr}, indent=2))
        print(f"logs and summary in {out}")
    print("method: Yosys synthesis + technology mapping (cell counts, pre-place-and-route)" +
          ("; nextpnr-ecp5 out-of-context P&R, Fmax = post-route static timing estimate" if a.pnr else "") +
          ". No power figures and no FPGA measurement.")
    return 0


# ------------------------------------------------------------------ assembler

def cmd_asm(a) -> int:
    from .emu import Machine, run
    from .isa import assemble, encode_program
    prog = assemble(Path(a.file).read_text())
    cfg = load_config(a.arch)
    prog.validate(cfg)
    print(f"assembled {len(prog)} instructions, {len(prog.kernels)} kernels: valid for {cfg.name}")
    if a.out:
        Path(a.out).write_bytes(encode_program(prog))
        print(f"wrote {a.out}")
    if a.run:
        r = run(Machine(cfg), prog, a.mode)
        print(json.dumps(r.summary(), indent=2))
    return 0


def cmd_disasm(a) -> int:
    from .isa import decode_program, disassemble_program
    print(disassemble_program(decode_program(Path(a.file).read_bytes())), end="")
    return 0


# ------------------------------------------------------------------ main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="titan", description="ROUGE TITAN / TITAN-X1 virtual accelerator toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)
    arch_help = f"architecture preset {list_presets()} or TOML path"

    s = sub.add_parser("info", help="architecture summary")
    s.add_argument("--arch", default="dev", help=arch_help)
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("backends", help="list execution backends")
    s.set_defaults(fn=cmd_backends)

    s = sub.add_parser("quickstart", help="train a model on the virtual accelerator")
    s.add_argument("--model", default="mlp", choices=["mlp", "transformer"])
    s.add_argument("--dataset", default=None)
    s.add_argument("--profile", default="small", choices=["tiny", "small", "medium"])
    s.add_argument("--batch", type=int, default=None)
    s.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    s.add_argument("--backend", default="functional", choices=["functional", "cycle", "rtl", "fpga", "hardware"])
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--lr", type=float, default=None)
    s.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--eval-every", type=int, default=50)
    s.add_argument("--ckpt-every", type=int, default=100)
    s.add_argument("--arch", default="dev", help=arch_help)
    s.add_argument("--out", default=None, help="run directory")
    s.add_argument("--resume", default=None, help="checkpoint .npz to resume from")
    s.add_argument("-i", "--interactive", action="store_true", help="guided selection")
    s.set_defaults(fn=cmd_quickstart)

    s = sub.add_parser("matmul", help="vertical slice: compile and execute one matrix multiplication")
    s.add_argument("--m", type=int, default=64)
    s.add_argument("--n", type=int, default=64)
    s.add_argument("--k", type=int, default=64)
    s.add_argument("--dtype", default="fp32")
    s.add_argument("--arch", default="dev", help=arch_help)
    s.add_argument("--macro", action="store_true", help="lower to the GEMM macro instruction")
    s.add_argument("--rtl", action="store_true", help="also execute MATMULs on the Verilator RTL model (INT8/INT4)")
    s.add_argument("--show", type=int, default=12, help="instructions to print")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_matmul)

    s = sub.add_parser("bench", help="run the reproducible benchmark suite")
    s.add_argument("--suite", default="quick", choices=["quick", "default"])
    s.add_argument("--arch", default="dev", help=arch_help)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="reports")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("tune", help="verified GEMM tiling/scheduling search vs baseline")
    s.add_argument("--m", type=int, default=256)
    s.add_argument("--n", type=int, default=256)
    s.add_argument("--k", type=int, default=256)
    s.add_argument("--dtype", default="bf16")
    s.add_argument("--arch", default="dev", help=arch_help)
    s.add_argument("--candidates", type=int, default=48)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=None)
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_tune)

    s = sub.add_parser("scale", help="multi-device strong-scaling study (cycle model + link model)")
    s.add_argument("--devices", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    s.add_argument("--topology", default="ring", choices=["ring", "fully_connected", "mesh2d", "switch"])
    s.add_argument("--chiplets", type=int, default=0, help="chiplets per package (hierarchical system)")
    s.add_argument("--link-gbps", type=float, default=900.0, help="ASSUMED inter-package link bandwidth")
    s.add_argument("--latency-us", type=float, default=2.0, help="ASSUMED per-hop latency")
    s.add_argument("--global-batch", type=int, default=256)
    s.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    s.add_argument("--arch", default="dev", help=arch_help)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_scale)

    s = sub.add_parser("dp-train", help="data-parallel training over N virtual devices")
    s.add_argument("--devices", type=int, default=4)
    s.add_argument("--steps", type=int, default=100)
    s.add_argument("--global-batch", type=int, default=64)
    s.add_argument("--lr", type=float, default=0.05)
    s.add_argument("--dataset", default="spirals")
    s.add_argument("--mode", default="cycle", choices=["functional", "cycle"])
    s.add_argument("--arch", default="dev", help=arch_help)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_dp_train)

    s = sub.add_parser("rtl-verify", help="RTL vs emulator correspondence (Verilator)")
    s.add_argument("--cases", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rows", type=int, default=8)
    s.add_argument("--cols", type=int, default=8)
    s.add_argument("--data-w", type=int, nargs="+", default=[8, 4])
    s.set_defaults(fn=cmd_rtl_verify)

    s = sub.add_parser("synth", help="Yosys resource estimates for the RTL")
    s.add_argument("--targets", nargs="+", default=["generic", "xc7", "ecp5"])
    s.add_argument("--pnr", action="store_true", help="also place & route on Lattice ECP5 with nextpnr")
    s.add_argument("--out", default="reports/synth")
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("asm", help="assemble a TITAN assembly file")
    s.add_argument("file")
    s.add_argument("--arch", default="dev", help=arch_help)
    s.add_argument("--out", default=None, help="write binary program")
    s.add_argument("--run", action="store_true")
    s.add_argument("--mode", default="functional", choices=["functional", "cycle"])
    s.set_defaults(fn=cmd_asm)

    s = sub.add_parser("disasm", help="disassemble a TITAN binary")
    s.add_argument("file")
    s.set_defaults(fn=cmd_disasm)
    return p


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
