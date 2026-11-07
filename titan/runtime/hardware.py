"""Discover and measure the compute connected to this machine, and configure TITAN for it.

    titan hardware [--json out.json]

Finds every device PyTorch can drive (CUDA GPUs incl. ROCm builds, Apple MPS, Intel XPU, the CPU with
its matrix extensions), MEASURES each one (dense GEMM in FP32 and BF16, memory bandwidth) and derives
the fastest configuration: which devices train (all accelerators of the fastest kind, data parallel),
precision (BF16 only where the exact reduced-precision GEMM path exists and is measured faster),
compilation, and the expected step speed-up. Numbers are measurements of this machine, labelled so.
"""
from __future__ import annotations

import os
import platform
import time
from pathlib import Path

CPU_FLAGS = ("avx2", "avx512f", "avx512_bf16", "amx_bf16", "amx_int8", "amx_tile", "neon", "asimd", "sve")


def cpu_info() -> dict:
    info = {"model": platform.processor() or platform.machine(), "logical_cores": os.cpu_count() or 1,
            "flags": []}
    try:
        txt = Path("/proc/cpuinfo").read_text()
        for line in txt.splitlines():
            if line.startswith("model name"):
                info["model"] = line.split(":", 1)[1].strip()
            if line.startswith(("flags", "Features")):
                have = set(line.split(":", 1)[1].split())
                info["flags"] = [f for f in CPU_FLAGS if f in have]
                break
    except OSError:
        pass
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return info


def devices() -> list[dict]:
    """Every compute device PyTorch can use here (CPU always)."""
    out = [{"device": "cpu", "kind": "cpu", "name": cpu_info()["model"]}]
    try:
        import torch
    except ImportError:
        return out
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            out.append({"device": f"cuda:{i}", "kind": "cuda", "name": p.name, "memory_gb": round(p.total_memory / 1e9, 1),
                        "capability": f"{p.major}.{p.minor}", "multiprocessors": p.multi_processor_count})
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            out.append({"device": f"xpu:{i}", "kind": "xpu", "name": torch.xpu.get_device_name(i)})
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        out.append({"device": "mps", "kind": "mps", "name": "Apple GPU (Metal)"})
    return out


def _sync(torch, dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize(dev)
    elif dev.startswith("xpu"):
        torch.xpu.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def measure(dev: str, n: int = 1024, seconds: float = 0.4) -> dict:
    """Dense GEMM TFLOP/s (FP32; BF16 operands with FP32 accumulation where exact) and copy bandwidth."""
    import torch
    from .accel import AccelDevice
    from ..dtypes import DType
    res = {}

    def rate(fn, flops):
        fn()
        _sync(torch, dev)
        k, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            fn()
            k += 1
        _sync(torch, dev)
        return flops * k / (time.perf_counter() - t0)

    a = torch.randn(n, n, device=dev)
    b = torch.randn(n, n, device=dev)
    res["fp32_tflops"] = round(rate(lambda: a @ b, 2 * n ** 3) / 1e12, 3)
    ad = AccelDevice("dev", device=dev)
    if DType.BF16 in ad.exact_reduced:
        ab, bb = a.bfloat16().float(), b.bfloat16().float()
        with ad._reduced_matmul():
            res["bf16_exact_tflops"] = round(rate(lambda: ab @ bb, 2 * n ** 3) / 1e12, 3)
    x = torch.empty(256 * 2 ** 20 // 4, device=dev)                  # 256 MiB: beyond the caches
    y = torch.empty_like(x)
    res["copy_gb_s"] = round(rate(lambda: y.copy_(x), 2 * x.numel() * 4) / 1e9, 1)
    return res


def recommend(devs: list[dict], perf: dict) -> dict:
    """Fastest configuration from the measurements."""
    accel = [d for d in devs if d["kind"] != "cpu"]
    kind = max({d["kind"] for d in accel}, key=lambda k: sum(perf.get(d["device"], {}).get("fp32_tflops", 0)
                                                         for d in accel if d["kind"] == k)) if accel else "cpu"
    use = [d["device"] for d in devs if d["kind"] == kind]
    p0 = perf.get(use[0], {})
    bf16 = p0.get("bf16_exact_tflops", 0) > 1.2 * p0.get("fp32_tflops", 0)
    total = sum(perf.get(u, {}).get("fp32_tflops", 0) for u in use)
    cpu = perf.get("cpu", {}).get("fp32_tflops", 0) or 1e-9
    return {"devices": use, "data_parallel": len(use) > 1, "precision": "bf16" if bf16 else "fp32",
            "compile": "auto (on for runs > 10 min)", "backend": "accel",
            "gemm_tflops_total_fp32": round(total, 3),
            "gemm_tflops_total_best": round(sum(max(perf.get(u, {}).get("fp32_tflops", 0),
                                                    perf.get(u, {}).get("bf16_exact_tflops", 0)) for u in use), 3),
            "vs_this_cpu_fp32": round(total / cpu, 1)}


def train_benchmark(steps: int = 30, d_model: int = 256, batch: int = 0, log=print) -> dict:
    """Measured training throughput on THIS machine: one device vs every connected accelerator
    (data parallel), plus the precision/compile defaults. Same model, same steps, tokens per second."""
    import tempfile
    import time as _t
    from ..train import TrainConfig, Trainer
    from .multidevice import parse_accelerators
    accs = parse_accelerators("auto", "accel")
    n = max(1, len(accs))
    batch = batch or 8 * n
    ov = {"d_model": d_model, "heads": max(1, d_model // 64), "d_ff": 4 * d_model, "layers": 4, "seq": 64}
    rows = {}
    for label, acc, prec in (("1 device, fp32", "1", "fp32"), ("1 device, precision auto", "1", "auto"),
                             (f"{n} devices, precision auto", "auto", "auto")):
        if label.startswith(f"{n} devices") and n == 1:
            continue
        with tempfile.TemporaryDirectory() as d:
            tr = Trainer(TrainConfig(model="transformer", profile="medium", backend="accel", precision=prec,
                                     accelerators=acc, torch_compile=False, steps=steps + 3, batch=batch,
                                     eval_every=0, ckpt_every=0, overrides=dict(ov), out_dir=d), log=None)
            tr.run(3)
            t0 = _t.perf_counter()
            tr.run(steps + 3)
            dt = (_t.perf_counter() - t0) / steps
            tr.close()
        rows[label] = {"ms_per_step": round(dt * 1e3, 1), "tokens_per_s": round(batch * 64 / dt),
                       "precision": tr.tc.precision, "devices": len(tr.dp.devs) if tr.dp else 1}
        log(f"  {label:28s} {rows[label]['ms_per_step']:8.1f} ms/step  {rows[label]['tokens_per_s']:>9,} tokens/s")
    base = rows["1 device, fp32"]["tokens_per_s"]
    return {"model": {**ov, "batch": batch}, "runs": rows,
            "speedup_vs_1_device_fp32": {k: round(v["tokens_per_s"] / base, 2) for k, v in rows.items()},
            "method": "measured training steps on this machine (eager; --compile auto adds more on long runs)"}


def profile(seconds: float = 0.4) -> dict:
    devs = devices()
    perf = {}
    for d in devs:
        try:
            perf[d["device"]] = measure(d["device"], seconds=seconds)
        except Exception as e:  # pragma: no cover - device-specific failures are reported, not fatal
            perf[d["device"]] = {"error": f"{type(e).__name__}: {e}"}
    return {"cpu": cpu_info(), "devices": devs, "measured": perf, "recommended": recommend(devs, perf),
            "method": "measured on this machine (dense 1024^3 GEMM, 256 MiB copy); not a vendor peak"}
