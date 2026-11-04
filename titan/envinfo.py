"""Inspect the environment TITAN runs in and derive the limits it implies (`titan env`)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .provenance import host_info

BYTES_PER_PARAM_TRAIN = 24          # FP32 weights + grads + Adam m, v + update temporaries (measured)


def gpu_info() -> list[dict]:
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        out.append({"index": i, "name": p.name, "memory_gb": total / 1e9, "free_gb": free / 1e9,
                    "bf16": bool(torch.cuda.is_bf16_supported())})
    return out


def environment(path: str | os.PathLike = ".") -> dict:
    from .scale import host_memory_available
    info = host_info()
    avail = host_memory_available()
    total = 0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                total = int(line.split()[1]) * 1024
    except OSError:
        pass
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    du = shutil.disk_usage(p)
    try:
        import torch
        torch_v = torch.__version__
        threads = torch.get_num_threads()
    except Exception:
        torch_v, threads = None, None
    gpus = gpu_info()
    ram_params = int(0.7 * avail / BYTES_PER_PARAM_TRAIN) if avail else 0
    gpu_params = int(0.8 * max((g["free_gb"] for g in gpus), default=0) * 1e9 / BYTES_PER_PARAM_TRAIN)
    backend = "accel (CUDA)" if gpus else "accel (CPU)" if torch_v else "fast"
    return {
        "host": info, "ram_total_gb": total / 1e9, "ram_available_gb": avail / 1e9,
        "storage_path": str(p.resolve()), "storage_free_gb": du.free / 1e9, "storage_total_gb": du.total / 1e9,
        "torch": torch_v, "torch_threads": threads, "gpus": gpus, "recommended_backend": backend,
        "max_trainable_params_host_ram": ram_params, "max_trainable_params_single_gpu": gpu_params,
        "max_corpus_tokens_on_storage": int(du.free * 0.9),
        "note": "Limits are estimates for single-machine numeric training (FP32 Adam, 24 bytes/parameter "
                "plus activations). Larger models: `titan plan` (multi-device TITAN plan).",
    }
