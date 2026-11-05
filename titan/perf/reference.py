"""External reference figures (NVIDIA B300) and the rules for comparing against them.

Rules enforced in code:
  * same precision on both sides (no INT8-vs-FP8, no BF16-vs-FP4),
  * same sparsity assumption (dense vs dense),
  * peak may only be compared with peak; a simulated/measured TITAN workload number is
    never divided by a vendor peak and called a speed-up.
"""
from __future__ import annotations

from ..dtypes import DType

B300 = {
    "device": "NVIDIA B300 (Blackwell Ultra), one GPU of an HGX B300 system",
    "dense_tflops": {"FP4": 13500.0, "FP8": 4500.0, "FP16": 2250.0, "BF16": 2250.0},
    "sparse_tflops": {"FP4": 18000.0, "FP8": 9000.0, "FP16": 4500.0, "BF16": 4500.0},
    "hbm_capacity_gb": 288.0,
    "hbm_capacity_note": "288 GB physical HBM3e; ~270 GB usable is reported for the HGX datasheet",
    "hbm_bandwidth_tbps": [7.7, 8.0],
    "hbm_bandwidth_note": "sources disagree (7.7 vs 8 TB/s); the upper value is used for ratios (conservative "
                          "for TITAN)",
    "not_listed": ["INT8", "INT4", "FP32", "FP64"],
    "sources": [
        "https://inferencex.semianalysis.com/chips/b300",
        "https://nebius.com/b300",
        "https://flopper.io/system/nvidia-hgx-b300-8-gpu/spec-sheet",
        "https://verda.com/blog/nvidia-b300-vs-b200-complete-gpu-comparison-to-date",
        "https://openmetal.io/resources/hardware-details/nvidia-hgx-b300-8-gpu-server/",
        "https://www.nvidia.com/en-us/data-center/hgx/",
    ],
    "retrieved": "2026-11-04",
    "confidence": "third-party aggregation of vendor figures; per-GPU dense values derived from 8-GPU HGX totals. "
                  "Verify against the official NVIDIA B300 datasheet before external use.",
}


class ComparisonError(ValueError):
    pass


def b300_peak_tflops(dt: DType | str, sparse: bool = False) -> float:
    key = dt.name if isinstance(dt, DType) else str(dt).upper()
    table = B300["sparse_tflops" if sparse else "dense_tflops"]
    if key not in table:
        raise ComparisonError(f"no sourced B300 {('sparse' if sparse else 'dense')} figure for {key}; "
                              f"comparison refused")
    return table[key]


def compare_peak(cfg, dt: DType, sparse: bool = False) -> dict:
    """Paper-peak vs paper-peak at identical precision and sparsity."""
    if sparse:
        raise ComparisonError("TITAN-X1 has no structured-sparsity mode; a sparse comparison would be invalid")
    if dt.name not in B300["dense_tflops"]:
        raise ComparisonError(f"no like-for-like B300 figure for {dt.name}")
    titan = cfg.peak_ops(dt) / 1e12
    ref = b300_peak_tflops(dt)
    return {"precision": dt.name, "sparsity": "dense", "titan_config": cfg.name,
            "titan_peak_tflops": titan, "b300_peak_tflops": ref, "ratio": titan / ref,
            "titan_hbm_tbps": cfg.hbm_gbps / 1e3, "b300_hbm_tbps": max(B300["hbm_bandwidth_tbps"]),
            "bandwidth_ratio": (cfg.hbm_gbps / 1e3) / max(B300["hbm_bandwidth_tbps"]),
            "label": "paper peak vs paper peak; TITAN value is an analytical estimate of an unbuilt configuration"}


def compare(titan_value: float, titan_kind: str, dt: DType, ref_kind: str = "peak") -> float:
    """Guarded generic ratio. Only like-for-like kinds are allowed."""
    if titan_kind != ref_kind:
        raise ComparisonError(f"cannot compare a TITAN '{titan_kind}' figure with a B300 '{ref_kind}' figure")
    return titan_value / (b300_peak_tflops(dt) * 1e12)
