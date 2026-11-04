"""TITAN-X1 architecture configuration.

Every architectural parameter that influences functional behaviour (capacities)
or the cycle / analytical models (rates, bandwidths, latencies) lives here.
Configs are loaded from TOML files in ``configs/`` and hashed for provenance.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .dtypes import DType

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass(frozen=True)
class ArchConfig:
    name: str = "titan-x1-dev"
    description: str = ""
    clock_ghz: float = 1.0
    # ---- compute fabric (per core = one tensor processing cluster)
    cores: int = 4
    mxu_rows: int = 16
    mxu_cols: int = 16
    # MACs per processing element per cycle, by matrix-engine input type
    mxu_rate_int4: float = 4.0
    mxu_rate_int8: float = 2.0
    mxu_rate_fp16: float = 1.0
    mxu_rate_bf16: float = 1.0
    mxu_rate_fp32: float = 0.25
    mxu_latency: int = 8            # extra pipeline cycles beyond array fill/drain
    vpu_lanes: int = 64             # elements / cycle for vector, reduce, convert
    vpu_latency: int = 4
    sfu_lanes: int = 16             # elements / cycle for transcendental / normalization
    sfu_latency: int = 12
    # ---- memory hierarchy
    local_sram_kib: int = 512       # per core scratchpad
    gemm_staging_kib: int = 128     # top of local SRAM reserved for the GEMM macro sequencer
    local_gbps: float = 512.0
    local_latency: int = 4
    shared_sram_kib: int = 4096     # on-chip SRAM shared by all cores
    shared_gbps: float = 512.0
    shared_latency: int = 24
    l2_kib: int = 1024              # cache in front of global memory
    l2_line: int = 128
    l2_ways: int = 8
    l2_gbps: float = 1024.0
    l2_latency: int = 40
    hbm_gbps: float = 256.0         # global memory (HBM abstraction) aggregate bandwidth
    hbm_latency: int = 200
    global_mem_gib: float = 16.0    # architectural capacity (analytical models)
    emulated_global_mib: int = 256  # backing store actually allocated by the emulator
    noc_link_gbps: float = 128.0    # per-core DMA link into the network-on-chip
    dma_engines: int = 2            # independent DMA engines per core
    # ---- command processor / control
    queue_depth: int = 64
    cp_dispatch_cycles: int = 1
    semaphores: int = 64
    barrier_latency: int = 16
    kernel_max_depth: int = 4
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    # ------------------------------------------------------------ derived
    @property
    def local_bytes(self) -> int:
        return self.local_sram_kib * 1024

    @property
    def staging_base(self) -> int:
        return (self.local_sram_kib - self.gemm_staging_kib) * 1024

    @property
    def shared_bytes(self) -> int:
        return self.shared_sram_kib * 1024

    @property
    def global_bytes(self) -> int:
        return self.emulated_global_mib * 1024 * 1024

    def bytes_per_cycle(self, gbps: float) -> float:
        return gbps / self.clock_ghz

    def mxu_rate(self, dt: DType) -> float:
        rate = {DType.INT4: self.mxu_rate_int4, DType.INT8: self.mxu_rate_int8,
                DType.FP16: self.mxu_rate_fp16, DType.BF16: self.mxu_rate_bf16,
                DType.FP32: self.mxu_rate_fp32}.get(dt, 0.0)
        return rate

    def peak_ops(self, dt: DType) -> float:
        """Analytical dense peak (ops/s, 1 MAC = 2 ops) of the configuration. Not a measurement."""
        return self.cores * self.mxu_rows * self.mxu_cols * self.mxu_rate(dt) * 2 * self.clock_ghz * 1e9

    def ridge_point(self, dt: DType) -> float:
        """Arithmetic intensity (ops/byte of global traffic) where compute and HBM roofs meet."""
        return self.peak_ops(dt) / (self.hbm_gbps * 1e9)

    # ------------------------------------------------------------ io
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("meta", None)
        return d

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    def replace(self, **kw) -> "ArchConfig":
        return dataclasses.replace(self, **kw)

    def validate(self) -> "ArchConfig":
        pos = ["clock_ghz", "cores", "mxu_rows", "mxu_cols", "vpu_lanes", "sfu_lanes", "local_sram_kib",
               "shared_sram_kib", "hbm_gbps", "noc_link_gbps", "dma_engines", "queue_depth", "semaphores",
               "l2_line", "l2_ways"]
        for k in pos:
            if getattr(self, k) <= 0:
                raise ValueError(f"config {self.name}: {k} must be > 0")
        if self.cores > 65535:
            raise ValueError("cores must be <= 65535 (16-bit core field in the ISA)")
        if not 0 <= self.gemm_staging_kib < self.local_sram_kib:
            raise ValueError("gemm_staging_kib must be smaller than local_sram_kib")
        if self.l2_kib * 1024 % (self.l2_line * self.l2_ways):
            raise ValueError("l2 size must be a multiple of line*ways")
        if not math.log2(self.l2_line).is_integer():
            raise ValueError("l2_line must be a power of two")
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "ArchConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d).validate()

    @classmethod
    def from_toml(cls, path: str | Path) -> "ArchConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        flat = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                flat.update(v)
            else:
                flat[k] = v
        return cls.from_dict(flat)


def load_config(name_or_path: str | Path | ArchConfig | None = None) -> ArchConfig:
    """Load a preset by name (``tiny``, ``dev``, ``target``) or a TOML path."""
    if isinstance(name_or_path, ArchConfig):
        return name_or_path
    if name_or_path is None:
        name_or_path = "dev"
    p = Path(str(name_or_path))
    if p.suffix != ".toml":
        short = str(name_or_path).removeprefix("titan-x1-")
        p = CONFIG_DIR / f"titan-x1-{short}.toml"
    if not p.exists():
        raise FileNotFoundError(f"no TITAN config {name_or_path!r} ({p})")
    return ArchConfig.from_toml(p)


def list_presets() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("titan-x1-*.toml"))
