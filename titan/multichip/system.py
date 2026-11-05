"""Multi-chip system model: chiplets -> packages -> system, links, collectives (functional + cost).

All link figures are ASSUMPTIONS of this model (documented per preset); they are
not properties of existing hardware. Collectives are executed functionally on real
data (chunked ring algorithm) and costed with an alpha-beta link model.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Level:
    name: str             # e.g. "d2d" (chiplets inside a package) or "inter-package"
    size: int             # participants at this level
    topology: str         # ring | fully_connected | mesh2d | switch
    link_gbps: float      # per-link, per-direction bandwidth
    latency_us: float     # per-hop latency (alpha)


@dataclass(frozen=True)
class SystemConfig:
    name: str
    levels: tuple[Level, ...]
    note: str = ""

    @property
    def devices(self) -> int:
        return math.prod(lv.size for lv in self.levels)

    def describe(self) -> dict:
        return {"name": self.name, "devices": self.devices, "levels": [asdict(lv) for lv in self.levels],
                "note": self.note}


def system_preset(devices: int, topology: str = "ring", link_gbps: float = 900.0, latency_us: float = 2.0,
                  chiplets_per_package: int = 0, d2d_gbps: float = 2000.0, d2d_latency_us: float = 0.5) -> SystemConfig:
    """Flat system of `devices` packages, or hierarchical (chiplets_per_package > 1)."""
    note = ("ASSUMED link parameters for modelling: inter-package %.0f GB/s/link, %.1f us/hop; "
            "d2d %.0f GB/s, %.1f us" % (link_gbps, latency_us, d2d_gbps, d2d_latency_us))
    if chiplets_per_package > 1:
        if devices % chiplets_per_package:
            raise ValueError("devices must be a multiple of chiplets_per_package")
        lv = (Level("d2d", chiplets_per_package, "ring", d2d_gbps, d2d_latency_us),
              Level("inter-package", devices // chiplets_per_package, topology, link_gbps, latency_us))
        return SystemConfig(f"{devices // chiplets_per_package}x{chiplets_per_package}-hier-{topology}", lv, note)
    return SystemConfig(f"{devices}-{topology}", (Level("inter-package", devices, topology, link_gbps,
                                                         latency_us),), note)


# ------------------------------------------------------------------ cost model (alpha-beta)

def _allreduce_time(lv: Level, nbytes: float) -> float:
    n = lv.size
    if n == 1:
        return 0.0
    bw = lv.link_gbps * 1e9
    a = lv.latency_us * 1e-6
    if lv.topology in ("ring", "switch"):
        return 2 * (n - 1) * a + 2 * (n - 1) / n * nbytes / bw
    if lv.topology == "fully_connected":       # direct reduce-scatter + all-gather over n-1 links in parallel
        return 2 * a + 2 * (n - 1) / n * nbytes / (bw * (n - 1))
    if lv.topology == "mesh2d":                 # two ring phases over sqrt(n) rows/cols
        s = int(round(math.sqrt(n)))
        if s * s != n:
            raise ValueError("mesh2d needs a square number of devices")
        return 2 * (2 * (s - 1) * a + 2 * (s - 1) / s * nbytes / bw)
    raise ValueError(lv.topology)


def allreduce_cost(sys_: SystemConfig, nbytes: float) -> dict:
    """Hierarchical all-reduce: reduce-scatter inside lower levels, all-reduce on top, all-gather back."""
    t, per = 0.0, {}
    payload = float(nbytes)
    for lv in sys_.levels:
        tl = _allreduce_time(lv, payload)
        per[lv.name] = tl
        t += tl
        payload /= lv.size                       # next level reduces only this level's shard
    return {"seconds": t, "per_level_s": per, "bytes": nbytes,
            "label": "analytical estimate (alpha-beta link model, assumed link parameters)"}


# ------------------------------------------------------------------ functional collectives

def ring_allreduce(buffers: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    """Chunked ring all-reduce (reduce-scatter + all-gather) on real data.

    Returns the reduced buffers (identical on every rank) and traffic statistics, which
    are checked against the cost model's byte count: 2 (n-1)/n * bytes per rank.
    """
    n = len(buffers)
    flat = [np.asarray(b, dtype=np.float64).ravel().copy() for b in buffers]
    if n == 1:
        return [flat[0].reshape(buffers[0].shape)], {"steps": 0, "bytes_sent_per_rank": 0}
    chunks = [np.array_split(f, n) for f in flat]
    sent = [0] * n
    steps = 0
    for s in range(n - 1):                       # reduce-scatter
        msgs = [(r, (r - s) % n, chunks[r][(r - s) % n].copy()) for r in range(n)]
        for r, idx, data in msgs:
            dst = (r + 1) % n
            chunks[dst][idx] = chunks[dst][idx] + data
            sent[r] += data.nbytes
        steps += 1
    for s in range(n - 1):                       # all-gather
        msgs = [(r, (r + 1 - s) % n, chunks[r][(r + 1 - s) % n].copy()) for r in range(n)]
        for r, idx, data in msgs:
            chunks[(r + 1) % n][idx] = data
            sent[r] += data.nbytes
        steps += 1
    out = [np.concatenate(c).reshape(buffers[0].shape) for c in chunks]
    return out, {"steps": steps, "bytes_sent_per_rank": max(sent), "total_bytes": flat[0].nbytes}
