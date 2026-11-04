#!/usr/bin/env python3
"""Transparent peak-throughput estimator; not a benchmark or silicon prediction.

Kept for backward compatibility with the D0 workflow. Since M9 it refuses
cross-precision ratios (the D0 version divided an INT8 estimate by a B300 FP8
peak, which is not a valid comparison). Full tooling: `python -m titan info`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from titan.perf.reference import B300  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--macs", type=int, default=16, help="MAC units")
    p.add_argument("--clock-mhz", type=float, default=100.0)
    p.add_argument("--ops-per-mac", type=int, default=2, help="Convention: multiply + add = 2 operations")
    p.add_argument("--utilization", type=float, default=1.0, help="Assumed utilization in [0,1], not measured")
    p.add_argument("--precision", default="INT8", help="precision of the MAC units (INT8, FP16, BF16, ...)")
    a = p.parse_args()
    if a.macs <= 0 or a.clock_mhz <= 0 or a.ops_per_mac <= 0:
        p.error("macs, clock-mhz and ops-per-mac must be positive")
    if not 0.0 <= a.utilization <= 1.0:
        p.error("utilization must be between 0 and 1")
    ops_s = a.macs * a.clock_mhz * 1e6 * a.ops_per_mac * a.utilization
    prec = a.precision.upper()
    print("ROUGE TITAN analytical peak model (not measured)")
    print(f"MAC units:                 {a.macs} ({prec})")
    print(f"Clock assumption:          {a.clock_mhz:g} MHz")
    print(f"Utilization assumption:    {a.utilization:.1%}")
    print(f"Estimated throughput:      {ops_s:,.0f} ops/s ({ops_s / 1e15:.6g} P-ops/s)")
    ref = B300["dense_tflops"].get(prec)
    if ref is None:
        print(f"B300 comparison:           refused - no sourced dense B300 figure for {prec} "
              f"(available: {', '.join(B300['dense_tflops'])})")
    else:
        print(f"B300 dense {prec} reference: {ref:g} TFLOPS ({B300['retrieved']}, see titan/perf/reference.py)")
        print(f"Ratio (peak vs peak):      {ops_s / (ref * 1e12):.6g}x")
    print("Caution: memory, software and workload are not modelled here; use `python -m titan bench`.")


if __name__ == "__main__":
    main()
