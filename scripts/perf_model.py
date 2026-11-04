#!/usr/bin/env python3
"""Transparent peak-throughput estimator; not a benchmark or silicon prediction."""
import argparse

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--macs", type=int, default=16, help="MAC units")
    p.add_argument("--clock-mhz", type=float, default=100.0)
    p.add_argument("--ops-per-mac", type=int, default=2,
                   help="Convention: multiply + add = 2 operations")
    p.add_argument("--utilization", type=float, default=1.0,
                   help="Assumed utilization in [0,1], not measured")
    p.add_argument("--b300-pflops", type=float, default=5.0,
                   help="Reference dense FP8 peak in PFLOP/s; verify vendor spec")
    a = p.parse_args()
    if a.macs <= 0 or a.clock_mhz <= 0 or a.ops_per_mac <= 0:
        p.error("macs, clock-mhz and ops-per-mac must be positive")
    if not 0.0 <= a.utilization <= 1.0:
        p.error("utilization must be between 0 and 1")
    ops_s = a.macs * a.clock_mhz * 1e6 * a.ops_per_mac * a.utilization
    pflops = ops_s / 1e15
    print("ROUGE TITAN analytical peak model (not measured)")
    print(f"MAC units:                 {a.macs}")
    print(f"Clock assumption:          {a.clock_mhz:g} MHz")
    print(f"Utilization assumption:    {a.utilization:.1%}")
    print(f"Operations per MAC:        {a.ops_per_mac}")
    print(f"Estimated throughput:      {ops_s:,.0f} ops/s ({pflops:.6g} PFLOP/s)")
    print(f"Reference B300:            {a.b300_pflops:g} PFLOP/s")
    print(f"Ratio to reference:        {pflops/a.b300_pflops:.6g}x")
    print("Caution: precision, sparsity, memory, software and workload are not normalized.")

if __name__ == "__main__":
    main()
