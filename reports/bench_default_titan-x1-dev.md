# TITAN-X1 benchmark report — suite `default`

All numbers are reproducible with `titan bench --suite default --arch titan-x1-dev --seed 0`.

| Provenance | |
|---|---|
| commit | `d9ed8afce2dc14c8757af572cdedeb875ab9e456-dirty` |
| software | titan 0.1.0, Python 3.13.18, NumPy 2.5.3 |
| host hardware (for *measured* rows) | Intel(R) Xeon(R) Processor @ 2.10GHz (4 logical CPUs) |
| architecture config | `titan-x1-dev` hash `ae5c9f6222b89391`, 4 cores, 16x16 MXU, 1.0 GHz (assumed), HBM 256.0 GB/s |
| timestamp | 2026-11-04T16:33:57+00:00 |
| suite duration | 1.294 s (host) |

## Method legend

* **verified** — verified: functional emulation vs float64 NumPy reference
* **simulated** — simulated: TITAN-X1 cycle model (transaction-level, cycle-approximate) at the configured clock
* **analytical** — analytical estimate (roofline; ideal overlap; not a measurement)
* **measured** — measured: host wall clock of the software emulator on the host CPU (not accelerator speed)

No row is a hardware measurement of TITAN silicon or FPGA: neither exists.

## Results

| Workload | Prec | Verified (max rel err) | Simulated cycles | Simulated TOPS | % of config peak | MXU util | HBM GB/s (sim) | L2 hit | Sync overhead | Analytical bound | Analytical est. (µs) | Sim / est. | Host wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GEMM 128x128x128 | FP32 | PASS (3.4e-07) | 9112 | 0.460 | 89.9% | 89.9% | 21.6 | 60% | 0.0% | compute | 8.19 | 1.11x | 0.001 |
| GEMM 256x256x256 | FP32 | PASS (6.0e-07) | 67224 | 0.499 | 97.5% | 97.5% | 11.7 | 75% | 0.0% | compute | 65.54 | 1.03x | 0.004 |
| GEMM 256x256x256 | BF16 | PASS (3.0e-07) | 17816 | 1.883 | 92.0% | 92.0% | 29.4 | 75% | 0.0% | compute | 16.38 | 1.09x | 0.007 |
| GEMM 256x256x256 | INT8 | PASS (0.0e+00) | 9112 | 3.682 | 89.9% | 89.9% | 43.2 | 75% | 0.0% | compute | 8.19 | 1.11x | 0.028 |
| GEMM 512x512x512 | BF16 | PASS (2.7e-07) | 132760 | 2.022 | 98.7% | 98.7% | 15.8 | 88% | 0.0% | compute | 131.07 | 1.01x | 0.053 |
| GEMM 512x64x512 | BF16 | PASS (2.2e-07) | 18072 | 1.857 | 90.7% | 90.7% | 39.9 | 25% | 0.0% | compute | 16.38 | 1.10x | 0.005 |
| GEMM 1024x1024x64 | BF16 | PASS (1.4e-07) | 66968 | 2.004 | 97.9% | 97.9% | 66.5 | 90% | 0.0% | compute | 65.54 | 1.02x | 0.038 |
| element-wise add 1024x1024 | FP32 | PASS (6.5e-08) | 49380 | 0.021 | 4.1% | 0.0% | 254.8 | 0% | 0.0% | global_memory | 49.15 | 1.00x | 0.035 |
| MLP training step (16-128-128-8, batch 64, fwd+bwd+SGD) | FP32 | PASS (8.4e-08) | 24560 | 0.300 | 58.7% | 59.4% | 29.6 | 79% | 9.7% | mxu_critical_core | 16.13 | 1.52x | 0.018 |
| transformer block forward (B4 T32 d64 h4) | FP32 | PASS (2.4e-07) | 40134 | 0.349 | 68.2% | 66.3% | 32.3 | 75% | 3.9% | mxu_critical_core | 26.62 | 1.51x | 0.031 |

*Sim / est.* = simulated time divided by the analytical lower bound (1.0 = roofline-limited).
*MXU util* = busy cycles of all matrix engines / (cores x simulated cycles).

