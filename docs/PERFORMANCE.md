# Performance methodology

## 1. Kinds of numbers

| Kind | Produced by | Meaning |
|---|---|---|
| verified | functional emulator vs float64 NumPy reference | numerical correctness only |
| simulated | cycle model at the configured (assumed) clock | time the modelled TITAN-X1 would take |
| analytical | roofline (`titan/perf/analytical.py`) | lower bound / estimate, ideal overlap |
| measured (host) | wall clock of the emulator on the host CPU | software speed, **not** accelerator speed |
| RTL / P&R | Verilator, Yosys, nextpnr | bit-exactness; cell counts; post-route STA Fmax estimate |

Every benchmark row records workload, precision, configuration hash, method, duration, host
hardware, software versions and commit SHA (`titan/provenance.py`). Reports: `reports/`.
Regenerate: `python3 scripts/make_reports.py`.

## 2. Cycle model scope

Transaction-level and cycle-approximate: per-core in-order issue, scoreboarded hazards,
unit occupancy, MXU fill/drain and tile quantization, DMA link/HBM/L2/SHARED bandwidth with
contention, L2 hit/miss, queue depth, command-processor dispatch, barrier and semaphore
latency. Not modelled: DRAM page/bank effects, NoC topology and routing, clock-domain
crossings, power/thermal throttling. Agreement with the analytical bound for large GEMMs is
1.01–1.11×; the model has not been validated against hardware (none exists).

## 3. Roofline

Intensity = ops / compulsory global traffic. Ridge point of a configuration =
peak ops / HBM bandwidth (dev BF16: 8 ops/B; target BF16: 721 ops/B). The benchmark report
lists the analytical bound for each workload next to the simulated time ("Sim / est.").

## 4. B300 reference and comparison rules

Source table with URLs, retrieval date (2026-11-04) and confidence note: `titan/perf/reference.py`.
Per GPU, dense: FP4 13.5 PF, FP8 4.5 PF, BF16/FP16 2.25 PF; HBM3e 288 GB, 7.7–8 TB/s (sources disagree).

Enforced in code (`ComparisonError`):
1. Same precision on both sides — INT8 has no sourced B300 figure, so no INT8 ratio is reported.
2. Dense vs dense — TITAN-X1 has no sparsity mode.
3. Peak only vs peak — a simulated workload number is never divided by a vendor peak.

## 5. Target design point vs the 10× goal (analytical only)

| | TITAN-X1 target (assumed) | B300 (vendor-class figure) | Ratio |
|---|---|---|---|
| BF16 dense peak | 23.07 PFLOPS | 2.25 PFLOPS | 10.25× |
| Memory bandwidth | 32 TB/s | 8 TB/s | 4.0× |
| Ridge point (BF16) | 721 ops/B | 281 ops/B | — |

Consequences: compute-bound GEMMs could at most approach the 10× paper ratio if the design
were built and ran at its assumed clock and utilization; memory-bound kernels (element-wise,
attention at short context, small-batch inference) are limited to ~4×. Reaching 10× paper peak
at balanced intensity would need ~80 TB/s of memory bandwidth
(`titan.perf.scale_requirements(22500, BF16, 2.0)` → 344 cores of 128×128, 80 TB/s).
None of this is achieved; it is the quantitative statement of the research problem.

## 6. Findings produced by the models so far

* Command-processor head-of-line blocking: emitting all of core 0's instructions first left the
  other cores idle (GEMM 1024×1024×64 BF16: 35.6 % MXU). Interleaving per-core streams
  → 97.9 % MXU, 2.75× fewer cycles (commit `f65c032`).
* Shared-resource FIFO artefact in an early cycle model version (future reservations blocking
  earlier requests) replaced by busy calendars (commit `24c737f`).
* Data-parallel scaling saturates: with assumed 900 GB/s links, an MLP step reaches 3.4× on 8
  and 2.9× on 16 devices because all-reduce volume stays constant while compute shrinks.
* INT4 multipliers are not DSP-mapped on Xilinx 7-series by Yosys (5790 LUT vs 2048 for INT8).

## 7. Large-model planner (Mode C)

`titan plan` / `titan.scale.plan` evaluates every combination of tensor (TP), pipeline (PP) and data
parallelism (DP), micro-batch size, ZeRO stage and activation recomputation for a transformer and a
device count, keeps the plans whose per-device memory fits (90 % of capacity) and returns the
fastest. Accounting: mixed-precision Adam 16 bytes/parameter; activations per layer
t·d·(34 + 5·a·T/d)/TP bytes (Korthikanti et al., 2022); GEMMs on a roofline with an assumed
efficiency of 0.8; attention and element-wise traffic on HBM; TP ring all-reduces, PP point-to-point
plus bubble (p−1)/(m+p−1); DP gradient all-reduce without overlap. Parameter and FLOP formulas are
tested for exact equality with compiled TITAN graphs.

| GPT-100B (101.7 B params), global batch 1024 × 2048 tokens | 64 × titan-x1-target | 64 × B300 |
|---|---|---|
| best plan | TP1 PP8 DP8 ZeRO-1 | TP2 DP32 ZeRO-3 (per planner) |
| step time (estimate) | 2.18 s | 15.3 s |
| tokens/s | 0.96 M | 0.14 M |
| MFU | 41 % | 60 % |

Ratio 7.0× under identical assumptions, below the 10.25× BF16 paper-peak ratio because the target
design point has 4× (not 10×) the memory bandwidth. Full table 1.3B-175B:
`reports/SPEED_AND_LARGE_MODELS.md`. The planner is not validated against real distributed runs.

## 8. Single-machine 100B (exact, from the compiled program)

`titan plan --params 100b --single-machine` builds and compiles the complete GPT-100B training step
(forward, loss, backward, global-norm clipping, AdamW) to TITAN IR: 101.70 B parameters, 433,137
kernels, 78,978 lazily initialized tensors, 34.7 s and 1.5 GB of host memory to compile. Exact
accounting of its kernel schedule with the FP32 executor: 1.63 TB in RAM without offload; with
storage offload (tensors ≥ 4 MiB) **12.5 GB RAM + 1.63 TB storage**. A single server with a
multi-terabyte NVMe can therefore run the 100B training step; it is storage-bandwidth bound
(each step streams the full optimizer state), which is why many devices remain the route to speed.
