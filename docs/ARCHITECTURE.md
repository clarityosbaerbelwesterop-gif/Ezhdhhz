# TITAN-X1 architecture

Executable definition: `titan/config.py` + `configs/*.toml`, `titan/emu/`, `titan/isa/`.
The D0 MAC tile (`rtl/rtl_mac_array.sv`) remains the reference for integer MAC numerics.

## 1. Device organisation

```
                 +--------------------------- TITAN-X1 device ----------------------------+
 host  --cmd-->  | Command processor: stream decode, kernel inlining, GEMM sequencer,     |
                 | barrier fan-out, per-core queues (depth Q), completion records, faults  |
                 |   |            |             |             |                          |
                 | core 0       core 1   ...  core N-1   (tensor processing clusters)      |
                 |  CTRL  DMA×E  MXU R×C  VPU (lanes)  SFU (lanes)  LOCAL SRAM            |
                 |   \______________ network-on-chip (per-core link) _____________/      |
                 |            SHARED SRAM            L2 cache (set-assoc., LRU)           |
                 |                                    |                                   |
                 |                       global memory (HBM abstraction)                  |
                 +------------------------------------------------------------------------+
```

| Parameter | tiny (tests) | dev (default) | target (research design point, assumptions) |
|---|---|---|---|
| cores | 2 | 4 | 352 |
| MXU per core | 4×4 | 16×16 | 128×128 |
| MACs/PE/cycle INT4 / INT8 / FP16=BF16 / FP32 | 4 / 2 / 1 / 0.25 | same | same |
| clock (assumed) | 1 GHz | 1 GHz | 2 GHz |
| LOCAL SRAM / core (GEMM staging) | 64 KiB (16) | 512 KiB (128) | 8 MiB (1 MiB) |
| SHARED SRAM / L2 | 256 KiB / 64 KiB | 4 MiB / 1 MiB | 256 MiB / 256 MiB |
| global memory bandwidth | 64 GB/s | 256 GB/s | 32 TB/s |
| BF16 dense paper peak | 0.064 TFLOPS | 2.05 TFLOPS | 23 068 TFLOPS |

## 2. Compute fabric

* **MXU** — systolic-style matrix engine, output-stationary model: an M×N×K product costs
  ⌈M/R⌉·⌈N/C⌉·⌈K/rate⌉ cycles plus R+C+latency fill/drain. Integer inputs: exact products,
  INT32 two's-complement accumulation. Float inputs: FP32 accumulation. RTL: `rtl/titan_mxu_tile.sv`
  (outer-product formulation, verified bit-exact against the emulator).
* **VPU** — element-wise arithmetic, reductions, accumulate, conversion (INT32/FP16/BF16/FP32).
* **SFU** — activations (ReLU, GELU, sigmoid, tanh, exp, log, sqrt, rsqrt, reciprocal, step,
  GELU′, square) and row normalizations (softmax, layernorm, RMSnorm).
* **Data conversion** — `CONVERT` with scale, round-half-even and saturation (quantization).
* Number and size of cores, arrays and lanes are configuration parameters.

## 3. Memory hierarchy

* Address spaces GLOBAL / SHARED / LOCAL (byte addressed, 2-D strided tensor operands).
* L2 cache in front of GLOBAL (set-associative LRU, write-through/no-allocate) — timing only.
* Bandwidth resources (HBM, L2, SHARED, per-core NoC link, LOCAL) are modelled as busy
  calendars; requests take the earliest idle gap (contention without FIFO artefacts).
* DMA engines per core; double buffering is a compiler strategy validated by the cycle model.
* Capacity limits are enforced by ISA validation (OUT_OF_BOUNDS).

## 4. Command processor and execution

The command processor decodes one stream, inlines kernels (`EXECUTE_KERNEL`, with global
relocation), expands `GEMM` into micro-ops in the staging region, fans out barriers and
dispatches into bounded per-core queues. Each core issues in order; a scoreboard tracks
RAW/WAR/WAW hazards on byte ranges; independent work overlaps across units. Cross-core
ordering only through `SYNCHRONIZE` (counting semaphores with release/acquire semantics)
and `BARRIER`. Faults are precise; `WAIT` without a matching `SIGNAL` is a detected deadlock.

## 5. Execution modes (never merged into one number)

| Mode | What it produces | Where |
|---|---|---|
| A functional | exact architectural results, instruction/op/byte counts | `titan/emu/engine.py` |
| B cycle model | simulated cycles, unit utilization, stalls, queue delays, HBM/L2 statistics (transaction-level, cycle-approximate) | `titan/emu/cycle.py` |
| C analytical | roofline estimates, bounds, scaling requirements | `titan/perf/analytical.py` |
| D RTL | bit-exact MXU results from Verilator; hybrid execution of INT8/INT4 MATMUL | `titan/rtl/cosim.py` |

Execution paths for Mode A: the **reference engine** (`titan/emu/engine.py`), the **replay
executor** (`titan/emu/replay.py`; same instructions and order, pre-bound to memory views,
bit-identical memory images, ~12-21× faster) and the **fast IR executor** (`titan/runtime/fast.py`;
the same compiled kernels on whole tensors, no ISA program, FP32 summation order differs). Mode B
reuses the simulated timing of a program because the cycle model is data-independent.

## 6. Software stack

Graph (2-D tensors, batched products, element-wise, reductions, row norms, casts, composite
cross-entropy) → autodiff (gradients are graph nodes) → TITAN IR kernels → passes:
element-wise and GEMM-epilogue fusion, in-place parameter updates, liveness-based global memory
planning, LOCAL tiling (double buffering, resident weights), multi-core distribution (row blocks
or round-robin tiles), partition-aware barrier elimination, per-core stream interleaving → ISA →
static validation. Runtime: devices, host transfers (timed separately), buffer sharing between
executables (train/eval).

## 7. Multi-chip

Chiplet → package → system hierarchy (`titan/multichip`): ring, fully connected, 2-D mesh and
switch levels with assumed link bandwidth/latency; hierarchical all-reduce cost model;
functional ring all-reduce; data-parallel training with replica-consistency checks.

## 8. Large models

Graphs can be built in meta mode (shapes only) for any size; parameter and FLOP counts are read
from the graph and compiled to TITAN IR without host memory. `titan/scale/planner.py` splits a
transformer over many packages (tensor, pipeline and data parallelism; ZeRO 1-3; activation
recomputation) and estimates memory per device and step time. Numeric training on a host is
limited by its RAM (255 M parameters measured on 16 GB with the fast executor).

## 9. Research targets and non-goals

* Targets: 10× one B300 per package, 100× per system, on like-for-like workloads — unvalidated.
* The cycle model is not RTL-accurate; the target configuration is not physically designed
  (no area, power, thermal or packaging analysis); no HBM PHY, die-to-die PHY or optical I/O RTL.
* Sparsity and FP8/FP4 are not implemented, so no comparison at those precisions is possible.
