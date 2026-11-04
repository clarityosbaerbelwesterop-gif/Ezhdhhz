# Roadmap and acceptance gates

## D0 — RTL functional tile
- [x] Define 16-lane INT8 MAC tile.
- [x] Add self-checking directed testbench.
- [x] Add reproducible Verilator Makefile.
- [ ] Run tests in a real tool environment and record version/logs.

## D1 — Robust digital verification
- Randomized signed operand tests against a software reference.
- Overflow behavior specification and tests.
- Assertions for reset, clear and enable priority.
- Lint with warnings resolved rather than suppressed.

## D2 — GEMM dataflow
- Input staging and output accumulation buffers.
- Tiled matrix multiplication controller.
- Explicit backpressure and valid/ready interfaces.
- SRAM inference and bandwidth model.

## D3 — Performance architecture
- Parameterized array sizes and data widths.
- FP16/BF16 design study (not simply widening INT8).
- Roofline model with bandwidth, utilization and memory traffic.
- Transformer kernel mapping and workload traces.

## D4 — FPGA proof
- Select an affordable supported FPGA board only after checking current availability and toolchain.
- Synthesize, place, route and measure actual clock, resources and power where instrumentation permits.

## D5 — ASIC exploration
- Select a compatible open PDK and shuttle.
- Run synthesis, floorplanning, place-and-route, DRC/LVS and timing analysis.
- Treat open-PDK results as process-specific research, not a shortcut to leading-edge manufacturing.

## D6 — Multi-chip system
- Die-to-die protocol and topology.
- Distributed memory and collective communication.
- Fault tolerance, checkpoint/restart and thermal/power management.
- Measure scaling efficiency; do not assume linear scaling.
