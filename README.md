# ROUGE TITAN

**Status:** digital research prototype — RTL baseline, not a fabricated ASIC.

ROUGE TITAN explores a scalable heterogeneous AI-compute architecture. The first deliverable (TITAN-X1-D0) is a small, synthesizable RTL compute tile plus reproducible simulation and analytical performance-model scaffolding.

## Reality and scope

Long-term targets:
- **Single package:** aspirational target of 10× one NVIDIA B300 on explicitly defined, like-for-like workloads.
- **System:** aspirational target of 100× one B300, subject to measured scaling efficiency.
- Workloads: LLM training, inference, HPC and data processing.

These are research targets, not achieved or demonstrated performance. Achieving them would require advanced process technology, HBM integration, high-speed die-to-die links, packaging, power delivery, cooling, verification, manufacturing and substantial capital.

## Prototype contents

- rtl/: synthesizable SystemVerilog 16-lane INT8 MAC tile.
- tb/: self-checking SystemVerilog testbench.
- scripts/: analytical throughput estimator with exposed assumptions.
- docs/: architecture, roadmap and verification notes.
- Makefile: Verilator simulation entry point.

## Quick start

Requirements: Verilator with SystemVerilog support, GNU Make, Python 3.10+.

    make test
    make lint
    python3 scripts/perf_model.py --macs 16 --clock-mhz 100 --ops-per-mac 2

The RTL tile performs 16 independent signed INT8 multiply-accumulate operations per enabled cycle. Each lane has a signed 32-bit accumulator. It is a functional architecture proof, not a competitive accelerator.

## Next gates

1. Pass simulation and lint.
2. Add formal properties and randomized reference-model tests.
3. Add tiled GEMM data movement and SRAM interfaces.
4. Build FPGA implementation and measure real clocks/resource use.
5. Evaluate ASIC RTL-to-GDS using a compatible open PDK and documented constraints.

No fabricated silicon, FPGA measurements, power figures, or B300-equivalent results are claimed.
