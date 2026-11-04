# TITAN-X1 Architecture — Design Note D0

## Objective
Establish a verifiable RTL foundation for a future heterogeneous AI accelerator. Long-term ambition: 10× one B300 for a single package and 100× for a multi-accelerator system. These are unvalidated targets.

## D0 compute tile
- 16 independent signed INT8 MAC lanes.
- 8-bit signed inputs, 16-bit signed product, 32-bit signed accumulation.
- One product accumulation per lane per enabled cycle.
- Flat packed input/output buses to keep the first RTL interface simple.
- Asynchronous active-low reset and synchronous accumulator clear.

At an assumed 100 MHz, the tile's arithmetic peak is 16 MAC × 2 operations × 100 MHz = 3.2 GOP/s, before utilization and implementation losses. This is intentionally a small functional prototype.

## Future package decomposition
1. Compute chiplets: tensor/matrix arrays, vector units, sparse engines.
2. SRAM/cache chiplets: operand reuse and local buffering.
3. I/O and memory base die: HBM PHY/controllers and die-to-die links.
4. Optical I/O research: board/rack scale links; not assumed inside D0.
5. Management die: RISC-V control, telemetry, security and scheduling.

## Critical research questions
- Which GEMM tile shapes maximize useful throughput for transformer kernels?
- How much HBM bandwidth and capacity are required for target model sizes?
- What fraction of end-to-end training time is compute vs communication vs optimizer state movement?
- Can sparsity or MoE routing reduce work without harming quality?
- What are the thermal and power-delivery limits of the package?
- Which functions benefit from optical links or in-memory compute, and which do not?

## Non-goals for D0
- No 27B model training.
- No HBM PHY, optical transceiver, analog compute or advanced packaging RTL.
- No claim of B300 equivalence, 10× performance, tapeout readiness or fabricated silicon.
