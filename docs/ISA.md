# TITAN ISA v1.0 — formal specification

Executable form: `titan/isa/spec.py` (validation), `titan/isa/encoding.py` (binary),
`titan/emu/semantics.py` (data semantics), `titan/emu/engine.py` (control semantics).
`tests/test_isa.py::test_isa_doc_in_sync` fails if this document misses an opcode,
flag, function code or error code defined in code.

Versioning: `major.minor`. A decoder rejects programs with a different major
version. Minor versions may only add opcodes, functions or flags in reserved
encodings.

## 1. Machine model

* A device has `cores` cores (tensor processing clusters). Each core has a
  DMA unit (`dma_engines` engines), a matrix engine (MXU, `mxu_rows × mxu_cols`
  processing elements), a vector unit (VPU), a special-function unit (SFU)
  and a private LOCAL scratchpad.
* Address spaces (2-bit code): `GLOBAL`=0 (global memory, HBM abstraction),
  `SHARED`=1 (on-chip SRAM visible to all cores), `LOCAL`=2 (scratchpad of the
  core executing the instruction). All addresses are byte addresses.
* The top `gemm_staging_kib` of every LOCAL scratchpad is reserved for the GEMM
  sequencer; software must not rely on its contents across GEMM instructions.
* Execution: the command processor reads one instruction stream and dispatches
  each instruction to the queue of the core named in its `core` field. Each
  core issues its queue **in order**; independent instructions may complete out
  of order, but every core observes its own memory accesses in program order
  (hazards are enforced by a scoreboard). Memory written by one core becomes
  visible to another core only through `SYNCHRONIZE` or `BARRIER`.

## 2. Data types (4-bit code)

| Code | Type | Storage | Notes |
|---|---|---|---|
| 0 | INT4 | two's complement, 2 per byte, element 2i in low nibble | cols and stride must be even |
| 1 | INT8 | int8 | |
| 2 | INT32 | int32 | accumulator type; arithmetic wraps modulo 2³² |
| 3 | FP16 | IEEE binary16 | |
| 4 | BF16 | upper 16 bits of binary32 | rounding: nearest-even |
| 5 | FP32 | IEEE binary32 | |

Float arithmetic is performed in FP32 and rounded to the destination type
(nearest-even). Integer destinations of `CONVERT` saturate; `INT32`
destinations of arithmetic wrap.

## 3. Tensor operands

A tensor operand describes a 2-D strided tensor: `space, addr, rows, cols,
stride, dtype` (`stride` in elements, `stride ≥ cols` when `rows > 1`).
Assembly: `G|S|L:<addr>:<rows>x<cols>[/<stride>]:<dtype>`.

Rules checked for every operand (static validation):
shape non-empty → `SHAPE_MISMATCH`; address aligned to the element size (INT4:
even cols/stride) → `MISALIGNED`; the whole span inside the space capacity →
`OUT_OF_BOUNDS`.

## 4. Binary encoding (little-endian 64-bit words)

```
header : [7:0] opcode [15:8] func [31:16] flags [47:32] core
         [51:48] n_operands [55:52] n_imm [63:56] reserved=0
operand: w0 [1:0] space [5:2] dtype [7:6] reserved=0 [47:8] addr [63:48] reserved=0
         w1 [19:0] rows [39:20] cols [63:40] stride
imm    : 64-bit; float immediates are IEEE binary32 in bits [31:0]
```

Program container: `"TITN" u16 major u16 minor u32 n_kernels u32 meta_len`,
JSON metadata (zero padded to 8 bytes), then per kernel `u32 id, u32 n_words,
words`, then `u32 n_words, words` of the main stream. Trailing bytes, unknown
opcodes, bad type/space codes and non-zero reserved bits are rejected
(`BAD_ENCODING`, `ILLEGAL_OPCODE`).

## 5. Flags (16 bits)

| Bit | Flag | Applies to | Meaning |
|---|---|---|---|
| 0 | ACC | MATMUL | C += op(A)·op(B) instead of C = … |
| 1 | TRANS_A | MATMUL, GEMM | op(A) = Aᵀ |
| 2 | TRANS_B | MATMUL, GEMM | op(B) = Bᵀ |
| 3 | TRANSPOSE | COPY | dst = srcᵀ |
| 4 | BCAST_A_ROW | vector binary | a is [1,N] broadcast over rows |
| 5 | BCAST_A_COL | vector binary | a is [M,1] broadcast over columns |
| 6 | BCAST_A_SCALAR | vector binary | a is [1,1] |
| 7 | BCAST_B_ROW | vector binary | b is [1,N] |
| 8 | BCAST_B_COL | vector binary | b is [M,1] |
| 9 | BCAST_B_SCALAR | vector binary | b is [1,1] |
| 15 | NOTIFY | all | post a completion record `(stream index, core, opcode, time)` when the instruction retires |

A flag not listed for an opcode → `BAD_FLAGS`. At most one broadcast mode per operand.

## 6. Instructions

Operands are listed as `dst, src…`. "L" = LOCAL only.

| Opcode | Mnemonic | Unit | Operands | Imm | Semantics |
|---|---|---|---|---|---|
| 0x00 | NOP | CTRL | — | — | No effect. |
| 0x01 | LOAD | DMA | dst ∈ {LOCAL,SHARED}, src ∈ {GLOBAL,SHARED} | — | dst ← src; same shape and dtype. |
| 0x02 | STORE | DMA | dst ∈ {GLOBAL,SHARED}, src ∈ {LOCAL,SHARED} | — | dst ← src. |
| 0x03 | COPY | DMA | any, any | — | dst ← src or srcᵀ (TRANSPOSE; not INT4). |
| 0x10 | MATMUL | MXU | C(L), A(L), B(L) | — | C ← op(A)·op(B) (+C with ACC). A,B ∈ {INT4, INT8, FP16, BF16, FP32}, same type; C is INT32 for integer inputs (exact products, wrap-around accumulation) and FP32 otherwise (FP32 accumulation, order implementation-defined). |
| 0x11 | GEMM | MXU (sequencer) | C(G), A(G), B(G) | alpha:f, beta:f | C ← alpha·op(A)·op(B) + beta·C. Macro-op: the command processor expands it into LOAD/MATMUL/VECTOR/STORE micro-ops on the issuing core using the staging region. Integer GEMM requires alpha = 1 and beta ∈ {0,1} (`BAD_IMMEDIATE`). |
| 0x20 | VECTOR_ADD | VPU | dst, a, b (L) | — | dst ← a + b |
| 0x21 | VECTOR_MULTIPLY | VPU | dst, a, b (L) | — | dst ← a · b |
| 0x22 | VECTOR_SUB | VPU | dst, a, b (L) | — | dst ← a − b |
| 0x23 | VECTOR_MAX | VPU | dst, a, b (L) | — | dst ← max(a, b) |
| 0x24 | VECTOR_DIV | VPU | dst, a, b (L) | — | dst ← a / b (float only; IEEE division by zero) |
| 0x25 | VECTOR_SCALE_ADD | VPU | dst, a [, b] (L) | alpha:f, beta:f | dst ← alpha·a + beta·b, or alpha·a + beta without b (float only) |
| 0x30 | REDUCE | VPU | dst, src (L) | axis:i | func ∈ ReduceOp {SUM, MAX, MEAN}; axis 0 = over rows → [1,N], 1 = over columns → [M,1], 2 = all → [1,1]. MEAN float only. |
| 0x31 | ACCUMULATE | VPU | dst, src (L) | — | dst ← dst + src; dst ∈ {INT32 (wrap), FP32}. |
| 0x40 | ACTIVATION | SFU | dst, src (L) | — | func ∈ Act (below), element-wise, float only. |
| 0x41 | NORMALIZATION | SFU | dst, src (L) | [eps:f] | func ∈ Norm: SOFTMAX (row-wise, max-subtracted), LAYERNORM ((x−μ)/√(σ²+eps), no affine), RMSNORM (x/√(mean(x²)+eps)). |
| 0x42 | CONVERT | VPU | dst, src (L) | [scale:f] | dst ← round_dst(src·scale); integer targets: round-half-even then saturate, NaN → 0; float targets: nearest-even. |
| 0x50 | SYNCHRONIZE | CTRL | — | sem:i, n:i | func ∈ SyncOp: SIGNAL — after every earlier instruction of this core has completed, sem += n (release). WAIT — block the core until sem ≥ n, then sem −= n (acquire). |
| 0x51 | BARRIER | CTRL | — | [first:i, count:i] | Every core in [first, first+count) (default: all cores) must arrive; each core's earlier work completes before arrival; no core proceeds before all have arrived. One stream instruction is dispatched to every core of the range. Not allowed inside kernels. |
| 0x60 | EXECUTE_KERNEL | CTRL | — | id:i [, offset:i] | Execute kernel `id` on this core (core fields of the body are replaced by this core). `offset` is added to every GLOBAL address of the body. Nesting depth ≤ `kernel_max_depth`. |

Shape rules: vector binary ops require a (and b) to equal dst shape, or the
broadcast shape selected by flags. MATMUL/GEMM require
`op(A): M×K, op(B): K×N, C: M×N`. Violations → `SHAPE_MISMATCH`.
Dtype rules: all vector operands share the dst type; DMA never converts
(`DTYPE_MISMATCH`).

### Function codes

* Act: RELU=0, GELU=1 (tanh approximation 0.5x(1+tanh(√(2/π)(x+0.044715x³)))),
  SIGMOID=2, TANH=3, EXP=4, LOG=5, SQRT=6, RSQRT=7, RECIP=8, STEP=9 (1 if x>0 else 0),
  GELU_GRAD=10 (derivative of the GELU approximation), SQUARE=11, IDENTITY=12.
* Norm: SOFTMAX=0, LAYERNORM=1, RMSNORM=2.
* ReduceOp: SUM=0, MAX=1, MEAN=2.
* SyncOp: SIGNAL=0, WAIT=1.

## 7. Errors

Validation runs before execution (whole program) and again after kernel
relocation. Runtime faults are precise: they report core, stream index and
opcode, and execution stops.

| Code | Name | Raised when |
|---|---|---|
| 1 | ILLEGAL_OPCODE | unknown opcode / mnemonic, BARRIER inside a kernel |
| 2 | BAD_ENCODING | malformed binary or assembly |
| 3 | OPERAND_COUNT | wrong number of operands or immediates |
| 4 | ILLEGAL_SPACE | operand in a space the opcode does not allow |
| 5 | OUT_OF_BOUNDS | operand span outside the space capacity |
| 6 | MISALIGNED | address not element aligned; odd INT4 cols/stride |
| 7 | DTYPE_MISMATCH | operand types violate the opcode's type rules |
| 8 | SHAPE_MISMATCH | operand shapes inconsistent |
| 9 | BAD_FUNC | function code not defined for the opcode |
| 10 | BAD_FLAGS | flag not allowed / conflicting broadcast flags |
| 11 | BAD_CORE | core id or barrier range outside the device |
| 12 | BAD_SEMAPHORE | semaphore id outside `semaphores` |
| 13 | KERNEL_NOT_FOUND | EXECUTE_KERNEL of an undefined kernel |
| 14 | KERNEL_DEPTH | kernel nesting deeper than `kernel_max_depth` |
| 15 | DEADLOCK | no core and no dispatch can make progress (e.g. WAIT without SIGNAL) |
| 16 | NUMERIC | value cannot be represented (e.g. out-of-range write to INT8 without CONVERT) |
| 17 | BAD_IMMEDIATE | immediate outside its legal set |

## 8. Assembly example

```
.kernel 1
    load @0 L:0x0:16x16:bf16, G:0x0:16x16:bf16
.end
execute_kernel @0 # 1, 0
load @0 L:0x200:16x16:bf16, G:0x200:16x16:bf16
matmul @0 L:0x400:16x16:fp32, L:0x0:16x16:bf16, L:0x200:16x16:bf16
activation.gelu @0 L:0x800:16x16:fp32, L:0x400:16x16:fp32
store @0 +notify G:0x1000:16x16:fp32, L:0x800:16x16:fp32
synchronize.signal @0 # 0, 1
```
