# NPU Compilation Pipeline

This document explains how the Merlin compiler lowers a `linalg.matmul`
or `iree_linalg_ext.attention` op all the way down to a simulator-ready
instruction stream, what gets tiled by which pass, how running state
(MXU accumulator, flash-attention softmax state) flows across tile
invocations, and how the memory planner allocates DRAM.

Every IR snippet cited here is the input to a lit test under
`compiler/src/merlin/Dialect/NPU/Transforms/tests/` — the doc will not
drift silently from the compiler's actual behavior.

## Four IR layers

```
 linalg IR                                 whole-tensor ops
    │
    │  ConvertLinalgToNPUKernel
    ▼
 npu_kernel IR                             per-op kernel "intents"
    │   (npu_kernel.matmul, ukernel_generic)
    │
    │  TileNPUKernelToSchedule            (outer M/N/batch scf.for;
    ▼                                      K / KV unrolled in body)
 npu_schedule IR                           tile-level schedule
    │   (npu_schedule.ukernel_launch)
    │
    │  ConvertNPUScheduleToISA            native-kernel-lowering
    ▼                                      expands each launch into
 npu_isa IR                                manifest instructions
    │   (npu_isa.dma_load, matmul_mxu0,…)  with DMA address patching
    │
    │  PlanNPUISAMemory                    assigns DMA base/flag
    ▼                                      attrs (liveness analysis)
 npu_isa IR (planned)                      concrete addresses
    │
    │  npu-translate --mlir-to-npu-text-isa
    ▼
 text ISA                                  simulator-loadable stream
```

## Tiling decisions per pass

### TileNPUKernelToSchedule

Tiling source: `compiler/src/merlin/Dialect/NPU/Transforms/TileNPUKernelToSchedule.cpp`.

**Outer loops** (M, N, optional batch) are emitted as `scf.for` with
the output accumulator carried as an iter_arg.

**K reduction / KV attention iteration** is **unrolled at IR
generation time**. Unrolling lets the compiler pick the right variant
symbol per iteration:

  * K-tile chain — `matmul_acc_first` (overwrite MXU accumulator),
    `matmul_acc_mid` (add to it), `matmul_acc_last` (add + drain to
    DRAM).
  * K/V tile chain for attention — `attention_acc_first` (initialize
    running (m, l, O) to -∞/0/0), `attention_acc_mid` (load prev
    state, update), `attention_acc_last` (load, update, divide by l,
    emit final output).

#### Worked example: matmul at K = 2

Input (see `tile_matmul_k.mlir`):

```mlir
func.func @matmul_two_k_tiles(
    %lhs: tensor<32x64xf8E4M3FN>,
    %rhs: tensor<64x32xf8E4M3FN>) -> tensor<32x32xbf16> {
  %0 = npu_kernel.matmul %lhs, %rhs
      : tensor<32x64xf8E4M3FN>, tensor<64x32xf8E4M3FN> -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}
```

After `tile-npu-kernel-to-schedule`:

```mlir
%1 = scf.for %m = %c0 to %c32 step %c32 iter_args(%acc0 = %init) {
  %2 = scf.for %n = %c0 to %c32 step %c32 iter_args(%acc1 = %acc0) {
    // K-loop unrolled (2 iterations → first + last)
    %a0 = tensor.extract_slice %lhs[%m, %c0]  [32, 32] [1, 1] : …
    %b0 = tensor.extract_slice %rhs[%c0, %n]  [32, 32] [1, 1] : …
    %3  = npu_schedule.ukernel_launch "npu_uk_matmul_acc_first"(%a0, %b0) : …
    %a1 = tensor.extract_slice %lhs[%m, %c32] [32, 32] [1, 1] : …
    %b1 = tensor.extract_slice %rhs[%c32, %n] [32, 32] [1, 1] : …
    %4  = npu_schedule.ukernel_launch "npu_uk_matmul_acc_last"(%a1, %b1) : …
    %ins = tensor.insert_slice %4 into %acc1[%m, %n] …
    scf.yield %ins
  }
  scf.yield %2
}
```

The outer scf.for is degenerate at this shape (M=N=32 → one iteration)
but is emitted unconditionally; trivial loop folding can simplify it.
At K=3 the middle iteration gets `matmul_acc_mid`; at K=N it gets N-2
mids.

### ConvertNPUScheduleToISA — address patching

Source: `compiler/src/merlin/Dialect/NPU/Transforms/ConvertNPUScheduleToISA.cpp`.

With `--iree-npu-native-kernel-lowering=true`, the pass loads the kernel
manifest (`npu_model/kernel_library/manifest.json`) and, for each
`ukernel_launch`, emits the manifest kernel's instruction stream inline
with its per-invocation `invocationIndex`. It then walks every
`patch_point` entry (`dram_in_N`, `dram_out_N`) and rewrites the
`lui+addi` chain that loads the DRAM address so each invocation lands
in its own DRAM region.

Memory patching happens here, not in `PlanNPUISAMemory`. The planner
only assigns the outer `base`/`flag` attributes on the ISA DMA ops.

### PlanNPUISAMemory — DMA base + flag assignment

Source: `compiler/src/merlin/Dialect/NPU/Transforms/PlanNPUISAMemory.cpp`.

Walks each `FunctionOpInterface` in op order. For every
`npu_isa.dma_load` / `dma_store`, assigns:

  * `base` — monotonically increasing allocation pointer (separate
    counters for activation-load, weight-load, store).
  * `flag` — modulo-rotated sync token. Each load/store increments
    `nextIssuedFlag`; each `dma_wait` consumes the next pending flag.

See `plan-memory.mlir`:

```mlir
// Before
npu_isa.dma_load rd=2, base=111, flag=9
npu_isa.dma_wait flag=99

// After
npu_isa.dma_load rd=2, base=0, flag=0
npu_isa.dma_wait flag=0
```

The current allocator has no liveness analysis — buffers stay
allocated forever (monotonic counters only). Liveness is on the
roadmap; see the next section.

## Memory model

### Three address spaces, three counters

  * **Load region** — DRAM buffers the kernel reads from (activations).
    Default base configurable via `NPUMemoryPlannerOptions::loadBase`.
  * **Weight region** — DRAM buffers feeding MXU weight loads
    (`dma.load.mxu0`, `.mxu1`). Separate base so weights don't
    interleave with activations.
  * **Store region** — DRAM buffers the kernel writes outputs to.

### Liveness analysis (planned)

Planned upgrade to `PlanNPUISAMemory`:

1. **Liveness pass**. Walk ops in order. For each DMA load/store,
   record `defOrder` (op index) and `lastUseOrder` (last dma_wait that
   consumes the buffer's flag). Produces `(size, defOrder, lastUseOrder)`
   per buffer class.
2. **Greedy allocation**. Free-slot list per class; when a buffer's
   `lastUseOrder` passes, return its slot. Smallest-fit allocation on
   new buffers.
3. **Overlap verifier**. Debug-build assertion that no two live
   buffers occupy overlapping ranges at the same op index.

Expected effect: the full-SmolVLA compile's total DRAM footprint
shrinks by roughly the K/V buffer reuse factor (each block's attention
temporaries go away after the block completes).

### Layout model (per-kernel DRAM conventions)

| Kernel | input_layout | output_layout | Physical bytes |
|---|---|---|---|
| `matmul` | fp8 contiguous 32×32 | bf16 split halves (two 32×16 halves stacked) | 1024 in, 2048 out |
| `attention` | fp8 Q/K/V tile + bf16 scale | bf16 split halves | 4×1024 in, 2×1024 out |
| `attention_acc_*` | fp8 Q/K/V + bf16 scale (+6 state tiles for mid/last) | 6 state tiles (first/mid) or 4 col-blocks of [32,16] bf16 output (last) | See `attention_acc_kernels.py` |
| `elementwise_add/sub/mul` | bf16 split halves (interpretable as row-major [32,32]) | same | 2×2048 in, 2048 out |
| `rms_norm` | bf16 split halves + inv_dim const + eps const | bf16 split halves | 4×1024 in, 2×1024 out |
| `silu` | contiguous bf16 32×32 at DRAM 0 | bf16 split halves | 2048 in, 2048 out |
| `requant` | bf16 split halves | fp8 contiguous | 2×1024 in, 1024 out |

`bf16_split_halves`: 2048 bytes, 1024 B for cols 0-15 as [32, 16],
then 1024 B for cols 16-31. When a kernel's two halves share the same
operation (e.g., elementwise add), the layout preserves under composition
— the result is also split-halves.

### MXU accumulator semantics

The MXU's accumulator is persistent hardware state. It survives across
kernel invocations as long as no explicit pop (`vmatpop.bf16.acc.mxu0`)
or reset (`vmatmul.mxu0` = overwrite) intervenes.

  * `vmatmul.mxu0 vd, vs1, vs2` — `acc = v[vs1] @ weight`. Overwrites
    prior accumulator content.
  * `vmatmul.acc.mxu0 vd, vs1, vs2` — `acc = acc + v[vs1] @ weight`.
  * `vmatpop.bf16.acc.mxu0 vd, vs1` — drain accumulator to MRF as two
    bf16 halves (v[vd] = first, v[vd+1] = second).
  * `vmatpop.fp8.acc.mxu0 vd, vs1` — drain accumulator as fp8 tile.
  * `vmatpush.weight.mxu0 vs1` — load weight from MRF into MXU weight
    buffer.
  * `vmatpush.acc.bf16.mxu0 vs1` / `vmatpush.acc.fp8.mxu0 vs1` —
    inject bf16/fp8 data into the accumulator (used for bf16⇄fp8
    roundtrips).

The matmul chain (`matmul_acc_first → mid → last`) exploits this:

  * `_first` uses `vmatmul.mxu0` — fresh accumulator from A_0 @ B_0.
  * `_mid` uses `vmatmul.acc.mxu0` — adds A_k @ B_k into the running
    sum.
  * `_last` uses `vmatmul.acc.mxu0`, then `vmatpop.bf16.acc` drains the
    final sum to MRF and DMA-stores to DRAM.

No DRAM state spilling between K-iterations — the accumulator lives in
hardware.

The attention chain is different: its running state is (m, l, O_col0..3)
in MRF registers, which DO NOT survive across invocations. So
`attention_acc_*` spills those 6 state tiles to DRAM between
invocations, re-loads at the next, and ping-pongs across two
state slots to avoid a write-before-read hazard. See
`npu_model/kernel_library/stitch.py:stitch_attention_chain`.

## Invariants that downstream code depends on

  * Every kernel in the manifest has at most one `dma.store`
    destination per ordinal slot. The stitcher's `store_overrides[N]`
    maps exactly to the Nth store ordinal.
  * `addi` immediates must be within the 12-bit signed range
    `[-2048, 2047]`. Generator scripts enforce this via
    `fix_manifest_addi_overflow.py`; the simulator sign-extends, which
    matches RISC-V semantics.
  * `vload` / `vstore` use `imm12` in units of 32 bytes (i.e. one
    vector register width). A 32-byte-imm12 offset = one 32x16 bf16
    register width in memory.
  * Every DMA load/store has a matching `dma.wait` consumer in the
    same kernel — the planner's flag queue depends on balanced
    issue/consume counts.

## Tested examples

The MLIR snippets in this doc are inputs to lit tests that run in CI:

  * `compiler/src/merlin/Dialect/NPU/Transforms/tests/tile_matmul_k.mlir`
    — K=2 and K=3 tile examples; FileCheck verifies the scf.for +
    variant-symbol sequence shown above.
  * `compiler/src/merlin/Dialect/NPU/Transforms/tests/tile_attention_kv.mlir`
    — attention K/V-tile example (2D and 3D batched).
  * `compiler/src/merlin/Dialect/NPU/Transforms/tests/plan-memory.mlir`
    — memory planner base/flag assignment.
  * `compiler/src/merlin/Dialect/NPU/Transforms/tests/text_isa.mlir`,
    `ukernel_symbol_to_isa.mlir`,
    `linalg_generic_fp8_ukernel_pipeline.mlir` — full
    linalg→kernel→schedule→ISA→text-ISA lowering.

## Runtime side

The kernel library — manifest, stitcher, fixtures, per-kernel
goldens — lives in `third_party/npu_model/npu_model/kernel_library/`.
Its own pytest suite under `kernel_library/tests/` runs in npu_model's
CI independently of merlin:

  * `test_manifest_round_trip.py` — every kernel's text-ISA emit →
    parse is lossless (19 kernels, 823+ instructions).
  * `test_kernel_goldens.py` — every `KernelFixture` runs in the
    simulator and `torch.allclose` against its canonical reference
    (11 kernels, all PASS).
  * `test_chains.py` — matmul K-tile chain at K ∈ {2, 3, 4} and
    flash-attention chain at KV ∈ {2, 3} validate end-to-end
    numerically vs ISA-exact torch references.

`test_programs.py --fast-sim` runs the full Program suite without
per-cycle trace emission (see `LoggerConfig.fast_sim`), needed for
long simulations like `SmolVLAFullProgram`.
