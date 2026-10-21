---
title: Compose integer accelerator products with Merlin host lowering
kind: guide
status: current
owner: compiler
last_verified: 2026-10-21
related: [shared_host_optimizations, shared_source_lowering, task_window_profile]
code_refs:
  - src/merlin/llvmlower/integer_matmul_calls.py
  - src/merlin/llvmlower/integer_epilogue.py
  - src/merlin/llvmlower/convolution_routes.py
  - src/merlin/llvmlower/fresh_tensor_writer.py
  - src/merlin/llvmlower/logical_borrowed_abi.py
  - src/merlin/llvmlower/lower.py
  - merlin/tests/ir/test_integer_matmul_calls.py
---

# Integer products with shared host composition

The retained EL4 V26 compiler emits its complete program through its own
`mixed_codegen.SourceEmitter`. Its output is already LLVM when it reaches
Merlin, so the ordinary tensor fusion and bufferization pipeline cannot improve
those tensor operations. Updating Merlin alone does not change this old path.

`merlin.llvmlower.integer_matmul_calls.lower_integer_matmul_program` provides
an explicit alternative at the tensor source boundary. Merlin recognizes exact
zero-seeded, signed i8 × i8 → i32 rank-two recurrences, asks an original caller
to select implementations, and replaces selected products with fresh tensor
writers. The backend receives geometry and supplies only product LLVM plus
implementation pins. Merlin's existing descriptor bridge, borrowed-buffer
lifetime checker, preprocessing, fusion, bufferization and ordinary LLVM
lowering own the surrounding computation. No alternate host tensor emitter is
introduced.

The original source's remaining users stay intact. In particular, a live raw
i32 result prevents treating the product as an int8-only computation. With an
explicit backend readout declaration, this route can now absorb a proved
saturation/conversion epilogue. Named, encoded, batched, nonzero-seeded
and unsupported recurrences retain their original source. A caller can decline
geometry its backend cannot implement. Malformed source remains an error.

## Using the composition

The original caller supplies `select(call)`, `emit_kernel(call)`, an explicit
`LogicalBorrowedABI` and the selected compiler's 64-bit data layout:

```python
program = lower_integer_matmul_program(
    original_source,
    workdir,
    select=backend_supports,
    emit_kernel=backend_emit,
    logical_abi=selected_logical_abi,
    data_layout=selected_data_layout,
)
```

`backend_emit(call)` returns `(raw_llvm_text, ((source_path, sha256), ...))`.
The result provides `program.host.ll_path`, `program.kernels` and
`program.bridges`. Compile and link these through the selected ordinary tools
and runtime. The external kernel symbol is `call.kernel_symbol`; it accepts
three dense row-major pointers, in lhs/rhs/out order, and must completely write
the selected i32 or i8 output before returning. No kernel or runtime is discovered by name.
The shared bridge handles descriptor offsets, layout/alignment borrowing,
bounded packing fallback and output disjointness checks. Its maximum fallback
scratch budget and the linked program's actual stack reserve must agree.

The supplied borrowed effects and format qualification remain caller premises;
this connection does not establish device semantics or physical qualification.
Independent complete-output execution is still required. The existing native
lifetime checker verifies actual forwarding bodies under those premises.

This API is selected by the original experiment/build driver. It does not
silently add imports to frozen author SDKs, modify a historical Phase 1 package,
switch every ordinary target build or restart an expired authoring campaign.
Prospective experiments should freeze the composed source, selected Merlin
implementation and backend together before timing comparisons.

## Captured capsule evidence

The Gemmini companion adapter `backend.gemmini_host_matmul.emit_raw_matmul`
reuses the exact original V26 Builder/contraction emitter for the product.
The new path passed all nine captured capsules: **8,491,008 exact int8 output
comparisons** under the original functional Spike runtime. The guest compares
every output element to the pinned independent captured reference after the
original entry-through-completion timing boundary. Input staging and output
comparison remain outside that boundary.

| Capsule (GEMM dimensions are M × N × K) | Original kernel instructions | Composed kernel instructions | Reduction |
| --- | ---: | ---: | ---: |
| GEMM 12544 × 256 × 64 | 363,373,737 | 29,475,439 | 12.33× |
| Conv 56, C64 | 419,277,506 | 57,276,525 | 7.32× |
| GEMM 3136 × 512 × 128 | 181,757,462 | 14,903,105 | 12.20× |
| Conv 28, C128 | 209,855,942 | 18,082,073 | 11.61× |
| GEMM 12544 × 64 × 256 | 91,657,029 | 7,945,117 | 11.54× |
| GEMM 784 × 1024 × 256 | 91,321,701 | 7,842,762 | 11.64× |
| Conv 14, C256 | 125,207,547 | 10,755,087 | 11.64× |
| GEMM 512 × 512 × 512 | 30,166,618 | 2,773,003 | 10.88× |
| GEMM 3136 × 128 × 512 | 46,167,977 | 4,317,998 | 10.69× |

These are retired-instruction counts, including accelerator command issue.
The new diagnostic counter adds a small amount of call-boundary overhead.
Functional Spike's `mcycle` is instruction based; these results establish
neither hardware cycle speedups nor roofline attainment. No timing-model fit
or new FireSim result is claimed. Convolution still has host im2col/layout
work, and device command/DMA/synchronization costs remain.

The gains primarily come from the existing upstream fusion, bufferization and
alignment handling. The additional scalar families imported from commit
`0cc289de5` are selected by ordinary lowering but report zero matches on these
integer sources; the improvement must not be attributed to those new families.

Task-local evidence is retained under
`out/artifacts/verification/merlin-host-composition-v4/`: `result.json`, each
prepared source, host LLVM, backend LLVM, descriptor bridge, ELF, independent
reference, complete comparison console and build commands. The task's
`merlin-host-composition-tails-v1` evidence additionally exercises target tails,
live raw i32 output, repeated invocation and output guards.

Native tests exercise complete outputs for 1×1×1, 3×7×5 and 17×31×19 geometry,
both with and without a live raw result, repeated calls and guard bytes. They
also check source refusal for unsigned widening, changed arithmetic, nonzero
initialization and unknown operation attributes, plus symbol collisions and
changed prepared files. Run with the ordinary compiler environment:

```bash
python -B -m pytest merlin/tests/ir/test_integer_matmul_calls.py
```

## Exact integer readout fusion

Pass `output_readout=selected_readout` to the composition API and supply the
original `LogicalBorrowedABI.readout_format_qualification_identity` premise.
The readout is a `DenseIntegerReadout`: typed scalar IR, output format and
backend instruction declaration. The backend separately accepts or declines
the proposed i8 `IntegerMatmulCall` with its readout identity. If declined, the
raw i32 product and original host epilogue remain.

The shared matcher follows a bounded, sole-use, same-coordinate chain of
parallel generics. It supports scalar constants, uniform tensor operands and
separate compare/select clamp stages. The actual scalar source is checked
against the selected readout with the existing QF_BV translation validator,
covering every i32 accumulator bit pattern. It rejects initializer reads,
changed bounds, unsigned clamp semantics, extra observers, unknown metadata
and intervening effects. This is a source-equivalence proof; target execution
and physical format/effect qualification remain separate obligations.

The Gemmini companion declares signed scale-one saturated i8 accumulator
readout and reuses the original V26 implementation. All six captured GEMMs
select that readout. Their surrounding clamp/cast loops disappear, and the
product's logical output transfer falls from four bytes per element to one.
The three convolution epilogues include bias and coordinate changes, so this
unary matcher conservatively retains their raw i32 route.

Complete-output Spike checks pass for all nine capsules (8,491,008 elements),
plus 2,028 comparisons across native i8 and connected raw-i32 tail cases with
repeated calls and output guards. Across the six GEMMs, the new readout cuts
kernel instructions from 67.26M with shared host lowering to 7.26M. The tall
GEMM falls from 29.48M to 2.45M. Across all nine, the total falls from 153.37M
to 93.37M. These instruction reductions are separate from cycle predictions.

The descriptor bridge now uses checked multiplication for span overflow and
exact unit-stride injectivity checks. This preserves the malformed-descriptor
and alias guards while avoiding dynamic division on dense buffers. The general
nonunit-stride packing fallback remains. The existing native bridge tests
exercise supported widths, layouts, alignment, aliasing and overflow cases.

## Bounded convolution preparation

`prepare_integer_convolution_panels(source, max_packing_bytes=budget)` reuses
the existing bounded im2col matcher, rewrite and realized-program verifier.
The composition API exposes it as optional `convolution_packing_bytes`.
It reads actual source windows, signed arithmetic, reduction order, layouts
and zero initialization; no selected accelerator is fabricated for matching.
Provenance metadata, zero tensor splats and intervening pure weight views
are supported. Unsupported or unnecessary rewrites return the original bytes.

The budget bounds the summed payload of distinct activation-panel tensors,
including a narrow tail. It does not bound total host memory or imply target
capacity admission. Each resulting product still needs separate backend
selection. The shared verifier checks actual windows, reduction order,
complete output coverage and private loop-carried panel reuse.

All three captured convolutions pass complete outputs with a 64 KiB budget.
Their activation packing payloads fall from 7,225,344 / 3,612,672 / 1,806,336
bytes to 32,256 bytes each. Instructions increase by approximately 11–16%
because the smaller products require extra calls. This option remains a
storage policy, with no default profitability claim. A separate experiment
with the existing generalize-before-fuse pass order improves two convolution
shapes and regresses the third; it also remains an experimental choice.

## Diagnostic timing

Timing is observed with the previously frozen native event engine and pricing
model. The new whole-invocation scope includes all internal host work and
ends after completion; staging and independent output comparison stay outside.
The first observations exposed a software tracking limit (65,536 memory
regions) and missing CPU-division prices. A separately recorded diagnostic
model raises only observer bookkeeping limits to 1,048,576 memory regions
and 2,097,152 scoreboard values. All hardware capacities, cache configuration,
prices, libraries and engine bytes remain unchanged. No prices were fitted.
The original model and unsuccessful observations are retained.

The division-free bridge allows the existing prices to cover the new dense
path. For example, 512³ GEMM predicts 5.24M cycles versus 52.27M originally;
the tall GEMM predicts 14.88M versus 577.08M. Their structural compute-floor
attainment is approximately 10.0% and 5.4%, respectively. Native event timing
remains unqualified, uncertainty intervals are unavailable, and structural
compute-floor attainment is not a complete attainable roofline. No new
FireSim measurement or 30% convergence claim follows from these predictions.

All nine final ELFs now have complete scoped timing estimates. The three
convolutions predict 117.93M, 56.75M and 50.42M cycles (56/28/14 output sides),
or approximately 1.5%, 3.2% and 3.6% of the structural peak. They are therefore
ahead of the GEMMs in the remaining-distance priority. Their estimates still
include host preparation and postprocessing. Raising software observation
budgets establishes no new hardware or model-domain qualification.

Task-local evidence includes `merlin-shared-epilogue-final-v1`,
`merlin-readout-tails-v1`, `merlin-shared-epilogue-panels-v1`,
`merlin-convolution-pass-order-v1`, and `merlin-shared-epilogue-timing-v2` /
`merlin-shared-epilogue-timing-v3` under `out/artifacts/verification/`.
The final directory's `timing-summary.json` joins each numerical verdict and
estimate by exact ELF digest and records the resulting capsule priority.
