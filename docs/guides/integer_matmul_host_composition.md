---
title: Compose integer accelerator products with Merlin host lowering
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [shared_host_optimizations, shared_source_lowering, task_window_profile]
code_refs:
  - src/merlin/llvmlower/integer_matmul_calls.py
  - src/merlin/llvmlower/integer_epilogue.py
  - src/merlin/llvmlower/convolution_routes.py
  - src/merlin/llvmlower/fresh_tensor_writer.py
  - src/merlin/llvmlower/logical_borrowed_abi.py
  - src/merlin/llvmlower/lower.py
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

## Complete-output verification

Historical captured-capsule measurements and exact products are retained under
`out/artifacts/verification/merlin-host-composition-v4/` and
`merlin-host-composition-tails-v1`. They are evidence for their recorded source,
backend, ELF and runtime, not a current universal speedup or hardware-cycle claim.
New selections need their own complete outputs, inputs, guards, source-to-object
joins and retained failure behavior.

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

Smaller panels can increase calls, preparation and synchronization even when
they reduce live payload. Compare complete emitted executions; a storage bound
does not establish a profitable panel size or pass order.

## Diagnostic timing

Use the selected whole-invocation scope, including host work and completion,
while keeping staging and independent output comparison explicitly separate.
Bind the actual timing engine, fixed model, observer budgets and executable.
Unknown domain qualification, prices, overlap or uncertainty remain unknown;
raising a software bookkeeping budget supplies no hardware authority. Retired
instruction counts, provisional event estimates and physical cycles are distinct.
Structural compute floors are not a complete attainable roofline.

Task-local evidence includes `merlin-shared-epilogue-final-v1`,
`merlin-readout-tails-v1`, `merlin-shared-epilogue-panels-v1`,
`merlin-convolution-pass-order-v1`, and `merlin-shared-epilogue-timing-v2` /
`merlin-shared-epilogue-timing-v3` under `out/artifacts/verification/`.
The final directory's `timing-summary.json` joins each numerical verdict and
estimate by exact ELF digest and records the resulting capsule priority.

This particular composition API consumes raw LLVM kernel text; that is its
explicit admitted boundary, not a restriction on all Merlin backends. Backends
that emit device binaries or command assets use their declared artifact and
launch integration through the public compiler service. They must not return a
non-LLVM artifact to this LLVM-specific callback.
