---
title: Shared scalar host optimizations
kind: guide
status: current
owner: compiler
last_verified: 2026-10-20
related: [optional_passes, shared_pass_phase2, ../reference/host_cpu_arithmetic, ../reference/quantized_host_optimizations]
code_refs:
  - src/merlin/compile/host_optimizations.py
  - src/merlin/compile/host_optimization_export.py
  - src/merlin/llvmlower/automatic_host_features.py
  - src/merlin/llvmlower/host_tensor_codegen.py
  - src/merlin/llvmlower/host_epilogue.py
  - src/merlin/llvmlower/requantization_match.py
  - src/merlin/runtime/host_rounding.py
---

# Shared scalar host optimizations

Merlin owns accelerator-independent host algorithms. Target adapters supply
their instruction selection, row layout, physical ABI, resources and device
schedules. Host CPU capabilities such as RV64GC/lp64d are selected explicitly
and independently of the accelerator. Compatibility imports in the companion
backend delegate to these implementations.

Ordinary serial Python `lower_model` compilation automatically composes three
checked tensor rewrites after ordinary fusion and before bufferization: ordered
single-accumulator contractions with a reduction extent greater than one,
ordered squared sums, and source math hoisting to a smaller proved broadcast
domain. Their typed matchers preserve operand and reduction order and reject
unsupported bodies, shapes and effects. Strict floating environment scopes
retain the original route. This joins the existing automatic row, destination,
permutation, bounded-rounding and copy passes.

`automatic-host-optimizations.json` records selected choices, actual rewrite
counts and before/after source hashes. It is bound into the lowering recipe and
returned in `stats["automatic_host_optimizations"]`. Missing or contradictory
completion reports refuse compilation. `--no-pass automatic-host-optimizations`
disables this composition; `--no-pass scalar-contraction`, `scalar-squared-sum`
and `broadcast-math-hoist` disable individual families. Explicit accumulator
tiles and source-bound numerical/effect routes retain their own selections.
Vector, parallel and custom transform schedules retain their existing route.

Packet widths, reduction unrolling and lookup tables remain explicit when
profitability is unknown. Approximation, contraction into FMA, masked arithmetic
omission and alternative libm precision still require their original declared
numerical/effect contracts and unchanged accuracy gates. The optional-pass
registry exposes the historical host choices without granting those permissions.
Selecting a named approximation is not evidence that a particular input permits it.

## Using the algorithms from a compiler generator

`merlin.compile.host_optimizations.rewrite_tensor_module(context, module)` runs
the same canonical native MLIR tensor matchers inside the caller's existing
context. Named tensor operations must first be generalized. The public module
also exposes bounded LLVM rounding, constant-clamp and explicitly permitted FMA
packet rewrites. Callers that construct xDSL LLVM can use `host_builder`,
`host_scalar_builder`, `host_tensor_codegen` and `host_epilogue` from
`merlin.llvmlower`; row alignment and scalar formats are caller inputs.
The historical branch-free polynomial primitives preserve their original
policy requirements and are never selected by the automatic tensor pass.

`integer_gemm`, `captured_requant` and `requantization_match` recognize logical
source contractions and epilogues without selecting a device schedule.
Requantization returns source dimensions and numerical proofs. An optional pool
source inspector is an explicit callback. Target adapters consume these results
to choose their own legal kernel/store implementation.

The retained EL4 V26 Phase 1 generator emits LLVM directly and imports it into
MLIR. Tensor passes cannot operate on tensors that have already been lowered
away. Such a generator can call the shared APIs at its source stage, or use the
LLVM and builder APIs at its existing stage. The frozen Phase 1 package remains
unchanged; this migration does not regenerate it.

For a prospective shared-source experiment, export a separate reviewable library:

```bash
python -m merlin.compile.host_optimization_export \
  --output-dir "$HOST_LIBRARY" --license-directory "$MERLIN_SOURCE"
```

The output directory must be empty and unlinked. The
`merlin.shared_host_library.v1` manifest lists exact source bytes, public modules,
dependencies and license resources. Review it with the existing
`freeze_compiler_library`, including `shared-host-library.json`, then select the
resulting `CompilerLibrarySelection` for both comparison arms as described in
[shared-pass preparation](shared_pass_phase2.md). This is an additional library;
it does not change frozen SDK inventories, profiles or grants. It excludes
device code, reference models, graders, measurements and compiler discovery.

## Coverage and execution boundaries

The shared library includes scalar/tensor/epilogue builders, explicit host
arithmetic policies, source-order and quantization contracts, loop controls,
packing and prepared-model interfaces. Target kernels, physical layouts,
instruction selection and device scheduling remain target responsibilities.

Use the individual API contracts and focused tests to determine supported
semantics. Host execution tests and source proofs do not establish target
adapter correctness, complete-model execution or a hardware performance result.
Record those separately for the selected compiler, inputs and runtime. Public
source availability does not supply historical binaries or their measurements.
