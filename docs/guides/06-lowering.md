---
title: 6. Lower through scalar and OOT implementations
kind: guide
status: current
owner: compiler
last_verified: 2026-11-22
related: [05-transformations, 07-building]
code_refs: [src/merlin/compiler/system.py, src/merlin/compiler/backend/interface.py, src/merlin/compiler/backend/source_proposals.py, src/merlin/compiler/backend/providers.py, src/merlin/compiler/backend/scalar.py, src/merlin/compiler/backend/boundary.py, src/merlin/compiler/backend/c_abi.py, src/merlin/compiler/lower/device/region_lowering.py, src/merlin/compiler/lower/native, src/merlin/compiler/frontend/intake/kernel_atomic_evidence.py, src/merlin/compiler/dialects/fp8.py, src/merlin/compiler/lower/options/atomic_compare_exchange.py, src/merlin/compiler/lower/options/atomic_word_views.py, src/merlin/compiler/lower/interface.py, src/merlin/compiler/lower/interface_readout.py, src/merlin/compiler/lower/interface_materialization.py, src/merlin/compiler/ir/interface_contract.py, src/merlin/compiler/build/completion.py, src/merlin/compiler/build/regions.py, src/merlin/compiler/compilation.py, src/merlin/compiler/records.py]
---

# 6. Lower through scalar and OOT implementations

[Previous](05-transformations.md) | [Guide index](../README.md) | [Next](07-building.md)


Lowering turns a selected implementation into code or another admitted artifact.
This is where a region enters its OOT compiler. Shared scalar computation and
control lowering remain reusable Merlin components.

## Find the scalar code

[`compiler/backend/scalar.py`](../../src/merlin/compiler/backend/scalar.py)
exposes the installed `merlin.scalar` provider.
`compiler/lower/native/` owns the shared scalar pass sequence, terminal LLVM
conversion, ABI handling and native invocation.

The scalar provider requires an explicit data layout and index width. Object
emission also requires explicit target flags, including a target triple.
`scalar.call-entry` selects its supported synchronous callable tensor route;
the resulting object still needs image linking and independent checking.

The small `merlin.layout` provider uses `compiler/lower/c/layout.py` for its
supported layout-only subset. It is useful for the quickstart, but does not
replace the scalar arithmetic compiler.

Saved-model orchestration lives in `compiler/lower/model/`; optional pass
selection lives in `compiler/lower/options/`. These routes compose shared
host/device and native lowering rather than defining a separate target backend.

## Find the OOT boundary

`PlannedCompiler(selection, backends, *, image=None, validate_completion=None)`
in `compiler/system.py` consumes a selected execution plan. It validates every
region before compiling any region, then invokes the bound providers.
`backend/source_proposals.py` binds a selected lowering recipe and checks the
returned artifacts against that exact selection. A failed conditional lowering
stops compilation; it does not silently replan or claim scalar fallback.
`lower/device/region_lowering.py` connects selected region emission to the
existing device object builder.

The OOT repository owns its dialect, instruction selection, device kernels,
target resource descriptions, encoders and target ABI glue. Merlin owns shared
host transformations, orchestration, buffer ownership and build infrastructure.
An optimization that is independent of the accelerator belongs in the shared
library even if it was first measured on one target.

The compiler description in `backend/interface.py` connects named source
families to declared target-IR pipelines. Merlin owns upstream preparation and
conversion to canonical Interface IR. The normal OOT input is
`Source.kind="merlin.interface.v1"`; target conversion starts there. A generated
compiler must lower those operations and integrate their ABI, beyond providing
operation definitions or kernel builders.

`compiler/compilation.py` calls `backend/boundary.py` for providers declaring
that contract. The backend receives only the prepared Interface request.
The compilation retains both input identities and saves `merlin.interface.mlir`
as a shared artifact. Selected-region acknowledgments bind the actual target
products and the final shared artifact roster separately.
Providers without that declared input contract retain their explicit existing route.

Open `compiler/lower/interface.py` for
`to_interface(module, *, function=None)` and `ToInterfacePass`. Source conversion
covers one complete static signed i8-by-i8 matmul or batched matmul initialized from zero, with raw
i32 output or the admitted readout described in [planning](04-planning.md).
`interface.matmul` carries its explicit numerical contract, followed by
`interface.commit` and the original function return. The tensor arguments,
result and source symbol are preserved. Uncovered work and unsupported
attributes cause a refusal. This limited domain does not establish full-model
coverage.

The public parser is in `compiler/ir/interface_contract.py`.
`parse_interface` registers builtin, func, Interface and arith for static
parameter constants. Structural admission accepts only the exact constant op
from arith, not arbitrary arithmetic. `validate_interface` checks that shared
grammar; an authored Interface module needs no conversion receipt. Each OOT backend separately checks its
supported numerical and resource subset before selecting target code.

`lower/interface_readout.py` matches the complete source readout before
conversion. It supports the admitted ordered f32 scale/bias chain and the exact
signed-integer saturating cast. Parameterized readouts retain their immutable
f32 constants as actual `interface.commit` operands, including final-axis
channel mapping. The selected OOT implementation must separately prove that its
chosen strategy preserves this arithmetic, argument order and result binding.

## Materialize arithmetic after the Interface boundary

[`lower/interface_materialization.py`](../../src/merlin/compiler/lower/interface_materialization.py)
exposes `materialize_interface(module, *, function=None)` and
`InterfaceMaterializationPass`. This explicit route creates ordinary structured
IR for a complete raw-i32 matmul/commit/return chain. It accepts dense static
rank-two or equal-batch rank-three signed i8 inputs, sign-extends before
wrapping i32 multiplication/addition, and constructs the zero initializer.
It preserves the tensor signature and refuses readout epilogues or other chains.

This is arithmetic lowering after source matching. It chooses no placement,
packing, physical ABI or target instruction. A provider must explicitly compose
the materialization with its supported structured-IR pipeline; its presence
does not make the scalar provider accept every Interface program.

RVV implementation code belongs in `rvv-mlir`. A CPU with scalar and RVV
implementations can still be one shared hardware resource. Scalar fallback
must not be reported as evidence that vector or accelerator code executed.

Shared lowering reaches provider-implemented components through
`backend/providers.py`: the selected vector provider's `<component>_api()`
accessors (register-block tables, weight panels, fused requantization) and its
accumulator runner hooks. The optimization registry resolves the features those
components implement from the provider's `feature_definitions()`. With no
provider selected, requesting one of them fails with that reason.

## Tensor-kernel native lowering

The custom TTIR route retains standard tensor/linalg/SCF data flow and source
annotations. Type readers live in `compiler/dialects`, receipt intake in
`compiler/frontend/intake`, and native orchestration in `compiler/lower/native`.
The selected atomic, diagnostic, low-precision and loop-destination legalizations
live in `compiler/lower/options`, with their original source and effect checks.

The private-invocation atomic intake admits i8/i16/i32/i64 integer words and
checks the original dtype, operation, order, scope and every old-value result.
`frontend/intake/kernel_atomic_evidence.py` keeps the producer's buffer and
source facts independently checked; width admission does not grant an alias,
an external participant or permission to omit an event. The native
`lower/options/atomic_compare_exchange.py` helper retains exact integer
comparison/update/old-value semantics for those byte-sized widths. Logical
word views and their active alignment/byte-order checks remain explicit.
Integer updates through f16 storage use a private i16 return transport, then
restore the original bitcast at the caller; the native helper checks its full
body rather than trusting the view annotation. The same word semantics apply
to explicitly ordered, actively aligned byte-backed i16 views.

Direct bitwise exchange also covers f16, BF16 and the four pinned FP8 formats.
The existing atomic-word feature derives the actual storage type and retains
each returned old value using i8/i16 helper arguments and results. It admits
default-space rank-one storage, casting static shapes to the helper ABI, and
checks the complete allocation-free borrowing body. Floating addition policies
remain separate. `compiler/dialects/fp8.py` includes MemRef's copied float
constraints in its standard type registration; reader admission grants no
arithmetic policy. The frozen replay records reviewed updates when an old
non-support assertion becomes a qualified capability, preserving the refusal
guard with a valid unqualified body and retaining the original source blobs.

`merlin.compiler.lower.native.pipeline.lower_to_llvm_ir` retains
`owned_result_allocations=` and its typed resource selection. Destination reuse
and returned-loop ownership checks run at their existing bufferization and
deallocation seams. Provenance does not authorize dropping effects, live storage
or useful tiling.

The Python native runner finalizes number min/max with explicit NaN selection.
MLIR `arith.maxnumf`/`minnumf` select the numeric operand when only one operand
is NaN; older LLVM `maxnum`/`minnum` intrinsics may propagate signaling NaNs.
`lower/native/number_minmax.py` retains the original intrinsic for two numeric
operands, including its signed-zero behavior, and selects the actual numeric
operand for either NaN position. Propagating maximum/minimum operations and
transport words remain unchanged. The retained LLVM-dialect product contains
the same guarded computation that is translated and compiled. This Python
runner legalization does not qualify the separately selected stock CLI driver.

`merlin.compiler.lower.model.codegen.build_host_shared` delegates to the shared
build owner and retains `owned_result_bridge=`. Caller-provided outputs and
execution-owned descriptors remain distinct transports. Generated runners and
receipts name the moved implementations. Rebuild products when source commitments
change; frozen receipts are not rewritten to claim custody of the new layout.

## Outputs do not have to be LLVM

A backend may return compatible IR, an object file, a device binary, a command
stream or another declared artifact kind. `Compilation`, defined in
`compiler/records.py` and exposed through `compiler/compilation.py`, records the
artifacts and their hashes. A device-program backend need not pass through LLVM.

What matters is the next boundary: accepted representation, data layout, target
ABI, runtime requirements and invocation method. Textual interchange between
xDSL and native MLIR requires an admitted dialect/type/version subset. Separate
processes avoid a shared C++ ABI but do not remove IR compatibility requirements.

## Make the region callable

For a C-dispatch image, each region needs an admitted callable wrapper. The wrapper
describes its symbol, arguments, memory use and completion behavior. The device
backend supplies the target-specific call or launch; shared control calls that
wrapper. A raw device binary alone is not a callable host function.

`backend/c_abi.py` defines the C call and explicit context/workspace contracts.
`build/regions.py` emits shared control through `emit_selected_c`. Region
products retain their nested manifests; the aggregate compilation qualifies
their roles as `region.*` rather than treating a region ELF as the whole image.
A synchronous wrapper must complete the work it promises before returning;
an asynchronous design needs explicit completion dependencies.

Proposed `c_call_requirements` and canonical Interface device calls pass through
`build/completion.py:admit_call`. The original caller's `validate_completion`
callback receives the exact compilation, ABI, resource, memory domain and
topology. It must check selected-platform completion and CPU visibility, then
return `None` or refuse. A successful object build or a truthy callback return
cannot discharge that obligation. Existing admitted legacy C-call contracts
retain their explicit route.

The planned route records `compiled_regions`, `c_dispatch` or `linked_image`
in `plan.json`, together with remaining obligations. Image linking still
requires startup qualification, independent backend correctness and execution
evidence; none of these stages produces those verdicts by itself.

This boundary does not require an extra copy by itself. Packing or transfer is
required only when the selected layout, address space or ABI requires it, and
must remain visible in the plan and cost accounting.

## Open the implementation

| Responsibility | Source |
| --- | --- |
| Compile selected regions | [`system.py`](../../src/merlin/compiler/system.py) |
| Invoke the exact chosen lowering | [`backend/source_proposals.py`](../../src/merlin/compiler/backend/source_proposals.py) |
| Provider-implemented components and features | [`backend/providers.py`](../../src/merlin/compiler/backend/providers.py) |
| Scalar provider | [`backend/scalar.py`](../../src/merlin/compiler/backend/scalar.py) |
| Canonical Interface IR conversion and admission | [`lower/interface.py`](../../src/merlin/compiler/lower/interface.py) |
| Explicit Interface-to-structured arithmetic | [`lower/interface_materialization.py`](../../src/merlin/compiler/lower/interface_materialization.py) |
| Scalar pass sequence | [`lower/native/policy_pipeline.py`](../../src/merlin/compiler/lower/native/policy_pipeline.py) |
| C wrapper contract | [`backend/c_abi.py`](../../src/merlin/compiler/backend/c_abi.py) |
| Selected wrapper admission and control | [`build/completion.py`](../../src/merlin/compiler/build/completion.py), [`build/regions.py`](../../src/merlin/compiler/build/regions.py) |
| Artifact records and invocation | [`records.py`](../../src/merlin/compiler/records.py), [`compilation.py`](../../src/merlin/compiler/compilation.py) |
