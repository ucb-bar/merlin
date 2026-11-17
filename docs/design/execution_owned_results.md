---
title: "Results sized during execution"
kind: design
status: current
owner: core
last_verified: 2026-11-17
related: [runtime]
code_refs: [src/merlin/llvmlower/owned_results.py, src/merlin/llvmlower/owned_result_lowering.py, src/merlin/llvmlower/owned_result_abi.py, src/merlin/llvmlower/owned_result_c.py, src/merlin/llvmlower/owned_result_allocations.py, src/merlin/frontends/compile_inputs.py, src/merlin/frontends/derived_compile_inputs.py, src/merlin/frontends/execution_model_correspondence.py, src/merlin/frontends/execution_call_contracts.py, merlin/runtime/abi/owned_results.c, merlin/runtime/abi/owned_result_views.c, merlin/runtime/abi/owned_result_views.h]
---

# Results sized during execution

## Lowering before runtime selection

`lower_owned_results` emits source/prepared MLIR, LLVM and the source-typed C
bridge without host linking or invocation data. Its immutable lowering object
rechecks original input/result types, prepared/bridge reconstruction, selected
LLVM layout and product/implementation digests. The ordinary host compiler now
delegates to this seam before its existing shared-library link and execution.
Saved host programs additionally pin the common lowerer.

The caller supplies the actual selected layout/index contract. Explicit default
pointer fields must agree with the descriptor width; the current bridge requires
64-bit indices and independently asserts C pointer/integer layout at compilation.
The lowerer does not observe target hardware or grant a target verdict. A target
builder must separately observe/bind its compiler selection, allocator context,
resource map, linked runtime and actual execution.

Tests consume this seam with the actual observed cross-compiler contract and
compile both model and descriptor-bridge objects. These are object-level build
checks, not a linked or executed target image. Missing/incompatible layouts,
changed products and runtime/zero-trip ownership generations retain their
existing refusals. Native execution, checked-C calls and saved programs continue
through the same ordinary ownership and allocation gates.

## Checked generated-C calls

The ordinary returned-descriptor bridge now also exports
`merlin_owned_result_invoke_checked`. Within an explicitly begun invocation it
executes the original entry once, then validates every returned result using
`owned_result_views`. Caller-generated rank, storage and static extents remain
fixed. Actual descriptors determine dynamic extents; no output-size predictor
or evaluator value enters the bridge.

Validation checks actual live owners, alignment, complete signed-stride byte
footprints, shape/product overflow, total logical output bytes and exact closure
of returned owners. Every descriptor is checked before copying any payload.
An invalid result clears all output views and returns a diagnostic status;
the caller must release the recorded invocation even after a failure. Empty
views retain their owners but evaluate no inactive address arithmetic.
The generic C helper requires matching `size_t`/address widths; the selected
bridge retains its existing explicit 64-bit C ABI assertions.

`merlin_owned_result_copy` rechecks live source ownership and metadata before
copying row-major logical words through original signed strides. Destinations
must be valid independent caller storage; overlapping source-owner storage
refuses. Copies remain valid after invocation cleanup. Duplicate results retain
their full logical byte charge but share source owner cleanup. The original
Python descriptor validator remains an independent execution path.

The C generator and runtime implementation/header join the existing saved
program implementation pins and build-input closure. Native C executables
exercise actual compiled content-derived compaction, empty/duplicate/borrowed
results and repeated calls. Separate corrupt-descriptor tests run optimized
and undefined-arithmetic-trapping C builds. A calling-contract fixture checks
one original call and exact by-value scalar transport; it is not a numerical
source-equivalence oracle. This implements the reusable checked C call/copy
seam. Staged generated model drivers, target allocator context, target linking
and dynamic-result readback remain separate integration requirements.

`compile_owned_results` explicitly selects `merlin.execution-owned-results.v1`
for serial host execution. Ranks and storage types come from the original
public function. Result extents come from its executed descriptors, after the
original computation runs once. This supports dimensions computed from vector
contents without a separate shape-prediction invocation or a source expression
restricted to public axes and singleton words.

`stage_compile_inputs(..., result_mode="execution-owned")` explicitly selects
`merlin.compile-signature.v6` for ordinary captures, standalone kernels and
bound models after their existing package/source checks. The default result
mode and signatures v1--v5 retain their original behavior. Triton2MLIR's selected
host workers now compile and invoke this mode; native evaluator, generated C
runtime and target harness integration remain separate requirements.

## Compiler staging and repeated invocation

Signature v6 reconstructs every public input/result rank, storage type and
static dimension from actual typed source. Unknown result extents remain null;
no evaluator value or captured result size predicts them. Existing input
dimension/scalar relationships and scalar bounds remain explicit invocation
restrictions. They supply no output prediction or numerical/effect permission.
The signature separately fixes the selected allocation limits and native width.
Changed/rehashed result declarations, extra size programs, resource policies
and invalid axis coordinates refuse independent source reconstruction. Older
preallocated runtime readers refuse this version.

Staging reads only the original compiler members and preserves source locations.
The selected capture interface reports that C runtime generation was not
performed; result descriptors belong to the later host invocation. Compiler
workers receive no invocation-array or reference arguments. The original parent
archives, weights, kernel correspondence and executable payload remain intact.

`owned_result_program.save/load` retains the source, prepared module, LLVM,
bridge and library identities, selected limits and implementation identities in
`merlin.owned-result-program.v1`. Readers open fixed local compiler filenames,
never serialized paths, and independently reconstruct prepared source/bridge
types. The runtime reopens that program without compilation; complete result
arrays and actual allocation observations stay with the evaluator. These
records establish product custody, not source equivalence or hermetic closure.

## Ownership and ABI

The selection reuses upstream bufferization and every ordinary ownership,
deallocation and lifetime gate. It omits `buffer-results-to-out-params`, retains
function result descriptors, and lowers ownership-created clones through the
official `convert-bufferization-to-memref` pass. That pass follows ownership;
it does not establish ownership on its own.

Upstream ownership semantics borrow argument buffers and transfer returned
allocations to the caller. A returned view of an argument therefore requires
a copy. The official C interface places a returned descriptor, or aggregate of
descriptors, behind the first pointer argument. A generated C bridge delegates
aggregate offsets to the selected C compiler; Python does not guess tuple
offsets. All public MLIR tensor/scalar types and original computation stay
intact. Source locations and provenance are retained, without acquiring numeric,
alias or lifetime authority.

The initial transport explicitly selects native 64-bit indices. C static
assertions verify pointer and `intptr_t` widths with the selected compiler, and
actual descriptor sizes are compared before invocation. Supported inputs are
ranked unencoded dense tensor storage and the existing native scalar carriers;
results are 1..64 ranked tensors with ranks at most 16. FP8 and BF16 use their
declared raw-word storage. Arithmetic legalization remains a separate feature
selection. Scalar results, non-native widths, public memref/encoded/unranked
types and source-provider combinations need separate contracts.

## Allocation and observation

The selected MemRefToLLVM pass uses its official generic allocation functions.
An invocation-local host runtime records actual allocations, capacities, peak
charged bytes and allocation count. Empty allocations receive one owned byte;
alignment padding and tracking headers count toward the byte budget. Hidden
callback definitions and locally bound function definitions keep another loaded
model from substituting its allocator or entry. Runtime state is thread local;
the synchronous invocation may not reenter the same state.

Before MemRefToLLVM computes byte sizes, native checks retain each actual
allocation dimension, reject negative sizes, and bound multiplication by guarded
division. An overflowing element product cannot become a small allocator
request. Empty products retain zero-access semantics. The initial allocation
profile checks identity-layout heap allocations in supported scalar storage
types; stack and nonidentity allocation layouts need further resource contracts.
The allocator separately enforces actual byte/count budgets and rejects unknown
or repeated frees. Resource failures abort the native process, like source
assertions; callers needing failure isolation use a bounded worker.

Every returned descriptor must name a live allocation in that invocation.
Static extents, signed-stride footprints, alignment, accessible bytes and total
logical output bytes are checked before any payload copy. Empty views perform
no payload access. Unreturned live allocations refuse as leaked owners.
Repeated result owners are copied independently and released once. Cleanup
frees only the runtime's tracked owners, including when observation refuses;
it never frees a pointer merely because a descriptor supplied it.

The source, prepared program, LLVM, bridge, library and implementation bytes
are checked across execution. Loading uses the existing private-image helper;
unchanged program objects reuse their loaded image. These checks retain build
correspondence, not a proof of arbitrary source equivalence or a hermetic
dependency closure.

## Executed model call correspondence

The separate `checked_execution_model` frontend primitive independently
reconstructs explicit `t2m.model-kernel-binding.v4` /
`t2m.execution-checked-calls.v1`. Its inputs are the original parsed parent,
the bound module, the binding record and retained located kernel text/ABIs.
It imports no producer frontend or evaluator and executes no size predictor.
Every original typed call, scalar/dimension/narrowing guard and complete kernel
body must reproduce the entire bound SSA graph, including ordinary surrounding
computation and source locations. Rehashed declarations cannot authorize a
changed operation, missing guard, reordered call or substituted source mapping.

Original unlocated kernel artifacts omit source locations. V4 additionally
binds the actual located bytes in `located_implementations`; stripping locations
must reproduce the retained unlocated implementation digest. This supplies
byte/expansion correspondence, not TTIR numeric/effect equivalence, shape
prediction, ownership or target permission. Existing downstream lifetime and
resource checks remain mandatory.

Derived `t2m.bound-model-receipt.v2` now explicitly selects this v4 binding and
requires `stage_compile_inputs(..., result_mode="execution-owned")`. Ordinary
parent/kernel compiler-member, memory and source-capsule checks still run.
The additional located kernel bytes are retained alongside their original
unlocated payloads. Complete independent reconstruction precedes borrowed-input
annotation and signature v6 selection. Default mode, receipt downgrades, unknown
profiles, missing located artifacts and changed original SSA or types refuse.
The source correspondence is explicitly versioned
`merlin.derived-source-correspondence.v2` and records the checked expansion.

Public input dimensions are bound only to their actual source axes. Original
source-domain guards remain in the full executable body. Result dimensions
are discovered after execution; no public-axis/word prediction or static
guard-scratch estimate grants admission for internal sizes. The selected native
allocation policy accounts for actual private and result allocations.

A v2 source-only unqualified candidate can enter compiler staging before its
producer observes results. Staging also accepts the final qualified receipt,
but never opens evaluator arrays, reference archives, metadata or qualification
reports and supplies no execution verdict from the receipt's status. The
producer publishes its qualified package only after actual host execution and
complete original reference comparison. This explicit qualification backend
does not constitute a second independent native compiler implementation.

## Validation boundary and remaining integration

`test_owned_result_execution.py` executes positive-value compaction with
content-derived sizes, empty and duplicate results, borrowed snapshots, mixed
ranks, scalar inputs and exact floating words. It also exercises corrupted
descriptors, output budgets, native negative/overflowing allocation products and
source/build substitutions. `test_owned_result_allocator.py` executes alignment,
zero allocation, byte/count limits and invalid release paths.

A fresh resizing-loop experiment remains an explicit refusal: the independent
generation checker currently separates replaced carried owners only for
positive static trip counts with one final same-block release. Runtime/zero-trip
generation state and conditional or returned ownership need a separate closed
proof. The transport does not disable that check to obtain a successful run.

The selected staging/host interface now qualifies complete input-count models
with two dependent Triton calls in tensor, buffer and guarded-buffer profiles.
Explicit v4 producer binding now directly executes a real internally computed
count feeding two later Triton allocations. The consumer correspondence reader
and derived v2 staging preserve its full graph through the packaged interface.
Further work includes the separate native evaluator, target harnesses and
remaining internally computed shape, view/effect and resizing ownership
compositions.
Existing input-axis and source-proved size programs remain useful for
preallocated interfaces; they do not silently acquire execution-derived extents. No full
arbitrary-TTIR, GPU correspondence, accelerator or performance claim follows.

The upstream contracts and reused implementation are documented in
[Ownership-based Buffer Deallocation](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/),
[LLVM IR Target](https://mlir.llvm.org/docs/TargetLLVMIR/), and the pinned
[clone conversion and its tests](https://github.com/llvm/llvm-project/tree/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/test/Conversion/BufferizationToMemRef).
