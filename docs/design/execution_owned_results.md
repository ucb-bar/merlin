---
title: "Results sized during execution"
kind: design
status: current
owner: core
last_verified: 2026-11-17
related: [runtime]
code_refs: [src/merlin/llvmlower/owned_results.py, src/merlin/llvmlower/owned_result_abi.py, src/merlin/llvmlower/owned_result_allocations.py, src/merlin/frontends/execution_model_correspondence.py, src/merlin/frontends/execution_call_contracts.py, merlin/runtime/abi/owned_results.c]
---

# Results sized during execution

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

This primitive is implemented and tested against handwritten fixtures and
actual count-producing Triton models. It is not yet connected to derived
package staging: old receipt/binding readers continue refusing v4 until its
located artifact retention and independent reconstruction are integrated.

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
count feeding two later Triton allocations. The consumer correspondence
primitive preserves its full graph. Further work must connect retained v4
artifacts and reconstruction to staging and extend producer/qualification,
native/target evaluators and remaining internally computed shape compositions.
Existing input-axis and source-proved size programs remain useful for
preallocated interfaces; they do not silently acquire execution-derived extents. No full
arbitrary-TTIR, GPU correspondence, accelerator or performance claim follows.

The upstream contracts and reused implementation are documented in
[Ownership-based Buffer Deallocation](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/),
[LLVM IR Target](https://mlir.llvm.org/docs/TargetLLVMIR/), and the pinned
[clone conversion and its tests](https://github.com/llvm/llvm-project/tree/a47bddccec30255619bb8c37fa59700e661d4e66/mlir/test/Conversion/BufferizationToMemRef).
