---
title: "Results sized during execution"
kind: design
status: current
owner: core
last_verified: 2026-11-17
related: [runtime, execution_resource_policy]
code_refs: [src/merlin/llvmlower/owned_results.py, src/merlin/llvmlower/owned_result_lowering.py, src/merlin/llvmlower/owned_result_abi.py, src/merlin/llvmlower/owned_result_c.py, src/merlin/llvmlower/owned_result_driver.py, src/merlin/runtime/owned_result_readback.py, src/merlin/runtime/backends/owned_result_model.py, merlin/runtime/baremetal/spike/owned_result_main.c, src/merlin/llvmlower/owned_result_allocations.py, src/merlin/frontends/compile_inputs.py, src/merlin/frontends/derived_compile_inputs.py, src/merlin/frontends/execution_model_correspondence.py, src/merlin/frontends/execution_call_contracts.py, merlin/runtime/c/owned_result_driver.c, merlin/runtime/c/owned_result_readback.c, merlin/runtime/c/owned_result_host_main.c, merlin/runtime/abi/owned_results.c, merlin/runtime/abi/owned_result_views.c, merlin/runtime/abi/owned_result_views.h]
---

# Results sized during execution

## Explicit functional mutation results

Executed model binding may explicitly select `t2m.functional-kernel-binding.v1`.
The original model2MLIR call retains `m2m.functional-kernel-bases.v1`: original
schema text, complete mutated argument names, typed operand/base/result ties and
every raw tuple slot. Ordinary observations precede updated bases; only a void
source op may account for an unused None tuple slot.

`execution_functional_state.py` independently checks the original call against
the retained translated memory ABI. Each tie must consume and return the same
readwrite source buffer at its actual schema operand and updated-result ordinal.
An equally typed fresh output cannot replace that state. All bases, slots and
source readwrite effects must be accounted for. Missing/unknown profiles and
incomplete or changed ties refuse before complete-SSA reconstruction. No producer
implementation or evaluator array is imported by this reader.

Public input or buffer state additionally has the original
`m2m.functional-state-results.v1` entry contract. Complete state-result/input
ties and ordered user results retain exact types. These are explicit returned
state values: staging continues to mark every borrowed entry tensor readonly.
Public physical aliases, automatic caller copyback and target synchronization
semantics require separate invocation/consumer support. The normal bufferization,
allocation, result-descriptor and lifetime checks stay active. Parent archives,
receipt v2 and compile-signature v6 retain their schemas.

The independent functional-state tests construct original SSA/ABI directly and
reject fresh-output substitution, readonly source substitution, missing/changed
profiles, invalid base/operand indices, wrong integer types, changed tuple slots
and malformed entry result roles. Whole-model source expansion and execution
remain separate gates.

## Ordinary serial target images

The ordinary `runtime.backends.spike_model.build` API now selects the returned
result path only for an explicit signature v6, positional invocation archive,
complete readback and complete original input observation. Earlier signatures
retain their original builder. Unsupported nondefault numeric, device,
transformation, profiling, console or effect options refuse before building;
none are silently discarded. The current selection is a serial HTIF executor.

`owned_result_model` reuses the observed compiler/index plan, shared source-only
lowerer, exact typed driver, existing bounded allocator/startup/linker/console
and explicitly resolved linked math archive. Runtime arrays become a separate
blob, including exact parameter/scalar words. Zero-input functions use a physical
sentinel without inventing a logical argument. Actual ELF intervals, stack and
allocator must be disjoint inside the declared span; an optional provider-decoded
map keeps its ordinary admission requirements. Simulator backing memory itself
does not establish a board's decoded map. Compilation records bind preparation,
actual compile/link commands and the completed image.

The bare-metal adapter takes input placement and size exclusively from the
builder. Shared readback preserves binary payloads through the existing console.
Success requires complete independently decoded shapes, payloads, inputs and
DONE as well as the actual process outcome. A guest allocator failure's encoded
status may truncate to host exit code zero; missing completion still refuses.
Source assertions, allocation/arena failures and arbitrary traps cannot become
successful tensor qualification.

Production builder checks execute dynamic empty/sparse/integer-extreme results,
zero-input functions and actual arena exhaustion. Triton2MLIR's explicit
`--result-mode execution-owned` target qualifier additionally stages the real
three-kernel model, compares every original PyTorch result and replays complete
independent source staging when verifying retained observations. This extends
functional serial target execution; source effects, arbitrary views/resizing,
numeric combinations, accelerators and performance remain separately qualified.

## Generated invocation and observed-shape readback

`owned_result_driver.generate` reconstructs signature-v6 source/input contracts
and emits `model_owned.c`, a separate `invocation.bin`, `signature.json` and a
versioned driver record. Reopening reconstructs all four members and selected
implementation identities from current source and actual invocation arrays.
Serialized paths cannot redirect inputs. The bounded little-endian transport
fixes input shapes for that driver; data and native scalars may change without
recompiling its model or driver. Scalar bounds, dimension equalities, alignment,
blob length and canonical Boolean storage remain executable checks on each call.
No invocation value or result reference enters source-only compiler staging.

The reusable C controller begins one source invocation and calls the checked
bridge exactly once. It copies every validated result into independent storage
before releasing tracked source owners or invoking any observer. Duplicate
results retain separate logical copies. Allocation/descriptor failures publish
no result; callback failures stop publication and free private copies. A failed
begin does not release another caller's active context. The hosted adapter reads
the separate bounded blob and writes DONE only after all observations succeed.

The explicitly selected `merlin.execution-owned-readback.v1` profile reuses
OUT_BIN v1 framing with paired `inN_shape`/`inN` and `outN_shape`/`outN` frames.
Shape payloads are unsigned little-endian 64-bit dimensions; rank-zero shapes
have zero payload bytes. Value payloads retain exact flat storage words, with
signed integer framing and unsigned IEEE/BF16/FP8 storage framing. The complete
source-typed decoder independently checks ranks/static extents, actual input
shapes and bytes, output product/byte limits, canonical Booleans and the complete
frame roster before returning arrays. Dynamic output dimensions come from
execution alone. Raw-payload parsing avoids per-word Python integer lists while
retaining the ordinary strict checksum/framing parser and existing APIs.

Checks cover real compiled content-derived results, native scalars, rank-zero
IEEE words, empty/duplicate results, changed inputs and cleanup failures. The
frontend additionally links the actual three-kernel TTIR model LLVM with the
production hosted driver and decoder. This API is explicit runtime transport;
it does not yet integrate normal target staging/build/readback, grant source
effects, or qualify accelerator execution. Numerical qualification still needs
complete independent references and the actual successful process outcome.

## Lowering before runtime selection

The allocator additionally exposes two explicit build selections:
`MERLIN_OWNED_RESULTS_SERIAL_CONTEXT=1` uses one nonconcurrent process-local
context, and `MERLIN_OWNED_RESULTS_EXTERNAL_FAILURE=1` delegates fatal reporting
to the executor's noreturn `merlin_owned_results_failure`. Neither is inferred
from a target name or source operation. Other selected macro values refuse;
the default remains native thread-local state with hosted diagnostics. The
executor must separately establish serial invocation, allocator/arena bounds
and failure behavior. The selection supplies no multihart/thread-safety proof.

A bare-metal fixture now links the common lowerer's actual model/bridge and
serial runtime objects with the existing startup, bounded allocator and console.
One unchanged image handles content-derived, empty, duplicate and input-snapshot
results; a separate allocation-budget selection fails before result publication.
Tests retain image, compiler, simulator, harness/source and output identities.
This initial standard-MLIR fixture retains its original runtime-seam scope.
The ordinary serial target integration above has its own generated-driver,
staging, readback and complete-model execution checks.

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
seam. The generated driver and ordinary serial target selection above add their
own complete invocation, linking and observed-shape requirements.

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
host workers now compile and invoke this mode; its explicitly selected serial
target qualifier adds separate image execution. A second native evaluator and
broader target/effect/shape compositions remain separate requirements.

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

[Executed call views](execution_call_views.md) additionally connect checked
dense adapters and static source-coordinate/root proofs through their explicit
v4 extension profile. All original schema operands and complete emitted SSA
remain accounted for. Fully runtime reshape and public physical/mutating view
contracts remain separate requirements.

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

[Returned loop ownership](promoted_loop_ownership.md) now separately proves
fresh direct generations for runtime and zero-trip serial loops. Actual ownership
flags, materialized zero-trip copies, closed initial uses and complete returned
coordinates establish the lifetime; the checker remains enabled. Native/saved/C
and actual target tests retain changing/empty results and source guards.
Arbitrary branch/while/nested ownership, retained aliases and other final
consumers still need further proofs. No transport flag disables the check.

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
