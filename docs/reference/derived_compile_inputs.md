---
title: Derived tensor compiler inputs
kind: reference
status: current
owner: compiler
last_verified: 2026-11-19
related: [architecture]
code_refs:
  - src/merlin/frontends/compile_inputs.py
  - src/merlin/frontends/derived_compile_inputs.py
  - src/merlin/frontends/derived_runtime_arguments.py
  - src/merlin/frontends/derived_live_arguments.py
  - src/merlin/frontends/source_index_ranges.py
  - src/merlin/frontends/derived_symbolic_shapes.py
  - src/merlin/frontends/model_shape_facts.py
  - src/merlin/frontends/shape_expression_ir.py
  - src/merlin/frontends/shape_expressions.py
  - src/merlin/frontends/shape_input_words.py
  - src/merlin/frontends/shape_word_bindings.py
  - src/merlin/llvmlower/runtime_invocation.py
  - src/merlin/llvmlower/c_runtime.py
  - src/merlin/frontends/derived_source_views.py
  - src/merlin/frontends/kernel_compile_inputs.py
  - src/merlin/frontends/kernel_allocation_evidence.py
  - src/merlin/frontends/kernel_barrier_evidence.py
  - src/merlin/frontends/kernel_atomic_evidence.py
  - src/merlin/frontends/kernel_volatile_evidence.py
  - src/merlin/xdsl_dialects/standard_llvm.py
  - src/merlin/frontends/kernel_shape_abi.py
  - src/merlin/xdsl_dialects/standard_tensor.py
  - src/merlin/xdsl_dialects/standard_arith.py
  - src/merlin/xdsl_dialects/standard_bufferization.py
  - src/merlin/xdsl_dialects/standard_memref.py
  - merlin/tests/ir/test_derived_compile_inputs.py
  - merlin/tests/ir/test_derived_live_arguments.py
  - merlin/tests/ir/test_derived_symbolic_shapes.py
  - merlin/tests/runtime/test_runtime_input_axes.py
  - merlin/tests/runtime/test_derived_size_invocation.py
  - merlin/tests/runtime/test_input_word_sizes.py
  - merlin/tests/ir/test_derived_size_sources.py
  - merlin/tests/ir/test_derived_runtime_arguments.py
  - merlin/tests/ir/test_derived_source_views.py
  - merlin/tests/ir/test_kernel_compile_inputs.py
  - merlin/tests/ir/test_standard_bufferization.py
  - merlin/tests/ir/test_standard_memref.py
  - merlin/tests/ir/test_standard_arith.py
---

# Derived tensor compiler inputs

Selected private volatile reads use memory admission v5 and source-word stages
v3. The independent source reader requires actual typed volatile observations,
complete intermediate identity bodies and exact nondestructive stage pipelines.
The exported payload must retain real standard LLVM volatile loads inside closed
SCF descriptor-address leaves. Check base/offset/stride SSA, storage byte scale,
ordering and nonescaping pointers; annotations cannot replace an observation.
Older profiles cannot admit these low-level leaves. Hardware cache policies,
external/shared publication and broader effect compositions remain separate.

Protected allocation source phases explicitly select ordinary barrier v1 or
`t2m.synchronous-program-barrier.v2` when actual source atomics compose with
synchronous single-CTA rendezvous. The independent reader reconstructs primitive
atomic operation/type/order/scope and the selected private-invocation numeric
policy from actual typed source. Protected atomics must retain those fields and
original effect order; source selection must match bound memory evidence.
Returned old values remain executable. No producer/evaluator import, external
publication, cross-program synchronization or ownership permission follows.
Existing compiled memory-conflict and native lifetime checks remain required.

`stage_compile_inputs` accepts the existing `m2m.capture-receipt.v1` path and an
explicit `t2m.bound-model-receipt.v1` derived-model path. It publishes a
`merlin.compile-inputs.v1` directory with `source_profile: derived-tensor-model`.
The latter retains a typed `forward` program, weights and argument
manifest, signature, original parent source/FX trace, TTIR implementations/specs,
and `merlin.derived-source-correspondence.v1` derivation evidence.
Public tensor inputs carry `bufferization.writable = false`: they are borrowed
functional values, including parameters, and must survive repeated invocations.
This prevents function-boundary bufferization's default writable-argument policy
from silently turning destination reuse into caller-visible mutation.

The original FX trace remains under `source/parent/`; it is not relabeled as
correspondence for the transformed model. A compiler-only projection of the
parent receipt binds selected source members. Evaluator arrays, goldens,
execution reports and their digests are absent from the compiler view. All
selected files are checked against the producer receipt and their parent/kernel
relations before an atomic publication. Unknown dependencies, unresolved calls,
unregistered operations and malformed lineage refuse without publishing a stage.

Bound-model correspondence v3 selects
`t2m.captured-source-tensor-views.v1` for additional original schema operands
that are views of a declared owning root. `derived_source_views` independently
recovers static canonical transpose/positive rank-preserving slice coordinates
from parent SSA, then compares every coordinate and type to captured typed
integer arguments and the executable allocation-root pointer binding. Identical
tensor casts are transparent. All original schema tensor operands remain
accounted for. Rehashed records, wrong roots/pointers, changed scalar facts,
word-changing regions and missing/downgraded profiles refuse before publication.
Dense row-major value reshapes retain their existing optional profile alongside
these source views. The public model ABI stays unchanged; logical relationships
grant no physical alias, mutation, numeric or target permission. Evaluator
archives/digests never supply view or scalar facts.

The additive `t2m.live-runtime-arguments.v1` argument profile retains actual
original scalar SSA operands and singleton-tensor elements in model calls.
`derived_live_arguments` checks the exact i1/i8/i16/i32/i64/f32/f64 type,
unencoded singleton shape, standard adapter and complete per-call record.
An index conversion additionally needs a representable original SSA constant
or static dimension fact. Source integer facts come only from standard integer
constants, static `tensor.dim` with a known axis, and singleton
`tensor.from_elements`, under a finite inspection bound. No saved input or
weight value supplies a compiler fact. Captured attributes and live operands
partition the source scalar formals; complete schema accounting remains.

`t2m.checked-index-runtime-arguments.v1` additionally admits an index enclosure
proved by original dominating executable assertions. The source must check an
unsigned index/integer/index round trip. Signed/unsigned comparisons and typed
extended literals constrain the fixed-width integer words; only a nonnegative
enclosure representable by the declared signed kernel type admits the standard
`arith.index_castui` adapter. No native index width or `nonNeg` promise is
invented. The separate `source_index_ranges` reader uses actual SSA and constant
tensor axes, never messages, tags, samples or producer analysis imports.

Check `unsigned_integer_range` and the exact adapter against the original parent
before staging. Assertions in sibling regions or after the call supply no proof;
downgraded profiles, altered intervals and invented constants refuse. Bounds
remain executable. Per-query budgets limit predecessors/condition visits to
4,096, enclosing blocks to 64, constant inspection depth to 16 and integer width
to 4,096 bits. Arithmetic/CFG/loop-carried closure remains separate. These ranges
do not provide concrete tensor dimensions, dynamic result allocation or scratch
costs by themselves. The separate symbolic profile below retains actual
input-axis relationships and independently reconstructs their storage bounds.

Unknown live data may control executable computation but cannot supply a
tensor dimension or scratch cost. Reopen the original parent before any
producer inlining; do not treat a prior opaque kernel's eventual literal result
as an original shape fact. Every named tensor extent still needs an actual
parent SSA/type fact, and declared scalar bounds retain their own executable
guards. A changed fact/type/adapter or downgraded profile refuses. The public
model signature remains static unless the explicit symbolic profile below is
selected. These custody/interface checks
grant no source equivalence, numeric, alias, mutation, effect or target proof.

## Source-proved dynamic model signatures

`t2m.symbolic-model-shapes.v1` selects independent shape reconstruction from
original parent SSA, actual kernel functions and the complete inlined body.
`model_shape_facts` follows standard input axes, explicit tensor/memref sizes,
casts, Linalg destinations, insert-slice/forall destinations, invariant for-loop
yields and matching if-region yields. Checked scalar adapters retain the
original dimension identity. Unknown/resizing producers stay unknown. Source
inspection is bounded to 64 nested queries and 16,384 visits per environment.

`derived_symbolic_shapes` checks dependent calls in original source order and
matches every input/result relationship to actual executable dimensions. It
compares original, declared and complete inlined contracts. Guards remain
executable; no sample, reference, producer import or provenance tag supplies a
dimension. Guard scratch is independently summed from each call's checked
source shape upper bounds, under `sum-of-source-shape-upper-bounds`.

The compiler publishes `merlin.compile-signature.v3`, with
`shape_scope: source-proved-input-axes`, unchanged dynamic public tensor types
and `dimension_bindings` naming actual function argument/axis coordinates.
Parameters retain their public positions; no scalar inputs are manufactured.
Compiler-only staging and evaluator exclusion remain unchanged. Missing or
downgraded profiles, altered axes/ranges/bodies/order and scratch costs refuse.

Runtime invocation independently reopens the standard body before resolving
result storage from actual input shapes. The C generator accepts the complete
positional argument roster and checks stored parameter/buffer/constant bytes
against the original weight container, retaining its ordinary blob transport.
This grants no parameter-update protocol, source-equivalence, alias, effect or
target implementation. Broader dimension expressions, dynamic views, resizing
state and data-dependent results remain separate work.

### Derived and conditional dimensions

`t2m.symbolic-model-shapes.v2` additionally selects `t2m.runtime-shapes.v3` and
the explicit `logical-size-program.v1` typed DAG. The compiler publishes
`merlin.compile-signature.v4`, with `shape_scope: source-proved-derived-dimensions`
and the checked `shape_program`. V1 model profiles/signature v3 retain their
original input-axis scope; older signatures refuse the new program field.
Public argument/result types, parent artifacts and receipt schemas are unchanged.

`shape_expression_ir` reconstructs actual scalar constants, fixed-width word
arithmetic/casts, comparisons and selections, plus rank-preserving slice sizes.
Widths, modular behavior and signed interpretation stay explicit. Index selection
and unsigned comparison choose no native index width. Only actual dominating
assertions transfer bounds between injectively transported input dimensions;
late, sibling-region and truncated-word equalities supply no such proof.

`shape_expressions` validates complete typed dependencies, content-bound node
identifiers and the exact reachable program. Source-derived expressions must
agree with actual kernel arguments/results and the complete inlined body.
When their syntax differs, bounded Z3 bit-vector queries prove equality over
original source assertions/enclosures. Only UNSAT admits the relationship;
missing or inconclusive solvers refuse. The optional `shape-proofs` extra declares
the solver dependency. Native workers need only pure program evaluation.
Programs have 4,096-node, 64-level, 16,384-visit and 4,096-bit-word budgets;
each proof has two seconds, one million resource units and 128 MiB. These limits
are resource refusals, not hardware facts or universal shape coverage.

Before allocating results, runtime invocation independently reconstructs the
actual standard body and checks the supplied expression contract. Canonical
but false replacement programs and rehashed result declarations refuse. Actual
input axes then size descriptors without changing compiler arguments or payload.
Conditional dimensions retain the original selections, and guarded scratch
still follows checked source upper bounds. These logical proofs grant no
physical alias, public mutation, effect, numeric policy or target implementation.
Runtime-supplied slice bounds, data-dependent sizes, resizing state and broader
model/effect compositions remain separate obligations.

### Runtime input words in result dimensions

`t2m.symbolic-model-shapes.v3` and `t2m.checked-word-runtime-arguments.v1`
separately select `t2m.runtime-shapes.v4` and `logical-size-program.v2`.
Signature v5 declares `shape_scope: source-proved-input-words-and-dimensions`
and exact `word_bindings`. Older programs/profiles/signatures reject these
fields and input-word leaves. The public model ABI and source artifact roster
stay unchanged.

`shape_input_words` recognizes only actual public singleton tensor reads or
native integer arguments, retaining their exact argument, coordinates and dtype.
Original dominating signed/unsigned assertions constrain the source words.
Injective extensions can transfer bounds; truncation, late/sibling assertions,
metadata and sampled values cannot. The checked size adapter keeps its exact
scalar SSA and independently reconstructed inclusive `integer_range`.

Source-only staging reconstructs each original call, executable kernel result
and complete inlined body without reading invocation data. Guard scratch still
uses checked source upper bounds. Runtime invocation rechecks that body before
`shape_word_bindings` reads declared i8/i16/i32/i64 source storage and evaluates
the program. Values supply invocation allocation, never a compiler constant or
proof premise. Negative logical sizes and wrong storage carriers refuse before
allocation. Rehashed false bindings, output declarations and canonical but
incorrect programs refuse. Earlier receipt and positional input transport schemas
remain intact.

This profile grants no physical alias, mutation, numeric, effect or target
implementation permission. General vector-data-dependent size computations,
runtime bounds/views, resizing state and effectful size reads remain separate
obligations. The existing finite program/proof budgets apply unchanged.

Standard parallel tensor operations have explicit xDSL bindings, including
multi-axis/dynamic-bound syntax. Standard buffer materialization retains compatible
shapes/element types independently of memref layout and checks its result and
writable/restrict attributes against native MLIR's rules. These bindings grant
no aliasing, ownership, numerical or target-execution proof.

Scalar standard `arith.index_castui` additionally has a registered binding in
the frontend and shared contexts. Retain the zero-extension/truncation operation
and optional native `nonNeg` promise rather than substituting a signed cast.
Exactly one side is index and the other a nonzero signless integer. Native
MLIR supplies execution semantics; parsing grants no source range, index-width
or symbolic public-model admission. Shaped unsigned casts remain separately
unsupported by this scalar binding.

The current derived
staging profile requires static tensor entry arguments/results. Runtime scalar/
dynamic signature admission and explicit effectful dependencies need separate
profiles. Staging establishes byte custody and declared lineage; it does not
establish complete source closure, semantic equivalence or compiled correctness.

Standalone `t2m.capture-receipt.v2` packages use the separate
`standalone-tensor-kernel` source profile. This admits a static functional tensor
ABI with read, fresh-write and readwrite buffers. Readwrite buffers retain their
incoming value and return an updated tensor; this profile does not mutate caller
storage. Argument/result types and ordering must agree with the declared spec,
receipt counts and input manifest. The parameter container must be empty.

The stager changes the single defined entry symbol to `forward` for the current
native ABI and records both names in `merlin.kernel-source-correspondence.v1`.
It retains exact original MLIR/TTIR/spec bytes and projects compiler provenance,
including distinct original and normalized operation inventories. Numeric
evaluation reports and reference data/digests are excluded from that projection.
No FX trace or Torch capture identity is manufactured. The standalone and derived
paths share compiler-only custody and borrowed-immutable input semantics; both
still require separate consumer lowering and execution evidence.

The standalone ABI v3 extension additionally admits explicitly typed by-value
runtime scalars at declared original TTIR positions. Their public arguments
follow the logical tensor inputs. The signature labels them `kind: scalar`,
with empty storage shape and an exact scalar dtype; the MLIR input remains a
plain scalar. These inputs receive no tensor bufferization attributes.
Invocation values are forbidden in scalar declarations and remain outside
compiler staging. Actual host execution uses the existing `ScalarArg` ABI;
neither staging nor this profile supplies dynamic shape/stride/grid admission.

ABI v4 additionally retains declared inclusive integer scalar bounds in each
signature entry. The interval must fit the signed scalar type; invocation values
remain excluded. This permits a consumer to validate values separately from the
frontend's executable entry assertions. Bound declarations do not grant a memory
ownership/initialization proof or certify the guard/payload semantics by staging
alone. Actual tensor dimensions remain static in this profile.

The explicit `shared-functional-buffer` payload profile retains the same
functional tensor/scalar boundary with standard memref/bufferization operations
inside it. Staging checks the additional memory witness, continuation input,
output and report against provenance and copies them into `source/` (or each
derived kernel's source directory). The projected receipt retains the payload
profile; evaluator values remain excluded. This custody check does not validate
the producer's memory proof or broaden source admission.

Standard `bufferization.to_tensor` accepts compatible strided memrefs with the
same shape/element checks as native MLIR. Standard `memref.reinterpret_cast`
accepts ranked or unranked sources while checking element type, memory space,
size/stride/offset and dynamic operand segments. These bindings preserve the
upstream raw-buffer representation without changing loop-carried SSA types.

Standalone ABI v5 additionally admits named dense runtime dimensions and
logical ceil-divided grid expressions over bounded nonnegative integer scalars.
Its `merlin.compile-signature.v2` retains symbolic shapes and named scalar
bindings for input checks and result allocation. Actual input/result dimensions
are dynamic; maximum bounds are declarations for proof, not captured sizes.
Legacy signatures remain v1. The compiler view receives no evaluator arrays or
example shape values. This declared-contract check does not establish that the
program enforces its guards, covers its outputs or has qualified numerics;
consumer lowering and repeated independent-shape execution remain separate.

The explicit standalone `shared-guarded-buffer` profile replaces the pure tensor
memory witness with `memory-admission.json`, retaining the same functional
tensor/scalar boundary and standard memory payload. Staging checks the report's
source/access/effect inventory, internal ABI layout, continuation hashes and
declared scratch cost per result element. It retains that cost in the
compile signature for evaluator allocation accounting, and copies the report
with the other compiler evidence. Evaluator values and references remain
excluded. These are declared-contract and byte-custody checks, not a proof that
the payload executes its guards or preserves source semantics. Actual host
compilation and independent repeated-invocation checks remain separate.

Static derived models can also retain guarded kernels. Staging derives declared
scratch from each kernel's typed result dimensions and the complete call
inventory, checks the model function/binding declarations, and records the sum
in `signature.json`. Host evaluation accounts for that scratch separately from
the final model outputs, which may be smaller than kernel intermediates. This
is declared resource consistency, not a proof of executable allocation lifetime
or semantic equivalence. Dynamic bound-model scratch requires another contract.

V2 admission additionally declares writable observations using sticky reader/
writer program tags and completed-write initialization checks. Staging checks
the protocol name and derives the 16/24/32-byte metadata cost from the maximum
logical grid's 31-bit program-ID fragments. The explicit `memory-effects`
continuation stops before CSE/canonicalization so the producer can preserve
source-base gather observations as actual memory loads before optimization.
V1 evidence remains readable with its original write-only admission scope,
16-byte cost and `memory-ir` continuation. Neither version grants an executable
memory or numerical proof through staging alone.

The explicit `shared-source-word-buffer` profile retains the same guarded
memory protocol and public ABI while selecting upstream's source-word address
stage. Additional `word-input.mlir`, `word-tensor.mlir` and `word-stage.json`
members bind its exact input/output and pass sequence, including the supported
64-bit offset option. Staging validates/copies these records and checks declared
computation pipelines: both the retained sequence and its exact additional
`math-expand-ops{ops=rsqrt}` pass are recognized. Unfiltered/different math
expansions, duplicate passes and changed offset widths refuse. These alternatives
do not change receipt schemas or the byte-bound event reconstruction obligation.
Staging checks declared
function access intent against admission. Logical-grid annotations retain the
same intent through producer model-call inlining. This custody check establishes
neither source equivalence nor numerical qualification and supplies no source
Triton frontend or evaluator data to the compiler.
