---
title: Shared compiler pass contracts
kind: reference
status: current
owner: core
last_verified: 2026-10-17
related: [../design/shared_phase1_scheduling.md]
code_refs:
  - src/merlin/transforms/pipeline.py
  - src/merlin/compile/scheduling/compact_dense.py
  - src/merlin/compile/scheduling/compact_verification.py
  - src/merlin/compile/scheduling/program_limits.py
  - src/merlin/compile/scheduling/temporary_storage.py
  - src/merlin/compile/scheduling/device_spills.py
  - src/merlin/compile/scheduling/spill_conformance.py
  - src/merlin/compile/scheduling/copy_elimination.py
  - src/merlin/compile/scheduling/schedule_pipeline.py
  - src/merlin/compile/scheduling/fusion.py
  - src/merlin/compile/scheduling/fusion_semantics.py
  - src/merlin/llvmlower/automatic_host.py
  - src/merlin/llvmlower/quant_round_check.py
---

# Shared compiler pass contracts

The ordinary captured-policy host preparation path automatically legalizes
eligible bounded quantization arithmetic. Compact dense scheduling is a shared
compiler entry point over the existing counted-loop kernel IR. Its encoder and
actual-operand decoder are explicit integration inputs. It has no automatic
interpretation of an opaque backend program.

## Inputs and ownership

Select the original hardware configuration and numerical policy. Supply resource
and code-growth budgets, with optional optimization controls. Structural RTL
facts can establish dimensions, widths and capacity; they do not establish
instruction arithmetic, completion semantics or measured performance.

Target repositories retain their encodings, physical mapping, instruction
implementations and execution qualification. Merlin owns candidate search,
ordering, transformations and shared checks. A contract identity binds a chosen
input; naming an identity supplies no authority or missing hardware evidence.

## Ordering and analysis freshness

`PassSpec` names a stage, IR level, prerequisites, produced analyses and
invalidations. `validate_pass_order` rejects duplicate passes, missing producers,
stage inversions and incompatible IR levels. The common pipeline validates an
explicit `PipelineOptions.pass_order` before executing any transformation.
The default order normalizes semantics, performs narrow source buffer reuse,
segments or unrolls ordered loops, schedules dependencies, then recomputes and
checks final storage and dependency analyses.

The narrow source buffer rewrite runs before loop segmentation because its
initializer checker needs the original rectangular nest. Physical device
allocation needs the selected tile representation and belongs after tiling.

`PipelineSession` stamps analyses with the current IR digest, IR level and
selected hardware, numerical, resource, execution and library identities.
Any actual IR mutation invalidates old analyses, including a mutation whose
pass omitted an invalidation declaration. Changing a selected identity likewise
makes an old analysis unusable. A failed action poisons its private session.
An analysis pass that mutates IR is an error.

`repair(stage=..., reason=...)` clears analyses and restarts an already selected
legal fallback within an explicit repair budget. It cannot revive a failed
session, manufacture a fallback or approve an unchecked change. Existing covered
source transformations retain their host verification hooks. `plan_only=True`
remains explicitly unverified.

## Automatic bounded host arithmetic

`prepare_for_lowering` invokes `prepare_automatic_host` when preparation selected
the existing `merlin.quantization_policy.v1` captured policy. Missing policy and
explicit pass ablation preserve the original route. An explicitly selected
alternative rounding lowering keeps ownership of that choice.
The policy's `prepared_mlir_sha256` must bind the exact current input bytes.
A missing or stale binding retains the legal prepared input and writes a
diagnostic instead of applying a policy from another preparation stage.

The supported rewrite is a pure, straight-line tensor `linalg.generic` carrying
an admitted round-to-even, integral clamp and signed integer conversion chain.
The existing matcher checks bounds, widths, uses and safe scale construction.
No approximation, reduction reassociation or numerical admission threshold is
introduced. Unknown scalar effects and unsupported bodies are barriers.
Added operations are bounded; the default budget is 4096.
Buffer generics, unknown tensor encodings and programs with CFG control flow
retain the unchanged route. Unchanged IR is checked structurally without
requiring the arithmetic checker to interpret that control flow.

The independent checker reconstructs the actual scalar output expressions and
checks unchanged surrounding interfaces, attributes, effects and dataflow.
The `automatic_host/report.json` artifact records source/output identities,
selected policy, library digest, actual operation and rounding-site counts,
refusals and checks. This is a conditional arithmetic check at tensor IR;
machine equivalence and hardware support remain separate obligations.
The receipt explicitly reports that this conditional floating-point check is
outside the existing covered integer transformation proof domains.

Use the ordinary optional-pass selector `automatic-host-numerics` to ablate
the rewrite in a paired experiment. No tuning is necessary to enable the
ordinary captured-policy route.
`MERLIN_AUTOMATIC_HOST_NUMERICS=0` also disables it. An explicit optional-pass
selection takes precedence over that environment fallback. Receipts include
the transformation, checker, matcher and pipeline source identities.

## Compact dense loops and operand retention

`schedule_compact_dense` extends `transforms.dense_planning`; it does not expand
the tensor or enumerate dynamic tiles. The existing bounded M/N search chooses
one largest legal K panel per candidate. A physical `k_limit(m, n)` narrows that
panel for separate operand stores, padding and available row windows before
ranking. Unknown costs use the documented traffic-first ranking, without cycle
estimates.

Inputs are a typed `DenseProblem`, `DenseResources`, assigned `DenseLoopStorage`
views and a `DenseLoopAdapter`. Each view declares physical backing, row byte
width, assigned base, usable capacity and alignment. The usable assignment must
already exclude packing scratch, DMA staging and other reservations. The three
live roles may use separate memories or disjoint windows in one backing.

The supported family requires explicit external-buffer disjointness, addressable
accumulator storage, synchronous completion and exact accumulator continuation.
It supports rectangular primitives, different element widths, both M/N outer
orders, peeled panel tails and primitive tails. The first K compute initializes
each output; subsequent computes continue its accumulator in source K order.
Bias seeds, gathers, asynchronous completion and unqualified numerical modes
retain their existing route or require a precise diagnostic.

The emitted program uses the existing `sched.ir.Kernel`, `Loop` and `Call`
objects. M/N output panels retain complete accumulators while successive K
panels load each operand once for its consumers. Partial panels are peeled;
tail dimensions and addresses remain parameterized expressions. Static call
sites are budgeted independently of executed call counts.

The adapter also selects a signed 32- or 64-bit loop index representation.
`CompactDenseResult.emit_c` uses that width through the existing C emitter.
Integer literals and select branches carry the same width, and conservative
intermediate-expression bounds reject possible arithmetic overflow without
expanding dynamic loops. This does not establish instruction argument widths,
physical address windows or pointer ownership; those remain target/runtime
obligations. Omitting the new width option on the existing emitter preserves
its historical emitted text.

`verify_compact_dense` projects actual calls, then uses bounded symbolic domains
to establish complete contraction/output coverage, valid tails and source
coordinates, initialization, source reduction order, physical residency and
absence of intervening overwrites. It checks padded high-water marks against
the current capacities and reconstructs executed commands and requested payload
bytes from actual operands and loop multiplicities. Unknown execution or
configuration effects refuse. No dynamic loop is unrolled by the checker.

The checker requires `z3-solver` and an explicit time budget. One bounded repair
may replace an unproved candidate with a simpler streaming loop and recheck its
actual output. Concrete invariant failures never trigger repair. Missing tools
or an unproved final alternative produce a diagnostic. If no candidate fits,
the scheduling entry returns `None` so the compiler can retain its legal
original route.

The adapter must independently decode actual instruction operands and establish
the selected instruction semantics. Reusing retained requests as a decoder is
not evidence. The shared receipt explicitly leaves final machine emission and
execution qualification unestablished. Tests compile the actual loop program
through the existing C emitter using neutral instruction implementations,
compare complete results to independent ordered arithmetic, and compare executed
commands and byte requests with reconstructed observations. These tests qualify
the shared mechanism's stated scope, not a hardware target.

Existing regular target integrations do not expose this new counted-loop
encoder/decoder contract. They keep their existing legal lowering. Integration
requires that public semantic boundary, actual addressing and tail legality,
completion/continuation qualification, and final emitted-effect checks. It does
not require changing a target's instruction implementation to fit Merlin.

## Executable schedule pipeline

`compile_schedule` connects the existing storage passes, effect checks and final
program checks through `PipelineSession`. Its finite straight-line family uses
the existing `sched.ir.Kernel`, without introducing another schedule IR or pass
manager. The session accepts explicit fingerprint and structural-verifier
callbacks for that representation; xDSL callers retain the original defaults.
Fingerprints, IR level, hardware, numerical policy, resource declarations,
execution contracts and shared-library source identity bind every analysis.
The callbacks establish structural custody, not numerical proof authority.

The default schedule sequence is `semantic_admission`, `storage`, `completion`,
`emission`, then `final_program`. Explicit ordering must contain all selected
passes exactly once and satisfy stages and prerequisites before a target
callback runs. Storage rewrites invalidate all earlier facts and produce newly
checked realization. Failed copy allocation and inserted spills call the same
session's bounded repair mechanism; both consume `ScheduleBudgets.max_repairs`.
Repair clears analyses before allocation and completion are recomputed.
Instruction changes invalidate existing emission and size facts even when a
pass omits an explicit invalidation declaration. Failed actions poison their
private session and return no compiled result.

Selecting `fusion_capabilities` inserts `fusion` between semantic admission and
storage. It constructs checked physical candidates privately; storage selects a
candidate only after allocation succeeds. Fusion allocation fallback consumes
the same repair budget as spills and copy fallback.

Replacing an analysis also invalidates its consumers transitively, even if the
owning IR is unchanged. Analyses produced together share invalidation: replacing
an encoded byte image cannot leave its old size or conformance result current.
Both explicit-order validation and runtime custody enforce these dependencies.

The target supplies a `ScheduleEncoding` with an encoder, actual-byte call
decoder, `ProgramDecoder`, numerical-policy identity and independent
qualification identity. Configuration and policy must match the selected
`StorageAdapter`. One actual decoded instruction corresponds to one typed call
in this family. The shared pass compares all actual decoded accesses and
completion behavior across both decoders and requires the complete numerical
and memory-effect sequence to match the verified schedule. The decoder cannot
substitute retained requests for actual operands. Independent qualification
must establish each decoder's interpretation, all state effects and executable
instruction semantics; the shared checks cannot establish that a lying decoder
describes its target correctly.

Final size, instruction count, address and branch checks run on the encoded
bytes after storage rewrites and spill insertion. Segmentation requires the
existing qualified continuation contract. Every declared external window is
conservatively treated as published, including inputs, aliases and carried
state. Each final segment is decoded again for numerical and memory conformance,
as well as the existing final-program checks. Receipts retain actual byte
hashes, final counts, requested transfer bytes, pass stamps, selected parameters
and shared source hashes. Timing remains unknown without paired timing evidence.

For an integration that supplies those public contracts:

```python
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, compile_schedule

result = compile_schedule(
    source_kernel, private_objects, selected_banks, storage_adapter,
    selected_encoding, selected_program_limits,
    external_windows=selected_runtime_windows,
    private_ownership=True,
    budgets=ScheduleBudgets(max_repairs=2),
)
checked_segments = result.program.segments
```

These declarations come from selected capabilities and the compiler's ownership
analysis, rather than inferred target names. Unsupported effects, counted loops,
asynchronous instructions, missing ownership and infeasible allocation produce
a diagnostic for the caller's existing legal lowering route. Counted dense
loops and asynchronous storage allocation retain their separately checked
interfaces; this finite encoded family does not silently claim their semantics.
Native C and byte-interpreter tests qualify shared mechanisms only. Existing
full-model target providers still need this public encoder/semantic-decoder
boundary before the schedule pipeline can enter their ordinary model lowering.
This entry is not evidence of that missing target integration or hardware
execution qualification.

## Private temporary storage

`allocate_temporary_storage` extends the existing semantic graph allocator with
an executable relocation pass over finite straight-line `sched.ir.Kernel`
programs. Each `TemporaryObject` declares one invocation-private backing object,
its virtual source interval, selected byte-addressed bank and alignment. All
typed views of that object share its backing and byte offsets. Separately named
banks can alias one physical memory; distinct physical memories retain their
own capacities. Existing `Reservation` intervals exclude packing scratch, DMA
staging and other unavailable storage across every alias of the backing.

The target's selected `StorageAdapter` decodes actual memory operands,
nonaddress semantics and executable completion instructions. It also rewrites
addresses into actual calls. Every local access must resolve to exactly one
private object, and external accesses need explicit fixed windows. Private
ownership and disjointness from external buffers are required. Complete initial
writes must precede reads, including reads through aliases; a read-modify-write
instruction cannot establish initialization by listing its write first.

Lifetimes extend through completion of actual asynchronous readers and writers.
A token annotation without an executable decoded wait is insufficient. Pending
conflicts, duplicate or missing tokens and a return before completion refuse.
The derived completion events reuse the existing allocation graph and its
independent checker; event positions are ordering facts, not estimated cycles.
Objects read and written by the same instruction remain simultaneously live
unless a different separately qualified transformation explicitly permits reuse.

The pass consumes selected addresses in the emitted calls. Its independent
checker derives assignments again from actual output operands, checks alias
agreement and current reservations/capacities, and compares actual nonaddress
semantics, access extents, external addresses and completion. Native C tests
execute baseline and relocated programs with complete byte comparisons and
independent command/traffic counters, including deferred reads completed by
actual waits. Storage footprint reduction does not imply a traffic reduction.

Infeasible allocation returns `None` for the compiler's existing legal route.
Unknown effects, loops, escaped or persistent storage, nonbyte address mappings
and unsupported completion produce a diagnostic. The entry does not perform
CPU register allocation or infer a target's physical mapping. Device spill
insertion requires the additional exact capabilities described below.
Existing targets need the public effect/address-rewrite adapter and
independent execution qualification before this pass can enter their ordinary
lowering. That integration remains incomplete.

## Private copy elimination

The ordinary `allocate_temporary_storage` entry defaults to
`eliminate_copies=True`. It identifies complete, synchronous, bit-preserving
private copies through the existing qualified `StorageAdapter`. A destination
that is only read may share its source's bytes when both typed address views
refer to the same physical backing and the source interval is unchanged until
the last destination read. Subrange offsets, other aliases, destination
alignment and every actual view's capacity remain constraints. Copy chains are
forwarded with work bounded by the original call and object budgets.

The rewritten calls are reprojected and allocation, liveness and completion are
recomputed. If extending the source lifetime makes the composition impossible
to allocate, the pass restores the original composition and replans it. A
decoder or invariant failure diagnoses rather than accepting an unchecked
rewrite. `eliminate_copies=False` selects the original fixed-operation family
for a paired comparison. Device spill repair keeps its insertion family fixed;
the outer allocator selects forwarding before that bounded repair.

`verify_storage_dataflow` independently reconstructs the original and final
byte definitions, permits omission only of qualified private copies, and
compares every retained operation's inputs and numerical semantics, physical
alias relations and complete external output and state windows. The receipt counts
eliminated copies, actual calls, and requested bytes from the decoded realized
program. It does not use the optimizer's list of removed calls as evidence.
The existing `verify_temporary_storage` checker retains its narrower role of
verifying relocation with the original operation count; it cannot certify
copy elimination. Native C tests execute both actual allocated programs on
identical input, compare all output bytes to an independent reference, preserve
unwritten output bytes, and count executed calls and byte requests.

This family does not eliminate packing or conversions unless the operation is
independently qualified as an exact byte copy. It does not infer asynchronous
handoffs, different-memory transfers, in-place epilogues, persistent ownership
or executable target support. Source/destination overwrites keep the copy.
Physical arithmetic fusion and target integration remain incomplete. Existing
targets require a public, independently qualified actual-effect decoder and
address rewriter before this mechanism can enter their normal lowering.
Traffic results from synthetic execution are not hardware latency results.

## Physical elementwise fusion

`compile_schedule(..., fusion_capabilities=selected_capabilities)` discovers
adjacent producer-consumer compositions from decoded effects and
`ElementwiseSemantics`. Each declaration contains actual typed scalar MLIR,
the element count, ordered stages, and an explicit byte order for every read
and write. Values are contiguous, with one corresponding scalar value per
element and no broadcast, hidden effects, volatile access or cross-element
dependency. All arguments of one element are read before its result is written.
Shifted or differently typed aliases are refused. Exact in-place aliases retain
that explicitly qualified execution rule.

This family supports total fixed-width integer arithmetic, constants, casts,
comparisons, selection and signed/unsigned min/max. It reuses the existing
scalar eligibility checks and admits additional exact standard scalar classes
under this new composition scope. Floating arithmetic, poison flags, unknown
operations, calls, reductions, side effects and unmodeled metadata remain
barriers. Existing frozen numerical proof domains and authoring grants are
unchanged. No approximation or new numerical policy is selected.

Merlin uses the actual completion-aware lifetime to require a newly initialized,
invocation-private intermediate with no other observer or writer. The complete
typed producer result must be consumed with the same byte layout. One selected
`ReadoutCapability` must apply every stage, and the returned instruction's
decoded readout identity must match it. Capabilities from different readouts
cannot be combined. The target-owned `FusionCapability.emit` either returns one
actual instruction or declines. Selection uses the first compatible qualified
capability in declaration order; it invents no cycle estimates.

`verify_fusion_step` reconstructs the collapsed pair from original and returned
calls without accepting a selected-pair witness. It checks all unaffected
effects, actual input and output byte intervals, explicit layouts, widths and
stage order. Typed scalar expression DAGs must match after substituting the
producer into the consumer. Widening, wrapping arithmetic, clamp order and
narrowing remain observable nodes; algebraic reassociation is not inferred.
The immutable graph cache and scalar work budgets bound compiler space by IR
size rather than element count.

The candidate's objects and lifetimes are then replanned through the ordinary
allocator. An incompatible physical view or insufficient capacity restores the
original composition and recomputes allocation. Scalar operation/source-byte
budgets and `ScheduleBudgets.max_fusions` can retain the original route. Actual
arithmetic mismatches diagnose; they never authorize an unchecked candidate.
If both routes fail allocation, the caller receives the existing precise
unavailable diagnostic. A receipt marked `allocation_fallback` describes a
discarded candidate, not eliminated traffic in the final program.

Allocation/spill verification binds the checked candidate to the realized
schedule; actual-byte decoding then checks the complete final instruction
effects and numerical declarations. Final receipts report actual commands,
encoded size and requested read/write bytes after all rewrites. Tests execute
the emitted byte program in native code and check complete outputs, input
state, intermediate signed wrap, clamp-before-narrowing and actual transfer
counters against an independent vector reference. Separate-memory and aliased
view models, different element widths/byte orders, tails, alignment fallback,
multiple consumers, extra operands, asynchronous barriers and mutation controls
exercise the shared mechanism. Negative controls expose both incorrect target
arithmetic and hidden temporary traffic despite plausible declarations.

Synthetic tests do not qualify a target. Existing providers must expose a
public, independently qualified scalar/effect projection, complete layouts and
the physical composition emitter, together with final actual-byte decoding.
The initial read-only target integrations do not provide these hooks. Physical
floating readouts, contraction-to-epilogue fusion, noncontiguous layouts,
cross-memory handoffs and asynchronous fusion remain incomplete. Traffic
reductions in these tests are not full-model latency results.

## Exact device register spills

The ordinary `allocate_temporary_storage` entry accepts selected `SpillClass`
capabilities and a `max_spills` repair budget. Each class declares its device
register bank, a separate addressable spill backing, exact byte width,
alignment, transfer emitter and independent qualification identity. It grants
no permission to replace the native toolchain's host CPU register allocation.

When allocation cannot fit, a bounded deterministic search selects unused gaps
in eligible register lifetimes, emits an exact save and reload, and rebases later
uses into a fresh private object. Unknown costs use the documented longest-gap
policy. Each insertion reconstructs actual effects and recomputes liveness,
allocation and completion; previous emission and program-size checks are
invalidated. Spill slots are allocated by the same allocator and can be reused
after their actual last read. Growth and repair budgets bound the search.

The emitted transfer must independently decode as `StorageEffects.exact_copy`:
one equal-sized read/write, no arithmetic or conversion, and actual synchronous
completion. This family supports synchronous straight-line device programs.
Unknown register classes, nonaddressable memory, missing exact transfers,
unsupported widths and asynchronous spill programs keep the existing route or
produce a diagnostic. No solver timeout or invariant failure becomes a legal
unchecked fallback.

`verify_device_spills` reconstructs interval byte provenance from the original
and actual final programs. It accepts no planned spill locations. It checks
every original operation and read in order, physical aliases, current capacity,
alignment and reservations, allowed exact copies, and complete external output
and state intervals. Provenance storage grows with written intervals, rather
than tensor element count. Receipts record observed commands and requested
bytes, with hardware timing and machine-emission qualification explicitly
unestablished. Native tests include a faulty copy implementation that the
independent numerical comparison rejects despite its declared copy effects.

Existing backends need independently qualified register-class and transfer
adapters before this mechanism can enter their ordinary target lowering. A
synthetic execution qualifies the shared mechanism's stated scope only.

## Final encoded programs

`finalize_program` checks actual encoded bytes through the selected target's
independently qualified `ProgramDecoder`. `ProgramLimits` declares instruction
alignment, byte and instruction budgets, memory windows and relative branch
encodings. Decoding must cover every byte exactly once. Every decoded access
must fit a declared window, and every branch must fit its declared encoded
reach and target an instruction boundary inside the program. Unknown structural
facts produce a diagnostic.

An oversized program can be segmented only with an explicit `Continuation`
contract: sequential completed launches, entry at zero, preserved memory spaces
and a launch budget. The caller supplies the original complete output intervals.
The shared algorithm preserves instruction bytes, keeps relative branch regions
closed, and checks future reads and published outputs before allowing volatile
storage to disappear at a launch boundary. It retains partial-overwrite values
conservatively. With branches or asynchronous operations it also retains future
reads conservatively rather than inferring path-specific liveness.

An asynchronous instruction prevents a boundary until an actual decoded drain.
Unknown completion and position-dependent instructions prevent segmentation.
Every resulting segment is decoded again: its actual normalized effects must
equal the original instruction effects and meet the final budgets. The receipt
records actual counts, sizes and hashes; it does not establish execution timing
or qualify a target decoder or runtime continuation contract. Unsupported cuts
refuse rather than inventing branch relaxation, state saves or launch support.

Neutral tests execute variable-width byte programs before and after segmentation
and cover a distinct fixed-word representation. Existing target integrations do
not supply this final decoder/continuation boundary automatically. Final machine
program checks therefore remain an explicit integration requirement for those
targets.

## Existing mechanisms and remaining integration requirements

| Concern | Existing mechanism | Requirement beyond the scope above |
| --- | --- | --- |
| Temporary storage | Alias/liveness analyses, source reuse, checked SSA allocation, LLVM arena binding and consumed private straight-line storage relocation | Target effect/rewrite adapters and ordinary lowering integration; general loop and persistent-storage allocation |
| Register pressure | Bounded exact device spill/reload insertion, reusable slots, fresh allocation/completion checks and independent actual byte provenance | Qualified device register/transfer adapters and ordinary target integration; CPU allocation stays with the native toolchain |
| Fusion and transfers | Readout composition admission and connected dispatch; exact private-copy forwarding; consumed total-integer elementwise physical composition with layout/numerical checks, allocation fallback and reconstructed final traffic | Floating/contraction epilogues, noncontiguous and asynchronous handoffs, and qualified full-model target integration |
| Reductions and attention | Ordered source loops, complete CPU reductions and explicit numerical policies | Shared multi-tile lowering with full domains, masks and intermediate storage; per-tile softmax is insufficient |
| Final program limits | Actual decoded instruction size, branch/address checks and closed-region segmentation through declared interfaces | Independently qualified target decoder and runtime continuation integration; ordinary targets retain their existing linked-image admission |
| Persistent state | Runtime state binding, immutable weight hashing/reuse and recurrent relinking | Qualified address windows, ownership and persistence for each device runtime |
| Whole-model coverage | Source routing/census and compiled CPU fallback machinery | Complete realized coverage and unchanged numerical gates for each selected full model |

An analysis, selected composition, metadata assignment or small executed kernel
does not establish these remaining integrations. Actual full outputs and carried
state need paired execution and independent references under unchanged policies.
Functional simulator instruction counts, traffic reductions, compilation cost
and qualified latency measurements are different results and must be reported
separately.
