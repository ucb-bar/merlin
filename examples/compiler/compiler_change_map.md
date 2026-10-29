# Compiler contribution map

Use this map with [the contributor guide](../../docs/guides/compiler_contributions.md)
and [the construction workflow](construction_rehearsal.md). It describes change
locations, not a mandatory pass order or a supplied target implementation.
The author starts with an empty target backend and the selected public library.
Only modules actually included in that selection are available to the author;
an API existing in this repository does not establish that it was staged or used.

The experiment asks whether shared representations and transformations reduce
construction effort and improve executable code. Calling an API, emitting a
receipt or importing a dialect cannot establish that result. Follow each useful
product through its consumer to the emitted program and numerical execution.

## Where the author may need to change the compiler

| Compiler layer | Existing starting point | Changes the author may need | How to demonstrate usefulness |
| --- | --- | --- | --- |
| Frontend and semantic admission | `xdsl_dialects.interface`, `compile.scheduling.dense_source`, `transforms.dense_ir` | Support additional source recurrences, indexing maps, initial values, convolution regions and layouts. Preserve signedness, accumulation, rounding and saturation. Replace duplicated pattern checks with a reusable reader when appropriate. | The same semantic route accepts independent shapes and rejects a near-miss with different semantics; its result supplies dimensions and operands to real lowering. |
| Graph and region construction | `compile.fusion_discovery`, `compile.fusion_workflow`, `compile.connected_regions` | Discover producer–consumer and shared-input regions; represent all live-ins/live-outs, aliases and other consumers; support mixed dense/pointwise regions beyond the existing narrow pointwise route. | Actual fused/unfused alternatives compile, compare every externally visible output and retain a legal unfused route. Merely enumerating candidates is discovery only. |
| Numeric and layout contracts | Selected numerical contracts, `xdsl_dialects.lowering.output_fusion`, shared host arithmetic | Express layout conversions, exact epilogue order and device-supported arithmetic; separate weight preparation from per-inference work. Add generic missing conversions in Merlin and target legality in the backend. | Tail/epilogue checks preserve exact semantics; measured preparation and recurring costs remain separate. A change in quantization is not an optimization under the old contract. |
| Pass registration and policy | `transforms.pipeline_state`, `transforms.pipeline_policy` | Register actual actions and preconditions, analysis dependencies and invalidations; select or branch pass sequences using current source shapes, graph properties and hardware resources. Add new passes when existing ones do not express the transformation. | A chosen branch produces the artifact later emitted. Each alternative has its own valid analyses; a no-op is recorded as such. Compare pass time and target time separately. |
| Logical scheduling and compact loops | `sched.ir`, `compile.scheduling.dense_source`, `compact_dense`, `block_schedule`, `llvm_program` | Connect source semantics to Kernel/Loop/Call or supported counted IR; choose tiling, loop order and reusable policies. Extend a representation when it cannot preserve required dependencies or compact iteration. | Returned schedule/loop products reach code generation. Large shapes retain compact loops instead of expanding into one host object per device command. |
| Physical scheduling and effects | `compile.scheduling.schedule_pipeline`, configuration and command-compaction transforms | Supply target command effects and decoding adapters; distinguish operand consumption, result availability, queue release and output completion. Expose legal overlap and barriers without treating instruction counts as a timeline. | The transformed physical program is emitted, satisfies actual effects and hazards, and improves a matched complete-call measurement. Finite scheduling APIs must not silently expand large counted loops. |
| Storage, transfers and preparation | `compile.scheduling.temporary_storage`, `device_spills`, `panel_transfers`, `copy_pipeline` | Model live ranges, capacity, bank/layout constraints, ping-pong slots, spills and reusable prepared buffers; implement the target storage/transfer adapter. | Allocated/transferred objects are those used by emitted commands; outputs remain correct across reuse and tails. Compare actual traffic and whole-region cost, not planned byte savings alone. |
| ISA dialect and instruction lowering | `targetgen.isa_dialect_gen`, generated typed operations, resource allocation utilities | Author operand packing, memory effects, instruction selection, pointer/address spaces, target ABI and completion. Extend generated definitions when the declared ISA subset is insufficient. | Encodings agree with independent public declarations; real emitted commands execute. Generation supplies syntax/encoding, not a legal schedule or mathematical semantics. |
| Host code generation and execution | Shared host arithmetic/building APIs, logical pointer ABI, `compile.host_lane`, `compile.owned_host_model` | Compose device calls with shared CPU conversion/epilogue code, compact dispatch, preparation caches and synchronization. Implement target-specific launch glue while retaining reusable host machinery in Merlin. | Linked complete programs compare every output and account for host preparation, dispatch, waiting and boundaries. Isolated accelerator timing does not prove whole-chain savings. |
| Model assembly and lifetime across calls | `model_check.chain`, `model_check.chain_image_plan`, connected-region/runtime interfaces | Connect mixed stages, preserve intermediates safely, pass distinct stream frames and recurrent state, support complete model outputs. Compiler-side composition and trusted reference execution remain separate. | Full chain results match independently produced references. Correct isolated stages or summing their cycles is insufficient. |

All module names above are under `merlin`. Existing implementations have narrower
domains than the entire row: for example, finite adjacent synchronous fusion is
not arbitrary graph fusion, and a logical shared-input schedule is not a target
instruction image. Preserve precise unsupported diagnostics and extend the layer
that lacks the capability rather than substituting success-looking output.

## Ownership and authority

The author may change its new backend and its explicitly editable generic
compiler selection. Generic transformations, graph policy, host arithmetic,
packing, ownership and orchestration belong in Merlin. Target instruction
definitions, device layouts, physical policies, effects adapters and target ABI
implementation belong in the newly authored backend. A mixed change uses an
explicit interface between those owners. A prototype should not leave two
diverging implementations of the same generic algorithm.

The author does not change trusted numerical references, graders, RTL facts,
hardware selections or timing coefficients to improve its score. A missing
export or trusted-tool defect is an operator issue, recorded for a new source
selection. Independently developed simulator corrections are separately checked
against public evidence and must not silently alter a running experiment.

No prerequisite backend code is added to the persistent target repository for a
fresh run. Generate mechanical definitions inside the author workspace from
public declaration data. Neither private optimized programs nor their scores
select the author's transformations.

## A small, useful record for each attempted change

Keep these fields with the immutable compiler snapshot, without putting full
traces into every agent context:

1. **Hypothesis and owner:** the missing capability or measured bottleneck, its
   compiler layer, files changed and why an existing abstraction is insufficient.
2. **Semantic domain:** operation/shape/layout/numerical/resource predicates;
   legality premises and the unsupported or baseline route.
3. **Consumed product:** API/pass name, source and output identities, and the
   next consumer. Link the actual IR, schedule or allocation to the emitted ELF.
4. **Cost and feedback:** author time/tokens when available, compiler action time,
   relevant numerical checks, timing model identity/coverage, complete-call cycles
   and resource/roofline signals. Unknown metrics remain unknown.
5. **Disposition:** keep, reject, no-op or unresolved; what was learned and which
   affected cases need rechecking. Preserve successful independent work in the
   same compiler while immutable evaluations run.

Evidence for the research premise needs comparisons: one library-backed change
with its consumed product versus the retained baseline, under matching semantics
and measurement scope. If integrating an abstraction costs more than it saves,
record that result and the specific representation or adapter mismatch. A single
successful tiny execution cannot establish faster construction, better generality
or a performance advantage for the full compiler.

## Supported composition boundaries

`DenseSourceResult.emit_llvm` uses the existing compact program and the caller's
LLVM builder; it does not introduce a second scheduling algorithm. The
[LLVM integration example](source_dense_llvm.md) documents the callback boundary.
The target adapter still owns instruction implementation and completion.

`compile_connected_region` and `ConnectedProgram.ordered_invocations()` retain
source-owned bindings across checked dense products and pointwise images.
[Connected host code generation](connected_host.md) emits an unfused synchronous
sequence with explicit buffer and stage contracts. This does not establish
cross-stage device fusion, target ABI qualification or complete-model execution.

`periodic_effects.check_drained_period` checks bounded storage/completion behavior
for independently drained blocks while preserving counted emission. It does not
prove arbitrary asynchronous recurrence or the complete numerical computation.
These narrower contracts should be extended only with corresponding semantics
and validation, rather than treated as complete target capabilities.
