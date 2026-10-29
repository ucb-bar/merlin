# Bounded fresh-compiler construction rehearsal

Purpose: discover whether the public abstractions shorten the path from an empty
backend to a correct, measurable binary, and identify the first concrete missing
abstraction. This is a bounded rehearsal, not completion of the 250-case campaign.
A negative result with the exact failing boundary is useful. Do not import a
retained target compiler or private reference kernel, and do not create a backend
outside the ordinary fresh author workspace.

The [compiler change map](compiler_change_map.md) covers changes beyond passes:
frontend semantics, graph regions, numerical/layout contracts, scheduling,
allocation, instruction lowering, host execution and model assembly. Use it to
locate the owner of a missing capability and record actual product consumption.

## Same mechanism offline and online

Use the selected public library, software/target/numerical declarations, corpus,
ISA declaration data and `construction_actions.py` staged by the ordinary session.
The action client supports ordinary commands and `agent_selfcheck.py` requests;
there is no operator-only shortcut to compilation or grading. Use explicit local
dependencies and a bounded number of concurrent actions. Submit independent tests
as soon as their immutable snapshot exists; do not stop authoring while they run.
Request/ELF/snapshot identities must remain attached to feedback.

The operator's public command declaration can save mechanical instruction work:
call `isa_spec.resolve_isa` and `isa_dialect_gen.generate_package` on that declared
data inside the fresh workspace. Generated command types and encoding are not
mathematical semantics, effect knowledge, payload packing or a target schedule.
Unknown effects remain unknown until the author supplies a justified reusable
adapter. Encoding checks alone do not satisfy the first executable gate.

## Small existing public case set

Shapes below use **M × N × K**. The companion `construction_rehearsal_cases.json`
records exact public capsule/source hashes and provenance. IDs are existing suite
members; this document neither changes their required tiers nor creates tests.

| Priority | Existing public capsule | Shape and purpose | Required engine coverage |
| --- | --- | --- | --- |
| 1 | `SY_contraction_i8_aligned` | 16 × 16 × 32, i8 inputs/i32 output. Small first target execution; resident weight, reduction and full output completion. | Core: Spike and gSIM, plus declared cheap checks. |
| 2 | `G_matmul_m32k32n32_raw` | 32³, same numerical operation and generalized compiler path; model-derived device-group fixture. | Core: Spike and gSIM. |
| 2 | `USER_matmul_m512k512n512_raw` | 512³, same raw i32 semantics at scale; explicit public compiler test. Keep loops compact; expose host issue/transfer costs. | Scalability: Spike only. |
| 3 | `SY_contraction_i8_partial` | 16 × 15 × 31, same operation with reduction/output tails. Use when an affected transformation changes bounds, storage or scheduling. | Core: Spike and gSIM. |
| 3 | `SY_group_bias_add_acc_scale_relu_int8` | 16 × 16 × 32 plus i32 bias, declared scale, ReLU and i8 output. Exact epilogue order and rounding; connected contraction/output chain. | Core: Spike and gSIM. |
| Alternative epilogue | `PF00_fused_matmul_bias_m16k16n16` | 16³ plus bias to i8; existing public fusion group. Use if this admitted source route becomes executable first. | Scalability: Spike only. |

The primary trajectory is first tiny executable → medium and large same path →
one controlled optimization comparison. Tail and epilogue cases cover the changes
that actually arise; do not turn this rehearsal into a broad sweep. A Spike-only
screen of a core case is progress, not a complete core pass. Large cases never
acquire a gSIM requirement here. Preserve native evidence reuse and relevance
checks rather than repeatedly submitting unchanged work.

The epilogue case is not a complete connected TinyLlama layer or a full ResNet
block. The existing `JC_tiny_llama_prepared_connected` source has several mixed
integer/float outputs and a broader host/accelerator ABI; do not quietly relabel
this smaller epilogue as that model-region coverage.

## Construction and consumption checklist

| Stage | Existing shared API to consider | Evidence that it was actually used |
| --- | --- | --- |
| Read typed source | `scheduling.dense_source.read_source_dense`, `read_zero_seeded_integer_matmul`; selected numerical and scheduling contracts | Source-derived dimensions, operand identity and exact recurrence; no capsule-name dispatch. Record unsupported/unknown reasons. |
| General dense construction | `dense_source.schedule_source_dense`, `compact_dense.schedule_compact_dense`, `block_schedule.schedule_contraction` | Actual returned Kernel/counted IR consumed by the backend, with selected geometry/resources and tails. Small and large cases use the same semantic route. |
| Compact real emission | `compact_dense.emit_compact_dense`, `DenseSourceResult.emit_llvm` when selected, `llvm_program.emit_index_program` | Their emitted program enters the candidate's ordinary target artifact, linking and execution. The LLVM callback supplies target instructions; shared loops retain tails, accumulation and final stores. Do not flatten a large loop into millions of Python calls. |
| Physical finite schedule | `schedule_pipeline.compile_schedule` with the author's justified decoder/effect/storage/encoding adapters | Final image and remaining objects enter ordinary emission, allocation and runtime. Keep its schedule report and actual final bytes; imports and receipts alone are insufficient. Use this only for its supported finite representation. |
| Mechanical instruction bridge | Generated typed ops; `resource_allocation.allocate_instruction_resources`, `emit_resource_allocation`, `validate_allocated_ir` where applicable | Explicit allocation maps SSA values to encoding resources. Consume returned allocated IR; generated raw encodings do not prove legal memory effects. |
| Shared pass composition | `pipeline_state.PipelineSession`, `PassSpec`; `pipeline_policy.RegisteredPass`, `PipelineVariant`, `evaluate_variants` | One real transformed artifact per legal alternative reaches emission. Use a common `prefix` once where valid; IR-bound analyses are rebuilt for each clone. Never share analysis references across branches. |
| Configuration/command opportunities | `configuration.eliminate_redundant_configuration`, `configuration_loops.hoist_loop_configuration`, `command_compaction.compact_command_stream` | Justified state/descriptor contracts, an actual before/after stream and consumed rewritten output. Unknown clobbers remain barriers. |
| Movement/storage opportunities | `panel_transfers.plan_packed_panels`, `temporary_storage.allocate_temporary_storage`, `device_spills.allocate_with_spills`, `copy_pipeline.copy_pipeline_motion` | Real strides, bounds, ownership, lifetime and completion constraints; decode actual encoded transfers. Double buffering needs nonoverlapping live slots, not a flag. |
| Epilogue and connected exploration | `output_fusion.decisions`, `fusion_workflow.discover` / `realize_candidate` / `evaluate_fusions`; `shared_input_schedule.schedule_shared_input` where actual inputs are shared | Preserve every visible output and exact arithmetic. Emit realized products and compare complete scopes against the unfused baseline. The generic connected bridge supports ordered pointwise stages; mixed dense/pointwise runtime composition still needs the author's reusable ABI work. |
| Host boundary | Shared host tensor/epilogue/requantization codegen and the public logical-pointer ABI | Correct argument order, buffer lifetimes, output completion and all-output comparison. Do not duplicate generic host arithmetic in the target dialect. |

These are options, not a required pass order or a suggested tile configuration.
Choose from actual selected exports; develop missing general adapters or improve
an editable pass when the source/contract exposes a real gap. Retain unknown and
unsupported diagnostics instead of generating a success-looking stub.

## Exactly one optimization change, measurable end to end

1. Retain baseline **A** as an immutable compiler snapshot with a numerically
   successful target ELF. Start the matching medium/large case as soon as possible.
2. Inspect the baseline's actual report: completed-call cycles, timing coverage,
   roofline bound and its assumptions, bytes moved, command/configuration counts,
   resource/stall attribution and host preparation/dispatch/completion costs when
   available. Unknown fields stay unknown. Host `host_pass_profile` times measure
   compiler actions/framework overhead, not target execution cycles.
3. Choose **one** applicable transformation, policy parameter or reusable pass
   change from that evidence. State a falsifiable expectation and the legality
   premises. Do not prescribe a known tile, schedule or reference performance.
4. Create **B** from A with only that change. Keep inputs, numerical contracts,
   hardware/model pins, measurement scope, host preparation and cache assumptions
   fixed. Check the emitted artifact actually differs where expected. If it is
   identical, record a no-op rather than claiming an optimization.
5. Submit the affected numerical/engine checks through the ordinary broker, and
   record A/B results joined to their exact ELF and source hashes. Newly changed
   core output bytes do not inherit an old gSIM certificate. A shared failure
   gets one repair at its owning boundary rather than many identical retries.
6. Compare only matching scopes. Use the selected tuned model if qualified;
   otherwise label provisional estimates and coverage limitations explicitly.
   Missing/UNKNOWN cycle estimates are not wins. Do not fit from private scores.
   A lower provisional estimate with correct outputs is promising evidence,
   not verified physical speedup or a model-accuracy claim.
7. Retain B only when the evidence supports it; otherwise preserve A and the
   failed hypothesis. The reusable compiler change, not a per-capsule answer,
   becomes the next starting point.

Use at most A+B for this initial comparison (the generic evaluator's larger
bound is not a requirement to fill it). Independent agents may own frontend,
encoding adapter, shared-pass integration and output or runtime work on the same
compiler with clear interfaces; integrate their changes before naming a baseline.
Continue local construction while immutable checks run. Do not spend the bounded
window repeatedly rediscovering APIs or grading superseded snapshots.

## Minimum end-of-rehearsal record

Report authored files and their snapshot, selected input/library/model identities,
requests and emitted ELF identities; completed numerical executions and exact
required-tier passes separately; source-to-pass-to-artifact consumption; the A/B
change, matching measurements and limitations; time spent authoring, emission,
linking, queueing, Spike, gSIM and feedback processing when observed. Identify the
first unresolved compiler/tooling boundary with its failing artifact. An external
rehearsal time limit is an interruption, not formal campaign completion.
