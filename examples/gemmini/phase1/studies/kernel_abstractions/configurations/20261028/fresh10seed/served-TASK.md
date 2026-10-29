# Phase 1: Gemmini compiler for complete models

Build one compiler in `submission/`. Fresh runs start with an empty target backend; declared continuations preserve the authored compiler. The required programs are:

| Model | Programs |
| --- | --- |
| ResNet50 | `model` |
| SmolVLA | `prefix_encode`, `flow_denoise`, `action_decode` |
| TinyLlama | `prefill`, `decode` |

All six programs must compile and link. During Phase 1, numerical checks require every device kernel
to replay bit-exactly on Spike, including sealed inputs and sealed shape variants, a fixed-denominator
offload census, and separately tracked physical obligations. Development defaults to Spike;
L3 gSIM certification of the original correctness cases runs at the end, best-effort;
performance uses the Spike cycle model unless --sim gsim is asked; separate selective
calibration witnesses. Full-model Spike development checks are available throughout
the remaining campaign in this execution mode;
per-kernel model gSIM replay is not requested.
`qa/verdict.json` reports your
visible-input results. Sealed-input results appear only in the official grade.

The frozen `task/PHASE2_TARGET.md` describes the planned FireSim deployment
target and its relationship to this release's selected RTL facts. Read the
selected facts and contract for the exact Phase 1 compilation target; retain
their identities when reporting any timing observation.

## Deliverable

Write a self-contained `submission/` with `manifest.yaml`, compiler sources, the four required
parse/lower/emit/LLVM entrypoints defined by the admitted contract, and `REPORT.md` stating
measured results and limitations. The model-check contract uses `numerics.approximations` in the
manifest for declared approximate kernels. Undeclared device kernels must match the integer
reference exactly. Declared approximate kernels and host f32 work must meet the contract's
error policy.

## Available tools

- `python agent_selfcheck.py --submission submission --capsules A,B`: focused public capsule checks.
  Without `--sim`, or with `--sim required`/`--sim auto`, this joint campaign selects Spike L0/L1/L2 development checks; this does not certify unrun physical tiers. Explicit `--sim gsim` accepts one named required original correctness capsule under its separate correctness budget, or one calibration witness subject to question review. Required correctness does not consume calibration permissions or qualify a fit. When every correctness gSIM slot is busy the request waits for one in arrival order (its progress says `queued for a gSIM slot (N ahead)`); collect it later with `--attach`. Unlisted cases, multi-capsule gSIM requests and scalability gSIM/L3 requests are refused.
  Add `--wait-budget 5` to get a request id back at once, keep working, and collect the verdict with
  `--attach ID`; `--list` shows your requests. Several requests may be in flight. Put every capsule an
  edit affects in one request: its capsules run in parallel and per-request overhead dominates.
  Check edits by naming the affected capsules; `--capsules changed` rebuilds every capsule (a slow
  sweep for occasional regression checks); `--capsules failing` re-checks your last check's failures.
- `--submission variants/NAME`: self-check an alternative kept in `variants/NAME/` (same layout as
  `submission/`) beside it. Variant results are labelled and never graded officially; only
  `submission/` counts, so copy the winner there.
- `python simjob.py --help`: asynchronous public capsule jobs (`--sim required` also selects Spike in this joint campaign).
- Optional cheap code-size feedback for applicable public accelerator members:
  `python agent_selfcheck.py --compile-only --scale-probe --capsules PUBLIC_ID --timeout 120`.
  Replace `PUBLIC_ID` with one or two declared public member IDs. The harness derives source-only
  scale/tail cases from those members and reports lowering failures, timeouts and emitted LLVM growth.
  Include public convolution and output-readout families when applicable; small matmuls alone do not
  test their compilation cost. This mode allocates no tensor values and reads no private model shapes.
- For bounded build feedback use `python agent_selfcheck.py --compile-only --capsules PUBLIC_ID --timeout 120` for bounded
  object compilation, linking and ABI/host-work audits. Inspect the reported failure stage before
  retrying. A successful emission or link grants no numerical, offload, trace or full-model credit.
- `python await_verdict.py --help`: wait for a new harness verdict.
- `python agent_model_check.py check --model MODEL --out REPORT.json`: visible model check through the broker.
- `python agent_model_check.py replay-kernel --model MODEL --program PROGRAM`
  `--group SOURCE_REGION_ID --engine spike`: visible kernel replay through the broker.
  A pending reply includes a `wait --job-id` command.

Selected treatment tools:
- `merlin_infra`: Answer-free support modules imported transitively by the xDSL, CCA and RTL-profile authoring tools. Not an oracle, grader or answer surface — granting them widens no moat, and withholding one disables the advertised tool that imports it.
- `xdsl_kit`: The xDSL authoring kit: dialect synthesis, the generators, the dialect definitions, the interface emitters and the out-of-tree starter kit.
- `cca_spine`: The Common-Compute-Abstraction spine — the where/how of modifying a compiler: extract a CCA, diff two, check the CCA<->action bijection, walk the escalation ladder, author a microkernel.
- `isa_tools`: Derived assembler, disassembler, static linter and lite debugger for the target's own ISA. Oracle-free: it encodes the syntax you chose and inspects the words you emitted.
- `cca_tools`: The two mandated CCA introspection calls — check_bijection and escalation_ladder — reachable as plain imports inside the sandbox. Oracle-free: public schema against public routes.
- `rtl_generators`: The CIRCT RTL-fact generators: derive an ISA encoder module, a distilled RTL digest and a numeric-shape checker from the target's elaborated RTL rather than from its documentation.
- `rtl_facts`: The facts already extracted from THIS target's RTL — the generators' output, granted directly.

## Sandbox and enforced rules

- Use only the frozen inputs and tools granted in this workspace. Do not access hidden inputs,
  goldens, prior backend submissions, or operator-private paths.
- `integrity_exempt: false`; the compiler must be agent-authored and use the selected xDSL/Python
  language and CLI contract. Do not import Merlin's reference or simulator implementation into
  the submission, or substitute C compute kernels for emitted device code.
- Keep `manifest.yaml` and the four entrypoints valid. The harness checks package integrity,
  public development capsules on Spike, required original gSIM correctness, separately admitted
  calibration witnesses, exact model-kernel replays,
  and the offload census. Unrun declared physical tiers remain pending.
  A pending or unavailable check is not a pass.
- The harness grades immutable snapshots. You may continue editing while a self-check or grade runs.
  A verdict names the snapshot it graded; check its source digest before treating it as current.

## Final status

Lead with a table of all six programs: model, program, compile/link, numerics, and census.
Report capsule RTL status separately. Then name the submission path, test artifacts, and remaining failures. Report only
measured results.


## Runtime scope (generated for this launch; authoritative)
- Core public/dev capsules: **109**, derived from the descriptor's graded roots, label filter, and exclusions. Completion is non-vacuous only when every required member passes.
- Held-out capsules: **14**, derived from the descriptor's hidden roots; their contents remain sealed.
- Core and separately selected scalability cases are distinct public rosters; the scalability section gives its additional count and source index. Use the default core suite for core IDs and `--suite scalability` for scalability IDs.
- **0 core capsules are final-only** (already included in the core count): none. They remain pending in development grades. Full-model execution, including small model capsules, is gated to the selected final Spike window; full target program obligations are separate from the capsule count.
- Automatic grading uses a rolling unresolved queue with bounded snapshot admission time. The optional 1–16-ID impact hint sets priorities; it is not an author access limit. Name any admitted public development IDs in focused requests through the appropriate suite; hidden members remain inaccessible and final-only/model and engine gates still apply.
- Active sandbox: **`bwrap`** (deny-by-default bwrap; allowed inputs are the frozen run snapshot and answer surfaces are masked).
- The harness re-grades a snapshot of your workspace on its own schedule and refreshes `qa/verdict.json` underneath you; you are NOT relaunched between grades. To wait for the next one instead of checking repeatedly, run **`python await_verdict.py`** — it blocks until a new grade lands and prints its score and failing capsules (`--timeout <s>` to bound the wait; a timeout is reported, not an error). Polling `qa/verdict.json` in a loop costs you a turn every time you look.
- If an older bundled document states a fixed capsule count, a different isolation mode, or that you are relaunched each round, this launch-generated block wins.


## Common compiler infrastructure and preservation policy

The versioned common substrate is available in every experiment level: `merlin.transforms.certificates`, `merlin.transforms.normalization`, `merlin.transforms.loops`, `merlin.transforms.planning`, `merlin.transforms.validation`, `merlin.transforms.pipeline`, `merlin.transforms.client`, `merlin.verify.lean_contract`, `merlin.transforms.pure_scalar`, `merlin.transforms.scalar_witness`, `merlin.transforms.unrolling`, `merlin.transforms.analysis_types`, `merlin.transforms.analysis_memory`, `merlin.transforms.analysis_liveness`, `merlin.transforms.analyses`, `merlin.transforms.resource_allocation`, `merlin.semantic_compiler.allocation_types`, `merlin.semantic_compiler.allocate`, `merlin.semantic_compiler.rules`, `merlin.semantic_compiler.model`, `merlin.transforms.dense_planning`, `merlin.transforms.dense_ir`, `merlin.compile.scheduling.block_schedule`, `merlin.compile.scheduling.shared_input_verification`, `merlin.compile.scheduling.shared_input_schedule`, `merlin.compile.scheduling.shared_input_groups`, `merlin.compile.scheduling.panel_transfers`, `merlin.compile.scheduling.copy_pipeline`, `merlin.compile.scheduling.configuration_loops`, `merlin.compile.scheduling.command_compaction`, `merlin.compile.scheduling.compact_dense`, `merlin.compile.scheduling.compact_loaded_sum`, `merlin.compile.scheduling.spatial_gather`, `merlin.compile.scheduling.dense_source`, `merlin.compile.scheduling.configuration`, `merlin.compile.scheduling.configuration_ir`, `merlin.compile.scheduling.device_spills`, `merlin.compile.scheduling.fusion`, `merlin.compile.scheduling.fusion_semantics`, `merlin.compile.scheduling.library_sources`, `merlin.compile.scheduling.llvm_program`, `merlin.compile.scheduling.pointwise_source`, `merlin.compile.scheduling.program_limits`, `merlin.compile.scheduling.row_softmax`, `merlin.compile.scheduling.schedule_pipeline`, `merlin.compile.scheduling.staged_dense`, `merlin.compile.scheduling.selected_contracts`, `merlin.compile.scheduling.spill_conformance`, `merlin.compile.scheduling.temporary_storage`, `merlin.compile.scheduling.window_transfer`, `merlin.llvmlower.cpu_softmax`, `merlin.sched.ir.expr`, `merlin.sched.ir.integer_domain`, `merlin.sched.ir.kernel`, `merlin.sched.ir.parse`, `merlin.sched.isa.types`, `merlin.targetgen.isa_dialect_gen`, `merlin.targetgen.isa_spec`, `merlin.transforms.pipeline_state`, `merlin.compile.scheduling.periodic_dense`, `merlin.compile.scheduling.spatial_producer`, `merlin.compile.scheduling.spatial_readout`, `merlin.llvmlower.spatial_scratch_abi`, `merlin.llvmlower.fresh_tensor_writer`, `merlin.llvmlower.private_workspace_pool`, `merlin.llvmlower.borrowed_effects`, `merlin.compile.scheduling.schedule_ir`, `merlin.xdsl_dialects.schedule`, `merlin.compile.scheduling.loop_schedule`, `merlin.compile.scheduling.dense_panel_pipeline`, `merlin.compile.scheduling.scf_commands`, `merlin.compile.scheduling.configuration_contexts`, `merlin.compile.scheduling.configuration_loop_cleanup`, `merlin.compile.scheduling.dense_panel_readout`, `merlin.compile.scheduling.tile_load_coalescing`, `merlin.compile.scheduling.panel_tile_loads`, `merlin.compile.scheduling.dense_segmented_pipeline`, `merlin.compile.scheduling.dense_array_retention`, `merlin.compile.scheduling.api_catalog`, `merlin.compile.scheduling.rolling_window`, `merlin.compile.scheduling.rolling_spatial`, `merlin.xdsl_dialects.rolling_window`, `merlin.compile.scheduling.spatial_readout_cache`, `merlin.compile.scheduling.dense_readout_overlap`, `merlin.llvmlower.host_builder`, `merlin.compile.scheduling.projection_views`, `merlin.transforms.tensor_views`, `merlin.llvmlower.source_region`, `merlin.llvmlower.source_region_host`, `merlin.compile.composition_catalog`, `merlin.compile.scheduling.dense_ready_readout`, `merlin.sched.ir.events`, `merlin.compile.scheduling.scf_carried_events`, `merlin.sched.primitives.event_loops`, `merlin.llvmlower.passes_xdsl`, `merlin.xdsl_dialects.lowering.outline`, `merlin.sched.primitives.loop_handles`, `merlin.compile.scheduling.completion_streams`, `merlin.compile.scheduling.configuration_motion`, `merlin.compile.scheduling.spatial_max_stripes`, `merlin.compile.logical_abi`, `merlin.compile.scheduling.convolution_exploration`, `merlin.compile.scheduling.dense_rhs_residency`, `merlin.transforms.pipeline_policy`, `merlin.transforms.pass_exploration`, `merlin.transforms.evaluated_exploration`, `merlin.transforms.shape_policy`, `merlin.compile.scheduling.dense_panel_epochs`, `merlin.compile.scheduling.pointwise_convolution`, `merlin.compile.scheduling.batched_dense`, `merlin.compile.scheduling.convolution_output_groups`, `merlin.compile.scheduling.loaded_sum_epochs`, `merlin.compile.tuning_knobs`, `merlin.transforms.tuning_selection`, `merlin.llvmlower.late_quant_rne`, `merlin.llvmlower.source_expression_interval`, `merlin.llvmlower.source_expression_interval_llvm`, `merlin.llvmlower.bounded_rne_word_cells`, `merlin.llvmlower.scaled_integer_finite`, `merlin.llvmlower.scaled_integer_finite_llvm`, `merlin.compile.scheduling.projection_dense`, `merlin.compile.scheduling.rolling_lookahead`, `merlin.compile.scheduling.operand_slots`, `merlin.compile.scheduling.dense_shared_rhs`, `merlin.compile.scheduling.horizontal_convolution`, `merlin.llvmlower.repeated_pointwise`, `merlin.llvmlower.contraction_row_slice`, `merlin.llvmlower.weight_prepack_runtime`, `merlin.transforms.dense_row_batching`, `merlin.transforms.shared_rhs_views`, `merlin.compile.scheduling.direct_source`, `merlin.compile.scheduling.source_tokens`.
Use its documented generic transformations and analyses. Their automatic verification is common infrastructure, not an additional level treatment.
Use `merlin.transforms.client.apply_transform(workspace, source_ir, options)` through the original host broker for verified common transformations. Submit only IR and documented parameters; the host selects admitted semantics, formal coverage and the frozen proof toolchain.
Within the admitted formal domain, successful Lean checking is required. Missing tools, unknown obligations, failed proofs and timeouts cannot discharge covered obligations. Outside that domain, independent correctness checks remain required and formal coverage must be reported explicitly. The admitted semantics and coverage policy are immutable.
Tool presence describes an interface, not a successful runtime qualification. Report actual availability and proof outcomes separately from numerical and performance results.
Use `merlin.transforms.client.check_transform(workspace, before_ir, after_ir, entry="...")` to submit actual compiler rewrites for covered host checking. The original host selects software/ISA semantics and effect premises from sealed inputs. A result establishes only its reported formal scope; hardware, numerical execution and performance need independent evidence.
The original command-buffer schema accepts an optional source-bound `host_optimization_request`. Only bounded documented choices are permitted; the original reviewed software profile independently supplies any required host semantic and runtime permissions. Omission of those permissions leaves the request unavailable. Numerical, instruction-policy and complete execution gates still apply.
During original compiler entrypoint calls, `MERLIN_COMMON_WORKSPACE` names the isolated request workspace for the same public transformation/check client. Replies establish only their reported scope.
Shared dense scheduling is available as `merlin.transforms.dense_planning`: provide explicit primitive geometry, available storage after reservations, supported block limits and a target static-command price. The planner returns a bounded choice, not a numerical or execution proof. Emit and verify its `merlin.transforms.dense_ir.DenseScheduleOp` against the realized commands. Version 4 retains version 3's formal coverage policy.
Shared `merlin.compile.scheduling.block_schedule` provides checked synchronous matmul/convolution schedules. Supply explicit fact-derived Geometry; check_residency verifies operand contents, ordered reductions, accumulator lifetimes and complete output stores. It does not establish asynchronous completion or external view aliasing. Project emitted operations explicitly; load counts alone cannot establish lifetime conformance. Version 5 keeps version 4's automatic checkers and formal coverage policy.
Every V24 treatment receives the same shared scheduling SDK: tiling and reuse, temporary storage, supported fusion/copy elimination, persistent configuration, device spills, complete row reductions and final program checks. Phase 2 may edit reusable shape/resource-based policy code to select legal alternatives. Keep numerical policy, hardware limits, qualification identities and verification gates fixed; checker work limits are not performance objectives. Check actual decoded output and complete outputs/state before performance comparisons. The exact original Phase 1 release is immutable; older runs cannot acquire these grants. V24 also makes generic ISA parser/generator algorithms common infrastructure, so historical ISA-tool ablations need a new interpretation. No target facts, instruction meanings, execution qualification or runtime tools follow from SDK availability. Missing capabilities remain unavailable.
Discover current executable scheduling APIs with `merlin.compile.scheduling.api_catalog.scheduling_capabilities(24)` or the selected `shared-library.json` scheduling_capabilities entry. It lists segmented-K issue ordering, exact final wide-load composition, array-weight reuse and source-axis loop transforms with actual symbols and required proofs. Emit the checked actual kernel; checking projections and discovery metadata supply no target execution admission.
V19 additionally provides a source-checked final-K producer/readout frontier on the original full accumulator view. It requires a separate target RAW retirement and pending-write read readiness capability, with original store snapshots/backpressure and a final full drain. It supplies no fixed-cycle spacing or loop-carried event ownership.
V22 keeps every V21 scheduling API, including loop-carried completion events and effect-checked schedule composition. Operator-only kernel packages and access-policy grader/oracle modules are not part of this SDK; the CCA calls remain available through their staged broker clients.
V24 additionally exposes source-checked shared-RHS sibling products through the scheduling dialect and the ordinary registered-pass A/B route. Bind original member operands and all outputs, immutable storage, readout contracts and selected ordering laws. Missing premises retain drains or decline; recheck emitted effects and complete numerical results before comparing whole-call cycles.


## Required public Spike scalability suite
Mathematical sources and exact source-group policy: /work/merlin/tasks/el4-grader-throughput-20261024/merlin/out/build/run-frameworks/afcf34be293c/out/artifacts/fresh-inputs-xs_capsules/source-release/payload/corpus/_public_scalability/public_sources/suite.json.
This separately selected suite has 40 cases and is required for completion. Use `python agent_selfcheck.py --suite scalability --capsules NAME1,NAME2 --wait-budget 5` for affected cases; collect asynchronous requests with `--attach ID`. Use `--capsules all` when ready for the complete functional denominator. Only Spike L0/L1/L2 is permitted here; gSIM and L3 requests are refused. A subset, unavailable contract or unrun case cannot complete this suite. Its passes give no RTL or timing credit. Core RTL requirements remain unchanged; hidden and full-model validation remain separate. Full-model execution follows the separately selected campaign window. In a joint campaign, start these cases immediately: their failures and performance feedback guide compiler architecture while core correctness is still evolving. Timing estimates, when explicitly selected, carry their own qualification status.


## Selected public connected programs
Cases: PG_source_attention_cache_representative, PG_source_repeated_inference_representative, PG_source_residual_block_medium. Sources: `public_program_sources/suite.json`. Use `python agent_selfcheck.py --suite programs --capsules NAME --sim spike --tiers L0,L1,L2 --wait-budget 5`. Only the operator-selected source roster is available. The original model owner compiles host/device work and checks every output; all selected programs are required final Spike obligations, tracked separately from original core conformance. They do not authorize gSIM or earn original-conformance/L3 credit. The scalability-only idle tuner does not automatically explore this separate endpoint.


## Joint editable compiler development

Develop one general compiler across all public capability families. A fresh run has an empty target backend; a declared continuation retains its authored compiler. Generic compiler sources under submission/compiler_library are editable: extend passes, dialects, numerical/layout representations, schedules, allocation, lowering and host generation as useful. There is no mandatory pass order, discovery sequence, tiny-canary prerequisite or minimum helper count. Choose component ownership and testing order; up to the selected helper ceiling can work concurrently toward the same compiler. Use all 20 CPUs as one shared pool.

Test useful eligible cases as soon as they can provide feedback, including large and connected cases. A tiny executable can cheaply expose ABI faults but does not gate unrelated work. Keep editing while immutable snapshots grade and integrate completed changes promptly. Consume transformed IR in real emission; preserve all outputs, numerical contracts, source placement and completion. No private answers, retained target backend seed, benchmark-name dispatch or forbidden CPU fallback. An import, receipt or build is not numerical evidence.

All 250 selected cases remain eventual obligations: 153 original correctness, 20 calibration and 77 scalability cases, with their actual required tiers. Optimize chiefly larger shapes and connected workloads from the start; small originals primarily establish correctness and model evidence. Full-model Spike development checks are available throughout the remaining campaign; there is no final-model-window waiting requirement in this execution mode. Without `--sim`, or with `--sim required`/`--sim auto`, this joint campaign selects Spike L0/L1/L2 development checks; this does not certify unrun physical tiers. Explicit `--sim gsim` accepts one named required original correctness capsule under its separate correctness budget, or one calibration witness subject to question review. Required correctness does not consume calibration permissions or qualify a fit. When every correctness gSIM slot is busy the request waits for one in arrival order (its progress says `queued for a gSIM slot (N ahead)`); collect it later with `--attach`. Unlisted cases, multi-capsule gSIM requests and scalability gSIM/L3 requests are refused.

Selected campaign duration: 12 hours from author launch; editing stops 60 minutes before deadline for freeze/drain. L3 gSIM certification of the selected correctness cases runs at the end, best-effort, on the final compiler; development grades stop at Spike L0-L2. Performance is measured on the Spike cycle model by default; ask for gSIM cycles with --sim gsim. Selective calibration gSIM concurrency 15, per-job budget 1200 seconds; initial diagnostics 2, bootstrap observations 8, later authorized batch limit 4. These are actual admission limits, not required exploration stages. Extra witnesses need matching numerical evidence and an unresolved question under the selected policy; required correctness has separate accounting. Unrun tiers and timeouts remain pending.

Use roofline and whole-SoC feedback with its scope and uncertainty. Prefer the selected tuned model when qualified; missing timing is not zero cost. Independent final FireSim validation uses the exact selected unchanged hardware configuration and complete output checks. Report observed correctness and performance separately; private grading remains independent.


Prohibited instruction role `loop_descriptor`: hand the endpoint a whole loop nest to run itself (a hardware-loop FSM). Build every schedule from the target's other commands; a case whose executed command stream issues a command with a prohibited role fails at L2.


## Selected public compilation resources

- Original public scale-emission resource policy: mode `enforce`, limits `max_code_bytes=1048576`; mandatory for completion: **True**. Use `python3 agent_selfcheck.py --compile-only --scale-probe --capsules SY_conv_k3x3_s1x1_d1x1_pad1x1 --timeout 120`. The harness supplies these limits; requests cannot relax them. The selected ladder also runs before background capsule simulations and publishes `qa/public_codegen_resources.json` with its candidate digest. Resource acceptance is separate from object compilation, semantic placement, numerical and trace qualification.


## Joint shared CPU scheduling

Author/helper affinity: `8,9,10,11,12,13,14,15,16,17,18,19,20,21,22`; shared compute affinity: `8,9,10,11,12,13,14,15,16,17,18,19,20,21,22`. Builds, Spike, gSIM and preparation may use any available core in the shared pool; there is no core partition by engine. Expensive harness steps acquire per-core leases. Author processes share their affinity; helpers receive no additional cores. Native simulation concurrency ceiling: 15; author/helper ceiling: 8. These are ceilings, not counts of running work. Selected gSIM concurrency ceilings: calibration witnesses 15; required correctness 15. Question/worker budgets still apply.

The native scheduler serves author-requested checks first (priority 1), background grading second (2), preparation third (3), and advisory probes last (4). Waiting jobs are FIFO within a class; running leases are not preempted. These classes are assigned by the harness, not a selfcheck priority flag. Prioritize critical work by requesting its focused check through the native broker; submit independent cases to rolling admission, reuse exact-input caches and avoid duplicate full sweeps. Continue compiler development after immutable snapshot acknowledgment; consume each finished result without waiting for unrelated slow cases. Keep heavy work on the documented leased harness routes, not unmanaged shell jobs.

## Editable compiler source

The complete reviewed public compiler source tree is in `submission/compiler_library/`. Edit frontend, graph transformations, dialects, scheduling, allocation, lowering and host code directly; add implementation modules as needed. The target backend remains your authored implementation. There is no required per-pass API menu or proof receipt for exploratory compiler calls. You choose component ownership, construction order and useful tests; no fixed pass order, first-capsule sequence or helper-count minimum is imposed by this guidance. Use up to 16 helpers within the selected ceiling when independent work exists, sharing all 20 CPUs dynamically. Evaluate useful independent cases concurrently against immutable snapshots while continuing edits and integrating completed helper work; there is no all-153 prerequisite for larger cases. Private numerical, ISA, timing and isolation grading uses independent evaluator code; editing compiler-side checks does not edit those gates. Do not add CPU fallbacks to evade acceleration obligations.

Public source inventory and explicitly unavailable dependencies: `FULL_COMPILER_SOURCE_SELECTION.json`. Unavailable private evaluators are not compiler dependencies you may read; use public interfaces or implement the required compiler behavior. No known target backend or private performance reference is seeded.

Initial source staging retained 8 member(s) different from the selected baseline. Read `COMPILER_SOURCE_VERSIONS.json` when an advertised API is missing or behaves differently. Retained source overrides take precedence; differences can be intentional author edits or older implementations. Inspect the relevant definitions and migrate explicitly without overwriting useful changes. This advisory describes initial staging, not the current evolving snapshot.


## Performance-driven compiler exploration

Strong performance across the admitted shape families is part of the work from
the start, not a phase after correctness: build on the scheduling representation
early and measure what each change does to cycles. Correctness and legal execution
still decide whether a variant is accepted. Pass counts, dialect coverage,
buffer counts and instruction counts are not the score. Keep a working baseline
and compare a few hypothesis-driven variants on relevant small checks and a
representative large Spike case under one timing epoch. Integrate wins as reusable
passes or shape/resource-derived heuristics; do not branch on capsule/model names.
A library edit inside the grant needs no operator transition. Spend effort on the
current dominant cost rather than implementing every available transformation.

For CPU-issued, tightly coupled accelerators, consider this mechanism menu only
when selected facts and observations support the premise:

| Suspected cost | Compiler/schedule options | Evidence needed |
| --- | --- | --- |
| CPU issue/configuration overhead | `configuration` elimination, `configuration_loops` hoisting, `command_compaction`, compact `Kernel` loops | Actual emitted commands/host work and complete-call cycles; keep unknown state clobbers. |
| Serialized transfers/compute | `copy_pipeline`, `compile_schedule` pass order, legal queue interleaving and buffer lifetimes | Dependencies, slot capacity, completion and same-epoch timing. Two buffers with per-operation fences do not demonstrate overlap. |
| Repeated operand traffic or spills | `compact_dense` blocking/traversal, `temporary_storage`, `device_spills` | Actual placement, live ranges, requested traffic, padding and a large case in the relevant capacity regime. |
| Boundary/packing overhead | `panel_transfers`, shared host codegen, `fusion`, `shared_input_schedule` | Numerical/layout/alias premises, complete live-outs, one-time versus per-call costs and whole-chain timing. |

This menu supplies hypotheses and API locations, not selected schedules or winning
parameters. Discover choices from typed IR, hardware facts and your own evaluations.
Preserve an independent serial/unfused alternative where useful. Report numerical
status, structural evidence and withheld RTL coverage separately; withheld gSIM is
not an instruction to simulate the whole corpus. Unknown complete timing is not zero
or a win: decide on the case's calibrated fast estimate meanwhile, and revert if a
complete estimate or measurement later disagrees.

## Construct capabilities for the whole public corpus

Read PUBLIC_CAPABILITIES.md once, then query PUBLIC_CAPABILITIES.json only for
relevant families and members. Use all public operations, shapes, layouts,
numerical boundaries and connected patterns to plan ONE general compiler.
Capsule IDs identify tests and provenance; never use them as production dispatch.
Before construction, make one compact capability DAG covering ALL declared public
families, with an owner, input/output contract, local dependencies and a fallback
or explicit missing capability for each node. Use these compiler boundaries:

- Frontend/source graphs and legal fusion -> typed semantic interface with numerical
  order, layouts, aliases and every live output -> reusable target lowering.
- Public RTL/ISA declarations -> generic ISA generation and typed operations -> the
  same target lowering. Generate neutral scaffolding inside the empty backend;
  do not import an existing target compiler. Encoding alone proves no memory effects,
  completion, legal fusion or instruction numerical semantics.
- Source ownership and live buffers -> Merlin host codegen/dispatch -> complete
  host/device execution ABI joined to target lowering.
- Shared scheduling IR, transformations and pass policy connect these layers: tune
  pass order and branches from source shapes, layouts, numeric contracts and resources;
  consume transformed products in actual emission rather than logging unused calls.

Construct independent nodes concurrently. Do not develop the compiler one capsule
at a time or wait for one family to pass before implementing another. Agree on typed
contracts and disjoint source ownership; every helper contributes to ONE compiler.
When CONSTRUCTION_PLAN.example.json is staged, adapt it after mandatory discovery
into submission/CONSTRUCTION_PLAN.json (and each independently tested variant).
Preserve its inventory_sha256 and complete family roster; assign real owners,
planned module paths and local depends_on edges, with integration_node depending
transitively on all capability nodes. Planned modules need not exist yet. This is
construction ownership, not evidence of implementation or a correctness certificate.
If the selected policy requires the plan, ordinary broad simulation requests check
it; editing, compile diagnostics and the policy's bounded tiny bootstrap remain
available. Missing code or a failed capsule never globally stops construction.
Existing construction_actions validates local action dependencies when used.

Write executable entrypoints early and prove one tiny end-to-end
emission/target-consumption/link/Spike/all-output path. Then immediately submit broad
eligible sets from the entire public roster through the rolling queue; retain
unimplemented families as active construction tasks. Further tiny canaries are for
changed boundaries, not a mandatory serial check for every case. Fix shared ABI
failures before repeating affected work; independent capabilities keep progressing.
Consume each result as it arrives, without waiting for a batch or source quiet.
Inspect the returned public source -> target IR -> LLVM/object artifacts and their
complete pointer/output ABI, actual commands, loops, readout modes and completion.
Trace the first incorrect boundary; a successful import, pass receipt or object
build alone does not establish executable semantics.

The goal is ALL cases in the current selected public inventory complete AND strong
performance. Use PUBLIC_CAPABILITIES.json and the launch contract for current counts,
roles and required tiers, including admitted supplements; historical corpus sizes
are not coverage limits. Correct declared refusals remain refusals.
There is no whole-correctness-cohort barrier before calibration or large-shape work.
Preserve each original Spike/gSIM obligation and use the normal calibration policy.
Begin scalable implementation and legal optimization during construction; use the
selected provisional model with its limits, then the selected tuned model when
available. Optimize chiefly the selected primary performance roles and connected
workloads; small correctness cases have low performance priority and supply
regression/calibration evidence. Prioritize broad Spike numerical coverage before
spending discretionary gSIM work. Every gSIM candidate needs matching passing
numerical evidence; original required L3 obligations and selected calibration
tiers remain intact. A runtime forecast does not waive a required physical tier.
As soon as informative passing Spike witnesses exist, automatic calibration can
progress while other compiler families are repaired; do not impose a global pass
barrier. Aim for below 5% error on independent held-out measurements within the
model's claimed domain, including large-shape regimes. Small held-outs alone do
not validate large-shape accuracy. Use a qualified selected model when available;
otherwise retain explicit provisional uncertainty and missing coverage. Keep
FireSim as independent final validation, never coefficient-fitting feedback.
Use compact family-level feedback and retrieve details on demand. For reusable
construction/check sequences use the staged `construction_actions.py` with explicit
commands, semantic capability, boundary and local dependencies. Automatic and
offline studies execute the same client and self-check broker; its command success
is not numerical certification. Keep source/request identities in the action log.
Prefer a bounded
machine-run search over legal pass choices to one agent turn per parameter trial.
If exported in the selected library, `pipeline_policy.evaluate_variants` evaluates
applicable policies on separate IR clones and returns transformed artifacts and
receipts. Feed those artifacts into real emission and independent checks; a search
receipt is neither a correctness certificate nor a timing measurement. When the
selected version provides `host_pass_profile`, inspect action_wall_ns separately
from framework_wall_ns to find expensive transformations versus orchestration.
These are host pass costs, not target cycles or capsule numerical evidence.

## Learn reusable schedule regions

Explore both the optimization mechanisms and their selection: edit or introduce
shared passes, transformations, schedules, knobs and flags when the existing menu
is insufficient. Keep semantics/lowering shared; specialize actual pass pipelines
using source geometry, layout, precision and hardware-derived resource facts.
Start with coarse regions, then refine where paired complete-call cycle evidence
reveals a crossover. Consider interacting dimensions and aspect ratios, not only
one global size cutoff. Storage-fit legality is independent of profitability.
Use cheap cycle-model comparisons first; spend additional simulations near uncertain
crossovers or on a specific unresolved mechanism, without rerunning the whole corpus.
Compare both sides of each proposed boundary and aligned/ragged neighbors. Freeze
separate geometries for validation; a holdout used to revise a rule becomes development
data and needs a fresh replacement. Preserve each region's incumbent where evidence
is missing, noisy or regresses. Avoid finer rules without useful performance support.
When exported, `shape_policy` assesses proposed typed guards and executes selected
registered pipelines; the agent still discovers the guards and supplies real paired
feedback. Extend shared predicate tooling for missing generic features rather than
encoding capsule identities. Record the rule, selected pass order/options, evidence
domain, gains/regressions and tested coverage; no universal generalization claim is
implied. This work runs alongside construction and adds no launch prerequisite.

Optimize across compiler levels when a local schedule plateaus. Compare graph
grouping and fusion/fission, producer/consumer layouts, immutable preparation,
storage lifetimes and reuse, instruction selection and command batching, as well
as device schedules. Preserve structured batch/head/reduction axes and explicit
effects until the relevant transformations have consumed them. Keep every chosen
alternative connected to ordinary host/device emission; annotations alone do not
change execution. Fusion can lose through spills or lost parallelism, so retain
unfused alternatives. Layout choices include downstream conversions and packing.
Only hoist preparation for truly immutable/versioned inputs; report cold setup
separately from repeated execution under the declared workload scope.

Use hardware-relative features where meaningful (tile counts, tail fractions,
live bytes/capacity, reuse distance and transfer/compute demand) alongside raw
dimensions. Do not invent measured activity from those static features. Explore
small interacting bundles when one-pass-at-a-time search misses enabling effects,
then ablate useful winners to explain their benefit. Rank next work by expected
whole-model time saved, occurrence count and evaluation cost; estimate only from
compatible timing scopes, and check complete chains for shared/overlapping work.
Screen legality, changed emitted code and duplicate programs cheaply; reuse valid
prior observations and evaluate informative boundary cases before broad sweeps.
Keep compiler search cost separate from target runtime, and prefer simpler rules
when timing uncertainty cannot distinguish alternatives. Record missed optimization
reasons (legality, missing representation/adapter, unsupported timing or measured
regression) so a helper improves the actual bottleneck rather than adding flags.

## One evolving compiler, parallel bounded work

Act as one compiler coordinator and delegate bounded, independent tasks to the
available parallel workers, up to the selected ceiling of 16 helpers when supported.
Start independent capability construction promptly; use fewer workers when ownership,
CPU or token budgets would make more redundant. Choose tasks from the current bottlenecks: frontend
operator coverage, connected-region legality and fusion, scheduling/pass policy,
shared host boundaries, or focused correctness/performance diagnosis. Every worker
contributes to the SAME reusable compiler; do not create separate benchmark backends.
Use the supported isolated lightweight git/task mechanism for edits, with narrow
file ownership and clear commits, rather than copying the whole compiler per worker.
The coordinator owns integration, compiler health and the next useful experiment.
Integrate compatible completed changes promptly; keep conflicting variants separate
until a focused same-scope comparison supports a choice.

Submit focused checks asynchronously against immutable compiler revisions. Continue
independent implementation while they run; neither the coordinator nor all workers
must wait for source quiet or each other's tests. A result belongs to its recorded
compiler/input/runtime hashes, never automatically to newer edits. Name the revision, touched files, consumed pass/IR APIs, local dependencies,
focused checks, first actionable failure, numerical/performance/model confidence and
artifact paths in each short handoff. Pass relevant contracts and a compact capability
brief to helpers; retrieve detailed source or logs only for their current boundary. Reuse an existing equivalent request/result
only when those relevant identities match. Avoid full-cohort checks on every edit;
reserve them for declared milestones/final validation. Report untested coverage.
All 20 selected CPUs are dynamically shared through the existing pool: no fixed
CPU assignments to author, compilation, grading, or calibration roles. Available
worker count is a concurrency ceiling, not a requirement to launch redundant work.
A protected evaluator/library upgrade may request a checkpoint; ordinary compiler
and editable-pass development continues while immutable evaluations run.

## Model deployment and numerical boundaries

Maximize accelerator coverage over ALL operations and connected regions admitted
by the selected hardware facts and numerical contract, not only contractions.
Inspect standalone and fused forms separately: residual/elementwise addition,
bias, scaling/requantization, activation, pooling and movement may be available
through different legal device forms. Every must-accelerate supported region needs
a legal fused OR unfused device path; a failed fusion attempt does not justify
silent CPU substitution. A readout-only epilogue capability does
not imply a standalone instruction. Derive eligibility from actual operand and
result types, shape/layout, numeric order and selected capabilities; an operator
name alone is insufficient. Preserve every live output when grouping or fusing.

Do not silently send a supported legal domain to the CPU because its compiler
adapter is unimplemented, inconvenient, or has an unmeasured cost. Hardware
unsupported, compiler unimplemented, numerically invalid and unresolved proof
are different outcomes. Missing target lowering is a compiler task. Profitability
guides device grouping, schedules and fusion choices; it is not blanket permission
to evade required accelerator coverage. Initially establish reusable lowering for
the supported domain. Subsequently compare legal CPU/device placements under the
same numeric and complete-call cost scope; a measured or supported-model CPU win
may justify CPU placement where the selected admission policy permits that choice.
If a current mandatory device gate prevents that comparison, report the precise
policy limitation for review; do not silently bypass it or label the hardware
unsupported. Retain real hardware-unsupported floating or nonlinear/control seams
through Merlin host codegen with explicit evidence.

Maintain source-bound placement diagnostics: source region/node IDs, standalone
versus fused-only capability, selected device form, host reason and remaining
compiler/proof gap. Report required/covered/eligible-but-host/unknown counts and
static boundary bytes separately from measured host time and transferred bytes.
Unknown measurements stay unknown. The fast model-check contraction census is
only a contraction census, not evidence that every supported operator is offloaded.
Inspect the native complete source-demand report for all-region obligations.

The selected deployment for ResNet50, TinyLlama and SmolVLA is W8A8:
signed INT8 weights and activations at accelerator contractions, with the selected
integer accumulation and output ABI. This does not require every intermediate or
model output to be INT8. Keep raw i32 outputs when the typed source requires them.
Preserve captured floating-point normalization, nonlinear functions, scaling and
control through Merlin host lowering; do not replace accelerator contractions with
CPU fallback or convert an entire mixed graph to integer arithmetic.

Use the selected public `target_contract.yaml`, typed source and numeric policies
as the authority. Its scaled i8 readout contract specifies f32 scale, half-even
rounding and saturation to [-128, 127]. That contract does not make arbitrary host
casts, raw accumulator stores or other source rounding operations equivalent.
Track operand/output dtype, scale identity and granularity, zero points, bias and
activation order, rounding, saturation, accumulation bounds and host/device
placement at each graph boundary. Fusion, packing and scheduling must preserve
these properties; never manufacture scales or loosen comparison tolerances.

The existing visible model-check endpoint accepts `resnet50`, `tiny_llama` and
`smolvla`: `python agent_model_check.py check --model MODEL`. ResNet has `model`;
TinyLlama has `prefill` and `decode`; SmolVLA has `prefix_encode`, `flow_denoise`
and `action_decode`, including its declared recurrent state. Early fast checks
replay selected kernels; they do not establish full-chain numerical correctness.
Use unchanged independent numerical contracts for capsule, connected and final
model checks. A format-compatible external signature alone does not establish
model calibration, boundary semantics or numerical equivalence. Missing evidence
stays unknown. The compiler must generalize through these source contracts, not
model-name dispatch.

## Scheduling integration and shared-pass development

Try the exported scheduling IR where useful in the production compiler:
`merlin.sched.ir.kernel` provides Kernel, Call, Loop and explicit dependency/stage
fields; `ir.expr` supplies symbolic indices and `ir.parse.parse_kernel` reopens
retained schedules. These are useful abstractions, not a dialect-use score or a
requirement to run every pass. Honor the selected shared-pass contract; an unused
import or a side demonstration does not establish production integration.
Keep symbolic counted loops for large problems; the finite `compile_schedule`
route is for bounded instruction regions, not unrolling a full large tensor.

If selected, `selected_contracts.scheduling_spec_draft` and
`inspect_scheduling_draft` expose missing declaration fields through the ordinary
resolver. Supply explicit authored facts/software paths; keep unknown effects
unknown until established. Raw ISA encodings alone do not provide logical memory
effects or completion. The current compact dense family requires synchronous
logical operations; queued acceptance is not completion. Existing block schedules
remain available while developing a sound asynchronous adapter or extension.

Represent buffer ownership and lifetimes explicitly in the chosen scheduling IR: host input,
DMA destination, resident compute operands, accumulator and output each need an
identified owner, read/write effects, completion and safe reuse boundary. Construct
legal load/compute/store interleaving across tiles, with prologue, steady state,
epilogue and tails. Double buffering and prefetch may overlap independent transfers
with compute only while their storage slots, banks, command resources and hardware
dependencies permit it. Preserve preload/compute accumulator dependencies, drain
stores before host consumption and fence before reuse where required. Model the
accelerator's actual command queues and dependency tracking; a CPU reorder buffer
is not a substitute. If the timing model rejects a legal overlap, retain the exact
schedule and rejection as an unsupported-model report for operator review; do not
silently serialize the intended production schedule or invent an overlap estimate.
Keep a legal serial schedule available for numerical diagnosis and comparison.

The selected baseline inventory gives inspectable module paths, signatures and
source hashes. When the editable generic fork is selected, improve its pass
implementations alongside target adapters and policy in your submission. The
compiler change may belong in semantic admission, graph/region construction,
numeric/layout contracts, allocation, instruction lowering or host execution,
not only a pass. Identify the missing boundary and retain a working route.
When the selected `DenseSourceResult` provides `emit_llvm`, it connects the
checked compact schedule to the ordinary `FnBuilder`: the target callback emits
instructions while the shared schedule owns loops, tails, accumulation and final
stores. Inspect that seam before reconstructing those semantics manually. This
emitter preserves the chosen schedule; it is not by itself a performance win.
Compare meaningful scheduling/policy alternatives through actual emitted code.
Use only available APIs and extend the editable library when a real gap remains.
The
following are available mechanisms, with independent
legality requirements; they are not a mandatory sequence of all passes:

| Layer / entrypoints | Applicability and tunable choices |
| --- | --- |
| `block_schedule.schedule_contraction`, `schedule_tiled_convolution` | Declared contraction/convolution semantics and geometry; tune tile dimensions, loop order, operand residency, load grouping, lookahead and placement through Geometry/Knobs; retain tails and reduction order. |
| `dense_source.schedule_source_dense`, `compact_dense.schedule_compact_dense`, `llvm_program.emit_index_program` | Recognized typed sources and admitted index/encoding adapter; keep large nests compact and derive dimensions from the input. |
| `shared_input_groups.find_shared_input_groups`, `shared_input_schedule.schedule_shared_input` | Actual shared operands and legal ordered reductions; choose groups/tiles under storage bounds, verify with `shared_input_verification`. |
| `panel_transfers.plan_packed_panels` | Actual source layouts, bounds, transfer widths and physical limits; tune packed panels and coalescing without assuming contiguous physical storage. |
| `schedule_pipeline.compile_schedule` | Finite Kernel plus effect/storage adapters, actual-byte decoder and resource contracts; use ScheduleBudgets and pass_order with validated prerequisites, current context stamps and bounded repair. |
| `temporary_storage.allocate_temporary_storage`, `device_spills.allocate_with_spills` | Private ownership, alias lifetimes, storage banks and admitted spill classes; choose allocation, copy elimination and spill limits, preserving completion. |
| `copy_pipeline.copy_pipeline_motion` | Explicit independent resources, in-flight limits, slot lifetimes and wait/completion facts; tune CopyPipelinePolicy and prefetch distance, preserving a legal serial fallback. Double buffering requires two actually nonoverlapping live slots. |
| `fusion.plan_fusion`, `pointwise_source.compile_source_pointwise` | Exact scalar/reduction correspondence plus effects and resource bounds; select legal fusion groups and bounded recomputation rather than model-name rules. |
| `configuration.eliminate_redundant_configuration`, `configuration_loops.hoist_loop_configuration` | Decoded reads/writes/clobbers and loop invariants; preserve unknown state changes. |
| `staged_dense.rewrite_staged_dense`, `command_compaction.compact_command_stream` | Admitted staged operand/descriptor contracts; select reuse, legal output-tile ordering and command compaction with their independent checks. |
| `program_limits.finalize_program`, `row_softmax.compile_source_rows` | Actual encoded bounds / continuation contracts; row numerics, ordering and storage remain explicit. Unsupported row arithmetic uses an admitted host route. |
| `host_optimizations`, `host_tensor_codegen.HostLinalg`, `host_epilogue.HostEpilogueEmitter`, `requantization` | Reuse shared host codegen and exact numerical policies. Tune graph partitioning, packing reuse and host/device handoffs; do not regenerate generic host arithmetic case by case. |

Use a source-to-realization fusion workflow where applicable. Distinguish adjacent
pointwise fusion, contraction epilogues, shared-input siblings, connected
producer/consumer residency, and preparation reuse. Their legality is different:
source adjacency alone proves neither scalar equivalence nor buffer lifetime.
Preserve all live-outs and independent output/epilogue state, including observers
outside the proposed region. Keep a correct unfused alternative. Record source IDs,
numeric/capability pins, legality premises, storage/lifetimes, selected schedule,
and applied/no-match/unsupported/unknown; do not call an unimplemented route
hardware unsupported. Compare complete-region costs, including boundaries, only
when scope and numerical contracts match; unknown model cost is not a gain.

If `merlin.compile.fusion_workflow` is present in the selected public API inventory,
`discover` records these source opportunities and native fanout-reader refusals.
`realize_adjacent` returns the checked Kernel and remaining temporary objects;
feed both to ordinary allocation/emission. `realize_shared_input` returns actual
logical effects with independent sibling outputs; target encoding/effect checks
remain required. `realize_multioutput` returns the existing guarded pointwise
compilation images. Consume returned products, not just receipts. Counted loops
stay on the compact route; finite fusion never unrolls them. Source epilogue proofs
still use `output_fusion.decisions`; general connected residency is not implemented
by this facade. Use existing `compiler_observations.install_prepared_observed` and
`lifetimes` for preparation installation and source lifetime evidence. Save pinned
connected_fusion sections with `compiler_observations.write_receipt` for the
existing diagnostic feedback path; planned/checked schedules do not prove runtime
traffic reduction or complete numerical execution. If the optional facade is not
exported, use its listed native APIs from the actual inventory; no protected import.

When the selected facade provides resumable discovery, bound each `discover`
page with `max_candidates` and retain its returned cursor for the same source and
contracts. `evaluate_fusions` evaluates isolated `FusionAlternative` products
alongside an unfused baseline; pass the actual product to the emitter. Use
`realize_candidate(..., connected=options)` with the selected
`connected_regions.compile_connected_pointwise` bridge for supported ordered
pointwise stages. Its explicit tensor bindings and runtime continuation preserve
live buffers across launches; this is not proof of on-chip retention or a speedup.
When selected, `compile_connected_region` also composes checked dense and pointwise
products with explicit live buffers. Its `ordered_invocations()` supplies the
actual products and bindings to ordinary target code generation; `execute()` is
a separate synchronous buffer-runtime interface. Mixed dense/pointwise composition
still requires the target's multi-stage pointer/launch ABI and completion checks.
Neither API alone establishes fused device execution or on-chip retention.
Implement missing reusable composition mechanisms through editable compiler
code; retain a correct route and all live outputs while exploring alternatives.

Maintain an input-derived pass applicability record for every compiler invocation:
family, legal premises, applied/no-match/unsupported/unknown, precise reason,
selected policy and actual source/result schedule digests. Unsupported means a
specific absent capability; unknown means an unresolved implementation or proof.
Do not declare everything unsupported to avoid implementing scheduling. Prioritize
the applicable mechanisms identified by the first small and large Spike results.

Persist the real Kernel text and the library's `merlin.schedule_compilation.v1`
report when using compile_schedule, including pass records, library/context
identity, resources, final schedule/image digests and completion checks. Feed its
actual result into emission. The host's V2 target transport receipt proves file
custody only; your JSON claims do not independently prove optimization use or
semantic equivalence. Do not report scheduling qualification until an operator
checks the selected schedule against actual emitted operands/completions. A valid
no-match is acceptable; an ignored transformed result is not. Numerical checks
and performance evidence continue while that qualification remains pending.

For modules in the selected editable fork, implement generic improvements directly
in `submission/compiler_library/`; each evaluation snapshots those edits together
with your backend. You need no operator epoch for an ordinary edit inside that
grant. Keep useful failure examples and independent checks with the change.
For a new shared module or protected implementation outside the editable grant,
put a patch and applicability example in `shared-library-proposals/` and continue
independent work while the operator reviews the extension. The operator can
request an epoch stop at the next native author turn boundary, drain owned jobs,
freeze your current compiler, prepare a new versioned library and start a new
native run with `--joint-epoch REVIEW_JSON --fresh-backend`. That continuation
retains your authored compiler, original empty origin, original 12-hour clock
and aggregate gSIM ledger. It grants no historical certificate. Never edit the
immutable baseline in place or import a prior backend to implement a proposal.

## Capability-selected optimization suggestions

Read `OPTIMIZATION_SUGGESTIONS.json`: supported suggestions and unknown discovery questions are separate. This is advice, not pass order, legality or performance proof. The initial prompt has no workload IR and cannot establish its prerequisites. When selected, `COMMAND_ORDERING_SELECTION.json` carries the exact public RTL-derived ordering record for schedule law construction; the convolution/dense epoch recipes discover it without access to operator paths or environment. This is source evidence, not a selected schedule or permission to ignore dependencies. Recompute for the actual public source with `python optimization_suggest.py --ir PATH --output suggestions.json`. Only selected API modules are import grants; design starter sets grant no new implementation. Missing declarations remain unknown. The abstractions map connects typed graph/contracts, legal region fusion and preparation, compact Kernel scheduling, target emission and shared host codegen. Locate their editable sources in COMPILER_LIBRARY_API.json. Distill useful changes into general passes and typed-IR/resource heuristics for additional models; do not dispatch on model or benchmark names. Dialect availability does not require using every dialect or replacing the existing compiler.

## Selected public software declaration

Read `/work/merlin/tasks/el4-grader-throughput-20261024/merlin/out/build/agent-workspaces/phase1/gemmini/merlin_assisted/gemmini-open-compiler-20261028-fresh10seed/workspace/public_declarations/software-spec.yaml` for the original selected software semantics (SHA-256 `3c1b8aa1c77775e1c0e590af871f05e038743d7000a99860fb12843f7053372f`). Use this readable data leaf with `read_scheduling_contracts`; the original backend tree remains withheld. This identical-byte alias grants no additional implementation or qualification.


## Build against the whole public corpus

Read `PUBLIC_CAPABILITIES.md` first. It summarizes 152 selected public cases; Use `python capability_query.py --groups`, `--group PREFIX` or `--capsule ID` for bounded queries over `PUBLIC_CAPABILITIES.json`. Build one reusable compiler around the required semantic capabilities and compiler boundaries: graph, numeric/layout, schedule/storage, ISA lowering, host/runtime, integration. Assign bounded disjoint edits, integrate shared interfaces early, and let admission gate only tests, never construction in an independent capability. For reproducible action sequences use `python construction_actions.py PLAN.json --out OUTPUT`; it invokes ordinary commands and agent_selfcheck.py, just as an offline study does. Test a tiny end-to-end case as each capability becomes executable; continue independent construction while immutable snapshots are evaluated. Calibration prerequisites and large shape cases can proceed before all original correctness cases pass. The scalability cohort contains both performance targets and diagnostic controls: inspect workload_role and performance_priority. Do not spend repeated performance rounds on small controls once their mechanism is resolved. Original small cases remain mandatory correctness regressions.


## Optimization principles by operation

`OPTIMIZATION_BY_OPERATION.json` holds reviewed optimization principles, pitfalls and the matching library mechanisms, skills and recipes for the operation families of this corpus (contraction, matrix_vector, convolution, elementwise_epilogue, data_movement, connected_chain), plus principles for every operation. Families come from each case's declaration, not its name; a self-check case's `optimization_next.operation` names its entry. The principles are hypotheses: judge them by cycles of the same evidence kind on the affected shape classes (`python agent_selfcheck.py --panel FAMILY`). Principles that need an instruction role the run policy prohibits are left out.

Compositions: the library's composition catalog (`merlin.compile.composition_catalog.composition_capabilities`) covers producer -> consumer chains: a product feeding bias, scale, activation, requantization or a pool, residual adds, sibling products sharing an input and per-head slices. `python .agents/skills/heavy-pass-recipes/scripts/composition_routes.py --case NAME` lists the routes that apply to one case, each with its catalog rows, a library `module:symbol`, whether `submission/` imports it and a recipe or contract command; self-check rows carry the same pointer as `optimization_next.composition`. Model and program checks also accept `submission/source_composition.json` requests (`prepack_weights`, `share_exact_tensor_preparations`, `dense_input_views`, `propagate_layouts`, `reuse_intermediate_storage`, `shared_input_siblings`, `independent_heads`, `op_profile`): the original owner applies a request only where the source proves it and keeps the ordinary route elsewhere, and each row's `source_composition` reports what was requested, selected and applied (`optimization_next.source_composition_request` names options a case may use). The heavy-pass-recipes skill's `reference/source-composition.md` gives the schema. Neither a route nor a request establishes a speedup: compare cycles on the same cases.


## Reference map

`REFERENCE_MAP.json` (read-only, operator-built) links regions of the expert references in `reference-kernels/` to the operation each implements: `operation.kind`, `dims`, `epilogue` and `dtypes`, the `techniques` observed there, whether the region `uses_hardware_loops`, and the `confidence` and `evidence` of the link. Each entry names a `file` and a `locator` (lines, function or address range). Once a case passes numerically and is a performance target, its self-check feedback names at most two matching entries (exact dimensions, same shape class, or same operation kind) beside its next step. A region that uses hardware loops shows a technique to express with ordinary commands. The map grants no numerical or performance credit: check each change with ordinary selfchecks.
A match describes operation geometry, not equal fusion, layouts, quantization or timing boundaries.


## Use selected kernel reference evidence

Read `REFERENCE_MAP.json` for expert reference regions linked to operation kinds and shapes, and lift a mapped region (its `file` and `locator`) with the analysis below to read its instructions, loops and data movement. Primary optimization objective: approach the supplied expert kernels' performance through compiler rules that generalize. Treat supplied expert measurements as trusted performance targets; retain their stated measurement conditions and distinguish them from our model estimates. For every representative, record expert versus emitted candidate evidence for fusion, operand retention/traffic, transfer shapes, tiling, overlap and completion. Prioritize the largest supported mechanism and performance gaps; a slower valid kernel remains unfinished optimization work. Missing observations require a tool/evidence repair task. Exact instruction matching is unnecessary; demonstrate equivalent useful mechanisms or a measured better alternative. Derive schedules from source semantics, dimensions, tails, layouts, capacities and target laws. Reference names and copied per-case schedules must not select the implementation. Use references during construction, before a case passes. Choose a declared public case and one mapped region; align its source graph, fusion/live-outs, shapes, dtypes, layouts and quantization first. Inspect with `python agent_selfcheck.py --reference-analysis lift --reference-entry ID` (or `--reference-path reference-kernels/FILE --reference-symbol SYMBOL`). Inventory and bounded pages use `--reference-analysis inventory` and `--reference-offset N --reference-limit 8`; ZIPs accept `--reference-member MEMBER`. Missing operands/ISA correspondence are gaps, not inferred zeroes. After compilation, rebuild returned lowered IR with the existing `.agents/skills/inspect-kernel/scripts/build_kernel.py LOWERED_IR --out variants/NAME/build/CASE`, For a mapped machine-code view, run `python agent_selfcheck.py --reference-analysis compare --reference-entry ID --candidate-path variants/NAME/build/CASE/kernel.dis --candidate-symbol SYMBOL`. Select one region on each side; do not silently compare whole-model orchestration to a leaf kernel. A zero comparison denominator is missing evidence. Declared semantic contracts can be compared with format `semantic_region`; equal declarations still require binding to actual compilation products. Establish a reference technique's applicability from current source semantics, shapes, layouts, numerics and resource facts. For each mechanism, keep a baseline plus alternative explanations/schedules until evidence separates them. Use the selected transformation catalog and pass-library/ab-library-pass APIs; construction feedback names applicable APIs and missing source/effect adapters. Recompute applicability with `python optimization_suggest.py --ir SOURCE --output suggestions.json` when staged. Express choices through schedule.plan and consume it in emission. Inspect an actual plan using `--reference-analysis candidate --reference-format schedule_mlir --reference-path variants/NAME/plan.mlir`; its decisions alone never prove realization. Keep the consumer receipt and emitted-artifact pins, then run existing diagnostics, compile checks and numerical selfchecks. If composition cannot express a supported expert mechanism, implement the missing pass, lowering, schedule consumer or general scheduling rule on permitted compiler surfaces, with explicit legality and fallback; evaluate it through the same loop. Use declared knobs or ordinary authored variants for bounded alternatives, then compare compatible complete-region timing and check unseen shapes. Classify unresolved source semantics, ISA, parser, library, effect and schedule-consumer bindings as explicit compiler/tooling tasks; continue independent construction without waiting for manual per-kernel decisions. Maintain the experiment's `OPTIMIZATION_RULES.json` as a mechanism-by-operation family/shape index: legality/resource guards, pass order/settings, positive and negative results, emitted witnesses, trusted reference targets, timing domain, exact receipt paths and remaining gaps. Reuse `merlin.perf.optimization_ledger` and pinned `merlin.mining.reconstruction_history` records for outcomes; link those receipts from the rules and `ITERATION_NOTES.md`. Authored rules remain hypotheses until checked. Test baseline, alternatives and ablations on neighboring shapes, tails, capacity boundaries and related operations; freeze candidates before independent held-out evaluation. Promote rules through existing `merlin.transforms.shape_policy` proposal/assessment, preserving fallback outside validated guards. This guidance grants no additional source access, numerical, mechanism or performance credit.
