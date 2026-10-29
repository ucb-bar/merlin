"""Source-only scheduling discovery for ordinary reviewed author libraries.

The catalog supplies callable locations and composition requirements. It
selects no target law, policy, benchmark, simulator, timing answer or grant.
"""

from __future__ import annotations


def scheduling_capabilities(version=16):
    """Return fresh usage metadata; the selected library contract grants access."""
    if type(version) is not int or version not in (16, 17, 18, 19, 20, 21, 22, 23, 24):
        raise ValueError("scheduling discovery requires explicit inventory 16 through 24")
    prefix = "merlin.compile.scheduling."
    catalog = {
        "schema": "merlin.scheduling_api_catalog.v1",
        "inventory_version": 16,
        "authority": "discovery only; explicit source/effects/target admission remains required",
        "execution": "same Python/xDSL APIs in ordinary automatic and offline authoring",
        "capabilities": [
            {
                "name": "segmented_k_retained_operand_pipeline",
                "symbols": [
                    prefix + name
                    for name in (
                        "dense_segmented_pipeline.DenseSegmentedAdmission",
                        "dense_segmented_pipeline.dense_segmented_blocks",
                        "dense_segmented_pipeline.schedule_dense_segmented_pipeline",
                        "dense_segmented_pipeline.build_dense_segmented_pipeline",
                        "dense_segmented_pipeline.verify_dense_segmented_pipeline",
                        "dense_segmented_pipeline.admit_segmented_tile_loads",
                        "dense_segmented_pipeline.batch_dense_segmented_readouts",
                        "schedule_ir.make_dense_schedule_plan",
                    )
                ],
                "usage": [
                    "Use dense_segmented_blocks for full-LHS/two-RHS capacity and alignment bounds.",
                    "Create a typed schedule plan with pipeline_kind='retained_lhs_segmented_k'.",
                    "Choose a supported pipeline_issue_order and capacity-legal block_tiles.",
                    "Call schedule_dense_segmented_pipeline with original source identity, storage, adapter and law.",
                    "Revalidate the result and emit its actual kernel through materialize_scf/emit_llvm.",
                ],
                "requires": [
                    "explicit conjunction of ordered reduction commits and disjoint future-load execution laws",
                    "complete original source, tails, aliases, capacities, numerical boundaries and epoch proof",
                ],
            },
            {
                "name": "array_weight_retention",
                "symbols": [
                    prefix + name
                    for name in (
                        "dense_array_retention.DenseArrayRetentionAdmission",
                        "dense_array_retention.propose_dense_array_retention",
                        "dense_array_retention.verify_dense_array_retention",
                        "dense_array_retention.verify_dense_array_tile_loads",
                    )
                ],
                "usage": [
                    "proposal = propose_dense_array_retention(checked.kernel, adapter, array_law)",
                    "Verify with original problem/block/resources/storage/adapter/base_admission and static_budget.",
                    "For final wide loads use verify_dense_array_tile_loads with actual project/encode_band callbacks.",
                    "Emit only proposal.kernel; proposal.checking_kernel is a conservative proof projection.",
                ],
                "requires": [
                    "pure K,N,M paired run; invariant RHS identity/depth/width and continued scratch residency",
                    "separate fresh-capture/STAY/output-descriptor target law plus original source/epoch admission",
                    "no cross-run or cross-epoch retention, hidden configuration effect or implicit drain removal",
                ],
            },
            {
                "name": "exact_final_wide_loads",
                "symbols": [
                    prefix + name
                    for name in (
                        "tile_load_coalescing.TileLoadAddressLaw",
                        "tile_load_coalescing.TileLoadMapping",
                        "tile_load_coalescing.propose_tile_load_coalescing",
                        "panel_tile_loads.expand_dense_panel_tile_loads",
                        "panel_tile_loads.admit_panel_tile_loads",
                    )
                ],
                "usage": [
                    "Decode the final emitted call, not an earlier primitive access list.",
                    "Project every physical band with its exact source pitch, local stride and completion epoch.",
                    "Use the typed panel, segmented or array consumer for complete final source/hazard checks.",
                ],
                "requires": ["independently selected address law; address projection grants no execution semantics"],
            },
            {
                "name": "source_axis_loop_scheduling",
                "symbols": [
                    prefix + name
                    for name in (
                        "dense_panel_pipeline.dense_retained_compute_nests",
                        "dense_panel_pipeline.dense_retained_compute_order",
                        "dense_panel_pipeline.dense_retained_compute_unroll",
                        "loop_schedule.LoopTransform",
                        "loop_schedule.propose_loop_transforms",
                    )
                ],
                "usage": [
                    "Discover axes from projected original operands, then request actual typed loop transforms.",
                    "Run the complete source/storage/effect verifier on the transformed command kernel.",
                ],
                "requires": ["axis discovery and transform construction are proposals, never legality certificates"],
            },
        ],
    }
    if version >= 17:
        catalog["inventory_version"] = version
        catalog["capabilities"].append(
            {
                "name": "source_exact_quantized_spatial_readout_cache",
                "symbols": [
                    prefix + "spatial_readout_cache." + name
                    for name in (
                        "SpatialReadoutCache",
                        "propose_spatial_readout_cache",
                        "emit_cached_spatial_max",
                    )
                ],
                "usage": [
                    "Propose retention from the original RollingPackedSpatialPlan and disjoint caller-owned windows.",
                    "Keep original Q1 stores, Q2 conversion, clipped halos and all source liveouts.",
                    "Emit counted initial/frontier reloads through the selected synchronous lower callback.",
                ],
                "requires": [
                    "supported 3x3 stride-2 pad-1 signed-i8 source with original producer ReLU",
                    "exact unchanged i8-to-i32 reload and original pooled readout binding",
                    "producer restricted to its assigned disjoint accumulator window",
                    "actual reload and readout completion; storage capacity supplies no target capability",
                ],
            }
        )
    if version >= 18:
        from merlin.compile.composition_catalog import composition_capabilities
        from merlin.compile.scheduling.library_sources import library_modules

        catalog["composition"] = composition_capabilities(selected_modules=library_modules(version))
        catalog["capabilities"].extend(
            [
                {
                    "name": "disjoint_accumulator_readout_compute",
                    "symbols": [
                        prefix + name
                        for name in (
                            "dense_panel_pipeline.DenseReadoutComputeAdmission",
                            "dense_readout_overlap.propose_dense_readout_overlap",
                            "dense_readout_overlap.schedule_dense_readout_overlap",
                            "dense_readout_overlap.admit_readout_overlap_tile_loads",
                        )
                    ],
                    "usage": [
                        "Start with an original checked DenseSegmentedPipelineResult and owned storage.",
                        "Supply the separate readout/compute capability and an actual counted output group_path.",
                        "Revalidate the checked result, then materialize_scf or emit_llvm its actual paired kernel.",
                        "Compose wide loads only through admit_readout_overlap_tile_loads on final physical bands.",
                    ],
                    "requires": [
                        "separate selected store snapshot, backpressure, disjoint ACC and full-drain completion law",
                        "actual epoch hazards before exact counted original-source transport; odd tails retained",
                        "two aligned owned ACC windows; original quantization, input lifetimes and full output cover",
                        "no causal-ready same-window readout permission or capacity-derived execution grant",
                    ],
                },
                {
                    "name": "declared_persistent_scalar_state_in_scf",
                    "symbols": [
                        prefix + "scf_commands." + name
                        for name in (
                            "materialize_scf_commands",
                            "verify_scf_commands",
                            "unroll_small_command_loops",
                            "group_scf_command_loops",
                            "emit_scf_commands",
                        )
                    ]
                    + [prefix + "llvm_program.FnBuilderIndexAdapter", "merlin.llvmlower.host_builder.FnBuilder"],
                    "usage": [
                        "Attach closed signed-i64 Const sets/assumes to actual command Calls.",
                        "Pass allow_persistent_state=True explicitly at every SCF bridge boundary.",
                        "Print/parse or transform actual command-bearing SCF, reverify, then emit "
                        "through the callback.",
                    ],
                    "requires": [
                        "unknown initial state; preceding setter, matching value and compatible loop reentry",
                        "selected target visibility/clobber law remains NOT_ESTABLISHED by this scalar-state checker",
                        "explicit async dependencies, source, alias, completion and numerical obligations remain",
                    ],
                },
            ]
        )
    if version >= 19:
        catalog["capabilities"].append(
            {
                "name": "final_k_causal_accumulator_readout_frontier",
                "symbols": [
                    prefix + name
                    for name in (
                        "dense_panel_pipeline.DenseReadyReadoutAdmission",
                        "dense_panel_pipeline.DenseReadoutConfigurationSnapshot",
                        "dense_ready_readout.propose_dense_ready_readout",
                        "dense_ready_readout.schedule_dense_ready_readout",
                        "dense_ready_readout.schedule_dense_pipeline",
                        "dense_ready_readout.admit_ready_readout_tile_loads",
                    )
                ],
                "usage": [
                    "Start with the exact original checked segmented schedule and a separate causal target capability.",
                    "Reorder final pure compute through typed M/N/K transforms and append each original output store.",
                    "Revalidate actual producer-before-readout hazards and original source, then emit actual SCF/LLVM.",
                    "Varying tiled coefficients require an independently selected per-store snapshot contract bound "
                    "to the exact original readout identity; incompatible stores remain distinct.",
                    "Use the typed final-wide consumer to decode every physical band before source transport.",
                ],
                "requires": [
                    "selected overlapping RAW producer retirement and accumulator pending-write read readiness law",
                    "original store snapshots and finite-queue backpressure, full drain before any ownership reuse",
                    "all overlapping producer instances precede each store; no later aliased writer in the epoch",
                    "unchanged full ACC geometry, input lifetimes and original scalar readout boundaries",
                    "no mixed DMA/init in the new causal epoch, no fixed-cycle or loop-carried event permission",
                ],
            }
        )
    if version >= 20:
        catalog["capabilities"].append({
            "name": "loop_carried_completion_events",
            "symbols": [
                "merlin.sched.ir.kernel.Loop",
                prefix + "scf_commands.materialize_scf_commands",
                prefix + "scf_commands.verify_scf_commands",
                prefix + "scf_commands.emit_scf_commands",
                prefix + "llvm_program.FnBuilderIndexAdapter",
            ],
            "usage": [
                "Declare Loop.iter_args, yields and results with explicit completion ownership.",
                "Materialize actual runtime.event SCF iter_args/results and verify lexical and staging dependencies.",
                "Supply actual target launch/wait callbacks; carried emission requires builder.loop_carried.",
            ],
            "requires": [
                "dominating original incoming events, exact yields and complete external completion",
                "checked cross-iteration storage dependencies and bounded stage-state reentry",
                "separate selected target completion, storage, source and numerical contracts",
            ],
        })
    if version >= 21:
        catalog["capabilities"].append({
            "name": "effect_checked_schedule_composition",
            "symbols": [
                prefix + "configuration_motion.promote_configuration_writes",
                prefix + "configuration_motion.verify_configuration_motion",
                prefix + "completion_streams.fuse_completion_streams",
                prefix + "spatial_max_stripes.plan_striped_spatial_max",
            ],
            "usage": [
                "Apply transformations to the current Kernel and emit the returned product.",
                "Compare complete elapsed-cycle observations with matching source, ABI and model scope.",
            ],
            "requires": [
                "original effect projections, state dependencies and complete source/output coverage",
                "fact-derived storage bounds and independent target completion implementation",
            ],
        })
    if version >= 24:
        catalog["capabilities"].append({
            "name": "current_source_quantized_residual_edges",
            "symbols": ["merlin.llvmlower.quantized_sum_source.discover_quantized_sum_sources"],
            "usage": [
                "Discover current sole-use contraction-to-residual edges from actual host SSA.",
                "Inspect each unchanged producer/readout and residual/readout program and source shape.",
                "Try the target composition adapter before selecting an ordinary QuantizedSumRoute.",
                "Keep unsupported alternatives separate; compare complete region cycles and private workspace.",
            ],
            "requires": [
                "current ordinary DeviceRewrite and private complete borrowed source declarations",
                "fresh unobserved intermediate, preserved Q1/Q2, target effects and actual completion",
                "independent numerical execution and target admission; discovery supplies neither",
            ],
        })
    if version >= 23:
        catalog["capabilities"].append({
            "name": "capacity_phased_convolution_rhs_residency",
            "symbols": [prefix + "convolution_exploration." + name for name in (
                "ConvolutionRhsPhaseLaw", "ConvolutionRhsPhases", "plan_convolution_rhs_phases",
                "emit_convolution_schedule", "convolution_pass_space",
                "ConvolutionStorageAlternative", "plan_convolution_storage_alternative",
                "convolution_storage_pass_space",
            )],
            "usage": [
                "Keep full-fit RHS residency, else derive complete-K output-column phases from assigned capacity.",
                "Pass the complete owner and exact resident input_rows to emit_convolution_schedule; never emit "
                "phases alone.",
                "The registered rhs_modes=(False, True, 'phased') route retains original column tails and actual-byte"
                " dedup.",
                "Check phase_emissions against the requested owner, retaining real preparation and phase-last-use "
                "drains.",
                "Jointly explore caller-qualified resident input maps and output group bounds before freezing RHS "
                "phase capacity; rank actual emitted complete-call cycles.",
                "Supply rhs_column_limits=(None, ...) to convolution_storage_pass_space for explicit narrower "
                "RHS phases; None retains the maximal-capacity default. plan_convolution_rhs_phases uses "
                "max_column_blocks and plan_convolution_storage_alternative uses rhs_column_blocks.",
                "Derive a useful finite column bound from actual ACC rows divided by requested live spatial "
                "tiles times primitive rows; keep the real operand capacity and exact RHS/input placements.",
                "Retain structured spatial coordinates outside K/N loops; resident input emit_row_coordinates "
                "maps them directly without repeated flattened-coordinate decoding.",
            ],
            "requires": [
                "original independent output columns, immutable inputs and no intermediate output observers",
                "exclusive assigned RHS rows disjoint from the actual resident input footprint",
                "separate RHS preservation and ConvolutionRhsPhaseLaw; capacity alone grants no reordering",
                "real full completion before private RHS replacement; source/effect/numerical checks remain required",
            ],
        })
    if version >= 23:
        catalog["capabilities"].append({
            "name": "source_qualified_convolution_readout_epochs",
            "symbols": [prefix + "window_transfer.ConvolutionReadoutEpochLaw",
                        prefix + "convolution_exploration.emit_convolution_schedule"],
            "usage": [
                "Bind a separate selected source readout law; compute-epoch or dense readout laws do not imply it.",
                "Pass readout_law explicitly to the convolution consumer with original source/target callbacks.",
                "Verify the returned readout_epoch, source/schedule and final block drain match the requested law.",
                "Keep intermediate compute, rolling input-ready/last-use and whole-call completion drains.",
                "Inspect actual emitted effects, full outputs and same-domain complete-call cycles; UNKNOWN keeps the"
                " default.",
            ],
            "requires": [
                "counted convolution with issued compute epochs and disjoint original output/ACC intervals",
                "selected ACC RAW/result visibility, ordered store-configuration snapshots and finite queue progress",
                "separate finite pending store capacity and bounded compiler-issued readouts; no extra hardware slots",
                "private ownership and a real final drain before input/ACC/RHS reuse",
                "target effect/completion adapter; consumer receipts do not independently prove emitted instruction "
                "semantics",
            ],
        })
    if version >= 23:
        catalog["capabilities"].append({
            "name": "parallel_emitted_variant_shape_evaluation",
            "symbols": ["merlin.transforms.evaluated_exploration." + name for name in (
                "EvaluationRequest", "EvaluationCompletion", "evaluate_emitted", "explore_evaluated",
            )],
            "usage": [
                "Apply real registered passes and emit distinct alternatives with explore_candidates.",
                "Interleave requests from representative shapes; each retains its original source and timing domain.",
                "Use one caller-owned executor and ordinary resource-leased evaluator, with bounded max_in_flight.",
                "Keep completion publication compact; verification runs in the evaluator and cleanup off the hot path.",
                "Consume each completed result immediately; identical programs reuse only exact-domain evidence.",
                "Rank complete same-ROI cycles within each shape; UNKNOWN never becomes an instruction-count score.",
                "Promote measured shape/resource rules through shape_policy while retaining unaffected incumbents.",
            ],
            "requires": [
                "target-owned emitted-program execution adapter and original numerical/provider/ROI identities",
                "existing shared resource admission; this library creates no CPU pool or simulator",
                "input lifetime covering all evaluation workers and the completion callback",
            ],
        })
    if version >= 23:
        catalog["capabilities"].append({
            "name": "pointwise_convolution_dense_view",
            "symbols": [prefix + "pointwise_convolution.pointwise_dense_view"],
            "usage": [
                "Derive a zero-copy matrix view from actual unit-stride unpadded 1x1 NHWC source geometry.",
                "Pass the returned dimensions to shared dense planning, retention and scheduling.",
                "Use the existing target encoder and preserve bias, quantization and all output observers.",
                "Retain spatial lowering when geometry or storage is incompatible; "
                "compare emitted alternatives by cycles.",
            ],
            "requires": ["original Window2D geometry and exact result shape",
                         "physical operand layout and ABI agreement; "
                         "target effects and numerical checks remain required"],
        })
    if version >= 22:
        catalog["capabilities"].append({
            "name": "registered_source_convolution_exploration",
            "symbols": [prefix + "convolution_exploration." + name for name in (
                "ConvolutionExplorationSource", "ConvolutionScheduleChoice", "plan_convolution_schedule",
                "ConvolutionRhsResidencyLaw", "ConvolutionRhsResidency", "plan_convolution_rhs_residency",
                "emit_convolution_schedule", "convolution_pass_space", "explore_convolution_schedules",
            )],
            "usage": [
                "Bind original counted convolution, complete scalar-safe column intervals and selected private rows.",
                "Register bounded choices with original-source verification and target realization callbacks.",
                "Enable compute_pair_limits with a separate checked epoch law; stores still fully drain.",
                "Select rhs_modes with an independent immutable-weight law and explicit private operand rows.",
                "Emit invocation-local RHS preparation once, or retain original transfers when capacity declines.",
                "Consume the checked schedule through actual target emission; deduplicate exact program bytes.",
                "Compare complete same-source/model/window cycle feedback; no count-based default ranking.",
            ],
            "requires": [
                "separate original input-retention, delayed-load and compute-epoch laws for enabled transformations",
                "actual completion callbacks, private ACC ownership and original source custody",
                "ordinary source/dependency admission; registry/identities confer no hardware or numerical authority",
            ],
        })
    if version >= 22:
        catalog["capabilities"].append({
            "name": "selected_logical_kernel_argument_binding",
            "symbols": ["merlin.compile.logical_abi." + name for name in (
                "logical_abi", "logical_abi_from_document", "logical_interface", "kernel_arg_order",
            )] + ["merlin.llvmlower.host_builder.FnBuilder.for_logical_kernel"],
            "usage": [
                "Parse the selected public ABI declaration, then call FnBuilder.for_logical_kernel(cb, abi=abi).",
                "Use the returned source-name to SSA-pointer map; semantic lhs/rhs order is not ABI order.",
                "Implicit COMMIT and pooled result shapes share the runner's declaration-only derivation.",
            ],
            "requires": [
                "original command buffer and selected public ABI declaration; no inferred private pointer contract",
                "independent runtime ownership and numerical implementation; no harness/reference code is exported",
            ],
        })
    if version >= 22:
        catalog["capabilities"].append({
            "name": "shape_resource_schedule_selection",
            "symbols": ["merlin.transforms.shape_policy." + name for name in (
                "PerformanceDomain", "ShapeScheduleClass", "ShapeCaseFeedback", "assess_shape_policy",
                "AssessedShapePolicy.execute", "ShapePolicyProposal", "propose_shape_policies",
            )] + ["merlin.transforms.pipeline_policy." + name for name in (
                "DimensionRef", "ResourceRef", "RatioFeature", "TileFeature", "FeaturePredicate",
            )] + ["merlin.transforms.evaluated_exploration.explore_evaluated",
                  "merlin.transforms.pass_exploration.ExplorationRefused",
                  "merlin.transforms.pass_exploration.MappingProposalKey",
                  "merlin.transforms.pass_exploration.MappingProposalCache"],
            "usage": [
                "Register bounded real pass paths and an original fallback, with separate legality predicates.",
                "Freeze distinct training/heldout geometries and supply exact paired completed observations.",
                "Declare shape/resource performance guards; include exact cutoffs and adjacent outside cases.",
                "Propose bounded dimension cutoffs and two-axis regions from paired TRAINING cycles only.",
                "Use proposal.requests for missing pairs/boundaries; freeze new boundary geometries before timing.",
                "Express exact ratios, tile counts and tails from current shape/resource facts; unknown stays unknown.",
                "Training selects; heldout and boundary cases veto. UNKNOWN/regression keeps each class incumbent.",
                "Execute on current IR with its selected performance domain; no cached training kernel is reused.",
                "Reuse only option-search proposals in an explicit bounded MappingProposalCache keyed by "
                "semantics/shapes/layouts/dtypes/epilogue and hardware/library/numerics; current factories, "
                "legality and emission still run.",
                "Use existing evaluate_variants(prefix=...) for preparation shared by every alternative; "
                "it discards prefix analyses before current-source branch checks.",
                "Exploration outcomes distinguish duplicates, construction refusals, "
                "pending/UNKNOWN timing and regressions.",
            ],
            "requires": [
                "complete numerical-pass same-source/input/reference/provider/ROI cycle observations",
                "original registered pass contracts and independent source/resource/effect legality checks",
                "new normal inventory22 selection; empirical panel support is not universal or physical qualification",
            ],
        })
    if version >= 19:
        catalog["capabilities"].append({
            "name": "checked_dense_pipeline_entry",
            "symbols": [prefix + "dense_ready_readout.schedule_dense_pipeline",
                        prefix + "schedule_ir.make_dense_schedule_plan"],
            "usage": [
                "Reuse one DenseProblem/resources/storage/actual-operation adapter across typed pipeline decisions.",
                "Dispatch compact, retained, two-window or segmented plans through their original source checker.",
                "Optionally batch stores or select ready/disjoint readout with its separate execution law.",
                "Emit the returned checked result with lower_call(instruction, operands, metadata=None).",
                "Retain incumbent and alternative artifacts when timing is UNKNOWN; inspect and verify legal changes.",
            ],
            "requires": [
                "original source-bound plan, actual allocated windows, exact numerical/alias/effect boundaries",
                "independently selected finite issue/backpressure/snapshot/completion laws; capacity is insufficient",
                "lower every actual completion; an explicit unsupported decision refuses rather than changing family",
            ],
        })
    if version >= 22:
        catalog["capabilities"].append({
            "name": "checked_dense_completion_epoch_emission",
            "symbols": [prefix + name for name in (
                "dense_panel_pipeline.DensePanelPipelineResult",
                "dense_segmented_pipeline.DenseSegmentedPipelineResult",
                "dense_ready_readout.DenseReadyReadoutResult",
                "dense_readout_overlap.DenseReadoutOverlapResult",
                "llvm_program.emit_index_effect_program",
            )],
            "usage": [
                "Consume result.emit_index_program(control, lower_effect, complete_epoch=real_full_completion).",
                "Map each actual checked data effect separately from the original full epoch drain.",
                "Reuse one target encoder for panel, segmented, ready and disjoint schedules "
                "and real pass alternatives.",
                "Keep counted control; explicit event/state annotations still require their supported SCF consumer.",
                "The ordinary checked Kernel and every emitted epoch are revalidated before callbacks.",
            ],
            "requires": [
                "separate selected RAW/WAR/WAW, storage, operand/result retirement and finite-progress laws",
                "a real full completion callback; no primitive-by-primitive synthetic fence substitution",
                "original complete outputs and numerical policies; source emission grants no performance claim",
            ],
        })
    if version >= 22:
        catalog["capabilities"].append({
            "name": "fact_backed_dense_panel_epochs",
            "symbols": [prefix + name for name in (
                "dense_panel_epochs.DensePanelEpochLaw",
                "dense_panel_epochs.propose_dense_panel_epochs",
                "dense_panel_epochs.schedule_dense_panel_epochs",
                "dense_panel_epochs.DensePanelEpochResult",
                "dense_ready_readout.schedule_dense_pipeline",
            )],
            "usage": [
                "Pass epoch_law=DensePanelEpochLaw(...) to schedule_dense_pipeline with a two-window decision.",
                "Set each flag from the target's command-ordering facts; a False flag is UNKNOWN and keeps its drains.",
                "Drains inside one output block's K stream go (and before its readout under result visibility).",
                "Emit the returned result: emit_llvm/emit_index_program invoke completion only for remaining drains.",
                "Time the epochs against the drained incumbent before keeping them.",
            ],
            "requires": [
                "selected RAW (compute waits for older load) and WAR (load waits for older reader) ordering",
                "in-order accumulator updates; readout waiting for its producer to merge the readout epoch",
                "every load of a K step read by the next step (proved); merged epochs within the issued budgets",
                "a real full completion callback for every remaining drain",
            ],
        })
    if version >= 23:
        catalog["capabilities"].extend([
            {
                "name": "bounded_declared_settings_cycle_selection",
                "symbols": [
                    "merlin.compile.tuning_knobs.KnobSettings",
                    "merlin.compile.tuning_knobs.bounded_settings",
                    "merlin.transforms.tuning_selection.KnobTrial",
                    "merlin.transforms.tuning_selection.unique_emissions",
                    "merlin.transforms.tuning_selection.select_settings",
                ],
                "usage": [
                    "Enumerate a declared knob subset: exact baseline, single changes and optional bounded pairs.",
                    "Emit and revalidate current choices; unique_emissions deduplicates exact same-domain bytes.",
                    "Use the caller-owned ordinary evaluator and identity-bound CompletedObservation cycle feedback.",
                    "select_settings returns an exact-shape proposal; use shape_policy for validated range rules.",
                ],
                "requires": [
                    "original declaration, complete settings and semantic/shape/layout/dtype/numerical/hardware pins",
                    "same-source/input/reference/provider/ROI observations; complete passing cycles alone rank",
                    "UNKNOWN, failure, conflicts or no demonstrated improvement retain the unchanged baseline",
                ],
            },
            {
                "name": "checked_disjoint_loaded_sum_epochs",
                "symbols": [prefix + "loaded_sum_epochs." + name for name in (
                    "LoadedSumEpochLaw", "LoadedSumEpochPlan", "plan_loaded_sum_epochs", "emit_loaded_sum_epochs",
                )],
                "usage": [
                    "Consume the original complete compact_loaded_sum source/scalar proof and ACC byte layout.",
                    "Plan bounded disjoint tiles; emit initialize, accumulate and store phases with real drains.",
                    "The caller-owned target encoder lowers every effect and complete_phase before private reuse.",
                ],
                "requires": [
                    "independent target pending-operation/configuration capture/progress/drain/ownership premises",
                    "original exact parameters, complete outputs, tails and disjoint live private byte windows",
                    "source/storage construction grants no target execution or performance qualification",
                ],
            },
            {
                "name": "checked_initialization_and_uniform_readout_batches",
                "symbols": [prefix + name for name in (
                    "dense_panel_pipeline.DenseRetainedReductionAdmission",
                    "dense_contracts.DenseTiledIntegerReadout",
                    "dense_panel_readout.batch_dense_panel_initializers",
                    "dense_panel_readout.batch_dense_panel_readouts",
                    "dense_panel_pipeline.verify_dense_retained_reduction",
                )],
                "usage": [
                    "Start with an original checked retained result and its actual source/effect adapter.",
                    "Select bounded initializer issue only with ordered configuration capture and disjoint ACC writes.",
                    "Batch initializers through the final checker; keep the real completion before computation.",
                    "Normalize an original scalar/vector with DenseTiledIntegerReadout.from_source_scales; never "
                    "average gains.",
                    "Uniform original tiled integer scales may batch stores under the selected snapshot/store law.",
                    "Emit the returned kernel and compare complete cycles; fewer drains alone prove no speedup.",
                ],
                "requires": [
                    "original symbolic data-call equality, source numerics, aliases, capacities and completion proofs",
                    "varying-scale readouts retain their original boundaries or require a separate supported law",
                ],
            },
            {
                "name": "source_exact_spatial_matrix_gathers",
                "symbols": [prefix + "affine_gather_runs." + name for name in (
                    "ConvolutionGatherRuns", "emit_convolution_gather_runs", "emit_convolution_matrix_gather",
                )],
                "usage": [
                    "Map logical contraction rows and reduction tiles to original valid-window NHWC addresses.",
                    "Use reduction tiles dividing the source channel dimension; preserve source tap/channel order.",
                    "Pass exact fragments to the ordinary target consumer without allocating host im2col payloads.",
                ],
                "requires": [
                    "original checked window geometry and independent dtype, storage, alias and completion admission",
                    "fragment construction supplies no permission to overlap transfers or change arithmetic",
                ],
            },
            {
                "name": "source_bound_rolling_convolution_rows",
                "symbols": [prefix + name for name in (
                    "window_transfer.ConvolutionResidentInputRows",
                    "convolution_exploration.emit_convolution_schedule",
                    "rolling_window.materialize_rolling_window",
                    "rolling_window.emit_rolling_window",
                    "rolling_window.materialize_grouped_rolling_window",
                    "rolling_window.verify_grouped_rolling_window",
                    "rolling_window.emit_grouped_rolling_window",
                )],
                "usage": [
                    "Bind original NHWC source cells to caller-owned tiled operand rows and a checked ring lifetime.",
                    "Pass input_rows to the ordinary convolution schedule emitter; consume all staging effects.",
                    "Derive transfer widths, row pitch and private storage from selected hardware/source contracts.",
                    "Padded integer convolution requires explicit original zero halos, not a host padded tensor.",
                    "Keep real input-ready and last-use completion; compare whole-call cycles and resource traffic.",
                ],
                "requires": [
                    "exact original source identity, source stride and disjoint live operand storage",
                    "nonunit compute row steps require an explicitly selected source/target row-step contract",
                    "target implementations of actual stage/fill effects and completion; mapping grants no overlap",
                    "nonzero affine padding values require a separately supported source and lowering contract",
                ],
            },
            {
                "name": "checked_cross_spatial_array_reuse",
                "symbols": [prefix + name for name in (
                    "convolution_output_groups.plan_rolling_convolution_output_group",
                    "convolution_output_groups.emit_rolling_convolution_output_group",
                    "dense_array_retention.DenseArrayRetentionAdmission",
                    "convolution_exploration.emit_convolution_schedule",
                )],
                "usage": [
                    "Derive source-row/M bounds from the original schedule, resident RHS or complete phase owner, and"
                    " live ring.",
                    "Pass the group or phase-aligned tuple and separate array_law to emit_convolution_schedule; None "
                    "retains an ordinary phase.",
                    "Consume actual fresh_weights through the same target encoder; preserve each original N readout.",
                    "Verify spatial_output_group and its source/law hashes; keep the incumbent when planning declines.",
                    "With a checked multirow ring union, grouped_transport=True completes staging once before the "
                    "actual consumer group and drains before reuse.",
                ],
                "requires": [
                    "resident full-K RHS per active original phase, checked source-row union in actual ring slots and"
                    " disjoint M/N ACC cells",
                    "separate independent-output-row/no-intermediate-observer source permissions",
                    "selected fresh/stay/output-descriptor behavior and actual bounded compute/readout completion",
                    "default one-row path retained; real group/ring drains and full phase-last-use before RHS "
                    "replacement",
                    "grouped transport separately checks actual row/halo/channel command count against the selected "
                    "issue bound",
                ],
            },
            {
                "name": "source_exact_convolution_readout_transfers",
                "symbols": [prefix + name for name in (
                    "window_transfer.ConvolutionReadoutTransferLaw",
                    "window_transfer.plan_convolution_readout_transfers",
                    "convolution_exploration.emit_convolution_schedule",
                )],
                "usage": [
                    "Bind a separate selected narrow block-column/row-stride map and original numerical states.",
                    "Pass readout_transfer_law with the independent producer/readout law to the original consumer.",
                    "Only compatible original adjacent stores merge; inspect actual emitted stores and complete "
                    "outputs.",
                    "Compose with checked spatial groups using each spatial slot's actual ACC block map, including "
                    "short source rows and RHS phases.",
                ],
                "requires": [
                    "scaled INT8 source exits; exact controller map, output columns and private physical ACC spacing",
                    "identical original numerical state within each combined run; full ACC and pooling excluded",
                    "separate producer visibility/store snapshot law and real final private-block drain",
                    "missing target/model support remains UNKNOWN, not an inferred speedup or qualification",
                ],
            },
            {
                "name": "checked_shared_input_group_blocks",
                "symbols": [prefix + "compact_dense_group." + name for name in (
                    "CompactDenseGroupResult", "schedule_compact_dense_group",
                )],
                "usage": [
                    "Select explicit per-member block_tiles while retaining the original shared input producer.",
                    "Keep independent output scales and complete storage/numerical/source checks for every member.",
                    "Unsupported requested geometry declines; compare whole-group cycles with independent execution.",
                ],
                "requires": [
                    "true original operand identity; similar values or fewer transfers establish no reuse law",
                ],
            },
        ])
    if version >= 24:
        catalog["capabilities"].append({
            "name": "factored_immutable_tensor_layouts",
            "symbols": [prefix + "tensor_storage." + name for name in (
                "FactoredTensorStorage", "column_panel_storage",
            )],
            "usage": [
                "Factor original logical axes and select their physical order from the target's transfer/storage contract.",
                "Carry the same checked map through actual constant preparation, call-buffer types, target addresses and readback.",
                "Prepare source-owned immutable bytes once; include preparation in comparisons for mutable or invocation-local operands.",
                "Inspect emitted transfer addresses and strides; a layout declaration alone does not change target code.",
            ],
            "requires": [
                "exact positive source factors, a complete dense permutation and original byte extent",
                "independent immutable-source authorization and an executable consumer of the declared physical map",
                "original arithmetic, column coefficients and observers; unsupported tails or affine-only consumers keep their original layout",
                "compare whole-region cycles and traffic with the actual prepared constant; no preferred panel width is supplied",
            ],
        })
        catalog["capabilities"].append({
            "name": "completed_input_with_rhs_frontier",
            "symbols": [prefix + "convolution_input_reuse." + name for name in (
                "ConvolutionInputReuseAdmission", "select_convolution_input_reuse",
                "InvariantConvolutionInput",
            )] + [prefix + "rolling_lookahead.RollingRhsFrontierAdmission",
                  prefix + "convolution_exploration.emit_convolution_schedule"],
            "usage": [
                "Establish a completed original full-input union through the first actual RHS phase.",
                "Select invariant_input_law and rhs_phase_prefetch_law independently, then pass both to the same ordinary emitter.",
                "The checked consumer advances complete reduction runs before issuing future RHS writes into their dead prefix.",
                "Keep all original preparation, group, input-last-use, phase and final drains; inspect actual reader/compute overlap and cycles.",
            ],
            "requires": [
                "the complete original input union fits its injective assigned rows and remains immutable",
                "every future RHS write is disjoint from retained input under its exact source/phase map",
                "separate target RAW/WAR, ordered configuration, queue progress and real completion contracts",
                "one complete column consumer per phase; unsupported compositions retain separate schedules",
            ],
        })
        catalog["capabilities"].append({
            "name": "capacity_derived_m_panels_with_rhs_reuse",
            "symbols": [prefix + "dense_panel_pipeline." + name for name in (
                "dense_m_panel_capacity", "schedule_dense_m_panels", "DenseMPanelAttempt",
            )],
            "usage": [
                "Select the original N width; derive M limits from assigned padded LHS/RHS/ACC rows and byte capacity.",
                "Construct one counted schedule with schedule_dense_m_panels; RHS staging is hoisted across M groups.",
                "Inspect per-shape capacity/issue/tail blockers and checked load counts; compare whole-call cycles.",
                "A constructed schedule is not an emitted mechanism. Reopen actual emission before crediting transfer.",
            ],
            "requires": [
                "original private rows, full-K fit, supported tails and selected ordered-reduction/read-only law",
                "source-bound scheduling dialect decisions and independent final source/storage/epoch checks",
                "partial M families keep their explicit reloads; array STAY needs a separate selected target law",
                "missing capacity or laws decline before emission; concrete verification failures propagate",
            ],
        })
        catalog["capabilities"].append({
            "name": "retained_ready_input_placement",
            "symbols": [prefix + "dense_ready_readout." + name for name in (
                "DenseReadyInputPlacement", "dense_ready_input_choices", "schedule_dense_pipeline",
            )],
            "usage": [
                "Filter retained-weight full-K choices for two advancing LHS slots, one RHS and one ACC panel.",
                "Build the selected ordinary plan and pass the SAME input_pipeline_placement to the ready consumer.",
                "Actual projected accesses, wide-load composition and revalidation consume the same placement.",
                "Optionally pass a separately selected array_prefix_admission: retain pure K-prefix K,N,M runs, "
                "then issue only fresh final-K pairs and their original causal stores.",
            ],
            "requires": [
                "original owned row views, padded physical footprints and combined operand byte capacity",
                "at least two complete advancing groups and the separate selected load/ready conjunction",
                "array prefix needs at least two K slices and a reusable M run; "
                "no STAY overlaps stores and no store-preserves-array premise is inferred",
                "explicit plans never shrink; counted source/products/completion checks "
                "and target qualification remain",
            ],
        })
        catalog["capabilities"].append({
            "name": "staged_spatial_producer_bands",
            "symbols": [prefix+"spatial_producer."+name for name in (
                "SpatialProducerBand","SpatialProducerBandAdmission","SpatialProducerBandAlternative",
                "spatial_producer_band_choices","emit_spatial_producer_band",
                "spatial_producer_pass_space","explore_spatial_producer",
            )]+[prefix+"convolution_exploration.emit_convolution_rhs_preparation"],
            "usage": [
                "Match the actual adjacent convolution/pool source and retain both original rounded readouts and every external observer.",
                "Enumerate band heights from assigned ACC and Q1 scratch capacity; inspect exact halo recomputation and MAC costs.",
                "Emit the producer union, actual Q1 stores/reloads and a complete multirow Q2 pool rectangle through selected target callbacks.",
                "Optionally compose an existing complete immutable RHS owner; real preparation and final completion remain required.",
                "Use the ordinary pass registry to explore independent band/RHS choices and legal planning orders, deduplicate actual emitted bytes, "
                "then compare exact whole-region cycles. Unknown cycles remain unranked.",
                "The ordinary connected host route uses the original band.source plan and full assigned scratch; rolling packed/readout-cache "
                "paths retain their separate representation prerequisites.",
            ],
            "requires": [
                "original pure producer with no additional live intermediate observer; original Q1/Q2 scalar coefficients and signed padding",
                "separate source-bound rectangular pooling address/configuration and command-tracking law with actual field limits",
                "exclusive assigned ACC/Q1 storage and actual completion of the entire pooled rectangle before reuse",
                "actual original source callback and native execution; this library supplies no host effects or target qualification",
            ],
        })
        catalog["capabilities"].append({
            "name": "source_checked_integer_observers",
            "symbols": [
                "merlin.llvmlower.late_quant_rne.rewrite",
                "merlin.llvmlower.source_expression_interval.close_scalar_i8_observer",
                "merlin.llvmlower.source_expression_interval.build_source_interval_table",
                "merlin.llvmlower.source_expression_interval.emit_source_interval_i8_lookup",
                "merlin.llvmlower.source_expression_interval_llvm.rewrite_source_interval_i8_lookup",
                "merlin.llvmlower.bounded_rne_word_cells.prepare_bounded_rne_word_cells",
                "merlin.llvmlower.scaled_integer_finite.prove_broadcast_scaled_integer_finite",
                "merlin.llvmlower.scaled_integer_finite_llvm.bind_finite_scale_helpers",
            ],
            "usage": [
                "Bind exact typed scalar arithmetic and complete saturated signed-byte observations.",
                "Retain ordinary bounded roundeven representation and the existing late host legalizer.",
                "Derive finite broadcast-scale scans and immutable tables from the current source only.",
                "Consume returned LLVM/C in the ordinary compiler before object identity; "
                "validate the complete output.",
            ],
            "requires": [
                "explicit RNE, gradual underflow, nontrapping and unobserved floating effects",
                "original source fallback, actual incoming-mode guard and exact original coefficients/order",
                "immutable input scales and fresh disjoint outputs for finite preparation",
                "unknown source operations such as exp require their own numerical contract; no approximation granted",
                "no target/provider, reference output, protected evaluator or timing authority is exported",
            ],
        })
        catalog["capabilities"].append({
            "name": "counted_projection_pointer_binding",
            "symbols": [prefix + "projection_dense." + name for name in (
                "ProjectionDenseBinding", "bind_dense_projection",
            )],
            "usage": [
                "Keep original sibling projection_views, independent pointers and exact scalar readouts.",
                "Retain the common input with the existing dense scheduler; choose panels dividing every member.",
                "Partition complete virtual-N panel loops with ordinary ordered loop transforms.",
                "Bind actual member/local-column/row-width operands to original target ABI pointers and pitches.",
            ],
            "requires": [
                "current typed source group and an independently checked original dense schedule",
                "complete original per-column readout words; physical rectangles cannot cross members",
                "original source aliases, private storage, actual target effects and real completion",
                "no tensor concatenation, dynamic command expansion or execution grant follows from the mapping",
            ],
        })
        catalog["capabilities"].append({
            "name": "completion_bound_operand_slots",
            "symbols": [prefix + "operand_slots." + name for name in (
                "PipelineOperand", "PipelineSlotResult", "allocate_pipeline_operand_slots",
                "verify_pipeline_operand_slots",
            )],
            "usage": [
                "Group original immutable TemporaryObject generations by operand, retaining actual views.",
                "Supply StorageAdapter projections for initialization, reads, launches and real waits.",
                "Use the returned Kernel: existing allocation chooses bases and checked phase loops rotate them.",
                "Materialize/emit the actual returned SCF and compare whole-call timing independently.",
            ],
            "requires": [
                "one bounded source counted loop with integer-decoded completion operands and owned byte banks",
                "complete original effects, aliases, reservations, alignments and external source/output windows",
                "already legal issue order; no new overlap permission or hardware slots follow from allocation",
                "nested or SSA-carried asynchronous boundaries and irregular rotations currently refuse",
            ],
        })
        catalog["capabilities"].append({
            "name": "disjoint_future_row_transport",
            "symbols": [prefix + "rolling_lookahead." + name for name in (
                "RollingLookaheadAdmission", "materialize_lookahead_rolling_window",
                "verify_lookahead_rolling_window", "emit_lookahead_rolling_window",
            )] + [prefix + "convolution_output_groups.ConvolutionGroupContinuation"],
            "usage": [
                "Keep the original rolling source, row ownership, and current consumer unchanged.",
                "Use actual ring capacity for the union of current and future live rows.",
                "Emit counted SCF with carried resident handles and real transport/consumer completion.",
                "Optionally derive consumer_runs = kernel_y * kernel_x * ceil(input_channels/channel_tile) "
                "from the original consumer; pass it unchanged to materialize, verify and emit.",
                "Bind start_consumer to initialization and a retained continuation; advance_consumer "
                "emits first_run/run_count complete reductions; finish_consumer performs final readout.",
                "For rolling convolution use emit_rolling_convolution_output_group(..., defer_consumer=True); "
                "its returned continuation checks advance(first, count) coverage before finish().",
                "For a complete invocation pass input_prefetch_law=RollingLookaheadAdmission(...) to "
                "emit_convolution_schedule with its checked output group and grouped transport; original K runs "
                "are derived from source. The separately selected RHS-frontier mode remains an alternative.",
                "Compare complete-call cycles; earlier issue alone does not establish hidden latency.",
            ],
            "requires": [
                "disjoint current/future writes and an immutable original input",
                "independently bound ordered configuration, finite queue progress, and completion callbacks",
                "split consumers additionally require retained_consumer_continuations=True, live ACC/input "
                "storage, complete fresh/stay runs and exactly one original initialization/readout",
                "no arbitrary nested asynchronous schedule or fixed-cycle completion is implied",
            ],
        })
        catalog["capabilities"].append({
            "name": "direct_ring_convolution_row_fragments",
            "symbols": [prefix + "convolution_output_groups." + name for name in (
                "ConvolutionRowFragmentGroup", "plan_rolling_convolution_row_fragments",
                "emit_rolling_convolution_output_group",
            )],
            "usage": [
                "Request a capacity-safe source-row band with plan_rolling_convolution_row_fragments.",
                "Pass its checked group to emit_convolution_schedule with the original rolling input, "
                "array/readout laws and real completions; the default whole-row planner is unchanged.",
                "Existing ConvolutionGatherRuns splits direct ring reads at original row boundaries. "
                "Fragments compute into disjoint subrows of packed ACC tiles; readout covers each complete original contiguous interval.",
                "Compare actual fragment/preload count, padded work, bytes and complete-call cycles; "
                "fewer ACC/readout tiles can require more compute fragments.",
            ],
            "requires": [
                "primitive-sized original row tiles, complete source-row bands and the actual staged ring horizon",
                "original immutable input/RHS, independent output rows and separately selected array/readout laws",
                "all original products, per-output reduction order, scalar exits, real pair boundaries and final drains",
                "no extra DMA bytes, local copy instruction or packed LHS allocation is inferred",
            ],
        })
        catalog["capabilities"].append({
            "name": "last_use_rhs_phase_transport",
            "symbols": [prefix + "rolling_lookahead." + name for name in (
                "RollingRhsFrontierAdmission", "RollingRhsFrontier",
            )] + [prefix + "convolution_exploration.emit_convolution_schedule"],
            "usage": [
                "Plan complete original RHS phases and their checked rolling output groups using actual SP/ACC capacity.",
                "Supply rhs_phase_prefetch_law=RollingRhsFrontierAdmission(...) to emit_convolution_schedule "
                "with grouped_transport=True and the original phase-aligned output_group tuple.",
                "The emitter derives the original reduction run count and stages each next-phase tile after "
                "the complete current K/N/M run in only the final source batch and spatial group.",
                "The next phase uses the same assigned RHS prefix; its column-block count must not grow. "
                "All original preparation, group, phase and final drains remain.",
                "Compare exact outputs and complete-call cycles with the same placement and numerical source.",
            ],
            "requires": [
                "separate selected target law covering exact-range scratchpad WAR/RAW until real source completion",
                "original immutable weights, ordered configuration, finite queue progress and full consumer/transport drain",
                "complete original batch/spatial domain and original channel tails; no second ACC or RHS window",
            ],
        })
        catalog["capabilities"].append({
            "name": "sole_observer_contraction_rows",
            "symbols": [
                "merlin.llvmlower.contraction_row_slice.pushback_contraction_row_slices",
                "merlin.llvmlower.contraction_row_slice.prepare_contraction_row_slices",
            ],
            "usage": [
                "Before source/device binding, push back a sole static contiguous row observer.",
                "Supply original immutable argument bytes for stored epilogue coefficients when needed.",
                "Retain the original ordered integer dot and each selected scalar epilogue instruction.",
            ],
            "requires": [
                "closed zero-seeded signed integer product; complete reduction and output columns",
                "static row-major views, fixed coefficients, pure parallel single-observer epilogue",
                "new ordinary source/interface/target qualification; no old device-ABI transfer",
            ],
        })
        catalog["capabilities"].append({
            "name": "bounded_horizontal_convolution_panels",
            "symbols": [prefix + "horizontal_convolution." + name for name in (
                "HorizontalConvolutionPanels", "plan_horizontal_convolution_panels",
                "emit_horizontal_convolution_panels",
            )],
            "usage": [
                "Bind the original CountedConvolutionPlan and explicit private operand/accumulator capacities.",
                "Plan bounded contiguous horizontal reductions and complete output-column intervals.",
                "Materialize the original zero halo; emit counted effects with actual completion before reuse.",
            ],
            "requires": [
                "unchanged source reduction order, output precision, bias and every tail output",
                "separate multiblock transport and fresh/stay array laws; private ownership is not a target grant",
                "complete-call cycles including halo/preparation cost; no speedup follows from transfer counts",
            ],
        })
        catalog["capabilities"].append({
            "name": "source_proven_shared_rhs_row_views",
            "symbols": [
                "merlin.llvmlower.weight_prepack_runtime.fixed_arguments",
                "merlin.llvmlower.repeated_pointwise.hoist_repeated_pointwise",
                "merlin.llvmlower.repeated_pointwise.prepare_repeated_pointwise",
                "merlin.transforms.shared_rhs_views.plan_shared_rhs_row_views",
                "merlin.transforms.dense_row_batching.DenseRowBatchingBudgets",
                "merlin.transforms.dense_row_batching.plan_source_dense_row_batching",
            ],
            "usage": [
                "Before device grouping, use exact broadcast/view/scalar source to prepare repeated pointwise work.",
                "Stored coefficients require fixed_arguments from the original generated caller "
                "plus exact bound bytes.",
                "Expose complete matrix groups with plan_shared_rhs_row_views, then use checked dense row batching.",
                "Retain every original result and dense observer; "
                "revalidate actual source/SSA before consuming output.",
            ],
            "requires": [
                "source-proven coordinates and unchanged scalar programs; equal runtime values grant no sharing",
                "fresh source/session/manifest ownership for immutable specialization; literals need no loader",
                "new ordinary source/interface/target/effects and execution qualification after regrouping",
                "late host preparation after device grouping cannot revise an already selected device ABI",
            ],
        })
        catalog["capabilities"].append({
            "name": "shared_rhs_epochs",
            "symbols": [prefix + "dense_shared_rhs." + name for name in (
                "schedule_dense_shared_rhs", "DenseRHSReuseLaw",
                "make_shared_rhs_schedule_plan", "read_shared_rhs_schedule_plan",
            )],
            "usage": [
                "Bind original sibling products sharing one immutable RHS, every member pointer and every live output.",
                "Consume a schedule dialect plan; use capacity-derived blocks and full-K LHS residency.",
                "Preserve original readout contracts, including scaled narrow outputs and qualification identities.",
                "Select dependency epochs from explicit target laws; UNKNOWN premises retain their drains.",
                "Register an A/B alternative through shared_rhs_epoch_recipe.py:prepare and emit the checked result.",
                "Reanalyze operands, transfers, configuration and completions; retain measured correct improvements.",
            ],
            "requires": [
                "original source-product equality, complete output roster and independent storage/alias admission",
                "selected immutable-RHS reuse, ordering and readout laws; no name-based applicability",
                "actual emitted-effect/source checks and a real final completion; command counts establish no speedup",
            ],
        })
    return catalog
