"""Source-only scheduling discovery for ordinary reviewed author libraries.

The catalog supplies callable locations and composition requirements. It
selects no target law, policy, benchmark, simulator, timing answer or grant.
"""

from __future__ import annotations


def scheduling_capabilities(version=16):
    """Return fresh usage metadata; the selected library contract grants access."""
    if type(version) is not int or version not in (16, 17, 18, 19, 20, 21, 22):
        raise ValueError("scheduling discovery requires explicit inventory 16, 17, 18, 19, 20, 21 or 22")
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
                        "dense_segmented_pipeline.schedule_dense_segmented_pipeline",
                        "dense_segmented_pipeline.build_dense_segmented_pipeline",
                        "dense_segmented_pipeline.verify_dense_segmented_pipeline",
                        "dense_segmented_pipeline.admit_segmented_tile_loads",
                        "dense_segmented_pipeline.batch_dense_segmented_readouts",
                        "schedule_ir.make_dense_schedule_plan",
                    )
                ],
                "usage": [
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
                "AssessedShapePolicy.execute",
            )] + ["merlin.transforms.evaluated_exploration.explore_evaluated"],
            "usage": [
                "Register bounded real pass paths and an original fallback, with separate legality predicates.",
                "Freeze distinct training/heldout geometries and supply exact paired completed observations.",
                "Declare shape/resource performance guards; include exact cutoffs and adjacent outside cases.",
                "Training selects; heldout and boundary cases veto. UNKNOWN/regression keeps each class incumbent.",
                "Execute on current IR with its selected performance domain; no cached training kernel is reused.",
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
    return catalog
