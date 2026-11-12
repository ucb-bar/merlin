"""Dense command scheduling, retention, completion and shared-input contracts.

Pure discovery metadata; no admission or execution authority.
"""

from __future__ import annotations


def records(prefix):
    """Build fresh records in their original family order."""
    return (
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
        },
        {
            "name": "loop_carried_completion_events",
            "symbols": [
                "merlin.compiler.kernel_ir.ir.Loop",
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
        },
        {
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
        },
        {
            "name": "checked_dense_pipeline_entry",
            "symbols": [
                prefix + "dense_ready_readout.schedule_dense_pipeline",
                prefix + "schedule_ir.make_dense_schedule_plan",
            ],
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
        },
        {
            "name": "checked_dense_completion_epoch_emission",
            "symbols": [
                prefix + name
                for name in (
                    "dense_panel_pipeline.DensePanelPipelineResult",
                    "dense_segmented_pipeline.DenseSegmentedPipelineResult",
                    "dense_ready_readout.DenseReadyReadoutResult",
                    "dense_readout_overlap.DenseReadoutOverlapResult",
                    "llvm_program.emit_index_effect_program",
                )
            ],
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
        },
        {
            "name": "fact_backed_dense_panel_epochs",
            "symbols": [
                prefix + name
                for name in (
                    "dense_panel_epochs.DensePanelEpochLaw",
                    "dense_panel_epochs.propose_dense_panel_epochs",
                    "dense_panel_epochs.schedule_dense_panel_epochs",
                    "dense_panel_epochs.DensePanelEpochResult",
                    "dense_ready_readout.schedule_dense_pipeline",
                )
            ],
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
        },
        {
            "name": "checked_disjoint_loaded_sum_epochs",
            "symbols": [
                prefix + "loaded_sum_epochs." + name
                for name in (
                    "LoadedSumEpochLaw",
                    "LoadedSumEpochPlan",
                    "plan_loaded_sum_epochs",
                    "emit_loaded_sum_epochs",
                )
            ],
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
            "symbols": [
                prefix + name
                for name in (
                    "dense_panel_pipeline.DenseRetainedReductionAdmission",
                    "dense_panel_contracts.DenseTiledIntegerReadout",
                    "dense_panel_readout.batch_dense_panel_initializers",
                    "dense_panel_readout.batch_dense_panel_readouts",
                    "dense_panel_pipeline.verify_dense_retained_reduction",
                )
            ],
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
            "name": "checked_shared_input_group_blocks",
            "symbols": [
                prefix + "compact_dense_group." + name
                for name in (
                    "CompactDenseGroupResult",
                    "schedule_compact_dense_group",
                )
            ],
            "usage": [
                "Select explicit per-member block_tiles while retaining the original shared input producer.",
                "Keep independent output scales and complete storage/numerical/source checks for every member.",
                "Unsupported requested geometry declines; compare whole-group cycles with independent execution.",
            ],
            "requires": [
                "true original operand identity; similar values or fewer transfers establish no reuse law",
            ],
        },
        {
            "name": "capacity_derived_m_panels_with_rhs_reuse",
            "symbols": [
                prefix + "dense_panel_pipeline." + name
                for name in (
                    "dense_m_panel_capacity",
                    "schedule_dense_m_panels",
                    "DenseMPanelAttempt",
                )
            ],
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
        },
        {
            "name": "retained_ready_input_placement",
            "symbols": [
                prefix + "dense_ready_readout." + name
                for name in (
                    "DenseReadyInputPlacement",
                    "dense_ready_input_choices",
                    "schedule_dense_pipeline",
                )
            ],
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
        },
        {
            "name": "shared_rhs_epochs",
            "symbols": [
                prefix + "dense_shared_rhs." + name
                for name in (
                    "schedule_dense_shared_rhs",
                    "DenseRHSReuseLaw",
                    "make_shared_rhs_schedule_plan",
                    "read_shared_rhs_schedule_plan",
                )
            ],
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
        },
    )
