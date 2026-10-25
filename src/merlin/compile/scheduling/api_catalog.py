"""Source-only scheduling discovery for ordinary reviewed author libraries.

The catalog supplies callable locations and composition requirements. It
selects no target law, policy, benchmark, simulator, timing answer or grant.
"""

from __future__ import annotations


def scheduling_capabilities(version=16):
    """Return fresh usage metadata; the selected library contract grants access."""
    if type(version) is not int or version not in (16, 17, 18, 19):
        raise ValueError("scheduling discovery requires explicit inventory 16, 17, 18 or 19")
    prefix = "merlin.compile.scheduling."
    catalog = {
        "schema": "merlin.scheduling_api_catalog.v1",
        "inventory_version": 16,
        "authority": "discovery only; explicit source/effects/target admission remains required",
        "execution": "same Python/xDSL APIs in ordinary automatic and offline authoring",
        "capabilities": [
            {
                "name": "segmented_k_retained_operand_pipeline",
                "symbols": [prefix + name for name in (
                    "dense_segmented_pipeline.DenseSegmentedAdmission",
                    "dense_segmented_pipeline.schedule_dense_segmented_pipeline",
                    "dense_segmented_pipeline.build_dense_segmented_pipeline",
                    "dense_segmented_pipeline.verify_dense_segmented_pipeline",
                    "dense_segmented_pipeline.admit_segmented_tile_loads",
                    "dense_segmented_pipeline.batch_dense_segmented_readouts",
                    "schedule_ir.make_dense_schedule_plan",
                )],
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
                "symbols": [prefix + name for name in (
                    "dense_array_retention.DenseArrayRetentionAdmission",
                    "dense_array_retention.propose_dense_array_retention",
                    "dense_array_retention.verify_dense_array_retention",
                    "dense_array_retention.verify_dense_array_tile_loads",
                )],
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
                "symbols": [prefix + name for name in (
                    "tile_load_coalescing.TileLoadAddressLaw",
                    "tile_load_coalescing.TileLoadMapping",
                    "tile_load_coalescing.propose_tile_load_coalescing",
                    "panel_tile_loads.expand_dense_panel_tile_loads",
                    "panel_tile_loads.admit_panel_tile_loads",
                )],
                "usage": [
                    "Decode the final emitted call, not an earlier primitive access list.",
                    "Project every physical band with its exact source pitch, local stride and completion epoch.",
                    "Use the typed panel, segmented or array consumer for complete final source/hazard checks.",
                ],
                "requires": ["independently selected address law; address projection grants no execution semantics"],
            },
            {
                "name": "source_axis_loop_scheduling",
                "symbols": [prefix + name for name in (
                    "dense_panel_pipeline.dense_retained_compute_nests",
                    "dense_panel_pipeline.dense_retained_compute_order",
                    "dense_panel_pipeline.dense_retained_compute_unroll",
                    "loop_schedule.LoopTransform",
                    "loop_schedule.propose_loop_transforms",
                )],
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
        catalog["capabilities"].append({
            "name": "source_exact_quantized_spatial_readout_cache",
            "symbols": [prefix + "spatial_readout_cache." + name for name in (
                "SpatialReadoutCache", "propose_spatial_readout_cache", "emit_cached_spatial_max",
            )],
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
        })
    if version >= 18:
        from merlin.compile.composition_catalog import composition_capabilities
        from merlin.compile.scheduling.library_sources import library_modules

        catalog["composition"] = composition_capabilities(selected_modules=library_modules(version))
        catalog["capabilities"].extend([
            {
                "name": "disjoint_accumulator_readout_compute",
                "symbols": [prefix + name for name in (
                    "dense_panel_pipeline.DenseReadoutComputeAdmission",
                    "dense_readout_overlap.propose_dense_readout_overlap",
                    "dense_readout_overlap.schedule_dense_readout_overlap",
                    "dense_readout_overlap.admit_readout_overlap_tile_loads",
                )],
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
                "symbols": [prefix + "scf_commands." + name for name in (
                    "materialize_scf_commands", "verify_scf_commands", "unroll_small_command_loops",
                    "group_scf_command_loops", "emit_scf_commands",
                )] + [prefix + "llvm_program.FnBuilderIndexAdapter", "merlin.llvmlower.host_builder.FnBuilder"],
                "usage": [
                    "Attach closed signed-i64 Const sets/assumes to actual command Calls.",
                    "Pass allow_persistent_state=True explicitly at every SCF bridge boundary.",
                    "Print/parse or transform actual command-bearing SCF, reverify, then emit through the callback.",
                ],
                "requires": [
                    "unknown initial state; preceding setter, matching value and compatible loop reentry",
                    "selected target visibility/clobber law remains NOT_ESTABLISHED by this scalar-state checker",
                    "explicit async dependencies, source, alias, completion and numerical obligations remain",
                ],
            },
        ])
    if version == 19:
        catalog["capabilities"].append({
            "name": "final_k_causal_accumulator_readout_frontier",
            "symbols": [prefix + name for name in (
                "dense_panel_pipeline.DenseReadyReadoutAdmission",
                "dense_ready_readout.propose_dense_ready_readout",
                "dense_ready_readout.schedule_dense_ready_readout",
                "dense_ready_readout.admit_ready_readout_tile_loads",
            )],
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
        })
    return catalog
