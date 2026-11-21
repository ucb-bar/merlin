"""Source-only scheduling discovery and applicability joins for the current SDK.

Declarations live with their dense, spatial and source-contract families. Their
ordering remains explicit here. Discovery selects no target law or import grant.
"""

from __future__ import annotations

from merlin.compiler.schedule import catalog_source
from merlin.compiler.schedule.catalog_effects import CAPABILITY_EFFECTS
from merlin.compiler.schedule.dense import capabilities as catalog_dense
from merlin.compiler.schedule.library_sources import CURRENT_SDK_VERSION, library_modules, require_current_sdk
from merlin.compiler.schedule.spatial import capabilities as catalog_spatial

# ---- api_catalog -------------------------------------------------------------------------------

# ---- api_catalog -------------------------------------------------------------------------------

_ORDER = (
    "segmented_k_retained_operand_pipeline",
    "array_weight_retention",
    "exact_final_wide_loads",
    "source_axis_loop_scheduling",
    "source_exact_quantized_spatial_readout_cache",
    "disjoint_accumulator_readout_compute",
    "declared_persistent_scalar_state_in_scf",
    "final_k_causal_accumulator_readout_frontier",
    "loop_carried_completion_events",
    "effect_checked_schedule_composition",
    "current_source_quantized_residual_edges",
    "capacity_phased_convolution_rhs_residency",
    "source_qualified_convolution_readout_epochs",
    "parallel_emitted_variant_shape_evaluation",
    "pointwise_convolution_dense_view",
    "registered_source_convolution_exploration",
    "selected_logical_kernel_argument_binding",
    "shape_resource_schedule_selection",
    "checked_dense_pipeline_entry",
    "checked_dense_completion_epoch_emission",
    "fact_backed_dense_panel_epochs",
    "bounded_declared_settings_cycle_selection",
    "checked_disjoint_loaded_sum_epochs",
    "checked_initialization_and_uniform_readout_batches",
    "source_exact_spatial_matrix_gathers",
    "source_bound_rolling_convolution_rows",
    "checked_cross_spatial_array_reuse",
    "source_exact_convolution_readout_transfers",
    "checked_shared_input_group_blocks",
    "factored_immutable_tensor_layouts",
    "completed_input_with_rhs_frontier",
    "capacity_derived_m_panels_with_rhs_reuse",
    "retained_ready_input_placement",
    "staged_spatial_producer_bands",
    "source_checked_integer_observers",
    "counted_projection_pointer_binding",
    "completion_bound_operand_slots",
    "disjoint_future_row_transport",
    "direct_ring_convolution_row_fragments",
    "last_use_rhs_phase_transport",
    "sole_observer_contraction_rows",
    "bounded_horizontal_convolution_panels",
    "source_proven_shared_rhs_row_views",
    "shared_rhs_epochs",
)
_OWNER = "merlin.compiler.schedule."


def scheduling_capabilities(version=CURRENT_SDK_VERSION):
    """Return fresh usage metadata; only the selected contract grants access."""
    require_current_sdk(version, what="scheduling discovery")

    records = {
        row["name"]: row
        for family in (catalog_dense, catalog_source, catalog_spatial)
        for row in family.records(_OWNER)
    }
    return {
        "schema": "merlin.scheduling_api_catalog.v1",
        "inventory_version": CURRENT_SDK_VERSION,
        "authority": "discovery only; explicit source/effects/target admission remains required",
        "execution": "same Python/xDSL APIs in ordinary automatic and offline authoring",
        "capabilities": [records[name] for name in _ORDER],
        "composition": composition_capabilities(selected_modules=library_modules()),
    }


def capability_effects(version=CURRENT_SDK_VERSION):
    """Join declared effects to the selected inventory in original effect order."""
    capabilities = {row["name"]: row for row in scheduling_capabilities(version)["capabilities"]}
    rows = []
    for name, effect in CAPABILITY_EFFECTS.items():
        if name not in capabilities:
            continue
        capability = capabilities[name]
        rows.append(
            dict(
                name=name,
                symbols=list(capability["symbols"]),
                usage=list(capability["usage"]),
                requires=list(capability["requires"]),
                families=list(effect["families"]),
                changes=dict(effect["changes"]),
                admission=effect["admission"],
            )
        )
    return rows


# ---- composition -------------------------------------------------------------------------------
# Discover completed source composition tools without selecting their authority.
#
# This pure metadata is safe to copy into an author library. Symbol availability
# comes from the selected source inventory; original host owners, implementation
# proofs, numerical policy and target effects remain independently selected.
# No model, captured parameter, reference implementation or measured score lives
# here. Original-owner routes remain distinct from author-side source preparation.


def composition_capabilities(*, selected_modules=None):
    """Describe APIs against an optional exact selected module inventory.

    The small shared SDK and an additional/full compiler source selection have
    different rosters. Membership here is data supplied by the original caller;
    it establishes neither symbol-version compatibility nor import/host grants.
    """
    if selected_modules is not None and (
        type(selected_modules) is not tuple
        or len(selected_modules) > 8192
        or any(
            type(name) is not str or not name or not all(part.isidentifier() for part in name.split("."))
            for name in selected_modules
        )
        or len(selected_modules) != len(set(selected_modules))
    ):
        raise ValueError("composition discovery needs an exact bounded module tuple")
    selected = None if selected_modules is None else frozenset(selected_modules)

    def row(name, symbols, *, consumer, stage, access="author_library", requires=(), limitations=()):
        return {
            "name": name,
            "symbols": list(symbols),
            "consumer": consumer,
            "stage": stage,
            "access": access,
            "requires": list(requires),
            "limitations": list(limitations),
        }

    catalog = {
        "schema": "merlin.composition_api_catalog.v1",
        "authority": "discovery only; no source, numerical, effects, target or execution grant",
        "selected_source_contract_required": True,
        "capabilities": [
            row(
                "original_head_batch_request",
                (
                    "merlin.compiler.lower.device.head.HeadBatchSourceRoute",
                    "merlin.compiler.build.owned_sources.OwnedHostModelCompiler",
                ),
                consumer="optional independent_heads, whole_head_batches and shared_rhs_row_views Booleans in ordinary "
                "source_composition.json",
                access="original_host_owner",
                stage="original_source_admission_and_normal_device_link",
                requires=(
                    "complete original eligible-operation evidence for every composed head",
                    "stock batch bridge, original per-product work, actual object/link and all-output checks",
                ),
                limitations=(
                    "unmatched groups retain ordinary accelerator routing; original readouts remain",
                    "joint opt-in checks disjoint sibling components after heads; "
                    "typed sibling interfaces remain original",
                    "whole_head_batches requires an actual target consumer of the complete ranked batch; "
                    "it grants no inter-head overlap, storage or completion law",
                    "shared_rhs_row_views freshly checks scalar invariance, generated immutable bindings "
                    "when needed, inverse row batching and every original member's exact row range",
                    "batch ABI calls are not matrix invocation counts or a performance estimate",
                ),
            ),
            row(
                "independent_head_and_plane_products",
                (
                    "merlin.compiler.transforms.dense.head_batching.plan_independent_head_batching",
                    "merlin.compiler.schedule.dense.batched.BatchMatrixView",
                    "merlin.compiler.schedule.dense.batched.BatchedDenseProblem",
                    "merlin.compiler.schedule.dense.batched.counted_batched_dense_products",
                    "merlin.compiler.lower.device.logical_borrowed_batch_abi.emit_borrowed_batch_translation_unit",
                    "merlin.compiler.lower.source.compositions.lower_integer_batched_matmul_program",
                ),
                consumer="ordinary source host with an explicitly supplied per-product primitive or ranked-batch consumer",
                access="additional_compiler_sources",
                stage="checked_source_refolding_and_ranked_borrowed_call",
                requires=(
                    "complete independent original head slices and exact signed integer scalar body",
                    "original readouts/plane sums, explicit primitive effects/format, immutable implementation pins",
                    "whole-batch nonalias, output injectivity and bounded per-matrix scratch",
                ),
                limitations=(
                    "all matrix products remain; one batch ABI call is not one accelerator command",
                    "default per-product bridge is synchronous; a ranked-batch consumer owns its "
                    "storage/effects/completion; no implied overlap or performance improvement",
                    "incomplete/unknown source retains its original form; selected module membership is required",
                ),
            ),
            row(
                "operation_profile_diagnostics",
                ("merlin.runtime.harness.output_observation.summarize",),
                consumer="original-owner op_profile request and independently pinned typed operation table",
                access="additional_compiler_sources",
                stage="optional_instrumented_observation",
                requires=("complete original table and marker trace; uninstrumented control",),
                limitations=(
                    "raw counter ticks do not establish elapsed cycles or performance ranking",
                    "semantic categories do not establish CPU/device placement or physical occupancy",
                ),
            ),
            row(
                "whole_model_source_coverage_and_work",
                (
                    "merlin.compiler.planning.source_coverage.plan_model_coverage",
                    "merlin.analysis.programs.model_group_work.summarize_group_work",
                ),
                consumer="current source groups, descriptive shape profiles and original SSA boundaries",
                access="additional_compiler_sources",
                stage="whole_source_planning_and_observation",
                requires=("actual original source module and its current ordered nonoverlapping groups",),
                limitations=(
                    "representatives describe coverage; they do not establish numerical or effect equivalence",
                    "source MACs and scalar work are not runtime traffic, timing or physical utilization",
                    "unassigned or unsupported work remains explicit; no missing region becomes zero cost",
                ),
            ),
            row(
                "source_fusion_exploration",
                (
                    "merlin.compiler.planning.exploration.discover",
                    "merlin.compiler.planning.exploration.plan_materialization_frontier",
                    "merlin.compiler.planning.exploration.realize_candidate",
                    "merlin.compiler.planning.exploration.evaluate_fusions",
                    "merlin.compiler.planning.exploration.compare_region_costs",
                ),
                consumer="registered passes and ordinary emission of the actual returned product",
                stage="bounded_source_discovery_and_existing_realizers",
                requires=("exact source/capability/numerical identities and original unfused alternative",),
                limitations=(
                    "discovery is not legality or profitability; unsupported routes retain source",
                    "logical materialization bytes prioritize bounded nonoverlapping proposals; actual region cycles select performance",
                    "allow_provisional=True compares complete same-model region observations for exploration only; "
                    "UNKNOWN remains unranked, and all-output/source/scope checks remain required",
                ),
            ),
            row(
                "checked_pointwise_fusion",
                (
                    "merlin.compiler.schedule.elementwise.fusion.plan_fusion",
                    "merlin.compiler.transforms.source_rewrites.multioutput.compile_source_multioutput",
                ),
                consumer="actual checked Kernel or guarded compilation images enter ordinary target emission",
                stage="checked_schedule_or_guarded_images",
                requires=("source scalar semantics, effects, aliases, storage and original target adapter",),
                limitations=("finite adjacent calls or supported pointwise fanout; not arbitrary graph fusion",),
            ),
            row(
                "connected_source_execution",
                (
                    "merlin.compiler.lower.connected.plan_connected_region",
                    "merlin.compiler.lower.connected.compile_connected_region",
                    "merlin.compiler.lower.connected.ConnectedProgram",
                    "merlin.compiler.lower.connected.emit_connected_host_c",
                    "merlin.compiler.partition.global_plan.GlobalPlan",
                    "merlin.compiler.partition.global_plan.verify_global_plan",
                    "merlin.compiler.partition.global_plan_emission.emit_global_plan",
                ),
                consumer="source-bound stages, ordered runtime/host calls and checked global-plan emission",
                stage="checked_sequential_or_explicit_global_plan_composition",
                requires=("explicit stage implementations, backing windows, lifetimes and completion",),
                limitations=("sequential composition alone establishes neither cross-stage fusion nor overlap",),
            ),
            row(
                "original_ssa_region_and_views",
                (
                    "merlin.compiler.lower.source.region.extract_source_region",
                    "merlin.compiler.transforms.canonicalization.canonicalize_tensor_views",
                ),
                consumer="exact extracted source with original free inputs and all externally observed outputs",
                stage="source_transform",
                requires=("contiguous original span; explicit literal/view closure; unchanged source ownership",),
                limitations=("no inferred runtime input values, purity of external calls or source closure",),
            ),
            row(
                "original_ssa_diagnostic_outputs",
                ("merlin.compiler.lower.source.region.append_source_observations",),
                consumer="optional diagnostic clone with bounded original SSA values appended to its outputs",
                stage="source_diagnostics",
                requires=("explicit original tensor values and total observation element bound",),
                limitations=(
                    "observations extend lifetimes and change the ABI; no timing or effect equivalence",
                    "original outputs and acceptance criteria remain separately checked",
                ),
            ),
            row(
                "independent_shared_input_components",
                ("merlin.compiler.lower.device.sibling_components.prepare_sibling_components",),
                consumer="original-owner partial sibling route and actual ordinary unmatched groups",
                access="additional_compiler_sources",
                stage="source_host_device_component_emission",
                requires=("original SSA and exact synchronous borrowed-effect selection",),
                limitations=(
                    "bounded independent components; original epilogues and all observers remain",
                    "unproven input layouts retain identity copies and original guards",
                    "source composition alone grants neither target legality nor numerical qualification",
                ),
            ),
            row(
                "source_program_component_emission",
                (
                    "merlin.compiler.lower.source.compositions.prepare_source_program",
                    "merlin.compiler.lower.source.compositions.emit_source_program",
                ),
                consumer="actual host/device emitter products and original logical pointer transports",
                access="additional_compiler_sources",
                stage="source_host_device_component_emission",
                requires=("exact supported sibling source, selected borrowed effects and stock host tools",),
                limitations=(
                    "initial sibling integer family only; not arbitrary graph lowering",
                    "component ABI checks do not establish physical work, numerics or final owner admission",
                ),
            ),
            row(
                "stable_loop_occurrences",
                (
                    "merlin.compiler.kernel_ir.loops.locate_loop",
                    "merlin.compiler.kernel_ir.loops.transform_loop",
                ),
                consumer="digest-bound structural occurrence and explicit transformed descendant handles",
                access="additional_compiler_sources",
                stage="loop_transformation",
                requires=("current kernel digest and explicit structural occurrence path",),
                limitations=(
                    "stale handles refuse; split/fission descendants require explicit selection",
                    "binder hygiene preserves lexical events, not new target dependency laws",
                ),
            ),
            row(
                "source_sibling_host_preparation",
                (
                    "merlin.compiler.lower.source.region.outline_sibling_contractions",
                    "merlin.compiler.lower.source.region.outline_source_siblings",
                    "merlin.compiler.lower.source.region.prepare_sibling_source_host",
                    "merlin.compiler.lower.source.region.emit_ranked_source_entry",
                ),
                consumer="explicit source-owner preparation and ordinary closed-core build",
                stage="source_host_preparation",
                requires=("same original LHS identity; exact zero-seeded integer roots; original borrowed effects",),
                limitations=("all original host epilogues/observers remain; owner route is separately selected",),
            ),
            row(
                "sibling_original_host_route",
                (
                    "merlin.compiler.lower.device.sibling_source_route.SiblingSourceRoute",
                    "merlin.compiler.build.owned_sources.OwnedHostModelCompiler",
                ),
                consumer="OwnedHostModelCompiler.compile(sibling_source=original_typed_route)",
                access="original_host_owner",
                stage="ordinary_source_host_shared_device_object_link",
                requires=(
                    "exact original source, separately qualified synchronous borrowed effects and logical ABI",
                    "complete source membership, original host quantizers/observers and linked-image checks",
                ),
                limitations=(
                    "explicit caller selection; automatic model checks need their composition audit adapter",
                    "source join and numerical match do not replace actual per-member device-work proof",
                ),
            ),
            row(
                "member_projection_views",
                (
                    "merlin.compiler.schedule.shared_operands.projections.plan_shared_projection_views",
                    "merlin.compiler.schedule.shared_operands.shared_inputs.find_shared_input_groups",
                    "merlin.compiler.schedule.shared_operands.shared_inputs.schedule_shared_input",
                ),
                consumer="original tensor pointers/pitches and actual complete target rectangles",
                stage="source_checked_virtual_index_mapping",
                requires=("every actual wide load/store stays within one original member; final band/effect checks",),
                limitations=("virtual matrices are never packed allocations; per-output quantizers remain separate",),
            ),
            row(
                "original_projection_work_identity",
                (
                    "merlin.compiler.schedule.shared_operands.projections.shared_projection_provenance",
                    "merlin.compiler.schedule.shared_operands.projections.projection_compute_member",
                    "merlin.compiler.schedule.shared_operands.projections.projection_ir_attributes",
                ),
                consumer="actual member-bound PRELOAD/COMPUTE sites retain original typed root identities",
                stage="source_to_interface_to_actual_device_work",
                requires=(
                    "complete unique original region/node identities and proved real compute column bounds",
                    "matching canonical interface transport and explicitly selected helper source",
                ),
                limitations=(
                    "cross-member or unknown intervals refuse attribution; identities grant no work credit",
                    "independent actual command, CFG, object and native checks remain mandatory",
                ),
            ),
            row(
                "connected_spatial_original_host_route",
                (
                    "merlin.compiler.lower.device.spatial.group_connected_spatial",
                    "merlin.compiler.lower.device.spatial.ConnectedSpatialRoute",
                    "merlin.compiler.build.owned_sources.OwnedHostModelCompiler",
                ),
                consumer="OwnedHostModelCompiler.compile(connected_spatial=original_typed_route)",
                access="original_host_owner",
                stage="ordinary_source_host_device_object_link",
                requires=(
                    "original SSA pair, ordered Q1/Q2, complete observers, checked plan/private workspace",
                    "independently selected fixed host, device package and final linked-image admission",
                ),
                limitations=(
                    "explicit caller selection; not automatically enabled by model-check or a candidate flag",
                    "author source availability cannot issue original host or target authority",
                ),
            ),
            row(
                "bounded_spatial_packing_and_rolling",
                (
                    "merlin.compiler.schedule.spatial.producer.PackedSpatialProducerPlan",
                    "merlin.compiler.schedule.spatial.readouts.propose_rolling_spatial_choices",
                    "merlin.compiler.schedule.spatial.readouts.emit_rolling_spatial_producer",
                    "merlin.compiler.schedule.spatial.rolling_window.materialize_rolling_window",
                    "merlin.compiler.lower.source.compositions.emit_nhwc_halo_panel",
                ),
                consumer="actual counted row/cache/panel transfers and existing dense producer emission",
                stage="checked_geometry_storage_and_emission",
                requires=("source halo/layout/padding, explicit capacities, row identities and completion",),
                limitations=("movement reuse supplies no accumulator-retention or asynchronous execution law",),
            ),
            row(
                "bounded_spatial_max_strips",
                ("merlin.compiler.schedule.spatial.max_stripes.plan_striped_spatial_max",),
                consumer="counted staging and pooled output commands through the selected target adapter",
                access="additional_compiler_sources",
                stage="source_window_and_private_capacity_planning",
                requires=("original spatial max/padding semantics, physical row mapping and selected capacities",),
                limitations=(
                    "strip planning grants no pooled target instruction, numerical or asynchronous law",
                    "actual encoded transfer, window and completion checks remain required",
                ),
            ),
            row(
                "private_intermediate_storage",
                (
                    "merlin.compiler.schedule.storage.temporaries.allocate_temporary_storage",
                    "merlin.compiler.lower.host.writers.rewrite_fresh_tensor_writers",
                    "merlin.compiler.lower.host.writers.pool_private_writer_workspaces",
                    "merlin.compiler.lower.options.weight_prepack.prepacked_bundle",
                    "merlin.compiler.diagnostics.install_prepared_observed",
                ),
                consumer="actual source owners, checked temporary placements and exact immutable preparation bytes",
                stage="storage_and_preparation_transform",
                requires=("full writes, borrowing/noescape, lifetime/alias proof and bounded owned storage",),
                limitations=("capacity is not completion; immutable prepacking requires original parameter identity",),
            ),
            row(
                "connected_host_layout_selection",
                (
                    "merlin.compiler.lower.host.layout.plan_islands",
                    "merlin.compiler.lower.host.layout.rewrite_module",
                    "merlin.compiler.lower.host.graph.optimize_host_graph",
                ),
                consumer="actual rewritten source enters ordinary host/device grouping and lowering",
                access="additional_compiler_sources",
                stage="source_layout_before_device_grouping",
                requires=(
                    "current source-coordinate and scalar-body conformance, fixed public layouts and observers",
                    "selected supported regions, explicit structural budget and original software permissions",
                ),
                limitations=(
                    "unknown operations retain boundary conversions; source layout is not device residency",
                    "composed source ownership needs a fresh checked bridge; no blanket ABI transfer",
                    "conversion bytes screen alternatives; complete-region cycles establish performance",
                ),
            ),
            row(
                "completed_host_allocation_reuse",
                (
                    "merlin.compiler.lower.host.host_storage_policy.prepare",
                    "merlin.compiler.lower.host.host_storage_policy.recheck",
                    "merlin.compiler.lower.host.arena.bind_arena",
                ),
                consumer="reuse_intermediate_storage request through the original host compiler and its exact LLVM",
                access="original_host_owner",
                stage="completed_call_lifetime_and_final_arena_binding",
                requires=(
                    "actual allocation aliases, real release/completion and checked nonescaping call effects",
                    "original sequential/non-reentrant allocation policy and current compiled product",
                ),
                limitations=(
                    "graph last use alone cannot release asynchronous accesses or persistent state",
                    "unknown effects, loop generations and unproved ownership retain original allocations",
                    "arena bytes describe selected storage, not measured peak occupancy or elapsed cycles",
                ),
            ),
            row(
                "ordered_quantization_boundaries",
                (
                    "merlin.compiler.schedule.spatial.readouts.prove_spatial_readout",
                    "merlin.compiler.schedule.spatial.readouts.propose_spatial_readout_cache",
                    "merlin.compiler.lower.quant.affine_pair.propose_exact_affine_pair",
                    "merlin.compiler.lower.quant.affine_pair.prove_ordered_loaded_sum",
                ),
                consumer="existing typed readout/finite-domain proofs plus actual target adapter emission",
                stage="source_exact_readout_proposals",
                requires=("literal ordered source scales, rounding, saturation, activation and output types",),
                limitations=("unsupported composition retains stages; per-column coefficients are not scalar",),
            ),
            row(
                "real_transform_ordering",
                (
                    "merlin.compiler.transforms.exploration.pass_orders.explore_candidates",
                    "merlin.compiler.transforms.exploration.pass_orders.rank_performance_variants",
                    "merlin.compiler.schedule.loops.counted_loops.tile_loop_nest_transforms",
                    "merlin.compiler.schedule.loops.counted_loops.partition_counted_loop",
                    "merlin.compiler.schedule.loops.scf_commands.materialize_scf_commands",
                    "merlin.compiler.schedule.loops.scf_commands.emit_scf_commands",
                    "merlin.compiler.kernel_ir.ir.Loop",
                ),
                consumer="registered real transforms, actual checked Kernel/SCF and target callbacks",
                stage="transforms_consumed_by_emission",
                requires=("caller factors/cuts/order; source, bounds, tails, dependencies and final verification",),
                limitations=(
                    "persistent state and lexical completion inputs/yields/results require explicit opt-in; "
                    "carried split/unroll/ordered partitions preserve issue order, while dependency-changing "
                    "transforms refuse; actual launch/wait callbacks and target completion law "
                    "remain required",
                ),
            ),
            row(
                "qualified_configuration_motion",
                (
                    "merlin.compiler.schedule.loops.configuration_loops.promote_configuration_writes",
                    "merlin.compiler.schedule.loops.configuration_loops.verify_configuration_motion",
                ),
                consumer="actual retained Kernel feeds ordinary configuration cleanup and emission",
                access="additional_compiler_sources",
                stage="effect_checked_configuration_motion",
                requires=("complete source state effects; pure writes and independent crossed calls",),
                limitations=(
                    "does not cross loops, unknown effects or completion metadata",
                    "no deletion or target timing law; native profitability requires cycle feedback",
                ),
            ),
            row(
                "counted_emitted_work",
                ("merlin.analysis.programs.emitted_work.summarize_emitted_work",),
                consumer="own-artifact diagnostics or explicit original emitted LLVM analysis",
                access="additional_compiler_sources",
                stage="emitted_program_observation",
                requires=("actual final LLVM function and supported exact control domains",),
                limitations=(
                    "counts IR sites, not callee bodies, bytes, cycles or physical utilization",
                    "unclassified commands remain visible; data-dependent control stays UNKNOWN",
                ),
            ),
            row(
                "independent_completion_stream_fusion",
                (
                    "merlin.compiler.schedule.loops.completion_events.fuse_completion_streams",
                    "merlin.compiler.schedule.loops.completion_events.CompletionEffectsAdapter",
                    "merlin.compiler.schedule.loops.completion_events.CompletionEffectEnvelope",
                ),
                consumer="actual checked Kernel and SCF with original ordered event recurrences and drains",
                access="additional_compiler_sources",
                stage="completion_aware_loop_transform",
                requires=(
                    "adjacent equal-trip recurrences with independent active/passive event channels",
                    "independently selected complete memory, state and observer effect projection",
                ),
                limitations=(
                    "unknown effects or cross-stream hazards refuse; stage rows alone do not grant motion",
                    "target completion law, numerical policy and final source/epoch checks remain required",
                ),
            ),
            row(
                "explicit_host_graph_optimization",
                (
                    "merlin.compiler.lower.host.graph.GraphOptimizationContract",
                    "merlin.compiler.lower.host.graph.optimize_host_graph",
                ),
                consumer="ordinary host compilation with source-bound graph selection",
                access="original_host_owner",
                stage="source_checked_host_graph_transform",
                requires=("exact source, explicit logical/physical layout permissions and bounded growth",),
                limitations=("structural savings are not measured traffic/time or target effect qualification",),
            ),
        ],
    }
    for capability in catalog["capabilities"]:
        modules = {symbol.rsplit(".", 1)[0] for symbol in capability["symbols"]}
        missing = None if selected is None else sorted(modules - selected)
        capability["source_availability"] = {
            "status": "selection_not_supplied"
            if missing is None
            else "missing_selected_modules"
            if missing
            else "modules_selected",
            "missing_modules": missing,
            "scope": "exact module membership only; symbol compatibility and original authority remain separate",
        }
    return catalog
