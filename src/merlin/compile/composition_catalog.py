"""Discover completed source composition tools without selecting their authority.

This pure metadata is safe to copy into an author library. Symbol availability
comes from the selected source inventory; original host owners, implementation
proofs, numerical policy and target effects remain independently selected.
No model, captured parameter, reference implementation or measured score lives
here. Original-owner routes remain distinct from author-side source preparation.
"""

from __future__ import annotations


def composition_capabilities(*, selected_modules=None):
    """Describe APIs against an optional exact selected module inventory.

    The small shared SDK and an additional/full compiler source selection have
    different rosters. Membership here is data supplied by the original caller;
    it establishes neither symbol-version compatibility nor import/host grants.
    """
    if selected_modules is not None and (
        type(selected_modules) is not tuple
        or len(selected_modules) > 8192
        or any(type(name) is not str or not name or not all(part.isidentifier() for part in name.split("."))
               for name in selected_modules)
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
            row("original_head_batch_request", (
                "merlin.llvmlower.head_source_route.HeadBatchSourceRoute",
                "merlin.compile.owned_host_model.OwnedHostModelCompiler",
            ), consumer="optional independent_heads, whole_head_batches and shared_rhs_row_views Booleans in ordinary "
                        "source_composition.json",
                access="original_host_owner", stage="original_source_admission_and_normal_device_link",
                requires=("complete original eligible-operation evidence for every composed head",
                          "stock batch bridge, original per-product work, actual object/link and all-output checks"),
                limitations=("unmatched groups retain ordinary accelerator routing; original readouts remain",
                             "joint opt-in checks disjoint sibling components after heads; "
                             "typed sibling interfaces remain original",
                             "whole_head_batches requires an actual target consumer of the complete ranked batch; "
                             "it grants no inter-head overlap, storage or completion law",
                             "shared_rhs_row_views freshly checks scalar invariance, generated immutable bindings "
                             "when needed, inverse row batching and every original member's exact row range",
                             "batch ABI calls are not matrix invocation counts or a performance estimate",)),
            row("independent_head_and_plane_products", (
                "merlin.transforms.head_batching.plan_independent_head_batching",
                "merlin.compile.scheduling.batched_dense.BatchMatrixView",
                "merlin.compile.scheduling.batched_dense.BatchedDenseProblem",
                "merlin.compile.scheduling.batched_dense.counted_batched_dense_products",
                "merlin.llvmlower.logical_borrowed_batch_abi.emit_borrowed_batch_translation_unit",
                "merlin.llvmlower.integer_batched_matmul_calls.lower_integer_batched_matmul_program",
            ), consumer="ordinary source host with an explicitly supplied per-product primitive or ranked-batch consumer",
                access="additional_compiler_sources", stage="checked_source_refolding_and_ranked_borrowed_call",
                requires=("complete independent original head slices and exact signed integer scalar body",
                          "original readouts/plane sums, explicit primitive effects/format, immutable implementation pins",
                          "whole-batch nonalias, output injectivity and bounded per-matrix scratch"),
                limitations=("all matrix products remain; one batch ABI call is not one accelerator command",
                             "default per-product bridge is synchronous; a ranked-batch consumer owns its "
                             "storage/effects/completion; no implied overlap or performance improvement",
                             "incomplete/unknown source retains its original form; selected module membership is required",)),
            row("operation_profile_diagnostics", (
                "merlin.runtime.operation_profile.summarize",
            ), consumer="original-owner op_profile request and independently pinned typed operation table",
                access="additional_compiler_sources", stage="optional_instrumented_observation",
                requires=("complete original table and marker trace; uninstrumented control",),
                limitations=("raw counter ticks do not establish elapsed cycles or performance ranking",
                             "semantic categories do not establish CPU/device placement or physical occupancy",)),
            row("whole_model_source_coverage_and_work", (
                "merlin.compile.model_coverage.plan_model_coverage",
                "merlin.perf.model_group_work.summarize_group_work",
            ), consumer="current source groups, descriptive shape profiles and original SSA boundaries",
                access="additional_compiler_sources", stage="whole_source_planning_and_observation",
                requires=("actual original source module and its current ordered nonoverlapping groups",),
                limitations=("representatives describe coverage; they do not establish numerical or effect equivalence",
                             "source MACs and scalar work are not runtime traffic, timing or physical utilization",
                             "unassigned or unsupported work remains explicit; no missing region becomes zero cost",)),
            row("source_fusion_exploration", (
                "merlin.compile.fusion_workflow.discover",
                "merlin.compile.fusion_workflow.plan_materialization_frontier",
                "merlin.compile.fusion_workflow.realize_candidate",
                "merlin.compile.fusion_workflow.evaluate_fusions",
                "merlin.compile.fusion_workflow.compare_region_costs",
            ), consumer="registered passes and ordinary emission of the actual returned product",
                stage="bounded_source_discovery_and_existing_realizers",
                requires=("exact source/capability/numerical identities and original unfused alternative",),
                limitations=("discovery is not legality or profitability; unsupported routes retain source",
                             "logical materialization bytes prioritize bounded nonoverlapping proposals; actual region cycles select performance",
                             "allow_provisional=True compares complete same-model region observations for exploration only; "
                             "UNKNOWN remains unranked, and all-output/source/scope checks remain required",)),
            row("checked_pointwise_fusion", (
                "merlin.compile.scheduling.fusion.plan_fusion",
                "merlin.compile.scheduling.multioutput_fusion.compile_source_multioutput",
            ), consumer="actual checked Kernel or guarded compilation images enter ordinary target emission",
                stage="checked_schedule_or_guarded_images",
                requires=("source scalar semantics, effects, aliases, storage and original target adapter",),
                limitations=("finite adjacent calls or supported pointwise fanout; not arbitrary graph fusion",)),
            row("connected_source_execution", (
                "merlin.compile.connected_regions.plan_connected_region",
                "merlin.compile.connected_regions.compile_connected_region",
                "merlin.compile.connected_runtime.ConnectedProgram",
                "merlin.compile.connected_host.emit_connected_host_c",
                "merlin.xdsl_dialects.lowering.global_plan.GlobalPlan",
                "merlin.xdsl_dialects.lowering.global_plan.verify_global_plan",
                "merlin.xdsl_dialects.lowering.global_plan_emission.emit_global_plan",
            ), consumer="source-bound stages, ordered runtime/host calls and checked global-plan emission",
                stage="checked_sequential_or_explicit_global_plan_composition",
                requires=("explicit stage implementations, backing windows, lifetimes and completion",),
                limitations=("sequential composition alone establishes neither cross-stage fusion nor overlap",)),
            row("original_ssa_region_and_views", (
                "merlin.llvmlower.source_region.extract_source_region",
                "merlin.transforms.tensor_views.canonicalize_tensor_views",
            ), consumer="exact extracted source with original free inputs and all externally observed outputs",
                stage="source_transform",
                requires=("contiguous original span; explicit literal/view closure; unchanged source ownership",),
                limitations=("no inferred runtime input values, purity of external calls or source closure",)),
            row("original_ssa_diagnostic_outputs", (
                "merlin.llvmlower.source_region.append_source_observations",
            ), consumer="optional diagnostic clone with bounded original SSA values appended to its outputs",
                stage="source_diagnostics",
                requires=("explicit original tensor values and total observation element bound",),
                limitations=("observations extend lifetimes and change the ABI; no timing or effect equivalence",
                             "original outputs and acceptance criteria remain separately checked",)),
            row("independent_shared_input_components", (
                "merlin.llvmlower.sibling_components.prepare_sibling_components",
            ), consumer="original-owner partial sibling route and actual ordinary unmatched groups",
                access="additional_compiler_sources", stage="source_host_device_component_emission",
                requires=("original SSA and exact synchronous borrowed-effect selection",),
                limitations=("bounded independent components; original epilogues and all observers remain",
                             "unproven input layouts retain identity copies and original guards",
                             "source composition alone grants neither target legality nor numerical qualification",)),
            row("source_program_component_emission", (
                "merlin.llvmlower.source_program.prepare_source_program",
                "merlin.llvmlower.source_program.emit_source_program",
            ), consumer="actual host/device emitter products and original logical pointer transports",
                access="additional_compiler_sources", stage="source_host_device_component_emission",
                requires=("exact supported sibling source, selected borrowed effects and stock host tools",),
                limitations=("initial sibling integer family only; not arbitrary graph lowering",
                             "component ABI checks do not establish physical work, numerics or final owner admission",)),
            row("stable_loop_occurrences", (
                "merlin.sched.primitives.loop_handles.locate_loop",
                "merlin.sched.primitives.loop_handles.transform_loop",
            ), consumer="digest-bound structural occurrence and explicit transformed descendant handles",
                access="additional_compiler_sources", stage="loop_transformation",
                requires=("current kernel digest and explicit structural occurrence path",),
                limitations=("stale handles refuse; split/fission descendants require explicit selection",
                             "binder hygiene preserves lexical events, not new target dependency laws",)),
            row("source_sibling_host_preparation", (
                "merlin.llvmlower.source_region.outline_sibling_contractions",
                "merlin.llvmlower.source_region_host.outline_source_siblings",
                "merlin.llvmlower.source_region_host.prepare_sibling_source_host",
                "merlin.llvmlower.source_region_host.emit_ranked_source_entry",
            ), consumer="explicit source-owner preparation and ordinary closed-core build",
                stage="source_host_preparation",
                requires=("same original LHS identity; exact zero-seeded integer roots; original borrowed effects",),
                limitations=("all original host epilogues/observers remain; owner route is separately selected",)),
            row("sibling_original_host_route", (
                "merlin.llvmlower.sibling_source_route.SiblingSourceRoute",
                "merlin.compile.owned_host_model.OwnedHostModelCompiler",
            ), consumer="OwnedHostModelCompiler.compile(sibling_source=original_typed_route)",
                access="original_host_owner", stage="ordinary_source_host_shared_device_object_link",
                requires=("exact original source, separately qualified synchronous borrowed effects and logical ABI",
                          "complete source membership, original host quantizers/observers and linked-image checks"),
                limitations=("explicit caller selection; automatic model checks need their composition audit adapter",
                             "source join and numerical match do not replace actual per-member device-work proof",)),
            row("member_projection_views", (
                "merlin.compile.scheduling.projection_views.plan_shared_projection_views",
                "merlin.compile.scheduling.shared_input_groups.find_shared_input_groups",
                "merlin.compile.scheduling.shared_input_schedule.schedule_shared_input",
            ), consumer="original tensor pointers/pitches and actual complete target rectangles",
                stage="source_checked_virtual_index_mapping",
                requires=("every actual wide load/store stays within one original member; final band/effect checks",),
                limitations=("virtual matrices are never packed allocations; per-output quantizers remain separate",)),
            row("original_projection_work_identity", (
                "merlin.compile.scheduling.projection_provenance.shared_projection_provenance",
                "merlin.compile.scheduling.projection_provenance.projection_compute_member",
                "merlin.compile.scheduling.projection_provenance.projection_ir_attributes",
            ), consumer="actual member-bound PRELOAD/COMPUTE sites retain original typed root identities",
                stage="source_to_interface_to_actual_device_work",
                requires=("complete unique original region/node identities and proved real compute column bounds",
                          "matching canonical interface transport and explicitly selected helper source"),
                limitations=("cross-member or unknown intervals refuse attribution; identities grant no work credit",
                             "independent actual command, CFG, object and native checks remain mandatory",)),
            row("connected_spatial_original_host_route", (
                "merlin.llvmlower.connected_spatial.group_connected_spatial",
                "merlin.llvmlower.connected_spatial_route.ConnectedSpatialRoute",
                "merlin.compile.owned_host_model.OwnedHostModelCompiler",
            ), consumer="OwnedHostModelCompiler.compile(connected_spatial=original_typed_route)",
                access="original_host_owner", stage="ordinary_source_host_device_object_link",
                requires=("original SSA pair, ordered Q1/Q2, complete observers, checked plan/private workspace",
                          "independently selected fixed host, device package and final linked-image admission"),
                limitations=("explicit caller selection; not automatically enabled by model-check or a candidate flag",
                             "author source availability cannot issue original host or target authority",)),
            row("bounded_spatial_packing_and_rolling", (
                "merlin.compile.scheduling.spatial_producer.PackedSpatialProducerPlan",
                "merlin.compile.scheduling.rolling_spatial.propose_rolling_spatial_choices",
                "merlin.compile.scheduling.rolling_spatial.emit_rolling_spatial_producer",
                "merlin.compile.scheduling.rolling_window.materialize_rolling_window",
                "merlin.llvmlower.convolution_packing.emit_nhwc_halo_panel",
            ), consumer="actual counted row/cache/panel transfers and existing dense producer emission",
                stage="checked_geometry_storage_and_emission",
                requires=("source halo/layout/padding, explicit capacities, row identities and completion",),
                limitations=("movement reuse supplies no accumulator-retention or asynchronous execution law",)),
            row("bounded_spatial_max_strips", (
                "merlin.compile.scheduling.spatial_max_stripes.plan_striped_spatial_max",
            ), consumer="counted staging and pooled output commands through the selected target adapter",
                access="additional_compiler_sources", stage="source_window_and_private_capacity_planning",
                requires=("original spatial max/padding semantics, physical row mapping and selected capacities",),
                limitations=("strip planning grants no pooled target instruction, numerical or asynchronous law",
                             "actual encoded transfer, window and completion checks remain required",)),
            row("private_intermediate_storage", (
                "merlin.compile.scheduling.temporary_storage.allocate_temporary_storage",
                "merlin.llvmlower.fresh_tensor_writer.rewrite_fresh_tensor_writers",
                "merlin.llvmlower.private_workspace_pool.pool_private_writer_workspaces",
                "merlin.llvmlower.weight_prepack.prepacked_bundle",
                "merlin.compile.compiler_observations.install_prepared_observed",
            ), consumer="actual source owners, checked temporary placements and exact immutable preparation bytes",
                stage="storage_and_preparation_transform",
                requires=("full writes, borrowing/noescape, lifetime/alias proof and bounded owned storage",),
                limitations=("capacity is not completion; immutable prepacking requires original parameter identity",)),
            row("connected_host_layout_selection", (
                "merlin.llvmlower.host_layout_planning.plan_islands",
                "merlin.llvmlower.host_layout_realization.rewrite_module",
                "merlin.llvmlower.graph_optimizations.optimize_host_graph",
            ), consumer="actual rewritten source enters ordinary host/device grouping and lowering",
                access="additional_compiler_sources", stage="source_layout_before_device_grouping",
                requires=("current source-coordinate and scalar-body conformance, fixed public layouts and observers",
                          "selected supported regions, explicit structural budget and original software permissions"),
                limitations=("unknown operations retain boundary conversions; source layout is not device residency",
                             "composed source ownership needs a fresh checked bridge; no blanket ABI transfer",
                             "conversion bytes screen alternatives; complete-region cycles establish performance",)),
            row("completed_host_allocation_reuse", (
                "merlin.llvmlower.host_storage_policy.prepare",
                "merlin.llvmlower.host_storage_policy.recheck",
                "merlin.llvmlower.arena_bind.bind_arena",
            ), consumer="reuse_intermediate_storage request through the original host compiler and its exact LLVM",
                access="original_host_owner", stage="completed_call_lifetime_and_final_arena_binding",
                requires=("actual allocation aliases, real release/completion and checked nonescaping call effects",
                          "original sequential/non-reentrant allocation policy and current compiled product"),
                limitations=("graph last use alone cannot release asynchronous accesses or persistent state",
                             "unknown effects, loop generations and unproved ownership retain original allocations",
                             "arena bytes describe selected storage, not measured peak occupancy or elapsed cycles",)),
            row("ordered_quantization_boundaries", (
                "merlin.compile.scheduling.spatial_readout.prove_spatial_readout",
                "merlin.compile.scheduling.spatial_readout_cache.propose_spatial_readout_cache",
                "merlin.llvmlower.quantized_affine_pair.propose_exact_affine_pair",
                "merlin.llvmlower.quantized_affine_pair.prove_ordered_loaded_sum",
            ), consumer="existing typed readout/finite-domain proofs plus actual target adapter emission",
                stage="source_exact_readout_proposals",
                requires=("literal ordered source scales, rounding, saturation, activation and output types",),
                limitations=("unsupported composition retains stages; per-column coefficients are not scalar",)),
            row("real_transform_ordering", (
                "merlin.transforms.pass_exploration.explore_candidates",
                "merlin.transforms.pass_exploration.rank_performance_variants",
                "merlin.compile.scheduling.loop_schedule.tile_loop_nest_transforms",
                "merlin.compile.scheduling.loop_schedule.partition_counted_loop",
                "merlin.compile.scheduling.scf_commands.materialize_scf_commands",
                "merlin.compile.scheduling.scf_commands.emit_scf_commands",
                "merlin.sched.ir.Loop",
            ), consumer="registered real transforms, actual checked Kernel/SCF and target callbacks",
                stage="transforms_consumed_by_emission",
                requires=("caller factors/cuts/order; source, bounds, tails, dependencies and final verification",),
                limitations=("persistent state and lexical completion inputs/yields/results require explicit opt-in; "
                             "carried split/unroll/ordered partitions preserve issue order, while dependency-changing "
                             "transforms refuse; actual launch/wait callbacks and target completion law "
                             "remain required",)),
            row("qualified_configuration_motion", (
                "merlin.compile.scheduling.configuration_motion.promote_configuration_writes",
                "merlin.compile.scheduling.configuration_motion.verify_configuration_motion",
            ), consumer="actual retained Kernel feeds ordinary configuration cleanup and emission",
                access="additional_compiler_sources", stage="effect_checked_configuration_motion",
                requires=("complete source state effects; pure writes and independent crossed calls",),
                limitations=("does not cross loops, unknown effects or completion metadata",
                             "no deletion or target timing law; native profitability requires cycle feedback",)),
            row("counted_emitted_work", (
                "merlin.perf.emitted_work.summarize_emitted_work",
            ), consumer="own-artifact diagnostics or explicit original emitted LLVM analysis",
                access="additional_compiler_sources", stage="emitted_program_observation",
                requires=("actual final LLVM function and supported exact control domains",),
                limitations=("counts IR sites, not callee bodies, bytes, cycles or physical utilization",
                             "unclassified commands remain visible; data-dependent control stays UNKNOWN",)),
            row("independent_completion_stream_fusion", (
                "merlin.compile.scheduling.completion_streams.fuse_completion_streams",
                "merlin.compile.scheduling.completion_streams.CompletionEffectsAdapter",
                "merlin.compile.scheduling.completion_streams.CompletionEffectEnvelope",
            ), consumer="actual checked Kernel and SCF with original ordered event recurrences and drains",
                access="additional_compiler_sources", stage="completion_aware_loop_transform",
                requires=("adjacent equal-trip recurrences with independent active/passive event channels",
                          "independently selected complete memory, state and observer effect projection"),
                limitations=("unknown effects or cross-stream hazards refuse; stage rows alone do not grant motion",
                             "target completion law, numerical policy and final source/epoch checks remain required",)),
            row("explicit_host_graph_optimization", (
                "merlin.llvmlower.graph_optimizations.GraphOptimizationContract",
                "merlin.llvmlower.graph_optimizations.optimize_host_graph",
            ), consumer="ordinary host compilation with source-bound graph selection",
                access="original_host_owner", stage="source_checked_host_graph_transform",
                requires=("exact source, explicit logical/physical layout permissions and bounded growth",),
                limitations=("structural savings are not measured traffic/time or target effect qualification",)),
        ],
    }
    for capability in catalog["capabilities"]:
        modules = {symbol.rsplit(".", 1)[0] for symbol in capability["symbols"]}
        missing = None if selected is None else sorted(modules - selected)
        capability["source_availability"] = {
            "status": "selection_not_supplied" if missing is None else
                      "missing_selected_modules" if missing else "modules_selected",
            "missing_modules": missing,
            "scope": "exact module membership only; symbol compatibility and original authority remain separate",
        }
    return catalog
