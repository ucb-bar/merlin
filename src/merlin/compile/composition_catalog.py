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
            row("source_fusion_exploration", (
                "merlin.compile.fusion_workflow.discover",
                "merlin.compile.fusion_workflow.realize_candidate",
                "merlin.compile.fusion_workflow.evaluate_fusions",
            ), consumer="registered passes and ordinary emission of the actual returned product",
                stage="bounded_source_discovery_and_existing_realizers",
                requires=("exact source/capability/numerical identities and original unfused alternative",),
                limitations=("discovery is not legality or profitability; unsupported routes retain source",)),
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
            row("ordered_quantization_boundaries", (
                "merlin.compile.scheduling.spatial_readout.prove_spatial_readout",
                "merlin.compile.scheduling.spatial_readout_cache.propose_spatial_readout_cache",
                "merlin.llvmlower.quantized_affine_pair.propose_exact_affine_pair",
            ), consumer="existing typed readout/finite-domain proofs plus actual target adapter emission",
                stage="source_exact_readout_proposals",
                requires=("literal ordered source scales, rounding, saturation, activation and output types",),
                limitations=("unsupported composition retains stages; per-column coefficients are not scalar",)),
            row("real_transform_ordering", (
                "merlin.transforms.pass_exploration.explore_candidates",
                "merlin.compile.scheduling.loop_schedule.tile_loop_nest_transforms",
                "merlin.compile.scheduling.loop_schedule.partition_counted_loop",
                "merlin.compile.scheduling.scf_commands.materialize_scf_commands",
                "merlin.compile.scheduling.scf_commands.emit_scf_commands",
            ), consumer="registered real transforms, actual checked Kernel/SCF and target callbacks",
                stage="transforms_consumed_by_emission",
                requires=("caller factors/cuts/order; source, bounds, tails, dependencies and final verification",),
                limitations=("persistent scalar state needs explicit opt-in; no loop-carried event ownership",)),
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
