"""Versioned source-only scheduling discovery and applicability joins.

Declarations live with their dense, spatial and source-contract families. Their
ordering remains explicit here. Discovery selects no target law or import grant.
Historical inventories retain their records; current inventories name canonical owners.
"""

from __future__ import annotations

from . import catalog_dense, catalog_source, catalog_spatial
from .catalog_effects import CAPABILITY_EFFECTS

_ORDER = (
    (16, "segmented_k_retained_operand_pipeline"),
    (16, "array_weight_retention"),
    (16, "exact_final_wide_loads"),
    (16, "source_axis_loop_scheduling"),
    (17, "source_exact_quantized_spatial_readout_cache"),
    (18, "disjoint_accumulator_readout_compute"),
    (18, "declared_persistent_scalar_state_in_scf"),
    (19, "final_k_causal_accumulator_readout_frontier"),
    (20, "loop_carried_completion_events"),
    (21, "effect_checked_schedule_composition"),
    (24, "current_source_quantized_residual_edges"),
    (23, "capacity_phased_convolution_rhs_residency"),
    (23, "source_qualified_convolution_readout_epochs"),
    (23, "parallel_emitted_variant_shape_evaluation"),
    (23, "pointwise_convolution_dense_view"),
    (22, "registered_source_convolution_exploration"),
    (22, "selected_logical_kernel_argument_binding"),
    (22, "shape_resource_schedule_selection"),
    (19, "checked_dense_pipeline_entry"),
    (22, "checked_dense_completion_epoch_emission"),
    (22, "fact_backed_dense_panel_epochs"),
    (23, "bounded_declared_settings_cycle_selection"),
    (23, "checked_disjoint_loaded_sum_epochs"),
    (23, "checked_initialization_and_uniform_readout_batches"),
    (23, "source_exact_spatial_matrix_gathers"),
    (23, "source_bound_rolling_convolution_rows"),
    (23, "checked_cross_spatial_array_reuse"),
    (23, "source_exact_convolution_readout_transfers"),
    (23, "checked_shared_input_group_blocks"),
    (24, "factored_immutable_tensor_layouts"),
    (24, "completed_input_with_rhs_frontier"),
    (24, "capacity_derived_m_panels_with_rhs_reuse"),
    (24, "retained_ready_input_placement"),
    (24, "staged_spatial_producer_bands"),
    (24, "source_checked_integer_observers"),
    (24, "counted_projection_pointer_binding"),
    (24, "completion_bound_operand_slots"),
    (24, "disjoint_future_row_transport"),
    (24, "direct_ring_convolution_row_fragments"),
    (24, "last_use_rhs_phase_transport"),
    (24, "sole_observer_contraction_rows"),
    (24, "bounded_horizontal_convolution_panels"),
    (24, "source_proven_shared_rhs_row_views"),
    (24, "shared_rhs_epochs"),
)
_LEGACY = "merlin.compile.scheduling."


def _canonical(value):
    if isinstance(value, str):
        from .library_sources import canonical_module

        if value == _LEGACY + "dense_contracts.DenseTiledIntegerReadout":
            return "merlin.compiler.schedule.dense_panel_contracts.DenseTiledIntegerReadout"
        if value.startswith("merlin."):
            renamed = canonical_module(value)
            if renamed != value:
                return renamed
            module, separator, symbol = value.rpartition(".")
            if separator and module not in {"merlin.compile", "merlin.compile.analysis"}:
                return canonical_module(module) + separator + symbol
        return value
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    return value


def scheduling_capabilities(version=27):
    """Return fresh usage metadata; only the selected contract grants access."""
    if type(version) is not int or version not in range(16, 28):
        raise ValueError("scheduling discovery requires explicit inventory 16 through 27")
    historical = min(version, 25)
    records = {
        row["name"]: row
        for family in (catalog_dense, catalog_source, catalog_spatial)
        for row in family.records(_LEGACY)
    }
    catalog = {
        "schema": "merlin.scheduling_api_catalog.v1",
        "inventory_version": historical,
        "authority": "discovery only; explicit source/effects/target admission remains required",
        "execution": "same Python/xDSL APIs in ordinary automatic and offline authoring",
        "capabilities": [records[name] for minimum, name in _ORDER if historical >= minimum],
    }
    if historical >= 18:
        from merlin.compiler.passes.composition import composition_capabilities

        from .library_sources import library_modules

        catalog["composition"] = composition_capabilities(selected_modules=library_modules(historical))
    if version in (26, 27):
        catalog = _canonical(catalog)
        catalog["inventory_version"] = version
    return catalog


def capability_effects(version=27):
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
