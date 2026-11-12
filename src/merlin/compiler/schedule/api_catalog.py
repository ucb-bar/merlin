"""Source-only scheduling discovery and applicability joins for the current SDK.

Declarations live with their dense, spatial and source-contract families. Their
ordering remains explicit here. Discovery selects no target law or import grant.
"""

from __future__ import annotations

from . import catalog_dense, catalog_source, catalog_spatial
from .catalog_effects import CAPABILITY_EFFECTS
from .library_sources import CURRENT_SDK_VERSION, library_modules, require_current_sdk

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
    from merlin.compiler.passes.composition import composition_capabilities

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
