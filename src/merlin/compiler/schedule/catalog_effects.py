"""Declared applicability effects for scheduling discoveries, never target laws or ranking."""

from __future__ import annotations

CAPABILITY_EFFECTS = {
    "segmented_k_retained_operand_pipeline": dict(
        families=("dense",),
        changes=dict(operand_load_per_compute="reduces"),
        admission="dense_segmented_pipeline.dense_segmented_blocks",
    ),
    "array_weight_retention": dict(
        families=("dense",), changes=dict(weight_load_per_compute="reduces"), admission=None
    ),
    "exact_final_wide_loads": dict(
        families=("dense", "convolution"),
        changes=dict(operand_transfer_extent="increases"),
        admission="selected_transfer_extent_field",
    ),
    "source_axis_loop_scheduling": dict(
        families=("dense",),
        changes=dict(operand_load_per_compute="reduces", weight_load_per_compute="reduces"),
        admission=None,
    ),
    "capacity_derived_m_panels_with_rhs_reuse": dict(
        families=("dense",),
        changes=dict(operand_load_per_compute="reduces"),
        admission="dense_panel_pipeline.dense_m_panel_capacity",
    ),
    "checked_shared_input_group_blocks": dict(
        families=("dense_sibling_group",), changes=dict(operand_load_per_compute="reduces"), admission=None
    ),
    "shared_rhs_epochs": dict(
        families=("dense_sibling_group",), changes=dict(operand_load_per_compute="reduces"), admission=None
    ),
    "capacity_phased_convolution_rhs_residency": dict(
        families=("convolution",),
        changes=dict(operand_load_per_compute="reduces"),
        admission="convolution_exploration.plan_convolution_rhs_residency",
    ),
    "registered_source_convolution_exploration": dict(
        families=("convolution",),
        changes=dict(operand_load_per_compute="reduces"),
        admission="convolution_exploration.plan_convolution_rhs_residency",
    ),
    "completed_input_with_rhs_frontier": dict(
        families=("convolution",), changes=dict(operand_load_per_compute="reduces"), admission=None
    ),
    "source_bound_rolling_convolution_rows": dict(
        families=("convolution",), changes=dict(operand_load_per_compute="reduces"), admission=None
    ),
    "checked_cross_spatial_array_reuse": dict(
        families=("convolution",), changes=dict(weight_load_per_compute="reduces"), admission=None
    ),
    "checked_initialization_and_uniform_readout_batches": dict(
        families=("dense",), changes=dict(broadcast_load_sites="increases"), admission=None
    ),
    "effect_checked_schedule_composition": dict(
        families=("dense", "convolution"),
        changes=dict(config_per_transfer="reduces", sync_sites="reduces"),
        admission=None,
    ),
    "checked_dense_completion_epoch_emission": dict(
        families=("dense",), changes=dict(sync_sites="reduces"), admission=None
    ),
    "fact_backed_dense_panel_epochs": dict(families=("dense",), changes=dict(sync_sites="reduces"), admission=None),
    "loop_carried_completion_events": dict(
        families=("dense", "convolution"), changes=dict(sync_sites="reduces"), admission=None
    ),
}
