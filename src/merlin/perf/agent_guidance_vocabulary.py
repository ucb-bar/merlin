"""The declared vocabularies the agent brief is written in, and the ladder it is ordered by.

Split out of :mod:`merlin.perf.agent_guidance` because it is a different kind of thing: these are
DECLARATIONS -- which CCA axes a finding is about, which effects an opportunity needs, the order the
macro tiers are descended in -- while the module they serve is the logic that reads evidence and
writes a brief. Keeping them together put 375 lines of table in front of every reader of that logic,
and it pushed the module past the size the structure gate allows.

One rule survives the move intact and is the reason the scope/effect vocabulary is not simply a
literal here: it is READ FROM THE MANIFEST SCHEMA. A second opinion about a declared vocabulary is
how the two come to disagree, and they did -- `memory_planning` was added to the schema and the copy
here was left behind, so 22 of 29 packages validated against the schema and then failed here with
"has invalid effects", aborting every launch before authoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _declared_surface_vocabulary(*, contract: str | Path | None = None) -> tuple[frozenset[str], frozenset[str]]:
    """``(scopes, effects)`` READ FROM THE MANIFEST SCHEMA, which is the one place they are declared.

    These were duplicated here as literal frozensets, and the copies drifted the moment one was
    updated: `memory_planning` was added to the schema (two committed package surfaces needed it for
    workspace-lifetime reuse) and this list was left behind, so 22 of 29 gemmini packages passed
    `oot_runner.load_package` -- which validates against the schema -- and then raised
    "has invalid effects" here, aborting every phase-2 launch before authoring. A second opinion
    about a declared vocabulary is how the two come to disagree, so there is now only one.
    """
    from merlin.targetgen.contract.schemas import load_schema

    surfaces = (load_schema("manifest", contract=contract).get("properties") or {}).get("optimization_surfaces") or {}
    fields = (surfaces.get("items") or {}).get("properties") or {}
    scopes = (fields.get("scope") or {}).get("enum")
    effects = ((fields.get("effects") or {}).get("items") or {}).get("enum")
    if not scopes or not effects:
        raise ValueError(
            "the manifest schema declares no optimization_surfaces scope/effects enum, so the "
            "surface vocabulary cannot be derived; refusing to fall back to a baked copy"
        )
    return frozenset(str(v) for v in scopes), frozenset(str(v) for v in effects)


OPPORTUNITY_EFFECTS = {
    "missing_occupancy": frozenset({"latency_hiding", "issue", "tiling"}),
    "accelerator_bubbles": frozenset({"synchronization", "issue", "tiling", "latency_hiding", "residency"}),
    "exposed_movement": frozenset({"movement", "residency", "layout", "encoding", "latency_hiding"}),
    "encoding_transitions": frozenset({"encoding", "layout", "fusion"}),
    "unknown_latency_hiding": frozenset({"latency_hiding", "issue"}),
    "unknown_encoding_transitions": frozenset({"encoding", "layout", "fusion"}),
}

STRUCTURAL_EFFECTS = {
    "residency_restaged": frozenset({"movement", "residency"}),
    "memory_round_trip": frozenset({"movement", "residency", "fusion"}),
    "unfused_single_consumer": frozenset({"fusion", "movement"}),
}

# CCA axes are more precise than broad effects.  They let an agent move from "movement is high" to
# "this exact surface controls whether an intermediate materializes" without baking a target opcode or
# compiler filename into the harness.  The effect match remains as a compatibility path for packages
# frozen before this optional field existed.
FINDING_AXES = {
    "host_memory_hotspot": frozenset(
        {
            "communication.intermediate_materialized",
            "coverage.non_contraction_op_fraction",
            "layout.transpose_materialized",
            "compute.epilogue",
            "compute.activation_vectorization",
        }
    ),
    "whole_model_lowering_declined": frozenset(
        {
            "coverage.claimed_mac_fraction",
            "coverage.unclaimed_op_classes",
            "envelope.calls_in_loop",
            "envelope.runtime_calls",
            "communication.intermediate_materialized",
            "dispatch.loop_offloaded",
        }
    ),
    "inter_lane_boundary_pressure": frozenset(
        {
            "communication.host_device_bytes",
            "communication.engine_engine_bytes",
            "communication.intermediate_materialized",
            "communication.copy_compute_overlap",
            "coverage.non_contraction_op_fraction",
        }
    ),
    "heterogeneous_model_placement": frozenset(
        {"coverage.claimed_mac_fraction", "coverage.unclaimed_op_classes", "coverage.non_contraction_op_fraction"}
    ),
    "memory_regime_pressure": frozenset(
        {"memory.capacity_fit", "memory.onchip_resident", "dispatch.double_buffered_banks", "dispatch.dma_overlap"}
    ),
    "arithmetic_expansion": frozenset(
        {"compute.contraction_form", "coverage.claimed_mac_fraction", "coverage.unclaimed_op_classes"}
    ),
    "movement_regression": frozenset(
        {
            "communication.host_device_bytes",
            "communication.engine_engine_bytes",
            "communication.intermediate_materialized",
            "communication.resident_across_calls",
            "communication.copy_compute_overlap",
            "dispatch.dma_overlap",
            "memory.onchip_resident",
            "layout.transpose_materialized",
            "layout.operand_major",
        }
    ),
    "unpriced_movement": frozenset(
        {
            "communication.host_device_bytes",
            "communication.engine_engine_bytes",
            "memory.dma_pattern",
            "memory.onchip_resident",
        }
    ),
    "synchronization_regression": frozenset({"communication.fences", "dispatch.dma_overlap", "simt.barriers_in_loop"}),
    # The trade spans every axis the single-term findings each own, because that is the point of
    # it: the edit it describes spent one of them to buy another.
    "resource_trade": frozenset(
        {
            "coverage.claimed_mac_fraction",
            "communication.host_device_bytes",
            "communication.engine_engine_bytes",
            "communication.fences",
            "compute.epilogue",
            "dispatch.loop_offloaded",
        }
    ),
    "residency_restaged": frozenset(
        {
            "compute.accumulator_resident",
            "spatial.accumulator_resident",
            "communication.resident_across_calls",
            "memory.onchip_resident",
        }
    ),
    "memory_round_trip": frozenset(
        {
            "communication.intermediate_materialized",
            "communication.resident_across_calls",
            "compute.accumulator_resident",
            "memory.onchip_resident",
        }
    ),
    "unfused_single_consumer": frozenset(
        {"compute.epilogue", "communication.intermediate_materialized", "layout.transpose_materialized"}
    ),
    "unverified_encoding_transitions": frozenset(
        {"layout.transpose_materialized", "layout.operand_major", "memory.access_pattern"}
    ),
    "unknown_occupancy_and_overlap": frozenset(
        {
            "dispatch.dma_overlap",
            "dispatch.double_buffered_banks",
            "dispatch.descriptor_reuse",
            "dispatch.loop_offloaded",
            "communication.copy_compute_overlap",
        }
    ),
    "issued_movement_regression": frozenset(
        {
            "memory.onchip_resident",
            "communication.resident_across_calls",
            "communication.intermediate_materialized",
            "layout.transpose_materialized",
        }
    ),
    "unresolved_target_encoding": frozenset(
        {"layout.operand_major", "layout.transpose_materialized", "memory.access_pattern"}
    ),
    "dispatch_configuration_regression": frozenset({"dispatch.descriptor_reuse", "dispatch.loop_offloaded"}),
    "loop_offload_regression": frozenset({"dispatch.loop_offloaded"}),
    "artifact_synchronization_regression": frozenset(
        {"communication.fences", "dispatch.dma_overlap", "simt.barriers_in_loop"}
    ),
    "trace_regression": frozenset(
        {
            "communication.resident_across_calls",
            "memory.onchip_resident",
            "communication.fences",
            "dispatch.descriptor_reuse",
            "compute.epilogue",
        }
    ),
}

OPPORTUNITY_AXES = {
    "missing_occupancy": FINDING_AXES["unknown_occupancy_and_overlap"],
    "accelerator_bubbles": FINDING_AXES["unknown_occupancy_and_overlap"],
    "exposed_movement": FINDING_AXES["movement_regression"],
    "encoding_transitions": FINDING_AXES["unverified_encoding_transitions"],
    "unknown_latency_hiding": FINDING_AXES["unknown_occupancy_and_overlap"],
    "unknown_encoding_transitions": FINDING_AXES["unverified_encoding_transitions"],
}

# The generic mechanism checklist for a complete-model optimization loop.  This is not a list of
# target tricks: every row is expressed in the cross-target CCA vocabulary.  Its purpose is to make
# omissions visible.  A row is covered only when the current artifact exposes evidence *and* the
# candidate package declares a real AST surface for one of the controlling axes.
GAP_AXES = {
    "whole_model_placement_and_coverage": frozenset(
        {
            "coverage.claimed_mac_fraction",
            "coverage.unclaimed_op_classes",
            "coverage.non_contraction_op_fraction",
            "envelope.calls_in_loop",
            "envelope.runtime_calls",
        }
    ),
    "arithmetic_lowering": frozenset(
        {"compute.contraction_form", "compute.reduction_form", "compute.epilogue", "compute.activation_vectorization"}
    ),
    "encoding_and_layout": frozenset(
        {"layout.operand_major", "layout.transpose_materialized", "memory.access_pattern"}
    ),
    "movement_and_materialization": frozenset(
        {
            "communication.host_device_bytes",
            "communication.engine_engine_bytes",
            "communication.intermediate_materialized",
            "memory.onchip_resident",
        }
    ),
    "residency_across_operations": frozenset(
        {
            "compute.accumulator_resident",
            "spatial.accumulator_resident",
            "communication.resident_across_calls",
            "memory.onchip_resident",
        }
    ),
    "fusion_and_host_boundaries": frozenset(
        {
            "compute.epilogue",
            "communication.intermediate_materialized",
            "coverage.non_contraction_op_fraction",
            "envelope.calls_in_loop",
        }
    ),
    "dispatch_and_loop_offload": frozenset({"dispatch.descriptor_reuse", "dispatch.loop_offloaded"}),
    "latency_hiding_and_double_buffering": frozenset(
        {"dispatch.dma_overlap", "dispatch.double_buffered_banks", "communication.copy_compute_overlap"}
    ),
    "synchronization": frozenset({"communication.fences", "simt.barriers_in_loop", "dispatch.dma_overlap"}),
    "capacity_and_contention": frozenset({"memory.capacity_fit", "dispatch.double_buffered_banks"}),
    "exact_quantized_epilogues_and_residuals": frozenset(
        {"compute.epilogue", "compute.activation_vectorization", "coverage.non_contraction_op_fraction"}
    ),
    "arena_lifetimes_and_reuse": frozenset({"memory.capacity_fit", "communication.intermediate_materialized"}),
    "entry_abi_and_runtime_overhead": frozenset({"envelope.runtime_calls", "envelope.calls_in_loop"}),
}


# Phase 2 is a whole-program optimization loop. A flat list of actionable findings can make a
# cheap local rewrite look as important as deleting a model-wide representation boundary. Keep
# the ordering target-neutral: tiers name compiler/dataflow scope, never a target opcode or model.
MACRO_OPTIMIZATION_LADDER = (
    {
        "tier": 0,
        "name": "correctness_and_regression_repair",
        "gaps": (),
        "purpose": "repair emission, correctness, work, movement, synchronization, or trace regressions",
    },
    {
        "tier": 1,
        "name": "whole_program_work_deletion",
        "gaps": (
            "whole_model_placement_and_coverage",
            "fusion_and_host_boundaries",
            "exact_quantized_epilogues_and_residuals",
            "arena_lifetimes_and_reuse",
            "entry_abi_and_runtime_overhead",
        ),
        "purpose": "delete whole-model tasks, materializations, boundaries, redundant representations, and runtime work",
    },
    {
        "tier": 2,
        "name": "global_dataflow_and_representation",
        "gaps": (
            "encoding_and_layout",
            "movement_and_materialization",
            "residency_across_operations",
        ),
        "purpose": "select encodings globally and keep values resident instead of converting, spilling, or reloading them",
    },
    {
        "tier": 3,
        "name": "global_execution_and_latency_hiding",
        "gaps": (
            "dispatch_and_loop_offload",
            "latency_hiding_and_double_buffering",
            "synchronization",
            "capacity_and_contention",
        ),
        "purpose": "keep engines occupied by issuing independent work early and minimizing exposed waits",
    },
    {
        "tier": 4,
        "name": "operator_layer_and_tile_efficiency",
        "gaps": ("arithmetic_lowering",),
        "purpose": "improve an operator or tile only after larger whole-program opportunities are terminal for this revision",
    },
    {
        "tier": 5,
        "name": "local_scalar_cleanup",
        "gaps": (),
        "purpose": "apply peepholes and scalar cleanup only when no higher-tier mechanism remains actionable",
    },
)


# Quantized-region plans name semantic ownership roles, never paths.  This host-owned table maps a
# known role onto CCA/effect vocabulary; the real editable coordinates still have to resolve through
# ``inspect_compiler_package`` and the immutable edit contract below.  A candidate receipt therefore
# cannot grant itself authority by inventing a path, effect, or new role.
QUANTIZED_REGION_EDIT_ROLES = {
    "source_epilogue_semantics": {
        "effects": frozenset({"quantization", "fusion", "dtype"}),
        "cca_axes": frozenset(
            {
                "compute.epilogue",
                "coverage.non_contraction_op_fraction",
            }
        ),
        "purpose": "recognize exact source quantization stages and ownership",
    },
    "global_quant_domain_planner": {
        "effects": frozenset({"quantization", "encoding", "residency", "fusion"}),
        "cca_axes": frozenset(
            {
                "compute.epilogue",
                "communication.intermediate_materialized",
                "communication.resident_across_calls",
            }
        ),
        "purpose": "choose compatible domains and legal producer/consumer regions globally",
    },
    "target_epilogue_emitter": {
        "effects": frozenset({"quantization", "dtype", "encoding", "fusion"}),
        "cca_axes": frozenset({"compute.epilogue", "layout.operand_major"}),
        "purpose": "lower an admitted epilogue through a target-supported physical readout",
    },
    "target_residual_region_emitter": {
        "effects": frozenset({"quantization", "fusion", "residency", "movement"}),
        "cca_axes": frozenset(
            {
                "compute.epilogue",
                "communication.intermediate_materialized",
                "communication.resident_across_calls",
                "memory.onchip_resident",
            }
        ),
        "purpose": "lower admitted wide-domain residual operations without host materialization",
    },
    "target_encoding_and_residency": {
        "effects": frozenset({"encoding", "layout", "residency", "movement"}),
        "cca_axes": frozenset(
            {
                "layout.operand_major",
                "communication.resident_across_calls",
                "memory.onchip_resident",
                "memory.capacity_fit",
            }
        ),
        "purpose": "realize direct encoded boundaries under target-derived capacity",
    },
}


def macro_optimization_order() -> dict[str, Any]:
    """Return the host-owned largest-to-smallest Phase-2 selection contract."""
    return {
        "schema": "macro_optimization_order_v1",
        "selection": (
            "choose the lowest numbered nonterminal tier; within it prefer quantified dynamic "
            "extent, then cross-model coverage; use a matched reduced cycle witness when unlike "
            "costs trade off"
        ),
        "descent_gate": (
            "every higher tier must have a retained structural change or an explicit source/plan-"
            "bound refusal or no-op for the current compiler revision"
        ),
        "prohibited_shortcut": (
            "an easy local rewrite, capsule win, or unknown cost cannot bypass an actionable "
            "whole-program or global-dataflow mechanism"
        ),
        "tiers": [dict(row) for row in MACRO_OPTIMIZATION_LADDER],
    }
