"""Current source owners of module names recorded in issued optimization catalog entries.

Issued catalog records (``resources/optimizations/authoring.json``) keep the module,
path, line and digest they were reviewed against, some under owners that have since
moved. Readers render those records against current owners through this map; the
records themselves are never rewritten. This module imports nothing from Merlin so
build tools can load it by path.
"""

from __future__ import annotations

_MOVED = {
    "merlin.verify": "merlin.verification",
    "merlin.verify.epilogue_applicability": "merlin.verification.epilogue_applicability",
    "merlin.compile": "merlin.compiler",
    "merlin.compile.analysis": "merlin.analysis",
    "merlin.compile.analysis.contraction_shapes": "merlin.compiler.ir.contraction_shapes",
    "merlin.compile.logical_abi": "merlin.compiler.ir.logical_abi",
    "merlin.compile.composition_catalog": "merlin.compiler.passes.composition",
    "merlin.compile.tuning_knobs": "merlin.compiler.selection.knobs",
    "merlin.compile.capacity": "merlin.compiler.selection.capacity",
    "merlin.compile.baremetal_model": "merlin.model",
    "merlin.compile.model_execution_inputs": "merlin.runtime.modelinputs",
    "merlin.compile.materialized_model_inputs": "merlin.compiler.frontend.materialized",
    "merlin.compile.route_before_build": "merlin.compiler.planning.model",
    "merlin.compile.complete_group_routing": "merlin.compiler.planning.complete",
    "merlin.compile.model_coverage": "merlin.compiler.planning.coverage",
    "merlin.compile.model_preflight": "merlin.compiler.frontend.preflight",
    "merlin.compile.model_interface": "merlin.compiler.partition.model",
    "merlin.compile.owned_host_model": "merlin.compiler.build.owned",
    "merlin.compile.owned_shared_source": "merlin.compiler.build.siblings",
    "merlin.compile.owned_spatial_source": "merlin.compiler.build.spatial",
    "merlin.compile.completed_model_link": "merlin.verification.completed_model",
    "merlin.compile.scalar_host_qualification": "merlin.verification.scalar_host",
    "merlin.compile.scheduling.block_schedule": "merlin.compiler.schedule.blocks",
    "merlin.compile.connected_regions": "merlin.compiler.lower.connected",
    "merlin.compile.connected_runtime": "merlin.runtime.connected",
    "merlin.compile.connected_host": "merlin.compiler.lower.connectedhost",
    "merlin.compile.fusion_discovery": "merlin.compiler.planning.frontiers",
    "merlin.compile.fusion_workflow": "merlin.compiler.planning.explore",
    "merlin.compile.compiler_observations": "merlin.compiler.observations",
    "merlin.llvmlower.passes_xdsl": "merlin.compiler.frontend.preprocess",
}


def current_owner(name: str) -> str:
    """The current owner of a recorded module name; current names map to themselves."""
    if name in _MOVED:
        return _MOVED[name]
    if name.startswith("merlin.verify."):
        return name.replace("merlin.verify", "merlin.verification", 1)
    if name.startswith("merlin.compile.analysis."):
        return name.replace("merlin.compile.analysis", "merlin.analysis", 1)
    return name.replace("merlin.compile.scheduling", "merlin.compiler.schedule", 1)
