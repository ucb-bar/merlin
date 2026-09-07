"""Staged lowering across the core dialects.

linalg input -> contract -> schedule -> interface -> target -> runtime -> command
buffer. Stages are plain module->module transforms; cross-op legality lives in
``analyses``. Entry point: :func:`pipeline.lower_module` (any generic-MLIR payload);
:func:`pipeline.lower_repeated_rhs_matmul` is the reference workload built on top of it.
"""
from __future__ import annotations

from importlib import import_module

# Preserve the public convenience API without importing execution/oracle modules
# when a submitted compiler needs only a pure leaf helper.
_EXPORTS = {
    name: module for module, names in (
        ("dispatch_program", ("DispatchProgram", "build_dispatch_program",
                              "lower_model_to_dispatch_program", "prune_dead_nodes", "verify_program")),
        ("interface_lowering", ("LoweringError",)),
        ("global_plan", ("BufferRepresentation", "CycleInterval", "GlobalPlan", "PlanDemand",
                         "RegionAlternative", "ResourceOccupancy", "TransitionAlternative",
                         "ValueRepresentation", "verify_global_plan")),
        ("global_plan_emission", ("BoundaryMapping", "EmittedComponent", "GlobalPlanEmission",
                                  "GlobalPlanEmitter", "dispatch_digest", "emit_global_plan",
                                  "verify_global_plan_emission")),
        ("outline", ("OutlineError", "OutlineResult", "outline_dispatches")),
        ("passes", ("CATALOG", "DialectPlaneResult", "catalog", "run_dialect_plane")),
        ("schedule_dispatch", ("Schedule", "partition_dispatches")),
        ("pipeline", ("LoweringResult", "execute", "lower_module", "lower_repeated_rhs_matmul")),
    ) for name in names
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
