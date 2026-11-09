"""Explicit region-cover fixtures shared by source and emitted-C conformance."""

from dataclasses import replace

from merlin.compiler.partition.semantic import extract_semantic_region
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation,
    CycleInterval,
    GlobalPlan,
    RegionAlternative,
    ValueRepresentation,
)


def plan_for(source, groups):
    pure = [node for node in source.nodes if node.effect == "pure"]
    rep = ValueRepresentation("memory", "dense", pure[0].type.dtype)
    alternatives = []
    for index, indices in enumerate(groups):
        region = extract_semantic_region(source, tuple(pure[i].id for i in indices))
        inputs = tuple(node.id for node in region.nodes if node.effect in {"input", "constant"})
        alternatives.append(
            RegionAlternative(
                f"region_{index}",
                tuple(indices),
                f"kernel_{index}",
                "device",
                CycleInterval.unknown("uncalibrated"),
                tuple(BufferRepresentation(key, replace(rep, dtype=source.node(key).type.dtype)) for key in inputs),
                tuple(
                    BufferRepresentation(key, replace(rep, dtype=source.node(key).type.dtype)) for key in region.outputs
                ),
            )
        )
    return GlobalPlan(tuple(alternatives), (), CycleInterval.unknown("uncalibrated"))
