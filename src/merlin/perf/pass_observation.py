"""Bounded observations of actual transformation IR; no performance prices."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from xdsl.dialects import builtin, memref, scf
from xdsl.traits import get_effects

from merlin.transforms.analyses import analyze_ir
from merlin.transforms.certificates import parse_ir
from merlin.transforms.planning import plan_dependencies, plan_storage

ANALYSES = ("ssa", "dominance", "shapes", "effects", "aliasing", "liveness", "storage", "dependencies", "cost")
MAX_IR_BYTES = 256 * 1024
MAX_OPERATIONS = 16384


def metrics(text: str, *, index_bits: int | None = None) -> dict:
    """Recompute analyses from the selected IR, never reuse a stale pass plan."""
    if not isinstance(text, str) or len(text.encode()) > MAX_IR_BYTES:
        raise ValueError("pass observation exceeds its IR byte budget")
    module = parse_ir(text)
    operations = []
    for operation in module.walk():
        if len(operations) >= MAX_OPERATIONS:
            raise ValueError("pass observation exceeds its operation budget")
        operations.append(operation)
    names = Counter(operation.name for operation in operations)
    storage, dependencies = plan_storage(module), plan_dependencies(module)
    structural = analyze_ir(module, index_bits=index_bits)
    payload, unknown_allocations = 0, 0
    for operation in operations:
        if not isinstance(operation, memref.AllocOp):
            continue
        dtype = operation.memref.type
        element = dtype.get_element_type()
        shape = dtype.get_shape()
        if not isinstance(element, builtin.IntegerType) or element.width.data % 8 or any(size < 0 for size in shape):
            unknown_allocations += 1
            continue
        size = element.width.data // 8
        for extent in shape:
            size *= extent
        payload += size
    return {
        "ir_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "ir_bytes": len(text.encode()),
        "static_operations": len(operations),
        "operation_counts": dict(sorted(names.items())),
        "static_loops": sum(isinstance(operation, scf.ForOp) for operation in operations),
        "unknown_effect_operations": sum(get_effects(operation) is None for operation in operations),
        "known_static_allocation_bytes": payload,
        "unknown_allocations": unknown_allocations,
        "planned_block_reservation_bytes": storage.peak_reserved_bytes if not storage.unknown else None,
        "storage_unknowns": list(storage.unknown),
        "dependency_edges": sum(len(witness.edges) for witness in dependencies.witnesses),
        "dependency_unknowns": sum(len(witness.unknown_effect_indices) for witness in dependencies.witnesses),
        "max_block_live_values": max((block.peak_live_values for block in structural.blocks), default=0),
        "max_block_peak_known_logical_bytes": max(
            (block.peak_known_logical_bytes for block in structural.blocks), default=0
        ),
        "unknown_logical_sizes": len({value for block in structural.blocks for value in block.unknown_sizes}),
        "unresolved_alias_values": sum(
            value.memory is not False and value.alias_kind == "unknown" for value in structural.values
        ),
        "structural_unknown_count": len(structural.unknown),
        "pressure_scope": "known whole-value logical subtotals by lexical block; not physical or runtime peak storage",
        "physical_traffic": "UNKNOWN",
        "dynamic_work": "UNKNOWN",
        "cycles": "UNKNOWN",
    }


def observe_changes(changes, *, profile_sha256: str, index_bits: int | None = None) -> dict:
    rows = []
    previous = None
    for index, change in enumerate(changes):
        before = metrics(change.before_ir, index_bits=index_bits)
        after = metrics(change.after_ir, index_bits=index_bits)
        if previous is not None and before["ir_sha256"] != previous:
            raise ValueError("pass observations do not describe one connected IR pipeline")
        rows.append(
            {
                "index": index,
                "pass": change.kind,
                "parameters": change.parameters,
                "before": before,
                "after": after,
                "invalidated_analyses": list(ANALYSES) if before["ir_sha256"] != after["ir_sha256"] else [],
                "recomputed_analyses": [
                    "effects",
                    "storage",
                    "dependencies",
                    "aliasing",
                    "liveness",
                    "logical_pressure",
                ],
                "scope": list(change.certificate.scope),
                "static_operation_delta": after["static_operations"] - before["static_operations"],
            }
        )
        previous = after["ir_sha256"]
    return {
        "schema": "merlin.pass_observation.v1",
        "compiler_tool_profile_sha256": profile_sha256,
        "observer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passes": rows,
        "basis": "actual ordered before/after IR; conservatively invalidated analyses recomputed",
        "capsule_impact": "UNKNOWN until joined to original emitted artifacts",
        "ranking_authority": "none",
    }
