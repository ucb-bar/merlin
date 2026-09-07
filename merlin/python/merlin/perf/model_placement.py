"""MAC-weighted placement of a captured model, independent of target lane names.

Region counts are useful for compiler coverage and useless as a performance weight: one large
contraction can dominate a model.  This joins the compiler's declared ``region -> lane`` placement to
Merlin's full-source MAC observer and reports exact contraction MACs per declared
lane.  It never decides which lane is "the accelerator"; that identity belongs to the target
descriptor, while this shared layer stays valid for meshes, vectors, GPUs, or multiple engines.
"""
from __future__ import annotations

import math
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.perf.model_macs import observe_model_macs as observe_contractions


@dataclass(frozen=True)
class PreparedCapturedSource:
    """Host-only parsed/outlined source reused across independent whole-graph audits."""

    source_sha256: str
    parsed_module: Any
    outlined: Any
    graph: Any
    logical_dispatch_digest: str


def prepare_captured_source(model_mlir: str | Path) -> PreparedCapturedSource:
    """Parse and outline immutable source once, retaining an exact byte/digest binding."""
    from merlin.common.mlir_query import parse
    from merlin.xdsl_dialects.lowering.dispatch_program import lower_model_to_dispatch_program
    from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest

    source_path = Path(model_mlir)
    source = source_path.read_bytes()
    parsed = parse(source_path)
    outlined, graph = lower_model_to_dispatch_program(parsed, prune=False)
    outlined.module.verify()
    return PreparedCapturedSource(
        source_sha256=hashlib.sha256(source).hexdigest(), parsed_module=parsed,
        outlined=outlined, graph=graph, logical_dispatch_digest=dispatch_digest(graph))


def _require_prepared_source(
        model_mlir: str | Path,
        prepared: PreparedCapturedSource,
) -> PreparedCapturedSource:
    from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest

    source_sha256 = hashlib.sha256(Path(model_mlir).read_bytes()).hexdigest()
    if prepared.source_sha256 != source_sha256:
        raise ValueError("prepared captured source does not match immutable source bytes")
    if dispatch_digest(prepared.graph) != prepared.logical_dispatch_digest:
        raise ValueError("prepared captured graph changed after host construction")
    return prepared


def captured_global_graph(
        model_mlir: str | Path, *,
        prepared_source: PreparedCapturedSource | None = None,
) -> dict[str, Any]:
    """Compile the actual capture into the graph consumed by the shared global emitter.

    The unpruned graph binds every driver operation, including scalar glue and constants. This is
    the logical objective, not a claim about what a candidate's target codegen has implemented.
    """
    prepared = (_require_prepared_source(model_mlir, prepared_source)
                if prepared_source is not None else prepare_captured_source(model_mlir))
    graph = prepared.graph
    return {
        "schema": "captured_global_graph_v1", "status": "verified",
        "source_sha256": prepared.source_sha256,
        "logical_dispatch_digest": prepared.logical_dispatch_digest,
        "nodes": len(graph.nodes), "dispatches": graph.n_dispatches,
        "buffers": len(graph.buffers), "arguments": len(graph.args),
        "results": len(graph.results),
        "dispatch_program": graph.to_dict(),
        "compiler_entrypoints": {
            "outline": "merlin.xdsl_dialects.lowering.dispatch_program.lower_model_to_dispatch_program",
            "plan": "merlin.xdsl_dialects.lowering.outlined_plan_emission.plan_dispatch_fusion",
            "emit": "merlin.xdsl_dialects.lowering.outlined_plan_emission.OutlinedGlobalPlanEmitter",
        },
        "licence": "logical full-model graph; candidate target emission remains independently checked",
    }


def _string_attr(op: Any, key: str) -> str | None:
    attr = getattr(op, "attributes", {}).get(key)
    value = getattr(attr, "data", attr)
    value = getattr(value, "data", value)
    return str(value) if isinstance(value, str) and value else None


def contraction_placement(model_mlir: str | Path,
                          placement_rows: Sequence[Mapping[str, Any]], *,
                          target: str | None = None, entry: str | None = None,
                          prepared_source: PreparedCapturedSource | None = None) -> dict[str, Any]:
    """Return exact contraction work by the lane declared for each captured region."""
    lanes: dict[str, str] = {}
    conflicts: list[str] = []
    for row in placement_rows:
        if not isinstance(row, Mapping):
            continue
        region, lane = str(row.get("region") or ""), str(row.get("lane") or "")
        if not region or not lane:
            continue
        if region in lanes and lanes[region] != lane:
            conflicts.append(region)
        lanes[region] = lane

    by_lane: Counter[str] = Counter()
    by_op: Counter[str] = Counter()
    by_regime: Counter[str] = Counter()
    macs_by_regime: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    source = (_require_prepared_source(model_mlir, prepared_source).parsed_module
              if prepared_source is not None else model_mlir)
    observed = (observe_contractions(source, entry=entry)
                if entry is not None else observe_contractions(source))
    for ordinal, (op, shape) in enumerate(observed):
        region = _string_attr(op, "prov.region_id")
        status = getattr(shape, "status", "derived")
        macs = math.prod(tuple(shape.parallel) + tuple(shape.reduction)) if status == "derived" else None
        lane = lanes.get(region or "")
        memory_regime = "unknown"
        capacity_rows = working_rows = None
        if target:
            from merlin.targetgen import memory_regime as MR
            dtype = shape.dtypes[0] if getattr(shape, "dtypes", ()) else None
            store, capacity_rows = MR.operand_store(target, dtype=dtype)
            if store is not None and capacity_rows:
                sized: list[int] = []
                for operand in getattr(op, "operands", ()):
                    ty = getattr(operand, "type", None)
                    if not hasattr(ty, "get_shape"):
                        sized = []
                        break
                    try:
                        operand_dtype = str(ty.get_element_type())
                        value = store.working_set_rows(list(ty.get_shape()), operand_dtype)
                    except Exception:  # noqa: BLE001 - one unsized operand keeps regime unknown
                        sized = []
                        break
                    if value is None:
                        sized = []
                        break
                    sized.append(int(value))
                if sized:
                    working_rows = sum(sized)
                    memory_regime = MR.classify(working_rows, working_rows, int(capacity_rows))
        record = {"ordinal": ordinal, "region": region, "op": shape.op,
                  "source_op_index": getattr(shape, "source_op_index", None),
                  "mac_status": status, "mac_basis": getattr(shape, "reason", "static contraction domain"),
                  "parallel": list(shape.parallel), "reduction": list(shape.reduction),
                  "macs": macs, "lane": lane, "memory_regime": memory_regime,
                  "working_set_rows": working_rows, "capacity_rows": capacity_rows}
        if macs is None or region is None or lane is None:
            unresolved.append(record)
            continue
        by_lane[lane] += macs
        by_op[shape.op] += macs
        by_regime[memory_regime] += 1
        macs_by_regime[memory_regime] += macs
        rows.append(record)

    known = sum(by_lane.values())
    unknown_domains = [row for row in unresolved if row["macs"] is None]
    unresolved_known_macs = sum(int(row["macs"]) for row in unresolved if row["macs"] is not None)
    unresolved_macs = None if unknown_domains else unresolved_known_macs
    total = None if unknown_domains else known + unresolved_known_macs
    status = ("complete" if total and not unresolved and not conflicts else
              "partial" if rows or unresolved else "UNKNOWN")
    return {
        "schema": "model_contraction_placement_v1",
        "status": status,
        "contraction_count": len(rows) + len(unresolved),
        "total_contraction_macs": total or None,
        "known_domain_macs": known + unresolved_known_macs,
        "unknown_mac_domain_count": len(unknown_domains),
        "mac_domain_coverage": "partial" if unknown_domains else "derived",
        "placement_basis": "compiler-declared region lanes; not issued accelerator work proof",
        "known_placement_macs": known,
        "unresolved_placement_macs": unresolved_macs,
        "macs_by_lane": dict(sorted(by_lane.items())),
        "mac_fraction_by_lane": ({lane: value / total for lane, value in sorted(by_lane.items())}
                                 if total else {}),
        "macs_by_contraction_op": dict(sorted(by_op.items())),
        "memory_regime": {
            "status": ("derived" if target and by_regime and set(by_regime) != {"unknown"}
                       else "UNKNOWN"),
            "region_counts": dict(sorted(by_regime.items())),
            "macs_by_regime": dict(sorted(macs_by_regime.items())),
            "mac_fraction_by_regime": ({name: value / known
                                        for name, value in sorted(macs_by_regime.items())}
                                       if known else {}),
            "double_buffer_eligible_regime": "fits_double",
            "note": ("full operand footprint using each operand's dtype; not a tiled allocation, "
                     "physical traffic or proved overlap schedule"),
        },
        "contractions": sorted(rows, key=lambda row: (-int(row["macs"]), int(row["ordinal"]))),
        "unresolved": unresolved,
        "conflicting_regions": sorted(set(conflicts)),
        "licence": (
            "exact structural contraction MAC weighting; non-contraction work and cycles are not "
            "priced here"
        ),
    }
