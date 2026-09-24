"""Answer-free, operation-complete Phase 0 application capture inventory.

Every parsed MLIR operation is accounted for. Admission is not lowering or compile acceptance.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def _application_operation_inventory(path: str | Path, target: str, cap_map: dict) -> dict:
    """Answer-free, operation-complete inventory for one application capture.

    A provenance tag names the frontend source; it is not the operation's computation. In particular,
    im2col gathers inherit their convolution's `prov.family=contraction`. The semantic family of a
    `linalg.generic` therefore comes from its body, and a nested body op is a component of that region,
    not a second independent accelerator demand. Unknowns remain rows, never disappear from a count.
    """
    from merlin.common import mlir_query as mq
    from merlin.targetgen import model_coverage as mc
    from merlin.targetgen import semantic_families as sf
    from merlin.targetgen.eligibility import RegionDescriptor, is_eligible
    from merlin.xdsl_dialects.lowering import contraction_coverage as cc

    p = Path(path)
    try:
        data = p.read_bytes()
        module = mq.parse(data.decode("utf-8"))
    except Exception as exc:
        raise ValueError(
            f"declared application capture {p}: cannot inventory MLIR: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        extents = mc._contraction_extents(module)  # noqa: PLC2701 -- region-coverage shape authority
    except Exception as exc:
        raise ValueError(
            f"declared application capture {p}: cannot inventory shapes: {type(exc).__name__}: {exc}"
        ) from exc

    module_quantization = mq.attr_str(module, "prov.quantization")
    grouped: dict[str, dict] = {}
    counts: Counter = Counter()
    n_operations = 0
    for ordinal, op in enumerate(mq.walk(module)):
        n_operations += 1
        name = mq.op_name(op)
        provenance = mq.provenance(op)
        frontend = provenance.get("prov.aten")
        source_op = provenance.get("prov.op")
        callee_attr = op.properties.get("callee") if name == "func.call" else None
        callee = getattr(getattr(callee_attr, "root_reference", None), "data", None)
        canonical = frontend or source_op or (f"func.call @{callee}" if callee else name)
        parent = op.parent_op()
        nested = False
        while parent is not None:
            if mq.op_name(parent).startswith("linalg."):
                nested = True
                break
            parent = parent.parent_op()

        structural = name in {"builtin.module", "func.func", "func.return", "linalg.yield", "scf.yield"}
        support = name == "arith.constant" or name.startswith(("tensor.", "memref.", "scf."))
        family: str | None = None
        family_basis = "unknown"
        if name == "linalg.generic":
            try:
                generic_kind = cc.classify_generic(op)
            except Exception as exc:
                raise ValueError(
                    f"declared application capture {p}: cannot classify operation {ordinal} ({name}): {exc}"
                ) from exc
            family = {
                "contraction": "contraction",
                "sum-reduction": "reduction",
                "max-reduction": "reduction",
                "absmax": "reduction",
                "other-reduction": "reduction",
                "movement": "movement",
                "elementwise": "elementwise_map",
            }.get(generic_kind)
            family_basis = "generic_body" if family else "unclassified_generic_body"
        elif name.startswith("linalg."):
            family = "reduction" if name == "linalg.reduce" else sf.from_op(name.rpartition(".")[2])
            family_basis = "linalg_op" if family else "unclassified_linalg_op"
        elif name.startswith(("arith.", "math.")):
            family, family_basis = "elementwise_map", "dialect_operation"
        elif name.startswith(("tensor.", "memref.")):
            family, family_basis = "movement", "dialect_operation"
        elif not structural:
            # A custom-dialect op may have a known semantic name. The source region's family tag is
            # not used as authority: it is copied onto unrelated operations in captured models.
            family = sf.from_op(name.rpartition(".")[2])
            family_basis = "operation_name" if family else "unclassified_operation"

        operand_types = [mq.type_shape_dtype(value.type) for value in op.operands]
        result_types = [mq.type_shape_dtype(value.type) for value in op.results]
        operand_dtypes = sorted({dtype for _shape, dtype in operand_types if dtype})
        result_dtypes = sorted({dtype for _shape, dtype in result_types if dtype})
        result_shapes = [shape for shape, _dtype in result_types if shape]
        operand_shapes = [shape for shape, _dtype in operand_types if shape]
        m, k, n, rank = extents.get(id(op), (None, None, None, None))
        if rank is None and result_shapes:
            rank = len(result_shapes[0])
        elif rank is None and operand_shapes:
            rank = len(operand_shapes[0])
        shape = {"M": m, "K": k, "N": n, "rank": rank} if m is not None else None
        shape_confidence = (
            "observed_iteration_space"
            if shape is not None
            else "result_type"
            if result_shapes
            else "operand_type"
            if operand_shapes
            else "unknown"
            if name.startswith("linalg.") and not structural and not nested
            else "not_applicable"
        )
        input_format = mc._elem_dtype(op)  # noqa: PLC2701 -- same dtype authority as region coverage
        if input_format is None:
            input_format = next(
                (mc._ELEM_DTYPE[dtype] for _shape, dtype in operand_types if dtype in mc._ELEM_DTYPE),  # noqa: PLC2701
                None,
            )
        accumulator_dtypes = None
        if family == "contraction":
            accumulator_dtypes = (
                sorted(
                    {
                        dtype
                        for child in mq.walk(op)
                        if mq.op_name(child) in {"arith.addf", "arith.addi"}
                        for value in child.results
                        if (dtype := mq.type_shape_dtype(value.type)[1])
                    }
                )
                or None
            )
        if structural:
            disposition, reason = "structural", "IR container or terminator; not an independent demand"
        elif nested:
            disposition, reason = "component", "inside a linalg region; accounted with its parent computation"
        elif support:
            disposition = "support_required"
            reason = "constant, tensor/memory, or control-flow op requires lowering; not a separate compute capsule"
        elif name == "func.call":
            disposition = "unclassified"
            reason = "call requires a resolved callee/body or a declared external host lowering"
        elif family is None or shape_confidence == "unknown" or (input_format is None and not operand_dtypes):
            disposition, reason = "unclassified", "semantic family, operand format, or required shape is unknown"
        elif input_format is None:
            disposition = "host_required"
            reason = "known operand dtype has no hardware format mapping; host lowering remains required"
        else:
            verdict = is_eligible(
                RegionDescriptor(
                    op=name.rpartition(".")[2], family=family, in_dtype=input_format, m=m, k=k, n=n, rank=rank
                ),
                cap_map,
            )
            disposition = (
                "hardware_admitted" if verdict.eligible else "unclassified" if verdict.undetermined else "host_required"
            )
            reason = "hardware capability only; lowering unverified" if verdict.eligible else verdict.reason

        quant_evidence = {
            key: value
            for key, value in sorted(provenance.items())
            if any(token in key for token in ("quant", "scale", "format"))
        }
        layout_evidence = {
            key: value
            for key, value in sorted(provenance.items())
            if any(token in key for token in ("layout", "transpose", "conv_path"))
        }
        signature = {
            "operation": canonical,
            "mlir_operation": name,
            "frontend_op": frontend,
            "provenance_op": source_op,
            "callee": callee,
            "semantic_family": family,
            "family_basis": family_basis,
            "operand_dtypes": operand_dtypes,
            "operand_format": input_format,
            "result_dtypes": result_dtypes,
            "result_shapes": result_shapes,
            "accumulator_dtypes": accumulator_dtypes,
            "contraction_shape": shape,
            "shape_confidence": shape_confidence,
            "quant_evidence": quant_evidence or None,
            "layout_evidence": layout_evidence or None,
            "disposition": disposition,
            "reason": reason,
            "provenance_present": bool(frontend or source_op),
        }
        key = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        slot = grouped.setdefault(key, {**signature, "count": 0, "ordinals": []})
        slot["count"] += 1
        slot["ordinals"].append(ordinal)
        counts[disposition] += 1
        if not (frontend or source_op) and not structural and not nested:
            counts["untagged"] += 1
        if shape_confidence == "unknown":
            counts["shape_unknown"] += 1

    if n_operations == 0 or sum(row["count"] for row in grouped.values()) != n_operations:
        raise ValueError(f"declared application capture {p}: parsed operations were not fully inventoried")
    rows = [grouped[key] for key in sorted(grouped)]
    return {
        "capture": f"{p.parent.name}/{p.name}",
        "capture_sha256": hashlib.sha256(data).hexdigest(),
        "capture_quantization": module_quantization,
        "n_operations": n_operations,
        "n_signatures": len(rows),
        "counts": dict(sorted(counts.items())),
        "status": "incomplete" if counts["unclassified"] else "inventoried",
        "signatures": rows,
        "scope": "parsed MLIR operations only; hardware admission is not compiler lowering or model execution",
    }


def application_demand_inventory(applications: dict[str, str | Path], target: str, *, detailed: bool = False) -> dict:
    """Inventory declared derivation applications; keep exact operation rows in a generated sidecar.

    The default is a reviewable requirement summary. ``detailed=True`` retains all grouped signatures
    and exact operation ordinals, suitable for a digest-checked generated sidecar, not a hand-edited spec.
    """
    if not applications:
        return {
            "schema_version": 1,
            "status": "not_declared",
            "coverage_status": "not_applicable",
            "applications": {},
            "n_operations": 0,
        }
    from merlin.targetgen import claim_models as cm
    from merlin.targetgen.eligibility import capability_map_for_target

    cap_map = capability_map_for_target(target)
    output: dict[str, dict] = {}
    for label, path in sorted(applications.items()):
        if cm.is_claim_bundle(label) or cm.is_claim_bundle(Path(path).resolve().parent.name):
            raise ValueError(f"application {label!r} is a held-out claim model and cannot derive Phase 0 demands")
        output[str(label)] = _application_operation_inventory(path, target, cap_map)
    full = {
        "schema_version": 1,
        "status": "incomplete" if any(row["status"] == "incomplete" for row in output.values()) else "inventoried",
        "coverage_status": "unverified",
        "applications": output,
        "n_operations": sum(row["n_operations"] for row in output.values()),
        "basis": "only explicitly supplied application captures; held-out workload_spec.models are not read",
        "coverage_note": "operation inventory has not been matched to generated capsules or a submitted compiler",
    }
    if detailed:
        return full

    digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    operation_groups: dict[str, dict] = {}
    compact_apps: dict[str, dict] = {}
    for label, app in output.items():
        compact_apps[label] = {
            key: app[key]
            for key in (
                "capture",
                "capture_sha256",
                "capture_quantization",
                "n_operations",
                "n_signatures",
                "counts",
                "status",
            )
        }
        for row in app["signatures"]:
            if row["disposition"] in {"structural", "component"}:
                continue
            shape = row["contraction_shape"]
            rank = shape["rank"] if shape else len(row["result_shapes"][0]) if row["result_shapes"] else None
            shape_class = f"{'contraction' if shape else 'result'}:rank_{rank if rank is not None else 'unknown'}"
            key_fields = {
                "operation": row["operation"],
                "mlir_operation": row["mlir_operation"],
                "semantic_family": row["semantic_family"],
                "operand_format": row["operand_format"],
                "disposition": row["disposition"],
                "shape_class": shape_class,
            }
            key = json.dumps(key_fields, sort_keys=True, separators=(",", ":"))
            group = operation_groups.setdefault(key, {**key_fields, "count": 0, "sources": {}})
            group["count"] += row["count"]
            source = group["sources"].setdefault(label, {"capture_sha256": app["capture_sha256"], "count": 0})
            source["count"] += row["count"]
    return {
        "schema_version": 1,
        "status": full["status"],
        "coverage_status": "unverified",
        "n_operations": full["n_operations"],
        "n_signatures": sum(app["n_signatures"] for app in output.values()),
        "applications": compact_apps,
        "operation_groups": [operation_groups[key] for key in sorted(operation_groups)],
        "full_inventory_sha256": digest,
        "basis": full["basis"],
        "coverage_note": full["coverage_note"],
    }
