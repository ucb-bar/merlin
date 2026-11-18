"""Answer-free, operation-complete Phase 0 application capture inventory.

Every operation in the normalized program handed to backends is accounted for. Admission is not
lowering or compile acceptance; raw and normalized identities are retained separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import weakref
from collections import Counter
from pathlib import Path

from merlin.common.digest import is_sha256
from merlin.compiler.ir import mlir_query as mq
from merlin.development.targetgen.applications.inventory import application_graph_inventory

# ---- application_inventory ---------------------------------------------------------------------


def quantization_parameters(op) -> dict[str, int | None] | None:
    """Read per-tensor quantization operands/properties from parsed SSA, not provenance.

    A dynamic zero point or an unrecognized producer stays unknown. In particular,
    a dtype-compatible capture is not evidence for a symmetric quantization rule.
    """
    from merlin.compiler.ir import mlir_query as mq

    if mq.op_name(op) not in {"quant_ext.quantize_per_tensor", "quant_ext.dequantize_per_tensor"}:
        return None

    def integer_attribute(attribute) -> int | None:
        value = getattr(getattr(attribute, "value", None), "data", None)
        return value if type(value) is int else None

    def attribute(owner, field):
        return owner.properties[field] if field in owner.properties else owner.attributes.get(field)

    def constant_integer(value) -> int | None:
        owner = getattr(value, "owner", None)
        if owner is None or not hasattr(owner, "properties"):
            return None
        if mq.op_name(owner) == "tensor.splat" and len(owner.operands) == 1:
            owner = getattr(owner.operands[0], "owner", None)
        if owner is None or not hasattr(owner, "properties") or mq.op_name(owner) != "arith.constant":
            return None
        return integer_attribute(attribute(owner, "value"))

    zero_point = constant_integer(op.operands[2]) if len(op.operands) == 3 else None
    parameters = {"zero_point": zero_point}
    for field in ("quant_min", "quant_max"):
        parameters[field] = integer_attribute(attribute(op, field))
    return parameters


def _reduction_provenance_hint(op, mq) -> tuple[dict, dict | None]:
    """Suggest a missing tag from adjacent operations without inventing source provenance.

    Even agreeing tags cannot prove this reduction's frontend semantics. The source tag
    remains absent for admission; the hint only helps diagnose capture metadata loss.
    """
    provenance = mq.provenance(op)
    if mq.op_name(op) != "linalg.reduce" or provenance.get("prov.aten") or len(op.operands) < 2 or not op.results:
        return provenance, None
    # The tag on a fill can name this reduction only when that fill initializes THIS reduction
    # exclusively. Two reductions may share one tagged zero while both have agreeing tagged
    # consumers; agreement on either side alone does not identify which reduction owns the tag.
    init_uses = tuple(op.operands[-1].uses)
    if len(init_uses) != 1 or init_uses[0].operation is not op:
        return provenance, None
    initializer = op.operands[-1].owner
    init_prov = mq.provenance(initializer) if hasattr(initializer, "attributes") else {}
    identity = (init_prov.get("prov.region_id"), init_prov.get("prov.aten"))
    users = [use.operation for result in op.results for use in result.uses]
    if (
        not all(identity)
        or not users
        or any(
            (mq.provenance(user).get("prov.region_id"), mq.provenance(user).get("prov.aten")) != identity
            for user in users
        )
    ):
        return provenance, None
    return provenance, {
        "method": "reduction_initializer_and_consumers_agree_v1",
        "status": "unverified",
        "suggested_frontend_op": identity[1],
        "region_id": identity[0],
        "consumer_count": len(users),
    }


_PREPARED_APPLICATION_SOURCES = weakref.WeakKeyDictionary()


class PreparedApplicationSource:
    """One program's parsed source, with exact byte and IR mutation checks.

    This is a call-local parsing optimization. Independent source witnesses
    still reopen and reprove their complete original source before acceptance.
    """

    def __init__(self, path: Path, normalized: str):
        from merlin.compiler.frontend.patterns.capture_normalization import normalize_capture_mlir
        from merlin.compiler.ir import mlir_query as mq

        path = Path(path)
        raw = path.read_bytes()
        actual, normalization = normalize_capture_mlir(raw.decode("utf-8"))
        if actual != normalized:
            raise ValueError("prepared application source differs from normalized capture bytes")
        self.module = mq.parse(actual)
        # Weak ownership retains no program after its call-local carrier dies.
        # Original issued values are not mutable attributes on the carrier.
        _PREPARED_APPLICATION_SOURCES[self] = (path, raw, actual, normalization, self.module, str(self.module))

    def reopen(self, path):
        """Refuse changed bytes, another path, or mutation of the parsed IR."""
        original = _PREPARED_APPLICATION_SOURCES.get(self)
        if original is None:
            raise ValueError("application source has no original issued preparation")
        selected, raw, normalized, normalization, module, rendered = original
        if (
            Path(path) != selected
            or selected.read_bytes() != raw
            or self.module is not module
            or str(module) != rendered
        ):
            raise ValueError("prepared application source or parsed module changed")
        return raw, normalized, copy.deepcopy(normalization), module


def _application_operation_inventory(
    path: str | Path,
    target: str,
    cap_map: dict,
    *,
    include_graph: bool = False,
    capability_contract: dict | None = None,
    software_spec: dict | None = None,
    host_capabilities: dict | None = None,
    prepared_source: PreparedApplicationSource | None = None,
) -> dict:
    """Answer-free, operation-complete inventory for one application capture.

    A provenance tag names the frontend source; it is not the operation's computation. In particular,
    im2col gathers inherit their convolution's `prov.family=contraction`. The semantic family of a
    `linalg.generic` therefore comes from its body, and a nested body op is a component of that region,
    not a second independent accelerator demand. Unknowns remain rows, never disappear from a count.
    """
    from merlin.compiler.frontend.patterns.capture_normalization import normalize_capture_mlir
    from merlin.compiler.ir import contraction_shapes
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition import outline as cc
    from merlin.development.targetgen.applications.inventory import operation_structure, verify_capture_receipt
    from merlin.development.targetgen.conformance import model_coverage as mc
    from merlin.target.capabilities import semantic_families as sf
    from merlin.target.capabilities.eligibility import RegionDescriptor, is_eligible

    p = Path(path)
    try:
        if prepared_source is None:
            data = p.read_bytes()
            normalized, normalization = normalize_capture_mlir(data.decode("utf-8"))
            module = mq.parse(normalized)
        elif type(prepared_source) is PreparedApplicationSource:
            data, normalized, normalization, module = prepared_source.reopen(p)
        else:
            raise ValueError("application inventory needs an original prepared source")
    except Exception as exc:
        raise ValueError(
            f"declared application capture {p}: cannot inventory MLIR: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        extents = contraction_shapes.contraction_extents(module)  # region-coverage shape authority
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
        provenance, provenance_hint = _reduction_provenance_hint(op, mq)
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
        # Keep the ordered ABI and linalg access pattern in the digest-bound sidecar. A dtype set plus
        # a family/shape class cannot distinguish this matmul from another generic of the same size.
        structure = operation_structure(op)
        from merlin.development.targetgen.applications import inventory as access_observations

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
        elif support or (family == "movement" and name.startswith("linalg.")):
            disposition = "support_required"
            reason = (
                "layout/data movement requires an explicit lowering; a movement-family capability "
                "does not prove this operation executes on the accelerator"
                if family == "movement" and name.startswith("linalg.")
                else "constant, tensor/memory, or control-flow op requires lowering; not a separate compute capsule"
            )
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
                    op=name.rpartition(".")[2],
                    family=family,
                    in_dtype=input_format,
                    out_dtype=result_dtypes[0] if len(result_dtypes) == 1 else None,
                    m=m,
                    k=k,
                    n=n,
                    rank=rank,
                    form=sf.operation_form(source_op or frontend or name.rpartition(".")[2], carrier_op=name),
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
            **({"provenance_hint": provenance_hint} if provenance_hint is not None else {}),
            "provenance_op": source_op,
            "callee": callee,
            "semantic_family": family,
            "family_basis": family_basis,
            "operand_dtypes": operand_dtypes,
            "ordered_operand_types": structure["ordered_operand_types"],
            "operand_format": input_format,
            "result_dtypes": result_dtypes,
            "ordered_result_types": structure["ordered_result_types"],
            "result_shapes": result_shapes,
            "indexing_maps": structure["indexing_maps"],
            "iterator_types": structure["iterator_types"],
            "body_operations": structure["body_operations"],
            "accumulator_dtypes": accumulator_dtypes,
            "contraction_shape": shape,
            "shape_confidence": shape_confidence,
            "quant_evidence": quant_evidence or None,
            "layout_evidence": layout_evidence or None,
            # Graph-derived access observations in the software declarations' vocabulary; None = unknown.
            **access_observations.observe(op, family),
            "disposition": disposition,
            "reason": reason,
            "provenance_present": bool(frontend or source_op),
        }
        parameters = quantization_parameters(op)
        if parameters is not None:
            signature["quantization_parameters"] = parameters
        if disposition == "unclassified" and software_spec is not None and host_capabilities is not None:
            # Independent hardware eligibility remains unknown for dtype-changing operations.
            # A separately pinned and reviewed host declaration can nevertheless resolve their
            # placement without duplication in accelerator software semantics. Use the same
            # exact-signature screen as whole-program admission.
            from merlin.development.targetgen.applications.operation_accounting import admit_operation_row

            admission = admit_operation_row(
                signature,
                software_spec=software_spec,
                capability_contract=capability_contract,
                capability_map=cap_map,
                host_capabilities=host_capabilities,
                source_operations=(op,),
            )
            if (
                admission["host_admission"]["status"] == "admitted"
                and admission["host_admission"]["reviewed"]
                and admission["accelerator_admission"]["status"] != "admitted"
            ):
                disposition = signature["disposition"] = "host_required"
                signature["reason"] = "exact reviewed pinned host declaration admits this signature"
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
    trace_path = p.parent / "frontend-trace.json"
    capture_receipt = verify_capture_receipt(p)
    integerization = capture_receipt.pop("capture_integerization", None)
    result = {
        "capture": f"{p.parent.name}/{p.name}",
        "capture_source_path": str(p.absolute()),
        "capture_sha256": hashlib.sha256(data).hexdigest(),
        "frontend_trace": (
            {"source_path": str(trace_path.absolute()), "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()}
            if trace_path.is_file() and not trace_path.is_symlink()
            else None
        ),
        "capture_receipt": capture_receipt,
        "capture_normalization": normalization,
        "capture_quantization": module_quantization,
        "n_operations": n_operations,
        "n_signatures": len(rows),
        "counts": dict(sorted(counts.items())),
        "status": "incomplete" if counts["unclassified"] else "inventoried",
        "signatures": rows,
        "scope": (
            "parsed operations in the canonically normalized capture only; hardware admission "
            "is not compiler lowering or model execution"
        ),
    }
    if integerization is not None and integerization.get("source_quantization") == module_quantization:
        result["capture_integerization"] = integerization
    if include_graph:
        result["operation_graph"] = application_graph_inventory(p)
    if prepared_source is not None:
        prepared_source.reopen(p)
    return result


def application_operation_inventory(
    path: str | Path,
    target: str,
    *,
    capability_contract: dict,
    workload_id: str,
    workload_role: str = "validation",
) -> dict:
    """Observe one complete validation program without deriving any capsule.

    Unlike ``application_demand_inventory``, this evaluator entrypoint can read
    held-out models. Its result labels the role explicitly; it cannot certify
    compiler support or make a validation capture a legal derivation source.
    """
    if not isinstance(workload_id, str) or not workload_id or workload_role != "validation":
        raise ValueError("evaluation inventory requires an explicit validation workload identity")
    from merlin.target.capabilities.eligibility import capability_map_from_contract

    result = _application_operation_inventory(
        path, target, capability_map_from_contract(capability_contract), include_graph=True
    )
    result["workload_identity"] = {
        "workload_id": workload_id,
        "workload_role": workload_role,
        "coverage_scope": "full_capture",
    }
    result["purpose"] = "evaluation_only; not a Phase 0 derivation input"
    return result


def application_demand_inventory(
    applications: dict[str, str | Path],
    target: str,
    *,
    detailed: bool = False,
    capability_contract: dict | None = None,
    software_spec: dict | None = None,
    host_capabilities: dict | None = None,
    include_graph: bool = False,
    application_metadata: dict[str, dict] | None = None,
) -> dict:
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
    from merlin.development.targetgen.conformance import claim_applications as cm
    from merlin.target.capabilities.eligibility import capability_map_for_target, capability_map_from_contract

    cap_map = (
        capability_map_from_contract(capability_contract)
        if capability_contract is not None
        else capability_map_for_target(target)
    )
    output: dict[str, dict] = {}
    for label, path in sorted(applications.items()):
        if cm.is_claim_bundle(label) or cm.is_claim_bundle(Path(path).resolve().parent.name):
            raise ValueError(f"application {label!r} is a held-out claim model and cannot derive Phase 0 demands")
        output[str(label)] = _application_operation_inventory(
            path,
            target,
            cap_map,
            include_graph=include_graph,
            capability_contract=capability_contract,
            software_spec=software_spec,
            host_capabilities=host_capabilities,
        )
        identity = (application_metadata or {}).get(str(label))
        if identity is not None:
            if not isinstance(identity, dict) or identity.get("workload_role", "unknown") not in {
                "iteration",
                "validation",
                "unknown",
            }:
                raise ValueError("application metadata requires an explicit iteration/validation/unknown workload role")
            if identity.get("coverage_scope", "unknown") not in {"full_capture", "representative_subset", "unknown"}:
                raise ValueError("application metadata requires an explicit full/representative/unknown coverage scope")
            if identity.get("workload_role") == "validation":
                raise ValueError("held-out validation captures cannot derive Phase 0 iteration demands")
            for field in ("workload_id", "source_workload_id"):
                source = identity.get(field)
                if source is not None and (not isinstance(source, str) or not source):
                    raise ValueError(f"application metadata {field} must be a nonempty source identity")
                if source is not None and cm.is_claim_bundle(source):
                    raise ValueError(f"application metadata {field} names a held-out validation source")
            output[str(label)]["workload_identity"] = dict(identity)
    full = {
        "schema_version": 2 if include_graph else 1,
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
                "capture_receipt",
                "capture_normalization",
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


# ---- source_demand_inventory -------------------------------------------------------------------
# Join routing demands to complete, independently numbered source operations.


def indexed_demand_operations(operations):
    """Retain full source indices while omitting only recognized fill initializers.

    The demand reader omits carriers tagged ``fill``. Those are not a license to
    omit arbitrary arithmetic: only a literal constant, its tensor splat, or an
    actual linalg fill can be outside the routing census. Callers must still
    account for *all* operations in source bindings and task ownership.

    Untagged operations remain in this inventory. If the demand reader missed
    one, exact operation identity/count checks must refuse the incomplete join.
    """
    indexed = []
    for index, operation in enumerate(operations):
        name = mq.op_name(operation)
        tag = mq.attr_str(operation, "prov.op")
        if tag == "fill" and name not in ("arith.constant", "tensor.splat", "linalg.fill"):
            raise ValueError(f"source operation {index} ({name}) has an unsupported omitted fill carrier")
        if name == "linalg.fill" or tag == "fill":
            continue
        indexed.append((index, operation))
    return indexed


def _declared_data_operand_format(op) -> str | None:
    """Observe a declared view/quantizer's data role, without granting support.

    Quantizer scales and zero points are metadata operands, not alternate data
    precisions. Unknown operations, malformed declarations and dynamic data
    stay unknown. No requested compile dtype participates in this observation.
    """
    from math import prod

    from xdsl.dialects.builtin import Float32Type, IntegerType, TensorType

    from merlin.common import target_data as qf
    from merlin.compiler.ir import mlir_query as mq

    if op.regions or op.successors or len(op.results) != 1:
        return None

    def static_tensor(value):
        typ = value.type
        return typ if isinstance(typ, TensorType) and all(dim > 0 for dim in typ.get_shape()) else None

    if not op.operands:
        return None
    data, result = static_tensor(op.operands[0]), static_tensor(op.results[0])
    if data is None or result is None:
        return None
    name = mq.op_name(op)
    if name in {"tensor.collapse_shape", "tensor.expand_shape", "tensor.cast"}:
        if (
            len(op.operands) != 1
            or data.element_type != result.element_type
            or prod(data.get_shape()) != prod(result.get_shape())
        ):
            return None
    elif name in {"quant_ext.quantize_per_tensor", "quant_ext.dequantize_per_tensor"}:
        if len(op.operands) != 3 or data.get_shape() != result.get_shape():
            return None
        scale, zero = (static_tensor(value) for value in op.operands[1:])
        if (
            scale is None
            or zero is None
            or scale.get_shape()
            or zero.get_shape()
            or not isinstance(scale.element_type, Float32Type)
            or not isinstance(zero.element_type, IntegerType)
            or zero.element_type.width.data != 64
        ):
            return None
        floating, integer = (data, result) if name == "quant_ext.quantize_per_tensor" else (result, data)
        if not isinstance(floating.element_type, Float32Type) or not isinstance(integer.element_type, IntegerType):
            return None
    else:
        return None
    spelling = str(data.element_type)
    return qf.get(spelling).name if qf.has(spelling) else None


def _observe_declared_data_formats(module, demands):
    """Join only previously absent observations to the exact original carriers."""
    from dataclasses import replace

    from merlin.capture.demands import ModelDemandIncomplete
    from merlin.compiler.ir import mlir_query as mq

    carriers = [op for op in mq.walk(module) if mq.attr_str(op, "prov.op") not in {None, "fill"}]
    if len(carriers) != len(demands):
        raise ModelDemandIncomplete("declared operand observation lacks complete ordered carrier identity")
    observed = []
    for op, demand in zip(carriers, demands, strict=True):
        if mq.op_name(op) != demand.carrier_op or mq.attr_str(op, "prov.region_id") != demand.region_id:
            raise ModelDemandIncomplete("declared operand observation names another source carrier")
        fmt = _declared_data_operand_format(op) if demand.captured_input_formats == () else None
        observed.append(replace(demand, elem_fmt=fmt, captured_input_formats=(fmt,)) if fmt is not None else demand)
    return observed


def model_op_demands_checked(
    linalg_mlir: str, in_fmt: str, weight_fmt: str | None = None, *, source_kind: str = "captured_provenance_v1"
) -> list:
    """Verify complete, ordered contraction identity and shapes.

    Captured mode requires provenance plus independent body/shape agreement.
    Typed mode reads exact initialized integer dots without provenance or admission.
    """
    from merlin.capture.demands import ModelDemandIncomplete, model_op_demands

    if type(source_kind) is not str or source_kind != "captured_provenance_v1":
        from merlin.development.targetgen.conformance.model_coverage import typed_static_integer_demands

        return typed_static_integer_demands(linalg_mlir, in_fmt, weight_fmt, source_kind=source_kind)
    from collections import Counter
    from math import prod

    from merlin.compiler.ir import contraction_shapes as ks
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition import outline as cc

    try:
        module = mq.parse(linalg_mlir)
    except Exception as exc:
        raise ModelDemandIncomplete(f"cannot parse whole-model MLIR for demand completeness: {exc}") from exc

    structural = []
    for ordinal, op in enumerate(mq.walk(module)):
        name = mq.op_name(op)
        if name in cc.MATMUL_OPS:
            structural.append((ordinal, op, name))
        elif name == "linalg.generic":
            try:
                if cc.classify_generic(op) == "contraction":
                    structural.append((ordinal, op, name))
            except Exception as exc:
                raise ModelDemandIncomplete(f"cannot classify linalg.generic at operation {ordinal}: {exc}") from exc

    # The shape observer is independent of the provenance-tag reader. It deliberately may decline a
    # structural contraction whose maps/extent are undecidable; that is an explicit refusal here.
    observed = {id(op): shape for op, shape in ks.observe_contractions(module)}
    expected: Counter = Counter()
    expected_sequence: list[tuple[str, int, int, int, int]] = []
    unshaped: list[str] = []
    untagged: list[str] = []
    for ordinal, op, name in structural:
        marker = f"operation {ordinal} ({name})"
        tag = getattr((op.attributes or {}).get("prov.op"), "data", None)
        if not isinstance(tag, str) or not tag:
            untagged.append(marker)
            continue
        shape = observed.get(id(op))
        if shape is None or len(shape.parallel) < 2 or not shape.reduction:
            unshaped.append(f"{marker}, prov.op={tag!r}")
            continue
        parallel = tuple(int(x) for x in shape.parallel)
        reduction = tuple(int(x) for x in shape.reduction)
        identity = (tag, prod(parallel[:-2]), parallel[-2], prod(reduction), parallel[-1])
        expected[identity] += 1
        expected_sequence.append(identity)

    if untagged or unshaped:
        raise ModelDemandIncomplete(
            f"structural contraction inventory incomplete: {len(structural)} contraction(s); "
            f"untagged={untagged[:8]}, unshaped={unshaped[:8]}"
        )

    demands = _observe_declared_data_formats(module, model_op_demands(linalg_mlir, in_fmt, weight_fmt))
    actual: Counter = Counter()
    actual_sequence: list[tuple[str, int, int, int, int]] = []
    missing_shape: list[str] = []
    for index, demand in enumerate(demands):
        if demand.family != "contraction":
            continue
        if not demand.has_shape or demand.batch is None:
            missing_shape.append(f"demand {index} ({demand.op})")
            continue
        identity = (demand.op, demand.batch, demand.m, demand.k, demand.n)
        actual[identity] += 1
        actual_sequence.append(identity)
    absent = expected - actual
    excess = actual - expected
    out_of_order = [
        {"ordinal": i, "structural": want, "demand": got}
        for i, (want, got) in enumerate(zip(expected_sequence, actual_sequence))
        if want != got
    ]
    if missing_shape or absent or excess or out_of_order:

        def _rows(counter: Counter) -> list[dict]:
            return [
                {"op": op, "batch": batch, "M": m, "K": k, "N": n, "count": count}
                for (op, batch, m, k, n), count in sorted(counter.items())[:8]
            ]

        raise ModelDemandIncomplete(
            f"contraction demands differ from parsed MLIR: structural={sum(expected.values())}, "
            f"demands={sum(actual.values())}, missing_shape={missing_shape[:8]}, "
            f"missing={_rows(absent)}, extra={_rows(excess)}, out_of_order={out_of_order[:8]}"
        )
    return demands
