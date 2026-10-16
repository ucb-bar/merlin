"""Inspectable Phase 0 operation partitions, without inventing frontend counts.

The denominator is the exact normalized MLIR inventory, not the global PyTorch
operator registry. Frontend annotations can fan out over many lowered operations;
their occurrence counts are explicitly not original graph invocation counts.
Admission screens declarations only and never certifies a compiler lowering.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping

from merlin.common.digest import is_sha256
from merlin.targetgen.software_spec import admit_operation

SCHEMA = "merlin.phase0.operation_accounting.v1"
_DISPOSITIONS = frozenset(
    {
        "structural",
        "component",
        "support_required",
        "hardware_admitted",
        "host_required",
        "unclassified",
    }
)
_NON_COMPUTE = {
    "structural": "structural",
    "component": "nested_component",
    "support_required": "support_lowering_required",
}


def _canonical(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _graph_identity(graph: dict | None) -> dict | None:
    """Retain an independent exact-graph roster for downstream ledger checks."""
    if not isinstance(graph, dict) or graph.get("schema") != "merlin.application_graph.v1":
        return None
    normalized = graph.get("normalized_graph") or {}
    operations, edges = normalized.get("operations") or [], normalized.get("edges") or []
    nodes = [
        {key: row.get(key) for key in ("operation_id", "ordinal", "mlir_operation", "parent_operation_id")}
        for row in operations
    ]
    uses = [
        {key: row.get(key) for key in ("id", "consumer_operation_id", "producer_operation_id", "value_id", "type")}
        for row in edges
    ]
    return {
        "normalized_mlir_sha256": normalized.get("mlir_sha256"),
        "n_operations": len(nodes),
        "n_edges": len(uses),
        "node_roster_sha256": hashlib.sha256(_canonical(nodes)).hexdigest(),
        "edge_roster_sha256": hashlib.sha256(_canonical(uses)).hexdigest(),
        "scope": "independent projection of the selected normalized capture graph",
    }


def _count(value, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"operation inventory {field} must be a nonnegative integer")
    return value


def _validate_application(label: str, application: dict) -> None:
    if not isinstance(application, dict) or not is_sha256(application.get("capture_sha256")):
        raise ValueError(f"operation inventory application {label!r} requires its capture SHA256")
    total = _count(application.get("n_operations"), f"{label}.n_operations")
    rows = application.get("signatures")
    if not isinstance(rows, list):
        raise ValueError(f"operation inventory {label!r} must retain detailed signatures")
    if _count(application.get("n_signatures"), f"{label}.n_signatures") != len(rows):
        raise ValueError(f"operation inventory {label!r} signature count differs from its rows")
    ordinals, dispositions = set(), Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"operation inventory {label!r} has a non-mapping signature")
        count = _count(row.get("count"), f"{label}.signature.count")
        positions = row.get("ordinals")
        if count == 0 or not isinstance(positions, list) or len(positions) != count:
            raise ValueError(f"operation inventory {label!r} signature count/ordinals disagree")
        for position in positions:
            if type(position) is not int or not 0 <= position < total or position in ordinals:
                raise ValueError(f"operation inventory {label!r} has duplicate or invalid ordinals")
            ordinals.add(position)
        for key in ("operation", "mlir_operation"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"operation inventory {label!r} signature requires {key}")
        if row.get("disposition") not in _DISPOSITIONS:
            raise ValueError(f"operation inventory {label!r} has an unknown disposition")
        dispositions[row["disposition"]] += count
    if len(ordinals) != total:
        raise ValueError(f"operation inventory {label!r} does not account for every MLIR operation")
    saved_counts = application.get("counts")
    if saved_counts is not None:
        if not isinstance(saved_counts, dict):
            raise ValueError(f"operation inventory {label!r} counts must be a mapping")
        for name, value in saved_counts.items():
            _count(value, f"{label}.counts.{name}")
            if value > total:
                raise ValueError(f"operation inventory {label!r} count exceeds its total")
        for disposition in _DISPOSITIONS:
            if saved_counts.get(disposition, 0) != dispositions[disposition]:
                raise ValueError(f"operation inventory {label!r} disposition counts disagree")


def _observed_signature(row: dict) -> dict:
    """Project actual ABI observations; do not turn annotations into semantics."""
    from merlin.targetgen.semantic_families import operation_form

    observed = {"family": row.get("semantic_family")}
    carrier = row.get("mlir_operation") or ""
    form = operation_form(carrier.rpartition(".")[2], carrier_op=carrier)
    if form is not None:
        observed["form"] = form
    for source, key in (
        ("ordered_operand_types", "ordered_operand_dtypes"),
        ("ordered_result_types", "ordered_result_dtypes"),
    ):
        if isinstance(row.get(source), list):
            observed[key] = [value.get("dtype") for value in row[source]]
    if row.get("operand_format"):
        observed["operand_dtype"] = row["operand_format"]
    elif len(row.get("operand_dtypes") or []) == 1:
        observed["operand_dtype"] = row["operand_dtypes"][0]
    for source, key in (("accumulator_dtypes", "accum_dtype"), ("result_dtypes", "readout_dtype")):
        values = row.get(source) or []
        if len(values) == 1:
            observed[key] = values[0]
    # Contraction iteration rank includes the reduction loop: it is not tensor rank.
    for source in ("ordered_result_types", "ordered_operand_types"):
        shape = next(
            (
                item.get("shape")
                for item in row.get(source) or []
                if isinstance(item, dict) and isinstance(item.get("shape"), list)
            ),
            None,
        )
        if shape is not None:
            observed["rank"] = len(shape)
            break
    # Access observations derived from the graph (application_inventory); absent ones stay unresolved.
    for key in ("layout", "tails", "broadcasting", "aliasing"):
        if row.get(key) is not None:
            observed[key] = row[key]
    if row.get("quantization_parameters") is not None:
        observed["quantization_parameters"] = row["quantization_parameters"]
    # Composition observed in a written program (an epilogue stage and the family that produced its
    # operand). Captured application rows carry none, so they stay unresolved exactly as before.
    for key, value in (row.get("composed_observation") or {}).items():
        if key in {"epilogues", "composed_with", "scale_granularity"} and value is not None:
            observed[key] = value
    if row.get("numeric_screen") is not None:
        observed["numeric_screen"] = row["numeric_screen"]
    geometry = row.get("contraction_shape") or {}
    observed["dimensions"] = {axis: geometry[axis] for axis in ("M", "K", "N") if type(geometry.get(axis)) is int}
    return observed


def _software_admissions(spec: dict | None, row: dict, signature: dict) -> list[dict]:
    if spec is None:
        return []
    declarations = spec.get("operations") or []
    direct = [
        entry
        for entry in declarations
        if row["operation"] in entry.get("ops", []) or (not entry.get("ops") and entry.get("id") == row["operation"])
    ]
    decisions = []
    for declaration in direct or declarations:
        placement = declaration["placement"]
        decision = admit_operation({**spec, "operations": [declaration]}, row["operation"], signature, placement)
        if "declaration" in decision:
            decisions.append({**decision, "placement": placement})
    return sorted(decisions, key=lambda item: item["declaration"])


def _hardware_admission(row: dict, signature: dict, contract: dict | None, capability_map: dict | None) -> dict:
    disposition = row["disposition"]
    if disposition in _NON_COMPUTE:
        return {
            "status": "not_applicable",
            "basis": "inventory_structure",
            "reason": "not an independent accelerator compute demand",
        }
    if contract is None:
        return {
            "status": "unknown",
            "basis": "no_selected_capability_contract",
            "reason": "saved inventory dispositions are not current hardware admission evidence",
        }
    family, dtype = signature.get("family"), signature.get("operand_dtype")
    if family is None or dtype is None or row.get("shape_confidence") == "unknown":
        return {
            "status": "unknown",
            "basis": "selected_capability_contract",
            "reason": "required semantic family, operand dtype or shape observation is absent",
        }
    from merlin.targetgen.eligibility import (
        RegionDescriptor,
        is_eligible,
        providers_from_contract,
        undetermined_families_from_contract,
    )
    from merlin.targetgen.semantic_families import primitives_of

    capabilities = (
        [capability_map[family]]
        if family in capability_map
        else [capability_map[primitive] for primitive in primitives_of(family) if primitive in capability_map]
    )
    if any(capability.ranks and signature.get("rank") is None for capability in capabilities):
        return {
            "status": "unknown",
            "basis": "selected_capability_contract",
            "reason": "declared rank constraints need an observed tensor rank",
        }
    if any(capability.layouts for capability in capabilities) and signature.get("layout") is None:
        return {
            "status": "unknown",
            "basis": "selected_capability_contract",
            "reason": "declared layout constraints need a reviewed layout observation",
        }
    dimensions = signature["dimensions"]
    verdict = is_eligible(
        RegionDescriptor(
            op=row["mlir_operation"].rpartition(".")[2],
            family=family,
            in_dtype=dtype,
            out_dtype=signature.get("readout_dtype"),
            weight_dtype=(
                (row.get("ordered_operand_types") or [{}, {}])[1].get("dtype")
                if family == "contraction" and len(row.get("ordered_operand_types") or []) >= 2
                else None
            ),
            rank=signature.get("rank"),
            form=signature.get("form"),
            layout=signature.get("layout"),
            m=dimensions.get("M"),
            k=dimensions.get("K"),
            n=dimensions.get("N"),
        ),
        capability_map,
        undetermined=undetermined_families_from_contract(contract),
        providers=providers_from_contract(contract),
        # A fused-only capability admits a stage only behind an observed producer of its operand.
        fused_with=tuple(signature.get("composed_with") or ()) or None,
    )
    return {
        "status": "admitted" if verdict.eligible else "unknown" if verdict.undetermined else "unsupported",
        "basis": "selected_capability_contract",
        "reason": verdict.reason,
        "refusal": verdict.refusal,
        "engines": list(verdict.engines),
        "units": list(verdict.units),
        "qualification": "declaration_screen_only; not hardware or compiler qualification",
    }


def _classification(row: dict, software: list[dict], hardware: dict, has_spec: bool) -> str:
    if row["disposition"] in _NON_COMPUTE:
        return _NON_COMPUTE[row["disposition"]]
    accepted = [decision for decision in software if decision["status"] == "admitted"]
    if hardware["status"] == "admitted" and any(
        decision["placement"] in {"accelerator", "fused_accelerator"} for decision in accepted
    ):
        return "accelerator_candidate"
    if any(decision["placement"] == "host" for decision in accepted):
        return "host_required"
    if hardware["status"] == "unsupported":
        return "host_required"
    if row.get("semantic_family") is None:
        return "unresolved"
    if has_spec and (not software or all(decision["status"] == "unsupported" for decision in software)):
        return "unsupported"
    return "unresolved"


def _accelerator_admission(row: dict, software: list[dict], hardware: dict, has_spec: bool) -> dict:
    candidates = [decision for decision in software if decision["placement"] in {"accelerator", "fused_accelerator"}]
    accepted = [decision for decision in candidates if decision["status"] == "admitted"]
    status = "unknown"
    non_compute = row["disposition"] in _NON_COMPUTE
    if non_compute:
        status = "not_applicable"
        candidates = []
        accepted = []
    elif hardware["status"] == "admitted" and accepted:
        status = "admitted"
    elif hardware["status"] == "unsupported" or (
        has_spec and candidates and all(decision["status"] == "unsupported" for decision in candidates)
    ):
        status = "unsupported"
    return {
        "status": status,
        "reviewed": bool(accepted),
        "review_status": "not_applicable" if non_compute else "reviewed" if accepted else "unknown",
        "hardware_admission": hardware,
        "software_admissions": candidates,
        "reason": (
            "not an independent accelerator compute demand; support lowering, when required, remains unverified"
            if non_compute
            else "independent selected accelerator declaration screen; placement and lowering remain unselected"
        ),
    }


def _support_partition(accelerator: dict, host: dict) -> str:
    if accelerator["status"] == "not_applicable":
        return "not_independent_compute"
    if accelerator["status"] == "admitted" and host["status"] == "admitted":
        return "accelerator_and_host_candidate"
    if accelerator["status"] == "admitted":
        return "accelerator_candidate_host_unknown" if host["status"] == "unknown" else "accelerator_only_candidate"
    if host["status"] == "admitted":
        return "host_candidate_accelerator_unknown" if accelerator["status"] == "unknown" else "host_only_candidate"
    return "neither_admitted" if accelerator["status"] == host["status"] == "unsupported" else "unresolved"


def admit_operation_row(
    row: dict,
    *,
    software_spec: dict | None,
    capability_contract: dict | None,
    capability_map: dict | None = None,
    host_capabilities: dict | None = None,
    observed: dict | None = None,
    source_operations: tuple | None = None,
    source_context: Mapping | None = None,
) -> dict:
    """Every lane's admission of ONE observed operation, exactly as a captured application's are judged.

    ``row`` is an inventory signature row (``operation``, ``mlir_operation``, ``disposition``, typed
    operands/results and access observations); ``observed`` overrides the signature projected from it.
    A written capsule's program is screened through this same function, so a capsule and the workload
    operation it stands for cannot be judged by two rules.
    """
    from merlin.targetgen.host_capabilities import admit_host_operation

    if capability_contract is not None and capability_map is None:
        from merlin.targetgen.eligibility import capability_map_from_contract

        capability_map = capability_map_from_contract(capability_contract)
    observed = observed if observed is not None else _observed_signature(row)
    software = _software_admissions(software_spec, row, observed)
    hardware = _hardware_admission(row, observed, capability_contract, capability_map)
    accelerator = _accelerator_admission(row, software, hardware, software_spec is not None)
    host = admit_host_operation(
        host_capabilities, row, observed, source_operations=source_operations, source_context=source_context
    )
    return {
        "observed_admission_signature": observed,
        "software_admissions": software,
        "matching_declarations": [decision["declaration"] for decision in software],
        "hardware_admission": hardware,
        "accelerator_admission": accelerator,
        "host_admission": host,
        "support_partition": _support_partition(accelerator, host),
        "classification": _classification(row, software, hardware, software_spec is not None),
    }


def _summary(entries: list[dict]) -> dict:
    classifications, dispositions, operations, hardware_admissions = Counter(), Counter(), Counter(), Counter()
    host_admissions, accelerator_admissions = Counter(), Counter()
    support_partitions = Counter()
    annotated_frontend, annotated_provenance, unattributed, standalone_unattributed = 0, 0, 0, 0
    frontend_groups = {}
    for entry in entries:
        row, count = entry["observed_signature"], entry["count"]
        classifications[entry["classification"]] += count
        hardware_admissions[entry["hardware_admission"]["status"]] += count
        host_admissions[entry["host_admission"]["status"]] += count
        accelerator_admissions[entry["accelerator_admission"]["status"]] += count
        support_partitions[entry["support_partition"]] += count
        dispositions[row["disposition"]] += count
        operations[row["mlir_operation"]] += count
        frontend, provenance = row.get("frontend_op"), row.get("provenance_op")
        annotated_frontend += count if frontend else 0
        annotated_provenance += count if provenance else 0
        if not frontend and not provenance:
            unattributed += count
            if row["disposition"] not in {"structural", "component"}:
                standalone_unattributed += count
        if frontend or provenance:
            key = _canonical({"frontend_op": frontend, "provenance_op": provenance})
            group = frontend_groups.setdefault(
                key,
                {
                    "frontend_op": frontend,
                    "provenance_op": provenance,
                    "annotated_mlir_operation_count": 0,
                    "classification_counts": Counter(),
                },
            )
            group["annotated_mlir_operation_count"] += count
            group["classification_counts"][entry["classification"]] += count
    groups = []
    for key in sorted(frontend_groups):
        group = frontend_groups[key]
        group["classification_counts"] = dict(sorted(group["classification_counts"].items()))
        groups.append(group)
    return {
        "n_mlir_operations": sum(classifications.values()),
        "n_signatures": len(entries),
        "classification_counts": dict(sorted(classifications.items())),
        "hardware_admission_counts": dict(sorted(hardware_admissions.items())),
        "host_admission_counts": dict(sorted(host_admissions.items())),
        "accelerator_admission_counts": dict(sorted(accelerator_admissions.items())),
        "support_partition_counts": dict(sorted(support_partitions.items())),
        "inventory_disposition_counts": dict(sorted(dispositions.items())),
        "mlir_operation_counts": dict(sorted(operations.items())),
        "standalone_compute_demands": sum(
            dispositions[key] for key in ("hardware_admitted", "host_required", "unclassified")
        ),
        "pytorch_provenance": {
            "original_pytorch_invocation_count": None,
            "status": "unknown",
            "count_basis": "lowered MLIR annotation occurrences; provenance fanout is not original graph invocations",
            "frontend_annotated_mlir_operations": annotated_frontend,
            "provenance_annotated_mlir_operations": annotated_provenance,
            "unattributed_mlir_operations": unattributed,
            "standalone_unattributed_mlir_operations": standalone_unattributed,
            "groups": groups,
        },
    }


def _framework_partition(catalog: dict | None, entries: list[dict], inventory_available: bool) -> dict:
    """Compare exact frontend identities with the selected registry, never guessed aliases."""
    if catalog is not None and not isinstance(catalog, dict):
        raise ValueError("selected framework catalog must be a mapping")
    selected = catalog or {}
    available = selected.get("status") in {"observed", "available"}
    names = selected.get("all_ops") if available else None
    if available:
        if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("observed framework catalog must enumerate registered ATen operator names")
        if len(set(names)) != len(names) or _count(selected.get("n_all_aten"), "framework.n_all_aten") != len(names):
            raise ValueError("framework catalog registered operator names/count disagree")
    groups = {}
    for entry in entries:
        name = entry["observed_signature"].get("frontend_op")
        if name:
            group = groups.setdefault(
                name, {"operator": name, "annotated_mlir_operation_count": 0, "classification_counts": Counter()}
            )
            group["annotated_mlir_operation_count"] += entry["count"]
            group["classification_counts"][entry["classification"]] += entry["count"]
    observed = []
    registered = set(names or [])
    for name, group in sorted(groups.items()):
        group["classification_counts"] = dict(sorted(group["classification_counts"].items()))
        group["registry_membership"] = (
            "unknown" if not available else "registered" if name in registered else "unlisted"
        )
        observed.append(group)
    workload_names = set(groups)
    return {
        "status": "observed" if available else "not_available",
        "schema": selected.get("schema"),
        "torch_version": selected.get("torch"),
        "scope": "registered ATen overloads in the selected PyTorch build, not all Python API functions",
        "registered_aten_operators": sorted(registered) if available else None,
        "n_registered_aten_operators": len(registered) if available else None,
        "core_aten_operators": copy.deepcopy(selected.get("ops")),
        "decomposition_operators": copy.deepcopy(selected.get("decomposed")),
        "components": copy.deepcopy(selected.get("components")),
        "observed_frontend_operators": observed,
        "observed_registered_aten_operators": sorted(workload_names & registered)
        if available and inventory_available
        else None,
        "unobserved_registered_aten_operators": sorted(registered - workload_names)
        if available and inventory_available
        else None,
        "unlisted_frontend_operators": sorted(workload_names - registered)
        if available and inventory_available
        else None,
        "accelerator_candidate_frontend_operators": sorted(
            name for name, group in groups.items() if group["classification_counts"].get("accelerator_candidate")
        )
        if inventory_available
        else None,
        "unresolved_frontend_operators": sorted(
            name for name, group in groups.items() if group["classification_counts"].get("unresolved")
        )
        if inventory_available
        else None,
        "count_basis": "annotation fanout over normalized MLIR, not original PyTorch invocation counts",
        "absence_meaning": "not observed in selected workloads; does not mean unsupported or lowerable",
    }


def build_operation_accounting(
    inventory: dict | None,
    software_spec: dict | None = None,
    *,
    capability_contract: dict | None = None,
    framework_catalog: dict | None = None,
    framework_catalogs: dict[str, dict] | None = None,
    frontend_traces: dict[str, dict] | None = None,
    application_graphs: dict[str, dict] | None = None,
    host_capabilities: dict | None = None,
    capture_execution_attestations: dict[str, dict] | None = None,
    application_sources: dict[str, bytes] | None = None,
) -> dict:
    """Build deterministic, complete workload partitions from selected immutable inputs.

    ``capture_execution_attestations`` are carried per application unchanged; this
    module never judges them (the coverage gate re-verifies each from disk).
    ``application_sources`` supplies the exact captured bytes for declarations
    requiring parsed source bodies. This enables host declaration screening,
    not numerical qualification or compiler lowering certification.
    ``None`` means the detailed inventory is unavailable, not an empty covered
    workload. Only a supplied selected contract is re-screened for hardware
    admission; no target registry or ambient facts are read here.
    """
    if software_spec is not None and not isinstance(software_spec, dict):
        raise ValueError("selected software spec must be a mapping")
    if capability_contract is not None and not isinstance(capability_contract, dict):
        raise ValueError("selected capability contract must be a mapping")
    if inventory is not None:
        if not isinstance(inventory, dict) or not isinstance(inventory.get("applications"), dict):
            raise ValueError("operation accounting requires a detailed application inventory")
        if inventory.get("schema_version") not in {1, 2}:
            raise ValueError("unsupported application inventory schema")
        total = _count(inventory.get("n_operations"), "n_operations")
        applications = inventory["applications"]
        for label, application in applications.items():
            if not isinstance(label, str) or not label:
                raise ValueError("operation inventory application labels must be nonempty strings")
            _validate_application(label, application)
        if sum(app["n_operations"] for app in applications.values()) != total:
            raise ValueError("operation inventory overall count differs from application counts")
        if not applications and inventory.get("status") != "not_declared":
            raise ValueError("empty operation inventory must explicitly declare status not_declared")
    else:
        applications = {}
    if application_sources is not None and set(application_sources) != set(applications):
        raise ValueError("parsed source admission requires the exact selected application roster")
    capability_map = None
    if capability_contract is not None:
        from merlin.targetgen.eligibility import capability_map_from_contract

        capability_map = capability_map_from_contract(capability_contract)
    result_applications, all_entries = {}, []
    from merlin.targetgen.framework_operation_partition import attach_frontend_partitions, combine_frontend_partitions
    from merlin.targetgen.operation_obligations import build_application_completeness

    for label, application in sorted(applications.items()):
        selected_graph = (application_graphs or {}).get(label) or application.get("operation_graph")
        parsed_operations = None
        if application_sources is not None:
            from merlin.common import mlir_query as mq
            from merlin.frontends.capture_normalization import normalize_capture_mlir

            raw = application_sources[label]
            if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != application["capture_sha256"]:
                raise ValueError("parsed source admission differs from selected capture bytes")
            normalized, _ = normalize_capture_mlir(raw.decode("utf-8"))
            if (
                selected_graph is not None
                and hashlib.sha256(normalized.encode()).hexdigest() != selected_graph.get("normalized_mlir_sha256")
            ):
                raise ValueError("parsed source admission differs from selected normalized graph")
            parsed_operations = tuple(mq.walk(mq.parse(normalized)))
            if len(parsed_operations) != application["n_operations"]:
                raise ValueError("parsed source admission operation count differs from selected inventory")
        entries = []
        for row in sorted(application["signatures"], key=lambda item: min(item["ordinals"])):
            entry = {
                "application": label,
                "capture_sha256": application["capture_sha256"],
                "operation": row["operation"],
                "count": row["count"],
                "ordinals": sorted(row["ordinals"]),
                "observed_signature": copy.deepcopy(row),
                **admit_operation_row(
                    row,
                    software_spec=software_spec,
                    capability_contract=capability_contract,
                    capability_map=capability_map,
                    host_capabilities=host_capabilities,
                    source_operations=(
                        tuple(parsed_operations[ordinal] for ordinal in row["ordinals"])
                        if parsed_operations is not None else None
                    ),
                ),
                "lowering_status": "unverified",
            }
            entries.append(entry)
        completeness = build_application_completeness(
            application,
            entries,
            application_graph=selected_graph,
            frontend_trace=(frontend_traces or {}).get(label),
            software_spec=software_spec,
            host_capabilities=host_capabilities,
        )
        summary = _summary(entries)
        source_trace = completeness["source_trace"]
        summary["pytorch_provenance"].update(
            {
                "status": "observed" if source_trace["status"] == "complete" else "unknown",
                "original_pytorch_invocation_count": source_trace["original_invocation_count"],
                "quantized_pytorch_invocation_count": source_trace["quantized_invocation_count"],
                "prepared_pytorch_invocation_count": source_trace["prepared_invocation_count"],
                "source_trace_status": source_trace["status"],
                "original_counting_unit": source_trace["counting_unit"],
            }
        )
        result_applications[label] = {
            "application": label,
            "workload_identity": copy.deepcopy(
                application.get("workload_identity")
                or {
                    "workload_id": label,
                    "workload_role": "unknown",
                    "coverage_scope": "unknown",
                    "note": "captured application identity does not establish full-network coverage or equivalence",
                }
            ),
            "capture": application.get("capture"),
            "capture_sha256": application["capture_sha256"],
            "capture_receipt": copy.deepcopy(application.get("capture_receipt")),
            "capture_execution_attestation": copy.deepcopy((capture_execution_attestations or {}).get(label)),
            "capture_quantization": application.get("capture_quantization"),
            "capture_integerization": copy.deepcopy(application.get("capture_integerization")),
            "operation_graph_identity": _graph_identity(selected_graph),
            **summary,
            "signatures": entries,
            "completeness": completeness,
            "framework_universe": attach_frontend_partitions(
                _framework_partition((framework_catalogs or {}).get(label, framework_catalog), entries, True),
                source_trace,
            ),
        }
        selected_catalog = (framework_catalogs or {}).get(label, framework_catalog) or {}
        source_versions = {}
        for stage in ("original", "quantized", "prepared"):
            graph = source_trace["graphs"].get(stage) or {}
            versions = graph.get("runtime_versions") or {}
            source_versions[stage] = (
                versions.get("torch") if graph.get("status") == "verified" and isinstance(versions, dict) else None
            )
        result_applications[label]["framework_universe"]["source_graph_version_match"] = {
            "status": "unknown"
            if selected_catalog.get("status") not in {"available", "observed"}
            or not selected_catalog.get("torch")
            or not all(source_versions.values())
            else "matched"
            if all(version == selected_catalog["torch"] for version in source_versions.values())
            else "mismatch",
            "catalog_status": selected_catalog.get("status", "not_available"),
            "catalog_version": selected_catalog.get("torch"),
            "catalog_document_sha256": hashlib.sha256(_canonical(selected_catalog)).hexdigest()
            if selected_catalog
            else None,
            "source_graph_versions": source_versions,
        }
        all_entries.extend(entries)
    declared = []
    for declaration in sorted((software_spec or {}).get("operations") or [], key=lambda item: item["id"]):
        observations = [
            entry
            for entry in all_entries
            if declaration["id"] in entry["matching_declarations"]
            and entry["observed_signature"]["disposition"] not in {"structural", "component"}
        ]
        declared.append(
            {
                **copy.deepcopy(declaration),
                "observed_mlir_operation_count": sum(entry["count"] for entry in observations),
                "observed_signature_count": len(observations),
                "observed_applications": sorted({entry["application"] for entry in observations}),
                "observation_status": (
                    "not_available"
                    if inventory is None
                    else "no_workloads_declared"
                    if not applications
                    else "observed"
                    if observations
                    else "absent_from_selected_workloads"
                ),
                "lowering_status": "unverified",
            }
        )
    overall = _summary(all_entries)
    for stage in ("original", "quantized", "prepared"):
        counts = [
            application["completeness"]["source_trace"][f"{stage}_invocation_count"]
            for application in result_applications.values()
        ]
        overall["pytorch_provenance"][f"{stage}_pytorch_invocation_count"] = (
            sum(counts) if counts and all(type(count) is int for count in counts) else None
        )
    overall["pytorch_provenance"]["original_counting_unit"] = "static_captured_call_sites"
    overall["pytorch_provenance"]["source_trace_status"] = (
        "complete"
        if result_applications
        and all(app["completeness"]["source_trace"]["status"] == "complete" for app in result_applications.values())
        else "unknown"
    )
    overall["pytorch_provenance"]["status"] = (
        "observed" if overall["pytorch_provenance"]["source_trace_status"] == "complete" else "unknown"
    )
    return {
        "schema": SCHEMA,
        "status": "not_available" if inventory is None else "not_declared" if not applications else "accounted",
        "qualification": "diagnostic_declaration_screen; no compiler lowering certification",
        "digest_basis": "SHA256 of canonical selected JSON documents; not original authored file bytes",
        "inventory_sha256": hashlib.sha256(_canonical(inventory)).hexdigest() if inventory is not None else None,
        "selected_software_spec_sha256": hashlib.sha256(_canonical(software_spec)).hexdigest()
        if software_spec is not None
        else None,
        "selected_capability_contract_sha256": hashlib.sha256(_canonical(capability_contract)).hexdigest()
        if capability_contract is not None
        else None,
        "scope": "only explicitly captured applications; not all PyTorch operators or unobserved networks",
        "applications": result_applications,
        "overall": (
            overall
            if inventory is not None
            else {
                **_summary([]),
                "n_mlir_operations": None,
                "n_signatures": None,
                "standalone_compute_demands": None,
            }
        ),
        "framework_universe": combine_frontend_partitions(
            _framework_partition(framework_catalog, all_entries, inventory is not None), result_applications
        ),
        "declared_support_universe": {
            "status": "not_available" if software_spec is None else software_spec.get("status", "unknown"),
            "basis": "authored software declarations, not verified support or the global PyTorch registry",
            "operations": declared,
        },
    }
