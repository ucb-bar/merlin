"""Source-derived admission before component stimulus or reference allocation.

Counts bound logical reference work and tensor payload, not process heap bytes,
compiler time or hardware cycles. Unknown numerical/frontend paths cannot use a
declared cost as a substitute. Policies and decisions remain private Phase 0
evidence; the ordinary generator still owns programs, stimuli and full goldens.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

from merlin.targetgen import component_program, corpus_spec

from .component_generation import digest

SCHEMA = "merlin.component_execution_budget.v1"
RECEIPT_SCHEMA = "merlin.component_execution_admission.v1"
_METRICS = ("reference_work", "materialized_elements", "tensor_payload_bytes")
_LIMITS = {"max_scalar_bits", *("max_" + key for key in _METRICS), *("max_total_" + key for key in _METRICS)}


def validate(policy):
    """Require explicit finite per-member and whole-generation limits."""
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema", *_LIMITS}
        or policy["schema"] != SCHEMA
        or any(type(policy[key]) is not int or policy[key] < 1 for key in _LIMITS)
    ):
        raise ValueError("execution budget requires the closed v1 schema and positive integer limits")
    return policy


def _bits(dtype):
    if not isinstance(dtype, str) or not dtype.startswith("i") or not dtype[1:].isdigit() or int(dtype[1:]) < 1:
        raise ValueError("execution budget has no independent cost for this scalar format")
    return int(dtype[1:])


def _extent(shape):
    if not isinstance(shape, (list, tuple)) or not shape or any(type(dim) is not int or dim < 1 for dim in shape):
        raise ValueError("execution budget requires checked positive static tensor extents")
    return math.prod(shape)


def measure(source):
    """Count every leaf, temporary node result and complete published output.

    A reference work unit is one element visit or scalar multiply/add in the
    mathematical tensor evaluator. Matmul assumes every operand is nonzero;
    zero-skipping never discounts admission. Each other node charges its source
    visit/computation and modular projection. Alias copies are charged because
    the independent evaluator materializes them. Payload sums allocations,
    including both pre-projection and projected results; it is not peak heap.
    """
    kind, program = source["kind"], source["program"]
    inputs, nodes, outputs = program["inputs"], program["nodes"], program["outputs"]
    types = {row["name"]: row for row in inputs + nodes}
    elements = sum(_extent(row["shape"]) for row in inputs)
    payload = sum(_extent(row["shape"]) * ((_bits(row["dtype"]) + 7) // 8) for row in inputs)
    scalar_bits = max(_bits(row["dtype"]) for row in inputs + nodes + outputs)
    work = elements * (2 if kind == "component_program" else 1)
    for node in nodes:
        count, bits = _extent(node["shape"]), _bits(node["dtype"])
        raw_bits = bits
        if node["op"] == "matmul":
            lhs, rhs = (types[name] for name in node["actual_inputs"])
            reduction = lhs["shape"][1]
            work += 2 * count * reduction + 2 * count
            raw_bits = max(bits, _bits(lhs["dtype"]) + _bits(rhs["dtype"]) + reduction.bit_length() + 1)
        else:
            work += 3 * count
            if node["op"] in {"add", "update"}:
                raw_bits += 1
        scalar_bits = max(scalar_bits, raw_bits)
        elements += 2 * count
        payload += count * (((raw_bits + 7) // 8) + ((bits + 7) // 8))
    for row in outputs:
        count = _extent(row["shape"])
        work += count
        elements += count
        payload += count * ((_bits(row["dtype"]) + 7) // 8)
    palette = source["input_palette"]
    if palette is not None:
        from merlin.targetgen.input_palette import validate as validate_palette

        validate_palette(palette)
        # corpus_spec.build realizes palette leaves once before the independent
        # evaluator realizes them again. Count this extra allocation/pass even
        # if a declaration selects only a subset of leaves.
        extra = sum(_extent(row["shape"]) for row in inputs)
        values = sum(len(row["values"]) for row in palette["inputs"])
        elements += extra + values
        payload += sum(_extent(row["shape"]) * ((_bits(row["dtype"]) + 7) // 8) for row in inputs)
        payload += values * ((scalar_bits + 7) // 8)
        work += extra + values
    return dict(zip(_METRICS, (work, elements, payload), strict=True), scalar_bits=scalar_bits)


def source_for_entry(entry, *, binding):
    """Inspect selected source semantics without invoking any builder or oracle."""
    if entry.get("_component_source_unavailable"):
        raise ValueError(entry["_component_source_unavailable"])
    if entry.get("source") not in {None, "mlir", "direct"} or entry.get("pytorch_ref") or entry.get("spec_ref"):
        raise ValueError("execution budget has no derived cost for the selected frontend/reference path")
    if binding.tile_dim is None and entry.get("op") != "component_program":
        raise ValueError("source-only execution budget supports explicit tensor DAGs without hardware geometry")
    regime, selected = corpus_spec.entry_binding(entry, binding)
    if regime != "int":
        raise ValueError("execution budget has no independent cost for this numerical engine")
    operand, accumulator = (
        selected.mlir_dtype(selected.operand_dtype),
        selected.mlir_dtype(selected.accum_dtype),
    )
    if entry.get("op") == "component_program":
        program = component_program.analyze(entry.get("program"), operand_dtype=operand, accumulator_dtype=accumulator)
    elif entry.get("op") == "movement":
        shape = [entry.get(axis, entry.get(axis + "_tiles", 1) * selected.tile_dim) for axis in ("M", "N")]
        _extent(shape)
        program = {
            "inputs": [{"name": entry.get("src", "X"), "role": "input", "shape": shape, "dtype": operand}],
            "nodes": [],
            "outputs": [{"name": entry.get("out", "Y0"), "shape": shape, "dtype": accumulator}],
        }
    else:
        raise ValueError("execution budget has no source-derived cost for this operation")
    return {
        "kind": entry["op"],
        "program": program,
        "input_palette": copy.deepcopy(entry.get("input_palette")),
        "stimulus_range": copy.deepcopy(entry.get("stimulus_range")),
    }


def source_for_capsule(capsule):
    """Re-derive from the actual generated source declaration, never saved costs."""
    operation = capsule["operation"]
    if operation["op"] == "component_program":
        typed = capsule["component_program"]
        storage = typed["selected_storage"]
        program = component_program.analyze(
            operation["attributes"]["program"],
            operand_dtype=storage["operand"],
            accumulator_dtype=storage["accumulator"],
        )
        if program != typed or program["inputs"] != capsule["inputs"]:
            raise ValueError("budgeted component source differs from its checked typed declaration")
    elif operation["op"] == "movement":
        inputs = capsule["inputs"]
        if len(inputs) != 1 or inputs[0]["role"] != "input":
            raise ValueError("budgeted movement needs exactly one actual input")
        attrs = operation["attributes"]
        if attrs["src"] != inputs[0]["name"]:
            raise ValueError("budgeted movement source differs from its actual input")
        program = {
            "inputs": inputs,
            "nodes": [],
            "outputs": [{"name": attrs["out"], "shape": inputs[0]["shape"], "dtype": attrs["output_dtype"]}],
        }
    else:
        raise ValueError("budget receipt has no independent cost for the written operation")
    return {
        "kind": operation["op"],
        "program": program,
        "input_palette": capsule.get("input_palette"),
        "stimulus_range": capsule.get("stimulus_range"),
    }


def _exceeded(policy, cost, totals):
    limits = [key for key in _METRICS if cost[key] > policy["max_" + key]]
    if cost["scalar_bits"] > policy["max_scalar_bits"]:
        limits.append("scalar_bits")
    limits += ["total_" + key for key in _METRICS if totals[key] + cost[key] > policy["max_total_" + key]]
    return limits


def admit(entries, *, binding, policy):
    """Preflight the complete generation roster before frontend/data allocation."""
    validate(policy)
    decisions, totals, seen = [], dict.fromkeys(_METRICS, 0), set()
    for entry in entries:
        name = entry["name"]
        if name in seen:
            raise ValueError("execution budget refuses duplicate generated member names")
        seen.add(name)
        row = {"name": name, "requested_member": entry["cat"] + "/" + name, "state": "unavailable"}
        try:
            source = source_for_entry(entry, binding=binding)
            cost = measure(source)
            row.update(source_sha256=digest(source), cost=cost)
            exceeded = _exceeded(policy, cost, totals)
            if exceeded:
                row["reason"] = "execution budget exceeded: " + ", ".join(exceeded)
            else:
                row.update(state="admitted", reason="source-derived bounded reference work and payload")
                for key in _METRICS:
                    totals[key] += cost[key]
        except ValueError as exc:
            row["reason"] = str(exc)
        decisions.append(row)
    return {
        "schema": RECEIPT_SCHEMA,
        "policy": copy.deepcopy(policy),
        "policy_sha256": digest(policy),
        "scope": "complete generation roster; logical reference counts, not process heap or hardware timing",
        "decisions": decisions,
        "totals": totals,
    }


def verify(root, report):
    """Reopen every admitted actual member and recalculate costs and totals."""
    receipt = report.get("execution_admission")
    policy = validate(report["declaration"]["execution_budget"])
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != RECEIPT_SCHEMA
        or digest(receipt.get("policy")) != digest(policy)
        or receipt.get("policy_sha256") != digest(policy)
        or report["generation_identity"].get("execution_budget_sha256") != digest(policy)
        or report["generation_identity"].get("execution_admission_sha256") != digest(receipt)
    ):
        raise ValueError("component execution budget binding changed")
    totals, seen = dict.fromkeys(_METRICS, 0), set()
    for row in receipt["decisions"]:
        name = row["name"]
        if name in seen:
            raise ValueError("component execution budget member duplicated")
        seen.add(name)
        if row["state"] != "admitted":
            raise ValueError("component execution budget contains an unavailable requested member")
        path = Path(row["requested_member"])
        if path.is_absolute() or len(path.parts) != 2 or any(part in {".", ".."} for part in path.parts):
            raise ValueError("component execution budget member path is invalid")
        directory = Path(root) / path
        capsule_file = directory / "capsule.yaml"
        if (
            directory.is_symlink()
            or directory.resolve().parent.parent != Path(root).resolve()
            or capsule_file.is_symlink()
        ):
            raise ValueError("component execution budget member escaped its actual generation root")
        capsule = yaml.safe_load(capsule_file.read_bytes())
        source = source_for_capsule(capsule)
        cost = measure(source)
        if (
            digest(source) != row["source_sha256"]
            or digest(cost) != digest(row["cost"])
            or _exceeded(policy, cost, totals)
        ):
            raise ValueError("component execution budget derived source/counts changed or exceeded limits")
        stamp = (capsule.get("component_coverage") or {}).get("generation_sha256")
        if stamp is None:
            stamp = (capsule.get("performance") or {}).get("component_generation_sha256")
        if stamp != digest(report["generation_identity"]):
            raise ValueError("component execution budget written generation roster binding changed")
        for key in _METRICS:
            totals[key] += cost[key]
    covered = {member["name"] for row in report["obligations"] for member in row["members"]}
    if not covered <= seen or digest(totals) != digest(receipt["totals"]):
        raise ValueError("component execution budget lost requested membership or total work")
