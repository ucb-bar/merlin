"""Ordinary public performance captures as complete connected source members.

The selected roster, sealed capture and normal software screen own admission.
This projection never substitutes an isolated contraction for a program's host
work, chooses new network geometry or grants target execution and timing.
"""

from __future__ import annotations

import copy
import hashlib
import json
from math import prod
from pathlib import Path

from merlin.targetgen import materialized_integer_capture as N
from merlin.targetgen.frontend_use_def import original_use_def_semantics

POLICY_SCHEMA = "merlin.phase0.public_program_selection.v1"
INTEGER_POLICY_SCHEMA = "merlin.phase0.public_program_selection.v2"
AFFINE_INTEGER_POLICY_SCHEMA = "merlin.phase0.public_program_selection.v3"
CAPTURE_SCHEMA = "merlin.materialized_public_capture.v2"
INTEGER_CAPTURE_SCHEMA = N.SCHEMA
BASIS_SCHEMA = "merlin.phase0.public_program_source_basis.v1"


def validate_selection(value):
    affine = isinstance(value, dict) and value.get("schema") == AFFINE_INTEGER_POLICY_SCHEMA
    integer = isinstance(value, dict) and value.get("schema") in {INTEGER_POLICY_SCHEMA, AFFINE_INTEGER_POLICY_SCHEMA}
    if (
        not isinstance(value, dict)
        or set(value)
        != {"schema", "selection", "max_members", "max_capture_bytes", "max_input_elements", "max_output_elements"}
        | ({"source_mode"} if integer else set())
        | ({"source_integer_proof"} if affine else set())
        or value["schema"] not in {POLICY_SCHEMA, INTEGER_POLICY_SCHEMA, AFFINE_INTEGER_POLICY_SCHEMA}
        or (integer and value.get("source_mode") != N.MODE)
        or (affine and value.get("source_integer_proof") != "checked_native_integer_affine_copy_v2")
        or value["selection"] != "declared_performance_applications"
        or any(
            type(value[key]) is not int or value[key] < 1
            for key in value
            if key not in {"schema", "selection", "source_mode", "source_integer_proof"}
        )
    ):
        raise ValueError("public program selection needs the declared performance roster and explicit finite budgets")
    return copy.deepcopy(value)


def capture_schema(selection):
    validate_selection(selection)
    return (
        INTEGER_CAPTURE_SCHEMA
        if selection["schema"] in {INTEGER_POLICY_SCHEMA, AFFINE_INTEGER_POLICY_SCHEMA}
        else CAPTURE_SCHEMA
    )


def observe_inventory(requirement, conformance_spec, observe):
    """Bind the adjacent public inventory to the ordinary frozen input owner."""
    selected = requirement["public_program_sources"]
    validate_selection(selected["selection"])
    if selected.get("schema") != POLICY_SCHEMA or selected.get("sidecar") != "performance-application-demands.json":
        raise ValueError("public program inventory must be one selected adjacent sidecar")
    location = Path(conformance_spec).parent / selected["sidecar"]
    if location.is_symlink():
        raise ValueError("public program inventory may not be a symlink")
    raw = observe(location, "public-program-inventory", required=True)
    content = json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(content).hexdigest() != selected.get("full_inventory_sha256"):
        raise ValueError("public program inventory differs from the requirement's commitment")


def source_semantics(trace, *, source_mode=None):
    """Actual original operations and closed uses; never parameter initialization."""
    semantics = original_use_def_semantics(trace)
    operations = list(semantics.operations)
    groups = {
        "movement": {
            "aten.clone.default",
            "aten._to_copy.default",
            "aten.cat.default",
            "aten.transpose.int",
            "aten.permute.default",
        },
        "conversion": {"aten.to.dtype", "aten._to_copy.default"},
        "reduction": {"aten.sum.dim_IntList", "aten.mean.dim", "aten.softmax.int", "aten._softmax.default"},
        "contraction": {
            "aten.matmul.default",
            "aten.mm.default",
            "aten.bmm.default",
            "aten.linear.default",
            "aten.conv2d.default",
            "aten.convolution.default",
        },
        "map": {"aten.add.Tensor", "aten.mul.Tensor", "aten.relu.default", "aten.silu.default", "aten.gelu.default"},
    }
    if source_mode is not None:
        if source_mode != N.MODE:
            raise ValueError("unsupported public program source mode")
        groups["contraction"].add("aten._int_mm.default")
    observed = {family: sorted(set(operations) & selected) for family, selected in groups.items()}
    values = {result["id"]: result for node in trace["graphs"]["original"]["nodes"] for result in node["results"]}
    observed["conversion"] = sorted(
        {
            node["target"]
            for node in trace["graphs"]["original"]["nodes"]
            if node["target"] in groups["conversion"]
            and node["args"]
            and isinstance(node["args"][0], dict)
            and "value_id" in node["args"][0]
            and any(
                result.get("dtype") != values[node["args"][0]["value_id"]].get("dtype") for result in node["results"]
            )
        }
    )
    return {
        **semantics.public_semantics(),
        "observed_families": observed,
        "purpose": (
            "complete connected source call, including every observed movement, conversion, "
            "reduction and compute operation"
        ),
        "missing_families": [family for family, selected in observed.items() if not selected],
        "missing_family_status": "UNKNOWN; no source case is invented for an unobserved family",
    }


def original_contractions(trace, *, source_mode=None):
    """Ordered actual tensor signatures, including contractions inside a DAG."""
    original_use_def_semantics(trace)
    graph = trace["graphs"]["original"]
    values = {result["id"]: result for node in graph["nodes"] for result in node["results"]}

    def typed_arguments(value):
        if isinstance(value, dict):
            if "value_id" in value:
                result = values[value["value_id"]]
                return {key: result[key] for key in ("kind", "shape", "dtype")}
            return {key: typed_arguments(child) for key, child in value.items()}
        if isinstance(value, list):
            return [typed_arguments(child) for child in value]
        return value

    rows = []
    contractions = {
        "aten.linear.default",
        "aten.matmul.default",
        "aten.mm.default",
        "aten.bmm.default",
        "aten.conv2d.default",
        "aten.convolution.default",
    }
    if source_mode is not None:
        if source_mode != N.MODE:
            raise ValueError("unsupported public program source mode")
        contractions.add("aten._int_mm.default")
    for node in graph["nodes"]:
        target = node["target"]
        if node["op"] not in {"call_function", "call_method", "call_module"} or target not in contractions:
            continue
        args = node["args"]
        operands = [values[arg["value_id"]] for arg in args[:2]]
        results = node["results"]
        tensors = operands + results
        if len(results) != 1 or any(
            row.get("kind") != "tensor"
            or not isinstance(row.get("shape"), list)
            or any(type(v) is not int or v < 1 for v in row["shape"])
            for row in tensors
        ):
            raise ValueError("public contraction requires actual complete positive tensor types")
        lhs, rhs = (row["shape"] for row in operands)
        if target == "aten._int_mm.default" and (
            [row.get("dtype") for row in tensors] != ["int8", "int8", "int32"]
            or len(lhs) != 2
            or len(rhs) != 2
            or results[0]["shape"] != [lhs[0], rhs[1]]
            or len(args) != 2
            or node["kwargs"]
        ):
            raise ValueError("public _int_mm needs its original rank-two signed i8/i8/i32 storage")
        if target == "aten.linear.default":
            if len(lhs) < 2 or len(rhs) != 2 or lhs[-1] != rhs[-1]:
                raise ValueError("public linear tensor shapes disagree")
            gemm = [prod(lhs[:-1]), lhs[-1], rhs[0]]
            conv = None
        elif target in {"aten.conv2d.default", "aten.convolution.default"}:
            if len(lhs) != 4 or len(rhs) != 4:
                raise ValueError("public convolution requires actual rank-four tensors")
            stride = args[3] if len(args) > 3 else node["kwargs"].get("stride", [1, 1])
            if isinstance(stride, int):
                stride = [stride, stride]
            if not isinstance(stride, list) or len(stride) != 2 or any(type(v) is not int or v < 1 for v in stride):
                raise ValueError("public convolution stride is unavailable")
            conv = [rhs[2], rhs[3], *stride, lhs[1], rhs[0]]
            gemm = [prod(results[0]["shape"][-2:]), prod(rhs[1:]), rhs[0]]
        else:
            if len(lhs) < 2 or len(rhs) < 2 or lhs[-1] != rhs[-2]:
                raise ValueError("public matmul tensor shapes disagree")
            gemm, conv = [lhs[-2], lhs[-1], rhs[-1]], None
        rows.append(
            {
                "operation": target,
                "ordered_inputs": [{key: row[key] for key in ("shape", "dtype")} for row in operands],
                "ordered_results": [{key: row[key] for key in ("shape", "dtype")} for row in results],
                "gemm": gemm,
                "conv": conv,
                "operator_schema": graph["operator_schemas"][target],
                "arguments": typed_arguments(args),
                "keyword_arguments": typed_arguments(node["kwargs"]),
            }
        )
    return rows


def check_budget(directory: Path, selection):
    selection = validate_selection(selection)
    receipt = json.loads((directory / "capture_receipt.json").read_bytes())
    artifacts = receipt["artifacts"]
    if sum(row["bytes"] for row in artifacts.values()) > selection["max_capture_bytes"]:
        raise ValueError("complete public capture exceeds the selected byte budget")
    meta = json.loads((directory / "meta.json").read_bytes())
    for field, budget in (("input_abi", "max_input_elements"), ("output_abi", "max_output_elements")):
        count = 0
        for tensor in meta[field]:
            shape = tensor["shape"]
            if not isinstance(shape, list) or any(type(v) is not int or v < 1 for v in shape):
                raise ValueError("public program requires complete positive static tensor shapes")
            count += prod(shape)
        if count > selection[budget]:
            raise ValueError("complete public program ABI exceeds the selected element budget")


def performance(entry, *, trace, forms, facts_sha256):
    """A whole-call objective with conditional existing hardware boundaries."""
    if (entry.get("public_source_role") or {}).get("selected_member", {}).get("role") == "functional_guard":
        raise ValueError("explicit source functional guard cannot become a performance objective")
    capture = entry["materialized_capture"]
    semantics = source_semantics(trace, source_mode=capture.get("source_mode"))
    label = capture["workload_id"]
    boundaries = [
        {"class": row["class_id"], "entry": member["entry"], "price": member["price"]}
        for row in forms.get("classes") or []
        for member in row.get("members") or []
        if member.get("application") == label
    ]
    basis = {
        "schema": BASIS_SCHEMA,
        "kind": "public_program",
        "capture_sha256": capture["capture_sha256"],
        "receipt_sha256": capture["receipt_sha256"],
        "operations": semantics["operation_semantics"],
        "numerical_reference": "host_torch_eager",
        "boundary": "complete_program_call",
    }
    return {
        "level": "L6_global",
        "family": "public_program",
        "lever": "complete connected public source call",
        "member_class": "OBJECTIVE",
        "claim": "DIFFERENTIAL",
        "comparand": {
            "kind": "candidate",
            "against": "same complete original source",
            "cancels": "identical inputs and weights",
            "demand_equal": "complete program and ordered outputs",
        },
        "falsifier": {
            "observation": "complete call cost and all numerical outputs",
            "fires_when": "any required output fails or complete cost increases",
            "negative_control": "unchanged compiler with identical source and input bytes",
        },
        "gate": {
            "traits": [],
            "source_execution_basis": basis,
            "instrument": "complete selected execution boundary",
            "capacity": "UNKNOWN physical source/resource mapping",
            "on_missing": "skip_with_evidence",
        },
        "regime": {
            "separation": "unchanged independent public program geometry",
            "layout": "actual captured typed source",
        },
        "emitter": {"status": "existing", "entry": "materialized_model_artifacts + write_model_capsule", "knobs": {}},
        "cost": {
            "tier": "source/reference",
            "runs": 1,
            "projected_cycles": "unbounded",
            "basis": "unmeasured complete call",
        },
        "curriculum": {
            "schema": "merlin.phase0.public_program_curriculum.v1",
            **semantics,
            "derivation": {
                key: capture[key]
                for key in ("capture_sha256", "receipt_sha256", "full_inventory_sha256", "loader_sha256")
            },
            "numerical_reference": "all receipt-bound host eager outputs; existing output accuracy gates retained",
            "evaluation_tiers": {
                "source_reference": "capture host eager",
                "compilation": "UNKNOWN",
                "numerical_execution": "UNKNOWN",
                "timing": "UNKNOWN",
            },
            "resource_boundary": {
                "facts_sha256": facts_sha256,
                "observed_forms": boundaries,
                "hardware_mapping": "UNKNOWN",
            },
            "limitations": [
                "not an isolated contraction certificate",
                "original call cost includes complete host work and output publication",
                "physical lifetimes, reuse, overlap and state protocols UNKNOWN",
                "passing seeded random-weight outputs does not establish model accuracy",
            ],
        },
    }


def verify_written(capsule, directory):
    """Reopen source, trace, receipt, weights and complete reference bindings."""
    from merlin.targetgen.capsule_source import _portable_weights_reference

    directory = Path(directory)
    capture = capsule.get("materialized_capture") or {}
    if (
        capture.get("schema") not in {CAPTURE_SCHEMA, INTEGER_CAPTURE_SCHEMA}
        or capture.get("workload_role") != "performance_scale"
        or capture.get("coverage_scope") != "full_capture"
    ):
        raise ValueError("public program lacks explicit performance capture selection")
    native_integer = N.selected(capture)
    frontend = capsule["frontend_trace"]
    receipt_raw = (directory / "source-capture-receipt.json").read_bytes()
    if hashlib.sha256(receipt_raw).hexdigest() != capture["receipt_sha256"]:
        raise ValueError("public program capture receipt changed")
    receipt = json.loads(receipt_raw)
    original = (directory / "frontend-source.mlir").read_bytes()
    interface = (directory / "capsule.interface.mlir").read_bytes()
    if original != interface or hashlib.sha256(original).hexdigest() != frontend["source_mlir_sha256"]:
        raise ValueError("public program differs from its portable original source")
    raw_trace = (directory / "frontend-trace.json").read_bytes()
    if (
        hashlib.sha256(raw_trace).hexdigest() != receipt["artifacts"]["frontend-trace.json"]["sha256"]
        or hashlib.sha256(raw_trace).hexdigest() != frontend["sha256"]
    ):
        raise ValueError("public program original graph changed")
    relocation = frontend["source_portability"]
    # Reconstruct the exact raw source using its receipt-bound recorded weights path.
    meta = None
    references = N.reference_members(receipt["artifacts"]) if native_integer else ("golden.npy",)
    for member in ("meta.json", "inputs.npz", "input_order.json", *references):
        raw = (directory / ("source-capture-" + member)).read_bytes()
        if (
            len(raw) != receipt["artifacts"][member]["bytes"]
            or hashlib.sha256(raw).hexdigest() != receipt["artifacts"][member]["sha256"]
        ):
            raise ValueError("public program reference closure changed")
        if member == "meta.json":
            meta = json.loads(raw)
    original_weights = meta["weights"]
    portable = 'prov.weights_file = "capsule.weights.safetensors"'
    if original.decode().count(portable) != 1:
        raise ValueError("public program requires one exact original weights reference")
    raw_source = original.decode().replace(portable, 'prov.weights_file = "' + original_weights + '"')
    emitted, edit = _portable_weights_reference(raw_source, original_weights, sidecar=True)
    raw_hash = hashlib.sha256(raw_source.encode()).hexdigest()
    if emitted.encode() != interface or any(relocation.get(k) != v for k, v in edit.items()):
        raise ValueError("public program portability binding changed")
    if raw_hash != capture["capture_sha256"] or receipt["artifacts"]["model.mlir"]["sha256"] != raw_hash:
        raise ValueError("public program raw source identity differs")
    for source, emitted in (
        ("weights.safetensors", "capsule.weights.safetensors"),
        ("weights.safetensors.manifest.json", "capsule.weights.safetensors.manifest.json"),
    ):
        raw = (directory / emitted).read_bytes()
        if (
            len(raw) != receipt["artifacts"][source]["bytes"]
            or hashlib.sha256(raw).hexdigest() != receipt["artifacts"][source]["sha256"]
        ):
            raise ValueError("public program weight closure changed")
    loader = (directory / "capsule.pytorch.py").read_bytes()
    if hashlib.sha256(loader).hexdigest() != capture["loader_sha256"]:
        raise ValueError("public program original loader changed")
    profile_sha = (meta.get("loader_provenance") or {}).get("profile_sha256")
    if profile_sha is not None or capture.get("profile_sha256") is not None:
        profile = directory / "profile.json"
        if (
            profile.is_symlink()
            or not profile.is_file()
            or hashlib.sha256(profile.read_bytes()).hexdigest() != profile_sha
            or profile_sha != capture.get("profile_sha256")
        ):
            raise ValueError("public program original profile changed")
    semantics = source_semantics(json.loads(raw_trace), source_mode=capture.get("source_mode"))
    if capsule.get("public_source_role"):
        from .public_source_roles import verify_written as verify_source_role

        role = verify_source_role(capsule, directory)
        selected = capsule["public_source_role"]["source_execution_basis"]
        if role == "objective" and selected != capsule["performance"]["gate"]["source_execution_basis"]:
            raise ValueError("objective source role differs from its actual complete-call basis")
    else:
        selected = capsule["performance"]["gate"]["source_execution_basis"]
    if (
        selected["operations"] != semantics["operation_semantics"]
        or selected["capture_sha256"] != raw_hash
        or selected["receipt_sha256"] != capture["receipt_sha256"]
    ):
        raise ValueError("public program rationale differs from its actual source")
    import numpy as np

    from merlin.targetgen.golden_store import load_golden

    golden = load_golden(directory)
    attributes = capsule["operation"]["attributes"]
    names = attributes.get("outs") or [attributes["out"]]
    if set(golden["outputs"]) != set(names) or len(set(names)) != len(names):
        raise ValueError("public program reference omits a published output")
    if native_integer:
        from merlin.targetgen.capsule_privacy import capture_reference_closure

        if capsule.get("reference_closure") != capture_reference_closure(receipt["artifacts"], source_mode=N.MODE):
            raise ValueError("integer public reference closure omits expected-output leaves")
        values = N.outputs(meta, (directory / "source-capture-golden.json").read_bytes())
        values = values if len(meta["output_abi"]) > 1 else [values]
        if len(names) != len(values):
            raise ValueError("integer public result names differ from its complete typed ABI")
        policies = capsule["numeric_policy"]
        for name, value, abi in zip(names, values, meta["output_abi"], strict=True):
            expected_policy = dict(compare="exact_int", dtype=abi["dtype"])
            policy = policies.get("outputs", {}).get(name) if policies.get("compare") == "per_output" else policies
            if policy != expected_policy:
                raise ValueError("integer public result changed its exact per-output policy")
            actual, expected = np.asarray(golden["outputs"][name]), np.asarray(value)
            if actual.dtype.kind not in "iu" or actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise ValueError("integer public complete reference differs from its actual capture")
        N.inputs(meta, (directory / "source-capture-inputs.npz").read_bytes())
    else:
        if len(names) != 1:
            raise ValueError("public program reference omits a published output")
        actual = np.asarray(golden["outputs"][names[0]], dtype=np.float32)
        expected = np.load(directory / "source-capture-golden.npy", allow_pickle=False)
        if actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
            raise ValueError("public program complete reference differs from its actual capture")
    with np.load(directory / "source-capture-inputs.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"in" + str(i) for i in range(len(capsule["inputs"]))}:
            raise ValueError("public program input closure differs")
        for i, tensor in enumerate(capsule["inputs"]):
            expected_input = archive["in" + str(i)]
            observed_input = golden["oracle_provenance"]["inputs"][tensor["name"]]
            actual_input = np.asarray(observed_input["decoded"], dtype=expected_input.dtype).reshape(
                observed_input["shape"]
            )
            if tensor["shape"] != list(expected_input.shape) or actual_input.tobytes() != expected_input.tobytes():
                raise ValueError("public program input values changed")
    if "public_source_frontier" in capsule:
        from .public_source_frontier import verify_written as verify_frontier

        verify_frontier(capsule, directory)
    return {
        "selection": selected,
        "source_sha256": hashlib.sha256(interface).hexdigest(),
        "typed_program_sha256": semantics["graph_sha256"],
        "reference_sha256": hashlib.sha256((directory / "golden.yaml").read_bytes()).hexdigest(),
        "status": "source_prepared_execution_unknown",
        "scope": "complete original public program/reference; compilation, execution and timing unknown",
    }
