"""Check source-bound buffer observations and explicit snapshot consent.

Producer observations are not numerical or reference-origin certificates. Only
the original caller may authorize freezing, and existing pure tensor/source and
independent preparation admission remain required.
"""

from __future__ import annotations

import hashlib
import json
import operator
from math import prod

from merlin.common.digest import is_sha256

SCHEMA = "merlin.capture.export-state.v1"
_FIELDS = {
    "schema",
    "status",
    "source_sha256",
    "prepared_graph_sha256",
    "manifest_sha256",
    "inputs",
    "outputs",
    "operator_effects",
    "placeholder_names",
    "observer_source_sha256",
    "snapshot_source_sha256",
    "scope",
}
_INPUT = {
    "ordinal",
    "kind",
    "target",
    "argument",
    "storage_key",
    "storage_group",
    "shape",
    "dtype",
    "strides",
    "storage_offset",
    "storage_bytes",
    "payload_bytes",
    "payload_sha256",
    "version",
}


def validate_observation(record, *, trace, source_sha256, manifest_sha256=None):
    """Validate one explicit schema against copied source correspondence only."""
    if (
        not isinstance(record, dict)
        or set(record) != _FIELDS
        or record.get("schema") != SCHEMA
        or record.get("status") != "observed"
    ):
        raise ValueError("buffer observation has an unsupported or incomplete schema")
    for name in (
        "source_sha256",
        "prepared_graph_sha256",
        "manifest_sha256",
        "observer_source_sha256",
        "snapshot_source_sha256",
    ):
        if not is_sha256(record[name]):
            raise ValueError("buffer observation requires exact source/producer identities")
    if (
        record["source_sha256"] != source_sha256
        or manifest_sha256 is not None
        and record["manifest_sha256"] != manifest_sha256
    ):
        raise ValueError("buffer observation differs from captured source/argument manifest")
    prepared = (trace.get("graphs") or {}).get("prepared") if isinstance(trace, dict) else None
    if (
        not isinstance(prepared, dict)
        or prepared.get("sha256") != record["prepared_graph_sha256"]
        or trace.get("status") != "complete"
        or trace.get("mlir", {}).get("sha256") != source_sha256
    ):
        raise ValueError("buffer observation differs from the complete prepared graph/IR trace")
    inputs, outputs, effects = record["inputs"], record["outputs"], record["operator_effects"]
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not outputs or not isinstance(effects, list):
        raise ValueError("buffer observation lacks complete input/output/effect rosters")
    from merlin.compiler.frontend.signature import _WEIGHT_DTYPES

    for ordinal, row in enumerate(inputs):
        if (
            not isinstance(row, dict)
            or set(row) != _INPUT
            or type(row["ordinal"]) is not int
            or row["ordinal"] != ordinal
            or row["kind"] not in {"BUFFER", "PARAMETER", "USER_INPUT", "CONSTANT_TENSOR"}
            or any(not isinstance(row[name], str) or not row[name] for name in ("argument", "storage_key"))
            or not isinstance(row["dtype"], str)
            or row["dtype"] not in _WEIGHT_DTYPES
            or not is_sha256(row["payload_sha256"])
        ):
            raise ValueError("buffer observation has an invalid typed input roster")
        if (
            not isinstance(row["shape"], list)
            or any(type(n) is not int or n < 0 for n in row["shape"])
            or not isinstance(row["strides"], list)
            or len(row["strides"]) != len(row["shape"])
            or any(type(n) is not int or n < 0 for n in row["strides"])
            or any(
                type(row[name]) is not int or row[name] < 0
                for name in ("storage_group", "storage_offset", "storage_bytes", "payload_bytes", "version")
            )
            or row["payload_bytes"] != prod(row["shape"]) * _WEIGHT_DTYPES[row["dtype"]][2]
            or (
                row["target"] is not None
                if row["kind"] == "USER_INPUT"
                else not isinstance(row["target"], str) or not row["target"]
            )
        ):
            raise ValueError("buffer observation has invalid source tensor storage/role metadata")
    placeholders = [node for node in prepared.get("nodes", ()) if node.get("op") == "placeholder"]
    if len(placeholders) != len(inputs) or any(
        node.get("input_kind") != row["kind"]
        or node.get("input_target") != row["target"]
        or node.get("target") != row["argument"]
        for node, row in zip(placeholders, inputs, strict=True)
    ):
        raise ValueError("buffer observation differs from the prepared input roles/targets")
    names = record["placeholder_names"]
    if (
        not isinstance(names, list)
        or len(names) != len(inputs)
        or any(
            not isinstance(name, dict)
            or set(name) != {"ordinal", "name", "target", "argument", "prepared_name"}
            or name["ordinal"] != row["ordinal"]
            or name["target"] != row["argument"]
            or name["argument"] != row["argument"]
            or name["prepared_name"] != row["argument"]
            or not isinstance(name["name"], str)
            or not name["name"]
            for name, row in zip(names, inputs, strict=True)
        )
    ):
        raise ValueError("buffer observation has incomplete live placeholder/name correspondence")
    if len({row["name"] for row in names}) != len(names) or len({row["argument"] for row in names}) != len(names):
        raise ValueError("buffer observation placeholder mapping is not one-to-one")
    if [row["ordinal"] for row in outputs if isinstance(row, dict)] != list(range(len(outputs))) or any(
        not isinstance(row, dict)
        or set(row) != {"ordinal", "kind", "argument", "target"}
        or row["kind"] != "USER_OUTPUT"
        or row["target"] is not None
        or not isinstance(row["argument"], str)
        for row in outputs
    ):
        raise ValueError("buffer observation contains mutation, unknown or incomplete output effects")
    nodes = [node for node in prepared.get("nodes", ()) if node.get("op") == "call_function"]
    if any(node.get("op") not in {"placeholder", "output", "call_function"} for node in prepared.get("nodes", ())):
        raise ValueError("prepared trace has unsupported or unknown effect nodes")
    if len(effects) != len(nodes):
        raise ValueError("buffer observation omitted a prepared operator's effects")
    schemas = prepared.get("operator_schemas", {})
    for effect, node in zip(effects, nodes, strict=True):
        if (
            not isinstance(effect, dict)
            or set(effect) != {"ordinal", "target", "schema", "writes"}
            or effect["ordinal"] != node.get("ordinal")
            or effect["target"] != node.get("target")
            or effect["writes"] != []
            or effect["schema"] != schemas.get(effect["target"])
            or effect["schema"] is None
            and effect["target"] != str(operator.getitem)
        ):
            raise ValueError("buffer observation has unknown, changed or writing operator schemas")
    for row in inputs:
        if row["kind"] == "BUFFER" and sum(item["storage_group"] == row["storage_group"] for item in inputs) != 1:
            raise ValueError("buffer observation has unsupported source storage aliasing")
    return json.loads(json.dumps(record, sort_keys=True, allow_nan=False))


def authorize_freeze(directory, staged, function, *, consent, max_bytes):
    """Return exact selected buffer snapshots after explicit original-caller consent."""
    if type(consent) is not bool:
        raise ValueError("freeze_readonly_buffers requires an explicit Boolean")
    buffers = {i: row for i, row in enumerate(staged.argument_bindings) if row["kind"] == "buffer"}
    if not consent:
        if buffers:
            raise ValueError("captured buffer binding requires an admitted immutable-state or state ABI contract")
        return None
    from merlin.compiler.frontend.archives import _read_member
    from merlin.compiler.frontend.inputs import load_lifted_argument_bytes
    from merlin.compiler.frontend.signature import _WEIGHT_DTYPES
    from merlin.llvmlower.prepared_operand_effects import validate_pure_tensor_function

    manifest = json.loads(_read_member(directory, "compile-inputs.json"))
    record = (manifest.get("source") or {}).get("export_state")
    trace = json.loads(_read_member(directory, "frontend-trace.json"))
    observation = validate_observation(record, trace=trace, source_sha256=staged.original_source_sha256)
    observed = {row["ordinal"]: row for row in observation["inputs"] if row["kind"] == "BUFFER"}
    if not buffers or set(observed) != set(buffers) or len(observation["inputs"]) != len(staged.argument_bindings):
        raise ValueError("freeze consent requires the exact observed BUFFER argument roster")
    validate_pure_tensor_function(function, {staged.entry: function}, allow_named_linalg=True)
    payloads = load_lifted_argument_bytes(directory, staged, kinds=("buffer",), max_bytes=max_bytes)
    for ordinal, row in buffers.items():
        actual = observed[ordinal]
        if (
            row["weight"] != actual["storage_key"]
            or row["shape"] != actual["shape"]
            or row["dtype"] != _WEIGHT_DTYPES[actual["dtype"]][0]
            or hashlib.sha256(payloads[ordinal]).hexdigest() != actual["payload_sha256"]
        ):
            raise ValueError("buffer snapshot differs from the original prepared source/storage bytes")
    return {
        "schema": "merlin.buffer-freeze.v1",
        "consent": True,
        "scope": "all_declared_observed_readonly_buffer_snapshots_for_compiled_invocations",
        "export_state_sha256": hashlib.sha256(
            json.dumps(observation, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "buffers": [
            {
                "argument_index": i,
                "kind": "buffer",
                "target": observed[i]["target"],
                "binding": "frozen_snapshot",
                "payload_sha256": observed[i]["payload_sha256"],
            }
            for i in buffers
        ],
        "qualification": "none; original-caller preparation and independent numerical/reference admission remain",
    }
