"""Reconcile staged invocation and lifted tensor bindings with parsed IR."""

from __future__ import annotations

import hashlib
import inspect
import json
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from math import prod
from pathlib import Path

from xdsl.dialects.builtin import TensorType

from merlin.capture.safetensors import read_header
from merlin.common import digest as _mdigest
from merlin.common.digest import is_sha256
from merlin.compiler.dialects.xdsl_context import text as module_text
from merlin.compiler.frontend.inputs import CONSTANT_FILE, _ArchiveInfo
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

# ---- signature ---------------------------------------------------------------------------------

_WEIGHT_DTYPES = {
    "float32": ("f32", "F32", 4),
    "float16": ("f16", "F16", 2),
    "bfloat16": ("bf16", "BF16", 2),
    "float64": ("f64", "F64", 8),
    "int64": ("i64", "I64", 8),
    "int32": ("i32", "I32", 4),
    "int16": ("i16", "I16", 2),
    "int8": ("i8", "I8", 1),
    "uint8": ("ui8", "U8", 1),
    "bool": ("i1", "BOOL", 1),
}


def _entry_type(module, entry: str):
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1:
        raise ValueError("compiler input signature does not name exactly one parsed function")
    return functions[0].function_type


def _tensor_abi(type_) -> dict:
    if not isinstance(type_, TensorType) or not type_.has_static_shape():
        raise ValueError("compiler input signature needs static ranked tensor arguments")
    return {"shape": list(type_.get_shape()), "dtype": str(type_.element_type)}


def _weight_header(raw: bytes | _ArchiveInfo) -> dict:
    header = raw.header if isinstance(raw, _ArchiveInfo) else read_header(BytesIO(raw))[0]
    return {name: row for name, row in header.items() if name != "__metadata__"}


def _payload_bytes(raw: bytes | _ArchiveInfo) -> int:
    return (
        raw.file_bytes - raw.payload_start
        if isinstance(raw, _ArchiveInfo)
        else (len(raw) - 8 - int.from_bytes(raw[:8], "little"))
    )


def _argument_bindings(
    function_type,
    runtime_inputs: list,
    members: dict[str, bytes | _ArchiveInfo],
    *,
    materialized: bool = False,
) -> list[dict]:
    """Join captured argument indices to runtime inputs or exact weight tensors."""
    manifest = json.loads(members["weights.safetensors.manifest.json"])
    order_raw = members.get("input_order.json")
    order = json.loads(order_raw) if order_raw is not None else None
    types = list(function_type.inputs.data)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {str(index) for index in range(len(types))}
        or not isinstance(order, dict)
        or len(order) != len(runtime_inputs)
        or any(type(index) is not int for index in order.values())
        or set(order.values()) != set(range(len(runtime_inputs)))
    ):
        raise ValueError("weight manifest or input order does not cover the parsed arguments")
    files = ("weights.safetensors", CONSTANT_FILE) if materialized else ("weights.safetensors",)
    if (CONSTANT_FILE in members) != materialized:
        raise ValueError("capture constant archive differs from signature schema")
    headers = {name: _weight_header(members[name]) for name in files}
    roles = {"param", "buffer", "constant"} if materialized else {"param", "buffer"}
    fields = {"kind", "weight", "dtype", "shape"} | ({"file"} if materialized else set())
    bindings = []
    used_inputs: set[int] = set()
    used_weights = {name: set() for name in files}
    ranges = {name: [] for name in files}
    payload_lengths = {name: _payload_bytes(members[name]) for name in files}
    for index, type_ in enumerate(types):
        row = manifest[str(index)]
        actual = _tensor_abi(type_)
        if not isinstance(row, dict):
            raise ValueError("weight manifest has a malformed argument row")
        kind = row.get("kind")
        if kind == "input":
            name = row.get("name")
            if set(row) != {"kind", "name"} or not isinstance(name, str) or name not in order:
                raise ValueError("weight manifest input has no declared runtime position")
            ordinal = order[name]
            if runtime_inputs[ordinal] != actual:
                raise ValueError("runtime input argument differs from parsed function")
            used_inputs.add(ordinal)
            bindings.append({"kind": "input", "name": name, "input_index": ordinal, **actual})
            continue
        name = row.get("weight")
        file = row.get("file") if materialized else "weights.safetensors"
        dtype_name = row.get("dtype")
        dtype = _WEIGHT_DTYPES.get(dtype_name) if isinstance(dtype_name, str) else None
        if (
            not isinstance(kind, str)
            or kind not in roles
            or set(row) != fields
            or not isinstance(file, str)
            or file not in files
            or kind == "param"
            and file != "weights.safetensors"
            or kind == "constant"
            and file != CONSTANT_FILE
            or not isinstance(name, str)
            or not name
            or dtype is None
            or not isinstance(row.get("shape"), list)
            or any(type(extent) is not int or extent < 0 for extent in row["shape"])
            or row["shape"] != actual["shape"]
            or dtype[0] != actual["dtype"]
        ):
            raise ValueError("weight argument shape or dtype differs from parsed function")
        stored = headers[file].get(name)
        if not isinstance(stored, dict) or stored.get("dtype") != dtype[1] or stored.get("shape") != actual["shape"]:
            raise ValueError("weight safetensors shape or dtype differs from manifest")
        offsets = stored.get("data_offsets")
        elements = 1
        for extent in actual["shape"]:
            elements *= extent
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] < 0
            or offsets[1] - offsets[0] != elements * dtype[2]
            or offsets[1] > payload_lengths[file]
        ):
            raise ValueError("weight safetensors payload differs from declared shape")
        if name not in used_weights[file]:
            ranges[file].append((offsets[0], offsets[1]))
        used_weights[file].add(name)
        bindings.append({"kind": row["kind"], "weight": name, **actual, **({"file": file} if materialized else {})})
    if used_inputs != set(range(len(runtime_inputs))) or any(
        used_weights[file] != set(headers[file]) for file in files
    ):
        raise ValueError("weight manifest leaves a runtime input or stored tensor unbound")
    for file in files:
        cursor = 0
        for start, end in sorted(ranges[file]):
            if start != cursor:
                raise ValueError("weight safetensors payload has an overlap or gap")
            cursor = end
        if cursor != payload_lengths[file]:
            raise ValueError("weight safetensors payload has undeclared bytes")
    return bindings


def _check_signature(module, signature: dict, members: dict[str, bytes]) -> None:
    """Reconcile the declared invocation ABI with the parsed entry function.

    The capture receipt authenticates bytes, but it cannot establish that its
    separately produced ABI describes the IR. Check this again when consuming
    a staged directory because its manifest is caller-owned data.
    """
    if not isinstance(signature, dict) or signature.get("schema") not in {
        "merlin.compile-signature.v1",
        "merlin.compile-signature.v2",
        "merlin.compile-signature.v3",
    }:
        raise ValueError("compiler input signature has an unsupported schema")
    entry = signature.get("entry")
    if not isinstance(entry, str) or not entry:
        raise ValueError("compiler input signature lacks an entry function")
    function_type = _entry_type(module, entry)
    if signature["schema"] != "merlin.compile-signature.v1":
        rows = signature.get("inputs")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("compiler input signature has invalid runtime inputs")
        expected = _argument_bindings(
            function_type,
            rows,
            members,
            materialized=signature["schema"] == "merlin.compile-signature.v3",
        )
        if signature.get("arguments") != expected:
            raise ValueError("compiler input signature arguments differ from parsed function")
    else:
        weight_manifest = json.loads(members["weights.safetensors.manifest.json"])
        if not isinstance(weight_manifest, dict):
            raise ValueError("compiler input weight manifest is not a mapping")
        if weight_manifest or CONSTANT_FILE in members:
            raise ValueError("legacy compiler input signature cannot bind externalized weights")
    directions = [("outputs", function_type.outputs.data)]
    if signature["schema"] == "merlin.compile-signature.v1":
        directions.insert(0, ("inputs", function_type.inputs.data))
    for direction, types in directions:
        rows = signature.get(direction)
        if not isinstance(rows, list) or len(rows) != len(types):
            raise ValueError(f"compiler input signature {direction} differ from parsed function")
        for row, type_ in zip(rows, types):
            if (
                not isinstance(row, dict)
                or set(row) != {"shape", "dtype"}
                or not isinstance(row["shape"], list)
                or any(type(size) is not int or size < 0 for size in row["shape"])
                or not isinstance(row["dtype"], str)
                or row != _tensor_abi(type_)
            ):
                raise ValueError(f"compiler input signature {direction} differ from parsed function")


# ---- state -------------------------------------------------------------------------------------
# Check source-bound buffer observations and explicit snapshot consent.
#
# Producer observations are not numerical or reference-origin certificates. Only
# the original caller may authorize freezing, and existing pure tensor/source and
# independent preparation admission remain required.

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
    from merlin.compiler.frontend.inputs import _read_member, load_lifted_argument_bytes
    from merlin.compiler.lower.source.prepared import validate_pure_tensor_function

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


# ---- argument_identity -------------------------------------------------------------------------
# Host-owned normalization replay that preserves captured entry argument identities.
#
# This proves an input-binding bridge, not numerical equivalence of normalization.
# Callbacks and their source pins must be chosen by the trusted host policy, never
# loaded from candidate code or a candidate-supplied recipe. Serialized receipts are
# audit output, not bearer permissions accepted from an untrusted caller.


@dataclass(frozen=True)
class ArgumentIdentityStage:
    apply: Callable
    reparse_before: bool = False


@dataclass(frozen=True)
class ArgumentIdentityBridge:
    source_sha256: str
    normalized_sha256: str
    entry: str
    argument_types: tuple[str, ...]
    stages: tuple[tuple[str, bool], ...]
    source_pins: tuple[tuple[str, str], ...]

    def to_evidence(self) -> dict:
        return {
            "schema": "entry_argument_identity_bridge_v1",
            "source_sha256": self.source_sha256,
            "normalized_sha256": self.normalized_sha256,
            "entry": self.entry,
            "argument_types": list(self.argument_types),
            "argument_index_map": list(range(len(self.argument_types))),
            "stages": [{"callable": name, "reparse_before": parse} for name, parse in self.stages],
            "source_pins": dict(self.source_pins),
            "scope": "entry argument identity/order/type only; not normalization numerical equivalence",
        }


_sha = _mdigest.sha256_text


def _arguments(module, entry: str):
    functions = [op for op in module.body.block.ops if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("argument identity requires one single-block source entry")
    return tuple(functions[0].body.block.args)


def replay_argument_identity(
    *,
    raw_text: str,
    source_sha256: str,
    normalized_sha256: str,
    entry: str,
    stages: Sequence[ArgumentIdentityStage],
    source_pins: Mapping[Path, str],
) -> ArgumentIdentityBridge:
    """Replay host-selected passes and reject replacement, permutation or retyping.

    Explicit print/parse boundaries preserve ordered typed arguments by syntax;
    every transform between those boundaries must preserve the actual argument
    objects. Both endpoint byte identities are checked, not inferred from shape.
    No model arithmetic is evaluated and no candidate is imported by this helper.
    """
    if _sha(raw_text) != source_sha256:
        raise ValueError("raw source differs from the host-pinned capture")
    pins = {Path(path).resolve(): digest for path, digest in source_pins.items()}
    own_source = Path(__file__).resolve()
    pins.setdefault(own_source, hashlib.sha256(own_source.read_bytes()).hexdigest())

    def verify_pins():
        for path, expected in pins.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("normalization implementation differs from host source pins")

    verify_pins()
    names = []
    for stage in stages:
        if not isinstance(stage, ArgumentIdentityStage) or type(stage.reparse_before) is not bool:
            raise ValueError("normalization stages must be explicit host-selected operations")
        implementation = inspect.getsourcefile(stage.apply)
        if implementation is None or Path(implementation).resolve() not in pins:
            raise ValueError("normalization callback has no pinned host implementation")
        names.append((f"{stage.apply.__module__}.{stage.apply.__qualname__}", stage.reparse_before))
    module = parse_mlir_text(raw_text)
    original = _arguments(module, entry)
    types = tuple(str(arg.type) for arg in original)
    for stage in stages:
        if stage.reparse_before:
            serialized = module_text(module)
            module = parse_mlir_text(serialized)
            reparsed = _arguments(module, entry)
            if tuple(str(arg.type) for arg in reparsed) != types:
                raise ValueError("serialization changed the ordered entry argument types")
        before = _arguments(module, entry)
        stage.apply(module)
        after = _arguments(module, entry)
        if (
            len(before) != len(after)
            or any(left is not right for left, right in zip(before, after))
            or tuple(str(arg.type) for arg in after) != types
        ):
            raise ValueError("normalization changed entry argument identity, order or type")
    normalized = module_text(module)
    if _sha(normalized) != normalized_sha256:
        raise ValueError("normalization replay does not reproduce the exact compiled source")
    verify_pins()
    return ArgumentIdentityBridge(
        source_sha256,
        normalized_sha256,
        entry,
        types,
        tuple(names),
        tuple(sorted((str(path), digest) for path, digest in pins.items())),
    )
