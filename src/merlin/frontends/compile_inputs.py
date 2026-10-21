"""Stage the declared compiler inputs from a materialized model capture.

This host-owned boundary is deliberately narrower than a capture bundle. The
compiler receives parsed IR, external weights, source correspondence, and a
static signature. Runtime inputs and reference outputs stay with the evaluator.
The capture producer's receipt establishes byte identity for copied members;
it does not establish source closure or numerical equivalence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from xdsl.dialects.builtin import StringAttr, TensorType

from merlin.capture.safetensors import read_header
from merlin.frontends.capture_arguments import CONSTANT_FILE, materialize_arguments, needs_materialization
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text as module_text

_COPIED = ("weights.safetensors", "weights.safetensors.manifest.json", "frontend-trace.json")
_SOURCE = "model.mlir"
_STAGED_REQUIRED = frozenset({"program.mlir", "signature.json", *_COPIED})
_STAGED_OPTIONAL = frozenset({"precision-policy.json", "input_order.json", CONSTANT_FILE})
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


class LiftedArgumentResourceLimit(ValueError):
    """The selected payload exceeds the caller's materialization budget."""

    status = "resource_limit"


@dataclass(frozen=True)
class StagedCompileInputs:
    program: str
    entry: str
    precision_policy: dict | None
    precision_policy_document: bytes | None
    original_source_sha256: str
    program_sha256: str
    precision_policy_sha256: str | None
    argument_bindings: tuple[dict, ...] = ()
    weights_sha256: str | None = None
    constants_sha256: str | None = None


@dataclass(frozen=True)
class _ArchiveInfo:
    header: dict
    payload_start: int
    file_bytes: int
    checked_sha256: str | None


def _archive_info(
    directory: Path, name: str, *, expected_sha256: str | None = None, typed: bool = True,
) -> _ArchiveInfo:
    """Read bounded tensor metadata and, when requested, hash without copying payloads."""
    member = directory / name
    if member.is_symlink() or not member.is_file():
        raise ValueError(f"capture member is absent or a symlink: {name}")
    with member.open("rb") as stream:
        before = os.fstat(stream.fileno())
        header, start = read_header(stream) if typed else ({}, 0)
        if start > before.st_size:
            raise ValueError("tensor archive header extends past its file")
        checked_sha256 = None
        if expected_sha256 is not None:
            stream.seek(0)
            checked_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            if checked_sha256 != expected_sha256:
                raise ValueError(f"compiler input member differs from manifest: {name}")
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ValueError("tensor archive changed while checking its metadata")
    return _ArchiveInfo(header, start, before.st_size, checked_sha256)


def _sha(data: bytes | _ArchiveInfo) -> str:
    if isinstance(data, _ArchiveInfo):
        if data.checked_sha256 is None:
            raise ValueError("tensor archive identity has not been checked")
        return data.checked_sha256
    return hashlib.sha256(data).hexdigest()


def _copy_archive(source: Path, destination: Path, info: _ArchiveInfo) -> None:
    """Copy and hash the same bounded chunks before publishing a staged archive."""
    digest, size = hashlib.sha256(), 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != info.file_bytes or digest.hexdigest() != _sha(info):
        raise ValueError("capture archive changed during staging")


def _read_member(directory: Path, name: str) -> bytes:
    member = directory / name
    if member.is_symlink() or not member.is_file():
        raise ValueError(f"capture member is absent or a symlink: {name}")
    return member.read_bytes()


def _checked_member(directory: Path, receipt: dict, name: str) -> bytes:
    entry = (receipt.get("artifacts") or {}).get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"capture receipt lacks {name}")
    raw = _read_member(directory, name)
    if entry.get("bytes") != len(raw) or entry.get("sha256") != _sha(raw):
        raise ValueError(f"capture member differs from receipt: {name}")
    return raw


def _selected_precision_policy(path: Path, source: Path, source_sha256: str) -> bytes:
    """Bind a separately selected policy to the captured source bytes.

    Staging does not qualify the policy's arithmetic or target legality. The
    selected target provider must check those before using it to lower IR.
    """
    if (
        path.is_symlink()
        or any(parent.is_symlink() for parent in path.parents)
        or not path.is_file()
        or path.resolve().is_relative_to(source)
    ):
        raise ValueError("precision policy must be a separate regular file")
    raw = path.read_bytes()
    policy = json.loads(raw)
    if (
        not isinstance(policy, dict)
        or not isinstance(policy.get("schema"), str)
        or not policy["schema"]
        or not isinstance(policy.get("source_sha256"), str)
        or len(policy["source_sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in policy["source_sha256"])
    ):
        raise ValueError("precision policy needs a schema and canonical source SHA-256")
    if policy["source_sha256"] != source_sha256:
        raise ValueError("precision policy differs from captured source bytes")
    return raw


def _entry_type(module, entry: str):
    functions = [
        op for op in module.walk()
        if op.name == "func.func" and op.sym_name.data == entry
    ]
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
    return raw.file_bytes - raw.payload_start if isinstance(raw, _ArchiveInfo) else (
        len(raw) - 8 - int.from_bytes(raw[:8], "little")
    )


def _argument_bindings(
    function_type, runtime_inputs: list, members: dict[str, bytes | _ArchiveInfo], *, materialized: bool = False,
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
            not isinstance(kind, str) or kind not in roles or set(row) != fields
            or not isinstance(file, str) or file not in files
            or kind == "param" and file != "weights.safetensors"
            or kind == "constant" and file != CONSTANT_FILE
            or not isinstance(name, str) or not name
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
            not isinstance(offsets, list) or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] < 0 or offsets[1] - offsets[0] != elements * dtype[2]
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
        "merlin.compile-signature.v1", "merlin.compile-signature.v2", "merlin.compile-signature.v3",
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
            function_type, rows, members, materialized=signature["schema"] == "merlin.compile-signature.v3",
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


def stage_compile_inputs(
    capture_dir: str | Path,
    destination: str | Path,
    *,
    precision_policy: str | Path | None = None,
) -> dict:
    """Publish a new, immutable compiler-only directory for one captured model.

    The destination must be absent. The caller owns capture permission and the
    evaluator's separate runtime bundle. This function never reads the runtime
    input or golden members, even if the producer receipt lists them.
    """
    source = Path(capture_dir).absolute()
    target = Path(destination).absolute()
    if (
        source.is_symlink()
        or any(parent.is_symlink() for parent in source.parents)
        or target.exists()
        or target.is_symlink()
    ):
        raise ValueError("capture must be a real directory and destination must be absent")
    if not source.is_dir():
        raise ValueError("capture directory is absent")
    receipt_raw = _read_member(source, "capture_receipt.json")
    receipt = json.loads(receipt_raw)
    if receipt.get("schema") != "m2m.capture-receipt.v1":
        raise ValueError("unsupported capture receipt")
    materialized_abi = receipt.get("materialized_abi") or {}
    if materialized_abi.get("complete") is not True:
        raise ValueError("capture has incomplete ABI or unbound lifted constants")
    raw = _checked_member(source, receipt, _SOURCE)
    copied = {name: _checked_member(source, receipt, name) for name in _COPIED if name != "weights.safetensors"}
    weight_manifest = json.loads(copied["weights.safetensors.manifest.json"])
    if not isinstance(weight_manifest, dict):
        raise ValueError("capture weight manifest is not a mapping")
    entry = (receipt.get("artifacts") or {}).get("weights.safetensors")
    if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
        raise ValueError("capture receipt lacks weights.safetensors identity")
    try:
        weights = _archive_info(
            source, "weights.safetensors", expected_sha256=entry["sha256"], typed=bool(weight_manifest),
        )
    except ValueError as error:
        raise ValueError("capture archive invalid or differs from receipt: " + str(error)) from error
    if entry.get("bytes") != weights.file_bytes:
        raise ValueError("capture member differs from receipt: weights.safetensors")
    copied["weights.safetensors"] = weights
    materialized = needs_materialization(weight_manifest, receipt)
    if weight_manifest:
        copied["input_order.json"] = _checked_member(source, receipt, "input_order.json")
    if precision_policy is not None:
        copied["precision-policy.json"] = _selected_precision_policy(
            Path(precision_policy).absolute(), source, _sha(raw)
        )
    metadata = json.loads(_checked_member(source, receipt, "meta.json"))
    if metadata.get("ok") is not True or metadata.get("opaque") != 0:
        raise ValueError("capture has opaque or failed frontend operations")
    if metadata.get("frontend_trace", {}).get("sha256") != _sha(copied["frontend-trace.json"]):
        raise ValueError("frontend trace differs from capture metadata")
    trace = json.loads(copied["frontend-trace.json"])
    if trace.get("status") != "complete" or trace.get("mlir", {}).get("sha256") != _sha(raw):
        raise ValueError("frontend trace does not bind complete captured IR")
    module = parse_mlir_text(raw.decode("utf-8"))
    old_reference = module.attributes.get("prov.weights_file")
    if not isinstance(old_reference, StringAttr) or old_reference.data != metadata.get("weights"):
        raise ValueError("parsed IR weights reference differs from capture metadata")
    module.attributes["prov.weights_file"] = StringAttr("weights.safetensors")
    module.verify()
    staged_text = module_text(module, generic=True)
    parsed = parse_mlir_text(staged_text)
    parsed.verify()
    if parsed.attributes.get("prov.weights_file") != StringAttr("weights.safetensors"):
        raise ValueError("staged IR lost its relative weights reference")
    staged = staged_text.encode("utf-8")
    entry = metadata.get("func_name")
    if entry is None:
        functions = [op for op in parsed.walk() if op.name == "func.func"]
        if len(functions) != 1:
            raise ValueError("capture metadata must name its entry when parsed IR has multiple functions")
        entry = functions[0].sym_name.data
    signature = {
        "schema": "merlin.compile-signature.v2" if weight_manifest else "merlin.compile-signature.v1",
        "entry": entry,
        "inputs": metadata.get("input_abi"),
        "outputs": metadata.get("output_abi"),
        "shape_scope": "static_captured_signature",
        "capture_dtype": metadata.get("dtype"),
    }
    if (
        not isinstance(signature["entry"], str)
        or not isinstance(signature["inputs"], list)
        or not isinstance(signature["outputs"], list)
        or materialized_abi.get("inputs") != len(signature["inputs"])
    ):
        raise ValueError("capture has an incomplete invocation signature")
    materialization = None
    if materialized:
        order = json.loads(copied.get("input_order.json", b"{}"))
        if not isinstance(order, dict) or set(order) & set(receipt.get("lifted_constants", {})):
            raise ValueError("lifted constants cannot occupy runtime input positions")
        weight_manifest, copied[CONSTANT_FILE], materialization = materialize_arguments(
            weight_manifest,
            [_tensor_abi(type_) for type_ in _entry_type(parsed, signature["entry"]).inputs.data],
            receipt, _checked_member(source, receipt, "extra.npz"), _WEIGHT_DTYPES,
        )
        copied["weights.safetensors.manifest.json"] = json.dumps(weight_manifest, sort_keys=True).encode()
        signature["schema"] = "merlin.compile-signature.v3"
    if weight_manifest:
        signature["arguments"] = _argument_bindings(
            _entry_type(parsed, signature["entry"]), signature["inputs"], copied, materialized=materialized,
        )
    _check_signature(parsed, signature, copied)
    signature_raw = (json.dumps(signature, sort_keys=True, indent=2) + "\n").encode()
    manifest = {
        "schema": "merlin.compile-inputs.v1",
        "scope": "compiler_only_static_capture; source_closure_and_numerical_admission_unverified",
        "source": {
            "capture_receipt_sha256": _sha(receipt_raw),
            "model_mlir_sha256": _sha(raw),
            **({"argument_materialization": materialization} if materialization is not None else {}),
        },
        "members": {
            "program.mlir": _sha(staged),
            "signature.json": _sha(signature_raw),
            **{name: _sha(value) for name, value in copied.items()},
        },
        "excluded_evaluator_members": ["inputs.npz", "inputs.json", "golden.npy", "golden.json", "extra.npz"],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        for name, value in {"program.mlir": staged, "signature.json": signature_raw, **copied}.items():
            if isinstance(value, _ArchiveInfo):
                _copy_archive(source / name, temporary / name, value)
            else:
                (temporary / name).write_bytes(value)
        (temporary / "compile-inputs.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        if target.exists() or target.is_symlink():
            raise ValueError("compile-input destination appeared during staging")
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def load_compile_inputs(directory: str | Path) -> StagedCompileInputs:
    """Recheck a compiler-only stage before handing its IR to a compiler.

    A precision policy is returned for explicit target validation. This loader
    grants no numerical transformation and never opens capture/evaluator files.
    """
    root = Path(directory).absolute()
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents) or not root.is_dir():
        raise ValueError("compiler input directory must be real and present")
    manifest_raw = _read_member(root, "compile-inputs.json")
    manifest = json.loads(manifest_raw)
    if not isinstance(manifest, dict) or manifest.get("schema") != "merlin.compile-inputs.v1":
        raise ValueError("unsupported compiler input manifest")
    members = manifest.get("members")
    if not isinstance(members, dict) or not _STAGED_REQUIRED <= set(members) <= (
        _STAGED_REQUIRED | _STAGED_OPTIONAL
    ):
        raise ValueError("compiler input manifest has missing or unknown members")
    if {member.name for member in root.iterdir()} != {"compile-inputs.json", *members}:
        raise ValueError("compiler input directory has missing or undeclared members")
    raw_members = {}
    for name, expected in members.items():
        if not isinstance(expected, str) or len(expected) != 64 or any(
            char not in "0123456789abcdef" for char in expected
        ):
            raise ValueError(f"compiler input member has invalid digest: {name}")
        if name in {"weights.safetensors", CONSTANT_FILE}:
            continue
        raw_members[name] = _read_member(root, name)
        if _sha(raw_members[name]) != expected:
            raise ValueError(f"compiler input member differs from manifest: {name}")
    signature = json.loads(raw_members["signature.json"])
    if not isinstance(signature, dict):
        raise ValueError("compiler input signature has an unsupported schema")
    # The legacy signature has no typed weight bindings. Preserve its opaque
    # member identity check without inventing a tensor-archive requirement.
    for name in sorted({"weights.safetensors", CONSTANT_FILE} & set(members)):
        raw_members[name] = _archive_info(
            root, name, expected_sha256=members[name], typed=signature.get("schema") != "merlin.compile-signature.v1",
        )
    _check_signature(parse_mlir_text(raw_members["program.mlir"].decode("utf-8")), signature, raw_members)
    policy = None
    if "precision-policy.json" in raw_members:
        policy = json.loads(raw_members["precision-policy.json"])
        source_record = manifest.get("source")
        if not isinstance(source_record, dict):
            raise ValueError("compiler input manifest lacks captured source identity")
        source_sha256 = source_record.get("model_mlir_sha256")
        if (
            not isinstance(policy, dict)
            or not isinstance(source_sha256, str)
            or policy.get("source_sha256") != source_sha256
        ):
            raise ValueError("staged precision policy differs from captured source identity")
    source_record = manifest.get("source")
    original_source_sha256 = source_record.get("model_mlir_sha256") if isinstance(source_record, dict) else None
    if (
        not isinstance(original_source_sha256, str)
        or len(original_source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in original_source_sha256)
    ):
        raise ValueError("compiler input manifest lacks a canonical captured source identity")
    return StagedCompileInputs(
        raw_members["program.mlir"].decode("utf-8"),
        signature["entry"],
        policy,
        raw_members.get("precision-policy.json"),
        original_source_sha256,
        members["program.mlir"],
        members.get("precision-policy.json"),
        tuple(signature.get("arguments", ())),
        members["weights.safetensors"],
        members.get(CONSTANT_FILE),
    )


def load_lifted_argument_bytes(
    directory: str | Path, staged: StagedCompileInputs, *, kinds: tuple[str, ...] | None = None,
    max_bytes: int | None = None,
) -> dict[int, bytes]:
    """Read exact declared parameter bytes for a previously checked stage.

    The staged signature identifies function argument roles. Rehash the weight
    file while extracting the selected ranges, so payloads and digest come
    from the same stream. ``kinds`` can select small lifted constants without
    materializing model parameters or mutable state. The optional byte budget
    counts unique selected tensors before any payload allocation. Other roles
    stay in the staged signature; selection does not change their meaning.
    Runtime tensors and reference outputs are never read.
    """
    roles = {"param", "buffer", "constant"}
    if kinds is not None and (
        not isinstance(kinds, tuple) or any(not isinstance(kind, str) or kind not in roles for kind in kinds)
        or len(set(kinds)) != len(kinds)
    ):
        raise ValueError("lifted argument selection needs distinct declared roles")
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("lifted argument byte budget must be a nonnegative integer")
    if not staged.argument_bindings:
        return {}
    root = Path(directory).absolute()
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError("compiler input directory must be real")
    selected = roles if kinds is None else set(kinds)
    archives = {}
    for file, digest in (("weights.safetensors", staged.weights_sha256), (CONSTANT_FILE, staged.constants_sha256)):
        if file == CONSTANT_FILE and digest is None:
            continue
        if digest is None:
            raise ValueError("staged tensor archive lacks its checked identity")
        archives[file] = (_archive_info(root, file), digest)
    ranges, indices, total = {}, {}, 0
    for index, binding in enumerate(staged.argument_bindings):
        if binding["kind"] not in roles | {"input"}:
            raise ValueError("staged argument has an unknown role")
        if binding["kind"] not in selected:
            continue
        name = binding["weight"]
        file = binding.get("file", "weights.safetensors")
        if file not in archives:
            raise ValueError("staged argument names an unchecked tensor archive")
        info, _ = archives[file]
        key = (file, name)
        row = info.header.get(name) if name != "__metadata__" else None
        if not isinstance(row, dict) or not isinstance(row.get("data_offsets"), list):
            raise ValueError("selected tensor is missing from its checked archive")
        offsets = row["data_offsets"]
        if len(offsets) != 2 or any(type(value) is not int for value in offsets):
            raise ValueError("selected tensor has malformed byte offsets")
        start, end = offsets
        if not 0 <= start <= end <= _payload_bytes(info):
            raise ValueError("selected tensor extends past its checked archive")
        if key not in ranges:
            ranges[key] = (start + info.payload_start, end + info.payload_start)
            total += end - start
        indices[index] = key
    if max_bytes is not None and total > max_bytes:
        raise LiftedArgumentResourceLimit("selected lifted arguments exceed the byte budget")
    by_weight = {}
    for file, (info, digest) in archives.items():
        portions = {key: [] for key in ranges if key[0] == file}
        hasher, position, header_parts = hashlib.sha256(), 0, []
        with (root / file).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
                if position < info.payload_start:
                    header_parts.append(chunk[:info.payload_start - position])
                for key, parts in portions.items():
                    start, end = ranges[key]
                    a, b = max(start, position), min(end, position + len(chunk))
                    if a < b:
                        parts.append(chunk[a - position:b - position])
                position += len(chunk)
        if hasher.hexdigest() != digest or position != info.file_bytes:
            role = "weights" if file == "weights.safetensors" else "constants"
            raise ValueError(f"staged {role} changed after compiler input validation")
        actual_header, actual_start = read_header(BytesIO(b"".join(header_parts)))
        if actual_header != info.header or actual_start != info.payload_start:
            raise ValueError("tensor archive metadata changed before payload extraction")
        by_weight.update((key, b"".join(parts)) for key, parts in portions.items())
    return {index: by_weight[key] for index, key in indices.items()}
