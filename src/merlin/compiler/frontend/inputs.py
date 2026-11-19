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
import struct
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from io import BytesIO
from math import prod
from pathlib import Path

import numpy as np
from xdsl.dialects.builtin import StringAttr

from merlin.capture.safetensors import read_header
from merlin.common.digest import sha256_file
from merlin.compiler.dialects.xdsl_context import text as module_text
from merlin.compiler.frontend.input_types import CONSTANT_FILE, _ArchiveInfo
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
from merlin.compiler.frontend.signature import (
    _WEIGHT_DTYPES,
    _argument_bindings,
    _check_signature,
    _entry_type,
    _payload_bytes,
    _tensor_abi,
)

# ---- archives ----------------------------------------------------------------------------------
# Checked compiler input bytes and bounded tensor-archive identities.


def _archive_info(
    directory: Path,
    name: str,
    *,
    expected_sha256: str | None = None,
    typed: bool = True,
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
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
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


# ---- constants ---------------------------------------------------------------------------------
# Normalize declared capture constants and buffer initializers for compilation.
#
# The capture producer stores these leaves in extra.npz. Only receipt-declared
# constants and manifest-declared buffers are selected. Runtime inputs, outputs
# and unrelated archive members never become compiler tensor arguments.


def needs_materialization(manifest: dict, receipt: dict) -> bool:
    lifted = receipt.get("lifted_constants", {})
    names = (receipt.get("materialized_abi") or {}).get("lifted_constants", [])
    if (
        not isinstance(lifted, dict)
        or not isinstance(names, list)
        or any(not isinstance(name, str) or not name.startswith("c_lifted_tensor_") for name in lifted)
        or names != sorted(lifted)
    ):
        raise ValueError("capture has unbound lifted constants or an invalid constant declaration")
    return bool(lifted) or any(
        isinstance(row, dict)
        and (row.get("kind") == "constant" or row.get("kind") == "buffer" and set(row) == {"kind", "name"})
        for row in manifest.values()
    )


def _array(archive: zipfile.ZipFile, key: str, shape: list[int]) -> np.ndarray:
    """Read one numeric NPY without pickle, extraction or shape inference."""
    member = key + ".npy"
    if member not in archive.namelist():
        raise ValueError("declared capture tensor is absent from the extra archive")
    with archive.open(member) as stream:
        version = np.lib.format.read_magic(stream)
        if version not in {(1, 0), (2, 0)}:
            raise ValueError("capture tensor has an unsupported NPY version")
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        stored_shape, fortran, dtype = reader(stream, max_header_size=65536)
        if stored_shape != tuple(shape) or dtype.hasobject or dtype.fields is not None or dtype.kind not in "biuf":
            raise ValueError("capture tensor shape or numeric storage differs from parsed IR")
        if dtype.itemsize not in (1, 2, 4, 8):
            raise ValueError("capture tensor has an unsupported scalar width")
        size = prod(shape) * dtype.itemsize
        if archive.getinfo(member).file_size - stream.tell() != size:
            raise ValueError("capture tensor payload length differs from its declaration")
        payload = stream.read(size)
        if len(payload) != size:
            raise ValueError("capture tensor payload is truncated")
    return np.frombuffer(payload, dtype=dtype).reshape(shape, order="F" if fortran else "C")


def _encode(array: np.ndarray, actual: dict, dtypes: dict) -> tuple[str, str, bytes, str]:
    matches = [(name, row) for name, row in dtypes.items() if row[0] == actual["dtype"]]
    if len(matches) != 1:
        raise ValueError("capture tensor has an unsupported source dtype")
    name, (_, encoding, width) = matches[0]
    canonical = array.astype(array.dtype.newbyteorder("<"), copy=False)
    if actual["dtype"] == "bf16":
        # The producer's documented BF16 -> f32 storage cast is invertible for
        # non-NaNs with zero low bits. Do not round a general f32 payload.
        if array.dtype.newbyteorder("=") != np.dtype("float32"):
            raise ValueError("captured BF16 requires its exact f32 storage representation")
        bits = np.frombuffer(canonical.tobytes(order="C"), dtype="<u4")
        if np.any(bits & 65535) or np.any((bits & 0x7FFFFFFF) > 0x7F800000):
            raise ValueError("captured BF16 storage is lossy or has unqualified NaN payloads")
        return name, encoding, (bits >> 16).astype("<u2").tobytes(), "exact-bf16-from-f32-storage"
    if array.dtype.newbyteorder("=") != np.dtype(name):
        raise ValueError("capture tensor dtype differs from parsed IR; no precision conversion is selected")
    payload = canonical.tobytes(order="C")
    if len(payload) != prod(actual["shape"]) * width or (actual["dtype"] == "i1" and any(x > 1 for x in payload)):
        raise ValueError("capture tensor storage differs from its logical type")
    return name, encoding, payload, "byte-preserving-little-endian-storage"


def materialize_arguments(
    manifest: dict,
    argument_types: list[dict],
    receipt: dict,
    extra: bytes,
    dtypes: dict,
) -> tuple[dict, bytes, dict]:
    """Produce a consumer manifest and one small safetensors archive.

    Existing model weights retain their bytes. Buffers remain state arguments;
    storing their initializer does not make them immutable parameters.
    """
    if set(manifest) != {str(i) for i in range(len(argument_types))}:
        raise ValueError("capture manifest does not cover the parsed function arguments")
    lifted = receipt.get("lifted_constants", {})
    needs_materialization(manifest, receipt)
    result, tensors, entries, used = {}, {}, [], set()
    try:
        archive = zipfile.ZipFile(BytesIO(extra))
    except zipfile.BadZipFile as exc:
        raise ValueError("capture extra archive is not a valid NPZ") from exc
    with archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("capture extra archive contains duplicate members")
        buffers = {}
        for member in names:
            if member.startswith("buf::") and member.endswith(".npy"):
                key = member[:-4]
                # This is the producer's capture ABI naming rule, not a model
                # heuristic. Reject collisions rather than choosing one leaf.
                buffers.setdefault("b_" + key[5:].replace(".", "_"), []).append(key)
        for index, actual in enumerate(argument_types):
            row = manifest[str(index)]
            if not isinstance(row, dict):
                raise ValueError("capture manifest has a malformed argument row")
            kind, name = row.get("kind"), row.get("name")
            if not isinstance(kind, str):
                raise ValueError("capture manifest has an invalid argument kind")
            if kind in {"input", "constant"} and isinstance(name, str) and name in lifted:
                if set(row) != {"kind", "name"}:
                    raise ValueError("lifted constant has unknown manifest fields")
                key, role = name, "constant"
                used.add(name)
            elif kind == "buffer" and set(row) == {"kind", "name"}:
                if not isinstance(name, str) or len(buffers.get(name, [])) != 1:
                    raise ValueError("capture buffer has no unique initializer in the extra archive")
                key, role = buffers[name][0], "buffer"
            else:
                result[str(index)] = {**row, "file": "weights.safetensors"} if kind in {"param", "buffer"} else row
                continue
            try:
                array = _array(archive, key, actual["shape"])
            except (zipfile.BadZipFile, EOFError) as exc:
                raise ValueError("capture tensor has invalid or truncated NPY storage") from exc
            original_sha = hashlib.sha256(array.tobytes(order="C")).hexdigest()
            if role == "constant":
                declaration = lifted[name]
                if (
                    not isinstance(declaration, dict)
                    or set(declaration) != {"shape", "dtype", "sha256"}
                    or declaration["shape"] != list(array.shape)
                    or declaration["dtype"] != str(array.dtype)
                    or declaration["sha256"] != original_sha
                ):
                    raise ValueError("lifted constant differs from its capture receipt")
            dtype, encoding, data, codec = _encode(array, actual, dtypes)
            tensor = (encoding, actual["shape"], data)
            if name in tensors and tensors[name] != tensor:
                raise ValueError("shared capture tensor has conflicting types or payloads")
            tensors[name] = tensor
            result[str(index)] = {
                "kind": role,
                "weight": name,
                "dtype": dtype,
                "shape": actual["shape"],
                "file": CONSTANT_FILE,
            }
            entries.append(
                {
                    "argument_index": index,
                    "key": key,
                    "role": role,
                    "codec": codec,
                    "source_tensor_sha256": original_sha,
                    "tensor_sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    if used != set(lifted) or not tensors:
        raise ValueError("capture has unbound lifted constants or no declared extra tensors")
    header, payloads, offset = {}, [], 0
    for name, (encoding, shape, data) in tensors.items():
        header[name] = {"dtype": encoding, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payloads.append(data)
        offset += len(data)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    raw = struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads)
    return (
        result,
        raw,
        {
            "schema": "merlin.capture_argument_materialization.v1",
            "extra_npz_sha256": hashlib.sha256(extra).hexdigest(),
            "entries": entries,
        },
    )


# ---- inputs ------------------------------------------------------------------------------------

_COPIED = ("weights.safetensors", "weights.safetensors.manifest.json", "frontend-trace.json")
_SOURCE = "model.mlir"
_STAGED_REQUIRED = frozenset({"program.mlir", "signature.json", *_COPIED})
_STAGED_OPTIONAL = frozenset({"precision-policy.json", "input_order.json", CONSTANT_FILE})


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
            source,
            "weights.safetensors",
            expected_sha256=entry["sha256"],
            typed=bool(weight_manifest),
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
    export_state = None
    if "export_state" in metadata:
        from merlin.compiler.frontend.signature import validate_observation

        export_state = validate_observation(
            metadata["export_state"],
            trace=trace,
            source_sha256=_sha(raw),
            manifest_sha256=_sha(copied["weights.safetensors.manifest.json"]),
        )
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
            receipt,
            _checked_member(source, receipt, "extra.npz"),
            _WEIGHT_DTYPES,
        )
        copied["weights.safetensors.manifest.json"] = json.dumps(weight_manifest, sort_keys=True).encode()
        signature["schema"] = "merlin.compile-signature.v3"
    if weight_manifest:
        signature["arguments"] = _argument_bindings(
            _entry_type(parsed, signature["entry"]),
            signature["inputs"],
            copied,
            materialized=materialized,
        )
    _check_signature(parsed, signature, copied)
    signature_raw = (json.dumps(signature, sort_keys=True, indent=2) + "\n").encode()
    manifest = {
        "schema": "merlin.compile-inputs.v1",
        "scope": "compiler_only_static_capture; source_closure_and_numerical_admission_unverified",
        "source": {
            "capture_receipt_sha256": _sha(receipt_raw),
            "model_mlir_sha256": _sha(raw),
            **({"export_state": export_state} if export_state is not None else {}),
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
    if not isinstance(members, dict) or not _STAGED_REQUIRED <= set(members) <= (_STAGED_REQUIRED | _STAGED_OPTIONAL):
        raise ValueError("compiler input manifest has missing or unknown members")
    if {member.name for member in root.iterdir()} != {"compile-inputs.json", *members}:
        raise ValueError("compiler input directory has missing or undeclared members")
    raw_members = {}
    for name, expected in members.items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)
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
            root,
            name,
            expected_sha256=members[name],
            typed=signature.get("schema") != "merlin.compile-signature.v1",
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
    if "export_state" in source_record:
        from merlin.compiler.frontend.signature import validate_observation

        validate_observation(
            source_record["export_state"],
            trace=json.loads(raw_members["frontend-trace.json"]),
            source_sha256=original_source_sha256,
        )
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
    directory: str | Path,
    staged: StagedCompileInputs,
    *,
    kinds: tuple[str, ...] | None = None,
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
        not isinstance(kinds, tuple)
        or any(not isinstance(kind, str) or kind not in roles for kind in kinds)
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
                    header_parts.append(chunk[: info.payload_start - position])
                for key, parts in portions.items():
                    start, end = ranges[key]
                    a, b = max(start, position), min(end, position + len(chunk))
                    if a < b:
                        parts.append(chunk[a - position : b - position])
                position += len(chunk)
        if hasher.hexdigest() != digest or position != info.file_bytes:
            role = "weights" if file == "weights.safetensors" else "constants"
            raise ValueError(f"staged {role} changed after compiler input validation")
        actual_header, actual_start = read_header(BytesIO(b"".join(header_parts)))
        if actual_header != info.header or actual_start != info.payload_start:
            raise ValueError("tensor archive metadata changed before payload extraction")
        by_weight.update((key, b"".join(parts)) for key, parts in portions.items())
    return {index: by_weight[key] for index, key in indices.items()}


# ---- materialized ------------------------------------------------------------------------------
# Exact input binding for a trusted materializer, separate from M2M receipts.
#
# Only the trusted caller selects this binding after its own source/ABI validator
# has completed. It establishes unchanged materialized bytes, never compiler or
# numerical qualification and never impersonates the original capture issuer.


def _direct_root(root):
    path = Path(root).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("materialized model root must not use symbolic links")
    return path.resolve(strict=True)


@dataclass(frozen=True)
class MaterializedModelInputs:
    root: Path
    identity_bytes: bytes
    validator_identity_bytes: bytes
    source_identity_bytes: bytes

    @classmethod
    def bind(cls, root, *, artifacts, validator_identity, source_identity):
        root = _direct_root(root)
        if not isinstance(validator_identity, dict) or not validator_identity.get("worker_sha256"):
            raise ValueError("trusted model materializer needs its selected validator identity")
        if (
            not isinstance(source_identity, dict)
            or not source_identity.get("content_sha256")
            or not source_identity.get("artifacts")
        ):
            raise ValueError("trusted model materializer needs exact frozen source identities")
        if not isinstance(artifacts, dict) or not {
            "model.mlir",
            "weights.safetensors",
            "weights.safetensors.manifest.json",
            "inputs.npz",
            "input_order.json",
            "golden.npy",
            "goldens.npz",
            "output_order.json",
        } <= set(artifacts):
            raise ValueError("materialized model needs complete source, ABI, and output artifacts")
        bound = cls(
            root,
            json.dumps(artifacts, sort_keys=True).encode(),
            json.dumps(validator_identity, sort_keys=True).encode(),
            json.dumps(source_identity, sort_keys=True).encode(),
        )
        bound.verify(root)
        return bound

    def verify(self, root):
        if type(self) is not MaterializedModelInputs or _direct_root(root) != self.root:
            raise ValueError("materialized model binding names another capture")
        artifacts = json.loads(self.identity_bytes)
        for name, record in artifacts.items():
            if Path(name).name != name or not isinstance(record, dict):
                raise ValueError("invalid materialized model artifact")
            path = self.root / name
            if path.is_symlink() or not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise ValueError(f"materialized model artifact changed: {name}")
            if path.stat().st_size != record.get("size_bytes"):
                raise ValueError(f"materialized model artifact extent changed: {name}")
        return artifacts

    def record(self):
        return {
            "scope": "trusted runtime materialization; no M2M issuer or compiled qualification claim",
            "artifacts": json.loads(self.identity_bytes),
            "validator": json.loads(self.validator_identity_bytes),
            "source": json.loads(self.source_identity_bytes),
        }


# ---- bundle ------------------------------------------------------------------------------------
# Capture bundles for the RVV whole-model lane, and the scalar datatype a bundle's IR carries.
#
# ``_bundle_dir`` resolves the bundle ``merlin-compile`` compiles, ``_capture_python`` picks the interpreter a
# model is captured with, and ``ir_scalar_dtype`` reads the element type the captured ``model.mlir``
# actually carries -- which, not the bundle's name, decides the scalar datapath.


def _bundle_dir(workload: str, dtype: str):
    """The capture bundle `merlin-compile` uses, preferring the FULL-FIDELITY recapture.

    Delegates to ``baselines.bundle.resolve``, which prefers ``<w>_<dtype>_full`` (the real/native
    architecture) and falls back to ``<w>_<dtype>_consistent``. This used to hard-code the
    ``_consistent`` suffix, which for tiny_llama is a 2-layer RANDOM-INIT stand-in rather than the
    real 22-layer TinyLlama-1.1B — so `merlin-compile` silently compiled a toy while every baseline
    arm (which already goes through ``resolve``) compiled the real model.
    """
    from merlin.capture.bundle import resolve

    return resolve(workload, dtype).root


def ir_scalar_dtype(bundle: "Path") -> str | None:  # noqa: UP037 -- Preserve the original callable annotation.
    """The dtype the bundle's IR ACTUALLY carries, read off ``model.mlir`` — the fact that decides which
    scalar/RVV datapath is correct.

    A bundle's NAME states how the model was quantized; it does not state what the compiled IR contains.
    A weight-only fake-quant capture stores fp8/int8 weights but emits f32 tensors end to end, so the
    correct scalar package for it is the f32 one — choosing by the name instead asks for a datapath the IR
    has no operands for. Returns ``None`` when no known element type dominates, so the caller fails closed
    rather than guessing.

    Selection is by PRESENCE of the narrowest datapath, not by majority. Every weight-only capture is
    mostly f32 by count -- an int8 bundle measures 46 i8 tensors against 1190 f32 ones -- so a majority vote
    hands back f32 and silently drops the int8 datapath the i8 operands require. One tensor of a narrow type
    means the narrow datapath is needed; only a bundle with none of them is an f32 model.

    Structural, not pattern-matched: it counts occurrences of each known tensor element-type spelling.
    (No regex — see the repo's no-regex rule.)"""
    mlir = bundle / "model.mlir"
    if not mlir.is_file():
        return None
    text = mlir.read_text(encoding="utf-8", errors="replace")
    # spelling in the IR -> the --dtype token that selects its datapath, narrowest datapath first
    present = [tok for tok in _IR_ELEMENT_ORDER if text.count(f"x{_IR_ELEMENT_SPELLING[tok]}>")]
    return present[0] if present else None


def _capture_python(m2m_dir: Path, workload: str) -> Path:
    """The interpreter to capture ``workload`` with: its own venv when its capture.toml names one.

    A workload's upstream stack is pinned per model (model2MLIR's `capture.toml [venv]`), so the
    capture must not run under merlin's interpreter just because that is what is executing.
    Falls back to model2MLIR's own venv, then to this interpreter, and never fails here — the
    caller surfaces a capture failure with the worker's output attached.
    """
    toml = m2m_dir / "workloads" / workload / "capture.toml"
    if toml.is_file():
        try:
            import tomllib

            cfg = tomllib.loads(toml.read_text())
        except Exception:  # noqa: BLE001
            cfg = {}
        venv = cfg.get("venv")
        if venv:
            cand = Path(venv)
            if not cand.is_absolute():
                cand = m2m_dir / "workloads" / workload / venv
            if (cand / "bin" / "python").is_file():
                return cand / "bin" / "python"
    fallback = m2m_dir / ".venv" / "bin" / "python"
    return fallback if fallback.is_file() else Path(sys.executable)


#: MLIR tensor element-type spelling for each ``--dtype`` token, used to read a bundle's ACTUAL IR dtype
#: back off ``model.mlir`` (:func:`ir_scalar_dtype`). Keys must stay a subset of ``_DTYPE_STRATEGY``.
_IR_ELEMENT_SPELLING = {"int8": "i8", "fp32": "f32", "fp16": "f16", "bf16": "bf16"}
#: narrowest datapath first — the first spelling PRESENT in the IR decides (see :func:`ir_scalar_dtype`).
_IR_ELEMENT_ORDER = ("int8", "fp16", "bf16", "fp32")
