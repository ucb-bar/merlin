"""Data-format-only caller bindings for explicit logical/physical storage contracts.

No operation interpretation, casting, quantization, or tensor algebra occurs here. The
caller supplies its dtype's storage words after resolving logical input values; the
encoding only relocates those words. An explicit map is complete, never a sparse hint.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any

from merlin.analysis.programs.storage_encoding import GroupedAxesStorage, _element_bytes
from merlin.common import digest as _mjson
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
from merlin.compiler.frontend.signature import ArgumentIdentityBridge
from merlin.runtime.engine.command_buffer import Tensor

# ---- storage_binding ---------------------------------------------------------------------------


class StoragePrepackRequired(ValueError):
    """An exact, inspectable obligation; a candidate's weight role is not authorization."""

    def __init__(self, obligations: Mapping[str, Any]):
        self.storage_obligations = dict(obligations)
        super().__init__(
            "explicit caller input requires_prepack without host immutable-input authorization: "
            + ", ".join(obligations)
        )


@dataclass(frozen=True)
class StorageBinding:
    encoding: GroupedAxesStorage
    access: str
    logical_values: tuple[Any, ...] | None
    prepack_authorization: Any = None

    @property
    def requires_prepack(self) -> bool:
        """Whether non-unit logical axes change order, rather than merely adding padding."""
        return self.encoding.reorders_logical_elements

    def setup_evidence(self) -> dict[str, Any]:
        return {
            "schema": "caller_storage_setup_v1",
            "encoding": self.encoding.to_dict(),
            "logical_copy_elements": prod(self.encoding.logical_shape),
            "requires_prepack": self.requires_prepack and self.access != "write",
            "packing_inside_compute_roi": False,
            "prepack_authorization": (
                self.prepack_authorization.to_evidence()
                if self.prepack_authorization is not None
                else "UNPROVEN"
                if self.requires_prepack and self.access != "write"
                else "not_required"
            ),
            "scope": "format setup/readback only; no model arithmetic or timing qualification",
        }

    def pack_words(self, words: Sequence[int]) -> list[int]:
        """Copy already encoded scalar storage words; retain each word unchanged."""
        encoding = self.encoding
        if len(words) != prod(encoding.logical_shape):
            raise ValueError("logical storage-word count disagrees with the encoding")
        words = tuple(words)
        if self.prepack_authorization is not None:
            width = _element_bytes(encoding.dtype)
            if width not in (1, 2, 4, 8) or any(type(word) is not int for word in words):
                raise ValueError("authorized prepack requires exact whole-byte integer storage words")
            modulus = 1 << (8 * width)
            if any(not -(modulus // 2) <= word < modulus for word in words):
                raise ValueError("authorized prepack storage word exceeds declared dtype width")
            payload = b"".join((word % modulus).to_bytes(width, "little") for word in words)
            self.prepack_authorization.validate_payload(payload)
        result = [0] * encoding.storage_elements
        terms = encoding.linear_index_terms()
        for linear, word in enumerate(words):
            offset = encoding.offset_elements + sum((linear // d % n) * s for d, n, s in terms)
            result[offset] = word
        return result

    def logical_offset_terms(self) -> tuple[tuple[int, int, int], ...]:
        """(row-major linear divisor, logical extent, physical stride) for readback."""
        return self.encoding.linear_index_terms()


def _logical_values(value: Any, shape: tuple[int, ...], dtype: str, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, Tensor):
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"logical input {name!r} Tensor shape/dtype disagrees with storage encoding")
        return tuple(value.data)

    def visit(node, dims):
        if not dims:
            if isinstance(node, (list, tuple, Mapping)):
                raise ValueError(f"logical input {name!r} has a non-scalar leaf")
            return [node]
        if not isinstance(node, (list, tuple)) or len(node) != dims[0]:
            raise ValueError(f"logical input {name!r} shape disagrees with storage encoding")
        return [item for child in node for item in visit(child, dims[1:])]

    return tuple(visit(value, shape))


def resolve_storage_bindings(
    cb: Mapping[str, Any],
    inputs: Mapping[str, Any] | None = None,
    *,
    max_storage_bytes: int,
    reference_only_allow_prepack: bool = False,
    prepack_authorizations: Mapping[str, Any] | None = None,
    describe_only: bool = False,
) -> dict[str, StorageBinding] | None:
    """Validate all storage before materializing inputs, or return None for the legacy ABI.

    ``max_storage_bytes`` is a host resource bound, not a claim about target capacity.
    Physical tensor descriptors retain unpadded extents; allocated strides/offset/padding
    come exclusively from the checked explicit encoding.

    ``reference_only_allow_prepack`` permits testing the data-format copy itself; it is
    NOT an authorization to exclude candidate-selected input work from a timed invocation.
    ``describe_only`` validates the same allocation map without creating even synthetic
    input values. It reports a format contract, not an executable caller or prepack grant.
    ``prepack_authorizations`` is an out-of-band host grant, never command-buffer
    metadata. Binding is checked here; the exact initializer words are checked by
    ``pack_words`` before relocation. Missing authorized inputs never materialize
    from deterministic names.
    """
    params = cb.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("whole-program params must be a mapping")
    if "storage_encodings" not in params:
        if prepack_authorizations is not None:
            raise ValueError("host prepack authorization requires explicit storage encodings")
        return None
    if type(max_storage_bytes) is not int or max_storage_bytes <= 0:
        raise ValueError("explicit storage requires a positive host allocation budget")
    # Derived arithmetic belongs in the submitted compiler, never in this format-only path.
    from merlin.runtime.engine.command_buffer import DERIVATION_RECIPE_KEYS

    if any(params.get(key) for key in DERIVATION_RECIPE_KEYS):
        raise ValueError("explicit storage cannot move tensor derivation into the caller")
    records, tensors, abi = params["storage_encodings"], cb.get("tensors"), cb.get("kernel_abi")
    if not isinstance(records, Mapping) or not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("explicit storage requires a nonempty complete tensor map")
    if not isinstance(abi, Mapping) or abi.get("kind") != "whole_program":
        raise ValueError("explicit storage caller requires the whole_program pointer ABI")
    args, outputs = abi.get("args"), abi.get("outputs")
    if not isinstance(args, list) or not isinstance(outputs, list) or not outputs:
        raise ValueError("explicit storage requires declared arguments and outputs")
    access = {}
    for arg in args:
        if (
            not isinstance(arg, Mapping)
            or not isinstance(arg.get("tensor"), str)
            or arg.get("access") not in ("read", "write", "readwrite")
            or arg["tensor"] in access
        ):
            raise ValueError("explicit storage has malformed or duplicate pointer arguments")
        access[arg["tensor"]] = arg["access"]
    if set(records) != set(tensors) or set(access) != set(tensors):
        raise ValueError("storage encodings must cover exactly every tensor and ABI argument")
    if any(not isinstance(name, str) or name not in access or access[name] == "read" for name in outputs) or len(
        outputs
    ) != len(set(outputs)):
        raise ValueError("explicit storage outputs must name distinct writable arguments")
    encodings, allocated = {}, 0
    for name, spec in tensors.items():
        if not isinstance(spec, Mapping):
            raise ValueError("explicit storage requires typed tensor descriptors")
        encoding = GroupedAxesStorage.from_dict(records[name])
        if (
            not isinstance(spec.get("shape"), list)
            or any(type(dim) is not int or dim <= 0 for dim in spec["shape"])
            or spec["shape"] != list(encoding.physical_shape)
            or spec.get("dtype") != encoding.dtype
        ):
            raise ValueError(f"tensor {name!r} physical shape/dtype disagrees with storage encoding")
        if (
            spec.get("role") in ("input", "weight", "bias", "scale")
            and access[name] == "write"
            or spec.get("role") in ("output", "intermediate")
            and access[name] == "read"
        ):
            raise ValueError("tensor role contradicts storage argument access")
        allocated += encoding.storage_elements * _element_bytes(encoding.dtype)
        if allocated > max_storage_bytes:
            raise ValueError("explicit storage exceeds the host aggregate allocation budget")
        encodings[name] = encoding
    prepack = {
        name: StorageBinding(encoding, access[name], None).setup_evidence()
        for name, encoding in encodings.items()
        if access[name] != "write" and StorageBinding(encoding, access[name], None).requires_prepack
    }
    provided = {} if inputs is None else inputs
    if not isinstance(provided, Mapping) or set(provided) - set(tensors):
        raise ValueError("explicit logical inputs must name declared tensors")
    if describe_only and provided:
        raise ValueError("layout description cannot consume input values")
    authorizations = {} if prepack_authorizations is None else prepack_authorizations
    if not isinstance(authorizations, Mapping) or set(authorizations) - set(tensors):
        raise ValueError("host prepack authorizations must name declared tensors")
    if authorizations:
        for name, authorization in authorizations.items():
            if type(authorization) is not HostPrepackAuthorization:
                raise ValueError("prepack requires an exact host authorization object, not candidate metadata")
            if access[name] != "read" or name not in provided:
                raise ValueError("authorized prepack requires an explicit read-only logical input")
            authorization.validate_binding(cb, name, encodings[name])
    missing = {name: obligation for name, obligation in prepack.items() if name not in authorizations}
    if missing and not reference_only_allow_prepack and not describe_only:
        raise StoragePrepackRequired(missing)
    result = {}
    for name, encoding in encodings.items():
        values = None
        if access[name] == "write":
            if name in provided:
                raise ValueError("write-only output cannot be initialized from supplied inputs")
        elif name in provided:
            values = _logical_values(provided[name], encoding.logical_shape, encoding.dtype, name=name)
        elif not describe_only:
            values = tuple(Tensor.deterministic(name, encoding.logical_shape, encoding.dtype).data)
        result[name] = StorageBinding(encoding, access[name], values, authorizations.get(name))
    return result


# ---- captured_constants ------------------------------------------------------------------------
# Read one pinned captured state tensor without executing or packing a model.
#
# This verifies capture bytes, not permission to prepack a source argument. The
# caller must independently prove the raw-source/manifest binding, preservation
# through normalization, and read-only use. Candidate role/recipe claims are not
# accepted here. No dtype conversion or interpretation of tensor bytes occurs.

# File-format storage types, not accelerator facts. Keep this explicit: unknown
# or sub-byte formats require their own byte-preserving contract, never a cast.
_TYPES = {
    "I8": ("i8", "int8", 1),
    "I16": ("i16", "int16", 2),
    "I32": ("i32", "int32", 4),
    "I64": ("i64", "int64", 8),
    "U8": ("ui8", "uint8", 1),
    "U16": ("ui16", "uint16", 2),
    "U32": ("ui32", "uint32", 4),
    "U64": ("ui64", "uint64", 8),
    "F16": ("f16", "float16", 2),
    "BF16": ("bf16", "bfloat16", 2),
    "F32": ("f32", "float32", 4),
    "F64": ("f64", "float64", 8),
    "BOOL": ("i1", "bool", 1),
}
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_HEADER_BYTES = 16 * 1024 * 1024
_CHUNK_BYTES = 64 * 1024


def _digest_pin(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("capture pins require lowercase SHA-256 hex digests")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate capture JSON key: {key!r}")
        result[key] = value
    return result


def _json_object(payload: bytes) -> dict:
    def invalid_constant(value):
        raise ValueError(f"non-finite capture JSON value: {value}")

    try:
        value = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("malformed capture JSON") from error
    if not isinstance(value, dict):
        raise ValueError("capture JSON root must be an object")
    return value


def _shape(value) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(type(dim) is not int or dim < 0 for dim in value):
        raise ValueError("capture shape requires nonnegative integer extents")
    return tuple(value)


@dataclass(frozen=True)
class CapturedConstant:
    """An immutable byte snapshot. Its construction alone grants no authority."""

    entry_argument_index: int
    manifest_path: str
    manifest_sha256: str
    safetensors_path: str
    safetensors_sha256: str
    tensor_key: str
    manifest_kind: str
    source_shape: tuple[int, ...]
    source_dtype: str
    storage_dtype: str
    header_sha256: str
    file_offset_bytes: int
    payload_sha256: str
    logical_payload: bytes

    def to_evidence(self) -> dict:
        return {
            "schema": "verified_capture_constant_bytes_v1",
            "entry_argument_index": self.entry_argument_index,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "safetensors_path": self.safetensors_path,
            "safetensors_sha256": self.safetensors_sha256,
            "tensor_key": self.tensor_key,
            "manifest_kind": self.manifest_kind,
            "source_shape": list(self.source_shape),
            "source_dtype": self.source_dtype,
            "storage_dtype": self.storage_dtype,
            "header_sha256": self.header_sha256,
            "file_offset_bytes": self.file_offset_bytes,
            "payload_bytes": len(self.logical_payload),
            "payload_sha256": self.payload_sha256,
            "scope": "pinned captured state bytes only; source binding and prepack authorization unproven",
            "prepack_authorized": False,
        }


def verify_capture_constant(
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    safetensors_path: str | Path,
    safetensors_sha256: str,
    entry_argument_index: int,
    source_shape: Sequence[int],
    source_dtype: str,
    max_payload_bytes: int,
) -> CapturedConstant:
    """Verify two pinned files and load only one explicitly bounded tensor.

    Every tensor header is checked for a contiguous, nonoverlapping valid file
    partition. The full blob is hashed in chunks; only the selected payload is
    retained. The selected bytes come from that same hashing pass, preventing a
    separate post-verification read from silently observing different bytes.
    """
    _digest_pin(manifest_sha256)
    _digest_pin(safetensors_sha256)
    if type(entry_argument_index) is not int or entry_argument_index < 0:
        raise ValueError("entry argument index must be a nonnegative integer")
    if type(max_payload_bytes) is not int or max_payload_bytes <= 0:
        raise ValueError("selected payload requires an explicit positive byte budget")
    wanted_shape = _shape(source_shape)
    manifest_path, safetensors_path = Path(manifest_path), Path(safetensors_path)
    with manifest_path.open("rb") as stream:
        manifest_bytes = stream.read(_MAX_MANIFEST_BYTES + 1)
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("capture manifest exceeds bounded JSON size")
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError("capture manifest SHA-256 mismatch")
    manifest = _json_object(manifest_bytes)
    if any(not key.isascii() or not key.isdecimal() or str(int(key)) != key for key in manifest):
        raise ValueError("capture manifest requires unique canonical argument indices")
    entry = _entry(manifest, entry_argument_index)
    if _shape(entry.get("shape")) != wanted_shape:
        raise ValueError("manifest shape differs from source argument")
    key = entry["weight"]
    with safetensors_path.open("rb") as stream:
        length_bytes, header_bytes, header, file_length, payload_base = _layout(stream)
        tensor = header.get(key)
        if key == "__metadata__" or not isinstance(tensor, dict):
            raise ValueError("manifest weight key is absent from safetensors")
        dtype = tensor["dtype"]
        normalized_dtype, manifest_dtype, _ = _TYPES[dtype]
        if (
            source_dtype != normalized_dtype
            or entry.get("dtype") != manifest_dtype
            or _shape(tensor["shape"]) != wanted_shape
        ):
            raise ValueError("capture storage type/shape differs from source or manifest")
        begin, end = tensor["data_offsets"]
        if end - begin > max_payload_bytes:
            raise ValueError("selected capture payload exceeds explicit byte budget")
        start, stop = payload_base + begin, payload_base + end
        # Capture the selected bytes from the exact stream being hashed. Also
        # compare the parsed header with the hashed header to reject a changed
        # preliminary view, rather than trusting file stat timestamps.
        stream.seek(0)
        digest = hashlib.sha256()
        prefix = length_bytes + header_bytes
        payload = bytearray()
        position = 0
        while chunk := stream.read(_CHUNK_BYTES):
            digest.update(chunk)
            prefix_end = min(len(prefix), position + len(chunk))
            if position < prefix_end and chunk[: prefix_end - position] != prefix[position:prefix_end]:
                raise ValueError("safetensors header changed during verification")
            lo, hi = max(start, position), min(stop, position + len(chunk))
            if lo < hi:
                payload.extend(chunk[lo - position : hi - position])
            position += len(chunk)
        if position != file_length or digest.hexdigest() != safetensors_sha256 or len(payload) != end - begin:
            raise ValueError("safetensors SHA-256 or file length mismatch")
    logical_payload = bytes(payload)
    if entry["kind"] == "constant" and hashlib.sha256(logical_payload).hexdigest() != entry["sha256"]:
        raise ValueError("captured constant payload differs from its source digest")
    return CapturedConstant(
        entry_argument_index,
        str(manifest_path.resolve()),
        manifest_sha256,
        str(safetensors_path.resolve()),
        safetensors_sha256,
        key,
        entry["kind"],
        wanted_shape,
        source_dtype,
        dtype,
        hashlib.sha256(header_bytes).hexdigest(),
        start,
        hashlib.sha256(logical_payload).hexdigest(),
        logical_payload,
    )


# ---- prepack_authority -------------------------------------------------------------------------
# Out-of-band host authority for prepacking one exact captured constant payload.
#
# The objects here are created inside the trusted host, after capture-byte and
# normalization-identity verification. They are never deserialized from candidate
# JSON, inferred from a tensor role, or accepted as a performance certificate.

_digest = _mjson.canonical_sha256


@dataclass(frozen=True)
class HostPrepackAuthorization:
    tensor: str
    command_buffer_digest: str
    encoding: GroupedAxesStorage
    constant: CapturedConstant
    argument_bridge: ArgumentIdentityBridge

    def validate_binding(self, cb: Mapping, tensor: str, encoding: GroupedAxesStorage) -> None:
        if tensor != self.tensor or _digest(cb) != self.command_buffer_digest or encoding != self.encoding:
            raise ValueError("prepack authority does not bind this command buffer/tensor/encoding")
        abi = cb.get("kernel_abi")
        args = abi.get("args") if isinstance(abi, Mapping) else None
        if not isinstance(args, list):
            raise ValueError("authorized prepacking requires the declared readonly pointer ABI")
        matching = [arg for arg in args if isinstance(arg, Mapping) and arg.get("tensor") == tensor]
        if len(matching) != 1 or matching[0].get("access") != "read":
            raise ValueError("authorized prepacking requires an exactly readonly input")

    def validate_payload(self, actual_bytes: bytes) -> None:
        if type(actual_bytes) is not bytes or actual_bytes != self.constant.logical_payload:
            raise ValueError("actual initializer differs from the authorized captured constant payload")

    def to_evidence(self) -> dict:
        return {
            "schema": "host_capture_prepack_authorization_v1",
            "tensor": self.tensor,
            "command_buffer_digest": self.command_buffer_digest,
            "encoding_digest": _digest(self.encoding.to_dict()),
            "capture": self.constant.to_evidence(),
            "argument_bridge": self.argument_bridge.to_evidence(),
            "prepack_authorized": True,
            "actual_initializer_payload": "must be checked again by the consuming caller",
            "scope": "offline format preparation of this fixed captured payload only",
            "numerical_equivalence": "UNPROVEN",
            "performance_promotion": False,
        }


def authorize_capture_prepack(
    *,
    constant: CapturedConstant,
    bridge: ArgumentIdentityBridge,
    raw_source_text: str,
    normalized_source_text: str,
    command_buffer: Mapping,
    tensor: str,
) -> HostPrepackAuthorization:
    """Join host evidence to a particular source argument and compiler-selected ABI.

    A changed capture, normalization, tensor binding, encoding or actual input
    invalidates this grant. Callback selection and evidence creation belong to
    the trusted host policy; callers cannot submit serialized grants in a CB.
    """
    from xdsl.dialects.builtin import TensorType

    if type(constant) is not CapturedConstant or type(bridge) is not ArgumentIdentityBridge:
        raise ValueError("prepack authorization requires host-created typed capture and identity evidence")
    if (
        hashlib.sha256(raw_source_text.encode()).hexdigest() != bridge.source_sha256
        or hashlib.sha256(normalized_source_text.encode()).hexdigest() != bridge.normalized_sha256
    ):
        raise ValueError("source bytes differ from the verified argument identity bridge")
    if hashlib.sha256(constant.logical_payload).hexdigest() != constant.payload_sha256:
        raise ValueError("capture evidence no longer binds its immutable byte snapshot")
    raw = parse_mlir_text(raw_source_text)
    weights_file = getattr(raw.attributes.get("prov.weights_file"), "data", None)
    if not isinstance(weights_file, str) or Path(weights_file).resolve() != Path(constant.safetensors_path).resolve():
        raise ValueError("raw capture source does not bind this verified weights file")
    normalized = parse_mlir_text(normalized_source_text)
    entries = [op for op in normalized.body.block.ops if op.name == "func.func" and op.sym_name.data == bridge.entry]
    if len(entries) != 1 or len(entries[0].body.blocks) != 1:
        raise ValueError("normalized source entry is not the verified identity boundary")
    arguments = tuple(entries[0].body.block.args)
    index = constant.entry_argument_index
    if (
        tuple(str(arg.type) for arg in arguments) != bridge.argument_types
        or type(index) is not int
        or not 0 <= index < len(arguments)
    ):
        raise ValueError("captured argument index/types disagree with normalization bridge")
    value_type = arguments[index].type
    if (
        not isinstance(value_type, TensorType)
        or tuple(value_type.get_shape()) != constant.source_shape
        or str(value_type.get_element_type()) != constant.source_dtype
    ):
        raise ValueError("captured constant type differs from the actual normalized argument")
    params = command_buffer.get("params")
    plan = params.get("global_program_plan") if isinstance(params, Mapping) else None
    encodings = params.get("storage_encodings") if isinstance(params, Mapping) else None
    if (
        not isinstance(plan, Mapping)
        or plan.get("source_sha256") != bridge.normalized_sha256
        or not isinstance(encodings, Mapping)
        or tensor not in encodings
    ):
        raise ValueError("compiler plan/encoding is not bound to the normalized source")
    bindings = plan.get("entry_bindings")
    if (
        not isinstance(bindings, list)
        or len(bindings) != len(arguments)
        or bindings[index] != tensor
        or bindings.count(tensor) != 1
    ):
        raise ValueError("captured argument is not uniquely bound to the requested ABI tensor")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("prepacking requires explicit source-task read/write accounting")
    for task in tasks:
        if not isinstance(task, Mapping) or not isinstance(task.get("writes"), list):
            raise ValueError("prepacking requires explicit source-task writes")
        if tensor in task["writes"]:
            raise ValueError("compiler plan writes the purported immutable input")
    encoding = GroupedAxesStorage.from_dict(encodings[tensor])
    if encoding.logical_shape != constant.source_shape or encoding.dtype != constant.source_dtype:
        raise ValueError("prepacking encoding does not preserve the captured logical type")
    specs = command_buffer.get("tensors")
    spec = specs.get(tensor) if isinstance(specs, Mapping) else None
    if (
        not isinstance(spec, Mapping)
        or spec.get("shape") != list(encoding.physical_shape)
        or spec.get("dtype") != encoding.dtype
    ):
        raise ValueError("prepacking encoding differs from the physical ABI tensor")
    authority = HostPrepackAuthorization(tensor, _digest(command_buffer), encoding, constant, bridge)
    authority.validate_binding(command_buffer, tensor, encoding)
    return authority


def _layout(stream):
    """One bounded safetensors header and its complete contiguous partition."""
    length_bytes = stream.read(8)
    if len(length_bytes) != 8:
        raise ValueError("truncated safetensors length prefix")
    header_length = struct.unpack("<Q", length_bytes)[0]
    if not 2 <= header_length <= _MAX_HEADER_BYTES:
        raise ValueError("safetensors header exceeds bounded JSON size")
    header_bytes = stream.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("truncated safetensors header")
    header = _json_object(header_bytes)
    file_length, payload_base = stream.seek(0, 2), 8 + header_length
    ranges = []
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or any(not isinstance(v, str) for v in tensor.values()):
                raise ValueError("invalid safetensors metadata")
            continue
        if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("malformed safetensors tensor header")
        dtype = tensor["dtype"]
        if not isinstance(dtype, str) or dtype not in _TYPES:
            raise ValueError("unsupported safetensors storage dtype")
        shape, offsets = _shape(tensor["shape"]), tensor["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError("malformed safetensors data offsets")
        begin, end = offsets
        if not 0 <= begin <= end <= file_length - payload_base:
            raise ValueError("safetensors data offsets outside file bounds")
        if end - begin != prod(shape) * _TYPES[dtype][2]:
            raise ValueError("safetensors shape/dtype byte length mismatch")
        ranges.append((begin, end))
    cursor = 0
    for begin, end in sorted(ranges):
        if begin != cursor:
            raise ValueError("safetensors payload overlaps or contains an unindexed hole")
        cursor = end
    if cursor != file_length - payload_base:
        raise ValueError("safetensors payload contains unindexed trailing bytes")
    return length_bytes, header_bytes, header, file_length, payload_base


def _entry(manifest, index):
    entry = manifest.get(str(index))
    if (
        not isinstance(entry, dict)
        or entry.get("kind") not in ("param", "buffer", "constant")
        or "error" in entry
        or entry.get("stub", False) is not False
        or not isinstance(entry.get("weight"), str)
        or not entry["weight"]
    ):
        raise ValueError("argument is missing a concrete captured parameter/buffer/constant")
    if entry["kind"] == "constant":
        if any(not isinstance(entry.get(key), str) or not entry[key] for key in ("name", "source_target", "sha256")):
            raise ValueError("captured constant lacks its exported source binding")
        _digest_pin(entry["sha256"])
    return entry


@dataclass(frozen=True)
class CaptureConstantTable:
    """Reuse already pinned immutable compiler bytes for several source constants.

    The streaming single-tensor reader remains bounded independently. Compiler
    staging already owns the full weight bytes: this table validates that blob
    once instead of rehashing a large model for every small lifted constant.
    """

    manifest_bytes: bytes = field(repr=False)
    safetensors_bytes: bytes = field(repr=False)
    manifest_sha256: str
    safetensors_sha256: str
    manifest_path: str = "weights.safetensors.manifest.json"
    safetensors_path: str = "weights.safetensors"
    _manifest: dict = field(init=False, repr=False)
    _layout: tuple = field(init=False, repr=False)

    def __post_init__(self):
        _digest_pin(self.manifest_sha256)
        _digest_pin(self.safetensors_sha256)
        if not isinstance(self.manifest_bytes, bytes) or not isinstance(self.safetensors_bytes, bytes):
            raise ValueError("compiler constant table requires immutable bytes")
        if len(self.manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise ValueError("capture manifest exceeds bounded JSON size")
        if (
            hashlib.sha256(self.manifest_bytes).hexdigest() != self.manifest_sha256
            or hashlib.sha256(self.safetensors_bytes).hexdigest() != self.safetensors_sha256
        ):
            raise ValueError("capture table SHA-256 mismatch")
        manifest = _json_object(self.manifest_bytes)
        if any(not key.isascii() or not key.isdecimal() or str(int(key)) != key for key in manifest):
            raise ValueError("capture manifest requires unique canonical argument indices")
        with io.BytesIO(self.safetensors_bytes) as stream:
            layout = _layout(stream)
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(self, "_layout", layout)

    def read(self, *, entry_argument_index, source_shape, source_dtype, max_payload_bytes):
        if type(entry_argument_index) is not int or entry_argument_index < 0:
            raise ValueError("entry argument index must be a nonnegative integer")
        if type(max_payload_bytes) is not int or max_payload_bytes <= 0:
            raise ValueError("selected payload requires an explicit positive byte budget")
        entry = _entry(self._manifest, entry_argument_index)
        wanted_shape = _shape(source_shape)
        _, header_bytes, header, _, payload_base = self._layout
        key = entry["weight"]
        tensor = header.get(key)
        if key == "__metadata__" or not isinstance(tensor, dict):
            raise ValueError("manifest weight key is absent from safetensors")
        dtype = tensor["dtype"]
        normalized, manifest_dtype, _ = _TYPES[dtype]
        if (
            _shape(entry.get("shape")) != wanted_shape
            or _shape(tensor["shape"]) != wanted_shape
            or source_dtype != normalized
            or entry.get("dtype") != manifest_dtype
        ):
            raise ValueError("capture storage type/shape differs from source or manifest")
        begin, end = tensor["data_offsets"]
        if end - begin > max_payload_bytes:
            raise ValueError("selected capture payload exceeds explicit byte budget")
        start = payload_base + begin
        payload = self.safetensors_bytes[start : payload_base + end]
        digest = hashlib.sha256(payload).hexdigest()
        if entry["kind"] == "constant" and digest != entry["sha256"]:
            raise ValueError("captured constant payload differs from its source digest")
        return CapturedConstant(
            entry_argument_index,
            self.manifest_path,
            self.manifest_sha256,
            self.safetensors_path,
            self.safetensors_sha256,
            key,
            entry["kind"],
            wanted_shape,
            source_dtype,
            dtype,
            hashlib.sha256(header_bytes).hexdigest(),
            start,
            digest,
            payload,
        )
