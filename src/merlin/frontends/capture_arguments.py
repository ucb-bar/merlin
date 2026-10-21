"""Normalize declared capture constants and buffer initializers for compilation.

The capture producer stores these leaves in extra.npz. Only receipt-declared
constants and manifest-declared buffers are selected. Runtime inputs, outputs
and unrelated archive members never become compiler tensor arguments.
"""

import hashlib
import json
import struct
import zipfile
from io import BytesIO
from math import prod

import numpy as np

CONSTANT_FILE = "capture-constants.safetensors"


def needs_materialization(manifest: dict, receipt: dict) -> bool:
    lifted = receipt.get("lifted_constants", {})
    names = (receipt.get("materialized_abi") or {}).get("lifted_constants", [])
    if (not isinstance(lifted, dict) or not isinstance(names, list)
            or any(not isinstance(name, str) or not name.startswith("c_lifted_tensor_") for name in lifted)
            or names != sorted(lifted)):
        raise ValueError("capture has unbound lifted constants or an invalid constant declaration")
    return bool(lifted) or any(isinstance(row, dict) and (
        row.get("kind") == "constant" or row.get("kind") == "buffer" and set(row) == {"kind", "name"}
    ) for row in manifest.values())


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
    manifest: dict, argument_types: list[dict], receipt: dict, extra: bytes, dtypes: dict,
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
                if (not isinstance(declaration, dict) or set(declaration) != {"shape", "dtype", "sha256"}
                        or declaration["shape"] != list(array.shape) or declaration["dtype"] != str(array.dtype)
                        or declaration["sha256"] != original_sha):
                    raise ValueError("lifted constant differs from its capture receipt")
            dtype, encoding, data, codec = _encode(array, actual, dtypes)
            tensor = (encoding, actual["shape"], data)
            if name in tensors and tensors[name] != tensor:
                raise ValueError("shared capture tensor has conflicting types or payloads")
            tensors[name] = tensor
            result[str(index)] = {"kind": role, "weight": name, "dtype": dtype,
                                  "shape": actual["shape"], "file": CONSTANT_FILE}
            entries.append({"argument_index": index, "key": key, "role": role, "codec": codec,
                            "source_tensor_sha256": original_sha, "tensor_sha256": hashlib.sha256(data).hexdigest()})
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
    return result, raw, {"schema": "merlin.capture_argument_materialization.v1",
                         "extra_npz_sha256": hashlib.sha256(extra).hexdigest(), "entries": entries}
