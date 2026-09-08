#!/usr/bin/env python3
"""Rebuild the copied q535 payload for this compiler ABI, including OIHW->HWIO."""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path

import numpy as np


ARTIFACT = Path(__file__).resolve().parents[1]
BUNDLE = ARTIFACT / "validation/full_spike_candidate"
CAPTURE = Path("/scratch/agustin/projects/oscar-merlin/out/artifacts/recaptures/"
               "resnet50_pt2e_w8a8_universal_dog_direct_20260906")
DTYPE = {"i1": np.dtype("uint8"), "i8": np.dtype("int8"),
         "i16": np.dtype("int16"), "i32": np.dtype("int32"),
         "i64": np.dtype("int64"), "f32": np.dtype("float32")}
SAFETENSOR_DTYPE = {
    "BOOL": np.dtype("bool"), "I8": np.dtype("int8"),
    "I16": np.dtype("int16"), "I32": np.dtype("int32"),
    "I64": np.dtype("int64"), "U8": np.dtype("uint8"),
    "U16": np.dtype("uint16"), "U32": np.dtype("uint32"),
    "U64": np.dtype("uint64"), "F16": np.dtype("float16"),
    "F32": np.dtype("float32"), "F64": np.dtype("float64"),
}


class SafeTensorReader:
    """Small dependency-free reader for the capture's uncompressed tensors."""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            header_bytes = struct.unpack("<Q", stream.read(8))[0]
            self.header = json.loads(stream.read(header_bytes))
        self.data_start = 8 + header_bytes

    def get_tensor(self, name):
        row = self.header[name]
        begin, end = map(int, row["data_offsets"])
        dtype = SAFETENSOR_DTYPE[row["dtype"]]
        count = math.prod(map(int, row["shape"]))
        array = np.memmap(self.path, mode="r", dtype=dtype,
                          offset=self.data_start + begin, shape=(count,))
        if array.nbytes != end - begin:
            raise ValueError(f"{name}: safetensor byte extent mismatch")
        return np.asarray(array).reshape(row["shape"])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def align(value, amount=64):
    return (int(value) + amount - 1) // amount * amount


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def physical_nbytes(name, tensor, encodings):
    dtype = DTYPE[tensor["dtype"]]
    encoding = encodings.get(name)
    if encoding:
        return int(encoding["storage_elements"]) * dtype.itemsize
    shape = list(map(int, tensor["shape"]))
    if not shape:
        return dtype.itemsize
    return math.prod(shape[:-1]) * align(shape[-1], 16) * dtype.itemsize


def padded_bytes(array, tensor):
    shape = list(map(int, tensor["shape"]))
    source = np.asarray(array, dtype=DTYPE[tensor["dtype"]])
    if source.size != math.prod(shape):
        raise ValueError(f"payload size {source.size} != ABI shape {shape}")
    if not shape:
        return source.tobytes(order="C")
    rows, cols = math.prod(shape[:-1]), shape[-1]
    result = np.zeros((rows, align(cols, 16)), dtype=source.dtype)
    result[:, :cols] = source.reshape(rows, cols)
    return result.tobytes(order="C")


def main():
    cb = json.loads((BUNDLE / "compiler/command_buffer.json").read_text())
    manifest = json.loads((CAPTURE / "weights.safetensors.manifest.json").read_text())
    inputs = np.asarray(json.loads((CAPTURE / "inputs.json").read_text()), dtype=np.float32)
    recipes = {row["tensor"]: row
               for row in cb["params"].get("weight_prepack_recipes", [])}
    args = cb["kernel_abi"]["args"]
    reads = [row for row in args if row["access"] == "read"]
    writes = [row for row in args if row["access"] != "read"]
    if args != reads + writes or [r["tensor"] for r in reads] != [f"arg{i}" for i in range(len(reads))]:
        raise ValueError("unexpected whole-program ABI order")
    layout, offset, hwio_checks = [], 0, []
    with SafeTensorReader(CAPTURE / "weights.safetensors") as weights:
        with (BUNDLE / "payload/const_blob.bin").open("wb") as stream:
            for index, arg in enumerate(reads):
                name = arg["tensor"]
                tensor = cb["tensors"][name]
                entry = manifest[str(index)]
                source_name = "image" if entry["kind"] == "input" else entry["weight"]
                array = inputs[0] if entry["kind"] == "input" else weights.get_tensor(source_name)
                source_shape = list(array.shape)
                recipe = recipes.get(name)
                if recipe:
                    if source_shape != recipe["source_shape"]:
                        raise ValueError(f"{name}: source/recipe shape mismatch")
                    pair = (recipe["source_layout"], recipe["packed_layout"])
                    if pair == ("NK", "KN_dim_padded"):
                        array = np.ascontiguousarray(array.T)
                    elif pair == ("OIHW", "CoK_dim_padded"):
                        array = np.ascontiguousarray(array.reshape(recipe["packed_shape"]))
                    elif pair == ("OIHW", "HWIO_dim_padded"):
                        if recipe.get("permutation") != [2, 3, 1, 0]:
                            raise ValueError("HWIO recipe lacks exact OIHW permutation")
                        source = np.asarray(array)
                        array = np.ascontiguousarray(source.transpose(2, 3, 1, 0).reshape(
                            recipe["packed_shape"]))
                        # Independent scalar index formula; do not validate transpose with itself.
                        o, i, h, w = source.shape
                        scalar = np.asarray([
                            source[oc, ic, ky, kx]
                            for ky in range(h) for kx in range(w)
                            for ic in range(i) for oc in range(o)
                        ], dtype=source.dtype).reshape(recipe["packed_shape"])
                        if not np.array_equal(array, scalar):
                            raise ValueError("OIHW->HWIO vector and scalar packers disagree")
                        hwio_checks.append({"tensor": name, "elements": int(array.size),
                                            "unpadded_sha256": sha_bytes(array.tobytes())})
                    else:
                        raise ValueError(f"unsupported prepack {recipe}")
                payload = padded_bytes(array, tensor)
                aligned = align(offset)
                stream.write(bytes(aligned - offset)); stream.write(payload)
                offset = aligned + len(payload)
                layout.append({"index": index, "tensor": name, "access": "read",
                               "storage": "const", "offset": aligned, "bytes": len(payload),
                               "logical_shape": tensor["shape"], "dtype": tensor["dtype"],
                               "source": source_name, "source_shape": source_shape,
                               "prepack_recipe": recipe})
    const_bytes = offset
    encodings = cb["params"].get("storage_encodings", {})
    offset = 0
    for index, arg in enumerate(writes, len(reads)):
        name = arg["tensor"]; tensor = cb["tensors"][name]
        aligned = align(offset); size = physical_nbytes(name, tensor, encodings)
        layout.append({"index": index, "tensor": name, "access": arg["access"],
                       "storage": "mutable", "offset": aligned, "bytes": size,
                       "logical_shape": tensor["shape"], "dtype": tensor["dtype"],
                       "storage_encoding": encodings.get(name)})
        offset = aligned + size
    mutable_bytes = align(offset)
    abi = {"alignment_bytes": 64, "row_padding_elements": 16,
           "const_blob_bytes": const_bytes, "mutable_blob_bytes": mutable_bytes,
           "kernel_arg_count": len(args), "args": layout}
    (BUNDLE / "payload/abi_layout.json").write_text(json.dumps(abi, indent=2) + "\n")

    lines = []
    for row in layout:
        base = "merlin_const_blob_start" if row["storage"] == "const" else "merlin_mutable_blob"
        lines.append(f"      (void *)({base} + {row['offset']})")
    replacement = "  gemmini_kernel(\n" + ",\n".join(lines) + ");\n"
    harness_hashes = {}
    for harness_name in ("single_run_harness.c", "single_run_harness_warm_progress.c"):
        harness_path = BUNDLE / "payload" / harness_name
        harness = harness_path.read_text()
        harness = re.sub(r"#define MERLIN_MUTABLE_BLOB_BYTES \(\(size_t\)\d+\)",
                         f"#define MERLIN_MUTABLE_BLOB_BYTES ((size_t){mutable_bytes})", harness)
        start = harness.index("static void run_model(void) {")
        call = harness.index("  gemmini_kernel(\n", start)
        end = harness.index(");\n", call) + 3
        harness = harness[:call] + replacement + harness[end:]
        harness_path.write_text(harness)
        harness_hashes[harness_name] = sha_bytes(harness_path.read_bytes())
        if harness_name.endswith("_warm_progress.c"):
            if harness.count("  run_model_warm_progress();\n  gemmini_fence();") != 1:
                raise ValueError("warm-progress harness does not invoke the instrumented warm entry")
    receipt = {"schema": "compute_only_loop_conv_payload_repack_v1", "status": "passed",
               "const_blob_bytes": const_bytes, "mutable_blob_bytes": mutable_bytes,
               "args": len(layout), "hwio_independent_checks": hwio_checks,
               "const_blob_sha256": sha_bytes((BUNDLE / "payload/const_blob.bin").read_bytes()),
               "harness_sha256": harness_hashes}
    (ARTIFACT / "validation/canonical_resnet50/payload_repack_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
