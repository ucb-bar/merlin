"""Compiler-only admission of explicitly standalone functional tensor kernels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntegerAttr, StringAttr, TensorType, i1
from xdsl.dialects.func import FuncOp

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text


def _stage_kernel_inputs(source: Path, target: Path, receipt: dict, receipt_raw: bytes) -> dict:
    def digest(raw):
        return hashlib.sha256(raw).hexdigest()

    def encoded(value):
        return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()

    def checked(name):
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("kernel compiler member absent or symlink: " + name)
        raw = path.read_bytes()
        if receipt.get("artifacts", {}).get(name) != {"bytes": len(raw), "sha256": digest(raw)}:
            raise ValueError("kernel compiler member differs from receipt: " + name)
        return raw

    if receipt.get("profile") != "shared-functional-tensor":
        raise ValueError("unsupported standalone kernel profile")
    raw, ttir, spec_raw, provenance_raw = (checked(name) for name in
                                         ("model.mlir", "source.ttir", "spec.json", "provenance.json"))
    spec, provenance = json.loads(spec_raw), json.loads(provenance_raw)
    if spec.get("schema") not in ("t2m.shared-tensor-abi.v1", "t2m.shared-tensor-abi.v2", "t2m.shared-tensor-abi.v3", "t2m.shared-tensor-abi.v4"):
        raise ValueError("unsupported standalone tensor ABI")
    if (provenance.get("spec") != spec or provenance.get("mlir_sha256") != digest(raw)
            or provenance.get("source_sha256") != digest(ttir)):
        raise ValueError("kernel source/provenance/ABI correspondence failed")
    weights = {name: checked(name) for name in ("weights.safetensors", "weights.safetensors.manifest.json")}
    if weights["weights.safetensors"] != struct.pack("<Q", 8) + b"{}      ":
        raise ValueError("standalone profile requires an empty parameter container")
    buffers = spec.get("buffers", [])
    if (not buffers or any(b.get("effect") not in ("read", "write", "readwrite") for b in buffers)
            or len({b.get("name") for b in buffers}) != len(buffers)):
        raise ValueError("standalone profile has an incomplete memory ABI")
    inputs = [b for b in buffers if b["effect"] != "write"]
    scalars = spec.get("runtime_scalars", [])
    if bool(scalars) != (spec["schema"] in ("t2m.shared-tensor-abi.v3", "t2m.shared-tensor-abi.v4")):
        raise ValueError("runtime scalar inputs require the explicit ABI v3 profile")
    if any(set(s) != {"name", "dtype", "source_arg"} | ({"bounds"} if "bounds" in s else set())
           or not isinstance(s["name"], str) or not s["name"].isidentifier() for s in scalars):
        raise ValueError("runtime scalar declarations carry names/types/positions, never invocation values")
    if (spec["schema"] == "t2m.shared-tensor-abi.v4") != any("bounds" in s for s in scalars):
        raise ValueError("guarded runtime scalar ranges require ABI v4")
    for scalar in scalars:
        if "bounds" not in scalar:
            continue
        bounds, dtype = scalar["bounds"], scalar["dtype"]
        if (dtype not in ("i8", "i16", "i32", "i64") or not isinstance(bounds, list) or len(bounds) != 2
                or any(type(n) is not int for n in bounds)
                or not -(1 << (int(dtype[1:]) - 1)) <= bounds[0] <= bounds[1] < 1 << (int(dtype[1:]) - 1)):
            raise ValueError("runtime scalar range is outside its signed integer type")
    positions = [s.get("source_arg") for s in scalars]
    names = [b.get("name") for b in buffers] + [s.get("name") for s in scalars]
    if (len(set(names)) != len(names)
            or any(type(p) is not int or not 0 <= p < len(buffers) + len(spec.get("scalar_constants", [])) + len(scalars) for p in positions)
            or len(set(positions)) != len(positions)
            or any(s.get("dtype") not in ("f32", "f64", "i1", "i8", "i16", "i32", "i64") for s in scalars)):
        raise ValueError("runtime scalar declaration has an invalid type/name/source position")
    inputs.extend({**s, "shape": [], "abi_kind": "scalar"} for s in scalars)
    outputs = [b for b in buffers if b["effect"] != "read"]
    if (not outputs or receipt.get("materialized_abi") !=
            {"complete": True, "inputs": len(inputs), "outputs": len(outputs)}):
        raise ValueError("kernel receipt does not bind its complete functional ABI")
    if json.loads(weights["weights.safetensors.manifest.json"]) != {
            str(i): {"kind": "input", "name": b["name"]} for i, b in enumerate(inputs)}:
        raise ValueError("kernel input manifest differs from declared buffer order")
    module = parse_mlir_text(raw.decode())
    module.verify()
    functions = [op for op in module.ops if isinstance(op, FuncOp)]
    if len(functions) != 1 or not functions[0].body.blocks or functions[0].sym_name.data != spec.get("entry"):
        raise ValueError("standalone profile requires its one declared tensor entry")
    for op in module.walk():
        if op.name in ("builtin.unregistered", "func.call") or "prov.kernel" in op.attributes:
            raise ValueError("standalone kernel retains an unresolved operation or dependency")
    entry = functions[0]

    def abi(types, declared):
        values = []
        if len(types) != len(declared):
            raise ValueError("kernel MLIR signature differs from declared buffer order")
        for type_, buffer in zip(types, declared, strict=True):
            if buffer.get("abi_kind") == "scalar":
                if isinstance(type_, TensorType) or str(type_) != buffer["dtype"]:
                    raise ValueError("kernel MLIR scalar type differs from declared ABI")
                values.append({"shape": [], "dtype": str(type_), "kind": "scalar",
                               **({"bounds": buffer["bounds"]} if "bounds" in buffer else {})})
                continue
            if not isinstance(type_, TensorType) or any(n <= 0 for n in type_.get_shape()):
                raise ValueError("standalone profile requires positive static tensor shapes")
            value = {"shape": list(type_.get_shape()), "dtype": str(type_.get_element_type())}
            if value != {"shape": buffer.get("shape"), "dtype": buffer.get("dtype", "f32")}:
                raise ValueError("kernel MLIR tensor type differs from declared ABI")
            values.append(value)
        return values

    signature = {"schema": "merlin.compile-signature.v1", "entry": "forward",
                 "inputs": abi(entry.function_type.inputs, inputs), "outputs": abi(entry.function_type.outputs, outputs),
                 "shape_scope": "static_captured_signature", "capture_dtype": "typed-per-argument",
                 "input_mutability": "borrowed-immutable-tensors"}
    if scalars:
        signature["scalar_abi"] = "native-by-value; invocation-data"
    # The native host's existing ABI names @forward. Record this symbol-only
    # adaptation explicitly; retain original source names and provenance IDs.
    entry.properties["sym_name"] = StringAttr("forward")
    attrs = list(entry.arg_attrs) if entry.arg_attrs is not None else [DictionaryAttr({}) for _ in inputs]
    entry.properties["arg_attrs"] = ArrayAttr(DictionaryAttr({**a.data, **(
        {"bufferization.writable": IntegerAttr(0, i1)} if isinstance(t, TensorType) else {})})
        for a, t in zip(attrs, entry.function_type.inputs, strict=True))
    staged = text(module, generic=True).encode()
    parse_mlir_text(staged.decode()).verify()
    # Project compiler provenance, excluding numeric admission/evaluator reports.
    fields = ("schema", "abi_difference", "correspondence", "emitted_operations", "intent", "memory",
              "mlir_sha256", "optimizer", "pipeline", "source_operations", "source_sha256", "spec",
              "transformations", "normalized_operations", "original_attributes", "original_blocks")
    correspondence = {"schema": "merlin.kernel-source-correspondence.v1", "producer": "triton2MLIR",
                      "input_model_sha256": digest(raw), "staged_model_sha256": digest(staged),
                      "source_provenance_sha256": digest(provenance_raw), "source_entry": spec["entry"],
                      "staged_entry": "forward", "scope": "byte-bound-lineage; semantic-admission-unverified"}
    members = {"program.mlir": staged, "signature.json": encoded(signature), **weights,
               "source/model.mlir": raw, "source/source.ttir": ttir, "source/spec.json": spec_raw,
               "source/compiler-provenance.json": encoded({k: provenance[k] for k in fields if k in provenance}),
               "source-correspondence.json": encoded(correspondence)}
    manifest = {"schema": "merlin.compile-inputs.v1", "source_profile": "standalone-tensor-kernel",
                "scope": "compiler_only_static_capture; source_closure_and_numerical_admission_unverified",
                "source": {"capture_receipt_sha256": digest(receipt_raw), "capture_schema": receipt["schema"],
                           "model_mlir_sha256": digest(raw), "correspondence": "source-correspondence.json"},
                "members": {name: digest(data) for name, data in members.items()},
                "excluded_evaluator_members": ["inputs.npz", "golden.npy", "extra.npz", "execution.json"]}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        for name, data in members.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (temporary / "compile-inputs.json").write_bytes(encoded(manifest))
        if target.exists() or target.is_symlink():
            raise ValueError("compile-input destination appeared during staging")
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
