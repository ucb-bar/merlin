"""Compiler-only staging of receipt-bound derived tensor models.

The original FX trace describes the parent program. A kernel-binding record
describes the derivation and points to exact TTIR implementations. Keep those
graphs separate; neither byte identity nor staging grants numeric admission.
This module imports no capture frontend, evaluator or Triton implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntegerAttr, StringAttr, TensorType, i1
from xdsl.dialects.func import FuncOp

from merlin.frontends.derived_runtime_arguments import checked_runtime_call_shapes
from merlin.frontends.kernel_allocation_evidence import checked_allocation_sources
from merlin.frontends.kernel_compile_inputs import _checked_memory_members, _checked_word_intent
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text as module_text


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _relative(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name or not path.parts:
        raise ValueError("invalid derived compiler artifact path")
    return path


def _stage_bound_model_inputs(source: Path, target: Path, receipt: dict, receipt_raw: bytes) -> dict:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or receipt.get("materialized_abi") != "unchanged-from-parent":
        raise ValueError("derived capture has an incomplete ABI/artifact inventory")

    def checked(name):
        relative = _relative(name)
        path = source / relative
        if any(source.joinpath(*relative.parts[:i]).is_symlink() for i in range(1, len(relative.parts) + 1)):
            raise ValueError("derived capture member is a symlink: " + name)
        if not path.is_file() or name not in artifacts:
            raise ValueError("derived capture lacks compiler member: " + name)
        raw = path.read_bytes()
        if artifacts[name] != {"bytes": len(raw), "sha256": _sha(raw)}:
            raise ValueError("derived capture member differs from receipt: " + name)
        return raw

    raw = checked("model.mlir")
    weights = {name: checked(name) for name in ("weights.safetensors", "weights.safetensors.manifest.json")}
    bindings_raw = checked("kernel-bindings.json")
    bindings = json.loads(bindings_raw)
    if (bindings.get("schema") not in ("t2m.model-kernel-binding.v1", "t2m.model-kernel-binding.v2")
            or bindings.get("mlir_sha256") != _sha(raw)):
        raise ValueError("derived source correspondence does not bind the current model")
    parent_raw = checked("parent/capture_receipt.json")
    parent = json.loads(parent_raw)
    if (receipt.get("parent_receipt") != {"bytes": len(parent_raw), "sha256": _sha(parent_raw)}
            or parent.get("schema") != "m2m.capture-receipt.v1"
            or (parent.get("materialized_abi") or {}).get("complete") is not True):
        raise ValueError("derived capture lacks a bound complete parent ABI")
    copied = {}
    parent_members = receipt.get("parent_artifact_bindings")
    if not isinstance(parent_members, dict):
        raise ValueError("derived capture lacks parent artifact bindings")
    # Select compiler members explicitly. Do not validate the entire capture by
    # reading runtime inputs, goldens or the evaluator's execution report.
    for name in ("model.mlir", "frontend-trace.json", "weights.safetensors", "weights.safetensors.manifest.json"):
        expected_path = name if name.startswith("weights.") else "parent/" + name
        if parent_members.get(name) != expected_path:
            raise ValueError("derived parent compiler-member mapping differs: " + name)
        data = weights[name] if name.startswith("weights.") else checked(expected_path)
        if (parent.get("artifacts") or {}).get(name) != {"bytes": len(data), "sha256": _sha(data)}:
            raise ValueError("derived parent compiler artifact relation failed: " + name)
        if not name.startswith("weights."):
            copied["source/parent/" + name] = data
    parent_model = copied["source/parent/model.mlir"]
    parent_trace = json.loads(copied["source/parent/frontend-trace.json"])
    if (bindings.get("input_model_sha256") != _sha(parent_model)
            or parent_trace.get("status") != "complete"
            or (parent_trace.get("mlir") or {}).get("sha256") != _sha(parent_model)):
        raise ValueError("derived correspondence does not bind the original parent graph")
    parent_module = parse_mlir_text(parent_model.decode())
    parent_module.verify()
    from .compiler_constants import checked_compiler_constants
    constants = checked_compiler_constants(parent, weights["weights.safetensors.manifest.json"], weights["weights.safetensors"],
                                           parent_module, parent_trace)
    if constants is not None:
        copied["compiler-constants.json"] = _json(constants)
    copied["source/parent/compiler-artifacts.json"] = _json({
        "schema": "merlin.capture-compiler-projection.v1", "original_receipt_sha256": _sha(parent_raw),
        "artifacts": {name: parent["artifacts"][name] for name in
                      ("model.mlir", "frontend-trace.json", "weights.safetensors", "weights.safetensors.manifest.json")}})
    implementations = receipt.get("kernel_artifacts")
    if not isinstance(implementations, dict) or set(implementations) != set(bindings.get("kernels", {})):
        raise ValueError("derived kernel implementation inventory differs from correspondence")
    calls = bindings.get("calls")
    if (not isinstance(calls, list)
            or {call.get("implementation_sha256") for call in calls} != set(implementations)
            or len({call.get("id") for call in calls}) != len(calls)):
        raise ValueError("derived kernel call correspondence is incomplete or ambiguous")
    tensor_views = bindings["schema"] == "t2m.model-kernel-binding.v2"
    if (tensor_views and (bindings.get("tensor_view_binding_profile") != "t2m.captured-dense-tensor-views.v1"
                        or not any(call.get("tensor_views") == "dense-row-major-v1" for call in calls))
            or not tensor_views and ("tensor_view_binding_profile" in bindings
                                     or any("tensor_views" in call for call in calls))
            or any(call.get("tensor_views") not in (None,"dense-row-major-v1") for call in calls)):
        raise ValueError("derived tensor views require the explicit v2 binding and known view policy")
    guard_storage, runtime_ordinals = 0, set()
    for digest, directory in implementations.items():
        relative = _relative(directory)
        if len(relative.parts) != 2 or relative.parts[0] != "kernels":
            raise ValueError("derived kernel artifacts need their own kernel directory")
        kernel = {name: checked(directory + "/" + name)
                  for name in ("model.mlir", "source.ttir", "spec.json", "provenance.json")}
        provenance, spec = json.loads(kernel["provenance.json"]), json.loads(kernel["spec.json"])
        kernel.update(_checked_memory_members(lambda name: checked(directory + "/" + name), provenance))
        kernel.update(checked_allocation_sources(lambda name: checked(directory + "/" + name), provenance, kernel["source.ttir"]))
        runtime_shapes = []
        selected_calls = [call for call in calls if call["implementation_sha256"] == digest]
        if spec.get("runtime_scalars") or any(call.get("tensor_views") for call in selected_calls):
            if (spec.get("runtime_scalars")
                    and bindings.get("argument_binding_profile") != "t2m.captured-runtime-arguments.v1"):
                raise ValueError("derived runtime kernel arguments require their explicit binding profile")
            for call in selected_calls:
                ordinal = call.get("parent_call_ordinal")
                if type(ordinal) is not int or ordinal in runtime_ordinals:
                    raise ValueError("derived runtime kernel call correspondence is ambiguous")
                runtime_ordinals.add(ordinal)
                runtime_shapes.append(checked_runtime_call_shapes(parent_module, spec, call, tensor_views=tensor_views))
        if provenance.get("profile") in ("shared-guarded-buffer", "shared-source-word-buffer"):
            outputs = [b for b in spec["buffers"] if b["effect"] != "read"]
            if not runtime_shapes and any(not b.get("shape") or any(type(n) is not int or n <= 0 for n in b["shape"]) for b in outputs):
                raise ValueError("guarded derived kernels require static positive scratch dimensions")
            kernel_module = parse_mlir_text(kernel["model.mlir"].decode())
            kernel_module.verify()
            entries = list(kernel_module.ops)
            if len(entries) != 1 or not isinstance(entries[0], FuncOp):
                raise ValueError("guarded derived kernel lacks one typed entry")
            if provenance["profile"] == "shared-source-word-buffer":
                _checked_word_intent(entries[0], kernel)
            contract = entries[0].attributes.get("prov.memory_guard_bytes_per_result_element")
            types = list(entries[0].function_type.outputs)
            cost = json.loads(kernel["memory-admission.json"])["guard_storage_bytes_per_output_element"]
            if (not isinstance(contract, IntegerAttr) or contract.value.data != cost or len(types) != len(outputs)
                    or any(not isinstance(t, TensorType) or len(t.get_shape()) != len(b["shape"])
                           or any((actual >= 0 if isinstance(expected, str) else actual != expected)
                                  for actual, expected in zip(t.get_shape(), b["shape"], strict=True))
                           or str(t.get_element_type()) != b["dtype"] for t, b in zip(types, outputs))):
                raise ValueError("guarded derived kernel scratch differs from its typed results")
            guard_storage += (cost * sum(math.prod(shape[b["name"]]) for shape in runtime_shapes for b in outputs)
                if runtime_shapes else cost * sum(math.prod(b["shape"]) for b in outputs) * sum(
                    call["implementation_sha256"] == digest for call in calls))
            if not 0 <= guard_storage < 2**63:
                raise ValueError("guarded derived kernel scratch exceeds its i64 domain")
        if (digest != _sha(kernel["model.mlir"]) or provenance.get("mlir_sha256") != digest
                or provenance.get("source_sha256") != _sha(kernel["source.ttir"])
                or provenance.get("spec") != spec or provenance != bindings["kernels"][digest]):
            raise ValueError("derived kernel source/ABI/provenance relation failed")
        if any(call.get("source_sha256") != provenance["source_sha256"]
               for call in calls if call["implementation_sha256"] == digest):
            raise ValueError("derived kernel call does not bind its original TTIR")
        copied.update(("source/" + directory + "/" + name, data) for name, data in kernel.items())
    module = parse_mlir_text(raw.decode())
    module.verify()
    functions = [op for op in module.ops if isinstance(op, FuncOp)]
    if len(functions) != 1 or not functions[0].body.blocks or functions[0].sym_name.data != "forward":
        raise ValueError("derived compile profile requires one defined forward tensor entry")
    for op in module.walk():
        if op.name == "builtin.unregistered" or "prov.kernel" in op.attributes or op.name == "func.call":
            raise ValueError("derived compile profile retains an unresolved operation or dependency")
    entry = functions[0]
    parent_entries = [op for op in parent_module.ops if isinstance(op, FuncOp) and op.sym_name.data == "forward"]
    if len(parent_entries) != 1 or entry.function_type != parent_entries[0].function_type:
        raise ValueError("derived model changes the original public function ABI")

    def abi(types):
        values = []
        for type_ in types:
            if not isinstance(type_, TensorType) or any(n < 0 for n in type_.get_shape()):
                raise ValueError("derived compile profile needs static ranked tensor arguments/results")
            values.append({"shape": list(type_.get_shape()), "dtype": str(type_.get_element_type())})
        return values

    if module.attributes.get("prov.weights_file") != StringAttr("weights.safetensors"):
        raise ValueError("derived model needs its canonical relative weight reference")
    # Public tensor values are borrowed functional inputs. Function-boundary
    # bufferization otherwise treats entry arguments as writable by default.
    # Keep this ABI premise explicit instead of allowing destination reuse to
    # mutate model parameters or runtime inputs across invocations.
    argument_attributes = list(entry.arg_attrs) if entry.arg_attrs is not None else [DictionaryAttr({}) for _ in entry.function_type.inputs]
    entry.properties["arg_attrs"] = ArrayAttr(DictionaryAttr({**attributes.data, "bufferization.writable": IntegerAttr(0, i1)})
                                             for attributes in argument_attributes)
    staged = module_text(module, generic=True).encode()
    parse_mlir_text(staged.decode()).verify()
    signature = {"schema": "merlin.compile-signature.v1", "entry": entry.sym_name.data,
                 "inputs": abi(entry.function_type.inputs), "outputs": abi(entry.function_type.outputs),
                 "shape_scope": "static_captured_signature", "capture_dtype": "typed-per-argument",
                 "input_mutability": "borrowed-immutable-tensors"}
    contract = entry.attributes.get("prov.memory_guard_scratch_bytes")
    if guard_storage:
        if (not isinstance(contract, IntegerAttr) or contract.value.data != guard_storage
                or bindings.get("guard_storage_bytes") != guard_storage):
            raise ValueError("guarded derived scratch declaration differs from kernel call storage")
        signature["guard_storage_bytes"] = guard_storage
    elif contract is not None or "guard_storage_bytes" in bindings:
        raise ValueError("derived model has an unbound guard storage declaration")
    correspondence = {"schema": "merlin.derived-source-correspondence.v1", "producer": "triton2MLIR",
                      "input_model_sha256": _sha(raw), "staged_model_sha256": _sha(staged),
                      "derivation": bindings, "scope": "byte-bound-lineage; semantic-admission-unverified"}
    members = {"program.mlir": staged, "signature.json": _json(signature), **weights, **copied,
               "source-correspondence.json": _json(correspondence)}
    manifest = {"schema": "merlin.compile-inputs.v1", "source_profile": "derived-tensor-model",
                "scope": "compiler_only_static_capture; source_closure_and_numerical_admission_unverified",
                "source": {"capture_receipt_sha256": _sha(receipt_raw), "model_mlir_sha256": _sha(raw),
                           "capture_schema": receipt["schema"], "correspondence": "source-correspondence.json"},
                "members": {name: _sha(data) for name, data in members.items()},
                "excluded_evaluator_members": ["inputs.npz", "inputs.json", "golden.npy", "goldens.npz", "tensor_words.npz", "golden.json", "extra.npz", "execution.json"]}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        for name, data in members.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (temporary / "compile-inputs.json").write_bytes(_json(manifest))
        if target.exists() or target.is_symlink():
            raise ValueError("compile-input destination appeared during staging")
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest
