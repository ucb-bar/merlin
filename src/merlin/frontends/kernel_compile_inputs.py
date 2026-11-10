"""Compiler-only admission of explicitly standalone functional tensor kernels."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, IntegerAttr, StringAttr, TensorType, i1
from xdsl.dialects.func import FuncOp

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.frontends.kernel_shape_abi import validate_dimensions
from merlin.xdsl_dialects._common import text


def _guard_layout(spec, version2, *, atomics=False):
    if not version2:
        return {"guard_storage_bytes_per_output_element": 16}
    validate_dimensions(spec, spec["buffers"], spec.get("runtime_scalars", []))
    bounds = {s["name"]: s.get("bounds") for s in spec.get("runtime_scalars", [])}
    counts = [n if type(n) is int else (bounds[n["dimension"]][1] + n["divisor"] - 1) // n["divisor"]
              for n in spec.get("grid", [1])]
    bits = 30 if atomics else 31
    words = max(1, ((math.prod(counts) - 1).bit_length() + bits - 1) // bits)
    return {"owner_words_per_element": words, "guard_storage_bytes_per_output_element": 8 * (words + 1),
            **({"owner_fragment_bits": bits} if atomics else {})}


def _checked_memory_members(checked, provenance):
    profile = provenance.get("profile", "shared-functional-tensor")
    if profile == "shared-functional-tensor":
        return {}
    word = profile == "shared-source-word-buffer"
    guarded = profile in ("shared-guarded-buffer", "shared-source-word-buffer")
    if profile not in ("shared-functional-buffer", "shared-guarded-buffer", "shared-source-word-buffer"):
        raise ValueError("unsupported kernel memory evidence profile")
    names = ("memory-admission.json" if guarded else "memory-witness.mlir", "memory-input.mlir", "memory-stage.mlir", "memory-stage.json")
    copied = {name: checked(name) for name in names}
    admission = json.loads(copied[names[0]]) if guarded else None
    version3 = admission is not None and admission.get("schema") in ("t2m.memory-admission.v3", "t2m.memory-admission.v4")
    version2 = admission is not None and admission.get("schema") in ("t2m.memory-admission.v2", "t2m.memory-admission.v3", "t2m.memory-admission.v4")
    memory = provenance.get("memory", {})
    for name, key in ((names[0], "admission_sha256" if guarded else "witness_sha256"), (names[1], "continuation_input_sha256"),
                      (names[2], "continuation_output_sha256")):
        if memory.get(key) != hashlib.sha256(copied[name]).hexdigest():
            raise ValueError("kernel memory evidence differs from its provenance")
    stage = json.loads(copied[names[3]])
    def record(name):
        return {"bytes": len(copied[name]), "sha256": hashlib.sha256(copied[name]).hexdigest()}
    if (stage.get("schema") != "t2m.triton-shared-stage.v1" or stage.get("stage") != ("memory-effects" if version2 else "memory-ir")
            or stage.get("stage_completed") is not True
            or stage.get("artifacts", {}).get("source.ttir") != record(names[1])
            or stage.get("artifacts", {}).get("lowered.mlir") != record(names[2])):
        raise ValueError("kernel memory report does not bind its input/output")
    if guarded:
        spec = provenance.get("spec")
        if not isinstance(spec, dict) or not isinstance(spec.get("buffers"), list):
            raise ValueError("guarded memory evidence lacks its logical buffer ABI")
        arguments = ([{"kind": "extent", "buffer": b["name"]} for b in spec["buffers"]]
            + [{"kind": kind, "buffer": b["name"]} for b in spec["buffers"] if b["effect"] != "read" for kind in ("owners", "visited")]
            + [{"kind": "program"}])
        layout = _guard_layout(spec, version2, atomics=version3)
        if (admission.get("schema") not in ("t2m.memory-admission.v1", "t2m.memory-admission.v2", "t2m.memory-admission.v3", "t2m.memory-admission.v4")
                or admission.get("strategy") != "runtime-guarded-private-output-state"
                or admission.get("spec") != spec or admission.get("internal_arguments") != arguments
                or admission.get("guarded_input_sha256") != record(names[1])["sha256"]
                or admission.get("continuation_output_sha256") != record(names[2])["sha256"]
                or any(admission.get(key) != value for key, value in layout.items())
                or version2 and admission.get("ownership_protocol") != ("sticky-complemented-atomic-participants-v1" if version3 else "sticky-complemented-program-fragments")
                or admission.get("writable_reads") != ("shared-readers-or-single-writer-program" if version2 else "refused")
                or admission.get("fresh_initialization") != "post-grid-complete-write-coverage"
                or not isinstance(admission.get("accesses"), list)):
            raise ValueError("guarded memory evidence has an inconsistent admission contract")
        buffers = {b["name"]: b for b in spec["buffers"]}
        records = provenance.get("normalized_operations", provenance.get("source_operations", []))
        inventory = {r["id"] for r in records}
        for access in admission["accesses"]:
            buffer = buffers.get(access.get("buffer"))
            atomic = version3 and access.get("opcode") in ("tts.unstructured_atomic_rmw", "tt.atomic_cas")
            write = access.get("opcode") in ("tts.store", "tts.scatter") or atomic
            allowed = ("tts.load", "tts.store", "tts.gather", "tts.scatter") + (("tts.unstructured_atomic_rmw", "tt.atomic_cas") if version3 else ())
            if (buffer is None or access.get("opcode") not in allowed
                    or access.get("effect") != ("atomic-readwrite" if atomic else "write" if write else "read")
                    or write and buffer["effect"] == "read"
                    or not write and buffer["effect"] != "read" and not version2
                    or not access.get("source_ids") or any(s not in inventory for s in access["source_ids"])
                    or not access.get("tile") or any(type(n) is not int or n <= 0 for n in access["tile"])):
                raise ValueError("guarded memory access inventory differs from its source/effect contract")
        if version3:
            from .kernel_atomic_evidence import checked_atomic_effects
            if not word:
                raise ValueError("atomic memory evidence requires its source-word address profile")
            checked_atomic_effects(admission, memory, records, buffers)
    if word:
        for name in ("word-input.mlir", "word-tensor.mlir", "word-stage.json"):
            copied[name] = checked(name)
        word_stage = json.loads(copied["word-stage.json"])
        if (not version2 or admission.get("address_policy") != "source-words-offset64"
                or memory.get("address_policy") != "source-words-offset64"
                or memory.get("word_stage_sha256") != record("word-stage.json")["sha256"]
                or admission.get("word_input_sha256") != record("word-input.mlir")["sha256"]
                or admission.get("word_tensor_sha256") != record("word-tensor.mlir")["sha256"]):
            raise ValueError("source-word address evidence does not bind its declared input/output/policy")
        from .kernel_word_evidence import checked_word_stages
        checked_word_stages(word_stage, copied["word-input.mlir"], copied["word-tensor.mlir"], memory, inventory)
    return copied


def _checked_word_intent(entry, members):
    if entry.attributes.get("prov.address_policy") != StringAttr("source-words-offset64"):
        raise ValueError("source-word kernel lacks its declared address intent")
    access = entry.attributes.get("prov.source_accesses")
    if not isinstance(access, StringAttr) or json.loads(access.data) != json.loads(members["memory-admission.json"])["accesses"]:
        raise ValueError("source-word kernel access intent differs from admission")
    admission = json.loads(members["memory-admission.json"])
    if admission.get("schema") in ("t2m.memory-admission.v3", "t2m.memory-admission.v4"):
        from .kernel_atomic_evidence import EFFECT_PROFILE, TYPED_EFFECT_PROFILE
        profile = TYPED_EFFECT_PROFILE if admission["schema"] == "t2m.memory-admission.v4" else EFFECT_PROFILE
        if entry.attributes.get("prov.effect_profile") != StringAttr(profile):
            raise ValueError("atomic kernel lacks its private-invocation effect intent")


def _stage_kernel_inputs(source: Path, target: Path, receipt: dict, receipt_raw: bytes) -> dict:
    observable = receipt.get("schema") == "t2m.effect-kernel-receipt.v1"
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

    if receipt.get("profile") not in ("shared-functional-tensor", "shared-functional-buffer", "shared-guarded-buffer", "shared-source-word-buffer"):
        raise ValueError("unsupported standalone kernel profile")
    raw, ttir, spec_raw, provenance_raw = (checked(name) for name in
                                         ("model.mlir", "source.ttir", "spec.json", "provenance.json"))
    spec, provenance = json.loads(spec_raw), json.loads(provenance_raw)
    if provenance.get("profile", "shared-functional-tensor") != receipt["profile"]:
        raise ValueError("kernel receipt and memory provenance profiles differ")
    memory_members = _checked_memory_members(checked, provenance)
    if spec.get("schema") not in tuple("t2m.shared-tensor-abi.v" + str(i) for i in range(1, 7)):
        raise ValueError("unsupported standalone tensor ABI")
    effect_only = spec["schema"] == "t2m.shared-tensor-abi.v6"
    if (effect_only and (not observable or spec.get("effect_only") is not True)
            or not effect_only and spec.get("effect_only", False) is not False):
        raise ValueError("effect-only ABI requires its explicit observable-effect receipt")
    if (provenance.get("spec") != spec or provenance.get("mlir_sha256") != digest(raw)
            or provenance.get("source_sha256") != digest(ttir)):
        raise ValueError("kernel source/provenance/ABI correspondence failed")
    weights = {name: checked(name) for name in ("weights.safetensors", "weights.safetensors.manifest.json")}
    if weights["weights.safetensors"] != struct.pack("<Q", 8) + b"{}      ":
        raise ValueError("standalone profile requires an empty parameter container")
    buffers = spec.get("buffers", [])
    if (not buffers and not effect_only or any(b.get("effect") not in ("read", "write", "readwrite") for b in buffers)
            or len({b.get("name") for b in buffers}) != len(buffers)):
        raise ValueError("standalone profile has an incomplete memory ABI")
    inputs = [b for b in buffers if b["effect"] != "write"]
    scalars = spec.get("runtime_scalars", [])
    if not effect_only and bool(scalars) != (spec["schema"] in ("t2m.shared-tensor-abi.v3", "t2m.shared-tensor-abi.v4", "t2m.shared-tensor-abi.v5")):
        raise ValueError("runtime scalar inputs require the explicit ABI v3 profile")
    if any(set(s) != {"name", "dtype", "source_arg"} | ({"bounds"} if "bounds" in s else set())
           or not isinstance(s["name"], str) or not s["name"].isidentifier() for s in scalars):
        raise ValueError("runtime scalar declarations carry names/types/positions, never invocation values")
    if spec["schema"] not in ("t2m.shared-tensor-abi.v5", "t2m.shared-tensor-abi.v6") and (spec["schema"] == "t2m.shared-tensor-abi.v4") != any("bounds" in s for s in scalars):
        raise ValueError("guarded runtime scalar ranges require ABI v4")
    for scalar in scalars:
        if "bounds" not in scalar:
            continue
        bounds, dtype = scalar["bounds"], scalar["dtype"]
        if (dtype not in ("i8", "i16", "i32", "i64") or not isinstance(bounds, list) or len(bounds) != 2
                or any(type(n) is not int for n in bounds)
                or not -(1 << (int(dtype[1:]) - 1)) <= bounds[0] <= bounds[1] < 1 << (int(dtype[1:]) - 1)):
            raise ValueError("runtime scalar range is outside its signed integer type")
    dynamic = validate_dimensions(spec, buffers, scalars)
    positions = [s.get("source_arg") for s in scalars]
    names = [b.get("name") for b in buffers] + [s.get("name") for s in scalars]
    if (len(set(names)) != len(names)
            or any(type(p) is not int or not 0 <= p < len(buffers) + len(spec.get("scalar_constants", [])) + len(scalars) for p in positions)
            or len(set(positions)) != len(positions)
            or any(s.get("dtype") not in ("f32", "f64", "i1", "i8", "i16", "i32", "i64") for s in scalars)):
        raise ValueError("runtime scalar declaration has an invalid type/name/source position")
    inputs.extend({**s, "shape": [], "abi_kind": "scalar"} for s in scalars)
    outputs = [b for b in buffers if b["effect"] != "read"]
    if (not outputs and not effect_only or receipt.get("materialized_abi") !=
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
                               **({"scalar": buffer["name"]} if dynamic else {}),
                               **({"bounds": buffer["bounds"]} if "bounds" in buffer else {})})
                continue
            if not isinstance(type_, TensorType):
                raise ValueError("standalone profile requires ranked tensor shapes")
            shape, declared = tuple(type_.get_shape()), buffer["shape"]
            if len(shape) != len(declared) or any((n < 0) != isinstance(d, str) or n >= 0 and n != d for n, d in zip(shape, declared)):
                raise ValueError("kernel MLIR tensor dimensions differ from declared ABI")
            value = {"shape": declared, "dtype": str(type_.get_element_type())}
            if value["dtype"] != buffer.get("dtype", "f32"):
                raise ValueError("kernel MLIR tensor type differs from declared ABI")
            values.append(value)
        return values

    signature = {"schema": "merlin.compile-signature.v2" if dynamic else "merlin.compile-signature.v1", "entry": "forward",
                 "inputs": abi(entry.function_type.inputs, inputs), "outputs": abi(entry.function_type.outputs, outputs),
                 "shape_scope": "runtime_scalar_bound_dimensions" if dynamic else "static_captured_signature", "capture_dtype": "typed-per-argument",
                 "input_mutability": "borrowed-immutable-tensors"}
    if observable:
        signature["effect_abi"] = "source-encoded-observable-effects; evaluator-outcome-excluded"
    if receipt["profile"] in ("shared-guarded-buffer", "shared-source-word-buffer"):
        contract = entry.attributes.get("prov.memory_guard_bytes_per_result_element")
        admission = json.loads(memory_members["memory-admission.json"])
        cost = admission["guard_storage_bytes_per_output_element"]
        if not isinstance(contract, IntegerAttr) or contract.value.data != cost:
            raise ValueError("guarded kernel lacks its declared scratch storage contract")
        signature["guard_storage_bytes_per_result_element"] = cost
        if receipt["profile"] == "shared-source-word-buffer":
            _checked_word_intent(entry, memory_members)
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
    fields = ("schema", "profile", "abi_difference", "correspondence", "emitted_operations", "intent", "memory",
              "mlir_sha256", "optimizer", "pipeline", "source_operations", "source_sha256", "spec",
              "transformations", "normalized_operations", "original_attributes", "original_blocks")
    correspondence = {"schema": "merlin.kernel-source-correspondence.v1", "producer": "triton2MLIR",
                      "input_model_sha256": digest(raw), "staged_model_sha256": digest(staged),
                      "source_provenance_sha256": digest(provenance_raw), "source_entry": spec["entry"],
                      "staged_entry": "forward", "scope": "byte-bound-lineage; semantic-admission-unverified"}
    members = {"program.mlir": staged, "signature.json": encoded(signature), **weights,
               "source/model.mlir": raw, "source/source.ttir": ttir, "source/spec.json": spec_raw,
               "source/compiler-provenance.json": encoded({k: provenance[k] for k in fields if k in provenance}),
               **{"source/" + name: data for name, data in memory_members.items()},
               "source-correspondence.json": encoded(correspondence)}
    manifest = {"schema": "merlin.compile-inputs.v1", "source_profile": "standalone-effect-kernel" if observable else "standalone-tensor-kernel",
                "payload_profile": receipt["profile"],
                "scope": "compiler_only_declared_runtime_shapes; source_closure_and_numerical_admission_unverified" if dynamic else "compiler_only_static_capture; source_closure_and_numerical_admission_unverified",
                "source": {"capture_receipt_sha256": digest(receipt_raw), "capture_schema": receipt["schema"],
                           "model_mlir_sha256": digest(raw), "correspondence": "source-correspondence.json"},
                "members": {name: digest(data) for name, data in members.items()},
                "excluded_evaluator_members": ["inputs.npz", "golden.npy", "extra.npz", "execution.json",
                    *(["reference.json", "native/"] if observable else [])]}
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
