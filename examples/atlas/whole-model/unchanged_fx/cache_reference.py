"""Independent original-FX cache phase replay using an authenticated frozen oracle.

Preparation is the source-hashed case0 embedding/mask/RoPE boundary. Empty
prefill K/V is constructed independently. Later K/V comes exclusively from the
explicit parent independent reference's expected outputs, resolved by cache
roles and global source identities. Generated execution is never an input.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


def checked(path):
    path = Path(path).absolute()
    for ancestor in reversed((path, *path.parents)):
        if ancestor.is_symlink():
            raise ValueError("explicit reference paths must not use symlinks")
    return path.resolve(strict=False)


def file_hash(path):
    digest = sha256()
    with checked(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(checked(path).read_text())


def record_hash(record):
    return sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, record):
    checked(path).write_text(json.dumps(record, indent=2) + "\n")


def load_reference_driver(path, binding_path, model_root, reference_root):
    """Bind the explicitly selected frozen driver before and after importing it."""
    path, binding_path = checked(path), checked(binding_path)
    binding = read_json(binding_path)
    if binding.get("status") != "EXPECTED" or binding.get("adapter_sha256") != file_hash(path):
        raise ValueError("frozen reference driver lacks its completed matching helper binding")
    sys.path[:0] = [str(path.parent), str(checked(model_root)), str(checked(reference_root))]
    name = "_atlas_frozen_cache_reference_" + file_hash(path)
    spec = importlib.util.spec_from_file_location(name, path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[name] = helper
    spec.loader.exec_module(helper)
    sources = dict(adapter_sha256=file_hash(path), base_adapter_sha256=file_hash(helper.base.__file__),
                   primitive_helper_sha256=file_hash(helper.primitives.__file__),
                   attention_adapter_sha256=file_hash(helper.primitives.attention_ref.__file__))
    if any(binding.get(key) != value for key, value in sources.items()):
        raise ValueError("frozen reference helper source differs from its completed binding")
    return helper, binding, sources


def qualify(helper, binding, sources, args):
    """Reuse the frozen ordinary, softmax, head, and optional native gates."""
    before = helper.validate_helpers(args.validation, args.model_root)
    policy = read_json(args.policy)
    for field, expected in helper.primitives.POLICIES.items():
        if policy.get(field) != expected and not (field == "already_quantized_mask"
                and policy.get(field) == helper.rowspan_softmax.POLICY):
            raise ValueError("selected policy differs from the frozen reference: " + field)
    if policy.get("host_bf16_linear") != helper.BF16_LINEAR_POLICY:
        raise ValueError("cache model head requires the explicit qualified BF16 linear policy")
    mask = read_json(args.mask_validation)
    if (mask.get("status") != "PASS" or mask.get("mismatches") != 0
            or mask.get("selected_policy") != helper.rowspan_softmax.POLICY
            or policy.get("already_quantized_mask") != helper.rowspan_softmax.POLICY
            or mask.get("model_arithmetic_sources") != before
            or mask.get("oracle_sha256") != helper.rowspan_softmax.source_hash()):
        raise ValueError("rowspan80 instruction qualification differs from selected oracle")
    linear = read_json(args.bf16_linear_validation)
    if (linear.get("status") != "PASS" or linear.get("mismatches") != 0
            or linear.get("selected_policy") != helper.BF16_LINEAR_POLICY
            or linear.get("model_arithmetic_sources") != before
            or linear.get("bf16_linear_oracle_sha256") != sha256(inspect.getsource(helper.bf16_linear).encode()).hexdigest()):
        raise ValueError("BF16 linear qualification differs from selected oracle")
    hashes = dict(validation_sha256=file_hash(args.validation), selected_policy_sha256=file_hash(args.policy),
                  mask_validation_sha256=file_hash(args.mask_validation),
                  bf16_linear_validation_sha256=file_hash(args.bf16_linear_validation))
    if (any(binding.get(key) != value for key, value in hashes.items())
            or binding.get("model_arithmetic_sources") != before
            or binding.get("selected_policies") != policy):
        raise ValueError("qualified helper/policy receipts differ from the completed frozen binding")
    engine = helper.primitives.configure_engine(args.engine, 1, args.native_qualification)
    proof = dict(**sources, **hashes, model_arithmetic_sources=before, selected_policies=policy,
                 bf16_linear_policy=helper.BF16_LINEAR_POLICY, engine=engine,
                 reference_sources=helper.primitives.attention_ref.reference_sources(),
                 helper_binding_sha256=file_hash(args.helper_binding))
    import torch
    proof["runtime"] = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__)
    if proof["reference_sources"] != binding.get("reference_sources"):
        raise ValueError("independent reference sources differ from completed helper binding")
    return policy, proof


def read_phase(capsules):
    """Read explicit extracted phase roles, not compiler publications."""
    root = checked(capsules)
    manifest = read_json(root / "manifest.json")
    whole = manifest["whole_model"]
    if whole.get("schema") != "atlas.sequential_cache_phase_fx.v1":
        raise ValueError("independent cache replay requires an explicit extracted cache phase")
    phase = whole["cache_phase"]
    if manifest.get("cache_phase") != phase:
        raise ValueError("top-level cache phase differs from the sequential phase binding")
    if phase.get("binding", {}).get("schema") != "atlas.source_cache_phase_binding.v1":
        raise ValueError("cache phase lacks its declared source ABI binding")
    if phase["binding"].get("model_pt2_sha256") != manifest.get("source_model_sha256"):
        raise ValueError("cache phase source archive binding differs from the capsule manifest")
    abi = phase["binding"]["abi"]
    count = abi["layer_count"]
    order = [f"layer{i}" for i in range(count)] + ["norm", "head"]
    if whole.get("regions") != order or len(whole.get("links", ())) != len(order):
        raise ValueError("cache phase must contain every ordered decoder and tail")
    embedded = {}
    for row in manifest.get("regions", ()):
        if row["kind"] in embedded:
            raise ValueError("duplicate embedded source region")
        embedded[row["kind"]] = row
    sources, parameters = {}, {}
    for name in order:
        source = read_json(root / name / "source.json")
        if source != embedded.get(name):
            raise ValueError("source region differs from embedded source metadata: " + name)
        if source.get("kind") != name or file_hash(root / name / "model.pt2") != source.get("model_sha256"):
            raise ValueError("source capsule model or stage identity differs: " + name)
        if source["input_keys"] != [f"input_{i}" for i in range(len(source["source_boundaries"]))]:
            raise ValueError("source input roster is not positional and complete")
        roles = source.get("cache_roles", {})
        for field, names in (("inputs", source["source_boundaries"]), ("outputs", source["source_outputs"])):
            entries = roles.get(field, ())
            if len(entries) != len(names) or any(row.get("index") != i or row.get("source_node") != names[i]
                                               for i, row in enumerate(entries)):
                raise ValueError("source region cache roles differ from source node order")
        for key, value in source["parameters"].items():
            if key in parameters and parameters[key] != value:
                raise ValueError("source parameter appears with different bytes across stages")
            parameters[key] = value
        sources[name] = source
    if not parameters:
        raise ValueError("cache phase requires an explicit immutable parameter inventory")
    raw_identity = manifest.get("source_provenance", {}).get("raw_capture")
    if not raw_identity or not phase.get("cache_capture_sha256"):
        raise ValueError("cache phase lacks authenticated common raw source identity")
    cache_inputs = {row["index"]: row for row in abi["inputs"] if row["role"] in {"past_key", "past_value"}}
    external = whole.get("external_cache_inputs", ())
    implicit_empty = abi.get("initial_cache_empty") is True
    if implicit_empty and (abi["past_length"] != 0 or len(abi["inputs"]) != 3 or cache_inputs):
        raise ValueError("implicit source cache requires a zero-prefix three-input ABI")
    expected_cache_inputs = 0 if implicit_empty else 2 * count
    if len(cache_inputs) != expected_cache_inputs or len(external) != len(cache_inputs):
        raise ValueError("every nonempty layer requires two declared external cache inputs")
    seen = set()
    for row in external:
        global_index = row["global_input_index"]
        expected = cache_inputs.get(global_index)
        if expected is None or global_index in seen or any(row.get(key) != expected[key]
                for key in ("source_node", "role", "layer", "shape", "dtype")):
            raise ValueError("external cache input differs from the global phase ABI")
        seen.add(global_index)
        source = sources[row["region"]]
        index = row["input_index"]
        if not 0 <= index < len(source["input_keys"]) or source["source_boundaries"][index] != row["boundary"]:
            raise ValueError("external cache input differs from its source region boundary")
        shape = [abi["batch_size"], abi["kv_heads"], abi["past_length"], abi["head_dim"]]
        if row["shape"] != shape or row["dtype"] != "torch.bfloat16":
            raise ValueError("past cache must preserve the declared BF16 prefix geometry")
    if {row["kind"] for row in whole["links"]} != set(order):
        raise ValueError("source activation links do not cover every stage exactly once")
    for row in whole["links"]:
        source = sources[row["kind"]]
        indices = [link["input_index"] for link in row["inputs"]]
        if len(indices) != len(set(indices)):
            raise ValueError("source activation input link is duplicated")
        for link in row["inputs"]:
            index = link["input_index"]
            if (link["region"] not in order[:order.index(row["kind"])] or not 0 <= index < len(source["input_keys"])
                    or source["source_boundaries"][index] != link["boundary"]
                    or not 0 <= link["output_index"] < len(sources[link["region"]]["source_outputs"])
                    or sources[link["region"]]["source_outputs"][link["output_index"]] != link["source_node"]):
                raise ValueError("source activation link differs from exact prior output identity")
    return dict(root=root, manifest=manifest, whole=whole, abi=abi, sources=sources,
                parameters=parameters, raw_identity=raw_identity, phase=phase["phase"])


def cache_output_locations(phase):
    """Global K/V output roles resolve to exact source-region output slots."""
    result = {}
    for entry in phase["abi"]["outputs"][1:]:
        key = (entry["layer"], entry["role"].removeprefix("updated_"))
        if entry["role"] not in {"updated_key", "updated_value"} or key in result:
            raise ValueError("global updated-cache output roles are incomplete or duplicated")
        matches = []
        for name, source in phase["sources"].items():
            for row in source["cache_roles"]["outputs"]:
                if (row.get("global_index") == entry["index"] and row.get("source_node") == entry["source_node"]
                        and row.get("role") == entry["role"] and row.get("layer") == entry["layer"]):
                    matches.append(dict(region=name, output_index=row["index"], global_output_index=entry["index"],
                                        source_node=entry["source_node"], role=entry["role"], layer=entry["layer"]))
        if len(matches) != 1:
            raise ValueError("updated cache output has no unique exact source-region producer")
        result[key] = matches[0]
    if set(result) != {(layer, operand) for layer in range(phase["abi"]["layer_count"]) for operand in ("key", "value")}:
        raise ValueError("updated cache output role roster differs from layer count")
    return result


def parent_caches(current, parent_reference, parent_capsules, proof):
    """Authenticate independent expected bytes; never load a generated actual."""
    if current["abi"]["past_length"] == 0:
        if parent_reference is not None or parent_capsules is not None:
            raise ValueError("empty prefill must not consume a parent reference")
        return {}, dict(parent_reference_receipt_sha256=None, parent_capsule_manifest_sha256=None,
                        parent_source_model_sha256=None)
    if parent_reference is None or parent_capsules is None:
        raise ValueError("nonempty cache requires explicit parent independent reference and parent capsules")
    parent = read_phase(parent_capsules)
    reference_root = checked(parent_reference)
    receipt = read_json(reference_root / "reference.json")
    if (receipt.get("status") != "EXPECTED" or receipt.get("oracle_mode") != "independent_cache_phase"
            or receipt.get("ordinary_framework_kernels_called") is not False
            or receipt.get("compiler_or_emitted_program_inputs") is not False
            or receipt.get("source_case") != 0
            or receipt.get("source_model_sha256") != parent["manifest"]["source_model_sha256"]
            or receipt.get("capsule_manifest_sha256") != file_hash(parent["root"] / "manifest.json")):
        raise ValueError("parent is not a completed independent source cache reference")
    for key in ("adapter_sha256", "base_adapter_sha256", "primitive_helper_sha256", "attention_adapter_sha256",
                "model_arithmetic_sources", "validation_sha256", "selected_policy_sha256",
                "mask_validation_sha256", "bf16_linear_validation_sha256"):
        if receipt.get(key) != proof[key]:
            raise ValueError("parent independent reference arithmetic binding differs: " + key)
    if (receipt.get("raw_source_identity") != parent["raw_identity"] or current["raw_identity"] != parent["raw_identity"]
            or receipt.get("parameter_inventory") != parent["parameters"] or current["parameters"] != parent["parameters"]
            or receipt.get("parameter_inventory_sha256") != record_hash(parent["parameters"])
            or receipt.get("cache_capture_sha256") != parent["whole"]["cache_phase"]["cache_capture_sha256"]
            or current["whole"]["cache_phase"]["cache_capture_sha256"] != receipt.get("cache_capture_sha256")):
        raise ValueError("cache phases do not share authenticated unchanged source parameters")
    for key in ("layer_count", "batch_size", "kv_heads", "head_dim"):
        if current["abi"][key] != parent["abi"][key]:
            raise ValueError("parent cache geometry differs: " + key)
    if current["abi"]["past_length"] != parent["abi"]["total_length"]:
        raise ValueError("parent cache length is not the current phase's complete prefix")
    if [row["stage"] for row in receipt["stages"]] != parent["whole"]["regions"]:
        raise ValueError("parent independent reference did not complete every source region")
    stages = {row["stage"]: row for row in receipt["stages"]}
    caches = {}
    for key, location in cache_output_locations(parent).items():
        stage, index = location["region"], location["output_index"]
        publication = read_json(reference_root / stage / "reference.json")
        if (publication != stages[stage] or publication.get("status") != "EXPECTED" or not publication.get("constant_bytes_identical")
                or publication.get("ordinary_framework_kernels_called") is not False
                or publication.get("semantic_json_or_emitted_program_inputs") is not False
                or publication.get("selected_policy_sha256") != proof["selected_policy_sha256"]):
            raise ValueError("parent expected stage publication differs from the completed reference")
        if publication["source_model_sha256"] != parent["sources"][stage]["model_sha256"]:
            raise ValueError("parent expected stage belongs to another source graph")
        with np.load(reference_root / stage / "expected.npz", allow_pickle=False) as arrays:
            value = np.ascontiguousarray(arrays[f"output_{index}"])
        metadata = publication["outputs"][f"output_{index}"]
        shape = [parent["abi"]["batch_size"], parent["abi"]["kv_heads"],
                 parent["abi"]["total_length"], parent["abi"]["head_dim"]]
        if (value.dtype != np.uint16 or list(value.shape) != shape or metadata.get("logical_dtype") != "bf16"
                or metadata.get("dtype") != value.dtype.str or metadata.get("shape") != shape
                or metadata.get("sha256") != sha256(value.tobytes()).hexdigest()):
            raise ValueError("parent expected BF16 cache words differ from their publication")
        caches[key] = (value.copy(), {**location, "reference_stage": stage,
            "parent_reference_receipt_sha256": file_hash(reference_root / "reference.json"),
            "parent_stage_reference_sha256": file_hash(reference_root / stage / "reference.json"),
            "parent_expected_sha256": file_hash(reference_root / stage / "expected.npz")})
    return caches, dict(parent_reference_receipt_sha256=file_hash(reference_root / "reference.json"),
                        parent_capsule_manifest_sha256=file_hash(parent["root"] / "manifest.json"),
                        parent_source_model_sha256=parent["manifest"]["source_model_sha256"])


def stage_inputs(helper, phase, name, prior, caches, fixture_getter):
    source = phase["sources"][name]
    links = {row["input_index"]: row for row in next(row["inputs"] for row in phase["whole"]["links"] if row["kind"] == name)}
    external = {row["input_index"]: row for row in phase["whole"]["external_cache_inputs"] if row["region"] == name}
    if set(links) & set(external):
        raise ValueError("source input cannot be both an activation seam and external cache")
    inputs, origins = [], []
    for index, key in enumerate(source["input_keys"]):
        if index in links:
            link = links[index]
            if link["region"] not in prior:
                raise ValueError("current-phase independent hidden predecessor is missing")
            value = prior[link["region"]][link["output_index"]]
            origin = dict(reference_stage=link["region"], output_index=link["output_index"], input=key)
        elif index in external:
            entry = external[index]
            operand = entry["role"].removeprefix("past_")
            if phase["abi"]["past_length"] == 0:
                value = helper.Tensor(np.zeros(entry["shape"], np.uint16), "bf16")
                origin = dict(input=key, role=entry["role"], layer=entry["layer"], independent_empty_cache=True)
            else:
                if (entry["layer"], operand) not in caches:
                    raise ValueError("independent parent cache missing; framework fixture fallback is forbidden")
                words, parent_origin = caches[(entry["layer"], operand)]
                value = helper.Tensor(words.copy(), "bf16")
                origin = dict(parent_origin, input=key, role=entry["role"], layer=entry["layer"],
                              independent_parent_cache=True)
            if value.dtype != "bf16" or value.raw.dtype != np.uint16 or list(value.raw.shape) != entry["shape"]:
                raise ValueError("independent cache input differs from source phase geometry")
        else:
            boundary = source["source_boundaries"][index]
            if boundary not in phase["whole"]["fixture_boundary_allowlist"]:
                raise ValueError("unresolved source input cannot fall back to a framework fixture")
            fixture = fixture_getter(key)
            value = helper.raw_tensor(fixture)
            expected = source["tensors"][key]
            if (str(fixture.dtype) != expected["dtype"] or list(value.raw.shape) != expected["shape"]
                    or sha256(value.raw.tobytes()).hexdigest() != expected["sha256"]):
                raise ValueError("source-hashed case0 preparation words changed")
            origin = dict(input=key, source_boundary=boundary, arithmetic="unchanged source preparation fixture")
        inputs.append(value)
        origins.append(origin)
    return inputs, origins


def run_phase(helper, capsules, output, policy, proof, *, parent_reference=None, parent_capsules=None, retain=()):
    import torch
    from safetensors import safe_open

    phase, output = read_phase(capsules), checked(output)
    caches, parent_binding = parent_caches(phase, parent_reference, parent_capsules, proof)
    locations = cache_output_locations(phase)
    output.mkdir(parents=True, exist_ok=False)
    records, prior = [], {}
    record = dict(schema="atlas.independent_original_fx_sequence.v1", status="RUNNING",
        oracle_mode="independent_cache_phase", source_case=0, cache_phase=phase["phase"], **proof, **parent_binding,
        source_model_sha256=phase["manifest"]["source_model_sha256"],
        capsule_manifest_sha256=file_hash(phase["root"] / "manifest.json"),
        cache_capture_sha256=phase["whole"]["cache_phase"]["cache_capture_sha256"],
        raw_source_identity=phase["raw_identity"], parameter_inventory=phase["parameters"],
        parameter_inventory_sha256=record_hash(phase["parameters"]),
        orchestration_sha256=file_hash(__file__), ordinary_framework_kernels_called=False,
        compiler_or_emitted_program_inputs=False, stages=records, generated_execution="UNRUN")
    write_json(output / "reference.json", record)
    try:
        for name in phase["whole"]["regions"]:
            started = time.monotonic()
            source, folder = phase["sources"][name], output / name
            model_file = phase["root"] / name / "model.pt2"
            program = torch.export.load(model_file)
            if (file_hash(model_file) != source["model_sha256"]
                    or read_json(phase["root"] / name / "source.json") != source):
                raise ValueError("source capsule archive or metadata changed before interpretation")
            state = {key: dict(dtype=str(value.dtype), shape=list(value.shape),
                bytes=value.numel() * value.element_size(), sha256=sha256(helper.raw_tensor(value).raw.tobytes()).hexdigest())
                for key, value in program.state_dict.items()}
            if state != source["parameters"]:
                raise ValueError("source capsule parameter inventory differs from saved program")
            with safe_open(phase["root"] / name / "tensors.safetensors", framework="pt", device="cpu") as fixtures:
                inputs, origins = stage_inputs(helper, phase, name, prior, caches, fixtures.get_tensor)
            folder.mkdir()
            incoming = {f"input_{i}": {**helper.info(value.raw), "logical_dtype": value.dtype, "origin": origins[i]}
                        for i, value in enumerate(inputs)}
            np.savez(folder / "runtime_inputs.npz", **{f"input_{i}": value.raw for i, value in enumerate(inputs)})
            write_json(folder / "inputs.json", incoming)
            interpreter = helper.Interpreter(program, policy, engine=proof["engine"]["engine"], threads=1,
                linear_policy=helper.BF16_LINEAR_POLICY, retain=retain,
                progress=lambda node, **rest: print(json.dumps(dict(stage=name, node=node, **rest)), flush=True))
            values = interpreter.run(inputs)
            values = values if isinstance(values, (tuple, list)) else (values,)
            if len(values) != len(source["source_outputs"]) or not all(isinstance(value, helper.Tensor) for value in values):
                raise ValueError("independent source stage output roster differs")
            if any(helper.info(value.raw) != {key: incoming[f"input_{i}"][key] for key in ("dtype", "shape", "sha256")}
                   for i, value in enumerate(inputs)):
                raise ValueError("independent reference mutated an input or parent cache")
            prior[name] = values
            np.savez(folder / "expected.npz", **{**interpreter.saved,
                     **{f"output_{i}": value.raw for i, value in enumerate(values)}})
            stage_record = dict(schema="atlas.independent_original_fx_layer.v1", stage=name, status="EXPECTED",
                source_model_sha256=source["model_sha256"], inputs=origins, constants=interpreter.constants,
                original_lifted_constants={key: value for key, value in interpreter.constants.items() if key in program.state_dict},
                ordinary_framework_kernels_called=False, semantic_json_or_emitted_program_inputs=False,
                selected_policy_sha256=proof["selected_policy_sha256"], boundaries=interpreter.boundaries,
                outputs={f"output_{i}": {**helper.info(value.raw), "logical_dtype": value.dtype}
                         for i, value in enumerate(values)}, constant_bytes_identical=True,
                elapsed_seconds=round(time.monotonic() - started, 6))
            write_json(folder / "reference.json", stage_record)
            records.append(stage_record)
            write_json(output / "reference.json", record)
            del interpreter, program
        if helper.primitives.authenticate_model(proof["model_root"]) != proof["model_arithmetic_sources"]:
            raise ValueError("instruction-model arithmetic sources changed during cache replay")
        sources_after = dict(adapter_sha256=file_hash(helper.__file__), base_adapter_sha256=file_hash(helper.base.__file__),
            primitive_helper_sha256=file_hash(helper.primitives.__file__),
            attention_adapter_sha256=file_hash(helper.primitives.attention_ref.__file__))
        if any(proof[key] != value for key, value in sources_after.items()):
            raise ValueError("frozen reference helper source changed during cache replay")
        for location in locations.values():
            value = prior[location["region"]][location["output_index"]]
            expected_shape = [phase["abi"]["batch_size"], phase["abi"]["kv_heads"],
                              phase["abi"]["total_length"], phase["abi"]["head_dim"]]
            if value.dtype != "bf16" or value.raw.dtype != np.uint16 or list(value.raw.shape) != expected_shape:
                raise ValueError("independent updated cache differs from the declared BF16 output geometry")
        record.update(status="EXPECTED", cache_outputs=[{**location,
            **helper.info(prior[location["region"]][location["output_index"]].raw)} for location in locations.values()],
            whole_model_device_exactness="PENDING separate generated cache execution comparison")
        write_json(output / "reference.json", record)
        return record
    except Exception as error:
        record.update(status="REFUSED", error=str(error))
        write_json(output / "reference.json", record)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    for name in ("reference-driver", "helper-binding", "model-root", "reference-root", "policy", "validation",
                 "mask-validation", "bf16-linear-validation", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--capsules", type=Path)
    parser.add_argument("--parent-reference", type=Path)
    parser.add_argument("--parent-capsules", type=Path)
    parser.add_argument("--engine", choices=("numpy", "native"), default="numpy")
    parser.add_argument("--native-qualification", type=Path)
    parser.add_argument("--retain", action="append", default=[])
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.qualify_only and args.capsules is None:
        parser.error("--capsules is required for phase replay")
    if checked(args.out).exists():
        parser.error("a fresh output path is required")
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    helper, binding, sources = load_reference_driver(args.reference_driver, args.helper_binding,
                                                    args.model_root, args.reference_root)
    from atlas_ref.ops import register
    register()
    policy, proof = qualify(helper, binding, sources, args)
    proof["model_root"] = str(checked(args.model_root))
    if args.qualify_only:
        checked(args.out).mkdir(parents=True)
        write_json(args.out / "reference_qualification.json", dict(status="QUALIFIED_NUMERICAL_HELPERS", **proof,
            orchestration_sha256=file_hash(__file__), graph_arithmetic_executed=False, generated_execution="UNRUN"))
    else:
        run_phase(helper, args.capsules, args.out, policy, proof, parent_reference=args.parent_reference,
                  parent_capsules=args.parent_capsules, retain=args.retain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
