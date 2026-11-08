"""Extract unchanged quantized contraction capsules from an upstream FX bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from cache_roles import bind_region_cache_roles, derive_cache_phase_abi, prioritize_stage_output
from cache_source import validate_phase


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_phase_inputs(inputs, abi):
    """Admit the exact eager boundary, including implicit-empty prefill."""
    if len(inputs) != len(abi["inputs"]):
        raise ValueError("source phase fixture input roster differs from ABI")
    if abi.get("initial_cache_empty") is True:
        from cache_source import validate_prefill_phase
        validate_prefill_phase(*inputs)
    else:
        validate_phase(inputs[0], inputs[1], inputs[2], inputs[3:])


def load_cache_phase(program, bundle, manifest, abi_path, receipt_path):
    """Authenticate a declared source phase and recompute its exact export ABI."""
    if abi_path is None and receipt_path is None:
        if len(program.graph_signature.user_inputs) != 1 or (bundle / "cache_phase_abi.json").exists():
            raise ValueError("cache source extraction requires --cache-abi and --cache-capture")
        return None
    if abi_path is None or receipt_path is None:
        raise ValueError("cache source extraction requires both --cache-abi and --cache-capture")
    binding = json.loads(abi_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (binding.get("schema") != "atlas.source_cache_phase_binding.v1"
            or receipt.get("schema") != "atlas.source_cache_capture.v1"
            or receipt.get("status") != "SOURCE_CACHE_CAPTURED"):
        raise ValueError("completed source cache capture and phase binding are required")
    provenance = manifest.get("provenance", {})
    phase = provenance.get("phase")
    captured = receipt.get("phases", {}).get(phase, {})
    if captured.get("status") != "SOURCE_CAPTURED":
        raise ValueError("phase is not authenticated by the completed cache capture")
    hashes = {"model_pt2_sha256": file_hash(bundle / "model.pt2"),
              "manifest_sha256": file_hash(bundle / "fx_manifest.json"),
              "cache_phase_abi_sha256": file_hash(abi_path),
              "verification_sha256": file_hash(bundle / "verification.json"),
              "fresh_process_verification_sha256": file_hash(bundle / "fresh_process_verification.json")}
    if any(captured.get(key) != value for key, value in hashes.items()):
        raise ValueError("cache capture phase file binding differs")
    if (binding.get("model_pt2_sha256") != hashes["model_pt2_sha256"]
            or binding.get("source_fx_manifest_sha256") != hashes["manifest_sha256"]
            or manifest["files"]["model.pt2"]["sha256"] != hashes["model_pt2_sha256"]):
        raise ValueError("cache phase sidecar source binding differs")
    if (provenance.get("boundary") != "causal_lm_tensor_bf16_cache"
            or provenance.get("raw_capture") != receipt.get("raw_identity")):
        raise ValueError("cache phase original source identity differs")
    expected = binding.get("abi", {})
    names = ("layer_count", "past_length", "query_length", "kv_heads", "head_dim", "batch_size")
    if any(name not in expected for name in names):
        raise ValueError("cache phase ABI is incomplete")
    actual = derive_cache_phase_abi(program, **{name: expected[name] for name in names})
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("cache phase ABI differs from actual exported program")
    if (captured.get("input_count") != len(actual["inputs"])
            or captured.get("output_count") != len(actual["outputs"])
            or provenance.get("past_length") != actual["past_length"]
            or provenance.get("seqlen") != actual["query_length"]
            or provenance.get("total_length") != actual["total_length"]):
        raise ValueError("cache phase capture geometry differs")
    return dict(phase=phase, binding=binding, cache_phase_abi_sha256=hashes["cache_phase_abi_sha256"],
                cache_capture_sha256=file_hash(receipt_path), generated_execution="UNRUN")


def sequential_region_manifest(regions, nodes, cache_phase=None):
    """Resolve generated seams; keep declared temporal cache inputs separate."""
    producers = {name: (region["kind"], index) for region in regions
                 for index, name in enumerate(region["source_outputs"])}
    node_by_name = {node.name: node for node in nodes}
    aliases = {}
    for region in regions:
        for name in region["source_boundaries"]:
            node = node_by_name[name]
            while node.name not in producers and node.op == "call_function" and str(node.target) in {
                    "aten.alias.default", "aten.detach.default"}:
                original = node.name
                node = node.args[0]
                aliases[original] = node.name
    cache_abi = None if cache_phase is None else cache_phase["binding"]["abi"]
    cache_inputs = {} if cache_abi is None else {
        entry["source_node"]: entry for entry in cache_abi["inputs"]
        if entry["role"] in {"past_key", "past_value"}}
    if cache_abi is not None:
        if cache_abi.get("initial_cache_empty") is True:
            if cache_abi["past_length"] != 0 or len(cache_abi["inputs"]) != 3 or cache_inputs:
                raise ValueError("implicit-empty prefill requires P=0 and exactly three source inputs")
        elif len(cache_inputs) != 2 * cache_abi["layer_count"]:
            raise ValueError("source phase requires every declared past K/V input")
    links, external_cache_inputs, available = [], [], set()
    initial_boundaries = set()
    for boundary in regions[0]["source_boundaries"]:
        source = boundary
        while source in aliases:
            source = aliases[source]
        if source not in cache_inputs:
            initial_boundaries.add(boundary)
    for region in regions:
        region_links = []
        for index, boundary in enumerate(region["source_boundaries"]):
            source = boundary
            while source in aliases:
                source = aliases[source]
            if source in producers:
                if source not in available:
                    raise ValueError(f"nonsequential region dependency: {source}")
                producer, output_index = producers[source]
                region_links.append(dict(input_index=index, boundary=boundary, source_node=source,
                                         region=producer, output_index=output_index))
            elif source in cache_inputs:
                original = cache_inputs[source]
                external_cache_inputs.append(dict(region=region["kind"], input_index=index, boundary=boundary,
                    source_node=source, global_input_index=original["index"], role=original["role"],
                    layer=original["layer"], shape=original["shape"], dtype=original["dtype"]))
            elif boundary not in initial_boundaries:
                raise ValueError(f"unresolved generated boundary cannot use framework fixture: {region['kind']}:{boundary}")
        links.append(dict(kind=region["kind"], inputs=region_links))
        available.update(region["source_outputs"])
    if any(not item["inputs"] for item in links[1:]):
        raise ValueError("a whole-model region has no generated activation predecessor")
    if cache_phase is not None:
        if set(cache_inputs) != {entry["source_node"] for entry in external_cache_inputs}:
            raise ValueError("cache phase extraction omitted a declared past K/V input")
        if any(entry["source_node"] not in producers
               for entry in cache_phase["binding"]["abi"]["outputs"][1:]):
            raise ValueError("cache phase extraction omitted a declared updated K/V output")
    result = dict(schema="atlas.sequential_unchanged_fx.v1", regions=[region["kind"] for region in regions],
                  links=links, aliases=aliases, fixture_boundary_allowlist=sorted(initial_boundaries),
                  initial_boundary_role="source_embedding_mask_and_rope_capture",
                  embedding_execution="framework_source_capture", framework_activation_substitution=False)
    if cache_phase is not None:
        result.update(schema="atlas.sequential_cache_phase_fx.v1", cache_phase=cache_phase,
                      external_cache_inputs=external_cache_inputs,
                      generated_cache_authentication="UNRUN")
    return result


def tensor_record(value):
    payload = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def fixture_inputs(fixtures, case_index=0):
    """Recover positional arguments numerically, including cache ABIs over ten inputs."""
    prefix = f"case_{case_index}.input_"
    selected = {}
    for key, value in fixtures.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        if not suffix.isascii() or not suffix.isdigit() or str(int(suffix)) != suffix:
            raise ValueError("fixture input index is not canonical")
        selected[int(suffix)] = value
    if not selected or sorted(selected) != list(range(len(selected))):
        raise ValueError("fixture input indices must be contiguous from zero")
    return tuple(selected[index] for index in range(len(selected)))


class Capture(torch.fx.Interpreter):
    def __init__(self, module):
        super().__init__(module, garbage_collect_values=False)
        self.calls = []

    def run_node(self, node):
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        result = super().run_node(node)
        if node.op == "call_function" and str(node.target).startswith("atlas."):
            self.calls.append((node, args, kwargs, result))
        return result


class LinearCapsule(torch.nn.Module):
    def __init__(self, args, quantization):
        super().__init__()
        self.register_buffer("weight_codes", args[2].detach().clone())
        self.register_buffer("weight_exp", args[3].detach().clone())
        self.register_buffer("bias", None if args[4] is None else args[4].detach().clone())
        self.out_dtype, self.weight_format = args[5:7]
        self.quantization = tuple(quantization)

    def forward(self, activation):
        codes, exponent = torch.ops.atlas.quantize_act.default(activation, *self.quantization)
        return torch.ops.atlas.qlinear.default(codes, exponent, self.weight_codes,
                                             self.weight_exp, self.bias,
                                             self.out_dtype, self.weight_format)


class ContractionCapsule(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, lhs_codes, lhs_exp, rhs_codes, rhs_exp):
        if self.kind == "qk_matmul":
            return torch.ops.atlas.qk_matmul.default(lhs_codes, lhs_exp, rhs_codes, rhs_exp)
        return torch.ops.atlas.pv_matmul.default(lhs_codes, lhs_exp, rhs_codes, rhs_exp)


def regional_capture_metadata(metadata):
    """Start a new FX capture namespace and archive its exact parent identities."""
    fresh = dict(metadata)
    parent = {key: fresh.pop(key) for key in ("_m2m_node_id", "_m2m_origin_ids") if key in fresh}
    custom = dict(fresh.get("custom") or {})
    captured = {key: custom.pop(key) for key in
                ("m2m_node_id", "m2m_lineage", "m2m_deferred_constant_users") if key in custom}
    if captured:
        parent["custom"] = captured
        if custom:
            fresh["custom"] = custom
        else:
            fresh.pop("custom", None)
    # The receipt is independent of the source's mutable metadata containers.
    return fresh, json.loads(json.dumps(parent, allow_nan=False))


def save_region(capture, output, label, outputs, selected, cache_abi=None):
    """Copy source operations and inputs into a new regional capture namespace."""
    if cache_abi is not None:
        outputs = prioritize_stage_output(outputs, cache_abi)
    graph, mapping, boundaries = torch.fx.Graph(), {}, []
    parent_nodes = []

    def copy(node):
        if node in mapping:
            return mapping[node]
        if node not in selected and node.op != "get_attr":
            value = capture.env[node]
            if not isinstance(value, torch.Tensor):
                return value
            replacement = graph.placeholder(f"input_{len(boundaries)}")
            boundaries.append((node, value.detach().clone()))
        else:
            replacement = graph.node_copy(node, copy)
        replacement.meta, identity = regional_capture_metadata(node.meta)
        parent_nodes.append(dict(source_node=node.name, source_op=node.op,
                                 source_target=str(node.target), region_node=replacement.name,
                                 role="boundary" if node not in selected and node.op != "get_attr" else "copied",
                                 identity_metadata=identity))
        mapping[node] = replacement
        return replacement

    graph.output(tuple(copy(node) for node in outputs))
    source_hash = getattr(capture, "source_model_sha256", None)
    if any(row["identity_metadata"] for row in parent_nodes) and (
            not isinstance(source_hash, str) or len(source_hash) != 64
            or any(char not in "0123456789abcdef" for char in source_hash)):
        raise ValueError("parent capture identities require an authenticated source model SHA-256")
    parent_capture = dict(schema="atlas.regional_capture_parent.v1", source_model_sha256=source_hash,
                          region_graph_sha256=hashlib.sha256(str(graph).encode()).hexdigest(),
                          node_map=parent_nodes)
    module = torch.fx.GraphModule(capture.module, graph).eval()
    inputs = tuple(value for _, value in boundaries)
    expected = tuple(capture.env[node] for node in outputs)
    with torch.no_grad():
        actual = module(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    exported = torch.export.export(module, inputs, strict=True)
    source_state = {**capture.source_program.state_dict, **capture.source_program.constants}
    for key, value in exported.state_dict.items():
        if key not in source_state or tensor_record(value) != tensor_record(source_state[key]):
            raise ValueError(f"source region parameter changed: {key}")
    folder = output / label
    folder.mkdir(exist_ok=True)
    torch.export.save(exported, folder / "model.pt2")
    tensors = {f"input_{index}": value for index, (_, value) in enumerate(boundaries)}
    tensors.update({f"framework_output_{index}": value for index, value in enumerate(expected)})
    save_file({key: value.detach().contiguous().clone() for key, value in tensors.items()},
              folder / "tensors.safetensors")
    (folder / "graph.txt").write_text(str(exported.graph))
    record = {"kind": label, "source_nodes": [node.name for node in capture.module.graph.nodes if node in selected and node in mapping],
              "source_outputs": [node.name for node in outputs],
              "source_boundaries": [node.name for node, _ in boundaries],
              "input_keys": [f"input_{index}" for index in range(len(boundaries))],
              "tensors": {key: tensor_record(value) for key, value in tensors.items()},
              "parameters": {key: tensor_record(value) for key, value in exported.state_dict.items()},
              "model_sha256": hashlib.sha256((folder / "model.pt2").read_bytes()).hexdigest(),
              "framework_replay_exact": True, "device_comparison": "PENDING",
              "parent_capture": parent_capture}
    if cache_abi is not None:
        record["cache_roles"] = bind_region_cache_roles(cache_abi, record)
    (folder / "source.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"stage": "REGION_EXTRACTED", "kind": label}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--register-module", default="atlas_fx_ops")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--whole-model", action="store_true", help="extract all decoder layers, final norm, and LM head")
    parser.add_argument("--cache-abi", type=Path, help="authenticated cache_phase_abi.json sidecar")
    parser.add_argument("--cache-capture", type=Path, help="completed source cache_capture.json receipt")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("threads must be in 1..4")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    importlib.import_module(args.register_module)
    print(json.dumps({"stage": "LOAD_BUNDLE"}), flush=True)
    manifest = json.loads((args.bundle / "fx_manifest.json").read_text())
    for filename, record in manifest["files"].items():
        actual = hashlib.sha256((args.bundle / filename).read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"upstream bundle hash mismatch: {filename}")
    program = torch.export.load(args.bundle / "model.pt2")
    cache_phase = load_cache_phase(program, args.bundle, manifest, args.cache_abi, args.cache_capture)
    cache_abi = None if cache_phase is None else cache_phase["binding"]["abi"]
    fixtures = load_file(args.bundle / "fixtures.safetensors")
    inputs = fixture_inputs(fixtures)
    if cache_abi is not None:
        validate_phase_inputs(inputs, cache_abi)
    print(json.dumps({"stage": "CAPTURE_SOURCE"}), flush=True)
    capture = Capture(program.module())
    capture.source_program = program
    capture.source_model_sha256 = manifest["files"]["model.pt2"]["sha256"]
    with torch.no_grad():
        result = capture.run(*inputs)
    outputs = result if isinstance(result, (tuple, list)) else (result,)
    for index, value in enumerate(outputs):
        if not torch.equal(value, fixtures[f"case_0.output_{index}"]):
            raise ValueError("upstream full graph no longer matches fixture")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"schema": "atlas.unchanged_fx_capsules.v1",
                  "source_model_sha256": manifest["files"]["model.pt2"]["sha256"],
                  "source_fixture_sha256": manifest["files"]["fixtures.safetensors"]["sha256"],
                  "source_provenance": manifest.get("provenance", {}),
                  "source_full_graph_fixture_exact": True,
                  "weight_decode_or_requantization": False, "capsules": []}
    if cache_phase is not None:
        provenance["cache_phase"] = cache_phase
    selected = set()
    calls_by_name = {node.name: (node, values, kwargs, value)
                     for node, values, kwargs, value in capture.calls}
    for node, values, kwargs, value in capture.calls:
        kind = str(node.target).split(".")[1]
        if kind not in {"qlinear", "qk_matmul", "pv_matmul"} or kind in selected:
            continue
        if kwargs or len(values) != (7 if kind == "qlinear" else 4):
            raise ValueError(f"capsule extraction requires the unfolded positional {kind} source ABI")
        selected.add(kind)
        folder = args.output / kind
        folder.mkdir(exist_ok=True)
        tensors = {"lhs_codes": values[0], "lhs_exp": values[1],
                   "rhs_codes": values[2], "rhs_exp": values[3],
                   "framework_output": value}
        if kind == "qlinear":
            producer = node.args[0].args[0]
            quant_node, quant_args, quant_kwargs, quant_result = calls_by_name[producer.name]
            if quant_kwargs:
                raise ValueError("capsule extraction cannot drop quantize_act keyword arguments")
            assert str(quant_node.target) == "atlas.quantize_act.default"
            assert torch.equal(quant_result[0], values[0])
            assert torch.equal(quant_result[1], values[1])
            tensors["activation"] = quant_args[0]
            if values[4] is not None:
                tensors["bias"] = values[4]
            module = LinearCapsule(values, quant_args[1:]).eval()
            capsule_inputs = (quant_args[0],)
        else:
            module = ContractionCapsule(kind).eval()
            capsule_inputs = tuple(values[:4])
        with torch.no_grad():
            check = module(*capsule_inputs)
        assert torch.equal(check, value)
        exported = torch.export.export(module, capsule_inputs, strict=True)
        torch.export.save(exported, folder / "model.pt2")
        save_file({key: tensor.detach().contiguous().clone() for key, tensor in tensors.items()},
                  folder / "tensors.safetensors")
        (folder / "graph.txt").write_text(str(exported.graph))
        record = {"kind": kind, "source_node": node.name,
                  "source_operand_nodes": [item.name if isinstance(item, torch.fx.Node) else item
                                           for item in node.args],
                  "tensors": {key: tensor_record(tensor) for key, tensor in tensors.items()},
                  "model_sha256": hashlib.sha256((folder / "model.pt2").read_bytes()).hexdigest(),
                  "framework_replay_exact": True,
                  "device_comparison": "PENDING"}
        (folder / "source.json").write_text(json.dumps(record, indent=2) + "\n")
        provenance["capsules"].append(record)
    if selected != {"qlinear", "qk_matmul", "pv_matmul"}:
        raise ValueError(f"missing expected graph contraction kinds: {selected}")
    attention_quantizers = {node for node, values, _, _ in capture.calls
                            if str(node.target) == "atlas.quantize_act.default" and len(values) >= 7
                            and values[6] in {f"model.layers.0.self_attn:{role}" for role in ("q", "k", "v")}}
    assert len(attention_quantizers) == 3
    memo = {}

    def depends(node):
        if node not in memo:
            memo[node] = node in attention_quantizers or any(depends(parent) for parent in node.all_input_nodes)
        return memo[node]

    attention_nodes = {node for node in capture.module.graph.nodes if depends(node)}
    pv_output = next(node for node, _, _, _ in capture.calls if str(node.target) == "atlas.pv_matmul.default")
    provenance["regions"] = [save_region(capture, args.output, "attention", (pv_output,), attention_nodes, cache_abi)]
    layer_nodes = {node for node in capture.module.graph.nodes
                   if any(path == "model.model.layers.0" for path, _ in node.meta.get("nn_module_stack", {}).values())}
    layer_outputs = tuple(node for node in layer_nodes if any(user not in layer_nodes for user in node.users))
    # Start with graph order; cached save_region prioritizes a unique hidden result.
    layer_outputs = tuple(node for node in capture.module.graph.nodes if node in layer_outputs)
    provenance["regions"].append(save_region(capture, args.output, "layer0", layer_outputs, layer_nodes, cache_abi))
    if args.whole_model:
        regions = provenance["regions"][1:]
        tail_regions = []
        for label, path in (("norm", "model.model.norm"), ("head", "model.lm_head")):
            selected_nodes = {node for node in capture.module.graph.nodes
                              if any(module_path == path for module_path, _ in node.meta.get("nn_module_stack", {}).values())}
            if not selected_nodes:
                raise ValueError(f"missing final {label} region at {path}")
            observed = tuple(node for node in capture.module.graph.nodes if node in selected_nodes
                             and any(user not in selected_nodes for user in node.users))
            tail_regions.append(save_region(capture, args.output, label, observed, selected_nodes, cache_abi))
        for index in range(1, 22 if cache_abi is None else cache_abi["layer_count"]):
            selected_nodes = {node for node in capture.module.graph.nodes
                              if any(path == f"model.model.layers.{index}"
                                     for path, _ in node.meta.get("nn_module_stack", {}).values())}
            if not selected_nodes:
                raise ValueError(f"missing decoder layer {index}")
            observed = tuple(node for node in capture.module.graph.nodes if node in selected_nodes
                             and any(user not in selected_nodes for user in node.users))
            regions.append(save_region(capture, args.output, f"layer{index}", observed, selected_nodes, cache_abi))
        regions.extend(tail_regions)
        provenance["whole_model"] = sequential_region_manifest(regions, capture.module.graph.nodes, cache_phase)
        if cache_phase is not None:
            provenance["regions"] = regions

    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": "EXTRACTED", "kinds": sorted(selected),
                      "source_model_sha256": provenance["source_model_sha256"]}))


if __name__ == "__main__":
    main()
