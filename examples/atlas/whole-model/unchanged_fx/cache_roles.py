"""Source-node identities for the tensor-only BF16 cache phase ABI.

These helpers inspect an existing ExportedProgram; they do not run its graph or
alter source operators. Value-dependent position/mask admission remains the
responsibility of cache_source.validate_phase before capture and execution.
"""
from __future__ import annotations

import torch


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be a static integer >= {minimum}")
    return value


def _tensor_entry(node, index, role, layer=None):
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"source ABI node lacks tensor metadata: {node.name}")
    shape = [_integer(dim, f"{node.name} dimension", 0) for dim in value.shape]
    entry = dict(index=index, role=role, source_node=node.name,
                 shape=shape, dtype=str(value.dtype))
    if layer is not None:
        entry["layer"] = layer
    return entry


def derive_cache_phase_abi(program, *, layer_count, past_length, query_length,
                           kv_heads=None, head_dim=None, batch_size=None):
    """Validate fixed phase geometry and bind roles to actual export names.

    TensorCacheLogits has ids, positions, mask, and ordered past K/V inputs.
    TensorPrefillLogits omits past tensors at P=0. Both return logits and ordered
    updated K/V. Lifted state is excluded using graph_signature.
    """
    layer_count = _integer(layer_count, "layer_count")
    past_length = _integer(past_length, "past_length", 0)
    query_length = _integer(query_length, "query_length")
    nodes = {node.name: node for node in program.graph.nodes}
    signature = program.graph_signature
    input_specs = [spec for spec in signature.input_specs if spec.kind.name == "USER_INPUT"]
    output_specs = list(signature.output_specs)
    if any(spec.kind.name != "USER_OUTPUT" for spec in output_specs):
        raise ValueError("cache phase must have tensor user outputs only, with no mutations")
    implicit_empty = len(input_specs) == 3 and past_length == 0
    if ((len(input_specs) != 3 + 2 * layer_count and not implicit_empty)
            or len(output_specs) != 1 + 2 * layer_count):
        raise ValueError("cache phase input/output roster differs from expected layer count")

    def resolve(spec, input_node=False):
        name = getattr(spec.arg, "name", None)
        if name not in nodes or (input_node and nodes[name].op != "placeholder"):
            raise ValueError("cache phase signature must identify source tensor nodes")
        return nodes[name]

    inputs = [resolve(spec, True) for spec in input_specs]
    outputs = [resolve(spec) for spec in output_specs]
    graph_outputs = next(node for node in program.graph.nodes if node.op == "output").args[0]
    if not isinstance(graph_outputs, (tuple, list)) or tuple(graph_outputs) != tuple(outputs):
        raise ValueError("source graph output identities differ from graph_signature")
    if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
        raise ValueError("cache phase tensor roles must have distinct source identities")
    in_entries = [_tensor_entry(node, index, role) for index, (node, role) in enumerate(zip(
        inputs[:3], ("input_ids", "position_ids", "attention_mask"), strict=True))]
    out_entries = [_tensor_entry(outputs[0], 0, "logits")]
    for layer in range(layer_count):
        for offset, operand in enumerate(("key", "value")):
            if not implicit_empty:
                index = 3 + 2 * layer + offset
                in_entries.append(_tensor_entry(inputs[index], index, f"past_{operand}", layer))
            index = 1 + 2 * layer + offset
            out_entries.append(_tensor_entry(outputs[index], index, f"updated_{operand}", layer))

    ids = in_entries[0]
    if len(ids["shape"]) != 2 or ids["dtype"] != "torch.int64":
        raise ValueError("source ids require int64 [B,Q]")
    batch = _integer(ids["shape"][0], "batch_size")
    if ids["shape"][1] != query_length or (batch_size is not None and batch_size != batch):
        raise ValueError("source ids differ from expected batch/query geometry")
    first = out_entries[1] if implicit_empty else in_entries[3]
    if len(first["shape"]) != 4:
        raise ValueError("source cache requires [B,Hkv,P,D]")
    heads, dimension = first["shape"][1], first["shape"][3]
    _integer(heads, "kv_heads")
    _integer(dimension, "head_dim")
    if (kv_heads is not None and kv_heads != heads) or (head_dim is not None and head_dim != dimension):
        raise ValueError("source cache differs from expected pre-GQA head geometry")

    def require(entry, shape, dtype):
        if entry["shape"] != list(shape) or entry["dtype"] != dtype:
            raise ValueError(f"source {entry['role']} dtype/geometry differs: {entry['source_node']}")

    require(in_entries[1], (batch, query_length), "torch.int64")
    require(in_entries[2], (batch, 1, query_length, past_length + query_length), "torch.bfloat16")
    logits = out_entries[0]
    if (len(logits["shape"]) != 3 or logits["shape"][:2] != [batch, query_length]
            or logits["shape"][2] < 1 or logits["dtype"] != "torch.bfloat16"):
        raise ValueError("source logits require BF16 [B,Q,V]")
    for entry in in_entries[3:]:
        require(entry, (batch, heads, past_length, dimension), "torch.bfloat16")
    for entry in out_entries[1:]:
        require(entry, (batch, heads, past_length + query_length, dimension), "torch.bfloat16")
    result = dict(schema="atlas.tensor_cache_phase_abi.v1", layer_count=layer_count,
                batch_size=batch, past_length=past_length, query_length=query_length,
                total_length=past_length + query_length, kv_heads=heads, head_dim=dimension,
                cache_representation="bf16_post_rope_pre_gqa", inputs=in_entries, outputs=out_entries,
                value_admission="cache_source.validate_phase_required")
    if implicit_empty:
        result["initial_cache_empty"] = True
        result["value_admission"] = "cache_source.validate_prefill_phase_required"
    return result


def prioritize_stage_output(outputs, abi):
    """Move a unique non-cache stage result first, preserving all node objects."""
    outputs = tuple(outputs)
    cache_names = {entry["source_node"] for entry in abi["outputs"][1:]}
    stage = [node for node in outputs if node.name not in cache_names]
    if len(stage) != 1:
        return outputs
    return (stage[0], *(node for node in outputs if node is not stage[0]))


def bind_region_cache_roles(abi, record):
    """Return role arrays aligned with an extracted source.json's saved order.

    Existing arrays and tensor files are not reordered. A decoder's sole
    non-cache output is the hidden stage result; other unbound tensors keep an
    explicit source_tensor role until extraction establishes their provenance.
    """
    known = {entry["source_node"]: entry for entry in (*abi["inputs"], *abi["outputs"])}
    result = {}
    for field, label in (("source_boundaries", "inputs"), ("source_outputs", "outputs")):
        names = record[field]
        if len(set(names)) != len(names):
            raise ValueError("region source tensor identities must be unique")
        entries = []
        for index, name in enumerate(names):
            original = known.get(name)
            entry = dict(index=index, source_node=name, role="source_tensor")
            if original is not None:
                entry.update({key: value for key, value in original.items() if key != "index"})
                entry["global_index"] = original["index"]
            entries.append(entry)
        result[label] = entries
    kind = record.get("kind", "")
    unknown = [entry for entry in result["outputs"] if entry["role"] == "source_tensor"]
    if kind.startswith("layer") and kind[5:].isascii() and kind[5:].isdigit() and len(unknown) == 1:
        layer = int(kind[5:])
        if layer >= abi["layer_count"]:
            raise ValueError("region decoder index exceeds source layer count")
        unknown[0].update(role="hidden_state", layer=layer)
    return result
