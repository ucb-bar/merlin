"""TinyLlama raw-state loader for the external adapter capture worker.

Select ATLAS_EXPORT_DIR, ATLAS_ARCHITECTURE, and ATLAS_INPUTS_SAFETENSORS.
Set ATLAS_EXPECTED_CONTRACT_SHA256 and ATLAS_EXPECTED_POLICY_SHA256 from the
authenticated source receipt. The input archive contains input_ids. Source
modules must be on PYTHONPATH.
Invoke the worker with --already-quantized --quant-adapter atlas_fp8 and the
export's manifest.json and policy.json as --contract and --policy, with the
same expected values as --expected-contract-sha256 and --expected-policy-sha256.

For a live layer-0 capture, set ATLAS_CAPTURE_BOUNDARY=layer0 and point
ATLAS_INPUTS_SAFETENSORS at its original boundary tensor archive. Also select
ATLAS_BOUNDARY_SOURCE_JSON and its frozen ATLAS_EXPECTED_BOUNDARY_SOURCE_SHA256.
Only the saved tensor words and their authenticated ABI are read; the decoder
executes directly from the reconstructed source modules, without a saved graph.
An authenticated six-input cache boundary selects the existing tensor-cache
append semantics and returns hidden state, updated K, and updated V.
"""
from __future__ import annotations

import os
import json
from hashlib import sha256
from pathlib import Path

from atlas_mlir_native.capture import normalize_capture, prepare_model


def layer0_boundary(model, inputs, source, *, tensor_record):
    """Wrap the original decoder, preserving its source parameter paths."""
    import torch
    from torch import nn

    if source.get("kind") != "layer0":
        raise ValueError("layer0 requires the original decoder source boundary")
    cache_roles = source.get("cache_roles")
    cached = cache_roles is not None
    keys = source.get("input_keys")
    if keys != [f"input_{index}" for index in range(6 if cached else 4)]:
        raise ValueError("layer0 requires hidden, cosine, sine, and additive mask")
    if cached:
        for field, expected in (("inputs", ["source_tensor", "source_tensor", "source_tensor",
                                            "past_key", "attention_mask", "past_value"]),
                                ("outputs", ["hidden_state", "updated_key", "updated_value"])):
            entries = cache_roles.get(field, [])
            if ([entry.get("role") for entry in entries] != expected
                    or [entry.get("index") for entry in entries] != list(range(len(expected)))
                    or any(entry.get("layer") != 0 for entry in entries
                           if entry.get("role") not in ("source_tensor", "attention_mask"))):
                raise ValueError("layer0 cache roles differ from the original ordered source ABI")
    if set(inputs) != set(source["tensors"]):
        raise ValueError("boundary archive tensor roster differs from selected source")
    for key, value in inputs.items():
        if tensor_record(value) != source["tensors"][key]:
            raise ValueError(f"boundary tensor words differ from selected source: {key}")
    arguments = tuple(inputs[key] for key in keys)
    hidden, cosine, sine = arguments[:3]
    mask = arguments[4 if cached else 3]
    config = model.config
    batch, query, width = hidden.shape
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    past = arguments[3].shape[2] if cached else 0
    if (any(value.dtype != torch.bfloat16 for value in arguments)
            or width != config.hidden_size or cosine.shape != (batch, query, head_dim)
            or sine.shape != cosine.shape or mask.shape != (batch, 1, query, past + query)):
        raise ValueError("layer0 boundary differs from the original BF16 decoder geometry")
    if cached:
        cache_shape = (batch, config.num_key_value_heads, past, head_dim)
        if past < 1 or arguments[3].shape != cache_shape or arguments[5].shape != cache_shape:
            raise ValueError("layer0 cache requires positive-length BF16 post-RoPE, pre-GQA K/V")

    class Layer0(nn.Module):
        def __init__(self):
            super().__init__()
            # Keep only this decoder in the registered hierarchy. Its exact
            # model.model.layers.0 names match the original CausalLogits export.
            self.model = nn.Module()
            self.model.model = nn.Module()
            self.model.model.layers = nn.ModuleList([model.model.layers[0]])

        def forward(self, hidden, cosine, sine, attention_mask):
            return self.model.model.layers[0](hidden, attention_mask=attention_mask,
                                              position_embeddings=(cosine, sine))

    if cached:
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location("_adapter_cache_source", Path(__file__).with_name("cache_source.py"))
        cache_source = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cache_source
        spec.loader.exec_module(cache_source)

        class CachedLayer0(Layer0):
            def forward(self, hidden, cosine, sine, past_key, attention_mask, past_value):
                cache = cache_source._AppendTensorCache(((past_key, past_value),))
                hidden = self.model.model.layers[0](hidden, attention_mask=attention_mask,
                    position_embeddings=(cosine, sine), past_key_values=cache, use_cache=True)
                return (hidden, *cache.tensors())

        boundary = CachedLayer0().eval()
    else:
        boundary = Layer0().eval()
    state = list(boundary.named_parameters()) + list(boundary.named_buffers())
    actual_state = {name: tensor_record(value) for name, value in state if value.device.type != "meta"}
    if actual_state != source["parameters"]:
        raise ValueError("live layer0 parameters differ from selected source boundary")
    return boundary, arguments


def get_model_and_inputs():
    import algos
    import fx_export
    import load_quantized
    import ref_attention
    import ref_profile
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    import importlib.util
    spec = importlib.util.spec_from_file_location("_adapter_raw_capture", Path(__file__).with_name("raw_capture.py"))
    raw_capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw_capture)
    inventory = raw_capture.inventory
    prohibit_weight_conversion = raw_capture.prohibit_weight_conversion
    reconstruct = raw_capture.reconstruct

    export = Path(os.environ["ATLAS_EXPORT_DIR"])
    architecture = Path(os.environ["ATLAS_ARCHITECTURE"])
    expected_contract = os.environ["ATLAS_EXPECTED_CONTRACT_SHA256"]
    expected_policy = os.environ["ATLAS_EXPECTED_POLICY_SHA256"]
    for path, expected in ((export / "manifest.json", expected_contract),
                           (export / "policy.json", expected_policy)):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"selected source receipt digest differs: {path.name}")
    # read_export authenticates files before any model reconstruction.
    tensors, policy, manifest = load_quantized.read_export(export, verify=True)
    if sha256((architecture / "config.json").read_bytes()).hexdigest() != manifest["model"]["checkpoint"]["config_json_sha256"]:
        raise ValueError("architecture differs from authenticated export")
    before = inventory(tensors)
    config = AutoConfig.from_pretrained(str(architecture), local_files_only=True)
    config.use_cache, config._attn_implementation = False, "eager"
    with prohibit_weight_conversion(), torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="eager")
    for module in model.modules():
        module._non_persistent_buffers_set.clear()
    replay = load_quantized.PolicyAlgo(policy, tensors)
    with prohibit_weight_conversion(), algos.active(replay):
        reconstruct(model, policy, tensors, replay=replay, linear_class=ref_profile.RefQuantLinear,
                    patch_attention=ref_attention.patch_attention)
    if inventory(tensors) != before:
        raise ValueError("raw source state changed during reconstruction")
    inputs = load_file(os.environ["ATLAS_INPUTS_SAFETENSORS"])
    selected = os.environ.get("ATLAS_CAPTURE_BOUNDARY", "model")
    if selected == "model":
        boundary = fx_export.CausalLogits(model).eval()
        if set(inputs) != {"input_ids"} or inputs["input_ids"].dtype != torch.int64:
            raise ValueError("input archive must contain only int64 input_ids")
        arguments = (inputs["input_ids"],)
    elif selected == "layer0":
        source_path = raw_capture.checked_path(os.environ["ATLAS_BOUNDARY_SOURCE_JSON"])
        source_bytes = source_path.read_bytes()
        if sha256(source_bytes).hexdigest() != os.environ["ATLAS_EXPECTED_BOUNDARY_SOURCE_SHA256"]:
            raise ValueError("selected boundary source receipt digest differs")
        boundary, arguments = layer0_boundary(model, inputs, json.loads(source_bytes),
                                              tensor_record=raw_capture.tensor_record)
    else:
        raise ValueError(f"unsupported live capture boundary: {selected}")
    with prohibit_weight_conversion(), algos.active(replay):
        return prepare_model(boundary, arguments, export / "manifest.json",
                             export / "policy.json", export_prefix="model.",
                             expected_contract_sha256=expected_contract,
                             expected_policy_sha256=expected_policy)
