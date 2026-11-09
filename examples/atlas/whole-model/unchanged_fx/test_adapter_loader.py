"""Exercise the example loader through real raw-state reconstruction and export."""
from hashlib import sha256
import json
from unittest.mock import patch

import algos
import pytest
import quant_export
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, LlamaConfig

import adapter_loader


@pytest.fixture
def source(tmp_path, monkeypatch, request):
    import fx_export

    torch.manual_seed(19)
    architecture, export = tmp_path / "architecture", tmp_path / "export"
    architecture.mkdir()
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=getattr(request, "param", 1),
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                         tie_word_embeddings=False, use_cache=False)
    config._attn_implementation = "eager"
    config.save_pretrained(architecture)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="eager").eval()
    for module in model.modules():
        module._non_persistent_buffers_set.clear()
    algo = algos.make("granularity:w=channel,a=token,rule=nc")
    options = {"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"}
    report = algos.quantize(model, "ref_conv", ["lm_head"], options, algo=algo)
    quant_export.export(model, export, mode="ref_conv", report=report, algo=algo,
                        algo_spec="granularity:w=channel,a=token,rule=nc", options=options, skip=["lm_head"])
    manifest_path = export / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"] = {"checkpoint": {"config_json_sha256": sha256((architecture / "config.json").read_bytes()).hexdigest()}}
    manifest_path.write_text(json.dumps(manifest))
    inputs = torch.arange(8, dtype=torch.int64).reshape(1, 8)
    with torch.no_grad():
        expected = fx_export.CausalLogits(model).eval()(inputs).clone()
    inputs_path = tmp_path / "inputs.safetensors"
    save_file({"input_ids": inputs}, inputs_path)
    values = {"ATLAS_EXPORT_DIR": str(export), "ATLAS_ARCHITECTURE": str(architecture),
              "ATLAS_INPUTS_SAFETENSORS": str(inputs_path),
              "ATLAS_EXPECTED_CONTRACT_SHA256": sha256(manifest_path.read_bytes()).hexdigest(),
              "ATLAS_EXPECTED_POLICY_SHA256": sha256((export / "policy.json").read_bytes()).hexdigest()}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ATLAS_CAPTURE_BOUNDARY", "model")
    return export, architecture, inputs_path, inputs, expected, model


def test_loader_reconstructs_native_attention_without_saved_program_or_weight_conversion(source):
    from ref_profile import RefQuantLinear
    import raw_capture
    from safetensors.torch import load_file

    export, _, _, inputs, expected, _ = source
    before = raw_capture.inventory(load_file(str(export / "model.safetensors")))
    with patch.object(torch.export, "load", side_effect=AssertionError("saved source program loaded")), \
         patch.object(RefQuantLinear, "__init__", side_effect=AssertionError("quantizing constructor called")):
        program, actual_inputs = adapter_loader.get_model_and_inputs()
    assert isinstance(program, torch.export.ExportedProgram)
    assert torch.equal(actual_inputs[0], inputs)
    assert torch.equal(program.module()(*actual_inputs), expected)
    assert raw_capture.inventory(load_file(str(export / "model.safetensors"))) == before
    calls = program._m2m_external_quantization_manifest["atlas"]["calls"]
    assert {"atlas.qlinear.default", "atlas.qk_matmul.default", "atlas.pv_matmul.default"} <= {
        call["target"] for call in calls}
    assert "saved_native_program" not in program._m2m_external_quantization_manifest["atlas"]


@pytest.mark.parametrize("role", ["contract", "policy"])
def test_loader_refuses_changed_receipt_bytes_before_architecture_construction(source, role):
    export, *_ = source
    path = export / ("manifest.json" if role == "contract" else "policy.json")
    path.write_bytes(path.read_bytes() + b"\n")
    with patch("transformers.AutoModelForCausalLM.from_config", side_effect=AssertionError("model constructed")):
        with pytest.raises(ValueError, match="source receipt digest"):
            adapter_loader.get_model_and_inputs()


def test_loader_refuses_unattributed_architecture(source):
    _, architecture, *_ = source
    path = architecture / "config.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="architecture differs"):
        adapter_loader.get_model_and_inputs()


def test_loader_refuses_noninteger_token_archive(source):
    _, _, inputs_path, inputs, *_ = source
    save_file({"input_ids": inputs.float()}, inputs_path)
    with pytest.raises(ValueError, match="only int64 input_ids"):
        adapter_loader.get_model_and_inputs()


def test_live_prefill_cache_exports_original_state_and_feeds_decode(source, tmp_path, monkeypatch):
    from cache_source import phase_positions_and_mask
    from raw_capture import inventory, tensor_record
    from ref_profile import RefQuantLinear
    from safetensors.torch import load_file
    from transformers import DynamicCache
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    export, _, inputs_path, ids, _, model = source
    raw = load_file(str(export / "model.safetensors"))
    before = inventory(raw)
    model_before = inventory(dict(list(model.named_parameters()) + list(model.named_buffers())))
    parameters = {f"model.{name}": tensor_record(value) for name, value in raw.items()
                  if name.startswith("model.layers.0.")}
    source_path = tmp_path / "layer0-source.json"

    def select(arguments, selection, cache_roles=None):
        inputs = {f"input_{index}": value for index, value in enumerate(arguments)}
        save_file(inputs, inputs_path)
        receipt = dict(kind="layer0", input_keys=list(inputs), parameters=parameters,
                       tensors=inventory(inputs))
        if cache_roles is not None:
            receipt["cache_roles"] = cache_roles
        source_path.write_text(json.dumps(receipt))
        monkeypatch.setenv("ATLAS_CAPTURE_BOUNDARY", selection)
        monkeypatch.setenv("ATLAS_BOUNDARY_SOURCE_JSON", str(source_path))
        monkeypatch.setenv("ATLAS_EXPECTED_BOUNDARY_SOURCE_SHA256", sha256(source_path.read_bytes()).hexdigest())
        with patch.object(torch.export, "load", side_effect=AssertionError("saved source program loaded")), \
             patch.object(RefQuantLinear, "__init__", side_effect=AssertionError("quantizing constructor called")):
            return adapter_loader.get_model_and_inputs()

    def state(program):
        return inventory({name: value for name, value in
                          list(program.named_parameters()) + list(program.named_buffers())
                          if value.device.type != "meta"})

    conventional = DynamicCache(config=model.config)
    with torch.no_grad():
        positions, mask = phase_positions_and_mask(ids, 0)
        hidden = model.model.embed_tokens(ids)
        cosine, sine = model.model.rotary_emb(hidden, positions)
        expected_hidden = model.model.layers[0](hidden, attention_mask=mask,
            position_embeddings=(cosine, sine), past_key_values=conventional, use_cache=True)
        expected_prefill = (expected_hidden, conventional.layers[0].keys, conventional.layers[0].values)
        # Independently establish that the returned keys precede GQA expansion
        # and follow the source's RoPE, while values retain the projected words.
        attention = model.model.layers[0].self_attn
        normalized = model.model.layers[0].input_layernorm(hidden)
        shape = (1, ids.shape[1], -1, attention.head_dim)
        query = attention.q_proj(normalized).view(shape).transpose(1, 2)
        key = attention.k_proj(normalized).view(shape).transpose(1, 2)
        value = attention.v_proj(normalized).view(shape).transpose(1, 2)
        _, rotated_key = apply_rotary_pos_emb(query, key, cosine, sine)
    arguments = (hidden, cosine, sine, mask)
    hidden_only, original_inputs = select(arguments, "layer0")
    prefill, actual_inputs = select(arguments, "layer0-prefill-cache")
    assert len(original_inputs) == len(actual_inputs) == 4
    assert len(hidden_only.graph_signature.user_outputs) == 1
    assert len(prefill.graph_signature.user_outputs) == 3
    assert state(prefill) == state(hidden_only) == parameters
    assert set(dict(prefill.named_parameters())) == set(dict(hidden_only.named_parameters()))
    actual_prefill = prefill.module()(*actual_inputs)
    assert [tensor_record(value) for value in actual_prefill] == [tensor_record(value) for value in expected_prefill]
    assert tensor_record(hidden_only.module()(*original_inputs)) == tensor_record(actual_prefill[0])
    assert tensor_record(actual_prefill[1]) == tensor_record(rotated_key)
    assert tensor_record(actual_prefill[2]) == tensor_record(value)
    assert actual_prefill[1].shape == (1, model.config.num_key_value_heads, ids.shape[1], attention.head_dim)
    assert all(value.dtype == torch.bfloat16 for value in actual_prefill)
    assert tensor_record(key) != tensor_record(rotated_key)

    with torch.no_grad():
        decode_ids = torch.tensor([[11]])
        positions, mask = phase_positions_and_mask(decode_ids, ids.shape[1])
        hidden = model.model.embed_tokens(decode_ids)
        cosine, sine = model.model.rotary_emb(hidden, positions)
        expected_hidden = model.model.layers[0](hidden, attention_mask=mask,
            position_embeddings=(cosine, sine), past_key_values=conventional, use_cache=True)
        expected_decode = (expected_hidden, conventional.layers[0].keys, conventional.layers[0].values)
    arguments = (hidden, cosine, sine, actual_prefill[1], mask, actual_prefill[2])
    frozen_arguments = [tensor_record(value) for value in arguments]
    roles = {field: [dict(index=index, role=role,
        **({"layer": 0} if role not in ("source_tensor", "attention_mask") else {}))
        for index, role in enumerate(names)] for field, names in (
            ("inputs", ["source_tensor", "source_tensor", "source_tensor", "past_key", "attention_mask", "past_value"]),
            ("outputs", ["hidden_state", "updated_key", "updated_value"]))}
    decode, decode_inputs = select(arguments, "layer0", cache_roles=roles)
    assert state(decode) == parameters
    assert [tensor_record(value) for value in decode.module()(*decode_inputs)] == [
        tensor_record(value) for value in expected_decode]
    assert [tensor_record(value) for value in arguments] == frozen_arguments
    assert inventory(load_file(str(export / "model.safetensors"))) == before
    assert inventory(dict(list(model.named_parameters()) + list(model.named_buffers()))) == model_before
    with pytest.raises(ValueError, match="original four-input boundary"):
        select(arguments, "layer0-prefill-cache", cache_roles=roles)


@pytest.mark.parametrize("source", [2], indirect=True)
def test_live_whole_cache_exports_original_state_and_independent_decode(source, tmp_path, monkeypatch):
    from cache_source import phase_positions_and_mask
    from raw_capture import inventory, tensor_record
    from ref_profile import RefQuantLinear
    from safetensors.torch import load_file
    from transformers import DynamicCache

    export, _, inputs_path, ids, original_logits, model = source
    raw_before = inventory(load_file(str(export / "model.safetensors")))
    state_before = inventory(dict(list(model.named_parameters()) + list(model.named_buffers())))
    parameters = {f"model.{name}": value for name, value in raw_before.items()}
    source_path = tmp_path / "model-source.json"
    baseline, _ = adapter_loader.get_model_and_inputs()
    baseline_state = inventory({name: value for name, value in
        list(baseline.named_parameters()) + list(baseline.named_buffers()) if value.device.type != "meta"})

    def select(arguments, selection, *, include_parameters=False):
        inputs = {f"input_{index}": value.contiguous() for index, value in enumerate(arguments)}
        save_file(inputs, inputs_path)
        receipt = dict(kind="model", input_keys=list(inputs), tensors=inventory(inputs))
        if include_parameters:
            receipt["parameters"] = parameters
        source_path.write_text(json.dumps(receipt))
        monkeypatch.setenv("ATLAS_CAPTURE_BOUNDARY", selection)
        monkeypatch.setenv("ATLAS_BOUNDARY_SOURCE_JSON", str(source_path))
        monkeypatch.setenv("ATLAS_EXPECTED_BOUNDARY_SOURCE_SHA256", sha256(source_path.read_bytes()).hexdigest())
        with patch.object(torch.export, "load", side_effect=AssertionError("saved source program loaded")), \
             patch.object(RefQuantLinear, "__init__", side_effect=AssertionError("quantizing constructor called")):
            return adapter_loader.get_model_and_inputs()

    def words(values):
        return [tensor_record(value) for value in values]

    conventional = DynamicCache(config=model.config)
    with torch.no_grad():
        positions, mask = phase_positions_and_mask(ids, 0)
        original = model(ids, position_ids=positions, attention_mask=mask,
                         past_key_values=conventional, use_cache=True)
        expected_prefill = (original.logits, *(value.clone() for layer in conventional.layers
                                              for value in (layer.keys, layer.values)))
        decode_ids = torch.tensor([[11]])
        decode_positions, decode_mask = phase_positions_and_mask(decode_ids, ids.shape[1])
        independent_arguments = (decode_ids, decode_positions, decode_mask,
                                 *(value.clone() for value in expected_prefill[1:]))
        original = model(decode_ids, position_ids=decode_positions, attention_mask=decode_mask,
                         past_key_values=conventional, use_cache=True)
        expected_decode = (original.logits, *(value.clone() for layer in conventional.layers
                                             for value in (layer.keys, layer.values)))
    assert tensor_record(expected_prefill[0]) == tensor_record(original_logits)
    prefill, prefill_inputs = select((ids, positions, mask), "model-prefill-cache")
    actual_prefill = prefill.module()(*prefill_inputs)
    decode, decode_inputs = select(independent_arguments, "model-decode-cache", include_parameters=True)
    frozen_inputs = words(decode_inputs)
    # Generated words enter only the invocation, after independent capture.
    actual_decode = decode.module()(*decode_inputs[:3], *actual_prefill[1:])
    count = model.config.num_hidden_layers
    assert len(prefill_inputs) == 3
    assert len(decode_inputs) == 3 + 2 * count
    assert len(actual_prefill) == len(actual_decode) == 1 + 2 * count
    assert words(actual_prefill) == words(expected_prefill)
    assert words(actual_decode) == words(expected_decode)
    assert words(decode_inputs) == frozen_inputs
    for prefix, updated in zip(actual_prefill[1:], actual_decode[1:]):
        assert tensor_record(prefix) == tensor_record(updated[:, :, :ids.shape[1]])
    for program in (prefill, decode):
        actual_state = inventory({name: value for name, value in
            list(program.named_parameters()) + list(program.named_buffers()) if value.device.type != "meta"})
        assert actual_state == baseline_state
        assert actual_state == {name: parameters[name] for name in actual_state}
        assert set(dict(program.named_parameters())) == set(dict(baseline.named_parameters()))
        assert {call["target"] for call in program._m2m_external_quantization_manifest["atlas"]["calls"]} >= {
            "atlas.qlinear.default", "atlas.qk_matmul.default", "atlas.pv_matmul.default"}
    assert inventory(load_file(str(export / "model.safetensors"))) == raw_before
    assert inventory(dict(list(model.named_parameters()) + list(model.named_buffers()))) == state_before

    with pytest.raises(ValueError, match="prefill positions"):
        select((ids, positions + 1, mask), "model-prefill-cache")
    with pytest.raises(ValueError, match="ordered source K/V"):
        select(independent_arguments[:-1], "model-decode-cache")
