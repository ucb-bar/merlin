"""Live decoder boundaries use original modules and authenticated tensor words."""
from copy import deepcopy

import pytest
import torch

from adapter_loader import layer0_boundary
from cache_source import phase_positions_and_mask
from raw_capture import tensor_record
from test_cache_source import tiny_source, state_inventory


def source_case():
    model = tiny_source()
    ids = torch.tensor([[2, 3, 5]])
    positions, mask = phase_positions_and_mask(ids, 0)
    with torch.no_grad():
        hidden = model.model.embed_tokens(ids)
        cosine, sine = model.model.rotary_emb(hidden, positions)
    values = (hidden, cosine, sine, mask)
    inputs = {f"input_{index}": value for index, value in enumerate(values)}
    source = dict(kind="layer0", input_keys=list(inputs),
                  tensors={key: tensor_record(value) for key, value in inputs.items()},
                  parameters={f"model.{name}": tensor_record(value) for name, value in
                              model.named_parameters() if name.startswith("model.layers.0.")})
    return model, inputs, source


def decode_case():
    from cache_source import _AppendTensorCache

    model, prefill, source = source_case()
    with torch.no_grad():
        cache = _AppendTensorCache(((None, None),))
        model.model.layers[0](prefill["input_0"], attention_mask=prefill["input_3"],
            position_embeddings=(prefill["input_1"], prefill["input_2"]),
            past_key_values=cache, use_cache=True)
        past_key, past_value = cache.tensors()
        ids = torch.tensor([[7]])
        positions, mask = phase_positions_and_mask(ids, past_key.shape[2])
        hidden = model.model.embed_tokens(ids)
        cosine, sine = model.model.rotary_emb(hidden, positions)
    inputs = {f"input_{index}": value for index, value in enumerate(
        (hidden, cosine, sine, past_key, mask, past_value))}
    source.update(input_keys=list(inputs),
                  tensors={key: tensor_record(value) for key, value in inputs.items()},
                  cache_roles={field: [dict(index=index, role=role,
                    **({"layer": 0} if role not in ("source_tensor", "attention_mask") else {}))
                    for index, role in enumerate(roles)] for field, roles in (
                      ("inputs", ["source_tensor", "source_tensor", "source_tensor", "past_key",
                                  "attention_mask", "past_value"]),
                      ("outputs", ["hidden_state", "updated_key", "updated_value"]))})
    return model, inputs, source


def test_live_layer_uses_original_module_and_parameter_names():
    model, inputs, source = source_case()
    before = state_inventory(model)
    boundary, arguments = layer0_boundary(model, inputs, source, tensor_record=tensor_record)
    assert boundary.model.model.layers[0] is model.model.layers[0]
    assert set(dict(boundary.named_parameters())) == {
        f"model.{name}" for name, _ in model.named_parameters() if name.startswith("model.layers.0.")}
    with torch.no_grad():
        actual = boundary(*arguments)
        expected = model.model.layers[0](arguments[0], attention_mask=arguments[3],
                                          position_embeddings=arguments[1:3])
    assert tensor_record(actual) == tensor_record(expected)
    assert state_inventory(model) == before


@pytest.mark.parametrize("cached", [False, True])
def test_live_layer_exports_original_attention_calls(cached):
    import fx_export
    import algos
    import load_quantized

    model, inputs, source = decode_case() if cached else source_case()
    before = state_inventory(model)
    boundary, arguments = layer0_boundary(model, inputs, source, tensor_record=tensor_record)
    policy = {"sites": {f"model.layers.{i}.self_attn": {
        "status": "quantized", "kind": "attention", "operands": {
            "q": {"source": "reference"}, "k": {"source": "reference"},
            "v": {"source": "reference"},
            "p": {"mode": "static", "exp": -8, "label": "static"}}}
        for i in range(model.config.num_hidden_layers)}}
    with algos.active(load_quantized.PolicyAlgo(policy, {})), torch.no_grad():
        with fx_export._native_sites(boundary, policy, storage="uint8"):
            program = torch.export.export(boundary, arguments, strict=True)
    calls = [str(node.target) for node in program.graph.nodes if node.op == "call_function"]
    assert calls.count("atlas.qk_matmul.default") == 1
    assert calls.count("atlas.pv_matmul.default") == 1
    assert len(program.graph_signature.user_inputs) == (6 if cached else 4)
    assert len(program.graph_signature.user_outputs) == (3 if cached else 1)
    assert state_inventory(model) == before


def test_decode_matches_original_framework_cache_words_without_mutation():
    from transformers import DynamicCache

    model, inputs, source = decode_case()
    before = state_inventory(model)
    input_words = {name: tensor_record(value) for name, value in inputs.items()}
    boundary, arguments = layer0_boundary(model, inputs, source, tensor_record=tensor_record)
    hidden, cosine, sine, past_key, mask, past_value = arguments
    conventional = DynamicCache(config=model.config)
    conventional.update(past_key, past_value, 0)
    with torch.no_grad():
        actual = boundary(*arguments)
        expected = model.model.layers[0](hidden, attention_mask=mask,
            position_embeddings=(cosine, sine), past_key_values=conventional, use_cache=True)
    assert [tensor_record(value) for value in actual] == [
        tensor_record(value) for value in (expected, conventional.layers[0].keys,
                                          conventional.layers[0].values)]
    assert state_inventory(model) == before
    assert {name: tensor_record(value) for name, value in inputs.items()} == input_words


def test_wrong_decode_cache_role_order_is_refused():
    model, inputs, source = decode_case()
    source["cache_roles"]["inputs"][3]["role"] = "past_value"
    with pytest.raises(ValueError, match="cache roles"):
        layer0_boundary(model, inputs, source, tensor_record=tensor_record)


def test_modified_boundary_words_are_refused():
    model, inputs, source = source_case()
    inputs["input_0"] = inputs["input_0"].clone()
    inputs["input_0"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="tensor words"):
        layer0_boundary(model, inputs, source, tensor_record=tensor_record)


def test_incomplete_boundary_roster_is_refused():
    model, inputs, source = source_case()
    source = deepcopy(source)
    source["input_keys"].reverse()
    with pytest.raises(ValueError, match="hidden, cosine"):
        layer0_boundary(model, inputs, source, tensor_record=tensor_record)


def test_parameters_from_another_source_are_refused():
    model, inputs, source = source_case()
    source["parameters"] = {}
    with pytest.raises(ValueError, match="parameters differ"):
        layer0_boundary(model, inputs, source, tensor_record=tensor_record)
