"""Explicit one-decoder logits boundaries preserve original source modules."""
from copy import deepcopy

import pytest
import torch

from adapter_loader import model_cache_boundary
from cache_source import phase_positions_and_mask
from raw_capture import tensor_record
from test_cache_source import tiny_source, state_inventory


def receipt(arguments, **options):
    inputs = {f"input_{index}": value for index, value in enumerate(arguments)}
    return inputs, dict(kind="model", input_keys=list(inputs), decoder_layers=[0],
                        tensors={key: tensor_record(value) for key, value in inputs.items()}, **options)


def test_declared_layer0_logits_prefill64_and_two_greedy_decodes():
    from transformers import DynamicCache

    torch.set_num_threads(1)
    model = tiny_source(max_position_embeddings=128)
    before = state_inventory(model)
    original_config = model.config.to_dict()
    authenticated = {f"model.{name}": tensor_record(value) for name, value in
                     list(model.named_parameters()) + list(model.named_buffers())}
    expected_roster = {name: value for name, value in authenticated.items()
                       if not name.startswith("model.model.layers.") or name.startswith("model.model.layers.0.")}
    reference = deepcopy(model)
    reference.model.layers = torch.nn.ModuleList([reference.model.layers[0]])
    reference.config.num_hidden_layers = 1
    cache = DynamicCache(config=reference.config)
    actual_ids = expected_ids = torch.arange(64, dtype=torch.int64).reshape(1, 64) % model.config.vocab_size
    previous = ()
    with torch.no_grad():
        for past in (0, 64, 65):
            positions, mask = phase_positions_and_mask(actual_ids, past)
            arguments = (actual_ids, positions, mask, *previous)
            inputs, source = receipt(arguments)
            boundary, values = model_cache_boundary(model, inputs, source, tensor_record=tensor_record,
                prefill_cache=past == 0, authenticated_parameters=authenticated)
            actual = boundary(*values)
            expected = reference(expected_ids, position_ids=positions, attention_mask=mask,
                                 past_key_values=cache, use_cache=True)
            expected_values = (expected.logits, cache.layers[0].keys, cache.layers[0].values)
            assert [tensor_record(value) for value in actual] == [tensor_record(value) for value in expected_values]
            assert boundary.model.model.layers[0] is model.model.layers[0]
            assert boundary.model.model.embed_tokens is model.model.embed_tokens
            assert boundary.model.model.norm is model.model.norm
            assert boundary.model.lm_head is model.lm_head
            assert boundary.model.config is not model.config
            assert boundary.model.config.num_hidden_layers == 1
            assert {name: tensor_record(value) for name, value in
                    list(boundary.named_parameters()) + list(boundary.named_buffers())} == expected_roster
            actual_ids = actual[0][:, -1].argmax(dim=-1, keepdim=True)
            expected_ids = expected.logits[:, -1].argmax(dim=-1, keepdim=True)
            assert torch.equal(actual_ids, expected_ids)
            previous = actual[1:]
    assert len(model.model.layers) == 2
    assert model.config.to_dict() == original_config
    assert state_inventory(model) == before


@pytest.mark.parametrize("selection", [None, [], [1], [0, 1], [False], "0", (0,)])
def test_invalid_declared_layer_roster_is_refused(selection):
    model = tiny_source()
    ids = torch.tensor([[2]])
    positions, mask = phase_positions_and_mask(ids, 0)
    inputs, source = receipt((ids, positions, mask))
    source["decoder_layers"] = selection
    with pytest.raises(ValueError, match="decoder_layers"):
        model_cache_boundary(model, inputs, source, tensor_record=tensor_record, prefill_cache=True)


def test_explicit_full_parameter_roster_is_refused_for_layer0_selection():
    model = tiny_source()
    ids = torch.tensor([[2]])
    positions, mask = phase_positions_and_mask(ids, 0)
    inputs, source = receipt((ids, positions, mask), parameters={f"model.{name}": tensor_record(value)
        for name, value in list(model.named_parameters()) + list(model.named_buffers())})
    with pytest.raises(ValueError, match="parameters differ"):
        model_cache_boundary(model, inputs, source, tensor_record=tensor_record, prefill_cache=True)
