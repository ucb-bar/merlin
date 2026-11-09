"""The live head retains original parameters and authenticates saved inputs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from adapter_loader import head_boundary
from raw_capture import tensor_record


def case():
    model = torch.nn.Module()
    model.config = SimpleNamespace(hidden_size=4)
    model.lm_head = torch.nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)
    inputs = {"input_0": torch.ones((1, 3, 4), dtype=torch.bfloat16)}
    source = dict(kind="head", input_keys=["input_0"],
                  tensors={key: tensor_record(value) for key, value in inputs.items()},
                  parameters={"model.lm_head.weight": tensor_record(model.lm_head.weight)})
    return model, inputs, source


def test_head_preserves_source_module_and_words():
    model, inputs, source = case()
    boundary, arguments = head_boundary(model, inputs, source, tensor_record=tensor_record)
    assert boundary.model.lm_head is model.lm_head
    assert tensor_record(boundary(*arguments)) == tensor_record(model.lm_head(inputs["input_0"]))
    assert list(dict(boundary.named_parameters())) == ["model.lm_head.weight"]


@pytest.mark.parametrize("change", ["kind", "keys", "words", "parameters", "dtype", "rank"])
def test_head_rejects_changed_boundary(change):
    model, inputs, source = case()
    source = deepcopy(source)
    if change == "kind": source["kind"] = "norm"
    if change == "keys": source["input_keys"] = ["other"]
    if change == "words": inputs["input_0"][0, 0, 0] = 2
    if change == "parameters": source["parameters"] = {}
    if change == "dtype":
        inputs["input_0"] = inputs["input_0"].float()
        source["tensors"]["input_0"] = tensor_record(inputs["input_0"])
    if change == "rank":
        inputs["input_0"] = inputs["input_0"].reshape(3, 4)
        source["tensors"]["input_0"] = tensor_record(inputs["input_0"])
    with pytest.raises(ValueError):
        head_boundary(model, inputs, source, tensor_record=tensor_record)
