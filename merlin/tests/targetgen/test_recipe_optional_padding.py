"""A model without string-padding Conv2d does not need its frontend normalizer."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from merlin.targetgen._recipe_quantizer import (
    RecipeError,
    _allow_exported_model_train_eval,
    _normalize_conv_padding,
)


def graph(*targets):
    return SimpleNamespace(
        graph=SimpleNamespace(nodes=[SimpleNamespace(op="call_function", target=t) for t in targets])
    )


def test_unrelated_model_never_imports_optional_padding_normalizer(monkeypatch):
    monkeypatch.setitem(sys.modules, "m2m.capture.pt2e_padding", None)
    assert _normalize_conv_padding(graph("matmul", "relu"), "conv_padding") == []


def test_string_padding_still_refuses_when_its_normalizer_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "m2m.capture.pt2e_padding", None)
    with pytest.raises(RecipeError, match="cannot normalize Conv2d string padding"):
        _normalize_conv_padding(graph("conv_padding"), "conv_padding")


def test_needed_normalizer_receives_the_original_graph_and_retains_ancestry(monkeypatch):
    selected = graph("conv_padding")
    receipt = [{"source_ids": ["original_conv"], "padding": "same"}]
    frontend = ModuleType("m2m.capture.pt2e_padding")

    def normalize(exported):
        assert exported is selected
        return receipt

    frontend.normalize_conv2d_padding = normalize
    monkeypatch.setitem(sys.modules, frontend.__name__, frontend)
    assert _normalize_conv_padding(selected, "conv_padding") is receipt


def test_old_frontend_uses_public_torchao_train_eval_helper(monkeypatch):
    monkeypatch.setitem(sys.modules, "m2m.capture.pt2e_compat", None)
    framework = ModuleType("torchao.quantization.pt2e")
    calls = []
    framework.allow_exported_model_train_eval = calls.append
    monkeypatch.setitem(sys.modules, framework.__name__, framework)
    model = object()
    _allow_exported_model_train_eval(model)
    assert calls == [model]
