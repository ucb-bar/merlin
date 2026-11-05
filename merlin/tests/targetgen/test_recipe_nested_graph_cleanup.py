"""Public tiny FX regressions for the selected capture interpreter's PT2E cleanup."""
import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
from torch.fx import Graph, GraphModule
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer import Quantizer

from merlin.common.paths import repo_root

spec = importlib.util.spec_from_file_location("recipe_cleanup_under_test", repo_root() / "src/merlin/targetgen/_recipe_quantizer.py")
RQ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RQ)


class IdentityQuantizer(Quantizer):
    def annotate(self, model): return model
    def validate(self, model): pass
    def transform_for_annotation(self, model): return model


def invoke(module, value):
    return module(value)


def nested():
    root = torch.nn.Module()
    root.submod_1 = torch.nn.ReLU()
    root.constants = torch.nn.Module()
    root.constants.register_buffer("scale", torch.tensor([2., 3.]))
    graph = Graph()
    value = graph.placeholder("value")
    scale = graph.get_attr("constants.scale")
    relu = graph.call_module("submod_1", (value,))
    relu.name = "submod_3"
    result = graph.call_function(torch.mul, (relu, scale))
    graph.output(result)
    child = GraphModule(root, graph)
    child.dead = torch.nn.Sigmoid()
    root = torch.nn.Module()
    root.body = child
    graph = Graph()
    value = graph.placeholder("value")
    body = graph.get_attr("body")
    result = graph.call_function(invoke, (body, value))
    graph.output(result)
    parent = GraphModule(root, graph)
    parent.dead = torch.nn.Identity()
    return parent


def test_actual_pt2e_conversion_preserves_nested_outputs_and_restores_framework():
    original_cleanup = GraphModule.delete_all_unused_submodules
    source = nested()
    inputs = (torch.tensor([-1., 2.]), torch.tensor([3., -4.]))
    expected = [source(value) for value in inputs]
    prepared = prepare_pt2e(copy.deepcopy(source), IdentityQuantizer())
    with RQ._pt2e_nested_graph_cleanup() as report:
        converted = convert_pt2e(prepared)
    assert GraphModule.delete_all_unused_submodules is original_cleanup
    assert report["adapted_calls"] >= 1 and report["retained_graph_references"] >= 1
    for value, want in zip(inputs, expected):
        assert torch.equal(converted(value), want)
    converted.graph.lint()
    converted.body.graph.lint()
    assert hasattr(converted.body, "submod_1") and hasattr(converted.body.constants, "scale")
    assert not hasattr(converted, "dead") and not hasattr(converted.body, "dead")
    # Conversion did not mutate the original model's live state or dead children.
    assert hasattr(source.body, "dead") and hasattr(source, "dead")


def test_missing_live_state_is_not_reconstructed_or_silently_pruned():
    model = nested()
    del model.body.submod_1
    original = GraphModule.delete_all_unused_submodules
    with pytest.raises(AttributeError):
        with RQ._pt2e_nested_graph_cleanup():
            model.delete_all_unused_submodules()
    assert GraphModule.delete_all_unused_submodules is original


def test_plain_graph_uses_original_pruning_and_exception_restores():
    model = torch.fx.symbolic_trace(torch.nn.ReLU())
    model.dead = torch.nn.Identity()
    original = GraphModule.delete_all_unused_submodules
    with RQ._pt2e_nested_graph_cleanup() as report:
        model.delete_all_unused_submodules()
    assert report["adapted_calls"] == 0 and not hasattr(model, "dead")
    with pytest.raises(RuntimeError, match="conversion failed"):
        with RQ._pt2e_nested_graph_cleanup():
            raise RuntimeError("conversion failed")
    assert GraphModule.delete_all_unused_submodules is original


class NestedMask(torch.nn.Module):
    def forward(self, indices):
        return torch.vmap(lambda q: torch.vmap(
            lambda k: q.new_ones((), dtype=torch.bool) & (k <= q)
        )(indices))(indices)


def test_vmap_control_effects_survive_quantization_and_later_capture_dce():
    from merlin.targetgen._capture_integer_mm import exact_integer_mm
    from torch._functorch import predispatch
    from torch.fx.node import _side_effectful_functions

    original_registry = set(_side_effectful_functions)
    model, indices = NestedMask(), torch.arange(4)
    expected = model(indices)
    exported = torch.export.export(model, (indices,)).module()
    controls = (predispatch._vmap_increment_nesting, predispatch._vmap_decrement_nesting)
    original_controls = [node.target for node in exported.graph.nodes if node.target in controls]
    assert len(original_controls) == 4
    # This exported source works with and without integer-mm dispatch. The
    # effect context must preserve it, never reconstruct a lost nesting stack.
    assert torch.equal(exported(indices), expected)
    with exact_integer_mm(torch):
        assert torch.equal(exported(indices), expected)
    with RQ.preserve_fx_vmap_effects():
        exported.graph.eliminate_dead_code()
        exported.recompile()
        prepared = prepare_pt2e(exported, IdentityQuantizer())
        with RQ._pt2e_nested_graph_cleanup():
            converted = convert_pt2e(prepared)
        for _ in range(2):  # Later capture cleanup, after conversion's scope.
            converted.graph.eliminate_dead_code()
            converted.recompile()
        assert [node.target for node in converted.graph.nodes if node.target in controls] == original_controls
        assert torch.equal(converted(indices), expected)
        with exact_integer_mm(torch):
            assert torch.equal(converted(indices), expected)
    assert _side_effectful_functions == original_registry


def test_effect_scope_keeps_preexisting_registration_and_restores_on_failure():
    from torch._functorch import predispatch
    from torch.fx.node import _side_effectful_functions

    before = set(_side_effectful_functions)
    _side_effectful_functions.add(predispatch._vmap_increment_nesting)
    selected = set(_side_effectful_functions)
    try:
        with pytest.raises(RuntimeError, match="capture failed"):
            with RQ.preserve_fx_vmap_effects():
                with RQ.preserve_fx_vmap_effects():
                    assert predispatch._vmap_decrement_nesting in _side_effectful_functions
                assert predispatch._vmap_decrement_nesting in _side_effectful_functions
                raise RuntimeError("capture failed")
        assert _side_effectful_functions == selected
    finally:
        _side_effectful_functions.clear()
        _side_effectful_functions.update(before)


def test_direct_recipe_call_has_effect_scope_and_keeps_introspection(monkeypatch):
    import inspect
    from torch._functorch import predispatch
    from torch.fx.node import _side_effectful_functions

    before = set(_side_effectful_functions)
    assert "calibration_samples" in inspect.signature(RQ.apply_recipe).parameters

    def refuse_recipe(recipe):
        assert predispatch._vmap_increment_nesting in _side_effectful_functions
        assert predispatch._vmap_decrement_nesting in _side_effectful_functions
        raise RQ.RecipeError("independent recipe refusal")

    monkeypatch.setattr(RQ, "_require", refuse_recipe)
    with pytest.raises(RQ.RecipeError, match="independent recipe refusal"):
        RQ.apply_recipe(NestedMask(), {}, example_inputs=(torch.arange(4),))
    assert _side_effectful_functions == before


def test_late_capture_cleanup_preserves_shared_live_graph_state_and_final_golden():
    source = nested()
    inputs = (torch.tensor([-2., 5.]), torch.tensor([7., -3.]))
    expected = [source(value) for value in inputs]
    source_nodes = tuple((node.op, node.target) for node in source.body.graph.nodes)
    source_scale = source.body.constants.scale.clone()
    original = GraphModule.delete_all_unused_submodules
    with RQ.preserve_fx_vmap_effects(), RQ._pt2e_nested_graph_cleanup() as outer:
        prepared = prepare_pt2e(copy.deepcopy(source), IdentityQuantizer())
        with RQ._pt2e_nested_graph_cleanup():
            converted = convert_pt2e(prepared)
        # Model2mlir may reconstruct a wrapper that shares nested modules with
        # the model subsequently used for the final eager golden evaluation.
        late = GraphModule(converted, copy.deepcopy(converted.graph))
        assert late.body is converted.body
        late.dead = torch.nn.Identity()
        late.delete_all_unused_submodules()
        late.graph.eliminate_dead_code()
        late.recompile()
        assert not hasattr(late, "dead")
        for value, want in zip(inputs, expected):
            assert torch.equal(late(value), want)
            assert torch.equal(converted(value), want)
            assert torch.equal(source(value), want)
        assert outer["adapted_calls"] >= 1
    assert GraphModule.delete_all_unused_submodules is original
    assert tuple((node.op, node.target) for node in source.body.graph.nodes) == source_nodes
    assert torch.equal(source.body.constants.scale, source_scale)
