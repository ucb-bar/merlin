"""Reduced sources preserve real nonlinear/stateful definitions and refuse drift."""

import ast
import hashlib
import json

import pytest
from merlin_experiments.phase0.development_reduced_models import KINDS, render


def test_complete_sources_keep_periods_full_decode_and_explicit_numeric_boundary():
    sources = {kind: render(kind) for kind in KINDS}
    for source in sources.values():
        ast.parse(source)
        assert "def get_model_and_inputs():" in source
    assert "ResNet(Bottleneck, [1, 1, 1, 1], num_classes=1000)" in sources["residual_classifier"]
    decoder = sources["grouped_decoder"]
    assert "range(32)" in decoder and "past_key_values=cache" in decoder
    assert "layer.keys, layer.values" in decoder
    action = sources["vision_action_flow"]
    assert "num_vlm_layers=2" in action and '"num_hidden_layers"] = 2' in action
    assert "self.network.sample_actions" in action and "noise=noise" in action
    assert "state_mean" in action and "action_std" in action
    assert "never loads pretrained weights" in action
    with pytest.raises(ValueError, match="complete public"):
        render("unknown")


def test_original_definition_body_and_pin_are_both_required(tmp_path):
    # Exercise the generated loader's actual extractor with a nontrivial body.
    source = "def nonlinear(x):\n    return (x * x + 3) // 2\n"
    leaf = tmp_path / "original.py"
    leaf.write_text(source)
    pins = {"original.py": hashlib.sha256(source.encode()).hexdigest()}
    (tmp_path / "source-pins.json").write_text(json.dumps(pins))
    program = tmp_path / "loader.py"
    program.write_text(render("residual_classifier"))
    # Only the shared pin/extraction definitions are needed; this is not a fake model execution.
    tree = ast.parse(program.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"leaf", "definitions"}]
    namespace = dict(ROOT=tmp_path, PINS=pins, hashlib=hashlib, ast=ast)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(program), "exec"), namespace)
    namespace["definitions"]("original.py", ["nonlinear"], namespace)
    assert namespace["nonlinear"](-5) == 14
    leaf.write_text(source.replace("+ 3", "+ 5"))
    with pytest.raises(ValueError, match="pin changed"):
        namespace["definitions"]("original.py", ["nonlinear"], namespace)
