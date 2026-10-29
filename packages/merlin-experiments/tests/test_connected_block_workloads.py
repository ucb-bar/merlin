"""Public connected graphs stage through the original roster and retain all observable results."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from merlin_experiments.phase0 import workload_roster as WR

from merlin.common.paths import repo_root

torch = pytest.importorskip("torch")
ROOT = repo_root() / "examples/workloads/connected_blocks"


def _load(root):
    spec = importlib.util.spec_from_file_location("connected_source", Path(root) / "loader.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def staged(tmp_path):
    return WR.stage(ROOT / "phase0-workloads.yaml", tmp_path / "staged")["performance_applications"]


@pytest.mark.parametrize("family", ["residual_block", "attention_cache", "repeated_inference"])
def test_all_selected_sizes_execute_with_exact_staged_profile_and_complete_outputs(staged, family):
    for size in ("small", "medium", "representative"):
        row = staged[f"{family}_{size}"]
        root = Path(row["workload_root"])
        loader = _load(root)
        model, inputs = loader.get_model_and_inputs()
        with torch.no_grad():
            outputs = model(*inputs)
        assert isinstance(outputs, tuple) and all(torch.isfinite(x).all() for x in outputs)
        profile = loader.validate_profile(json.loads((root / "profile.json").read_text()))
        assert model.session_provenance["profile_sha256"] == row["profile_sha256"]
        assert model.session_provenance["source_role"] == "handauthored_compiler_test"
        assert hashlib.sha256((root / "loader.py").read_bytes()).hexdigest() == row["loader_sha256"]
        if family == "residual_block":
            side = (profile["side"] + profile["stride"] - 1) // profile["stride"]
            assert len(outputs) == 1 and outputs[0].shape == (1, profile["out_channels"], side, side)
            assert not isinstance(model.blocks[0].shortcut, torch.nn.Identity)
        elif family == "attention_cache":
            width = profile["hidden"] // profile["heads"]
            assert outputs[0].shape == inputs[0].shape and len(outputs) == 3
            assert outputs[1].shape == outputs[2].shape == (
                1, profile["kv_heads"], profile["prefix"] + profile["tokens"], width)
            torch.testing.assert_close(outputs[1][:, :, :profile["prefix"]], inputs[1], rtol=0, atol=0)
            torch.testing.assert_close(outputs[2][:, :, :profile["prefix"]], inputs[2], rtol=0, atol=0)
        else:
            assert len(outputs) == profile["steps"]
            for x, output in zip(inputs[2:], outputs, strict=True):
                torch.testing.assert_close(output, torch.relu(x @ inputs[0]) @ inputs[1], rtol=0, atol=0)
            assert not torch.equal(inputs[2], inputs[3])
            assert all(x.shape == (profile["rows"], profile["output"]) for x in outputs)


@pytest.mark.parametrize("family", ["residual_block", "attention_cache", "repeated_inference"])
def test_torch_export_preserves_complete_connected_program_and_outputs(staged, family):
    loader = _load(staged[f"{family}_small"]["workload_root"])
    model, inputs = loader.get_model_and_inputs()
    exported = torch.export.export(model, inputs)
    with torch.no_grad():
        expected, actual = model(*inputs), exported.module()(*inputs)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    calls = [n for n in exported.graph.nodes if n.op == "call_function"]
    if family == "attention_cache":
        names = {str(n.target) for n in calls}
        assert any("softmax" in name for name in names)
        assert any("silu" in name for name in names)
        assert any("rms_norm" in name for name in names)
        assert any("cat" in name for name in names)
    if family == "repeated_inference":
        placeholders = [n for n in exported.graph.nodes if n.op == "placeholder"]
        assert len(placeholders[0].users) == len(inputs) - 2
        assert len(placeholders[1].users) == len(inputs) - 2


def test_attention_is_causal_and_cache_inputs_are_borrowed_not_mutated(staged):
    loader = _load(staged["attention_cache_small"]["workload_root"])
    model, inputs = loader.get_model_and_inputs()
    saved = tuple(x.clone() for x in inputs)
    changed = inputs[0].clone()
    changed[:, -1] += 0.5
    with torch.no_grad():
        before = model(*inputs)
        after = model(changed, *inputs[1:])
    torch.testing.assert_close(before[0][:, :-1], after[0][:, :-1], rtol=0, atol=0)
    assert not torch.equal(before[1][:, :, -1], after[1][:, :, -1])
    for a, b in zip(inputs, saved, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("change", [dict(heads=3), dict(hidden=15), dict(kv_heads=0),
                                    dict(prefix=-1), dict(tokens=True), dict(target_schedule="invented"),
                                    dict(hidden=8192, ffn=8192)])
def test_invalid_or_excessive_source_profiles_are_refused(change):
    loader = _load(ROOT)
    profile = dict(schema=loader.PROFILE_SCHEMA, workload_id=loader.WORKLOAD_ID, kind="attention")
    with pytest.raises(ValueError):
        loader.validate_profile({**profile, **change})


def test_zero_prefix_and_identity_shortcut_are_valid_semantic_variants():
    loader = _load(ROOT)
    common = dict(schema=loader.PROFILE_SCHEMA, workload_id=loader.WORKLOAD_ID)
    model, inputs = loader.build(dict(**common, kind="attention", prefix=0))
    assert model(*inputs)[1].shape[2] == inputs[0].shape[1]
    model, inputs = loader.build(dict(**common, kind="residual", channels=8, out_channels=8, stride=1))
    assert isinstance(model.blocks[0].shortcut, torch.nn.Identity)
    assert model(*inputs)[0].shape == inputs[0].shape


@pytest.mark.parametrize('size', ['small', 'representative'])
def test_conv_relu_pool_profiles_are_real_connected_sources_with_all_outputs(staged, size):
    root = Path(staged['conv_pool_'+size]['workload_root'])
    loader = _load(root)
    profile = loader.validate_profile(json.loads((root/'profile.json').read_text()))
    model, inputs = loader.get_model_and_inputs()
    with torch.no_grad():
        output, = model(*inputs)
        expected = torch.nn.functional.max_pool2d(torch.relu(torch.nn.functional.conv2d(
            inputs[0], model.convolution.weight, stride=profile['conv_stride'], padding=profile['conv_padding'])),
            profile['pool_kernel'], stride=profile['pool_stride'], padding=profile['pool_padding'])
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.shape == ((1, 16, 5, 6) if size == 'small' else (1, 64, 56, 56))
    exported = torch.export.export(model, inputs)
    calls = [node for node in exported.graph.nodes if node.op == 'call_function']
    convolution = next(node for node in calls if 'conv2d' in str(node.target))
    relu = next(node for node in calls if 'relu' in str(node.target))
    pool = next(node for node in calls if 'max_pool2d' in str(node.target))
    assert relu.args[0] is convolution and pool.args[0] is relu
    with torch.no_grad():
        actual, = exported.module()(*inputs)
    torch.testing.assert_close(actual, output, rtol=0, atol=0)


def test_conv_pool_geometry_is_not_tied_to_three_by_three_stride_two():
    loader = _load(ROOT)
    profile = dict(schema=loader.PROFILE_SCHEMA, workload_id=loader.WORKLOAD_ID, kind='conv_pool',
                   channels=2, out_channels=5, height=11, width=13, conv_kernel=5, conv_padding=2,
                   conv_stride=1, pool_kernel=2, pool_stride=1, pool_padding=0)
    model, inputs = loader.build(profile)
    output, = model(*inputs)
    assert output.shape == (1, 5, 10, 12)
    for change in ({'conv_kernel': 19, 'conv_padding': 0}, {'pool_padding': 2},
                   {'conv_stride': 0}, {'height': True}, {'out_channels': 8192, 'channels': 8192}):
        with pytest.raises(ValueError):
            loader.validate_profile({**profile, **change})
