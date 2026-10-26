"""Synthetic fixtures must preserve frozen quantization and functional cache state."""

import importlib.util

import numpy as np
import pytest

from merlin.common.paths import repo_root

torch = pytest.importorskip("torch")


@pytest.fixture
def source():
    path = repo_root() / "examples/gemmini/comparisons/tvm/synthetic_tinyllama.py"
    spec = importlib.util.spec_from_file_location("synthetic_source_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rne_negative_ties_and_asymmetric_activation_endpoint(source):
    values = np.array([-129, -128, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 127, 128], np.float32)
    np.testing.assert_array_equal(source.rne_codes(values, 1), [-128, -128, -2, -2, 0, 0, 2, 2, 127, 127])


def test_weight_groups_share_scale_and_keep_partial_tail(source):
    weights = np.ones((19, 3), np.float32)
    weights[0, 0] = -127
    weights[16:] *= 2
    codes, scales = source.grouped_weight(weights)
    np.testing.assert_array_equal(scales[:16], np.ones(16))
    assert len(set(scales[16:].tolist())) == 1
    assert scales[16] != scales[0]
    assert codes.min() == -127 and codes.max() == 127


def test_projection_matches_independent_integer_reference_and_does_not_recalibrate(source):
    generator = torch.Generator().manual_seed(17)
    module = source.StaticLinear(torch.randn(19, 7, generator=generator), 0.125)
    values = torch.randn(3, 7, generator=generator)
    scales = module.activation_scale.clone(), module.weight_scale.clone()
    torch.testing.assert_close(module(values), module.run(values, oracle=True), rtol=0, atol=0)
    module(values * 10000)
    assert torch.equal(scales[0], module.activation_scale)
    assert torch.equal(scales[1], module.weight_scale)


def test_wide_integer_oracle_refuses_overflow(source):
    left = torch.full((1, 140000), -128, dtype=torch.int8)
    right = torch.full((140000, 1), -128, dtype=torch.int8)
    with pytest.raises(ValueError, match="exceeds INT32"):
        source.integer_mm(left, right, oracle=True)


def test_connected_attention_cache_is_functional_and_keeps_prefix(source):
    config = dict(num_attention_heads=2, num_key_value_heads=1, hidden_size=8, rope_theta=10000)
    generator = torch.Generator().manual_seed(23)
    arrays = {
        "model.layers.0." + key: torch.ones(8) for key in ("input_layernorm.weight", "post_attention_layernorm.weight")
    }
    for name, group, n, k in (
        ("q", "self_attn", 8, 8),
        ("k", "self_attn", 4, 8),
        ("v", "self_attn", 4, 8),
        ("o", "self_attn", 8, 8),
        ("gate", "mlp", 16, 8),
        ("up", "mlp", 16, 8),
        ("down", "mlp", 8, 16),
    ):
        arrays[f"model.layers.0.{group}.{name}_proj.weight"] = torch.randn(n, k, generator=generator)

    class Archive:
        def get_tensor(self, name):
            return arrays[name]

    scales = {name: 0.125 for name in ("qkv", "o", "gate_up", "down", "query", "key", "value")}
    scales["probability"] = float(np.float32(1 / 127))
    layer = source.Layer(Archive(), 0, scales, config)
    hidden = torch.randn(3, 8, generator=generator)
    empty = torch.zeros(2, 1, 64, 4, dtype=torch.int8)
    _, cache = layer.attention(hidden, empty, torch.tensor([0]), True)
    previous = cache.clone()
    args = (hidden[:1], cache, torch.tensor([3]))
    candidate = source.Chunk(layer, "attention", False)
    actual = candidate(*args)
    expected = candidate.run(args, oracle=True)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(previous, cache)
    assert torch.equal(actual[1][..., :3, :], previous[..., :3, :])
    assert not torch.count_nonzero(actual[1][..., 4:, :])


def test_replay_preserves_frozen_reference_bytes_instead_of_recomputing(tmp_path, monkeypatch):
    pytest.importorskip("tvm")
    folder = repo_root() / "examples/gemmini/comparisons/tvm"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("synthetic_prepare_test", folder / "prepare_synthetic_tinyllama.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)

    class Module(torch.nn.Module):
        kind, prefill = "o", False

        def forward(self, value):
            return value + 1

        def run(self, *args, **kwargs):
            raise AssertionError("Frozen reference must not be replaced by another oracle evaluation")

    value = np.array([[1, -2]], np.float32)
    frozen = tmp_path / "frozen.npz"
    np.savez_compressed(frozen, input_0=value, expected=value + 1)
    result = prepare.freeze_chunk(tmp_path, "example", Module(), (torch.zeros(1, 2),), [], frozen_fixture=frozen)
    produced = tmp_path / "chunks/example/fixture.npz"
    assert produced.read_bytes() == frozen.read_bytes()
    assert result["oracle_max_abs_errors"] == [0.0]
