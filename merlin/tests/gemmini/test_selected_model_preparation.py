"""Check algebraic preparation and the existing functional cache ABI independently of weights."""

import importlib

import numpy as np
import pytest

from merlin.common.paths import repo_root

torch = pytest.importorskip("torch")


@pytest.fixture
def source(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/comparisons/tvm"))
    return importlib.import_module("selected_tinyllama_loader")


def test_norm_migration_preserves_shared_projection_math(source):
    torch.manual_seed(67)
    norm = torch.nn.LayerNorm(8, bias=False, dtype=torch.float64)
    projections = [torch.nn.Linear(8, n, bias=False, dtype=torch.float64) for n in (16, 4, 4)]
    values = torch.randn(3, 8, dtype=torch.float64)
    before = [p(norm(values)).detach() for p in projections]
    scale = source.migration_scale(torch.arange(1, 9, dtype=torch.float64), projections)
    source.fold_migration(norm, projections, scale)
    for projection, expected in zip(projections, before, strict=True):
        torch.testing.assert_close(projection(norm(values)), expected, rtol=1e-12, atol=1e-12)


def test_o_migration_preserves_projection_and_bias(source):
    torch.manual_seed(71)
    projection = torch.nn.Linear(8, 5, dtype=torch.float64)
    values = torch.randn(4, 8, dtype=torch.float64)
    expected = projection(values).detach()
    scale = source.migration_scale(values.abs().amax(0), [projection])
    prepared = source.ScaledLinear(projection, scale)
    torch.testing.assert_close(prepared(values), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("all_zero_statistics", [False, True])
def test_gated_channel_split_and_zero_padding_preserve_complete_mlp(source, all_zero_statistics):
    torch.manual_seed(73)

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(8, 19, bias=False, dtype=torch.float64)
            self.up_proj = torch.nn.Linear(8, 19, bias=False, dtype=torch.float64)
            self.down_proj = torch.nn.Linear(19, 8, bias=False, dtype=torch.float64)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    model = MLP()
    x = torch.randn(3, 8, dtype=torch.float64)
    before = model(x).detach()
    statistics = torch.zeros(19) if all_zero_statistics else torch.arange(1, 20).float()
    row = source.split_mlp(model, statistics)
    torch.testing.assert_close(model(x), before, rtol=1e-12, atol=1e-12)
    assert model.gate_proj.weight.shape[0] == row["padded_width"]
    assert row["padded_width"] % 16 == 0
    assert not torch.count_nonzero(model.down_proj.weight[:, row["expanded_width"] :])


def test_canonical_prefill_decode_retains_floating_cache_and_layer_positions():
    pytest.importorskip("m2m")
    from m2m.capture.causal_session import FixedCacheDecodeStep, FixedCachePrefillStage
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        values = FixedCachePrefillStage(model, 64)(torch.arange(32).reshape(1, 32))
        logits, cache, positions = values
        assert logits.shape == (1, 64)
        assert cache.shape == (2, 2, 1, 1, 64, 8) and cache.dtype == torch.float32
        np.testing.assert_array_equal(positions.numpy(), [32, 32])
        previous = cache.clone()
        output = FixedCacheDecodeStep(model, 64)(torch.tensor([[32]]), cache, positions)
        assert torch.equal(cache, previous)
        np.testing.assert_array_equal(output[2].numpy(), [33, 33])
        assert torch.equal(output[1][..., :32, :], cache[..., :32, :])


def test_decode_calibration_freezes_real_contexts_and_replays_without_recalculation(source, tmp_path, monkeypatch):
    pytest.importorskip("m2m")
    from m2m.capture.causal_session import FixedCacheDecodeStep, FixedCachePrefillStage
    from transformers import LlamaConfig, LlamaForCausalLM

    monkeypatch.setenv("SELECTED_RELEASE", str(tmp_path))
    monkeypatch.setenv("SELECTED_INPUTS", str(tmp_path))
    torch.set_num_threads(1)
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=256,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    tokens = np.tile(np.arange(160, dtype=np.int64), (128, 1))
    np.savez(tmp_path / "tiny-calibration.npz", tokens=tokens)
    (tmp_path / "tiny-range-identity-v2.json").write_text('{"test": "independent tiny model"}')
    with torch.no_grad():
        _, cache, positions = FixedCachePrefillStage(model, 64)(torch.from_numpy(tokens[:1, :32].copy()))
    decode = FixedCacheDecodeStep(model, 64).eval()
    selected = {}
    for index, arguments in enumerate(source.get_calibration_inputs(decode, (torch.tensor([[32]]), cache, positions))):
        if index in (0, 7, 8, 1023):
            selected[index] = tuple(value.clone() for value in arguments)
    assert index + 1 == 1024
    for index, arguments in selected.items():
        assert arguments[0].item() == 32 + index % 8
        assert arguments[2].item() == 32 + index % 8
        assert not torch.count_nonzero(arguments[1][..., 32 + index % 8 :, :])

    def refuse(*args, **kwargs):
        raise AssertionError("Replay must not regenerate calibration state")

    monkeypatch.setattr(decode, "forward", refuse)
    for index, arguments in enumerate(source.get_calibration_inputs(decode, (torch.tensor([[32]]), cache, positions))):
        if index in selected:
            for actual, expected in zip(arguments, selected[index], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (tmp_path / "tiny-range-identity-v2.json").write_text('{"test": "changed"}')
    with pytest.raises(ValueError, match="identity changed"):
        next(source.get_calibration_inputs(decode, (torch.tensor([[32]]), cache, positions)))


def test_frozen_export_loader_keeps_data_dependent_outputs_and_eval_lifecycle(source, tmp_path, monkeypatch):
    loader = importlib.import_module("selected_export_loader")

    class Program(torch.nn.Module):
        def forward(self, value, position):
            return value * 3 + 1, position + 1

    original = Program().eval()
    inputs = (torch.ones(2, 3), torch.tensor([32, 32]))
    path = tmp_path / "program.pt2"
    torch.export.save(torch.export.export(original, inputs), path)
    monkeypatch.setenv("SELECTED_PREPARED_PROGRAM", str(path))
    module, captured = loader.get_model_and_inputs()
    module.eval()
    alternate = (captured[0] + 10, captured[1] + 7)
    for actual, expected in zip(module(*alternate), original(*alternate), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_causal_functional_boundary_supports_repeated_export_with_real_module_inventory(source):
    pytest.importorskip("m2m")
    from m2m.capture.causal_session import FixedCachePrefillStage
    from transformers import LlamaConfig, LlamaForCausalLM

    adapter = importlib.import_module("selected_tinyllama_functional_loader")
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=256,
        max_position_embeddings=128,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    inputs = (torch.arange(32).reshape(1, 32),)
    stage = FixedCachePrefillStage(model, 64).eval()
    with torch.no_grad():
        expected = stage(*inputs)
    functional = adapter.functional_boundary(stage, inputs)
    assert isinstance(functional.get_submodule("lm.model.layers.0.mlp.gate_proj"), torch.nn.Linear)
    for _ in range(2):
        exported = torch.export.export(functional, inputs)
        with torch.no_grad():
            actual = exported.module()(*inputs)
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
