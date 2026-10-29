"""Optional iteration-workload profile keys: strict validation, and defaults that leave the model alone."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import types

import pytest

from merlin.common.paths import repo_root

_SCHEMA = "merlin.iteration_workload_profile.v1"
_WORKLOADS = ("residual_cnn", "causal_decoder", "multimodal_policy")


def _load(tmp_path, monkeypatch, workload, profile=None):
    """Import a private copy of a loader, next to ``profile`` when one is given.

    Validation needs no framework: without torch installed, a placeholder module satisfies the
    loader's class definitions, and only the profile functions may be called.
    """
    if importlib.util.find_spec("torch") is None:
        stub = types.ModuleType("torch")
        stub.nn = types.SimpleNamespace(Module=object)
        monkeypatch.setitem(sys.modules, "torch", stub)
    directory = tmp_path / workload
    directory.mkdir(parents=True)
    source = directory / "loader.py"
    shutil.copyfile(repo_root() / "examples/workloads" / workload / "loader.py", source)
    if profile is not None:
        (directory / "profile.json").write_text(json.dumps(profile))
    spec = importlib.util.spec_from_file_location(f"profile_probe_{workload}_{tmp_path.name}", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile(workload, **fields):
    return {"schema": _SCHEMA, "workload_id": workload, **fields}


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_absent_profile_selects_the_defaults_without_provenance(tmp_path, monkeypatch, workload):
    module = _load(tmp_path, monkeypatch, workload)
    profile, digest = module._selected_profile()
    assert digest is None
    required = {"channels": 8, "spatial_side": 16, "blocks": 1} if workload == "residual_cnn" else {}
    explicit = module.validate_profile(_profile(workload, **required))
    assert profile == {key: value for key, value in explicit.items() if key not in {"schema", "workload_id"}}


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_identity_and_unknown_keys_are_refused(tmp_path, monkeypatch, workload):
    module = _load(tmp_path, monkeypatch, workload)
    required = {"channels": 8, "spatial_side": 16, "blocks": 1} if workload == "residual_cnn" else {}
    base = _profile(workload, **required)
    module.validate_profile(base)
    with pytest.raises(ValueError, match="unsupported shape"):
        module.validate_profile({**base, "unrecognized": 1})
    with pytest.raises(ValueError, match="unsupported shape"):
        module.validate_profile([base])
    with pytest.raises(ValueError, match="unsupported identity"):
        module.validate_profile({**base, "workload_id": "coverage_mlp"})
    with pytest.raises(ValueError, match="unsupported identity"):
        module.validate_profile({**base, "schema": "merlin.iteration_workload_profile.v2"})


def test_residual_profile_keys_are_strictly_typed(tmp_path, monkeypatch):
    module = _load(tmp_path, monkeypatch, "residual_cnn")
    base = _profile("residual_cnn", channels=12, spatial_side=20, blocks=2)
    # The variant producer's required-only shape keeps every optional pattern off.
    assert module.validate_profile(base) == {
        **base,
        "stem": "conv3",
        "stem_pool": False,
        "downsample": False,
        "bottleneck": False,
        "batchnorm": False,
    }
    full = {**base, "stem": "conv7s2", "stem_pool": True, "downsample": True, "bottleneck": True, "batchnorm": True}
    assert module.validate_profile(full) == full
    with pytest.raises(ValueError, match="unsupported shape"):
        module.validate_profile({key: value for key, value in full.items() if key != "blocks"})
    for key, value in (
        ("stem", "conv5"),
        ("stem", 7),
        ("stem_pool", 1),
        ("downsample", "true"),
        ("bottleneck", None),
        ("batchnorm", 0),
        ("channels", True),
        ("blocks", 17),
    ):
        with pytest.raises(ValueError, match=f"invalid {key}"):
            module.validate_profile({**full, key: value})


def test_decoder_profile_keys_are_strictly_typed(tmp_path, monkeypatch):
    module = _load(tmp_path, monkeypatch, "causal_decoder")
    defaults = module.validate_profile(_profile("causal_decoder"))
    assert {key: defaults[key] for key in ("seq", "hidden", "heads", "kv_heads", "ffn")} == {
        "seq": 8,
        "hidden": 32,
        "heads": 4,
        "kv_heads": 4,
        "ffn": 64,
    }
    assert defaults["rope"] is False and defaults["decode_step"] is False and defaults["cache_len"] is None
    grouped = module.validate_profile(_profile("causal_decoder", hidden=48, heads=6, kv_heads=2, seq=12, rope=True))
    assert grouped["kv_heads"] == 2 and grouped["cache_len"] is None
    step = module.validate_profile(_profile("causal_decoder", seq=12, decode_step=True))
    assert step["cache_len"] == 11
    assert module.validate_profile(_profile("causal_decoder", decode_step=True, cache_len=5))["cache_len"] == 5
    for fields, message in (
        ({"hidden": 72, "heads": 8}, "invalid hidden"),
        ({"seq": 17}, "invalid seq"),
        ({"seq": 0}, "invalid seq"),
        ({"heads": 3}, "heads dividing hidden"),
        ({"kv_heads": 3}, "kv_heads dividing heads"),
        ({"heads": 32, "rope": True}, "even head width"),
        ({"rope": 1}, "invalid rope"),
        ({"decode_step": "yes"}, "invalid decode_step"),
        ({"kv_heads": None}, "null field"),
        ({"cache_len": 4}, "without a decode step"),
        ({"decode_step": True, "cache_len": 16}, "invalid cache_len"),
        ({"decode_step": True, "seq": 1}, "invalid cache_len"),
        ({"ffn": 2.0}, "invalid ffn"),
    ):
        with pytest.raises(ValueError, match=message):
            module.validate_profile(_profile("causal_decoder", **fields))


def test_policy_profile_keys_are_strictly_typed(tmp_path, monkeypatch):
    module = _load(tmp_path, monkeypatch, "multimodal_policy")
    assert module.validate_profile(_profile("multimodal_policy")) == {
        "tokens": 4,
        "queries": 1,
        "gelu_mlp": False,
        "time_embedding": False,
    }
    chosen = _profile("multimodal_policy", tokens=13, queries=3, gelu_mlp=True, time_embedding=True)
    assert module.validate_profile(chosen) == {"tokens": 13, "queries": 3, "gelu_mlp": True, "time_embedding": True}
    for key, value in (("tokens", 0), ("tokens", 33), ("queries", True), ("gelu_mlp", 1), ("time_embedding", None)):
        with pytest.raises(ValueError, match=f"invalid {key}"):
            module.validate_profile({**chosen, key: value})


@pytest.mark.parametrize("workload", _WORKLOADS)
def test_profile_must_be_an_ordinary_adjacent_file(tmp_path, monkeypatch, workload):
    module = _load(tmp_path, monkeypatch, workload)
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps(_profile(workload)))
    (tmp_path / workload / "profile.json").symlink_to(target)
    with pytest.raises(ValueError, match="adjacent ordinary file"):
        module._selected_profile()


def test_residual_patterns_build_and_run(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    plain = _load(tmp_path / "plain", monkeypatch, "residual_cnn")
    model, (image,) = plain.get_model_and_inputs()
    assert not hasattr(model, "session_provenance")
    assert isinstance(model.stem, torch.nn.Conv2d) and model.stem.kernel_size == (3, 3)
    assert not any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules())
    assert tuple(model(image).shape) == (1, 4)

    profile = _profile(
        "residual_cnn",
        channels=12,
        spatial_side=20,
        blocks=2,
        stem="conv7s2",
        stem_pool=True,
        downsample=True,
        bottleneck=True,
        batchnorm=True,
    )
    module = _load(tmp_path, monkeypatch, "residual_cnn", profile)
    model, (image,) = module.get_model_and_inputs()
    stem = model.stem[0]
    assert (stem.kernel_size, stem.stride, stem.padding, stem.bias) == ((7, 7), (2, 2), (3, 3), None)
    assert [model.conv1[0].kernel_size, model.conv2[0].kernel_size, model.expand[0].kernel_size] == [
        (1, 1),
        (3, 3),
        (1, 1),
    ]
    assert model.conv1[0].out_channels == 3 and model.expand[0].out_channels == 12
    assert len(model.residuals) == 1 and len(model.residuals[0]) == 3
    assert model.down_conv1[0].stride == (2, 2) and model.down_shortcut[0].kernel_size == (1, 1)
    assert model.down_shortcut[0].stride == (2, 2) and model.head.in_features == 24
    norms = [layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d)]
    assert len(norms) == 1 + 3 + 3 + 3 and all(not layer.training for layer in norms)
    assert all(not torch.equal(layer.running_var, torch.ones_like(layer.running_var)) for layer in norms)
    assert all(bool((layer.running_var > 0).all()) for layer in norms)
    result = model(image)
    assert tuple(result.shape) == (1, 4) and bool(torch.isfinite(result).all())
    assert torch.equal(result, model(image))


def test_decode_step_matches_the_full_sequence_at_its_position(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    shape = {"seq": 10, "hidden": 48, "heads": 6, "kv_heads": 2, "ffn": 80, "rope": True}
    full = _load(tmp_path / "full", monkeypatch, "causal_decoder", _profile("causal_decoder", **shape))
    model, (tokens,) = full.get_model_and_inputs()
    assert tuple(tokens.shape) == (1, 10)
    with torch.no_grad():
        logits = model(tokens)
        _, key, value = model.qkv(model.norm(model.embedding(tokens))).split(model.split_sizes, dim=-1)
        key = model._rotate(key.reshape(1, 10, 2, 8).transpose(1, 2))
        value = value.reshape(1, 10, 2, 8).transpose(1, 2)
    assert tuple(logits.shape) == (1, 10, 64)

    step = _load(tmp_path, monkeypatch, "causal_decoder", _profile("causal_decoder", **shape, decode_step=True))
    step_model, (token, key_cache, value_cache) = step.get_model_and_inputs()
    assert tuple(token.shape) == (1, 1) and tuple(key_cache.shape) == tuple(value_cache.shape) == (1, 2, 9, 8)
    assert not hasattr(step_model, "mask")
    assert step_model.session_provenance["workload_id"] == "causal_decoder"
    with torch.no_grad():
        assert tuple(step_model(token, key_cache, value_cache).shape) == (1, 1, 64)
        cached = step_model(tokens[:, 9:], key[:, :, :9], value[:, :, :9])
    torch.testing.assert_close(cached, logits[:, 9:], rtol=1e-5, atol=1e-5)


def test_policy_patterns_build_and_run(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    profile = _profile("multimodal_policy", tokens=13, queries=3, gelu_mlp=True, time_embedding=True)
    module = _load(tmp_path, monkeypatch, "multimodal_policy", profile)
    model, inputs = module.get_model_and_inputs()
    image, tokens, state, timestep = inputs
    assert tuple(tokens.shape) == (1, 13) and tuple(timestep.shape) == (1,)
    result = model(*inputs)
    assert tuple(result.shape) == (1, 3, 4) and bool(torch.isfinite(result).all())
    shifted = model(image, tokens, state, timestep + 1)
    assert not torch.equal(result, shifted)
    assert len(model.session_provenance["profile_sha256"]) == 64
