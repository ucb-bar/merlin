"""Tests for the frontend-adapter registry + extended bundle variants."""
from __future__ import annotations

import pytest

from merlin.baselines import bundle as _bundle
from merlin.frontends import registry as fr


def test_registry_lists_adapters_and_resolves_modules():
    names = fr.list_adapters()
    assert {"m2m", "gguf"} <= set(names)
    for name in names:
        mod = fr.get_adapter(name)          # lazy import must succeed
        assert mod.NAME == name
        assert callable(mod.can_handle) and callable(mod.ingest)


def test_source_routing_picks_specific_adapter_over_catch_all():
    assert fr.for_source("/models/tinyllama-Q6_K.gguf").NAME == "gguf"
    assert fr.for_source("TinyLlama/TinyLlama-1.1B-Chat-v1.0").NAME == "m2m"
    assert fr.for_source("tiny_llama").NAME == "m2m"


def test_gguf_adapter_recognises_and_rejects_a_missing_checkpoint():
    """GGUF ingest is implemented now, so it must fail LOCALLY on a bad path.

    This test used to assert `NotImplementedError` and went stale when ingest landed. What it
    caught instead was worse than a stale assertion: ingest shelled out to model2MLIR, whose
    transformers treated the unresolvable path as a Hub repo id and hit the network, so the
    suite failed with a HuggingFace 404 inside a CalledProcessError — slow, offline-hostile,
    and silent about the real mistake (a wrong path).
    """
    gguf = fr.get_adapter("gguf")
    assert gguf.can_handle("x.gguf") and not gguf.can_handle("tiny_llama")
    with pytest.raises(FileNotFoundError, match="GGUF checkpoint not found"):
        gguf.ingest("x.gguf", model="tiny_llama", variant="fp6")


def test_m2m_adapter_resolves_bundle_without_requiring_files():
    m2m = fr.get_adapter("m2m")
    assert m2m.can_handle("tiny_llama") and not m2m.can_handle("x.gguf")
    b = m2m.resolve("tiny_llama", "int8")
    assert isinstance(b, _bundle.CaptureBundle)
    assert b.model == "tiny_llama" and b.variant == "int8"


def test_bundle_variants_extended():
    for v in ("fp32", "fp16", "bf16", "int8", "fp8", "fp6", "fp4", "mixed"):
        b = _bundle.resolve("tiny_llama", v)          # must not raise
        assert b.variant == v
    with pytest.raises(ValueError):
        _bundle.resolve("tiny_llama", "not_a_variant")


def test_per_variant_tolerance_override(monkeypatch):
    # Default falls back to the per-model tolerance regardless of variant.
    assert _bundle.tolerance("tiny_llama", "fp4") == _bundle.tolerance("tiny_llama")
    # A measured override wins when present.
    monkeypatch.setitem(_bundle._VARIANT_TOL, ("tiny_llama", "fp4"), (0.95, 5e-2))
    assert _bundle.tolerance("tiny_llama", "fp4") == (0.95, 5e-2)
    assert _bundle.resolve("tiny_llama", "fp4").tolerance == (0.95, 5e-2)
