"""The optional causal LM checkpoint pin must reach HF and fail closed offline."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

LOADER = Path(__file__).resolve().parents[3] / "examples/workloads/causal_lm_section/loader.py"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"


@pytest.fixture
def loader(monkeypatch):
    # Import only the loader's pin boundary; never allocate a model or import Torch here.
    torch = types.ModuleType("torch")
    torch.float32 = object()
    torch.nn = types.SimpleNamespace(Module=object)
    monkeypatch.setitem(sys.modules, "torch", torch)
    spec = importlib.util.spec_from_file_location("test_causal_lm_loader", LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AutoModel:
    def __init__(self, commit):
        self.commit = commit
        self.calls = []

    def from_pretrained(self, model_id, **kwargs):
        self.calls.append((model_id, kwargs))
        return types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash=self.commit))


def test_unpinned_path_keeps_original_model_arguments(loader, monkeypatch):
    monkeypatch.delenv("CAUSAL_LM_REVISION", raising=False)
    auto = AutoModel(None)
    assert loader._load_pretrained(auto, "TinyLlama/example", "eager").config._commit_hash is None
    assert auto.calls == [
        (
            "TinyLlama/example",
            {
                "dtype": loader.torch.float32,
                "use_cache": False,
                "attn_implementation": "eager",
            },
        )
    ]


def test_pinned_offline_revision_reaches_cache_and_model(loader, monkeypatch):
    monkeypatch.setenv("CAUSAL_LM_REVISION", REVISION)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    auto = AutoModel(REVISION)
    loader._load_pretrained(auto, "TinyLlama/example", "eager")
    assert calls == [{"repo_id": "TinyLlama/example", "revision": REVISION, "local_files_only": True}]
    assert auto.calls[0][1]["revision"] == REVISION
    assert auto.calls[0][1]["local_files_only"] is True


def test_missing_cached_revision_fails_before_model_load(loader, monkeypatch):
    monkeypatch.setenv("CAUSAL_LM_REVISION", REVISION)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    hub = types.ModuleType("huggingface_hub")

    def missing(**_kwargs):
        raise FileNotFoundError("exact revision absent from offline cache")

    hub.snapshot_download = missing
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    auto = AutoModel(REVISION)
    with pytest.raises(FileNotFoundError, match="exact revision absent"):
        loader._load_pretrained(auto, "TinyLlama/example", "eager")
    assert auto.calls == []


def test_wrong_loaded_revision_is_rejected(loader, monkeypatch):
    monkeypatch.setenv("CAUSAL_LM_REVISION", REVISION)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **_kwargs: "/cached/exact-revision"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    auto = AutoModel("0" * 40)
    with pytest.raises(RuntimeError, match="differs from CAUSAL_LM_REVISION"):
        loader._load_pretrained(auto, "TinyLlama/example", "eager")
    assert auto.calls[0][1]["revision"] == REVISION


def test_revision_requires_exact_commit(loader, monkeypatch):
    monkeypatch.setenv("CAUSAL_LM_REVISION", "main")
    auto = AutoModel(None)
    with pytest.raises(RuntimeError, match="40-character lowercase commit SHA"):
        loader._load_pretrained(auto, "TinyLlama/example", "eager")
    assert auto.calls == []
