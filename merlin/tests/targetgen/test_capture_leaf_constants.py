"""The capture worker binds lifted constants to the exact exported ABI name."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from merlin.targetgen import _m2m_capture_worker as worker


class _Model:
    def named_buffers(self):
        return iter(())


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.int64)

    def detach(self):
        return self


def _export(monkeypatch, *, value):
    torch = ModuleType("torch")
    torch.__path__ = []
    torch_export = ModuleType("torch.export")
    torch_export.__path__ = []
    graph_signature = ModuleType("torch.export.graph_signature")
    graph_signature.InputKind = SimpleNamespace(CONSTANT_TENSOR="constant")
    m2m = ModuleType("m2m")
    m2m.__path__ = []
    m2m_capture = ModuleType("m2m.capture")
    m2m_capture.__path__ = []
    bundle = ModuleType("m2m.capture.bundle")
    bundle._numpy_safe = lambda tensor: tensor.values
    for name, module in (
        ("torch", torch),
        ("torch.export", torch_export),
        ("torch.export.graph_signature", graph_signature),
        ("m2m", m2m),
        ("m2m.capture", m2m_capture),
        ("m2m.capture.bundle", bundle),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        constants={} if value is None else {"lm.model.lifted_tensor_0": _Tensor(value)},
        state_dict={},
        graph_signature=SimpleNamespace(
            input_specs=[
                SimpleNamespace(
                    kind="constant",
                    arg=SimpleNamespace(name="c_lm_model_lifted_tensor_0"),
                    target="lm.model.lifted_tensor_0",
                )
            ]
        ),
    )


def _manifest(path: Path) -> str:
    weights = path / "weights.safetensors"
    Path(str(weights) + ".manifest.json").write_text(
        json.dumps({"0": {"kind": "input", "name": "c_lm_model_lifted_tensor_0"}})
    )
    return str(weights)


def test_lifted_constant_uses_exact_exported_name_and_bytes(tmp_path, monkeypatch):
    exported = _export(monkeypatch, value=[2**53 + 1, -(2**53 + 7)])
    dest = tmp_path / "extra.npz"
    worker._write_leaf_constants(_Model(), (), _manifest(tmp_path), dest, exported_program=exported)
    with np.load(dest, allow_pickle=False) as extra:
        assert extra.files == ["c_lm_model_lifted_tensor_0"]
        np.testing.assert_array_equal(extra[extra.files[0]], [2**53 + 1, -(2**53 + 7)])


def test_missing_lifted_constant_fails_without_partial_extra(tmp_path, monkeypatch):
    exported = _export(monkeypatch, value=None)
    dest = tmp_path / "extra.npz"
    with pytest.raises(RuntimeError, match="could not be recovered"):
        worker._write_leaf_constants(_Model(), (), _manifest(tmp_path), dest, exported_program=exported)
    assert not dest.exists()
