"""Native-free transitional adapter checks; no hardware qualification."""

from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native launch"))
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    from merlin.targetgen.plugins import load_declared

    module = load_declared("saturn", "matrix_lowering")
    assert Path(module.__file__).resolve() == root / "matrix_lowering.py"
    return module


def test_geometry_and_build_delegate_exact_inputs(adapter, monkeypatch, tmp_path):
    opu_shim = adapter.opu_shim

    calls = []
    result = object()
    monkeypatch.setattr(opu_shim, "load_contract", lambda unit: (
        calls.append(unit) or SimpleNamespace(geometry=lambda config: (calls.append(config) or (8, 16)))
    ))
    assert adapter.geometry(unit="fixture-unit", config="fixture-config") == (8, 16)
    assert calls == ["fixture-unit", "fixture-config"]
    monkeypatch.setattr(opu_shim, "build_object", lambda *args, **kwargs: (
        calls.append((args, kwargs)) or result
    ))
    signatures = {"fixture": (8, 8, 8)}
    assert adapter.build_object(signatures, tmp_path, unit="u", config="c", cc="cc", cflags=["-O2"],
                                scalar_tile=True, parallel_tiles=True) is result
    assert calls[-1] == ((signatures, tmp_path), dict(unit="u", config="c", cc="cc", cflags=["-O2"],
                                                    scalar_tile=True, parallel_tiles=True))


def test_selector_threshold(adapter):
    choose = adapter.selector(4)
    assert choose(SimpleNamespace(parallel=(4, 4)))
    assert not choose(SimpleNamespace(parallel=(3, 4)))
    with pytest.raises(ValueError):
        adapter.selector(0)


def test_actual_shared_rewrite_and_signature_roundtrip(adapter, tmp_path):
    prepared = tmp_path / "input.mlir"
    prepared.write_text("""builtin.module {
  func.func @fixture(%a: tensor<2x2xi8>, %b: tensor<2x2xi8>) -> tensor<2x2xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<2x2xi32>
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<2x2xi32>) -> tensor<2x2xi32>
    %r = linalg.matmul ins(%a, %b : tensor<2x2xi8>, tensor<2x2xi8>)
         outs(%f : tensor<2x2xi32>) -> tensor<2x2xi32>
    func.return %r : tensor<2x2xi32>
  }
}
""")
    rewrite = adapter.rewrite_prepared_file(prepared, tmp_path, select=adapter.selector(2), tile_edge=2)
    assert rewrite.count == 1
    assert adapter.load_signatures(tmp_path) == rewrite.signatures
    assert list(rewrite.signatures.values()) == [(2, 2, 2)]
    assert "bufferization.access" in prepared.read_text()
