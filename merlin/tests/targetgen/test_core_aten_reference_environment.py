"""Source references use the declared capture interpreter, independently of compiler Python."""

import json
import os
from types import SimpleNamespace

import pytest

from merlin.targetgen import core_aten_numeric


@pytest.mark.parametrize("arguments", [(), ("--native",)])
def test_source_reference_uses_declared_capture_python(tmp_path, monkeypatch, arguments):
    from merlin.common.paths import python_import_roots
    from merlin.integrations import model2mlir

    python = tmp_path / "capture-python"
    monkeypatch.setattr(model2mlir, "capture_python", lambda: python)
    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", "/unavailable/compiler-python")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(core_aten_numeric.subprocess, "run", run)
    assert core_aten_numeric._source_reference(tmp_path, *arguments).returncode == 0
    command, kwargs = calls.pop()
    assert command == [str(python), "-m", "merlin.targetgen.core_aten_w8a8_reference", str(tmp_path), *arguments]
    assert kwargs["env"]["PYTHONPATH"] == os.pathsep.join(str(p) for p in python_import_roots())
    assert not calls


@pytest.mark.parametrize("native", [False, True])
def test_source_reference_executes_without_a_compiler_environment(tmp_path, monkeypatch, native):
    import numpy as np

    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", "/unavailable/compiler-python")
    prefix = "native-int8" if native else "w8a8"
    source = "model.native_reference.mlir" if native else "model.before_w8a8.mlir"
    (tmp_path / source).write_text('''builtin.module {
  func.func @forward(%x: tensor<2xi32>) -> tensor<2xi32> {
    func.return %x : tensor<2xi32>
  }
}''')
    values = np.array([-(2**31), 2**31 - 1], dtype=np.int32)
    np.savez(tmp_path / "inputs.npz", in0=values)
    (tmp_path / f"{prefix}-request.json").write_text(json.dumps({
        "selected_ordinals": [], "quant_min": -128, "quant_max": 127,
        "eps": 2**-12, "dispatch": [],
    }))
    process = core_aten_numeric._source_reference(tmp_path, *(("--native",) if native else ()))
    assert process.returncode == 0, process.stdout + process.stderr
    reference = json.loads((tmp_path / f"{prefix}-reference.json").read_text())
    assert reference["outputs"] == [[-(2**31), 2**31 - 1]]
    assert reference["output_bytes"] == [values.tobytes().hex()]
