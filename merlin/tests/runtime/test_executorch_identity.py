"""Board-free fail-closed tests for the ExecuTorch exporter/runtime identity gate."""

from pathlib import Path

import pytest

from merlin.baselines import executorch as et
from merlin.baselines import executorch_identity as identity


EXPORTER = "4af91c3d6a4c16c5b0e5745620d7b3d208ba6928"
SOURCE = "7fc34bf6f53d2098e3e16c1fa71c23222f607330"


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    python, source = tmp_path / "python", tmp_path / "executorch"
    python.write_text("")
    source.mkdir()
    return python, source


def test_exact_exporter_source_identity_passes(monkeypatch, tmp_path):
    python, source = _paths(tmp_path)
    monkeypatch.setattr(identity, "_exporter_probe",
                        lambda _python: {"version": "test", "git_sha": SOURCE})
    monkeypatch.setattr(identity, "_source_commit", lambda _source: SOURCE)

    found = identity.require_matching_executorch(python, source)

    assert found.matches
    assert found.as_dict()["source_git_sha"] == SOURCE


def test_exporter_source_mismatch_blocks_every_precision(monkeypatch, tmp_path):
    python, source = _paths(tmp_path)
    monkeypatch.setattr(identity, "_exporter_probe",
                        lambda _python: {"version": "1.4.0a0", "git_sha": EXPORTER})
    monkeypatch.setattr(identity, "_source_commit", lambda _source: SOURCE)

    with pytest.raises(identity.ExecuTorchIdentityError, match="blocks FP32 and quantized"):
        identity.require_matching_executorch(python, source)


@pytest.mark.parametrize("reported", ["", "4af91c3", "not-a-commit", "0" * 39])
def test_missing_or_abbreviated_exporter_identity_is_a_blocker(
        monkeypatch, tmp_path, reported):
    python, source = _paths(tmp_path)
    monkeypatch.setattr(identity, "_exporter_probe",
                        lambda _python: {"version": "unknown", "git_sha": reported})

    with pytest.raises(identity.ExecuTorchIdentityError, match="no reliable full git identity"):
        identity.inspect_executorch_identity(python, source)


def test_et_venv_available_uses_identity_gate(monkeypatch):
    monkeypatch.setattr(et, "et_identity", lambda: (_ for _ in ()).throw(
        identity.ExecuTorchIdentityError("mismatch")))
    assert not et.et_venv_available()


@pytest.mark.parametrize("quantize", [False, True], ids=["fp32", "w8a8"])
def test_export_and_runtime_build_entrypoints_reject_mismatch(
        monkeypatch, tmp_path, quantize):
    def mismatch():
        raise identity.ExecuTorchIdentityError("exporter/runtime source identity mismatch")

    monkeypatch.setattr(et, "et_identity", mismatch)
    with pytest.raises(et.ExecuTorchError, match="identity mismatch"):
        et.export_pte("unused", object(), tmp_path, quantize=quantize)
    with pytest.raises(et.ExecuTorchError, match="identity mismatch"):
        et.cross_compile_runner(tmp_path)
