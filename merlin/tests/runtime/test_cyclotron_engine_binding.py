"""Cyclotron L2 measurements carry the executable/config identity that produced them."""
from __future__ import annotations

import json

import pytest

from merlin.runtime.backends.base import get_backend
from merlin.targetgen import evaluation_cohort as EC


BACKEND = get_backend("muon")
MO = BACKEND.muon_oracles
MU = BACKEND.muon


def _cb() -> dict:
    return {
        "target": "radiance",
        "tensors": {
            "X": {"shape": [1], "dtype": "f32", "role": "input"},
            "Y": {"shape": [1], "dtype": "f32", "role": "output"},
        },
        "commands": [{"opcode": "VECTOR_MAP", "operands": {"arg0": "X", "dst": "Y"}}],
        "canonical_inputs": {"X": {"shape": [1], "values": [1.0]}},
        "_oracle_expected_outputs": {"Y": [1.0]},
        "_oracle_numeric_policy": {"compare": "exact"},
    }


def _stub_oracle(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("MERLIN_MUON_TRUSTED_COMPACT_NUMERIC", raising=False)
    monkeypatch.setattr(MU, "available", lambda simulator: simulator == "cyclotron")
    monkeypatch.setattr(MU, "is_mlir_artifact", lambda _source: True)
    monkeypatch.setattr(
        MU, "compile_mlir_forkfree", lambda *args, **kwargs: tmp_path / "kernel.elf")
    monkeypatch.setattr(
        MU, "run_elf", lambda *args, **kwargs: (
            "OUT Y 1 1 1.0\nDONE\nsimulation finished after 100 cycles\n", 100, {}))


def test_adapter_publishes_binding_only_after_stable_success(monkeypatch, tmp_path) -> None:
    _stub_oracle(monkeypatch, tmp_path)
    binding = {"engine": "cyclotron", "binding_sha256": "a" * 64}
    monkeypatch.setattr(EC, "cyclotron_l2_engine_binding", lambda _target: dict(binding))

    MO.cyclotron_adapter()(_cb(), "builtin.module { llvm.func @k() }", tmp_path, 60)

    assert json.loads((tmp_path / "cyclotron_engine_binding.json").read_text()) == binding


def test_adapter_refuses_engine_change_during_measurement(monkeypatch, tmp_path) -> None:
    _stub_oracle(monkeypatch, tmp_path)
    bindings = iter([
        {"engine": "cyclotron", "binding_sha256": "a" * 64},
        {"engine": "cyclotron", "binding_sha256": "b" * 64},
    ])
    monkeypatch.setattr(EC, "cyclotron_l2_engine_binding", lambda _target: next(bindings))

    with pytest.raises(MU.MuonUnavailable, match="changed during the measured L2 invocation"):
        MO.cyclotron_adapter()(_cb(), "builtin.module { llvm.func @k() }", tmp_path, 60)
    assert not (tmp_path / "cyclotron_engine_binding.json").exists()
