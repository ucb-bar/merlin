"""Installed whole-model ownership: exact cache bytes and early input refusal."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import contracts as CONTRACTS
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2.contracts import StageGateError


def test_installed_cache_reuses_only_exact_bytes_and_identity(tmp_path):
    binding = {"root": str(tmp_path / "cache"), "compiler_dependencies_sha256": "d" * 64}
    identity = EA.baseline_emission_cache_identity(
        baseline_sha256="a" * 64,
        capsule_sha256="b" * 64,
        source_sha256="c" * 64,
        target="synthetic",
        compiler_dependencies_sha256="d" * 64,
        compiler_api_schema={"path": "/explicit/schema.json", "sha256": "e" * 64},
        entrypoints=("emit_analysis_bundle",),
    )
    assert EA.load_baseline_emission_cache(binding, identity) is None
    first = EA.store_baseline_emission_cache(
        binding,
        identity,
        lowered_text="module {}\n",
        command_buffer_text='{"commands":[]}\n',
        emission_wall_seconds=0.25,
    )
    assert EA.load_baseline_emission_cache(binding, identity) == first
    assert first["lowered_text"] == "module {}\n"
    assert first["command_buffer_text"] == '{"commands":[]}\n'
    changed_identity = {**identity, "compiler_dependencies_sha256": "f" * 64}
    assert EA.load_baseline_emission_cache(binding, changed_identity) is None
    with pytest.raises(StageGateError, match="different bytes"):
        EA.store_baseline_emission_cache(
            binding,
            identity,
            lowered_text="changed",
            command_buffer_text='{"commands":[]}\n',
            emission_wall_seconds=0.25,
        )
    lowered = tmp_path / "cache" / first["key"] / "lowered.mlir"
    lowered.chmod(0o644)
    lowered.write_text("changed")
    with pytest.raises(StageGateError, match="digest changed"):
        EA.load_baseline_emission_cache(binding, identity)


def test_changed_frozen_model_refuses_before_compiler_execution(tmp_path, monkeypatch):
    from merlin.targetgen import oot_runner

    baseline, candidate, model = (tmp_path / name for name in ("baseline", "candidate", "model"))
    for path in (baseline, candidate, model):
        path.mkdir()
        (path / "input.txt").write_text("frozen")
    sentinel = SimpleNamespace(
        frozen_source_path=str(model), capsule_sha256=CONTRACTS.exact_tree_record(model)["sha256"]
    )
    (model / "input.txt").write_text("changed")

    def forbidden(*args, **kwargs):
        raise AssertionError("changed inputs must refuse before compiler admission")

    monkeypatch.setattr(oot_runner, "load_package", forbidden)
    with pytest.raises(StageGateError, match="sentinel bytes changed"):
        EA.analyze_whole_model_emission(
            baseline,
            candidate,
            sentinel,
            timeout_s=1,
            peak_macs_per_cycle=None,
            achievable_macs_per_cycle=None,
            target="synthetic",
            contract_root=tmp_path / "external-contract",
        )
