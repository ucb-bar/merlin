"""Installed measured calibration uses exact recorded work, without native imports."""

import builtins
import json
import subprocess

import pytest


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"perf_model", "perf_agent_stage", "perf_capsule_verdict"}:
            pytest.fail("installed calibration imported native owner")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native launch"))


def test_harvest_exact_work_and_refuse_missing_buffer(tmp_path):
    from merlin_experiments.phase2 import calibration as C

    for name in ("measured", "missing"):
        root = tmp_path / name
        root.mkdir()
        (root / "capsule_result.json").write_text(json.dumps({"tiers": {"L3": {"cycles": 8}}}))
    generated = tmp_path / "measured/generated"
    generated.mkdir()
    (generated / "command_buffer.json").write_text(
        json.dumps(
            {
                "tensors": {"a": {"shape": [2, 3], "dtype": "i8"}, "b": {"shape": [3, 4], "dtype": "i8"}},
                "commands": [{"opcode": "MATMUL", "operands": {"lhs": "a", "rhs": "b"}}],
            }
        )
    )
    points, skipped = C.harvest_measured_points(tmp_path)
    assert len(points) == 1
    assert (points[0].macs, points[0].cycles, points[0].reduction_depths) == (24, 8, (3,))
    assert skipped == ["missing: measured 8 cycles but emitted no command buffer, so its work cannot be priced"]
    assert not C.achievable_ceiling(points, provenance="synthetic").known
    assert C.achievable_ceiling([*points, C.MeasuredPoint("second", 48, 16, "synthetic")], provenance="synthetic").known
    assert not C.achievable_ceiling([], provenance="synthetic").known


def test_capsule_verdict_import_is_installed_and_unknown_stays_refused():
    from merlin_experiments.phase2 import capsule_verdict as V

    assert V.ceiling_dispersion([]) is None
    assert (
        V.capsule_verdict(
            capsule="fixture",
            declared_macs=None,
            achievable_rate=None,
            baseline_cycles=None,
            candidate_cycles=None,
            dispersion=None,
        )["verdict"]
        == V.REFUSED
    )
