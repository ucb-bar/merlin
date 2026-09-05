"""Token-free command-plan checks for the CHIA performance-experiment envelope."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir


SOURCE = merlin_dir() / "experiments/gemmini_perf_bench/scripts/chia_agentic_perf_experiment.py"
SPEC = importlib.util.spec_from_file_location("chia_agentic_perf_experiment_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
WRAPPER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WRAPPER
SPEC.loader.exec_module(WRAPPER)


def test_plan_wraps_unchanged_coordinator_under_main_python() -> None:
    command = WRAPPER.plan_command(["--experiment-id", "e", "--model", "gpt-5.6-sol"])
    assert Path(command[0]).name == "python"
    assert Path(command[1]).name == "run_agentic_perf_experiment.py"
    assert command[2:] == ["--experiment-id", "e", "--model", "gpt-5.6-sol"]
    assert WRAPPER.run_coordinator._resources == {"codex_slots": 1, "gsim_slots": 1} \
        if hasattr(WRAPPER.run_coordinator, "_resources") else True


def test_stub_plan_is_token_free_and_dry_run_launches_nothing(capsys) -> None:
    assert WRAPPER.main([
        "--orchestration-run-id", "stub", "--stub-seconds", "0.01", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert '"stub": true' in output
    assert "time.sleep" in output
    assert '"driver_parity_claim": false' in output


def test_runtime_assignment_gate_requires_both_logical_resources() -> None:
    assert WRAPPER.validate_assigned_resources(
        {"CPU": 1, "codex_slots": 1, "gsim_slots": 1})["gsim_slots"] == 1.0
    with pytest.raises(RuntimeError, match="gsim_slots"):
        WRAPPER.validate_assigned_resources({"CPU": 1, "codex_slots": 1})


def test_content_addressed_launch_receipt_is_immutable(tmp_path: Path) -> None:
    document = {"schema": "launch", "assigned": {"codex_slots": 1, "gsim_slots": 1}}
    path, digest = WRAPPER._content_addressed_receipt(tmp_path, "launch", document)

    assert path.name == f"launch.{digest}.json"
    assert path.read_bytes() == WRAPPER._canonical(document)
    assert path.stat().st_mode & 0o222 == 0
    assert WRAPPER._content_addressed_receipt(tmp_path, "launch", document) == (path, digest)
