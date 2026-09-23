"""Pure installed-owner resource decisions; no native controller or execution."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import portfolio_resources as PR
from merlin_experiments.phase2.telemetry import _package_source_record


def test_allocation_and_measurement_validation(tmp_path):
    (tmp_path / "capsule.yaml").write_text("interface_mlir: input.mlir\n")
    (tmp_path / "input.mlir").write_bytes(b"module {}")
    member = SimpleNamespace(frozen_source_path=tmp_path, capsule="fixture", capsule_sha256="a" * 64)
    result = PR.portfolio_member_analysis_allocation(90, [member])
    assert result["allocated_seconds"] == 90
    assert result["interface_bytes"] == 9
    for digest, cost in [("A" * 64, 1), ("a" * 64, True), ("a" * 64, float("nan"))]:
        with pytest.raises(ValueError, match="malformed"):
            PR.portfolio_member_analysis_allocation(90, [member], emission_seconds_by_capsule_sha256={digest: cost})


def test_memory_admission_and_tie_order():
    result = PR.portfolio_analysis_concurrency(
        requested_workers=4, members=3, memory_available_bytes=80 * 1024**3, minimum_memory_available_bytes=48 * 1024**3
    )
    assert result["admitted_workers"] == 2
    result = PR.portfolio_concurrent_schedule([2, 2, 1], 2)
    assert result["submission_order"] == [0, 1, 2]
    assert result["worker_estimated_seconds"] == [3, 2]


def test_owner_is_in_existing_recursive_source_inventory():
    path = Path(PR.__file__).resolve()
    record = _package_source_record()
    assert "portfolio_resources.py" in record["members"]
    assert record["members"]["portfolio_resources.py"] == hashlib.sha256(path.read_bytes()).hexdigest()
