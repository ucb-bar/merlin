"""Calibration of the analytical cost model vs the npu_model cycle-level simulator.

Skipped unless the npu_model simulator (and torch) are importable. Report-only: asserts the
analytical and simulated single-invocation cycles agree within a loose factor — a validity
check on the analytical model's scale, not an exact match.
"""
import pytest

from merlin.dse import calibrate_npu as C

pytestmark = pytest.mark.skipif(not C.available(),
                                reason="npu_model simulator (torch) not importable")


def test_calibration_runs_and_agrees(tmp_path):
    report = C.calibrate(out_dir=tmp_path)
    assert (tmp_path / "calibration.yaml").is_file()
    assert report["rows"], "no calibration points ran"
    for row in report["rows"]:
        assert row["simulated_cycles"] > 0
        # Loose agreement band: analytical within 0.2x..5x of simulated.
        assert 0.2 <= row["ratio"] <= 5.0, row
