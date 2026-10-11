"""The operator-side reference column: candidate/reference cycles per capsule, refused across engines."""

from __future__ import annotations

import json

import pytest
from merlin_experiments.phase2 import reference_comparison as RC


def _reference(tmp_path, *, engine="e" * 64):
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "schema": RC.SCHEMA,
                "label": "reference compiler",
                "engine": {"binary_sha256": engine},
                "capsules": {"PM00": {"gsim_cycles": 100}, "PM01": {"gsim_cycles": 1000}},
            }
        )
    )
    return path


def _paired():
    row = {"family": "PM", "replicate": "r000", "simulator": "gsim", "comparable": True}
    return [
        dict(row, capsule="PM00", baseline_cycles=400, candidate_cycles=200),
        dict(row, capsule="PM01", baseline_cycles=3000, candidate_cycles=1500),
        dict(row, capsule="PM02", baseline_cycles=50, candidate_cycles=40),
    ]


def test_ratios_and_unmeasured_capsules_are_stated(tmp_path):
    (tmp_path / "campaign").mkdir()
    (tmp_path / "campaign" / "paired_cycles.json").write_text(json.dumps(_paired()))
    out = RC.write_comparison(tmp_path / "campaign", _reference(tmp_path), campaign_engine_sha256="e" * 64)
    document = json.loads(out.read_text())
    rows = {r["capsule"]: r for r in document["rows"]}
    assert rows["PM00"]["candidate_over_reference"] == 2.0 and rows["PM00"]["baseline_over_reference"] == 4.0
    assert rows["PM02"]["state"] == "reference_unmeasured" and rows["PM02"]["candidate_over_reference"] is None
    assert document["visibility"] == "operator_only"
    assert document["summary"]["compared"] == 2
    assert document["summary"]["geomean_candidate_over_reference"] == pytest.approx((2.0 * 1.5) ** 0.5, rel=1e-3)
    assert document["summary"]["total_candidate_over_total_reference"] == pytest.approx(1700 / 1100, rel=1e-3)


def test_a_reference_from_another_engine_is_never_a_divisor(tmp_path):
    rows = RC.reference_rows(_paired(), RC.load_reference(_reference(tmp_path)), campaign_engine_sha256="f" * 64)
    assert {r["state"] for r in rows} == {"engine_mismatch"}
    assert all(r["candidate_over_reference"] is None for r in rows)


def test_a_non_reference_document_is_refused(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"schema": "other", "capsules": {}}))
    with pytest.raises(RC.ReferenceError):
        RC.load_reference(path)
