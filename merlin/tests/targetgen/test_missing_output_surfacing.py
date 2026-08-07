"""A DROPPED declared output must be surfaced distinctly — never as a phantom "N mismatches, 0 error".

Regression for the AT6_resident_reuse class: a kernel that emits the first output but silently drops a
second (a mis-lowered / mis-addressed store) produces `mismatch_count > 0` while `max_abs_error == 0`
and `first_mismatch is None` — a self-contradictory signal unless the missing output is named. The
grader's `compare()` already records the per-output "missing from observed" reason; the runner must
carry it into `capsule_result.json`'s `numeric` block (so the self-check shows it) and into the failure
detail. Reveals no golden VALUE — only the identity of a declared output the agent already holds.
"""
from __future__ import annotations

from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import capsule_runner as CR

_POLICY = {"compare": "tolerance_float", "atol": 0.25, "rtol": 0.02}


def _rep_with_missing_second_output() -> dict:
    """Two declared outputs Y0/Y1; the kernel wrote only Y0 (Y1 absent from observed)."""
    expected = {"Y0": [[1.0, 2.0]], "Y1": [[3.0, 4.0]]}
    observed = {"Y0": [[1.0, 2.0]]}                        # Y1 never written
    rep = CG.compare(expected, observed, _POLICY, golden_source="specir_refmodel_fp8_bf16")
    return rep, expected, observed


def test_compare_records_missing_output_reason():
    rep, _, _ = _rep_with_missing_second_output()
    assert rep["status"] == "fail"
    assert rep["per_output"]["Y0"]["status"] == "pass"
    assert rep["per_output"]["Y1"]["status"] == "fail"
    assert "missing" in rep["per_output"]["Y1"]["reason"]
    # the self-contradictory-looking pair that WOULD baffle the agent without the missing-output name
    assert rep["mismatch_count"] > 0
    assert rep["max_abs_error"] == 0
    assert rep["first_mismatch"] is None


def test_absent_outputs_extracts_the_dropped_name():
    rep, _, _ = _rep_with_missing_second_output()
    assert CR._absent_outputs(rep) == ["Y1"]


def test_absent_output_detail_names_it_and_counts_written():
    rep, expected, observed = _rep_with_missing_second_output()
    detail = CR._absent_output_detail(rep, "atlas-functional", expected, observed)
    assert detail is not None
    assert "Y1" in detail
    assert "1 of 2" in detail                              # produced 1 of 2 declared outputs
    assert "never wrote" in detail
    # honesty: no golden value leaks into the detail
    assert "3.0" not in detail and "4.0" not in detail


def test_length_mismatch_also_flagged_structural():
    expected = {"Y0": [[1.0, 2.0, 3.0]]}
    observed = {"Y0": [[1.0, 2.0]]}                        # wrong length, not a value error
    rep = CG.compare(expected, observed, _POLICY, golden_source="specir_refmodel_fp8_bf16")
    assert CR._absent_outputs(rep) == ["Y0"]


def test_value_mismatch_is_NOT_flagged_as_absent():
    # a genuine value error (both outputs present, one wrong) must stay a value mismatch, not "absent"
    expected = {"Y0": [[1.0, 2.0]]}
    observed = {"Y0": [[9.0, 9.0]]}
    rep = CG.compare(expected, observed, _POLICY, golden_source="specir_refmodel_fp8_bf16")
    assert CR._absent_outputs(rep) == []
    assert CR._absent_output_detail(rep, "sim", expected, observed) is None
    assert rep["max_abs_error"] > 0                        # a real magnitude, unlike the dropped-store case
