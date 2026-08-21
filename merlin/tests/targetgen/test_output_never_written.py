"""An unwritten output buffer must be NAMED, not reported as ordinary numeric drift.

Measured case that motivated this: 12 capsules held byte-identical mismatch counts for six rounds while
the emitted artifact changed underneath them. Their observed output was uniformly 0.0 -- the kernel never
wrote the buffer -- so the mismatch count was a function of the GOLDEN's zero distribution, not of the
kernel, and could not move no matter what the agent emitted. Reported as 'functional_mismatch', it sent
the agent chasing numerics for six rounds.
"""

from __future__ import annotations

from merlin.targetgen.capsule_golden import compare


def _cmp(expected, observed, **policy):
    pol = {"compare": "tolerance_float", "atol": 0.25, "rtol": 0.02}
    pol.update(policy)
    return compare(expected, observed, pol)


def test_unwritten_output_is_named_not_called_a_mismatch():
    exp = {"Y0": [0.5, 1.5, -2.0, 0.0]}
    obs = {"Y0": [0.0, 0.0, 0.0, 0.0]}          # buffer never touched
    rep = _cmp(exp, obs)
    assert rep["status"] == "fail"
    assert rep["failure_class"] == "output_never_written"
    assert rep["outputs_never_written"] == ["Y0"]
    assert rep["per_output"]["Y0"]["observed_constant"] == 0.0


def test_a_genuinely_wrong_but_written_output_is_not_misclassified():
    exp = {"Y0": [0.5, 1.5, -2.0, 3.0]}
    obs = {"Y0": [0.4, 9.9, -1.0, 0.0]}          # wrong, but varied -> a real attempt
    rep = _cmp(exp, obs)
    assert rep["status"] == "fail"
    assert "failure_class" not in rep
    assert "outputs_never_written" not in rep


def test_a_constant_golden_never_triggers_the_class():
    """If the EXPECTED values are themselves constant, a constant observed output is not evidence of an
    unwritten buffer -- it could be correct, or wrong for ordinary reasons."""
    exp = {"Y0": [0.0, 0.0, 0.0, 0.0]}
    obs = {"Y0": [1.0, 1.0, 1.0, 1.0]}
    rep = _cmp(exp, obs)
    assert rep["status"] == "fail"
    assert "outputs_never_written" not in rep


def test_saturation_is_reported_so_a_pinned_count_is_visible():
    """A 16-element output that is 100% wrong is PINNED: the count cannot move until it is partly right,
    so 'count unchanged' is not evidence the kernel is unchanged."""
    exp = {"Y0": [float(i + 1) for i in range(16)]}
    obs = {"Y0": [999.0 + i for i in range(16)]}
    rep = _cmp(exp, obs)
    po = rep["per_output"]["Y0"]
    assert po["n_elements"] == 16
    assert po["mismatch_count"] == 16
    assert po["saturated"] is True


def test_partial_failure_is_not_saturated():
    exp = {"Y0": [1.0, 2.0, 3.0, 4.0]}
    obs = {"Y0": [1.0, 2.0, 3.0, 99.0]}
    po = _cmp(exp, obs)["per_output"]["Y0"]
    assert po["mismatch_count"] == 1 and po["saturated"] is False


def test_reproduces_the_measured_af6_signature():
    """AF6_add_bf16_pt, rounds 3 and 4: 195/256 mismatches, max_rel_error exactly 1.0, observed 0.0.
    A rel error of exactly 1.0 with a zero observation is the fingerprint of an unwritten buffer."""
    # 256 elements; 61 of the golden values are within atol of zero, so 195 register as mismatches --
    # exactly the measured count, and it is set by the GOLDEN, not by the kernel.
    exp = {"Y0": [0.0] * 61 + [1.0] * 195}
    obs = {"Y0": [0.0] * 256}
    rep = _cmp(exp, obs)
    assert rep["mismatch_count"] == 195
    assert rep["max_rel_error"] == 1.0
    assert rep["failure_class"] == "output_never_written"


def test_the_agent_facing_detail_names_a_writeback_failure_not_a_numeric_one():
    """The message AF6 got for six rounds was 'does not compute the declared operation within tolerance'.
    With the class detected, the agent is told the store never landed and that arithmetic changes will
    not move the count -- the two facts that would have redirected it on round 1."""
    from merlin.targetgen.capsule_runner import _unwritten_output_detail
    rep = _cmp({"Y0": [0.0] * 61 + [1.0] * 195}, {"Y0": [0.0] * 256})
    detail = _unwritten_output_detail(rep, "atlas-arc-arcilator-cosim")
    assert detail is not None
    assert "WRITEBACK failure" in detail
    assert "will NOT move if you only change arithmetic" in detail
    assert "Y0 (all 0.0)" in detail


def test_the_detail_leaks_no_golden_VALUE():
    """It may name the output and the observed constant (both of which the agent already holds from its
    own readback); it must never carry a reference value."""
    from merlin.targetgen.capsule_runner import _unwritten_output_detail
    secret = 0.546875123
    rep = _cmp({"Y0": [secret, secret * 2, 0.0, 0.0]}, {"Y0": [0.0, 0.0, 0.0, 0.0]})
    detail = _unwritten_output_detail(rep, "sim")
    assert detail is not None
    for tok in (str(secret), str(secret * 2), "0.546875"):
        assert tok not in detail, f"golden value {tok} leaked into agent-facing text"


def test_no_detail_when_the_output_was_genuinely_computed():
    from merlin.targetgen.capsule_runner import _unwritten_output_detail
    rep = _cmp({"Y0": [1.0, 2.0, 3.0]}, {"Y0": [1.0, 9.0, 3.0]})
    assert _unwritten_output_detail(rep, "sim") is None


def test_a_shape_error_is_named_not_folded_into_the_value_count():
    """mismatch_count for a length error is |len delta| + 1, which is not a count of wrong values. An
    8-element output emitted as 512 reads as '505 mismatches'; the next round, correct shape and every
    value wrong, reads as '5'. Both were misread as value counts on a real run."""
    rep = _cmp({"Y0": [1.0] * 8}, {"Y0": [1.0] * 512})
    po = rep["per_output"]["Y0"]
    assert po["failure_class"] == "output_shape_mismatch"
    assert po["n_expected"] == 8 and po["n_observed"] == 512
    assert rep["outputs_wrong_shape"] == ["Y0"]
    assert rep["mismatch_count"] == 505          # documents the ambiguous legacy number


def test_correct_shape_all_values_wrong_is_not_confused_with_near_success():
    rep = _cmp({"Y0": [0.00128] * 8}, {"Y0": [24183284.0] * 5 + [0.00128] * 3})
    po = rep["per_output"]["Y0"]
    assert po["mismatch_count"] == 5 and po["n_elements"] == 8
    assert "failure_class" not in po or po.get("failure_class") != "output_shape_mismatch"
    assert po["saturated"] is False               # 5 of 8, not pinned -- but 62% wrong, not near-passing
