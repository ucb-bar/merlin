"""The shared evaluation schema must not overstate what an evaluation established.

Every rule tested here fails in the FLATTERING direction if it regresses -- an execution cert reads
as a correctness pass, an unwinnable task reads as a failure by the method, a substituted reference
kernel reads as the submission's own result. Those are the errors a study cannot afford, so they are
pinned rather than left to reviewer attention.

The redaction test is the load-bearing one: the grader's own mismatch report embeds the expected
value, so a blacklist that missed a key would hand an agent the answer key.
"""
import pytest

from merlin.benchharness import evaluation as EV

CERT = frozenset({"L2"})


def _result(**over):
    base = {
        "status": "pass",
        "tiers": {"L2": {"status": "pass", "cycles": 1000, "gflops": 1.5, "pct_fp_peak": 0.1,
                         "timing": {"build_s": 0.5, "adapter_wall_s": 3.0}}},
        "numeric": {"status": "pass", "mismatch_count": 0, "max_abs_diff": 0,
                    "max_rel_error": 0.0, "policy": "float"},
        "failure": None,
        "toolchain_shas": {},
    }
    base.update(over)
    return base


def _ev(res, **kw):
    return EV.from_capsule_result(res, task_id="T", config_id="C0", target="t",
                                  certifying_tiers=CERT, **kw)


def test_a_certifying_tier_pass_is_correct():
    ev = _ev(_result())
    assert ev.verdict == "match"
    assert ev.certifying_tier == "L2"
    assert ev.counts_as_correct is True
    assert ev.perf_valid is True


def test_execution_only_tier_is_not_a_correctness_pass():
    """The core rule: a tier that ran the kernel but compared nothing cannot certify numerics."""
    res = _result(
        tiers={"L3": {"status": "pass", "completion_only": True, "cycles": 42}},
        numeric={"status": "skipped"},
    )
    ev = _ev(res)
    assert ev.tier_reached == "L3"
    assert ev.certifying_tier is None, "a non-certifying tier must not certify"
    assert ev.verdict == "not_certified"
    assert ev.counts_as_correct is False
    assert {c.code for c in ev.caveats} >= {"tier_certifies_execution_only"}


def test_an_unwinnable_task_is_excluded_not_failed():
    """No correct submission could win it, so it leaves both sides of the ratio."""
    ev = _ev(_result(status="fail", numeric={"status": "fail", "mismatch_count": 247}),
             unwinnable_reason="golden requires target-private block scales")
    assert ev.status == "unsupported"
    assert ev.verdict == "structurally_unwinnable"
    assert ev.is_scoreable is False
    assert ev.counts_as_correct is False


def test_an_incorrect_kernel_gets_no_performance_credit():
    """A wrong kernel can be arbitrarily fast by not doing the work."""
    ev = _ev(_result(status="fail", numeric={"status": "fail", "mismatch_count": 5,
                                             "max_abs_diff": 3.5, "max_rel_error": 0.9}))
    assert ev.verdict == "mismatch"
    assert ev.cycles == 1000, "cycles are still recorded"
    assert ev.perf_valid is False, "but they must not count as performance"


def test_a_substituted_reference_kernel_is_not_attributed_to_the_submission():
    res = _result()
    res["tiers"]["L2"]["toolchain"] = "mx-reference-kernel(not-the-submission;fixture)"
    ev = _ev(res)
    assert "result_not_attributable_to_submission" in {c.code for c in ev.caveats}


def test_tier_rank_is_numeric_not_lexicographic():
    """L10 must outrank L2; string ordering would silently report the lower tier."""
    res = _result(tiers={"L2": {"status": "pass", "cycles": 1},
                         "L10": {"status": "pass", "cycles": 2}})
    assert _ev(res).tier_reached == "L10"


def test_redaction_never_leaks_a_golden_value():
    """The load-bearing test: the grader's report embeds `expected`, which is the answer key."""
    res = _result(
        status="fail",
        numeric={"status": "fail", "mismatch_count": 3, "max_abs_diff": 2.5,
                 "max_rel_error": 0.4, "policy": "float",
                 "first_mismatch": {"output": "Y0", "index": 7,
                                    "expected": 1234.5, "observed": 0.0},
                 "per_output": {"Y0": {"expected_values": [1234.5, 6.0]}}},
    )
    red = _ev(res).redact()
    blob = repr(red)
    assert "expected" not in blob and "1234.5" not in blob, f"golden leaked into {red}"
    assert "first_mismatch" not in red and "per_output" not in red
    # What a developer legitimately sees about their own kernel survives.
    assert red["mismatch_count"] == 3
    assert red["max_abs_error"] == 2.5
    assert red["correct"] is False


def test_redaction_is_a_whitelist_so_new_grader_keys_cannot_leak():
    """A blacklist would pass through any key a future grader adds; this must not."""
    res = _result()
    res["numeric"]["a_future_key_holding_the_answer"] = [9.9, 8.8]
    res["tiers"]["L2"]["golden_path"] = "/capsules/x/golden.yaml"
    red = _ev(res).redact()
    assert "a_future_key_holding_the_answer" not in repr(red)
    assert "golden.yaml" not in repr(red)


def test_unavailable_is_not_reported_as_a_failure():
    """A crashed oracle and a wrong answer are different findings."""
    res = _result(status="error", tiers={"L2": {"status": "error", "reason": "sim crashed"}},
                  numeric={"status": "skipped"},
                  failure={"plane": "oracle", "category": "tool_crash", "detail": "boom"})
    ev = _ev(res)
    assert ev.status == "error"
    assert ev.verdict == "unknown", "no run happened, so nothing is certified either way"
    assert ev.counts_as_correct is False
    assert ev.failure_category == "tool_crash"


@pytest.mark.parametrize("field", ["task_id", "config_id", "target"])
def test_identity_is_carried_so_results_can_be_joined(field):
    assert getattr(_ev(_result()), field)


def test_an_rtl_tier_without_cycles_leaves_the_latency_a_model_estimate():
    """The trap this pins: a cert run whose RTL engine reports no timing.

    GSIM on this target passes L3 with cycles=None -- it certifies that the kernel RAN, not how long
    it took. The latency then still comes from the functional model, and a record saying "cert"
    beside a model number is how a perf claim becomes unfalsifiable.
    """
    res = _result(
        tiers={
            "L2": {"status": "pass", "cycles": 573042, "cycle_accurate": False},
            "L3": {"status": "pass", "cycles": None, "cycle_accurate": True,
                   "derived_from_rtl": True},
        },
    )
    ev = _ev(res)
    assert ev.tier_reached == "L3", "the RTL tier did pass"
    assert ev.cycles == 573042
    assert ev.cycles_tier == "L2", "the number came from the model, and must say so"
    assert ev.cycles_cycle_accurate is False
    assert "latency_is_a_model_estimate" in {c.code for c in ev.caveats}
    assert "latency_is_a_model_estimate" in repr(ev.redact()), "the agent-facing view must carry it"


def test_a_cycle_accurate_tier_that_reports_cycles_needs_no_caveat():
    """The control: when RTL DOES time the kernel, nothing is hedged."""
    res = _result(tiers={"L2": {"status": "pass", "cycles": 99, "cycle_accurate": False},
                         "L3": {"status": "pass", "cycles": 12345, "cycle_accurate": True}})
    ev = EV.from_capsule_result(res, task_id="T", config_id="C0", target="t",
                                certifying_tiers=frozenset({"L2", "L3"}))
    assert ev.cycles == 12345 and ev.cycles_tier == "L3"
    assert ev.cycles_cycle_accurate is True
    assert "latency_is_a_model_estimate" not in {c.code for c in ev.caveats}
