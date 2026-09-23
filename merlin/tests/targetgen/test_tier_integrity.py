"""A cert tier that did not certify may not be reported as a pass, and never without a reason.

Every test here is a MUTATION of a state that a real graded run actually produced. The states are not
invented: `merlincirct_radiance_pr1_cohort29_20260907` carried 24 rows with `status: pass`,
`L3: unavailable` and `failure_detail: null`, four more with `status: pass` next to `L3: fail`, and a
gemmini run in the same week carried twelve `status: pass` rows whose cert tier had been killed at its
1200 s ceiling. Each of those is one case below.
"""

from __future__ import annotations

from merlin.targetgen import tier_integrity as TI
from merlin.targetgen.capsule_common import NOT_MEASURED_STATUSES


def _result(**tiers) -> dict:
    """A passing capsule result whose tier records are given as keyword arguments."""
    return {"capsule": "C", "status": "pass", "tiers": dict(tiers)}


def _rtl(status: str, **fields) -> dict:
    return {"status": status, "derived_from_rtl": True, "mandatory": False, **fields}


def _screen(status: str = "pass", **fields) -> dict:
    return {"status": status, "derived_from_rtl": False, "mandatory": True, **fields}


# --- the headline rule ---------------------------------------------------------------------------


def test_never_run_cert_tier_is_not_reported_as_a_pass():
    """The 24-row case. A cheaper tier passing is a SCREEN; it may never certify."""
    r = _result(L2=_screen(), L3=_rtl("unavailable", reason="the console carries no completion witness"))
    verdict = TI.qualify(r)
    assert verdict is not None, "a pass resting on a cert tier that produced no verdict must be rewritten"
    assert verdict["status"] == TI.CERT_NOT_MEASURED != "pass"
    assert verdict["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
    assert verdict["failure"]["tier"] == "L3"
    assert "no completion witness" in verdict["failure"]["tier_reason"]
    assert "NOT a verdict on this capsule" in verdict["failure"]["detail"]


def test_cert_not_measured_is_outside_the_scored_denominator():
    """It must be neither numerator nor denominator -- the ``screened_only`` precedent."""
    assert TI.CERT_NOT_MEASURED in NOT_MEASURED_STATUSES


def test_executed_and_refuted_cert_tier_is_not_reported_as_a_pass():
    """The 4-row case: L3 ran, refused the program, and the row still read ``pass``."""
    r = _result(L2=_screen(), L3=_rtl("fail", reason="injected operand of dtype 'f8E4M3FN' has no decoder"))
    verdict = TI.qualify(r)
    assert verdict is not None and verdict["status"] == "fail"
    assert verdict["failure"]["category"] == "EXECUTED_CERT_TIER_REFUTED"
    # The tier's own sentence must travel, so a reader can tell a harness gap from a wrong lowering.
    assert "no decoder" in verdict["failure"]["tier_reason"]


def test_cert_tier_abandoned_on_the_clock_is_not_a_refutation():
    """The 12-row gemmini case. A cert the engine was KILLED on says nothing about the program."""
    r = _result(
        L2=_screen(),
        L3=_rtl(
            "fail", reason=("elaborated_rtl crash: Command '['emulator', 'kernel.elf']' timed out after 1200 seconds")
        ),
    )
    verdict = TI.qualify(r)
    assert verdict is not None, "still not a pass -- nothing certified it"
    assert verdict["status"] == TI.CERT_NOT_MEASURED, "a clock is not evidence of incorrectness"
    assert verdict["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"


def test_a_clean_cert_tier_leaves_the_pass_alone():
    """The rule must not fire on the state it exists to protect."""
    assert TI.qualify(_result(L2=_screen(), L3=_rtl("pass", cycles=222584))) is None


def test_a_result_with_no_cert_tier_at_all_is_untouched():
    """Screen-only ladders are a different question; this rule has nothing to say about them."""
    assert TI.qualify(_result(L0=_screen("skipped"), L2=_screen())) is None


# --- the silence ---------------------------------------------------------------------------------


def test_every_non_passing_tier_gets_a_reason_and_never_none():
    """``failure_detail: null`` on 24 of 29 rows is the defect. No branch may return ``None``."""
    tiers = {
        "L0": _screen("skipped", reason="integer reference not applicable"),
        "L1": _screen("skipped"),  # no reason field at all
        "L2": _screen("pass"),
        "L3": _rtl("unavailable"),  # ran, said nothing
        "L4": "fail",
    }  # bare-string carrier
    why = TI.not_certified(tiers, ladder=("L0", "L1", "L2", "L3", "L4"))
    assert "L2" not in why, "a passing tier is not in the not-certified set"
    assert set(why) == {"L0", "L1", "L3", "L4"}
    assert list(why) == ["L0", "L1", "L3", "L4"], "the declared ladder orders the report"
    for tier, sentence in why.items():
        assert isinstance(sentence, str) and sentence.strip(), f"{tier} recorded no reason"
    assert why["L0"] == "integer reference not applicable", "the tier's OWN text leads"
    assert why["L1"].startswith(TI.UNKNOWN_PREFIX)
    assert why["L3"].startswith(TI.UNKNOWN_PREFIX) and "L3" in why["L3"]
    assert why["L4"].startswith(TI.UNKNOWN_PREFIX), "a bare status string carries no reason"


def test_an_absent_tier_record_still_names_itself():
    assert TI.reason_for("L3", None).startswith(TI.UNKNOWN_PREFIX)
    assert "L3" in TI.reason_for("L3", None)


def test_a_deliberately_unbought_tier_says_so_rather_than_unknown():
    why = TI.reason_for("L3", _rtl("skipped", budget_deferred=True))
    assert not why.startswith(TI.UNKNOWN_PREFIX)
    assert "not purchased" in why


# --- what the rule must NOT break ----------------------------------------------------------------


def test_not_applicable_cert_tier_stays_exempt():
    """The advisory legality smoke is documented never to block; it is flagged ``not_applicable``."""
    r = _result(L2=_screen(), L3=_rtl("fail", not_applicable=True, reason="advisory smoke"))
    assert TI.qualify(r) is None


def test_budget_deferred_cert_tier_stays_with_the_screened_only_verdict():
    """``screened_only`` already covers a tier deliberately not bought; do not double-count it."""
    r = _result(L2=_screen(), L3=_rtl("skipped", budget_deferred=True))
    assert TI.qualify(r) is None


def test_an_existing_failure_is_never_downgraded():
    r = {
        "capsule": "C",
        "status": "fail",
        "failure": {"plane": "cyclotron", "category": "FUNCTIONAL_MISMATCH", "detail": "wrong numbers"},
        "tiers": {"L2": _screen("fail"), "L3": _rtl("unavailable")},
    }
    assert TI.qualify(r) is None
    TI.apply(r)
    assert r["status"] == "fail" and r["failure"]["plane"] == "cyclotron"


def test_apply_keeps_a_more_specific_failure_that_is_already_present():
    r = _result(L2=_screen(), L3=_rtl("fail", reason="refused"))
    r["failure"] = {"plane": "lanes", "category": "ALREADY_KNOWN", "detail": "keep me"}
    TI.apply(r)
    assert r["status"] == "fail" and r["failure"]["category"] == "ALREADY_KNOWN"


# --- derived, not named --------------------------------------------------------------------------


def test_the_cert_tier_is_derived_from_the_record_not_from_its_name():
    """A tier named L3 that is NOT elaborated RTL must not be treated as the cert tier, and a tier
    named anything at all that IS must be. Tier names are per-target spellings, not facts."""
    named_but_functional = _result(L2=_screen(), L3=_screen("unavailable"))
    assert TI.qualify(named_but_functional) is None
    unnamed_but_rtl = _result(L2=_screen(), GOLD=_rtl("unavailable", reason="engine absent"))
    verdict = TI.qualify(unnamed_but_rtl)
    assert verdict is not None and verdict["failure"]["tier"] == "GOLD"


def test_no_target_name_or_engine_name_is_baked_into_this_module():
    from pathlib import Path

    from merlin.common.paths import merlin_dir

    source = (Path(merlin_dir()) / "python/merlin/targetgen/tier_integrity.py").read_text()
    body = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("#", '"', "*")))
    for banned in ("gsim", "verilator", "cyclotron", "spike"):
        assert banned not in body.lower(), f"{banned!r} is a per-target fact, not a rule"


def test_the_kind_of_non_certification_is_not_recoverable_from_the_tier_status():
    """Both an abandonment and a refutation are recorded ``fail``; only ``kind`` separates them."""
    abandoned = TI.cert_tier_outcome({"L3": _rtl("fail", reason="timed out after 900 seconds")})
    refused = TI.cert_tier_outcome({"L3": _rtl("fail", reason="does not compute the declared operation")})
    assert abandoned[2] == refused[2] == "fail"
    assert abandoned[0] == TI.NOT_MEASURED and refused[0] == TI.REFUTED
