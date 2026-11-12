from __future__ import annotations

from verification_witnesses.lean_verification import (
    AFTER,
    BEFORE,
    LeanToolchain,
    Path,
    accepted_pipeline_receipt,
    accepted_receipt,
    coverage_policy_record,
    digest,
    os,
    pytest,
    real_toolchain,
    replace,
    verify_pipeline_change,
    verify_scalar_transform,
)


def test_missing_tool_and_forged_report_never_prove():
    result = verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=None)
    assert result.status == "unavailable"
    forged = replace(result, status="verified")
    assert not accepted_receipt(forged, BEFORE, AFTER)
    assert not hasattr(__import__("merlin.verification.lean_contract", fromlist=["issue"]), "issue")


def test_real_lean_checks_actual_scalar_pair_and_rejects_wrong_pair(real_toolchain):
    valid = verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=real_toolchain)
    assert valid.verified, valid.reason
    assert accepted_receipt(valid, BEFORE, AFTER)
    cached = verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=real_toolchain)
    assert cached.verified and cached.cache_hit
    assert accepted_receipt(cached, BEFORE, AFTER)
    assert not accepted_receipt(replace(valid), BEFORE, AFTER)
    wrong = AFTER.replace("func.return %x", "%one = arith.constant 1 : i8 %r = arith.addi %x, %one : i8 func.return %r")
    invalid = verify_scalar_transform(BEFORE, wrong, entry="kernel", toolchain=real_toolchain)
    assert not invalid.verified
    assert not accepted_receipt(valid, BEFORE, wrong)
    assert not set(valid.axioms) - {"propext", "Classical.choice", "Quot.sound"}


def test_real_unapproved_axiom_does_not_create_completion(real_toolchain):
    from merlin.verification.lean_checker import _check_generated

    unproved = verify_scalar_transform(BEFORE, AFTER, entry="kernel", toolchain=None)
    query = "import Merlin.Foundation\naxiom injected : False\ntheorem merlin_transform : False := injected\n#print axioms merlin_transform\n"
    result = _check_generated(query, unproved, real_toolchain)
    assert not result.verified and "unapproved axiom" in result.reason
    assert not accepted_receipt(result, BEFORE, AFTER)
