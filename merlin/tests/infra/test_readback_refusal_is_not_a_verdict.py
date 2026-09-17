"""A refused readback is NOT MEASURED, and must never be recorded as the submission's failure.

WHY THIS EXISTS. `readback_integrity` refuses a buffer whose every Nth word came back exactly zero
while the rest carries data -- a transport defect, not an arithmetic result -- BEFORE any value is
compared. That refusal travels out through the target's oracle as an ordinary exception, and with
nothing to distinguish it, `capsule_runner`'s adapter handler recorded it as `status: fail`,
`failure_category: tool_crash`, `failure_plane: verilator`.

Every part of that was wrong. Verilator had not crashed; nothing had crashed; and the capsules
carrying it had `numeric_status: pass` with `mismatch_count: 0` and `L2: pass`.

MEASURED 2026-09-09 on radiance: three capsules -- the same `squareish_gemm` shape at bf16, f32 and
i8 -- were scored as agent failures on that basis, 3 of that run's 11 non-passes.

`capsule_runner` already carries four reclassifications in this handler, each rescuing a verdict that
WAS the agent's from an infra-shaped label. This is the fifth and the only one that runs the other
way: it rescues the agent from a verdict that was never measured at all.
"""

from __future__ import annotations

import pytest

from merlin.common.readback_integrity import ReadbackIntegrityError, require_intact
from merlin.targetgen.capsule_common import NOT_MEASURED_STATUSES
from merlin.targetgen.capsule_runner import _is_readback_refusal, _readback_refusal_detail
from merlin.targetgen.oot_runner import INFRASTRUCTURE_PLANE, CertFailure, InfraCategory, InfraFailure

DETAIL = (
    "gsim_evaluator_owned_gmem_dump: readback_residue_class_zeroed: every 4-byte word at "
    "index % 2 == 1 is exactly zero (512 of 1024)"
)


def _rewrapped_twice() -> BaseException:
    """The real shape: the oracle re-wraps, then the runner re-wraps again."""
    try:
        try:
            try:
                raise ReadbackIntegrityError(DETAIL)
            except ReadbackIntegrityError as exc:
                raise RuntimeError(f"muon: {exc}") from exc
        except RuntimeError as e:
            raise RuntimeError(f"verilator invocation failed: {e}") from e
    except RuntimeError as outer:
        return outer


# --- detection is by TYPE, through the re-wraps -------------------------------------------------


def test_a_refusal_is_detected_through_two_rewraps():
    """THE REGRESSION. The refusal is unrecognisable by the time it reaches the handler unless the
    cause chain is walked."""
    assert _is_readback_refusal(_rewrapped_twice()) is True


def test_the_innermost_detail_is_recovered_not_the_outer_wrapper_text():
    """The outer message says 'verilator invocation failed', which is the lie being fixed."""
    assert _readback_refusal_detail(_rewrapped_twice()) == DETAIL


def test_a_real_crash_is_not_misdetected_as_a_refusal():
    """Widening this would hide genuine tool crashes -- the opposite defect."""
    assert _is_readback_refusal(RuntimeError("verilator: segmentation fault")) is False
    assert _is_readback_refusal(RuntimeError("terminate called: std::out_of_range")) is False


@pytest.mark.timeout(15)
def test_detection_survives_a_cycle_in_the_exception_chain():
    """A cyclic cause chain must TERMINATE. Without the `seen` guard the walk spins forever, and an
    unbounded loop inside the grader's exception handler is worse than the mislabel it replaces --
    so this asserts with a timeout, and fails rather than hanging the suite."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert _is_readback_refusal(a) is False


# --- the recorded outcome is NOT a verdict ------------------------------------------------------


def test_the_category_exists_and_serializes_to_an_honest_token():
    assert InfraCategory.READBACK_TRANSPORT_REFUSED.value == "readback_transport_refused"
    assert str(InfraCategory.READBACK_TRANSPORT_REFUSED) == "readback_transport_refused"


def test_an_infra_failure_lands_outside_the_pass_fail_denominator():
    """`infrastructure_fault` is what capsule_runner maps InfraFailure to, and it must stay a
    NOT_MEASURED status -- otherwise the refusal is still counted against the agent."""
    assert "infrastructure_fault" in NOT_MEASURED_STATUSES


def test_infra_failure_is_still_a_cert_failure_so_existing_recorders_work():
    """The narrower type must be caught FIRST; being a subclass is what keeps the recorders honest."""
    exc = InfraFailure(INFRASTRUCTURE_PLANE, InfraCategory.READBACK_TRANSPORT_REFUSED, DETAIL)
    assert isinstance(exc, CertFailure)
    assert exc.plane == INFRASTRUCTURE_PLANE
    assert exc.plane not in ("verilator", "oracle", "numeric")


# --- the refusal itself, since readback_integrity ships with this change ------------------------


def _buf(words: list[int]) -> bytes:
    return b"".join(int(w).to_bytes(4, "little") for w in words)


def test_a_periodic_hole_is_refused():
    """Every other word zero, the rest live: no arithmetic kernel produces this."""
    words = [0 if i % 2 else (i + 1) for i in range(64)]
    with pytest.raises(ReadbackIntegrityError):
        require_intact(_buf(words), transport="t")


def test_a_clean_buffer_is_accepted():
    require_intact(_buf([i + 1 for i in range(64)]), transport="t")


def test_an_all_zero_buffer_is_NOT_refused():
    """Deliberate: there is no live complement to contrast against, and an all-zero buffer is
    already visible to any value comparison. Refusing it here would mask a real all-zeros bug."""
    require_intact(_buf([0] * 64), transport="t")


def test_a_short_buffer_is_not_evidence():
    """A few words can hold a zero residue class by arithmetic accident."""
    require_intact(_buf([0, 1, 0, 2]), transport="t")
