"""Explicit fast numerical/count feedback, with full source effects pending.

The original live selector and numerical-product reopener remain the owners.
This module supplies no source-effect, timing, incumbent or convergence grant.
"""

from merlin.runtime import _native_files as F
from merlin.targetgen import checked_source_model_diagnostic as S

from . import archive_native_screen as NATIVE
from . import ordinary_checked_source as CHECKED
from .contracts import StageGateError, canonical_json, document_sha256

EFFECTS_SCHEMA = "merlin.public_numerical_counts_pending_effects.v1"


def is_fast(record):
    if not isinstance(record, dict) or "verification_protocol" not in record:
        return False
    if (record.get("schema") != S.NUMERICAL_COUNTS_SCHEMA
            or record.get("verification_protocol") != S.NUMERICAL_COUNTS_PROTOCOL):
        raise StageGateError("queued numerical/count feedback requires its explicit verification protocol")
    return True


def selection_record(policy, *, deadline=None):
    checked = CHECKED.record(
        getattr(policy, "_checked_source_diagnostic", None),
        getattr(policy, "_compiler_pass_selection", None), deadline=deadline,
    )
    return checked if is_fast(checked) else None


def grade_matches(record, grade):
    """A new label under an old selection is a refusal, never a functional PASS."""
    fast = is_fast(record)
    claimed = isinstance(grade, dict) and (
        grade.get("schema") == S.FAST_RESULT_SCHEMA
        or grade.get("checked_source_status") == S.FAST_STATUS
        or "verification_protocol" in grade
    )
    matches = S.numerical_grade_matches_selection(record, grade)
    tiers = grade.get("tiers", {}) if isinstance(grade, dict) else {}
    complete_numerical_claim = all((tiers.get(t) or {}).get("status") == "pass" for t in ("L0", "L1", "L2"))
    if not matches and (claimed or fast and complete_numerical_claim):
        raise StageGateError("queued numerical grade differs from its original live verification selection")
    return matches


def pending_effects(record):
    if not is_fast(record):
        raise StageGateError("pending source effects require the explicit numerical/count selection")
    return dict(
        schema=EFFECTS_SCHEMA, status="PENDING", verification_protocol=S.NUMERICAL_COUNTS_PROTOCOL,
        checked_source_diagnostic_sha256=document_sha256(record),
        required_before="compiler acceptance: complete source/product/host-effect audit of frozen compiler",
        formal_admission=False, promotion_authority="NONE", incumbent_authority="NONE",
        roofline_convergence_authority="NONE", timing_cycles=None,
    )


def validate_projection(value, record):
    if canonical_json(value) != canonical_json(pending_effects(record)):
        raise StageGateError("queued numerical/count feedback changed its pending source-effect obligations")


def recheck_child_result(policy, document, *, work_root, selected_members, deadline):
    """Reopen every PASS arm even when no native count receipt was produced."""
    record = selection_record(policy, deadline=deadline)
    if record is None:
        return
    limit = CHECKED._deadline(deadline)
    root = F.canonical(work_root)
    if not root.is_relative_to(F.canonical(policy.feedback_evaluator.work_root)):
        raise StageGateError("numerical/count job escaped its original private owner")
    members = sorted(policy.search_corpus.capsules, key=lambda row: (row.family, row.capsule))
    if (type(document) is not dict or document.get("selected_members") != [list(k) for k in selected_members]
            or type(document.get("cells")) is not list or len(document["cells"]) != len(members)):
        raise StageGateError("numerical/count feedback changed its original admitted roster")
    validate_projection(document.get("source_effects"), record)
    for index, (member, cell) in enumerate(zip(members, document["cells"], strict=True)):
        if (type(cell) is not dict or (cell.get("family"), cell.get("capsule"), cell.get("source_sha256")) != (
            member.family, member.capsule, member.source_sha256
        )):
            raise StageGateError("numerical/count feedback changed its original source member")
        if (member.family, member.capsule) not in selected_members:
            continue
        for arm in ("baseline", "candidate"):
            CHECKED._completed(limit)
            value = cell.get(arm)
            if type(value) is not dict:
                raise StageGateError("numerical/count feedback lost its original arm")
            validate_projection(value.get("source_effects"), record)
            if value.get("functional_status") == "PASS":
                numerical = {k: v for k, v in value.items() if k != "native_exploration"}
                NATIVE.original_numerical_observation(
                    F.canonical(root / f"m{index:03d}_{arm}"), numerical,
                    checked_source=(policy._checked_source_diagnostic, policy._compiler_pass_selection),
                )
    CHECKED._completed(limit)
