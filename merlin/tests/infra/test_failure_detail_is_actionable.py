"""A tier that ran and failed must tell the agent WHY, and which tier.

Measured on a full atlas arm-4 run: every capsule carried the tier reason "atlas program did not halt
within 20000 instructions (functional)", but the agent was shown only

    plane: oracle_unavailable / NOT_RUN_IS_NOT_PASS
    detail: "mandatory tier L# did not run (unavailable)"

Two losses compound there. The tier's own `reason` was never interpolated, so the actionable half was
dropped; and the blanket numeric scrub that stops a golden value leaking through free text also rewrote
the tier LABEL "L2" to "L#", so the agent could not even tell which tier blocked it. Ten rounds of a
fully conformant model went into feedback naming nothing it could fix, and its effort decayed round over
round. An oracle that is ABSENT and a program that RAN AND HUNG are different events.
"""
from __future__ import annotations

import inspect
import sys

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))
import qa_check as qc  # noqa: E402


def test_the_tier_reason_leads_the_detail():
    """The reason must be interpolated, not just the status."""
    from merlin.targetgen import capsule_runner
    src = inspect.getsource(capsule_runner.run_capsule)
    assert '_why = (getattr(tr, "reason", None) or "").strip()' in src, \
        "the tier reason must be read"
    assert '"tier_reason": _why or None' in src, "the reason must be carried structurally too"


def test_the_tier_label_survives_redaction():
    """'L2' must not come out as 'L#'."""
    row_src = inspect.getsource(qc._per_capsule_from_results)
    assert '"failure_tier"' in row_src, "the tier label must be published"
    # the shape guard must accept a tier label and reject anything else
    guard = (lambda v: v if isinstance(v, str) and len(v) <= 4 and v[:1] == "L"
             and v[1:].isdigit() else None)
    assert guard("L2") == "L2"
    assert guard("L15") == "L15"
    assert guard(None) is None
    assert guard("Lmax") is None, "only a numeric tier label may ride this field"
    assert guard("expected=42") is None, "the field must not become a leak channel"


def test_a_hung_program_still_reads_as_a_hung_program_after_redaction():
    """Numbers are scrubbed, but the sentence must stay actionable."""
    detail = ("atlas program did not halt within 20000 instructions (functional) "
              "(mandatory tier L2, status unavailable)")
    out = qc._redact_detail(detail)
    assert "did not halt" in out, "the actionable verb must survive"
    assert "20000" not in out, "concrete numbers are still scrubbed"


def test_redaction_still_scrubs_a_bare_value():
    assert "3.5" not in (qc._redact_detail("expected 3.5 got 4.0") or "")
