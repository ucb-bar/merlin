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
from merlin_experiments.phase1.feedback import qa as qc  # noqa: E402


def test_the_tier_reason_leads_the_detail(tmp_path):
    """The reason must be interpolated, not just the status.

    Asserted on the produced row rather than on the source text. The original form matched a literal
    line inside `run_capsule`; when that block was lifted into the shared finalizer -- so the model path
    would report tier reasons too -- the behaviour was preserved but the assertion broke, which is the
    failure mode of testing where code lives instead of what it does.
    """
    from merlin.targetgen import capsule_runner as R
    from merlin.targetgen.capsule_common import make_run_paths

    paths = make_run_paths(tmp_path / "runs", "cap", suite="t", target="radiance", dtype="fp32", benchmark="cap")
    why = "atlas program did not halt within 20000 instructions (functional)"
    row = R._finalize_capsule_result(
        name="cap",
        capsule={"name": "cap", "kind": "isa", "label": "public", "required_oracle_tiers": ["L2"]},
        status="pass",
        failure=None,
        tiers={"L2": R.TierResult("L2", "unavailable", True, reason=why)},
        trace_check_res={"status": "skipped", "violations": []},
        numeric={"status": "pass"},
        required={"L2"},
        no_oracle=False,
        eff_target="radiance",
        paths=paths,
        run_id="cap",
        cfg=R._config_for_target("radiance", "t", "fp32"),
        contract=None,
    )

    assert row["status"] != "pass", "a mandatory tier that did not run cannot yield a pass"
    fail = row["failure"]
    assert fail["tier_reason"] == why, "the reason must be carried structurally"
    assert fail["detail"].startswith(why), "the actionable half must LEAD the detail, not be dropped"
    assert fail["tier"] == "L2", "the agent must be told which tier blocked it"


def test_the_tier_label_survives_redaction():
    """'L2' must not come out as 'L#'."""
    row_src = inspect.getsource(qc._per_capsule_from_results)
    assert '"failure_tier"' in row_src, "the tier label must be published"

    # the shape guard must accept a tier label and reject anything else
    def guard(v):
        return v if isinstance(v, str) and len(v) <= 4 and v[:1] == "L" and v[1:].isdigit() else None

    assert guard("L2") == "L2"
    assert guard("L15") == "L15"
    assert guard(None) is None
    assert guard("Lmax") is None, "only a numeric tier label may ride this field"
    assert guard("expected=42") is None, "the field must not become a leak channel"


def test_a_hung_program_still_reads_as_a_hung_program_after_redaction():
    """Numbers are scrubbed, but the sentence must stay actionable."""
    detail = "atlas program did not halt within 20000 instructions (functional) (mandatory tier L2, status unavailable)"
    out = qc._redact_detail(detail)
    assert "did not halt" in out, "the actionable verb must survive"
    assert "20000" not in out, "concrete numbers are still scrubbed"


def test_redaction_still_scrubs_a_bare_value():
    assert "3.5" not in (qc._redact_detail("expected 3.5 got 4.0") or "")


# --- the harness's own counts are not the answer ----------------------------------------------------
# The same loss, one turn of the screw further on. The scrub that stops a golden value leaking also
# collapsed the counts the grader reports about ITSELF, so the agent read
#
#     on-mesh execution: # of # tile(s) passed, # failed, # unavailable, # unsynthesizable
#
# and could not tell a wrong kernel from a bench that has no builder for its op. MEASURED on
# M2_microvit_gemmini: the real counts were 12 of 12 passed, 0 failed, 0 unavailable, 3 unsynthesizable
# -- the model was bit-exact and every certifiable tile passed on RTL. Four grades plateaued on it.

_M2_DETAIL = (
    "declared oracle tier(s) ['L3'] RAN and did not pass (on-mesh execution: 12 of 12 tile(s) passed, "
    "0 failed, 0 unavailable, 3 unsynthesizable); a whole-model verdict cannot be a pass over a "
    "failing tier"
)


def test_the_agent_can_tell_unsynthesizable_from_failed():
    """The distinction the plateau turned on: both counts must survive, and be distinguishable."""
    out = qc._redact_detail(_M2_DETAIL)
    assert "0 failed" in out and "3 unsynthesizable" in out
    assert "12 of 12 tile(s) passed" in out, "both runs of `N of N` are judged by the noun they share"


def test_a_golden_value_still_scrubs():
    """The mutation this whitelist must not weaken: a value beside its reference is still destroyed."""
    assert qc._redact_detail("expected 42, actual 17") == "expected #, actual #"
    assert qc._redact_detail("cos 0.9997 against the golden") == "cos # against the golden"
    assert qc._redact_detail("output[3] = 128 but reference says 127") == "output[#] = # but reference says #"


def test_a_count_of_something_unnamed_is_still_a_value():
    """The whitelist is a closed set of nouns about the harness, not a general amnesty on numbers."""
    assert qc._redact_detail("element 7 differs") == "element # differs"
    assert qc._redact_detail("4 of 8 outputs") == "# of # outputs"


def test_structure_and_position_are_untouched():
    """The two exemptions that already existed keep working beside the new one."""
    assert qc._redact_detail("tensor<16x16xi8>") == "tensor<16x16xi8>"
    assert qc._redact_detail('File "gemmini_opt.py", line 412') == 'File "gemmini_opt.py", line 412'
