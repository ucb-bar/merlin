"""A certification budget must be fitted on the cycle-accurate tier, not on the sum over tiers.

`timing_diagnostic` recorded only `build_s`/`sim_active_s`/`oracle_wait_s`, each SUMMED across every
tier that ran an oracle -- and the per-tier breakdown was computed in `capsule_grade` and thrown away.
The two tiers are not comparable magnitudes: measured on PC00_k64, L2 (spike, functional) took 0.009s
while L3 (verilator, RTL) took 698.234s, a factor of ~78,000. So a cost model fitted on the sum is
fitted on whichever tiers happened to run, and cannot answer the question a capsule's size has to be
sized against -- how long does CERTIFYING this cost.

The fix has two halves and both are checked here: the grader records per-tier timing carrying the
flags that say what kind of oracle produced each number, and `cert_cost` selects the cycle-accurate
tier rather than assuming a tier NAME implies an oracle kind (a target certifies on whichever rung its
contract declares). A capsule whose run never reached a cycle-accurate tier contributes nothing, since
a fit that absorbed its functional time would read a near-zero cost for a capsule nobody certified.
"""
from __future__ import annotations

import json

from merlin.targetgen import cert_cost as CC


def _entry(by_tier: dict | None, summed: float | None = None) -> dict:
    out: dict = {"build_s": 0.5, "oracle_wait_s": 0.0}
    if summed is not None:
        out["sim_active_s"] = summed
    if by_tier is not None:
        out["by_tier"] = by_tier
    return out


def test_the_functional_tier_is_never_read_as_a_certification_cost():
    """The real shape of the bug: a fast functional tier beside a slow cycle-accurate one."""
    secs, basis = CC._cycle_accurate_seconds(_entry({
        "L2": {"sim_active_s": 0.009, "cycle_accurate": False, "derived_from_rtl": False},
        "L3": {"sim_active_s": 698.234, "cycle_accurate": True, "derived_from_rtl": True},
    }, summed=698.243))
    assert secs == 698.234, "the cycle-accurate tier's own time is the certification cost"
    assert "cycle_accurate_tier" in basis and "L3" in basis
    assert secs != 698.243, "the summed value must not be used when a per-tier block exists"


def test_a_run_that_never_certified_contributes_nothing():
    """Not its functional time. A near-zero cost for an uncertified capsule is worse than no sample."""
    secs, basis = CC._cycle_accurate_seconds(_entry({
        "L2": {"sim_active_s": 0.009, "cycle_accurate": False, "derived_from_rtl": False},
    }, summed=0.009))
    assert secs is None, f"expected no sample, got {secs} ({basis})"
    assert "no cycle-accurate tier" in basis


def test_the_tier_is_selected_by_its_declared_kind_not_by_its_name():
    """A target certifies on whichever rung its contract declares, so the NAME cannot be the test."""
    secs, basis = CC._cycle_accurate_seconds(_entry({
        "L4": {"sim_active_s": 42.0, "cycle_accurate": True, "derived_from_rtl": True},
    }))
    assert secs == 42.0 and "L4" in basis


def test_derived_from_rtl_is_accepted_as_the_older_spelling():
    secs, _ = CC._cycle_accurate_seconds(_entry({
        "L3": {"sim_active_s": 7.5, "derived_from_rtl": True},
    }))
    assert secs == 7.5


def test_a_legacy_score_file_still_fits_but_says_so():
    """Existing score files predate the per-tier block; they must remain usable AND be labelled.

    Silently mixing a summed sample into a per-substrate fit is the provenance error this whole change
    exists to remove, so the basis travels with the number.
    """
    secs, basis = CC._cycle_accurate_seconds(_entry(None, summed=100.5))
    assert secs == 100.5
    assert "legacy" in basis, f"a summed sample must be labelled, got {basis!r}"


def test_the_grader_records_the_per_tier_block_it_used_to_discard():
    """The producing half: without this the reader above has nothing to select from."""
    from merlin.common.paths import repo_root

    src = (repo_root() / "merlin" / "python" / "merlin" / "targetgen"
           / "capsule_grade.py").read_text(encoding="utf-8")
    assert '"by_tier"' in src and 'entry_tm["by_tier"]' in src, (
        "capsule_grade must write the per-tier timing block, not just the summed scalars")
    for flag in ("cycle_accurate", "derived_from_rtl", "evidence"):
        assert flag in src, f"the per-tier block must carry {flag!r} so a consumer can select on it"


def test_the_budget_refuses_to_extrapolate_far_past_the_evidence():
    """A budget answer outside the measured range is a guess, and this one is clamped rather than given.

    Measured today: gemmini's fit spans 256..4096 elements while capsules in the corpus reach 262,144,
    so the honest answer for a large budget is the largest size the evidence supports -- not the
    arithmetic solution of the line.
    """
    fit = CC.CostFit(target="t", intercept_s=100.0, per_element_s=0.01, r2=0.9, n_samples=10,
                     elements_min=256, elements_max=4096, metric="max_operand_elements", sources=())
    huge = CC.max_elements_within(fit, 10_000_000.0)
    assert huge is not None and huge <= int(4096 * 2), (
        f"a budget far beyond the evidence must clamp to the measured range, got {huge}")
    assert CC.max_elements_within(fit, 50.0) is None, (
        "a budget under the fixed floor admits no capsule of any size, which is a statement about "
        "the budget rather than about the shape")


def test_a_single_capsule_run_is_a_cost_sample(tmp_path):
    """A calibration run writes capsule_result.json and NO score file, and was invisible to the model.

    `_timing_records` only read score files -- the batch grader's roll-up -- so the one kind of run you
    would deliberately make to calibrate the model (a single capsule at a chosen size) contributed
    nothing. The per-capsule result is the primary record and always carries per-tier timing; the score
    file is derived from it. Measured after this was fixed: the visible sample count for gemmini went
    from 32 to 75, because every capsule run under out/runs/ had been ignored.
    """
    import json

    run = tmp_path / "runs" / "t-capsule-bench" / "CAL_probe"
    run.mkdir(parents=True)
    (run / "capsule_result.json").write_text(json.dumps({
        "capsule": "CAL_probe",
        "tiers": {
            "L2": {"timing": {"sim_active_s": 0.006}, "cycle_accurate": False,
                   "derived_from_rtl": False, "evidence": "spike_console.log"},
            "L3": {"timing": {"sim_active_s": 177.249}, "cycle_accurate": True,
                   "derived_from_rtl": True, "evidence": "rtl_verilator_console.log"},
        },
    }), encoding="utf-8")

    recs = CC._timing_records("t", root=tmp_path)
    assert "CAL_probe" in recs, f"a single-capsule run must contribute a sample: {recs}"
    seconds, source = recs["CAL_probe"]
    assert seconds == 177.249, "the cycle-accurate tier's time, not the functional one or the sum"
    assert "cycle_accurate_tier:L3" in source


def test_a_result_whose_only_tier_is_functional_contributes_nothing(tmp_path):
    import json

    run = tmp_path / "runs" / "t-capsule-bench" / "FUNC_only"
    run.mkdir(parents=True)
    (run / "capsule_result.json").write_text(json.dumps({
        "capsule": "FUNC_only",
        "tiers": {"L2": {"timing": {"sim_active_s": 0.006}, "cycle_accurate": False,
                          "derived_from_rtl": False}},
    }), encoding="utf-8")
    assert CC._timing_records("t", root=tmp_path) == {}, (
        "0.006s from a functional oracle must never enter a certification cost model")
