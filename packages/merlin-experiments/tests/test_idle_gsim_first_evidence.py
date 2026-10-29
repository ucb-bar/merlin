"""The idle gSIM lane gives every family first evidence early and keeps long jobs in the long lane."""

from merlin_experiments.phase1 import joint_gsim_measurement as M
from merlin_experiments.phase1.feedback import idle_gsim as G
from merlin_experiments.phase1.feedback import performance_brief as PB


def _entry(name, family, work, *, evidence, headroom):
    return {"capsule": name, "family": family, "work_cycles": work, "timing_evidence": evidence,
            "headroom_cycles": headroom, "measured_before": False}


def test_smallest_untimed_case_per_family_leads_then_headroom_order():
    entries = [
        _entry("gemm_big", "compute", 5000, evidence=True, headroom=9000),
        _entry("conv_large", "compute", 4000, evidence=False, headroom=8000),
        _entry("conv_small", "compute", 300, evidence=False, headroom=10),
        _entry("decode_big", "read requests", 9000, evidence=False, headroom=7000),
        _entry("decode_small", "read requests", 800, evidence=False, headroom=5),
    ]
    ranked = [c["capsule"] for c in G.order(entries)]
    # One first-evidence case per family, smallest declared work first; the rest by headroom.
    assert ranked == ["conv_small", "decode_small", "gemm_big", "conv_large", "decode_big"]
    # A family that already has evidence contributes no first-evidence case.
    timed = [dict(e, timing_evidence=True) for e in entries]
    assert [c["capsule"] for c in G.order(timed)][0] == "gemm_big"


def test_candidates_classify_untimed_cases_and_their_family_from_declared_work():
    roofline = {"useful_macs": 1 << 20, "structural_peak_macs_per_cycle": 256}
    records = {
        "small": {"elf_sha256": "a", "structural_roofline": roofline},
        "large": {"elf_sha256": "b", "structural_roofline": {**roofline, "useful_macs": 1 << 30}},
    }
    model = {"classes": {"all": {"overhead_cycles": 0.0, "factor": 2.0, "margin": 1.5}}, "worst_ratio": 3.0}
    ranked = G.candidates(["large", "small"], {"large": "pass", "small": "pass"}, {}, "d",
                          records=records, model=model)
    # Headroom alone would put "large" first; the smallest untimed case of the family leads instead.
    assert [c["capsule"] for c in ranked] == ["small", "large"]
    assert not ranked[0]["timing_evidence"] and ranked[0]["family"] == "compute"
    assert ranked[0]["work_cycles"] < ranked[1]["work_cycles"]


def test_a_job_long_relative_to_pending_smaller_work_uses_only_the_long_lane():
    entries = [{"capsule": n} for n in ("a", "b", "c", "jt")]
    expected = {"a": 100.0, "b": 200.0, "c": 300.0, "jt": 9000.0}
    assert G.long_lane(entries, expected) == {"jt"}
    # Nothing else pending, or no single natural break: it may take a normal slot.
    assert G.long_lane([{"capsule": "jt"}], {"jt": 9000.0}) == set()
    assert G.long_lane(entries[:3], {"a": 100.0, "b": 200.0, "c": 400.0}) == set()
    fit = {"setup_s": 0.0, "cycles_per_s": 1.0, "margin": 2.0}
    window = dict(remaining_drain_s=1e6, remaining_end_s=2e6, final_window_s=1000, final_slots_free=True)
    plan, why = M.job_plan(fit, 9000, 9000, long_allowed=False, long_lane=True, **window)
    assert plan is None and "long lane" in why
    plan, _ = M.job_plan(fit, 9000, 9000, long_allowed=True, long_lane=True, **window)
    assert plan["long"] is True
    # Without the classification the same job would have held a normal slot.
    assert M.job_plan(fit, 9000, 9000, long_allowed=False, **window)[0]["long"] is False


def test_brief_says_unknown_timing_is_missing_evidence_and_how_to_measure_one_case(tmp_path):
    rows = [{"capsule": "untimed", "tiers": {"L2": {"status": "pass"}}, "performance": {"status": "UNKNOWN"}}]
    brief = PB.build(rows, workspace=tmp_path)
    unknown = brief["unknown_timing"]
    assert "missing evidence" in unknown["meaning"]
    assert unknown["commands"] == ["python agent_selfcheck.py --capsules untimed --sim gsim --wait-budget 5"]
    assert "--attach" in unknown["how_to_measure"]
