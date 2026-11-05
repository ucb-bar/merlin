"""The fast estimate is labelled order-blind and says when it cannot rank two variants."""

import json

from merlin_experiments.phase1.feedback import fast_estimate as FE
from merlin_experiments.phase1.feedback import joint_outcome_brief as J
from merlin_experiments.phase1.feedback import joint_performance_report as R
from merlin_experiments.phase1.feedback import seed_timing
from merlin_experiments.phase1.feedback import shape_classes as S
from merlin_experiments.phase1.providers.authoring_prompt import COEVOLUTION_FEEDBACK_GUIDANCE

CALIBRATED = {"cycles_per_command": 16.0, "cycles_per_fence": 50.0, "ratio_quantiles": {"p10": 0.9, "p90": 1.2}}
ROOFLINE = {"structural_compute_floor_cycles": 100, "useful_macs": 25_600, "structural_peak_macs_per_cycle": 256}


def _site(klass, funct, runs, width=None):
    return {"class": klass, "funct": funct, "executions": runs,
            "rs2": {"raw": None, "kind": "unknown", "known_bits": {"mask": 1, "value": width or 0, "width": 64}}}


def _trace(*, loads=64, load_width=4, computes=128, fences=4, swap=False):
    sites = [_site("MVIN", 2, loads, load_width), _site("COMPUTE_PRELOADED", 4, computes)]
    if swap:
        sites.reverse()
    return {"instructions": [_site("FENCE", None, 1), *sites, _site("FENCE", None, fences - 1)]}


def _block(**shape):
    return FE.estimate(FE.census(_trace(**shape)), ROOFLINE, CALIBRATED)


def test_label_and_block_name_what_the_estimate_cannot_see():
    block = _block()
    assert block["status"] == "ESTIMATED" and block["order_blind"] is True
    assert FE.ORDER_BLIND in block["basis"]
    assert block["blind_to"] == ["transfer size", "load/compute overlap", "issue order", "prefetch"]
    label = FE.label({"fast_estimate": block})
    assert label.startswith("estimated: fast census estimate") and FE.ORDER_BLIND in label
    assert S.evidence_kind(label) == "fast_estimate"
    assert block["census"]["families"] == {"contraction": 128, "movement": 64}


def test_ranking_refuses_to_rank_what_the_census_cannot_distinguish():
    base = _block()
    wider = FE.ranking(base, _block(loads=16, load_width=1))
    assert wider["status"] == "NOT_RANKABLE" and "transfer sizes differ" in wider["reason"]
    assert wider["candidate_to_baseline_ratio"] is None and wider["recommendation"] == FE.COMPLETE_ESTIMATE_ADVICE
    drains = FE.ranking(base, _block(fences=40))
    assert drains["status"] == "NOT_RANKABLE" and "fences" in drains["reason"]
    reordered = FE.ranking(base, _block(swap=True))
    assert reordered["status"] == "NOT_RANKABLE" and "issue order" in reordered["reason"]
    operands = FE.ranking(base, _block(load_width=8))
    assert operands["status"] == "NOT_RANKABLE" and "command stream" in operands["reason"]
    # Performance records work too; a count difference the estimate prices ranks, with its caveat.
    priced = FE.ranking({"fast_estimate": base}, {"fast_estimate": _block(computes=256, loads=128)})
    assert priced["status"] == "RANKABLE_WITH_CAVEAT" and priced["candidate_to_baseline_ratio"] > 1
    assert FE.ORDER_BLIND in priced["reason"]
    assert FE.ranking(base, base)["status"] == "RANKABLE_WITH_CAVEAT"
    assert FE.ranking(base, {"status": "UNKNOWN"})["status"] == "UNKNOWN"
    # Older blocks without a stream digest: equal counts cannot be told apart.
    legacy = {k: v for k, v in base.items() if k != "census"}
    legacy["census"] = {"commands": base["census"]["commands"], "fences": base["census"]["fences"]}
    assert FE.ranking(legacy, dict(legacy, cycles=legacy["cycles"] + 1))["status"] == "NOT_RANKABLE"


def test_brief_action_is_guidance_with_the_caveat_and_unchanged_otherwise():
    outcomes = {"next_priority": "timing_model_or_observation"}
    J.decide_on_fast_estimate(outcomes, _block())
    assert outcomes["next_priority"] == "performance_search"
    assert FE.ORDER_BLIND in outcomes["next_action"] and FE.COMPLETE_ESTIMATE_ADVICE in outcomes["next_action"]
    complete = {"next_priority": "performance_search", "next_action": "kept"}
    J.decide_on_fast_estimate(complete, _block())
    assert complete == {"next_priority": "performance_search", "next_action": "kept"}


def test_prompt_states_the_limit_and_the_complete_estimate_route():
    text = " ".join(COEVOLUTION_FEEDBACK_GUIDANCE.split())
    assert "order-blind" in text and "transfer size, load/compute overlap, issue order and prefetch" in text
    assert "obtain a complete Spike estimate before deciding" in text
    assert "prefer the change the estimate favors" not in text


def test_like_with_like_comparisons_carry_the_caveat_and_models_stay_unchanged():
    best = {("case", "fast_estimate"): {"cycles": 100}, ("case", "provisional_model", "d"): {"cycles": 100}}
    fast = S.compare("case", 90, "fast_estimate", best, "dense")
    assert fast["ratio_vs_best"] == 0.9 and FE.ORDER_BLIND in fast["verdict"]
    modeled = S.compare("case", 90, "provisional_model", best, "dense", domain="d")
    assert FE.ORDER_BLIND not in modeled["verdict"] and modeled["ratio_vs_best"] == 0.9


def test_seed_comparisons_on_fast_estimates_carry_the_caveat(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "joint_campaign.json").write_text("{}")
    (run / seed_timing.BASELINE).write_text(json.dumps({"cases": {
        "fast": {"cycles": 100, "evidence": "estimated: fast census estimate (seed)"},
        "modeled": {"cycles": 100, "evidence": "modeled: Spike timing model (seed)"}}}))
    entries = [{"test_case": "fast", "cycles": 90, "evidence": FE.label({"fast_estimate": _block()})},
               {"test_case": "modeled", "cycles": 90, "evidence": "modeled: Spike timing model (self-check)"}]
    summary = seed_timing.annotate(entries, root=run)
    assert summary["compared"] == 2 and entries[0]["vs_seed"] == 0.9
    assert FE.ORDER_BLIND in entries[0]["vs_seed_caveat"]
    assert "vs_seed_caveat" not in entries[1]


def test_frozen_evaluation_never_derives_a_ratio_from_the_fast_estimate():
    identity = {k: "x" for k in ("capsule_sha256", "selection_sha256", "model_sha256", "comparison_domain_sha256",
                                 "scope_sha256")}
    fast_only = dict(identity, status="UNKNOWN", fast_estimate=_block())
    complete = dict(identity, status="PROVISIONAL", estimated_whole_soc_cycles=1000)
    for baseline, candidate in ((fast_only, complete), (complete, fast_only), (fast_only, fast_only)):
        result = R.compare(baseline, candidate)
        assert result["status"] == "UNKNOWN" and result["candidate_to_baseline_ratio"] is None
        assert "fast_estimate" in result["reason"] and FE.ORDER_BLIND in result["reason"]
    both = R.compare(dict(complete, fast_estimate=_block()), dict(complete, estimated_whole_soc_cycles=900))
    assert both["status"] == "PROVISIONAL" and both["candidate_to_baseline_ratio"] == 0.9
