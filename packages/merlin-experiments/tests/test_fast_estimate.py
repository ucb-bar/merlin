"""Every passing case gets a calibrated cycle estimate from its emitted command census.

The Spike timing observer needs far longer for a large convolution than a grade gives it; such a case
used to read ``timing_status: UNKNOWN`` and stay unranked. The fast estimate prices the case's decoded
command census against its structural floor with coefficients fitted to complete timing-model estimates,
and the headroom ranking uses it (labelled) when no complete estimate exists.
"""

import json

import pytest

from merlin_experiments.phase1.feedback import fast_estimate as FE
from merlin_experiments.phase1.feedback import performance_headroom as H
from merlin_experiments.phase1.feedback import shape_classes as S

FACTS = "f" * 64
PER_COMMAND, PER_FENCE = 16.0, 50.0


def _trace(commands, fences, *, unproved=False):
    """A decoded stream: one command site in a counted loop, one fence site, plus a bracketing fence."""
    return {
        "instructions": [
            {"index": 0, "class": "FENCE", "funct": None, "executions": 1},
            {"index": 1, "class": "COMPUTE_CMD", "funct": 4, "executions": None if unproved else commands},
            {"index": 2, "class": "FENCE", "funct": None, "executions": fences - 1},
            {"index": 3, "class": "UNKNOWN", "funct": None, "raw": ".insn other"},  # neither: not counted
        ]
    }


def _roofline(floor):
    return {"structural_compute_floor_cycles": floor, "useful_macs": floor * 256, "structural_peak_macs_per_cycle": 256}


def _truth(floor, commands, fences):
    return max(floor, commands * PER_COMMAND) + fences * PER_FENCE


def _points():
    out = []
    for floor, commands, fences in [(1_000, 50, 4), (200_000, 300_000, 17_000), (80_000, 1_000, 10),
                                    (5_000, 20_000, 9_000), (1_800_000, 270_000, 20_000), (400, 30, 2),
                                    (60_000, 10_000, 100), (900_000, 120_000, 600), (2_000, 8_000, 4_000),
                                    (300_000, 4_000, 2), (10_000, 40_000, 40), (700, 100, 50), (50_000, 2_000, 1_000)]:
        out.append((floor, commands, fences, _truth(floor, commands, fences)))
    return out


def test_census_multiplies_sites_by_proved_executions():
    counted = FE.census(_trace(129_024, 17_370))
    assert counted == {"commands": 129_024, "fences": 17_370, "sites": 3, "unproved_sites": 0, "complete": True}
    partial = FE.census(_trace(10, 3, unproved=True))
    assert partial["commands"] == 1 and partial["complete"] is False  # an unproved site counts once
    assert FE.census({"instructions": []}) is None and FE.census(None) is None


def test_fit_recovers_issue_and_fence_costs_and_states_its_band():
    fitted = FE.fit(_points())
    assert fitted["cycles_per_command"] == pytest.approx(PER_COMMAND, rel=0.1)
    assert fitted["cycles_per_fence"] == pytest.approx(PER_FENCE, rel=0.1)
    assert fitted["ratio_quantiles"]["p10"] <= 1.0 <= fitted["ratio_quantiles"]["p90"]
    assert FE.fit(_points()[: FE.MIN_PAIRS - 1]) is None  # too few pairs: no calibration


def test_estimate_prices_census_with_band_and_unknown_without_calibration():
    fitted = FE.fit(_points())
    block = FE.estimate(FE.census(_trace(270_000, 20_000)), _roofline(1_800_000), fitted)
    assert block["evidence_kind"] == "fast_estimate" and block["status"] == "ESTIMATED"
    assert block["cycles"] == pytest.approx(_truth(1_800_000, 270_000, 20_000), rel=0.1)
    assert block["low_cycles"] <= block["cycles"] <= block["high_cycles"]
    assert block["limited_by"] == "command issue" and block["lower_bound"] is False
    unknown = FE.estimate(FE.census(_trace(10, 3)), _roofline(100), None)
    assert unknown["status"] == "UNKNOWN" and unknown["cycles"] is None and "calibration" in unknown["reason"]


def test_run_ledger_calibrates_and_otherwise_the_seed_must_match_the_facts(tmp_path, monkeypatch):
    ledger = tmp_path / FE.LEDGER
    for floor, commands, fences, cycles in _points():
        counted = {"commands": commands, "fences": fences, "complete": True}
        FE.record(_roofline(floor), counted, cycles, facts_sha256=FACTS, path=ledger)
    FE.record(_roofline(10), {"commands": 1, "fences": 1, "complete": False}, 10**9, facts_sha256=FACTS, path=ledger)
    assert len(json.loads(ledger.read_text())["pairs"]) == len(_points())  # a partial census never calibrates
    own = FE.calibration(FACTS, path=ledger)
    assert own["source"].startswith("this run") and own["cycles_per_command"] == pytest.approx(PER_COMMAND, rel=0.1)
    assert FE.calibration("0" * 64, path=ledger) is None  # other facts: no run fit, no matching seed
    seed = json.loads((FE.Path(FE.__file__).parents[2] / "resources" / FE.SEED).read_text())
    shipped = seed["calibrations"][0]["facts_sha256"]
    found = FE.calibration(shipped, path=tmp_path / "absent.json")
    assert found["source"].startswith("reviewed seed") and found["pairs"] >= FE.MIN_PAIRS


def test_attach_estimates_a_deferred_case_and_records_a_complete_one(tmp_path):
    ledger = tmp_path / FE.LEDGER
    case = tmp_path / "case"
    (case / "generated").mkdir(parents=True)
    (case / "generated" / FE.TRACE).write_text(json.dumps(_trace(270_000, 20_000)))
    for floor, commands, fences, cycles in _points():
        FE.record(_roofline(floor), {"commands": commands, "fences": fences, "complete": True}, cycles,
                  facts_sha256=FACTS, path=ledger)
    deferred = {"status": "UNKNOWN", "estimated_whole_soc_cycles": None}
    FE.attach(deferred, case, _roofline(1_800_000), facts_sha256=FACTS, path=ledger)
    assert FE.cycles_of(deferred) == deferred["fast_estimate"]["cycles"] > 1_800_000
    complete = {"complete_event_accounting": True, "estimated_whole_soc_cycles": 6_000_000}
    FE.attach(complete, case, _roofline(1_800_000), facts_sha256=FACTS, path=ledger)
    assert json.loads(ledger.read_text())["pairs"][-1]["cycles"] == 6_000_000
    missing = {"estimated_whole_soc_cycles": None}
    FE.attach(missing, tmp_path / "no-case", _roofline(10), facts_sha256=FACTS, path=ledger)
    assert missing["fast_estimate"]["status"] == "UNKNOWN" and FE.cycles_of(missing) is None


def test_headroom_ranks_a_case_on_its_fast_estimate_only_without_a_complete_one(tmp_path):
    fitted = FE.fit(_points())
    fast = FE.estimate(FE.census(_trace(270_000, 20_000)), _roofline(1_800_000), fitted)
    row = {"capsule": "conv", "performance": {"status": "UNKNOWN", "structural_roofline": _roofline(1_800_000),
                                              "fast_estimate": fast}}
    entry = H._entry("conv", row, {}, {}, tmp_path)
    assert entry["cycles"] == fast["cycles"] and entry["evidence"].startswith("estimated: fast census estimate")
    assert S.evidence_kind(entry["evidence"]) == "fast_estimate" and "fast_estimate" in S.RANKABLE
    complete = dict(row["performance"], status="PROVISIONAL", complete_event_accounting=True,
                    estimated_whole_soc_cycles=3_000_000)
    preferred = H._entry("conv", {"capsule": "conv", "performance": complete}, {}, {}, tmp_path)
    assert preferred["cycles"] == 3_000_000 and preferred["evidence"].startswith("modeled")


def test_self_check_brief_ranks_a_deferred_primary_case_on_its_fast_estimate(tmp_path):
    from merlin_experiments.phase1.feedback import performance_brief as PB

    (tmp_path / "PUBLIC_CAPABILITIES.json").write_text(json.dumps(dict(
        schema="merlin.public_capability_inventory.v1", members={"conv": dict(workload_role="large_kernel")})))
    fast = FE.estimate(FE.census(_trace(270_000, 20_000)), _roofline(1_800_000), FE.fit(_points()))
    row = {"capsule": "conv", "tiers": {"L2": "pass"}, "performance": {
        "status": "UNKNOWN", "estimated_whole_soc_cycles": None, "structural_roofline": _roofline(1_800_000),
        "reason": "observation budget too small", "fast_estimate": fast}}
    brief = PB.build([row], workspace=tmp_path)
    entry = brief["largest_headroom"][0]
    assert entry["test_case"] == "conv" and entry["cycles"] == fast["cycles"]
    assert entry["evidence_kind"] == "fast_estimate"
