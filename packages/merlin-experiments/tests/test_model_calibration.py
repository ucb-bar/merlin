"""The Spike timing model is calibrated only against gSIM cycles of the same executable, and labelled."""

from __future__ import annotations

import json

import pytest

from merlin.common import stat_memo
from merlin_experiments.phase1.feedback import model_calibration as M
from merlin_experiments.phase1.feedback import performance_brief as B


@pytest.fixture(autouse=True)
def _fresh_memo():
    stat_memo.clear()
    yield
    stat_memo.clear()


def _roofline(read_cycles=100):
    return {"structural_compute_floor_cycles": 10,
            "logical_request_ideal_service": {"read": {"ideal_requested_service_cycles": read_cycles}},
            "useful_macs": 4096, "structural_peak_macs_per_cycle": 256}


def _modeled_row(name, digest, cycles):
    return {"capsule": name, "execution_digest": digest, "tiers": {"L2": "pass"},
            "performance": {"status": "PROVISIONAL", "complete_event_accounting": True,
                            "estimated_whole_soc_cycles": cycles, "structural_roofline": _roofline()}}


def _measured_row(name, digest, cycles):
    return {"capsule": name, "execution_digest": digest, "barrier_tier": "L3", "pass": True, "barrier_cycles": cycles}


def _channel(tmp_path, modeled, measured):
    channel = tmp_path / ".qa_channel"
    channel.mkdir(parents=True)
    (channel / "resp_1.json").write_text(json.dumps({"per_capsule": modeled}))
    (channel / "simresp_1.json").write_text(json.dumps({"sim": "gsim", "per_capsule": measured}))
    return tmp_path


def test_pairs_only_match_the_same_test_case_and_executable(tmp_path):
    ws = _channel(tmp_path,
                  [_modeled_row("a", "d1", 1000), _modeled_row("b", "d2", 1000)],
                  [_measured_row("a", "d1", 500), _measured_row("b", "OTHER", 100), _measured_row("c", "d2", 100)])
    found = M.pairs(ws)
    assert [(p["test_case"], p["execution_digest"], p["ratio"]) for p in found] == [("a", "d1", 0.5)]


def test_robust_ratio_and_uncalibrated_levels(tmp_path):
    ratios = [0.4, 0.45, 0.5, 0.55, 3.0]  # one outlier must not move the median much
    modeled = [_modeled_row(f"t{i}", f"d{i}", 1000) for i in range(5)]
    measured = [_measured_row(f"t{i}", f"d{i}", int(1000 * r)) for i, r in enumerate(ratios)]
    fitted = M.fit(M.pairs(_channel(tmp_path, modeled, measured)))
    overall = fitted["levels"]["all"]["all"]
    assert overall["n"] == 5 and overall["median_measured_over_modeled"] == 0.5
    assert overall["interquartile"][0] <= 0.5 <= overall["interquartile"][1]
    few = M.fit(M.pairs(_channel(tmp_path / "x", modeled[:2], measured[:2])))
    assert few["levels"]["all"]["all"] == {"n": 2, "status": "uncalibrated"}
    assert M.apply(1000, few)["model_confidence"]["status"] == "uncalibrated"


def test_apply_is_labelled_an_estimate_with_an_interval():
    fitted = {"pairs": 6, "levels": {"all": {"all": {"n": 6, "median_measured_over_modeled": 0.5,
                                                      "interquartile": [0.4, 0.7], "spread": 0.3}}}}
    out = M.apply(2000, fitted, bound_class="memory requests")
    assert out["calibrated_estimate"] == 1000 and out["calibrated_interval"] == [800, 1400]
    assert out["model_confidence"]["level"] == "all"
    assert "not a measurement" in out["calibration_label"]
    assert M.apply(2000, None) is None


def test_brief_calibrates_modeled_entries_only(tmp_path):
    modeled = [_modeled_row(f"t{i}", f"d{i}", 1000) for i in range(5)]
    measured = [_measured_row(f"t{i}", f"d{i}", 500) for i in range(5)]
    ws = _channel(tmp_path, modeled, measured)
    rows = [_modeled_row("new", "dz", 4000),
            {**_measured_row("gsim_case", "dg", 600), "performance": {"structural_roofline": _roofline()}}]
    brief = B.build(rows, workspace=ws, debuggable=False)
    by_name = {e["test_case"]: e for e in brief["largest_headroom"]}
    assert by_name["new"]["calibrated_estimate"] == 2000 and by_name["new"]["cycles"] == 4000
    assert "calibrated_estimate" not in by_name["gsim_case"]  # a measurement is never "calibrated"
    assert brief["model_calibration"]["pairs"] == 5 and "use" in brief["model_calibration"]
