"""Joint feedback keeps expensive launches and unqualified claims bounded."""

import copy
import json
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1.feedback import joint_performance as J
from merlin_experiments.phase1.feedback import joint_performance_report as P


def observed():
    report = dict(estimated_cycles=100, model_sha256="a" * 64, scope_sha256="b" * 64,
                  domain={"scope_sha256": "b" * 64}, timing_complete=True, unresolved=[],
                  feature_counts={"cpu.load": 23, "target.class.compute": 5},
                  counter_windows=[dict(begin_pc=10, end_pc=20, estimated_cycles=100)],
                  private_path="/operator/reference", expected_output=[42],
                  cycle_breakdown=dict(scope="whole_trace", resources=[dict(
                      resource_id=1, busy_unit_cycles=900, reservations=10, unknown_events=0,
                      peak_queue_entries=3, private_path="/hidden")]))
    observation = dict(status="observed", complete_counts=True,
                       scope_binding=dict(begin_pc=10, end_pc=20, scope_sha256="b" * 64))
    return report, observation


def projection(report, observation):
    return P.project(report, observation, dict(structural_compute_floor_cycles=30),
                     selection_sha256="c" * 64, reason="independent qualification pending")


def test_scoped_cycles_never_make_structural_attainment_qualified():
    report, observation = observed()
    value = projection(report, observation)
    assert value["estimated_whole_soc_cycles"] == 100
    assert value["structural_floor_to_estimate_ratio"] == .3
    assert value["status"] == "PROVISIONAL"
    assert value["qualified_attainable_fraction"] is None
    assert not value["performance_target_met"] and not value["timing_qualified"]
    assert "operator" not in json.dumps(value) and "expected_output" not in json.dumps(value)
    assert not value["modeled_resource_activity"]["usable_as_kernel_decomposition"]


@pytest.mark.parametrize("fault", ["unpriced", "window", "incomplete", "below_floor"])
def test_ineligible_estimates_do_not_guide_performance(fault):
    report, observation = observed()
    if fault == "unpriced":
        report["unresolved"] = ["FP occupancy missing"]
    elif fault == "window":
        report["counter_windows"][0]["end_pc"] = 24
    elif fault == "incomplete":
        observation["complete_counts"] = False
    else:
        report["estimated_cycles"] = report["counter_windows"][0]["estimated_cycles"] = 20
    value = projection(report, observation)
    assert value["estimated_whole_soc_cycles"] is None
    assert value["status"] in {"UNKNOWN", "REFUTED"}
    assert not value["performance_target_met"]


def test_comparisons_require_same_source_scope_and_evaluator():
    report, observation = observed()
    first = projection(report, observation)
    first["capsule_sha256"] = "d" * 64
    second = copy.deepcopy(first)
    second["estimated_whole_soc_cycles"] = 80
    assert P.compare(first, second)["candidate_to_baseline_ratio"] == .8
    for key in ("model_sha256", "comparison_domain_sha256", "scope_sha256", "capsule_sha256"):
        changed = dict(second, **{key: "e" * 64})
        assert P.compare(first, changed)["candidate_to_baseline_ratio"] is None


@pytest.mark.parametrize("numeric,engine,budget", [("fail", "spike", 30), ("pass", "gsim", 30),
                                                   ("pass", "spike", 0)])
def test_failed_or_other_engine_or_exhausted_request_does_not_launch(monkeypatch, numeric, engine, budget):
    choice = ({"sha256": "a" * 64}, {"runtime": {"path": "/must/not/read"}})
    monkeypatch.setattr(J, "selected", lambda: choice)
    monkeypatch.setattr(J, "structural_feedback", lambda *a, **kw: {"structural_compute_floor_cycles": 30})
    monkeypatch.setattr(J, "_runtime", lambda *_: pytest.fail("must not launch or read runtime"))
    row = {}
    J.attach(row, "unused", result=dict(numeric=dict(status=numeric), tiers=dict(L2=dict(status="pass", engine=engine))),
             context=SimpleNamespace(target="neutral"), compiler="unused", capsule={"name": "neutral"},
             remaining_s=budget)
    assert row["performance"]["status"] == "UNKNOWN"


def test_operator_selection_drift_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(dict(schema=J.SCHEMA, per_request_seconds=30)))
    monkeypatch.setenv(J.ENV, json.dumps(dict(path=str(path), sha256=J.F.digest(path))))
    assert J.selected()[1]["per_request_seconds"] == 30
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        J.selected()


def test_capsule_identity_survives_native_snapshot_relocation():
    capsule = {"name": "neutral", "source_sha256": "a" * 64}
    assert J.capsule_identity(capsule) == J.capsule_identity(dict(capsule, __dir__="/changed/snapshot"))
    assert J.capsule_identity(capsule) != J.capsule_identity(dict(capsule, source_sha256="b" * 64))
