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


def test_model_cannot_contradict_extracted_capacity():
    from merlin_experiments.phase1.feedback.joint_timing_facts import verify

    alignment = dict(schema="merlin.joint_timing_fact_alignment.v1", mappings=[dict(
        fact_paths=[["banks"], ["rows"]], model_path=["capacity"], role="logical local rows")])
    assert verify(alignment, {"banks": 4, "rows": 128}, {"capacity": 512})["status"] == "consistent"
    with pytest.raises(ValueError, match="contradicts"):
        verify(alignment, {"banks": 4, "rows": 128}, {"capacity": 1024})


def test_tuned_model_is_selected_and_drift_never_falls_back():
    from merlin_experiments.phase1.feedback.joint_timing_selection import choose, within_coverage

    epoch = dict(epoch=1, mode="tuned", runtime={"path": "tuned"}, calibration_evidence={"path": "evidence"},
                 coverage=dict(capsule_sha256=["a" * 64], scope_sha256="scope", memory_regime_sha256="memory",
                               feature_ranges={"cpu.load": [10, 50]}))
    selection = dict(runtime={"path": "untuned"}, timing_epochs=[epoch])
    chosen = choose(selection, "a" * 64)
    assert chosen == epoch
    assert choose(selection, "b" * 64)["mode"] == "untuned"
    assert within_coverage(chosen, dict(scope_sha256="scope", domain=dict(memory_regime_sha256="memory"),
                                        feature_counts={"cpu.load": 25}))
    assert not within_coverage(chosen, dict(scope_sha256="scope", domain=dict(memory_regime_sha256="memory"),
                                            feature_counts={"cpu.load": 60}))


def test_native_ceiling_overlay_resolves_only_exact_pinned_source():
    source = dict(name="neutral", required_oracle_tiers=["L0", "L1", "L2", "L3"],
                  numeric_policy={"compare": "exact_int"}, interface_mlir="program.mlir")
    selection = {"members": {"neutral": {"capsule_sha256": J.capsule_identity(source)}}}
    staged = dict(source, oracle_tier_ceiling="L3", __dir__="/native/private/public/neutral")
    assert J.selected_capsule(selection, staged) == source
    assert staged["oracle_tier_ceiling"] == "L3"  # no mutation of execution contract
    for changed in (dict(staged, interface_mlir="other.mlir"),
                    dict(staged, required_oracle_tiers=["L2"]),
                    dict(staged, numeric_policy={"compare": "approximate"}),
                    dict(staged, oracle_tier_ceiling="unknown")):
        with pytest.raises(ValueError, match="different capsule bytes"):
            J.selected_capsule(selection, changed)
    declared = dict(source, oracle_tier_ceiling="L2")
    selected = {"members": {"neutral": {"capsule_sha256": J.capsule_identity(declared)}}}
    with pytest.raises(ValueError):
        J.selected_capsule(selected, dict(declared, oracle_tier_ceiling="L3"))


def test_staged_source_routes_to_tuned_epoch_without_waiving_missing_rtl(monkeypatch):
    source = dict(name="neutral", required_oracle_tiers=["L2", "L3"])
    identity = J.capsule_identity(source)
    staged = dict(source, oracle_tier_ceiling="L3")
    report, observation = observed()
    tuned = dict(epoch=1, mode="tuned", runtime={"path": "tuned"},
                 calibration_evidence={"path": "evidence"}, coverage={})
    choice = ({"sha256": "a"*64}, {"members": {"neutral": {"capsule_sha256": identity}},
              "facts": {}, "runtime": {"path": "untuned"}})
    from merlin_experiments.phase1.feedback import joint_timing_selection as S
    def choose(selection, key):
        return tuned if key == identity else dict(epoch=0, mode="untuned")
    monkeypatch.setattr(S, "choose", choose)
    monkeypatch.setattr(S, "within_coverage", lambda *_: True)
    monkeypatch.setattr(J, "selected", lambda: choice)
    monkeypatch.setattr(J, "structural_feedback", lambda *_a, **_k: {"structural_compute_floor_cycles":30})
    monkeypatch.setattr(J, "_read", lambda *_: {})
    def runtime(pin, **kwargs):
        assert pin == tuned["runtime"]
        return SimpleNamespace(review_id="explicit review")
    monkeypatch.setattr(J, "_runtime", runtime)
    monkeypatch.setattr(J, "_observe", lambda *_a, **_k: (report, observation))
    from merlin_experiments.phase1 import joint_calibration_service as C
    recorded=[]
    monkeypatch.setattr(C, "register", lambda *_a, **k: recorded.append(k["capsule"]))
    result=dict(numeric={"status":"pass"},tiers={"L2":{"status":"pass","engine":"spike"},
                "L3":{"status":"unavailable"}},trace_check={"status":"unavailable_fields"})
    row={}
    J.attach(row,"unused",result=result,context=SimpleNamespace(target="neutral"),
             compiler="unused",capsule=staged,remaining_s=120)
    assert row["performance"]["status"] == "PROVISIONAL"
    assert row["performance"]["timing_mode"] == "tuned"
    assert row["performance"]["capsule_sha256"] == identity
    assert row["performance"]["staged_capsule_sha256"] == J.capsule_identity(staged)
    assert recorded == [source]
    assert result["tiers"]["L3"]["status"] == "unavailable"
    assert not row["performance"]["promotion_authority"]


def test_capture_metadata_requires_reopened_exact_policy_source(tmp_path):
    import yaml
    source = dict(name="neutral", required_oracle_tiers=["L2", "L3"],
                  capture_tool={"status":"not_available", "reason":"source metadata"})
    selection = {"members":{"neutral":{"capsule_sha256":J.capsule_identity(source)}}}
    public=tmp_path/"public"/"neutral";public.mkdir(parents=True)
    staged={k:v for k,v in source.items() if k!="capture_tool"}
    staged.update(oracle_tier_ceiling="L3",__dir__=str(public))
    with pytest.raises(ValueError):J.selected_capsule(selection,staged)
    origin=tmp_path/"policy"/"0"/"neutral"/"capsule.yaml"
    origin.parent.mkdir(parents=True);origin.write_text(yaml.safe_dump(source))
    assert J.selected_capsule(selection,staged)==source
    origin.write_text(yaml.safe_dump(dict(source,capture_tool={"status":"changed"})))
    with pytest.raises(ValueError):J.selected_capsule(selection,staged)


def test_original_elf_slot_ignores_retained_execution_copy(tmp_path):
    from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME
    direct=tmp_path/PACKAGE_ELF_NAME;direct.write_bytes(b'producer')
    retained=tmp_path/'executed'/'spike'/'digest'/PACKAGE_ELF_NAME
    retained.parent.mkdir(parents=True);retained.write_bytes(b'producer')
    assert J.original_executable(tmp_path)==direct
    direct.unlink()
    with pytest.raises(ValueError):J.original_executable(tmp_path)
    direct.symlink_to(retained)
    with pytest.raises(ValueError):J.original_executable(tmp_path)
