"""Idle reserved CPUs A/B-time candidate changes on the passing cases with the most headroom, as advice."""

from __future__ import annotations

import json
import shutil

import pytest
from merlin_experiments.phase1.brokers import selfcheck_tuning as T
from merlin_experiments.phase1.feedback import idle_performance as I
from merlin_experiments.phase1.feedback import idle_tuning as U
from merlin_experiments.phase1.feedback import performance_brief as PB
from merlin_experiments.phase1.feedback import scratch_retention

MODEL = {"model_sha256": "m1", "timing_mode": "untuned", "timing_epoch": 0}
CATALOGUE = [
    {"id": "readout", "family": "overlap", "api": "lib.readout", "symbol": "propose", "status": "supported",
     "requirements": {"memory.accumulator": "supported"}, "evidence": "accumulator readout"},
    {"id": "coalesce", "family": "movement", "api": "lib.copies", "symbol": "merge", "status": "supported",
     "requirements": {"memory.explicit_dma": "supported"}, "evidence": "transfer count"},
]


def _performance(cycles, *, floor=100, model=MODEL):
    return {"status": "PROVISIONAL", "complete_event_accounting": True, "estimated_whole_soc_cycles": cycles,
            "structural_roofline": {"structural_compute_floor_cycles": floor, "useful_macs": 1000,
                                    "structural_peak_macs_per_cycle": 10},
            "scoped_resource_feedback": {"resources": [
                {"name": "accelerator.storage.accumulator.all", "capacity_normalized_activity": 0.9}]},
            **model}


def _workspace(tmp_path, cohort=("big", "mid", "small")):
    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / ".qa_channel").mkdir(parents=True)
    (ws / "submission").mkdir()
    (ws / "submission" / "backend.py").write_text("import os\n")
    (ws / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps({"all_mechanisms": CATALOGUE}))
    (run / "qa_history").mkdir(parents=True)
    verdict = {"joint_test_selection": {"performance_cohorts": {"primary": list(cohort)}},
               "per_capsule": [{"capsule": n, "tiers": {"L2": "pass"}} for n in cohort]}
    (run / "qa_history" / "verdict_fast_1.json").write_text(json.dumps(verdict))
    return ws, run


def _baseline(ws, digest, cycles, rid="3_1_base"):
    cases = [{"capsule": n, "status": "pass", "tiers": {"L2": "pass"}, "performance": _performance(c)}
             for n, c in cycles.items()]
    (ws / ".qa_channel" / f"resp_{rid}.json").write_text(json.dumps(
        {"submission_sha256": digest, "graded_at_unix_ns": 1, "selfcheck_submission": "submission", "cases": cases}))


def _variant(ws, name="alt", text="from lib.readout import propose\n"):
    (ws / "variants" / name).mkdir(parents=True, exist_ok=True)
    (ws / "variants" / name / "backend.py").write_text(text)


def _knobs(ws, values=(False, True), current=False):
    (ws / "submission" / "tuning.json").write_text(json.dumps({"ready_readout": current}))
    (ws / "submission" / T.DECLARATION).write_text(json.dumps({
        "schema": T.SCHEMA, "settings_file": "tuning.json",
        "knobs": {"ready_readout": {"values": list(values), "mechanism": "readout"}}}))


@pytest.fixture
def gates(monkeypatch):
    for name in (I.ENV, U.ENV, "MERLIN_SELFCHECK_BROKER_JOBS", *I.KNOBS, *U.KNOBS):
        monkeypatch.delenv(name, raising=False)
    free = {"slots": 20}
    monkeypatch.setattr(I, "free_slots", lambda environ=None: free["slots"])
    monkeypatch.setattr(scratch_retention, "free_space_ok", lambda *a, **k: True)
    return free


def _measurer(ws, run, trees):
    """A measurer whose tree digests come from ``trees`` (label -> digest), so a test can move them."""
    clock = [1000.0]
    measurer = I.IdleMeasurer(ws, run, digest=lambda root: trees[root.relative_to(ws).as_posix()],
                              clock=lambda: clock[0])

    def tick(grading=False):
        clock[0] += max(I.knob("MERLIN_IDLE_PERF_INTERVAL_S"), I.knob("MERLIN_IDLE_PERF_CHECK_S"))
        return measurer.tick(grading=grading)

    return measurer, tick


def _requests(ws):
    return {p.name[len("req_"):-len(".json")]: json.loads(p.read_text())
            for p in (ws / ".qa_channel").glob(f"req_{U.REQUEST_PREFIX}*.json")}


def _answer(ws, rid, body, cycles, *, passed=True, graded=None, model=MODEL):
    """The broker's response to one experiment request, as ``_decorate_response`` shapes it."""
    variant = body.get("submission", "submission") != "submission"
    sha = graded or (body["submission_sha256"] if variant else "copy-sha")
    cases = [{"capsule": n, "status": "pass" if passed else "fail", "tiers": {"L2": "pass" if passed else "fail"},
              "performance": _performance(cycles[n], model=model)} for n in body["capsules"].split(",")]
    document = {"cases": cases, "selfcheck_request_id": rid, "graded_at_unix_ns": 9,
                "submission_sha256": body["submission_sha256"] if not variant else sha,
                "selfcheck_submission": body.get("submission", "submission"),
                "variant": {"sha256": sha, "official": False}, "official": False,
                "tuning_experiment": body["tuning_experiment"]}
    (ws / ".qa_channel" / f"resp_{rid}.json").write_text(json.dumps(document))


def _ledger(ws):
    return json.loads((ws / U.LEDGER).read_text())


def test_tuning_runs_only_on_idle_capacity_once_estimates_are_current(tmp_path, gates):
    ws, run = _workspace(tmp_path)
    _variant(ws)
    trees = {"submission": "d1", "variants/alt": "v1"}
    measurer, tick = _measurer(ws, run, trees)
    assert tick() == "submitted"  # baseline timing comes first
    assert not _requests(ws)
    (timing,) = (ws / ".qa_channel").glob("req_op_idle_*.json")
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200}, rid=timing.name[len("req_"):-len(".json")])
    gates["slots"] = 2
    assert tick(grading=True) == "cpus reserved for the admitting grade"  # the grade reserve is kept
    (ws / ".qa_channel" / "req_3_5_author.json").write_text(json.dumps({"deadline_unix_ns": 2**62}))
    gates["slots"] = 20
    assert tick() == "author request waiting"
    (ws / ".qa_channel" / "req_3_5_author.json").unlink()
    assert not _requests(ws)
    gates["slots"] = 2
    assert tick() == "tuning submitted"
    ((rid, body),) = _requests(ws).items()
    # Sized from free capacity: two free slots, two cases (largest headroom first), one in flight.
    assert body["capsules"] == "big,mid" and body["origin"] == U.ORIGIN == I.TUNING_ORIGIN
    assert rid.startswith(I.REQUEST_PREFIX) and body["suite"] == "scalability" and body["sim"] == "spike"
    assert body["submission"] == "variants/alt" and body["submission_sha256"] == "v1"
    experiment = body["tuning_experiment"]
    assert experiment["base_sha256"] == "d1" and experiment["mechanism_ids"] == ["readout"]
    assert experiment["api"] == "lib.readout:propose" and "readout" in experiment["suggested_for_these_cases"]
    assert (ws / ".qa_channel" / f"detached_{rid}").exists()
    assert tick() in ("background requests in flight", "tuning experiments in flight")
    assert len(_requests(ws)) == 1


def test_in_flight_and_per_source_bounds_follow_free_slots_and_knobs(tmp_path, gates, monkeypatch):
    ws, run = _workspace(tmp_path)
    for name in ("a", "b", "c"):
        _variant(ws, name, f"value = {name!r}\n")
    trees = {"submission": "d1", "variants/a": "va", "variants/b": "vb", "variants/c": "vc"}
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    gates["slots"] = 8
    measurer, tick = _measurer(ws, run, trees)
    statuses = [tick() for _ in range(4)]
    # 8 free: three cases per experiment (the cap) and at most min(2, 8 // 3) = 2 experiments at once.
    assert statuses[:2] == ["tuning submitted"] * 2 and "tuning submitted" not in statuses[2:]
    assert all(len(b["capsules"].split(",")) == 3 for b in _requests(ws).values())
    for rid, body in _requests(ws).items():
        _answer(ws, rid, body, {"big": 800, "mid": 500, "small": 200})
    monkeypatch.setenv("MERLIN_IDLE_TUNE_MAX_PER_SOURCE", "2")
    assert tick() == "tuning budget for this source used"
    monkeypatch.setenv("MERLIN_IDLE_TUNE_MAX_PER_SOURCE", "3")
    assert tick() == "tuning submitted"
    for rid, body in _requests(ws).items():
        _answer(ws, rid, body, {"big": 800, "mid": 500, "small": 200})
    assert tick() == "tuning budget for this source used"
    monkeypatch.setenv("MERLIN_IDLE_TUNE_MAX_PER_SOURCE", "4")
    assert tick() == "estimates current; no untried tuning candidate"


def test_results_are_advisory_bound_to_the_source_digest_and_dropped_when_it_moves(tmp_path, gates):
    ws, run = _workspace(tmp_path)
    _variant(ws)
    trees = {"submission": "d1", "variants/alt": "v1"}
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    measurer, tick = _measurer(ws, run, trees)
    assert tick() == "tuning submitted"
    ((rid, body),) = _requests(ws).items()
    _answer(ws, rid, body, {"big": 600, "mid": 550, "small": 200})
    tick()
    ledger = _ledger(ws)
    assert ledger["advisory_only"] is True and ledger["promotion_authority"] is False
    assert ledger["source_sha256"] == "d1"
    rows = {r["test_case"]: r for r in ledger["experiments"]}
    big = rows["big"]
    assert big["measured_cycles_before"] == 900 and big["measured_cycles_after"] == 600
    assert big["delta_pct"] == -33.33 and big["numerics"].startswith("unchanged")
    assert big["mechanism"] == "readout" and big["api"] == "lib.readout:propose"
    assert "on source d1" in big["evidence"] and big["candidate"]["graded_sha256"] == "v1"
    assert big["how_to_apply"].startswith("compare variants/alt/")
    assert rows["mid"]["delta_pct"] == 10.0  # a slower candidate is reported like a faster one
    assert [r["test_case"] for r in ledger["experiments"]][0] == "big"
    assert (ws / "submission" / "backend.py").read_text() == "import os\n"  # nothing applied

    trees["variants/alt"] = "v2"  # the variant moved on: its old result no longer describes it
    (ws / "variants" / "alt" / "backend.py").write_text("from lib.readout import propose\nchanged = 1\n")
    tick()
    assert _ledger(ws)["experiments"] == []
    trees["variants/alt"] = "v1"
    (ws / "variants" / "alt" / "backend.py").write_text("from lib.readout import propose\n")
    tick()
    assert len(_ledger(ws)["experiments"]) == 3
    trees["submission"] = "d2"  # the source moved on: every result bound to d1 is dropped
    tick()
    assert _ledger(ws)["source_sha256"] == "d2" and _ledger(ws)["experiments"] == []
    # Once d2 has its own estimates, the unchanged variant's timing is compared with them, not re-run.
    timed_v1 = [b for b in _requests(ws).values() if b["submission_sha256"] == "v1"]
    _baseline(ws, "d2", {"big": 1200, "mid": 500, "small": 200}, rid="3_2_base")
    tick()
    rows = {r["test_case"]: r for r in _ledger(ws)["experiments"]}
    assert rows["big"]["source_sha256"] == "d2" and rows["big"]["measured_cycles_before"] == 1200
    assert rows["big"]["delta_pct"] == -50.0 and rows["big"]["candidate_timed_with_source"] == "d1"
    assert [b for b in _requests(ws).values() if b["submission_sha256"] == "v1"] == timed_v1


def test_numerical_changes_and_incomparable_estimates_are_stated_not_compared(tmp_path, gates):
    ws, run = _workspace(tmp_path, cohort=("big",))
    _variant(ws)
    trees = {"submission": "d1", "variants/alt": "v1"}
    _baseline(ws, "d1", {"big": 900})
    measurer, tick = _measurer(ws, run, trees)
    assert tick() == "tuning submitted"
    ((rid, body),) = _requests(ws).items()
    _answer(ws, rid, body, {"big": 100}, passed=False, model={**MODEL, "timing_epoch": 1})
    tick()
    (record,) = _ledger(ws)["experiments"]
    assert record["numerics"].startswith("changed") and record["delta_pct"] is None
    assert record["measured_cycles_after"] is None and "timing_epoch" in record["not_compared"]


def test_declared_knob_values_are_timed_on_a_private_copy(tmp_path, gates):
    ws, run = _workspace(tmp_path)
    _knobs(ws)
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    trees = {"submission": "d1"}
    measurer, tick = _measurer(ws, run, trees)
    assert tick() == "tuning submitted"
    ((rid, body),) = _requests(ws).items()
    assert body["submission"] == "submission" and body["submission_sha256"] == "d1"
    assert body[T.REQUEST_KEY] == {"ready_readout": True}
    assert body["tuning_experiment"]["candidate"]["kind"] == "knob"

    # What the broker does with it: copy, verify, then set the value in the copy only.
    snapshot = tmp_path / "snapshot" / "submission"
    shutil.copytree(ws / "submission", snapshot)
    applied = T.apply_request(snapshot, body, digest=lambda root: "copy-sha")
    assert json.loads((snapshot / "tuning.json").read_text()) == {"ready_readout": True}
    assert json.loads((ws / "submission" / "tuning.json").read_text()) == {"ready_readout": False}
    fields = T.response_fields(applied)
    assert fields["official"] is False and fields["variant"]["sha256"] == "copy-sha"
    assert fields["variant"]["tuning_overrides"] == {"ready_readout": True}
    assert fields["tuning_experiment"]["base_sha256"] == "d1"
    with pytest.raises(ValueError):
        T.apply_request(snapshot, {**body, T.REQUEST_KEY: {"ready_readout": "maybe"}}, digest=str)
    with pytest.raises(ValueError):
        T.apply_request(snapshot, {**body, T.REQUEST_KEY: {"undeclared": True}}, digest=str)
    assert T.apply_request(snapshot, {"x": 1, T.APPLIED_KEY: {"forged": True}}, digest=str) == {"x": 1}

    _answer(ws, rid, body, {"big": 700, "mid": 500, "small": 210})
    tick()
    rows = {r["test_case"]: r for r in _ledger(ws)["experiments"]}
    assert rows["big"]["delta_pct"] == -22.22 and rows["big"]["mechanism"] == "readout"
    assert rows["big"]["how_to_apply"] == "set ready_readout to true in submission/tuning.json"
    # A knob copy is made of one submission digest: a new digest drops the result and asks again.
    trees["submission"] = "d2"
    _baseline(ws, "d2", {"big": 900, "mid": 500, "small": 200}, rid="3_2_base")
    assert tick() == "tuning submitted"
    assert _ledger(ws)["experiments"] == []
    assert sorted(b["submission_sha256"] for b in _requests(ws).values()) == ["d1", "d2"]


def test_a_malformed_knob_declaration_offers_nothing(tmp_path):
    root = tmp_path / "submission"
    root.mkdir()
    (root / "tuning.json").write_text("{}")
    for declaration in ({"schema": "other"}, {"schema": T.SCHEMA, "settings_file": "../x.json", "knobs": {"a": {
            "values": [1]}}}, {"schema": T.SCHEMA, "settings_file": "tuning.json", "knobs": {"a": {"values": [{}]}}}):
        (root / T.DECLARATION).write_text(json.dumps(declaration))
        assert T.declaration(root) is None and T.alternatives(root) == []
    (root / T.DECLARATION).write_text(json.dumps(
        {"schema": T.SCHEMA, "settings_file": "tuning.json", "knobs": {"a": {"values": [1, 2]}}}))
    assert [change["value"] for change in T.alternatives(root)] == [1, 2]  # an unset knob: every value


def test_the_knob_disables_tuning_and_its_publication(tmp_path, gates, monkeypatch):
    ws, run = _workspace(tmp_path)
    _variant(ws)
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    monkeypatch.setenv(U.ENV, "0")
    measurer, tick = _measurer(ws, run, {"submission": "d1", "variants/alt": "v1"})
    assert tick() == "estimates current; tuning disabled"
    assert not _requests(ws) and not (ws / U.LEDGER).exists()
    document = {"submission_sha256": "d1", "performance_brief": {"largest_headroom": []}}
    assert "idle_tuning" not in U.attach(document, ws)["performance_brief"]
    monkeypatch.delenv(U.ENV)
    monkeypatch.setenv(I.ENV, "0")  # disabling idle timing disables what runs on its capacity
    assert tick() == "disabled" and not _requests(ws)


def test_the_brief_shows_only_results_of_its_own_source(tmp_path, gates):
    ws, run = _workspace(tmp_path)
    _variant(ws)
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    measurer, tick = _measurer(ws, run, {"submission": "d1", "variants/alt": "v1"})
    assert tick() == "tuning submitted"
    ((rid, body),) = _requests(ws).items()
    _answer(ws, rid, body, {"big": 600, "mid": 500, "small": 200})
    tick()
    rows = [{"capsule": "big", "tiers": {"L2": "pass"}, "performance": _performance(900)}]
    document = PB.attach({"submission_sha256": "d1", "selfcheck_submission": "submission"}, rows, workspace=ws)
    brief = U.attach(document, ws)["performance_brief"]
    block = brief["idle_tuning"]
    assert block["advisory_only"] is True and block["promotion_authority"] is False
    assert block["count"] == 3 and block["experiments"][0]["test_case"] == "big"
    assert "TUNING_KNOBS.json" in block["offer"] and block["all"] == "qa/idle_tuning.json"
    (entry,) = brief["largest_headroom"]
    assert entry["idle_tuning_experiments"][0]["delta_pct"] == -33.33
    other = PB.attach({"submission_sha256": "d0", "selfcheck_submission": "submission"}, rows, workspace=ws)
    assert U.attach(other, ws)["performance_brief"]["idle_tuning"]["experiments"] == []
    variant = PB.attach({"submission_sha256": "v1", "variant": {"sha256": "v1"}}, rows, workspace=ws)
    assert "idle_tuning" not in U.attach(variant, ws)["performance_brief"]
    assert U.attach({"performance_brief": "broken"}, ws) == {"performance_brief": "broken"}


def test_variant_mechanisms_come_from_new_imports_in_changed_files(tmp_path):
    submission, variant = tmp_path / "submission", tmp_path / "variant"
    for root in (submission, variant):
        (root / "lib").mkdir(parents=True)
        (root / "lib" / "same.py").write_text("from lib.copies import merge\n")
    (submission / "backend.py").write_text("from lib.copies import merge\n")
    (variant / "backend.py").write_text("from lib.copies import merge\nfrom lib import readout\n")
    selection = {"all_mechanisms": CATALOGUE}
    assert U.changed_mechanisms(submission, variant, selection) == ["readout"]
    assert U.changed_mechanisms(submission, submission, selection) == []


def test_a_failing_tuning_step_never_stops_baseline_timing(tmp_path, gates, monkeypatch):
    ws, run = _workspace(tmp_path)
    measurer, tick = _measurer(ws, run, {"submission": "d1"})

    def broken(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(measurer.tuner, "refresh", broken)
    assert tick() == "submitted"  # the baseline timing request still goes out
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    monkeypatch.setattr(measurer.tuner, "tick", broken)
    assert tick() == "error"


@pytest.mark.parametrize('kind', ['knob', 'variant'])
def test_new_model_renews_candidate_and_per_source_budget(tmp_path, gates, monkeypatch, kind):
    ws, run = _workspace(tmp_path, cohort=('big',))
    trees = {'submission': 'd1'}
    if kind == 'knob':
        _knobs(ws)
    else:
        _variant(ws)
        trees['variants/alt'] = 'v1'
    monkeypatch.setenv('MERLIN_IDLE_TUNE_MAX_PER_SOURCE', '1')
    _baseline(ws, 'd1', {'big': 900})
    _, tick = _measurer(ws, run, trees)
    assert tick() == 'tuning submitted'
    ((rid, body),) = _requests(ws).items()
    _answer(ws, rid, body, {'big': 800})
    assert tick() == 'tuning budget for this source used'
    _baseline(ws, 'd1', {'big': 700}, rid='3_2_new_model')
    path = ws / '.qa_channel' / 'resp_3_2_new_model.json'
    doc = json.loads(path.read_text())
    doc['graded_at_unix_ns'] = 20
    doc['cases'][0]['performance'].update(model_sha256='m2', timing_epoch=1)
    path.write_text(json.dumps(doc))
    assert tick() == 'tuning submitted'
    requests = list(_requests(ws).values())
    assert len(requests) == 2
    assert requests[0]['tuning_experiment']['identity'] != requests[1]['tuning_experiment']['identity']
    assert all(row['tuning_experiment']['base_sha256'] == 'd1' for row in requests)


@pytest.mark.parametrize('field', ['selection_sha256', 'scope_sha256', 'comparison_domain_sha256'])
def test_idle_comparison_rejects_different_selection_scope_or_domain(field):
    before = _performance(100)
    after = {**before, field: 'changed'}
    assert field in U._comparable(before, after)


def test_old_inflight_response_cannot_display_new_model_tuning_results(tmp_path, gates):
    ws, run = _workspace(tmp_path, cohort=('big',))
    _variant(ws)
    _baseline(ws, 'd1', {'big': 900})
    _, tick = _measurer(ws, run, {'submission': 'd1', 'variants/alt': 'v1'})
    assert tick() == 'tuning submitted'
    ((rid, body),) = _requests(ws).items()
    _answer(ws, rid, body, {'big': 800})
    tick()
    rows = [dict(capsule='big', tiers={'L2': 'pass'}, performance=_performance(
        100, model={**MODEL, 'model_sha256': 'older-model'}))]
    response = PB.attach(dict(submission_sha256='d1', cases=rows), rows, workspace=ws)
    U.attach(response, ws)
    assert response['performance_brief']['idle_tuning']['experiments'] == []
    assert not response['performance_brief']['largest_headroom'][0].get('idle_tuning_experiments')
