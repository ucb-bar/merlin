"""Idle reserved CPUs refresh the performance cohort's timing estimates, behind every other kind of work."""

import json

import pytest
from merlin_experiments.phase1.feedback import idle_performance as I
from merlin_experiments.phase1.feedback import performance_headroom as H
from merlin_experiments.phase1.feedback import scratch_retention


def _run(tmp_path, cohort=("big", "small", "never"), passing=("big", "small", "never")):
    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / ".qa_channel").mkdir(parents=True)
    (ws / "submission").mkdir()
    (run / "qa_history").mkdir(parents=True)
    verdict = {"joint_test_selection": {"performance_cohorts": {"primary": list(cohort)}},
               "per_capsule": [{"capsule": n, "tiers": {"L2": "pass" if n in passing else "fail"}} for n in cohort]}
    (run / "qa_history" / "verdict_fast_1.json").write_text(json.dumps(verdict))
    return ws, run


def _response(ws, rid, digest, cycles, *, variant=False):
    cases = [{"capsule": name, "performance": {"estimated_whole_soc_cycles": c,
                                               "structural_roofline": {"structural_compute_floor_cycles": 10}}}
             for name, c in cycles.items()]
    (ws / ".qa_channel" / f"resp_{rid}.json").write_text(json.dumps(
        {"submission_sha256": digest, "graded_at_unix_ns": len(rid), "variant": variant,
         "selfcheck_submission": "submission", "cases": cases}))


@pytest.fixture
def open_gates(monkeypatch):
    for name in (I.ENV, "MERLIN_SELFCHECK_BROKER_JOBS", *I.KNOBS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 20)
    monkeypatch.setattr(scratch_retention, "free_space_ok", lambda *a, **k: True)


def _measurer(ws, run, digest="d1"):
    clock = [1000.0]
    measurer = I.IdleMeasurer(ws, run, digest=lambda _root: digest, clock=lambda: clock[0])
    return measurer, clock


def test_gates_only_avoid_adding_work(tmp_path, monkeypatch, open_gates):
    ws, run = _run(tmp_path)
    measurer, clock = _measurer(ws, run)
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 2)  # all within an admitting grade's reserve
    assert measurer.tick(grading=True) == "cpus reserved for the admitting grade"
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 20)
    clock[0] += I.knob("MERLIN_IDLE_PERF_CHECK_S")
    (ws / ".qa_channel" / "req_3_1_a.json").write_text(json.dumps({"deadline_unix_ns": 2**62}))
    assert measurer.tick(grading=False) == "author request waiting"
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 0)
    (ws / ".qa_channel" / "claimed_3_1_a").write_text("{}")  # a running author check is not waiting
    clock[0] += I.knob("MERLIN_IDLE_PERF_CHECK_S")
    assert measurer.tick(grading=False) == "cpus busy"
    monkeypatch.setattr(I, "free_slots", lambda environ=None: None)
    clock[0] += I.knob("MERLIN_IDLE_PERF_CHECK_S")
    assert measurer.tick(grading=False) == "single-lane broker"
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 20)
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "1")
    clock[0] += I.knob("MERLIN_IDLE_PERF_CHECK_S")
    assert measurer.tick(grading=False) == "single-lane broker"
    monkeypatch.delenv("MERLIN_SELFCHECK_BROKER_JOBS")
    monkeypatch.setattr(scratch_retention, "free_space_ok", lambda *a, **k: False)
    clock[0] += I.knob("MERLIN_IDLE_PERF_CHECK_S")
    assert measurer.tick(grading=False) == "low disk"
    monkeypatch.setenv(I.ENV, "0")
    assert measurer.tick(grading=False) == "disabled"
    assert not list((ws / ".qa_channel").glob("req_op_idle_*"))


def test_a_request_is_a_labelled_detached_scalability_check_at_the_lowest_useful_class(tmp_path, open_gates):
    from merlin_experiments.phase1.brokers import selfcheck as broker

    ws, run = _run(tmp_path)
    measurer, _ = _measurer(ws, run)
    assert measurer.tick(grading=False) == "submitted"
    (request,) = (ws / ".qa_channel").glob("req_op_idle_*.json")
    body = json.loads(request.read_text())
    rid = request.name[len("req_"):-len(".json")]
    assert body["request_id"] == rid and body["origin"] == I.ORIGIN == broker.IDLE_ORIGIN
    assert rid.startswith(broker.IDLE_REQUEST_PREFIX)
    assert body["suite"] == "scalability" and body["sim"] == "spike" and body["submission_sha256"] == "d1"
    assert (ws / ".qa_channel" / f"detached_{rid}").exists()
    assert broker._requester_alive(body) is None  # its deadline is its lease


def test_work_is_sized_from_free_capacity(tmp_path, monkeypatch, open_gates):
    names = tuple(f"c{i}" for i in range(12))
    ws, run = _run(tmp_path, cohort=names, passing=names)
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 8)
    measurer, clock = _measurer(ws, run)
    statuses = []
    for _ in range(4):
        statuses.append(measurer.tick(grading=False))
        clock[0] += I.knob("MERLIN_IDLE_PERF_INTERVAL_S")
    requests = [json.loads(p.read_text()) for p in (ws / ".qa_channel").glob("req_op_idle_*.json")]
    # 8 free slots: 4 cases per request (the cap) and at most 8 // 4 = 2 requests in flight.
    assert statuses.count("submitted") == 2 and statuses[-1] == "background requests in flight"
    assert all(len(r["capsules"].split(",")) == 4 for r in requests)
    monkeypatch.setenv("MERLIN_IDLE_PERF_MAX_CASES", "2")
    assert I.knob("MERLIN_IDLE_PERF_MAX_CASES") == 2
    monkeypatch.setenv("MERLIN_IDLE_PERF_MAX_CASES", "nonsense")
    assert I.knob("MERLIN_IDLE_PERF_MAX_CASES") == I.KNOBS["MERLIN_IDLE_PERF_MAX_CASES"]


def test_only_cases_without_an_estimate_of_the_current_source_are_requested(tmp_path, open_gates):
    ws, run = _run(tmp_path, cohort=("big", "small", "fresh", "never", "failing"),
                   passing=("big", "small", "fresh", "never"))
    _response(ws, "3_9_x", "old", {"big": 900, "small": 50})
    _response(ws, "3_99_y", "d1", {"fresh": 70})
    _response(ws, "3_999_z", "d1", {"never": 5}, variant=True)  # a variant never describes submission/
    known = I.estimates(ws / ".qa_channel")
    cohort, l2 = I._cohort_l2(I._verdicts(run))
    chosen = I.select(cohort, l2, known, "d1", set(), limit=3)
    assert chosen == ["never", "big", "small"]  # untimed and stale alternate; stale largest first
    assert I.select(cohort, l2, known, "d1", {("never", "d1"), ("big", "d1")}, limit=3) == ["small"]


def test_headroom_uses_self_check_estimates_and_marks_older_source(tmp_path):
    ws, _ = _run(tmp_path)
    _response(ws, "3_9_x", "old", {"big": 900})
    _response(ws, "3_99_y", "new", {"small": 50})
    verdict = {"per_capsule": [{"capsule": "big", "tiers": {"L2": "pass"}},
                               {"capsule": "small", "tiers": {"L2": "pass"}}]}
    block = H.build(ws, verdict)
    by_name = {e["test_case"]: e for e in block["largest_headroom"]}
    assert by_name["big"]["evidence"].startswith("modeled") and by_name["big"]["measured_on_earlier_source"]
    assert "measured_on_earlier_source" not in by_name["small"]


def test_a_joined_certificate_for_the_current_executable_is_current(tmp_path):
    ws, _ = _run(tmp_path)
    (ws / "qa").mkdir()
    row = {"capsule": "big", "barrier_cycles": 400, "execution_digest": "e_old",
           "performance": {"structural_roofline": {"structural_compute_floor_cycles": 100}}}
    (ws / ".qa_channel" / "simresp_1.json").write_text(json.dumps({"sim": "gsim", "per_capsule": [row]}))
    l2 = {"status": "pass", "execution_digest": "e_new", "components": {"c": "2"}}
    l3 = {"status": "pass", "execution_digest": "e_old", "components": {"c": "1"}}
    state = {"big": {"L2": l2, "L3": l3, "<certs>": {"L3": {
        "e_old": l3, "e_new": {"status": "joined", "execution_digest": "e_new"}}}}}
    (ws / "qa" / "tier_state.json").write_text(json.dumps(state))
    (entry,) = H.build(ws, {"per_capsule": [{"capsule": "big", "tiers": {"L2": "pass"}}]})["largest_headroom"]
    assert entry["evidence"].startswith("measured") and "measured_on_earlier_source" not in entry


def test_an_admitting_grade_keeps_its_reserve_but_does_not_stop_measurement(tmp_path, monkeypatch, open_gates):
    """An editing author keeps some grade admitting almost always; free slots beyond the reserve are used."""
    names = tuple(f"c{i}" for i in range(12))
    ws, run = _run(tmp_path, cohort=names, passing=names)
    monkeypatch.setattr(I, "pool_size", lambda environ=None: 20)
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 7)
    measurer, _ = _measurer(ws, run)
    assert I.grade_reserve() == 4  # a fifth of a 20-slot pool
    assert measurer.tick(grading=True) == "submitted"
    (request,) = (ws / ".qa_channel").glob("req_op_idle_*.json")
    assert len(json.loads(request.read_text())["capsules"].split(",")) == 3  # 7 free - 4 reserved
    monkeypatch.setenv("MERLIN_IDLE_PERF_GRADE_RESERVE_SLOTS", "7")
    assert I.grade_reserve() == 7


def test_new_model_refreshes_same_compiler_after_a_previous_request(tmp_path, monkeypatch, open_gates):
    ws, run = _run(tmp_path, cohort=('big',), passing=('big',))
    selected = ['model-selection-1']
    monkeypatch.setattr(I, '_selection_identity', lambda: selected[0])
    measurer, clock = _measurer(ws, run)
    assert measurer.tick(grading=False) == 'submitted'
    (request,) = (ws / '.qa_channel').glob('req_op_idle_*.json')
    rid = request.name[len('req_'):-len('.json')]
    _response(ws, rid, 'd1', {'big': 100})
    answer = ws / '.qa_channel' / f'resp_{rid}.json'
    doc = json.loads(answer.read_text())
    doc['cases'][0]['performance']['selection_sha256'] = selected[0]
    answer.write_text(json.dumps(doc))
    assert I.estimates(ws / '.qa_channel', selection_sha256=selected[0])['big']
    selected[0] = 'model-selection-2'
    assert I.estimates(ws / '.qa_channel', selection_sha256=selected[0]) == {}
    clock[0] += I.knob('MERLIN_IDLE_PERF_INTERVAL_S')
    assert measurer.tick(grading=False) == 'submitted'
    assert len(list((ws / '.qa_channel').glob('req_op_idle_*.json'))) == 2
    assert json.loads(answer.read_text()) == doc  # retained old-model evidence is immutable


def test_idle_model_identity_reads_latest_owner_selection_without_environment_mutation(monkeypatch):
    from merlin_experiments.phase1 import joint_control, joint_model_activation
    from merlin_experiments.phase1.feedback import joint_performance as P

    pin = dict(path='/owner/current-selection.json', sha256='a' * 64)
    monkeypatch.setenv(joint_control.ENV, '/owner/campaign.json')
    monkeypatch.setenv(P.ENV, json.dumps(dict(path='/owner/initial.json', sha256='b' * 64)))
    monkeypatch.setattr(joint_model_activation, 'current', lambda path: pin)
    seen = []
    monkeypatch.setattr(P, '_read', lambda selected: seen.append(selected))
    initial = P.environment()
    assert I._selection_identity() == pin['sha256']
    assert seen == [pin] and P.environment() == initial
