"""Idle reserved CPUs refresh the performance cohort's timing estimates, behind every other kind of work."""

import json
import os
import time

import pytest
from merlin_experiments.phase1.feedback import idle_performance as I
from merlin_experiments.phase1.feedback import performance_headroom as H
from merlin_experiments.phase1.feedback import scratch_retention


def _run(tmp_path, cohort=("big", "small", "never"), passing=("big", "small", "never")):
    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / ".qa_channel").mkdir(parents=True)
    (ws / "submission").mkdir()
    (run / "qa_history").mkdir(parents=True)
    roles = {name: 'large_kernel' for name in cohort}
    (ws / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {name: {'workload_role': role, 'selfcheck_suite': 'scalability'}
                    for name, role in roles.items()}}))
    verdict = {"joint_test_selection": {"performance_cohorts": {"primary": list(cohort)}, 'workload_roles': roles},
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
    # A queued author request is served first and keeps the CPUs it is about to take (here: all).
    assert measurer.tick(grading=False) == "cpus promised to queued requests"
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 0)
    (ws / ".qa_channel" / "claimed_3_1_a").write_text("{}")  # a running author check holds its own CPUs
    (ws / ".qa_channel" / "progress_3_1_a.json").write_text("{}")
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

    def requests():
        return [json.loads(p.read_text()) for p in (ws / ".qa_channel").glob("req_op_idle_*.json")]

    # 8 free CPUs: requests of 4 cases (the cap) until they are spoken for, in one tick.
    assert measurer.tick(grading=False) == "submitted"
    assert sorted(len(r["capsules"].split(",")) for r in requests()) == [4, 4]
    clock[0] += I.knob("MERLIN_IDLE_PERF_INTERVAL_S")
    assert measurer.tick(grading=False) == "cpus promised to queued requests"
    for body in requests():  # the broker started them: their handlers hold their own leases now
        (ws / ".qa_channel" / f"claimed_{body['request_id']}").write_text("{}")
        (ws / ".qa_channel" / f"progress_{body['request_id']}.json").write_text("{}")
    clock[0] += I.knob("MERLIN_IDLE_PERF_INTERVAL_S")
    assert measurer.tick(grading=False) == "submitted"
    assert sorted(len(r["capsules"].split(",")) for r in requests()) == [4, 4, 4]
    monkeypatch.setenv("MERLIN_IDLE_PERF_MAX_REQUESTS", "3")  # an optional operator cap
    clock[0] += I.knob("MERLIN_IDLE_PERF_INTERVAL_S")
    assert measurer.tick(grading=False) == "background requests in flight"
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


def _declared_dense(ws, dimensions):
    path = ws / "PUBLIC_CAPABILITIES.json"
    document = json.loads(path.read_text())
    for name, (m, k, n) in dimensions.items():
        document["members"][name].update(declared_operation="matmul", family="contraction",
            selfcheck_suite="scalability", input_tensors=[
            {"name": "A", "role": "input", "shape": [m, k], "dtype": "i8"},
            {"name": "B", "role": "weight", "shape": [k, n], "dtype": "i8"}])
    path.write_text(json.dumps(document))


def test_largest_dense_slot_uses_declared_work_not_case_names_or_cohort_order(tmp_path):
    ws, _ = _run(tmp_path, cohort=("dense_largest", "z", "a", "missing"))
    _declared_dense(ws, {"dense_largest": (2, 2, 2), "z": (64, 64, 64), "a": (32, 32, 32)})
    # A very large noncontraction is not a dense target, regardless of its name.
    document = json.loads((ws / "PUBLIC_CAPABILITIES.json").read_text())
    document["members"]["dense_largest"].update(declared_operation="relu", family="elementwise_map")
    (ws / "PUBLIC_CAPABILITIES.json").write_text(json.dumps(document))
    assert I._largest_dense(ws, document["members"]) == ["z", "a"]


def test_largest_dense_gets_sole_idle_slot_before_gsim_and_tuning(tmp_path, monkeypatch, open_gates):
    ws, run = _run(tmp_path, cohort=("small", "large"), passing=("small", "large"))
    _declared_dense(ws, {"small": (2, 2, 2), "large": (1024, 1024, 1024)})
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 1)
    measurer, _ = _measurer(ws, run)
    measurer._timing_turn = False  # ordinary admission would try a tuning experiment first
    offered = []
    monkeypatch.setattr(measurer.gsim, "tick", lambda *a, **k: offered.append(k["free"]))
    monkeypatch.setattr(measurer.tuner, "refresh", lambda *a, **k: None)
    monkeypatch.setattr(measurer.tuner, "plan", lambda *a, **k: [{"must_not_run_before_baseline": True}])
    assert measurer.tick(grading=False) == "submitted"
    (path,) = (ws / ".qa_channel").glob("req_op_idle_*.json")
    body = json.loads(path.read_text())
    assert body["capsules"] == "large" and body["sim"] == "spike" and body["origin"] == I.ORIGIN
    assert body["timeout"] == I.knob("MERLIN_IDLE_PERF_TIMEOUT_S")
    assert offered == [0]  # gSIM still uses its original policy and only remaining free capacity


def test_early_allowance_survives_source_edits_then_normal_coverage_resumes(tmp_path, monkeypatch, open_gates):
    names = ("small", "medium", "largest", "second")
    ws, run = _run(tmp_path, cohort=names, passing=names)
    _declared_dense(ws, {"small": (2, 2, 2), "medium": (4, 4, 4),
                         "largest": (1024, 1024, 1024), "second": (512, 512, 512)})
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 1)
    selected, digest = ["model-1"], ["d1"]
    monkeypatch.setattr(I, "_selection_identity", lambda: selected[0])
    measurer, clock = _measurer(ws, run)
    measurer._digest = lambda _root: digest[0]
    offered = []
    monkeypatch.setattr(measurer.gsim, "tick", lambda *a, **k: offered.append(k["free"]))
    monkeypatch.setattr(measurer.tuner, "refresh", lambda *a, **k: None)
    monkeypatch.setattr(measurer.tuner, "plan", lambda *a, **k: [])

    def tick():
        before = set((ws / ".qa_channel").glob("req_op_idle_*.json"))
        assert measurer.tick(grading=False) == "submitted"
        (path,) = set((ws / ".qa_channel").glob("req_op_idle_*.json")) - before
        body = json.loads(path.read_text())
        # A refusal/no timing result still spends the finite early allowance.
        (path.parent / ("resp_" + body["request_id"] + ".json")).write_text("{}")
        clock[0] += I.knob("MERLIN_IDLE_PERF_INTERVAL_S")
        digest[0] += "x"
        return body["capsules"]

    # Normal coverage then times the untimed cases largest declared work first (every edit makes all
    # of them untimed again here: no response carried an estimate).
    assert [tick(), tick(), tick()] == ["largest", "second", "largest"]
    assert offered == [0, 0, 1]  # the gSIM lane's normal opportunity resumes too
    assert measurer._early_dense_attempted == {"largest", "second"}
    # A newly selected timing epoch gets its own bounded first-evidence allowance.
    selected[0] = "model-2"
    assert tick() == "largest"


@pytest.mark.parametrize("numerical_pass", [True, False])
def test_early_dense_slot_requires_l2_and_does_not_credit_failed_timing(tmp_path, monkeypatch, open_gates,
                                                                      numerical_pass):
    names = ("small", "largest", "second")
    ws, run = _run(tmp_path, cohort=names, passing=("small", "second"))
    _declared_dense(ws, {"small": (2, 2, 2), "largest": (1024, 1024, 1024), "second": (512, 512, 512)})
    _response(ws, "previous", "old", {"second": 1000})
    response = ws / ".qa_channel/resp_previous.json"
    document = json.loads(response.read_text())
    document["cases"][0]["numerical_execution_passed"] = numerical_pass
    response.write_text(json.dumps(document))
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 1)
    measurer, _ = _measurer(ws, run)
    monkeypatch.setattr(measurer.gsim, "tick", lambda *a, **k: None)
    assert measurer.tick(grading=False) == "submitted"
    (path,) = (ws / ".qa_channel").glob("req_op_idle_*.json")
    assert json.loads(path.read_text())["capsules"] == ("small" if numerical_pass else "second")
    assert measurer._early_dense_attempted == (set() if numerical_pass else {"second"})


def _completed_l2(ws, rid, *, status="pass", complete=True, stamp=1):
    from merlin_experiments.phase1.brokers.selfcheck import _submission_digest

    channel = ws / ".qa_channel"
    digest = _submission_digest(ws / "submission")
    (channel / f"req_{rid}.json").write_text(json.dumps(dict(submission="submission")))
    (channel / f"resp_{rid}.json").write_text(json.dumps(dict(
        selfcheck_request_id=rid, selfcheck_submission="submission", submission_sha256=digest,
        graded_at_unix_ns=stamp, per_capsule=[dict(capsule="large", tiers=dict(L2=status))])))
    if complete:
        (channel / f"done_{rid}").write_text("ok")
    return digest


def test_selfcheck_only_readiness_invalidates_both_idle_caches_and_source_edits_remove_it(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import family_rollup

    ws, run = _run(tmp_path, cohort=("small", "large"), passing=())
    _declared_dense(ws, {"small": (2, 2, 2), "large": (1024, 1024, 1024)})
    (ws / "submission/backend.py").write_text("import json\n")
    (run / "qa_history/verdict_fast_1.json").unlink()
    monkeypatch.setattr(family_rollup, "_edge", lambda: 16)
    measurer, _ = _measurer(ws, run)

    def readiness():
        cohort, l2 = I._cohort_l2([], workspace=ws)
        return l2, measurer._preferred(cohort), measurer.tuner._rollup_bundles(cohort, l2, {})

    _completed_l2(ws, "new", complete=False)
    assert readiness() == ({}, frozenset(), [])
    # No grade or source changed: only the completed channel join becomes available.
    (ws / ".qa_channel/done_new").write_text("ok")
    l2, preferred, bundles = readiness()
    assert l2 == {"large": "pass"} and preferred == {"large"}
    assert bundles[0]["representative"] == "large" and bundles[0]["label"] != "largest headroom"
    _completed_l2(ws, "failure", status="fail", stamp=2)
    assert readiness() == ({"large": "fail"}, frozenset(), [])
    (ws / "submission/backend.py").write_text("import math\n")
    assert readiness() == ({}, frozenset(), [])


def test_current_selfcheck_gets_early_slot_without_any_grade_and_only_on_public_scalability(tmp_path, monkeypatch,
                                                                                        open_gates):
    ws, run = _run(tmp_path, cohort=("small", "large", "core"), passing=())
    _declared_dense(ws, {"small": (2, 2, 2), "large": (1024, 1024, 1024), "core": (2048, 2048, 2048)})
    (ws / "submission/backend.py").write_text("import json\n")
    (run / "qa_history/verdict_fast_1.json").unlink()
    inventory = ws / "PUBLIC_CAPABILITIES.json"
    document = json.loads(inventory.read_text())
    document["members"]["core"]["selfcheck_suite"] = "core"
    inventory.write_text(json.dumps(document))
    digest = _completed_l2(ws, "baseline")
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 1)
    measurer, clock = _measurer(ws, run, digest=digest)
    monkeypatch.setattr(measurer.gsim, "tick", lambda *a, **k: None)
    assert measurer.tick(grading=False) == "submitted"
    (path,) = (ws / ".qa_channel").glob("req_op_idle_*.json")
    body = json.loads(path.read_text())
    assert body["capsules"] == "large" and body["submission_sha256"] == digest
    assert I._cohort_l2([], workspace=ws)[0] == ["small", "large"]
    (path.parent / f"resp_{body['request_id']}.json").write_text("{}")
    (ws / "submission/backend.py").write_text("import math\n")
    clock[0] += I.knob("MERLIN_IDLE_PERF_INTERVAL_S")
    measurer.tick(grading=False)
    assert len(list((ws / ".qa_channel").glob("req_op_idle_*.json"))) == 1


def _author_request(ws, rid, capsules, *, claimed_s_ago=None):
    channel = ws / ".qa_channel"
    (channel / f"req_{rid}.json").write_text(json.dumps(
        {"deadline_unix_ns": 2**62, "capsules": capsules, "submission": "variants/v", "suite": "scalability"}))
    if claimed_s_ago is not None:
        marker = channel / f"claimed_{rid}"
        marker.write_text("{}")
        at = time.time() - claimed_s_ago
        os.utime(marker, (at, at))


def test_a_long_running_author_check_without_progress_does_not_keep_idle_cpus_shut(tmp_path, monkeypatch,
                                                                                   open_gates):
    # Measured on two live runs: variant checks publish no progress record while they run, so counting
    # each claimed one as "about to lease" (26 and 2**20 CPUs on 20-CPU pools) shut this lane for hours.
    names = tuple(f"c{i}" for i in range(6))
    ws, run = _run(tmp_path, cohort=names, passing=names)
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 8)
    _author_request(ws, "3_1_running", ",".join(f"x{i}" for i in range(13)), claimed_s_ago=600)
    _author_request(ws, "3_2_corpus", "all", claimed_s_ago=I.CLAIM_GRACE_S + 5)
    assert I.queued(ws / ".qa_channel")["author_cases"] == 0
    _author_request(ws, "3_3_starting", "x1,x2", claimed_s_ago=1)  # still starting: its CPUs are promised
    _author_request(ws, "3_4_waiting", "x3")  # not claimed yet
    assert I.queued(ws / ".qa_channel")["author_cases"] == 3
    measurer, _ = _measurer(ws, run)
    assert measurer.tick(grading=False) == "submitted"
    sizes = [len(json.loads(p.read_text())["capsules"].split(","))
             for p in (ws / ".qa_channel").glob("req_op_idle_*.json")]
    assert sum(sizes) == 8 - 3


def test_timing_yields_to_higher_priority_waiters_and_is_retried_when_cpus_are_idle(tmp_path, monkeypatch,
                                                                                  open_gates):
    names = tuple(f"c{i}" for i in range(8))
    ws, run = _run(tmp_path, cohort=names, passing=names)
    channel = ws / ".qa_channel"
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 8)
    measurer, clock = _measurer(ws, run)
    assert measurer.tick(grading=False) == "submitted"
    timing = sorted(channel.glob("req_op_idle_*.json"),
                    key=lambda p: json.loads(p.read_text())["requested_at_unix_ns"])
    assert len(timing) == 2
    others = []
    for origin in (I.TUNING_ORIGIN, I.GSIM_ORIGIN):  # their lanes keep their own rules
        rid = I.submit(channel, ["c0"], "d1", origin=origin, prefix=I.REQUEST_PREFIX, timeout=60,
                       extra={"sim": "gsim"} if origin == I.GSIM_ORIGIN else None)
        others.append(channel / f"req_{rid}.json")
    started = time.time() - I.CLAIM_GRACE_S - 1
    for path in [*timing, *others]:  # the broker started every request a while ago
        marker = channel / ("claimed_" + path.name[len("req_"):-len(".json")])
        marker.write_text("{}")
        os.utime(marker, (started, started))
    newest = json.loads(timing[1].read_text())["capsules"].split(",")
    contended = [3]  # a grade, an author check and a draining grade wait while no slot is free
    monkeypatch.setattr(I, "contention", lambda environ=None: contended[0])
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 0)
    assert measurer.tick(grading=True) == "yielded"
    assert not timing[1].exists() and timing[0].exists()  # the newest request covers the three waiters
    assert all(path.exists() for path in others)
    assert not any((name, "d1") in measurer.requested for name in newest)
    contended[0] = 0
    monkeypatch.setattr(I, "free_slots", lambda environ=None: 4)
    clock[0] += I.knob("MERLIN_IDLE_PERF_INTERVAL_S")
    assert measurer.tick(grading=False) == "submitted"
    retried = [json.loads(p.read_text()) for p in channel.glob("req_op_idle_*.json")
               if json.loads(p.read_text())["origin"] == I.ORIGIN and p != timing[0]]
    assert len(retried) == 1 and sorted(retried[0]["capsules"].split(",")) == sorted(newest)
    assert retried[0]["sim"] == "spike"  # timing never asks for gSIM


def test_contention_counts_only_waiters_ranked_above_idle_timing(tmp_path, monkeypatch):
    from merlin.common import cpu_pool

    keys = [(level, 1, 2, 3, "n") for level in (cpu_pool.AUTHOR, cpu_pool.GRADE, cpu_pool.PREPARATION,
                                                cpu_pool.AUXILIARY, cpu_pool.VERIFICATION)]
    free = [()]
    monkeypatch.setattr(cpu_pool, "_shared_compute_selection", lambda environ: ((0, 1), tmp_path))
    monkeypatch.setattr(cpu_pool, "_free_cpus", lambda root, cpus, fresh=False: free[0])
    monkeypatch.setattr(cpu_pool, "_queued_keys", lambda directory: keys)
    assert I.contention() == 3
    free[0] = (1,)  # a free slot goes to the waiters without withdrawing anything
    assert I.contention() == 0
    monkeypatch.setattr(cpu_pool, "_shared_compute_selection", lambda environ: None)
    assert I.contention() == 0


def test_untimed_cases_are_timed_largest_declared_work_first(tmp_path):
    names = ("small", "largest", "middle", "undeclared")
    ws, run = _run(tmp_path, cohort=names, passing=names)
    _declared_dense(ws, {"small": (2, 2, 2), "largest": (256, 256, 256), "middle": (16, 16, 16)})
    cohort, l2 = I._cohort_l2(I._verdicts(run))
    work = I._declared_work(ws, cohort)
    assert I.select(cohort, l2, {}, "d1", set(), limit=4, work=work) == ["largest", "middle", "small", "undeclared"]
