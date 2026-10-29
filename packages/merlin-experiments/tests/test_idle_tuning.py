"""Idle reserved CPUs A/B-time candidate changes on the passing bundles with the most headroom, as advice.

Measured on two runs: the tuner recorded no experiment at all although the authors built 38 variants,
because it waited until every case had an estimate of the current source (which never held while eight
authors edited) and dropped each result as soon as the source moved on. These tests hold the
replacement: experiments start beside baseline timing, are sized from free CPUs, keep their results
with a baseline label across edits, and every candidate the authors check is ranked.
"""

from __future__ import annotations

import json
import shutil
import time

import pytest
from merlin_experiments.phase1.brokers import selfcheck_tuning as T
from merlin_experiments.phase1.feedback import family_rollup, scratch_retention
from merlin_experiments.phase1.feedback import idle_performance as I
from merlin_experiments.phase1.feedback import idle_tuning as U
from merlin_experiments.phase1.feedback import idle_tuning_results as R
from merlin_experiments.phase1.feedback import performance_brief as PB

MODEL = {"model_sha256": "m1", "timing_mode": "untuned", "timing_epoch": 0}
CATALOGUE = [
    {"id": "readout", "family": "overlap", "api": "lib.readout", "symbol": "propose", "status": "supported",
     "requirements": {"memory.accumulator": "supported"}, "evidence": "accumulator readout"},
    {"id": "coalesce", "family": "movement", "api": "lib.copies", "symbol": "merge", "status": "supported",
     "requirements": {"memory.explicit_dma": "supported"}, "evidence": "transfer count"},
]


def _performance(cycles, *, floor=100, model=MODEL, elf="e1"):
    return {"status": "PROVISIONAL", "complete_event_accounting": True, "estimated_whole_soc_cycles": cycles,
            "structural_roofline": {"structural_compute_floor_cycles": floor, "useful_macs": 1000,
                                    "structural_peak_macs_per_cycle": 10},
            "scoped_resource_feedback": {"resources": [
                {"name": "accelerator.storage.accumulator.all", "capacity_normalized_activity": 0.9}]},
            "elf_sha256": elf, **model}


def _workspace(tmp_path, cohort=("big", "mid", "small")):
    ws, run = tmp_path / "ws", tmp_path / "run"
    (ws / ".qa_channel").mkdir(parents=True)
    (ws / "submission").mkdir()
    (ws / "submission" / "backend.py").write_text("import os\n")
    (ws / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps({"all_mechanisms": CATALOGUE}))
    (ws / "PUBLIC_CAPABILITIES.json").write_text(json.dumps({
        "schema": "merlin.public_capability_inventory.v1",
        "members": {name: {"workload_role": "large_kernel", "selfcheck_suite": "scalability"} for name in cohort},
    }))
    (run / "qa_history").mkdir(parents=True)
    verdict = {"joint_test_selection": {"performance_cohorts": {"primary": list(cohort)}},
               "per_capsule": [{"capsule": n, "tiers": {"L2": "pass"}} for n in cohort]}
    (run / "qa_history" / "verdict_fast_1.json").write_text(json.dumps(verdict))
    return ws, run


def _baseline(ws, digest, cycles, rid="3_1_base", graded=None, elf="e1"):
    cases = [{"capsule": n, "status": "pass", "tiers": {"L2": "pass"}, "performance": _performance(c, elf=elf)}
             for n, c in cycles.items()]
    (ws / ".qa_channel" / f"resp_{rid}.json").write_text(json.dumps(
        {"submission_sha256": digest, "graded_at_unix_ns": graded or time.time_ns(),
         "selfcheck_submission": "submission",
         "cases": cases}))


def _variant(ws, name="alt", text="from lib.readout import propose\n"):
    (ws / "variants" / name).mkdir(parents=True, exist_ok=True)
    (ws / "variants" / name / "backend.py").write_text(text)


def _knobs(ws, values=(False, True), current=False):
    (ws / "submission" / "tuning.json").write_text(json.dumps({"ready_readout": current}))
    (ws / "submission" / T.DECLARATION).write_text(json.dumps({
        "schema": T.SCHEMA, "settings_file": "tuning.json",
        "knobs": {"ready_readout": {"values": list(values), "mechanism": "readout"}}}))


def _bundles(ws, monkeypatch, bundles):
    """The family roll-up's bundles (representative and siblings), as of one completed grade."""
    (ws / "qa").mkdir(exist_ok=True)
    (ws / "qa" / "latest_completed_grade.json").write_text("{}")
    monkeypatch.setattr(family_rollup, "build", lambda workspace=None, **k: {"bundles": bundles})


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


def _requests(ws, prefix=U.REQUEST_PREFIX):
    return {p.name[len("req_"):-len(".json")]: json.loads(p.read_text())
            for p in (ws / ".qa_channel").glob(f"req_{prefix}*.json")}


def _timing(ws):
    return {rid: body for rid, body in _requests(ws, I.REQUEST_PREFIX).items() if not rid.startswith(U.REQUEST_PREFIX)}


def _running(ws):
    """The broker claimed every request and its handlers started (they now hold their own leases)."""
    for request in (ws / ".qa_channel").glob("req_*.json"):
        rid = request.name[len("req_"):-len(".json")]
        (ws / ".qa_channel" / f"claimed_{rid}").write_text("{}")
        (ws / ".qa_channel" / f"progress_{rid}.json").write_text("{}")


def _answer(ws, rid, body, cycles, *, passed=True, graded=None, model=MODEL, at=9):
    """The broker's response to one experiment request, as ``_decorate_response`` shapes it."""
    variant = body.get("submission", "submission") != "submission"
    sha = graded or (body["submission_sha256"] if variant else "copy-" + body["submission_sha256"])
    cases = [{"capsule": n, "status": "pass" if passed else "fail", "tiers": {"L2": "pass" if passed else "fail"},
              "performance": _performance(cycles[n], model=model, elf="cand")} for n in body["capsules"].split(",")]
    document = {"cases": cases, "selfcheck_request_id": rid, "graded_at_unix_ns": at,
                "submission_sha256": sha if variant else body["submission_sha256"],
                "selfcheck_submission": body.get("submission", "submission"), "official": False,
                "tuning_experiment": body["tuning_experiment"]}
    if variant:
        document["variant"] = {"name": body["submission"].partition("/")[2], "sha256": sha, "official": False}
    else:
        document["variant"] = {"name": None, "path": "submission", "kind": "knob_overrides", "sha256": sha,
                               "tuning_overrides": body[T.REQUEST_KEY], "settings_file": "tuning.json"}
    (ws / ".qa_channel" / f"resp_{rid}.json").write_text(json.dumps(document))


def _ledger(ws):
    return json.loads((ws / U.LEDGER).read_text())


def test_reference_requests_reuse_private_knob_trials_and_do_not_repeat_as_generic_candidates(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import reference_experiments

    submission = tmp_path / "submission"
    submission.mkdir()
    _knobs(tmp_path)
    request = dict(
        task_id="retention", request_id="controlled", packet_identity="a" * 64, context_identity="b" * 64,
        role="variant", status="ready", split="training", sim="spike", tiers=None,
        broker_payload=dict(suite="scalability"), tuning_overrides=dict(ready_readout=True),
        settings=dict(ready_readout=True), capsule="public-case", owner_observed=False,
    )
    monkeypatch.setattr(reference_experiments, "pending_requests", lambda *a, **k: (request,))
    tuner = U.IdleTuner(tmp_path)
    proposed = tuner.candidates("a" * 64, None)
    assert len(proposed) == 1 and proposed[0]["overrides"] == {"ready_readout": True}
    assert proposed[0]["reference_experiment"]["request_id"] == "controlled"
    request["owner_observed"] = True
    assert tuner.candidates("a" * 64, None) == []


@pytest.mark.parametrize("change", [
    dict(split="held_out"), dict(sim="gsim"), dict(broker_payload=dict(suite="core")),
    dict(tuning_overrides=dict(undeclared=True)), dict(settings=dict(ready_readout=True, unrelated="changed")),
])
def test_reference_request_cannot_change_idle_lane_or_undeclared_settings(tmp_path, monkeypatch, change):
    from merlin_experiments.phase1.feedback import reference_experiments

    (tmp_path / "submission").mkdir()
    _knobs(tmp_path)
    request = dict(
        task_id="retention", request_id="controlled", packet_identity="a" * 64, context_identity="b" * 64,
        role="variant", status="ready", split="training", sim="spike", tiers=None,
        broker_payload=dict(suite="scalability"), tuning_overrides=dict(ready_readout=True),
        settings=dict(ready_readout=True), capsule="public-case", owner_observed=False,
    ) | change
    monkeypatch.setattr(reference_experiments, "pending_requests", lambda *a, **k: (request,))
    proposed = U.IdleTuner(tmp_path).candidates("a" * 64, None)
    assert len(proposed) == 1 and "reference_experiment" not in proposed[0]


def test_pending_or_refused_library_pass_variants_are_not_discovered(tmp_path):
    for name in ('ready', 'pending', 'refused'):
        _variant(tmp_path, name)
    root = tmp_path / 'variants'
    (root / (U.LIBRARY_PASS_PENDING_PREFIX + 'pending')).write_text('building')
    (root / (U.LIBRARY_PASS_PENDING_PREFIX + 'refused')).symlink_to('missing-status-file')
    assert U._variant_names(tmp_path, 8) == ['ready']


def test_experiments_start_while_some_cases_lack_estimates(tmp_path, gates):
    """The old tuner waited until every passing case had a current estimate, which never held."""
    ws, run = _workspace(tmp_path)
    _variant(ws)
    trees = {"submission": "d1", "variants/alt": "v1"}
    _baseline(ws, "d0", {"big": 900})  # one case timed on older source; "mid" and "small" never timed
    gates["slots"] = 12
    measurer, tick = _measurer(ws, run, trees)
    status = tick()
    assert "submitted" in status and "tuning submitted" in status
    ((rid, body),) = _requests(ws).items()
    assert body["submission"] == "variants/alt" and body["submission_sha256"] == "v1"
    assert body["origin"] == U.ORIGIN == I.TUNING_ORIGIN and rid.startswith(I.REQUEST_PREFIX)
    assert body["suite"] == "scalability" and body["sim"] == "spike"
    assert body["capsules"] == "big,mid,small"  # the bundle: representative first, then siblings
    experiment = body["tuning_experiment"]
    assert experiment["base_sha256"] == "d1" and experiment["mechanism_ids"] == ["readout"]
    assert experiment["api"] == "lib.readout:propose" and experiment["origin"] == "idle_tuning"
    # The baselines the experiment needs were timed beside it, and the timing lane took the rest.
    timed = sorted(name for body in _timing(ws).values() for name in body["capsules"].split(","))
    assert timed == ["big", "mid", "small"]
    assert (ws / ".qa_channel" / f"detached_{rid}").exists()


def test_in_flight_work_is_sized_from_free_cpus(tmp_path, gates, monkeypatch):
    ws, run = _workspace(tmp_path)
    for name in ("a", "b", "c", "d"):
        _variant(ws, name, f"value = {name!r}\n")
    trees = {"submission": "d1", **{f"variants/{n}": f"v{n}" for n in "abcd"}}
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    gates["slots"] = 7
    measurer, tick = _measurer(ws, run, trees)
    assert tick() == "tuning submitted"
    # Seven free CPUs: two experiments of three cases and one of the representative alone.
    sizes = sorted(len(b["capsules"].split(",")) for b in _requests(ws).values())
    assert sizes == [1, 3, 3]
    assert tick() == "cpus promised to queued requests"  # queued requests own the CPUs they will take
    _running(ws)
    gates["slots"] = 3
    assert tick() == "tuning submitted"
    assert len(_requests(ws)) == 5  # finish b's two siblings, then start a in the remaining slot
    _running(ws)
    assert tick() == "tuning submitted"  # a's two siblings are still owed, even after its representative ran
    assert sum(len(body["capsules"].split(",")) for body in _requests(ws).values()) == 12
    _running(ws)
    assert tick() == "estimates current; no untried tuning candidate"
    # The caps are optional: an operator may still bound experiments in flight.
    monkeypatch.setenv("MERLIN_IDLE_PERF_MAX_REQUESTS", "2")
    assert tick() == "background requests in flight"


def test_author_requests_keep_their_cpus_without_stopping_background_work(tmp_path, gates):
    ws, run = _workspace(tmp_path)
    _variant(ws)
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    measurer, tick = _measurer(ws, run, {"submission": "d1", "variants/alt": "v1"})
    author = ws / ".qa_channel" / "req_3_5_author.json"
    author.write_text(json.dumps({"deadline_unix_ns": 2**62, "capsules": ",".join(f"c{i}" for i in range(20))}))
    gates["slots"] = 20
    assert tick() == "cpus promised to queued requests"  # all 20 free CPUs go to the author first
    author.write_text(json.dumps({"deadline_unix_ns": 2**62, "capsules": "c1,c2"}))
    assert tick() == "tuning submitted"  # the rest are used, not left idle
    (body,) = _requests(ws).values()
    assert body["origin"] == U.ORIGIN and len(body["capsules"].split(",")) == 3
    author.write_text(json.dumps({"deadline_unix_ns": 2**62, "capsules": "all"}))
    assert tick() == "cpus promised to queued requests"


def test_small_pools_do_not_give_every_slot_to_missing_baselines(tmp_path, gates):
    ws, run = _workspace(tmp_path, cohort=tuple(f"case{i}" for i in range(20)))
    _variant(ws)
    gates["slots"] = 4
    _, tick = _measurer(ws, run, {"submission": "d1", "variants/alt": "v1"})
    status = tick()
    assert "submitted" in status and "tuning submitted" in status
    assert _timing(ws) and _requests(ws)
    assert sum(len(r["capsules"].split(",")) for r in [*_timing(ws).values(), *_requests(ws).values()]) <= 4


def test_results_survive_a_source_change_with_their_baseline_label(tmp_path, gates):
    ws, run = _workspace(tmp_path)
    _variant(ws)
    trees = {"submission": "d1", "variants/alt": "v1"}
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    measurer, tick = _measurer(ws, run, trees)
    assert "tuning submitted" in tick()
    ((rid, body),) = _requests(ws).items()
    _answer(ws, rid, body, {"big": 600, "mid": 550, "small": 200})
    _running(ws)
    tick()
    ledger = _ledger(ws)
    assert ledger["advisory_only"] is True and ledger["promotion_authority"] is False
    rows = {r["test_case"]: r for r in ledger["experiments"]}
    big = rows["big"]
    assert big["measured_cycles_before"] == 900 and big["measured_cycles_after"] == 600
    assert big["delta_pct"] == -33.33 and big["numerics"].startswith("unchanged")
    assert big["baseline_status"] == "current" and big["baseline"]["sha256"] == "d1"
    assert big["mechanism"] == "readout" and big["api"] == "lib.readout:propose"
    assert big["candidate"]["graded_sha256"] == "v1" and big["how_to_apply"].startswith("compare variants/alt/")
    assert rows["mid"]["delta_pct"] == 10.0  # a slower candidate is reported like a faster one
    assert (ws / "submission" / "backend.py").read_text() == "import os\n"  # nothing applied
    ranked = ledger["ranked_candidates"][0]
    assert ranked["candidate"] == "variants/alt" and ranked["faster"] == 1 and ranked["slower"] == 1
    assert ledger["ranked_cases"][0]["test_case"] == "big" and ledger["ranked_cases"][0]["best"]["delta_pct"] == -33.33

    trees["submission"] = "d2"  # the author edits: results stay, labelled as an older baseline
    tick()
    rows = {r["test_case"]: r for r in _ledger(ws)["experiments"]}
    assert len(rows) == 3 and rows["big"]["baseline_status"] == "older_source"
    assert rows["big"]["baseline"]["sha256"] == "d1" and rows["big"]["baseline"]["age_s"] is not None
    assert rows["big"]["delta_pct"] == -33.33
    # d2 re-timed: "big" emits the same ELF (still a valid comparison, now current); "mid" changed.
    _baseline(ws, "d2", {"big": 900}, rid="3_2_base")
    _baseline(ws, "d2", {"mid": 400}, rid="3_3_base", elf="e2")
    timed_v1 = [b for b in _requests(ws).values() if b["submission_sha256"] == "v1"]
    tick()
    rows = {r["test_case"]: r for r in _ledger(ws)["experiments"]}
    assert rows["big"]["baseline_status"] == "current" and rows["big"]["source_sha256"] == "d2"
    # A variant's estimate depends on its own bytes: it is re-compared with the newer baseline, not re-run.
    assert rows["mid"]["baseline_status"] == "current" and rows["mid"]["measured_cycles_before"] == 400
    assert rows["mid"]["delta_pct"] == 37.5 and rows["small"]["baseline_status"] == "older_source"
    assert [b for b in _requests(ws).values() if b["submission_sha256"] == "v1"] == timed_v1

    trees["variants/alt"] = "v2"  # the variant itself moved on: its old result stays, marked as such
    (ws / "variants" / "alt" / "backend.py").write_text("from lib.readout import propose\nchanged = 1\n")
    tick()
    rows = {r["test_case"]: r for r in _ledger(ws)["experiments"]}
    assert rows["big"]["candidate_current"] is False


def test_a_knob_copy_is_re_timed_only_when_the_case_emission_changes(tmp_path, gates, monkeypatch):
    ws, run = _workspace(tmp_path, cohort=("big",))
    _knobs(ws)
    _baseline(ws, "d1", {"big": 900})
    trees = {"submission": "d1"}
    measurer, tick = _measurer(ws, run, trees)
    assert tick() == "tuning submitted"
    ((rid, body),) = _requests(ws).items()
    assert body["submission"] == "submission" and body["submission_sha256"] == "d1"
    assert body[T.REQUEST_KEY] == {"ready_readout": True}
    _answer(ws, rid, body, {"big": 700})
    _running(ws)
    tick()
    (record,) = _ledger(ws)["experiments"]
    assert record["delta_pct"] == -22.22 and record["baseline_status"] == "current"
    assert record["how_to_apply"] == "set ready_readout to true in submission/tuning.json"
    trees["submission"] = "d2"
    _baseline(ws, "d2", {"big": 900}, rid="3_2_base")  # same emitted ELF for this case
    assert "tuning submitted" not in tick()
    (record,) = _ledger(ws)["experiments"]
    assert record["baseline_status"] == "current" and record["source_sha256"] == "d1"
    trees["submission"] = "d3"
    _baseline(ws, "d3", {"big": 800}, rid="3_3_base", elf="e3")  # the case's emission changed
    assert tick() == "tuning submitted"
    (record,) = _ledger(ws)["experiments"]
    assert record["baseline_status"] == "superseded"
    assert sorted(b["submission_sha256"] for b in _requests(ws).values()) == ["d1", "d3"]


def test_numerical_changes_and_incomparable_estimates_are_stated_not_compared(tmp_path, gates):
    ws, run = _workspace(tmp_path, cohort=("big",))
    _variant(ws)
    _baseline(ws, "d1", {"big": 900})
    measurer, tick = _measurer(ws, run, {"submission": "d1", "variants/alt": "v1"})
    assert tick() == "tuning submitted"
    ((rid, body),) = _requests(ws).items()
    _answer(ws, rid, body, {"big": 100}, passed=False)
    _running(ws)
    tick()
    (record,) = _ledger(ws)["experiments"]
    assert record["numerics"].startswith("changed") and record["delta_pct"] == -88.89
    assert not R.usable(record) and _ledger(ws)["ranked_cases"][0].get("best") is None
    _answer(ws, rid, body, {"big": 100}, model={**MODEL, "timing_epoch": 1})
    measurer.tuner._responses.clear()
    tick()
    (record,) = _ledger(ws)["experiments"]
    assert record["delta_pct"] is None and "comparable" in record["not_compared"]


def test_declared_knob_values_are_timed_on_a_private_copy(tmp_path, gates):
    ws, run = _workspace(tmp_path)
    _knobs(ws)
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    measurer, tick = _measurer(ws, run, {"submission": "d1"})
    assert tick() == "tuning submitted"
    ((rid, body),) = _requests(ws).items()
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


def test_author_variant_checks_are_ranked_and_not_timed_again(tmp_path, gates, monkeypatch):
    ws, run = _workspace(tmp_path)
    for name in ("mine", "other"):
        _variant(ws, name, f"value = {name!r}\n")
    _bundles(ws, monkeypatch, [{"label": "dense tall", "representative": "mid", "siblings": ["small"]},
                               {"label": "dense wide", "representative": "big", "siblings": []}])
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    # The author already checked variants/mine on "mid" with the timing model.
    author = {"request_id": "3_7_a", "submission": "variants/mine", "submission_sha256": "vm",
              "suite": "scalability", "capsules": "mid", "deadline_unix_ns": 2**62}
    (ws / ".qa_channel" / "req_3_7_a.json").write_text(json.dumps(author))
    _answer(ws, "3_7_a", {**author, "tuning_experiment": None}, {"mid": 450}, at=5)
    gates["slots"] = 3
    measurer, tick = _measurer(ws, run, {"submission": "d1", "variants/mine": "vm", "variants/other": "vo"})
    assert tick() == "tuning submitted"
    jobs = sorted((b["submission"], b["capsules"]) for b in _requests(ws).values())
    # Bundles in value order (newest variant first within one); "mine" is not timed again on "mid",
    # where the author already timed it; its still-unchecked sibling is retained.
    assert jobs == [("variants/mine", "small"), ("variants/other", "mid,small")]
    _running(ws)
    gates["slots"] = 1
    assert tick() == "tuning submitted"
    assert ("variants/other", "big") in [(b["submission"], b["capsules"]) for b in _requests(ws).values()]
    _running(ws)
    assert tick() == "tuning submitted"
    assert ("variants/mine", "big") in [(b["submission"], b["capsules"]) for b in _requests(ws).values()]
    rows = {(r["candidate"]["label"], r["test_case"]): r for r in _ledger(ws)["experiments"]}
    assert rows[("variants/mine", "mid")]["delta_pct"] == -10.0 and rows[("variants/mine", "mid")]["origin"] == "author"
    assert _ledger(ws)["bundles"][0] == {"label": "dense tall", "representative": "mid", "siblings": ["small"]}


def test_comparison_history_survives_baseline_replacement_and_restart(tmp_path, gates):
    ws, run = _workspace(tmp_path, cohort=("big",))
    _variant(ws)
    trees = {"submission": "d1", "variants/alt": "v1"}
    _baseline(ws, "d1", {"big": 900}, graded=1)
    _, tick = _measurer(ws, run, trees)
    assert tick() == "tuning submitted"
    ((rid, body),) = _requests(ws).items()
    _answer(ws, rid, body, {"big": 600}, at=2)
    tick()
    (old,) = _ledger(ws)["comparison_history"]
    trees["submission"] = "d2"
    _baseline(ws, "d2", {"big": 800}, rid="3_2_base", graded=3, elf="new-baseline")
    tick()
    ledger = _ledger(ws)
    assert len(ledger["experiments"]) == 1 and len(ledger["comparison_history"]) == 2
    history = {r["baseline"]["sha256"]: r for r in ledger["comparison_history"]}
    assert history["d1"]["identity"] == old["identity"]
    assert history["d1"]["baseline_status"] == "superseded"
    assert history["d2"]["baseline_status"] == "current"
    _, restarted = _measurer(ws, run, trees)
    restarted()
    assert {r["identity"] for r in _ledger(ws)["comparison_history"]} == {
        r["identity"] for r in ledger["comparison_history"]}


def test_candidate_model_epochs_and_knob_copies_have_separate_immutable_results():
    candidate = {"kind": "knob", "key": "knob:k=true", "overrides": {"k": True}, "settings_file": "t.json"}
    base = {"candidate": candidate, "base": "d1", "graded": "copy-d1", "graded_at": 2,
            "request_id": "r1", "rows": [{"test_case": "big", "passed": True,
                                          "performance": _performance(600, elf="candidate-1")}]}
    changed = {**base, "base": "d2", "graded": "copy-d2", "graded_at": 4, "request_id": "r2",
               "rows": [{"test_case": "big", "passed": True,
                         "performance": _performance(500, model={**MODEL, "model_sha256": "m2"},
                                                     elf="candidate-2")}]}
    known = {"big": [(3, "d2", _performance(800, model={**MODEL, "model_sha256": "m2"}, elf="e2"), True),
                     (1, "d1", _performance(900), True)]}
    rows = R.records([base, changed], known, "d2", variant_digests={})
    assert len(rows) == 2 and len({r["identity"] for r in rows}) == 2
    assert rows[0]["baseline_status"] == "current" and rows[1]["baseline_status"] == "superseded"


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
    assert tick() == "estimates current; no untried tuning candidate"
    assert not _requests(ws) and not (ws / U.LEDGER).exists()
    document = {"submission_sha256": "d1", "performance_brief": {"largest_headroom": []}}
    assert "idle_tuning" not in U.attach(document, ws)["performance_brief"]
    monkeypatch.delenv(U.ENV)
    monkeypatch.setenv(I.ENV, "0")  # disabling idle timing disables what runs on its capacity
    assert tick() == "disabled" and not _requests(ws)


def test_the_brief_shows_results_whose_baseline_still_describes_the_case(tmp_path, gates):
    ws, run = _workspace(tmp_path)
    _variant(ws)
    _baseline(ws, "d1", {"big": 900, "mid": 500, "small": 200})
    trees = {"submission": "d1", "variants/alt": "v1"}
    measurer, tick = _measurer(ws, run, trees)
    assert tick() == "tuning submitted"
    ((rid, body),) = _requests(ws).items()
    _answer(ws, rid, body, {"big": 600, "mid": 500, "small": 200})
    _running(ws)
    tick()
    rows = [{"capsule": "big", "tiers": {"L2": "pass"}, "performance": _performance(900)}]
    document = PB.attach({"submission_sha256": "d1", "selfcheck_submission": "submission"}, rows, workspace=ws)
    brief = U.attach(document, ws)["performance_brief"]
    block = brief["idle_tuning"]
    assert block["advisory_only"] is True and block["promotion_authority"] is False
    assert block["count"] == 3 and block["experiments"][0]["test_case"] == "big"
    assert block["experiments"][0]["baseline"].startswith("d1 (") and "current" in block["experiments"][0]["baseline"]
    assert block["ranked_candidates"][0]["candidate"] == "variants/alt"
    assert "TUNING_KNOBS.json" in block["offer"] and "--sweep" in block["offer"]
    assert block["all"] == "qa/idle_tuning.json"
    (entry,) = brief["largest_headroom"]
    assert entry["idle_tuning_experiments"][0]["delta_pct"] == -33.33
    assert len(json.dumps(block)) < 8192  # lean: details stay in the ledger
    trees["submission"] = "d2"
    tick()
    other = PB.attach({"submission_sha256": "d2", "selfcheck_submission": "submission"}, rows, workspace=ws)
    shown = U.attach(other, ws)["performance_brief"]["idle_tuning"]["experiments"]
    assert shown and all("older_source" in row["baseline"] for row in shown)
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
    _variant(ws)
    measurer, tick = _measurer(ws, run, {"submission": "d1", "variants/alt": "v1"})

    def broken(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(measurer.tuner, "refresh", broken)
    monkeypatch.setattr(measurer.tuner, "plan", broken)
    assert tick() == "submitted"  # the baseline timing request still goes out
    assert not _requests(ws)


@pytest.mark.parametrize('kind', ['knob', 'variant'])
def test_new_model_renews_candidates(tmp_path, gates, monkeypatch, kind):
    ws, run = _workspace(tmp_path, cohort=('big',))
    trees = {'submission': 'd1'}
    if kind == 'knob':
        _knobs(ws)
    else:
        _variant(ws)
        trees['variants/alt'] = 'v1'
    selected = ['s1']
    monkeypatch.setattr(I, '_selection_identity', lambda: selected[0])
    _baseline(ws, 'd1', {'big': 900})
    path = ws / '.qa_channel' / 'resp_3_1_base.json'
    doc = json.loads(path.read_text())
    doc['cases'][0]['performance']['selection_sha256'] = 's1'
    path.write_text(json.dumps(doc))
    _, tick = _measurer(ws, run, trees)
    assert tick() == 'tuning submitted'
    ((rid, body),) = _requests(ws).items()
    _answer(ws, rid, body, {'big': 800}, model={**MODEL, 'selection_sha256': 's1'})
    _running(ws)
    assert 'tuning submitted' not in tick()
    selected[0] = 's2'
    doc['cases'][0]['performance']['selection_sha256'] = 's2'
    doc['graded_at_unix_ns'] = 20
    (ws / '.qa_channel' / 'resp_3_2_new_model.json').write_text(json.dumps(doc))
    assert 'tuning submitted' in tick()
    requests = list(_requests(ws).values())
    assert len(requests) == 2
    assert requests[0]['tuning_experiment']['identity'] != requests[1]['tuning_experiment']['identity']


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
    _running(ws)
    tick()
    rows = [dict(capsule='big', tiers={'L2': 'pass'}, performance=_performance(
        100, model={**MODEL, 'model_sha256': 'older-model'}))]
    response = PB.attach(dict(submission_sha256='d1', cases=rows), rows, workspace=ws)
    U.attach(response, ws)
    assert response['performance_brief']['idle_tuning']['experiments'] == []
    assert not response['performance_brief']['largest_headroom'][0].get('idle_tuning_experiments')


def test_progress_distinguishes_missing_adapter_from_missing_declared_alternatives(tmp_path, gates):
    ws, run = _workspace(tmp_path, cohort=('big',))
    _baseline(ws, 'd1', {'big': 900})
    _, tick = _measurer(ws, run, {'submission': 'd1'})
    tick()
    progress = _ledger(ws)['progress']
    assert progress['ready_baselines'] == 1 and progress['declared_candidates'] == 0
    assert progress['outstanding_candidate_cases'] == 0 and progress['requests_in_flight'] == 0
    assert progress['status'] == 'construction_required'
    task = progress['construction_task']
    assert task['kind'] == 'bind_source_effects' and 'binding/identity/evaluate' in task['action']
    assert '--adapter AUTHORED_OWNER.py' in task['command'] and '--facts SELECTED_FACTS.json' in task['command']
    assert 'neutral compilation is not numerics or timing' in task['qualification']
    # A declared custom path is discovered without importing it or prescribing
    # a filename. Presence does not prove its action or legality contracts.
    (ws/'submission'/'custom_adapter.py').write_text("raise RuntimeError('must never import in coordinator')")
    (ws/'submission'/'LIBRARY_PASS_SELECTION.json').write_text(json.dumps({'adapter': 'custom_adapter.py'}))
    tick()
    task = _ledger(ws)['progress']['construction_task']
    assert task['kind'] == 'declare_alternatives' and 'TUNING_KNOBS.json' in task['action']
    assert task['adapter_discovery'] == 'file_present_mapping_unconfirmed'


def test_progress_updates_immediately_and_is_bound_to_source_and_model(tmp_path, gates):
    ws, run = _workspace(tmp_path, cohort=('big',))
    _knobs(ws, values=(False, True, 'other'))
    _baseline(ws, 'd1', {'big': 900})
    _, tick = _measurer(ws, run, {'submission': 'd1'})
    assert tick() == 'tuning submitted'
    ledger = _ledger(ws)
    progress = ledger['progress']
    assert progress['ready_baselines'] == 1 and progress['declared_candidates'] == 2
    assert progress['requests_in_flight'] == 2 and progress['outstanding_candidate_cases'] == 0
    assert progress['status'] == 'running'
    response = dict(submission_sha256='d1', cases=[dict(performance=_performance(900))])
    assert U.matching_progress(ledger, response) == progress
    assert U.matching_progress(ledger, {**response, 'submission_sha256': 'd2'}) is None
    response['cases'][0]['performance']['model_sha256'] = 'old-model'
    assert U.matching_progress(ledger, response) is None


def test_registered_selection_alternatives_use_existing_broker_overrides(tmp_path, gates):
    ws, run = _workspace(tmp_path, cohort=('big',))
    root = ws/'submission'
    (root/'LIBRARY_PASS_SELECTION.json').write_text(json.dumps({'mechanism': None}))
    (root/T.DECLARATION).write_text(json.dumps({
        'schema': T.SCHEMA, 'settings_file': 'LIBRARY_PASS_SELECTION.json',
        'knobs': {'mechanism': {'values': [None, 'registered_a', 'registered_b']}}}))
    _baseline(ws, 'd1', {'big': 900})
    _, tick = _measurer(ws, run, {'submission': 'd1'})
    assert tick() == 'tuning submitted'
    requests = list(_requests(ws).values())
    assert len(requests) == 2
    assert {r[T.REQUEST_KEY]['mechanism'] for r in requests} == {'registered_a', 'registered_b'}
    assert all(r['origin'] == I.TUNING_ORIGIN and r['sim'] == 'spike' for r in requests)
    assert json.loads((root/'LIBRARY_PASS_SELECTION.json').read_text()) == {'mechanism': None}


def test_partial_current_cohort_sweeps_while_refreshing_missing_baselines(tmp_path, gates):
    ws, run = _workspace(tmp_path)
    _knobs(ws)
    _baseline(ws, 'd1', {'big': 900})
    _, tick = _measurer(ws, run, {'submission': 'd1'})
    status = tick()
    assert 'submitted' in status and 'tuning submitted' in status
    assert _requests(ws) and _timing(ws)
    assert _ledger(ws)['progress']['ready_baselines'] == 1
    timed = {n for r in _timing(ws).values() for n in r['capsules'].split(',')}
    assert {'mid', 'small'} <= timed
    work = [(r['submission_sha256'], n) for r in _requests(ws).values() for n in r['capsules'].split(',')]
    assert len(work) == len(set(work))


def test_mixed_public_bundle_routes_every_case_and_preserves_overrides(tmp_path, gates, monkeypatch):
    """The live failure: a scalability representative carried core/calibration siblings."""
    ws, _run = _workspace(tmp_path, cohort=("primary", "control", "program"))
    suites = {"primary": "scalability", "control": "core", "program": "programs"}
    inventory = json.loads((ws / "PUBLIC_CAPABILITIES.json").read_text())
    for name, suite in suites.items():
        inventory["members"][name]["selfcheck_suite"] = suite
    (ws / "PUBLIC_CAPABILITIES.json").write_text(json.dumps(inventory))
    _knobs(ws)
    _baseline(ws, "d1", {"primary": 900, "control": 300, "program": 500})
    _bundles(ws, monkeypatch, [{"label": "mixed", "representative": "primary",
                              "siblings": ["control", "program"]}])
    tuner = U.IdleTuner(ws)
    channel = ws / ".qa_channel"
    jobs = tuner.plan(channel, "d1", list(suites), dict.fromkeys(suites, "pass"), I.estimates(channel))
    assert {(job["suite"], tuple(job["cases"])) for job in jobs} == {
        (suite, (name,)) for name, suite in suites.items()}
    for job in jobs:
        tuner.submit(channel, job, job["cases"], "d1", timeout=37)
    bodies = list(_requests(ws).values())
    assert len(bodies) == 3
    for body in bodies:
        assert body["suite"] == suites[body["capsules"]]
        assert body["tuning_overrides"] == {"ready_readout": True}
        assert body["submission_sha256"] == body["tuning_experiment"]["base_sha256"] == "d1"
        assert body["sim"] == "spike" and body["timeout"] == 37
        assert body.get("tiers") == ("L0,L1,L2" if body["suite"] == "core" else None)
    assert json.loads((ws / "submission/tuning.json").read_text()) == {"ready_readout": False}


def test_refused_or_partial_requests_do_not_cover_unreturned_cases(tmp_path, gates):
    ws, _run = _workspace(tmp_path, cohort=("primary", "control"))
    _knobs(ws)
    channel = ws / ".qa_channel"
    body = {"request_id": "bad", "submission": "submission", "submission_sha256": "d1",
            "suite": "scalability", "capsules": "primary,control",
            "tuning_overrides": {"ready_readout": True}, "deadline_unix_ns": 2**62}
    (channel / "req_bad.json").write_text(json.dumps(body))
    (channel / "resp_bad.json").write_text(json.dumps({
        "error": "public scalability request names unknown or empty cases", "all_pass": False}))
    assert U.IdleTuner(ws)._tried(channel, {}, None)[1] == set()
    body["request_id"] = "partial"
    (channel / "req_partial.json").write_text(json.dumps(body))
    _answer(ws, "partial", {**body, "capsules": "primary", "tuning_experiment": None}, {"primary": 100})
    assert U.IdleTuner(ws)._tried(channel, {}, None)[1] == {("knob:ready_readout=true", "primary")}


def test_core_candidate_missing_baseline_uses_same_route_without_extra_cpu_budget(tmp_path, gates):
    ws, run = _workspace(tmp_path, cohort=("control",))
    inventory = json.loads((ws / "PUBLIC_CAPABILITIES.json").read_text())
    inventory["members"]["control"]["selfcheck_suite"] = "core"
    (ws / "PUBLIC_CAPABILITIES.json").write_text(json.dumps(inventory))
    _knobs(ws)
    measurer = I.IdleMeasurer(ws, run, digest=lambda _root: "d1")
    candidate = measurer.tuner.candidates("d1", None)[0]
    job = {"candidate": candidate, "cases": ["control"], "suite": "core"}
    assert measurer._experiment(ws / ".qa_channel", job, "d1", {}, 1, {}) == 0
    assert not _requests(ws, I.REQUEST_PREFIX)
    assert measurer._experiment(ws / ".qa_channel", job, "d1", {}, 2, {}) == 2
    bodies = list(_requests(ws, I.REQUEST_PREFIX).values())
    assert len(bodies) == 2 and all(b["suite"] == "core" and b["tiers"] == "L0,L1,L2" for b in bodies)


def test_empty_result_retries_are_bounded_without_claiming_case_coverage(tmp_path, gates):
    ws, _run = _workspace(tmp_path, cohort=("big",))
    _knobs(ws)
    _baseline(ws, "d1", {"big": 900})
    channel = ws / ".qa_channel"
    tuner = U.IdleTuner(ws)
    known = I.estimates(channel)
    for _attempt in range(2):
        job, = tuner.plan(channel, "d1", ["big"], {"big": "pass"}, known)
        rid = tuner.submit(channel, job, ["big"], "d1")
        (channel / f"resp_{rid}.json").write_text(json.dumps({"all_pass": False, "error": "no case result"}))
    assert tuner._tried(channel, known, None)[1] == set()
    assert tuner.plan(channel, "d1", ["big"], {"big": "pass"}, known) == []
    progress = tuner.progress(channel, "d1", known, cohort=["big"], l2={"big": "pass"})
    assert progress["status"] == "retry_deferred" and progress["outstanding_candidate_cases"] == 1
    assert progress["retry_deferred_candidate_cases"] == 1
    assert tuner.plan(channel, "d1", ["big"], {"big": "pass"}, known, selection="new-model")
    assert tuner._empty_attempts(channel, "d2", None, candidates=tuner.candidates("d2", None)) == {}
