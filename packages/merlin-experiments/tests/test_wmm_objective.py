"""The objective a measured loop optimises: ties within the store's own noise, solo confirmation of a
new best, certification feasibility, coverage eligibility, the stagnation signal, and the policy a
config must carry."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import config as C
from merlin_experiments.phase2.whole_model_measured import feedback as F
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import objective as O
from merlin_experiments.phase2.whole_model_measured import service as S

from merlin.perf import whole_model_verdict as V


def _covered(tmp_path: Path, entries) -> O.WholeModelObjective:
    """A screen store holding measured results with given (digest, cycles, package groups), oldest first."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec, pin = FX.write_builder(tmp_path)
    service = S.MeasurementService(
        tmp_path / "store", target="toy", builder=spec, builder_sha256=pin, machine=FX.spike_machine(tmp_path)
    )
    for index, (digest, cycles, answered) in enumerate(entries):
        job_dir = service.root / digest
        job_dir.mkdir(parents=True)
        (job_dir / "job.json").write_text(
            json.dumps({"package_sha256": digest, "job_key": digest, "state": J.DONE, "requested_epoch": index})
        )
        verdict = {
            "timing_status": V.TIMING_MEASURED,
            "objective_cycles": cycles,
            "whole_window_cycles": cycles,
            "groups": [],
        }
        routes = [{"group": g, "op": "conv2d", "on": "package" if g in answered else "vendor"} for g in (1, 2, 3)]
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "package_sha256": digest,
                    "timing_status": V.TIMING_MEASURED,
                    "objective_cycles": cycles,
                    "verdict": verdict,
                    "build": {"groups": routes},
                    "finished_at": f"2026092{index}T000000Z",
                }
            )
        )
    reference = tmp_path / "reference.json"
    groups = [{"group": g, "cycles": c} for g, c in ((1, 600), (2, 300), (3, 100))]
    reference.write_text(
        json.dumps({"timing_status": V.TIMING_MEASURED, "verdict": {"whole_window_cycles": 1000, "groups": groups}})
    )
    return O.WholeModelObjective(screen=service, screen_reference=reference)


def _repeat(objective, digest, cycles):
    job_dir = objective.screen.root / f"{digest}.r1"
    job_dir.mkdir()
    (job_dir / "job.json").write_text(
        json.dumps({"package_sha256": digest, "job_key": f"{digest}.r1", "replicate": 1, "state": J.DONE})
    )
    verdict = {"timing_status": V.TIMING_MEASURED, "objective_cycles": cycles}
    (job_dir / "result.json").write_text(
        json.dumps({"timing_status": V.TIMING_MEASURED, "objective_cycles": cycles, "verdict": verdict})
    )


def test_a_tie_never_displaces_the_best_and_is_recorded(tmp_path):
    objective = _covered(tmp_path, [("seed", 1000, {1, 2}), ("first", 900, {1, 2}), ("later", 900, {1, 2})])
    assert objective.noise_margin() == O.NOISE_FLOOR  # no repeats yet: the floor
    assert objective._screen_best_unretracted()["package_sha256"] == "first"
    rows = {r["package_sha256"]: r for r in objective.summary()["history"]}
    assert rows["later"]["tie_with"] == "first" and "tie_with" not in rows["first"]


def test_the_noise_margin_is_the_stores_own_measured_repeat_spread(tmp_path):
    """A fixed 0.1% floor undercounted a board whose repeats of one package landed up to 0.4% apart: a
    candidate crowned inside that spread is not a measured improvement."""
    objective = _covered(tmp_path, [("seed", 1000, {1, 2}), ("a", 900, {1, 2}), ("b", 890, {1, 2})])
    for base, again in (("seed", 1020), ("a", 909)):
        _repeat(objective, base, again)
    assert objective.repeat_spread() == 0.02  # the median of {1%, 2%}
    assert objective.noise_margin() == 0.02
    # b is 1.1% below a: inside the store's measured 2% spread, so the earlier best keeps the title
    assert objective._screen_best_unretracted()["package_sha256"] == "a"


def test_the_margin_never_drops_below_the_floor(tmp_path):
    objective = _covered(tmp_path, [("seed", 1000, {1, 2})])
    _repeat(objective, "seed", 1000)
    assert objective.repeat_spread() == 0.0 and objective.noise_margin() == O.NOISE_FLOOR


def test_a_new_best_that_does_not_hold_up_solo_is_demoted_to_a_tie(tmp_path, monkeypatch):
    objective = _covered(tmp_path, [("seed", 1000, {1, 2}), ("old", 900, {1, 2}), ("new", 890, {1, 2})])
    monkeypatch.setattr(objective, "repeat_spread", lambda: None)  # isolate from margin derivation
    assert objective._screen_best_unretracted()["package_sha256"] == "new"
    _repeat(objective, "old", 905)
    assert objective._screen_best_unretracted()["package_sha256"] == "new"  # its own solo not in yet
    _repeat(objective, "new", 905)
    assert objective._screen_best_unretracted()["package_sha256"] == "old"
    rows = {r["package_sha256"]: r for r in objective.summary()["history"] if not r.get("replicate")}
    assert rows["new"]["tie_with"] == "old" and "noise margin" in rows["new"]["tie_reason"]


def test_a_candidate_that_declines_work_to_the_library_is_never_the_best(tmp_path):
    objective = _covered(tmp_path, [("seed", 1000, {1, 2}), ("declines", 500, {2})])
    assert objective._screen_best_unretracted()["package_sha256"] == "seed"
    coverage = objective.coverage(objective.screen.result("declines"))
    assert coverage["eligible"] is False and coverage["declined_vs_seed"] == ["1"]
    assert "INELIGIBLE" in objective.ineligible_text(objective.screen.result("declines"))


def _stamp(seconds: float) -> str:
    return (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)).strftime("%Y%m%dT%H%M%SZ")


def _feasibility(tmp_path, board_cycles, *, timeout_seconds=144000.0, reference_seconds=9000):
    tmp_path.mkdir(parents=True, exist_ok=True)
    reference_dir = tmp_path / "cert_reference"
    reference_dir.mkdir()
    (reference_dir / "job.json").write_text(
        json.dumps({"dispatched_at": _stamp(0), "finished_at": _stamp(reference_seconds)})
    )
    (reference_dir / "result.json").write_text(json.dumps({"timing_status": V.TIMING_MEASURED}))
    screen_reference = tmp_path / "screen_reference.json"
    screen_reference.write_text(json.dumps({"verdict": {"whole_window_cycles": 30_000_000}}))
    requested = []
    screen = SimpleNamespace(
        root=tmp_path / "screen",
        result=lambda digest: {"package_sha256": digest, "objective_cycles": board_cycles},
        result_by_key=lambda key: None,
        request=lambda *a, **k: {"job_key": f"r{k.get('replicate')}"},
    )
    certifier = SimpleNamespace(
        timeout_seconds=timeout_seconds,
        request=lambda snapshot, **k: requested.append(snapshot) or {"package_sha256": "best"},
        result=lambda digest: None,
    )
    objective = O.WholeModelObjective(
        screen=screen,
        screen_reference=screen_reference,
        certifier=certifier,
        certifier_reference=reference_dir / "result.json",
    )
    return objective, requested


def test_a_certification_that_cannot_finish_in_time_is_deferred_with_its_projection(tmp_path):
    objective, requested = _feasibility(tmp_path, 6_000_000_000)
    objective._promote("best")
    record = objective._promoted["best"]
    assert requested == [] and record["state"] == O.DEFERRED_INFEASIBLE
    assert record["projection"]["projected_seconds"] == 1_800_000
    objective._refresh("best", record)
    assert record["state"] == O.DEFERRED_INFEASIBLE


def test_a_promoted_bests_repeat_is_requested_at_the_elevated_priority(tmp_path):
    """So batch.batch_candidates can outrank an older solo repeat already in the same queue with a
    freshly-promoted best's confirmation."""
    requested_screen = []
    objective, requested = _feasibility(tmp_path, 29_000_000)
    objective.screen.request = lambda *a, **k: requested_screen.append(k) or {"job_key": f"r{k.get('replicate')}"}
    objective._promote("best")
    assert requested_screen and all(k.get("priority") == O.PROMOTED_REPEAT_PRIORITY for k in requested_screen)
    assert all(k.get("solo") is True for k in requested_screen)


def test_an_unknown_certification_rate_fails_closed(tmp_path):
    objective, requested = _feasibility(tmp_path, 29_000_000)
    objective.screen_reference = None
    projection = objective.certification_projection("best")
    assert projection["feasible"] is False and projection["reason"].startswith("UNKNOWN")
    ok, requested = _feasibility(tmp_path / "ok", 29_000_000)
    ok._promote("best")
    assert len(requested) == 1 and ok._promoted["best"]["state"] == O.CERT_PENDING


# --------------------------------------------------------------- the stagnation signal
def _history():
    return [
        (100.0, {"1": 6_300_000, "70": 2_700_000}),
        (150.0, {"1": 6_310_000, "70": 2_690_000}),
        (250.0, {"1": 5_000_000, "70": 2_699_000}),
    ]


def test_a_group_improved_this_session_moved_and_one_that_did_not_is_named():
    document = F.stagnation(_history(), ["1", "70"], since_epoch=200.0, noise=0.001)
    rows = {r["group"]: r for r in document["groups"]}
    assert rows["1"]["moved"] and rows["70"]["moved"] is False and document["unmoved"] == ["70"]


def test_a_session_with_no_measurement_says_so_rather_than_claiming_stagnation():
    document = F.stagnation(_history(), ["1"], since_epoch=1_000.0, noise=0.001)
    assert document["groups"][0]["moved"] is None and document["unmoved"] == []
    assert "not measured this session" in F.render_stagnation(document)


def test_the_rendered_signal_states_counts_and_never_a_remedy():
    text = F.render_stagnation(F.stagnation(_history(), ["1", "70"], since_epoch=200.0, noise=0.001))
    for word in ("try", "should", "consider", "tile", "im2col"):
        assert word not in text.lower()


def test_the_objective_reads_its_best_gap_holders_against_the_marked_session(tmp_path, monkeypatch):
    objective = _covered(tmp_path, [("seed", 1000, {1, 2})])
    assert objective.stagnation() is None
    objective.mark_session(200.0)
    best = {"verdict": {"groups": []}, "package_sha256": "b"}
    monkeypatch.setattr(O.WholeModelObjective, "_screen_best_unretracted", lambda self: best)
    monkeypatch.setattr(O.F, "compare", lambda found, reference: {"gap_holders": [{"group": "1"}]})
    results = {
        "a": {
            "timing_status": "MEASURED",
            "finished_at": "19700101T000140Z",
            "verdict": {"groups": [{"group": "1", "cycles": 10}]},
        },
        "b": {
            "timing_status": "MEASURED",
            "finished_at": "19700101T000500Z",
            "verdict": {"groups": [{"group": "1", "cycles": 8}]},
        },
    }
    objective.screen = SimpleNamespace(
        jobs=lambda: [{"package_sha256": k, "job_key": k} for k in results], result_by_key=results.get
    )
    monkeypatch.setattr(O.WholeModelObjective, "coverage", lambda self, found: None)
    assert objective.stagnation()["groups"] == [{"group": "1", "before_session": 10, "this_session": 8, "moved": True}]


# --------------------------------------------------------------- the config's policy
def _config(tmp_path, roles, screen_roles, certifier_roles=None, *, sealed=True):
    spec, pin = FX.write_builder(tmp_path)
    machine = FX.spike_machine(tmp_path)

    def section(roles):
        return {"machine": machine, "build_options": {"prohibited_roles": roles} if roles is not None else {}}

    document = {
        "schema": C.CONFIG_SCHEMA,
        "builder": {"spec": spec, "sha256": pin},
        "store": str(tmp_path / "store"),
        "prohibited_instruction_roles": roles,
        "screen": section(screen_roles),
    }
    if roles and sealed:
        document[C.SEALED_POLICY] = FX.sealed_policy(roles)
    if certifier_roles is not None:
        document["certifier"] = section(certifier_roles)
    return document


def test_a_candidate_section_that_dropped_the_declared_roles_refuses_the_launch(tmp_path):
    """A relaunch that silently dropped the no-FSM roles once ran a no-FSM campaign without them."""
    with pytest.raises(C.ConfigError, match="prohibited roles"):
        C.from_config(_config(tmp_path, ["loop_descriptor"], None), target="toy")
    with pytest.raises(C.ConfigError, match="certifier"):
        C.from_config(_config(tmp_path, ["loop_descriptor"], ["loop_descriptor"], []), target="toy")
    objective = C.from_config(_config(tmp_path, ["loop_descriptor"], ["loop_descriptor"]), target="toy")
    assert objective.screen.build_options["prohibited_roles"] == ["loop_descriptor"]
    assert objective.rule()["prohibited_roles"] == ["loop_descriptor"]
    # The sealed policy rides every candidate job, so the instruction gate can hold the program to it.
    assert objective.screen.instruction_policy == FX.sealed_policy(["loop_descriptor"])


def test_declared_roles_without_an_enforceable_sealed_policy_refuse_the_launch(tmp_path):
    """Roles with no sealed Phase 0 policy, or one that is vacuous or unresolved, never start a run."""
    with pytest.raises(C.ConfigError, match="no enforceable sealed Phase 0"):
        C.from_config(_config(tmp_path, ["loop_descriptor"], ["loop_descriptor"], sealed=False), target="toy")
    vacuous = _config(tmp_path, ["loop_descriptor"], ["loop_descriptor"])
    vacuous[C.SEALED_POLICY]["prohibited_instructions"] = {"loop_descriptor": []}
    with pytest.raises(C.ConfigError, match="prohibits no instruction"):
        C.from_config(vacuous, target="toy")
    unknown = _config(tmp_path, ["loop_descriptor"], ["loop_descriptor"])
    unknown[C.SEALED_POLICY]["status"] = "UNKNOWN"
    with pytest.raises(C.ConfigError, match="not 'resolved'"):
        C.from_config(unknown, target="toy")


def test_the_sealed_policy_is_read_from_the_phase0_manifest_by_value(tmp_path):
    manifest = FX.write_phase0_manifest(tmp_path)
    document = C.with_policy(_config(tmp_path, [], None, [], sealed=False), ["loop_descriptor"])
    sealed = C.seal_policy(document, target="toy", manifest=manifest)
    assert sealed[C.SEALED_POLICY]["prohibited_instructions"] == FX.SEALED_POLICY["prohibited_instructions"]
    assert sealed[C.SEALED_POLICY]["sealed_source"]["path"] == str(manifest.resolve())
    assert C.check_policy(sealed) == ["loop_descriptor"]
    # A relaunch keeps the policy it was sealed under; a manifest naming another one is refused.
    assert C.seal_policy(sealed, target="toy") == sealed
    other = FX.write_phase0_manifest(tmp_path / "other", FX.sealed_policy(["loop_descriptor", "sync"]))
    with pytest.raises(C.ConfigError, match="differs"):
        C.seal_policy(sealed, target="toy", manifest=other)


def test_with_policy_writes_the_declared_roles_into_every_candidate_section(tmp_path):
    document = C.with_policy(_config(tmp_path, [], None, []), ["loop_descriptor"])
    document[C.SEALED_POLICY] = FX.sealed_policy()
    assert C.check_policy(document) == ["loop_descriptor"]
    assert document["certifier"]["build_options"]["prohibited_roles"] == ["loop_descriptor"]


def test_a_machine_of_another_target_is_refused(tmp_path):
    with pytest.raises(C.ConfigError, match="not this run's"):
        C.from_config(_config(tmp_path, [], None), target="other")


def test_the_first_request_after_a_relaunch_carries_the_coverage_gate(tmp_path, monkeypatch):
    """A gate armed only on poll left the first request of every relaunch ungated: a candidate that
    handed a group back to the library reached the board."""
    objective = _covered(tmp_path, [("seed", 1000, {1, 2})])
    monkeypatch.setattr(S, "spawn", lambda argv, **kw: SimpleNamespace(pid=999_999_999))
    assert objective.screen.coverage_gate is None  # a fresh process: nothing armed yet
    job = objective.screen.request(FX.package(tmp_path, "declines", argmax=9), label="first after relaunch")
    assert job["coverage_gate"]["floor_package_sha256"] == "seed"
    assert sorted(job["coverage_gate"]["floor_groups"]) == ["1", "2"] and job["coverage_gate"]["price"]["1"] == 600
    reference_job = objective.screen.request(FX.package(tmp_path, "ref", argmax=8), role=J.ROLE_REFERENCE)
    assert reference_job["coverage_gate"] is None


def test_a_one_off_request_with_no_objective_derives_the_same_gate_from_the_store(tmp_path, monkeypatch):
    """A cell win transplanted to the whole model arrives as an external request, with no objective
    polling the store; it must still carry the seed's coverage floor."""
    objective = _covered(tmp_path, [("seed", 1000, {1, 2})])
    monkeypatch.setattr(S, "spawn", lambda argv, **kw: SimpleNamespace(pid=999_999_999))
    service = S.MeasurementService(
        objective.screen.root,
        target="toy",
        builder=objective.screen.builder["spec"],
        builder_sha256=objective.screen.builder["sha256"],
        machine=objective.screen.machine,
        reference=objective.screen_reference_path,
    )
    expected = objective.coverage_gate_document()
    job = service.request(FX.package(tmp_path, "transplant", argmax=9), label="cell transplant")
    assert {k: job["coverage_gate"][k] for k in expected} == expected
    young = S.MeasurementService(
        tmp_path / "young",
        target="toy",
        builder=service.builder["spec"],
        builder_sha256=service.builder["sha256"],
        machine=service.machine,
    )
    assert young.derive_coverage_gate() is None
