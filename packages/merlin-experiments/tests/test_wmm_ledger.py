"""The run's OOT history, written by the harness: a commit per candidate whose tree digest IS the store's
package digest, ``measured/<n>`` when its measurement lands, ``best`` only on a confirmed win, and the
champion evidence assembled from the store, refused where the store cannot support it."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import ledger as L
from merlin_experiments.phase2.whole_model_measured import objective as O
from merlin_experiments.phase2.whole_model_measured import service as S
from merlin_experiments.phase2.whole_model_measured.identity import read_json, write_json_atomic

from merlin.common import oot_repo
from merlin.perf import whole_model_verdict as V
from merlin.targetgen import champions


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "spawn", lambda argv, **kw: SimpleNamespace(pid=999_999_999))
    monkeypatch.setattr(S, "alive", lambda pid, owner: pid == 999_999_999)
    phase1 = tmp_path / "phase1_oot"
    oot_repo.init(phase1)
    frozen = oot_repo.commit_candidate(phase1, FX.package(tmp_path, "p1"), label="round 1", when="20260929T000000Z")
    oot_repo.tag(phase1, oot_repo.FROZEN_TAG, frozen.commit)
    repo = oot_repo.init_from(tmp_path / "run" / "oot", phase1, ref=oot_repo.FROZEN_TAG)
    spec, pin = FX.write_builder(tmp_path)
    screen = S.MeasurementService(
        tmp_path / "store", target="toy", builder=spec, builder_sha256=pin, machine=FX.spike_machine(tmp_path)
    )
    objective = O.WholeModelObjective(screen=screen, screen_reference=None, repeats_on_best=1)
    stamps = iter(f"2026093{d}T0{h}0000Z" for d in range(10) for h in range(10))
    objective.ledger = L.OotLedger(
        repo,
        run_id="run",
        records=tmp_path / "run" / L.ITERATIONS,
        sandbox_roots=(tmp_path / "workspace",),
        clock=lambda: next(stamps),
    )
    return SimpleNamespace(objective=objective, repo=repo, frozen=frozen, store=tmp_path / "store", tmp=tmp_path)


def _land(store: Path, digest: str, cycles: int, **extra) -> None:
    job = read_json(store / digest / "job.json")
    job["state"] = J.DONE
    write_json_atomic(store / digest / "job.json", job)
    verdict = {
        "timing_status": V.TIMING_MEASURED,
        "objective_cycles": cycles,
        "whole_window_cycles": cycles,
        "groups": [],
    }
    write_json_atomic(
        store / digest / "result.json",
        {**J.result(job, timing_status=V.TIMING_MEASURED, objective_cycles=cycles, verdict=verdict), **extra},
    )


def test_each_candidate_is_one_harness_commit_whose_tree_is_the_measured_bytes(run):
    first = run.objective.measure(FX.package(run.tmp, "a"), label="first")
    again = run.objective.measure(FX.package(run.tmp, "a"), label="the same bytes")
    rows = run.objective.ledger.candidates()
    assert len(rows) == 1 and first["package_sha256"] == again["package_sha256"] == rows[0]["package_sha256"]
    commit = rows[0]["commit"]
    assert oot_repo.tree_digest(run.repo, commit["commit"]) == rows[0]["package_sha256"] == commit["package_digest"]
    assert oot_repo.is_ancestor(run.repo, run.frozen.commit, commit["commit"])
    assert oot_repo.history(run.repo)[-1].label == "candidate 1"


def test_a_landed_measurement_is_tagged_and_best_moves_only_to_a_confirmed_win(run):
    a = run.objective.measure(FX.package(run.tmp, "a"), label="a")["package_sha256"]
    b = run.objective.measure(FX.package(run.tmp, "b", argmax=4), label="b")["package_sha256"]
    run.objective.poll()
    assert oot_repo.BEST_TAG not in oot_repo.tags(run.repo)  # nothing landed yet
    _land(run.store, a, 1000)
    run.objective.poll()
    tags = oot_repo.tags(run.repo)
    assert "measured/1" in tags and "measured/2" not in tags and tags[oot_repo.BEST_TAG] == tags["measured/1"]
    _land(run.store, b, 900)
    run.objective.poll()
    tags = oot_repo.tags(run.repo)
    assert tags[oot_repo.BEST_TAG] == tags["measured/2"]
    kinds = [json.loads(line)["kind"] for line in (run.tmp / "run" / L.ITERATIONS).read_text().splitlines()]
    assert kinds == ["candidate", "candidate", "measured", "best", "measured", "best"]


def test_an_unconfirmed_best_does_not_move_the_tag(run, monkeypatch):
    a = run.objective.measure(FX.package(run.tmp, "a"), label="a")["package_sha256"]
    _land(run.store, a, 1000)
    monkeypatch.setattr(O.WholeModelObjective, "_confirmed", lambda self, digest: False)
    run.objective.poll()
    assert "measured/1" in oot_repo.tags(run.repo) and oot_repo.BEST_TAG not in oot_repo.tags(run.repo)


def test_the_champion_evidence_comes_from_the_store_and_is_refused_when_it_cannot(run):
    a = run.objective.measure(FX.package(run.tmp, "a"), label="a")["package_sha256"]
    provenance = dict(
        phase1_run="r1", frozen_commit=run.frozen.commit, corpus_seal_digest="c" * 64, phase0_evidence_digest="e" * 64
    )
    _land(
        run.store,
        a,
        1000,
        device={"artifact": "board"},
        build={"parameter_header_sha256": "h" * 64, "isa_census": {"per_group": {}}},
    )
    solo = L.champion_records(run.objective, a, roles=["loop_descriptor"], **provenance)
    missing = champions.missing_evidence(solo)
    assert "measurements.firesim.control.in_batch" in missing and "certification.gsim.verdict" in missing
    scanned = {"verdict": "clean", "scope": "whole_elf", "prohibited": {"8": "LOOP_0"}}
    _land(
        run.store,
        a,
        1000,
        device={"artifact": "board"},
        build={"parameter_header_sha256": "h" * 64, "isa_census": {"per_group": {}}, "isa_prohibition": scanned},
        batch={"batch": "b1", "size": 3, "control": {"ok": True, "ratio": 1.001}},
    )
    run.objective.certifier = SimpleNamespace(
        result=lambda digest: {"timing_status": V.TIMING_MEASURED, "device": {"artifact": "emu"}}
    )
    complete = L.champion_records(run.objective, a, roles=["loop_descriptor"], **provenance)
    assert champions.missing_evidence(complete) == []
    assert complete["isa_prohibition"]["prohibited_instructions"] == {"8": "LOOP_0"}
    unscanned = L.champion_records(run.objective, a, roles=[], **provenance)
    assert "isa_prohibition.verdict" in champions.missing_evidence(unscanned)
    # A build whose scan recorded no prohibited set (a store written before the gate kept it, or a
    # rule that matched nothing) is not clean evidence, whatever its census says.
    _land(
        run.store,
        a,
        1000,
        device={"artifact": "board"},
        build={"parameter_header_sha256": "h" * 64, "isa_census": {"per_group": {}}},
        batch={"batch": "b1", "size": 3, "control": {"ok": True, "ratio": 1.001}},
    )
    vacuous = L.champion_records(run.objective, a, roles=["loop_descriptor"], **provenance)
    assert {"isa_prohibition.verdict", "isa_prohibition.prohibited_instructions"} <= set(
        champions.missing_evidence(vacuous)
    )
