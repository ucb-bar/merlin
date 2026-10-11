"""Search arithmetic controls only; synthetic reports grant no execution authority."""

import time
from copy import deepcopy
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_workflow as CW
from merlin_experiments.phase2 import search_coordinator as SC
from merlin_experiments.phase2.contracts import StageGateError

ROSTER = (("compute", "contraction"), ("host", "copy"))
CORPUS, ENGINE = "a" * 64, "b" * 64


@pytest.fixture(autouse=True)
def diagnostic_projection(monkeypatch):
    # The live pool validates the full existing feedback schema. These fixtures
    # isolate objective/range arithmetic; they issue no runtime or RTL verdict.
    monkeypatch.setattr(CW, "validate_component_feedback", deepcopy)


def observation(source, compute, host, *, index=0, complete=True):
    cells = [
        {
            "family": family,
            "capsule": capsule,
            "baseline_gsim_cycles": baseline,
            "candidate_gsim_cycles": candidate,
            "measured": complete,
            "comparable": complete,
            "declared_macs": 100 if family == "compute" else None,
        }
        for (family, capsule), baseline, candidate in zip(ROSTER, (100, 10_000), (compute, host), strict=True)
    ]
    return {
        "state": "completed",
        "kind": "component-rtl-feedback",
        "source_sha256": source,
        "job_id": f"{source}-{index}",
        "revision_id": source,
        "options": {"edit_family": source},
        "result": {
            "tier": "certified_component_rtl",
            "candidate_sha256": source,
            "corpus_sha256": CORPUS,
            "configuration_sha256": ENGINE,
            "evidence": {"cells": cells},
        },
    }


def rank(*observations):
    return SC.rank_measured_contenders(
        observations,
        roster=ROSTER,
        corpus_sha256=CORPUS,
        configuration_sha256=ENGINE,
    )


def test_host_cycles_outweigh_large_contraction_speedup():
    result = rank(observation("c" * 64, 1, 10_000), observation("d" * 64, 100, 5_000))
    assert result["ranked"][0]["source_sha256"] == "d" * 64
    assert result["ranked"][0]["ratio_hi"] == pytest.approx(5_100 / 10_100)


def test_incumbent_priorities_share_exact_ranking_weights_and_source():
    winner = observation("c" * 64, 50, 5_000)
    winner["result"]["evidence"]["cells"][0]["roofline"] = {"status": "derived", "roofline_cycles": 10}
    loser = observation("d" * 64, 100, 10_000)
    loser["result"]["evidence"]["cells"][0]["roofline"] = {"status": "derived", "roofline_cycles": 1}
    status = diagnostic_coordinator([winner, loser]).status()
    assert status["incumbent"]["source_sha256"] == "c" * 64
    priority = status["priorities"]
    assert priority["weighted_cycles"] == status["ranked"][0]["candidate_weighted_cycles"]
    assert priority["ranked"][0]["potential_weighted_savings"] == {"lo": 20, "hi": 20}
    assert priority["unpriced_weighted_cycles"] == {"lo": 2_500, "hi": 2_500}


def test_repeated_measurements_never_pick_only_the_fastest_sample():
    result = rank(observation("c" * 64, 50, 5_000), observation("c" * 64, 200, 12_000, index=1))
    row = result["ranked"][0]
    assert row["ratio_lo"] < 1 < row["ratio_hi"]
    assert len(row["jobs"]) == 2


def test_partial_correctness_cannot_rank_or_count_as_a_plateau_trial():
    result = rank(observation("c" * 64, 1, 1, complete=False))
    assert result["ranked"] == []
    assert len(result["excluded"]) == 1


@pytest.mark.parametrize("mutation", ["source", "corpus", "engine", "roster"])
def test_source_context_or_global_roster_drift_refuses(mutation):
    row = observation("c" * 64, 50, 5_000)
    if mutation == "roster":
        row["result"]["evidence"]["cells"].pop()
    else:
        field = {"source": "candidate_sha256", "corpus": "corpus_sha256", "engine": "configuration_sha256"}[mutation]
        row["result"][field] = "e" * 64
    with pytest.raises(StageGateError):
        rank(row)


def test_analytical_predictions_cannot_enter_measured_ranking():
    row = observation("c" * 64, 1, 1)
    row["kind"] = "component-analytical-feedback"
    row["result"] = {"predicted_cycles": 1}
    assert rank(row)["ranked"] == []


@pytest.mark.parametrize(
    "settings",
    [
        SC.SearchSettings(beam_width=4),
        SC.SearchSettings(hard_seconds=28_801),
        SC.SearchSettings(final_reserve_seconds=14_400),
        SC.SearchSettings(minimum_improvement=float("nan")),
    ],
)
def test_unbounded_or_unreserved_search_refuses(settings):
    with pytest.raises(StageGateError):
        settings.verify()


def diagnostic_coordinator(rows, *, beam=3):
    # Isolate state/ranking arithmetic. This bypass issues no source or runtime
    # authority; actual restoration and cleanup use separate live-owner tests.
    baseline = {"revision_id": "f" * 64, "source_sha256": "f" * 64, "options": {}}
    coordinator = SC.ComponentSearchCoordinator.__new__(SC.ComponentSearchCoordinator)
    coordinator.pool = SimpleNamespace(
        observations=lambda: tuple(rows),
        max_jobs=256,
        baseline_revision=SimpleNamespace(public_record=lambda: deepcopy(baseline)),
        revisions=SimpleNamespace(
            get=lambda identity: SimpleNamespace(
                public_record=lambda: {
                    "revision_id": identity,
                    "source_sha256": identity,
                    "options": {},
                }
            )
        ),
    )
    coordinator.settings = SC.SearchSettings(beam_width=beam)
    coordinator.roster, coordinator.corpus_sha256, coordinator.configuration_sha256 = ROSTER, CORPUS, ENGINE
    coordinator.started = time.monotonic()
    coordinator._incumbent, coordinator._plateau = baseline, 0
    coordinator._finalized, coordinator.selected = False, None
    return coordinator


def test_late_repeat_revokes_prior_superiority_and_restores_baseline_fallback():
    rows = [observation("c" * 64, 90, 9_000)]
    coordinator = diagnostic_coordinator(rows)
    assert coordinator.status()["incumbent"]["source_sha256"] == "c" * 64
    rows.append(observation("c" * 64, 120, 12_000, index=1))
    assert coordinator.status()["incumbent"]["source_sha256"] == "f" * 64


def test_baseline_repeats_do_not_count_as_exploration_batches():
    rows = [observation("f" * 64, 100, 10_000), observation("c" * 64, 110, 11_000)]
    assert diagnostic_coordinator(rows, beam=2).status()["plateau_batches"] == 0


def test_plateau_is_recomputed_after_new_repeat_changes_batch_bounds():
    rows = [observation("c" * 64, 90, 9_000), observation("d" * 64, 120, 12_000)]
    coordinator = diagnostic_coordinator(rows, beam=2)
    assert coordinator.status()["plateau_batches"] == 0
    rows.append(observation("c" * 64, 130, 13_000, index=1))
    assert coordinator.status()["plateau_batches"] == 1


def test_source_selection_refuses_before_queue_finalization():
    with pytest.raises(StageGateError, match="cleanup"):
        diagnostic_coordinator([]).select()


def test_selection_uses_complete_ranked_observation_after_partial_attempt():
    source = "c" * 64
    partial = observation(source, 1, 1, complete=False)
    complete = observation(source, 50, 5_000, index=1)
    coordinator = diagnostic_coordinator([partial, complete])
    coordinator._finalized = True
    coordinator.pool.policy = SimpleNamespace(candidate="synthetic-candidate")
    coordinator.pool.revisions.restore = lambda identity, candidate: {
        "revision_id": identity,
        "source_sha256": identity,
    }
    selection = coordinator.select()
    assert selection["feedback"] is complete["result"]
    assert selection["selected"]["source_sha256"] == source
    assert selection["promotion"] == "NO_FINAL_ACCEPTANCE"
