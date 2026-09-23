"""Installed analytical evaluation binds and serializes host-only provider calls."""

import copy
import socket
import subprocess
import threading
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import portfolio_evaluation as PE

from merlin.perf.phase2_portfolio import FastEvaluationPolicy, standard_four_model_quality_schema

MODELS = tuple(C.document_sha256({"model": index}) for index in range(4))
SENTINELS = tuple(SimpleNamespace(capsule_sha256=model) for model in MODELS)
SURFACE = {"id": "schedule", "path": "compiler.py", "symbol": "schedule", "scope": "codegen", "effects": ["movement"]}
EDIT_CONTRACT = {"existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}]}


@pytest.fixture(autouse=True)
def refuse_processes_and_listeners(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("analytical evaluation must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


def metrics(cycles):
    improved = cycles < 100
    count = 1 if improved else 2
    return {
        "cycles": {"lo": cycles, "hi": cycles, "provenance": ["synthetic calibrated model"]},
        "movement_bytes": cycles,
        "movement_scope": "physical",
        "occupancy": {
            "total_cycles": cycles,
            "busy_cycles": {"compute": cycles * (0.6 if improved else 0.5), "move": cycles * 0.5},
            "compute_resources": ["compute"],
            "movement_resources": ["move"],
            "movement_elapsed_cycles": cycles * 0.5,
            "overlap_cycles": cycles * (0.25 if improved else 0.125),
            "overlap_available_cycles": cycles * 0.5,
            "movement_bytes": cycles,
            "encoding_transitions": count,
            "provenance": ["explicit synthetic activity schedule"],
        },
        "coverage": {
            "supported_work_total": 100,
            "supported_work_placed": 80 if improved else 70,
            "largest_connected_region_work": 70 if improved else 60,
            "connected_region_work": [70, 10] if improved else [60, 10],
            "host_islands": [{"taxonomy": "unsupported-control", "count": count, "work": 5}],
            "boundary_crossings": 8 if improved else 10,
            "boundary_bytes": cycles,
            "work_unit": "synthetic source work",
            "provenance": ["source ownership"],
        },
        "roofline": {
            "lower_bound_cycles": cycles * 0.5,
            "resource_floors": {"compute": cycles * 0.5},
            "limiting_resources": ["compute"],
            "optimization_effects": ["movement"],
            "composition": "explicit max",
            "provenance": ["synthetic resource rate"],
        },
        "encoding_conversions": {
            "count": count,
            "bytes": cycles,
            "cycles": {"lo": count, "hi": count, "provenance": ["synthetic conversion"]},
        },
        "risk_score": 0.05,
        "provenance": ["host analytical fixture"],
    }


def provider(**kwargs):
    classification = kwargs["sentinel"].capsule_sha256 == MODELS[0]
    before = (
        {"top1_degradation_percentage_points": 0.0}
        if classification
        else {"cosine_similarity": 1.0, "normalized_root_mean_square_error": 0.0}
    )
    after = (
        {"top1_degradation_percentage_points": 0.4}
        if classification
        else {"cosine_similarity": 0.995, "normalized_root_mean_square_error": 0.01}
    )
    return {
        "baseline": metrics(100),
        "candidate": metrics(80),
        "baseline_quality": {"values": before, "complete": True, "provenance": ["held-out reference"]},
        "candidate_quality": {"values": after, "complete": True, "provenance": ["held-out reference"]},
    }


@pytest.fixture
def configuration():
    quality = standard_four_model_quality_schema(
        MODELS,
        classification_member_sha256=MODELS[0],
        corpus_sha256_by_member={model: C.document_sha256({"corpus": model}) for model in MODELS},
    )
    return {
        "portfolio_sentinels": SENTINELS,
        "provider": provider,
        "policy": FastEvaluationPolicy(),
        "quality_budgets": quality.budget_map,
        "provider_binding": {
            "schema": "host_fast_analytical_evaluator_binding_v1",
            "implementation_sha256": MODELS[0],
            "execution": "host_analytical_only",
            "full_model_simulation_allowed": False,
            "resource_admission": "serialized_one_model_at_a_time",
            "maximum_model_seconds": 1,
            "calibration_sha256s": [MODELS[1]],
        },
    }


def evaluate(owner):
    analysis = {"optimization_brief": {"ranked_actions": [{"edit_surfaces": [copy.deepcopy(SURFACE)]}]}}
    return owner.evaluate(
        [(sentinel, copy.deepcopy(analysis), {"lowered": "synthetic"}) for sentinel in SENTINELS],
        edit_contract=EDIT_CONTRACT,
        target_descriptor={"name": "synthetic-target"},
        target_sha256=MODELS[2],
        portfolio_sha256=MODELS[3],
    )


def test_unconfigured_is_exact_only():
    owner = PE.FastPortfolioEvaluation(SENTINELS)
    owner.check_integrity()
    report = evaluate(owner)
    assert report["status"] == "exact_only_fallback"
    assert report["approximation_allowed"] is False
    assert report["maximum_parallel_model_evaluations"] == 1
    assert owner.binding is None


@pytest.mark.parametrize("missing", ["provider", "policy", "quality_budgets", "provider_binding"])
def test_partial_configuration_refused(configuration, missing):
    configuration[missing] = None
    with pytest.raises(ValueError, match="requires provider"):
        PE.FastPortfolioEvaluation(**configuration)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("implementation_sha256", "bad"),
        ("execution", "simulation"),
        ("full_model_simulation_allowed", True),
        ("resource_admission", "parallel"),
        ("maximum_model_seconds", True),
        ("maximum_model_seconds", 0),
        ("maximum_model_seconds", 61),
        ("calibration_sha256s", []),
        ("calibration_sha256s", "abc"),
        ("calibration_sha256s", ["bad"]),
    ],
)
def test_provider_admission_binding_refused(configuration, field, value):
    configuration["provider_binding"][field] = value
    with pytest.raises(ValueError, match="exact implementation binding"):
        PE.FastPortfolioEvaluation(**configuration)


@pytest.mark.parametrize("mutation", ["count", "missing_budget", "budget_type", "profiles", "policy_type"])
def test_exact_four_model_quality_configuration(configuration, mutation):
    if mutation == "count":
        configuration["portfolio_sentinels"] = SENTINELS[:3]
    elif mutation == "missing_budget":
        configuration["quality_budgets"].pop(MODELS[0])
    elif mutation == "budget_type":
        configuration["quality_budgets"][MODELS[0]] = {}
    elif mutation == "profiles":
        configuration["quality_budgets"][MODELS[0]] = configuration["quality_budgets"][MODELS[1]]
    else:
        configuration["policy"] = {}
    with pytest.raises((ValueError, TypeError)):
        PE.FastPortfolioEvaluation(**configuration)


@pytest.mark.parametrize("mutation", ["binding", "provider", "provider_binding", "policy", "quality"])
def test_integrity_refuses_state_substitution(configuration, mutation):
    owner = PE.FastPortfolioEvaluation(**configuration)
    if mutation == "binding":
        owner.binding["execution"] = "simulation"
    elif mutation == "provider":
        owner.provider = lambda **kwargs: {}
    elif mutation == "provider_binding":
        owner.provider_binding["calibration_sha256s"].append(MODELS[2])
    elif mutation == "policy":
        owner.policy = FastEvaluationPolicy(require_improvement=False)
    else:
        owner.quality_budgets[MODELS[0]] = owner.quality_budgets[MODELS[1]]
    with pytest.raises(ValueError):
        owner.check_integrity()


def test_binding_detached_from_caller(configuration):
    owner = PE.FastPortfolioEvaluation(**configuration)
    configuration["provider_binding"]["calibration_sha256s"].append(MODELS[2])
    configuration["quality_budgets"].clear()
    owner.check_integrity()
    assert owner.binding_sha256 == C.document_sha256(owner.binding)


def test_real_scientific_gate_and_host_arguments(configuration):
    calls = []

    def recorded(**kwargs):
        calls.append(kwargs["sentinel"].capsule_sha256)
        assert kwargs["target_descriptor"] == {"name": "synthetic-target"}
        assert kwargs["target_sha256"] == MODELS[2]
        assert kwargs["portfolio_sha256"] == MODELS[3]
        kwargs["provider_binding"]["calibration_sha256s"].append(MODELS[3])
        return {**provider(**kwargs), "model_id": "provider-cannot-substitute-model"}

    owner = PE.FastPortfolioEvaluation(**{**configuration, "provider": recorded})
    report = evaluate(owner)
    assert report["status"] == "retain"
    assert report["portfolio_conservative_cycle_speedup_geomean"] == pytest.approx(1.25)
    assert report["ordered_model_sha256s"] == calls == list(MODELS)
    assert report["provider_evaluation_order"] == calls
    assert report["resource_admission"] == "serialized_one_model_at_a_time"
    assert "never summed" in report["aggregation"]
    assert report["binding"] == owner.binding
    owner.check_integrity()
    for model in report["models"]:
        for lever in model["recommended_levers"]:
            assert all(surface["id"] == "schedule" for surface in lever["authorized_surfaces"])


def test_faster_candidate_cannot_bypass_quality_gate(configuration):
    def unsafe(**kwargs):
        row = provider(**kwargs)
        if kwargs["sentinel"].capsule_sha256 == MODELS[0]:
            row["candidate_quality"]["values"]["top1_degradation_percentage_points"] = 0.8
        return row

    report = evaluate(PE.FastPortfolioEvaluation(**{**configuration, "provider": unsafe}))
    assert report["status"] == "reject"
    assert report["models"][0]["quality_gate"]["status"] == "failed"


@pytest.mark.parametrize("failure", ["exception", "non_mapping", "elapsed"])
def test_provider_failures_preserve_analysis_but_refuse_scientific_admission(configuration, monkeypatch, failure):
    calls = []

    def failed(**kwargs):
        calls.append(kwargs["sentinel"].capsule_sha256)
        if failure == "exception":
            raise RuntimeError("synthetic provider failure")
        return None if failure == "non_mapping" else provider(**kwargs)

    if failure == "elapsed":
        times = iter(range(0, 16, 2))
        monkeypatch.setattr(PE.time, "monotonic", lambda: next(times))
    report = evaluate(PE.FastPortfolioEvaluation(**{**configuration, "provider": failed}))
    assert calls == list(MODELS)
    assert report["status"] == "needs_evidence"
    assert all(row["status"] == "needs_evidence" for row in report["models"])
    reason = {"exception": "RuntimeError", "non_mapping": "TypeError", "elapsed": "TimeoutError"}[failure]
    assert all(reason in row["reason"] for row in report["models"])


def test_authorized_surfaces_exact_match_dedup_and_detachment():
    allowed = copy.deepcopy(SURFACE)
    surfaces = [
        allowed,
        copy.deepcopy(allowed),
        {**allowed, "symbol": "other"},
        {**allowed, "id": "other"},
        {**allowed, "path": "other.py"},
    ]
    analysis = {"optimization_brief": {"ranked_actions": [{"edit_surfaces": surfaces}]}}
    owner = PE.FastPortfolioEvaluation(SENTINELS)
    assert owner.authorized_surfaces(analysis, edit_contract=None) == ()
    selected = owner.authorized_surfaces(analysis, edit_contract=EDIT_CONTRACT)
    assert selected == (SURFACE,)
    allowed["effects"].append("untrusted")
    assert selected == (SURFACE,)


def test_concurrent_evaluations_serialize_provider_calls(configuration):
    entered, release, second_waiting = threading.Event(), threading.Event(), threading.Event()
    guard = threading.Lock()
    active = peak = calls = 0
    results, errors = [], []

    def serialized(**kwargs):
        nonlocal active, peak, calls
        with guard:
            calls += 1
            active += 1
            peak = max(peak, active)
            first = calls == 1
        try:
            if first:
                entered.set()
                assert release.wait(5), "test never released first analytical call"
            return provider(**kwargs)
        finally:
            with guard:
                active -= 1

    configuration["provider_binding"]["maximum_model_seconds"] = 60
    owner = PE.FastPortfolioEvaluation(**{**configuration, "provider": serialized})
    original_lock = owner._lock

    class ObservedLock:
        """Observe actual contention while retaining the owner's real mutex."""

        def __enter__(self):
            if threading.current_thread().name == "second-evaluation" and not second_waiting.is_set():
                assert original_lock.locked()
                second_waiting.set()
            return original_lock.__enter__()

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    owner._lock = ObservedLock()

    def run():
        try:
            results.append(evaluate(owner))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=run, daemon=True)
    second = threading.Thread(target=run, name="second-evaluation", daemon=True)
    first.start()
    try:
        assert entered.wait(5)
        second.start()
        assert second_waiting.wait(5)
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert calls == 8 and peak == 1
    assert len(results) == 2 and all(result["status"] == "retain" for result in results)
