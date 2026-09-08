from __future__ import annotations

import pytest

from merlin.perf.phase2_portfolio import (
    AnalyticalMetrics,
    FastEvaluationPolicy,
    QualityBudget,
    QualityLimit,
    evaluate_fast_portfolio,
)

MODELS = tuple(f"model-{index}" for index in range(4))


def _quality_budget() -> QualityBudget:
    return QualityBudget(
        limits=(QualityLimit("relative_error", "at_most", 0.10, maximum_degradation=0.02),),
        reference="independent complete-output reference",
    )


def _metrics(
    cycles: float,
    *,
    movement: float,
    utilization: float,
    overlap: float,
    encoding_count: int,
    encoding_bytes: float,
    encoding_cycles: float,
    risk: float = 0.05,
):
    compute_busy = utilization * cycles
    movement_busy = 0.5 * cycles
    return {
        "cycles": {"lo": cycles, "hi": cycles, "provenance": ["calibrated analytical whole-plan model"]},
        "movement_bytes": movement,
        "movement_scope": "physical",
        "occupancy": {
            "total_cycles": cycles,
            "busy_cycles": {"compute-engine": compute_busy, "movement-engine": movement_busy},
            "compute_resources": ["compute-engine"],
            "movement_resources": ["movement-engine"],
            "movement_elapsed_cycles": movement_busy,
            "overlap_cycles": overlap * movement_busy,
            "overlap_available_cycles": movement_busy,
            "idle_cycles": max(0.0, cycles - max(compute_busy, movement_busy)),
            "movement_bytes": movement,
            "encoding_transitions": encoding_count,
            "provenance": ["explicit target-adapter activity schedule"],
        },
        "coverage": {
            "supported_work_total": 1000,
            "supported_work_placed": 800 if cycles < 100 else 700,
            "largest_connected_region_work": 500 if cycles < 100 else 400,
            "connected_region_work": [500, 300] if cycles < 100 else [400, 300],
            "host_islands": [
                {"taxonomy": "unsupported-compute", "count": 2 if cycles < 100 else 3, "work": 50},
            ],
            "boundary_crossings": 8 if cycles < 100 else 10,
            "boundary_bytes": 80 if cycles < 100 else 100,
            "work_unit": "exact arithmetic operations",
            "provenance": ["verified source ownership and global-plan connectivity"],
        },
        "roofline": {
            "lower_bound_cycles": cycles * 0.5,
            "resource_floors": {"compute": cycles * 0.5, "movement": cycles * 0.4},
            "limiting_resources": ["compute"],
            "optimization_effects": ["movement", "tiling"],
            "composition": "explicit adapter max of independently overlappable resources",
            "provenance": ["target descriptor and calibrated resource rates"],
        },
        "encoding_conversions": {
            "count": encoding_count,
            "bytes": encoding_bytes,
            "cycles": {"lo": encoding_cycles, "hi": encoding_cycles, "provenance": ["warm reduced conversion witness"]},
        },
        "risk_score": risk,
        "provenance": ["host-owned analytical adapter v1"],
    }


def _row(
    model: str,
    *,
    candidate_cycles: float = 80,
    candidate_movement: float = 80,
    candidate_error: float = 0.06,
    surfaces=(),
):
    return {
        "model_id": model,
        "baseline": _metrics(
            100, movement=100, utilization=0.50, overlap=0.25, encoding_count=2, encoding_bytes=20, encoding_cycles=8
        ),
        "candidate": _metrics(
            candidate_cycles,
            movement=candidate_movement,
            utilization=0.60,
            overlap=0.50,
            encoding_count=1,
            encoding_bytes=10,
            encoding_cycles=4,
        ),
        "baseline_quality": {
            "values": {"relative_error": 0.05},
            "complete": True,
            "provenance": ["baseline complete-output oracle"],
        },
        "candidate_quality": {
            "values": {"relative_error": candidate_error},
            "complete": True,
            "provenance": ["candidate complete-output oracle"],
        },
        "surfaces": surfaces,
    }


def _evaluate(rows, *, surfaces=None, policy=None):
    return evaluate_fast_portfolio(
        rows,
        quality_budgets={model: _quality_budget() for model in MODELS},
        policy=policy or FastEvaluationPolicy(),
        authorized_surfaces=surfaces,
        expected_models=MODELS,
    )


def test_four_model_portfolio_retains_only_a_quality_safe_pareto_win():
    report = _evaluate([_row(model) for model in MODELS])

    assert report["status"] == "retain"
    assert report["models_evaluated"] == 4
    assert report["portfolio_conservative_cycle_speedup_geomean"] == pytest.approx(1.25)
    assert all(row["quality_gate"]["status"] == "passed" for row in report["models"])
    assert all(row["status"] == "pareto_admissible" for row in report["models"])
    assert report["models"][0]["candidate"]["roofline"]["headroom_to_lower_bound"] == pytest.approx(2.0)
    assert report["models"][0]["recommended_levers"][-1]["lever"] == "roofline_headroom"
    assert "never summed" in report["aggregation"]


def test_quality_budget_rejects_a_faster_candidate():
    rows = [_row(model) for model in MODELS]
    rows[2] = _row(MODELS[2], candidate_cycles=50, candidate_error=0.12)

    report = _evaluate(rows)

    assert report["status"] == "reject"
    failed = report["models"][2]
    assert failed["quality_gate"]["status"] == "failed"
    assert any("exceeds its budget" in failure for failure in failed["failures"])


def test_unknown_occupancy_and_encoding_are_not_treated_as_zero():
    rows = [_row(model) for model in MODELS]
    incomplete = dict(rows[0]["candidate"])
    incomplete["occupancy"] = None
    incomplete["encoding_conversions"] = {}
    rows[0] = {**rows[0], "candidate": incomplete}

    report = _evaluate(rows)

    assert report["status"] == "needs_evidence"
    blocked = report["models"][0]
    assert blocked["status"] == "needs_evidence"
    assert any("compute_utilization is UNKNOWN" in item for item in blocked["blockers"])
    assert blocked["candidate"]["encoding_conversions"]["count"] is None
    assert blocked["candidate"]["encoding_conversions"]["cycles"]["resolved"] is False


def test_one_model_regression_rejects_instead_of_being_averaged_away():
    rows = [_row(model) for model in MODELS]
    rows[3] = _row(MODELS[3], candidate_movement=101)

    report = _evaluate(rows)

    assert report["status"] == "reject"
    assert any("movement_bytes regressed" in item for item in report["failures"])


def test_risk_gate_refuses_an_overwide_or_high_risk_estimate():
    rows = [_row(model) for model in MODELS]
    risky = dict(rows[1]["candidate"])
    risky["risk_score"] = 0.40
    risky["cycles"] = {"lo": 40, "hi": 80, "provenance": ["insufficient calibration"]}
    risky_occupancy = dict(risky["occupancy"])
    risky_occupancy["total_cycles"] = 80
    risky["occupancy"] = risky_occupancy
    rows[1] = {**rows[1], "candidate": risky}

    report = _evaluate(rows)

    assert report["status"] == "reject"
    assert any("calibration risk exceeds policy" in item for item in report["failures"])
    assert any("cycle interval is too wide" in item for item in report["failures"])


def test_recommended_levers_only_include_host_authorized_matching_surfaces():
    rows = [_row(model) for model in MODELS]
    rows[0] = _row(MODELS[0], candidate_movement=101)
    surfaces = {
        MODELS[0]: (
            {
                "id": "movement-pass",
                "path": "compiler/move.py",
                "symbol": "plan",
                "scope": "pass",
                "effects": ["movement", "encoding"],
            },
            {
                "id": "quality-pass",
                "path": "compiler/math.py",
                "symbol": "quantize",
                "scope": "pass",
                "effects": ["quantization"],
            },
        )
    }

    report = _evaluate(rows, surfaces=surfaces)

    movement = next(item for item in report["models"][0]["recommended_levers"] if item["lever"] == "data_movement")
    assert [surface["id"] for surface in movement["authorized_surfaces"]] == ["movement-pass"]


def test_metric_consistency_refuses_disagreement_between_timeline_and_totals():
    value = _metrics(
        100, movement=100, utilization=0.5, overlap=0.2, encoding_count=2, encoding_bytes=20, encoding_cycles=8
    )
    value["movement_bytes"] = 99

    with pytest.raises(ValueError, match="movement bytes disagree"):
        AnalyticalMetrics.from_mapping(value)


def test_candidate_cannot_change_supported_source_work_denominator():
    rows = [_row(model) for model in MODELS]
    candidate = rows[0]["candidate"]
    candidate["coverage"] = {**candidate["coverage"], "supported_work_total": 2000}

    report = _evaluate(rows)

    assert report["status"] == "needs_evidence"
    assert "source-work denominator" in report["models"][0]["reason"]


def test_connected_regions_must_partition_all_placed_work():
    value = _metrics(
        100,
        movement=100,
        utilization=0.5,
        overlap=0.2,
        encoding_count=2,
        encoding_bytes=20,
        encoding_cycles=8,
    )
    value["coverage"] = {
        **value["coverage"],
        "connected_region_work": [400],
    }

    with pytest.raises(ValueError, match="exactly partition"):
        AnalyticalMetrics.from_mapping(value)


def test_more_placed_source_work_alone_is_not_a_global_benefit():
    rows = [_row(model, candidate_cycles=100, candidate_movement=100) for model in MODELS]
    for row in rows:
        candidate = row["candidate"]
        candidate["occupancy"] = dict(row["baseline"]["occupancy"])
        candidate["encoding_conversions"] = dict(row["baseline"]["encoding_conversions"])
        coverage = dict(candidate["coverage"])
        coverage.update(
            supported_work_placed=800,
            largest_connected_region_work=500,
            connected_region_work=[500, 300],
            host_islands=[{"taxonomy": "unsupported-compute", "count": 2, "work": 50}],
            boundary_crossings=10,
            boundary_bytes=100,
        )
        candidate["coverage"] = coverage

    report = _evaluate(rows)

    assert report["status"] == "reject"
    assert report["failures"] == ["portfolio: no robust global benefit; increased placement alone is insufficient"]
    assert report["models"][0]["candidate"]["coverage"]["host_islands"] == [
        {"taxonomy": "unsupported-compute", "count": 2, "work": 50.0}
    ]
