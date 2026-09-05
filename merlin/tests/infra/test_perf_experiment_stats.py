"""Phase-P statistics retain the exact matrix denominator and pair every comparison."""
from __future__ import annotations

import importlib.util
import random
import sys

import pytest

from merlin.common.paths import merlin_dir


_SOURCE = merlin_dir() / "experiments/gemmini_perf_bench/scripts/perf_experiment_stats.py"
_SPEC = importlib.util.spec_from_file_location("perf_experiment_stats_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
STATS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = STATS
_SPEC.loader.exec_module(STATS)


def _trials() -> list[dict]:
    return [
        {"trial": f"t{index}", "agent_run_id": f"agent-{index}"}
        for index in (1, 2, 3)
    ]


def _evidence() -> list[dict]:
    return [
        {**trial, "agent_evidence_sha256": str(index) * 64}
        for index, trial in enumerate(_trials(), 1)
    ]


def _declaration() -> dict:
    return STATS.predeclare(
        trials=_trials(),
        capsules=[{"family": "shape", "capsule": "wide"},
                  {"family": "reuse", "capsule": "deep"}],
        replicates=("r2", "r0", "r1"),
    )


def _rows(declaration: dict) -> list[dict]:
    speedups = {"t1": 2, "t2": 4, "t3": 8}
    rows = []
    for identity in declaration["matrix"]:
        speedup = speedups[identity["trial"]]
        cycles = 800 if identity["subject"] == "baseline" else 800 // speedup
        rows.append({
            "identity": dict(identity), "tier": "L3", "correct": True,
            "cycle_accurate": True, "cycles": cycles,
            "oracle": {"kind": f"rtl_{identity['simulator']}", "derived_from_rtl": True},
        })
    return rows


def test_predeclaration_has_every_exact_identity_and_is_deterministic() -> None:
    declaration = _declaration()
    assert len(declaration["matrix"]) == 3 * 2 * 2 * 3
    assert set(declaration["matrix"][0]) == {
        "trial", "subject", "family", "capsule", "simulator", "replicate"}
    assert {row["simulator"] for row in declaration["matrix"]} == {"gsim"}
    assert declaration["primary_simulator"] == "gsim"
    assert declaration == _declaration()


def test_all_trials_feed_paired_geometric_means_and_aggregate_median() -> None:
    declaration = _declaration()
    result = STATS.evaluate(declaration, _rows(declaration), trial_evidence=_evidence())

    assert result["status"] == "admitted"
    assert [row["geometric_mean_speedup"] for row in result["per_trial"]] == pytest.approx([2, 4, 8])
    assert result["aggregate"]["median_speedup"] == pytest.approx(4)
    assert result["aggregate"]["uncertainty"] == {
        "method": "across_trial_min_max_and_median_absolute_deviation",
        "minimum": pytest.approx(2), "maximum": pytest.approx(8),
        "median_absolute_deviation": pytest.approx(2), "n_independent_trials": 3,
    }
    assert result["aggregate"]["selection"] == "all_predeclared_trials_no_best_of_selection"
    assert set(result["aggregate"]["family_aggregate"]) == {"reuse", "shape"}
    for family in ("reuse", "shape"):
        assert result["aggregate"]["family_aggregate"][family][
            "median_speedup"] == pytest.approx(4)
        assert result["aggregate"]["family_aggregate"][family][
            "all_trial_speedups"] == pytest.approx([2, 4, 8])
    assert "reported separately" in result["aggregate"]["generalization_policy"]
    assert all(row["paired_cells"] == 6 for row in result["per_trial"])


def test_missing_cell_remains_in_denominator_and_refuses_claim() -> None:
    declaration = _declaration()
    rows = _rows(declaration)[:-1]
    result = STATS.evaluate(declaration, rows, trial_evidence=_evidence())
    assert result["status"] == "refused" and result["aggregate"] is None
    assert result["accounting"]["declared_cells"] == len(declaration["matrix"])
    assert result["accounting"]["missing_cells"] == 1
    assert result["accounting"]["declared_agent_trials"] == 3
    assert result["accounting"]["incomplete_agent_trials"] == 1


def test_generalization_family_regression_cannot_be_masked_by_other_family_gain() -> None:
    declaration = _declaration()
    rows = _rows(declaration)
    for row in rows:
        row["cycles"] = 100
        if row["identity"]["subject"] == "candidate":
            row["cycles"] = 50 if row["identity"]["family"] == "reuse" else 200
    result = STATS.evaluate(declaration, rows, trial_evidence=_evidence())

    assert result["status"] == "admitted"
    assert result["aggregate"]["median_speedup"] == pytest.approx(1.0)
    assert result["aggregate"]["family_aggregate"]["reuse"][
        "median_speedup"] == pytest.approx(2.0)
    assert result["aggregate"]["family_aggregate"]["shape"][
        "median_speedup"] == pytest.approx(0.5)


def test_duplicate_and_failed_cells_refuse_instead_of_selecting_best() -> None:
    declaration = _declaration()
    rows = _rows(declaration)
    rows.append(dict(rows[0]))
    rows[1] = {**rows[1], "correct": False, "cycles": 1}
    result = STATS.evaluate(declaration, rows, trial_evidence=_evidence())
    assert result["status"] == "refused"
    assert result["accounting"]["duplicate_cells"] == 1
    assert result["accounting"]["failed_cells"] == 1
    assert result["per_trial"] == []


@pytest.mark.parametrize("change", [
    {"cycle_accurate": False},
    {"oracle": {"derived_from_rtl": False}},
    {"cycles": None},
    {"tier": "L2"},
])
def test_only_correct_cycle_accurate_rtl_l3_rows_are_admitted(change: dict) -> None:
    declaration = _declaration()
    rows = _rows(declaration)
    rows[0] = {**rows[0], **change}
    result = STATS.evaluate(declaration, rows, trial_evidence=_evidence())
    assert result["status"] == "refused" and result["accounting"]["failed_cells"] == 1


def test_spike_cycles_are_excluded_and_cannot_fill_an_elaborated_rtl_cell() -> None:
    declaration = _declaration()
    rows = _rows(declaration)
    missing = rows.pop(0)
    spike = {**missing, "identity": {**missing["identity"], "simulator": "spike"}, "cycles": 1}
    result = STATS.evaluate(declaration, [*rows, spike], trial_evidence=_evidence())
    assert result["status"] == "refused"
    assert result["accounting"]["missing_cells"] == 1
    assert result["accounting"]["excluded_spike_rows"] == 1


def test_verilator_cannot_be_predeclared_as_final_timing_authority() -> None:
    with pytest.raises(STATS.EvidenceError, match="must be GSIM"):
        STATS.predeclare(
            trials=_trials(), capsules=[{"family": "f", "capsule": "c"}],
            replicates=("r0", "r1", "r2"), primary_simulator="verilator")


def test_gsim_identity_cannot_masquerade_as_verilator_oracle() -> None:
    declaration = _declaration()
    rows = _rows(declaration)
    rows[0] = {**rows[0], "oracle": {"kind": "rtl_verilator", "derived_from_rtl": True}}
    result = STATS.evaluate(declaration, rows, trial_evidence=_evidence())
    assert result["status"] == "refused"
    assert result["accounting"]["failed_cells"] == 1


def test_post_run_evidence_must_cover_every_predeclared_trial_exactly() -> None:
    declaration = _declaration()
    missing = STATS.evaluate(
        declaration, _rows(declaration), trial_evidence=_evidence()[:-1])
    assert missing["status"] == "refused"
    assert any("every predeclared trial" in issue for issue in missing["issues"])

    repeated = _evidence()
    repeated[2]["agent_evidence_sha256"] = repeated[1]["agent_evidence_sha256"]
    duplicate = STATS.evaluate(
        declaration, _rows(declaration), trial_evidence=repeated)
    assert duplicate["status"] == "refused"
    assert any("distinct content-addressed" in issue for issue in duplicate["issues"])


def test_declaration_rejects_too_few_or_nonindependent_trials_and_replicates() -> None:
    with pytest.raises(STATS.EvidenceError, match="at least 3 independent"):
        STATS.predeclare(trials=_trials()[:2], capsules=[{"family": "f", "capsule": "c"}],
                         replicates=("r0", "r1", "r2"))
    repeated = _trials()
    repeated[2]["agent_run_id"] = repeated[1]["agent_run_id"]
    with pytest.raises(STATS.EvidenceError, match="distinct"):
        STATS.predeclare(trials=repeated, capsules=[{"family": "f", "capsule": "c"}],
                         replicates=("r0", "r1", "r2"))
    with pytest.raises(STATS.EvidenceError, match="at least 3 unique paired"):
        STATS.predeclare(trials=_trials(), capsules=[{"family": "f", "capsule": "c"}],
                         replicates=("r0", "r1"))


def test_corrupted_predeclared_matrix_digest_refuses() -> None:
    declaration = _declaration()
    declaration["matrix"][0]["capsule"] = "post-hoc-substitution"
    result = STATS.evaluate(declaration, _rows(_declaration()), trial_evidence=_evidence())
    assert result["status"] == "refused"
    assert any("matrix digest" in issue for issue in result["issues"])


def test_trials_cannot_predeclare_different_capsule_cohorts() -> None:
    declaration = _declaration()
    declaration["matrix"] = [
        identity for identity in declaration["matrix"]
        if not (identity["trial"] == "t3" and identity["capsule"] == "deep")]
    declaration["matrix_sha256"] = STATS._sha256(declaration["matrix"])
    declaration["declaration_sha256"] = STATS._sha256({
        key: value for key, value in declaration.items() if key != "declaration_sha256"})
    result = STATS.evaluate(declaration, _rows(declaration), trial_evidence=_evidence())
    assert result["status"] == "refused"
    assert any("identical performance matrix" in issue for issue in result["issues"])


def test_result_order_and_serialization_do_not_depend_on_arrival_order() -> None:
    declaration = _declaration()
    rows = _rows(declaration)
    shuffled = list(rows)
    random.Random(91273).shuffle(shuffled)
    first = STATS.evaluate(declaration, rows, trial_evidence=_evidence())
    second = STATS.evaluate(declaration, shuffled, trial_evidence=list(reversed(_evidence())))
    assert first == second
    assert STATS.canonical_json(first) == STATS.canonical_json(second)
