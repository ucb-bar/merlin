"""Cross-run ratios must share a denominator, or they are not a comparison.

The defect. ``capsule_grade`` sets ``n_capsules`` to the rows it MEASURED, excluding
``NOT_MEASURED_STATUSES`` (``not_graded``, ``gated``, ``screened_only``, ``budget_exhausted``,
``infrastructure_fault``). For a single run that is right -- a deferred row is not a verdict, and
counting one as a failure puts ``all_pass`` out of reach and disables an agent loop's only early
exit. Across runs it is a lie, and it flatters the WEAKEST run.

Measured on the g3arm gemmini batch. Arms 1-3 reached an op pass fraction of ~0.96, so the
whole-model gate OPENED: ``M2_microvit_gemmini`` and ``SY_micro_model`` ran, FAILED, and stayed in a
denominator of 97 -> ``93/97`` (95.9%). Arm 4 reached ~0.79, the gate stayed shut, those same two
capsules were deferred (``status: gated``) and left its denominator -> ``75/95`` (78.9%). Arm 4's
percentage was computed over a cohort with the two hardest rows deleted while every other arm carried
them as failures, so the reported gap UNDERSTATED the real one: on the common cohort it is 93/95
against 75/95.

The fix is NOT to re-add deferred rows as failures -- they were not measured, and "not run is not a
pass" cuts both ways. It is to INTERSECT: score every run on the rows every compared run measured,
and name what each run did not measure beside its number. This file pins that, and pins the two ways
it would go quiet: a cohort computed as a union (which invents verdicts), and a table that prints two
ratios with different denominators next to each other as if they answered the same question.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.capsule_common import NOT_MEASURED_STATUSES

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _load(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 -- harness deps absent in this env
        pytest.skip(f"{name} not importable here: {type(exc).__name__}: {exc}")
    return mod


@pytest.fixture()
def agg():
    return _load("agg_agentic_results")


# --------------------------------------------------------------------------------------------------
# synthetic verdicts, shaped exactly like the archived qa_history/verdict_*.json rows
# --------------------------------------------------------------------------------------------------


def _row(name: str, status: str, l3: str | None = None) -> dict:
    r = {"capsule": name, "status": status}
    if l3 is not None:
        r["tiers"] = {"L3": l3}
    return r


def _verdict(rows: list[dict], **extra) -> dict:
    """A verdict whose n_passed/n_capsules are computed the way ``capsule_grade`` computes them --
    over the MEASURED rows only. That arithmetic is the premise of the whole file, so the fixture
    reproduces it rather than hardcoding numbers that could drift away from the grader."""
    measured = [r for r in rows if r["status"] not in NOT_MEASURED_STATUSES]
    return {
        "per_capsule": rows,
        "n_capsules": len(measured),
        "n_passed": sum(1 for r in measured if r["status"] == "pass"),
        **extra,
    }


def _runs(**by_id) -> dict:
    """``{run_id: {"arm": ..., "verdict": ...}}`` -- the shape ``cohort_report`` consumes."""
    return {rid: {"arm": rid.split("_")[0], "verdict": v, "verdict_file": f"{rid}.json"} for rid, v in by_id.items()}


#: Five rows both runs below agree on, plus the sixth row ("X") they disagree about MEASURING.
def _open_gate_rows(x_status: str = "fail") -> list[dict]:
    return [
        _row("p0", "pass", "pass"),
        _row("p1", "pass", "pass"),
        _row("p2", "pass", "pass"),
        _row("p3", "pass", "pass"),
        _row("f0", "fail"),
        _row("X", x_status),
    ]


def _shut_gate_rows(x_status: str) -> list[dict]:
    return [
        _row("p0", "pass", "pass"),
        _row("p1", "pass", "pass"),
        _row("p2", "pass", "pass"),
        _row("p3", "fail"),
        _row("f0", "fail"),
        _row("X", x_status),
    ]


# --------------------------------------------------------------------------------------------------
# the intersection itself
# --------------------------------------------------------------------------------------------------


def test_a_gated_row_leaves_the_cohort_for_BOTH_runs(agg):
    """The g3arm shape in miniature: one run measured X and failed it, the other deferred it.

    X must leave the cohort for BOTH -- including the run that DID measure it. Keeping it (a union)
    would ask the deferring run for a verdict it never produced; dropping it only from the deferring
    run is the original defect.
    """
    runs = _runs(open_gate=_verdict(_open_gate_rows("fail")), shut_gate=_verdict(_shut_gate_rows("gated")))
    rep = agg.cohort_report(runs)

    assert rep["common_cohort"] == ["f0", "p0", "p1", "p2", "p3"]
    assert rep["common_cohort_size"] == 5
    assert "X" not in rep["common_cohort"]
    assert rep["union_measured_size"] == 6, "X was measured by one run -- the union must still see it"
    # Equal denominators, which is the entire point.
    assert rep["runs"]["open_gate"]["cohort"]["of"] == rep["runs"]["shut_gate"]["cohort"]["of"] == 5
    assert rep["runs"]["open_gate"]["cohort"]["passed_ratio"] == "4/5"
    assert rep["runs"]["shut_gate"]["cohort"]["passed_ratio"] == "3/5"


def test_the_own_ratios_of_that_pair_are_over_different_denominators(agg):
    """Both facts survive: the own figures are kept verbatim AND flagged as non-comparable."""
    runs = _runs(open_gate=_verdict(_open_gate_rows("fail")), shut_gate=_verdict(_shut_gate_rows("gated")))
    rep = agg.cohort_report(runs)

    assert rep["runs"]["open_gate"]["own_ratio"] == "4/6"  # X measured, failed, in the denominator
    assert rep["runs"]["shut_gate"]["own_ratio"] == "3/5"  # X deferred, out of the denominator
    assert rep["own_denominators"] == {"open_gate": 6, "shut_gate": 5}
    assert rep["own_ratios_comparable"] is False
    assert rep["comparable_metric"] == "cohort"


def test_the_deferred_row_is_never_scored_as_a_failure(agg):
    """`not_run_is_not_pass` cuts both ways: the fix must not smuggle X back in as a fail."""
    runs = _runs(open_gate=_verdict(_open_gate_rows("fail")), shut_gate=_verdict(_shut_gate_rows("gated")))
    rep = agg.cohort_report(runs)
    shut = rep["runs"]["shut_gate"]
    # 3 of 5 -- not 3 of 6 (which would count the unmeasured X against it).
    assert (shut["cohort"]["passed"], shut["cohort"]["of"]) == (3, 5)
    assert "X" not in rep["common_cohort"]


def test_equal_own_denominators_over_DIFFERENT_rows_are_still_not_comparable(agg):
    """The subtle case. Two runs each measured 5 of 6 rows -- but not the same 5.

    Comparing 4/5 with 3/5 here is as wrong as comparing 4/6 with 3/5: the denominators match only
    numerically. Only measuring exactly the cohort makes an own ratio comparable.
    """
    a = _verdict(
        [
            _row("p0", "pass"),
            _row("p1", "pass"),
            _row("p2", "pass"),
            _row("p3", "pass"),
            _row("X", "gated"),
            _row("Y", "fail"),
        ]
    )
    b = _verdict(
        [
            _row("p0", "pass"),
            _row("p1", "pass"),
            _row("p2", "pass"),
            _row("p3", "fail"),
            _row("X", "fail"),
            _row("Y", "budget_exhausted"),
        ]
    )
    rep = agg.cohort_report(_runs(a_run=a, b_run=b))

    assert set(rep["own_denominators"].values()) == {5}, "numerically identical denominators"
    assert rep["common_cohort_size"] == 4 and rep["common_cohort"] == ["p0", "p1", "p2", "p3"]
    assert rep["own_ratios_comparable"] is False
    assert rep["runs"]["a_run"]["cohort"]["passed_ratio"] == "4/4"
    assert rep["runs"]["b_run"]["cohort"]["passed_ratio"] == "3/4"


def test_screened_only_and_budget_exhausted_are_handled_exactly_like_gated(agg):
    """Every not-measured status is the same fact: no verdict was produced for that row."""
    runs = _runs(
        screened=_verdict([_row("p0", "pass"), _row("p1", "pass"), _row("X", "screened_only")]),
        budget=_verdict([_row("p0", "pass"), _row("p1", "fail"), _row("X", "budget_exhausted")]),
        measured=_verdict([_row("p0", "pass"), _row("p1", "pass"), _row("X", "fail")]),
    )
    rep = agg.cohort_report(runs)

    assert rep["common_cohort"] == ["p0", "p1"]
    assert rep["runs"]["screened"]["not_measured"] == ["X[screened_only]"]
    assert rep["runs"]["budget"]["not_measured"] == ["X[budget_exhausted]"]
    assert rep["runs"]["measured"]["not_measured"] == []
    assert {r["cohort"]["of"] for r in rep["runs"].values()} == {2}


def test_infrastructure_fault_and_not_graded_leave_the_cohort_too(agg):
    """Both are in NOT_MEASURED_STATUSES, and neither is a verdict on the submission."""
    runs = _runs(
        harness=_verdict([_row("p0", "pass"), _row("X", "infrastructure_fault")]),
        ineligible=_verdict([_row("p0", "pass"), _row("X", "not_graded")]),
        ran=_verdict([_row("p0", "pass"), _row("X", "pass", "pass")]),
    )
    rep = agg.cohort_report(runs)
    assert rep["common_cohort"] == ["p0"]
    assert rep["runs"]["harness"]["not_measured_status"] == {"X": "infrastructure_fault"}
    assert rep["runs"]["ineligible"]["not_measured_status"] == {"X": "not_graded"}


# --------------------------------------------------------------------------------------------------
# what each run did not measure has to travel WITH that run
# --------------------------------------------------------------------------------------------------


def test_the_not_measured_names_travel_with_each_run_and_carry_their_status(agg):
    """A dropped row must be visible beside the number, not silently absent from it."""
    runs = _runs(
        shut_gate=_verdict([_row("p0", "pass"), _row("M2_microvit_gemmini", "gated"), _row("SY_micro_model", "gated")]),
        open_gate=_verdict([_row("p0", "pass"), _row("M2_microvit_gemmini", "fail"), _row("SY_micro_model", "fail")]),
    )
    shut = rep_shut = agg.cohort_report(runs)["runs"]["shut_gate"]
    assert shut["not_measured"] == ["M2_microvit_gemmini[gated]", "SY_micro_model[gated]"]
    assert shut["not_measured_status"] == {"M2_microvit_gemmini": "gated", "SY_micro_model": "gated"}
    assert shut["n_not_measured"] == 2
    # and the rows its NEIGHBOUR measured while it did not, which is what shrank its denominator
    assert rep_shut["missing_vs_union"] == ["M2_microvit_gemmini", "SY_micro_model"]


def test_the_grader_supplied_not_measured_map_is_preferred_over_re_deriving_it(agg):
    """When the verdict carries the grader's own map, read THAT -- one copy of the exclusion rule."""
    v = _verdict(
        [_row("p0", "pass"), _row("X", "gated")], not_measured_status={"X": "screened_only"}
    )  # deliberately disagrees with the row
    assert agg.not_measured(v) == {"X": "screened_only"}
    assert agg.not_measured_labels(v) == ["X[screened_only]"]


def test_a_verdict_without_the_map_still_classifies_from_per_capsule(agg):
    """Every verdict already on disk predates the field; deriving must still work."""
    v = _verdict([_row("p0", "pass"), _row("X", "gated")])
    assert "not_measured_status" not in v
    assert agg.not_measured(v) == {"X": "gated"}


# --------------------------------------------------------------------------------------------------
# the easy path must not change, and the hard path must not be printable as if it were easy
# --------------------------------------------------------------------------------------------------


def test_identical_cohorts_compare_exactly_as_before(agg):
    """No behaviour change when every run measured the same rows: cohort == own, and the own ratios
    are declared comparable so a reader is not warned off a number that is fine."""
    a = _verdict([_row("p0", "pass", "pass"), _row("p1", "pass", "pass"), _row("f0", "fail")])
    b = _verdict([_row("p0", "pass", "pass"), _row("p1", "fail"), _row("f0", "fail")])
    rep = agg.cohort_report(_runs(a_run=a, b_run=b))

    assert rep["common_cohort_size"] == 3 and rep["union_measured_size"] == 3
    assert rep["own_ratios_comparable"] is True
    assert rep["comparable_metric"] == "own"
    for rid, own in (("a_run", "2/3"), ("b_run", "1/3")):
        r = rep["runs"][rid]
        assert r["own_ratio"] == own == r["cohort"]["passed_ratio"]
        assert r["own_n_capsules"] == r["cohort"]["of"] == 3
        assert r["not_measured"] == [] and r["n_not_measured"] == 0
        assert r["own_is_cohort"] is True
    table = agg.format_cohort_table(rep)
    assert "NOT COMPARABLE" not in table
    assert "per-run ONLY" not in table


def test_the_table_never_puts_two_different_denominators_side_by_side_unmarked(agg):
    """Rule of the whole change: a ratio whose denominator differs from its neighbour's may not be
    presented as comparable. The cohort columns lead; the own column is last and labelled."""
    rep = agg.cohort_report(
        _runs(open_gate=_verdict(_open_gate_rows("fail")), shut_gate=_verdict(_shut_gate_rows("gated")))
    )
    table = agg.format_cohort_table(rep)

    assert "NOT COMPARABLE" in table
    assert "per-run ONLY" in table
    header = [ln for ln in table.splitlines() if "pass/cohort" in ln and "not measured" in ln][0]
    assert header.index("pass/cohort") < header.index("own"), "comparable columns lead"
    assert "L3/cohort" in header
    # the deferred rows are named in the row itself, not only in the JSON
    assert "X[gated]" in table
    # and the non-comparable denominators are both stated
    assert "5" in table and "6" in table


def test_l3_clean_is_cohort_normalized_too(agg):
    """L3 is the metric that gets cited, so it needs the same denominator discipline as the gate."""
    a = _verdict([_row("p0", "pass", "pass"), _row("p1", "pass", "fail"), _row("X", "pass", "pass")])
    b = _verdict([_row("p0", "pass", "pass"), _row("p1", "pass", "pass"), _row("X", "gated")])
    rep = agg.cohort_report(_runs(a_run=a, b_run=b))
    assert rep["runs"]["a_run"]["cohort"]["l3_clean_ratio"] == "1/2"  # p1 passed the gate, not L3
    assert rep["runs"]["b_run"]["cohort"]["l3_clean_ratio"] == "2/2"
    # X cleared L3 in run a and must NOT be credited: run b never measured it.
    assert rep["runs"]["a_run"]["cohort"]["l3_clean"] == 1


def test_a_measured_row_that_is_not_a_pass_is_never_credited_as_one(agg):
    """`declined` and `incomplete` ARE measured -- they stay in the denominator -- and neither is a
    pass. Only `pass` counts: a declared coverage gap must not be scored as a win, and an incomplete
    measurement (a mandatory tier came back unavailable) says nothing about the submission.
    """
    a = _verdict([_row("p0", "pass", "pass"), _row("d0", "declined"), _row("i0", "incomplete")])
    b = _verdict([_row("p0", "pass", "pass"), _row("d0", "pass", "pass"), _row("i0", "fail")])
    rep = agg.cohort_report(_runs(a_run=a, b_run=b))

    assert rep["common_cohort_size"] == 3, "declined/incomplete were MEASURED -- they stay in"
    assert rep["runs"]["a_run"]["cohort"]["passed_ratio"] == "1/3"
    assert rep["runs"]["b_run"]["cohort"]["passed_ratio"] == "2/3"
    assert rep["runs"]["a_run"]["cohort"]["l3_clean_ratio"] == "1/3"


def test_an_empty_comparison_is_not_a_vacuous_pass(agg):
    """Fail closed. No runs -> no cohort, and nothing is declared comparable."""
    assert agg.common_cohort({}) == set()
    rep = agg.cohort_report({})
    assert rep["common_cohort_size"] == 0 and rep["n_runs"] == 0
    assert rep["own_ratios_comparable"] is False


def test_runs_with_no_capsule_in_common_say_so_instead_of_printing_zeros(agg):
    """Selecting across batches graded on different capsule sets gives an EMPTY cohort. Every ratio
    is then 0/0 -- correct arithmetic, worthless table -- so the reason has to be stated."""
    rep = agg.cohort_report(
        _runs(batch_a=_verdict([_row("a0", "pass", "pass")]), batch_b=_verdict([_row("b0", "pass", "pass")]))
    )
    assert rep["common_cohort_size"] == 0 and rep["union_measured_size"] == 2
    assert rep["empty_cohort_reason"] and "share no measured capsule" in rep["empty_cohort_reason"]
    table = agg.format_cohort_table(rep)
    assert "NO COMPARISON POSSIBLE" in table
    # ... and a comparison that IS possible must not carry the warning
    ok = agg.cohort_report(_runs(a=_verdict([_row("a0", "pass", "pass")]), b=_verdict([_row("a0", "fail")])))
    assert ok["empty_cohort_reason"] is None
    assert "NO COMPARISON POSSIBLE" not in agg.format_cohort_table(ok)


# --------------------------------------------------------------------------------------------------
# reading the runs off disk
# --------------------------------------------------------------------------------------------------


def _write_verdict(run_dir, name: str, verdict: dict):
    qh = run_dir / "qa_history"
    qh.mkdir(parents=True, exist_ok=True)
    p = qh / name
    p.write_text(json.dumps(verdict), encoding="utf-8")
    return p


def test_l3_evidence_carries_what_the_run_did_not_measure(agg, tmp_path):
    """A single run's record keeps its own (measured) denominator AND names the rows missing from it,
    so the shrink is visible without a second run to compare against."""
    run = tmp_path / "shut_gate_run"
    _write_verdict(
        run,
        "verdict_round_00.json",
        _verdict([_row("p0", "pass", "pass"), _row("f0", "fail"), _row("SY_micro_model", "gated")]),
    )
    ev = agg._l3_evidence(run)

    assert (ev["gate_passed"], ev["n_capsules"]) == (1, 2)  # unchanged: measured rows only
    assert ev["rtl_clean"] == 1
    assert ev["n_not_measured"] == 1
    assert ev["not_measured"] == ["SY_micro_model[gated]"]
    assert ev["not_measured_status"] == {"SY_micro_model": "gated"}


def test_l3_evidence_reports_not_measured_as_unknown_when_there_is_no_verdict(agg, tmp_path):
    """None, never zero: "no verdict" must not read as "nothing was deferred"."""
    ev = agg._l3_evidence(tmp_path / "never_graded")
    assert ev["n_capsules"] is None and ev["n_not_measured"] is None
    assert ev["not_measured"] is None and ev["not_measured_status"] is None


def test_the_fast_l2_only_snapshot_is_not_a_comparable_grade(agg, tmp_path):
    """The in-turn fast snapshot is written minutes into a run so the agent gets early feedback.
    Scoring an arm from it compares somebody's warm-up against a finished run."""
    run = tmp_path / "live_run"
    _write_verdict(run, "verdict_round_00.json", _verdict([_row("p0", "pass", "pass"), _row("p1", "pass", "pass")]))
    round0 = run / "qa_history" / "verdict_round_00.json"
    _write_verdict(
        run, "verdict_fast_900.json", _verdict([_row("p0", "fail"), _row("p1", "fail")], stage="first_grade_loop_tier")
    )
    import os

    os.utime(round0, (10**9, 10**9))  # the fast snapshot is the NEWEST on disk

    j, name = agg.latest_verdict(run, skip_stages=agg._NOT_COMPARABLE_STAGES)
    assert name == "verdict_round_00.json" and j["n_passed"] == 2
    # and the historical behaviour of "newest verdict, whatever it is" is unchanged
    j2, name2 = agg.latest_verdict(run)
    assert name2 == "verdict_fast_900.json" and j2["n_passed"] == 0


def test_a_run_with_nothing_graded_is_named_not_dropped(agg, tmp_path, monkeypatch):
    """A run that has not produced a comparable grade must be reported as excluded, with a reason --
    silently omitting it is how a comparison loses an arm."""
    root = tmp_path / "runs"
    (root / "raw_baseline" / "graded_run").mkdir(parents=True)
    (root / "raw_baseline" / "ungraded_run").mkdir(parents=True)
    _write_verdict(
        root / "raw_baseline" / "graded_run", "verdict_round_00.json", _verdict([_row("p0", "pass", "pass")])
    )
    monkeypatch.setattr(agg.C, "RUNS", root)
    monkeypatch.setattr(agg, "RUN_DIRS", ["raw_baseline"])

    runs, skipped = agg.collect_comparable_runs(("_run",))
    assert list(runs) == ["graded_run"]
    assert "ungraded_run" in skipped and "nothing graded" in skipped["ungraded_run"]


# --------------------------------------------------------------------------------------------------
# the grader side: the names are recorded, and a single run's own counts do not move
# --------------------------------------------------------------------------------------------------


def _grade(monkeypatch, results):
    """Run the REAL ``capsule_grade.grade`` over a fixed result list (package/build/oracle stubbed),
    so the score-assembly under test is the shipped one."""
    from merlin.targetgen import capsule_grade as CG

    monkeypatch.setattr(CG, "load_package", lambda *a, **k: type("P", (), {"integrity_exempt": False})())
    monkeypatch.setattr(CG, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CG, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CG.CR, "discover_capsules", lambda *a, **k: [{"name": r["capsule"]} for r in results])
    monkeypatch.setattr(CG.CR, "run_suite", lambda *a, **k: results)
    return CG.grade("pkg", capsules_root=["root"], runs_root="runs", target="t", max_workers=1)


def _op(name, status, l3=None):
    r = {"capsule": name, "kind": "op", "label": "public", "status": status}
    if status == "pass":
        r.update({"numeric": {"status": "pass"}, "trace_check": {"status": "pass"}})
    if l3 is not None:
        r["tiers"] = {"L3": l3}
    return r


def test_a_single_runs_own_denominator_and_numerator_are_unchanged(monkeypatch):
    """The per-run semantics stay exactly as they were: the deferred row is in neither bucket."""
    s = _grade(
        monkeypatch,
        [
            _op("p0", "pass", "pass"),
            _op("p1", "pass", "pass"),
            _op("f0", "fail"),
            {
                "capsule": "M0",
                "kind": "model",
                "label": "public",
                "status": "gated",
                "failure": {"plane": "gate", "category": "GATED", "detail": "op pass fraction 0.67 < gate 0.8"},
            },
        ],
    )
    assert (s["n_passed"], s["n_capsules"]) == (2, 3)
    assert s["n_gated_deferred"] == 1 and s["gated_deferred"] == ["M0"]
    assert s["functional_pass"] == 0


def test_the_score_records_every_not_measured_row_by_name_and_status(monkeypatch):
    """One place to read "what is missing from this denominator", spanning every not-measured status
    -- so an aggregator does not have to re-implement the exclusion rule over per_capsule."""
    s = _grade(
        monkeypatch,
        [
            _op("p0", "pass", "pass"),
            _op("s0", "screened_only"),
            _op("b0", "budget_exhausted"),
            _op("n0", "not_graded"),
            {
                "capsule": "M0",
                "kind": "model",
                "label": "public",
                "status": "gated",
                "failure": {"plane": "gate", "category": "GATED", "detail": "op pass fraction 0.5 < gate 0.8"},
            },
        ],
    )
    assert (s["n_passed"], s["n_capsules"]) == (1, 1)
    assert s["n_not_measured"] == 4
    assert s["not_measured"] == ["M0", "b0", "n0", "s0"]
    assert s["not_measured_status"] == {
        "M0": "gated",
        "b0": "budget_exhausted",
        "n0": "not_graded",
        "s0": "screened_only",
    }
    # the per-status lists it unions are untouched
    assert s["n_gated_deferred"] == 1 and s["n_screened_only"] == 1
    assert s["n_budget_exhausted"] == 1 and s["n_not_graded_ineligible"] == 1


def test_a_run_that_measured_everything_records_a_zero_and_no_name_list(monkeypatch):
    """Mirrors the existing per-status fields: the count is always present, the name list only when
    there is something to name."""
    s = _grade(monkeypatch, [_op("p0", "pass", "pass"), _op("f0", "fail")])
    assert s["n_not_measured"] == 0
    assert "not_measured" not in s and "not_measured_status" not in s
    assert (s["n_passed"], s["n_capsules"]) == (1, 2)


def test_the_aggregator_and_the_grader_use_ONE_definition_of_not_measured(agg):
    """Both sides import the grader's tuple. A private copy in either place would drift the moment a
    new not-measured status is added -- and drift silently, as a wrong denominator."""
    assert agg.NOT_MEASURED_STATUSES is NOT_MEASURED_STATUSES
    assert "gated" in NOT_MEASURED_STATUSES and "pass" not in NOT_MEASURED_STATUSES
