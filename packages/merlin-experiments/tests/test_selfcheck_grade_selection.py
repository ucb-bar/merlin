"""One self-check request runs the grade's own tiers and certifies, or says precisely why not."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers import simjob
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import dispatch, promotion, selfcheck

from merlin.common.paths import data_path

SUITE = "fixture-suite"
TIERS = ("L0", "L1", "L2", "L3")


def _context(root: Path) -> InvocationContext:
    return InvocationContext(
        root, root / "target.yaml", root, "fixture", root / "runs", root / "reports", root / "bundles", ()
    )


def _adapter(engine: str):
    def run(*_args):  # never executed: the grade is replaced below
        raise AssertionError("fixture adapter must not run")

    run._merlin_simulator_engine = engine
    return run


def _capsule(corpus: Path, name: str) -> None:
    capsule = corpus / name
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text(json.dumps({
        "name": name,
        "kind": "isa",
        "source_role": "handauthored_compiler_test",
        "label": "public",
        "operation": {"op": "movement"},
        "numeric_policy": {"compare": "exact_int", "dtype": "i8"},
        "expected": {"instruction_classes": []},
        "required_oracle_tiers": list(TIERS),
    }))


def _result(name: str, tiers: dict[str, str]) -> dict:
    """A capsule_result as the runner writes it: every declared tier mandatory."""
    unreached = [tier for tier, status in tiers.items() if status == "unavailable"]
    status = "pass" if all(s == "pass" for s in tiers.values()) else "incomplete" if unreached else "fail"
    result = {
        "capsule": name,
        "kind": "op",
        "status": status,
        "numeric": {"status": "pass" if tiers.get("L2") == "pass" else "skipped"},
        "trace_check": {"status": "pass", "trace_requirement_status": "pass"},
        "tiers": {tier: {"status": s, "mandatory": True} for tier, s in tiers.items()},
    }
    if unreached:
        reason = f"this capsule declares required oracle tier {unreached[0]} but this phase supplies no adapter"
        result["tiers"][unreached[0]]["reason"] = reason
        result["failure"] = {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS",
                             "tier": unreached[0], "tier_status": "unavailable", "detail": reason}
    return result


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Drive the worker's real classification over fixed capsule results, recording what it graded."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MERLIN_SELFCHECK_PROMOTION_WS", raising=False)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("{}")
    corpus = tmp_path / "public"
    for name in ("cap_certified", "cap_unreached"):
        _capsule(corpus, name)
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda *a: ("fixture", "chipyard"))
    monkeypatch.setattr(selfcheck.CR, "suite_for", lambda *a: SUITE)
    monkeypatch.setattr(selfcheck, "_log_telemetry", lambda *a: None)
    official = {"L2": _adapter("spike"), "L3": _adapter("gsim")}
    seen = SimpleNamespace(selection=[], graded_with=[])

    def adapters(sim, target, sim_via, **_kwargs):
        seen.selection.append(sim)
        if sim == selfcheck.REQUIRED_SIM:
            return dict(official), sim
        return {"L2": official["L2"]}, sim

    monkeypatch.setattr(selfcheck, "_adapters", adapters)
    monkeypatch.setattr(selfcheck.CR, "oracle_adapters", lambda *a, **k: dict(official))

    def run(argv, results):
        def grade(_submission, *, runs_root, oracle_adapters, **_kwargs):
            seen.graded_with.append(dict(oracle_adapters))
            rows = []
            for name, tiers in results.items():
                parent = Path(runs_root) / "runs" / SUITE / name
                parent.mkdir(parents=True)
                result = _result(name, {t: s for t, s in tiers.items() if t in TIERS})
                (parent / "capsule_result.json").write_text(json.dumps(result))
                rows.append({"capsule": name, "status": result["status"]})
            return {"n_capsules": len(rows), "n_passed": 0, "per_capsule": rows}

        monkeypatch.setattr(selfcheck.CG, "grade", grade)
        code = selfcheck.main(
            ["--submission", str(submission), "--capsules", ",".join(results), *argv],
            context=_context(tmp_path), capsules_root=corpus, contract=data_path("contract"),
        )
        return code

    seen.run = run
    seen.official = official
    return seen


ALL_PASS = dict.fromkeys(TIERS, "pass")
NO_RTL = {"L0": "pass", "L1": "pass", "L2": "pass", "L3": "unavailable"}


def test_the_default_request_grades_with_the_grade_adapters_and_certifies(harness, capsys):
    code = harness.run([], {"cap_certified": ALL_PASS})
    report = json.loads(capsys.readouterr().out)
    assert harness.selection == [selfcheck.REQUIRED_SIM]
    assert harness.graded_with == [harness.official], "every tier the grade selects ran in this one request"
    assert code == 0 and report["all_pass"] is True
    assert report["n_certified"] == report["n_passed"] == 1
    assert report["oracle_tiers"] == ["L2", "L3"]
    assert report["tier_engines"] == {"L2": "spike", "L3": "gsim"}
    [row] = report["per_capsule"]
    assert row["pass"] is True and row["certified"] is True
    assert row["tiers"] == ALL_PASS and row["barrier_tier"] == "L3"
    assert row["numeric_consistency_passed"] is row["numerical_execution_passed"] is True


def test_pre_execution_crash_cannot_credit_command_buffer_consistency_as_target_execution(
    harness, capsys, monkeypatch
):
    original_result = _result

    def failed_before_execution(name, tiers):
        result = original_result(name, tiers)
        result["numeric"] = {"status": "pass", "mismatch_count": 0}
        result["failure"] = {"plane": "spike", "category": "tool_crash",
                             "detail": "kernel stack-frame preflight rejected missing entrypoint"}
        return result

    monkeypatch.setitem(globals(), "_result", failed_before_execution)
    code = harness.run([], {"cap_certified": {"L0": "pass", "L1": "pass", "L2": "fail"}})
    report = json.loads(capsys.readouterr().out)
    [row] = report["per_capsule"]
    assert code == 1 and report["n_passed"] == report["n_certified"] == 0
    assert row["numeric_consistency_passed"] is True
    assert row["numerical_execution_passed"] is False
    assert row["tiers"]["L2"] == "fail" and row["failure"]["category"] == "tool_crash"


def test_a_tier_that_could_not_run_is_never_a_pass_under_the_grade_selection(harness, capsys):
    code = harness.run([], {"cap_certified": ALL_PASS, "cap_unreached": NO_RTL})
    report = json.loads(capsys.readouterr().out)
    rows = {row["capsule"]: row for row in report["per_capsule"]}
    unreached = rows["cap_unreached"]
    assert code == 1 and report["all_pass"] is False
    assert unreached["pass"] is False and unreached["certified"] is False
    assert unreached["screened_only"] is True
    assert unreached["tiers"]["L3"] == "unavailable"
    assert unreached["failure"]["category"] == "NOT_RUN_IS_NOT_PASS"
    assert report["n_passed"] == report["n_certified"] == 1
    assert report["n_screened_only"] == 1
    assert "could NOT RUN although the grade's own selection was requested" in report["note"]


def test_a_named_screen_keeps_its_legacy_accounting(harness, capsys):
    """Backwards compatibility: --sim spike still screens, and a screen still never certifies."""
    code = harness.run(["--sim", "spike"], {"cap_unreached": NO_RTL})
    report = json.loads(capsys.readouterr().out)
    [row] = report["per_capsule"]
    assert harness.selection == ["spike"] and harness.graded_with == [{"L2": harness.official["L2"]}]
    assert code == 1 and report["all_pass"] is False
    assert row["pass"] is True and report["n_certified"] == 0 and report["n_screened_only"] == 1
    assert "certified" not in row


def test_a_narrowed_grade_selection_keeps_promotion_screen_accounting(harness, capsys):
    """A promotion asks for its cert tier alone; that L3 result must still read as an L3 pass."""
    l3_only = {"L0": "pass", "L1": "pass", "L2": "unavailable", "L3": "pass"}
    code = harness.run(["--sim", selfcheck.REQUIRED_SIM, "--tiers", "L3"], {"cap_certified": l3_only})
    report = json.loads(capsys.readouterr().out)
    [row] = report["per_capsule"]
    assert harness.graded_with == [{"L3": harness.official["L3"]}]
    assert code == 1 and report["n_certified"] == 0
    assert row["pass"] is True and row["barrier_tier"] == "L3" and row["barrier_status"] == "pass"
    assert "narrowed by --tiers" in report["note"]


def test_a_named_engine_carries_the_grade_selection_record_only_when_the_grade_selects_it(monkeypatch):
    record = {"engine": "gsim", "fidelity": "elaborated_rtl", "reason": "fixture", "considered": []}
    monkeypatch.setattr(
        "merlin.targetgen.oracle_policy.chipyard_tier_plan", lambda target: SimpleNamespace(selection=record)
    )
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    built = []
    monkeypatch.setattr(selfcheck.CR, "simulator_adapter", lambda engine, target, **kw: built.append((engine, kw)))
    selfcheck._adapters("gsim", "fixture", "chipyard")
    selfcheck._adapters("verilator", "fixture", "chipyard")
    assert built == [("spike", {}), ("gsim", {"selection": record}), ("spike", {}), ("verilator", {})]


def test_the_cert_tier_is_not_bought_twice_after_a_grade_selection_check():
    assert promotion.cert_already_executed({"oracle_tiers": ["L2", "L3"]}, "L3")
    assert not promotion.cert_already_executed({"oracle_tiers": ["L2"]}, "L3")
    assert not promotion.cert_already_executed({"per_capsule": []}, "L3"), "an undeclared selection keeps promoting"
    assert not promotion.cert_already_executed(None, "L3")


def test_async_jobs_on_the_grade_selection_take_the_rtl_budget_and_admission(monkeypatch):
    from merlin.targetgen.rtl_engine_policy import capsule_worker_cap

    assert simjob._job_timeout_s(dispatch.REQUIRED_SIM, 3, 300) == 900
    assert simjob._job_timeout_s("spike", 3, 300) == simjob._SCREEN_TIMEOUT_S
    tightest = min(capsule_worker_cap(engine) for engine in simjob._rtl_engines())
    assert simjob._capsule_workers(0, sim=dispatch.REQUIRED_SIM, n_capsules=64) == tightest
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    assert not simjob._uses_verilator_slots(dispatch.REQUIRED_SIM)
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE")
    assert simjob._uses_verilator_slots(dispatch.REQUIRED_SIM), "an unpinned selection may pick Verilator"
