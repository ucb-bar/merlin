"""Installed-capable feedback execution; no native controller, hardware or paid agent.

The process fixture substitutes only external capsule execution and target availability.
Package validation, grade roll-up, redaction, broker protocol, promotion, source freezing
and certificate recording execute the production implementations. Synthetic tier records
are protocol fixtures, not hardware qualification.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import dispatch, qa, snapshots
from merlin_experiments.phase1.feedback import selfcheck as selfcheck_feedback

from merlin.common.paths import data_path, module_source_path, python_import_roots


def _context(root):
    return InvocationContext(
        root, root / "target.yaml", root, "fixture", root / "runs", root / "reports", root / "bundles", ()
    )


def test_legacy_model_failure_without_candidate_branch_keeps_its_failure_plane(tmp_path):
    result_path = tmp_path / "runs" / "fixture-suite" / "model_case" / "capsule_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "capsule": "model_case",
                "kind": "model",
                "status": "incomplete",
                "failure": {
                    "plane": "model",
                    "category": "NOT_RUN_IS_NOT_PASS",
                    "detail": "model runtime unavailable",
                },
                "model_execution_check": {
                    "kind": "model_accelerator_execution",
                    "candidate_native_model_check": None,
                },
            }
        )
    )

    projected = qa._per_capsule_from_results(tmp_path)["model_case"]
    assert projected["candidate_native_verification"] is None
    assert projected["failure_plane"] == "model"
    assert projected["failure_detail"] == "model runtime unavailable"


def test_entered_candidate_branch_with_missing_verification_stays_unverified(tmp_path):
    result_path = tmp_path / "runs" / "fixture-suite" / "model_case" / "capsule_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "capsule": "model_case",
                "kind": "model",
                "status": "incomplete",
                "legacy_model_diagnostic": {"status": "incomplete"},
                "model_execution_check": {
                    "kind": "model_accelerator_execution",
                    "candidate_native_model_check": None,
                },
            }
        )
    )

    projected = qa._per_capsule_from_results(tmp_path)["model_case"]
    assert projected["candidate_native_verification"] == {
        "status": "unverified",
        "reason": "candidate_verification_record_unrecognized",
    }
    assert projected["failure_plane"] == "model_execution"


def test_null_top_level_candidate_record_is_not_treated_as_legacy(tmp_path):
    result_path = tmp_path / "runs" / "fixture-suite" / "model_case" / "capsule_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "capsule": "model_case",
                "kind": "model",
                "status": "incomplete",
                "candidate_native_model_check": None,
                "model_execution_check": {"kind": "model_accelerator_execution"},
            }
        )
    )

    projected = qa._per_capsule_from_results(tmp_path)["model_case"]
    assert projected["candidate_native_verification"] == {
        "status": "unverified",
        "reason": "candidate_verification_record_unrecognized",
    }


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("native_numeric", [None, "pass", "fail"])
def test_qa_keeps_candidate_verification_separate_from_legacy_coverage(tmp_path, monkeypatch, nested, native_numeric):
    """The actual QA result-to-verdict path must not certify the runner's other program."""
    check = {
        "schema": "merlin_candidate_native_model_check_v1",
        "status": "incomplete",
        "violations": ["candidate_completed_dispatch_unverified", "candidate_required_tiers_unverified"],
        "emitted_host_compute": {"status": "clean", "detail": "PRIVATE_ANSWER_SENTINEL"},
        "source_placement": {"status": "clean"},
        "completed_dispatch": {"status": "unverified", "console": "PRIVATE_ANSWER_SENTINEL"},
        "candidate_source_coverage": {"status": "unverified"},
        "candidate_required_tiers": {
            "status": "unverified",
            "required_tiers": ["L0", "L1", "L3"],
            "tiers": {
                "L0": {"status": "pass"},
                "L1": {"status": "pass"},
                "L3": {"status": "unverified", "numeric": {"expected": "PRIVATE_ANSWER_SENTINEL"}},
            },
        },
        "native_status": "numeric_match_diagnostic",
    }
    row = {
        "capsule": "model_case",
        "status": "incomplete",
        "numeric": {"status": "pass", "mismatch_count": 0},
        "tiers": {"L3": {"status": "pass", "cycles": 999}},
        "model_execution_check": {"lowering_coverage": {"operations": 6, "on_accelerator": 6, "coverage": 1.0}},
        "failure": {
            "plane": "legacy_host_graph",
            "category": "PRIVATE_ANSWER_SENTINEL",
            "tier": "L2",
            "detail": "PRIVATE_ANSWER_SENTINEL",
        },
    }
    if native_numeric is not None:
        row["candidate_native_execution"] = {
            "numeric": {
                "status": native_numeric,
                "mismatch_count": 3 if native_numeric == "fail" else 0,
                "first_mismatch": {"expected": "PRIVATE_ANSWER_SENTINEL"},
            }
        }
    (row["model_execution_check"] if nested else row)["candidate_native_model_check"] = check
    result_path = tmp_path / "runs" / "fixture-suite" / "model_case" / "capsule_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(row))
    monkeypatch.setattr(qa, "_loop_target_sim_via", lambda context: ("fixture", ""))
    monkeypatch.setattr(qa.CR, "qa_checkpoint_adapters", lambda *a: {"L3": object()})
    monkeypatch.setattr(qa, "_emitted_cost", lambda *a: {"dram_movements": 999})
    monkeypatch.setattr(qa, "_liveness_screen", lambda *a: {"status": "ok"})
    monkeypatch.setattr(
        qa.CG,
        "grade",
        lambda *a, **k: {
            "n_capsules": 1,
            "n_passed": 0,
            "per_capsule": [
                {
                    "capsule": "model_case",
                    "label": "public",
                    "status": "incomplete",
                    "tiers": {"L3": "unverified"},
                    "cost_plane": {"status": "measured", "measured_cycles": 999},
                }
            ],
        },
    )
    verdict = qa.run("submission", str(tmp_path), tmp_path, {"public"}, False, 1, context=_context(tmp_path))
    projected = verdict["per_capsule"][0]
    feedback = projected["candidate_native_verification"]
    assert feedback["status"] == "incomplete"
    assert feedback["violations"] == check["violations"]
    assert feedback["components"]["completed_dispatch"] == "unverified"
    assert feedback["required_tiers"] == ["L0", "L1", "L3"]
    assert feedback["tiers"] == {"L0": "pass", "L1": "pass", "L3": "unverified"}
    assert feedback["source_coverage"] == {"status": "unverified"}
    assert projected["failure_tier"] == "L3"
    assert projected["failure_plane"] == "model_execution"
    assert projected["failure_category"] == "NOT_RUN_IS_NOT_PASS"
    assert projected["failure_detail"] == ", ".join(check["violations"])
    assert projected["numeric_status"] == native_numeric
    assert projected["mismatch_count"] == (None if native_numeric is None else 3 if native_numeric == "fail" else 0)
    assert projected["tiers"] == feedback["tiers"]
    assert projected["tier_cycles"] == {}
    assert "cost_plane" not in projected
    rich = qa._per_capsule_from_results(tmp_path)["model_case"]
    assert rich["emitted_cost"] is None
    assert rich["liveness"] is None
    assert "placement_coverage" not in projected
    assert "PRIVATE_ANSWER_SENTINEL" not in json.dumps(verdict)
    assert not verdict["all_pass"]
    from merlin_experiments.phase1.feedback import promotion

    monkeypatch.setattr(promotion, "execution_digest", lambda *a: "f" * 64)
    assert qa._execution_digest_from_result(result_path) is None


def test_candidate_verification_feedback_only_exposes_verified_counts(tmp_path):
    check = {
        "schema": "merlin_candidate_native_model_check_v1",
        "status": "pass",
        "violations": [],
        "candidate_source_coverage": {
            "status": "verified",
            "n_source_operations": 25,
            "n_eligible": 6,
            "n_completed_eligible": 6,
            "golden": "PRIVATE_ANSWER_SENTINEL",
            "source_sha256": "PRIVATE_ANSWER_SENTINEL",
        },
    }
    record = {"candidate_native_model_check": check}
    summary = qa._candidate_native_feedback(record)
    assert summary["source_coverage"] == {
        "status": "verified",
        "n_source_operations": 25,
        "n_eligible": 6,
        "n_completed_eligible": 6,
    }
    assert "PRIVATE_ANSWER_SENTINEL" not in json.dumps(summary)
    for bad_count in (True, -1, 3, 7, "PRIVATE_ANSWER_SENTINEL"):
        check["candidate_source_coverage"]["n_completed_eligible"] = bad_count
        assert qa._candidate_native_feedback(record)["source_coverage"] == {"status": "unverified"}
    assert qa._candidate_native_feedback({}) is None
    for malformed in (None, [], {"schema": "PRIVATE_ANSWER_SENTINEL"}):
        summary = qa._candidate_native_feedback({"candidate_native_model_check": malformed})
        assert summary["status"] == "unverified"
        assert "PRIVATE_ANSWER_SENTINEL" not in json.dumps(summary)
    check["violations"] = ["PRIVATE_ANSWER_SENTINEL"]
    check["status"] = ["PRIVATE_ANSWER_SENTINEL"]
    check["completed_dispatch"] = {"status": ["PRIVATE_ANSWER_SENTINEL"]}
    summary = qa._candidate_native_feedback(record)
    assert summary["status"] == "unverified"
    assert "PRIVATE_ANSWER_SENTINEL" not in json.dumps(summary)


@pytest.mark.parametrize(
    "code", ["output_writer_source_result_ownership_unverified", "PRIVATE_ANSWER_SENTINEL", None, []]
)
def test_candidate_native_binding_refusal_survives_closed_selfcheck(tmp_path, code):
    check = {
        "schema": "merlin_candidate_native_model_check_v1",
        "status": "incomplete",
        "violations": ["candidate_full_model_native_unverified"],
    }
    result = {
        "capsule": "model_case",
        "kind": "model",
        "status": "incomplete",
        "candidate_native_model_check": check,
        "candidate_native_execution": {
            "schema": "merlin_candidate_native_model_execution_v1",
            "status": "incomplete",
            "failure": {"type": "NativeModelExecutionError", "code": code, "detail": "PRIVATE_ANSWER_SENTINEL"},
        },
    }
    result_path = tmp_path / "runs" / "fixture-suite" / "model_case" / "capsule_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(result))
    projected = qa._per_capsule_from_results(tmp_path)["model_case"]
    row, certified = selfcheck_feedback._candidate_selfcheck_row(
        result, projected, name="model_case", barrier_tier="L3"
    )
    assert not certified
    assert "PRIVATE_ANSWER_SENTINEL" not in json.dumps(row)
    summary = row["candidate_native_verification"]
    if code == "output_writer_source_result_ownership_unverified":
        assert summary["native_failure"] == {"stage": "source_binding", "code": code}
    else:
        assert "native_failure" not in summary


def test_candidate_verified_numeric_mismatch_has_closed_failure_feedback(tmp_path):
    check = {
        "schema": "merlin_candidate_native_model_check_v1",
        "status": "fail",
        "violations": ["candidate_verified_numeric_mismatch"],
        "candidate_required_tiers": {
            "status": "fail",
            "required_tiers": ["L2", "L3"],
            "tiers": {"L2": {"status": "fail"}, "L3": {"status": "pass"}},
        },
        "native_status": "numeric_match_diagnostic",
    }
    projected = qa._candidate_native_feedback({"candidate_native_model_check": check})
    assert projected["status"] == "fail"
    assert projected["violations"] == ["candidate_verified_numeric_mismatch"]
    assert projected["tiers"] == {"L2": "fail", "L3": "pass"}
    result_path = tmp_path / "runs" / "synthetic-suite" / "M" / "capsule_result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "capsule": "M",
                "kind": "model",
                "status": "fail",
                "numeric": {"status": "pass"},  # unrelated legacy graph
                "candidate_native_execution": {"numeric": {"status": "fail", "mismatch_count": 1}},
                "candidate_native_model_check": check,
            }
        )
    )
    row = qa._per_capsule_from_results(tmp_path)["M"]
    assert row["numeric_status"] == "fail"
    assert row["failure_plane"] == "candidate_model_numeric"
    assert row["failure_category"] == "FUNCTIONAL_MISMATCH"


def test_model_only_selfcheck_reports_gate_instead_of_harness_failure():
    score = {
        "per_capsule": [
            {
                "capsule": "model_case",
                "kind": "model",
                "status": "gated",
                "gate_reason": "op pass fraction 0.00 < gate 0.8",
            }
        ]
    }
    rows = selfcheck_feedback._gated_without_result_rows(
        score, requested={"model_case"}, models={"model_case"}, represented=set()
    )
    assert rows == [
        {"capsule": "model_case", "pass": False, "status": "gated", "reason": "op pass fraction 0.00 < gate 0.8"}
    ]


def test_mixed_selfcheck_retains_unexecuted_model_gate_beside_operation_result():
    score = {
        "per_capsule": [
            {"capsule": "operation_case", "status": "pass"},
            {
                "capsule": "model_case",
                "kind": "model",
                "status": "gated",
                "gate_reason": "operation evidence below gate",
            },
        ]
    }
    gated = selfcheck_feedback._gated_without_result_rows(
        score, requested={"operation_case", "model_case"}, models={"model_case"}, represented={"operation_case"}
    )
    assert gated == [
        {"capsule": "model_case", "pass": False, "status": "gated", "reason": "operation evidence below gate"}
    ]
    counts = selfcheck_feedback._selfcheck_counts(
        result_rows=1, gated_rows=len(gated), passed=1, certified=1, requested_size=2, suite_size=2, scope="all"
    )
    assert counts == {
        "n_capsules": 2,
        "n_result_rows": 1,
        "n_gated": 1,
        "n_unchecked": 1,
        "n_unknown": 0,
        "certified_complete": False,
        "all_pass": False,
        "n_passed": 1,
        "n_certified": 1,
    }
    subset = selfcheck_feedback._selfcheck_counts(
        result_rows=1, gated_rows=len(gated), passed=1, certified=1, requested_size=2, suite_size=3, scope="subset"
    )
    assert subset["n_unknown"] == 1
    assert subset["n_unchecked"] == 2


def test_selfcheck_ungated_count_projection_retains_existing_pass_semantics():
    counts = selfcheck_feedback._selfcheck_counts(
        result_rows=1, gated_rows=0, passed=1, certified=1, requested_size=1, suite_size=1, scope="all"
    )
    assert counts["n_capsules"] == counts["n_result_rows"] == 1
    assert counts["n_gated"] == counts["n_unchecked"] == counts["n_unknown"] == 0
    assert counts["all_pass"] and counts["certified_complete"]


@pytest.mark.parametrize(
    "rows,requested,represented",
    [
        ([{"kind": "model", "status": "gated", "gate_reason": "deferred"}], {"model_case"}, set()),
        (
            [{"capsule": "model_case", "kind": "model", "status": "gated", "gate_reason": "deferred"}],
            {"operation_case"},
            set(),
        ),
        (
            [
                {"capsule": "model_case", "kind": "model", "status": "gated", "gate_reason": "deferred"},
                {"capsule": "model_case", "kind": "model", "status": "gated", "gate_reason": "deferred"},
            ],
            {"model_case"},
            set(),
        ),
        (
            [{"capsule": "model_case", "kind": "model", "status": "gated", "gate_reason": "deferred"}],
            {"model_case"},
            {"model_case"},
        ),
        (
            [{"capsule": "model_case", "kind": "operation", "status": "gated", "gate_reason": "deferred"}],
            {"model_case"},
            set(),
        ),
    ],
)
def test_selfcheck_rejects_unbound_or_inconsistent_score_gates(rows, requested, represented):
    with pytest.raises(ValueError):
        selfcheck_feedback._gated_without_result_rows(
            {"per_capsule": rows}, requested=requested, models=requested & {"model_case"}, represented=represented
        )


def test_selfcheck_rejects_score_claiming_operation_capsule_is_gated_model():
    with pytest.raises(ValueError):
        selfcheck_feedback._gated_without_result_rows(
            {"per_capsule": [{"capsule": "unit_case", "kind": "model", "status": "gated", "gate_reason": "deferred"}]},
            requested={"unit_case"},
            models=set(),
            represented=set(),
        )


def test_selfcheck_rejects_result_missing_from_grader_score():
    with pytest.raises(ValueError):
        selfcheck_feedback._gated_without_result_rows(
            {"per_capsule": []},
            requested={"operation_case"},
            models=set(),
            represented={"operation_case"},
        )


@pytest.mark.parametrize(
    "result_rows,gated_rows,requested_size,suite_size",
    [(1, 0, 2, 2), (1, 0, 1, 0)],
)
def test_selfcheck_missing_or_unknown_requested_cohort_cannot_pass(result_rows, gated_rows, requested_size, suite_size):
    counts = selfcheck_feedback._selfcheck_counts(
        result_rows=result_rows,
        gated_rows=gated_rows,
        passed=1,
        certified=1,
        requested_size=requested_size,
        suite_size=suite_size,
        scope="all",
    )
    assert counts["all_pass"] is False
    assert counts["certified_complete"] is False


def test_selfcheck_rejects_more_reported_rows_than_requested_capsules():
    with pytest.raises(ValueError):
        selfcheck_feedback._selfcheck_counts(
            result_rows=2, gated_rows=0, passed=2, certified=2, requested_size=1, suite_size=2, scope="all"
        )


_HOST = r"""
import importlib.abc, json, os, runpy, sys
from pathlib import Path
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"_common", "qa_check", "tier_promote", "agent_selfcheck",
                        "selfcheck_broker", "simjob_broker", "run_baseline_qa_loop"}:
            raise AssertionError("native fallback: " + fullname)
sys.meta_path.insert(0, NoNative())
from merlin.targetgen import capsule_grade as CG, capsule_runner as CR
from merlin.common import provenance
CR.oracle_adapters = lambda *args, **kw: {} if os.environ.get("NO_ORACLE") else {"L2": object(), "L3": object()}
CR.qa_loop_adapters = lambda *args, **kw: {} if os.environ.get("NO_ORACLE") else {"L2": object()}
CR._rtl_tiers_of = lambda target: ()
CR.suite_for = lambda target: target + "-capsule-bench"
provenance.load_pins = lambda: {}
def external_execution(caps, package_dir, *, runs_root, oracle_adapters, target, **kwargs):
    rows = []
    for cap in caps:
        tiers = {tier: {"status": "pass", "derived_from_rtl": False} for tier in oracle_adapters}
        missing = not tiers
        row = {"capsule": cap["name"], "label": "public", "kind": "isa",
               "status": "incomplete" if missing else "pass", "tiers": tiers,
               "highest_tier": max(tiers) if tiers else None,
               "numeric": {"status": "pass", "mismatch_count": 0,
                           "first_mismatch": {"index": 0, "observed": 7, "expected": "PRIVATE_ANSWER_SENTINEL"}},
               "trace_check": {"status": "pass", "violations": []}}
        if missing:
            row["failure"] = {"plane": "oracle", "category": "NOT_RUN_IS_NOT_PASS",
                              "tier": "L2", "tier_status": "unavailable", "detail": "no fixture oracle"}
        result = Path(runs_root) / "runs" / CR.suite_for(target) / cap["name"]
        result.mkdir(parents=True)
        (result / "capsule_result.json").write_text(json.dumps(row))
        rows.append(row)
    return rows
CR.run_suite = external_execution
real_grade = CG.grade
def observed_grade(*args, **kwargs):
    score = real_grade(*args, **kwargs)
    with open(os.environ["GRADE_OBSERVER"], "a") as f:
        f.write(json.dumps({"integrity": score["integrity_status"], "tiers": sorted(kwargs["oracle_adapters"]),
                            "capsules": score["n_capsules"], "passed": score["n_passed"]}) + "\n")
    return score
CG.grade = observed_grade
role = sys.argv.pop(1)
if role == "worker":
    sys.argv[0] = "merlin_experiments.phase1.feedback.selfcheck"
    runpy.run_module(sys.argv[0], run_name="__main__")
else:
    from merlin_experiments.phase1.brokers import simjob, selfcheck
    broker = simjob if role == "simjob" else selfcheck
    real_command = broker.worker_command
    def fixture_worker(*args):
        command = real_command(*args)
        assert command[:3] == [sys.executable, "-m", "merlin_experiments.phase1.feedback.selfcheck"]
        return [sys.executable, __file__, "worker"] + command[3:]
    broker.worker_command = fixture_worker
    raise SystemExit(broker.main())
"""


def _inputs(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "phase1_feedback_fixtures", Path(__file__).with_name("phase1_feedback_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.inputs(tmp_path, host_program=_HOST)


def _wait(predicate, *, process, log, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if process.poll() is not None:
            pytest.fail(f"broker exited {process.returncode}: {log.read_text()}")
        time.sleep(0.03)
    pytest.fail(f"broker did not complete: {log.read_text()}")


@pytest.mark.parametrize("missing_oracle", [False, True])
def test_installed_async_broker_grades_and_records_promoted_source(tmp_path, missing_oracle):
    ws, corpus, descriptor, host, env, observer = _inputs(tmp_path)
    if missing_oracle:
        env["NO_ORACLE"] = "1"
    log = tmp_path / "broker.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                str(host),
                "simjob",
                "--descriptor",
                str(descriptor),
                "--repo",
                str(tmp_path),
                "--capsules-root",
                str(corpus),
                "--contract",
                str(data_path("contract")),
                "--ws",
                str(ws),
                "--poll",
                ".02",
                "--per-capsule-timeout",
                "10",
            ],
            cwd=tmp_path,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        try:
            channel = ws / ".qa_channel"
            _wait(channel.is_dir, process=process, log=log)
            # Public standalone client, not a handcrafted request, owns protocol serialization.
            client = subprocess.run(
                [
                    sys.executable,
                    str(ws / "simjob.py"),
                    "submit",
                    "--sim",
                    "contract",
                    "--capsules",
                    "A",
                    "--workers",
                    "1",
                ],
                cwd=ws,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert client.returncode == 0, client.stderr + client.stdout
            _wait(lambda: bool(list(channel.glob("simresp_*.json"))), process=process, log=log)
            first = json.loads(next(channel.glob("simresp_*.json")).read_text())
            assert not first.get("error"), first
            if not missing_oracle:

                def certified():
                    path = ws / "qa/tier_state.json"
                    return (
                        path.exists()
                        and '"status": "pass"' in path.read_text()
                        and bool(list(channel.glob("simreq_promo*.json")))
                        and len(list(channel.glob("simresp_*.json"))) >= 2
                    )

                _wait(certified, process=process, log=log)
            reports = [json.loads(p.read_text()) for p in channel.glob("simresp_*.json")]
            assert all("PRIVATE_ANSWER_SENTINEL" not in json.dumps(r) for r in reports)
            assert observer.is_file(), log.read_text()
            observed = [json.loads(line) for line in observer.read_text().splitlines()]
            assert all(row["integrity"] == "clean" for row in observed), observed
            if missing_oracle:
                assert not any(r.get("all_pass") for r in reports)
                assert not list(channel.glob("simreq_promo*.json"))
            else:
                assert {tuple(row["tiers"]) for row in observed} >= {("L2", "L3"), ("L3",)}
                assert all(row["capsules"] == 1 for row in observed)
                assert all(r.get("all_pass") for r in reports), reports
                assert (ws / "submission/compiler.py").read_text() == "print('synthetic compiler')\n"
        finally:
            (ws / ".qa_channel").mkdir(exist_ok=True)
            (ws / ".qa_channel/STOP").write_text("stop")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        assert process.returncode == 0, log.read_text()


def test_qa_context_keeps_missing_and_disjoint_actual_inventory_distinct(tmp_path, monkeypatch):
    context = _context(tmp_path)
    monkeypatch.setattr(qa, "_loop_target_sim_via", lambda context: ("fixture", ""))
    monkeypatch.setattr(qa.CR, "qa_checkpoint_adapters", lambda *a: {})
    monkeypatch.setattr(qa.CR, "oracle_adapters", lambda *a: {"L3": object()})
    with pytest.raises(SystemExit, match="no declared tier is reachable"):
        qa.run("submission", str(tmp_path), tmp_path, {"public"}, False, 1, context=context)
    monkeypatch.setattr(qa.CR, "oracle_adapters", lambda *a: {})
    marker = RuntimeError("actual grader invoked with no adapters")

    def grade(*args, **kwargs):
        assert kwargs["oracle_adapters"] == {}
        raise marker

    monkeypatch.setattr(qa.CG, "grade", grade)
    with pytest.raises(RuntimeError, match="actual grader invoked"):
        qa.run("submission", str(tmp_path), tmp_path, {"public"}, False, 1, context=context)


@pytest.mark.parametrize("loop_only", [True, False], ids=["loop-tier-request", "cert-tier-already-ran"])
def test_installed_sync_broker_grades_and_considers_promotion(tmp_path, loop_only):
    ws, corpus, descriptor, host, env, observer = _inputs(tmp_path)
    shutil.copyfile(module_source_path("merlin_experiments.phase1.tools.selfcheck"), ws / "agent_selfcheck.py")
    log = tmp_path / "sync.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                str(host),
                "selfcheck",
                "--descriptor",
                str(descriptor),
                "--repo",
                str(tmp_path),
                "--capsules-root",
                str(corpus),
                "--contract",
                str(data_path("contract")),
                "--ws",
                str(ws),
                "--poll",
                ".02",
            ],
            cwd=tmp_path,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        channel = ws / ".qa_channel"
        try:
            _wait(channel.is_dir, process=process, log=log)
            client = subprocess.run(
                [
                    sys.executable,
                    str(ws / "agent_selfcheck.py"),
                    "--sim",
                    "spike",
                    # The loop tier alone is what promotion serves. A check that already ran the cert
                    # tier has nothing to promote: buying it again is the duplicate this guards.
                    *(["--tiers", "L2"] if loop_only else []),
                    "--capsules",
                    "A",
                    "--workers",
                    "1",
                    "--timeout",
                    "5",
                ],
                cwd=ws,
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert client.returncode == 0, client.stderr + client.stdout + log.read_text()
            tiers = ["L2"] if loop_only else ["L2", "L3"]
            if loop_only:
                _wait(lambda: bool(list(channel.glob("simreq_promo*.json"))), process=process, log=log)
            else:
                _wait(lambda: "[promote] not enqueued" in log.read_text(), process=process, log=log)
            response = json.loads(next(channel.glob("resp_*.json")).read_text())
            assert response["all_pass"] is True
            assert response["selfcheck_protocol"] == 3
            assert response["oracle_tiers"] == tiers
            assert "PRIVATE_ANSWER_SENTINEL" not in json.dumps(response)
            records = [json.loads(line) for line in observer.read_text().splitlines()]
            assert records == [{"integrity": "clean", "tiers": tiers, "capsules": 1, "passed": 1}]
            if loop_only:
                request = json.loads(next(channel.glob("simreq_promo*.json")).read_text())
                assert request["promoted"] is True
                assert request["submission_snapshot"]
            else:
                assert not list(channel.glob("simreq_promo*.json"))
        finally:
            channel.mkdir(exist_ok=True)
            (channel / "STOP").write_text("stop")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        assert process.returncode == 0, log.read_text()


@pytest.mark.parametrize("module", ["feedback.qa", "feedback.selfcheck", "brokers.selfcheck", "brokers.simjob"])
def test_import_and_help_do_not_initialize_native_target(tmp_path, module):
    script = """
import importlib.abc, os, runpy, subprocess, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"_common", "run_baseline_qa_loop", "qa_check", "tier_promote"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoNative())
def refused(*args, **kwargs):
    raise AssertionError("help/import launched a process")
subprocess.run = subprocess.Popen = refused
before = dict(os.environ)
sys.argv = [sys.argv[1], "--help"]
try:
    runpy.run_module(sys.argv[0], run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("help did not exit")
assert dict(os.environ) == before
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(map(str, python_import_roots())),
        MERLIN_TARGET_EXPERIMENT=str(tmp_path / "unreadable.yaml"),
        MERLIN_REPO_ROOT=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-c", script, "merlin_experiments.phase1." + module],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--descriptor" in result.stdout


@pytest.mark.parametrize("token", ["", "../private", "A" * 32, "0" * 31, "0" * 33, "0" * 32 + "\n", None])
def test_snapshot_token_rejects_noncanonical_paths(tmp_path, token):
    assert snapshots.promotion_snapshot_path(tmp_path, token) is None


def test_dispatch_keeps_neutral_contract_selection_without_metadata_guess(tmp_path):
    context = _context(tmp_path)
    context.descriptor.write_text("target: fixture\n")
    assert dispatch.allowed_sims(context) == ("contract",)
    assert dispatch.cert_sim("L3", context=context) == "contract"


@pytest.mark.parametrize("guard_pass", [0, 1])
def test_qa_preserves_device_score_and_separate_guard_counts(tmp_path, monkeypatch, guard_pass):
    monkeypatch.setattr(qa, "_loop_target_sim_via", lambda context: ("fixture", ""))
    monkeypatch.setattr(qa.CR, "qa_checkpoint_adapters", lambda *a: {"L2": object()})
    rows = [
        dict(
            capsule="device",
            label="public",
            status="pass",
            lane="device",
            executed_instructions=7,
            cohort="public",
            scored=True,
            tiers={"L2": "pass"},
        ),
        dict(
            capsule="guard",
            label="public",
            status="pass" if guard_pass else "fail",
            lane="host",
            executed_instructions=0,
            cohort="guard",
            scored=False,
            tiers={"L2": "pass"},
        ),
    ]
    monkeypatch.setattr(
        qa.CG,
        "grade",
        lambda *a, **k: dict(
            n_capsules=1,
            n_passed=1,
            per_capsule=rows,
            device_lane_scored_pass=1,
            device_evidence_count=1,
            host_guard_pass=guard_pass,
            host_guard_total=1,
            structural_evidence_all_pass=True,
            # The guard is an admitted capsule outside the scored denominator.
            cohort_admission={
                "n_admitted_capsules": 2,
                "admitted_name_set_sha256": qa.CG._name_set_sha256(["device", "guard"]),
            },
        ),
    )
    verdict = qa.run("submission", str(tmp_path), tmp_path, {"public"}, False, 1, context=_context(tmp_path))
    assert verdict["n_passed"] == 1 and verdict["n_capsules"] == 1
    assert verdict["original_cohort_complete"] == bool(guard_pass)
    assert verdict["host_guard_pass"] == guard_pass and verdict["host_guard_total"] == 1
    assert verdict["device_lane_scored_pass"] == 1
    assert verdict["all_pass"] == bool(guard_pass)
    assert [(r["lane"], r["executed_instructions"], r["cohort"]) for r in verdict["per_capsule"]] == [
        ("device", 7, "public"),
        ("host", 0, "guard"),
    ]
