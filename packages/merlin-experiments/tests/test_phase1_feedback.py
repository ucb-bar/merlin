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

from merlin.common.paths import data_path, module_source_path, python_import_roots


def _context(root):
    return InvocationContext(
        root, root / "target.yaml", root, "fixture", root / "runs", root / "reports", root / "bundles", ()
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


def test_installed_sync_broker_grades_and_considers_promotion(tmp_path):
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
            _wait(lambda: bool(list(channel.glob("simreq_promo*.json"))), process=process, log=log)
            response = json.loads(next(channel.glob("resp_*.json")).read_text())
            assert response["all_pass"] is True
            assert response["selfcheck_protocol"] == 3
            assert "PRIVATE_ANSWER_SENTINEL" not in json.dumps(response)
            records = [json.loads(line) for line in observer.read_text().splitlines()]
            assert records == [{"integrity": "clean", "tiers": ["L2", "L3"], "capsules": 1, "passed": 1}]
            request = json.loads(next(channel.glob("simreq_promo*.json")).read_text())
            assert request["promoted"] is True
            assert request["submission_snapshot"]
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
