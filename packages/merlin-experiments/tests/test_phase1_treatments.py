"""Treatment adapters are explicit, invocation-local and source-attributed."""

from __future__ import annotations

import ast
import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import run_inputs
from merlin_experiments.phase1 import treatments as T

from merlin.common.paths import repo_root

HARNESS = repo_root() / "merlin/experiments/capsule_bench/harness"


@pytest.mark.parametrize("operation_fails", [False, True])
def test_timing_sink_failure_preserves_strict_finally_behavior(monkeypatch, operation_fails):
    times = iter([10.0, 11.23456])
    monkeypatch.setattr(T.time, "time", lambda: next(times))
    calls = []
    primary = ValueError("operation failed")

    def operation():
        if operation_fails:
            raise primary
        return "result"

    def sink(kind, elapsed):
        calls.append((kind, elapsed))
        raise RuntimeError("sink failed")

    assert T.timed(operation, "qa", None) is operation
    with pytest.raises(RuntimeError, match="sink failed") as caught:
        T.timed(operation, "qa", sink)()
    assert calls == [("qa", 1.235)]
    assert caught.value.__context__ is (primary if operation_fails else None)


def _function(filename, name, scope):
    tree = ast.parse((HARNESS / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(HARNESS / filename), "exec"), scope)
    return scope[name]


def test_fullsuite_task_callback_accepts_native_sandbox_keyword(tmp_path):
    task = tmp_path / "TASK_full.md"
    task.write_text("exact historical task\n")
    workspace, run = tmp_path / "workspace", tmp_path / "run"
    workspace.mkdir()
    run.mkdir()
    scope = {"Path": Path, "TASK_FULL": task, "shutil": shutil, "RI": run_inputs}
    build = _function("run_fullsuite.py", "_full_build", scope)
    build("raw_baseline", workspace, run, sandbox="none", bundle_dir=tmp_path / "bundle")
    assert (workspace / "TASK.md").read_bytes() == task.read_bytes()
    assert (run / "TASK.md").read_bytes() == task.read_bytes()


def test_fullsuite_repeated_invocations_do_not_accumulate_wrappers_or_timing(tmp_path, monkeypatch):
    import argparse

    calls = []
    controller = SimpleNamespace(PILOT_SUBSET=None, CR=SimpleNamespace(discover_capsules=lambda *a, **k: []))
    sentinel = object()
    controller.launch_agent = controller.qa_grade = controller._build_task = sentinel

    def invoke(argv, *, treatment):
        calls.append((list(argv), treatment))
        treatment.on_duration("agent", 1.5)
        treatment.on_duration("qa", 2.25)
        return 0

    controller.main = invoke
    run = tmp_path / "raw_baseline/run"
    run.mkdir(parents=True)
    scope = {
        "__name__": "fullsuite_under_test",
        "Path": Path,
        "sys": sys,
        "argparse": argparse,
        "C": SimpleNamespace(RUNS=tmp_path),
        "FULL_CAPSULES": tmp_path / "capsules",
        "TASK_FULL": tmp_path / "TASK_full.md",
        "Treatment": T.Treatment,
        "L": controller,
        "_full_build": lambda *a, **k: None,
        "yaml": yaml,
    }
    main = _function("run_fullsuite.py", "main", scope)
    before_argv = list(sys.argv)
    for _ in range(2):
        assert main(["--run-id", "run"]) == 0
    timing = yaml.safe_load((run / "fullsuite_agent_sim_timing.yaml").read_text())
    assert timing["agent_active_s"] == 3.0
    assert timing["sim_wait_s"] == 4.5
    assert timing["invocations"] == 2
    assert calls[0][1].on_duration is not calls[1][1].on_duration
    assert calls[0][1].capsules_root == tmp_path / "capsules"
    assert controller.PILOT_SUBSET is None
    assert controller.launch_agent is controller.qa_grade is controller._build_task is sentinel
    assert sys.argv == before_argv


@pytest.mark.parametrize("frozen", [False, True])
def test_qa_treatment_preserves_real_staging_and_persisted_failed_verdict(tmp_path, monkeypatch, frozen):
    monkeypatch.syspath_prepend(str(HARNESS))
    loop = importlib.import_module("run_baseline_qa_loop")
    grading = importlib.import_module("merlin_experiments.phase1.feedback.loop_grading")
    base = importlib.import_module("merlin_experiments.phase1.feedback.qa")
    monkeypatch.setattr(base, "run", lambda *a, **k: pytest.fail("wrong default grader"))
    monkeypatch.setattr(grading, "_language_ok", lambda *a: (True, "fixture"))
    for name in ("_write_stage_ledger", "_attach_shape_generalization", "_record_plateau"):
        monkeypatch.setattr(grading, name, lambda *a, **k: None)
    promotion = importlib.import_module("merlin_experiments.phase1.feedback.promotion")
    policy_calls = []
    monkeypatch.setattr(
        promotion, "resolve_tiers", lambda ws, **kwargs: policy_calls.append(kwargs) or (None, None, None)
    )
    ws, run, corpus = (tmp_path / name for name in ("workspace", "run", "public"))
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("fixture: true\n")
    corpus.mkdir()
    calls = []
    expected = {
        "all_pass": False,
        "n_capsules": 1,
        "n_passed": 0,
        "per_capsule": [{"status": "fail"}],
        "rtl_checks": [{"verdict": "advisory-failure"}],
    }

    def grade(submission, capsules_root, runs_root, labels, no_oracle, timeout, **kwargs):
        calls.append((submission, capsules_root, labels, no_oracle, timeout))
        assert (Path(submission) / "manifest.yaml").read_text() == "fixture: true\n"
        assert kwargs == ({"contract": tmp_path / "frozen-contract"} if frozen else {})
        return dict(expected)

    inputs = {"policy_root": tmp_path / "frozen-policy", "contract": tmp_path / "frozen-contract"} if frozen else {}
    config = grading.GradingInputs(
        loop.C.CONTEXT, "raw_baseline", corpus, (), inputs.get("contract"), inputs.get("policy_root")
    )
    result = grading.grade(ws, run, 0, True, 7, inputs=config, qa_runner=grade)
    assert policy_calls == [{"context": loop.C.CONTEXT, "capsules_root": inputs.get("policy_root")}]
    assert result["graded_at"]
    expected["graded_at"] = result["graded_at"]
    assert result == expected
    assert calls[0][1:] == (str(corpus), {"public", "dev"}, True, 7)
    assert json.loads((ws / "qa/verdict.json").read_text()) == expected
    assert json.loads((run / "qa_history/verdict_round_00.json").read_text()) == expected


def test_native_checkpoint_uses_explicit_nonempty_advisory_callback(tmp_path):
    from merlin.common.paths import module_source_path

    tree = ast.parse(module_source_path("merlin_experiments.phase1.authoring").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_verilator_grade")
    index = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_rtl_block" for t in n.targets)
    )
    seen = []

    def feedback(root, *, capsule_roots):
        seen.append((root, capsule_roots))
        return [{"verdict": "synthetic advisory", "capsule": "public-case"}]

    scope = {
        "treatment": T.Treatment(checkpoint_feedback=feedback),
        "vruns": tmp_path,
        "_roots": [str(tmp_path / "frozen")],
        "Path": Path,
    }
    code = compile(ast.Module(body=fn.body[index : index + 2], type_ignores=[]), "native-checkpoint-feedback", "exec")
    exec(code, scope)
    assert seen == [(tmp_path, (tmp_path / "frozen",))]
    assert scope["rtl_checks"] == [{"verdict": "synthetic advisory", "capsule": "public-case"}]
    scope["treatment"] = T.Treatment()
    exec(code, scope)
    assert scope["rtl_checks"] is None


def test_rtl_wrapper_has_no_global_swaps_and_dispatches_exact_callbacks(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_EXT_CHIPYARD", str(tmp_path / "unused-external-toolchain"))
    monkeypatch.syspath_prepend(str(HARNESS))
    base = importlib.import_module("qa_check")
    launcher = importlib.import_module("run_agent_experiment")
    bundles = dict(launcher.ARM_BUNDLE)
    before_argv = list(sys.argv)
    spec = importlib.util.spec_from_file_location("isolated_rtl_treatment", HARNESS / "run_rtlchecks_qa_loop.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    assert sys.modules["qa_check"] is base
    assert launcher.ARM_BUNDLE == bundles
    bundle = "merlin_assisted_rtlchecks_fixture"
    directory = tmp_path / bundle
    directory.mkdir()
    (directory / "input_bundle_manifest.yaml").write_text(
        yaml.safe_dump({"bundle_id": bundle, "arm": "merlin_rtlchecks"})
    )
    monkeypatch.setattr(wrapper, "C", SimpleNamespace(BUNDLES=tmp_path, REPO=tmp_path, CONTEXT=wrapper.C.CONTEXT))
    calls = []
    monkeypatch.setattr(wrapper.L, "main", lambda argv, **kwargs: calls.append((argv, kwargs)) or 0)
    assert wrapper.main(["--run-id", "fixture", "--bundle", bundle]) == 0
    argv, kwargs = calls[0]
    assert argv[argv.index("--bundle") + 1] == bundle
    assert argv[argv.index("--arm") + 1] == "merlin_assisted"
    assert kwargs["treatment"].qa_runner.__module__ == "merlin_experiments.phase1.feedback.rtlchecks"
    assert kwargs["treatment"].checkpoint_feedback.__module__ == "merlin_experiments.phase1.feedback.rtlchecks"
    assert sys.modules["qa_check"] is base
    assert launcher.ARM_BUNDLE == bundles
    assert sys.argv == before_argv


@pytest.mark.parametrize("all_pass", [True, False])
def test_rtl_checks_remain_advisory_in_real_wrapper(tmp_path, monkeypatch, all_pass):
    monkeypatch.setenv("MERLIN_EXT_CHIPYARD", str(tmp_path / "unused-external-toolchain"))
    monkeypatch.syspath_prepend(str(HARNESS))
    wrapper = importlib.import_module("merlin_experiments.phase1.feedback.rtlchecks")
    base = {"all_pass": all_pass, "n_capsules": 1, "n_passed": int(all_pass)}
    monkeypatch.setattr(wrapper._base, "run", lambda *args, **kwargs: dict(base))
    monkeypatch.setattr(
        wrapper, "feedback", lambda root, **kwargs: [{"verdict": "fail", "findings": [{"id": "fixture"}]}]
    )
    result = wrapper.run(
        "submission", "public", tmp_path, {"public"}, True, 1, context=SimpleNamespace(target="fixture")
    )
    assert {key: result[key] for key in base} == base
    assert result["rtl_checks"] == [{"verdict": "fail", "findings": [{"id": "fixture"}]}]
    assert "ADVISORY" in result["rtl_checks_note"]
