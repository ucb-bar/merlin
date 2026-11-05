"""Live workflow feedback uses real persisted evidence, without agents or capsule grades."""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import authoring_conformance as AC
from merlin_experiments.phase1 import conformance, session
from merlin_experiments.phase1.options import parse_options

from merlin.common.paths import module_source_path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    workspace, run = tmp_path / "workspace", tmp_path / "private-run"
    (workspace / "submission").mkdir(parents=True)
    (run / "rounds").mkdir(parents=True)
    monkeypatch.setattr(conformance, "_vendored_xdsl_hashes", lambda: frozenset())
    return workspace, run


def _append(run, command, result="ok", *, name="bash", round_index=0, inp=None):
    path = run / "rounds" / f"round_{round_index:02d}.transcript.jsonl"
    identity = str(time.monotonic_ns())
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": identity,
                        "name": name,
                        "input": inp or {"command": command},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": identity,
                        "content": result,
                        "is_error": False,
                    }
                ]
            },
        },
    ]
    with path.open("a") as stream:
        stream.write("".join(json.dumps(event) + "\n" for event in events))
    return path


def _read(workspace):
    return json.loads((workspace / "qa/workflow_conformance.json").read_text())


def _until(workspace, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = _read(workspace)
        if predicate(value):
            return value
        time.sleep(0.01)
    pytest.fail("live advisory did not reflect current persisted evidence")


def test_real_thread_reports_regex_and_recovery_before_launch_finishes(workspace):
    ws, run = workspace
    backend = ws / "submission/backend.py"
    backend.write_text("import re\nre.compile('example')\n")
    transcript = _append(run, "python agent_selfcheck.py --capsules all", '{"n_capsules": 1, "n_passed": 0}')
    with AC.live_feedback(ws, run, 0, "merlin_assisted", "", lambda: ("xdsl_kit",), interval_s=0.01):
        initial = _read(ws)
        assert initial["running"] and initial["checks"]["no_regex_ok"] is False
        assert initial["regex_hits"] == [{"file": "backend.py", "line": 2}]
        assert initial["checks"]["full_selfcheck"] is True
        backend.write_text("import ast\n")
        fixed = _until(ws, lambda value: value["conformant"])
        formal = conformance.compute(transcript, backend.parent, "merlin_assisted", "", resolved_tools=("xdsl_kit",))
        assert fixed["checks"] == formal["checks"]
        assert fixed["regex_hits"] == []
    assert _read(ws)["running"] is False
    assert not any(thread.name == "workflow-conformance-feedback" for thread in threading.enumerate())
    assert not list((ws / "qa").glob(".workflow-*"))


def test_discovery_after_first_mutation_stays_false_in_live_and_formal_gate(workspace):
    ws, run = workspace
    (ws / "submission/backend.py").write_text("import ast\n")
    tools = ("xdsl_kit", "cca_spine", "rtl_generators", "rtl_facts")
    transcript = _append(run, "", name="write_file", inp={"path": "submission/backend.py", "content": "import ast"})
    with AC.live_feedback(ws, run, 0, "merlin_assisted", "", lambda: tools, interval_s=0.01):
        assert _read(ws)["checks"]["arm4_discovery_before_submission_mutation"] is False
        _append(run, "python cca_contract.py check-bijection fixture")
        _append(run, "python action_catalog.py escalation-ladder fixture")
        _append(
            run,
            'python -c "from rtl_backend import derived_levers, target_profile; '
            "derived_levers(target_profile('fixture'))\"",
            "['spatial.dataflow']",
        )
        _append(
            run,
            "python -c \"from merlin.targetgen.rtl.facts import load_facts; load_facts('fixture')\"",
            '{"target":"fixture","arrays":[{}]}',
        )
        _append(
            run,
            'python -c "from generate import xdsl; xdsl.generate(plan)"',
            "[Artifact(relpath='dialect.py', content='...')]",
        )
        current = _until(ws, lambda value: value["checks"]["scaffold_generators_used"] is True)
        assert current["checks"]["arm4_discovery_before_submission_mutation"] is False
    formal = conformance.compute(transcript, ws / "submission", "merlin_assisted", "", resolved_tools=tools)
    assert formal["checks"] == _read(ws)["checks"]
    assert formal["conformant"] is False


def test_scanner_errors_are_private_and_never_publish_a_pass(workspace):
    ws, run = workspace
    (ws / "submission/backend.py").write_text("def SECRET_PRIVATE_ERROR(:\n")
    with AC.live_feedback(ws, run, 0, "merlin_assisted", "", lambda: ("xdsl_kit",)):
        value = _read(ws)
        assert value["available"] is False and value["conformant"] is False
        assert value["checks"]["no_regex_ok"] is False
        text = (ws / "qa/workflow_conformance.json").read_text()
        assert str(run) not in text and str(ws) not in text and "Error" not in text
    assert "regex scan failed" in (run / "workflow_conformance_errors.jsonl").read_text()
    assert (run / "workflow_conformance_errors.jsonl").stat().st_mode & 0o077 == 0


def test_projection_omits_raw_evidence_and_paths_outside_submission(workspace, monkeypatch):
    ws, run = workspace
    private = "PRIVATE_SENTINEL /operator/private/hidden_answers.yaml"
    (ws / "submission/escape.py").symlink_to(run / "secret.py")
    monkeypatch.setattr(
        conformance,
        "compute",
        lambda *args, **kwargs: {
            "conformant": False,
            "checks": {"no_regex_ok": False},
            "regex_scan_error": private,
            "tool_evidence": {"raw": private},
            "transcript": private,
            "regex_hits": [
                {"file": "/private/x.py", "line": 1},
                {"file": "../outside.py", "line": 2},
                {"file": "escape.py", "line": 3},
                {"file": "backend.py", "line": 4, "kind": private},
            ],
        },
    )
    with AC.live_feedback(ws, run, 0, "merlin_assisted", "", lambda: ("xdsl_kit",)):
        value = _read(ws)
        assert value["regex_hits"] == [{"file": "backend.py", "line": 4}]
        assert "PRIVATE" not in json.dumps(value)
    assert private in (run / "workflow_conformance_errors.jsonl").read_text()


def test_resolved_tool_error_clears_previous_available_feedback(workspace):
    ws, run = workspace
    (ws / "submission/backend.py").write_text("import ast\n")
    broken = [False]

    def tools():
        if broken[0]:
            raise RuntimeError("PRIVATE_TOOL_SELECTION")
        return ("xdsl_kit",)

    with AC.live_feedback(ws, run, 0, "merlin_assisted", "", tools, interval_s=0.01):
        assert _read(ws)["available"] is True
        broken[0] = True
        assert _until(ws, lambda value: not value["available"])["conformant"] is False
    assert "PRIVATE_TOOL_SELECTION" not in json.dumps(_read(ws))


def test_unchanged_polls_reuse_advisory_but_final_flush_recomputes(workspace, monkeypatch):
    ws, run = workspace
    checks, polls = [], []

    def compute(*args, **kwargs):
        checks.append(kwargs["resolved_tools"])
        return {"conformant": True, "checks": {"test": True}}

    def tools():
        polls.append(1)
        return ("xdsl_kit",)

    monkeypatch.setattr(conformance, "compute", compute)
    with AC.live_feedback(ws, run, 0, "merlin_assisted", "", tools, interval_s=0.01):
        _until(ws, lambda value: len(polls) >= 4)
        assert len(checks) == 1
    assert len(checks) == 2  # The terminal view never relies on advisory reuse.
    AC.workflow_conformance(run / "rounds/round_00.transcript.jsonl", ws / "submission", "merlin_assisted", "", ())
    assert len(checks) == 3  # Independent formal consumers remain uncached.


def test_live_reuse_invalidates_on_transcript_membership_tools_and_publication(workspace, monkeypatch):
    ws, run = workspace
    checks = []
    selected = ["xdsl_kit"]

    def compute(*args, **kwargs):
        checks.append(1)
        return {"conformant": True, "checks": {"test": True}}

    monkeypatch.setattr(conformance, "compute", compute)
    with AC.live_feedback(ws, run, 0, "merlin_assisted", "", lambda: selected, interval_s=0.01):
        for mutate in (
            lambda: _append(run, "echo test"),
            lambda: (ws / "submission/new.py").write_text("pass\n"),
            lambda: (ws / "submission/new.py").unlink(),
            lambda: selected.append("rtl_facts"),
            lambda: (ws / "qa/workflow_conformance.json").unlink(),
        ):
            count = len(checks)
            mutate()
            deadline = time.monotonic() + 5
            while len(checks) <= count and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(checks) > count
            _until(ws, lambda value: value["available"])


@pytest.mark.parametrize("linked", [False, True])
def test_live_reuse_detects_same_size_edit_with_restored_mtime(workspace, linked):
    ws, run = workspace
    source = ws / "submission/backend.py"
    if linked:
        target = ws / "linked-source.py"
        target.touch()
        source.symlink_to(target)
    forbidden = "import re\nre.compile('x')\n"
    source.write_text("import os\n".ljust(len(forbidden)))
    with AC.live_feedback(ws, run, 0, "merlin_assisted", "", lambda: ("xdsl_kit",), interval_s=0.01):
        assert _read(ws)["checks"]["no_regex_ok"] is True
        original = source.stat()
        time.sleep(0.02)  # Separate filesystem timestamp ticks before restoring mtime.
        source.write_text(forbidden)
        os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))
        updated = _until(ws, lambda value: value["checks"]["no_regex_ok"] is False)
        assert bool(updated["regex_hits"]) is not linked  # Escaping paths remain private.


def test_live_reuse_is_bounded_even_if_metadata_does_not_change(workspace, monkeypatch):
    ws, run = workspace
    checks = []

    def compute(*args, **kwargs):
        checks.append(1)
        return {"conformant": True, "checks": {"test": True}}

    monkeypatch.setattr(conformance, "compute", compute)
    monkeypatch.setattr(AC, "_MAX_ADVISORY_REUSE_S", 0.02)
    with AC.live_feedback(ws, run, 0, "merlin_assisted", "", lambda: (), interval_s=0.01):
        _until(ws, lambda value: len(checks) >= 3)


def test_live_reuse_retries_errors_and_inputs_changed_during_check(workspace, monkeypatch):
    ws, run = workspace
    checks = []

    def compute(*args, **kwargs):
        checks.append(1)
        if len(checks) == 1:
            return {"conformant": False, "error": "private transient"}
        if len(checks) == 2:
            (ws / "submission/new.py").write_text("pass\n")
        return {"conformant": True, "checks": {"test": True}}

    monkeypatch.setattr(conformance, "compute", compute)
    with AC.live_feedback(ws, run, 0, "merlin_assisted", "", lambda: (), interval_s=0.01):
        _until(ws, lambda value: len(checks) >= 3 and value["available"])
    assert len(checks) >= 4


@pytest.mark.parametrize("round_index", [0, 1, 99])
@pytest.mark.parametrize("timeout", [False, True])
def test_shared_launch_wrapper_flushes_success_and_timeout_without_grading(workspace, round_index, timeout):
    ws, run = workspace
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        assert _read(ws)["running"] is True
        _append(
            run, "python agent_selfcheck.py --capsules all", '{"n_capsules": 1, "n_passed": 0}', round_index=round_index
        )
        if timeout:
            raise TimeoutError("synthetic author timeout")
        return 7, "original transcript result"

    wrapped = AC.live_launch(launch, True, ws, run, "raw_baseline", "", lambda: ())
    args = (ws, run, "model", "effort", "none", {}, round_index, 10)
    if timeout:
        with pytest.raises(TimeoutError):
            wrapped(*args, arm="raw_baseline", continuous=True)
    else:
        assert wrapped(*args, arm="raw_baseline", continuous=True) == (7, "original transcript result")
    assert calls == [(args, {"arm": "raw_baseline", "continuous": True})]
    assert _read(ws)["running"] is False
    assert _read(ws)["checks"]["full_selfcheck"] is True
    assert sorted(path.name for path in run.iterdir()) == ["rounds"]


def test_disabled_launch_is_original_callable_and_does_no_observation(workspace, monkeypatch):
    ws, run = workspace
    monkeypatch.setattr(conformance, "compute", lambda *a, **k: pytest.fail("disabled watcher ran"))

    def launch(*args, **kwargs):
        return "unchanged"

    assert AC.live_launch(launch, False, ws, run, "raw_baseline", "", lambda: ()) is launch
    assert not (ws / "qa").exists()


def test_actual_controller_routes_all_three_launch_paths_through_the_live_wrapper(workspace):
    ws, run = workspace
    tree = ast.parse(module_source_path("merlin_experiments.phase1.authoring").read_text())
    execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    wrapper = next(
        node
        for node in execute.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_live_launch"
    )
    launches = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_launch"
    ]
    assert len(launches) == 3
    observed = []

    def launch(*args, **kwargs):
        assert _read(ws)["running"]
        observed.append(args[6])
        return 0, "original-result"

    options = SimpleNamespace(
        live_conformance_feedback=True,
        model="model",
        effort="effort",
        sandbox="none",
        round_timeout=10,
        schedule="continuous",
    )
    scope = {
        "_live_launch": AC.live_launch,
        "_launch": launch,
        "a": options,
        "ws": ws,
        "run_dir": run,
        "arm": "raw_baseline",
        "_endpoint_kind": "",
        "resolved_tools": lambda: (),
        "bundle": {},
        "rnd": 2,
    }
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), "actual-launch-wrapper", "exec"), scope)
    for call in launches:
        assert call.lineno > wrapper.lineno
        assert eval(compile(ast.Expression(body=call), "actual-author-launch", "eval"), scope) == (0, "original-result")
        assert _read(ws)["running"] is False
    assert sorted(observed) == [0, 2, 2]


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_selection_is_recorded_only_when_enabled_and_resume_refuses_changes(enabled):
    options = parse_options(["--run-id", "test"] + (["--live-conformance-feedback"] if enabled else []), environ={})
    request = SimpleNamespace(options=options, launcher_argv=(), context=SimpleNamespace(readback_policy=None))
    record = session.RunRequest.run_config.fget(request)
    assert ("live_conformance_feedback" in record) is enabled
    AC.verify_live_selection(options, {"run_config": record})
    with pytest.raises(RuntimeError, match="feedback selection changed"):
        AC.verify_live_selection(
            dataclasses.replace(options, live_conformance_feedback=not enabled), {"run_config": record}
        )
    with pytest.raises(RuntimeError, match="feedback selection changed"):
        AC.verify_live_selection(options, {"run_config": {"live_conformance_feedback": 1}})


@pytest.mark.parametrize("selected", [None, False, True])
def test_experiment_definition_emits_only_explicit_enabled_selection(tmp_path, selected):
    from merlin_experiments.adapters import ADAPTERS
    from merlin_experiments.spec import SpecError

    adapter = ADAPTERS["capsule_bench"]
    spec = SimpleNamespace(target="fixture", id="functional", path=tmp_path / "x.yaml", resolve=lambda v: tmp_path / v)
    config = {"descriptor": "d.yaml", "arm": "merlin_assisted"}
    if selected is not None:
        config["live_conformance_feedback"] = selected
        adapter.options["live_conformance_feedback"].validate("live_conformance_feedback", selected)
    command = adapter.resolve(spec, config, tmp_path, tmp_path / "run")
    assert command["argv"].count("--live-conformance-feedback") == int(selected is True)
    with pytest.raises(SpecError):
        adapter.options["live_conformance_feedback"].validate("live_conformance_feedback", "true")
