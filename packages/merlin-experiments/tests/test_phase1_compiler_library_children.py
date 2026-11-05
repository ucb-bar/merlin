"""Original CLI and thread transports preserve initial-run library selection."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import compiler_library as C
from merlin_experiments.phase1.feedback import lifecycle as L
from test_compiler_libraries import selected
from test_phase1_session import _external_substitutes, _request, _run
from test_phase1_session import project as project

from merlin.targetgen.compiler_library import active_compiler_library


def bound(project, tmp_path):
    host, _ = selected(tmp_path)
    return replace(_request(project), compiler_library_selection=host), host


def test_real_fresh_cli_consumers_load_only_original_explicit_snapshot(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    request, host = bound(project, tmp_path)

    def admitted(prepared):
        with C.execution(prepared):
            arguments = C.child_arguments()
            program = r"""
import json,sys
from pathlib import Path
from merlin.targetgen.compiler_library import active_compiler_library
from merlin_experiments.phase1 import compiler_library as C
from merlin_experiments.phase1.brokers import selfcheck,simjob
from merlin_experiments.phase1.feedback import selfcheck as worker,formal
flags=sys.argv[2:]
root=Path(sys.argv[1])
seen=[]
class ReachedOriginalContext(Exception): pass
def original_context(*a,**k):
    owner=active_compiler_library()
    assert owner is not None
    assert tuple(C.child_arguments())==tuple(flags)
    seen.append(owner.record())
    raise ReachedOriginalContext
for module,args in ((selfcheck,['--ws',str(root)]),(simjob,['--ws',str(root),'--max-jobs','1']),(worker,[])):
    module.resolve_context=original_context
    try: module.main([*args,*flags])
    except ReachedOriginalContext: pass
    else: raise AssertionError('original consumer not reached')
    assert active_compiler_library() is None
formal._main=lambda *a,**k: original_context()
try: formal.main(flags)
except ReachedOriginalContext: pass
else: raise AssertionError('formal consumer not reached')
assert len(seen)==4 and all(record==seen[0] for record in seen)
assert active_compiler_library() is None
print(json.dumps(seen))
"""
            result = subprocess.run(
                [sys.executable, "-B", "-c", program, str(prepared.workspace), *arguments],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout) == [active_compiler_library().record()] * 4
            assert str(host.root) not in arguments
        return 0

    assert _run(request, admitted) == 0


def test_original_command_builders_forward_same_library_not_request_data(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    from merlin_experiments.phase1 import formal_invocation
    from merlin_experiments.phase1.brokers.selfcheck import _handler_command
    from merlin_experiments.phase1.feedback.selfcheck import worker_command

    request, _ = bound(project, tmp_path)

    def admitted(prepared):
        with C.execution(prepared):
            original = C.child_arguments()
            commands = []
            saved_clients, saved_brokers, saved_for = L._TR.COMMON_CLIENTS, L._TR.COMMON_BROKERS, L._TR.brokers_for
            saved_popen = L.subprocess.Popen
            monkeypatch.setattr(L._TR, "COMMON_CLIENTS", ())
            names = ("merlin_experiments.phase1.brokers.selfcheck", "merlin_experiments.phase1.brokers.simjob")
            monkeypatch.setattr(
                L._TR,
                "COMMON_BROKERS",
                tuple(
                    SimpleNamespace(module=name, log=f"{index}.log", channel=f".channel{index}")
                    for index, name in enumerate(names)
                ),
            )
            monkeypatch.setattr(L._TR, "brokers_for", lambda _: ())
            monkeypatch.setattr(L.subprocess, "Popen", lambda argv, **kw: commands.append(argv) or SimpleNamespace())
            L.start_brokers(prepared.workspace, L.BrokerConfig(request.context, (), project / "timing"))
            assert len(commands) == 2
            for command in commands:
                assert tuple(command[-6:]) == original
            actual = worker_command(request.context, prepared.public_root)
            assert tuple(actual[-6:]) == original
            channel = prepared.workspace / ".qa_channel"
            channel.mkdir(exist_ok=True)
            proposal = channel / "req_attempt.json"
            proposal.write_text(json.dumps({"compiler_library_root": "/candidate/override"}))
            options = SimpleNamespace(ws=prepared.workspace, poll=0.1, rtl_facts=None, compiler_bundle_manifest=None)
            handler = _handler_command(
                options, request.context, prepared.public_root, None, None, proposal, channel / "stop"
            )
            assert tuple(handler[-6:]) == original
            assert "/candidate/override" not in handler
            monkeypatch.setattr(L._TR, "COMMON_CLIENTS", saved_clients)
            monkeypatch.setattr(L._TR, "COMMON_BROKERS", saved_brokers)
            monkeypatch.setattr(L._TR, "brokers_for", saved_for)
            monkeypatch.setattr(L.subprocess, "Popen", saved_popen)
            captured = []
            original_run = formal_invocation.subprocess.run

            def child_run(command, **kwargs):
                if "merlin_experiments.phase1.feedback.formal" in command:
                    captured.append(command)
                    return SimpleNamespace(returncode=0)
                return original_run(command, **kwargs)

            monkeypatch.setattr(
                formal_invocation.subprocess,
                "run",
                child_run,
            )
            assert formal_invocation.run(prepared, public_capsules=prepared.public_root, selected_rtl_facts=None) == 0
            assert tuple(captured[0][-6:]) == original
        return 0

    assert _run(request, admitted) == 0


@pytest.mark.parametrize("damage", ["none", "contract", "source"])
def test_actual_background_grader_retains_owner_and_revalidates_join(project, monkeypatch, tmp_path, damage):
    _external_substitutes(monkeypatch)
    request, _ = bound(project, tmp_path)

    def admitted(prepared):
        with C.execution(prepared):
            selected = active_compiler_library()
            entered, completed = threading.Event(), threading.Event()
            backup = {}

            def grade(*args, **kwargs):
                assert active_compiler_library() is selected
                entered.set()
                if damage != "none":
                    path = (
                        selected.root.parent / "contract.json"
                        if damage == "contract"
                        else selected.root / "merlin/portable.py"
                    )
                    backup.update(path=path, raw=path.read_bytes())
                    path.chmod(0o644)
                    path.write_text("changed = True\n")
                    path.chmod(0o444)
                completed.set()
                return {"all_pass": None}

            handle = L.start_background(
                prepared.workspace,
                prepared.run_dir,
                L.GradeCadence(60, False, 900),
                interval_grades=False,
                grade_callback=grade,
                fast_grade_callback=grade,
            )
            try:
                assert entered.wait(5) and completed.wait(5)
                if damage == "none":
                    L.stop_background(handle)
                else:
                    with pytest.raises(ValueError, match="library|member"):
                        L.stop_background(handle)
            finally:
                handle[1].set()
                handle[0].join()
                if backup:
                    backup["path"].chmod(0o644)
                    backup["path"].write_bytes(backup["raw"])
                    backup["path"].chmod(0o444)
        return 0

    assert _run(request, admitted) == 0


def test_default_late_addition_and_initial_child_missing_flags_refuse(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    host, _ = selected(tmp_path)

    def legacy(prepared):
        with pytest.raises(ValueError, match="binding changed"):
            with C.execution(prepared):
                prepared.environment[C.KEY] = {"unissued": True}
        prepared.environment.pop(C.KEY)
        return 0

    assert _run(_request(project), legacy) == 0
    request = replace(_request(project), compiler_library_selection=host)
    # Use a new run identity because the first legacy run cannot be upgraded.
    request = replace(request, options=replace(request.options, run_id="selected"))

    def initial(prepared):
        with C.execution(prepared), pytest.raises(ValueError, match="explicit original library"):
            C.selected_child(lambda argv: pytest.fail("unbound child executed"))([])
        return 0

    assert _run(request, initial) == 0


def test_legacy_continuous_actual_grader_thread_uses_same_owner(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    from merlin_experiments.phase1 import authoring

    request, _ = bound(project, tmp_path)
    request = replace(request, options=replace(request.options, continuous=True))
    observed = []
    ready = threading.Event()

    class ReachedFinalGrade(Exception):
        pass

    def admitted(prepared):
        with C.execution(prepared):
            original = active_compiler_library()

            def first(*args, **kwargs):
                assert active_compiler_library() is original
                assert threading.current_thread().name == "continuous-grader"
                observed.append("background")
                ready.set()
                return {"all_pass": False}

            def final(*args, **kwargs):
                assert active_compiler_library() is original
                observed.append("final")
                raise ReachedFinalGrade

            def local_author(*args, **kwargs):
                assert active_compiler_library() is original
                assert ready.wait(5)
                path = prepared.run_dir / "local-transcript.jsonl"
                path.write_text("")
                return 0, path

            monkeypatch.setattr(authoring.LG, "fast_grade", first)
            monkeypatch.setattr(authoring.LG, "grade", final)
            monkeypatch.setattr(authoring.EX, "launch", local_author)
            with pytest.raises(ReachedFinalGrade):
                authoring.execute(
                    prepared, authoring.AuthoringRuntime("fixture", project / "timing.json", prepared.public_root)
                )
        return 0

    assert _run(request, admitted) == 0
    assert observed == ["background", "final"]


def test_thread_entry_failure_survives_restored_source_before_join(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    request, _ = bound(project, tmp_path)

    def admitted(prepared):
        with C.execution(prepared):
            selected = active_compiler_library()
            target, joined = C.background_target(lambda: pytest.fail("invalid source entered grader"))
            leaf = selected.root / "merlin/portable.py"
            original = leaf.read_bytes()
            leaf.chmod(0o644)
            leaf.write_text("changed = True\n")
            leaf.chmod(0o444)
            thread = threading.Thread(target=target)
            thread.start()
            thread.join()
            leaf.chmod(0o644)
            leaf.write_bytes(original)
            leaf.chmod(0o444)
            with pytest.raises(RuntimeError, match="background owner failed"):
                joined()
        return 0

    assert _run(request, admitted) == 0
