"""Reviewed library scopes cross ordinary author, broker and final guard seams."""

import inspect
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import compiler_profile_child as CHILD
from merlin_experiments.phase2 import agent_workspace as W
from merlin_experiments.phase2 import authoring as A
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import candidate_record as RECORD
from merlin_experiments.phase2 import compiler_libraries as L
from test_agent_workspace import snapshot  # noqa: F401
from test_compiler_libraries import selected
from test_functional_emission_guard import inputs as guard_inputs  # noqa: F401

from merlin.targetgen.compiler_library import active_compiler_library, scoped_compiler_library


def test_explicit_stage_selection_precedes_any_work(tmp_path, monkeypatch):
    selection, _ = selected(tmp_path)
    arguments = {
        name: None
        for name, row in inspect.signature(A.run_stage).parameters.items()
        if row.default is inspect.Parameter.empty
    }
    observed = []

    def admission(*args, **kwargs):
        observed.append(active_compiler_library())
        raise RuntimeError("preflight boundary reached")

    monkeypatch.setattr(A, "admit_authoring_workflow", admission)
    with pytest.raises(RuntimeError, match="preflight boundary"):
        A.run_stage(**arguments, compiler_library_selection=selection)
    assert observed == [selection]
    with pytest.raises(RuntimeError, match="preflight boundary"):
        A.run_stage(**arguments, compiler_library_selection=selection, diagnostic_library_stage=True)
    assert observed == [selection, selection] and active_compiler_library() is None
    with scoped_compiler_library(selection):
        with pytest.raises(A.StageGateError, match="unbound"):
            A.run_stage(**arguments)


def test_prompt_declares_only_the_selected_source_leaves(tmp_path):
    selection, _ = selected(tmp_path)

    @dataclass(frozen=True)
    class Inputs:
        allowed_paths: tuple

    original = Inputs(("/original",))
    assert L.prompt_inputs(original) is original
    with scoped_compiler_library(selection):
        updated = L.prompt_inputs(original)
    assert updated.allowed_paths == (
        "/original",
        *(str(L.guest_root(selection) / row.path) for row in selection.contract.members),
    )
    assert original.allowed_paths == ("/original",)


@pytest.mark.parametrize("plane", ["inner", "outer"])
def test_author_planes_receive_exact_leaves_and_inner_verification_reopens_them(
    tmp_path,
    monkeypatch,
    snapshot,  # noqa: F811
    plane,
):
    selection, candidate = selected(tmp_path)
    monkeypatch.setattr(W.BW, "base_argv", lambda *args, **kwargs: ["bwrap"])
    monkeypatch.setattr(W.TC, "toolchain_binds", lambda *args, **kwargs: [])
    monkeypatch.setattr(W.TC, "sandbox_env", lambda *args, **kwargs: "export ORIGINAL=1; ")
    probes = (W.TC.ToolProbe("python", "python --version"),)
    monkeypatch.setattr(W.TC, "required_tool_probes", lambda *args, **kwargs: probes)
    inputs = W.CAMPAIGN.PackageSandboxInputs(
        W.TC.ToolchainPaths(tmp_path, *[str(tmp_path / value) for value in ("venv", "llvm", "compat", "clang", "uv")]),
        W.TC.SimToolchain(),
        "",
        (),
    )
    with scoped_compiler_library(selection):
        if plane == "inner":
            policy = W.inner_execution_policy(SimpleNamespace(), candidate, snapshot, inputs=inputs)
            policy.verify_execution()
            with pytest.raises(A.StageGateError, match="candidate-owned"):
                W.inner_execution_policy(SimpleNamespace(), selection.root, snapshot, inputs=inputs)
        else:
            policy = W.outer_codex_policy(candidate, snapshot, (), SimpleNamespace(), inputs=inputs)
        assert str(selection.root) not in policy.argv
        assert str(selection.root / "merlin/portable.py") in policy.argv
        assert str(selection.root / "private-answer.txt") not in policy.argv
        assert policy.compiler_library_selection is selection
        if plane == "inner":
            with pytest.raises(A.StageGateError, match="leaves|mount selection"):
                replace(policy, argv=policy.argv[:-3]).verify_execution()
    if plane == "inner":
        with pytest.raises(A.StageGateError, match="captured owner"):
            policy.verify_execution()


def test_broker_dispatch_installs_bound_library_in_a_fresh_thread(tmp_path, monkeypatch):
    selection, _ = selected(tmp_path)
    broker = object.__new__(B.Broker)
    broker._compiler_library_selection = selection

    def action(_request):
        assert active_compiler_library() is selection
        raise RuntimeError("request body reached")

    class Actions:
        def __contains__(self, name):
            action(name)

    broker.actions = Actions()
    observed = []

    def worker():
        try:
            broker.execute({"action": "probe"})
        except RuntimeError as error:
            observed.append(str(error))
        assert active_compiler_library() is None

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive() and observed == ["request body reached"]


@pytest.mark.parametrize("baseline_status", ["clean", "offending"])
def test_final_guard_requalifies_both_arms_under_one_library(tmp_path, baseline_status):
    selection, _ = selected(tmp_path)
    calls = []

    @L.paired_functional_guard
    def guard(baseline, candidate, *, grade_root):
        assert active_compiler_library() is selection
        calls.append((baseline, candidate, grade_root))
        return {"status": baseline_status if baseline == candidate else "clean", "offenders": []}

    with scoped_compiler_library(selection):
        result = guard("baseline", "candidate", grade_root=tmp_path / "grade")
    assert calls == [
        ("baseline", "baseline", tmp_path / "grade_library_baseline"),
        ("baseline", "candidate", tmp_path / "grade"),
    ]
    assert result["additional_compiler_library"] == selection.record()
    assert result["status"] == ("clean" if baseline_status == "clean" else "offending")


@pytest.mark.parametrize("batch", [False, True])
def test_model_children_cannot_lose_original_owner(tmp_path, batch):
    selection, candidate = selected(tmp_path)
    with scoped_compiler_library(selection):
        with pytest.raises(A.StageGateError):
            if batch:
                with CHILD.prepare_model_children(spec_paths=[], environment={}, package_dir=candidate):
                    pytest.fail("child launched")
            else:
                with CHILD.prepare_model_child(spec_path=tmp_path / "spec.json", environment={}, package_dir=candidate):
                    pytest.fail("child launched")


def test_legacy_measurement_cannot_drop_the_additional_library_selection():
    record = {
        "schema_version": RECORD.SCHEMA_VERSION,
        "kind": "arm4_performance_candidate",
        "additional_compiler_library": {"untrusted": "not authority"},
        "admission": {"consumable": True},
    }
    with pytest.raises(A.StageGateError, match="same-release"):
        RECORD.validate_candidate_record(record)


def test_original_final_guard_grades_baseline_and_candidate_without_historical_reuse(
    tmp_path,
    monkeypatch,
    guard_inputs,  # noqa: F811
):
    from merlin_experiments.phase2 import campaign as C

    from merlin.targetgen import capsule_grade, capsule_runner, oot_runner

    frozen, _, public, _ = guard_inputs
    selection, candidate = selected(tmp_path)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    for package in (baseline, candidate):
        (package / "compiler.py").write_text("same compiler source")
    inputs = C.PackageSandboxInputs(
        C.TC.ToolchainPaths(tmp_path, *[str(tmp_path / str(index)) for index in range(5)]),
        C.TC.SimToolchain(),
        "",
        (),
    )
    monkeypatch.setattr(oot_runner, "load_package", lambda path: SimpleNamespace(directory=Path(path)))
    monkeypatch.setattr(
        C,
        "package_sandbox_policy",
        lambda target, workspace, package, **kwargs: C.PackageSandboxPolicy(
            (), (), (), Path(workspace), Path(package), target, env_prefix=""
        ),
    )

    def emit(package, interface, scratch, tag, timeout):
        assert active_compiler_library() is selection
        assert C.active_sandbox_policy().package == package.directory
        return 0, "same LLVM", "same buffer"

    monkeypatch.setattr(A.EA, "emit_pair", emit)
    monkeypatch.setattr(capsule_runner, "qa_checkpoint_adapters", lambda *args: {})
    graded = []

    def grade(package, **kwargs):
        assert active_compiler_library() is selection
        assert C.active_sandbox_policy().package == package
        assert kwargs["capsules_root"] == public / "public"
        graded.append(package)
        return {
            "integrity_status": "clean",
            "per_capsule": [{"capsule": "visible", "status": "pass", "numeric": "pass", "cert_verdict": None}],
        }

    monkeypatch.setattr(capsule_grade, "grade", grade)
    with scoped_compiler_library(selection):
        result = A.functional_emission_guard(
            baseline,
            candidate,
            SimpleNamespace(target="synthetic", sim_via="selected"),
            frozen_functional=frozen,
            sandbox_inputs=inputs,
            grade_root=tmp_path / "final",
        )
    assert len(graded) == 2 and graded[0] != graded[1]
    assert result["status"] == result["library_baseline_guard"]["status"] == "clean"
    assert result["proved_unchanged"] == result["library_baseline_guard"]["proved_unchanged"] == 0


def test_selected_library_isolated_call_works_without_external_hard_boundary(tmp_path, monkeypatch):
    from merlin_experiments.phase2 import campaign as C
    from merlin_experiments.phase2 import functional_guard_execution as G

    selection, candidate = selected(tmp_path)
    workspace = tmp_path / "work"
    workspace.mkdir()
    incoming = tmp_path / "input.mlir"
    incoming.write_text("module {}")
    outgoing = tmp_path / "output.json"
    inputs = C.PackageSandboxInputs(
        C.TC.ToolchainPaths(tmp_path, *[str(tmp_path / str(index)) for index in range(5)]),
        C.TC.SimToolchain(),
        "",
        (),
    )
    monkeypatch.setattr(
        C,
        "package_sandbox_policy",
        lambda target, work, package, **kwargs: C.PackageSandboxPolicy((), (), (), Path(work), Path(package), target),
    )
    calls = []

    def execute(pkg, name, source, output, **kwargs):
        assert active_compiler_library() is selection
        assert source != incoming and source.read_text() == incoming.read_text()
        assert C.active_sandbox_policy().workspace == source.parent
        assert G.active_guard() is None
        output.write_text("{}")
        calls.append(name)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(C, "_run_boxed", execute)
    with scoped_compiler_library(selection), G.boxed_guard_package(None, workspace, candidate, inputs=inputs):
        G.execute_isolated(
            G.active_guard().policy,
            object(),
            "emit",
            incoming,
            outgoing,
            timeout=1,
            invocation_directory=tmp_path / "invocation",
        )
    assert calls == ["emit"] and outgoing.read_text() == "{}"
