"""Original profile composition controls; no runtime or proof qualification."""

import hashlib
import subprocess
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_component_runtime_support as runtime_units
from merlin_experiments.phase1 import component_package_execution as E
from merlin_experiments.phase1.brokers import compiler_call_channel as channel
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2.component_experiment import ComponentView, RuntimeGrant
from merlin_experiments.phase2.contracts import StageGateError

from merlin.targetgen import package_runtime as P


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    return runtime_units.prepared.__wrapped__(tmp_path, monkeypatch)


def test_original_scope_composes_source_public_grants_and_component_transport(tmp_path, monkeypatch):
    candidate, evidence = tmp_path / "candidate", tmp_path / "private" / "grade"
    candidate.mkdir()
    evidence.mkdir(parents=True)
    tool = candidate / "compiler.py"
    tool.write_text("actual selected compiler source")
    source, output = evidence / "input.mlir", evidence / "output.json"
    source.write_text("module {}")
    sandbox = tmp_path / "bwrap"
    sandbox.write_text("reviewed runtime tool")
    runtime = (RuntimeGrant(sandbox, "/usr/bin/bwrap", hashlib.sha256(sandbox.read_bytes()).hexdigest()),)
    view = ComponentView(tmp_path / "public", "1" * 64, "2" * 64, "3" * 64)
    owner = R.CommonTransformService.__new__(R.CommonTransformService)
    monkeypatch.setattr(R.CommonTransformService, "compiler_imports", lambda _self: nullcontext())
    monkeypatch.setattr(R.CommonTransformService, "_verify", lambda _self: None)
    public = tmp_path / "frozen-public.py"
    public.write_text("public compiler API")
    root = tmp_path / "frozen-python-root"
    monkeypatch.setattr(R, "_public_python_roots", lambda _owner: (root,))
    public_calls = []

    def public_policy(policy, *, target, owner):
        public_calls.append(owner)
        return replace(
            policy, argv=(*policy.argv, "--ro-bind", str(public), "/public/api.py", "--tmpfs", "/protected-answer")
        )

    monkeypatch.setattr(R, "public_package_policy", public_policy)
    monkeypatch.setattr(
        P, "_resolve_argv", lambda _pkg, _name, incoming, outgoing: ["compiler", str(incoming), str(outgoing)]
    )
    monkeypatch.setattr(P, "_needs_interpreter", lambda *_args: False)
    monkeypatch.setattr(
        E,
        "strict_tool_policy",
        lambda *_args, **_kwargs: (
            str(sandbox),
            "--ro-bind",
            str(candidate),
            str(candidate),
            "--setenv",
            "PYTHONPATH",
            "/component-inputs/compiler",
        ),
    )
    observed = []

    @contextmanager
    def transport(selected_owner, *, workspace, package, input_path, command_name, timeout_s):
        assert selected_owner is owner and package == candidate and input_path == source
        assert not source.is_relative_to(workspace)
        yield SimpleNamespace(
            environment=lambda: (("MERLIN_COMMON_WORKSPACE", str(workspace)),), remaining_timeout=lambda: timeout_s
        )

    monkeypatch.setattr(channel, "compiler_call_transport", transport)

    def run(command, **_kwargs):
        observed.append(command)
        Path(command[-1]).write_text('{"actual":"compiler output"}')
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(E.subprocess, "run", run)
    package = P.Package(candidate, {"language": "c"}, tool)
    with R.original_compiler_execution(
        owner, target_experiment=None, candidate=candidate, work_root=tmp_path / "scope"
    ):
        original = P.active_package_executor()
        with E.qualified_package_execution(candidate=candidate, view=view, runtime=runtime, evidence_root=evidence):
            composed = P.active_package_executor()
            assert type(composed) is E.ComponentPackageExecutor
            assert composed.original_profile_execution() is original
            assert composed.view is view and composed.runtime is runtime
            composed.run_entrypoint(package, "emit", source, output, invocation_directory=evidence)
            command = observed[0]
            assert str(root) + ":/component-inputs/compiler" in command
            assert "/protected-answer" in command and "/public/api.py" in command
            assert "MERLIN_COMMON_WORKSPACE" in command and str(evidence) not in command
            assert output.read_text() == '{"actual":"compiler output"}' and public_calls == [owner]
            tool.write_text("changed compiler source")
            with pytest.raises(ValueError, match="another source revision"):
                composed.build_package(package)
            tool.write_text("actual selected compiler source")
        assert P.active_package_executor() is original
    assert P.active_package_executor() is None


def test_component_wrapper_refuses_an_unrelated_executor(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    view = ComponentView(tmp_path / "public", "1" * 64, "2" * 64, "3" * 64)
    foreign = SimpleNamespace(build_package=lambda *_a: None, run_entrypoint=lambda *_a: None)
    with P.scoped_package_executor(foreign), pytest.raises(StageGateError, match="foreign package execution owner"):
        with E.qualified_package_execution(
            candidate=candidate, view=view, runtime=(), evidence_root=tmp_path / "grade"
        ):
            pass


def test_composed_container_selection_is_retained_without_native_fallback(tmp_path, monkeypatch):
    """Wrapper data composition only; no container/service authority is issued."""
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("same compiler source")
    owner = R.CommonTransformService.__new__(R.CommonTransformService)
    monkeypatch.setattr(R.CommonTransformService, "compiler_imports", lambda _self: nullcontext())
    monkeypatch.setattr(R.CommonTransformService, "_verify", lambda _self: None)
    view = ComponentView(tmp_path / "public", "1" * 64, "2" * 64, "3" * 64)
    declared_transport = object()
    with R.original_compiler_execution(
        owner, target_experiment=None, candidate=candidate, work_root=tmp_path / "scope"
    ):
        original = P.active_package_executor()
        with E.qualified_package_execution(
            candidate=candidate,
            view=view,
            runtime=(),
            evidence_root=tmp_path / "grade",
            container_transport=declared_transport,
        ):
            composed = P.active_package_executor()
            assert composed.container_transport is declared_transport
            assert composed.original_profile_execution() is original
        assert P.active_package_executor() is original


def test_runtime_grader_composes_original_scope_before_component_execution(prepared, tmp_path, monkeypatch):
    from merlin_experiments.phase2 import component_runtime_support as support

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("same original compiler")
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "capsule.yaml").write_text("name: scoped_component\n")
    view = ComponentView(tmp_path / "public", "1" * 64, "2" * 64, "3" * 64)
    prepared = replace(prepared, compiler_view=view)
    monkeypatch.setattr(support.PreparedIndependentRuntimeContext, "_fixture_for", lambda *_a: None)
    owner = R.CommonTransformService.__new__(R.CommonTransformService)
    monkeypatch.setattr(R.CommonTransformService, "compiler_imports", lambda _self: nullcontext())
    monkeypatch.setattr(R.CommonTransformService, "_verify", lambda _self: None)
    observed = []

    def probe(**_kwargs):
        executor = P.active_package_executor()
        assert type(executor) is E.ComponentPackageExecutor and executor.view is view
        observed.append(executor.original_profile_execution())
        raise RuntimeError("component body reached; no native evaluation")

    monkeypatch.setattr(support, "execute_component", probe)
    with R.original_compiler_execution(
        owner, target_experiment=None, candidate=candidate, work_root=tmp_path / "scope"
    ):
        original = P.active_package_executor()
        with pytest.raises(RuntimeError, match="component body reached"):
            prepared.grade(
                candidate,
                capsules_root=[capsule],
                runs_root=tmp_path / "grade",
                contract=prepared.contract_root,
                target=prepared.build_service.target,
                timeout=5,
            )
        assert observed == [original] and P.active_package_executor() is original
