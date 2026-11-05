"""Original final-grade routing controls; no kernel setup or real evaluator runs."""

import subprocess
import threading
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import campaign as C
from merlin_experiments.phase2 import functional_guard_execution as G
from merlin_experiments.phase2 import normal_resource_launch as N
from merlin_experiments.phase2.contracts import StageGateError
from test_functional_emission_guard import inputs as functional_inputs_fixture  # noqa: F401
from test_normal_resource_launch import ordinary_prefix
from test_normal_resource_launch import routing_spy as kernel_routing_fixture  # noqa: F401

from merlin.targetgen import capsule_common, capsule_scheduling, oot_runner


@pytest.fixture
def isolated(tmp_path, monkeypatch, kernel_routing_fixture):  # noqa: F811 - imported pytest fixture
    package = tmp_path / "compiler"
    package.mkdir()
    (package / "compiler.py").write_text("original compiler source")
    workspace = tmp_path / "grade"
    workspace.mkdir()
    selection = C.PackageSandboxInputs(
        C.TC.ToolchainPaths(tmp_path, *[str(tmp_path / str(i)) for i in range(5)]),
        C.TC.SimToolchain(),
        "",
        (),
    )

    def policy(target, workspace, package, *, inputs):
        assert inputs is selection
        return C.PackageSandboxPolicy(
            argv=(*ordinary_prefix(workspace), "--unshare-net", "--ro-bind", str(package), str(package)),
            workspace=Path(workspace).resolve(),
            package=Path(package).resolve(),
            target_experiment=None,
            coverage_gap=(),
            required_tools=(),
        )

    monkeypatch.setattr(C, "package_sandbox_policy", policy)
    monkeypatch.setattr(oot_runner, "_resolve_argv", lambda pkg, name, inp, out: ["fixed-compiler", str(inp), str(out)])
    monkeypatch.setattr(oot_runner, "_needs_interpreter", lambda *args: False)
    return SimpleNamespace(package=package, workspace=workspace, selection=selection, resource=kernel_routing_fixture)


def test_imported_execution_owners_are_from_the_managed_source():
    task = Path.cwd()
    assert (task / ".git").is_file()
    for module in (C, capsule_scheduling, capsule_common):
        assert Path(module.__file__).is_relative_to(task)


def test_retained_original_function_uses_one_interface_and_private_outputs(isolated, monkeypatch):
    value = isolated
    incoming = value.workspace / "input.mlir"
    incoming.write_text("module {}\n")
    secret = value.workspace / ".private_model_sources"
    secret.mkdir()
    (secret / "reference.npy").write_text("grading answers")
    output = value.workspace / "caller.json"
    original = oot_runner.run_entrypoint
    private = []

    def run(argv, **kwargs):
        inp, out = map(Path, argv[-2:])
        private.append(inp.parent)
        assert inp.read_text() == incoming.read_text()
        assert inp.parent != value.workspace
        binds = [Path(argv[i + 1]) for i, flag in enumerate(argv) if flag == "--bind"]
        assert value.workspace not in binds and secret not in binds
        assert not any(str(secret) in arg for arg in argv)
        assert ["--ro-bind", str(value.package), str(value.package)] in [argv[i : i + 3] for i in range(len(argv) - 2)]
        assert "--disable-userns" in argv
        out.write_text('{"actual": true}')
        return subprocess.CompletedProcess(argv, 0, "actual stdout", "")

    monkeypatch.setattr(C.subprocess, "run", run)
    pkg = SimpleNamespace(directory=value.package, manifest={})
    with (
        N.selected_boundary(value.resource.owner),
        G.boxed_guard_package(None, value.workspace, value.package, inputs=value.selection),
    ):
        result = original(pkg, "parse", incoming, output, timeout=2)
        assert result.stdout == "actual stdout" and output.read_text() == '{"actual": true}'
        assert all(path.is_dir() for path in private)
        assert G.owns_executor(oot_runner.active_package_executor())
    assert all(not path.exists() for path in private)
    assert C.active_sandbox_policy() is None and G.active_guard() is None
    assert oot_runner.run_entrypoint is original and oot_runner.active_package_executor() is None


@pytest.mark.parametrize("linked", ["symlink", "hardlink"])
def test_compiler_linked_outputs_refuse(isolated, monkeypatch, linked):
    value = isolated
    incoming = value.workspace / "input.mlir"
    incoming.write_text("module {}")
    output = value.workspace / "caller.json"

    def run(argv, **kwargs):
        out = Path(argv[-1])
        other = out.parent / "other"
        other.write_text("foreign output")
        if linked == "symlink":
            out.symlink_to(other)
        else:
            out.hardlink_to(other)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(C.subprocess, "run", run)
    with (
        N.selected_boundary(value.resource.owner),
        G.boxed_guard_package(None, value.workspace, value.package, inputs=value.selection),
        pytest.raises((StageGateError, OSError)),
    ):
        oot_runner.run_entrypoint(SimpleNamespace(directory=value.package), "parse", incoming, output)
    assert not output.exists()


def test_original_package_and_workspace_checks_remain(isolated, monkeypatch):
    value = isolated
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: pytest.fail("foreign path launched"))
    incoming = value.workspace / "input.mlir"
    incoming.write_text("module {}")
    with (
        N.selected_boundary(value.resource.owner),
        G.boxed_guard_package(None, value.workspace, value.package, inputs=value.selection),
    ):
        with pytest.raises(C.CampaignGateError, match="outside the snapshot"):
            oot_runner.run_entrypoint(SimpleNamespace(directory=value.workspace), "parse", incoming)
        with pytest.raises(C.CampaignGateError, match="input is outside"):
            oot_runner.run_entrypoint(SimpleNamespace(directory=value.package), "parse", value.package / "foreign")
        with pytest.raises(C.CampaignGateError, match="output is outside"):
            oot_runner.run_entrypoint(
                SimpleNamespace(directory=value.package), "parse", incoming, value.package / "out"
            )
        linked = value.workspace / "linked-output"
        linked.symlink_to(incoming)
        with pytest.raises(C.CampaignGateError, match="output destination is linked"):
            oot_runner.run_entrypoint(SimpleNamespace(directory=value.package), "parse", incoming, linked)


def test_grade_workers_keep_original_box_and_live_owner(isolated):
    value = isolated
    parent = threading.get_ident()
    seen = []

    def grade(cap, parallel):
        assert threading.get_ident() != parent
        assert C.active_sandbox_policy().package == value.package
        assert G.active_guard() is not None
        assert N.resources.active_resource_boundary() is value.resource.owner
        assert G.owns_executor(oot_runner.active_package_executor())
        seen.append(cap["name"])
        return {"capsule": cap["name"]}

    with (
        N.selected_boundary(value.resource.owner),
        G.boxed_guard_package(None, value.workspace, value.package, inputs=value.selection),
    ):
        rows = capsule_scheduling.run_pipelined(
            [{"name": "one"}, {"name": "two", "extends": "one"}],
            grade,
            max_workers=2,
            calibrate=lambda: set(),
            calibration_cap=1,
        )
    assert seen == ["one", "two"] and rows == [{"capsule": "one"}, {"capsule": "two"}]


def test_lower_interface_overlap_copies_original_context(tmp_path, monkeypatch):
    marker = ContextVar("focused-original-invocation", default=None)
    parent = threading.get_ident()
    threads = []
    inp = tmp_path / "interface.mlir"
    inp.write_text("module {}")
    pkg = SimpleNamespace(manifest={"commands": {"emit_command_buffer": {}}})
    monkeypatch.setattr(capsule_common.schemas, "validate_command_buffer", lambda *a, **k: None)

    def invoke(pkg, name, inp, out=None, **kwargs):
        assert marker.get() == "issued context"
        threads.append((name, threading.get_ident()))
        if out:
            out.write_text("{}")
        return subprocess.CompletedProcess([], 0, "module {}", "")

    token = marker.set("issued context")
    try:
        capsule_common.lower_interface(
            pkg, inp, tmp_path / "generated", contract=None, timeout=1, invoke=invoke, overlap=True
        )
    finally:
        marker.reset(token)
    assert any(name == "emit_command_buffer" and worker != parent for name, worker in threads)


def test_only_selected_budgeted_whole_model_child_refuses(isolated, monkeypatch):
    value = isolated
    model = value.workspace / "model"
    model.mkdir()
    (model / "capsule.yaml").write_text("kind: model\n")
    monkeypatch.setenv("MERLIN_MODEL_BUDGET_S", "2")
    G.require_inline_models([model])  # Exact default path without a live hard owner.
    with N.selected_boundary(value.resource.owner):
        with pytest.raises(StageGateError, match="exec-based whole-model budget child"):
            G.require_inline_models([model])
        monkeypatch.delenv("MERLIN_MODEL_BUDGET_S")
        G.require_inline_models([model])


def test_original_complete_numerical_grade_and_both_emissions_are_boxed(
    functional_inputs_fixture,  # noqa: F811 - imported pytest fixture
    isolated,
    monkeypatch,
):
    from merlin_experiments.phase2 import authoring

    from merlin.targetgen import capsule_grade, capsule_runner

    frozen_inputs, _, frozen, _ = functional_inputs_fixture
    value = isolated
    candidate = value.workspace / "candidate"
    candidate.mkdir()
    (candidate / "runtime.py").write_text("changed runtime")
    monkeypatch.setattr(oot_runner, "load_package", lambda path: SimpleNamespace(directory=Path(path)))
    monkeypatch.setattr(capsule_runner, "qa_checkpoint_adapters", lambda *a: {})
    observed = []

    def emit(pkg, interface, scratch, tag, timeout):
        assert C.active_sandbox_policy().package == pkg.directory
        assert interface.is_relative_to(C.active_sandbox_policy().workspace)
        assert interface.read_text() == "module {}\n"
        observed.append(tag)
        return 0, "same LLVM", "same buffer"

    def grade(package, **kwargs):
        assert C.active_sandbox_policy().package == package
        assert G.active_guard() is not None
        assert kwargs["capsules_root"] == frozen / "public"
        assert kwargs["labels"] == {"public", "dev"}
        observed.append("full grade")
        return {
            "integrity_status": "clean",
            "per_capsule": [{"capsule": "visible", "status": "pass", "numeric": "fail", "cert_verdict": None}],
        }

    monkeypatch.setattr(authoring.EA, "emit_pair", emit)
    monkeypatch.setattr(capsule_grade, "grade", grade)
    with N.selected_boundary(value.resource.owner):
        result = authoring.functional_emission_guard(
            value.package,
            candidate,
            SimpleNamespace(target="synthetic", sim_via="selected"),
            frozen_functional=frozen_inputs,
            sandbox_inputs=value.selection,
        )
    assert observed == ["baseline", "candidate", "full grade"]
    assert result["status"] == "offending" and result["full_output_regrade"]["status"] == "fail"
