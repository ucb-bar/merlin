"""Initial authoring and later tuning retain one explicitly reviewed library."""

from __future__ import annotations

import copy
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import compiler_library as L
from merlin_experiments.phase2 import compiler_library_state as S
from test_compiler_libraries import selected
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs
from test_phase1_session import _external_substitutes, _request, _run, assemble, probe, resolved_tools, stage
from test_phase1_session import project as project

from merlin.common.paths import module_source_path
from merlin.targetgen.compiler_library import (
    CompilerLibrarySelection,
    active_compiler_library,
    freeze_compiler_library,
    scoped_compiler_library,
)


def request(project, selection, *, resume=False):
    return replace(_request(project, resume=resume), compiler_library_selection=selection)


def test_real_initial_admission_freezes_inventory_and_resume_same_owner(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    host, _ = selected(tmp_path)
    observed = []

    def admitted(prepared):
        prepared.verify_inputs()
        binding = prepared.environment[L.KEY]
        snapshot = S.verify_binding(binding, host)
        assert snapshot.root.is_relative_to(prepared.run_dir)
        assert not (snapshot.root / "private-answer.txt").exists()
        assert not snapshot.root.stat().st_mode & 0o222
        task = (prepared.workspace / "TASK.md").read_text()
        assert "merlin.portable" in task
        assert str(host.root) not in task
        assert "private-answer" not in task
        assert (prepared.run_dir / "TASK.md").read_text() == task
        with L.execution(prepared):
            active = active_compiler_library()
            assert active.record() == snapshot.record()
            assert active is not host
        assert active_compiler_library() is None
        observed.append(copy.deepcopy(binding))
        return 0

    assert _run(request(project, host), admitted) == 0
    assert _run(request(project, host, resume=True), admitted) == 0
    assert observed[0] == observed[1]
    with pytest.raises(ValueError, match="added or removed"):
        _run(_request(project, resume=True), lambda _: pytest.fail("library was dropped"))
    other = tmp_path / "different"
    other.mkdir()
    different, _ = selected(other)
    with pytest.raises(S.StageGateError, match="independently reviewed"):
        _run(request(project, different, resume=True), lambda _: pytest.fail("library was substituted"))


def test_legacy_run_cannot_acquire_library_on_resume(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    assert _run(_request(project), lambda p: 0 if L.KEY not in p.environment else 1) == 0
    host, _ = selected(tmp_path)
    with pytest.raises(ValueError, match="added or removed"):
        _run(request(project, host, resume=True), lambda _: pytest.fail("old run was retroactively granted access"))


@pytest.mark.parametrize("change", ["host", "snapshot", "binding", "remove", "owner"])
def test_original_and_frozen_sources_reopen_after_execution(project, monkeypatch, tmp_path, change):
    _external_substitutes(monkeypatch)
    host, _ = selected(tmp_path)

    def admitted(prepared):
        with pytest.raises((ValueError, S.StageGateError), match="library|member|reviewed"):
            with L.execution(prepared):
                if change == "host":
                    (host.root / "merlin/portable.py").write_text("changed = True\n")
                elif change == "snapshot":
                    snapshot = active_compiler_library()
                    leaf = snapshot.root / "merlin/portable.py"
                    leaf.chmod(0o644)
                    leaf.write_text("changed = True\n")
                    leaf.chmod(0o444)
                elif change == "binding":
                    prepared.environment[L.KEY]["snapshot_sha256"] = "f" * 64
                elif change == "remove":
                    del prepared.environment[L.KEY]
                else:
                    object.__setattr__(prepared.request, "compiler_library_selection", None)
        return 0

    assert _run(request(project, host), admitted) == 0


def test_initial_selection_refuses_ambient_and_unisolated_transport(tmp_path):
    host, _ = selected(tmp_path)
    original = SimpleNamespace(compiler_library_selection=host, options=SimpleNamespace(sandbox="none"))
    with pytest.raises(ValueError, match="readonly bwrap"):
        L.freeze_for_run(original, tmp_path / "run", prior=None, resuming=False)
    with scoped_compiler_library(host), pytest.raises(ValueError, match="ambient"):
        L.freeze_for_run(original, tmp_path / "run", prior=None, resuming=False)


def test_exact_request_leaf_executes_in_isolated_python_and_hides_host_implementations(tmp_path):
    root = tmp_path / "reviewed"
    modules = (
        "merlin",
        "merlin.llvmlower",
        "merlin.llvmlower.source_optimization_request",
        "merlin.llvmlower.optional_passes",
    )
    sources = []
    for name in modules:
        source = module_source_path(name)
        relative = name.replace(".", "/") + ("/__init__.py" if source.name == "__init__.py" else ".py")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources.append((relative, name))
    contract = freeze_compiler_library(
        root, review_id="prospective-request-data-only", public_modules=(modules[2],), sources=tuple(sources)
    )
    host = CompilerLibrarySelection(contract, root)
    binding = S.freeze_selection(host, tmp_path / "frozen/python")
    snapshot = S.verify_binding(binding, host)
    script = f"""
import hashlib,sys
sys.path.insert(0, {str(snapshot.root)!r})
from merlin.llvmlower.source_optimization_request import SourceOptimizationRequest
source = "module {{}}"
r = SourceOptimizationRequest(hashlib.sha256(source.encode()).hexdigest(), True, False, 3)
states,budget = r.bind(source, permitted_passes=("automatic-host-permutations",), maximum_added_operations=3)
assert states.state("automatic-host-permutations") is True and budget == 3
for name in ("merlin.llvmlower.lower", "merlin.llvmlower.im2col_pack", "merlin.verify"):
    try: __import__(name)
    except ModuleNotFoundError: pass
    else: raise AssertionError(name)
for name,module in tuple(sys.modules.items()):
    if name == "merlin" or name.startswith("merlin."):
        assert module.__file__.startswith({str(snapshot.root)!r} + "/"), (name,module.__file__)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert S.verify_binding(binding, host).record() == snapshot.record()
    from merlin.targetgen.package_runtime import CertFailure, integrity_scan

    candidate = tmp_path / "compiler"
    candidate.mkdir()
    package = SimpleNamespace(directory=candidate, integrity_exempt=False, manifest={})
    tool = candidate / "tool.py"
    tool.write_text("from merlin.llvmlower.source_optimization_request import SourceOptimizationRequest\n")
    with scoped_compiler_library(snapshot):
        integrity_scan(package)
        for module in ("merlin.llvmlower.lower", "merlin.verify", "merlin.llvmlower.optional_passes"):
            tool.write_text("import " + module + "\n")
            with pytest.raises(CertFailure, match="imports"):
                integrity_scan(package)


def test_actual_provider_overlay_precedes_final_answer_masks(project, monkeypatch, tmp_path):
    from merlin_experiments.phase1.providers import execution

    from merlin.targetgen.sandbox import bwrap as B
    from merlin.targetgen.sandbox import claude_home, toolchain

    host, _ = selected(tmp_path)
    binding = S.freeze_selection(host, tmp_path / "frozen/python")
    snapshot = S.verify_binding(binding, host)
    context = _request(project).context
    seen = []
    monkeypatch.setattr(B, "base_argv", lambda *a, **k: ["bwrap"])
    monkeypatch.setattr(claude_home, "without_operator_state", lambda _: [])
    monkeypatch.setattr(B, "claude_runtime_binds", lambda: [])
    monkeypatch.setattr(toolchain, "toolchain_binds", lambda _: [])
    monkeypatch.setattr(toolchain, "sandbox_env", lambda *a, **k: "export PYTHONPATH=/original;")
    monkeypatch.setattr(B, "reapply_bundle_snapshot", lambda parts, *a, **k: [*parts, "frozen-original"])

    def masks(parts, *a, **k):
        seen.extend(parts)
        assert "frozen-original" in parts
        assert str(snapshot.root / "merlin/portable.py") in parts
        assert str(host.root / "private-answer.txt") not in parts
        return [*parts, "final-answer-mask"]

    monkeypatch.setattr(B, "apply_final_answer_masks", masks)
    with scoped_compiler_library(snapshot):
        command = execution.sandbox_command("true", project / "workspace", {}, context=context)
    assert seen
    argv = shlex.split(command)
    assert (
        argv.index("frozen-original")
        < argv.index(str(snapshot.root / "merlin/portable.py"))
        < argv.index("final-answer-mask")
    )
    assert "/perf-compiler-library/" in argv[-1]
    assert argv[-1].index("PYTHONPATH=/original") < argv[-1].index("/perf-compiler-library/")


def test_phase2_requires_same_independent_initial_selection(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    host, _ = selected(tmp_path)

    def admitted(prepared):
        functional = SimpleNamespace(run_dir=prepared.run_dir)
        assert S.require_phase1_selection(functional, host) == prepared.environment[L.KEY]
        with pytest.raises(S.StageGateError, match="independently reviewed"):
            S.require_phase1_selection(functional, None)
        other = tmp_path / "other"
        other.mkdir()
        foreign, _ = selected(other)
        with pytest.raises(S.StageGateError, match="independently reviewed"):
            S.require_phase1_selection(functional, foreign)
        leaf = S.verify_binding(prepared.environment[L.KEY], host).root / "merlin/portable.py"
        leaf.chmod(0o644)
        with pytest.raises(S.StageGateError, match="mutable"):
            S.require_phase1_selection(functional, host)
        return 0

    assert _run(request(project, host), admitted) == 0


def test_phase2_legacy_absence_does_not_claim_initial_access(tmp_path):
    host, _ = selected(tmp_path)
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "environment.yaml").write_text(yaml.safe_dump({"run_id": "old"}))
    assert S.require_phase1_selection(SimpleNamespace(run_dir=run), host) is None


def test_default_has_no_snapshot_inventory_or_overlay(tmp_path):
    request = SimpleNamespace(compiler_library_selection=None, options=SimpleNamespace(sandbox="none"))
    assert L.freeze_for_run(request, tmp_path / "run", prior=None, resuming=False) is None
    parts = ["unchanged"]
    assert L.sandbox_overlay(parts, workspace=tmp_path, bundle={}, context=None, private_run_dir=None) == (parts, "")
    assert not (tmp_path / "run").exists()


def test_actual_controller_carries_snapshot_through_readonly_box(project, monkeypatch, tmp_path):
    from merlin_experiments.phase1 import authoring, controller, task_staging, workspace_transport

    binary = shutil.which("bwrap")
    if binary is None:
        pytest.skip("requires existing Linux bwrap for actual readonly leaf transport")
    _external_substitutes(monkeypatch)
    host, _ = selected(tmp_path)
    base = _request(project)
    monkeypatch.setattr(
        task_staging, "callbacks", lambda config: SimpleNamespace(stage_task=stage, resolved_tools=resolved_tools)
    )
    monkeypatch.setattr(workspace_transport, "assemble", assemble)
    monkeypatch.setattr(workspace_transport, "probe", probe)
    observed = []

    def local_author(prepared, runtime):
        # External author process only is replaced; real controller admission,
        # readonly snapshot, import selection and mount builder remain in use.
        prepared.verify_inputs()
        active = active_compiler_library()
        assert active.record() == S.verify_binding(prepared.environment[L.KEY], host).record()
        argv = [binary, "--unshare-all", "--die-with-parent", "--clearenv", "--ro-bind", "/usr", "/usr"]
        for system_root in ("/lib", "/lib64"):
            if Path(system_root).exists():
                argv += ["--ro-bind", system_root, system_root]
        argv, _ = L.sandbox_overlay(
            argv,
            workspace=prepared.workspace,
            bundle={},
            context=prepared.request.context,
            private_run_dir=prepared.run_dir,
        )
        guest = "/perf-compiler-library/" + active.contract.sha256 + "/python"
        code = f"import sys;sys.path.insert(0,{guest!r});from merlin.portable import lower;assert lower(2)==7\n"
        code += (
            f"try: open({(guest + '/merlin/portable.py')!r}, 'w')\n"
            "except OSError: pass\nelse: raise AssertionError('writable library')\n"
        )
        result = subprocess.run(
            [*argv, "/usr/bin/python3", "-I", "-S", "-B", "-c", code], text=True, capture_output=True, check=False
        )
        assert result.returncode == 0, result.stderr
        observed.append(active.record())
        return 0

    monkeypatch.setattr(authoring, "execute", local_author)
    assert (
        controller.run(
            base.context,
            base.options,
            bundle_manifest=base.bundle_manifest,
            bundle_id="fixture",
            oracle_timing=project / "unused-timing.json",
            treatment=base.treatment,
            compiler_library_selection=host,
        )
        == 0
    )
    assert len(observed) == 1
    assert active_compiler_library() is None


def test_ordinary_phase2_retains_initial_binding_and_refuses_key_deletion(ordinary_inputs, tmp_path):
    from merlin_experiments.phase2 import ordinary_search_policy as O

    # Existing fixture substitutes external qualification, not library custody
    # or ordinary policy identity. No measurements or runtime qualification run.
    library = tmp_path / "reviewed-host"
    library.mkdir()
    host, _ = selected(library)
    path = ordinary_inputs.functional.run_dir / "environment.yaml"
    environment = yaml.safe_load(path.read_text())
    environment[L.KEY] = S.freeze_selection(host, tmp_path / "initial/python")
    path.write_text(yaml.safe_dump(environment))
    policy = O.prepare_ordinary_search_policy(**ordinary_inputs.arguments, compiler_library_selection=host)
    original = policy.binding()
    assert original["additional_compiler_library"] == host.record()
    del environment[L.KEY]
    path.write_text(yaml.safe_dump(environment))
    with pytest.raises(S.StageGateError, match="binding changed"):
        policy.binding()


def test_installed_cli_loads_explicit_original_library(project, monkeypatch, tmp_path):
    from merlin_experiments.phase1 import __main__, controller

    host, _ = selected(tmp_path)
    contract = tmp_path / "reviewed.json"
    contract.write_text(json.dumps(host.contract.record()))
    observed = []

    def continuation(context, options, **kwargs):
        observed.append(kwargs["compiler_library_selection"].record())
        assert kwargs["bundle_manifest"] == project / "input_bundles/fixture/input_bundle_manifest.yaml"
        return 0

    monkeypatch.setattr(controller, "run", continuation)
    argv = [
        "--run-id",
        "new-library",
        "--descriptor",
        str(project / "target_experiment.yaml"),
        "--repo",
        str(project),
        "--bundle-manifest",
        str(project / "input_bundles/fixture/input_bundle_manifest.yaml"),
        "--bundle",
        "fixture",
        "--oracle-timing",
        str(project / "timing.json"),
        "--compiler-library-root",
        str(host.root),
        "--compiler-library-contract",
        str(contract),
        "--compiler-library-contract-sha256",
        host.contract.sha256,
    ]
    assert __main__.main(argv) == 0
    assert observed == [host.record()]
    with pytest.raises(S.StageGateError, match="required together"):
        __main__.main(argv[:-2])
    assert len(observed) == 1


def test_library_is_frozen_before_trusted_task_callback(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    from merlin_experiments.phase1 import session

    host, _ = selected(tmp_path)
    source_paths = session.SI.paths

    def inventory(**kwargs):
        return {**source_paths(**kwargs), "phase1:source:test-library-stager": str(Path(__file__).resolve())}

    monkeypatch.setattr(session.SI, "paths", inventory)
    observed = []

    def stager(arm, workspace, run_dir, *, sandbox, bundle_dir):
        frozen = run_dir / "compiler_library/python/merlin/portable.py"
        observed.append(frozen.read_bytes())
        assert not frozen.stat().st_mode & 0o222
        (workspace / "TASK.md").write_text("Original task.\n")
        (run_dir / "TASK.md").write_bytes((workspace / "TASK.md").read_bytes())
        # A trusted callback cannot change the original release while benefiting
        # from the already-copied old bytes; admission must reopen both owners.
        (host.root / "merlin/portable.py").write_text("changed = True\n")

    base = request(project, host)
    base = replace(base, treatment=replace(base.treatment, stage_task=stager))
    with pytest.raises(ValueError, match="library|member"):
        _run(base, lambda _: pytest.fail("changed original release reached authoring"))
    assert len(observed) == 1


def test_overlay_reopens_original_archived_profile_and_preserves_exact_grants(tmp_path):
    from merlin.common.paths import repo_root
    from merlin.targetgen.sandbox import bwrap
    from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3, attach_common_substrate

    repo = repo_root()
    workspace, run = tmp_path / "workspace", tmp_path / "run"
    workspace.mkdir()
    run.mkdir()
    bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V3, python_source_root=repo / "src", host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(repo / "src" / row["path"].removeprefix("merlin/python/"))
    manifest = run / "input_bundle_manifest.yaml"
    manifest.write_text(yaml.safe_dump(bundle))
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo)
    host, _ = selected(tmp_path)
    binding = S.freeze_selection(host, tmp_path / "initial/python")
    snapshot = S.verify_binding(binding, host)
    with scoped_compiler_library(snapshot):
        parts, prefix = L.sandbox_overlay(
            ["box"], workspace=workspace, bundle=bundle, context=SimpleNamespace(repo=repo), private_run_dir=run
        )
        assert any(part.endswith("/merlin/transforms/normalization.py") for part in parts)
        assert str(snapshot.root / "merlin/portable.py") in parts
        assert not any(part.endswith("/merlin/verify/lean_checker.py") for part in parts)
        assert not any(part.endswith("/private-answer.txt") for part in parts)
        assert "/perf-compiler-library/" in prefix
        manifest.write_text("{}")
        with pytest.raises(ValueError, match="profile"):
            L.sandbox_overlay(
                ["box"], workspace=workspace, bundle=bundle, context=SimpleNamespace(repo=repo), private_run_dir=run
            )


def test_deleted_initial_binding_cannot_resume_as_legacy_or_enter_phase2(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    host, _ = selected(tmp_path)
    runs = []
    assert _run(request(project, host), lambda p: runs.append(p.run_dir) or 0) == 0
    path = runs[0] / "environment.yaml"
    environment = yaml.safe_load(path.read_text())
    del environment[L.KEY]
    path.write_text(yaml.safe_dump(environment))
    with pytest.raises(ValueError, match="binding is missing"):
        _run(_request(project, resume=True), lambda _: pytest.fail("bound run resumed as legacy"))
    with pytest.raises(S.StageGateError, match="binding is missing"):
        S.require_phase1_selection(SimpleNamespace(run_dir=runs[0]), None)


def test_reused_admission_owners_are_pinned_and_source_replacement_refuses(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    from merlin_experiments.phase1 import source_inputs as SI
    from merlin_experiments.spec import SpecError

    host, _ = selected(tmp_path)
    owners = (
        "merlin.targetgen.compiler_library",
        "merlin.targetgen.compiler_profile_admission",
        "merlin_experiments.phase2.compiler_library_state",
        "merlin_experiments.phase2.compiler_libraries",
        "merlin_experiments.phase2.compiler_library_guidance",
        "merlin_experiments.phase2.contracts",
    )

    def admitted(prepared):
        inputs = prepared.environment["implementation_sources"]["inputs"]
        paths = {row["path"] for row in inputs.values()}
        assert str(module_source_path("merlin_experiments.phase1.telemetry.report")) in paths
        for module in owners:
            source = module_source_path(module)
            assert str(source) in paths
            changed = tmp_path / (module + ".py")
            changed.write_bytes(source.read_bytes() + b"\n")
            original = SI._source
            with monkeypatch.context() as patch:
                patch.setattr(SI, "_source", lambda name: changed if name == module else original(name))
                with pytest.raises(SpecError, match="source identity changed"):
                    prepared.verify_inputs()
        prepared.verify_inputs()
        return 0

    assert _run(request(project, host), admitted) == 0


def test_authoring_companions_preserve_existing_patch_points(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import authoring
    from merlin_experiments.phase1.telemetry import report

    monkeypatch.setattr(authoring, "_PASS_LINE", "original chosen status")
    assert authoring._verified_status_line({}) == "original chosen status"
    monkeypatch.setattr(authoring, "_STATUS_LINES", ("earlier original status",))
    status = tmp_path / "REPORT.md"
    status.write_text("earlier original status\n")
    assert authoring._stamp_report_status(status, "current original status")
    assert status.read_text() == "current original status\n"
    observed = tmp_path / "run_timing.json"
    observed.write_text('{"observed": null}')
    calls = []
    monkeypatch.setattr(report, "write_run_timing", lambda root: calls.append(root) or observed)
    assert authoring._emit_run_timing(tmp_path, []) == {"observed": None}
    assert calls == [tmp_path]
