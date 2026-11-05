"""Compiler-product reuse primitives; no model execution or old verdict promotion."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import private_completed_build as C

from merlin.common import bundle_snapshot as B
from merlin.common import invocation_record as I
from merlin.compile import baremetal_model as BM
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.targetgen.sandbox import bwrap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def build(tmp_path, monkeypatch):
    """Actual returned Python/recipe observations around a synthetic compiler."""
    snapshot = tmp_path / "producer"
    relative = "packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_completed_build.py"
    helper = snapshot / relative
    core = snapshot / "src/merlin/compile/baremetal_model.py"
    core.parent.mkdir(parents=True)
    core.write_text("# selected synthetic compiler owner\n")
    helper.parent.mkdir(parents=True)
    helper.write_bytes(Path(C.__file__).read_bytes())
    seal = snapshot / "snapshot.json"
    seal.write_text("{}")
    selection = dict(
        schema=C.SCHEMA,
        cache_root=str(tmp_path / "cache"),
        producer_run=str(tmp_path / "original-run"),
        source_snapshot=C._pin(seal),
        sdk_roots=["selected"],
    )
    producer = {"source_snapshot": selection["source_snapshot"], "profile": "unchanged", "sdk": {}}
    release = {"files": {relative: sha(helper)}}
    populate, calls, admissions = [], [], []

    def selected(value, *, populate=False):
        assert value == selection
        if populate:
            build_state.populate.append(True)
            if not build_state.can_produce:
                raise ValueError("only the live selected original executor may populate completed builds")
        return deepcopy(producer), None, release

    def material(kwargs, context, *_):
        return {
            "source": deepcopy(producer),
            "capture": context["capture"],
            "candidate": context["candidate"],
            "policy": context["policy"],
            "option": kwargs.get("blob_weights", True),
        }

    def compiler(**kwargs):
        calls.append(kwargs["output"])
        output = BM._output_root(kwargs["output"])
        work = output / "build"
        work.mkdir(parents=True)
        source, obj, elf = (work / name for name in ("source.c", "source.o", "model.elf"))
        source.write_text("int main(void) { return 0; }\n")
        recipe = CompilationRecipe(work, producer=core)
        tool = Path(shutil.which("cc")).resolve()
        for src, dst, flags in ((source, obj, ["-c"]), (obj, elf, [])):
            recipe.run(
                [tool, *flags, src, "-o", dst],
                runner=lambda argv: subprocess.run(argv, capture_output=True),
                inputs=[src],
                output=dst,
            )
        recipe.completed(elf)
        result = {
            "schema": "merlin.baremetal-saved-model.v1",
            "status": "compiled",
            "output": {"elf": str(elf), "elf_sha256": sha(elf), "compilation_recipe": C._pin(recipe.path)},
        }
        (output / "baremetal_model.json").write_text(json.dumps(result))
        return result

    # Observe a real Python function with its actual separately selected source.
    scope = dict(C.__dict__)
    exec(compile(inspect.getsource(C._produce), str(helper), "exec"), scope)
    monkeypatch.setattr(C, "_produce", scope["_produce"])
    monkeypatch.setattr(C, "_producer", selected)
    monkeypatch.setattr(C, "_material", material)
    monkeypatch.setattr(BM, "compile_saved_model", compiler)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(C, "build_dir", lambda: tmp_path)
    original_source = C.module_source_path
    monkeypatch.setattr(
        C,
        "module_source_path",
        lambda module: core if module == "merlin.compile.baremetal_model" else original_source(module),
    )
    monkeypatch.setattr(C, "_admit", lambda receipt, kwargs, directory: admissions.append(directory) or {"new": True})
    build_state = SimpleNamespace(
        selection=selection,
        producer=producer,
        release=release,
        populate=populate,
        calls=calls,
        admissions=admissions,
        can_produce=True,
        context={"capture": "capture-1", "candidate": "candidate-1", "policy": "policy-1"},
        root=tmp_path,
        helper=helper,
        core=core,
    )

    def invoke(name, **kwargs):
        return C.compile_saved_model(
            completed_build_reuse=selection,
            build_context=build_state.context,
            output=tmp_path / "out" / name,
            capture=tmp_path / "capture",
            package=tmp_path / "package",
            **kwargs,
        )

    build_state.invoke = invoke
    return build_state


def test_completed_production_survives_later_checker_refusal_and_rechecks_current_isa(build):
    first = build.invoke("first")
    # A later checker result is neither cached nor required to issue production.
    checker = Path(first["output"]["elf"]).parent / "current-checker-refused.json"
    checker.write_text('{"passed": false}')
    second = build.invoke("second")
    assert len(build.calls) == 1
    assert first["completed_build_reuse"]["hit"] is False
    assert second["completed_build_reuse"]["hit"] is True
    assert first["completed_build_reuse"]["production_transaction_seconds"] > 0
    assert second["completed_build_reuse"]["production_transaction_seconds"] == 0
    assert len(build.admissions) == 1
    assert first["output"] == second["output"]
    assert second["completed_build_reuse"]["producer"] == build.producer
    assert (
        "all current private source/host/offload/numerical gates required" in second["completed_build_reuse"]["scope"]
    )
    retained = json.loads(Path(second["completed_build_reuse"]["production"]["path"]).read_bytes())
    assert "passed" not in retained
    invocation = I.verify(Path(retained["invocation"]["path"]))
    assert invocation["outcome"] == "returned"


@pytest.mark.parametrize("changed", ["source.c", "source.o", "model.elf", "compilation_recipe.json"])
def test_changed_consumed_product_or_recipe_refuses_hit(build, changed):
    first = build.invoke("first")
    path = Path(first["output"]["elf"]).parent / changed
    old_stat = path.stat()
    path.write_bytes(path.read_bytes() + b"changed")
    import os

    os.utime(path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    with pytest.raises((ValueError, RuntimeError)):
        build.invoke("second")
    assert len(build.calls) == 1


@pytest.mark.parametrize("field", ["capture", "candidate", "policy"])
def test_distinct_codegen_material_never_aliases(build, field):
    build.invoke("first")
    build.context[field] = "different"
    assert build.invoke("second")["completed_build_reuse"]["hit"] is False
    assert len(build.calls) == 2


def test_old_unissued_build_cannot_seed_cache_without_live_producer(build):
    build.can_produce = False
    with pytest.raises(ValueError, match="only the live selected original executor"):
        build.invoke("old-v7")
    assert not build.calls
    assert not list(build.root.rglob("production.json"))


@pytest.mark.parametrize("change", ["arguments", "executable", "outcome"])
def test_altered_production_invocation_cannot_authorize_reuse(build, change):
    first = build.invoke("first")
    entry = Path(first["completed_build_reuse"]["production"]["path"])
    record = json.loads(entry.read_bytes())
    invocation = Path(record["invocation"]["path"])
    document = json.loads(invocation.read_bytes())
    document[change] = {"different": True} if change != "outcome" else "interrupted"
    invocation.write_text(json.dumps(document))
    record["invocation"] = C._pin(invocation)
    entry.write_text(json.dumps(record))
    with pytest.raises((ValueError, KeyError)):
        build.invoke("second")
    assert len(build.calls) == 1


def test_fresh_output_guard_is_not_weakened_on_hit(build):
    build.invoke("first")
    with pytest.raises(BM.BaremetalModelError, match="fresh directory"):
        build.invoke("first")
    with pytest.raises(ValueError, match="canonical ordinary paths"):
        build.invoke("../../outside")


def test_current_policy_failure_does_not_turn_into_old_accepted_verdict(build, monkeypatch):
    build.invoke("first")

    def refuse(*_):
        raise ValueError("new policy refused")

    monkeypatch.setattr(C, "_admit", refuse)
    with pytest.raises(ValueError, match="new policy refused"):
        build.invoke("second")
    assert not (build.root / "out/second/baremetal_model.json").exists()
    assert len(build.calls) == 1


def test_default_has_no_reuse_or_selection_requirement(monkeypatch):
    calls = []
    monkeypatch.setattr(BM, "compile_saved_model", lambda **kwargs: calls.append(kwargs) or {"ordinary": True})
    assert C.compile_saved_model(capture="ordinary") == {"ordinary": True}
    assert calls == [{"capture": "ordinary"}]


def test_sdk_matches_real_original_archive_and_detects_membership_bytes_and_mode(tmp_path):
    repo, sdk, workspace = (tmp_path / name for name in ("repo", "sdk", "workspace"))
    for path in (repo, sdk, workspace):
        path.mkdir()
    library = sdk / "libshared.so"
    library.write_bytes(b"original-runtime")
    bundle = {"allowed": [{"path": str(sdk), "mode": "ro"}], "host_inputs": [], "denied": []}
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo)
    (frozen,) = B.snapshot_input_paths(workspace, bundle, [sdk], repo=repo)
    original = C._sdk(sdk, frozen)
    library.chmod(0o700)
    assert C._sdk(sdk, frozen) != original
    (sdk / "unexpected").write_bytes(b"new")
    with pytest.raises(ValueError, match="SDK membership differs"):
        C._sdk(sdk, frozen)
    (sdk / "unexpected").unlink()
    library.write_bytes(b"changed-runtime!")
    with pytest.raises(ValueError, match="SDK differs"):
        C._sdk(sdk, frozen)


def test_sdk_alias_cannot_hide_additional_runtime_members(tmp_path):
    sdk, frozen, other = (tmp_path / name for name in ("sdk", "frozen", "other"))
    for path in (sdk, frozen, other):
        path.mkdir()
    (sdk / "lib64").symlink_to(sdk, target_is_directory=True)
    (frozen / "lib64").mkdir()
    with pytest.raises(ValueError, match="alias cycle"):
        C._sdk(sdk, frozen)


def test_exact_v26_style_route_is_accepted_but_unknown_hooks_refuse(tmp_path):
    from dataclasses import replace

    from merlin.compile.complete_group_routing import select_complete_group
    from merlin.llvmlower.device_build import DeviceRouting

    device = DeviceRouting(
        device="test",
        package_dir=tmp_path,
        operand_dtype="i8",
        accum_dtype="i32",
        select=select_complete_group,
        granularity="group",
        require_complete_groups=True,
    )
    assert C._route(device)["require_complete_groups"] is True
    with pytest.raises(ValueError, match="unknown routing hooks"):
        C._route(replace(device, post_offload_transform=lambda *_: None))


def test_historical_profile_repo_is_not_actual_framework_source(tmp_path, monkeypatch):
    from merlin_experiments import frozen_python as FP

    monkeypatch.setattr(FP, "active_source_identity", lambda: None)
    with pytest.raises(ValueError, match="actual guarded frozen producer"):
        C._executing_release(tmp_path, {"path": "old-profile-seal"}, {"files": {}})


def test_actual_guarded_framework_release_accepts_distinct_original_profile_owner(tmp_path, monkeypatch):
    """Exercise the real immutable snapshot/import transport, without a compiler."""
    import sys

    from merlin_experiments import frozen_python as FP
    from merlin_experiments import source_snapshot as S

    from merlin.common.paths import repo_root

    root = repo_root()
    snapshot = tmp_path / "frozen-source"
    S.create(
        root,
        snapshot,
        output_root=tmp_path / "out",
        source_roots=(
            "src/merlin",
            "packages/merlin-experiments/src",
            "merlin/runtime",
            "merlin/contract",
            "merlin/schemas",
        ),
        python_roots=("src", "packages/merlin-experiments/src"),
        legacy_roots=(),
        runtime_links=False,
        root_files=(),
    )
    seal, release = S.load_seal(snapshot, "snapshot")
    program = (
        "from pathlib import Path; from merlin_experiments import source_snapshot as S; "
        "from merlin_experiments.phase1.feedback import private_completed_build as C; "
        "from merlin_experiments.phase1.feedback import compiler_profile_runtime as R; "
        f'root=Path({str(snapshot)!r}); seal,_=S.load_seal(root,"snapshot"); '
        "C._executing_release(root,C._pin(seal),S.verify(root)); "
        'assert R.active_original_execution() is None; print("guarded-producer-accepted")'
    )
    monkeypatch.delenv(FP.CONTEXT, raising=False)
    monkeypatch.delenv("MERLIN_RUNTIME_DIR", raising=False)
    try:
        result = subprocess.run(
            FP.python_command(snapshot, [sys.executable, "-c", program], verifier_source=Path(S.__file__)),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "guarded-producer-accepted"
        # It is the guarded framework seal, not source_root of an unrelated profile.
        assert release["source_root"] == str(root)
        assert C._pin(seal)["sha256"] == sha(seal)
    finally:
        for path in (snapshot, *snapshot.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)


def test_material_binds_actual_selected_tools_runtime_options_and_declared_inputs(tmp_path, monkeypatch):
    import sys

    from merlin.model_check import full_build_identity as BI

    monkeypatch.setenv("MERLIN_COMPILER_PYTHON", sys.executable)
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(tmp_path))
    monkeypatch.setenv("MERLIN_SCHEMAS_DIR", str(tmp_path))

    tool = tmp_path / "tool"
    tool.write_bytes(b"original compiler tool")
    capture, package = tmp_path / "capture", tmp_path / "package"
    for path in (capture, package):
        path.mkdir()
        (path / "payload").write_bytes(b"original")
    kwargs = dict(
        capture=capture,
        package=package,
        board_catalog=tool,
        board="board",
        dts=tool,
        output=tmp_path / "out",
        target="test",
        run="none",
        arena_mb=8,
        device=None,
        host_math_policy="native",
    )
    producer = dict(
        source_snapshot={"path": "independent-seal", "sha256": "a" * 64},
        sdk={
            str(tmp_path): "sdk-digest",
            str(Path(sys.executable).resolve().parent): "python-digest",
            str(Path(sys.executable).absolute().parent.parent.resolve()): "venv-digest",
        },
    )
    monkeypatch.setattr(
        BI,
        "build_material",
        lambda *_: {
            "source": {"actual-builder": "unchanged"},
            "tools": {"cc": C._pin(tool)},
            "compiler_driver": "python",
        },
    )
    monkeypatch.setattr(B, "snapshot_input_paths", lambda _ws, _bundle, paths, **_kw: paths)
    for name in (
        "LD_PRELOAD",
        "LD_AUDIT",
        "GCC_EXEC_PREFIX",
        "COMPILER_PATH",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "CPATH",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    archive = {"workspace": str(tmp_path), "repo": str(tmp_path), "manifest": {}}
    monkeypatch.setattr(C, "_read", lambda _: {})
    monkeypatch.setattr(C, "_runtime_paths", lambda tools: set(tools))
    monkeypatch.setattr(C, "_tool_defaults", lambda *_: set())
    owner = SimpleNamespace(workspace=tmp_path, bundle={}, repo=tmp_path)
    context = {"capability": {"entry": "explicit"}, "facts": {"config": "original"}}
    first = C._material(kwargs, context, {"runtime_archive": archive}, producer, owner)
    assert first["build"]["source"] == producer["source_snapshot"]
    arms = [
        C._material(
            {**kwargs, "host_elementwise_fusion": choice}, context, {"runtime_archive": archive}, producer, owner
        )
        for choice in (False, True)
    ]
    assert [arm["options"]["host_elementwise_fusion"] for arm in arms] == [False, True]
    assert C._digest(arms[0]) != C._digest(arms[1])
    (package / "payload").write_bytes(b"changed")
    assert C._material(kwargs, context, {"runtime_archive": archive}, producer, owner) != first
    with pytest.raises(ValueError, match="unsupported build options"):
        C._material({**kwargs, "graph_optimizations": object()}, context, {"runtime_archive": archive}, producer, owner)
    monkeypatch.setenv("CFLAGS", "-fplugin=/unknown/provider")
    with pytest.raises(ValueError, match="external compiler hooks"):
        C._material(kwargs, context, {"runtime_archive": archive}, producer, owner)


def test_bin_only_roots_do_not_cover_actual_runtime_defaults(monkeypatch):
    import sys

    monkeypatch.setenv("PATH", str(Path(sys.executable).resolve().parent))
    paths = C._runtime_paths([Path(sys.executable).resolve()])
    assert Path(sys.base_prefix).resolve() in paths
    assert Path("/usr/lib").resolve() in paths
    assert not all(path.is_relative_to(Path(sys.executable).resolve().parent) for path in paths)


def test_sdk_directory_alias_binds_its_target_and_complete_membership(tmp_path):
    sdk, frozen, other = (tmp_path / name for name in ("sdk", "frozen", "other"))
    for path in (sdk, frozen, other):
        path.mkdir()
    (other / "runtime").write_bytes(b"complete")
    (sdk / "lib64").symlink_to(other, target_is_directory=True)
    (frozen / "lib64").mkdir()
    (frozen / "lib64/runtime").write_bytes(b"complete")
    first = C._sdk(sdk, frozen)
    assert first == C._sdk(sdk, frozen)
    (other / "extra").write_bytes(b"unbound")
    with pytest.raises(ValueError, match="membership differs"):
        C._sdk(sdk, frozen)


def test_selected_default_queries_reopen_actual_archived_tool_and_explicit_namespaces(tmp_path, monkeypatch):
    tool = tmp_path / "query-tool"
    tool.write_text(
        '#!/bin/sh\ncase "$1" in\n-print-search-dirs) echo "programs: =/usr/bin"; '
        'echo "libraries: =/usr/lib";;\n-print-sysroot) echo /usr;;\n-p) '
        'echo "libtest.so => /usr/lib";;\n*) echo "statically linked";;\nesac\n'
    )
    tool.chmod(0o700)
    saved = tmp_path / "saved-tool"
    saved.write_bytes(tool.read_bytes())
    archive = {"workspace": str(tmp_path), "repo": str(tmp_path), "manifest": {}}
    monkeypatch.setattr(C, "_read", lambda _: {})
    monkeypatch.setattr(B, "snapshot_input_paths", lambda *_a, **_k: [saved])
    monkeypatch.setattr(C.shutil, "which", lambda _name: str(tool))
    tools = {name: C._pin(tool) for name in ("clang", "gcc")}
    paths = C._tool_defaults(tools, archive)
    assert {Path("/usr"), Path("/usr/bin"), Path("/usr/lib")} <= paths
    tool.write_text(tool.read_text() + "# changed default-query tool\n")
    with pytest.raises(ValueError, match="differs from selected archive"):
        C._tool_defaults(tools, archive)


def test_source_or_tool_dependency_mutation_refuses_returned_production(build):
    build.invoke("first")
    build.core.write_text("# changed actual compiler owner\n")
    with pytest.raises(ValueError, match="input, dependency or product changed"):
        build.invoke("second")
    assert len(build.calls) == 1


def test_actual_rpath_dependency_outside_tool_prefix_is_required(tmp_path, monkeypatch):
    tool = Path(shutil.which("cc")).resolve()
    vendor = tmp_path / "external-vendor"
    vendor.mkdir()
    source, library = vendor / "runtime.c", vendor / "libreuse_test.so"
    source.write_text("int actual_value(void) { return 7; }\n")
    main, executable = tmp_path / "main.c", tmp_path / "tool-prefix/bin/program"
    executable.parent.mkdir(parents=True)
    main.write_text("extern int actual_value(void); int main(void) { return actual_value(); }\n")
    subprocess.run(
        [str(tool), "-shared", "-fPIC", str(source), "-o", str(library)], capture_output=True, check=True, timeout=10
    )
    subprocess.run(
        [str(tool), str(main), "-L" + str(vendor), "-lreuse_test", "-Wl,-rpath," + str(vendor), "-o", str(executable)],
        capture_output=True,
        check=True,
        timeout=10,
    )
    original = subprocess.run

    def metadata(argv, **kwargs):
        if argv[1:] == ["-p"]:
            return subprocess.CompletedProcess(argv, 0, f"libreuse_test.so => {library}\n", "")
        return original(argv, **kwargs)

    monkeypatch.setattr(C.subprocess, "run", metadata)
    monkeypatch.setattr(C, "_read", lambda _: {})
    monkeypatch.setattr(B, "snapshot_input_paths", lambda _ws, _bundle, paths, **_kw: paths)
    archive = {"workspace": str(tmp_path), "repo": str(tmp_path), "manifest": {}}
    required = C._tool_defaults({"clang": C._pin(tool), "gcc": C._pin(tool), "program": C._pin(executable)}, archive)
    assert library in required
    assert not library.is_relative_to(executable.parent.parent)


def _current_policy(*, elf, evidence_root):
    """A real source-pinned service fixture, not an instruction decoder."""
    evidence_root.mkdir(parents=True)
    report = evidence_root / "report.json"
    result = {"status": "accepted", "elf_sha256": sha(elf)}
    report.write_text(json.dumps(result))
    return {**result, "report_path": str(report), "report_sha256": sha(report)}


def test_current_admission_uses_real_service_and_returned_invocation_each_time(tmp_path):
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    service = LinkedElfAdmissionService(
        target="test", evaluator=_current_policy, source_pins=((str(Path(__file__).resolve()), sha(Path(__file__))),)
    )
    device = DeviceRouting(
        device="test",
        package_dir=tmp_path,
        operand_dtype="i8",
        accum_dtype="i32",
        select=None,
        linked_elf_admission=service,
    )
    elf = tmp_path / "existing.elf"
    elf.write_bytes(b"synthetic artifact for service transport")
    receipt = {"output": {"elf": str(elf), "elf_sha256": sha(elf)}}
    for index in range(2):
        assert C._admit(receipt, {"device": device}, tmp_path / f"current-{index}")["status"] == "accepted"
    observations = list(tmp_path.glob("current-*/invocations/*/invocation.json"))
    assert len(observations) == 2
    assert all(I.verify(path)["outcome"] == "returned" for path in observations)
