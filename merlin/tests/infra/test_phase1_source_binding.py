"""Native source attribution: real inventory and gate code, no agent or hardware execution."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import session as S
from merlin_experiments.phase1 import source_inputs as SI
from merlin_experiments.phase1 import treatments as T
from merlin_experiments.spec import SpecError

from merlin.common.paths import module_source_path, repo_root


@pytest.fixture
def native_sources(tmp_path):
    harness = tmp_path / "merlin/experiments/capsule_bench/harness"
    harness.mkdir(parents=True)
    for name in ("run_baseline_qa_loop.py", "_common.py", "sandbox_toolchain.py", "grade_agent_run.py", "broker.py"):
        (harness / name).write_text(f"# synthetic {name}\n")
    descriptor = tmp_path / "target/target_experiment.yaml"
    descriptor.parent.mkdir()
    descriptor.write_text("target: synthetic\n")
    (descriptor.parent / "scripts").symlink_to(harness, target_is_directory=True)
    return {"repo": tmp_path, "entrypoint": harness / "run_baseline_qa_loop.py", "descriptor": descriptor}


def _native_main(name="main"):
    source = repo_root() / "merlin/experiments/capsule_bench/harness/run_baseline_qa_loop.py"
    return next(
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_declared_numeric_recipe_bytes_and_absence_bind_resume(native_sources):
    original = SI.record(**native_sources)
    descriptor = native_sources["descriptor"]
    profile = native_sources["repo"] / "numeric.yaml"
    profile.write_text("datapath: {operand_dtype: fp32, subnormal_operand_flush: true}\n")
    descriptor.write_text("target: synthetic\nnumeric_profile: numeric.yaml\n")
    with pytest.raises(SpecError, match="identity changed"):
        SI.verify(original, **native_sources)
    declared = SI.record(**native_sources)
    assert declared["inputs"]["phase1:startup:numeric_profile"]["path"] == str(profile)
    SI.verify(declared, **native_sources)
    profile.write_text("datapath: {operand_dtype: fp32, subnormal_operand_flush: false}\n")
    with pytest.raises(SpecError, match="identity changed"):
        SI.verify(declared, **native_sources)
    descriptor.write_text("target: synthetic\n")
    with pytest.raises(SpecError, match="identity changed"):
        SI.verify(declared, **native_sources)


def _bind_native(native_sources, run_dir, *, resume, treatment=None):
    """Execute the package's actual pre-workspace statements and PreparedRun verifier."""
    source = module_source_path("merlin_experiments.phase1.session")
    body = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare"
    ).body
    start = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_source_context" for t in node.targets)
    )
    end = next(
        i
        for i, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_corpus_record" for t in node.targets)
    )
    callback_file = native_sources["entrypoint"].with_name("session_transports.py")
    callback_file.write_text("def callback(*args, **kwargs):\n    return None\n")
    scope = {"__name__": "session_transports"}
    exec(compile(callback_file.read_text(), str(callback_file), "exec"), scope)
    callback = scope["callback"]
    transport = S.WorkspaceTransport(callback, callback)
    request = SimpleNamespace(
        source_context={**native_sources, "require_native": True},
        resolved_tools=callback,
        treatment=treatment or T.Treatment(),
        context=SimpleNamespace(repo=native_sources["repo"]),
        options=SimpleNamespace(sandbox="copy"),
    )
    namespace = {
        "request": request,
        "transport": transport,
        "stage_task": callback,
        "SI": SI,
        "Path": Path,
        "yaml": yaml,
        "run_dir": run_dir,
        "_resuming": resume,
        "__file__": str(native_sources["entrypoint"]),
        "T": T,
        "treatment": request.treatment,
    }
    exec(compile(ast.Module(body=body[start:end], type_ignores=[]), "session-source-binding", "exec"), namespace)
    prepared = S.PreparedRun(
        request,
        run_dir,
        run_dir / "workspace",
        run_dir / "bundle",
        {},
        {
            "implementation_sources": namespace["_implementation_sources"],
            "invocation_treatment": namespace["_invocation_treatment"],
        },
        None,
        None,
        None,
        None,
        resume,
        None,
        transport,
        callback,
    )
    namespace["_verify_implementation_sources"] = prepared.verify_inputs
    return namespace


def test_shared_inventory_and_fingerprint_have_one_owner(native_sources):
    from merlin_experiments import runner

    assert runner.fingerprint is SI.fingerprint
    record = SI.record(**native_sources)
    command = {
        "env": {"MERLIN_REPO_ROOT": str(native_sources["repo"])},
        "entrypoint": str(native_sources["entrypoint"]),
        "inputs": {"descriptor": str(native_sources["descriptor"])},
    }
    assert runner._phase1_source_inputs(command) == {name: value["path"] for name, value in record["inputs"].items()}
    assert record["inputs"]["phase1:startup:formal_grader"]["path"] == str(
        module_source_path("merlin_experiments.phase1.feedback.formal")
    )
    assert record["inputs"]["phase1:startup:source_discovery_helper"]["path"] == str(
        module_source_path("merlin.common.source_membership")
    )
    assert "phase1:native:broker.py" in record["inputs"]
    assert "phase1:source:providers/bedrock_agent.py" in record["inputs"]
    SI.verify(record, **native_sources)


def test_prepared_prelaunch_binds_ownership_marker(native_sources, tmp_path):
    from merlin.targetgen.sandbox import bwrap as BW

    context = _bind_native(native_sources, tmp_path / "run", resume=False)
    verify = context["_verify_implementation_sources"]
    prepared = verify.__self__
    prepared.workspace.mkdir(parents=True)
    public = native_sources["repo"] / "public.txt"
    public.write_text("public input")
    prepared.bundle["allowed"] = [{"path": "public.txt"}]
    manifest = BW.materialize_bundle_inputs(prepared.workspace, prepared.bundle, repo=native_sources["repo"])
    prepared.environment["bundle_input_snapshot"] = BW.snapshot_record(prepared.workspace)
    prepared.request.options.sandbox = "bwrap"
    verify()
    marker = BW.bundle_snapshot_root(prepared.workspace) / "snapshot.json"
    manifest["support_ownership"]["owners"].append(str(tmp_path / "different-owner"))
    marker.chmod(0o600)
    marker.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="differs from host-owned run record"):
        verify()


@pytest.mark.parametrize("change", ["bytes", "member", "removed", "symlink", "fifo"])
def test_native_implementation_drift_prevents_gated_launch(native_sources, tmp_path, change):
    context = _bind_native(native_sources, tmp_path / "run", resume=False)
    harness = native_sources["entrypoint"].parent
    if change == "bytes":
        (harness / "broker.py").write_text("# changed source\n")
    elif change == "member":
        (harness / "new_broker.py").write_text("# new source\n")
    elif change == "removed":
        (harness / "broker.py").unlink()
    elif change == "symlink":
        (harness / "new_broker.py").symlink_to(harness / "broker.py")
    else:
        os.mkfifo(harness / "new_broker.py", 0o444)
    called = []
    # This is the actual package verifier used by the continuation, not a substitute.
    with pytest.raises(
        SpecError, match="source identity changed|outside the declared source inventory|nonregular entries"
    ):
        context["_verify_implementation_sources"]()
        called.append("launch")
    assert not called


def test_formal_binding_ignores_retired_target_alias_but_refuses_foreign_package_owner(
    native_sources, tmp_path, monkeypatch
):
    record = SI.record(**native_sources)
    scripts = native_sources["descriptor"].parent / "scripts"
    scripts.unlink()
    SI.verify(record, **native_sources)
    scripts.mkdir()
    foreign = tmp_path / "foreign.py"
    foreign.write_text("raise AssertionError('not an implementation')\n")
    (scripts / "grade_agent_run.py").symlink_to(foreign)
    SI.verify(record, **native_sources)
    real_source = SI._source
    monkeypatch.setattr(
        SI,
        "_source",
        lambda module: foreign if module == "merlin_experiments.phase1.feedback.formal" else real_source(module),
    )
    with pytest.raises(SpecError, match="source identity changed"):
        SI.verify(record, **native_sources)


@pytest.mark.parametrize("historical_record", [None, {}, {"version": 0}])
def test_native_resume_refuses_missing_attribution_without_rewriting_history(
    native_sources, tmp_path, historical_record
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    environment = run_dir / "environment.yaml"
    environment.write_text(yaml.safe_dump({"implementation_sources": historical_record, "old_evidence": "preserve"}))
    before = environment.read_bytes()
    with pytest.raises(SpecError, match="create a new qualified run"):
        _bind_native(native_sources, run_dir, resume=True)
    assert environment.read_bytes() == before


def test_native_fresh_record_and_resume_use_exact_same_identity(native_sources, tmp_path):
    run_dir = tmp_path / "run"
    fresh = _bind_native(native_sources, run_dir, resume=False)
    run_dir.mkdir()
    environment = run_dir / "environment.yaml"
    environment.write_text(
        yaml.safe_dump(
            {
                "implementation_sources": fresh["_implementation_sources"],
                "invocation_treatment": fresh["_invocation_treatment"],
            }
        )
    )
    before = environment.read_bytes()
    resumed = _bind_native(native_sources, run_dir, resume=True)
    resumed["_verify_implementation_sources"]()
    assert resumed["_implementation_sources"] == fresh["_implementation_sources"]
    assert environment.read_bytes() == before
    native_sources["entrypoint"].with_name("grade_agent_run.py").write_text("# grader changed after resume\n")
    with pytest.raises(SpecError, match="source identity changed"):
        resumed["_verify_implementation_sources"]()


def test_native_repo_override_cannot_disguise_controller_as_synthetic_adapter(native_sources, tmp_path):
    foreign = dict(native_sources, repo=tmp_path / "unrelated-work-root")
    with pytest.raises(SpecError, match="outside its declared repository binding"):
        _bind_native(foreign, tmp_path / "run", resume=False)


@pytest.mark.parametrize("change", ["absent", "name", "root", "callback"])
def test_native_resume_refuses_changed_treatment_before_workspace_mutation(native_sources, tmp_path, change):
    run_dir = tmp_path / "run"
    fresh = _bind_native(native_sources, run_dir, resume=False)
    run_dir.mkdir()
    treatment = dict(fresh["_invocation_treatment"])
    if change == "name":
        treatment["name"] = "different"
    elif change == "root":
        treatment["capsules_root"] = str(tmp_path / "other-corpus")
    elif change == "callback":
        treatment["callbacks"] = {"qa_runner": {"source_input": "forged"}}
    environment = run_dir / "environment.yaml"
    environment.write_text(
        yaml.safe_dump(
            {
                "implementation_sources": fresh["_implementation_sources"],
                "invocation_treatment": None if change == "absent" else treatment,
            }
        )
    )
    before = environment.read_bytes()
    with pytest.raises(RuntimeError, match="invocation treatment changed or is absent"):
        _bind_native(native_sources, run_dir, resume=True)
    assert environment.read_bytes() == before
    assert sorted(path.name for path in run_dir.iterdir()) == ["environment.yaml"]


def test_native_treatment_binds_real_callback_owner_and_refuses_uninventoried_function(native_sources, tmp_path):
    callback_file = native_sources["entrypoint"].with_name("treatment_fixture.py")
    callback_file.write_text("def feedback(root, *, capsule_roots):\n    return [{'verdict':'advisory-synthetic'}]\n")
    scope = {"__name__": "treatment_fixture"}
    exec(compile(callback_file.read_text(), str(callback_file), "exec"), scope)
    selected = T.Treatment(name="fixture", checkpoint_feedback=scope["feedback"])
    context = _bind_native(native_sources, tmp_path / "run", resume=False, treatment=selected)
    binding = context["_invocation_treatment"]["callbacks"]["checkpoint_feedback"]
    assert binding == {
        "module": "treatment_fixture",
        "qualname": "feedback",
        "source_input": "phase1:native:treatment_fixture.py",
    }
    assert selected.checkpoint_feedback(tmp_path, capsule_roots=()) == [{"verdict": "advisory-synthetic"}]
    with pytest.raises(ValueError, match="outside the implementation inventory"):
        _bind_native(
            native_sources, tmp_path / "foreign-run", resume=False, treatment=T.Treatment(qa_runner=lambda *args: {})
        )
    assert not (tmp_path / "foreign-run").exists()


def test_native_source_checks_dominate_authoring_and_official_grading():
    launcher = _native_main()
    invocation = launcher.body[-1].value
    assert ast.unparse(invocation.func) == "controller.run"
    assert any(k.arg == "require_native_source" and ast.literal_eval(k.value) is True for k in invocation.keywords)
    controller = next(
        node
        for node in ast.parse(module_source_path("merlin_experiments.phase1.controller").read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    scope = controller.body[-1]
    assert isinstance(scope, ast.With)
    assert ast.unparse(scope.items[0].context_expr) == "runtime_environment.applied_environment(runtime)"
    assert ast.unparse(scope.body[-3].value.func) == "session.prepare"
    assert ast.unparse(scope.body[-2].test) == "isinstance(prepared, int)"
    assert ast.unparse(scope.body[-2].body[0]) == "return prepared"
    assert ast.unparse(scope.body[-1].value.func) == "authoring.execute"
    tree = ast.parse(module_source_path("merlin_experiments.phase1.authoring").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute")
    assert any(
        isinstance(node, ast.Assign) and ast.unparse(node) == "_verify_implementation_sources = prepared.verify_inputs"
        for node in main.body
    )
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    checks = sorted(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "_verify_implementation_sources"
    )
    assert len(checks) >= 7  # continuous, regular, certification, repair, finalization, official pre/post
    for block in (node for node in ast.walk(main) if isinstance(node, (ast.While, ast.If, ast.FunctionDef))):
        if isinstance(block, ast.FunctionDef) and block.name == "_verilator_grade":
            assert ast.unparse(block.body[0]) == "_verify_implementation_sources()"
        if isinstance(block, ast.While) and ast.unparse(block.test) == "_keep_going()":
            assert ast.unparse(block.body[0]) == "_verify_implementation_sources()"
    grade = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "grade_proc" for t in node.targets)
    )
    owner = next(
        node for node in ast.walk(main) if hasattr(node, "body") and isinstance(node.body, list) and grade in node.body
    )
    index = owner.body.index(grade)
    assert ast.unparse(owner.body[index + 1]) == "_verify_implementation_sources()"
    argv = next(
        i
        for i, node in enumerate(owner.body)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "grade_cmd" for t in node.targets)
    )
    assert ast.unparse(owner.body[argv - 1]) == "_verify_implementation_sources()"
    # The binding must occur before workspace selection/lease acquisition, and be embedded in the
    # existing environment dictionary (not a second receipt or replacement treatment snapshot).
    preparation = next(
        node
        for node in ast.parse(module_source_path("merlin_experiments.phase1.session").read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare"
    )
    binding = next(
        node
        for node in preparation.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_invocation_treatment" for t in node.targets)
    )
    workspace = next(
        node
        for node in ast.walk(preparation)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "select_workspace_root"
    )
    assert binding.end_lineno < workspace.lineno
    assert any(
        isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant)
            and key.value == "implementation_sources"
            and isinstance(value, ast.Name)
            and value.id == "_implementation_sources"
            for key, value in zip(node.keys, node.values, strict=True)
        )
        for node in ast.walk(preparation)
    )
