"""Real admission snapshots and process-resume; external execution is synthetic."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import session as S
from merlin_experiments.phase1.context import load_context
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.treatments import Treatment
from merlin_experiments.phase1.workspaces import workspace_parent, workspace_session

from merlin.common import storage_lifecycle
from merlin.common.paths import data_path, module_source_path, python_import_roots
from merlin.targetgen.sandbox import bwrap as BW


def assemble(bundle, ws, sandbox, *, context):
    assert sandbox == "bwrap"
    BW.materialize_bundle_inputs(ws, bundle, repo=Path(os.environ["SESSION_TEST_ROOT"]))
    (ws / "submission").mkdir(parents=True)
    return S.AssemblyEvidence([], [], None)


def probe(ws, bundle, sandbox, *, context):
    # Only the external sandbox probe is substituted; real snapshot/mask policies
    # below still verify private inputs are withheld. No OS sandbox claim here.
    return {"pilot_golden_visible_to_agent": os.environ.get("SESSION_TEST_MASK", "OK")}


def stage(arm, ws, run_dir, *, sandbox, task_scope, policy_root):
    (ws / "TASK.md").write_text("Synthetic task: " + json.dumps(task_scope, sort_keys=True))
    shutil.copyfile(ws / "TASK.md", run_dir / "TASK.md")
    # Tools must be observed AFTER trusted task staging, as in the native controller.
    (Path(os.environ["SESSION_TEST_ROOT"]) / "input_bundles/fixture/tools.txt").write_text("after-stage\n")


def resolved_tools():
    path = Path(os.environ["SESSION_TEST_ROOT"]) / "input_bundles/fixture/tools.txt"
    return tuple(path.read_text().splitlines())


def stage_treatment(arm, ws, run_dir, *, sandbox, bundle_dir):
    """Real admission supplies the authored directory, not the effective archive."""
    assert bundle_dir == Path(os.environ["SESSION_TEST_ROOT"]) / "input_bundles/fixture"
    (ws / "TASK.md").write_text("Treatment task from " + str(bundle_dir))
    shutil.copyfile(ws / "TASK.md", run_dir / "TASK.md")
    (bundle_dir / "tools.txt").write_text("treatment-after-stage\n")


def _capsule(root, name, label):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "capsule.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "kind": "isa",
                "source_role": "handauthored_compiler_test",
                "label": label,
                "operation": {"op": "matmul", "attributes": {}},
                "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
                "expected": {"instruction_classes": [], "modes": {}},
                "required_oracle_tiers": ["L0", "L2"],
                "interface_mlir": "capsule.interface.mlir",
            }
        )
    )
    (directory / "capsule.interface.mlir").write_text("module {}\n")
    (directory / "golden.yaml").write_text("outputs: {value: PRIVATE_SENTINEL}\n")


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    shutil.copytree(data_path("contract", "schemas"), root / "merlin/contract/schemas")
    shutil.copytree(data_path("schemas"), root / "merlin/schemas")
    _capsule(root / "corpus/isa", "public_member", "public")
    _capsule(root / "corpus/hidden", "hidden_member", "hidden")
    (root / "target_experiment.yaml").write_text(
        yaml.safe_dump(
            {
                "target": "fixture",
                "capsule_corpus": str(root / "corpus/isa"),
            }
        )
    )
    directory = root / "input_bundles/fixture"
    directory.mkdir(parents=True)
    (directory / "input_bundle_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "bundle_id": "fixture",
                "allowed": [{"path": str(root / "corpus/isa")}],
                "denied": [],
                "host_inputs": [{"path": str(root / "corpus/hidden")}],
            }
        )
    )
    (directory / "tools.txt").write_text("before-stage\n")
    monkeypatch.setenv("SESSION_TEST_ROOT", str(root))
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(root))
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(root / "target_experiment.yaml"))
    monkeypatch.setenv("MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT", "")
    monkeypatch.setenv("MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED", "")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(root / "out"))
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(root / "merlin/contract"))
    monkeypatch.setenv("MERLIN_SCHEMAS_DIR", str(root / "merlin/schemas"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(root / "public-cas"))
    monkeypatch.delenv("MERLIN_CORPUS_SEAL", raising=False)
    return root


def _request(root, *, resume=False, no_oracle=False):
    context = load_context(root / "target_experiment.yaml", repo=root)
    options = parse_options(
        ["--run-id", "one"] + (["--resume"] if resume else []) + (["--no-oracle"] if no_oracle else [])
    )
    return S.RunRequest(
        context,
        options,
        Treatment(capsules_root=root / "corpus/isa"),
        root / "input_bundles/fixture/input_bundle_manifest.yaml",
        (),
        module_source_path("merlin_experiments.phase1.session"),
        False,
        {"config_dir": None, "email": None, "orgId": None},
        resolved_tools,
    )


def _external_substitutes(monkeypatch, *, oracle=True, codegen=None):
    from merlin.targetgen import capsule_runner
    from merlin.targetgen.sandbox import preflight

    paths = S.SI.paths

    def fixture_inventory(**kwargs):
        result = paths(**kwargs)
        # Explicit test-only external execution substitute: its bytes participate
        # in the same inventory/verification. Production admits no unlisted owner.
        result["phase1:source:test-transport"] = str(Path(__file__).resolve())
        return result

    monkeypatch.setattr(S.SI, "paths", fixture_inventory)
    # This admission fixture substitutes only host sandbox operability. Its
    # separate mask/snapshot checks still run through their production owners.
    monkeypatch.setattr(
        preflight, "require_working_sandbox", lambda **kwargs: preflight.SandboxProbe("ok", "synthetic host")
    )
    monkeypatch.setattr(capsule_runner, "oracle_available", lambda *args: (oracle, "synthetic oracle probe"))
    monkeypatch.setattr(capsule_runner, "codegen_smoke", lambda *args, **kwargs: (codegen, "synthetic codegen probe"))


def _run(request, continuation):
    @workspace_session
    def invocation(*, _workspace_leases):
        result = S.prepare(request, S.WorkspaceTransport(assemble, probe), stage, workspace_leases=_workspace_leases)
        return result if isinstance(result, int) else continuation(result)

    return invocation()


def test_real_prepare_retains_views_and_post_staging_tool_observation(project, monkeypatch):
    _external_substitutes(monkeypatch)
    request = _request(project)
    observed = []

    def continuation(prepared):
        prepared.verify_inputs()
        assert storage_lifecycle.blockers(prepared.workspace.parent)
        assert prepared.environment["resolved_tools"] == ["after-stage"]
        assert prepared.environment["treatment_snapshot"]["resolved_tool_ids"] == ["after-stage"]
        assert prepared.policy_root != prepared.public_root
        assert prepared.hidden_root.is_dir()
        assert not BW.is_exposed(
            BW.base_argv(prepared.workspace, prepared.bundle, repo=project),
            prepared.hidden_root / "hidden_member/golden.yaml",
        )
        observed.append(prepared)
        return 0

    assert _run(request, continuation) == 0
    assert not storage_lifecycle.blockers(observed[0].workspace.parent, require_terminal=True)


def test_treatment_gets_authored_bundle_and_resumes_without_restaging(project, monkeypatch):
    _external_substitutes(monkeypatch)
    request = _request(project)
    treatment = dataclasses.replace(request.treatment, stage_task=stage_treatment)
    request = dataclasses.replace(request, treatment=treatment)
    observed = []

    def continuation(prepared):
        prepared.verify_inputs()
        assert prepared.environment["resolved_tools"] == ["treatment-after-stage"]
        task = (prepared.workspace / "TASK.md").read_text()
        assert task == "Treatment task from " + str(project / "input_bundles/fixture")
        observed.append(prepared.resuming)
        return 0

    assert _run(request, continuation) == 0
    resumed = dataclasses.replace(_request(project, resume=True), treatment=treatment)
    assert _run(resumed, continuation) == 0
    assert observed == [False, True]


def test_admission_observation_order_remains_source_then_snapshot_task_receipt_probe(project, monkeypatch):
    _external_substitutes(monkeypatch)
    from merlin_experiments.phase1 import run_inputs, treatments

    from merlin.common import storage_lifecycle as storage

    labels = {
        function.__code__: name
        for name, function in (
            ("sources", S.SI.record),
            ("treatment", treatments.record),
            ("workspace", S.select_workspace_root),
            ("lease", storage.acquire),
            ("corpus", S.CI.prepare_bundle),
            ("assembly", assemble),
            ("hidden", run_inputs.subtree_snapshot_record),
            ("scope", S.task_scope),
            ("task", stage),
            ("tools", resolved_tools),
            ("receipt", run_inputs.treatment_snapshot_record),
            ("mask", probe),
        )
    }
    observed = []
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code in labels:
            observed.append(labels[frame.f_code])

    try:
        request = _request(project)
        sys.setprofile(observe)
        assert _run(request, lambda value: 0) == 0
    finally:
        sys.setprofile(previous)
    assert observed == [
        "sources",
        "treatment",
        "workspace",
        "lease",
        "corpus",
        "assembly",
        "hidden",
        "scope",
        "task",
        "tools",
        "receipt",
        "mask",
    ]


@pytest.mark.parametrize(
    ("oracle", "codegen", "no_oracle", "expected"),
    [
        (False, None, False, 4),
        (False, None, True, 0),
        (True, False, False, 4),
        (True, True, False, 0),
        (True, None, False, 0),
    ],
)
def test_actual_preflight_refusals_do_not_invoke_continuation(
    project, monkeypatch, oracle, codegen, no_oracle, expected
):
    _external_substitutes(monkeypatch, oracle=oracle, codegen=codegen)
    invoked = []
    assert _run(_request(project, no_oracle=no_oracle), lambda prepared: invoked.append(prepared) or 0) == expected
    assert bool(invoked) == (expected == 0)
    if oracle:
        receipts = list(project.rglob("codegen_smoke.yaml"))
        assert len(receipts) == 1
        assert yaml.safe_load(receipts[0].read_text())["codegen_ok"] is codegen
        assert yaml.safe_load(receipts[0].read_text())["backend_target"] is None


def test_inoperable_sandbox_refuses_before_workspace_snapshot_or_authoring(project, monkeypatch):
    from merlin.targetgen.sandbox import preflight

    def refuse(**kwargs):
        raise preflight.SandboxUnavailable(preflight.SandboxProbe("inoperable", "userns_uid_map_denied"))

    monkeypatch.setattr(preflight, "require_working_sandbox", refuse)
    assert _run(_request(project), lambda prepared: pytest.fail("inoperable host reached authoring")) == 4
    assert not (project / "out").exists()


@pytest.mark.parametrize("codegen", [True, False, None])
def test_explicit_codegen_provider_routes_original_target_and_persists_selection(project, monkeypatch, codegen):
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import capsule_runner

    dispatcher = capsule_runner.codegen_smoke
    _external_substitutes(monkeypatch)
    monkeypatch.setattr(capsule_runner, "codegen_smoke", dispatcher)
    descriptor = project / "target_experiment.yaml"
    document = yaml.safe_load(descriptor.read_text())
    document["preflight"] = {"codegen_backend": "declared_compiler"}
    descriptor.write_text(yaml.safe_dump(document))
    observed = []

    def hook(*, target):
        observed.append(("hook", target))
        return codegen, "selected compiler evidence"

    def backend(name):
        observed.append(("backend", name))
        return SimpleNamespace(preflight_codegen_smoke=hook)

    monkeypatch.setattr(backends, "get_backend", backend)
    invoked = []
    assert _run(_request(project), lambda prepared: invoked.append(prepared) or 0) == (4 if codegen is False else 0)
    assert bool(invoked) is (codegen is not False)
    assert observed == [("backend", "declared_compiler"), ("hook", "fixture")]
    receipts = list(project.rglob("codegen_smoke.yaml"))
    assert len(receipts) == 1
    assert yaml.safe_load(receipts[0].read_text()) == {
        "target": "fixture",
        "backend_target": "declared_compiler",
        "codegen_ok": codegen,
        "reason": "selected compiler evidence",
    }


@pytest.mark.parametrize("change", ["task", "frozen", "source", "treatment", "review"])
def test_resume_refuses_changed_evidence(project, monkeypatch, change):
    _external_substitutes(monkeypatch)
    prepared = []
    assert _run(_request(project), lambda value: prepared.append(value) or 0) == 0
    state = prepared[0]
    request = _request(project, resume=True)
    if change == "task":
        (state.workspace / "TASK.md").write_text("tampered task")
    elif change == "frozen":
        member = state.public_root / "public_member/golden.yaml"
        member.chmod(0o600)
        member.write_text("tampered private bytes")
    elif change == "treatment":
        request = dataclasses.replace(
            request, treatment=Treatment(name="different", capsules_root=project / "corpus/isa")
        )
    else:
        path = state.run_dir / "environment.yaml"
        data = yaml.safe_load(path.read_text())
        if change == "source":
            data["implementation_sources"]["inputs"]["phase1:source:test-transport"]["sha256"] = "0" * 64
        else:
            data["corpus_review"] = {"different": "review"}
        path.write_text(yaml.safe_dump(data))
    with pytest.raises((RuntimeError, ValueError)):
        _run(request, lambda value: pytest.fail("refused resume reached authoring"))


@pytest.mark.parametrize("mask_status", ["LEAK", "UNPROVEN"])
def test_mask_refusal_releases_lease_and_uncertain_continuation_retains_it(project, monkeypatch, mask_status):
    _external_substitutes(monkeypatch)
    monkeypatch.setenv("SESSION_TEST_MASK", mask_status)
    assert _run(_request(project), lambda value: pytest.fail("leak reached authoring")) == 5
    monkeypatch.delenv("SESSION_TEST_MASK")

    def fail(prepared):
        raise RuntimeError("uncertain child lifetime")

    with pytest.raises(RuntimeError, match="uncertain child"):
        _run(_request(project, resume=True), fail)
    request = _request(project, resume=True)
    environment = yaml.safe_load((request.context.runs / "raw_baseline/one/environment.yaml").read_text())
    assert storage_lifecycle.blockers(Path(environment["workspace_path"]).parent)


def test_callback_owner_is_not_silently_added_to_inventory(project, monkeypatch):
    request = _request(project)
    with pytest.raises(ValueError, match="outside the implementation inventory"):
        _run(request, lambda value: 0)
    assert not request.context.runs.exists()


def test_shared_callback_attribution_preserves_treatment_record_and_eager_shape_check():
    from merlin_experiments.phase1 import treatments

    with pytest.raises(KeyError, match="inputs"):
        treatments.record(Treatment(), {})
    path = Path(__file__).resolve()
    sources = {"inputs": {"fixture": {"path": str(path), "sha256": S.SI.fingerprint(path)}}}
    expected = {"module": stage.__module__, "qualname": stage.__qualname__, "source_input": "fixture"}
    assert treatments.callback_reference(stage, sources, label="admission callback") == expected
    assert treatments.record(Treatment(stage_task=stage), sources)["callbacks"]["stage_task"] == expected
    sources["inputs"]["fixture"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="treatment stage_task source changed"):
        treatments.record(Treatment(stage_task=stage), sources)


def test_noncanonical_bundle_filename_cannot_read_one_manifest_and_freeze_another(project):
    request = dataclasses.replace(_request(project), bundle_manifest=project / "different.yaml")
    request.bundle_manifest.write_text("bundle_id: different\n")
    with pytest.raises(ValueError, match="canonical input_bundle_manifest.yaml"):
        _run(request, lambda value: pytest.fail("wrong manifest reached authoring"))
    assert not request.context.runs.exists()


def test_repository_identity_uses_explicit_context_and_resume_retains_it(project, monkeypatch, tmp_path):
    _external_substitutes(monkeypatch)
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.name=Synthetic Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    expected = subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()
    request = _request(project)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(unrelated))
    observed = []
    assert _run(request, lambda value: observed.append(value) or 0) == 0
    assert observed[0].environment["repo_sha"] == expected
    original = (observed[0].run_dir / "environment.yaml").read_bytes()
    request = dataclasses.replace(request, options=dataclasses.replace(request.options, resume=True))
    assert _run(request, lambda value: 0) == 0
    assert (observed[0].run_dir / "environment.yaml").read_bytes() == original


@pytest.mark.parametrize("condition", ["stale", "seed-overlap", "resume-seed"])
def test_preserved_workspace_and_seed_are_refused_before_mutation(project, monkeypatch, condition):
    _external_substitutes(monkeypatch)
    request = _request(project)
    workspace = workspace_parent("fixture", request.options.arm) / request.options.run_id / "workspace"
    source = workspace / "submission"
    source.mkdir(parents=True)
    manifest = source / "manifest.yaml"
    manifest.write_text("language: python\n")
    if condition != "stale":
        options = dataclasses.replace(request.options, seed_submission=str(source), resume=condition == "resume-seed")
        request = dataclasses.replace(request, options=options)
    with pytest.raises((FileExistsError, RuntimeError)):
        _run(request, lambda value: pytest.fail("unsafe seed reached authoring"))
    assert manifest.read_text() == "language: python\n"
    assert not request.context.runs.exists()
    assert not storage_lifecycle.blockers(workspace.parent)


def test_session_import_does_not_initialize_target_or_process_environment(project):
    code = """
import os, subprocess
before = dict(os.environ)
def unexpected(*args, **kwargs):
    raise AssertionError('subprocess at import')
subprocess.run = unexpected
import merlin_experiments.phase1.session
assert dict(os.environ) == before
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(map(str, python_import_roots())),
        MERLIN_TARGET_EXPERIMENT="/missing/descriptor.yaml",
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=project, env=env, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr


def test_two_process_prepare_checkpoint_and_resume_without_native_imports(project):
    program = r"""
import importlib.abc, importlib.util, json, sys
from pathlib import Path
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in {'_common', 'run_baseline_qa_loop', 'run_agent_experiment'}:
            raise AssertionError('native import: ' + name)
sys.meta_path.insert(0, NoNative())
spec = importlib.util.spec_from_file_location('session_fixture', sys.argv[1])
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
from pytest import MonkeyPatch
with MonkeyPatch.context() as patch:
    fixture._external_substitutes(patch)
    request = fixture._request(Path(sys.argv[2]), resume=sys.argv[3] == 'resume')
    def continuation(prepared):
        prepared.verify_inputs()
        checkpoint = prepared.run_dir / 'checkpoint.json'
        if prepared.resuming:
            original = json.loads(checkpoint.read_text())
            assert original['environment'] == (prepared.run_dir / 'environment.yaml').read_text()
            assert original['task'] == (prepared.workspace / 'TASK.md').read_text()
            assert original['hidden'] == str(prepared.hidden_root)
        else:
            checkpoint.write_text(json.dumps({
                'environment': (prepared.run_dir / 'environment.yaml').read_text(),
                'task': (prepared.workspace / 'TASK.md').read_text(),
                'hidden': str(prepared.hidden_root),
            }))
        return 0
    assert fixture._run(request, continuation) == 0
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, python_import_roots())))
    for mode in ("fresh", "resume"):
        result = subprocess.run(
            [sys.executable, "-c", program, __file__, str(project), mode],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            timeout=40,
        )
        assert result.returncode == 0, result.stdout + result.stderr
