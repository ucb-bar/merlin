"""Original compiler owner controls; invocation stubs do not qualify hardware."""

import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2 import campaign as C

from merlin.targetgen import package_runtime as P


@pytest.fixture
def execution(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("compiler source")
    owner = R.CommonTransformService.__new__(R.CommonTransformService)
    from merlin.targetgen.sandbox import bwrap

    owner.repo = tmp_path / "selected-repo"
    owner.repo.mkdir()
    selected = owner.repo / "input"
    selected.write_text("selected original input")
    owner.workspace = tmp_path / "selected-workspace"
    owner.workspace.mkdir()
    owner.bundle = {"allowed": [{"path": str(selected), "mode": "ro"}], "host_inputs": [], "denied": []}
    bwrap.materialize_bundle_inputs(owner.workspace, owner.bundle, repo=owner.repo)
    monkeypatch.setattr(R.CommonTransformService, "compiler_imports", lambda self: nullcontext())
    from merlin.verify import lean_emission_context
    # This synthetic transport fixture has no original sealed admission. The
    # real emission context is exercised separately with an actual snapshot.
    monkeypatch.setattr(lean_emission_context, "original_emission_checks", lambda service: nullcontext())
    def policy(target, workspace, package, *, inputs):
        return C.PackageSandboxPolicy(("box", "--bind", str(workspace), str(workspace),
            "--ro-bind", str(package), str(package)), (), (), Path(workspace), Path(package), target)
    monkeypatch.setattr(C, "package_sandbox_policy", policy)
    from contextlib import contextmanager

    from merlin_experiments.phase1.brokers import compiler_call_channel
    @contextmanager
    def transport(*args, **kwargs):
        yield SimpleNamespace(env_prefix=lambda prefix:prefix,remaining_timeout=lambda:kwargs["timeout_s"])
    monkeypatch.setattr(compiler_call_channel,"compiler_call_transport",transport)
    return owner, candidate, tmp_path / "private"


def test_only_exact_original_owner_installs_executor(execution):
    owner, candidate, work = execution
    with pytest.raises(ValueError, match="exact owner"):
        with R.original_compiler_execution(SimpleNamespace(), target_experiment=None,
                candidate=candidate, work_root=work):
            pass
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        assert P.active_package_executor() is not None
    assert P.active_package_executor() is None
    assert list(work.iterdir()) == []
    with R.original_compiler_execution(None, target_experiment=None, candidate=candidate, work_root=work):
        assert P.active_package_executor() is None


def test_byte_identical_copy_allowed_changed_revision_and_exemption_refuse(execution):
    owner, candidate, work = execution
    relocated = candidate.parent / "copy"
    shutil.copytree(candidate, relocated)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        pkg = SimpleNamespace(directory=relocated, integrity_exempt=False)
        executor._verify(pkg)
        pkg.integrity_exempt = True
        with pytest.raises(ValueError, match="exemption"):
            executor._verify(pkg)
        pkg.integrity_exempt = False
        (relocated / "compiler.py").write_text("another compiler")
        with pytest.raises(ValueError, match="another source"):
            executor._verify(pkg)


def test_original_call_only_mounts_one_input_and_private_output(execution, monkeypatch):
    owner, candidate, work = execution
    incoming = candidate.parent / "input.mlir"
    incoming.write_text("module {}")
    output = candidate.parent / "output.json"
    sibling = candidate.parent / "private-reference"
    sibling.write_text("answer")
    pkg = SimpleNamespace(directory=candidate, integrity_exempt=False, manifest={})
    monkeypatch.setattr(P, "_resolve_argv", lambda pkg, name, inp, out: ["compiler", str(inp), str(out)])
    monkeypatch.setattr(P, "_needs_interpreter", lambda *args: False)
    observed = []
    def invoke(self, policy, package, argv, **kwargs):
        inp, out = map(Path, argv[-2:])
        assert inp.read_bytes() == incoming.read_bytes()
        assert inp.parent != incoming.parent
        assert str(sibling) not in policy.argv and str(incoming.parent) not in policy.argv
        assert ("--ro-bind", str(inp), str(inp)) == policy.argv[-3:]
        observed.append(inp.parent)
        out.write_text('{"real_output":1}')
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(R._Executor, "_invoke", invoke)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        result = P.active_package_executor().run_entrypoint(pkg, "emit", incoming, output)
        assert result.returncode == 0
    assert output.read_text() == '{"real_output":1}'
    assert all(not path.exists() for path in observed)


def test_work_owner_cannot_be_candidate_or_alias(execution):
    owner, candidate, work = execution
    for root in (candidate / "work", candidate.parent / "alias"):
        if root.name == "alias":
            root.symlink_to(candidate, target_is_directory=True)
        with pytest.raises(ValueError, match="independent canonical"):
            with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=root):
                pass


def test_frozen_public_policy_uses_only_original_read_only_grants(tmp_path, monkeypatch):
    original = tmp_path / "original"
    selected = tmp_path / "other-source/merlin/transforms/client.py"
    selected.parent.mkdir(parents=True)
    selected.write_text("public source")
    owner = SimpleNamespace(workspace=original, bundle={"frozen":True}, repo=tmp_path / "repo",
        profile={"common_sources":[{"path":"merlin/python/merlin/transforms/client.py",
            "sha256":__import__("hashlib").sha256(selected.read_bytes()).hexdigest()}],"treatment_tools":[]})
    policy = C.PackageSandboxPolicy(("box",), (), (), tmp_path / "private", tmp_path / "pkg", None,
                                   "export PYTHONPATH=selected; ")
    from merlin.targetgen.sandbox import bwrap as B
    monkeypatch.setattr(B, "_snapshot_grants", lambda *args: ({}, [(str(selected), selected, selected)]))
    monkeypatch.setattr(B, "_bundle_mount_args", lambda ws, bundle, repo: ["--ro-bind", str(selected), str(selected)])
    masked = []
    def mask(argv, target, workspace, bundle, *, repo):
        assert workspace == original and bundle is owner.bundle and repo == owner.repo
        masked.append(True)
        return [*argv, "--tmpfs", str(tmp_path / "private-checker")]
    monkeypatch.setattr(B, "apply_final_answer_masks", mask)
    result = R.public_package_policy(policy, target=None, owner=owner)
    assert "--bind" not in result.argv and str(original) not in result.argv
    assert result.argv[-2:] == ("--tmpfs", str(tmp_path / "private-checker"))
    assert masked and str(tmp_path / "other-source") in result.env_prefix
    assert str(owner.repo / "merlin/python") not in result.env_prefix
    selected.write_text("mutated public source")
    with pytest.raises(ValueError, match="bytes differ"):
        R.public_package_policy(policy, target=None, owner=owner)


@pytest.mark.parametrize("reject", [False, True])
def test_public_policy_preparation_leases_and_releases_real_shared_cpu(tmp_path, monkeypatch, reject):
    import fcntl
    import os

    from merlin.common import cpu_pool
    from merlin.targetgen.sandbox import bwrap as B

    original_affinity = os.sched_getaffinity(0)
    cpu = min(original_affinity)
    slots = tmp_path / "slots"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))
    owner = SimpleNamespace(workspace=tmp_path, bundle={}, repo=tmp_path)
    policy = C.PackageSandboxPolicy(("box",), (), (), tmp_path, tmp_path, None)

    def roots(selected):
        assert selected is owner
        assert cpu_pool.current_thread_owns_cpu_lease()
        assert os.sched_getaffinity(0) == {cpu}
        return (tmp_path,)

    def masks(argv, *args, **kwargs):
        assert cpu_pool.current_thread_owns_cpu_lease()
        with cpu_pool.shared_compute_pool(timeout_s=0):
            assert os.sched_getaffinity(0) == {cpu}
        if reject:
            raise RuntimeError("private runtime alias rejected")
        return [*argv, "--tmpfs", str(tmp_path / "private")]

    monkeypatch.setattr(R, "_public_python_roots", roots)
    monkeypatch.setattr(B, "_bundle_mount_args", lambda *args: [])
    monkeypatch.setattr(B, "apply_final_answer_masks", masks)
    if reject:
        with pytest.raises(RuntimeError, match="private runtime alias rejected"):
            R.public_package_policy(policy, target=None, owner=owner)
    else:
        assert R.public_package_policy(policy, target=None, owner=owner).argv[-2:] == (
            "--tmpfs", str(tmp_path / "private"))
    assert os.sched_getaffinity(0) == original_affinity
    assert not cpu_pool.current_thread_owns_cpu_lease()
    with (slots / f"cpu-{cpu}.lock").open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_actual_frozen_source_box_imports_selected_leaf_at_an_independent_root(tmp_path):
    """Real Linux mount/import control only; no compiler or deployment qualification."""
    import hashlib

    import yaml

    from merlin.common.paths import repo_root
    from merlin.targetgen.sandbox import bwrap as B
    from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3, attach_common_substrate

    if shutil.which("bwrap") is None:
        pytest.skip("bwrap is absent")
    repo = repo_root()
    selected = tmp_path / "selected-source"
    shutil.copytree(repo / "src/merlin", selected / "merlin", ignore=shutil.ignore_patterns("__pycache__"))
    workspace = tmp_path / "original"
    workspace.mkdir()
    bundle = {"arm":"raw_baseline","tools":[],"allowed":[],"denied":[],"host_inputs":[]}
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V3, python_source_root=selected, host_index_bits=64)
    for row in bundle["allowed"]:
        row["path"] = str(selected / row["path"].removeprefix("merlin/python/"))
    manifest = tmp_path / "archive.yaml"
    manifest.write_text(yaml.safe_dump(bundle))
    B.materialize_bundle_inputs(workspace, bundle, repo=repo)
    owner = R.CommonTransformService(workspace=workspace, bundle_manifest=manifest, repo=repo)
    policy = C.PackageSandboxPolicy(("bwrap", "--unshare-user", "--unshare-net", "--unshare-pid",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", str(selected)), (), (), tmp_path, tmp_path, None)
    # Final-mask selection is target-owned; this small Linux import control uses
    # only the materialized allow/deny mounts, and tests their private siblings.
    from unittest.mock import patch
    with owner.compiler_imports(), patch.object(B, "apply_final_answer_masks", side_effect=lambda argv,*a,**k:argv):
        actual = R.public_package_policy(policy, target=None, owner=owner)
        code = ("from merlin.verify.lean_contract import coverage_policy_record; "
                "assert coverage_policy_record()['mode']=='strict_within_coverage'; "
                "import importlib.util; assert importlib.util.find_spec('merlin.verify.lean_checker') is None; "
                "print('selected-public-source')")
        proc = subprocess.run([*actual.argv,"bash","-c",actual.env_prefix+'exec "$@"',
            "probe","/usr/bin/python3","-c",code], capture_output=True,text=True,timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "selected-public-source"
    assert str(selected) in actual.env_prefix and str(repo / "merlin/python") not in actual.env_prefix
    assert hashlib.sha256((selected / "merlin/verify/lean_contract.py").read_bytes()).hexdigest() == next(
        row['sha256'] for row in owner.profile['common_sources'] if row['module']=='merlin.verify.lean_contract')


def test_budget_child_cannot_drop_original_executor_and_inline_keeps_it(execution, monkeypatch):
    from merlin_experiments.phase2.contracts import StageGateError

    from merlin.targetgen import capsule_runner as CR

    owner, candidate, work = execution
    monkeypatch.setattr(CR, "_grade_candidate_model_capsule_inline", lambda *args,**kwargs: {
        "scope_owner_present": P.active_package_executor() is not None})
    monkeypatch.setattr(subprocess, "Popen", lambda *args,**kwargs: pytest.fail("unboxed child launch"))
    capsule = {"name":"public-program","kind":"model"}
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        # This fixture deliberately substitutes an empty import scope. Genuine
        # sealed child delegation has separate actual-interpreter controls.
        with pytest.raises(StageGateError, match="incomplete original execution owner"):
            CR._grade_model_capsule_unlocked(capsule, timeout=1, budget_s=1, candidate_context={})
        result = CR._grade_model_capsule_unlocked(capsule, timeout=1, budget_s=-1, candidate_context={})
        assert result["scope_owner_present"] is True


def test_development_executor_consumes_the_original_import_scope(execution, monkeypatch):
    from contextvars import ContextVar

    from merlin_experiments.phase2 import development_feedback as D

    owner, _, work = execution
    entered = ContextVar("test_original_scope",default=False)
    from contextlib import contextmanager
    @contextmanager
    def admission(self):
        token = entered.set(True)
        try:
            yield
        finally:
            entered.reset(token)
    monkeypatch.setattr(R.CommonTransformService, "compiler_imports", admission)
    def executor(**kwargs):
        assert entered.get() is True
        assert kwargs["timeout_s"] == 3
        return {"scope": "original"}
    feedback = D.DevelopmentGsimFeedback(None,None,work,"a"*64,None,{},work,{},executor=executor,
        _compiler_profile_service=owner)
    result = feedback._execute(arm="candidate",package=work,package_sha256="a"*64,
        member=SimpleNamespace(descriptor={}),decision=None,workspace=work,timeout_s=3)
    assert result == {"scope":"original"} and entered.get() is False


def test_original_declared_build_keeps_source_and_aggregate_deadline(execution, monkeypatch):
    owner, candidate, work = execution
    pkg = SimpleNamespace(directory=candidate,integrity_exempt=False,
        manifest={"build":{"configure":["configure"],"command":["build"]}})
    from merlin.targetgen.contract import toolchain
    monkeypatch.setattr(toolchain,"mlir_cmake_dir",lambda:work / "mlir")
    monkeypatch.setattr(toolchain,"mlir_install",lambda:work / "llvm")
    timeouts = []
    def invoke(self,policy,package,argv,**kwargs):
        assert policy.argv[-3:] == ("--bind",str(candidate),str(candidate))
        assert "CM=cmake" in policy.env_prefix
        timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(argv,0,"","")
    monkeypatch.setattr(R._Executor,"_invoke",invoke)
    with R.original_compiler_execution(owner,target_experiment=None,candidate=candidate,work_root=work):
        P.active_package_executor().build_package(pkg,timeout=5)
    assert len(timeouts)==2 and 0 < timeouts[1] <= timeouts[0] <= 5


def test_phase1_checkpoint_qa_and_emit_probes_keep_original_scope(execution, monkeypatch):
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.feedback import loop_grading as G
    from merlin_experiments.phase1.feedback import promotion

    owner, candidate, _ = execution
    root = candidate.parent
    ws = root / "workspace"
    shutil.copytree(candidate,ws / "submission")
    (ws / "submission/manifest.yaml").write_text("language: python\n")
    run = root / "run"
    context = InvocationContext(root,root / "descriptor.yaml",root,"fixture",run,
        root / "reports",root / "bundles",())
    inputs = G.GradingInputs(context,"raw_baseline",root / "public",(),None,compiler_profile_service=owner)
    monkeypatch.setattr(G,"load_target_experiment",lambda path:None)
    monkeypatch.setattr(G,"_language_ok",lambda *args:(True,""))
    monkeypatch.setattr(G.OH,"commit_graded",lambda *args,**kwargs:None)
    monkeypatch.setattr(G,"_write_stage_ledger",lambda *args,**kwargs:None)
    monkeypatch.setattr(G,"_record_plateau",lambda *args:None)
    monkeypatch.setattr(G.FL,"record_channel_health",lambda *args:None)
    monkeypatch.setattr(promotion,"resolve_tiers",lambda *args,**kwargs:(None,None,None))
    seen = []
    def qa(selected,*args,**kwargs):
        assert P.active_package_executor() is not None
        assert Path(selected) != ws / "submission"
        seen.append("numeric")
        return {"all_pass":True,"n_capsules":1,"n_passed":1,"per_capsule":[]}
    def emit(*args,**kwargs):
        assert P.active_package_executor() is not None
        seen.append("emit")
    monkeypatch.setattr(G,"_attach_shape_generalization",emit)
    monkeypatch.setattr(G,"_attach_codegen_scalability",emit)
    result = G.grade(ws,run,0,False,5,inputs=inputs,qa_runner=qa)
    assert result["all_pass"] is True
    assert seen == ["numeric","emit","emit"]
    assert P.active_package_executor() is None


def test_setup_scope_is_closed_before_grading(execution):
    from merlin.common import bundle_snapshot

    owner, candidate, work = execution
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        assert bundle_snapshot._ACTIVE_VERIFICATION.get() is None
        assert P.active_package_executor() is not None


def test_setup_mutation_prevents_grading_and_unwinds_execution(execution, monkeypatch):
    from merlin.common import bundle_snapshot

    owner, candidate, work = execution
    payload = bundle_snapshot.snapshot_path(
        bundle_snapshot.bundle_snapshot_root(owner.workspace), owner.repo / "input", owner.repo)
    original = R._Executor.__init__

    def changing_setup(self, *args, **kwargs):
        original(self, *args, **kwargs)
        payload.chmod(0o600)
        payload.write_text("tampered during setup")

    monkeypatch.setattr(R._Executor, "__init__", changing_setup)
    with pytest.raises(RuntimeError, match="content differs"):
        with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
            pytest.fail("unverified setup reached grading")
    assert P.active_package_executor() is None
    assert bundle_snapshot._ACTIVE_VERIFICATION.get() is None
