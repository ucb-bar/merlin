"""Snapshot treatment: actual isolated processes, no simulator or agent launch."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen import package_runtime as P
from merlin_experiments.phase1.feedback import snapshot_execution as S


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    from merlin_experiments.phase2 import campaign as C, compiler_libraries
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
    candidate = tmp_path / "candidate"; candidate.mkdir()
    tool = candidate / "tool.py"
    tool.write_text('''import json,os,sys
from pathlib import Path
incoming,outgoing=map(Path,sys.argv[1:3])
assert not Path(sys.argv[3]).exists(), "private sibling exposed"
assert os.environ.get("PRIVATE_FIXTURE_SECRET") is None
try:
    Path(__file__).write_text("mutate")
except OSError:
    pass
else:
    raise RuntimeError("compiler source is writable")
outgoing.write_text(json.dumps({"observed":incoming.read_text(),"cwd":str(Path.cwd())}))
''')
    private = tmp_path / "private-answer"; private.write_text("withheld")
    def policy(target, workspace, package, **kwargs):
        return SimpleNamespace(argv=("/usr/bin/bwrap", "--die-with-parent", "--unshare-pid", "--unshare-net",
            "--clearenv", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", str(package), str(package), "--bind", str(workspace), str(workspace)),
            env_prefix="export PATH=/usr/bin:/bin; ", workspace=workspace)
    monkeypatch.setattr(C, "package_sandbox_policy", policy)
    monkeypatch.setattr(R, "public_package_policy", lambda policy, **kw: policy)
    monkeypatch.setattr(compiler_libraries, "apply", lambda policy: policy)
    monkeypatch.setenv("PRIVATE_FIXTURE_SECRET", "must-not-reach-child")
    executor = S.SnapshotExecutor(SimpleNamespace(snapshot_digest="bundle", profile={"sha256":"profile"}),
        None, candidate, tmp_path / "host-work")
    pkg = P.Package(candidate, {"language":"python", "commands":{"parse":{"argv":[
        "/usr/bin/python3", "{tool}", "{input_mlir}", "{output_json}", str(private)]}}}, tool)
    return executor, pkg


def test_mode_is_explicit_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv(S.ENV, raising=False)
    assert S.selected_mode() == S.STRICT and not S.selected()
    monkeypatch.setenv(S.ENV, S.MODE)
    assert S.selected()
    monkeypatch.setenv(S.ENV, "arbitrary")
    with pytest.raises(ValueError, match="unknown compiler execution mode"):
        S.selected()


def test_real_boxed_invocations_preserve_outputs_and_private_boundary(snapshot, tmp_path):
    from merlin.common import invocation_record as I
    executor, pkg = snapshot
    incoming = tmp_path / "interface.mlir"; incoming.write_text("public source")
    evidence = tmp_path / "results"
    with P.scoped_package_executor(executor):
        assert S.active_snapshot() is executor
        for number in range(2):
            output = tmp_path / f"result{number}.json"
            result = P.run_entrypoint(pkg, "parse", incoming, output,
                                     invocation_directory=evidence, timeout=10)
            assert result.returncode == 0, result.stderr
            assert json.loads(output.read_text())["observed"] == "public source"
    receipts = list(evidence.glob("invocations/*/invocation.json"))
    assert len(receipts) == 2
    for receipt in receipts:
        record = I.verify(receipt)
        assert record["stage"] == "snapshot_external_package"
        assert len(record["outputs"]) == 1
        assert record["inputs_unchanged"]
    calls = list((evidence / "compiler-calls").iterdir())
    assert len(calls) == 2 and calls[0] != calls[1]
    executor.close()
    assert S.active_snapshot() is None
    with pytest.raises(ValueError, match="escaped"):
        executor.run_entrypoint(pkg, "parse", incoming)


def test_integrity_once_and_exit_rejects_source_mutation(snapshot):
    executor, pkg = snapshot
    called = []
    executor.check_integrity_once(pkg, lambda: called.append("scan"))
    executor.check_integrity_once(pkg, lambda: called.append("repeated"))
    assert called == ["scan"]
    pkg.tool.write_text("changed")
    with pytest.raises(ValueError, match="changed during evaluation"):
        executor.close()
    assert not executor.live


def test_failed_integrity_is_never_cached(snapshot):
    executor, pkg = snapshot
    def reject():
        raise ValueError("not admitted")
    with pytest.raises(ValueError, match="not admitted"):
        executor.check_integrity_once(pkg, reject)
    with pytest.raises(ValueError, match="not admitted"):
        executor.check_integrity_once(pkg, reject)
    executor.close()


def test_explicit_mode_cannot_fall_back_to_unboxed_missing_bundle(monkeypatch, tmp_path):
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
    monkeypatch.setenv(S.ENV, S.MODE)
    with pytest.raises(ValueError, match="selected input bundle"):
        with R.selfcheck_execution(None, context=SimpleNamespace(), submission=tmp_path,
                                  work_root=tmp_path / "work"):
            pytest.fail("missing input owner must not run")


def test_external_mode_has_no_cohort_curriculum(monkeypatch):
    from merlin_experiments.phase1.feedback import rolling_model_gate as G
    monkeypatch.setenv(S.ENV, S.MODE)
    # No broker promotion state is needed and no prerequisite is invented.
    assert G.prepare(context=None, contract=None, source=None, runs=None,
                     graded_root=None) == (None, None)


def test_native_workers_inherit_external_snapshot_without_strict_proofs(snapshot, monkeypatch, tmp_path):
    import threading
    from merlin.targetgen import capsule_runner as C
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
    executor, pkg = snapshot
    monkeypatch.setenv("MERLIN_OPERATOR_GRADE_PROCESSES", "0")
    monkeypatch.setattr(C, "load_package", lambda *a, **k: pkg)
    monkeypatch.setattr(C, "integrity_scan", lambda p: None)
    monkeypatch.setattr(C, "build_package", lambda p: None)
    monkeypatch.setattr(C, "_split_ineligible", lambda caps, target: (caps, []))
    monkeypatch.setattr(C._tier_policy, "covering_set", lambda caps: [])
    monkeypatch.setattr(C, "_CALIBRATION_CAP", 3)
    monkeypatch.setenv(S.ENV, S.MODE)
    barrier = threading.Barrier(2)
    threads = []
    def execute(cap, *args, **kwargs):
        assert S.active_snapshot() is executor
        assert R.active_original_execution() is None
        threads.append(threading.get_ident())
        barrier.wait(timeout=10)
        return {"capsule": cap["name"], "status": "incomplete", "tiers": {}}
    monkeypatch.setattr(C, "run_capsule", execute)
    with P.scoped_package_executor(executor):
        result = C._run_suite([{"name": "first"}, {"name": "second"}], pkg.directory,
                              runs_root=tmp_path / "runs", max_workers=2)
    assert len(set(threads)) == 2
    assert len(result) == 2 and all(r["status"] == "incomplete" for r in result)
    assert executor._workers == 0
    executor.close()


def test_worker_scope_refuses_foreign_owner_and_drains_on_error(snapshot):
    executor, _ = snapshot
    with pytest.raises(ValueError, match="live original owner"):
        with executor.concurrent_entrypoint_scope_v1():
            pytest.fail("foreign context entered")
    with P.scoped_package_executor(executor):
        with pytest.raises(RuntimeError, match="worker failed"):
            with executor.concurrent_entrypoint_scope_v1():
                raise RuntimeError("worker failed")
    assert executor._workers == 0
    executor.close()


@pytest.mark.parametrize("mutation", [None, "spec", "compiler"])
def test_sealed_external_model_transport_reaches_only_bound_child(snapshot, monkeypatch, tmp_path, mutation):
    import dataclasses
    import os
    import subprocess
    import sys
    from merlin_experiments.phase1.feedback import snapshot_model_child as M
    executor, pkg = snapshot
    @dataclasses.dataclass
    class Target:
        target: str
        path: Path
        source_root: object = None
    descriptor = tmp_path / "descriptor.json"; descriptor.write_text("{}")
    executor.target = Target("fixture", descriptor)
    executor.inputs = object()
    monkeypatch.setattr(M.C, "_inputs_record", lambda value: {})
    executor.owner.record = lambda: {}
    executor.owner._verify = lambda: None
    monkeypatch.setenv(S.ENV, S.MODE)
    spec = tmp_path / "model.json"
    spec.write_text(json.dumps({"package_dir": str(pkg.directory), "target": "fixture"}))
    script = """from pathlib import Path
import json,sys
from merlin_experiments.phase1.feedback import snapshot_model_child as M
p=Path(sys.argv[2])
d=M.read_selection(int(sys.argv[1]),spec_path=p,spec_bytes=p.read_bytes())
print(json.dumps(M.receipt(d)))
"""
    with P.scoped_package_executor(executor):
        with M.prepare(spec_path=spec, environment=dict(os.environ), package_dir=pkg.directory) as transport:
            if mutation == "spec": spec.write_text("{}")
            if mutation == "compiler": pkg.tool.write_text("changed")
            result = subprocess.run([sys.executable, "-c", script, str(transport.fd), str(spec)],
                                    pass_fds=transport.pass_fds, capture_output=True, text=True, timeout=15)
            if mutation is None:
                assert result.returncode == 0, result.stderr
                value = json.loads(result.stdout)
                assert transport.verify_result({M.FIELD: value, "status": "incomplete"})["status"] == "incomplete"
                with pytest.raises(ValueError, match="receipt"):
                    transport.verify_result({M.FIELD: {}})
            else:
                assert result.returncode != 0 and "changed" in result.stderr


def test_new_model_mode_cannot_use_unboxed_legacy_child(monkeypatch, tmp_path):
    from merlin_experiments.phase1.feedback import compiler_profile_child as C
    monkeypatch.setenv(S.ENV, S.MODE)
    spec = tmp_path / "model.json"; spec.write_text("{}")
    with pytest.raises(C.StageGateError, match="sealed child transport"):
        C.run_model_grade(["--model-grade", str(spec), "--model-grade-out", str(tmp_path / "out.json")])


def test_input_owner_binds_current_evaluator_without_claiming_legacy_proofs(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import snapshot_inputs as I
    workspace = tmp_path / "workspace"; workspace.mkdir()
    archive = tmp_path / "bundle.yaml"; archive.write_text("{}")
    current = tmp_path / "checker.py"; current.write_text("current independent checker")
    profile = {"sha256": "historical-public-profile", "automatic_checker_sources": []}
    monkeypatch.setattr(I, "verify_tool_profile", lambda bundle: profile)
    monkeypatch.setattr(I.B, "verify_bundle_snapshot", lambda *a, **kw: {"content_sha256": "inputs"})
    monkeypatch.setattr(I.B, "snapshot_marker_digest", lambda *a: "marker")
    monkeypatch.setattr(I, "module_source_path", lambda name: current)
    owner = I.SnapshotInputs(workspace=workspace, bundle_manifest=archive, repo=tmp_path)
    owner._verify()
    assert owner.record()["schema"] == "merlin.snapshot_external_inputs.v1"
    current.write_text("changed checker")
    with pytest.raises(ValueError, match="evaluator revision changed"):
        owner._verify()


def test_external_startup_tool_probe_does_not_require_proof_rpc(monkeypatch):
    from merlin_experiments.phase1 import tooling_readiness as R
    monkeypatch.setenv(S.ENV, S.MODE)
    monkeypatch.setattr(R, '_selected_library_probe', lambda: pytest.fail('legacy library probe selected'))
    text = R._sandbox_probe('fixture', (), '', common_substrate=True)
    assert 'apply_transform' not in text and 'lean_kernel' not in text
    assert 'AUTHORING_AND_BROKER_ROUNDTRIPS_OK' in text


def test_external_grade_binding_keeps_exact_current_input_owner(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
    from merlin_experiments.phase1.feedback.snapshot_inputs import SnapshotInputs
    monkeypatch.setenv(S.ENV, S.MODE)
    owner = object.__new__(SnapshotInputs)
    owner.bundle_manifest = tmp_path / 'input_bundle_manifest.yaml'
    checked = []
    owner._verify = lambda: checked.append('inputs')
    owner.record = lambda: {'schema': 'merlin.snapshot_external_inputs.v1'}
    assert R.compiler_profile_binding(owner, run_dir=tmp_path) == owner.record()
    assert checked == ['inputs']
    with pytest.raises(ValueError, match='another snapshot'):
        R.compiler_profile_binding(SimpleNamespace(), run_dir=tmp_path)


def test_external_mode_has_no_legacy_whole_corpus_proof_prerequisite(monkeypatch):
    from merlin_experiments.phase1.feedback import public_scalability_gate as G
    monkeypatch.setenv(S.ENV, S.MODE)
    monkeypatch.setattr(G, '_observe', lambda **kw: pytest.fail('legacy proof ladder selected'))
    assert G.observe_authoring(owner=None, candidate=None, context=None, capsules_root=None,
        contract=None, facts_path=None, work_root=None) is None


def test_ordinary_source_capture_uses_live_evaluator_evidence(snapshot, tmp_path):
    from merlin.targetgen import capsule_source_identity as source
    executor, pkg = snapshot
    generated = tmp_path / "generated"
    generated.mkdir()
    with P.scoped_package_executor(executor):
        executor.prepare_result_source(tmp_path / "results")
        captured = source._capture_compiler_source(pkg.directory, generated)
        assert captured["schema"] == "merlin.capsule_compiler_source.v2"
        assert Path(captured["snapshot"]).parent == tmp_path / "results" / "compiler-source"
        assert source._finish_compiler_source(captured, generated)["status"] == "stable"
        pkg.tool.write_text("changed source")
        assert source._finish_compiler_source(captured, generated)["status"] == "source_changed"
    with pytest.raises(ValueError, match="changed during evaluation"):
        executor.close()
