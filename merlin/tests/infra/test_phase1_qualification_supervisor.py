"""Qualification orchestration uses real bounded toy child trees, never models."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.digest import sha256_file
from merlin.common.paths import repo_root


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/phase1/launch"))
    return importlib.import_module("qualify_phase1")


def test_qualifier_command_keeps_inputs_without_author_options(helper, tmp_path):
    selected = [sys.executable, "-m", "merlin_experiments.phase1", "--run-id", "new",
                "--descriptor", "/selected/descriptor", "--private-full-model-spec", "/private/spec",
                "--codex-binary", "/client", "--codex-auth-source", "/credential-path-not-opened",
                "--codex-home-root", "/home-input", "--live-conformance-feedback",
                "--round-timeout", "7200", "--sandbox", "bwrap", "--bundle", "reviewed"]
    argv = helper.qualifier_argv(SimpleNamespace(argv=selected), tmp_path / "frozen")
    assert argv[-2:] == ["--qualify-submission", str(tmp_path / "frozen")]
    assert "--private-full-model-spec" in argv and "--descriptor" in argv
    assert not any("codex-" in arg or "conformance" in arg or "round-timeout" in arg for arg in argv)


def test_handoff_requires_original_oot_and_clipped_deadline(helper, tmp_path):
    submission = tmp_path / "frozen"
    submission.mkdir()
    package = {"path": str(submission), "hash_tree_sha256": "actual", "model_check_tree_sha256": "full"}
    F = SimpleNamespace(read=lambda p: json.loads(p.read_text()), package_identity=lambda p: package,
                        oot_repo=SimpleNamespace(tree_digest=lambda *_: "actual"))
    proof = tmp_path / "proof.json"
    value = {"schema": "merlin.operator.fresh_author_handoff.v1", "qualification_only": True,
             "authoring_converged": False, "original_run_dir": str(tmp_path / "old"),
             "native_oot_commit": "native", "frozen_submission": package, "original_hard_deadline_unix": 10}
    proof.write_text(json.dumps(value))
    assert helper.admit_handoff(F, proof, sha256_file(proof), submission, 9) == package
    with pytest.raises(ValueError, match="correspondence"):
        helper.admit_handoff(F, proof, sha256_file(proof), submission, 11)
    F.oot_repo.tree_digest = lambda *_: "different"
    with pytest.raises(ValueError, match="correspondence"):
        helper.admit_handoff(F, proof, sha256_file(proof), submission, 9)


def test_formal_result_requires_exact_frozen_source_and_qualification_scope(helper, tmp_path):
    path = tmp_path / "submission_qualification.json"
    value = {"schema": "merlin.phase1.submission_qualification.v1", "qualification_only": True,
             "authoring_converged": False, "selected_source_tree_sha256": "source",
             "graded_source_sha256": "source", "official_grade": {"complete": True}}
    F = SimpleNamespace(read=lambda p: json.loads(p.read_text()))
    native = {"process": SimpleNamespace(poll=lambda: 0)}
    path.write_text(json.dumps(value))
    assert helper.formal_complete(F, native, tmp_path, {"hash_tree_sha256": "source"})
    for key, bad in [("graded_source_sha256", "different"), ("qualification_only", False),
                     ("authoring_converged", True), ("official_grade", {"complete": False})]:
        path.write_text(json.dumps({**value, key: bad}))
        assert not helper.formal_complete(F, native, tmp_path, {"hash_tree_sha256": "source"})


def test_operator_checkpoint_reopens_actual_terminal_source_without_native_claim(helper, tmp_path):
    F = importlib.import_module("full_model_verification")
    original = tmp_path / "original"
    source = tmp_path / "workspace/submission"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text("entrypoints: {}\n")
    (source / "compiler.py").write_text("# genuinely authored terminal fixture\n")
    original.mkdir()
    (original / "environment.yaml").write_text(json.dumps({
        "run_id": original.name, "workspace_path": str(source.parent)}))
    checkpoint = tmp_path / "operator-oot"
    F.oot_repo.init(checkpoint)
    record = F.oot_repo.commit_candidate(checkpoint, source, label="operator terminal checkpoint",
                                         when=1, run_id=original.name)
    compiler = tmp_path / "frozen"
    F.oot_repo.export(checkpoint, record.commit, compiler)
    package = F.package_identity(compiler)
    proof = {"schema": "merlin.operator.fresh_author_handoff.v1", "qualification_only": True,
        "authoring_converged": False, "original_run_dir": str(original), "frozen_submission": package,
        "original_hard_deadline_unix": 10, "freeze_mode": "terminal_author_operator_checkpoint",
        "operator_oot_path": str(checkpoint), "operator_oot_commit": record.commit,
        "terminal_authored_source_path": str(source), "original_source_revision": "a" * 40,
        "original_empty_origin": True, **{k: package["hash_tree_sha256"] for k in (
            "actual_terminal_source_tree_sha256", "source_sha256_before", "source_sha256_after")}}
    path = tmp_path / "handoff.json"

    def admit(value):
        path.write_text(json.dumps(value))
        return helper.admit_handoff(F, path, sha256_file(path), compiler, 9)

    assert "native_oot_commit" not in proof and not (original / "oot").exists()
    assert admit(proof) == package
    for key, value in [("source_sha256_before", "wrong"), ("source_sha256_after", "wrong"),
                       ("terminal_authored_source_path", str(compiler)), ("original_empty_origin", False)]:
        with pytest.raises(ValueError, match="correspondence"):
            admit({**proof, key: value})
    (source / "compiler.py").write_text("# changed after captured terminal identity\n")
    with pytest.raises(ValueError, match="correspondence"):
        admit(proof)


@pytest.mark.parametrize("audit_status,native_exit,expected", [
    ("pass", 0, True), ("fail", 0, False), ("pass", 1, False)])
def test_all_four_roles_and_all_output_gate_are_required(helper, tmp_path, audit_status, native_exit, expected):
    result = _toy_supervisor(helper, tmp_path, audit_status=audit_status, native_exit=native_exit)
    state = json.loads((tmp_path / "prep/status.json").read_text())
    assert result.returncode == (0 if expected else 1), result.stderr
    assert state["phase1_complete"] is expected
    assert state["qualification_only"] is True and state["authoring_converged"] is False
    assert set(state["job_exit_codes"]) == {"native", "one", "two", "three"}
    assert state["cleanup"]["complete"] is True
    starts = []
    for name in state["job_exit_codes"]:
        pin = json.loads((tmp_path / f"prep/{name}-pin.json").read_text())
        assert pin["boot_id"] and pin["start_ticks"] and pin["process_group"] == pin["pid"]
        starts.append(float((tmp_path / f"{name}.started").read_text()))
    assert max(starts) - min(starts) < 1


def test_absolute_deadline_covers_native_and_escaped_descendants(helper, tmp_path):
    result = _toy_supervisor(helper, tmp_path, hang=True)
    state = json.loads((tmp_path / "prep/status.json").read_text())
    assert result.returncode == 2, result.stderr
    assert not state["phase1_complete"] and state["status"] == "stopped_incomplete"
    assert "DeadlineReached" in state["termination_reason"]
    assert state["cleanup"]["complete"] is True
    for path in tmp_path.glob("*.descendant"):
        pid, ticks = map(int, path.read_text().split())
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != ticks


def test_original_deadline_also_interrupts_offline_audit(helper, tmp_path):
    result = _toy_supervisor(helper, tmp_path, audit_status="block")
    state = json.loads((tmp_path / "prep/status.json").read_text())
    assert result.returncode == 2, result.stderr
    assert "DeadlineReached" in state["termination_reason"]
    assert state["phase1_complete"] is False and state["cleanup"]["complete"] is True


def _toy_supervisor(helper, tmp_path, *, audit_status="pass", native_exit=0, hang=False):
    """Run the real owner in its own process so pytest never becomes a subreaper."""
    script = tmp_path / "driver.py"
    script.write_text('''
import contextlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace as N
sys.path.insert(0, sys.argv[1])
import qualify_phase1 as Q
root=Path(sys.argv[2]); hang=sys.argv[3]=='True'; audit_status=sys.argv[4]; native_exit=int(sys.argv[5])
package={'path':str(root/'compiler'),'hash_tree_sha256':'source','model_check_tree_sha256':'full'}
(root/'compiler').mkdir(); (root/'proof').write_text('{}')
def command(name):
    code="import pathlib,time,sys,subprocess; p=pathlib.Path(sys.argv[1]); p.write_text(str(time.time())); "
    if hang:
        child=("import os,time,pathlib; s=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split(); "
               "pathlib.Path("+repr(str(root/(name+'.descendant')))+").write_text(str(os.getpid())+' '+s[19]); "
               "time.sleep(60)")
        code+="subprocess.Popen([sys.executable,'-c',"+repr(child)+"],start_new_session=True); time.sleep(60)"
    else:
        code+="time.sleep(.15); "
        if name!='native': code+="p.with_name('prep').joinpath("+repr(name+'.json')+").write_text('{}'); "
        code+="sys.exit("+str(native_exit if name=='native' else 0)+")"
    return [sys.executable,'-c',code,str(root/(name+'.started'))]
models=N(model=lambda n:N(name=n),path=root/'models')
output=N(validate=lambda *a,**k:{'actual':'fixture'},
         audit=lambda *a,**k:time.sleep(60) if audit_status=='block' else {'status':audit_status},
         finalize=lambda *a:{'status':'pass'})
F=N(selection=lambda l:{'names':['one','two','three'],'models':models},
    selected_identity=lambda l:{'source':'selected'},output_verification=lambda:output,
    package_identity=lambda p:package,model_command=lambda l,n,*a:command(n),
    read=lambda p:{'original_hard_deadline_unix':time.time()+10},report_complete=lambda *a:True)
old=Q.P.operator_module
Q.P.operator_module=lambda n:F if n=='full_model_verification' else old(n)
Q.P.prepare=lambda *a:{'status':'ready'};Q.P.memory=lambda r:(0,1000000)
Q.P.configured_environment=lambda l:contextlib.nullcontext()
Q.admit_handoff=lambda *a:package
Q.sha256_file=lambda p:'pinned'
Q.qualifier_argv=lambda *a:command('native')
Q.formal_complete=lambda F,n,*a:n['process'].poll()==0
policy={'output_scope':'all_declared','all_output_plan':{},'input':'visible/v00'}
cpus=sorted(os.sched_getaffinity(0))[:1]
launch=N(document={'full_model_verification':policy,'inputs':{'repo':str(root)},
                   'identity':{'target':'fixture','arm':'fixture'},
                   'cpu':{'author':{'cpus':[]},'compute_pool':{'cpus':cpus}}},
         variables={'RUN':str(root/'newrun'),'REFERENCE_CPUS':'','SUPERVISOR_PSS_MAX_BYTES':'1000000',
                    'SUPERVISOR_HOST_HEADROOM_BYTES':'1'},run_id='fresh-qualified',path=lambda p:Path(p))
raise SystemExit(Q.run(launch,root/'prep',root/'compiler',root/'proof','pinned',
                       time.time()+(1.2 if hang or audit_status=='block' else 8),poll_s=.02))
''')
    return subprocess.run([sys.executable, str(script), str(Path(helper.__file__).parent), str(tmp_path),
                           str(hang), audit_status, str(native_exit)], capture_output=True, text=True, timeout=15,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
