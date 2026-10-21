"""Qualification orchestration uses real bounded toy child trees, never models."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.digest import sha256_file
from merlin.common.paths import repo_root


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/phase1/launch"))
    return importlib.import_module("qualify_phase1")


def test_selected_config_path_uses_launcher_virtualenv_for_manifest_python(helper, tmp_path, monkeypatch):
    config = yaml.safe_load((repo_root() / "examples/gemmini/phase1/launch/phase1_el4_config.yaml").read_text())
    prefix = config["environment"]["search_paths"]["PATH"]["prepend"]
    assert prefix.split(os.pathsep)[0] == str(Path(config["variables"]["PYTHON"]).parent)
    prefix = prefix.replace("${TASK_ROOT}", str(tmp_path)).replace("${WORKSPACE_ROOT}", str(tmp_path / "work"))
    selected_bin = tmp_path / "merlin/.venv/bin"
    selected_bin.mkdir(parents=True)
    (selected_bin / "python3").symlink_to(sys.executable)
    unrelated_bin = tmp_path / "unrelated"
    unrelated_bin.mkdir()
    unrelated = unrelated_bin / "python3"
    unrelated.write_text("#!/bin/sh\nexit 9\n")
    unrelated.chmod(0o755)
    monkeypatch.setenv("PATH", str(unrelated_bin))
    launch = SimpleNamespace(environment={"PATH": helper.P.L.Prepend(prefix)}, launch_environment={})
    with helper.P.configured_environment(launch):
        result = subprocess.run(
            ["python3", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, check=True
        )
    assert result.stdout.strip() == str(selected_bin / "python3")
    assert os.environ["PATH"] == str(unrelated_bin)


def test_qualifier_command_keeps_inputs_without_author_options(helper, tmp_path):
    selected = [sys.executable, "-m", "merlin_experiments.phase1", "--run-id", "new",
                "--descriptor", "/selected/descriptor", "--private-full-model-spec", "/private/spec",
                "--codex-binary", "/client", "--codex-auth-source", "/credential-path-not-opened",
                "--codex-home-root", "/home-input", "--live-conformance-feedback",
                "--round-timeout", "7200", "--sandbox", "bwrap", "--bundle", "reviewed",
                "--validation-scope", "capsules"]
    argv = helper.qualifier_argv(SimpleNamespace(argv=selected), tmp_path / "frozen")
    assert argv[-2:] == ["--qualify-submission", str(tmp_path / "frozen")]
    assert "--private-full-model-spec" in argv and "--descriptor" in argv
    assert argv[argv.index("--validation-scope") + 1] == "capsules"
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


@pytest.fixture
def continued_budget(tmp_path):
    authorized = time.time() - 5
    submission = tmp_path / "compiler"
    submission.mkdir()
    package = {"path": str(submission), "hash_tree_sha256": "a" * 64, "model_check_tree_sha256": "b" * 64}
    proof = {"schema": "merlin.operator.fresh_author_handoff.v1", "qualification_only": True,
             "authoring_converged": False, "original_run_dir": str(tmp_path / "old"),
             "native_oot_commit": "actual", "frozen_submission": package,
             "original_hard_deadline_unix": authorized - 100}
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(proof))
    authority = tmp_path / "continuation.json"
    authority.write_text(json.dumps({"authorized_by_user_text": "Continue for this bounded qualification",
        "new_author_allowed": False, "started_unix": authorized, "absolute_deadline_unix": authorized + 200}))
    value = {"schema": "merlin.operator.qualification_budget_extension.v1",
             "scope": "qualification_of_existing_frozen_compiler_only", "qualification_only": True,
             "new_author_allowed": False, "original_run_dir": proof["original_run_dir"],
             "original_handoff_sha256": sha256_file(path),
             "original_hard_deadline_unix": proof["original_hard_deadline_unix"],
             "submission_sha256": package["hash_tree_sha256"], "authorized_at_unix": authorized,
             "wall_budget_s": 300, "absolute_deadline_unix": authorized + 200,
             "authorization": {"path": str(authority), "sha256": sha256_file(authority)}}
    extension = tmp_path / "budget.json"
    extension.write_text(json.dumps(value))
    F = SimpleNamespace(read=lambda p: json.loads(p.read_bytes()), package_identity=lambda p: package,
                        oot_repo=SimpleNamespace(tree_digest=lambda *_: package["hash_tree_sha256"]))
    return SimpleNamespace(F=F, proof=path, original=path.read_bytes(), compiler=submission, package=package,
                           extension=extension, value=value, authority=authority, deadline=authorized + 100)


def admit_continued(helper, fixture):
    return helper.admit_handoff(fixture.F, fixture.proof, sha256_file(fixture.proof), fixture.compiler,
        fixture.deadline, budget_extension=fixture.extension,
        budget_extension_sha256=sha256_file(fixture.extension))


def test_explicit_continuation_keeps_original_handoff_and_candidate_checks(helper, continued_budget):
    f = continued_budget
    with pytest.raises(ValueError, match="correspondence"):
        helper.admit_handoff(f.F, f.proof, sha256_file(f.proof), f.compiler, f.deadline)
    assert admit_continued(helper, f) == f.package
    assert f.proof.read_bytes() == f.original
    f.F.oot_repo.tree_digest = lambda *_: "different"
    with pytest.raises(ValueError, match="correspondence"):
        admit_continued(helper, f)


@pytest.mark.parametrize("field,value", [
    ("scope", "new_author"), ("new_author_allowed", True), ("qualification_only", False),
    ("original_run_dir", "/another-run"), ("original_handoff_sha256", "f" * 64),
    ("submission_sha256", "f" * 64), ("original_hard_deadline_unix", 1),
    ("wall_budget_s", True), ("wall_budget_s", 1), ("absolute_deadline_unix", float("inf")),
    ("authorized_at_unix", float("nan")), ("extra", "not allowed"),
])
def test_budget_continuation_refuses_changed_scope_identity_or_unbounded_time(helper, continued_budget, field, value):
    f = continued_budget
    f.extension.write_text(json.dumps({**f.value, field: value}))
    with pytest.raises(ValueError):
        admit_continued(helper, f)


def test_budget_continuation_refuses_expired_selection_changed_authority_and_symlink(helper, continued_budget):
    f = continued_budget
    f.deadline = f.value["absolute_deadline_unix"] + 1
    with pytest.raises(ValueError):
        admit_continued(helper, f)
    f.deadline -= 100
    original = f.authority.read_bytes()
    f.authority.write_text("{}")
    with pytest.raises(ValueError, match="authorization"):
        admit_continued(helper, f)
    f.authority.write_bytes(original)
    indirect = f.extension.with_name("indirect.json")
    indirect.symlink_to(f.extension)
    f.extension = indirect
    with pytest.raises(ValueError, match="ordinary"):
        admit_continued(helper, f)


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


def _toy_supervisor(helper, tmp_path, *, audit_status="pass", native_exit=0, hang=False, scope="full",
                    scenario="ordinary"):
    """Run the real owner in its own process so pytest never becomes a subreaper."""
    script = tmp_path / "driver.py"
    script.write_text('''
import contextlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace as N
sys.path.insert(0, sys.argv[1])
import qualify_phase1 as Q
root=Path(sys.argv[2]); hang=sys.argv[3]=='True'; audit_status=sys.argv[4]
native_exit=int(sys.argv[5]); scope=sys.argv[6]; scenario=sys.argv[7]
package={'path':str(root/'compiler'),'hash_tree_sha256':'source','model_check_tree_sha256':'full'}
(root/'compiler').mkdir(); (root/'proof').write_text('{}')
def command(name):
    code="import pathlib,time,sys,subprocess; p=pathlib.Path(sys.argv[1]); p.write_text(str(time.time())); "
    if hang and not (scenario!='ordinary' and name=='one'):
        child=("import os,time,pathlib; s=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split(); "
               "pathlib.Path("+repr(str(root/(name+'.descendant')))+").write_text(str(os.getpid())+' '+s[19]); "
               "time.sleep(60)")
        code+="subprocess.Popen([sys.executable,'-c',"+repr(child)+"],start_new_session=True); time.sleep(60)"
    else:
        delay=(.1 if scenario=='native_failure' and name=='native' else 60 if name=='native'
               else .1 if name=='one' else .6) if scenario!='ordinary' else .15
        code+="time.sleep("+str(delay)+"); "
        if name!='native': code+="p.with_name('prep').joinpath("+repr(name+'.json')+").write_text('{}'); "
        if scenario=='job_integrity' and name=='one':
            code+="p.with_name('prep').joinpath('one-pin.json').write_text('changed original job'); "
        code+="p.with_suffix('.finished').write_text(str(time.time())); "
        if scenario=='signal' and name=='one': code+="__import__('os').kill(__import__('os').getpid(),9); "
        if scenario=='operator' and name=='one': code+="__import__('os').kill(__import__('os').getppid(),15); "
        model_fails=scenario not in {'ordinary','native_failure','invalid_report'} and name=='one'
        code+="sys.exit("+str(native_exit if name=='native' else 7 if model_fails else 0)+")"
    return [sys.executable,'-c',code,str(root/(name+'.started'))]
models=N(model=lambda n:N(name=n),path=root/'models')
output=N(validate=lambda *a,**k:{'actual':'fixture'},
         audit=lambda *a,**k:time.sleep(60) if audit_status=='block' else {'status':audit_status},
         finalize=lambda *a:{'status':'pass'})
F=N(selection=lambda l:{'names':['one','two','three'],'models':models},
    selected_identity=lambda l:{'source':'selected'},output_verification=lambda:output,
    package_identity=lambda p:package,model_command=lambda l,n,*a:command(n),
    read=lambda p:{'original_hard_deadline_unix':time.time()+10},report_complete=lambda *a:True)
if scenario=='invalid_report': F.report_complete=lambda report,model,*a:model.name!='one'
old=Q.P.operator_module
Q.P.operator_module=lambda n:F if n=='full_model_verification' else old(n)
Q.P.prepare=lambda *a:{'status':'ready'};Q.P.memory=lambda r:(0,1000000)
if scenario=='memory': Q.P.memory=lambda r:(1000001 if (root/'one.finished').exists() else 0,1000000)
if scenario=='integrity':
    F.package_identity=lambda p:{**package,'hash_tree_sha256':'changed'} if (root/'one.finished').exists() else package
Q.P.configured_environment=lambda l:contextlib.nullcontext()
Q.admit_handoff=lambda *a:package
Q.sha256_file=lambda p:'pinned'
if scenario=='job_integrity':
    import hashlib
    Q.sha256_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest() if p.name=='one-pin.json' else 'pinned'
Q.qualifier_argv=lambda *a:command('native')
Q.formal_complete=lambda F,n,*a,**kw:n['process'].poll()==0
if scope=='capsules':
    def forbidden(*a,**kw): raise AssertionError('external model path selected')
    F.selection=forbidden; F.output_verification=forbidden; F.model_command=forbidden
policy={'output_scope':'all_declared','all_output_plan':{},'input':'visible/v00'}
cpus=sorted(os.sched_getaffinity(0))[:1]
launch=N(document={'validation_scope':scope,'full_model_verification':policy,'inputs':{'repo':str(root),'descriptor':str(root/'descriptor')},
                   'identity':{'target':'fixture','arm':'fixture'},
                   'cpu':{'author':{'cpus':[]},'compute_pool':{'cpus':cpus}}},
         variables={'RUN':str(root/'newrun'),'REFERENCE_CPUS':'','SUPERVISOR_PSS_MAX_BYTES':'1000000',
                    'SUPERVISOR_HOST_HEADROOM_BYTES':'1'},run_id='fresh-qualified',path=lambda p:Path(p))
raise SystemExit(Q.run(launch,root/'prep',root/'compiler',root/'proof','pinned',
                       time.time()+(1.2 if hang or audit_status=='block' else 8),poll_s=.02))
''')
    return subprocess.run([sys.executable, str(script), str(Path(helper.__file__).parent), str(tmp_path),
                           str(hang), audit_status, str(native_exit), scope, scenario],
                          capture_output=True, text=True, timeout=15,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


@pytest.mark.parametrize("exit_code", [0, 1])
def test_capsule_qualification_starts_only_native_and_never_claims_full_completion(helper, tmp_path, exit_code):
    result = _toy_supervisor(helper, tmp_path, native_exit=exit_code, scope="capsules")
    state = json.loads((tmp_path / "prep/status.json").read_text())
    assert result.returncode == (0 if exit_code == 0 else 1), result.stderr
    assert set(state["job_exit_codes"]) == {"native"}
    assert state["models"] == {}
    assert state["validation_scope"] == "capsules"
    assert state["capsule_complete"] is (exit_code == 0)
    assert state["phase1_complete"] is state["external_full_models_complete"] is False
    assert state["overall_release_qualified"] is state["authoring_converged"] is False
    assert state["cleanup"]["complete"] is True
    assert not any((tmp_path / f"{name}.started").exists() for name in ("one", "two", "three"))


def test_capsule_formal_completion_reopens_technical_receipt_and_exact_source(helper, tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import capsule_certification as C

    path = tmp_path / "submission_qualification.json"
    record = {"schema": "merlin.phase1.submission_qualification.v1", "qualification_only": True,
              "validation_scope": "capsules", "authoring_converged": False,
              "selected_source_tree_sha256": "source", "graded_source_sha256": "source",
              "official_grade": {"complete": True, "validation_scope": "capsules",
                                 "capsule_certification": {"path": "technical", "sha256": "exact"}}}
    path.write_text(json.dumps(record))
    evidence = {"submission_sha256": "source", "capsule_complete": True,
                "receipt": {"path": "technical", "sha256": "exact"}}
    monkeypatch.setattr(C, "verify", lambda *a, **kw: evidence)
    F = SimpleNamespace(read=lambda p: json.loads(p.read_text()))
    native = {"process": SimpleNamespace(poll=lambda: 0)}
    package = {"hash_tree_sha256": "source"}
    args = dict(validation_scope="capsules", descriptor=tmp_path / "descriptor")
    assert helper.formal_complete(F, native, tmp_path, package, **args)
    assert not helper.formal_complete(F, native, tmp_path, package)  # No full-scope promotion.
    for changed in ({"submission_sha256": "other"}, {"capsule_complete": False},
                    {"receipt": {"path": "technical", "sha256": "changed"}}):
        monkeypatch.setattr(C, "verify", lambda *a, **kw: {**evidence, **changed})
        assert not helper.formal_complete(F, native, tmp_path, package, **args)

    def missing(*a, **kw):
        raise ValueError("actual selected capsule evidence is incomplete")

    monkeypatch.setattr(C, "verify", missing)
    assert not helper.formal_complete(F, native, tmp_path, package, **args)


def test_capsule_qualification_deadline_reaps_its_only_native_tree(helper, tmp_path):
    result = _toy_supervisor(helper, tmp_path, scope="capsules", hang=True)
    state = json.loads((tmp_path / "prep/status.json").read_text())
    assert result.returncode == 2, result.stderr
    assert set(state["job_exit_codes"]) == {"native"}
    assert state["capsule_complete"] is state["phase1_complete"] is False
    assert state["cleanup"]["complete"] is True
    assert "DeadlineReached" in state["termination_reason"]
    for path in tmp_path.glob("*.descendant"):
        pid, ticks = map(int, path.read_text().split())
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != ticks


def test_capsule_qualification_refuses_more_than_one_hour_before_owning_processes(helper, tmp_path):
    with pytest.raises(ValueError, match="at most one hour"):
        helper.run(SimpleNamespace(document={"validation_scope": "capsules"}), tmp_path / "prep",
                   tmp_path / "candidate", tmp_path / "proof", "sha", time.time() + 3602)
    assert not (tmp_path / "prep").exists()


@pytest.mark.parametrize("scenario,exit_code", [("independent", 7), ("invalid_report", 0)])
def test_failed_model_preserves_sibling_results_then_closes_native_waiter(helper, tmp_path, scenario, exit_code):
    result = _toy_supervisor(helper, tmp_path, scenario=scenario)
    state = json.loads((tmp_path / "prep/status.json").read_text())
    assert result.returncode == 1, result.stderr
    assert state["status"] == "finished"
    assert state["failed_models"] == ["one"]
    assert state["models"]["one"]["exit_code"] == exit_code
    assert state["models"]["one"]["status"] == "failed"
    for name in ("two", "three"):
        assert state["models"][name]["status"] == "passed"
        assert state["models"][name]["full_chain_complete"] is True
        assert (tmp_path / f"{name}.finished").is_file()
    assert "all model workers terminal" in state["native_owner_stop_reason"]
    assert state["job_exit_codes"]["native"] < 0
    assert state["phase1_complete"] is state["external_full_models_complete"] is False
    assert state["all_outputs_finalization"]["status"] == "incomplete"
    assert state["cleanup"]["complete"] is True


@pytest.mark.parametrize("scenario,reason", [("integrity", "frozen inputs changed"), ("memory", "memory guard"),
                                            ("signal", "terminated by signal"),
                                            ("job_integrity", "qualification job changed"),
                                            ("operator", "signal 15")])
def test_global_integrity_and_memory_errors_still_abort_independent_workers(helper, tmp_path, scenario, reason):
    result = _toy_supervisor(helper, tmp_path, scenario=scenario)
    state = json.loads((tmp_path / "prep/status.json").read_text())
    assert result.returncode == 2, result.stderr
    assert reason in state["termination_reason"]
    assert not (tmp_path / "two.finished").exists()
    assert not (tmp_path / "three.finished").exists()
    assert state["cleanup"]["complete"] is True
    assert state["phase1_complete"] is False


def test_deadline_after_one_failure_still_reaps_remaining_escaped_workers(helper, tmp_path):
    result = _toy_supervisor(helper, tmp_path, scenario="independent", hang=True)
    state = json.loads((tmp_path / "prep/status.json").read_text())
    assert result.returncode == 2, result.stderr
    assert state["failed_models"] == ["one"]
    assert "DeadlineReached" in state["termination_reason"]
    assert state["cleanup"]["complete"] is True
    assert state["models"]["two"]["full_chain_complete"] is False
    assert state["phase1_complete"] is False
    for path in tmp_path.glob("*.descendant"):
        pid, ticks = map(int, path.read_text().split())
        stat = Path(f"/proc/{pid}/stat")
        assert not stat.exists() or int(stat.read_text().rsplit(")", 1)[1].split()[19]) != ticks


def test_partial_model_roster_cannot_publish_static_or_numerical_success(helper, tmp_path, monkeypatch):
    chain = importlib.import_module("selected_chain_supervision")
    root = tmp_path / "partitions"
    root.mkdir()
    for name in ("two", "three"):
        (root / name).mkdir()
        (root / name / "partition-ready.json").write_text("{}")
    context = {"root": tmp_path, "partition_root": root, "run_dir": tmp_path / "native"}
    calls = []
    monkeypatch.setattr(chain.P, "save", lambda *a: calls.append(a))
    names = ["one", "two", "three"]
    chain.publish_static(context, names)
    state = {"models": {name: {"full_chain_complete": name != "one"} for name in names}}
    chain.publish_numerical(context, None, children={}, names=names, state=state, identity={}, package={},
                            output_context={}, output=SimpleNamespace(finalize=lambda *a: calls.append(a)))
    assert not calls
    assert not (tmp_path / "static-index.json").exists()
    assert not (tmp_path / "full-chain-qualification.json").exists()


def test_ordinary_native_failure_does_not_cancel_independent_model_workers(helper, tmp_path):
    result = _toy_supervisor(helper, tmp_path, scenario="native_failure", native_exit=1)
    state = json.loads((tmp_path / "prep/status.json").read_text())
    assert result.returncode == 1, result.stderr
    assert state["status"] == "finished"
    assert state["job_exit_codes"]["native"] == 1
    assert state["failed_jobs"] == ["native"]
    assert state["failed_models"] == []
    assert all(row["full_chain_complete"] for row in state["models"].values())
    assert all((tmp_path / f"{name}.finished").exists() for name in ("one", "two", "three"))
    assert state["native_formal_complete"] is False
    assert state["phase1_complete"] is False
    assert state["cleanup"]["complete"] is True
