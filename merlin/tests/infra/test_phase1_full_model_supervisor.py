"""Exact operator readiness and real bounded child trees, without author/simulator jobs."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common import oot_repo as O
from merlin.common.digest import sha256_file
from merlin.common.paths import python_import_roots, repo_root


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root() / "examples/gemmini/phase1/launch"))
    return importlib.import_module("full_model_verification"), importlib.import_module("launch_supervisor")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def world(tmp_path, modules):
    F, supervisor = modules
    repo, prep, bundle = tmp_path / "repo", tmp_path / "private-prep", tmp_path / "private-bundle"
    repo.mkdir()
    prep.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "source.py").write_text("# original source\n")
    subprocess.run(["git", "add", "source.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    config = {
        "schema": "merlin.model_check.models.v1",
        "target": "fixture",
        "models": {
            name: {
                "workload": name,
                "chain": {"kind": "single"},
                "programs": [{"name": "body", "steps": count, "inputs": []}],
                "judged": {"kind": "logits", "outputs": [{"program": "body", "output": 0}]},
            }
            for name, count in [("one", 1), ("two", 2), ("three", 3)]
        },
    }
    models = bundle / "configuration/models.yaml"
    models.parent.mkdir(parents=True)
    models.write_text(yaml.safe_dump(config))
    write(bundle / "MANIFEST.json", {"models": {name: {} for name in config["models"]}})
    (bundle / "SHA256SUMS").write_text("fixture input seal\n")
    write(bundle / "binding.json", {"status": "pass", "fixture": "actual selected binding bytes"})
    for name in config["models"]:
        path = bundle / name / "visible/v00/int8.npz"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture reference bytes; no simulator is invoked")
    private_spec = prep / "private-models.yaml"
    private_spec.write_text(
        yaml.safe_dump(
            {"numerics_bundle": {"path": str(bundle)}, "models": [{"id": name} for name in config["models"]]}
        )
    )
    runtime = tmp_path / "runtime"
    runtime.write_text("independently selected runtime bytes\n")
    run = repo / "out/runs/fixture/phase1/fresh"
    author = repo / "out/build/agent-workspaces/phase1/fixture/merlin_assisted/fresh/workspace/submission"
    author.mkdir(parents=True)
    (author / "manifest.yaml").write_text("entrypoints: {}\n")
    (author / "backend.py").write_text("# generated compiler fixture\n")
    candidate = run / "_qa_work/cand_00/submission"
    candidate.parent.mkdir(parents=True)
    shutil.copytree(author, candidate)
    O.init(run / "oot")
    record = O.commit_candidate(run / "oot", candidate, label="post-author", when=1, run_id="fresh")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 3:
        pytest.skip("supervisor allocation fixture requires three available CPUs")
    policy = {
        "comparison": "quantized-reference",
        "atol": 2e-6,
        "rtol": 1e-5,
        "split": "visible",
        "input": "visible/v00",
        "readiness_receipt": str(prep / "readiness.json"),
        "readiness_sha256": "pending",
    }
    document = {
        "inputs": {"repo": str(repo), "private_full_model_spec": str(private_spec)},
        "runtime_pins": {"runtime": {"path": str(runtime), "sha256": sha256_file(runtime)}},
        "full_model_verification": policy,
        "launcher": {"python": sys.executable},
        "identity": {"target": "fixture", "arm": "merlin_assisted"},
        "budgets": {"supervisor_backstop_s": 120},
        "cpu": {"author": {"cpus": [cpus[1]]}, "compute_pool": {"cpus": [cpus[0]]}},
    }
    launch = SimpleNamespace(
        document=document,
        environment={"MERLIN_MODEL_CHECK_BUNDLE": str(bundle)},
        launch_environment={},
        variables={
            "SOURCE_REVISION": commit,
            "RUN": str(run),
            "REFERENCE_CPUS": str(cpus[2]),
            "SUPERVISOR_FINAL_RESERVE_S": "1",
            "SUPERVISOR_PSS_MAX_BYTES": "1000000",
            "SUPERVISOR_HOST_HEADROOM_BYTES": "1",
        },
        path=lambda value: Path(value),
        expand=lambda value: value,
        manager_cpus=[cpus[0]],
    )
    return SimpleNamespace(
        F=F,
        supervisor=supervisor,
        repo=repo,
        prep=prep,
        bundle=bundle,
        run=run,
        author=author,
        candidate=candidate,
        record=record,
        launch=launch,
        policy=policy,
    )


@pytest.fixture
def allocation_prepare(world, monkeypatch):
    """Exercise real launch allocation checks with unrelated input gates substituted."""
    launcher = importlib.import_module("launch_phase1")
    w = world
    # The ordinary repository ignores generated output. This tiny git fixture
    # must do the same before prepare's real clean-source check.
    (w.repo / ".git/info/exclude").write_text("out/\n")
    w.launch.document["origin"] = {"initial_submission": "empty", "seed_submission": None, "resume": False}
    w.launch.document["launcher"].update(environment_file=str(w.prep / "environment.sh"),
                                         plan=str(w.prep / "plan.json"))
    manifest = w.bundle / "MANIFEST.json"
    contents = json.loads(manifest.read_text())
    contents["schema"] = "merlin.model_check.bundle.v1"
    manifest.write_text(json.dumps(contents))
    w.launch.document["runtime_pins"].update({
        "private_bundle_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "private_bundle_sums": {"path": str(w.bundle / "SHA256SUMS"), "sha256": sha256_file(w.bundle / "SHA256SUMS")},
    })
    monkeypatch.setattr(launcher.L, "render", lambda *a, **kw: None)
    monkeypatch.setattr(launcher.L, "launch_plan", lambda *a, **kw: {})
    monkeypatch.setattr(launcher.L, "verify", lambda *a, **kw: SimpleNamespace(drifts=[]))
    monkeypatch.setattr(launcher.L, "report_record", lambda *a, **kw: {"config": "allocation fixture"})
    monkeypatch.setattr(
        launcher.S, "verify", lambda *a, **kw: {"status": "pass", "manifest_sha256": sha256_file(manifest)}
    )
    monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda _: set(range(18)))
    monkeypatch.setattr(launcher, "source_preparation", lambda *_: {
        "source_preflight_passed": True, "candidate_qualified": False,
    })
    readiness_calls = []
    monkeypatch.setattr(launcher, "operator_module", lambda _: SimpleNamespace(
        validate_readiness=lambda launch: readiness_calls.append(launch),
    ))
    return launcher, w, readiness_calls


@pytest.mark.parametrize("references", ["", " , , ", None, "13,14,15"])
def test_prepare_accepts_all18_with_finished_or_reserved_reference_cpus(allocation_prepare, references):
    launcher, w, readiness_calls = allocation_prepare
    if references is None:
        w.launch.variables.pop("REFERENCE_CPUS")
    else:
        w.launch.variables["REFERENCE_CPUS"] = references
    reserved = bool(references and references.strip(" ,"))
    w.launch.document["cpu"]["compute_pool"]["cpus"] = list(range(13 if reserved else 16))
    w.launch.document["cpu"]["author"]["cpus"] = [16, 17]
    result = launcher.prepare(w.launch, w.prep)
    assert result["assigned_cpus"] == list(range(18))
    assert w.supervisor.assigned_cpus(w.launch) == set(range(18))
    assert readiness_calls == [w.launch] and result["phase1_complete"] is False


@pytest.mark.parametrize("damage", ["author_overlap", "reference_overlap", "missing_cpu", "unavailable_cpu"])
def test_prepare_refuses_overlap_or_missing_assigned_cpu(allocation_prepare, monkeypatch, damage):
    launcher, w, readiness_calls = allocation_prepare
    w.launch.variables["REFERENCE_CPUS"] = ""
    w.launch.document["cpu"]["compute_pool"]["cpus"] = list(range(16))
    w.launch.document["cpu"]["author"]["cpus"] = [16, 17]
    if damage == "author_overlap":
        w.launch.document["cpu"]["compute_pool"]["cpus"].append(16)
    elif damage == "reference_overlap":
        w.launch.variables["REFERENCE_CPUS"] = "0"
    elif damage == "missing_cpu":
        w.launch.document["cpu"]["compute_pool"]["cpus"].remove(15)
    else:
        monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda _: set(range(17)))
    with pytest.raises(ValueError, match="overlap|18-CPU allocation is unavailable"):
        launcher.prepare(w.launch, w.prep)
    assert not readiness_calls


def complete_report(w, name, package):
    model = w.F.selection(w.launch)["models"].model(name)
    count = sum(program.steps for program in model.programs)
    identity = w.F.selected_identity(w.launch)
    reference = w.bundle / name / "visible/v00/int8.npz"
    outputs = [{"program": "body", "step": step, "output": 0, "dtype": "<f4", "shape": [1]} for step in range(count)]
    return {
        "command": "check",
        "status": "pass",
        "model": name,
        "split": "visible",
        "agent_view": False,
        "failures": [],
        "submission": {"tree_sha256": package["model_check_tree_sha256"]},
        "tool": {"commit": w.launch.variables["SOURCE_REVISION"]},
        "binding": {
            "status": "pass",
            "receipt": "binding.json",
            "receipt_sha256": sha256_file(w.bundle / "binding.json"),
        },
        "comparison": {
            "selected": "quantized-reference",
            "float_policy": {"atol": 2e-6, "rtol": 1e-5, "equal_nan": False, "finite_required": True},
        },
        "gates": {
            "build": {"status": "pass"},
            "kernel_exact": {"status": "pass", "partial": False},
            "quantized_reference": {
                "status": "pass",
                "partial": False,
                "inputs_completed": 1,
                "inputs_expected": 1,
                "steps_completed": count,
                "steps_expected": count,
                "steps_declared": count,
                "outputs_expected": count,
                "outputs_compared": count,
                "float_policy": {"atol": 2e-6, "rtol": 1e-5, "equal_nan": False, "finite_required": True},
            },
        },
        "model_metrics": {
            "source": "measured",
            "inputs": 1,
            "per_input": [
                {
                    "id": "visible/v00",
                    "split": "visible",
                    "pass": True,
                    "reference_consistent": None,
                    "reference_validation": {
                        "source": "sealed_int8",
                        "strict_arity": True,
                        "unchanged": True,
                        "path": str(reference),
                        "sha256": sha256_file(reference),
                        "outputs": outputs,
                        "manifest_sha256": identity["bundle_manifest_sha256"],
                        "sums_sha256": identity["bundle_sums_sha256"],
                        "models_sha256": identity["models_config_sha256"],
                    },
                    "quantized_reference": [{**row, "pass": True, "mismatched": 0} for row in outputs],
                }
            ],
        },
        "programs": [
            {
                "program": p.name,
                "runs": [
                    {"step": step, "input": "visible/v00", "engine": "spike", "status": "completed"}
                    for step in range(p.steps)
                ],
            }
            for p in model.programs
        ],
    }


def test_operator_reexports_the_same_core_full_chain_predicate(world):
    from merlin.model_check.full_chain_validation import report_complete

    w = world
    assert w.F.report_complete is report_complete
    package = w.F.package_identity(w.author)
    model = w.F.selection(w.launch)["models"].model("two")
    report = complete_report(w, "two", package)
    assert report_complete(report, model, w.policy, package)
    report["gates"]["kernel_exact"]["partial"] = True
    assert not report_complete(report, model, w.policy, package)
    report["gates"]["kernel_exact"]["partial"] = False
    report["programs"][0]["runs"][-1]["engine"] = "host"
    assert not report_complete(report, model, w.policy, package)


def readiness(w):
    package = w.F.package_identity(w.author)
    identity = w.F.selected_identity(w.launch)
    checks = {}
    for name in w.F.selection(w.launch)["names"]:
        report = w.prep / f"retained-{name}.json"
        pin = w.prep / f"retained-{name}-pin.json"
        work = w.repo / f"out/build/retained/{name}"
        write(report, complete_report(w, name, package))
        command = w.F.model_command(w.launch, name, w.author, report, work, 100)
        write(
            pin,
            {
                "selected_identity": identity,
                "submission": package,
                "argv": command,
                "work": str(work),
                "timeout_s": 100,
            },
        )
        checks[name] = {
            "report": {"path": str(report), "sha256": sha256_file(report)},
            "pin": {"path": str(pin), "sha256": sha256_file(pin)},
        }
    receipt = {
        "schema": "merlin.phase1.full_model_readiness.v1",
        "selected_identity": identity,
        "retained_submission": package,
        "checks": checks,
    }
    path = Path(w.policy["readiness_receipt"])
    write(path, receipt)
    w.policy["readiness_sha256"] = sha256_file(path)
    return receipt


def test_real_hash_bound_readiness_and_distinct_package_algorithms(world):
    w = world
    receipt = readiness(w)
    assert (
        receipt["retained_submission"]["hash_tree_sha256"] != receipt["retained_submission"]["model_check_tree_sha256"]
    )
    assert w.F.validate_readiness(w.launch) == receipt
    del w.launch.document["full_model_verification"]
    assert w.F.validate_readiness(w.launch) is None


@pytest.mark.parametrize("damage", ["receipt", "runtime", "compiler", "report", "pin", "source"])
def test_readiness_refuses_changed_actual_evidence(world, damage):
    w = world
    receipt = readiness(w)
    if damage == "receipt":
        path = Path(w.policy["readiness_receipt"])
    elif damage == "runtime":
        path = Path(w.launch.document["runtime_pins"]["runtime"]["path"])
    elif damage == "compiler":
        path = w.author / "backend.py"
    elif damage in ("report", "pin"):
        path = Path(receipt["checks"]["one"][damage]["path"])
    else:
        (w.repo / "source.py").write_text("# changed harness\n")
        subprocess.run(["git", "add", "source.py"], cwd=w.repo, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "changed"],
            cwd=w.repo,
            check=True,
        )
        w.launch.variables["SOURCE_REVISION"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=w.repo, text=True
        ).strip()
        path = None
    if path:
        path.write_text(path.read_text() + "\nchanged")
    with pytest.raises(ValueError):
        w.F.validate_readiness(w.launch)


@pytest.mark.parametrize(
    "damage",
    [
        "status_only",
        "missing_steps",
        "partial",
        "wrong_input",
        "host_twin",
        "failure",
        "kernel_skip",
        "changed_reference",
        "unproved_arity",
        "pretend_recomputation",
    ],
)
def test_cli_pass_alone_never_qualifies_a_full_chain(world, damage):
    w = world
    package = w.F.package_identity(w.author)
    report = complete_report(w, "two", package)
    if damage == "status_only":
        report["gates"] = {}
    if damage == "missing_steps":
        report["programs"][0]["runs"].pop()
    if damage == "partial":
        report["gates"]["quantized_reference"]["partial"] = True
    if damage == "wrong_input":
        report["programs"][0]["runs"][0]["input"] = "visible/v01"
    if damage == "host_twin":
        report["comparison"]["selected"] = "host-twin"
    if damage == "failure":
        report["failures"] = [{"kind": "infrastructure"}]
    if damage == "kernel_skip":
        report["gates"].pop("kernel_exact")
    if damage == "changed_reference":
        report["model_metrics"]["per_input"][0]["reference_validation"]["unchanged"] = False
    if damage == "unproved_arity":
        report["model_metrics"]["per_input"][0]["reference_validation"]["strict_arity"] = False
    if damage == "pretend_recomputation":
        report["model_metrics"]["per_input"][0]["reference_consistent"] = True
    assert not w.F.report_complete(report, w.F.selection(w.launch)["models"].model("two"), w.policy, package)


def test_native_commit_can_start_before_ledger_but_cannot_complete_before_publication(world):
    w = world
    snapshot = w.F.native_snapshot(w.run, w.author)
    assert snapshot["native_oot_commit"] == w.record.commit
    package = w.F.copy_snapshot(w.run, snapshot, w.prep / "frozen-new-compiler")
    pending = w.F.final_correspondence(w.run, snapshot, package)
    assert pending["native_ledger"] == pending["native_final_publication"] == "pending"
    write(w.run / "oot_commits.jsonl", {"oot": w.record.as_record()})
    shutil.copytree(w.author, w.run / "submission")
    final = w.F.final_correspondence(w.run, snapshot, package)
    assert final == {
        "native_oot_commit_verified": True,
        "native_ledger": "exact",
        "native_final_publication": "exact",
        "frozen_copy_unchanged": True,
    }
    (w.run / "submission/backend.py").write_text("# another compiler\n")
    assert w.F.final_correspondence(w.run, snapshot, package)["native_final_publication"] == "different_compiler"


@pytest.mark.parametrize("output_scope", ["legacy", "all_declared", "incomplete_all_declared"])
def test_models_start_during_native_regrade_and_supervisor_waits_for_all_children(world, monkeypatch, output_scope):
    w = world
    if output_scope != "legacy":
        all_output_plan(w, incomplete=output_scope == "incomplete_all_declared")
    transcript = w.run / "rounds/round_00.transcript.jsonl"
    finished = w.prep / "native-finished"
    code = """import json,pathlib,shutil,sys,time
p=json.loads(sys.argv[1]);t=pathlib.Path(p['transcript']);t.parent.mkdir(parents=True);t.write_text('{"type":"result"}\\n')
time.sleep(.6)
shutil.copytree(p['author'],p['published'])
pathlib.Path(p['ledger']).write_text(json.dumps({'oot':p['record']})+'\\n')
pathlib.Path(p['finished']).write_text(str(time.monotonic()))
"""
    w.launch.argv = [
        sys.executable,
        "-c",
        code,
        json.dumps(
            {
                "transcript": str(transcript),
                "author": str(w.author),
                "published": str(w.run / "submission"),
                "ledger": str(w.run / "oot_commits.jsonl"),
                "record": w.record.as_record(),
                "finished": str(finished),
            }
        ),
    ]

    def command(launch, name, package, report, work, timeout):
        if output_scope != "legacy":
            (
                work
                / w.F.package_identity(package)["model_check_tree_sha256"][:16]
                / name
                / "runs/original/visible/v00"
            ).mkdir(parents=True)
        value = complete_report(w, name, w.F.package_identity(package))
        worker = (
            "import json,pathlib,sys,time;p=pathlib.Path(sys.argv[1]);"
            "p.with_suffix('.started').write_text(str(time.monotonic()));time.sleep(.8);p.write_text(sys.argv[2])"
        )
        return [sys.executable, "-c", worker, str(report), json.dumps(value)]

    monkeypatch.setattr(w.F, "model_command", command)
    incomplete = output_scope == "incomplete_all_declared"
    assert w.supervisor.run(w.launch, w.prep, time.monotonic(), memory=lambda _: (0, 10000000), poll_s=0.02) == int(
        incomplete
    )
    state = json.loads((w.prep / "status.json").read_text())
    assert state["full_model_execution_complete"] is not incomplete
    assert not state["phase1_complete"]
    assert len(state["models"]) == 3
    assert all(
        float(path.read_text()) < float(finished.read_text()) for path in w.prep.glob("full-model-checks-*/*.started")
    )
    assert all(row["exit_code"] == 0 and row["judged_full_chain_complete"] for row in state["models"].values())
    assert all(row["full_chain_complete"] is not incomplete for row in state["models"].values())
    assert state["output_scope"] == ("judged_only" if output_scope == "legacy" else "all_declared")
    if output_scope != "legacy":
        assert len((w.prep / "offline-calls.jsonl").read_text().splitlines()) == 3
        assert len(list(w.prep.glob("full-model-checks-*/*-all-outputs.json"))) == 3


@pytest.mark.parametrize("guard", ["deadline", "memory"])
def test_same_campaign_guard_stops_controller_tree_without_orphans(world, guard):
    w = world
    w.launch.document.pop("full_model_verification")
    w.launch.document["budgets"]["supervisor_backstop_s"] = 2
    pidfile = w.prep / "grandchild.pid"
    code = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
    )
    w.launch.argv = [sys.executable, "-c", code, str(pidfile)]
    start = time.monotonic()

    def memory(_):
        return (2000000 if guard == "memory" and time.monotonic() - start > 0.3 else 0), 10000000

    assert w.supervisor.run(w.launch, w.prep, start, memory=memory, poll_s=0.02) == 2
    pid = int(pidfile.read_text())
    stat = Path("/proc") / str(pid) / "stat"
    assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    state = json.loads((w.prep / "status.json").read_text())
    assert state["termination_reason"] == ("campaign_deadline" if guard == "deadline" else "campaign_memory_guard")


def test_absent_full_model_policy_preserves_successful_legacy_controller_exit(world):
    w = world
    w.launch.document.pop("full_model_verification")
    w.launch.argv = [sys.executable, "-c", "pass"]
    assert w.supervisor.run(w.launch, w.prep, time.monotonic(), memory=lambda _: (0, 10000000), poll_s=0.02) == 0


def test_readiness_accepts_exact_observed_instrumented_transport(world):
    w = world
    receipt = readiness(w)
    for row in receipt["checks"].values():
        pin_path = Path(row["pin"]["path"])
        pin = json.loads(pin_path.read_text())
        wrapper = w.prep / "retained-observer.py"
        wrapper.write_text("# reviewed observer fixture; not executed\n")
        pin["transport"] = {"path": str(wrapper), "sha256": sha256_file(wrapper)}
        pin["inner_argv"] = pin["argv"][4:]
        pin["argv"] = [sys.executable, "-u", str(wrapper), row["report"]["path"], pin["work"], *pin["inner_argv"]]
        write(pin_path, pin)
        row["pin"]["sha256"] = sha256_file(pin_path)
    write(Path(w.policy["readiness_receipt"]), receipt)
    w.policy["readiness_sha256"] = sha256_file(Path(w.policy["readiness_receipt"]))
    assert w.F.validate_readiness(w.launch) == receipt


def test_operator_helpers_load_exact_sibling_sources_under_safe_python_startup(tmp_path):
    launcher = repo_root() / "examples/gemmini/phase1/launch/launch_phase1.py"
    program = """import pathlib,runpy,sys
path=pathlib.Path(sys.argv[1]);owner=runpy.run_path(str(path),run_name='operator_fixture')
for name in ('full_model_verification','full_model_outputs','launch_supervisor'):
    module=owner['operator_module'](name)
    assert pathlib.Path(module.__file__).parent == path.parent
"""
    env = dict(os.environ, PYTHONSAFEPATH="1", PYTHONPATH=os.pathsep.join(map(str, python_import_roots())))
    result = subprocess.run(
        [sys.executable, "-P", "-c", program, str(launcher)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def all_output_plan(w, *, incomplete=False):
    """Only the operator transport is synthetic; regular reports and guards remain real."""
    w.launch.run_id = "fresh"
    identity = w.F.selected_identity(w.launch)
    names = w.F.selection(w.launch)["names"]
    expected = {name: w.F.selection(w.launch)["models"].model(name).programs[0].steps for name in names}
    helper, requirement, expected_file = (
        w.prep / name for name in ("offline_helper.py", "requirement.txt", "identity.json")
    )
    requirement.write_text("Every declared output must be compared.\n")
    write(expected_file, identity)
    helper.write_text(
        "import json,hashlib\nfrom pathlib import Path\n"
        "FLOAT_POLICY={'atol':2e-6,'rtol':1e-5,'equal_nan':False,'finite_required':True}\n"
        f"EXPECTED={expected!r}\nINCOMPLETE={incomplete!r}\nCALLS={str(w.prep / 'offline-calls.jsonl')!r}\n"
        "def verify(**kw):\n"
        "    name=kw['report_path'].stem\n"
        "    with Path(CALLS).open('a') as stream: stream.write(name+'\\n')\n"
        "    package=json.loads(kw['job_pin_path'].read_text())['submission']\n"
        "    return {'schema':'merlin.operator.full_declared_output_audit.v1','status':'pass',"
        "'unchanged':True,'model':name,'input_id':'visible/v00','selected_identity':kw['expected_identity'],"
        "'artifacts':[{'path':str(kw['report_path']),'sha256':hashlib.sha256(kw['report_path'].read_bytes()).hexdigest()}],"
        "'submission':package,'steps_compared':EXPECTED[name],'steps_expected':EXPECTED[name],"
        "'outputs_compared':EXPECTED[name]*2-int(INCOMPLETE),'n_declared_outputs':EXPECTED[name]*2}\n"
    )
    def pin(path):
        return {"path": str(path), "sha256": sha256_file(path)}
    references = {}
    bundle_keys = ("bundle_manifest_sha256", "bundle_sums_sha256", "models_config_sha256")
    for name in names:
        path = w.prep / f"{name}-all-reference.json"
        write(
            path,
            {
                "schema": "merlin.operator.full_declared_output_reference.v1",
                "status": "complete",
                "model": name,
                "input_id": "visible/v00",
                "representation": "declared_storage",
                "judged_crossjoin": {"bit_exact": True},
                "selected_identity": {key: identity[key] for key in bundle_keys},
                "programs": [
                    {
                        "program": "body",
                        "steps_declared": expected[name],
                        "declared_outputs": [{"output": 0}, {"output": 1}],
                        "steps": [{"step": step} for step in range(expected[name])],
                    }
                ],
            },
        )
        references[name] = pin(path)
    plan = {
        "schema": "merlin.operator.all_output_plan.v1",
        "selected_identity": identity,
        "expected_identity": pin(expected_file),
        "helper": pin(helper),
        "references": references,
        "requirement": pin(requirement),
        "extension_runtime_key": "runtime",
        "comparison": {
            "float": {"atol": 2e-6, "rtol": 1e-5, "equal_nan": False, "finite_required": True},
            "integer_bool": "exact",
        },
    }
    path = w.prep / "all-output-plan.json"
    write(path, plan)
    w.policy.update(output_scope="all_declared", all_output_plan=pin(path))
    return path, plan


@pytest.mark.parametrize(
    "damage", ["plan_bytes", "helper", "reference", "requirement", "selection", "run", "roster", "comparison"]
)
def test_all_output_metadata_is_admitted_before_authoring(world, damage):
    w = world
    readiness(w)
    path, plan = all_output_plan(w)
    assert w.F.validate_readiness(w.launch)["schema"] == "merlin.phase1.full_model_readiness.v1"
    if damage in {"helper", "reference", "requirement"}:
        pin = plan["references"]["one"] if damage == "reference" else plan[damage]
        Path(pin["path"]).write_text("changed byte identity\n")
    elif damage == "selection":
        plan["selected_identity"]["source_revision"] = "f" * 40
    elif damage == "run":
        plan["run_id"] = "some-other-run"
    elif damage == "roster":
        del plan["references"]["one"]
    elif damage == "comparison":
        plan["comparison"]["integer_bool"] = "close"
    if damage == "plan_bytes":
        path.write_text("changed plan\n")
    elif damage not in {"helper", "reference", "requirement"}:
        write(path, plan)
        w.policy["all_output_plan"]["sha256"] = sha256_file(path)
    with pytest.raises(ValueError):
        w.F.validate_readiness(w.launch)


@pytest.mark.parametrize("changed", ["report", "job", "audit"])
def test_all_output_audit_is_once_only_and_changed_completed_report_loses_credit(world, changed):
    w = world
    all_output_plan(w)
    operator = w.F.output_verification()
    selected = w.F.selection(w.launch)
    context = operator.validate(w.launch, identity=w.F.selected_identity(w.launch), models=selected["models"])
    package = w.F.package_identity(w.author)
    report, job, work = w.prep / "one.json", w.prep / "one-pin.json", w.repo / "work"
    write(report, {"final": True})
    write(job, {"work": str(work), "submission": package})
    (work / package["model_check_tree_sha256"][:16] / "one/runs/original/visible/v00").mkdir(parents=True)
    child = {"report": report, "pin": job, "input_id": "visible/v00"}
    def call():
        return operator.audit(context, name="one", child=child, package=package, models_path=selected["models"].path)
    assert call()["status"] == "pass"
    assert call()["status"] == "pass"
    assert (w.prep / "offline-calls.jsonl").read_text().splitlines() == ["one"]
    target = {"report": report, "job": job, "audit": Path(child["all_outputs"]["path"])}[changed]
    target.write_text('{"final":"changed"}')
    assert call()["status"] == "error"
    assert (w.prep / "offline-calls.jsonl").read_text().splitlines() == ["one"]


def test_all_output_finalization_reopens_retained_artifacts_without_regrading(world):
    w = world
    all_output_plan(w)
    operator = w.F.output_verification()
    selected = w.F.selection(w.launch)
    context = operator.validate(w.launch, identity=w.F.selected_identity(w.launch), models=selected["models"])
    original = w.prep / "original-readback.bin"
    original.write_bytes(b"original successful output bytes")
    children = {}
    for name in selected["names"]:
        path = w.prep / f"{name}-all-outputs.json"
        write(path, {"status": "pass", "artifacts": [{"path": str(original), "sha256": sha256_file(original)}]})
        children[name] = {"report": w.prep / f"{name}.json", "all_outputs": {
            "status": "pass", "path": str(path), "sha256": sha256_file(path)}}
    assert operator.finalize(context, children)["status"] == "pass"
    original.write_bytes(b"changed output bytes")
    assert operator.finalize(context, children)["status"] == "error"
    assert not (w.prep / "offline-calls.jsonl").exists()


def test_in_process_supervisor_uses_selected_runtime_environment_and_restores_ambient(world, monkeypatch):
    launcher = importlib.import_module("launch_phase1")
    world.launch.environment["MERLIN_SELECTED_OPERATOR_RUNTIME"] = "selected"
    monkeypatch.setenv("MERLIN_SELECTED_OPERATOR_RUNTIME", "ambient")

    def supervised(*_args, **_kwargs):
        assert os.environ["MERLIN_SELECTED_OPERATOR_RUNTIME"] == "selected"
        return 7

    monkeypatch.setattr(launcher, "operator_module", lambda _name: SimpleNamespace(run=supervised))
    assert launcher.run(world.launch, world.prep, time.monotonic()) == 7
    assert os.environ["MERLIN_SELECTED_OPERATOR_RUNTIME"] == "ambient"


@pytest.fixture
def infrastructure(world, monkeypatch):
    """Real subprocess artifact stand-ins exercise validation; they establish no target readiness."""
    import io

    import numpy as np

    from merlin.common import invocation_record as IR
    from merlin.common import owned_process as OP
    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.llvmlower.compilation_recipe import CompilationRecipe
    from merlin.model_check import full_build as FB
    from merlin.model_check import replay as RP
    from merlin.runtime.backends import spike_model

    w = world

    def pin(path):
        return {"path": str(path.resolve()), "sha256": sha256_file(path)}

    producer = w.prep / "collector.py"
    producer.write_text("# explicitly selected unit-test collector stand-in\n")
    w.launch.document["runtime_pins"]["operator_infrastructure_witness"] = pin(producer)
    roots, rows = {}, []
    for name in w.F.selection(w.launch)["names"]:
        capture_root = w.prep / "captures" / name
        capture = capture_root / "run/capture"
        capture.mkdir(parents=True)
        (capture / "model.mlir").write_text("module {}\n")
        selection = capture_root / "selection.json"
        write(selection, {"fixture": "independently selected capture"})
        attestation = capture_root / "attestation.json"
        write(attestation, {"selection": pin(selection), "capture": {
            "capture_path": str(capture), "capture_tree_sha256": "attested tree identity",
            "model_sha256": sha256_file(capture / "model.mlir"),
        }})
        rows.append({"id": name, "capture": str(capture), "capture_selection": str(selection),
                     "capture_selection_sha256": sha256_file(selection),
                     "capture_execution_attestation": str(attestation),
                     "capture_execution_attestation_sha256": sha256_file(attestation)})
        roots[name] = str(capture_root)
        write(w.bundle / name / "programs/body/catalog.json", {"stage": str(capture)})
        model = w.F.selection(w.launch)["models"].model(name)
        np.savez(w.bundle / name / "visible/v00/int8.npz",
                 **{f"out{i}": np.array([i], np.float32) for i in range(model.programs[0].steps)})
    write(w.prep / "private-models.yaml", {"models": rows})
    manifest = read_manifest = json.loads((w.bundle / "MANIFEST.json").read_text())
    manifest["selected_capture_roots"] = roots
    sums = w.bundle / "SHA256SUMS"
    members = sorted(path for path in w.bundle.rglob("*")
                     if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS"})
    sums.write_text("".join(f"{sha256_file(path)}  {path.relative_to(w.bundle)}\n" for path in members))
    read_manifest["seal"] = {"sums": "SHA256SUMS", "sums_sha256": sha256_file(sums)}
    write(w.bundle / "MANIFEST.json", manifest)
    compiler = w.prep / "compiler.py"
    compiler.write_text("import pathlib,sys\n"
                        "pathlib.Path(sys.argv[-1]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes())\n")
    stage = w.prep / "stage"
    stage.mkdir()
    (stage / "model.mlir").write_text("module {}\n")

    def compile_image(folder, input_path, producer_module):
        folder.mkdir()
        elf = folder / "model.elf"
        recipe = CompilationRecipe(folder, producer=Path(producer_module.__file__))
        recipe.run([sys.executable, compiler, input_path, "-o", elf],
                   runner=lambda argv: subprocess.run(argv, capture_output=True, check=True),
                   inputs=[compiler, input_path], output=elf)
        recipe.completed(elf)
        return elf, recipe.path

    original_input = w.prep / "original.input"
    original_input.write_bytes(b"original image fixture\n")
    original, recipe = compile_image(w.prep / "original", original_input, spike_model)
    build_path = w.prep / "build.json"
    write(build_path, {"status": "linked", "elf": str(original), "elf_sha256": sha256_file(original),
                       "build_hash": "fixture", "compilation_recipe": pin(recipe)})
    engine = w.prep / "engine.py"
    engine.write_text("""import io,numpy as np,sys
x=np.load(sys.argv[1],allow_pickle=False)['x'];out=io.BytesIO()
np.savez(out,out0=(x+1).astype('int32'),out1=(x/2).astype('float32'));sys.stdout.buffer.write(out.getvalue())
""")
    monkeypatch.setattr(RP, "spike_argv", lambda _target, elf, **_kw: ([sys.executable, str(engine), str(elf)], {}))

    def read_outputs(console, _stage, _hash):
        with np.load(io.BytesIO(console), allow_pickle=False) as data:
            return [np.asarray(data[f"out{i}"]) for i in range(len(data.files))], {}

    monkeypatch.setattr(FB, "read", read_outputs)
    runs = []
    for step in range(2):
        inputs, expected = w.prep / f"inputs-{step}.npz", w.prep / f"expected-{step}.npz"
        x = np.array([step, step + 2], np.int32)
        np.savez(inputs, x=x)
        np.savez(expected, out0=x + 1, out1=(x / 2).astype(np.float32))
        elf, recipe = compile_image(w.prep / f"step-{step}", inputs, FB)
        invocation = IR.Invocation(w.prep, stage="phase1.infrastructure.target",
                                   argv=[sys.executable, str(engine), str(elf)],
                                   inputs=[elf, inputs], dependencies=[engine])
        result = subprocess.run(invocation.document["argv"], capture_output=True, check=True)
        invocation.complete(result)
        runs.append({"inputs": pin(inputs), "expected": pin(expected), "elf": pin(elf),
                     "compilation_recipe": pin(recipe), "execution": pin(invocation.path)})
    target = w.prep / "target-witness.json"
    write(target, {"schema": "merlin.phase1.infrastructure_target_witness.v1", "producer": pin(producer),
                   "source_revision": w.launch.variables["SOURCE_REVISION"], "target": "fixture",
                   "stage": {"path": str(stage), "sha256": strict_tree_sha256(stage)["sha256"]},
                   "build_record": pin(build_path), "dram_base": 0, "dram_bytes": 1024, "runs": runs})
    child = w.prep / "child.py"
    child.write_text("""import json,os,pathlib,subprocess,sys,time
def report():
 print(json.dumps({'pid':os.getpid(),'start_ticks':int(pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}),flush=True)
report()
if len(sys.argv)==1: subprocess.Popen([sys.executable,__file__,'nested'],start_new_session=True)
time.sleep(30)
""")
    invocation = IR.Invocation(w.prep, stage="phase1.infrastructure.timeout", argv=[sys.executable, str(child)],
                               inputs=[child])
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        OP.run(invocation.document["argv"], timeout=0.5)
    stdout = invocation.directory / "stdout.bin"
    stdout.write_bytes(caught.value.output)
    invocation.document["stdout"] = pin(stdout)
    invocation.failed(caught.value)
    cleanup = w.prep / "cleanup-witness.json"
    write(cleanup, {"schema": "merlin.phase1.infrastructure_cleanup_witness.v1", "producer": pin(producer),
                    "source_revision": w.launch.variables["SOURCE_REVISION"], "invocation": pin(invocation.path),
                    "stdout": pin(stdout), "children": [json.loads(line) for line in stdout.read_text().splitlines()],
                    "owned_process_cleanup": caught.value.owned_process_cleanup})
    retained = w.F.package_identity(w.author)
    diagnostics = {}
    for name in w.F.selection(w.launch)["names"]:
        report = complete_report(w, name, retained)
        report.update(status="fail", failures=[{"kind": "gate", "gate": "quantized_reference"}])
        path, pin_path = w.prep / f"historic-{name}.json", w.prep / f"historic-{name}-pin.json"
        if name == "one":
            report = {"schema": "merlin.readiness.partial.v1", "reason": "interrupted_by_signal_15",
                      "complete": False, "model_check_report_so_far": report}
        write(path, report)
        write(pin_path, {"submission": retained, "selected_identity": {"source_revision": "historical identity"}})
        diagnostics[name] = {"report": path, "pin": pin_path}
    receipt = w.F.build_infrastructure_receipt(w.launch, w.author, diagnostics, target, cleanup)
    write(Path(w.policy["readiness_receipt"]), receipt)
    w.policy["readiness_sha256"] = sha256_file(Path(w.policy["readiness_receipt"]))
    return SimpleNamespace(w=w, receipt=receipt, target=target, cleanup=cleanup, pin=pin)


def test_infrastructure_v2_admits_fresh_author_without_qualifying_failed_retained_models(infrastructure):
    t = infrastructure
    assert t.w.F.validate_readiness(t.w.launch) == t.receipt
    for name, records in t.receipt["retained_diagnostics"].items():
        report = json.loads(Path(records["report"]["path"]).read_text())
        report = report.get("model_check_report_so_far", report)
        assert report["status"] == "fail"
        assert not t.w.F.report_complete(report, t.w.F.selection(t.w.launch)["models"].model(name),
                                         t.w.policy, t.receipt["retained_submission"])


@pytest.mark.parametrize("damage", ["source", "collector", "console", "expected", "recipe", "input",
                                    "boolean_only", "capture", "catalog", "reference", "cleanup",
                                    "comparison", "repeated_input", "execution_scope", "alive_child",
                                    "attested_graph", "historical_attribution"])
def test_infrastructure_v2_rejects_unbound_or_incomplete_actual_evidence(infrastructure, damage):
    t = infrastructure
    target = json.loads(t.target.read_text())
    if damage == "source":
        target["source_revision"] = "0" * 40
    elif damage == "collector":
        target["producer"]["sha256"] = "0" * 64
    elif damage == "boolean_only":
        target = {"schema": target["schema"], "status": "pass"}
    elif damage in {"console", "expected", "recipe", "input"}:
        run = target["runs"][0]
        selected = run[{"expected": "expected", "recipe": "compilation_recipe", "input": "inputs"}
                       .get(damage, "execution")]
        if damage == "console":
            selected = json.loads(Path(selected["path"]).read_text())["stdout"]
        Path(selected["path"]).write_bytes(b"changed actual evidence")
    elif damage in {"capture", "catalog"}:
        path = (t.w.prep / "private-models.yaml" if damage == "capture"
                else t.w.bundle / "one/programs/body/catalog.json")
        if damage == "capture":
            doc = yaml.safe_load(path.read_text())
            doc["models"][0]["capture"] = str(t.w.prep / "older-128-token-capture")
            path.write_text(yaml.safe_dump(doc))
            t.receipt["selected_identity"] = t.w.F.selected_identity(t.w.launch)
        else:
            write(path, {"stage": str(t.w.prep / "different-stage")})
    elif damage == "reference":
        (t.w.bundle / "one/visible/v00/int8.npz").write_bytes(b"changed reference")
    elif damage == "comparison":
        import numpy as np

        path = Path(target["runs"][0]["expected"]["path"])
        with np.load(path) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        arrays["out0"] += 1
        np.savez(path, **arrays)
        target["runs"][0]["expected"] = t.pin(path)
    elif damage == "repeated_input":
        target["runs"][1] = target["runs"][0]
    elif damage == "execution_scope":
        record = target["runs"][0]["execution"]
        path = Path(record["path"])
        doc = json.loads(path.read_text())
        doc["argv"] = ["wrong simulator"]
        write(path, doc)
        target["runs"][0]["execution"] = t.pin(path)
    elif damage == "attested_graph":
        (t.w.prep / "captures/one/run/capture/model.mlir").write_text("different captured model\n")
    elif damage == "historical_attribution":
        path = Path(t.receipt["retained_diagnostics"]["one"]["report"]["path"])
        doc = json.loads(path.read_text())
        doc["model_check_report_so_far"]["model"] = "another model"
        write(path, doc)
        t.receipt["retained_diagnostics"]["one"]["report"] = t.pin(path)
    else:
        cleanup = json.loads(t.cleanup.read_text())
        if damage == "alive_child":
            cleanup["children"][0] = {"pid": os.getpid(), "start_ticks": int(
                Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
            stdout = Path(cleanup["stdout"]["path"])
            stdout.write_text("\n".join(json.dumps(item) for item in cleanup["children"]) + "\n")
            cleanup["stdout"] = t.pin(stdout)
            invocation = Path(cleanup["invocation"]["path"])
            doc = json.loads(invocation.read_text())
            doc["stdout"] = cleanup["stdout"]
            write(invocation, doc)
            cleanup["invocation"] = t.pin(invocation)
        else:
            cleanup["owned_process_cleanup"]["guardian_reaped"] = False
        write(t.cleanup, cleanup)
        t.receipt["cleanup_witness"] = t.pin(t.cleanup)
    write(t.target, target)
    t.receipt["target_witness"] = t.pin(t.target)
    write(Path(t.w.policy["readiness_receipt"]), t.receipt)
    t.w.policy["readiness_sha256"] = sha256_file(Path(t.w.policy["readiness_receipt"]))
    with pytest.raises((ValueError, OSError)):
        t.w.F.validate_readiness(t.w.launch)
