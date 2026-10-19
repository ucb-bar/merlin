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


def test_models_start_during_native_regrade_and_supervisor_waits_for_all_children(world, monkeypatch):
    w = world
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
        value = complete_report(w, name, w.F.package_identity(package))
        worker = (
            "import json,pathlib,sys,time;p=pathlib.Path(sys.argv[1]);"
            "p.with_suffix('.started').write_text(str(time.monotonic()));time.sleep(.8);p.write_text(sys.argv[2])"
        )
        return [sys.executable, "-c", worker, str(report), json.dumps(value)]

    monkeypatch.setattr(w.F, "model_command", command)
    assert w.supervisor.run(w.launch, w.prep, time.monotonic(), memory=lambda _: (0, 10000000), poll_s=0.02) == 0
    state = json.loads((w.prep / "status.json").read_text())
    assert state["full_model_execution_complete"] and not state["phase1_complete"]
    assert len(state["models"]) == 3
    assert all(
        float(path.read_text()) < float(finished.read_text()) for path in w.prep.glob("full-model-checks-*/*.started")
    )
    assert all(row["exit_code"] == 0 and row["full_chain_complete"] for row in state["models"].values())


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
for name in ('full_model_verification','launch_supervisor'):
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
