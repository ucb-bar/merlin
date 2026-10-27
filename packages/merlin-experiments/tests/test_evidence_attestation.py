"""Operator attestation binds real local identities without claiming proof."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin_experiments.phase0 import evidence_attestation as EA
from merlin_experiments.phase0 import spec_validation as SV

DRIVER = "build_tools/scripts/run_spec_validation.py"
MODULE = "packages/merlin-experiments/src/merlin_experiments/phase0/evidence_attestation.py"
TOUCHED = {DRIVER, MODULE,
           "packages/merlin-experiments/tests/test_evidence_attestation.py"}
STUB = '''import argparse
import json
from pathlib import Path

def stamp(args):
    return json.loads((args.repo.parent / "current-stamp.json").read_bytes())

def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ("repo", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--label", required=True)
    args, _ = p.parse_known_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    block = stamp(args)
    (args.output / args.label / "provenance.json").write_text(json.dumps(block))
    raise AssertionError("execution must never be reached")
'''


def git(repo, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo), "-c", "user.name=Attestation Test", "-c", "user.email=attestation@example.invalid",
         *args], stderr=subprocess.PIPE
    )


def write(repo, path, text):
    destination = repo / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return destination


def commit(repo):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "test state")
    return git(repo, "rev-parse", "HEAD").decode().strip()


def stamp(repo, revision):
    return {"merlin_commit": revision, "merlin_source_label": revision,
            "merlin": {"commit": revision, "source_label": revision, "dirty_files": 0},
            "merlin_source_kind": "git_worktree", "merlin_diff_sha256": EA.EMPTY_SHA256,
            "sources": [str(repo / "src/pkg/root.py")], "source_digest": "a" * 64,
            "software_spec_sha256": "b" * 64}


@pytest.fixture
def history(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo, ".gitignore", "__pycache__/\n")
    write(repo, EA.RUNNER, STUB)
    write(repo, DRIVER, (repo_root() / DRIVER).read_text())
    write(repo, "src/pkg/__init__.py", "from . import leaf\n")
    write(repo, "src/pkg/root.py", "import pkg.direct\nfrom . import relative\n__import__('pkg.dynamic')\n")
    write(repo, "src/pkg/direct.py", "from . import root\n")  # cycle
    write(repo, "src/pkg/relative.py", "from .leaf import value\n")
    write(repo, "src/pkg/dynamic.py", "value = 1\n")
    write(repo, "src/pkg/leaf.py", "value = 1\n")
    write(repo, "downstream.py", "value = 1\n")
    previous = commit(repo)
    write(repo, "downstream.py", "value = 2\n")
    current = commit(repo)
    trace = tmp_path / "runtime-trace.json"
    trace.write_text(json.dumps({"schema": EA.TRACE_SCHEMA, "runs": [
        {"case_id": "sample", "phase": phase, "py_files": [str(repo / "src/pkg/root.py")]}
        for phase in ("capture", "execute")]}))
    policy = {"schema": EA.SCHEMA, "shard_commit": previous,
              "allow": ["downstream.py"], "attested_by": "Operator", "submitted_by": "Submitter",
              "statement": "I authorize reuse with the disclosed residual risks.",
              "evidence": {"runtime_trace": {"path": str(trace), "sha256": EA.sha256(trace.read_bytes())},
                           "notes": ["Trace is operator-authored."]}}
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps(policy))
    return SimpleNamespace(repo=repo, previous=previous, current=current, saved=stamp(repo, previous),
                           fresh=stamp(repo, current), policy=policy, attestation=attestation, trace=trace)


def decide(h):
    h.attestation.write_text(json.dumps(h.policy))
    return EA.decide(h.saved, h.fresh, repo=h.repo, attestation=h.attestation)


@pytest.mark.parametrize("field", ["software_spec_sha256", "source_digest", "sources", "artifact_digests"])
def test_stamp_differences_refuse(history, field):
    history.fresh[field] = ["changed"] if field == "sources" else "changed"
    with pytest.raises(EA.AttestationRefused, match="outside commit labels"):
        decide(history)


@pytest.mark.parametrize("stamp_name,field,value,message", [
    ("saved", "merlin_commit", "f" * 40, "saved commit differs"),
    ("fresh", "merlin_commit", "f" * 40, "current stamp commit differs"),
    ("saved", "merlin_source_kind", "plain_source_export", "not git_worktree"),
    ("fresh", "merlin_diff_sha256", "f" * 64, "diff is not clean"),
])
def test_invalid_stamp_identity(history, stamp_name, field, value, message):
    getattr(history, stamp_name)[field] = value
    with pytest.raises(EA.AttestationRefused, match=message):
        decide(history)


def test_same_commit_refuses(history):
    h = history
    h.policy["shard_commit"] = h.current
    h.saved = copy.deepcopy(h.fresh)
    with pytest.raises(EA.AttestationRefused, match="equals current"):
        decide(h)


def test_diverged_commit_refuses(history):
    h = history
    git(h.repo, "checkout", "-q", "--detach", h.previous)
    write(h.repo, "branch.py", "other = 1\n")
    diverged = commit(h.repo)
    git(h.repo, "checkout", "-q", "--detach", h.current)
    h.policy["shard_commit"] = diverged
    h.saved = stamp(h.repo, diverged)
    with pytest.raises(EA.AttestationRefused, match="merge-base --is-ancestor"):
        decide(h)


@pytest.mark.parametrize("dirty", ["tracked", "untracked", "staged"])
def test_dirty_tree_refuses(history, dirty):
    write(history.repo, "downstream.py" if dirty != "untracked" else "untracked.py", "dirty = True\n")
    if dirty == "staged":
        git(history.repo, "add", "-A")
    with pytest.raises(EA.AttestationRefused, match="worktree is dirty"):
        decide(history)


@pytest.mark.parametrize("stamp_name", ["saved", "fresh"])
def test_nested_dirty_files_refuse(history, stamp_name):
    getattr(history, stamp_name)["merlin"]["dirty_files"] = 1
    with pytest.raises(EA.AttestationRefused, match="dirty_files is not zero"):
        decide(history)


def test_outside_allow_refuses(history):
    history.policy["allow"] = ["unrelated.py"]
    with pytest.raises(EA.AttestationRefused, match="outside attestation allow"):
        decide(history)


@pytest.mark.parametrize("raw,message", [('{"schema":1,"schema":2}', "duplicate attestation key"),
                                        ('{', "Expecting"), ('[]', "must be an object"),
                                        ('{}', "missing required")])
def test_malformed_json_refuses(history, raw, message):
    history.attestation.write_text(raw)
    with pytest.raises(EA.AttestationRefused, match=message):
        EA.decide(history.saved, history.fresh, repo=history.repo, attestation=history.attestation)


def load_driver(path):
    spec = importlib.util.spec_from_file_location("test_attestation_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def driver_fixture(history, tmp_path, monkeypatch):
    driver = load_driver(history.repo / DRIVER)
    real_run = subprocess.run
    def no_launch(command, *args, **kwargs):
        assert command[0] == "git", "runner subprocess must never launch"
        return real_run(command, *args, **kwargs)
    monkeypatch.setattr(driver.subprocess, "run", no_launch)
    selection = {"cases": []}
    monkeypatch.setattr(driver.SV, "load_selection", lambda *a: selection)
    monkeypatch.setattr(driver.SV, "runner_suite", lambda *a: {"selected_cases": []})
    output = tmp_path / "output"
    base = output / "spec-validation"
    base.mkdir(parents=True)
    (base / "selection-suite.json").write_bytes(json.dumps({"selected_cases": []}, sort_keys=True).encode())
    (base / "provenance.json").write_text(json.dumps(history.saved))
    (tmp_path / "current-stamp.json").write_text(json.dumps(history.fresh))
    argv = ["--selection", "selection.json", "--output", str(output), "--software-spec", "spec.yaml",
            "--host-capabilities", "host.yaml", "--component-host-capabilities", "component.yaml",
            "--", "--target", "test-target", "--facts", "facts.json", "--repo", str(history.repo)]
    return driver, base, argv


def test_driver_missing_provenance_refuses(history, tmp_path, monkeypatch, capsys):
    driver, base, argv = driver_fixture(history, tmp_path, monkeypatch)
    (base / "provenance.json").unlink()
    monkeypatch.setenv("MERLIN_SPEC_VALIDATION_ATTESTATION", str(history.attestation))
    assert driver.main(argv) == 1
    assert "attestation runner provenance is missing" in capsys.readouterr().err
    assert not (base / "provenance.json").exists()
    assert not (base / "evidence-attestation.json").exists()


def test_driver_refusal_never_falls_back_or_changes_provenance(history, tmp_path, monkeypatch, capsys):
    driver, base, argv = driver_fixture(history, tmp_path, monkeypatch)
    old = (base / "provenance.json").read_bytes()
    history.policy["allow"] = ["unrelated.py"]
    history.attestation.write_text(json.dumps(history.policy))
    monkeypatch.setenv("MERLIN_SPEC_VALIDATION_ATTESTATION", str(history.attestation))
    assert driver.main(argv) == 1
    assert "changed paths outside attestation allow" in capsys.readouterr().err
    assert (base / "provenance.json").read_bytes() == old
    assert not (base / "evidence-attestation.json").exists()


@pytest.mark.parametrize("record_only", [False, True])
def test_env_unset_keeps_existing_driver_behavior(history, tmp_path, monkeypatch, record_only):
    driver, base, argv = driver_fixture(history, tmp_path, monkeypatch)
    monkeypatch.delenv("MERLIN_SPEC_VALIDATION_ATTESTATION", raising=False)
    commands = []
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: commands.append(a))
    record = {"sha256": "test", "verdict": "validated", "verdict_counts": {}, "mismatches": [], "unmodeled": []}
    seen = []
    monkeypatch.setattr(driver.SV, "prepare", lambda *a, **kw: seen.append(kw) or record)
    if record_only:
        argv.insert(0, "--record-only")
    assert driver.main(argv) == 0
    assert len(commands) == (0 if record_only else 1)
    assert "evidence_attestation" not in seen[0]
    assert not (base / "evidence-attestation.json").exists()


@pytest.mark.parametrize("record_only", [False, True])
def test_driver_accept_builds_record_loadable_by_derivation(history, tmp_path, monkeypatch, capsys, record_only):
    # Reuse the ordinary real record fixture; only execution is replaced by the tiny runner.
    from test_phase0_spec_validation import Run, _case, _write_selection, spec

    h = history
    run = Run(tmp_path, spec.__wrapped__(tmp_path))
    h.saved.update(run.stamp)
    h.fresh.update(run.stamp)
    run.stamp = h.saved
    run.add(_case("host", "aten.sort.default"), admitted=False, lane="host")
    path, suites, _ = _write_selection(tmp_path, run.cases)
    selection = SV.load_selection(path, suites)
    base = run.root / "spec-validation"
    (base / "provenance.json").write_text(json.dumps(h.saved))
    old = (base / "provenance.json").read_bytes()
    (base / "selection-suite.json").write_bytes(json.dumps(SV.runner_suite(selection), sort_keys=True).encode())
    (tmp_path / "current-stamp.json").write_text(json.dumps(h.fresh))
    monkeypatch.setenv("MERLIN_SPEC_VALIDATION_ATTESTATION", str(h.attestation))
    driver = load_driver(h.repo / DRIVER)
    argv = ["--selection", str(path), "--bounded", str(suites["bounded"]), "--output", str(run.root),
            "--software-spec", str(run.spec), "--host-capabilities", str(run.descriptor_host),
            "--component-host-capabilities", str(run.component_host), "--no-reverify-evidence",
            "--", "--target", "gemmini", "--facts", str(repo_root() / "merlin/tests/fixtures/core_aten_device/facts.json"),
            "--repo", str(h.repo)]
    if record_only:
        argv.insert(0, "--record-only")
    assert driver.main(argv) == 0
    assert "ATTESTED, not proven" in capsys.readouterr().out
    assert (base / "provenance.json").read_bytes() == old
    receipt_raw = (base / "evidence-attestation.json").read_bytes()
    receipt = json.loads(receipt_raw)
    assert receipt["status"] == "ATTESTED_NOT_PROVEN"
    assert receipt["residual_risks"] == EA.RESIDUAL_RISKS
    assert receipt["proof"] == "none: operator attestation; see residual_risks"
    record_path = base / "record.json"
    record = SV.verify(record_path)
    assert record["evidence_attestation"] == {"status": "ATTESTED_NOT_PROVEN", "receipt_sha256": EA.sha256(receipt_raw),
                                              "shard_commit": h.previous, "current_commit": h.current}
    # requirements.py uses bind(), whose _load_bound calls verify().
    bound = SV.bind(record_path, software_spec=run.spec, workload_spec={SV.GATE_KEY: "required"},
                    input_identity=record["input_identity"])
    assert bound["status"] == "bound"
    SV.verify_binding(bound, software_spec_sha256=record["software_spec_sha256"],
                      workload_spec={SV.GATE_KEY: "required"}, input_identity=record["input_identity"])


def test_patch_files_are_outside_literal_stamp_sources():
    runner_path = repo_root() / EA.RUNNER
    tree = ast.parse(runner_path.read_bytes())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stamp")
    sources = {EA.RUNNER}
    for node in ast.walk(function):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "relative" and isinstance(node.iter, ast.Tuple):
            sources.update(ast.literal_eval(node.iter))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__import__":
            module = ast.literal_eval(node.args[0])
            assert module == "merlin.targetgen.core_aten_bounded_runner"
            candidates = [root / (module.replace(".", "/") + ".py")
                          for root in [repo_root() / "src", *(repo_root() / "packages").glob("*/src")]]
            matches = [p for p in candidates if p.is_file()]
            assert len(matches) == 1
            sources.add(matches[0].relative_to(repo_root()).as_posix())
    paths = [node for node in ast.walk(function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "Path"
             and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "__file__"]
    assert len(paths) == 1
    assert len(sources) > 40
    assert not TOUCHED & sources
    changed = git(repo_root(), "diff", "HEAD", "--name-only").decode().splitlines()
    changed += git(repo_root(), "ls-files", "--others", "--exclude-standard").decode().splitlines()
    assert not set(changed) & sources


@pytest.mark.parametrize("override", ["--worker=capture", "--work=capture", "--evaluate-only"])
def test_active_driver_refuses_pre_stamp_execution_overrides(history, tmp_path, monkeypatch, capsys, override):
    driver, base, argv = driver_fixture(history, tmp_path, monkeypatch)
    monkeypatch.setenv("MERLIN_SPEC_VALIDATION_ATTESTATION", str(history.attestation))
    old = (base / "provenance.json").read_bytes()
    assert driver.main([*argv, override]) == 1
    assert "attestation refuses runner control overrides" in capsys.readouterr().err
    assert (base / "provenance.json").read_bytes() == old


def test_active_driver_refuses_changed_retained_input(history, tmp_path, monkeypatch, capsys):
    driver, base, argv = driver_fixture(history, tmp_path, monkeypatch)
    monkeypatch.setenv("MERLIN_SPEC_VALIDATION_ATTESTATION", str(history.attestation))
    (base / "selection-suite.json").write_bytes(b"different input")
    assert driver.main(argv) == 1
    assert "identical retained runner input" in capsys.readouterr().err
    assert (base / "selection-suite.json").read_bytes() == b"different input"


def test_empty_commit_diff_refuses(history):
    h = history
    git(h.repo, "checkout", "-q", "--detach", h.previous)
    git(h.repo, "commit", "--allow-empty", "-qm", "empty test commit")
    h.current = git(h.repo, "rev-parse", "HEAD").decode().strip()
    h.fresh = stamp(h.repo, h.current)
    with pytest.raises(EA.AttestationRefused, match="commit diff is empty"):
        decide(h)


def test_source_path_relocation_refuses(history):
    history.fresh["sources"] = ["/different/absolute/src/pkg/root.py"]
    with pytest.raises(EA.AttestationRefused, match="outside commit labels"):
        decide(history)


def test_real_runner_writes_provenance_only_after_stamp():
    # The production runner remains unchanged; guard the ordering relied on by the sentinel.
    tree = ast.parse((repo_root() / EA.RUNNER).read_bytes())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    stamp = next(node for node in main.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "stamp")
    write_call = next(node for node in main.body if isinstance(node, ast.Expr)
                      and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                      and node.value.func.id == "atomic_json")
    assert stamp.lineno < write_call.lineno



def test_accept_records_attestation_and_informational_evidence(history):
    h = history
    receipt = decide(h)
    assert receipt["schema"] == EA.RECEIPT_SCHEMA
    assert receipt["status"] == "ATTESTED_NOT_PROVEN"
    assert receipt["decision"] == "attested_reuse_of_shards"
    assert receipt["proof"] == "none: operator attestation; see residual_risks"
    assert (receipt["shard_commit"], receipt["current_commit"]) == (h.previous, h.current)
    assert receipt["changed"] == [{"path": "downstream.py", "sha256_at_P": EA.sha256(b"value = 1\n"),
                                   "sha256_at_C": EA.sha256(b"value = 2\n")}]
    assert receipt["saved_stamp_sha256"] == EA.sha256(EA.canonical(h.saved))
    assert receipt["current_stamp_sha256"] == EA.sha256(EA.canonical(h.fresh))
    assert receipt["attestation_sha256"] == EA.sha256(h.attestation.read_bytes())
    for field in ("attested_by", "submitted_by", "statement", "allow"):
        assert receipt[field] == h.policy[field]
    assert receipt["evidence"]["notes"] == h.policy["evidence"]["notes"]
    trace = receipt["evidence"]["runtime_trace"]
    assert trace["runs"] == {"capture": 1, "execute": 1}
    assert trace["distinct_py_file_count"] == trace["repo_file_count"] == 1
    assert trace["allowed_or_changed_files"] == []
    assert receipt["residual_risks"] == EA.RESIDUAL_RISKS
    assert len(EA.RESIDUAL_RISKS) == 8
    assert receipt["created_utc"].endswith("+00:00")


@pytest.mark.parametrize("stamp_name", ["saved", "fresh"])
@pytest.mark.parametrize("field,value,message", [
    ("merlin_source_kind", "export", "not git_worktree"),
    ("merlin_diff_sha256", "f" * 64, "diff is not clean"),
    ("merlin_source_label", "f" * 40, "labels are inconsistent"),
    ("merlin", {}, "nested merlin stamp is malformed"),
])
def test_each_stamp_identity_refuses(history, stamp_name, field, value, message):
    getattr(history, stamp_name)[field] = value
    with pytest.raises(EA.AttestationRefused, match=message):
        decide(history)


@pytest.mark.parametrize("stamp_name", ["saved", "fresh"])
@pytest.mark.parametrize("field,value,message", [
    ("commit", "f" * 40, "labels are inconsistent"),
    ("source_label", "f" * 40, "labels are inconsistent"),
    ("dirty_files", False, "dirty_files is not zero"),
    ("extra", 0, "nested merlin stamp is malformed"),
])
def test_nested_stamp_identity_refuses(history, stamp_name, field, value, message):
    getattr(history, stamp_name)["merlin"][field] = value
    with pytest.raises(EA.AttestationRefused, match=message):
        decide(history)


@pytest.mark.parametrize("change,message", [
    ({"unknown": True}, "unknown keys"), ({"current_commit": "f" * 40}, "unknown keys"),
    ({"schema": "wrong"}, "schema must"), ({"shard_commit": "x" * 40}, "40-hex"),
    ({"attested_by": "  "}, "attested_by is empty"), ({"submitted_by": 1}, "submitted_by is empty"),
    ({"statement": ""}, "statement is empty"), ({"allow": []}, "nonempty list"),
    ({"allow": ["../escape"]}, "invalid repository-relative"),
    ({"allow": ["/absolute"]}, "invalid repository-relative"),
    ({"allow": ["src/*.py"]}, "invalid repository-relative"),
    ({"allow": ["a", "a"]}, "duplicate paths"),
    ({"allow": ["a/./b"]}, "invalid repository-relative"),
    ({"allow": ["a//b"]}, "invalid repository-relative"), ({"allow": [1]}, "nonempty strings"),
    ({"evidence": {}}, "exactly runtime_trace and notes"),
])
def test_malformed_attestation_refuses(history, change, message):
    history.policy.update(change)
    with pytest.raises(EA.AttestationRefused, match=message):
        decide(history)


@pytest.mark.parametrize("notes", [None, "text", [1]])
def test_malformed_notes_refuse(history, notes):
    history.policy["evidence"]["notes"] = notes
    with pytest.raises(EA.AttestationRefused, match="list of strings"):
        decide(history)


@pytest.mark.parametrize("reference,message", [
    (None, "exactly path and sha256"),
    ({"path": "relative.json", "sha256": "a" * 64}, "path must be absolute"),
    ({"path": "/missing.json", "sha256": "invalid"}, "sha256 must be 64-hex"),
    ({"path": "/missing.json", "sha256": "a" * 64, "extra": True}, "exactly path and sha256"),
])
def test_bad_trace_reference_refuses(history, reference, message):
    history.policy["evidence"]["runtime_trace"] = reference
    with pytest.raises(EA.AttestationRefused, match=message):
        decide(history)


def test_trace_hash_mismatch_refuses(history):
    history.trace.write_text("{}")
    with pytest.raises(EA.AttestationRefused, match="runtime_trace sha256 mismatch"):
        decide(history)


def test_missing_trace_refuses(history):
    history.trace.unlink()
    with pytest.raises(EA.AttestationRefused, match="No such file"):
        decide(history)


def update_trace(h, document):
    h.trace.write_text(json.dumps(document))
    h.policy["evidence"]["runtime_trace"]["sha256"] = EA.sha256(h.trace.read_bytes())


def test_allowed_traced_file_is_recorded_without_refusal(history):
    h = history
    document = json.loads(h.trace.read_bytes())
    document["runs"][0]["py_files"] += [str(h.repo / "downstream.py"), str(h.repo / "downstream.py"),
                                         str(h.repo.parent / "external.py")]
    update_trace(h, document)
    trace = decide(h)["evidence"]["runtime_trace"]
    assert trace["allowed_or_changed_files"] == ["downstream.py"]
    assert trace["distinct_py_file_count"] == 3
    assert trace["repo_file_count"] == 2


@pytest.mark.parametrize("document", [
    {"schema": EA.TRACE_SCHEMA, "runs": []},
    {"schema": EA.TRACE_SCHEMA, "runs": [{"phase": "capture", "py_files": []}]},
    {"schema": "wrong", "runs": []}, [],
])
def test_trace_completeness_and_format_do_not_gate(history, document):
    update_trace(history, document)
    assert decide(history)["status"] == "ATTESTED_NOT_PROVEN"


def test_invalid_trace_json_is_informational(history):
    history.trace.write_text("{")
    history.policy["evidence"]["runtime_trace"]["sha256"] = EA.sha256(history.trace.read_bytes())
    assert decide(history)["evidence"]["runtime_trace"]["errors"]


def test_direct_importers_are_informational_in_both_revisions(history):
    h = history
    git(h.repo, "checkout", "-q", "--detach", h.previous)
    write(h.repo, "src/pkg/eager.py", "from . import leaf\n")
    write(h.repo, "src/pkg/lazy.py", "def load():\n    from . import leaf\n")
    write(h.repo, "packages/extra/src/extra/__init__.py", "import pkg.leaf\n")
    write(h.repo, "build_tools/scripts/observer.py", "import pkg.leaf\n")
    write(h.repo, "src/pkg/broken.py", "this is invalid ! python\n")
    h.previous = commit(h.repo)
    write(h.repo, "src/pkg/leaf.py", "value = 7\n")
    h.current = commit(h.repo)
    h.saved, h.fresh = stamp(h.repo, h.previous), stamp(h.repo, h.current)
    h.policy.update(shard_commit=h.previous, allow=["src/pkg/leaf.py"])
    summary = decide(h)["evidence"]["direct_importers"]
    for revision in ("P", "C"):
        eager = {row["path"] for row in summary[revision]["module_level"]["importers"]}
        all_bodies = {row["path"] for row in summary[revision]["all_bodies"]["importers"]}
        assert {"src/pkg/eager.py", "packages/extra/src/extra/__init__.py",
                "build_tools/scripts/observer.py"} <= eager
        assert "src/pkg/lazy.py" not in eager
        assert "src/pkg/lazy.py" in all_bodies
        assert any(error["path"] == "src/pkg/broken.py" for error in summary[revision]["all_bodies"]["errors"])


@pytest.mark.parametrize("help_flag", ["-h", "--help", "--he", "--hel", "--help=true"])
@pytest.mark.parametrize("record_only", [False, True])
def test_runner_help_refuses_before_import(history, tmp_path, monkeypatch, capsys, help_flag, record_only):
    driver, base, argv = driver_fixture(history, tmp_path, monkeypatch)
    monkeypatch.setenv("MERLIN_SPEC_VALIDATION_ATTESTATION", str(history.attestation))
    def no_import(*args):
        pytest.fail("runner must not be imported for help")
    monkeypatch.setattr(driver.importlib.util, "spec_from_file_location", no_import)
    old = (base / "provenance.json").read_bytes()
    if record_only:
        argv.insert(0, "--record-only")
    assert driver.main([*argv, help_flag]) == 1
    assert "refuses runner help flags" in capsys.readouterr().err
    assert (base / "provenance.json").read_bytes() == old


@pytest.mark.parametrize("behavior", ["mutate", "delete", "exit_zero", "return", "mutate_exit_zero", "import_exit_zero"])
@pytest.mark.parametrize("record_only", [False, True])
def test_bad_runner_restores_provenance_and_refuses(history, tmp_path, monkeypatch, capsys, behavior, record_only):
    h = history
    action = {
        "mutate": '(args.output / args.label / "provenance.json").write_text("changed")',
        "delete": '(args.output / args.label / "provenance.json").unlink()',
        "exit_zero": 'raise SystemExit(0)',
        "return": 'return 0',
        "mutate_exit_zero": '(args.output / args.label / "provenance.json").write_text("changed"); raise SystemExit(0)',
        "import_exit_zero": 'raise SystemExit(0)',
    }[behavior]
    source = STUB.replace('    block = stamp(args)', '    ' + action + '\n    block = stamp(args)')
    if behavior == "import_exit_zero":
        source = 'raise SystemExit(0)\n' + STUB
    write(h.repo, EA.RUNNER, source)
    h.current = commit(h.repo)
    h.fresh = stamp(h.repo, h.current)
    driver, base, argv = driver_fixture(h, tmp_path, monkeypatch)
    monkeypatch.setenv("MERLIN_SPEC_VALIDATION_ATTESTATION", str(h.attestation))
    old = (base / "provenance.json").read_bytes()
    if record_only:
        argv.insert(0, "--record-only")
    assert driver.main(argv) == 1
    error = capsys.readouterr().err
    assert ("restored" if behavior in {"mutate", "delete", "mutate_exit_zero"} else
            "without calling stamp" if behavior == "return" else "SystemExit(0)") in error
    assert (base / "provenance.json").read_bytes() == old
    assert not (base / "evidence-attestation.json").exists()


def test_added_and_deleted_file_hashes(history):
    h = history
    write(h.repo, "added.py", "value = 3\n")
    (h.repo / "downstream.py").unlink()
    h.current = commit(h.repo)
    h.fresh = stamp(h.repo, h.current)
    h.policy["allow"] = ["downstream.py", "added.py"]
    assert decide(h)["changed"] == [
        {"path": "added.py", "sha256_at_P": None, "sha256_at_C": EA.sha256(b"value = 3\n")},
        {"path": "downstream.py", "sha256_at_P": EA.sha256(b"value = 1\n"), "sha256_at_C": None},
    ]


@pytest.mark.parametrize("with_argv", [True, False])
def test_probe_sentinel_preserves_provenance_and_argv(history, tmp_path, with_argv):
    h = history
    if not with_argv:
        # Test the real runner's no-argument signature too.
        runner = tmp_path / "no-argv.py"
        runner.write_text(STUB.replace("def main(argv=None):", "def main():").replace("p.parse_known_args(argv)",
                                                                                  "p.parse_known_args()"))
    else:
        runner = h.repo / EA.RUNNER
    output = tmp_path / "output"
    (output / "spec-validation").mkdir(parents=True)
    path = output / "spec-validation/provenance.json"
    path.write_bytes(b"old stamp bytes\n")
    (tmp_path / "current-stamp.json").write_text(json.dumps(h.fresh))
    driver = load_driver(repo_root() / DRIVER)
    import sys
    original = sys.argv
    assert driver._probe_stamp(runner, ["--repo", str(h.repo), "--output", str(output),
                                        "--label", "spec-validation"]) == h.fresh
    assert sys.argv is original
    assert path.read_bytes() == b"old stamp bytes\n"


@pytest.mark.parametrize("field", ["schema", "shard_commit", "allow", "attested_by", "submitted_by", "statement", "evidence"])
def test_missing_attestation_key_refuses(history, field):
    del history.policy[field]
    with pytest.raises(EA.AttestationRefused, match="missing required keys"):
        decide(history)


def test_nested_duplicate_key_refuses(history):
    raw = history.attestation.read_text().replace('"notes": [', '"notes": [], "notes": [')
    history.attestation.write_text(raw)
    with pytest.raises(EA.AttestationRefused, match="duplicate attestation key: notes"):
        EA.load_attestation(history.attestation)


@pytest.mark.parametrize("gate,message", [
    ("saved_commit", "saved commit differs"), ("current_commit", "current stamp commit differs"),
    ("saved_dirty", "diff is not clean"), ("current_dirty", "dirty_files is not zero"),
    ("stamp_difference", "outside commit labels"), ("source_kind", "not git_worktree"),
    ("outside_allow", "outside attestation allow"), ("trace_sha", "runtime_trace sha256 mismatch"),
    ("dirty_worktree", "worktree is dirty"),
])
def test_driver_hard_gates_preserve_provenance(history, tmp_path, monkeypatch, capsys, gate, message):
    h = history
    if gate == "saved_commit":
        h.saved["merlin_commit"] = "f" * 40
    elif gate == "current_commit":
        h.fresh["merlin_commit"] = "f" * 40
    elif gate == "saved_dirty":
        h.saved["merlin_diff_sha256"] = "f" * 64
    elif gate == "current_dirty":
        h.fresh["merlin"]["dirty_files"] = 1
    elif gate == "stamp_difference":
        h.fresh["source_digest"] = "f" * 64
    elif gate == "source_kind":
        h.saved["merlin_source_kind"] = "export"
    elif gate == "outside_allow":
        h.policy["allow"] = ["other.py"]
    elif gate == "trace_sha":
        h.trace.write_text("{}")
    elif gate == "dirty_worktree":
        write(h.repo, "downstream.py", "dirty\n")
    h.attestation.write_text(json.dumps(h.policy))
    driver, base, argv = driver_fixture(h, tmp_path, monkeypatch)
    monkeypatch.setenv("MERLIN_SPEC_VALIDATION_ATTESTATION", str(h.attestation))
    old = (base / "provenance.json").read_bytes()
    assert driver.main(argv) == 1
    assert message in capsys.readouterr().err
    assert (base / "provenance.json").read_bytes() == old
    assert not (base / "evidence-attestation.json").exists()
