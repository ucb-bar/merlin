"""The performance phase starts from one explicit, frozen Arm-4 compiler or not at all.

These tests pin the boundary that keeps a performance run attributable: no "latest" lookup, no live
submission directory, no vacuous 0/0 completion, and no untrusted entrypoint outside the derived bwrap
policy.  The expensive simulator is not launched here; the mount table and completion arithmetic are
pure and therefore exercise the refusal paths in CI.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from merlin.benchharness import hash_tree
from merlin.common.paths import repo_root


def _load_campaign():
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("_perf_campaign_gate", scripts / "perf_campaign.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PC = _load_campaign()


def _functional_run(tmp_path: Path, run_id: str = "arm4_explicit") -> tuple[Path, str]:
    run = tmp_path / "merlin_assisted" / run_id
    sub = run / "submission"
    sub.mkdir(parents=True)
    (sub / "manifest.yaml").write_text(yaml.safe_dump({
        "artifact_type": "mlir_oot_target_backend",
        "target": "fixture",
        "language": "python",
        "entrypoints": {"tool": "tool.py"},
        "commands": {},
    }))
    (sub / "tool.py").write_text("print('fixture')\n")
    digest = hash_tree(sub)["sha256"]

    (run / "environment.yaml").write_text(yaml.safe_dump({
        "run_id": run_id,
        "bundle_id": "merlin_assisted_rtlchecks_hwbringup_v0",
        "sandbox": "bwrap",
        "bundle_input_snapshot": {
            "version": 2, "content_sha256": "a" * 64, "n_files": 7, "n_bytes": 41,
        },
        "isolation_violations": [],
        "golden_mask_selftest": {"n_answer_files_masked": 3, "leaked_answer_files": []},
    }))
    (run / "qa_loop_summary.yaml").write_text(yaml.safe_dump({
        "converged": True,
        "rounds": [{"answer_access_clean": True, "audit_hits": []}],
        "finalize": {"answer_access_clean": True, "audit_hits": [], "regrade_all_pass": True},
    }))
    (run / "freeze.json").write_text(json.dumps({
        "submission_sha256": digest, "submission_sha256_recheck": digest,
        "workspace_mutable_after_freeze": False, "frozen_at": "2026-08-31T00:00:00Z",
    }))
    (run / "run_manifest.yaml").write_text(yaml.safe_dump({
        "run_id": run_id,
        "submission_sha256": digest,
        "integrity_status": "clean",
        "integrity_exempt": False,
        "gradeable": True,
        "public_dev": {"functional_pass": 1, "passed": "2/2", "highest_tier": "L3"},
        "hidden": {"functional_pass": 1, "passed": "1/1"},
    }))
    for phase, names in (("public", ("p0", "p1")), ("hidden", ("h0",))):
        d = run / f"grading_{phase}"
        d.mkdir()
        rows = [{"capsule": n, "status": "pass", "tiers": {"L2": "pass", "L3": "pass"}}
                for n in names]
        (d / "score_capsule.json").write_text(json.dumps({
            "n_capsules": len(rows), "n_passed": len(rows), "functional_pass": 1,
            "gradeable": True, "integrity_status": "clean", "integrity_exempt": False,
            "per_capsule": rows,
        }))
    return run, digest


def test_one_exact_arm4_run_and_digest_are_required(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    rec = PC.inspect_functional_run(tmp_path, run.name, digest)
    assert rec.run_dir == run.resolve()
    assert rec.digest == digest
    assert rec.public_capsules == 2 and rec.hidden_capsules == 1

    with pytest.raises(PC.CampaignGateError, match="explicit functional run id"):
        PC.inspect_functional_run(tmp_path, "", digest)
    with pytest.raises(PC.CampaignGateError, match="simple directory name"):
        PC.inspect_functional_run(tmp_path, "../arm4_explicit", digest)
    with pytest.raises(PC.CampaignGateError, match="does not match"):
        PC.inspect_functional_run(tmp_path, run.name, "0" * 64)


def test_a_non_arm4_or_vacuous_functional_run_is_refused(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    env = yaml.safe_load((run / "environment.yaml").read_text())
    env["bundle_id"] = "merlin_assisted_hwbringup_v0"
    (run / "environment.yaml").write_text(yaml.safe_dump(env))
    with pytest.raises(PC.CampaignGateError, match="Arm-4 RTL-checks bundle"):
        PC.inspect_functional_run(tmp_path, run.name, digest)

    run, digest = _functional_run(tmp_path, "arm4_zero")
    manifest = yaml.safe_load((run / "run_manifest.yaml").read_text())
    manifest["hidden"]["passed"] = "0/0"
    (run / "run_manifest.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(PC.CampaignGateError, match="non-vacuous"):
        PC.inspect_functional_run(tmp_path, run.name, digest)


def test_functional_run_must_have_frozen_bundle_inputs_v2(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    environment = yaml.safe_load((run / "environment.yaml").read_text())
    environment.pop("bundle_input_snapshot")
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    with pytest.raises(PC.CampaignGateError, match="immutable bundle-input snapshot v2"):
        PC.inspect_functional_run(tmp_path, run.name, digest)


def test_the_perf_workspace_is_a_copy_and_is_digest_checked(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    rec = PC.inspect_functional_run(tmp_path, run.name, digest)
    snapshot = PC.materialize_perf_workspace(rec, tmp_path / "perf")
    assert snapshot != rec.submission_dir
    assert hash_tree(snapshot)["sha256"] == digest

    (rec.submission_dir / "tool.py").write_text("print('functional tree moved later')\n")
    assert hash_tree(snapshot)["sha256"] == digest
    assert "fixture" in (snapshot / "tool.py").read_text()


def test_digest_excluded_submission_state_is_refused(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    cache = run / "submission" / "__pycache__"
    cache.mkdir()
    (cache / "tool.pyc").write_bytes(b"unhashed executable bytes")
    with pytest.raises(PC.CampaignGateError, match="digest-excluded path"):
        PC.inspect_functional_run(tmp_path, run.name, digest)


def test_perf_fork_detects_snapshot_drift(tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    rec = PC.inspect_functional_run(tmp_path, run.name, digest)
    snapshot = PC.materialize_perf_workspace(rec, tmp_path / "perf")
    fork = PC.functional_fork(rec)
    held = PC.check_fork(fork, snapshot)
    assert held.ok is True and held.state == "held"

    # The host can still change bytes; the bwrap mount is the runtime boundary and this digest check is
    # the before/after backstop. Restore one write bit to model a bug in that boundary.
    tool = snapshot / "tool.py"
    tool.chmod(0o644)
    tool.write_text("print('mutated')\n")
    assert PC.check_fork(fork, snapshot).ok is None


def test_completion_is_non_vacuous_and_every_expected_cell_must_finish() -> None:
    with pytest.raises(PC.CampaignGateError, match="zero expected"):
        PC.completion_counts([], {})

    expected = {"k0": ("spike", "verilator")}
    complete = [{"kernel": "k0", "approaches": {"arm4": {"per_sim": {
        "spike": {"correct": True, "cycles": 7},
        "verilator": {"correct": True, "cycles": 9},
    }}}}]
    counts = PC.completion_counts(complete, expected)
    assert counts == {"expected": 2, "reported": 2, "correct": 2, "cycles_measured": 2,
                      "failed": 0, "missing": 0, "complete": True}

    broken = [{"kernel": "k0", "approaches": {"arm4": {"per_sim": {
        "spike": {"correct": True, "cycles": 7},
    }}}}]
    assert PC.completion_report(broken, expected) == {
        "expected": 2, "reported": 1, "correct": 1, "cycles_measured": 1,
        "failed": 0, "missing": 1, "complete": False,
    }
    with pytest.raises(PC.CampaignGateError, match="1 of 2 expected"):
        PC.completion_counts(broken, expected)


def test_package_sandbox_is_answer_closed_networkless_and_submission_read_only(tmp_path: Path) -> None:
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = (repo_root() / "merlin/experiments/capsule_bench/targets/gemmini"
                  / "target_experiment.yaml")
    te = load_target_experiment(descriptor)
    ws = tmp_path / "workspace"
    pkg = ws / "submission"
    pkg.mkdir(parents=True)
    policy = PC.package_sandbox_policy(te, ws, pkg)
    assert policy.coverage_gap == ()
    assert "--unshare-net" in policy.argv
    assert "--clearenv" in policy.argv
    assert str(Path.home() / ".claude") not in policy.argv
    pairs = list(zip(policy.argv, policy.argv[1:]))
    assert ("--ro-bind", str(pkg)) in pairs
    assert policy.required_tools, "tool enforcement must not be an empty loop"


def test_actual_entrypoint_and_every_tool_probe_use_the_bwrap_policy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from merlin.targetgen import oot_runner
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = (repo_root() / "merlin/experiments/capsule_bench/targets/gemmini"
                  / "target_experiment.yaml")
    te = load_target_experiment(descriptor)
    ws = tmp_path / "workspace"
    pkg_dir = tmp_path / "frozen" / "submission"
    pkg_dir.mkdir(parents=True)
    tool = pkg_dir / "tool.py"
    tool.write_text("print('boxed')\n")
    package = oot_runner.Package(pkg_dir, {
        "language": "python", "build": None,
        "commands": {"parse": {"argv": ["{tool}", "{input_mlir}"]}},
    }, tool)
    inp = ws / "generated" / "input.interface.mlir"
    inp.parent.mkdir(parents=True)
    inp.write_text("module {}\n")
    policy = PC.package_sandbox_policy(te, ws, pkg_dir)
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(PC.subprocess, "run", fake_run)
    probe_rows = PC.run_tool_probes(policy)
    assert len(probe_rows) == len(policy.required_tools)
    with PC.boxed_entrypoints(policy):
        result = oot_runner.run_entrypoint(package, "parse", inp)
        untrusted_build = oot_runner.Package(pkg_dir, {
            **package.manifest, "build": {"command": ["touch", "escaped-host-build"]},
        }, tool)
        with pytest.raises(PC.CampaignGateError, match="no host build"):
            oot_runner.build_package(untrusted_build)
    assert result.returncode == 0
    assert len(calls) == len(policy.required_tools) + 1
    entry_call = calls[-1]
    assert entry_call[0] == "bwrap"
    assert "--unshare-net" in entry_call and "--clearenv" in entry_call
    assert "perf-package" in entry_call and str(tool) in entry_call


def test_perf_runner_has_no_latest_or_mtime_submission_selection() -> None:
    source = (repo_root() / "merlin/experiments/gemmini_perf_bench/scripts/run_perf_bench.py").read_text()
    assert "_latest_submission" not in source
    assert "st_mtime" not in source
    assert "--functional-run-id" in source
    assert "--functional-submission-sha256" in source
