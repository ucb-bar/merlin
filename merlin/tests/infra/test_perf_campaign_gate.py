"""The performance phase starts from one explicit, frozen Arm-4 compiler or not at all.

These tests pin the boundary that keeps a performance run attributable: no "latest" lookup, no live
submission directory, no vacuous 0/0 completion, and no untrusted entrypoint outside the derived bwrap
policy.  The expensive simulator is not launched here; the mount table and completion arithmetic are
pure and therefore exercise the refusal paths in CI.
"""
from __future__ import annotations

import importlib.util
import contextlib
import concurrent.futures
import json
import multiprocessing
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
import perf_pk_claim as PK  # noqa: E402


def _load_runner():
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    spec = importlib.util.spec_from_file_location("_perf_runner_gate", scripts / "run_perf_bench.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _functional_input_snapshot(tmp_path: Path, run_id: str) -> tuple[dict, dict]:
    from merlin.targetgen.target_experiment import load_target_experiment

    root = tmp_path / "_qa_ws" / run_id / "bundle_inputs"
    package_rel = Path("out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8")
    source = repo_root() / package_rel
    package = root / "repo" / package_rel
    package.parent.mkdir(parents=True)
    shutil.copytree(source, package)
    frozen_models = root / "repo/merlin/contract/capsules/model"
    frozen_models.mkdir(parents=True)
    for name in ("M2_microvit_gemmini", "M3_host_island_seam_gemmini"):
        shutil.copytree(repo_root() / "merlin/contract/capsules/model" / name,
                        frozen_models / name)
    # A smaller mislabeled hidden capsule must never become the public admission sentinel.
    hidden_decoy = root / "repo/merlin/contract/capsules/hidden/M0_hidden_decoy"
    hidden_decoy.mkdir(parents=True)
    (hidden_decoy / "capsule.yaml").write_text(yaml.safe_dump({
        "name": "M0_hidden_decoy", "label": "public", "kind": "model",
        "lanes": {"require": ["on_mesh", "scalar_rvv_lane"]},
        "required_oracle_tiers": ["L2", "L3"],
    }))
    identity = PC._bundle_snapshot_content(root)
    snapshot = {"path": str(root), **identity, "version": 2}
    contract_rel = Path("merlin/contract")
    (root / "snapshot.json").write_text(json.dumps({
        "version": 2, "repo": str(repo_root()),
        "allowed": [package_rel.as_posix(), contract_rel.as_posix()],
        "grants": [
            {"path": package_rel.as_posix(),
             "destination": str((repo_root() / package_rel).absolute()),
             "snapshot": f"repo/{package_rel.as_posix()}"},
            {"path": contract_rel.as_posix(),
             "destination": str((repo_root() / contract_rel).absolute()),
             "snapshot": f"repo/{contract_rel.as_posix()}"},
        ], **identity,
    }, indent=2, sort_keys=True) + "\n")
    descriptor = (repo_root() / "merlin/experiments/capsule_bench/targets/gemmini"
                  / "target_experiment.yaml")
    te = load_target_experiment(descriptor)
    _resolved, host = te.resolve_host_lane(root=root / "repo")
    host["run_snapshot"] = dict(snapshot)
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(path.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)
    return snapshot, host


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

    bundle_snapshot, host_snapshot = _functional_input_snapshot(tmp_path, run_id)
    (run / "environment.yaml").write_text(yaml.safe_dump({
        "run_id": run_id,
        "bundle_id": "merlin_assisted_rtlchecks_hwbringup_v0",
        "sandbox": "bwrap",
        "bundle_input_snapshot": bundle_snapshot,
        "model_host_lane_snapshot": host_snapshot,
        "isolation_violations": [],
        "golden_mask_selftest": {"n_answer_files_masked": 3, "leaked_answer_files": []},
    }))
    (run / "qa_loop_summary.yaml").write_text(yaml.safe_dump({
        "converged": True, "numeric_all_pass": True, "workflow_conformant": True,
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


def _performance_block(family: str = "PK") -> dict:
    return {
        "level": "L1_tile", "family": family, "lever": "reduction_depth",
        "claim": "PREDICTS",
        "comparand": {"kind": "fitted_prediction", "against": "measured",
                       "cancels": ["M"], "demand_equal": ["operation"]},
        "falsifier": {"observation": "residual", "fires_when": "unbounded",
                       "negative_control": "fixed_extents"},
        "gate": {"traits": ["structural_pipeline_depth"], "instrument": "cycle_count",
                 "capacity": "two_points", "on_missing": "skip_with_evidence"},
        "regime": {"separation": "K_only", "layout": "direct"},
        "emitter": {"status": "existing", "entry": "fixture.build", "knobs": {}},
        "cost": {"tier": 1, "runs": 2, "projected_cycles": "derived", "basis": "fixture"},
        "acceptance": PK.supported_acceptance(),
    }


def _performance_corpus(tmp_path: Path, *, target: str = "fixture"):
    corpus_root = tmp_path / "capsules"
    primary = corpus_root / "isa"
    primary.mkdir(parents=True)
    phase = corpus_root / "_perf"
    capsule = phase / "PK00_k8"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text(yaml.safe_dump({
        "name": capsule.name, "kind": "model_slice", "label": "dev",
        "source_role": "derived_sweep", "performance": _performance_block(),
    }))
    (capsule / "capsule.interface.mlir").write_text("module {}\n")
    template = tmp_path / "_perf.yaml"
    template_doc = {"sweeps": [{"id": "PK"}]}
    template.write_text(yaml.safe_dump(template_doc))
    record = {
        "shared_template": {"path": str(template), "sha256": PC._document_digest(template_doc)},
        "facts": {"target": target, "sha256": "a" * 64},
        "phase": {"category": "_perf", "label": "dev",
                  "included_in_functional_grade": False},
        "families": [{"family": "PK", "claim": "PREDICTS", "fit_axes": ["K"]}],
        "counts": {"declared_families": 1, "generated_families": 1,
                   "generated_members": 1,
                   "by_family": {"PK": {"admitted_members": 1, "written_members": 1}}},
        "skipped_inapplicable": [], "blocked_unimplemented": [], "errors": [],
    }
    manifest = {"generated": ["_perf/PK00_k8"], "hand_authored": [],
                "performance_generation": {target: record}}
    (corpus_root / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest))

    class FixtureExperiment:
        capsule_corpus = primary

        def __init__(self, name: str):
            self.target = name

        def graded_roots(self):
            return [self.capsule_corpus]

    return FixtureExperiment(target), capsule, template


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


@pytest.mark.parametrize("field", ("numeric_all_pass", "workflow_conformant"))
def test_converged_functional_summary_without_numeric_workflow_conformance_is_refused(
        tmp_path: Path, field: str) -> None:
    run, digest = _functional_run(tmp_path)
    summary_path = run / "qa_loop_summary.yaml"
    summary = yaml.safe_load(summary_path.read_text())
    summary[field] = False
    summary_path.write_text(yaml.safe_dump(summary))
    with pytest.raises(PC.CampaignGateError, match="numeric and workflow conformance"):
        PC.inspect_functional_run(tmp_path, run.name, digest)


def test_functional_host_lane_is_reconstructed_only_from_the_run_snapshot(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, digest = _functional_run(tmp_path)
    live_decoy = tmp_path / "live-repo"
    live_decoy.mkdir()
    (live_decoy / "changed-after-launch.txt").write_text("not snapshotted\n")
    monkeypatch.setattr(PC, "repo_root", lambda: live_decoy)
    record = PC.inspect_functional_run(tmp_path, run.name, digest)
    assert record.model_host_package == (
        Path(record.bundle_input_snapshot["path"]) / "repo"
        / record.model_host_lane_snapshot["package"])
    assert record.model_host_lane_snapshot["package_sha256"] == hash_tree(
        record.model_host_package)["sha256"]

    # Sentinel discovery maps the target descriptor's repository-relative corpus into the
    # immutable functional snapshot.  Restore the real repository path after proving that host-lane
    # admission itself did not consult it.
    monkeypatch.undo()

    from merlin.targetgen.target_experiment import load_target_experiment
    descriptor = (repo_root() / "merlin/experiments/capsule_bench/targets/gemmini"
                  / "target_experiment.yaml")
    sentinel = PC.select_full_model_sentinel(record, load_target_experiment(descriptor))
    # The SMALLEST public L2/L3 model spanning both lanes, which is now the host-island seam capsule
    # rather than the micro-model: M3 exists to be exactly that shape (three tiles of arithmetic around
    # one host island) and became eligible once it declared the lane contract it had always been
    # testing implicitly. Asserting the cheapest-wins PROPERTY as well as the name, so a future capsule
    # that is smaller still is accepted rather than read as a regression.
    assert sentinel.capsule == "M3_host_island_seam_gemmini"
    snapshot_corpus = sentinel.source_dir.parent
    others = [c for c in sorted(snapshot_corpus.iterdir())
              if c.is_dir() and c.name != sentinel.capsule]
    assert others, "a one-candidate corpus would make the cheapest-wins rule vacuous"
    assert all(sentinel.n_bytes <= sum(f.stat().st_size for f in o.rglob("*") if f.is_file())
               for o in others), "the sentinel must be the cheapest qualifying model"
    assert sentinel.source_dir.is_relative_to(
        Path(record.bundle_input_snapshot["path"]) / "repo")


def test_measurement_contract_resolves_only_through_the_authenticated_frozen_grant(
        tmp_path: Path) -> None:
    run, digest = _functional_run(tmp_path)
    functional = PC.inspect_functional_run(tmp_path, run.name, digest)
    marker = Path(functional.bundle_input_snapshot["path"]) / "snapshot.json"
    marker_sha = PC._sha256_bytes(marker.read_bytes())
    contract = PC.frozen_bundle_grant_path(
        marker, marker_sha, repo_root() / "merlin/contract", label="measurement contract")
    assert contract == (Path(functional.bundle_input_snapshot["path"])
                        / "repo/merlin/contract").resolve()
    assert not (contract.stat().st_mode & 0o222)
    with pytest.raises(PC.CampaignGateError, match="absent|ambiguously"):
        PC.frozen_bundle_grant_path(
            marker, marker_sha, repo_root() / "not-granted", label="measurement contract")


def test_full_model_sentinel_is_a_candidate_l2_l3_admission_not_a_timing_row(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    package = tmp_path / "candidate"
    package.mkdir()
    sentinel_dir = tmp_path / "frozen-public" / "M2"
    sentinel_dir.mkdir(parents=True)
    sentinel = PC.FullModelSentinel(
        "M2", sentinel_dir, {"kind": "model"}, "a" * 64, 2, 180_000)
    workspace = tmp_path / "admission"
    workspace.mkdir()
    contract = tmp_path / "frozen-contract"
    contract.mkdir()
    target = type("Target", (), {"target": "gemmini"})()
    observed: dict = {}

    monkeypatch.setattr(runner.PC, "package_sandbox_policy",
                        lambda *args: object())
    monkeypatch.setattr(runner.PC, "boxed_entrypoints",
                        lambda _policy: contextlib.nullcontext())
    monkeypatch.setattr(runner.CR, "load_capsule",
                        lambda source, contract: {"name": "M2", "kind": "model",
                                                  "required_oracle_tiers": ["L0", "L1", "L2", "L3"]})

    def fake_grade(capsule, candidate, **kwargs):
        observed.update(capsule=capsule, candidate=candidate, kwargs=kwargs)
        # A whole model emits exactly ONE execution tier -- the target's citable RTL tier.  The
        # fixture used to hand back an L2 entry as well, which no model path can produce.
        return {"status": "pass", "numeric": {"status": "pass"},
                "tiers": {"L3": {"status": "pass", "cycles": 11}}}

    monkeypatch.setattr(runner.CR, "run_capsule", fake_grade)
    monkeypatch.setattr(runner, "_verify_frozen_contract", lambda *_args: None)
    evidence = runner.run_full_model_admission(
        package, sentinel, workspace, 90, target, contract, "b" * 64)
    assert observed["candidate"] == str(package)
    assert observed["capsule"]["required_oracle_tiers"] == ["L3"]
    assert observed["kwargs"]["contract"] == str(contract)
    assert evidence["passed"] is True
    assert evidence["cycles_recorded"] is False
    assert "cycles" not in evidence
    assert evidence["role"] == "correctness_admission_not_performance_claim"


def test_frozen_contract_drift_is_refused_before_a_measurement_cell(tmp_path: Path) -> None:
    runner = _load_runner()
    contract = tmp_path / "contract"
    contract.mkdir()
    schema = contract / "schema.yaml"
    schema.write_text("version: 1\n")
    expected = PC._exact_tree_record(contract)["sha256"]
    runner._verify_frozen_contract(contract, expected)
    schema.write_text("version: 2\n")
    with pytest.raises(runner.PC.CampaignGateError, match="differ"):
        runner._verify_frozen_contract(contract, expected)


def test_claim_bearing_launch_requires_ready_preflight_and_smoke_is_explicit() -> None:
    runner = _load_runner()
    with pytest.raises(runner.PC.CampaignGateError, match="preflight refused"):
        runner.measurement_mode("claim")
    assert runner.measurement_mode("measurement-smoke") == {
        "experiment_mode": "measurement_smoke_only",
        "claim_launch_status": "NOT_REQUESTED",
        "claim_launch_blocker": runner.PC.SMOKE_CLAIM_NONCLAIM,
        "claim_preflight": None,
    }
    preflight = {
        "schema_version": 1, "family": "PK", "claim": "PREDICTS", "status": "READY",
        "declaration": {}, "cohort": {}, "expected_identities": [], "refusal_reasons": [],
    }
    assert runner.measurement_mode("claim", preflight) == {
        "experiment_mode": "formal_claim", "claim_launch_status": "GO",
        "claim_launch_blocker": None, "claim_preflight": preflight,
    }


@pytest.mark.parametrize("failure", ("escape", "digest", "run_snapshot"))
def test_functional_host_lane_path_digest_and_snapshot_mismatch_fail_closed(
        tmp_path: Path, failure: str) -> None:
    run, digest = _functional_run(tmp_path)
    environment = yaml.safe_load((run / "environment.yaml").read_text())
    if failure == "escape":
        environment["model_host_lane_snapshot"]["package"] = "../outside"
    elif failure == "digest":
        environment["model_host_lane_snapshot"]["package_sha256"] = "0" * 64
    else:
        environment["model_host_lane_snapshot"]["run_snapshot"]["content_sha256"] = "0" * 64
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    with pytest.raises(PC.CampaignGateError, match="host-lane|host lane"):
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
        PC.completion_counts([], ())

    expected = (
        PC.PerfCellIdentity("PK", "PK00", "spike", "r000"),
        PC.PerfCellIdentity("PK", "PK00", "verilator", "r000"),
    )
    complete = [
        {"identity": expected[0].as_dict(), "tier": "L2", "purpose": "correctness_screen",
         "citable": False, "correct": True, "tier_status": "pass", "grade_status": "pass",
         "numeric_status": "pass", "cycles": None},
        {"identity": expected[1].as_dict(), "tier": "L3",
         "purpose": "performance_certification", "citable": True,
         "correct": True, "tier_status": "pass", "grade_status": "pass",
         "numeric_status": "pass", "cycles": 9},
    ]
    counts = PC.completion_counts(complete, expected)
    assert counts == {"expected": 2, "reported": 2, "correct": 2, "failed": 0,
                      "missing": 0, "screen_expected": 1, "screen_passed": 1,
                      "citable_expected": 1, "citable_measured": 1,
                      "citable_passed": 1, "complete": True}

    broken = complete[:1]
    assert PC.completion_report(broken, expected) == {
        "expected": 2, "reported": 1, "correct": 1, "failed": 0, "missing": 1,
        "screen_expected": 1, "screen_passed": 1, "citable_expected": 1,
        "citable_measured": 0, "citable_passed": 0, "complete": False,
    }
    with pytest.raises(PC.CampaignGateError, match="1 of 2 expected"):
        PC.completion_counts(broken, expected)


@pytest.mark.parametrize(
    ("field", "value"),
    (("grade_status", "fail"), ("numeric_status", "fail"), ("error", "runner failed"),
     ("failure", {"plane": "numeric"}), ("cycles", 9.5)),
)
def test_completion_refuses_evidence_the_reporter_cannot_accept(field: str, value) -> None:
    expected = (
        PC.PerfCellIdentity("PK", "PK00", "spike", "r000"),
        PC.PerfCellIdentity("PK", "PK00", "verilator", "r000"),
    )
    common = {"correct": True, "tier_status": "pass", "grade_status": "pass",
              "numeric_status": "pass"}
    rows = [
        {**common, "identity": expected[0].as_dict(), "tier": "L2",
         "purpose": "correctness_screen", "citable": False, "cycles": None},
        {**common, "identity": expected[1].as_dict(), "tier": "L3",
         "purpose": "performance_certification", "citable": True, "cycles": 9},
    ]
    rows[1][field] = value
    report = PC.completion_report(rows, expected)
    assert report["complete"] is False and report["failed"] == 1
    with pytest.raises(PC.CampaignGateError, match="1 reported cell"):
        PC.completion_counts(rows, expected)


def test_generated_performance_corpus_is_descriptor_derived_and_frozen_exactly(
        tmp_path: Path) -> None:
    te, source, _ = _performance_corpus(tmp_path)
    corpus = PC.discover_performance_corpus(te)
    assert corpus.phase_root == source.parent.resolve()
    assert corpus.phase_root not in {Path(path).resolve() for path in te.graded_roots()}
    assert [(row.family, row.capsule) for row in corpus.capsules] == [("PK", "PK00_k8")]

    frozen = PC.freeze_performance_corpus(corpus, tmp_path / "run" / "_frozen_workload")
    verified = PC.verify_frozen_performance_corpus(frozen)
    assert verified["verified"] is True and verified["capsules"] == 1
    manifest = json.loads(frozen.manifest_path.read_text())
    assert manifest["capsules"][0]["source_sha256"] == manifest["capsules"][0]["snapshot_sha256"]
    assert manifest["capsules_sha256"] == frozen.capsules_sha256
    assert not (frozen.manifest_path.stat().st_mode & 0o222)

    copied = frozen.capsules[0].source_dir / "capsule.interface.mlir"
    copied.chmod(0o644)
    copied.write_text("module { func.func @changed() }\n")
    with pytest.raises(PC.CampaignGateError, match="bytes changed"):
        PC.verify_frozen_performance_corpus(frozen)


def test_frozen_candidate_corpus_load_ignores_live_source_drift_and_checks_digests(
        tmp_path: Path) -> None:
    target_experiment, source, _template = _performance_corpus(tmp_path)
    discovered = PC.discover_performance_corpus(target_experiment)
    frozen = PC.freeze_performance_corpus(discovered, tmp_path / "stage" / "_frozen_corpus")
    loaded = PC.load_frozen_performance_corpus(
        frozen.root, manifest_sha256=frozen.manifest_sha256,
        capsules_sha256=frozen.capsules_sha256, expected_target=target_experiment.target)
    source.joinpath("capsule.interface.mlir").write_text("module { func.func @live_changed() }\n")
    loaded_again = PC.load_frozen_performance_corpus(
        frozen.root, manifest_sha256=frozen.manifest_sha256,
        capsules_sha256=frozen.capsules_sha256, expected_target=target_experiment.target)
    assert loaded_again.capsules[0].source_sha256 == loaded.capsules[0].source_sha256
    with pytest.raises(PC.CampaignGateError, match="digest|manifest"):
        PC.load_frozen_performance_corpus(
            frozen.root, manifest_sha256="0" * 64,
            capsules_sha256=frozen.capsules_sha256, expected_target=target_experiment.target)


@pytest.mark.parametrize("failure", (None, "functional", "corpus"))
def test_measurement_handoff_crosschecks_functional_and_frozen_corpus_before_running(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None) -> None:
    runner = _load_runner()
    run, digest = _functional_run(tmp_path / "functional")
    functional = PC.inspect_functional_run(tmp_path / "functional", run.name, digest)
    discovered_te, _capsule, _template = _performance_corpus(tmp_path / "live-corpus")
    frozen = PC.freeze_performance_corpus(
        PC.discover_performance_corpus(discovered_te), tmp_path / "stage/frozen-corpus")
    loaded = runner.PC.load_frozen_performance_corpus(
        frozen.root, manifest_sha256=frozen.manifest_sha256,
        capsules_sha256=frozen.capsules_sha256, expected_target="fixture")
    candidate = tmp_path / "stage/candidate"
    base = tmp_path / "stage/base"
    runner.PC.materialize_readonly_tree(functional.submission_dir, candidate)
    runner.PC.materialize_readonly_tree(functional.submission_dir, base)
    record = tmp_path / "stage/performance_candidate.json"
    record.write_text("{}\n")
    record.chmod(0o444)
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text("target: fixture\n")
    descriptor_sha = runner._file_sha256(descriptor)
    bundle_manifest = Path(functional.bundle_input_snapshot["path"]) / "snapshot.json"
    sentinel_source = (Path(functional.bundle_input_snapshot["path"])
                       / "repo/merlin/contract/capsules/model/M2_microvit_gemmini")
    sentinel_tree = runner.PC._exact_tree_record(sentinel_source)
    expected_cells = tuple(
        identity.as_dict() for identity in runner.PC.expected_perf_cells(loaded.capsules, 1))
    handoff = SimpleNamespace(
        functional_run_id=functional.run_id,
        functional_submission_sha256=("0" * 64 if failure == "functional" else digest),
        candidate_path=candidate, candidate_sha256=digest,
        candidate_initial_sha256=digest, functional_base_path=base,
        target_descriptor=descriptor, target_descriptor_sha256=descriptor_sha,
        functional_bundle_snapshot_sha256=functional.bundle_input_snapshot["content_sha256"],
        functional_bundle_manifest=bundle_manifest,
        functional_bundle_manifest_sha256=runner._file_sha256(bundle_manifest),
        host_lane=dict(functional.model_host_lane_snapshot), corpus_root=frozen.root,
        corpus_manifest=frozen.manifest_path,
        corpus_manifest_sha256=("0" * 64 if failure == "corpus"
                                else frozen.manifest_sha256),
        corpus_sha256=frozen.capsules_sha256, replicates=1,
        expected_cells=expected_cells, families=({"family": "PK"},),
        e2e_sentinel={
            "capsule": "M2_microvit_gemmini", "frozen_source_path": str(sentinel_source),
            "capsule_sha256": sentinel_tree["sha256"],
            "required_lanes": ["on_mesh", "scalar_rvv_lane"],
            "required_tiers": ["L2", "L3"],
        },
    )
    import perf_agent_stage as agent_stage
    monkeypatch.setattr(agent_stage, "verify_candidate_handoff", lambda *args, **kwargs: handoff)
    target = SimpleNamespace(target="fixture", path=descriptor)
    if failure is not None:
        with pytest.raises(runner.PC.CampaignGateError, match="functional|manifest"):
            runner.load_measurement_candidate(record, functional, target)
    else:
        admitted = runner.load_measurement_candidate(record, functional, target)
        assert admitted.package == candidate.resolve()
        assert admitted.corpus.manifest_sha256 == frozen.manifest_sha256
        assert admitted.contract_root.is_relative_to(
            Path(functional.bundle_input_snapshot["path"]) / "repo")


@pytest.mark.parametrize("failure", ("manual", "nonperformance", "foreign", "stale", "empty"))
def test_generated_performance_discovery_rejects_untrusted_or_stale_corpora(
        tmp_path: Path, failure: str) -> None:
    te, capsule, template = _performance_corpus(tmp_path)
    manifest_path = capsule.parent.parent / "MANIFEST.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    if failure == "manual":
        manifest["generated"] = []
        manifest["hand_authored"] = ["_perf/PK00_k8"]
        manifest_path.write_text(yaml.safe_dump(manifest))
    elif failure == "nonperformance":
        descriptor = yaml.safe_load((capsule / "capsule.yaml").read_text())
        descriptor.pop("performance")
        (capsule / "capsule.yaml").write_text(yaml.safe_dump(descriptor))
    elif failure == "foreign":
        manifest["performance_generation"][te.target]["facts"]["target"] = "someone_else"
        manifest_path.write_text(yaml.safe_dump(manifest))
    elif failure == "stale":
        template.write_text(yaml.safe_dump({"sweeps": [{"id": "changed"}]}))
    else:
        for path in capsule.iterdir():
            path.unlink()
        capsule.rmdir()
        manifest["generated"] = []
        counts = manifest["performance_generation"][te.target]["counts"]
        counts.update({"generated_families": 0, "generated_members": 0})
        counts["by_family"]["PK"] = {"admitted_members": 0, "written_members": 0}
        manifest_path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(PC.CampaignGateError):
        PC.discover_performance_corpus(te)


def test_pk_discovery_requires_the_frozen_quantitative_acceptance_contract(
        tmp_path: Path) -> None:
    target, capsule, _template = _performance_corpus(tmp_path)
    descriptor_path = capsule / "capsule.yaml"
    descriptor = yaml.safe_load(descriptor_path.read_text())
    descriptor["performance"].pop("acceptance")
    descriptor_path.write_text(yaml.safe_dump(descriptor))
    with pytest.raises(PC.CampaignGateError, match="acceptance"):
        PC.discover_performance_corpus(target)


def test_a_differential_family_is_canonical_without_an_acceptance_contract() -> None:
    """An acceptance block freezes the decision rule for a FIT, and a differential fits nothing.

    Its verdict is its falsifier over an A/B on identical work; there is no coefficient for a
    threshold to bound. Demanding one of every family refused the whole corpus on the first
    differential capsule, and the only way to satisfy that demand would have been to invent six
    thresholds nobody measured -- which is precisely what the frozen-contract discipline forbids.
    """
    block = _performance_block(family="PC")
    block["claim"] = "DIFFERENTIAL"
    block.pop("acceptance")

    assert PC._validate_performance_block(block, owner="fixture") is block


def test_a_predicts_family_without_an_acceptance_contract_is_still_refused() -> None:
    block = _performance_block()
    block.pop("acceptance")

    with pytest.raises(PC.CampaignGateError, match="acceptance"):
        PC._validate_performance_block(block, owner="fixture")


def test_an_acceptance_that_is_not_a_mapping_is_refused_whatever_the_claim() -> None:
    """Absent is canonical; present-and-malformed never is, or the contract is unreadable."""
    block = _performance_block(family="PC")
    block["claim"] = "DIFFERENTIAL"
    block["acceptance"] = "frozen"

    with pytest.raises(PC.CampaignGateError, match="frozen mapping"):
        PC._validate_performance_block(block, owner="fixture")


def test_every_family_the_shipped_template_declares_passes_this_gate() -> None:
    """The regression itself: the gate refused declarations the generator is allowed to write.

    Read from the one shared template rather than from a fixture, because the disagreement was
    between two real ends of the chain -- `generate_corpus` has never required `acceptance`, and this
    consumer did -- and only the real declarations can show whether they still agree.
    """
    from merlin.common.paths import merlin_dir

    template = merlin_dir() / "contract" / "capsules" / "profiles" / "_perf.yaml"
    document = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    blocks = [sweep["base"]["performance"] for sweep in (document.get("sweeps") or [])]
    blocks += [row["performance"] for row in (document.get("blocked_unimplemented") or [])]
    assert blocks, "the shared performance template declares no families"

    for block in blocks:
        PC._validate_performance_block(block, owner=str(block.get("family")))


def test_results_are_canonical_read_only_and_digest_verified(tmp_path: Path) -> None:
    path = tmp_path / "perf_results.json"
    rows = [{"identity": {"family": "PK", "capsule": "PK00",
                           "simulator": "verilator", "replicate": "r000"},
             "cycles": 17}]
    record = PC.write_immutable_json(path, rows)
    assert PC.verify_immutable_json(path, record["sha256"]) == rows
    assert not (path.stat().st_mode & 0o222)

    path.chmod(0o644)
    path.write_text("[]\n")
    path.chmod(0o444)
    with pytest.raises(PC.CampaignGateError, match="digest mismatch"):
        PC.verify_immutable_json(path, record["sha256"])


def test_package_sandbox_is_answer_closed_credential_free_and_submission_read_only(
        tmp_path: Path) -> None:
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = (repo_root() / "merlin/experiments/capsule_bench/targets/gemmini"
                  / "target_experiment.yaml")
    te = load_target_experiment(descriptor)
    ws = tmp_path / "workspace"
    pkg = ws / "submission"
    pkg.mkdir(parents=True)
    pkg.chmod(0o555)
    policy = PC.package_sandbox_policy(te, ws, pkg)
    assert policy.coverage_gap == ()
    assert "--unshare-net" not in policy.argv
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
    tool.chmod(0o444)
    pkg_dir.chmod(0o555)
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
        oot_runner.build_package(untrusted_build)
    assert result.returncode == 0
    assert len(calls) == len(policy.required_tools) + 2
    entry_call = calls[-2]
    assert entry_call[0] == "bwrap"
    assert "--unshare-net" not in entry_call and "--clearenv" in entry_call
    assert "perf-package" in entry_call
    assert str(policy.execution_package / "tool.py") in entry_call
    assert str(tool) not in entry_call
    assert calls[-1][0] == "bwrap" and "perf-build" in calls[-1]
    assert str(policy.execution_package) in calls[-1]


def test_cpp_build_runs_only_in_the_per_cell_bwrap_copy_and_failure_is_preserved(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from merlin.targetgen import oot_runner
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor = (repo_root() / "merlin/experiments/capsule_bench/targets/gemmini"
                  / "target_experiment.yaml")
    te = load_target_experiment(descriptor)
    source = tmp_path / "sealed" / "submission"
    source.mkdir(parents=True)
    tool = source / "build" / "target-opt"
    manifest = {
        "language": "cpp",
        "build": {"configure": ["cmake", "-S", ".", "-B", "build"],
                  "command": ["cmake", "--build", "build"],
                  "tool_output": "build/target-opt"},
        "commands": {"parse": {"argv": ["{tool}", "{input_mlir}"]}},
    }
    (source / "manifest.yaml").write_text("fixture\n")
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    workspace = tmp_path / "cell"
    workspace.mkdir()
    policy = PC.package_sandbox_policy(te, workspace, source)
    source_package = oot_runner.Package(source, manifest, tool)
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(PC.subprocess, "run", fake_run)
    with PC.boxed_entrypoints(policy):
        oot_runner.build_package(source_package)
    assert len(calls) == 2
    assert all(call[0] == "bwrap" and "perf-build" in call for call in calls)
    assert all(str(policy.execution_package) in call for call in calls)
    assert hash_tree(source)["sha256"] == policy.package_sha256

    def failed_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 17, stdout="", stderr="compiler refused")

    monkeypatch.setattr(PC.subprocess, "run", failed_run)
    with PC.boxed_entrypoints(policy):
        with pytest.raises(oot_runner.CertFailure, match="rc=17"):
            oot_runner.build_package(source_package)

    outside = oot_runner.Package(tmp_path / "other", manifest, tmp_path / "other/tool")
    with PC.boxed_entrypoints(policy):
        with pytest.raises(PC.CampaignGateError, match="outside the sealed candidate"):
            oot_runner.build_package(outside)


def test_real_bwrap_builds_and_executes_only_the_writable_per_cell_copy(
        tmp_path: Path) -> None:
    """Exercise the real subprocess boundary, not only its argv construction."""
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this host")
    from merlin.targetgen import oot_runner
    from merlin.targetgen.target_experiment import load_target_experiment

    target = load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/gemmini"
        / "target_experiment.yaml")
    source = tmp_path / "sealed" / "submission"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text("target: gemmini\nlanguage: cpp\n")
    tool = source / "build" / "fixture-opt"
    script = ("#!/usr/bin/env python3\nimport pathlib,sys\n"
              "print(pathlib.Path(sys.argv[1]).read_text().strip())\n")
    manifest = {
        "target": "gemmini", "language": "cpp",
        "build": {
            "command": [
                "python3", "-c",
                ("from pathlib import Path; p=Path('build/fixture-opt'); "
                 f"p.parent.mkdir(exist_ok=True); p.write_text({script!r}); p.chmod(0o755)"),
            ],
            "tool_output": "build/fixture-opt",
        },
        "commands": {"parse": {"argv": ["{tool}", "{input_mlir}"]}},
    }
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    source_before = hash_tree(source)["sha256"]
    workspace = tmp_path / "cell"
    input_mlir = workspace / "generated" / "input.interface.mlir"
    input_mlir.parent.mkdir(parents=True)
    input_mlir.write_text("module {}\n")
    policy = PC.package_sandbox_policy(target, workspace, source)
    package = oot_runner.Package(source, manifest, tool)

    with PC.boxed_entrypoints(policy):
        oot_runner.build_package(package)
        result = oot_runner.run_entrypoint(package, "parse", input_mlir)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "module {}"
    assert (policy.execution_package / "build/fixture-opt").is_file()
    assert not tool.exists()
    assert hash_tree(source)["sha256"] == source_before == policy.package_sha256


def test_actual_spawn_workers_preserve_predeclared_task_and_row_order(tmp_path: Path) -> None:
    """Exercise process isolation with deterministic map ordering on the fail-closed cell path."""
    import run_perf_bench as runner

    missing_package = tmp_path / "absent-sealed-candidate"
    contract = tmp_path / "contract"
    contract.mkdir()
    (contract / "schema.yaml").write_text("version: 1\n")
    contract_sha = PC._exact_tree_record(contract)["sha256"]
    target = SimpleNamespace(target="fixture")
    tasks = []
    for capsule in ("PK00_k16", "PK01_k32"):
        workspace = tmp_path / capsule
        workspace.mkdir()
        member = SimpleNamespace(family="PK", capsule=capsule)
        tasks.append((missing_package, member, "r000", workspace, 1, target,
                      contract, contract_sha))

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(runner._run_cell_task, tasks))

    assert [[row["identity"] for row in pair] for pair in results] == [
        [
            {"family": "PK", "capsule": capsule,
             "simulator": simulator, "replicate": "r000"}
            for simulator in ("spike", "verilator")
        ]
        for capsule in ("PK00_k16", "PK01_k32")
    ]
    assert all("CampaignGateError" in row["error"] for pair in results for row in pair)


def test_perf_runner_has_no_latest_or_mtime_submission_selection() -> None:
    source = (repo_root() / "merlin/experiments/gemmini_perf_bench/scripts/run_perf_bench.py").read_text()
    assert "_latest_submission" not in source
    assert "st_mtime" not in source
    assert "PB.KERNELS" not in source
    assert "kernel_corpus.yaml" not in source
    assert "--functional-run-id" in source
    assert "--functional-submission-sha256" in source
    assert "--candidate-record" in source and "--replicates" in source and "--workers" in source
    assert "discover_performance_corpus" not in source and "freeze_performance_corpus" not in source
    assert "_CONTRACT" not in source
    assert "select_full_model_sentinel" not in source
