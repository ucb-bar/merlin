"""Real formal receipts reach Phase 2; synthetic observations are NOT OS/RTL proof.

Only external engine results, source revision observation, and prior authoring /
sandbox observations are fixture data. Grading, snapshot verification, freeze,
formal manifest publication and both Phase-2 admission layers remain real.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.context import load_context
from merlin_experiments.phase1.feedback import formal, freeze
from merlin_experiments.phase2 import campaign
from merlin_experiments.phase2.global_inputs import FrozenPhase1

from merlin.benchharness import hash_tree
from merlin.common import artifacts, provenance
from merlin.common.paths import data_path
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.sandbox import bwrap as BW


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    return build_handoff(tmp_path, monkeypatch)


def build_handoff(
    tmp_path, monkeypatch, *, reviewed=None, authored_submission=None, post_freeze_failure=False, execution=None
):
    """Join real frozen inputs and formal receipts; external oracle observations stay synthetic."""
    # The explicit legacy context updates environment during formal.main.
    monkeypatch.setattr(os, "environ", os.environ.copy())

    def forbidden(*args, **kwargs):
        pytest.fail("formal handoff must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    spec = importlib.util.spec_from_file_location(
        "formal_handoff_inputs", Path(__file__).with_name("phase1_feedback_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    seed_root = tmp_path if reviewed is None else tmp_path / "synthetic-submission-seed"
    resources = (
        {}
        if reviewed is None
        else {
            "contract_root": Path(reviewed.fixture["environment"]["MERLIN_CONTRACT_DIR"]),
            "schemas_root": Path(reviewed.fixture["environment"]["MERLIN_SCHEMAS_DIR"]),
        }
    )
    workspace, public, descriptor, _, env, _ = helper.inputs(seed_root, **resources)
    for name in ("MERLIN_OUT_ROOT", "TMPDIR"):
        monkeypatch.setenv(name, env[name])
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "public-cas"))
    monkeypatch.setenv("MERLIN_MESH_SIM", "synthetic")
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_EXPERIMENT", raising=False)
    repo = tmp_path
    contract = data_path("contract")
    if reviewed is None:
        hidden = tmp_path / "hidden/H"
        hidden.mkdir(parents=True)
        capsule = json.loads((public / "A/capsule.yaml").read_text())
        capsule.update(name="H", label="hidden")
        (hidden / "capsule.yaml").write_text(json.dumps(capsule))
        bundle = {
            "bundle_id": "merlin_assisted_rtlchecks_synthetic_handoff",
            "allowed": [{"path": str(public)}],
            "denied": [],
            "host_inputs": [{"path": str(hidden.parent)}],
        }
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
        frozen_public, frozen_hidden = BW.snapshot_input_paths(workspace, bundle, [public, hidden.parent], repo=repo)
        public_count = 1
    else:
        seed_submission = workspace / "submission"
        seed_manifest = seed_submission / "manifest.yaml"
        if authored_submission is None:
            seed_document = json.loads(seed_manifest.read_text())
            seed_document["target"] = reviewed.target.target
            seed_manifest.write_text(json.dumps(seed_document))
        else:
            assert (authored_submission / "manifest.yaml").is_file()
            assert (authored_submission / "compiler.py").is_file()
            shutil.copytree(authored_submission, seed_submission, dirs_exist_ok=True)
            assert hash_tree(seed_submission) == hash_tree(authored_submission)
        workspace = reviewed.workspace
        shutil.copytree(seed_submission, workspace / "submission")
        descriptor = reviewed.descriptor
        repo = reviewed.fixture["workspace"]
        contract = Path(reviewed.fixture["environment"]["MERLIN_CONTRACT_DIR"])
        bundle = reviewed.prepared.bundle
        frozen_public = reviewed.view.public
        hidden_roots = reviewed.target.hidden_roots()
        assert len(hidden_roots) == 1
        (frozen_hidden,) = BW.snapshot_input_paths(workspace, bundle, hidden_roots, repo=repo)
        public_count = 2
    snapshot = BW.snapshot_record(workspace)
    assert snapshot["version"] == 4
    runs = tmp_path / "runs"
    run = runs / "merlin_assisted" / "formal-handoff"
    run.mkdir(parents=True)
    shutil.copytree(workspace / "submission", run / "submission")
    manifest = run / "input_bundle_manifest.yaml"
    if reviewed is None:
        manifest.write_text(yaml.safe_dump(bundle))
    else:
        shutil.copyfile(reviewed.run / "input_bundle_manifest.yaml", manifest)
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == reviewed.prepared.effective_sha256
    environment = {
        "run_id": run.name,
        "bundle_id": bundle["bundle_id"],
        "bundle_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "workspace_path": str(workspace),
        "bundle_input_snapshot": snapshot,
        "sandbox": "bwrap",
        "isolation_violations": [],
        "golden_mask_selftest": {"leaked_answer_files": [], "n_answer_files_masked": 1},
        "task_scope": {"required_public_dev_capsules": public_count, "held_out_capsules": 1},
        "fixture_observations": "synthetic external sandbox/authoring; no OS isolation claim",
    }
    if reviewed is not None:
        environment.update(
            authored_bundle_manifest_sha256=reviewed.prepared.authored_sha256,
            bundle_manifest_sha256=reviewed.prepared.effective_sha256,
            public_corpus_input=reviewed.prepared.corpus_record,
        )
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    (run / "qa_loop_summary.yaml").write_text(
        yaml.safe_dump(
            {
                "converged": True,
                "rounds": [{"answer_access_clean": True, "audit_hits": []}],
                "finalize": {"answer_access_clean": True, "audit_hits": [], "regrade_all_pass": True},
                "fixture_observations": "synthetic authoring observations, not an executed agent",
            }
        )
    )
    monkeypatch.setattr(CR, "_config_for_target", lambda *args: SimpleNamespace(rtl_tiers={"L3"}))
    monkeypatch.setattr(CR, "_rtl_tiers_of", lambda target: {"L3"})
    monkeypatch.setattr(
        CR,
        "describe_l3_engine",
        lambda target: {"available": True, "engine": "fixture", "fidelity": "elaborated_rtl"},
    )
    monkeypatch.setattr(CR, "oracle_adapters", lambda *args: {"L0": object(), "L2": object(), "L3": object()})
    monkeypatch.setattr(CR, "suite_for", lambda target: target + "-capsule-bench")
    monkeypatch.setattr(provenance, "load_pins", lambda: {})
    monkeypatch.setattr(provenance, "_git", lambda *args: None)
    monkeypatch.setattr(artifacts, "git_sha7", lambda: "fixture")
    monkeypatch.setattr(freeze, "repo_sha", lambda **kwargs: "synthetic-source-observation")
    events = []

    def external_execution(caps, package_dir, *, runs_root, oracle_adapters, target, **kwargs):
        is_hidden = {cap["label"] for cap in caps} == {"hidden"}
        assert (run / "freeze.json").exists() == is_hidden
        events.append("hidden" if is_hidden else "public")
        rows = []
        for cap in caps:
            row = {
                "capsule": cap["name"],
                "label": cap["label"],
                "kind": "isa",
                "status": "pass",
                "tiers": {name: {"status": "pass", "derived_from_rtl": name == "L3"} for name in oracle_adapters},
                "highest_tier": "L3",
                "numeric": {"status": "pass", "mismatch_count": 0},
                "trace_check": {"status": "pass", "violations": []},
            }
            result = Path(runs_root) / "runs" / CR.suite_for(target) / cap["name"]
            result.mkdir(parents=True)
            (result / "capsule_result.json").write_text(json.dumps(row))
            rows.append(row)
        return rows

    monkeypatch.setattr(CR, "run_suite", external_execution)
    # This fixture exercises the formal->Phase-2 handoff, not a compiler build.
    # Supply an explicit synthetic external gate boundary just as the capsule
    # oracle above is synthetic; production PFM.run is never replaced.
    from merlin_experiments.phase1.feedback import private_full_models as PFM

    from merlin.compile.model_execution_inputs import strict_tree_sha256

    private_spec = tmp_path / "synthetic-private-spec.yaml"
    target = load_context(descriptor, repo=repo).target
    firrtl = tmp_path / "selected.fir"
    firrtl.write_text("synthetic FIRRTL input\n")
    firrtl_sha = hashlib.sha256(firrtl.read_bytes()).hexdigest()
    raw_facts = tmp_path / "raw-facts.json"
    facts = {
        "inputs": {
            "target": target,
            "fir_sha256": firrtl_sha,
            "firrtl_inputs": [{"path": str(firrtl), "sha256": firrtl_sha}],
        },
        "facts": {"source": {"config": "FixtureConfig"}},
    }
    raw_facts.write_text(json.dumps(facts))
    effective_facts = tmp_path / "effective-facts.json"
    effective_facts.write_text(json.dumps(dict(facts, effective_host_view=True)))
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(effective_facts))
    private_spec.write_text(
        yaml.safe_dump(
            {
                "schema": PFM.SCHEMA,
                "target": target,
                "models": [
                    {
                        "id": "fixture_complete_model",
                        "rtl_config": "FixtureConfig",
                        "rtl_facts": str(raw_facts),
                        "rtl_facts_sha256": hashlib.sha256(raw_facts.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(PFM, "requirements_for", lambda _descriptor: ("fixture_complete_model",))
    monkeypatch.setattr(PFM, "program_requirements_for", lambda _descriptor: {"fixture_complete_model": ("model",)})
    monkeypatch.setattr(
        PFM,
        "loader_env_requirements_for",
        lambda _descriptor: {"fixture_complete_model": {"required": {}, "forbidden": ()}},
    )

    linked_with: list = []

    def external_private_gate(submission, _spec, *, target, required_models, required_programs, **_kwargs):
        assert os.environ["MERLIN_RTL_FACTS"] == str(raw_facts)
        linked_with.append(_kwargs.get("build_options"))
        sha = strict_tree_sha256(Path(submission))["sha256"]
        spec_sha = hashlib.sha256(Path(_spec).read_bytes()).hexdigest()
        return {
            "schema": PFM.RESULT_SCHEMA,
            "target": target,
            "passed": True,
            "private_spec_sha256": spec_sha,
            "authored_source_freeze": {
                "schema": "merlin.phase1.private_source_freeze.v1",
                "spec_sha256": spec_sha,
                "record_sha256": "f" * 64,
                "root": str(run / "private_full_model_input" / "sources"),
            },
            "candidate_tree_sha256": sha,
            "required_models": list(required_models),
            "required_programs": {name: list(required_programs[name]) for name in required_models},
            "full_model_numerical_equivalence": "not_run",
            "paper_accuracy": "not_claimed",
            "models": [
                {
                    "model": "fixture_complete_model",
                    "status": "pass",
                    "checks": {
                        "input_provenance": {
                            "model": {
                                "paper_ready": None,
                                "synthetic_inputs": None,
                                "meta_sha256": "a" * 64,
                                "scope": "input provenance only; no paper accuracy or full-model numerical result",
                            }
                        },
                        "build": {
                            "status": "capture_lower_codegen_link_verified",
                            "candidate_tree_sha256": sha,
                            "linked_device_groups": 1,
                            "programs": [
                                {
                                    "program": "model",
                                    "status": "capture_lower_codegen_link_verified",
                                    "candidate_tree_sha256": sha,
                                    "elf_sha256": "e" * 64,
                                    "linked_device_groups": 1,
                                    "static_host_compute_audit": [
                                        {
                                            "verdict": "clean_static_host_compute_audit",
                                            "artifact_sha256": "a" * 64,
                                            "object_sha256": "b" * 64,
                                            "audit": {"budget": {}, "groups": [{"verdict": "clean"}]},
                                        }
                                    ],
                                }
                            ],
                        },
                    },
                }
            ],
        }

    monkeypatch.setattr(PFM, "run", external_private_gate)
    # This synthetic handoff fixture replaces the complete external private
    # build boundary, including its separately frozen authored-source record.
    freeze_checks = 0

    def synthetic_source_freeze(*_args, **_kwargs):
        nonlocal freeze_checks
        freeze_checks += 1
        if post_freeze_failure and freeze_checks == 2:
            raise RuntimeError("frozen workspace source changed after linked build")
        return {"fixture": True}

    monkeypatch.setattr(formal, "_private_source_freeze_for_formal", synthetic_source_freeze)
    if execution is not None:
        # The descriptor's execution declaration and the engine run are the external boundary here;
        # formal's wiring (when it runs, with which facts, and what completion it allows) stays real.
        from merlin_experiments.phase1.feedback import private_full_model_execution as PFX

        declared = {
            "required": True,
            "programs": {"fixture_complete_model": {"model": {"engine": "gsim", "timeout_s": 1}}},
        }
        monkeypatch.setattr(PFX, "gate_for", lambda _descriptor, *, required_programs: declared)

        def external_execution(static, gate, *, target, static_out, **_kwargs):
            assert gate is declared and static["passed"] is True
            assert os.environ["MERLIN_RTL_FACTS"] == str(raw_facts)
            assert Path(static_out) == run / "grading_private_full_models"
            execution_calls.append(target)
            return {"schema": PFX.SCHEMA, "passed": execution == "pass", "programs": [], "deferred": []}

        execution_calls: list = []
        monkeypatch.setattr(PFX, "run", external_execution)
    status = formal.main(
        [
            "--run-dir",
            str(run),
            "--arm",
            "merlin_assisted",
            "--capsules",
            str(frozen_public),
            "--hidden-capsules",
            str(frozen_hidden),
            "--contract",
            str(contract),
            "--private-full-model-spec",
            str(private_spec),
        ],
        context=load_context(descriptor, repo=repo),
    )
    assert status == (1 if post_freeze_failure or execution == "fail" else 0)
    if execution is not None:
        assert execution_calls == [target]
        # The static gate links each executed program for the execution gate: every result read back.
        assert linked_with[0] == {"fixture_complete_model": {"model": {"readback": "full", "group_profile": True}}}
    assert os.environ["MERLIN_RTL_FACTS"] == str(effective_facts)
    assert events == ["public", "hidden"]
    digest = hash_tree(run / "submission")["sha256"]
    return SimpleNamespace(run=run, runs=runs, digest=digest, environment=environment, events=events)


def test_formal_receipts_admit_without_mocking_phase2(handoff):
    frozen = campaign.inspect_functional_run(handoff.runs, handoff.run.name, handoff.digest)
    assert not frozen.deviations
    binding = FrozenPhase1(handoff.runs, handoff.run.name, handoff.digest, (), 1, 1, ()).verify(
        handoff.run / "submission"
    )
    assert binding["public_passed"] == binding["public_total"] == 1
    assert len(binding["evidence_sha256"]) == 6


def test_successful_private_build_cannot_survive_postbuild_snapshot_refusal(tmp_path, monkeypatch):
    handoff = build_handoff(tmp_path, monkeypatch, post_freeze_failure=True)
    manifest = yaml.safe_load((handoff.run / "run_manifest.yaml").read_text())
    assert manifest["private_full_models"]["passed"] is False
    assert "changed after linked build" in manifest["private_full_models"]["reason"]
    assert manifest["completion"]["formal_grade_complete"] is False
    assert "private_full_models:incomplete" in manifest["completion"]["failures"]


@pytest.mark.parametrize("execution", ["pass", "fail"])
def test_a_declared_execution_gate_runs_after_the_static_gate_and_holds_completion(tmp_path, monkeypatch, execution):
    handoff = build_handoff(tmp_path, monkeypatch, execution=execution)
    manifest = yaml.safe_load((handoff.run / "run_manifest.yaml").read_text())
    assert manifest["private_full_models"]["passed"] is True
    assert manifest["private_full_model_execution"]["passed"] is (execution == "pass")
    assert manifest["completion"]["formal_grade_complete"] is (execution == "pass")
    assert ("private_full_model_execution:incomplete" in manifest["completion"]["failures"]) is (execution == "fail")


def test_an_undeclared_execution_gate_leaves_the_manifest_unchanged(handoff):
    manifest = yaml.safe_load((handoff.run / "run_manifest.yaml").read_text())
    assert "private_full_model_execution" not in manifest


def test_actual_diagnostic_observations_still_refuse(handoff):
    environment = dict(handoff.environment, sandbox="none", golden_mask_selftest={})
    (handoff.run / "environment.yaml").write_text(yaml.safe_dump(environment))
    with pytest.raises(campaign.CampaignGateError, match="sandbox_not_bwrap.*answer_mask_vacuous"):
        campaign.inspect_functional_run(handoff.runs, handoff.run.name, handoff.digest)


@pytest.mark.parametrize("mutation", ["submission", "score", "snapshot"])
def test_formal_handoff_refuses_changed_evidence(handoff, mutation):
    if mutation == "submission":
        path = handoff.run / "submission/compiler.py"
    elif mutation == "score":
        path = handoff.run / "grading_public/score_capsule.json"
    else:
        path = Path(handoff.environment["bundle_input_snapshot"]["path"]) / "snapshot.json"
    path.chmod(0o600)
    if mutation == "score":
        document = json.loads(path.read_text())
        document["n_passed"] = 0
        path.write_text(json.dumps(document))
    else:
        path.write_bytes(path.read_bytes() + b"\n# altered evidence\n")
    with pytest.raises((campaign.CampaignGateError, ValueError)):
        campaign.inspect_functional_run(handoff.runs, handoff.run.name, handoff.digest)
