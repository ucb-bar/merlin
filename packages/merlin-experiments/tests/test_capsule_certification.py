"""Synthetic grading boundaries exercise real observations, OOT and publication.

These fixtures make no hardware/numerical qualification claim. The external
grader is a declared fixture; receipt custody, freeze and admission stay real.
"""

import ast
import hashlib
import inspect
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import oot_history
from merlin_experiments.phase1.feedback import capsule_certification as C
from merlin_experiments.phase1.feedback import certification as CERT
from merlin_experiments.phase1.feedback import formal, freeze, public_owned_host_selection


def score(package, count, *, functional=False, hidden=False):
    label = "hidden" if hidden else "public"
    result = {
        "package": package,
        "n_capsules": count,
        "n_passed": count,
        "functional_pass": 1,
        "gradeable": True,
        "integrity_status": "clean",
        "integrity_exempt": False,
        "numeric_all_exact": True,
        "trace_all_pass": True,
        "model_execution_all_pass": True,
        "provided_host_execution_all_pass": True,
        "structural_evidence_all_pass": True,
        "structural_evidence_scope": {"n_instruction_trace_capsules": count, "n_model_execution_capsules": 0},
        "highest_tier": "L2" if functional else "L3",
        "tier_reached": {"L2": count, "L3": 0 if functional else count},
        "pass_evidence": {"rtl_backed": 0 if functional else count},
        "cohort_admission": {
            "policy": "all_discovered",
            "n_source_capsules": count,
            "n_admitted_capsules": count,
            "n_capability_excluded": 0,
            "n_resource_excluded": 0,
            "excluded_name_set_sha256": "a" * 64,
            "admitted_name_set_sha256": hashlib.sha256(
                json.dumps(
                    sorted(f"{label}-synthetic-{i}" for i in range(count)),
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "per_capsule": [
            {
                "capsule": f"{label}-synthetic-{i}",
                "label": label,
                "status": "pass",
                "numeric": "pass",
                "tiers": {"L2" if functional else "L3": "pass"},
            }
            for i in range(count)
        ],
        **{
            key: 0
            for key in (
                "n_not_graded_ineligible",
                "n_gated_deferred",
                "n_screened_only",
                "n_budget_exhausted",
                "n_incomplete",
                "n_not_gradeable_no_oracle",
            )
        },
    }
    if functional:
        result["functional_hidden_evidence"] = {
            "tier": "L2",
            "engine": "spike",
            "fidelity": "functional_model",
            "claim": "functional_corroboration",
            "n_admitted_capsules": count,
            "n_spike_passed": count,
            "n_validated_refusals": 0,
            "n_higher_tier_measured": 0,
        }
    return result


@pytest.fixture
def selected(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "submission").mkdir(parents=True)
    (run / "submission/compiler.py").write_text("print('synthetic external compiler fixture')\n")
    (run / "submission/manifest.yaml").write_text("{}\n")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "target": "fixture",
                "grading": {
                    "expected_cohort": {"source_capsules": 2, "admitted_capsules": 2},
                    "hidden_capability_admission": {"source_capsules": 1, "admitted_capsules": 1},
                },
            }
        )
    )
    public, hidden, contract = (tmp_path / name for name in ("public", "hidden", "contract"))
    for root in (public, hidden, contract):
        root.mkdir()
        (root / "input.txt").write_text("selected synthetic input\n")
    environment = {
        "validation_scope": "capsules",
        "arm": "baseline",
        "sandbox": "bwrap",
        "isolation_violations": [],
        "golden_mask_selftest": {"leaked_answer_files": [], "n_answer_files_masked": 1},
        "bundle_input_snapshot": {"version": 4, "n_files": 1, "n_bytes": 1, "content_sha256": "a" * 64},
        "resolved_tools": [],
        "task_scope": {"required_public_dev_capsules": 2, "held_out_capsules": 1},
        "started_at": datetime.now(UTC).isoformat(),
    }
    (run / "environment.yaml").write_text(yaml.safe_dump(environment))
    (run / "input_bundle_manifest.yaml").write_text("{}\n")
    (run / "qa_loop_state.yaml").write_text(
        yaml.safe_dump(
            {
                "cumulative": {"started_at": (datetime.now(UTC) - timedelta(seconds=10)).isoformat()},
            }
        )
    )
    context = SimpleNamespace(target="fixture", descriptor=descriptor, repo=tmp_path, readback_policy=None)
    oot_history.start(run)
    monkeypatch.setattr(freeze, "repo_sha", lambda **kwargs: "synthetic-source-observation")
    monkeypatch.setattr(
        formal,
        "_score",
        lambda package, roots, runs_root, labels, *args, **kwargs: score(
            package,
            1 if labels == {"hidden"} else 2,
            functional=kwargs.get("hidden_capsule_mode") == "functional_spike",
            hidden=labels == {"hidden"},
        ),
    )
    from merlin_experiments.phase1.feedback import compiler_profile_runtime

    monkeypatch.setattr(compiler_profile_runtime, "functional_service", lambda run_dir: None)
    return SimpleNamespace(
        run=run,
        descriptor=descriptor,
        public=public,
        hidden=hidden,
        contract=contract,
        context=context,
        resources={"status": "not_selected", "required": False, "resource_complete": False},
    )


def grade_and_freeze(s, *, hidden_mode="rtl"):
    C.grade(
        s.run,
        capsules=s.public,
        labels={"public", "dev"},
        context=s.context,
        contract=s.contract,
        facts=None,
        timeout_s=10,
    )
    freeze.freeze(s.run, repo=s.context.repo)
    C.grade(
        s.run,
        capsules=s.hidden,
        labels={"hidden"},
        context=s.context,
        contract=s.contract,
        facts=None,
        timeout_s=10,
        hidden_mode=hidden_mode,
    )


def test_receipt_reopens_actual_grades_and_readonly_exact_oot_export(selected):
    s = selected
    grade_and_freeze(s)
    C.publish(s.run, descriptor=s.descriptor, resources=s.resources)
    record = C.verify(s.run, descriptor=s.descriptor)
    assert record["capsule_complete"] is True
    assert record["formal_grade_complete"] is record["external_full_models_complete"] is False
    assert record["phase2_iteration_ready"] is False
    assert record["public"]["summary"]["n_passed"] == 2
    assert record["hidden"]["summary"]["n_passed"] == 1
    assert record["capsule_clock"]["wall_seconds"] >= 10
    assert record["handoff"]["compiler"]["hash_tree_sha256"] == record["submission_sha256"]
    assert not Path(record["handoff"]["compiler"]["path"]).stat().st_mode & 0o222
    with pytest.raises(FileExistsError):
        C.publish(s.run, descriptor=s.descriptor, resources=s.resources)


@pytest.mark.parametrize("fault", ["candidate", "roster", "descriptor", "score", "invocation", "export", "freeze"])
def test_receipt_refuses_changed_selected_or_produced_bytes(selected, fault):
    s = selected
    grade_and_freeze(s)
    C.publish(s.run, descriptor=s.descriptor, resources=s.resources)
    record = C.verify(s.run, descriptor=s.descriptor)
    path = {
        "candidate": s.run / "submission/compiler.py",
        "roster": s.public / "input.txt",
        "descriptor": s.descriptor,
        "score": s.run / "grading_hidden/score_capsule.json",
        "invocation": Path(record["public"]["invocation"]["path"]),
        "export": Path(record["handoff"]["compiler"]["path"]) / "compiler.py",
        "freeze": s.run / "freeze.json",
    }[fault]
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises((ValueError, KeyError)):
        C.verify(s.run, descriptor=s.descriptor)


@pytest.mark.parametrize("fault", ["empty", "partial", "trace", "budget", "tier", "cohort"])
def test_publisher_does_not_turn_missing_capsule_evidence_into_success(selected, monkeypatch, fault):
    s = selected
    original = formal._score

    def failed(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == "empty":
            result.update(n_capsules=0, n_passed=0)
        elif fault == "partial":
            result["n_passed"] -= 1
        elif fault == "trace":
            result["trace_all_pass"] = False
        elif fault == "budget":
            result["n_budget_exhausted"] = 1
        elif fault == "tier":
            result["tier_reached"]["L3"] = 0
        else:
            result["cohort_admission"]["n_admitted_capsules"] -= 1
        return result

    monkeypatch.setattr(formal, "_score", failed)
    grade_and_freeze(s)
    with pytest.raises(ValueError, match="incomplete"):
        C.publish(s.run, descriptor=s.descriptor, resources=s.resources)
    assert not (s.run / "capsule_certification.json").exists()


def test_selected_functional_hidden_stays_l2_and_never_rtl(selected):
    s = selected
    environment = yaml.safe_load((s.run / "environment.yaml").read_bytes())
    from merlin_experiments.phase1.feedback.hidden_capsule_policy import record

    environment["hidden_capsule_validation"] = record("functional_spike")
    (s.run / "environment.yaml").write_text(yaml.safe_dump(environment))
    grade_and_freeze(s, hidden_mode="functional_spike")
    C.publish(s.run, descriptor=s.descriptor, resources=s.resources)
    receipt = C.verify(s.run, descriptor=s.descriptor)
    assert receipt["hidden"]["summary"]["rtl_certified"] is False
    assert receipt["hidden"]["summary"]["required_tier"] == "L2"


def test_formal_capsules_preserves_public_host_projection_and_skips_only_external_waits(selected, monkeypatch):
    s = selected
    monkeypatch.setattr(formal, "_install_formal_model_simulator", lambda target: ({"engine": "synthetic"}, None))
    for name in ("requirements_for", "program_requirements_for", "loader_env_requirements_for"):
        monkeypatch.setattr(formal.PFM, name, lambda descriptor: {})

    def forbidden(*args, **kwargs):
        pytest.fail("capsule scope must not enter whole-model validation")

    monkeypatch.setattr(formal.PFM, "run", forbidden)
    from merlin_experiments.phase1.feedback import model_check_official, public_scalability_gate

    monkeypatch.setattr(model_check_official, "run_gate", forbidden)
    monkeypatch.setattr(public_scalability_gate, "run_gate", lambda **kwargs: s.resources)
    projected = []
    monkeypatch.setattr(public_owned_host_selection, "prepare", lambda *args, **kwargs: projected.append(args) or {})
    spec = s.run / "private-spec.yaml"
    spec.write_text("{}")
    rc = formal._main(
        [
            "--run-dir",
            str(s.run),
            "--arm",
            "baseline",
            "--capsules",
            str(s.public),
            "--hidden-capsules",
            str(s.hidden),
            "--contract",
            str(s.contract),
            "--validation-scope",
            "capsules",
            "--private-full-model-spec",
            str(spec),
        ],
        context=s.context,
    )
    assert rc == 0 and len(projected) == 1
    assert C.verify(s.run, descriptor=s.descriptor)["capsule_complete"] is True
    assert yaml.safe_load((s.run / "run_manifest.yaml").read_bytes())["completion"]["formal_grade_complete"] is False


@pytest.mark.parametrize("recorded,requested", [(None, "capsules"), ("full", "capsules"), ("capsules", "full")])
def test_scope_cannot_be_downgraded_by_resume_or_formal_argv(recorded, requested):
    environment = {} if recorded is None else {"validation_scope": recorded}
    with pytest.raises(ValueError, match="admitted run"):
        C.verify_scope(requested, environment)
    C.verify_scope("full", {})


@pytest.mark.parametrize("checkpoint", [False, True])
def test_capsule_completion_does_not_claim_external_or_operator_checkpoint_convergence(checkpoint):
    result = CERT._completion_for_scope(
        validation_scope="capsules",
        numeric_all_pass=True,
        workflow_conformant=True,
        official_grade={"complete": True},
        telemetry={"complete": True},
        checkpoint=checkpoint,
    )
    assert result["formal_complete"] is result["external_full_models_complete"] is False
    assert result["converged"] is result["phase2_iteration_ready"] is (not checkpoint)
    assert result["capsule_complete"] is True


def publish_author(s):
    grade_and_freeze(s)
    technical = C.publish(s.run, descriptor=s.descriptor, resources=s.resources)
    transcript = s.run / "rounds/round_00.transcript.jsonl"
    transcript.parent.mkdir()
    transcript.write_text(
        "\n".join(
            map(
                json.dumps,
                [
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "check",
                                    "name": "bash",
                                    "input": {"command": "python agent_selfcheck.py --capsules all"},
                                }
                            ]
                        },
                    },
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "check",
                                    "content": '{"n_capsules": 2, "n_passed": 2}',
                                    "is_error": False,
                                }
                            ]
                        },
                    },
                ],
            )
        )
        + "\n"
    )
    telemetry = s.run / "timing_detailed.json"
    telemetry.write_text(json.dumps({"telemetry_integrity": {"complete": True}}))
    summary = {
        "validation_scope": "capsules",
        "converged": True,
        "authoring_converged": True,
        "authoring_complete": True,
        "numeric_all_pass": True,
        "workflow_conformant": True,
        "rounds": [{"answer_access_clean": True, "audit_hits": []}],
        "finalize": {"skipped": "synthetic fixture has no paid finalizer"},
        "phase2_iteration_ready": True,
        "capsule_complete": True,
        "formal_complete": False,
        "external_full_models_complete": False,
        "telemetry": {"complete": True, **C.pin(telemetry)},
        "official_grade": {"capsule_certification": C.pin(s.run / "capsule_certification.json")},
    }
    (s.run / "qa_loop_summary.yaml").write_text(yaml.safe_dump(summary))
    publication = C.publish_phase2_handoff(
        s.run,
        descriptor=s.descriptor,
        transcript=transcript,
        endpoint_kind="",
        resolved_tools=[],
    )
    return technical, publication


def test_final_publication_requires_actual_reopened_workflow_and_durable_telemetry(selected):
    technical, publication = publish_author(selected)
    assert technical["phase2_iteration_ready"] is False
    assert publication["phase2_iteration_ready"] is True
    assert publication["handoff"] == technical["handoff"]
    assert publication["formal_grade_complete"] is publication["external_full_models_complete"] is False
    assert C.verify_phase2_handoff(selected.run, descriptor=selected.descriptor) == publication


@pytest.mark.parametrize("fault", ["workflow", "transcript", "qa", "telemetry", "compiler", "technical"])
def test_final_publication_refuses_post_publication_mutations(selected, fault):
    s = selected
    _, publication = publish_author(s)
    paths = {
        "workflow": Path(publication["workflow"]["path"]),
        "transcript": s.run / "rounds/round_00.transcript.jsonl",
        "qa": s.run / "qa_loop_summary.yaml",
        "telemetry": s.run / "timing_detailed.json",
        "compiler": Path(publication["handoff"]["compiler"]["path"]) / "compiler.py",
        "technical": s.run / "capsule_certification.json",
    }
    path = paths[fault]
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises((ValueError, KeyError)):
        C.verify_phase2_handoff(s.run, descriptor=s.descriptor)


def test_stale_grade_is_refused_before_another_observation(selected):
    s = selected
    (s.run / "grading_public").mkdir()
    (s.run / "grading_public/score_capsule.json").write_text(json.dumps(score("other", 2)))
    with pytest.raises(ValueError, match="stale"):
        C.grade(
            s.run,
            capsules=s.public,
            labels={"public", "dev"},
            context=s.context,
            contract=s.contract,
            facts=None,
            timeout_s=10,
        )


@pytest.mark.parametrize("scope,expected", [("capsules", 0), ("full", 120), (None, 120)])
def test_actual_authoring_execution_config_capsule_scope_never_starts_visible_models(
    tmp_path, monkeypatch, scope, expected
):
    from merlin_experiments.phase1 import authoring
    from merlin_experiments.phase1.feedback import lifecycle, visible_model_gate
    from merlin_experiments.phase1.providers import execution as EX

    # Evaluate the actual production keyword; then drive the real provider route
    # up to its paid-driver boundary, substituting only external brokers/driver.
    tree = ast.parse(inspect.getsource(authoring.execute))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ExecutionConfig"
    )
    expression = next(item.value for item in call.keywords if item.arg == "model_gate_interval")
    options = SimpleNamespace(grade_interval=120, experiment="full")
    if scope is not None:
        options.validation_scope = scope
    context = SimpleNamespace(target="gemmini")
    interval = eval(
        compile(ast.Expression(expression), "<actual ExecutionConfig>", "eval"), {"a": options, "context": context}
    )
    assert interval == expected
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "TASK.md").write_text("Synthetic driver boundary; no agent invocation.\n")
    started = []
    monkeypatch.setattr(lifecycle, "start_brokers", lambda *args: None)
    monkeypatch.setattr(lifecycle, "stop_brokers", lambda *args, **kwargs: None)
    monkeypatch.setattr(visible_model_gate, "start", lambda *args, **kwargs: started.append(kwargs) or None)

    def stop(*args, **kwargs):
        raise RuntimeError("synthetic paid-driver boundary")

    monkeypatch.setattr(EX, "resolve_driver", stop)
    config = EX.ExecutionConfig(
        context, EX.ProviderConfig(), lambda: (), tmp_path / "timing.json", model_gate_interval=interval
    )
    with pytest.raises(RuntimeError, match="paid-driver boundary"):
        EX.launch(workspace, tmp_path / "run", "synthetic", "low", "bwrap", {}, 0, 1, config=config)
    assert len(started) == (1 if expected else 0)


def test_publication_failure_clears_persisted_convergence_and_exit_claim(selected, monkeypatch):
    s = selected
    publish_author(s)
    summary = yaml.safe_load((s.run / "qa_loop_summary.yaml").read_bytes())
    (s.run / "cost_time_toolcalls.yaml").write_text("exit_code: 0\nwall_time_seconds: 123\n")

    def fail(*args, **kwargs):
        raise ValueError("synthetic lost workflow custody")

    monkeypatch.setattr(C, "publish_phase2_handoff", fail)
    result = CERT._publish_capsule_handoff(
        s.run,
        summary,
        validation_scope="capsules",
        descriptor=s.descriptor,
        transcript=s.run / "rounds/round_00.transcript.jsonl",
        endpoint_kind="",
        tools=[],
    )
    assert result["converged"] is result["authoring_converged"] is result["phase2_iteration_ready"] is False
    assert yaml.safe_load((s.run / "qa_loop_summary.yaml").read_bytes()) == result
    cost = yaml.safe_load((s.run / "cost_time_toolcalls.yaml").read_bytes())
    assert cost == {"exit_code": 1, "wall_time_seconds": 123}
    assert yaml.safe_load((s.run / "run_manifest.yaml").read_bytes())["completion"]["phase2_iteration_ready"] is False


def test_original_required_resource_policy_cannot_be_skipped(selected, monkeypatch):
    from merlin_experiments.phase1.feedback import compiler_profile_runtime, public_scalability_gate

    owner = SimpleNamespace(profile={"sha256": "a" * 64})
    monkeypatch.setattr(compiler_profile_runtime, "functional_service", lambda run: owner)
    monkeypatch.setattr(public_scalability_gate, "selected_policy", lambda selected: {"required": True})
    with pytest.raises(ValueError, match="admitted profile"):
        C._resources(selected.run, selected.resources, "b" * 64)


def test_identical_verifier_and_descriptor_copies_keep_original_observed_pins(selected, monkeypatch):
    from merlin_experiments.phase1 import authoring_conformance as AC

    s = selected
    _, published = publish_author(s)
    copied = s.run.parent / "copied-descriptor.yaml"
    shutil.copyfile(s.descriptor, copied)
    for module in (C, AC):
        destination = s.run.parent / (Path(module.__file__).name + ".copy")
        shutil.copyfile(module.__file__, destination)
        monkeypatch.setattr(module, "__file__", str(destination))
    reopened = C.verify_phase2_handoff(s.run, descriptor=copied)
    assert reopened == published
    assert reopened["descriptor"]["path"] == str(s.descriptor)
    s.descriptor.write_bytes(s.descriptor.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        C.verify_phase2_handoff(s.run, descriptor=copied)


@pytest.mark.parametrize("fault", ["answer_hit", "missing_audit", "leaked_mask", "isolation"])
def test_publication_preserves_answer_access_and_isolation_integrity(selected, fault):
    environment = yaml.safe_load((selected.run / "environment.yaml").read_bytes())
    summary = {
        "rounds": [{"answer_access_clean": True, "audit_hits": []}],
        "finalize": {"skipped": "no finalizer turn"},
    }
    C._author_integrity(selected.run, environment, summary)
    if fault == "answer_hit":
        summary["rounds"][0]["audit_hits"] = [{"kind": "unknown_content_read"}]
    elif fault == "missing_audit":
        del summary["rounds"][0]["audit_hits"]
    elif fault == "leaked_mask":
        environment["golden_mask_selftest"]["leaked_answer_files"] = ["private.bin"]
    else:
        environment["isolation_violations"] = ["private mount exposed"]
    with pytest.raises(ValueError, match="provenance|audit"):
        C._author_integrity(selected.run, environment, summary)


def test_same_public_and_hidden_capsule_identity_cannot_certify(selected, monkeypatch):
    s = selected
    original = formal._score

    def reused(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[3] == {"hidden"}:
            result["per_capsule"][0]["capsule"] = "public-synthetic-0"
            result["cohort_admission"]["admitted_name_set_sha256"] = hashlib.sha256(
                json.dumps(["public-synthetic-0"], separators=(",", ":")).encode(),
            ).hexdigest()
        return result

    monkeypatch.setattr(formal, "_score", reused)
    grade_and_freeze(s)
    with pytest.raises(ValueError, match="identities must be distinct"):
        C.publish(s.run, descriptor=s.descriptor, resources=s.resources)
