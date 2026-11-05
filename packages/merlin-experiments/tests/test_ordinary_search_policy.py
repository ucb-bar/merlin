"""Normal retained feedback routing with synthetic measurements; no engines run."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import corpus_feedback as CF
from merlin_experiments.phase2 import development_feedback as DF
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2 import gsim_gate as GATE
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2 import paired_measurement as PME
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore, executing_revision
from test_component_workflow import generated_corpus

from merlin.benchharness import hash_tree
from merlin.perf.phase2_edit_contract import seal
from merlin.targetgen.contract.artifact_retesting import active_artifact_retests
from merlin.targetgen.target_experiment import TargetExperiment


@pytest.fixture
def ordinary_inputs(tmp_path, monkeypatch, request):
    """Stub only external admission/execution evidence, retain actual normal routing."""
    monkeypatch.setenv(DF.SWEEP_WORKERS_ENV, "1")
    monkeypatch.delenv(DF.AUTHORING_GSIM_FLOOR_BUDGET_ENV, raising=False)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    selected = getattr(request, "param", "def schedule():\n    return 1\n")
    source = selected["source"] if isinstance(selected, dict) else selected
    components = selected.get("new_python_components") if isinstance(selected, dict) else None
    (candidate / "compiler.py").write_text(source)
    (candidate / "manifest.yaml").write_text(
        json.dumps({"entrypoints": {"tool": "compiler.py"}, "commands": {"compile": {"argv": ["python", "{tool}"]}}})
    )
    baseline = tmp_path / "baseline"
    digest = PC.materialize_readonly_tree(candidate, baseline)
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    authority = FrozenEditAuthority(authority_root)
    authority.freeze(
        candidate,
        seal(
            {
                "schema": "compiler_edit_contract_v2" if components is not None else "compiler_edit_contract_v1",
                "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
                "helper_extensions": [],
                **({"new_python_components": components} if components is not None else {}),
            }
        ),
        has_iterations=False,
    )
    corpus = generated_corpus(tmp_path)
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: fixture\n")
    target = TargetExperiment(
        target="fixture",
        isa_headers=(),
        hwbringup_set=None,
        curated_harness=None,
        capsule_corpus=corpus.capsules_root,
        sim_via="synthetic",
        rtl_via="synthetic",
        rtl_repo=None,
        prior_backends=(),
        path=descriptor,
        descriptor_sha256=sha256_file(descriptor),
    )
    run = tmp_path / "functional-run"
    run.mkdir()
    for name in OS._FUNCTIONAL_FILES:
        path = run / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")
    (run / "environment.yaml").write_text(
        json.dumps(
            {
                "task_scope": {"target": target.target},
                "public_corpus_input": {"descriptor_sha256": target.descriptor_sha256},
            }
        )
    )

    def score(name):
        return {"per_capsule": [{"capsule": name, "tiers": {"L2": "pass", "L3": "pass"}}]}

    functional = FI.StageFunctionalRun(
        run, baseline, run.name, digest, 1, 1, score("public"), score("hidden"), "synthetic", {}, {}, baseline
    )
    admission = []

    def inspect(run_root, run_id, expected_digest, *, waive):
        admission.append((run_root, run_id, expected_digest, waive))
        assert (run_root, run_id, expected_digest) == (tmp_path, run.name, digest)
        return functional

    monkeypatch.setattr(FI, "inspect_stage_functional_run", inspect)
    # This fixture substitutes external qualification and measurements; the
    # real evidence reader has independent tests and never runs a simulator.
    monkeypatch.setattr(FI, "require_formal_qualification", lambda functional, target: None)
    monkeypatch.setattr(FI, "verify_functional_host_lane_snapshot", lambda snapshot: None)
    monkeypatch.setattr(PC, "verify_private_input_snapshot", lambda run_dir, environment: None)
    certificate_path = tmp_path / "certificate.json"
    certificate_path.write_text('{"synthetic_not_qualified":true}')
    workload = {"operation": "synthetic", "shape": {"m": 1}, "semantics": {"dtype": "i32"}}
    identity = GATE.workload_sha256(workload)
    certificate = GATE.CertificateRecord(
        certificate_path, sha256_file(certificate_path), target.target, {}, {identity: {}}, {}, {}
    )

    def load(path, *, expected_sha256):
        assert sha256_file(path) == expected_sha256
        return certificate

    monkeypatch.setattr(GATE, "load_certificate", load)
    monkeypatch.setattr(PME, "gsim_workload", lambda member: workload)
    facts = tmp_path / "facts.json"
    facts.write_text("{}")
    monkeypatch.setattr(DF.MS, "load_rtl_identity", lambda path, target: {"sha256": sha256_file(path)})
    monkeypatch.setattr(DF.FM, "derived_peak_macs_per_cycle", lambda *args: (None, "synthetic unknown"))
    monkeypatch.setattr(DF, "derive_machine_bounds", lambda *args: {})
    monkeypatch.setattr(DF.MS, "executed_profile", lambda *args, **kwargs: {"status": "unknown"})
    executions = []

    def execute(spec, workspace, timeout, target, rtl_identity, **kwargs):
        assert active_artifact_retests() is not None
        executions.append((spec, workspace, timeout, executing_revision(spec.package)))
        decision = spec.gsim_decision
        return {
            "measurement": {
                "status": "pass",
                "numeric": "pass",
                "per_sim": {"spike": {"correct": True}, "gsim": {"correct": True, "cycles": 10}},
                "gsim_qualification": {"admitted": True, "decision": decision.to_dict()},
            }
        }

    monkeypatch.setattr(PME, "run_execution", execute)
    contract = tmp_path / "contract"
    contract.mkdir()
    (contract / "schema.json").write_text("{}")
    arguments = dict(
        functional_runs_root=tmp_path,
        functional_run_id=run.name,
        functional_submission_sha256=digest,
        corpus=corpus,
        baseline=baseline,
        candidate=candidate,
        target_experiment=target,
        receipt_path=tmp_path / "control/receipts.jsonl",
        work_root=tmp_path / "feedback",
        edit_authority=authority,
        contract_root=contract,
        certificate_path=certificate_path,
        certificate_sha256=certificate.sha256,
        rtl_facts_path=facts,
    )
    return SimpleNamespace(
        arguments=arguments, executions=executions, admission=admission, certificate=certificate, functional=functional
    )


def revisions(selected, tmp_path):
    store = RetainedRevisionStore(
        tmp_path / "revisions",
        edit_authority=selected.edit_authority,
        max_bytes=16_384,
        max_files=16,
        max_total_bytes=65_536,
    )
    baseline = store.freeze(selected.search_baseline)
    (selected.candidate / "compiler.py").write_text("def schedule():\n    return 2\n")
    candidate = store.freeze(selected.candidate, parent_revision=baseline.revision_id)
    return candidate, baseline


def test_normal_factory_uses_original_admission_and_fixed_executor(ordinary_inputs):
    f = ordinary_inputs
    selected = OS.prepare_ordinary_search_policy(**f.arguments)
    assert f.admission == [(f.arguments["functional_runs_root"], f.functional.run_id, f.functional.digest, frozenset())]
    assert type(selected) is OS.OrdinarySearchPolicy
    assert isinstance(selected, CF.CorpusFeedbackPolicy)
    assert selected.workflow_id == BP.CORPUS_FEEDBACK_V1
    assert selected.verify_receipts is CF.verify_broker_receipts
    assert selected.feedback_evaluator.executor.func is DF.development_executor
    assert selected.search_baseline == f.arguments["baseline"]
    assert selected.search_corpus is f.arguments["corpus"]
    assert selected.validate_search_inputs() == {("generated-family", "member")}
    assert not f.executions
    binding = selected.binding()
    assert binding["scope"] == "frozen_development_corpus_simulator_cycles"
    assert "FreshCompilerOrigin" not in json.dumps(binding)
    assert len(selected.search_closure_sha256) == 64
    binding["scope"] = "changed local copy"
    assert selected.binding()["scope"] == "frozen_development_corpus_simulator_cycles"


def test_retained_dispatch_executes_original_complete_feedback_and_ignores_live_edits(ordinary_inputs, tmp_path):
    f = ordinary_inputs
    selected = OS.prepare_ordinary_search_policy(**f.arguments)
    candidate, baseline = revisions(selected, tmp_path)
    (selected.candidate / "compiler.py").write_text("unfinished author edit: [\n")
    observed = selected.evaluate_retained(candidate, baseline, work_root=tmp_path / "job", call_index=0, timeout_s=10)
    assert observed["candidate_sha256"] == candidate.source_sha256
    assert observed["tuning_corpus_sha256"] == selected.search_corpus.capsules_sha256
    assert observed["certificate_sha256"] == f.certificate.sha256
    assert observed["summary"]["comparable"] == len(selected.search_corpus.capsules)
    assert observed["summary"]["all_correct"]
    assert {spec.arm for spec, *_ in f.executions} == {"baseline", "candidate"}
    assert {spec.package for spec, *_ in f.executions} == {baseline.source, candidate.source}
    assert all(issued is (baseline if spec.arm == "baseline" else candidate) for spec, _, _, issued in f.executions)
    assert not (tmp_path / "job/round_00/call_000/_measured_candidate").exists()
    assert hash_tree(candidate.source)["sha256"] == candidate.source_sha256
    assert callable(selected._feedback)  # the normal broker's original handler was not shadowed


@pytest.mark.parametrize("changed", ["target", "functional", "contract", "corpus", "executor", "descriptor", "owner"])
def test_drift_refuses_before_execution(ordinary_inputs, tmp_path, changed):
    f = ordinary_inputs
    selected = OS.prepare_ordinary_search_policy(**f.arguments)
    candidate, baseline = revisions(selected, tmp_path)
    if changed == "executor":
        selected.feedback_evaluator.executor = lambda **kwargs: {"pass": True}
    elif changed == "descriptor":
        selected.search_corpus.capsules[0].descriptor["label"] = "protected"
    elif changed == "owner":
        selected.functional_base = selected.candidate
    else:
        path = {
            "target": selected.target_experiment.path,
            "functional": f.functional.run_dir / "freeze.json",
            "contract": f.arguments["contract_root"] / "schema.json",
            "corpus": selected.search_corpus.capsules[0].source_dir / "capsule.yaml",
        }[changed]
        path.chmod(0o644)
        path.write_text('{"changed":true}')
    with pytest.raises((StageGateError, AssertionError), match="changed|differs|fixed development|bytes"):
        selected.evaluate_retained(candidate, baseline, work_root=tmp_path / "job", call_index=0, timeout_s=10)
    assert not f.executions


def test_other_authority_and_wrong_baseline_refuse(ordinary_inputs, tmp_path):
    selected = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    candidate, baseline = revisions(selected, tmp_path)
    with pytest.raises(StageGateError, match="Phase 1 functional fork"):
        selected.evaluate_retained(candidate, candidate, work_root=tmp_path / "job", call_index=0, timeout_s=10)
    tampered = replace(candidate, edit_authority_sha256="a" * 64)
    with pytest.raises(StageGateError):
        selected.evaluate_retained(tampered, baseline, work_root=tmp_path / "job", call_index=0, timeout_s=10)
    assert not ordinary_inputs.executions


@pytest.mark.parametrize("timeout", [True, 0, 601])
def test_original_reduced_execution_ceiling_remains(ordinary_inputs, tmp_path, timeout):
    selected = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    candidate, baseline = revisions(selected, tmp_path)
    with pytest.raises(StageGateError, match="reduced-witness deadline"):
        selected.evaluate_retained(candidate, baseline, work_root=tmp_path / "job", call_index=0, timeout_s=timeout)
    assert not ordinary_inputs.executions


def test_original_functional_refusal_cannot_be_replaced_with_a_saved_pass(ordinary_inputs, monkeypatch):
    def refuse(*args, **kwargs):
        raise StageGateError("original functional admission refused")

    monkeypatch.setattr(FI, "inspect_stage_functional_run", refuse)
    with pytest.raises(StageGateError, match="original functional admission refused"):
        OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    assert not ordinary_inputs.executions


def test_formal_model_refusal_prevents_feedback_preparation(ordinary_inputs, monkeypatch):
    def refuse(functional, target):
        assert functional is ordinary_inputs.functional
        raise StageGateError("Phase 1 formal qualification is incomplete")

    monkeypatch.setattr(FI, "require_formal_qualification", refuse)
    with pytest.raises(StageGateError, match="formal qualification"):
        OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    assert not ordinary_inputs.executions


def test_ordinary_search_cannot_waive_functional_gaps(ordinary_inputs):
    with pytest.raises(StageGateError, match="unwaived"):
        OS.prepare_ordinary_search_policy(
            **ordinary_inputs.arguments, waive_functional_gate=("phase_grade_incomplete",)
        )
    assert not ordinary_inputs.executions


@pytest.mark.parametrize("identity", ["target", "descriptor_sha256"])
def test_phase1_target_selection_cannot_be_substituted(ordinary_inputs, identity):
    f = ordinary_inputs
    path = f.functional.run_dir / "environment.yaml"
    environment = json.loads(path.read_text())
    environment["task_scope" if identity == "target" else "public_corpus_input"][identity] = "another-selection"
    path.write_text(json.dumps(environment))
    with pytest.raises(StageGateError, match="original Phase 1 environment"):
        OS.prepare_ordinary_search_policy(**f.arguments)
    assert not f.executions


def test_member_fanout_cannot_exceed_the_explicit_worker_selection(ordinary_inputs, tmp_path, monkeypatch):
    selected = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    candidate, baseline = revisions(selected, tmp_path)
    monkeypatch.setenv(DF.SWEEP_WORKERS_ENV, "2")
    with pytest.raises(StageGateError, match="explicit member worker grant"):
        selected.evaluate_retained(candidate, baseline, work_root=tmp_path / "job", call_index=0, timeout_s=10)
    assert not ordinary_inputs.executions
