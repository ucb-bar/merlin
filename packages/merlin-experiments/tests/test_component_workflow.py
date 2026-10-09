"""Real component broker admission/receipts; synthetic services, no engines or agents."""

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from component_baseline_fixture import (
    synthetic_component_controller as synthetic_component_controller,
)
from component_baseline_fixture import (
    unissued_baseline,
    unissued_runtime,
)
from merlin_experiments.phase2 import authoring, authoring_cli, broker, stage_inputs, stage_prompt
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import component_workflow as CW
from merlin_experiments.phase2 import corpus as C
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, sha256_file

from merlin.benchharness import hash_tree
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

pytestmark = pytest.mark.usefixtures("synthetic_component_controller")


def generated_corpus(tmp_path, *, kind=None):
    source = tmp_path / "generated"
    member_dir = source / "_tuning" / "member"
    member_dir.mkdir(parents=True)
    descriptor = {
        "name": "member",
        "label": "dev",
        "source_role": "derived_sweep",
        "performance": {"family": "generated-family", "claim": "RECOVERS"},
    }
    if kind is not None:
        descriptor["kind"] = kind
    (member_dir / "capsule.yaml").write_text(json.dumps(descriptor))
    provenance = source / "MANIFEST.yaml"
    provenance.write_text(
        json.dumps(
            {
                "generated": ["_tuning/member"],
                "hand_authored": [],
                "performance_generation": {
                    "fixture": {
                        "errors": [],
                        "phase": {
                            "category": "_tuning",
                            "label": "dev",
                            "included_in_functional_grade": False,
                        },
                    }
                },
            }
        )
    )
    public = source / "public"
    public.mkdir()
    live = C.discover_performance_corpus(
        SimpleNamespace(
            target="fixture",
            capsule_corpus=public,
            graded_roots=lambda: [public],
        )
    )
    return C.freeze_performance_corpus(live, tmp_path / "frozen")


def context(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "compiler").write_text("# synthetic tool\n")
    (candidate / "manifest.yaml").write_text(
        json.dumps(
            {
                "entrypoints": {"tool": "compiler"},
                "commands": {"parse": {"argv": ["{tool}", "{input}", "{output}"]}},
            }
        )
    )
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: fixture\n")
    result = dict(
        candidate=candidate,
        target_experiment=SimpleNamespace(target="fixture", path=descriptor),
        receipt_path=tmp_path / "control" / "receipts.jsonl",
        component_corpus=generated_corpus(tmp_path),
    )
    baseline = tmp_path / "fresh-functional-baseline"
    shutil.copytree(candidate, baseline)
    result["baseline_admission"] = unissued_baseline(
        baseline=baseline, corpus=result["component_corpus"], target_descriptor=descriptor,
    )
    result["target_experiment"].fixture_inputs = result
    return result


def analytical(candidate, corpus, timeout_s):
    assert timeout_s > 0 and corpus.capsules
    return {
        (member.family, member.capsule): (
            CycleInterval(10, 20, provenance=("controlled calibration",)),
            CycleInterval.unknown("cache domain is not calibrated"),
        )
        for member in corpus.capsules
    }


def provider(tmp_path, monkeypatch, target):
    from merlin_experiments.phase2 import component_analytical as CA

    from merlin.perf.component_cost import ComponentCostScope

    adapter = tmp_path / "adapter.json"
    adapter.write_text("{}")
    calibration = {"target_sha256": sha256_file(target.path), "controlled": True}
    for module in (CW, CA):
        monkeypatch.setattr(module, "prepare_phase2_calibration",
                            lambda _: {"status": "ready", "calibration": calibration})
    source = Path(__file__)
    selected = target.fixture_inputs
    admission = selected["baseline_admission"]
    runtime = unissued_runtime(target_descriptor=target.path, feature_provider=analytical,
                               source_pins=((adapter, sha256_file(adapter)),))
    selected["independent_runtime"] = runtime
    bound = CA.build_component_analytical_provider(
        baseline=admission.baseline, corpus=selected["component_corpus"], target_descriptor=target.path,
        feature_provider=analytical, calibration_adapter=adapter,
        scope=ComponentCostScope(*(document_sha256(value) for value in ("timer", "accuracy", "inputs"))),
        output=tmp_path / "analytical-output", lease_path=tmp_path / "engine.lease",
        dependencies={}, memory_per_worker_bytes=1 << 20, baseline_admission=admission,
        independent_runtime=runtime,
    )
    return replace(bound, evaluate=analytical, implementation=source, implementation_sha256=sha256_file(source))


def test_registry_contains_no_model_or_descriptor_probe_actions(tmp_path, monkeypatch):
    selected = context(tmp_path)
    monkeypatch.setattr(BP.TC, "required_tool_probes", lambda *_: pytest.fail("descriptor probes were selected"))
    actions = BP.action_registry(BP.COMPONENT_ONLY_V1, selected["candidate"], selected["target_experiment"])
    names = {action.name for action in actions}
    assert names == {
        "candidate-parse",
        CW.ANALYTICAL_ACTION,
        CW.RTL_ACTION,
        CW.CCA_ACTION,
        BP.INVENTORY_ACTION,
        BP.ANALYSIS_ACTION,
    }
    assert {action.name for action in actions if action.required} == {"candidate-parse"}
    assert not any(action.available for action in actions if action.name in (CW.ANALYTICAL_ACTION, CW.RTL_ACTION))
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **selected)
    assert not policy.admission(CW.ANALYTICAL_ACTION).reserved
    assert policy.admission(CW.RTL_ACTION).limit == 2
    with pytest.raises(AttributeError):
        policy.workflow_id = BP.CORPUS_FEEDBACK_V1


@pytest.mark.parametrize("name", CW._GLOBAL_INPUTS)
def test_legacy_capability_is_not_inherited(tmp_path, name):
    with pytest.raises(StageGateError, match="refuses|disagrees"):
        BP.select_workflow(BP.COMPONENT_ONLY_V1, **context(tmp_path), **{name: object()})


@pytest.mark.parametrize(
    "services",
    [
        BP.BrokerServices(whole_model_analysis=lambda: None),
        BP.BrokerServices(global_analysis_view=lambda: None),
    ],
)
def test_whole_model_services_are_rejected(tmp_path, services):
    with pytest.raises(StageGateError, match="whole-model services"):
        BP.select_workflow(BP.COMPONENT_ONLY_V1, **context(tmp_path), services=services)


def test_actual_broker_estimates_unknowns_and_closes_scoped_receipts(tmp_path, monkeypatch):
    selected = context(tmp_path)
    selected["component_analytical"] = provider(tmp_path, monkeypatch, selected["target_experiment"])
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **selected)
    actions = policy.build_registry()
    engine = broker.Broker(
        SimpleNamespace(),
        selected["target_experiment"],
        selected["candidate"],
        actions,
        selected["receipt_path"],
        workflow=policy,
        deadline=time.monotonic() + 30,
        max_calls=5,
        max_tool_seconds=10,
    )
    result = engine.execute({"action": CW.ANALYTICAL_ACTION})
    assert result["returncode"] == 0
    feedback = json.loads(result["stdout"])
    assert feedback["tier"] == "calibrated_component_analytical"
    assert feedback["promotion"] == "NO_FINAL_ACCEPTANCE"
    assert feedback["evidence"][0]["candidate"]["lo"] is None
    assert feedback["evidence"][0]["candidate"]["missing"] == ["cache domain is not calibrated"]
    rows = [json.loads(line) for line in selected["receipt_path"].read_text().splitlines()]
    # Required compiler commands still need successful receipts. Use a feedback-only
    # projection here to qualify the actual host action, without a fake compiler run.
    feedback_actions = tuple(action for action in actions if not action.required)
    audit = {
        "broker_invocations": [
            {"action": row["action"], "bindings_sha256": row["bindings_command_sha256"]} for row in rows
        ]
    }
    qualified = policy.verify_receipts(
        selected["receipt_path"], feedback_actions, audit, candidate_sha256=hash_tree(selected["candidate"])["sha256"]
    )
    assert qualified["final_acceptance"] == "NOT_ESTABLISHED"
    with pytest.raises(StageGateError):
        engine.execute({"action": BP.E2E_ANALYSIS_ACTION})
    rows[0]["workflow_id"] = BP.CORPUS_FEEDBACK_V1
    selected["receipt_path"].write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises((ValueError, StageGateError), match="workflow"):
        policy.verify_receipts(selected["receipt_path"], feedback_actions, audit)


def test_unavailable_tiers_and_forged_model_registry_refuse(tmp_path):
    selected = context(tmp_path)
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **selected)
    assert policy.unavailable[CW.RTL_ACTION]
    actions = {action.name: action for action in policy.build_registry()}
    actions[BP.E2E_ANALYSIS_ACTION] = broker.BrokerAction(BP.E2E_ANALYSIS_ACTION, ("host",), (), "forged", False)
    with pytest.raises(StageGateError, match="action contract"):
        policy.validate_actions(actions)
    with pytest.raises(StageGateError, match="outside"):
        policy.execute({}, BP.OCCUPANCY_PROFILE_ACTION, {}, 0, 1, time.monotonic())


def test_provider_configuration_corpus_and_selection_drift_refuse(tmp_path, monkeypatch):
    selected = context(tmp_path)
    own = provider(tmp_path, monkeypatch, selected["target_experiment"])
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **selected, component_analytical=own)
    own.calibration_adapter.write_text('{"changed": true}')
    with pytest.raises(StageGateError, match="identity changed"):
        policy._validate()
    own.calibration_adapter.write_text("{}")
    policy.component_analytical = None
    with pytest.raises(StageGateError, match="selection changed"):
        policy._validate()
    policy.component_analytical = own
    policy.component_corpus = replace(selected["component_corpus"], capsules_sha256="a" * 64)
    with pytest.raises(StageGateError, match="bytes changed"):
        policy._validate()


def _untrusted_interval(value):
    # A callback can bypass a Python constructor. The broker must still perform
    # its independent finite/provenance checks, even with strict value objects.
    interval = object.__new__(CycleInterval)
    for name, selected in (("lo", value), ("hi", value), ("provenance", ("claimed",)), ("missing", ())):
        object.__setattr__(interval, name, selected)
    return interval


@pytest.mark.parametrize(
    "interval",
    [
        _untrusted_interval(float("nan")),
        _untrusted_interval(float("inf")),
        CycleInterval(1, 2),
    ],
)
def test_unqualified_analytical_cost_cannot_become_a_measurement(interval):
    with pytest.raises(StageGateError):
        CW._interval(interval)


def test_rtl_does_not_accept_arbitrary_callback_or_spike_tier(tmp_path):
    with pytest.raises(StageGateError, match="certified GSIM"):
        BP.select_workflow(BP.COMPONENT_ONLY_V1, **context(tmp_path), component_rtl=lambda: None)
    with pytest.raises(StageGateError, match="unsupported"):
        CW.validate_component_feedback(
            {
                "schema": CW.SCHEMA,
                "workflow_id": BP.COMPONENT_ONLY_V1,
                "tier": "spike",
                "candidate_sha256": "a" * 64,
                "corpus_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
                "provider_sha256": "d" * 64,
                "configuration_sha256": "e" * 64,
                "evidence": {},
                "promotion": "NO_FINAL_ACCEPTANCE",
            }
        )


def test_normal_authoring_profile_refuses_before_legacy_telemetry_or_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(authoring, "_require_executable", lambda *_a, **_k: pytest.fail("legacy launch was entered"))
    with pytest.raises(StageGateError, match="verified runtime isolation and zero-history"):
        authoring.run_stage(
            suite="fixture",
            contract_root=tmp_path,
            source_root=tmp_path,
            functional_runs_root=tmp_path,
            functional_run_id="fixture",
            functional_submission_sha256="a" * 64,
            target_experiment=SimpleNamespace(),
            sandbox_inputs=None,
            stage_root=tmp_path / "stage",
            model="unused",
            effort="high",
            wall_budget_seconds=10,
            rounds=1,
            round_timeout_seconds=10,
            max_tool_calls=1,
            tool_timeout_seconds=1,
            workflow_id=BP.COMPONENT_ONLY_V1,
        )
    assert not (tmp_path / "stage").exists()


def test_normal_cli_profile_refuses_without_legacy_certificate_or_model_inputs(monkeypatch, capsys):
    monkeypatch.setattr(authoring_cli, "load_target_experiment", lambda *_a, **_k: pytest.fail("descriptor accessed"))
    assert authoring_cli.main(["--workflow", BP.COMPONENT_ONLY_V1]) == 2
    assert "zero-history session transport" in capsys.readouterr().err


def test_generated_model_and_mutated_descriptor_are_not_component_inputs(tmp_path):
    model = generated_corpus(tmp_path / "model", kind="model")
    with pytest.raises(StageGateError, match="not models"):
        CW._corpus(model)
    selected = context(tmp_path / "ordinary")
    selected["component_corpus"].capsules[0].descriptor["source_role"] = "manual"
    with pytest.raises(StageGateError, match="generated development components"):
        BP.select_workflow(BP.COMPONENT_ONLY_V1, **selected)


def test_component_prompt_uses_only_generated_inputs_and_selected_capabilities(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage_inputs, "select_e2e_sentinel", lambda *_a, **_k: pytest.fail("model selection was entered")
    )
    selected = context(tmp_path)
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **selected)
    paths = ("/approved/corpus-manifest.json", "candidate", "/broker/tool", "/broker/receipts")
    prepared = stage_inputs.prepare_component_prompt_inputs(
        policy,
        corpus_manifest_path=paths[0],
        candidate_path=paths[1],
        allowed_paths=paths,
        wall_budget_seconds=60,
        max_tool_calls=5,
        tool_timeout_seconds=10,
        broker_path=paths[2],
        broker_receipt_path=paths[3],
    )
    text = stage_prompt.render_component_prompt(prepared)
    from merlin.targetgen.generalization_prompt import GENERAL_COMPILER_CONTRACT_V1

    assert text.count(GENERAL_COMPILER_CONTRACT_V1) == 1
    document = json.loads(text.split("```json\n")[1].split("\n```")[0])
    assert document["workflow_id"] == BP.COMPONENT_ONLY_V1
    assert document["final_acceptance"] == "NOT_ESTABLISHED"
    assert document["launch_admission"] == "REQUIRED_SEPARATELY"
    assert document["corpus"]["members"] == [{"family": "generated-family", "capsule": "member"}]
    assert all(action["name"] != BP.E2E_ANALYSIS_ACTION for action in document["actions"])
    assert not next(action for action in document["actions"] if action["name"] == CW.RTL_ACTION)["available"]
    for action in prepared.actions:
        assert f"- `{action.name}`: {action.purpose}" in text
        if not action.available:
            assert f"UNAVAILABLE: {action.unavailable_reason}" in text
    assert "e2e_sentinel" not in document and "functional_bundle_snapshot" not in document
    assert str(selected["component_corpus"].root) not in text
    forged = broker.BrokerAction(BP.E2E_ANALYSIS_ACTION, ("host",), (), "forged", False)
    with pytest.raises(StageGateError, match="outside"):
        stage_prompt.render_component_prompt(replace(prepared, actions=(*prepared.actions, forged)))
    with pytest.raises(StageGateError, match="declared view"):
        stage_prompt.render_component_prompt(replace(prepared, allowed_paths=()))


def _analytic_broker(tmp_path, monkeypatch):
    selected = context(tmp_path)
    selected["component_analytical"] = provider(tmp_path, monkeypatch, selected["target_experiment"])
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **selected)
    actions = policy.build_registry()
    engine = broker.Broker(
        SimpleNamespace(),
        selected["target_experiment"],
        selected["candidate"],
        actions,
        selected["receipt_path"],
        workflow=policy,
        deadline=time.monotonic() + 30,
        max_calls=5,
        max_tool_seconds=10,
    )
    return selected, policy, actions, engine


@pytest.mark.parametrize(
    "failure", ["missing-member", "invented-member", "missing-cost", "unqualified-cost", "timeout"]
)
def test_analytic_provider_cannot_fabricate_coverage_or_return_after_budget(tmp_path, monkeypatch, failure):
    selected = context(tmp_path)
    original = provider(tmp_path, monkeypatch, selected["target_experiment"])

    def bad(candidate, corpus, timeout_s):
        if failure == "timeout":
            pytest.fail("provider was called after its deadline")
        if failure == "missing-member":
            return {}
        if failure == "invented-member":
            return {("foreign", "member"): (CycleInterval.point(1, "fixture"), CycleInterval.point(1, "fixture"))}
        if failure == "missing-cost":
            return {("generated-family", "member"): (CycleInterval.point(1, "fixture"),)}
        if failure == "unqualified-cost":
            return {("generated-family", "member"): (CycleInterval(1, 2), CycleInterval(1, 2))}
        return analytical(candidate, corpus, timeout_s)

    # A trusted host can select a new provider; the running policy cannot change selection.
    provider_bad = replace(original, evaluate=bad)
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **{**selected, "component_analytical": provider_bad})
    if failure == "timeout":
        result, feedback = policy.execute({}, CW.ANALYTICAL_ACTION, {}, 0, 1, time.monotonic() - 2)
    else:
        engine = broker.Broker(
            SimpleNamespace(),
            selected["target_experiment"],
            selected["candidate"],
            policy.build_registry(),
            selected["receipt_path"],
            workflow=policy,
            deadline=time.monotonic() + 30,
            max_calls=5,
            max_tool_seconds=10,
        )
        result = engine.execute({"action": CW.ANALYTICAL_ACTION})
        feedback = None
    assert result["returncode"] == 125 and not result["stdout"] and feedback is None
    assert "foreign" not in result["stderr"]


@pytest.mark.parametrize("failure", ["rejected-success", "bool-returncode", "wrong-tier", "wrong-stdout", "unbound"])
def test_new_receipt_identity_and_tier_integrity_fail_closed(tmp_path, monkeypatch, failure):
    selected, policy, actions, engine = _analytic_broker(tmp_path, monkeypatch)
    assert engine.execute({"action": CW.ANALYTICAL_ACTION})["returncode"] == 0
    rows = [json.loads(line) for line in selected["receipt_path"].read_text().splitlines()]
    if failure == "rejected-success":
        rows[0]["state"] = "rejected"
    elif failure == "bool-returncode":
        rows[0]["returncode"] = False
    elif failure == "wrong-tier":
        rows[0]["action"] = CW.RTL_ACTION
    elif failure == "wrong-stdout":
        rows[0]["stdout_sha256"] = "a" * 64
    else:
        del rows[0]["workflow_id"]
    selected["receipt_path"].write_text(json.dumps(rows[0]) + "\n")
    audit = {
        "broker_invocations": [{"action": rows[0]["action"], "bindings_sha256": rows[0]["bindings_command_sha256"]}]
    }
    with pytest.raises(StageGateError):
        policy.verify_receipts(
            selected["receipt_path"], tuple(action for action in actions if not action.required), audit
        )


def test_unavailable_calibration_and_changed_target_refuse(tmp_path, monkeypatch):
    selected, policy, _, _ = _analytic_broker(tmp_path, monkeypatch)
    selected["target_experiment"].path.write_text("target: other\n")
    with pytest.raises(StageGateError, match="target configuration changed"):
        policy._validate()
    selected["target_experiment"].path.write_text("target: fixture\n")
    monkeypatch.setattr(CW, "prepare_phase2_calibration", lambda _: {"status": "incomplete", "calibration": None})
    with pytest.raises(StageGateError, match="calibration is unavailable"):
        policy._validate()


def test_same_source_callback_replacement_and_code_replacement_refuse(tmp_path, monkeypatch):
    selected, policy, _, _ = _analytic_broker(tmp_path, monkeypatch)
    original = policy.component_analytical

    def another(candidate, corpus, timeout_s):
        return {}

    # Frozen dataclass integrity must not depend on the caller respecting Python's
    # conventional immutability: both callables have the same pinned owner file.
    object.__setattr__(original, "evaluate", another)
    with pytest.raises(StageGateError, match="callable binding changed"):
        policy._validate()
    object.__setattr__(original, "evaluate", analytical)
    saved = analytical.__code__
    try:
        analytical.__code__ = another.__code__
        with pytest.raises(StageGateError, match="callable binding changed"):
            policy._validate()
    finally:
        analytical.__code__ = saved
    policy._validate()


def test_unknown_interval_rechecks_reason_after_constructor(tmp_path):
    interval = CycleInterval.unknown("fixture missing cost")
    object.__setattr__(interval, "missing", ())
    with pytest.raises(StageGateError, match="lacks its reason"):
        CW._interval(interval)


def test_feedback_does_not_replace_required_commands_or_bind_later_compiler(tmp_path, monkeypatch):
    selected, policy, actions, engine = _analytic_broker(tmp_path, monkeypatch)
    assert engine.execute({"action": CW.ANALYTICAL_ACTION})["returncode"] == 0
    row = json.loads(selected["receipt_path"].read_text())
    audit = {"broker_invocations": [{"action": row["action"], "bindings_sha256": row["bindings_command_sha256"]}]}
    with pytest.raises(StageGateError, match="required component command"):
        policy.verify_receipts(selected["receipt_path"], actions, audit)
    (selected["candidate"] / "compiler").write_text("# later compiler bytes\n")
    with pytest.raises(StageGateError, match="sealed compiler identity"):
        policy.verify_receipts(
            selected["receipt_path"],
            tuple(action for action in actions if not action.required),
            audit,
            candidate_sha256=hash_tree(selected["candidate"])["sha256"],
        )


def no_engine_executor(**kwargs):
    pytest.fail("synthetic policy test cannot execute an engine")


def _synthetic_rtl_policy(tmp_path, monkeypatch):
    from merlin_experiments.phase2 import development_feedback as DF
    from merlin_experiments.phase2 import gsim_gate, paired_measurement

    selected = context(tmp_path)
    certificate = tmp_path / "certificate.json"
    certificate.write_text("{}")
    record = SimpleNamespace(path=certificate, sha256=sha256_file(certificate), target="fixture")
    decision = SimpleNamespace(
        admitted=True,
        eligible=True,
        use_gsim=True,
        selected_engine="gsim",
        to_dict=lambda: {"certificate_sha256": record.sha256, "scope": "fixture"},
    )
    calls = []

    def load(path, expected_sha256):
        assert path == record.path and expected_sha256 == record.sha256
        calls.append("certificate")
        return record

    monkeypatch.setattr(gsim_gate, "load_certificate", load)
    monkeypatch.setattr(paired_measurement, "gsim_workload", lambda member: {"synthetic": member.capsule})
    monkeypatch.setattr(gsim_gate, "plan_evaluation", lambda *_a, **_k: decision)
    rtl = DF.DevelopmentGsimFeedback(
        certificate=record,
        corpus=selected["component_corpus"],
        baseline=selected["baseline_admission"].baseline,
        baseline_sha256=selected["baseline_admission"].baseline_sha256,
        target_experiment=selected["target_experiment"],
        rtl_identity={"fixture": True},
        work_root=tmp_path / "host-rtl",
        decisions={("generated-family", "member"): decision},
        executor=no_engine_executor,
    )
    selected["independent_runtime"] = unissued_runtime(
        target_descriptor=selected["target_experiment"].path, rtl_executor=no_engine_executor,
    )
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **selected, component_rtl=rtl, feedback_round=0)
    return policy, rtl, calls


def test_optional_rtl_delegates_to_unchanged_redaction_gate(tmp_path, monkeypatch):
    from merlin_experiments.phase2 import development_feedback as DF

    policy, rtl, calls = _synthetic_rtl_policy(tmp_path, monkeypatch)
    observed = []

    def evaluate(self, candidate, *, round_index, call_index, timeout_s):
        observed.append((self is rtl, candidate, round_index, call_index, timeout_s))
        return {"golden": [1]}  # Deliberately invalid raw data: the unchanged validator must reject.

    monkeypatch.setattr(DF.DevelopmentGsimFeedback, "evaluate", evaluate)
    result, evidence = policy.execute({}, CW.RTL_ACTION, {}, 2, 10, time.monotonic())
    assert result["returncode"] == 125 and result["stdout"] == "" and evidence is None
    assert "golden" not in result["stderr"]
    assert len(observed) == 1 and observed[0][:4] == (True, policy.candidate, 0, 2)
    assert 0 < observed[0][4] <= 10
    assert len(calls) >= 2 and policy.unavailable.get(CW.RTL_ACTION) is None


@pytest.mark.parametrize("failure", ["config", "baseline", "decision", "target"])
def test_rtl_identity_and_complete_workload_decisions_cannot_drift(tmp_path, monkeypatch, failure):
    policy, rtl, _ = _synthetic_rtl_policy(tmp_path, monkeypatch)
    if failure == "config":
        rtl.rtl_identity = {"changed": True}
    elif failure == "baseline":
        (rtl.baseline / "compiler").write_text("changed\n")
    elif failure == "decision":
        rtl.decisions = {}
    else:
        rtl.certificate.target = "other"
    with pytest.raises(StageGateError):
        policy._validate()
