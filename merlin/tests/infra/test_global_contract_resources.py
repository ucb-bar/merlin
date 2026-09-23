"""Explicit contract resources stay pinned across native controller handoffs."""

import json
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest
from merlin_experiments.phase2.portfolio_sandbox import PortfolioSandboxFactory
from test_global_perf_experiment import (
    EA,
    P2_CONTRACTS,
    G,
    _portfolio_authoring,
    _synthetic_sandbox_inputs,
    setup_experiment,
)

from merlin.common.paths import repo_root
from merlin.perf.agent_guidance import inspect_compiler_package
from merlin.perf.analysis_worker import IsolatedAnalysisWorker
from merlin.perf.phase2_edit_contract import seal

RESOURCES = (
    "gate_phases.yaml",
    "hardware_pins.yaml",
    "schemas/command_buffer.schema.json",
    "schemas/manifest.schema.json",
)


@pytest.fixture(autouse=True)
def refuse_execution(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("contract resource tests must not launch processes or bind sockets")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


def copy_resources(destination):
    for relative in RESOURCES:
        selected = destination / relative
        selected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root() / "merlin/contract" / relative, selected)
    return destination


@pytest.fixture
def contract_root(tmp_path):
    return copy_resources(tmp_path / "selected-contract")


def test_policy_pins_selected_resources_without_default_resource_aliases(contract_root):
    policy = G.host_verification_policy_record(contract_root=contract_root)
    for relative in RESOURCES:
        selected = (contract_root / relative).resolve()
        assert policy["sources"][str(selected)] == P2_CONTRACTS.sha256_file(selected)
        assert str((repo_root() / "merlin/contract" / relative).resolve()) not in policy["sources"]


@pytest.mark.parametrize("resource", RESOURCES)
def test_selected_resource_mutation_refuses_before_analyzer(tmp_path, contract_root, resource):
    experiment, candidate, calls = setup_experiment(tmp_path, contract_root=contract_root)
    selected = contract_root / resource
    selected.write_bytes(selected.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="host verification policy changed"):
        experiment.analysis.analyze(candidate, hypothesis="Changed resources cannot reinterpret prior evidence")
    assert calls == []


@pytest.mark.parametrize("blocked", [False, True])
def test_selected_root_analysis_checkpoint_consumption_requires_same_root(tmp_path, contract_root, blocked):
    experiment, candidate, calls = setup_experiment(tmp_path, contract_root=contract_root, verified=not blocked)
    record = experiment.analysis.analyze(candidate, hypothesis="Bind the explicitly selected contract")
    checkpoint = (
        experiment.revision_session.checkpoint_authoring(candidate, name="blocked")
        if blocked
        else experiment.revision_session.seal(candidate)
    )
    consume = G.consume_authoring_checkpoint if blocked else G.consume_global_candidate
    result = consume(checkpoint, contract_root=contract_root)
    assert calls and result["candidate_sha256"] == record["candidate_sha256"]
    assert result["global_speedup_proven"] is False
    with pytest.raises(ValueError, match="host verification policy changed"):
        consume(checkpoint)
    other = copy_resources(contract_root.parent / "other-contract")
    with pytest.raises(ValueError, match="host verification policy changed"):
        consume(checkpoint, contract_root=other)


def test_external_contract_resources_are_explicitly_bound_and_reverified(tmp_path):
    with tempfile.TemporaryDirectory(prefix="merlin-contract-outside-owner-", dir="/tmp") as temporary:
        outside = copy_resources(Path(temporary))
        assert not outside.resolve().is_relative_to(repo_root().resolve())
        experiment, candidate, calls = setup_experiment(tmp_path, contract_root=outside)
        for relative in RESOURCES:
            selected = (outside / relative).resolve()
            assert experiment.inputs.host_policy["sources"][str(selected)] == P2_CONTRACTS.sha256_file(selected)
        record = experiment.analysis.analyze(
            candidate, hypothesis="Analyze with explicitly selected external resources"
        )
        checkpoint = experiment.revision_session.seal(candidate)
        consumed = G.consume_global_candidate(checkpoint, contract_root=outside)
        assert consumed["candidate_sha256"] == record["candidate_sha256"]
        assert consumed["host_verification_policy"]["sources"] == experiment.inputs.host_policy["sources"]
        assert consumed["global_speedup_proven"] is False
        calls_before = len(calls)
        selected_schema = outside / "schemas/command_buffer.schema.json"
        selected_schema.write_bytes(selected_schema.read_bytes() + b"\n")
        with pytest.raises(ValueError, match="host verification policy changed"):
            G.consume_global_candidate(checkpoint, contract_root=outside)
        with pytest.raises(ValueError, match="host verification policy changed"):
            experiment.analysis.analyze(candidate, hypothesis="Changed external schema must refuse before analysis")
        assert len(calls) == calls_before


@pytest.mark.parametrize("blocked", [False, True])
def test_sequence_retains_explicit_compiler_source_owner(tmp_path, contract_root, blocked):
    shared = tmp_path / "selected-shared-merlin"
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    (shared / "schedule.py").write_text("VALUE = 1\n")
    experiment, candidate, calls = setup_experiment(
        tmp_path, contract_root=contract_root, compiler_shared_source_root=shared, verified=not blocked
    )
    (candidate / "compiler.py").write_text("from merlin.schedule import VALUE\n")

    def author(current, *, round_index, round_timeout_s):
        (current / "source.txt").write_text(f"revision {round_index}")
        experiment.analysis.analyze(current, hypothesis="Keep the selected compiler source owner")
        return {"status": "authored"}

    result = _portfolio_authoring(experiment, stage_root=tmp_path / "stage", run_round=author).run_sequence(
        candidate, max_rounds=2, total_authoring_seconds=60, round_seconds=30
    )
    checkpoint = Path(result["last_good_checkpoint"]["path"])
    consume = G.consume_authoring_checkpoint if blocked else G.consume_global_candidate
    consumed = consume(checkpoint, contract_root=contract_root, compiler_shared_source_root=shared)
    assert len(calls) == 3 and result["failures"] == []
    assert consumed["compiler_dependencies"]["shared_source_root"] == str(shared.resolve())
    with pytest.raises(ValueError, match="dependencies|comparison baseline"):
        consume(checkpoint, contract_root=contract_root)


def test_production_worker_configuration_retains_selected_root(tmp_path, contract_root, monkeypatch):
    experiment, _, calls = setup_experiment(tmp_path, contract_root=contract_root)
    experiment.analysis.analyzer = EA.analyze_whole_model_emission
    experiment.analysis.completion_contract = {"synthetic": True}
    monkeypatch.setattr(
        PortfolioSandboxFactory, "__call__", lambda *args: pytest.fail("configuration must not execute a sandbox")
    )
    G.configure_global_analysis(
        experiment,
        target_experiment=object(),
        agent_inputs=object(),
        frozen_functional=object(),
        frozen_corpus_manifest=tmp_path / "unused-manifest.json",
        stage_root=tmp_path / "stage",
        sandbox_inputs=_synthetic_sandbox_inputs(tmp_path),
    )
    assert isinstance(experiment.analysis.analyzer, IsolatedAnalysisWorker)
    assert experiment.analysis.analyzer.contract_root == contract_root.resolve()
    configured = experiment.analysis.analyzer.sandbox_factory
    assert isinstance(configured, PortfolioSandboxFactory)
    assert configured.analysis is experiment.analysis
    assert configured.analysis.session.inputs.contract_root == contract_root.resolve()
    assert (
        configured.analysis.session.inputs.compiler_shared_source_root == experiment.inputs.compiler_shared_source_root
    )
    assert experiment.analysis._member_analyzer().contract_root == contract_root.resolve()
    assert calls == []


def test_direct_analysis_receives_selected_contract_and_schema(tmp_path, contract_root, monkeypatch):
    experiment, candidate, _ = setup_experiment(tmp_path, contract_root=contract_root)
    delegate = experiment.analysis.analyzer
    observed = []

    def analyzer(*args, **kwargs):
        observed.append(kwargs)
        return delegate(*args, **kwargs)

    monkeypatch.setattr(EA, "analyze_whole_model_emission", analyzer)
    experiment.analysis.analyzer = analyzer
    experiment.analysis.analyze(candidate, hypothesis="Inspect selected analysis resources")
    assert len(observed) == 1
    assert observed[0]["contract_root"] == contract_root.resolve()
    assert Path(observed[0]["compiler_api_schema"]["path"]) == contract_root / "schemas/command_buffer.schema.json"


@pytest.mark.parametrize("recover", [False, True])
def test_sequence_consumption_and_recovery_keep_selected_root(tmp_path, contract_root, recover):
    experiment, candidate, _ = setup_experiment(tmp_path, contract_root=contract_root)
    inherited = []

    def author(current, *, round_index, round_timeout_s):
        inherited.append((current / "source.txt").read_text())
        (current / "source.txt").write_text(f"draft {round_index}")
        experiment.analysis.analyze(current, hypothesis="Synthetic sequence analysis")
        if recover and round_index == 0:
            experiment._write(
                "agent_round_0000.json",
                {
                    "agent_exit_code": -15,
                    "audit": {"clean": True, "broker_invocations": []},
                    "broker_evidence": {"all_required_succeeded": True},
                },
            )
            raise ValueError("synthetic recoverable author failure")
        return {"status": "authored"}

    result = _portfolio_authoring(experiment, stage_root=tmp_path / "stage", run_round=author).run_sequence(
        candidate, max_rounds=2, total_authoring_seconds=60, round_seconds=30, on_round_failure="resume-last-checkpoint"
    )
    assert inherited == ["candidate", "candidate" if recover else "draft 0"]
    assert len(result["failures"]) == int(recover)
    consumed = G.consume_global_candidate(Path(result["last_good_checkpoint"]["path"]), contract_root=contract_root)
    assert consumed["global_speedup_proven"] is False


@pytest.mark.parametrize("frozen", [False, True])
def test_native_guidance_uses_selected_manifest_vocabulary(tmp_path, contract_root, frozen):
    schema_path = contract_root / "schemas/manifest.schema.json"
    schema = json.loads(schema_path.read_text())
    properties = schema["properties"]["optimization_surfaces"]["items"]["properties"]
    properties["scope"]["enum"].append("selected_schedule")
    schema_path.write_text(json.dumps(schema))
    experiment, candidate, _ = setup_experiment(tmp_path, contract_root=contract_root)
    declaration = {
        "id": "schedule",
        "scope": "selected_schedule",
        "path": "compiler.py",
        "symbol": "schedule",
        "effects": ["movement"],
        "mechanism": "Select a source schedule",
        "emitted_delta": "Changed issue order",
        "validation": "Static dependency proof",
        "abandonment": "Invalid dependencies",
    }
    (candidate / "compiler.py").write_text("def schedule():\n    return 1\n")
    (candidate / "manifest.yaml").write_text(
        json.dumps(
            {
                "components": {"emit": ["compiler.py"]},
                "optimization_surfaces": [declaration],
            }
        )
    )
    with pytest.raises(ValueError, match="invalid id/scope/symbol"):
        inspect_compiler_package(candidate)
    if frozen:
        contract = seal(
            {
                "schema": "compiler_edit_contract_v1",
                "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
                "helper_extensions": [],
            }
        )
        experiment.freeze_edit_scope(candidate, contract, host_surface_declarations=[declaration])
    inventory = experiment.revision_session.inspect_optimization_surfaces(candidate)
    assert inventory["surfaces"][0]["scope"] == "selected_schedule"
    assert inventory["surfaces"][0]["symbol"] == "schedule"
    if frozen:
        assert experiment.edit_authority.guidance_inventory.to_dict()["surfaces"][0]["scope"] == "selected_schedule"
        assert inventory["host_guidance_binding"]["permission_scope"] == "unchanged host-frozen edit contract"
