"""Installed worker admission and helper semantics without authoring or native tools."""

import hashlib
import json
import socket
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from merlin_experiments.phase2 import portfolio_options as O
from merlin_experiments.phase2 import portfolio_worker as W


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("worker tests cannot launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


def policy(tmp_path):
    return dict(
        target_experiment=object(),
        agent_inputs=object(),
        frozen_functional=object(),
        frozen_corpus_manifest=tmp_path / "corpus.json",
        sandbox_inputs=object(),
    )


@pytest.mark.parametrize("ready", [False, True])
def test_analysis_only_configures_imports_seed_and_analyzes_without_sealing(tmp_path, monkeypatch, ready):
    events = []
    candidate = tmp_path / "candidate"
    seed = tmp_path / "seed.json"

    def analyze(path, **kwargs):
        assert path == candidate
        assert kwargs["hypothesis"] == "Host-requested full-objective compile/static preflight"
        events.append("analyze")
        return dict(
            readiness={"status": "ready_for_probe_admission" if ready else "blocked"},
            candidate_sha256="a" * 64,
            portfolio={"members_ready": int(ready), "members_total": 1},
            iteration=2,
        )

    def import_seed(path, **kwargs):
        assert path == candidate
        assert kwargs == {"checkpoint": seed, "checkpoint_sha256": "b" * 64}
        events.append("seed")

    experiment = SimpleNamespace(
        analysis=SimpleNamespace(analyze=analyze),
        output=tmp_path / "iterations",
        inputs=SimpleNamespace(
            portfolio_identity_sha256="c" * 64, baseline_sha256="d" * 64, optimization_baseline_sha256="e" * 64
        ),
        static_analysis_import=SimpleNamespace(import_checkpoint=import_seed),
    )
    selected = policy(tmp_path)

    def configure(actual, **kwargs):
        assert actual is experiment
        assert kwargs == {"stage_root": tmp_path, **selected}
        events.append("configure")

    monkeypatch.setattr(W, "configure_global_analysis", configure)
    assert W.run_analysis_only(
        experiment,
        candidate,
        stage_root=tmp_path,
        static_analysis_seed_checkpoint=seed,
        static_analysis_seed_sha256="b" * 64,
        **selected,
    ) == (0 if ready else 1)
    assert events == ["configure", "seed", "analyze"]
    receipt = json.loads((tmp_path / "analysis_only.json").read_text())
    for field in (
        "phase1_rerun",
        "authoring_launched",
        "simulators_executed",
        "candidate_sealed",
        "global_speedup_proven",
    ):
        assert receipt[field] is False
    assert receipt["objective_numerical_qualification"] == "UNPROVEN"
    assert receipt["iteration_record"] == str(tmp_path / "iterations/iteration_0002.json")


@pytest.mark.parametrize("failure_stage", ["configure", "sequence"])
def test_terminal_receipt_preserves_original_failure_and_completed_evidence(tmp_path, failure_stage):
    evidence = tmp_path / "global_iterations"
    evidence.mkdir()
    (evidence / "agent_round_0001.json").write_text("{}")
    iteration = evidence / "iteration_0001.json"
    iteration.write_text("{}")
    error = ValueError("original failure")

    def action(stage):
        if stage == failure_stage:
            raise error

    with pytest.raises(ValueError) as caught:
        W.run_authoring_with_terminal_receipt(
            tmp_path,
            configure=lambda: action("configure"),
            sequence=lambda: action("sequence"),
        )
    assert caught.value is error
    receipt = json.loads((tmp_path / "terminal_failure.json").read_text())
    assert receipt["stage"] == (
        "configure_global_analysis" if failure_stage == "configure" else "initial_analysis_or_authoring_sequence"
    )
    assert receipt["completed_round_receipts"] == 1
    assert receipt["iteration_receipts"] == [str(iteration)]
    assert receipt["promotion_status"] == "unqualified"


def test_guidance_requires_exact_explicit_inventory_pin(tmp_path):
    contract = tmp_path / "contract.json"
    inventory = tmp_path / "inventory.json"
    raw = json.dumps({"surfaces": [{"id": "declared"}]}).encode()
    inventory.write_bytes(raw)
    assert W.load_host_guidance_declarations(contract, {}) is None
    pin = {"guidance_inventory_sha256": hashlib.sha256(raw).hexdigest()}
    assert W.load_host_guidance_declarations(contract, pin) == [{"id": "declared"}]
    inventory.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="pinned catalog bytes"):
        W.load_host_guidance_declarations(contract, pin)


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_baseline_identity_is_never_relabelled(legacy):
    checkpoint = {"baseline_sha256" if legacy else "optimization_baseline_sha256": "a" * 64}
    W.validate_optimization_baseline_resume(checkpoint, optimization_baseline_sha256="a" * 64)
    with pytest.raises(ValueError, match="differs"):
        W.validate_optimization_baseline_resume(checkpoint, optimization_baseline_sha256="b" * 64)


def test_instruction_derivation_failure_is_explicit_unknown(monkeypatch):
    from merlin.perf import task_instruction_evidence as evidence

    def unavailable(target):
        raise ValueError("synthetic unavailable facts")

    monkeypatch.setattr(evidence, "target_instruction_facts", unavailable)
    result = W.declared_instruction_set_brief("synthetic")
    assert result["status"] == "UNKNOWN"
    assert result["declared_count"] is None
    assert "synthetic unavailable facts" in result["reason"]


@pytest.mark.parametrize("selection", ["unregistered", "absent", "valid", "malformed", "raises"])
def test_completion_capability_is_selected_without_silent_declared_failure(tmp_path, monkeypatch, selection):
    selected = ModuleType("synthetic_backend")
    expected = {"schema": "synthetic completion"}
    if selection in {"valid", "raises"}:
        adapter = ModuleType("explicit_completion")

        def derive():
            if selection == "raises":
                raise RuntimeError("declared completion failed")
            return expected

        adapter.derive_completion_contract = derive
        selected.completion_contract = adapter
    elif selection == "malformed":
        selected.completion_contract = None

    def lookup(target):
        assert target == "synthetic"
        if selection == "unregistered":
            raise KeyError(target)
        return selected

    monkeypatch.setattr(W, "get_backend", lookup)
    experiment = SimpleNamespace(
        inputs=SimpleNamespace(target="synthetic"),
        analysis=SimpleNamespace(completion_contract=None, analyzer=object()),
    )
    if selection in {"malformed", "raises"}:
        with pytest.raises(ValueError if selection == "malformed" else RuntimeError):
            W.configure_global_analysis(experiment, stage_root=tmp_path, **policy(tmp_path))
    else:
        W.configure_global_analysis(experiment, stage_root=tmp_path, **policy(tmp_path))
        assert experiment.analysis.completion_contract == (expected if selection == "valid" else None)


def test_worker_descriptor_mismatch_refuses_before_output_or_functional_inspection(tmp_path, monkeypatch):
    from merlin_experiments import source_snapshot as snapshot
    from merlin_experiments.phase1.providers import agent_bridge

    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic\n")
    invocation = O.parse_invocation(
        [
            "--campaign-config",
            str(tmp_path / "config.json"),
            "--candidate",
            str(tmp_path / "candidate"),
            "--output",
            str(tmp_path / "stage"),
        ]
    )
    context = W.PortfolioWorkerContext(
        snapshot_root=tmp_path,
        functional_runs_root=tmp_path / "functional",
        contract_root=tmp_path / "contract",
        compiler_shared_source_root=tmp_path / "shared",
        controller_source=tmp_path / "controller.py",
        prior_shared_source_relative=Path("shared"),
        prior_shared_source_fallback=tmp_path / "shared",
        guidance_contract=None,
        sandbox_inputs=object(),
        target_experiment=SimpleNamespace(path=descriptor, descriptor_sha256="0" * 64),
    )
    monkeypatch.setattr(snapshot, "verify", lambda root: {"schema": snapshot.SCHEMA})
    monkeypatch.setattr(snapshot, "remap_input", lambda root, receipt, path, **kwargs: path)
    monkeypatch.setattr(agent_bridge, "bind_frozen_proxy_config", lambda *args: None)
    monkeypatch.setattr(W.FI, "inspect_stage_functional_run", lambda *a, **k: pytest.fail("premature functional read"))
    with pytest.raises(ValueError, match="descriptor differs"):
        W.run(invocation, {"descriptor": str(descriptor)}, context=context)
    assert not invocation.args.output.exists()
