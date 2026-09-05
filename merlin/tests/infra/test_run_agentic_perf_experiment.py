"""Offline orchestration tests; no agent or simulator subprocess is launched."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(_SCRIPTS))
_SOURCE = _SCRIPTS / "run_agentic_perf_experiment.py"
_SPEC = importlib.util.spec_from_file_location("run_agentic_perf_experiment_under_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
ORCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ORCH
_SPEC.loader.exec_module(ORCH)


def _config(tmp_path: Path, *, max_cycles: int | None = 9000):
    price_table = tmp_path / "prices.yaml"
    if not price_table.exists():
        price_table.write_text("gpt-model: [1, 2, 0.1, 1]\n", encoding="utf-8")
        price_table.chmod(0o444)
    return ORCH.Config(
        experiment_id="exp", root=tmp_path / "experiment", functional_run_id="functional",
        functional_submission_sha256="a" * 64, descriptor=tmp_path / "target.yaml",
        rtl_facts=tmp_path / "rtl.json", perf_profile=tmp_path / "perf.yaml",
        gsim_certificate=tmp_path / "certificate.json", gsim_certificate_sha256="b" * 64,
        model="gpt-model", effort="high", wall_budget_seconds=60, rounds=2,
        round_timeout_seconds=30, max_tool_calls=5, tool_timeout_seconds=10,
        smoke_replicates=1, holdout_count=4, measurement_timeout=90,
        gsim_max_cycles=max_cycles,
        functional_gsim_certificate=tmp_path / "functional-certificate.json",
        functional_gsim_certificate_sha256="c" * 64,
        telemetry_price_table=price_table, chia_python=tmp_path / "chia-python")


def _mock_preflight_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    binary = tmp_path / "pinned-gsim"
    binary.write_bytes(b"exact pinned gsim")
    pins = {name: {"path": str(binary), "sha256": ORCH._sha_file(binary)}
            for name in ORCH.GATE.REQUIRED_PINS}
    certificate = SimpleNamespace(
        target="gemmini", sha256="b" * 64,
        pins=pins)
    monkeypatch.setattr(ORCH, "load_target_experiment",
                        lambda _path: SimpleNamespace(target="gemmini"))
    monkeypatch.setattr(
        ORCH, "_functional_grade_cohort",
        lambda _target: ORCH.FunctionalGradeCohort((), (), 1, 1))
    monkeypatch.setattr(ORCH.PAS, "inspect_stage_functional_run",
                        lambda *_args, **_kwargs: SimpleNamespace(
                            run_id="functional", digest="a" * 64))
    monkeypatch.setattr(ORCH.GATE, "load_certificate", lambda *_args, **_kwargs: certificate)
    monkeypatch.setattr(ORCH, "_verify_functional_certificate",
                        lambda *_args: {"public_descriptors": 1, "hidden_descriptors": 1})
    monkeypatch.setattr(ORCH, "_verify_functional_certificate_provenance",
                        lambda *_args: {"declaration_sha256": "7" * 64})
    monkeypatch.setattr(ORCH, "_verify_tuning_certificate",
                        lambda *_args: {"members": 1, "workload_sha256": ["e" * 64]})
    monkeypatch.setattr(ORCH.HOLDOUT, "derive_domain",
                        lambda *_args, **_kwargs: {"target": "gemmini", "legal": [1, 2, 3, 4]})
    monkeypatch.setattr(ORCH.HOLDOUT, "verify_rtl_facts_provenance",
                        lambda *_args, **_kwargs: {"replay_sha256": "d" * 64})
    telemetry_sources = {
        name: {"path": str(tmp_path / name), "sha256": f"{index:064x}"}
        for index, name in enumerate(sorted(ORCH.PAS.TELEMETRY_TREATMENT_SOURCES), start=1)}
    monkeypatch.setattr(ORCH.PAS, "telemetry_preflight", lambda **_kwargs: {
        "required": True, "driver": "codex", "billing_mode": "subscription_notional",
        "model_resolution": {"requested_model": "gpt-5.6-sol",
                             "resolved_model": "gpt-5.6-sol",
                             "codex_model_map": ""},
        "sources": telemetry_sources})
    monkeypatch.setattr(ORCH, "_chia_canary", lambda _path: {
        "available": True, "campaign_scheduler": "resume_safe_content_addressed_host_checkpoint_chain",
        "driver_parity_claim": False})
    return certificate


def test_preflight_declares_exactly_three_identical_trials_and_postseal_blocker(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_preflight_dependencies(tmp_path, monkeypatch)
    config = _config(tmp_path)
    blocked = ORCH.preflight(config)
    ready = ORCH.preflight(config, heldout_certificate_provider_available=True)

    assert blocked["status"] == "NO_GO"
    assert "post-seal" in blocked["blockers"][0]
    assert ready["status"] == "GO" and ready["blockers"] == []
    assert ready["trials"] == ["trial_00", "trial_01", "trial_02"]
    assert ready["replicates"] == list(ORCH.REPLICATES)
    assert ready["measurement_engine_policy"] == {
        "semantic_screen": "spike_no_timing",
        "rtl_execution_backends": ["gsim"],
        "timing_authority": "gsim",
        "verilator": "prelaunch_certificate_qualification_only",
    }
    contracts = list(ready["trial_contracts"].values())
    assert contracts[0] == contracts[1] == contracts[2]
    assert ready["selection"] == "all_trials_all_cells_no_best_of_no_drop"


def test_dry_run_never_calls_command_or_holdout_mutators(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_preflight_dependencies(tmp_path, monkeypatch)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("dry run launched a mutating action")

    result = ORCH.run(_config(tmp_path), dry_run=True, command_runner=forbidden,
                      commit_holdout=forbidden, reveal_holdout=forbidden)
    assert result["status"] == "GO"
    assert calls == [] and not (tmp_path / "experiment").exists()


def test_actual_launch_rejects_a_bare_random_chia_plan_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERLIN_CHIA_ENVELOPE_PLAN_SHA256", "a" * 64)
    monkeypatch.delenv("MERLIN_CHIA_LAUNCH_RECEIPT", raising=False)
    monkeypatch.delenv("MERLIN_CHIA_LAUNCH_RECEIPT_SHA256", raising=False)

    with pytest.raises(ORCH.ExperimentError, match="launch receipt"):
        ORCH._verify_chia_launch_receipt()


def test_chia_launch_receipt_attests_exact_command_sources_and_resources(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chia_source = tmp_path / "trace.py"
    chia_source.write_text("# pinned CHIA trace\n", encoding="utf-8")
    wrapper = (_SCRIPTS / "chia_agentic_perf_experiment.py").resolve()
    arguments = ["--experiment-id", "exp"]
    monkeypatch.setattr(sys, "argv", [str(_SOURCE), *arguments])
    command = [sys.executable, str(_SOURCE.resolve()), *arguments]
    command_artifacts = [
        {"index": index, "path": str(Path(command[index]).resolve()),
         "sha256": ORCH._sha_file(Path(command[index]))}
        for index in (0, 1)]
    wrapper_record = {"path": str(wrapper), "sha256": ORCH._sha_file(wrapper)}
    chia_record = {"path": str(chia_source.resolve()), "sha256": ORCH._sha_file(chia_source)}
    plan = {
        "schema_version": 1, "command": command, "command_artifacts": command_artifacts,
        "wrapper": wrapper_record, "chia_trace": chia_record,
    }
    plan["sha256"] = ORCH._sha_bytes(ORCH._canonical(plan))
    receipt = {
        "schema": "merlin.chia-agentic-perf-launch.v1",
        "status": "assigned_before_coordinator",
        "plan": plan, "plan_sha256": plan["sha256"], "command": command,
        "command_artifacts": command_artifacts,
        "required_resources": {"codex_slots": 1, "gsim_slots": 1},
        "assigned_resources": {"CPU": 1.0, "codex_slots": 1.0, "gsim_slots": 1.0},
        "wrapper": wrapper_record, "chia_trace": chia_record,
    }
    receipt_path = tmp_path / "launch.json"
    receipt_path.write_bytes(ORCH._canonical(receipt))
    receipt_path.chmod(0o444)
    receipt_sha = ORCH._sha_file(receipt_path)
    monkeypatch.setenv("MERLIN_CHIA_ENVELOPE_PLAN_SHA256", plan["sha256"])
    monkeypatch.setenv("MERLIN_CHIA_LAUNCH_RECEIPT", str(receipt_path))
    monkeypatch.setenv("MERLIN_CHIA_LAUNCH_RECEIPT_SHA256", receipt_sha)

    verified = ORCH._verify_chia_launch_receipt()
    assert verified["sha256"] == receipt_sha
    assert verified["assigned_resources"]["gsim_slots"] == 1.0
    assert verified["command"] == command
    assert verified["command_artifacts"] == command_artifacts

    receipt["assigned_resources"].pop("gsim_slots")
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(ORCH._canonical(receipt))
    receipt_path.chmod(0o444)
    monkeypatch.setenv("MERLIN_CHIA_LAUNCH_RECEIPT_SHA256", ORCH._sha_file(receipt_path))
    with pytest.raises(ORCH.ExperimentError, match="exact assigned invocation"):
        ORCH._verify_chia_launch_receipt()


def test_chia_resume_adopts_same_plan_across_new_receipt_location_and_assignment() -> None:
    saved = {
        "path": "/chia/run-one/launch.json", "sha256": "1" * 64,
        "plan_sha256": "2" * 64,
        "required_resources": {"codex_slots": 1, "gsim_slots": 1},
        "assigned_resources": {"CPU": 1.0, "codex_slots": 1.0, "gsim_slots": 1.0},
        "wrapper": {"path": "/repo/chia-wrapper.py", "sha256": "3" * 64},
        "chia_trace": {"path": "/chia/trace.py", "sha256": "4" * 64},
        "command": ["/repo/.venv/bin/python", "/repo/coordinator.py", "--root", "/run"],
        "command_artifacts": [
            {"index": 0, "path": "/python", "sha256": "5" * 64},
            {"index": 1, "path": "/repo/coordinator.py", "sha256": "6" * 64}],
    }
    current = {**saved, "path": "/chia/run-two/launch.json", "sha256": "7" * 64,
               "assigned_resources": {**saved["assigned_resources"], "node:abc": 1.0}}

    ORCH._verify_resume_chia_identity(saved, current)


@pytest.mark.parametrize("field", [
    "plan_sha256", "required_resources", "wrapper", "chia_trace", "command",
    "command_artifacts",
])
def test_chia_resume_refuses_predeclared_command_or_source_drift(field: str) -> None:
    saved = {
        "plan_sha256": "2" * 64,
        "required_resources": {"codex_slots": 1, "gsim_slots": 1},
        "wrapper": {"path": "/repo/chia-wrapper.py", "sha256": "3" * 64},
        "chia_trace": {"path": "/chia/trace.py", "sha256": "4" * 64},
        "command": ["/python", "/repo/coordinator.py", "--root", "/run"],
        "command_artifacts": [
            {"index": 0, "path": "/python", "sha256": "5" * 64},
            {"index": 1, "path": "/repo/coordinator.py", "sha256": "6" * 64}],
    }
    current = dict(saved)
    current[field] = "changed"

    with pytest.raises(ORCH.ExperimentError, match="saved predeclaration"):
        ORCH._verify_resume_chia_identity(saved, current)


def test_resume_refuses_any_saved_experiment_declaration_drift() -> None:
    original = {"schema": ORCH.SCHEMA, "agent_treatment": {"codex_binary_sha256": "a" * 64}}
    ORCH._verify_resume_declaration({"declaration": original}, dict(original))

    changed = {**original, "agent_treatment": {"codex_binary_sha256": "b" * 64}}
    with pytest.raises(ORCH.ExperimentError, match="saved predeclaration"):
        ORCH._verify_resume_declaration({"declaration": original}, changed)


def test_all_three_trial_handoffs_must_match_the_predeclared_treatment() -> None:
    sources = {name: f"{index:064x}" for index, name in enumerate(
        sorted(ORCH.PAS.TELEMETRY_TREATMENT_SOURCES), start=1)}
    expected = {
        "telemetry_preflight_sha256": "a" * 64,
        "codex_binary_sha256": sources["codex_binary"],
        "authoring_stage_sha256": sources["performance_authoring_stage"],
        "telemetry_source_sha256": sources,
        "requested_model": "gpt-5.6-sol",
        "resolved_model": "gpt-5.6-sol",
        "codex_model_map": "",
    }

    def handoff(**changes):
        return SimpleNamespace(agent_contract={
            "treatment_identity": {**expected, **changes}})

    complete = {trial: handoff() for trial in ORCH.TRIALS}
    ORCH._verify_trial_treatments(complete, expected)

    drifted = dict(complete)
    drifted["trial_02"] = handoff(codex_binary_sha256="f" * 64)
    with pytest.raises(ORCH.ExperimentError, match="trial_02.*predeclaration"):
        ORCH._verify_trial_treatments(drifted, expected)

    with pytest.raises(ORCH.ExperimentError, match="all three"):
        ORCH._verify_trial_treatments({"trial_00": handoff()}, expected)


def test_stale_stage_must_match_the_complete_predeclared_trial_contract() -> None:
    expected = {
        "model": "gpt-5.6-sol", "resolved_model": "gpt-5.6-sol", "effort": "high",
        "wall_budget_seconds": 4800, "rounds": 3, "round_timeout_seconds": 1600,
        "max_tool_calls": 300, "tool_timeout_seconds": 900, "smoke_replicates": 1,
        "measurement_replicates": 3, "functional_run_id": "functional",
        "functional_submission_sha256": "a" * 64, "telemetry_required": True,
        "telemetry_preflight_sha256": "b" * 64, "treatment_identity": {"pinned": True},
    }
    ORCH._verify_trial_contract(
        "trial_00", SimpleNamespace(agent_contract=dict(expected)), expected)

    for field, changed in (
            ("effort", "low"), ("wall_budget_seconds", 1), ("rounds", 1),
            ("functional_submission_sha256", "c" * 64),
            ("resolved_model", "gpt-5.5")):
        stale = {**expected, field: changed}
        with pytest.raises(ORCH.ExperimentError, match="predeclared trial contract"):
            ORCH._verify_trial_contract(
                "trial_00", SimpleNamespace(agent_contract=stale), expected)


def test_paid_trial_rechecks_live_treatment_before_launch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_preflight_dependencies(tmp_path, monkeypatch)
    config = _config(tmp_path)
    declaration = ORCH.preflight(config, heldout_certificate_provider_available=True)
    expected = declaration["agent_treatment"]
    ORCH._verify_live_agent_treatment(config, expected)

    original = ORCH.PAS.telemetry_preflight(model=config.model)
    drifted = json.loads(json.dumps(original))
    drifted["sources"]["codex_binary"]["sha256"] = "f" * 64
    monkeypatch.setattr(ORCH.PAS, "telemetry_preflight", lambda **_kwargs: drifted)
    with pytest.raises(ORCH.ExperimentError, match="saved predeclaration"):
        ORCH._verify_live_agent_treatment(config, expected)


def test_preflight_refuses_functional_certificate_from_a_different_gsim_build(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tuning = _mock_preflight_dependencies(tmp_path, monkeypatch)
    functional_pins = {name: dict(value) for name, value in tuning.pins.items()}
    functional_pins["gsim_binary"]["sha256"] = "9" * 64
    functional = SimpleNamespace(
        target="gemmini", sha256="c" * 64, pins=functional_pins)

    def load(path, **_kwargs):
        return functional if Path(path).name == "functional-certificate.json" else tuning

    monkeypatch.setattr(ORCH.GATE, "load_certificate", load)
    with pytest.raises(ORCH.ExperimentError, match="changed pinned build artifacts"):
        ORCH.preflight(_config(tmp_path), heldout_certificate_provider_available=True)


def _functional_qualification_fixture(tmp_path: Path, *, baseline_sha256: str):
    root = tmp_path / "qualification"
    root.mkdir()
    tuning_path = tmp_path / "tuning-certificate.json"
    tuning_path.write_text("{}\n", encoding="utf-8")
    pins = {name: {"path": str(tmp_path / name), "sha256": str(index) * 64}
            for index, name in enumerate(sorted(ORCH.GATE.REQUIRED_PINS), start=1)}
    tuning = SimpleNamespace(
        path=tuning_path, sha256=ORCH._sha_file(tuning_path), target="gemmini", pins=pins)
    workload = "8" * 64
    certificate_payload = ORCH._canonical({"certificate": "functional"})
    certificate_sha = ORCH._sha_bytes(certificate_payload)
    certificate_path = root / f"functional-certificate.{certificate_sha}.json"
    certificate_path.write_bytes(certificate_payload)
    certificate = SimpleNamespace(
        path=certificate_path, sha256=certificate_sha, target="gemmini", pins=pins,
        members={workload: {}})
    descriptor_path = tmp_path / "target.yaml"
    descriptor_path.write_text("target: gemmini\n", encoding="utf-8")
    declaration = {
        "schema": "merlin.functional-gsim-qualification.v1",
        "policy": "formal-public-plus-hidden-admission-distinct-workloads.v1",
        "target": "gemmini",
        "functional_baseline": {"sha256": baseline_sha256},
        "target_descriptor": {"path": str(descriptor_path.resolve()),
                              "sha256": ORCH._sha_file(descriptor_path)},
        "source_certificate": {
            "path": str(tuning_path.resolve()), "sha256": tuning.sha256,
            "pins": {name: pins[name]["sha256"] for name in sorted(pins)},
        },
        "cases": [{"workload_sha256": workload}],
    }
    declaration_payload = ORCH._canonical(declaration)
    declaration_sha = ORCH._sha_bytes(declaration_payload)
    (root / f"declaration.{declaration_sha}.json").write_bytes(declaration_payload)
    completion = {
        "schema": "merlin.functional-gsim-qualification.v1", "status": "complete",
        "declaration_sha256": declaration_sha,
        "source_certificate": {"path": str(tuning_path.resolve()), "sha256": tuning.sha256},
        "functional_certificate": {
            "path": str(certificate_path.resolve()), "sha256": certificate.sha256,
            "workload_sha256": [workload],
        },
    }
    completion_payload = ORCH._canonical(completion)
    completion_sha = ORCH._sha_bytes(completion_payload)
    completion_path = root / f"completion.{completion_sha}.json"
    completion_path.write_bytes(completion_payload)
    return certificate, tuning, completion_path


def test_functional_certificate_provenance_binds_exact_baseline(tmp_path: Path) -> None:
    baseline = "a" * 64
    certificate, tuning, _completion = _functional_qualification_fixture(
        tmp_path, baseline_sha256=baseline)
    evidence = ORCH._verify_functional_certificate_provenance(
        certificate, tuning, baseline)
    assert evidence["functional_submission_sha256"] == baseline

    with pytest.raises(ORCH.ExperimentError, match="exact sealed functional submission"):
        ORCH._verify_functional_certificate_provenance(certificate, tuning, "b" * 64)


def test_functional_certificate_provenance_refuses_tampered_completion(tmp_path: Path) -> None:
    baseline = "a" * 64
    certificate, tuning, completion = _functional_qualification_fixture(
        tmp_path, baseline_sha256=baseline)
    completion.write_bytes(completion.read_bytes() + b" \n")
    with pytest.raises(ORCH.ExperimentError, match="not content-addressed"):
        ORCH._verify_functional_certificate_provenance(certificate, tuning, baseline)


def test_functional_certificate_provenance_refuses_target_descriptor_drift(
        tmp_path: Path) -> None:
    baseline = "a" * 64
    certificate, tuning, _completion = _functional_qualification_fixture(
        tmp_path, baseline_sha256=baseline)
    (tmp_path / "target.yaml").write_text(
        "target: gemmini\n# changed after qualification\n", encoding="utf-8")

    with pytest.raises(ORCH.ExperimentError, match="exact sealed functional"):
        ORCH._verify_functional_certificate_provenance(certificate, tuning, baseline)


def test_child_environment_uses_certificate_pin_and_not_ambient_selection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    certificate = _mock_preflight_dependencies(tmp_path, monkeypatch)
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_EMU", "/ambient/wrong")
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_MAXCYCLES", "17")
    environment = ORCH.child_environment(_config(tmp_path), certificate)
    assert environment["MERLIN_GEMMINI_GSIM_EMU"] == certificate.pins["gsim_binary"]["path"]
    assert environment["MERLIN_REQUIRED_RTL_ENGINE"] == "gsim"
    assert environment["MERLIN_GEMMINI_GSIM_MAXCYCLES"] == "9000"
    no_cap = ORCH.child_environment(_config(tmp_path, max_cycles=None), certificate)
    assert "MERLIN_GEMMINI_GSIM_MAXCYCLES" not in no_cap


def test_post_reveal_qualification_receives_configured_gsim_cap_not_ambient(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def qualify(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path / "extension.json", "f" * 64

    monkeypatch.setattr(ORCH.HQUAL, "qualify_revealed_holdout", qualify)
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_MAXCYCLES", "17")
    config = _config(tmp_path, max_cycles=100_000_000)
    result = ORCH._qualify_heldout_with_config(
        tmp_path / "reveal.json", tmp_path / "qualification", SimpleNamespace(),
        functional_base=tmp_path / "functional", functional_base_sha256="a" * 64,
        reveal_manifest_sha256="b" * 64, reveal_corpus_sha256="c" * 64,
        config=config, target=SimpleNamespace(target="gemmini"))

    assert result == (tmp_path / "extension.json", "f" * 64)
    assert captured["gsim_max_cycles"] == 100_000_000
    assert captured["timeout"] == config.heldout_qualification_timeout


def test_checkpoints_are_append_only_content_addressed_and_resume_safe(tmp_path: Path) -> None:
    state = ORCH.Checkpoints(tmp_path / "state", "c" * 64)
    first = state.append("predeclared", {"x": 1})
    second = state.append("holdout_committed", {"y": 2})
    loaded = state.load()

    assert [row["stage"] for row in loaded] == ["predeclared", "holdout_committed"]
    assert second["previous_sha256"] == first["sha256"]
    assert all(Path(row["path"]).stat().st_mode & 0o222 == 0 for row in loaded)
    assert state.evidence("holdout_committed") == {"y": 2}
    with pytest.raises(ORCH.ExperimentError, match="duplicated"):
        state.append("holdout_committed", {"y": 3})


def test_uncheckpointed_child_artifact_adopts_complete_and_refuses_partial(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    final = attempt / "final.json"
    assert ORCH._uncheckpointed_state(attempt, final, label="child") == "absent"
    attempt.mkdir()
    with pytest.raises(ORCH.ExperimentError, match="partial; refusing in-place rerun"):
        ORCH._uncheckpointed_state(attempt, final, label="child")
    final.write_text("{}\n", encoding="utf-8")
    assert ORCH._uncheckpointed_state(attempt, final, label="child") == "complete"


def test_checkpointed_file_digest_is_rechecked_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("original\n", encoding="utf-8")
    saved = {"path": str(path), "sha256": ORCH._sha_file(path)}
    assert ORCH._verify_saved_file(saved, path, label="test") == path
    path.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ORCH.ExperimentError, match="changed across resume"):
        ORCH._verify_saved_file(saved, path, label="test")


def test_contract_inputs_are_content_addressed_readonly_snapshots(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.rtl_facts.write_text('{"facts": {}}\n', encoding="utf-8")
    config.perf_profile.write_text("profiles: []\n", encoding="utf-8")
    prepared, evidence = ORCH.snapshot_contract_inputs(config)

    assert prepared.rtl_facts != config.rtl_facts
    assert prepared.perf_profile != config.perf_profile
    for field in ("rtl_facts", "perf_profile"):
        snapshot = Path(evidence[field]["snapshot"])
        assert snapshot.is_file() and snapshot.stat().st_mode & 0o222 == 0
        assert ORCH._sha_file(snapshot) == evidence[field]["sha256"]
    resumed, resumed_evidence = ORCH.snapshot_contract_inputs(config)
    assert resumed == prepared and resumed_evidence == evidence


def test_functional_regrade_requires_candidate_digest_and_both_formal_phases(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "grade"
    run_dir.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("pass\n", encoding="utf-8")
    digest = str(ORCH.hash_tree(candidate)["sha256"])
    handoff = SimpleNamespace(candidate_path=candidate, candidate_sha256=digest)
    manifest = {"submission_sha256": digest,
                "completion": {"formal_grade_complete": True},
                "public_dev": {"formal_complete": True}, "hidden": {"formal_complete": True}}
    (run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    assert ORCH._verify_regrade(run_dir, handoff)["submission_sha256"] == digest
    manifest["hidden"]["formal_complete"] = False
    (run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ORCH.ExperimentError, match="full public.*hidden"):
        ORCH._verify_regrade(run_dir, handoff)


def test_measurement_adoption_checks_exact_trial_and_evidence_identities(tmp_path: Path) -> None:
    cells = tmp_path / "cells.json"
    raw_rows = [{
        "simulator": "gsim", "phase": "tuning", "family": "PK", "capsule": "pk0",
        "arm": "baseline", "replicate": "r000", "correct": True, "cycles": 10,
        "citable": True, "qualification": {"admitted": True},
        "provenance": {"tier": "L3", "simulator": "gsim", "cycle_accurate": True,
                       "oracle_kind": "rtl_gsim", "derived_from_rtl": True,
                       "elf_sha256": "3" * 64},
    }]
    cells.write_text(json.dumps({"cells": raw_rows}), encoding="utf-8")
    before = {"candidate_sha256": "e" * 64}
    expected_results = [{key: row[key] for key in
                         ("phase", "arm", "family", "capsule", "simulator", "replicate")}
                        for row in raw_rows]
    plan = {"expected_results": expected_results}
    completion = ORCH.PAIRED.completion_report(
        raw_rows, tuple(ORCH.PAIRED.ResultIdentity(**row) for row in expected_results))
    manifest = {
        "status": "GO", "phase": "tuning", "functional_run_id": "functional",
        "functional_submission_sha256": "a" * 64,
        "candidate_record_sha256": "d" * 64, "candidate_sha256": "e" * 64,
        "gsim_certificate": {"sha256": "f" * 64},
        "frozen_corpus": {"manifest_sha256": "1" * 64,
                          "capsules_sha256": "2" * 64, "visibility": "tuning"},
        "measurement_plan": plan,
        "measurement_plan_sha256": ORCH.PAIRED._sha256_bytes(
            ORCH.PAIRED._canonical_bytes(plan)),
        "completion": completion,
        "engine_policy": {"rtl_execution_backends": ["gsim"],
                          "timing_authority": "gsim",
                          "verilator": "prelaunch_certificate_qualification_only"},
        "identity_before": before,
        "identity_after": before, "fork_before": {"ok": True}, "fork_after": {"ok": True},
        "raw_results": {"paired_cells": str(cells),
                        "paired_cells_sha256": ORCH._sha_file(cells)},
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    handoff = SimpleNamespace(record_sha256="d" * 64, candidate_sha256="e" * 64)
    adopted = ORCH._verify_measurement_manifest(
        path, phase="tuning", functional_run_id="functional",
        functional_submission_sha256="a" * 64, handoff=handoff,
        corpus_manifest_sha256="1" * 64, corpus_capsules_sha256="2" * 64,
        certificate_sha256="f" * 64)
    assert adopted["sha256"] == ORCH._sha_file(path)

    manifest["candidate_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ORCH.ExperimentError, match="identity differs"):
        ORCH._verify_measurement_manifest(
            path, phase="tuning", functional_run_id="functional",
            functional_submission_sha256="a" * 64, handoff=handoff,
            corpus_manifest_sha256="1" * 64, corpus_capsules_sha256="2" * 64,
            certificate_sha256="f" * 64)


def test_stats_projection_keeps_every_gsim_cell_and_excludes_nonprimary(tmp_path: Path) -> None:
    cells_path = tmp_path / "cells.json"
    cells = [
        {"phase": "held_out", "arm": arm, "family": "PK", "capsule": "k17",
         "simulator": simulator, "replicate": replicate, "correct": True,
         "cycles": 100, "provenance": {"tier": "L3", "cycle_accurate": True,
                                        "oracle_kind": f"rtl_{simulator}",
                                        "derived_from_rtl": simulator != "spike"}}
        for arm in ("baseline", "candidate") for replicate in ORCH.REPLICATES
        for simulator in ("spike", "gsim", "verilator")
    ]
    cells_path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    manifest = {"status": "GO", "raw_results": {"paired_cells": str(cells_path),
                                                   "paired_cells_sha256": ORCH._sha_file(cells_path)}}
    manifest_path = tmp_path / "campaign.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows = ORCH._paired_rows(manifest_path, "trial_00")
    assert len(rows) == 2 * len(ORCH.REPLICATES)
    assert {row["identity"]["simulator"] for row in rows} == {"gsim"}
    assert {row["identity"]["family"] for row in rows} == {"held_out:PK"}


def test_statistics_predeclaration_projects_out_agent_evidence_hash() -> None:
    full = [{"trial": "trial_00", "agent_run_id": "run-0",
             "agent_evidence_sha256": "a" * 64}]
    assert ORCH._statistics_trials(full) == [
        {"trial": "trial_00", "agent_run_id": "run-0"}]


def _capsule(root: Path, name: str, *, k: int) -> Path:
    capsule = root / name
    capsule.mkdir(parents=True)
    manifest = capsule / "capsule.yaml"
    manifest.write_text(yaml.safe_dump({
        "name": name,
        "inputs": [
            {"name": "W", "role": "weight", "shape": [k, 16], "dtype": "i8"},
            {"name": "X", "role": "input", "shape": [16, k], "dtype": "i8"},
        ],
        "operation": {"op": "matmul", "attributes": {
            "lhs": "X", "weight": "W", "out": "Y0", "epilogue": [],
            "output_dtype": "i32"}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
    }, sort_keys=False), encoding="utf-8")
    return manifest


def _certificate(members: dict[str, dict], *, changed_pin: str | None = None):
    pins = {name: {"path": f"/{name}", "sha256": str(index) * 64}
            for index, name in enumerate(sorted(ORCH.GATE.REQUIRED_PINS), start=1)}
    if changed_pin is not None:
        pins[changed_pin] = {**pins[changed_pin], "sha256": "f" * 64}
    return SimpleNamespace(pins=pins, members=members)


def _revealed_corpus(root: Path, points: list[tuple[str, int]]) -> tuple[Path, list[Path]]:
    manifests = [_capsule(root / "_perf", name, k=k) for name, k in points]
    rows = [{"name": name, "path": f"_perf/{name}", "family": "PK",
             "cohort": "PK_predictor", "M": 16, "N": 16, "K": k}
            for name, k in points]
    tree = ORCH.HQUAL._tree_without_manifest(root, root / "holdout_manifest.json")
    document = {
        "schema_version": 2, "kind": "generated_performance_holdout_reveal",
        "domain": {"target": "gemmini"},
        "cohorts": {"PK_predictor": {"family": "PK", "member_count": len(rows)}},
        "members": rows, "corpus": tree,
    }
    manifest = root / "holdout_manifest.json"
    manifest.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)
    return manifest, manifests


def test_functional_certificate_must_cover_exact_public_and_hidden_descriptors(
        tmp_path: Path) -> None:
    public = tmp_path / "public"
    hidden = tmp_path / "hidden"
    manifests = [_capsule(public, "p0", k=17), _capsule(hidden, "h0", k=31)]
    identities = [ORCH.GATE.workload_sha256(ORCH.PAIRED.CERTPROD.derive_workload(path))
                  for path in manifests]
    members = tuple(ORCH.FunctionalCapsule(
        name=path.parent.name, kind="operator", manifest=path,
        manifest_sha256=ORCH._sha_file(path),
        workload_sha256=identity) for path, identity in zip(manifests, identities, strict=True))
    cohort = ORCH.FunctionalGradeCohort((members[0],), (members[1],), 1, 1)
    complete = _certificate({identity: {} for identity in identities})
    coverage = ORCH._verify_functional_certificate(complete, cohort)
    assert coverage["public_descriptors"] == 1
    assert coverage["hidden_descriptors"] == 1

    incomplete = _certificate({identities[0]: {}})
    with pytest.raises(ORCH.ExperimentError, match=r"exact admitted public\+hidden cohort"):
        ORCH._verify_functional_certificate(incomplete, cohort)

    extra = _certificate({**complete.members, "f" * 64: {}})
    with pytest.raises(ORCH.ExperimentError, match="extras"):
        ORCH._verify_functional_certificate(extra, cohort)


def test_real_functional_cohort_matches_canonical_descriptor_admission() -> None:
    target = ORCH.load_target_experiment(
        merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml")

    cohort = ORCH._functional_grade_cohort(target)
    full_identities = {
        capsule.workload_sha256 for capsule in (*cohort.public, *cohort.hidden)}
    gsim_cases = ORCH._functional_gsim_cases(cohort)
    certificate_identities = {capsule.workload_sha256 for capsule in gsim_cases}

    assert cohort.public_source_count == 48
    assert len(cohort.public) == 34
    assert cohort.hidden_source_count == 11
    assert len(cohort.hidden) == 10
    assert len(full_identities) == 33
    assert len(gsim_cases) == 42
    assert len(certificate_identities) == 31
    assert sum(capsule.kind == "model" for capsule in (*cohort.public, *cohort.hidden)) == 2
    assert not ({capsule.name for capsule in cohort.public} & set(target.graded_exclude))


def test_regrade_inputs_are_the_same_canonical_admitted_cohort() -> None:
    target = ORCH.load_target_experiment(
        merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml")
    cohort = ORCH._functional_grade_cohort(target)

    public_spec, hidden_spec = ORCH._functional_regrade_inputs(target, cohort)
    materialized = ORCH.discover_capsules(
        public_spec, labels={"public", "dev"}, contract=merlin_dir() / "contract")

    assert {str(cap["name"]) for cap in materialized} == {
        capsule.name for capsule in cohort.public}
    assert hidden_spec == ",".join(str(path) for path in target.hidden_roots())


def test_extension_certificate_retains_tuning_and_covers_exact_revealed_workloads(
    tmp_path: Path) -> None:
    holdout = tmp_path / "heldout"
    reveal, manifests = _revealed_corpus(holdout, [("k17", 17), ("k31", 31)])
    identities = [ORCH.GATE.workload_sha256(ORCH.PAIRED.CERTPROD.derive_workload(path))
                  for path in manifests]
    tuning = _certificate({"a" * 64: {"source": "tuning"}})
    extension = _certificate({**tuning.members,
                              **{identity: {"source": "heldout"}
                                 for identity in identities}})

    coverage = ORCH._verify_extension_certificate(tuning, extension, reveal)

    assert coverage["holdout_workload_sha256"] == sorted(identities)
    assert coverage["heldout_workloads_covered"] == 2
    assert coverage["pins_unchanged"] is True


def test_extension_certificate_rejects_missing_revealed_workload(tmp_path: Path) -> None:
    holdout = tmp_path / "heldout"
    reveal, (manifest,) = _revealed_corpus(holdout, [("k17", 17)])
    identity = ORCH.GATE.workload_sha256(ORCH.PAIRED.CERTPROD.derive_workload(manifest))
    tuning = _certificate({"a" * 64: {"source": "tuning"}})
    extension = _certificate(dict(tuning.members))

    with pytest.raises(ORCH.ExperimentError, match="does not cover every revealed workload"):
        ORCH._verify_extension_certificate(tuning, extension, reveal)
    assert identity not in extension.members


@pytest.mark.parametrize("changed_pin", ["gsim_model", "gsim_binary"])
def test_extension_certificate_rejects_changed_model_or_binary_pin(
        tmp_path: Path, changed_pin: str) -> None:
    holdout = tmp_path / "heldout"
    reveal, (manifest,) = _revealed_corpus(holdout, [("k17", 17)])
    identity = ORCH.GATE.workload_sha256(ORCH.PAIRED.CERTPROD.derive_workload(manifest))
    tuning = _certificate({"a" * 64: {"source": "tuning"}})
    extension = _certificate({**tuning.members, identity: {"source": "heldout"}},
                             changed_pin=changed_pin)

    with pytest.raises(ORCH.ExperimentError, match="changed pinned build artifacts"):
        ORCH._verify_extension_certificate(tuning, extension, reveal)


def test_extension_certificate_rejects_workloads_outside_exact_envelope(tmp_path: Path) -> None:
    holdout = tmp_path / "heldout"
    reveal, (manifest,) = _revealed_corpus(holdout, [("k17", 17)])
    identity = ORCH.GATE.workload_sha256(ORCH.PAIRED.CERTPROD.derive_workload(manifest))
    tuning = _certificate({"a" * 64: {"source": "tuning"}})
    extension = _certificate({**tuning.members, identity: {"source": "heldout"},
                              "b" * 64: {"source": "not revealed"}})

    with pytest.raises(ORCH.ExperimentError, match="outside the exact"):
        ORCH._verify_extension_certificate(tuning, extension, reveal)
