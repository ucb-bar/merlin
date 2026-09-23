"""Offline orchestration tests; no agent or simulator subprocess is launched."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import checkpoint_admission as AD
from merlin_experiments.phase2 import checkpoint_controller as CTRL
from merlin_experiments.phase2 import chia_launch as CHIA
from merlin_experiments.phase2 import corpus as P2_CORPUS
from merlin_experiments.phase2 import functional_cohort as FC
from merlin_experiments.phase2 import gsim_workload as WORKLOAD
from merlin_experiments.phase2 import heldout_qualification as HQUAL
from merlin_experiments.phase2 import measurement_evidence as ME
from merlin_experiments.phase2 import paired_measurement as PME
from merlin_experiments.phase2 import revealed_corpus as RC

from merlin.common.paths import merlin_dir, module_source_path, repo_root

pytestmark = pytest.mark.target("gemmini")

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
_SOURCE = _SCRIPTS / "run_agentic_perf_experiment.py"


def test_chia_canary_uses_shared_host_transport(tmp_path, monkeypatch):
    from merlin_experiments import frozen_python

    trace = tmp_path / "trace.py"
    trace.write_text("# synthetic import-canary source\n")
    seen = []
    monkeypatch.setattr(AD, "_resolve_chia_python", lambda _: Path(sys.executable))

    def transport(argv):
        seen.append(argv)
        return ["guarded-canary", *argv]

    def run(argv, **kwargs):
        assert argv == ["guarded-canary", *seen[0]]
        assert kwargs["timeout"] == 30
        return SimpleNamespace(returncode=0, stdout=json.dumps({"chia_trace": str(trace), "ray": "fixture"}))

    monkeypatch.setattr(frozen_python, "inherited_python_command", transport)
    monkeypatch.setattr(AD.subprocess, "run", run)
    result = AD._chia_canary(None, context=_config(tmp_path).context)
    assert len(seen) == 1
    assert seen[0][:2] == [sys.executable, "-c"]
    assert result["chia_trace_sha256"] == AD._sha_file(trace)
    assert result["ray_version"] == "fixture"


def _config(tmp_path: Path, *, max_cycles: int | None = 9000):
    price_table = tmp_path / "prices.yaml"
    if not price_table.exists():
        price_table.write_text("gpt-model: [1, 2, 0.1, 1]\n", encoding="utf-8")
        price_table.chmod(0o444)
    return AD.Config(
        context=AD.ExecutionContext(
            source_root=repo_root(),
            contract_root=merlin_dir() / "contract",
            functional_runs_root=tmp_path / "functional-runs",
            stage_root=tmp_path / "stages",
            measurement_root=tmp_path / "measurements",
            holdout_sources=AD.HOLDOUT.HoldoutSourceContext(
                source_root=repo_root(),
                catalog_path=repo_root() / "experiments/catalog.yaml",
                core_package_root=module_source_path("merlin").parent,
                experiments_package_root=module_source_path("merlin_experiments").parent,
                experiments_namespace_root=module_source_path("merlin.targetgen.capsule_runner").parents[1],
            ),
            chia_wrapper=_SCRIPTS / "chia_agentic_perf_experiment.py",
            invocation=(sys.executable, str(_SOURCE)),
            suite="synthetic-test",
        ),
        experiment_id="exp",
        root=tmp_path / "experiment",
        functional_run_id="functional",
        functional_submission_sha256="a" * 64,
        descriptor=tmp_path / "target.yaml",
        rtl_facts=tmp_path / "rtl.json",
        perf_profile=tmp_path / "perf.yaml",
        gsim_certificate=tmp_path / "certificate.json",
        gsim_certificate_sha256="b" * 64,
        model="gpt-model",
        effort="high",
        wall_budget_seconds=60,
        rounds=2,
        round_timeout_seconds=30,
        max_tool_calls=5,
        tool_timeout_seconds=10,
        smoke_replicates=1,
        holdout_count=4,
        measurement_timeout=90,
        gsim_max_cycles=max_cycles,
        functional_gsim_certificate=tmp_path / "functional-certificate.json",
        functional_gsim_certificate_sha256="c" * 64,
        telemetry_price_table=price_table,
        chia_python=tmp_path / "chia-python",
    )


def _mock_preflight_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from merlin.runtime.backends import base as backends

    def runtime_environment(*, binaries, gsim_max_cycles, environment):
        result = dict(environment)
        result["SYNTHETIC_GSIM"] = str(binaries["gsim"])
        result["SYNTHETIC_REFERENCE"] = str(binaries["verilator"])
        if gsim_max_cycles is None:
            result.pop("SYNTHETIC_MAXCYCLES", None)
        else:
            result["SYNTHETIC_MAXCYCLES"] = str(gsim_max_cycles)
        return result

    monkeypatch.setattr(
        backends, "get_backend", lambda target: SimpleNamespace(runtime_environment=runtime_environment)
    )
    binary = tmp_path / "pinned-gsim"
    binary.write_bytes(b"exact pinned gsim")
    pins = {name: {"path": str(binary), "sha256": AD._sha_file(binary)} for name in AD.GATE.REQUIRED_PINS}
    certificate = SimpleNamespace(target="gemmini", sha256="b" * 64, pins=pins)
    monkeypatch.setattr(AD, "load_target_experiment", lambda _path, **kw: SimpleNamespace(target="gemmini"))
    monkeypatch.setattr(
        FC,
        "functional_grade_cohort_from_run",
        lambda _target, _functional, **_kw: FC.FunctionalGradeCohort((), (), 1, 1),
    )
    monkeypatch.setattr(
        AD.FI,
        "inspect_stage_functional_run",
        lambda *_args, **_kwargs: SimpleNamespace(run_id="functional", digest="a" * 64),
    )
    monkeypatch.setattr(AD.GATE, "load_certificate", lambda *_args, **_kwargs: certificate)
    monkeypatch.setattr(
        AD, "_verify_functional_certificate", lambda *_args: {"public_descriptors": 1, "hidden_descriptors": 1}
    )
    monkeypatch.setattr(
        AD, "_verify_functional_certificate_provenance", lambda *_args: {"declaration_sha256": "7" * 64}
    )
    monkeypatch.setattr(
        AD,
        "_functional_qualification_descriptor",
        lambda *_args: (tmp_path / "frozen-target.yaml", {"sha256": "8" * 64}),
    )
    monkeypatch.setattr(AD, "_verify_tuning_certificate", lambda *_args: {"members": 1, "workload_sha256": ["e" * 64]})
    monkeypatch.setattr(
        AD.HOLDOUT, "derive_domain", lambda *_args, **_kwargs: {"target": "gemmini", "legal": [1, 2, 3, 4]}
    )
    monkeypatch.setattr(
        AD.HOLDOUT, "verify_rtl_facts_provenance", lambda *_args, **_kwargs: {"replay_sha256": "d" * 64}
    )
    telemetry_sources = {
        name: {"path": str(tmp_path / name), "sha256": f"{index:064x}"}
        for index, name in enumerate(sorted(AD.TEL.TREATMENT_SOURCES), start=1)
    }
    telemetry_sources["performance_package_sources"] = AD.TEL._package_source_record()
    monkeypatch.setattr(
        AD.TEL,
        "prepare",
        lambda **_kwargs: {
            "schema_version": 4,
            "source_policy_version": 4,
            "required": True,
            "driver": "codex",
            "billing_mode": "subscription_notional",
            "model_resolution": {
                "requested_model": "gpt-5.6-sol",
                "resolved_model": "gpt-5.6-sol",
                "codex_model_map": "",
            },
            "sources": telemetry_sources,
        },
    )
    monkeypatch.setattr(
        AD,
        "_chia_canary",
        lambda _path, **_kwargs: {
            "available": True,
            "campaign_scheduler": "resume_safe_content_addressed_host_checkpoint_chain",
            "driver_parity_claim": False,
        },
    )
    return certificate


def test_preflight_declares_exactly_three_identical_trials_and_postseal_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_preflight_dependencies(tmp_path, monkeypatch)
    config = _config(tmp_path)
    blocked = AD.preflight(config)
    ready = AD.preflight(config, heldout_certificate_provider_available=True)

    assert blocked["status"] == "NO_GO"
    assert "post-seal" in blocked["blockers"][0]
    assert ready["status"] == "GO" and ready["blockers"] == []
    assert ready["trials"] == ["trial_00", "trial_01", "trial_02"]
    assert ready["replicates"] == list(AD.REPLICATES)
    assert ready["measurement_engine_policy"] == {
        "semantic_screen": "spike_no_timing",
        "rtl_execution_backends": ["gsim"],
        "timing_authority": "gsim",
        "verilator": "prelaunch_certificate_qualification_only",
    }
    contracts = list(ready["trial_contracts"].values())
    assert contracts[0] == contracts[1] == contracts[2]
    assert ready["selection"] == "all_trials_all_cells_no_best_of_no_drop"


def test_dry_run_never_calls_command_or_holdout_mutators(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_preflight_dependencies(tmp_path, monkeypatch)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("dry run launched a mutating action")

    result = CTRL.run(
        _config(tmp_path), dry_run=True, command_runner=forbidden, commit_holdout=forbidden, reveal_holdout=forbidden
    )
    assert result["status"] == "GO"
    assert calls == [] and not (tmp_path / "experiment").exists()


def test_actual_launch_rejects_a_bare_random_chia_plan_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERLIN_CHIA_ENVELOPE_PLAN_SHA256", "a" * 64)
    monkeypatch.delenv("MERLIN_CHIA_LAUNCH_RECEIPT", raising=False)
    monkeypatch.delenv("MERLIN_CHIA_LAUNCH_RECEIPT_SHA256", raising=False)

    with pytest.raises(AD.ExperimentError, match="launch receipt"):
        CHIA.verify_launch_receipt(
            command=[sys.executable, str(_SOURCE)],
            wrapper=_SCRIPTS / "chia_agentic_perf_experiment.py",
            environment=os.environ,
        )


def test_chia_launch_receipt_attests_exact_command_sources_and_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chia_source = tmp_path / "trace.py"
    chia_source.write_text("# pinned CHIA trace\n", encoding="utf-8")
    wrapper = (_SCRIPTS / "chia_agentic_perf_experiment.py").resolve()
    arguments = ["--experiment-id", "exp"]
    monkeypatch.setattr(sys, "argv", [str(_SOURCE), *arguments])
    command = [sys.executable, str(_SOURCE.resolve()), *arguments]
    command_artifacts = [
        {"index": index, "path": str(Path(command[index]).resolve()), "sha256": AD._sha_file(Path(command[index]))}
        for index in (0, 1)
    ]
    wrapper_record = {"path": str(wrapper), "sha256": AD._sha_file(wrapper)}
    chia_record = {"path": str(chia_source.resolve()), "sha256": AD._sha_file(chia_source)}
    plan = {
        "schema_version": 1,
        "command": command,
        "command_artifacts": command_artifacts,
        "wrapper": wrapper_record,
        "chia_trace": chia_record,
        "launch_policy": CHIA.policy_identity(),
    }
    plan["sha256"] = AD._sha_bytes(AD._canonical(plan))
    receipt = {
        "schema": "merlin.chia-agentic-perf-launch.v2",
        "status": "assigned_before_coordinator",
        "plan": plan,
        "plan_sha256": plan["sha256"],
        "command": command,
        "command_artifacts": command_artifacts,
        "required_resources": {"codex_slots": 1, "gsim_slots": 1},
        "assigned_resources": {"CPU": 1.0, "codex_slots": 1.0, "gsim_slots": 1.0},
        "wrapper": wrapper_record,
        "chia_trace": chia_record,
        "launch_policy": CHIA.policy_identity(),
    }
    receipt_path = tmp_path / "launch.json"
    receipt_path.write_bytes(AD._canonical(receipt))
    receipt_path.chmod(0o444)
    receipt_sha = AD._sha_file(receipt_path)
    monkeypatch.setenv("MERLIN_CHIA_ENVELOPE_PLAN_SHA256", plan["sha256"])
    monkeypatch.setenv("MERLIN_CHIA_LAUNCH_RECEIPT", str(receipt_path))
    monkeypatch.setenv("MERLIN_CHIA_LAUNCH_RECEIPT_SHA256", receipt_sha)

    verified = CHIA.verify_launch_receipt(command=command, wrapper=wrapper, environment=os.environ)
    assert verified["sha256"] == receipt_sha
    assert verified["assigned_resources"]["gsim_slots"] == 1.0
    assert verified["command"] == command
    assert verified["command_artifacts"] == command_artifacts

    receipt["assigned_resources"].pop("gsim_slots")
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(AD._canonical(receipt))
    receipt_path.chmod(0o444)
    monkeypatch.setenv("MERLIN_CHIA_LAUNCH_RECEIPT_SHA256", AD._sha_file(receipt_path))
    with pytest.raises(AD.ExperimentError, match="exact assigned invocation"):
        CHIA.verify_launch_receipt(command=command, wrapper=wrapper, environment=os.environ)


def test_chia_resume_adopts_same_plan_across_new_receipt_location_and_assignment() -> None:
    saved = {
        "path": "/chia/run-one/launch.json",
        "sha256": "1" * 64,
        "plan_sha256": "2" * 64,
        "required_resources": {"codex_slots": 1, "gsim_slots": 1},
        "assigned_resources": {"CPU": 1.0, "codex_slots": 1.0, "gsim_slots": 1.0},
        "wrapper": {"path": "/repo/chia-wrapper.py", "sha256": "3" * 64},
        "chia_trace": {"path": "/chia/trace.py", "sha256": "4" * 64},
        "command": ["/repo/.venv/bin/python", "/repo/coordinator.py", "--root", "/run"],
        "command_artifacts": [
            {"index": 0, "path": "/python", "sha256": "5" * 64},
            {"index": 1, "path": "/repo/coordinator.py", "sha256": "6" * 64},
        ],
    }
    current = {
        **saved,
        "path": "/chia/run-two/launch.json",
        "sha256": "7" * 64,
        "assigned_resources": {**saved["assigned_resources"], "node:abc": 1.0},
    }

    AD._verify_resume_chia_identity(saved, current)


@pytest.mark.parametrize(
    "field",
    [
        "plan_sha256",
        "required_resources",
        "wrapper",
        "chia_trace",
        "command",
        "command_artifacts",
    ],
)
def test_chia_resume_refuses_predeclared_command_or_source_drift(field: str) -> None:
    saved = {
        "plan_sha256": "2" * 64,
        "required_resources": {"codex_slots": 1, "gsim_slots": 1},
        "wrapper": {"path": "/repo/chia-wrapper.py", "sha256": "3" * 64},
        "chia_trace": {"path": "/chia/trace.py", "sha256": "4" * 64},
        "command": ["/python", "/repo/coordinator.py", "--root", "/run"],
        "command_artifacts": [
            {"index": 0, "path": "/python", "sha256": "5" * 64},
            {"index": 1, "path": "/repo/coordinator.py", "sha256": "6" * 64},
        ],
    }
    current = dict(saved)
    current[field] = "changed"

    with pytest.raises(AD.ExperimentError, match="saved predeclaration"):
        AD._verify_resume_chia_identity(saved, current)


def test_resume_refuses_any_saved_experiment_declaration_drift() -> None:
    original = {"schema": AD.SCHEMA, "agent_treatment": {"codex_binary_sha256": "a" * 64}}
    AD._verify_resume_declaration({"declaration": original}, dict(original))

    changed = {**original, "agent_treatment": {"codex_binary_sha256": "b" * 64}}
    with pytest.raises(AD.ExperimentError, match="saved predeclaration"):
        AD._verify_resume_declaration({"declaration": original}, changed)


def test_all_three_trial_handoffs_must_match_the_predeclared_treatment() -> None:
    sources = {name: f"{index:064x}" for index, name in enumerate(sorted(AD.TEL.TREATMENT_SOURCES), start=1)}
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
        return SimpleNamespace(agent_contract={"treatment_identity": {**expected, **changes}})

    complete = {trial: handoff() for trial in AD.TRIALS}
    AD._verify_trial_treatments(complete, expected)

    drifted = dict(complete)
    drifted["trial_02"] = handoff(codex_binary_sha256="f" * 64)
    with pytest.raises(AD.ExperimentError, match="trial_02.*predeclaration"):
        AD._verify_trial_treatments(drifted, expected)

    with pytest.raises(AD.ExperimentError, match="all three"):
        AD._verify_trial_treatments({"trial_00": handoff()}, expected)


def test_stale_stage_must_match_the_complete_predeclared_trial_contract() -> None:
    expected = {
        "model": "gpt-5.6-sol",
        "resolved_model": "gpt-5.6-sol",
        "effort": "high",
        "wall_budget_seconds": 4800,
        "rounds": 3,
        "round_timeout_seconds": 1600,
        "max_tool_calls": 300,
        "tool_timeout_seconds": 900,
        "smoke_replicates": 1,
        "measurement_replicates": 3,
        "functional_run_id": "functional",
        "functional_submission_sha256": "a" * 64,
        "telemetry_required": True,
        "telemetry_preflight_sha256": "b" * 64,
        "treatment_identity": {"pinned": True},
    }
    AD._verify_trial_contract("trial_00", SimpleNamespace(agent_contract=dict(expected)), expected)

    for field, changed in (
        ("effort", "low"),
        ("wall_budget_seconds", 1),
        ("rounds", 1),
        ("functional_submission_sha256", "c" * 64),
        ("resolved_model", "gpt-5.5"),
    ):
        stale = {**expected, field: changed}
        with pytest.raises(AD.ExperimentError, match="predeclared trial contract"):
            AD._verify_trial_contract("trial_00", SimpleNamespace(agent_contract=stale), expected)


def test_paid_trial_rechecks_live_treatment_before_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_preflight_dependencies(tmp_path, monkeypatch)
    config = _config(tmp_path)
    declaration = AD.preflight(config, heldout_certificate_provider_available=True)
    expected = declaration["agent_treatment"]
    AD._verify_live_agent_treatment(config, expected)

    original = AD.TEL.prepare(authoring_stage=Path(AD.AUTHORING.__file__).resolve(), model=config.model)
    drifted = json.loads(json.dumps(original))
    drifted["sources"]["codex_binary"]["sha256"] = "f" * 64
    monkeypatch.setattr(AD.TEL, "prepare", lambda **_kwargs: drifted)
    with pytest.raises(AD.ExperimentError, match="saved predeclaration"):
        AD._verify_live_agent_treatment(config, expected)


def test_author_trials_use_installed_cli_with_explicit_resources(tmp_path, monkeypatch):
    config = _config(tmp_path)
    commands = []
    target = SimpleNamespace(target="fixture-target")
    monkeypatch.setattr(AD, "_verify_live_agent_treatment", lambda *_: None)
    monkeypatch.setattr(CTRL, "_uncheckpointed_state", lambda *_args, **_kwargs: "absent")
    monkeypatch.setattr(CTRL, "_run_checked", lambda runner, command, **kwargs: commands.append(command))

    class StopAfterLaunch(Exception):
        pass

    def launch_only(stages, *, workers):
        for stage in stages:
            stage.launch()
        raise StopAfterLaunch

    monkeypatch.setattr(CTRL, "run_child_stages", launch_only)
    with pytest.raises(StopAfterLaunch):
        CTRL._author_candidates(
            config,
            SimpleNamespace(evidence=lambda _: None),
            target,
            {},
            environment={},
            expected_treatment={},
            command_runner=None,
            workers=1,
        )
    assert len(commands) == len(AD.TRIALS)
    for trial, command in zip(AD.TRIALS, commands, strict=True):
        assert command[:3] == [sys.executable, "-m", "merlin_experiments.phase2.authoring_cli"]
        expected = {
            "--suite": config.context.suite,
            "--functional-runs-root": str(config.context.functional_runs_root),
            "--stage-root": str(config.context.stage_root / f"exp__{trial}"),
            "--source-root": str(config.context.source_root),
            "--contract-root": str(config.context.contract_root),
        }
        for flag, value in expected.items():
            assert command[command.index(flag) + 1] == value
        assert "--run-id" not in command


def test_preflight_refuses_functional_certificate_from_a_different_gsim_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tuning = _mock_preflight_dependencies(tmp_path, monkeypatch)
    functional_pins = {name: dict(value) for name, value in tuning.pins.items()}
    functional_pins["gsim_binary"]["sha256"] = "9" * 64
    functional = SimpleNamespace(target="gemmini", sha256="c" * 64, pins=functional_pins)

    def load(path, **_kwargs):
        return functional if Path(path).name == "functional-certificate.json" else tuning

    monkeypatch.setattr(AD.GATE, "load_certificate", load)
    with pytest.raises(AD.ExperimentError, match="changed pinned build artifacts"):
        AD.preflight(_config(tmp_path), heldout_certificate_provider_available=True)


def _functional_qualification_fixture(tmp_path: Path, *, baseline_sha256: str):
    from merlin_experiments.phase2 import contracts, functional_qualification

    root = tmp_path / "qualification"
    root.mkdir()
    tuning_path = tmp_path / "tuning-certificate.json"
    tuning_path.write_text("{}\n", encoding="utf-8")
    pins = {
        name: {"path": str(tmp_path / name), "sha256": str(index) * 64}
        for index, name in enumerate(sorted(AD.GATE.REQUIRED_PINS), start=1)
    }
    tuning = SimpleNamespace(path=tuning_path, sha256=AD._sha_file(tuning_path), target="gemmini", pins=pins)
    workload = "8" * 64
    certificate_payload = AD._canonical({"certificate": "functional"})
    certificate_sha = AD._sha_bytes(certificate_payload)
    certificate_path = root / f"functional-certificate.{certificate_sha}.json"
    certificate_path.write_bytes(certificate_payload)
    certificate = SimpleNamespace(
        path=certificate_path, sha256=certificate_sha, target="gemmini", pins=pins, members={workload: {}}, document={}
    )
    descriptor_path = tmp_path / "target.yaml"
    descriptor_path.write_text("target: gemmini\n", encoding="utf-8")
    contract = root / "inputs" / "contract"
    contract.mkdir(parents=True)
    (contract / "schema.json").write_text("{}\n")
    (contract / "schema.json").chmod(0o444)
    contract.chmod(0o555)
    declaration = {
        "schema": functional_qualification.SCHEMA,
        "contract_snapshot": {"path": str(contract), **contracts.exact_tree_record(contract)},
        "policy": "formal-public-plus-hidden-admission-distinct-workloads.v1",
        "target": "gemmini",
        "functional_baseline": {"sha256": baseline_sha256},
        "target_descriptor": {"path": str(descriptor_path.resolve()), "sha256": AD._sha_file(descriptor_path)},
        "source_certificate": {
            "path": str(tuning_path.resolve()),
            "sha256": tuning.sha256,
            "pins": {name: pins[name]["sha256"] for name in sorted(pins)},
        },
        "cases": [{"workload_sha256": workload}],
    }
    declaration_payload = AD._canonical(declaration)
    declaration_sha = AD._sha_bytes(declaration_payload)
    (root / f"declaration.{declaration_sha}.json").write_bytes(declaration_payload)
    completion = {
        "schema": functional_qualification.SCHEMA,
        "status": "complete",
        "declaration_sha256": declaration_sha,
        "source_certificate": {"path": str(tuning_path.resolve()), "sha256": tuning.sha256},
        "functional_certificate": {
            "path": str(certificate_path.resolve()),
            "sha256": certificate.sha256,
            "workload_sha256": [workload],
        },
    }
    completion_payload = AD._canonical(completion)
    completion_sha = AD._sha_bytes(completion_payload)
    completion_path = root / f"completion.{completion_sha}.json"
    completion_path.write_bytes(completion_payload)
    return certificate, tuning, completion_path


def test_functional_certificate_provenance_binds_exact_baseline(tmp_path: Path) -> None:
    baseline = "a" * 64
    certificate, tuning, _completion = _functional_qualification_fixture(tmp_path, baseline_sha256=baseline)
    evidence = AD._verify_functional_certificate_provenance(certificate, tuning, baseline)
    assert evidence["functional_submission_sha256"] == baseline

    with pytest.raises(AD.ExperimentError, match="exact sealed functional submission"):
        AD._verify_functional_certificate_provenance(certificate, tuning, "b" * 64)


def test_functional_certificate_provenance_refuses_tampered_completion(tmp_path: Path) -> None:
    baseline = "a" * 64
    certificate, tuning, completion = _functional_qualification_fixture(tmp_path, baseline_sha256=baseline)
    completion.write_bytes(completion.read_bytes() + b" \n")
    with pytest.raises(AD.ExperimentError, match="not content-addressed"):
        AD._verify_functional_certificate_provenance(certificate, tuning, baseline)


def test_functional_certificate_provenance_refuses_target_descriptor_drift(tmp_path: Path) -> None:
    baseline = "a" * 64
    certificate, tuning, _completion = _functional_qualification_fixture(tmp_path, baseline_sha256=baseline)
    (tmp_path / "target.yaml").write_text("target: gemmini\n# changed after qualification\n", encoding="utf-8")

    with pytest.raises(AD.ExperimentError, match="exact sealed functional"):
        AD._verify_functional_certificate_provenance(certificate, tuning, baseline)


def test_child_environment_uses_certificate_pin_and_not_ambient_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    certificate = _mock_preflight_dependencies(tmp_path, monkeypatch)
    monkeypatch.setenv("SYNTHETIC_GSIM", "/ambient/wrong")
    monkeypatch.setenv("SYNTHETIC_MAXCYCLES", "17")
    environment = AD.child_environment(_config(tmp_path), certificate)
    assert environment["SYNTHETIC_GSIM"] == certificate.pins["gsim_binary"]["path"]
    assert environment["SYNTHETIC_REFERENCE"] == certificate.pins["verilator_binary"]["path"]
    assert environment["MERLIN_REQUIRED_RTL_ENGINE"] == "gsim"
    assert environment["MERLIN_CACHE_STATE"] == "warm"
    assert environment["SYNTHETIC_MAXCYCLES"] == "9000"
    no_cap = AD.child_environment(_config(tmp_path, max_cycles=None), certificate)
    assert "SYNTHETIC_MAXCYCLES" not in no_cap


@pytest.mark.parametrize("engine", ["gsim", "verilator"])
def test_child_environment_refuses_changed_binary_before_provider(tmp_path, monkeypatch, engine):
    from merlin.runtime.backends import base as backends

    certificate = _mock_preflight_dependencies(tmp_path, monkeypatch)
    certificate.pins[f"{engine}_binary"]["sha256"] = "0" * 64
    monkeypatch.setattr(
        backends,
        "get_backend",
        lambda target: SimpleNamespace(
            runtime_environment=lambda **kw: pytest.fail("invalid binary reached provider policy")
        ),
    )
    with pytest.raises(AD.ExperimentError, match=f"runtime {engine} binary"):
        AD.child_environment(_config(tmp_path), certificate)


def test_child_environment_requires_explicit_provider_capability(tmp_path, monkeypatch):
    from merlin.runtime.backends import base as backends

    certificate = _mock_preflight_dependencies(tmp_path, monkeypatch)
    monkeypatch.setattr(backends, "get_backend", lambda target: SimpleNamespace())
    with pytest.raises(AD.ExperimentError, match="pure runtime environment"):
        AD.child_environment(_config(tmp_path), certificate)


def test_post_reveal_qualification_receives_configured_gsim_cap_not_ambient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def qualify(*args, **kwargs):
        captured.update(kwargs)
        return tmp_path / "extension.json", "f" * 64

    monkeypatch.setattr(HQUAL, "qualify_revealed_holdout", qualify)
    monkeypatch.setenv("MERLIN_GEMMINI_GSIM_MAXCYCLES", "17")
    config = _config(tmp_path, max_cycles=100_000_000)
    result = CTRL._qualify_heldout_with_config(
        tmp_path / "reveal.json",
        tmp_path / "qualification",
        SimpleNamespace(),
        functional_base=tmp_path / "functional",
        functional_base_sha256="a" * 64,
        reveal_manifest_sha256="b" * 64,
        reveal_corpus_sha256="c" * 64,
        config=config,
        target=SimpleNamespace(target="gemmini"),
    )

    assert result == (tmp_path / "extension.json", "f" * 64)
    assert captured["gsim_max_cycles"] == 100_000_000
    assert captured["timeout"] == config.heldout_qualification_timeout


def test_checkpoints_are_append_only_content_addressed_and_resume_safe(tmp_path: Path) -> None:
    state = AD.Checkpoints(tmp_path / "state", "c" * 64)
    first = state.append("predeclared", {"x": 1})
    second = state.append("holdout_committed", {"y": 2})
    loaded = state.load()

    assert [row["stage"] for row in loaded] == ["predeclared", "holdout_committed"]
    assert second["previous_sha256"] == first["sha256"]
    assert all(Path(row["path"]).stat().st_mode & 0o222 == 0 for row in loaded)
    assert state.evidence("holdout_committed") == {"y": 2}
    with pytest.raises(AD.ExperimentError, match="duplicated"):
        state.append("holdout_committed", {"y": 3})


def test_uncheckpointed_child_artifact_adopts_complete_and_refuses_partial(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    final = attempt / "final.json"
    assert CTRL._uncheckpointed_state(attempt, final, label="child") == "absent"
    attempt.mkdir()
    with pytest.raises(AD.ExperimentError, match="partial; refusing in-place rerun"):
        CTRL._uncheckpointed_state(attempt, final, label="child")
    final.write_text("{}\n", encoding="utf-8")
    assert CTRL._uncheckpointed_state(attempt, final, label="child") == "complete"


def test_checkpointed_file_digest_is_rechecked_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("original\n", encoding="utf-8")
    saved = {"path": str(path), "sha256": AD._sha_file(path)}
    assert CTRL._verify_saved_file(saved, path, label="test") == path
    path.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(AD.ExperimentError, match="changed across resume"):
        CTRL._verify_saved_file(saved, path, label="test")


def test_contract_inputs_are_content_addressed_readonly_snapshots(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.rtl_facts.write_text('{"facts": {}}\n', encoding="utf-8")
    config.perf_profile.write_text("profiles: []\n", encoding="utf-8")
    prepared, evidence = AD.snapshot_contract_inputs(config)

    assert prepared.rtl_facts != config.rtl_facts
    assert prepared.perf_profile != config.perf_profile
    for field in ("rtl_facts", "perf_profile"):
        snapshot = Path(evidence[field]["snapshot"])
        assert snapshot.is_file() and snapshot.stat().st_mode & 0o222 == 0
        assert AD._sha_file(snapshot) == evidence[field]["sha256"]
    resumed, resumed_evidence = AD.snapshot_contract_inputs(config)
    assert resumed == prepared and resumed_evidence == evidence


def test_functional_regrade_requires_candidate_digest_and_both_formal_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "grade"
    run_dir.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("pass\n", encoding="utf-8")
    digest = str(CTRL.hash_tree(candidate)["sha256"])
    handoff = SimpleNamespace(candidate_path=candidate, candidate_sha256=digest)
    manifest = {
        "submission_sha256": digest,
        "completion": {"formal_grade_complete": True},
        "public_dev": {"formal_complete": True},
        "hidden": {"formal_complete": True},
    }
    (run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    assert CTRL._verify_regrade(run_dir, handoff)["submission_sha256"] == digest
    manifest["hidden"]["formal_complete"] = False
    (run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(AD.ExperimentError, match="full public.*hidden"):
        CTRL._verify_regrade(run_dir, handoff)


def test_measurement_adoption_checks_exact_trial_and_evidence_identities(tmp_path: Path) -> None:
    cells = tmp_path / "cells.json"
    raw_rows = [
        {
            "simulator": "gsim",
            "phase": "tuning",
            "family": "PK",
            "capsule": "pk0",
            "arm": "baseline",
            "replicate": "r000",
            "correct": True,
            "cycles": 10,
            "citable": True,
            "qualification": {"admitted": True},
            "provenance": {
                "tier": "L3",
                "simulator": "gsim",
                "cycle_accurate": True,
                "oracle_kind": "rtl_gsim",
                "derived_from_rtl": True,
                "elf_sha256": "3" * 64,
            },
        }
    ]
    cells.write_text(json.dumps({"schema": "paired_arm4_result_cells_v2", "cells": raw_rows}), encoding="utf-8")
    before = {"candidate_sha256": "e" * 64}
    expected_results = [
        {key: row[key] for key in ("phase", "arm", "family", "capsule", "simulator", "replicate")} for row in raw_rows
    ]
    plan = {"schema": "paired_arm4_measurement_plan_v3", "expected_results": expected_results}
    completion = ME.completion_report(raw_rows, tuple(ME.ResultIdentity(**row) for row in expected_results))
    manifest = {
        "schema": "paired_arm4_performance_campaign_v2",
        "status": "GO",
        "phase": "tuning",
        "functional_run_id": "functional",
        "functional_submission_sha256": "a" * 64,
        "candidate_record_sha256": "d" * 64,
        "candidate_sha256": "e" * 64,
        "gsim_certificate": {"sha256": "f" * 64},
        "frozen_corpus": {"manifest_sha256": "1" * 64, "capsules_sha256": "2" * 64, "visibility": "tuning"},
        "measurement_plan": plan,
        "measurement_plan_sha256": PME.sha256_bytes(PME.canonical_bytes(plan)),
        "completion": completion,
        "engine_policy": {
            "rtl_execution_backends": ["gsim"],
            "timing_authority": "gsim",
            "verilator": "prelaunch_certificate_qualification_only",
        },
        "identity_before": before,
        "identity_after": before,
        "fork_before": {"ok": True},
        "fork_after": {"ok": True},
        "raw_results": {"paired_cells": str(cells), "paired_cells_sha256": AD._sha_file(cells)},
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    handoff = SimpleNamespace(record_sha256="d" * 64, candidate_sha256="e" * 64)
    adopted = CTRL._verify_measurement_manifest(
        path,
        phase="tuning",
        functional_run_id="functional",
        functional_submission_sha256="a" * 64,
        handoff=handoff,
        corpus_manifest_sha256="1" * 64,
        corpus_capsules_sha256="2" * 64,
        certificate_sha256="f" * 64,
    )
    assert adopted["sha256"] == AD._sha_file(path)

    manifest["candidate_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AD.ExperimentError, match="identity differs"):
        CTRL._verify_measurement_manifest(
            path,
            phase="tuning",
            functional_run_id="functional",
            functional_submission_sha256="a" * 64,
            handoff=handoff,
            corpus_manifest_sha256="1" * 64,
            corpus_capsules_sha256="2" * 64,
            certificate_sha256="f" * 64,
        )


def test_stats_projection_keeps_every_gsim_cell_and_excludes_nonprimary(tmp_path: Path) -> None:
    cells_path = tmp_path / "cells.json"
    cells = [
        {
            "phase": "held_out",
            "arm": arm,
            "family": "PK",
            "capsule": "k17",
            "simulator": simulator,
            "replicate": replicate,
            "correct": True,
            "cycles": 100,
            "provenance": {
                "tier": "L3",
                "cycle_accurate": True,
                "oracle_kind": f"rtl_{simulator}",
                "derived_from_rtl": simulator != "spike",
            },
        }
        for arm in ("baseline", "candidate")
        for replicate in AD.REPLICATES
        for simulator in ("spike", "gsim", "verilator")
    ]
    cells_path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    manifest = {
        "status": "GO",
        "raw_results": {"paired_cells": str(cells_path), "paired_cells_sha256": AD._sha_file(cells_path)},
    }
    manifest_path = tmp_path / "campaign.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    rows = ME.read_statistics_rows(manifest_path, trial="trial_00")
    assert len(rows) == 2 * len(AD.REPLICATES)
    assert {row["identity"]["simulator"] for row in rows} == {"gsim"}
    assert {row["identity"]["family"] for row in rows} == {"held_out:PK"}


def test_statistics_predeclaration_projects_out_agent_evidence_hash() -> None:
    full = [{"trial": "trial_00", "agent_run_id": "run-0", "agent_evidence_sha256": "a" * 64}]
    assert CTRL._statistics_trials(full) == [{"trial": "trial_00", "agent_run_id": "run-0"}]


def _capsule(root: Path, name: str, *, k: int) -> Path:
    capsule = root / name
    capsule.mkdir(parents=True)
    manifest = capsule / "capsule.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "inputs": [
                    {"name": "W", "role": "weight", "shape": [k, 16], "dtype": "i8"},
                    {"name": "X", "role": "input", "shape": [16, k], "dtype": "i8"},
                ],
                "operation": {
                    "op": "matmul",
                    "attributes": {"lhs": "X", "weight": "W", "out": "Y0", "epilogue": [], "output_dtype": "i32"},
                },
                "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return manifest


def _certificate(members: dict[str, dict], *, changed_pin: str | None = None):
    pins = {
        name: {"path": f"/{name}", "sha256": str(index) * 64}
        for index, name in enumerate(sorted(AD.GATE.REQUIRED_PINS), start=1)
    }
    if changed_pin is not None:
        pins[changed_pin] = {**pins[changed_pin], "sha256": "f" * 64}
    return SimpleNamespace(pins=pins, members=members, document={})


def _revealed_corpus(root: Path, points: list[tuple[str, int]]) -> tuple[Path, list[Path]]:
    manifests = [_capsule(root / "_perf", name, k=k) for name, k in points]
    rows = [
        {"name": name, "path": f"_perf/{name}", "family": "PK", "cohort": "PK_predictor", "M": 16, "N": 16, "K": k}
        for name, k in points
    ]
    tree = RC.tree_without_manifest(root, root / "holdout_manifest.json")
    document = {
        "schema_version": 2,
        "kind": "generated_performance_holdout_reveal",
        "domain": {"target": "gemmini"},
        "cohorts": {"PK_predictor": {"family": "PK", "member_count": len(rows)}},
        "members": rows,
        "corpus": tree,
    }
    manifest = root / "holdout_manifest.json"
    manifest.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)
    return manifest, manifests


def test_functional_certificate_must_cover_exact_public_and_hidden_descriptors(tmp_path: Path) -> None:
    public = tmp_path / "public"
    hidden = tmp_path / "hidden"
    manifests = [_capsule(public, "p0", k=17), _capsule(hidden, "h0", k=31)]
    identities = [AD.GATE.workload_sha256(WORKLOAD.derive_workload(path)) for path in manifests]
    members = tuple(
        FC.FunctionalCapsule(
            name=path.parent.name,
            kind="operator",
            manifest=path,
            manifest_sha256=AD._sha_file(path),
            workload_sha256=identity,
        )
        for path, identity in zip(manifests, identities, strict=True)
    )
    cohort = FC.FunctionalGradeCohort((members[0],), (members[1],), 1, 1)
    complete = _certificate({identity: {} for identity in identities})
    coverage = AD._verify_functional_certificate(complete, cohort)
    assert coverage["public_descriptors"] == 1
    assert coverage["hidden_descriptors"] == 1

    incomplete = _certificate({identities[0]: {}})
    with pytest.raises(AD.ExperimentError, match=r"exact admitted public\+hidden cohort"):
        AD._verify_functional_certificate(incomplete, cohort)

    extra = _certificate({**complete.members, "f" * 64: {}})
    with pytest.raises(AD.ExperimentError, match="extras"):
        AD._verify_functional_certificate(extra, cohort)


def _require_real_corpus_fixture(target) -> None:
    """Skip absent generated/private inputs, never an admission failure.

    Existing descriptors are parsed first so malformed available data still
    fails. An excluded name alone is not evidence of a missing fixture: require
    its authored model loader before classifying its absent manifest as such.
    """
    public_roots, hidden_roots = target.graded_roots(), target.hidden_roots()
    required_hidden_roots = list(hidden_roots)
    if getattr(target, "hidden_expected_source_capsules", 0) and not hidden_roots:
        # TargetExperiment.hidden_roots filters absent sibling resources. The
        # descriptor's nonzero seal makes that conventional resource required.
        required_hidden_roots.append(target.capsule_corpus.parent / "hidden")
    contract = merlin_dir() / "contract"
    FC.discover_capsules(public_roots, labels={"public", "dev"}, contract=contract)
    FC.discover_capsules(hidden_roots, labels={"hidden"}, contract=contract)
    missing = []
    for name in target.graded_exclude:
        for root in public_roots:
            directory = Path(root) / name
            manifest = directory / "capsule.yaml"
            if (directory / "capsule.pytorch.py").is_file() and not manifest.exists() and not manifest.is_symlink():
                missing.append(str(manifest))
    for root in required_hidden_roots:
        path = Path(root)
        if not path.exists() and not path.is_symlink():
            missing.append(str(path))
    if missing:
        pytest.skip("real corpus fixture unavailable; missing declared generated/private inputs: " + ", ".join(missing))


def test_missing_exclusion_still_refuses_available_synthetic_corpus(tmp_path) -> None:
    public, hidden = tmp_path / "public", tmp_path / "hidden"
    public.mkdir()
    hidden.mkdir()
    target = SimpleNamespace(
        graded_roots=lambda: [public], hidden_roots=lambda: [hidden], graded_exclude=("missing-member",)
    )
    _require_real_corpus_fixture(target)
    with pytest.raises(FC.ExperimentError, match="exclusions name absent capsules"):
        FC.functional_grade_cohort(target, contract_root=merlin_dir() / "contract")


def test_real_corpus_prerequisite_check_does_not_skip_malformed_available_data(tmp_path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / "capsule.yaml").write_text("[malformed", encoding="utf-8")
    target = SimpleNamespace(
        graded_roots=lambda: [public], hidden_roots=lambda: [tmp_path / "absent-private"], graded_exclude=()
    )
    with pytest.raises(yaml.YAMLError):
        _require_real_corpus_fixture(target)


def test_real_functional_cohort_matches_canonical_descriptor_admission() -> None:
    target = AD.load_target_experiment(
        merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    )

    _require_real_corpus_fixture(target)
    cohort = FC.functional_grade_cohort(target, contract_root=merlin_dir() / "contract")
    full_identities = {capsule.workload_sha256 for capsule in (*cohort.public, *cohort.hidden)}
    gsim_cases = FC.functional_gsim_cases(cohort)
    certificate_identities = {capsule.workload_sha256 for capsule in gsim_cases}

    # The descriptor was re-sealed after corpus expansion. Test its authoritative counts,
    # not a historical denominator that predates the frozen functional submission.
    assert cohort.public_source_count == target.graded_expected_source_capsules
    assert len(cohort.public) == target.graded_expected_admitted_capsules
    assert cohort.hidden_source_count == target.hidden_expected_source_capsules
    assert len(cohort.hidden) == target.hidden_expected_admitted_capsules
    assert len(full_identities) >= len(certificate_identities) > 0
    models = [capsule for capsule in (*cohort.public, *cohort.hidden) if capsule.kind == "model"]
    assert models
    assert len(gsim_cases) + len(models) == len(cohort.public) + len(cohort.hidden)
    assert not ({capsule.name for capsule in models} & {capsule.name for capsule in gsim_cases})
    assert not ({capsule.name for capsule in cohort.public} & set(target.graded_exclude))


def test_frozen_functional_cohort_ignores_later_live_corpus_growth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_repo = tmp_path / "live"
    snapshot = tmp_path / "phase1-inputs"
    frozen_parent = snapshot / "repo/merlin/contract/capsules"
    live_parent = live_repo / "merlin/contract/capsules"
    for parent in (frozen_parent, live_parent):
        (parent / "isa").mkdir(parents=True)
        (parent / "hidden").mkdir()
    public = _capsule(frozen_parent / "isa", "p0", k=17)
    hidden = _capsule(frozen_parent / "hidden", "h0", k=31)
    excluded = _capsule(frozen_parent / "hidden", "hx", k=47)
    _capsule(live_parent / "isa", "p0", k=17)
    _capsule(live_parent / "isa", "added_after_phase1", k=63)
    _capsule(live_parent / "hidden", "h0", k=31)
    for path, label in ((public, "public"), (hidden, "hidden"), (excluded, "hidden")):
        document = yaml.safe_load(path.read_text())
        document.update({"label": label, "kind": "op"})
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def discover(roots, *, labels=None, contract=None):
        roots = [roots] if isinstance(roots, (str, Path)) else roots
        found = []
        for root in roots:
            for manifest in sorted(Path(root).rglob("capsule.yaml")):
                document = yaml.safe_load(manifest.read_text())
                document["__dir__"] = str(manifest.parent)
                if labels is None or document.get("label") in labels:
                    found.append(document)
        return found

    monkeypatch.setattr(FC, "discover_capsules", discover)
    monkeypatch.setattr(AD, "discover_capsules", discover)
    descriptor_sha = "d" * 64
    public_admission = {
        "n_source_capsules": 1,
        "n_admitted_capsules": 1,
        "n_capability_excluded": 0,
        "n_resource_excluded": 0,
        "admitted_name_set_sha256": FC.name_set_sha256(("p0",)),
        "excluded_name_set_sha256": FC.name_set_sha256(()),
        "descriptor_sha256": descriptor_sha,
    }
    hidden_admission = {
        "n_source_capsules": 2,
        "n_admitted_capsules": 1,
        "n_capability_excluded": 1,
        "n_resource_excluded": 0,
        "admitted_name_set_sha256": FC.name_set_sha256(("h0",)),
        "excluded_name_set_sha256": FC.name_set_sha256(("hx",)),
    }
    functional = SimpleNamespace(
        bundle_input_snapshot={"path": str(snapshot)},
        public_capsules=1,
        hidden_capsules=1,
        public_score={"n_capsules": 1, "per_capsule": [{"capsule": "p0"}], "cohort_admission": public_admission},
        hidden_score={"n_capsules": 1, "per_capsule": [{"capsule": "h0"}], "cohort_admission": hidden_admission},
    )
    target = SimpleNamespace(target="gemmini", capsule_corpus=live_parent / "isa")

    cohort = FC.functional_grade_cohort_from_run(target, functional, source_root=live_repo)

    assert [capsule.name for capsule in cohort.public] == ["p0"]
    assert [capsule.name for capsule in cohort.hidden] == ["h0"]
    assert (cohort.public_source_count, cohort.hidden_source_count) == (1, 2)
    assert "added_after_phase1" not in {capsule.name for capsule in cohort.public}
    assert all(capsule.manifest.is_relative_to(snapshot) for capsule in (*cohort.public, *cohort.hidden))

    public_spec, hidden_spec, contract = AD._frozen_functional_regrade_inputs(tmp_path / "phase2", cohort)
    assert str(live_repo) not in public_spec + hidden_spec
    assert {Path(path).name for path in hidden_spec.split(",")} == {"h0", "hx"}
    assert contract == (snapshot / "repo/merlin/contract").resolve()
    admission = json.loads(
        (tmp_path / "phase2/functional_regrade_inputs/public_admission/.cohort_admission.json").read_text()
    )
    assert admission == public_admission


def test_regrade_inputs_are_the_same_canonical_admitted_cohort(tmp_path) -> None:
    target = AD.load_target_experiment(
        merlin_dir() / "experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    )
    _require_real_corpus_fixture(target)
    cohort = FC.functional_grade_cohort(target, contract_root=merlin_dir() / "contract")

    public_spec, hidden_spec = AD._functional_regrade_inputs(
        target, cohort, contract_root=merlin_dir() / "contract", public_destination=tmp_path / "public"
    )
    materialized = AD.discover_capsules(public_spec, labels={"public", "dev"}, contract=merlin_dir() / "contract")

    assert {str(cap["name"]) for cap in materialized} == {capsule.name for capsule in cohort.public}
    assert hidden_spec == ",".join(str(path) for path in target.hidden_roots())


def test_extension_certificate_retains_tuning_and_covers_exact_revealed_workloads(tmp_path: Path) -> None:
    holdout = tmp_path / "heldout"
    reveal, manifests = _revealed_corpus(holdout, [("k17", 17), ("k31", 31)])
    identities = [AD.GATE.workload_sha256(WORKLOAD.derive_workload(path)) for path in manifests]
    tuning = _certificate({"a" * 64: {"source": "tuning"}})
    extension = _certificate({**tuning.members, **{identity: {"source": "heldout"} for identity in identities}})

    coverage = CTRL._verify_extension_certificate(tuning, extension, reveal)

    assert coverage["holdout_workload_sha256"] == sorted(identities)
    assert coverage["heldout_workloads_covered"] == 2
    assert coverage["pins_unchanged"] is True


def test_extension_certificate_rejects_missing_revealed_workload(tmp_path: Path) -> None:
    holdout = tmp_path / "heldout"
    reveal, (manifest,) = _revealed_corpus(holdout, [("k17", 17)])
    identity = AD.GATE.workload_sha256(WORKLOAD.derive_workload(manifest))
    tuning = _certificate({"a" * 64: {"source": "tuning"}})
    extension = _certificate(dict(tuning.members))

    with pytest.raises(AD.ExperimentError, match="does not cover every revealed workload"):
        CTRL._verify_extension_certificate(tuning, extension, reveal)
    assert identity not in extension.members


@pytest.mark.parametrize("changed_pin", ["gsim_model", "gsim_binary"])
def test_extension_certificate_rejects_changed_model_or_binary_pin(tmp_path: Path, changed_pin: str) -> None:
    holdout = tmp_path / "heldout"
    reveal, (manifest,) = _revealed_corpus(holdout, [("k17", 17)])
    identity = AD.GATE.workload_sha256(WORKLOAD.derive_workload(manifest))
    tuning = _certificate({"a" * 64: {"source": "tuning"}})
    extension = _certificate({**tuning.members, identity: {"source": "heldout"}}, changed_pin=changed_pin)

    with pytest.raises(AD.ExperimentError, match="changed pinned build artifacts"):
        CTRL._verify_extension_certificate(tuning, extension, reveal)


def test_extension_certificate_rejects_workloads_outside_exact_envelope(tmp_path: Path) -> None:
    holdout = tmp_path / "heldout"
    reveal, (manifest,) = _revealed_corpus(holdout, [("k17", 17)])
    identity = AD.GATE.workload_sha256(WORKLOAD.derive_workload(manifest))
    tuning = _certificate({"a" * 64: {"source": "tuning"}})
    extension = _certificate({**tuning.members, identity: {"source": "heldout"}, "b" * 64: {"source": "not revealed"}})

    with pytest.raises(AD.ExperimentError, match="outside the exact"):
        CTRL._verify_extension_certificate(tuning, extension, reveal)


# --- launching without the functional cross-validation certificate --------------------------------
# That certificate compares OUTPUT BYTES between GSIM and the reference engine on one shared ELF, so
# what it establishes is a property of the ENGINE PAIR, not of timing. Measured 2026-09-06: the
# reference leg costs ~45 min/capsule against ~10 s on GSIM -- ~6.7 engine-hours for an 80-workload
# cohort, spent in front of a run that cannot start. Phase 1 graded the same submission on GSIM at L3
# and required no certificate at all. The waiver answers that asymmetry; every test below exists to
# keep it NARROW (it may relax nothing else) and LOUD (a reader can never mistake a waived run for a
# corroborated one).


def _uncertified(tmp_path: Path, *, waived: bool):
    return dataclasses.replace(
        _config(tmp_path),
        functional_gsim_certificate=None,
        functional_gsim_certificate_sha256=None,
        waive_functional_gsim_certificate=waived,
    )


def _certificate_blockers(declaration) -> list[str]:
    return [line for line in declaration["blockers"] if "functional-suite GSIM certificate" in line]


def test_absent_functional_certificate_blocks_launch_unless_explicitly_waived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONTROL then TREATMENT: the blocker must really fire before the waiver is worth anything."""
    _mock_preflight_dependencies(tmp_path, monkeypatch)

    refused = AD.preflight(_uncertified(tmp_path, waived=False), heldout_certificate_provider_available=True)
    assert refused["status"] == "NO_GO"
    assert _certificate_blockers(refused), "control: an absent certificate must block the launch"

    waived = AD.preflight(_uncertified(tmp_path, waived=True), heldout_certificate_provider_available=True)
    assert waived["status"] == "GO"
    assert _certificate_blockers(waived) == []


def test_the_waiver_records_the_claim_it_withdraws_and_is_not_gate_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A waiver that only deleted a blocker would be indistinguishable from a certificate that was
    checked and passed. The predeclaration must carry what was given up."""
    _mock_preflight_dependencies(tmp_path, monkeypatch)
    declaration = AD.preflight(_uncertified(tmp_path, waived=True), heldout_certificate_provider_available=True)

    waiver = declaration["functional_gsim_certificate_waiver"]
    assert waiver["waived"] is True
    assert waiver["gate_clean"] is False
    assert "uncorroborated" in waiver["claim_withdrawn"]
    # The timing authority is the one thing a perf experiment must never lose.
    assert "TUNING" in waiver["claim_retained"]
    assert declaration["functional_gsim_certificate_sha256"] is None
    assert declaration["functional_gsim_coverage"] is None


def test_no_waiver_is_recorded_when_the_certificate_is_actually_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The waiver field must distinguish the two worlds, so a supplied certificate never reads as a
    waived one and a waived run never reads as a certified one."""
    _mock_preflight_dependencies(tmp_path, monkeypatch)
    certified = AD.preflight(_config(tmp_path), heldout_certificate_provider_available=True)
    assert certified["functional_gsim_certificate_waiver"] is None
    assert certified["functional_gsim_coverage"] is not None


def test_the_waiver_suppresses_no_other_blocker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: waive the certificate AND withhold the held-out provider. The unrelated blocker must
    still refuse the launch, or the waiver is a general-purpose gate opener."""
    _mock_preflight_dependencies(tmp_path, monkeypatch)
    declaration = AD.preflight(_uncertified(tmp_path, waived=True), heldout_certificate_provider_available=False)
    assert declaration["status"] == "NO_GO"
    assert any("post-seal" in line for line in declaration["blockers"])


def test_the_waiver_never_relaxes_the_tuning_certificate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: the tuning certificate pins the TIMING authority. Breaking it must still refuse even
    with the functional certificate waived."""
    _mock_preflight_dependencies(tmp_path, monkeypatch)

    def _refuse(*_args, **_kwargs):
        raise AD.ExperimentError("tuning certificate does not cover the corpus")

    monkeypatch.setattr(AD, "_verify_tuning_certificate", _refuse)
    with pytest.raises(AD.ExperimentError):
        AD.preflight(_uncertified(tmp_path, waived=True), heldout_certificate_provider_available=True)


def _fake_corpus(rows):
    return SimpleNamespace(
        capsules=[
            SimpleNamespace(family=family, capsule=name, source_dir=Path("/corpus") / name) for family, name in rows
        ]
    )


def _identity_by_capsule(mapping, monkeypatch):
    monkeypatch.setattr(WORKLOAD, "derive_workload", lambda path: {"identity": mapping[path.parent.name]})
    monkeypatch.setattr(AD.GATE, "workload_sha256", lambda workload: workload["identity"])


def test_tuning_corpus_may_measure_one_workload_under_several_family_levers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Several families measure the same workload under a different lever and the campaign measures
    one member per identity -- which is what this function's contract says. A 1:1 assumption here
    refused a real launch on 2026-09-06: PK00_k16, PM00_m16n16 and PR00_fits_double_k16 are all
    16x16x16, and each family REQUIRES its own anchor (PK needs exactly four descriptors, PR needs
    three depths in its `fits_double` band), so no capsule could be dropped to satisfy it."""
    rows = [("PK", "PK00_k16"), ("PM", "PM00_m16n16"), ("PR", "PR00_fits_double_k16"), ("PK", "PK01_k32")]
    _identity_by_capsule(
        {"PK00_k16": "i16", "PM00_m16n16": "i16", "PR00_fits_double_k16": "i16", "PK01_k32": "i32"}, monkeypatch
    )
    monkeypatch.setattr(P2_CORPUS, "discover_performance_corpus", lambda *_args, **_kwargs: _fake_corpus(rows))
    certificate = SimpleNamespace(members={"i16": {}, "i32": {}})

    coverage = AD._verify_tuning_certificate(certificate, SimpleNamespace(target="gemmini"))

    assert coverage["members"] == 2, "the certificate covers distinct WORKLOADS, not capsules"
    assert coverage["corpus_capsules"] == 4
    assert coverage["shared_identities"] == {"i16": ["PK/PK00_k16", "PM/PM00_m16n16", "PR/PR00_fits_double_k16"]}


def test_shared_identities_do_not_hide_a_workload_the_certificate_never_covered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTATION: grouping must not weaken the coverage comparison. A workload absent from the
    certificate is still a refusal, whether or not other members share an identity."""
    rows = [("PK", "PK00_k16"), ("PM", "PM00_m16n16"), ("PK", "PK01_k32")]
    _identity_by_capsule({"PK00_k16": "i16", "PM00_m16n16": "i16", "PK01_k32": "i32"}, monkeypatch)
    monkeypatch.setattr(P2_CORPUS, "discover_performance_corpus", lambda *_args, **_kwargs: _fake_corpus(rows))

    with pytest.raises(AD.ExperimentError, match="does not cover the derived corpus"):
        AD._verify_tuning_certificate(SimpleNamespace(members={"i16": {}}), SimpleNamespace(target="gemmini"))
    with pytest.raises(AD.ExperimentError, match="does not cover the derived corpus"):
        AD._verify_tuning_certificate(
            SimpleNamespace(members={"i16": {}, "i32": {}, "i64": {}}), SimpleNamespace(target="gemmini")
        )


def test_a_narrowed_campaign_may_use_the_whole_corpus_certificate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A campaign that measures one claim's families still verifies against the corpus-wide tuning
    certificate. Refusing the un-measured identities as `extras` blocked a 16-member PM launch
    against the 36-workload certificate on 2026-09-06 -- the exact case selection exists to create."""
    full = [("PM", "PM00"), ("PM", "PM01"), ("PK", "PK00"), ("PR", "PR00")]
    _identity_by_capsule({"PM00": "i0", "PM01": "i1", "PK00": "i2", "PR00": "i3"}, monkeypatch)

    def _discover(_target, capsules=None, families=None):
        rows = full if capsules is None else [r for r in full if r[1] in capsules.split(",")]
        return _fake_corpus(rows)

    monkeypatch.setattr(P2_CORPUS, "discover_performance_corpus", _discover)
    certificate = SimpleNamespace(members={"i0": {}, "i1": {}, "i2": {}, "i3": {}})

    coverage = AD._verify_tuning_certificate(certificate, SimpleNamespace(target="gemmini"), capsules="PM00,PM01")
    assert coverage["members"] == 2, "only the selected members are the campaign's cohort"


def test_a_narrowed_campaign_still_needs_its_own_members_certified(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: widening `extras` must not weaken `missing`. A SELECTED member absent from the
    certificate is still a refusal, and a certificate workload foreign to the whole corpus still is."""
    full = [("PM", "PM00"), ("PM", "PM01"), ("PK", "PK00")]
    _identity_by_capsule({"PM00": "i0", "PM01": "i1", "PK00": "i2"}, monkeypatch)

    def _discover(_target, capsules=None, families=None):
        rows = full if capsules is None else [r for r in full if r[1] in capsules.split(",")]
        return _fake_corpus(rows)

    monkeypatch.setattr(P2_CORPUS, "discover_performance_corpus", _discover)

    with pytest.raises(AD.ExperimentError, match="missing="):
        AD._verify_tuning_certificate(
            SimpleNamespace(members={"i1": {}, "i2": {}}), SimpleNamespace(target="gemmini"), capsules="PM00,PM01"
        )
    with pytest.raises(AD.ExperimentError, match="extras="):
        AD._verify_tuning_certificate(
            SimpleNamespace(members={"i0": {}, "i1": {}, "i2": {}, "foreign": {}}),
            SimpleNamespace(target="gemmini"),
            capsules="PM00,PM01",
        )
