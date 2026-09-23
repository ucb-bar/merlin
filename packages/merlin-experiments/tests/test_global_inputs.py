"""Staged input ownership: local evidence admission, not full-workflow qualification."""

import json
import socket
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import global_inputs as GI
from merlin_experiments.phase2 import host_policy as HP
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.mechanism_program import MechanismProgram
from merlin_experiments.phase2.portfolio_evaluation import FastPortfolioEvaluation
from merlin_experiments.phase2.stage_inputs import StageE2ESentinel

from merlin.benchharness import hash_tree
from merlin.perf.historical_reference import reference_summary


@pytest.fixture(autouse=True)
def refuse_processes_and_listeners(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("input admission must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.fixture
def inputs(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "compiler.py").write_text("from merlin.helper import VALUE\n\ndef schedule():\n    return VALUE\n")
    (baseline / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    (shared / "helper.py").write_text("VALUE = 1\n")
    sentinels = []
    for index in range(2):
        source = tmp_path / f"capsule-{index}"
        source.mkdir()
        (source / "capsule.yaml").write_text("interface_mlir: capsule.interface.mlir\n")
        (source / "capsule.interface.mlir").write_text(f"module {{ // synthetic {index}\n}}\n")
        sentinels.append(
            StageE2ESentinel(
                source.name, str(source), str(source), C.exact_tree_record(source)["sha256"], ("lane",), ("L2",)
            )
        )
    descriptor = tmp_path / "target.json"
    descriptor.write_text('{"synthetic_target":true}\n')
    controller = tmp_path / "controller.py"
    controller.write_text("SYNTHETIC_CONTROLLER = True\n")
    contract = tmp_path / "contract"
    for relative in HP.RESOURCE_FILES:
        path = contract / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"synthetic_resource": relative}))
    return dict(
        baseline=baseline,
        baseline_sha256=hash_tree(baseline)["sha256"],
        sentinel=sentinels[0],
        portfolio_sentinels=sentinels[1:],
        target="synthetic-unregistered-target",
        target_sha256=C.sha256_file(descriptor),
        target_descriptor=descriptor,
        output=tmp_path / "run",
        compiler_shared_source_root=shared,
        contract_root=contract,
        controller_source=controller,
    )


def materialize(inputs):
    prepared = GI.GlobalExperimentInputs.prepare(**inputs)
    inputs["output"].mkdir()
    return prepared.materialize(inputs["output"])


def collaborators(owner):
    output = owner.baseline.parent / "run"
    authority = FrozenEditAuthority(output)
    mechanism = MechanismProgram(
        output,
        authority,
        portfolio_identity=owner.portfolio_identity,
        portfolio_identity_sha256=owner.portfolio_identity_sha256,
    )
    return dict(
        edit_authority=authority,
        mechanism_program=mechanism,
        fast_evaluation=FastPortfolioEvaluation(owner.portfolio_sentinels),
    )


def retained_sources(inputs):
    comparison = inputs["baseline"].parent / "comparison"
    comparison.mkdir()
    (comparison / "compiler.py").write_text(
        "from merlin.helper import VALUE\n\ndef schedule():\n    return VALUE + 1\n"
    )
    inputs["optimization_baseline"] = comparison
    inputs["optimization_baseline_sha256"] = hash_tree(comparison)["sha256"]
    historical = comparison.parent / "historical.json"
    historical.write_bytes(
        C.canonical_json({"schema": "historical_reference_bundle_v1", "records": [], "summary": reference_summary([])})
    )
    inputs["historical_reference_path"] = historical
    inputs["historical_reference_sha256"] = C.sha256_file(historical)


def test_prepare_does_not_materialize_and_legacy_baseline_is_not_copied(inputs):
    prepared = GI.GlobalExperimentInputs.prepare(**inputs)
    assert not inputs["output"].exists()
    inputs["output"].mkdir()
    owner = prepared.materialize(inputs["output"])
    assert owner.optimization_baseline == inputs["baseline"]
    assert owner.optimization_baseline_binding["selection"] == "frozen_phase1_compiler"
    assert owner.optimization_baseline_binding["phase1_regraded"] is False
    assert owner.portfolio_identity["members"][0]["role"] == "primary"
    assert owner.portfolio_identity["members"][1]["role"] == "training"
    assert owner.host_policy == HP.build_record(
        controller_source=inputs["controller_source"], contract_root=inputs["contract_root"]
    )
    assert owner.compiler_dependencies(inputs["baseline"]) == owner.baseline_dependencies
    owner.verify(**collaborators(owner))


@pytest.mark.parametrize(
    "invalid",
    [
        "target_digest",
        "baseline_digest",
        "comparison_pair",
        "reason",
        "output",
        "source_pair",
        "historical_pair",
        "duplicate_member",
        "descriptor",
        "workers",
        "memory",
    ],
)
def test_prepare_refuses_without_writing_output(inputs, invalid):
    if invalid == "target_digest":
        inputs["target_sha256"] = "bad"
    elif invalid == "baseline_digest":
        inputs["baseline_sha256"] = "0" * 64
    elif invalid == "comparison_pair":
        inputs["optimization_baseline"] = inputs["baseline"]
    elif invalid == "reason":
        inputs["optimization_baseline_reason"] = " "
    elif invalid == "output":
        inputs["output"].mkdir()
        (inputs["output"] / "untouched").write_text("keep")
    elif invalid == "source_pair":
        inputs["source_snapshot_root"] = inputs["baseline"]
    elif invalid == "historical_pair":
        inputs["historical_reference_path"] = inputs["target_descriptor"]
    elif invalid == "duplicate_member":
        inputs["portfolio_sentinels"] = [inputs["sentinel"]]
    elif invalid == "descriptor":
        inputs["target_descriptor"].write_text("changed")
    elif invalid == "workers":
        inputs["portfolio_analysis_workers"] = 0
    else:
        inputs["minimum_memory_available_bytes"] = -1
    with pytest.raises(ValueError):
        GI.GlobalExperimentInputs.prepare(**inputs)
    if invalid == "output":
        assert [path.name for path in inputs["output"].iterdir()] == ["untouched"]
    else:
        assert not inputs["output"].exists()


def test_early_digest_refusal_precedes_policy_discovery(inputs, monkeypatch):
    inputs["baseline_sha256"] = "0" * 64
    monkeypatch.setattr(HP, "build_record", lambda **kwargs: pytest.fail("policy observed before baseline admission"))
    with pytest.raises(ValueError, match="compiler digest"):
        GI.GlobalExperimentInputs.prepare(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("linked", ["controller", "comparison"])
def test_explicit_linked_source_refused_before_output(inputs, linked):
    if linked == "controller":
        original = inputs["controller_source"]
        alias = original.with_name("controller-link.py")
        alias.symlink_to(original)
        inputs["controller_source"] = alias
    else:
        retained_sources(inputs)
        original = inputs["optimization_baseline"]
        alias = original.with_name("comparison-link")
        alias.symlink_to(original, target_is_directory=True)
        inputs["optimization_baseline"] = alias
    with pytest.raises(ValueError):
        GI.GlobalExperimentInputs.prepare(**inputs)
    assert not inputs["output"].exists()


def test_materialize_requires_prepared_directory_and_single_use(inputs):
    prepared = GI.GlobalExperimentInputs.prepare(**inputs)
    with pytest.raises(ValueError, match="prepared output"):
        prepared.materialize(inputs["output"])
    other = inputs["output"].parent / "other-output"
    other.mkdir()
    with pytest.raises(ValueError, match="prepared output"):
        prepared.materialize(other)
    assert not list(other.iterdir())
    inputs["output"].mkdir()
    prepared.materialize(inputs["output"])
    with pytest.raises(ValueError, match="only once"):
        prepared.materialize(inputs["output"])


def test_comparison_changed_after_prepare_refuses_materialization(inputs):
    retained_sources(inputs)
    prepared = GI.GlobalExperimentInputs.prepare(**inputs)
    (inputs["optimization_baseline"] / "compiler.py").write_text("changed between prepare and capture")
    inputs["output"].mkdir()
    with pytest.raises(ValueError, match="changed while capturing"):
        prepared.materialize(inputs["output"])


def test_machine_policy_change_precedes_collaborator_checks(inputs, monkeypatch):
    owner = materialize(inputs)
    monkeypatch.setattr(GI, "current_machine_build_policy", lambda target: {"changed": True})

    class Unreachable:
        def check_integrity(self):
            pytest.fail("delegated check ran before machine policy refusal")

    with pytest.raises(ValueError, match="machine build toolchain policy"):
        owner.verify(edit_authority=Unreachable(), mechanism_program=Unreachable(), fast_evaluation=Unreachable())


def test_materialize_retains_readonly_reference_and_comparison(inputs):
    retained_sources(inputs)
    original_reference = inputs["historical_reference_path"].read_bytes()
    prepared = GI.GlobalExperimentInputs.prepare(**inputs)
    assert not inputs["output"].exists()
    inputs["output"].mkdir()
    owner = prepared.materialize(inputs["output"])
    historical = Path(owner.historical_reference["path"])
    assert historical == inputs["output"] / "historical_reference.json"
    assert historical.read_bytes() == original_reference
    assert not historical.stat().st_mode & 0o222
    assert owner.optimization_baseline == inputs["output"] / "optimization_baseline"
    assert hash_tree(owner.optimization_baseline)["sha256"] == inputs["optimization_baseline_sha256"]
    assert all(not path.stat().st_mode & 0o222 for path in owner.optimization_baseline.rglob("*"))
    assert not owner.optimization_baseline.stat().st_mode & 0o222
    assert owner.optimization_baseline_binding["selection"] == "explicit_host_seed"
    assert owner.optimization_baseline_binding["objective_numerical_qualification"] == "UNPROVEN"
    (inputs["optimization_baseline"] / "compiler.py").write_text("original source changed after capture")
    inputs["historical_reference_path"].write_text("original reference changed after capture")
    owner.verify(**collaborators(owner))


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("controller", "host verification"),
        ("resource", "host verification"),
        ("portfolio", "portfolio identity"),
        ("sentinel", "portfolio identity"),
        ("target", "target descriptor"),
        ("baseline", "frozen compiler"),
        ("dependencies", "shared compiler dependencies"),
        ("capsule", "objective changed"),
        ("comparison", "optimization baseline"),
        ("comparison_binding", "optimization baseline"),
        ("historical", "historical reference bytes"),
        ("historical_binding", "historical reference binding"),
    ],
)
def test_verify_refuses_real_input_and_identity_drift(inputs, mutation, match):
    retained_sources(inputs)
    owner = materialize(inputs)
    checks = collaborators(owner)
    owner.verify(**checks)
    if mutation == "controller":
        inputs["controller_source"].write_text("CHANGED = True\n")
    elif mutation == "resource":
        (inputs["contract_root"] / HP.RESOURCE_FILES[0]).write_text("{}")
    elif mutation == "portfolio":
        owner.portfolio_identity["members"][0]["capsule"] = "other"
    elif mutation == "sentinel":
        owner.portfolio_sentinels = (replace(owner.sentinel, required_lanes=("other",)), *owner.portfolio_sentinels[1:])
    elif mutation == "target":
        inputs["target_descriptor"].write_text("changed")
    elif mutation == "baseline":
        (owner.baseline / "compiler.py").write_text("changed")
    elif mutation == "dependencies":
        (inputs["compiler_shared_source_root"] / "helper.py").write_text("VALUE = 2\n")
    elif mutation == "capsule":
        (Path(owner.sentinel.frozen_source_path) / "capsule.interface.mlir").write_text("changed")
    elif mutation == "comparison":
        path = owner.optimization_baseline / "compiler.py"
        path.chmod(0o644)
        path.write_text("changed")
    elif mutation == "comparison_binding":
        owner.optimization_baseline_binding["reason"] = "substituted"
    elif mutation == "historical":
        path = Path(owner.historical_reference["path"])
        path.chmod(0o644)
        path.write_text("changed")
    else:
        owner.historical_reference["summary"] = {}
    with pytest.raises(ValueError, match=match):
        owner.verify(**checks)


def test_integrity_owners_run_in_original_order_and_stop_on_refusal(inputs):
    owner = materialize(inputs)
    events = []

    class Check:
        def __init__(self, name, refuse=False):
            self.name, self.refuse = name, refuse

        def check_integrity(self):
            events.append(self.name)
            if self.refuse:
                raise ValueError(f"{self.name} changed")

    for failure, expected in [
        (None, ["evaluation", "edit", "mechanism"]),
        ("evaluation", ["evaluation"]),
        ("edit", ["evaluation", "edit"]),
        ("mechanism", ["evaluation", "edit", "mechanism"]),
    ]:
        events.clear()
        checks = {
            "fast_evaluation": Check("evaluation", failure == "evaluation"),
            "edit_authority": Check("edit", failure == "edit"),
            "mechanism_program": Check("mechanism", failure == "mechanism"),
        }
        if failure is None:
            owner.verify(**checks)
        else:
            with pytest.raises(ValueError, match=failure):
                owner.verify(**checks)
        assert events == expected


def test_phase1_receipt_binding_is_rechecked_without_regrading(inputs):
    # This protocol fixture tests owner delegation, not Phase-1 qualification semantics.
    receipt = inputs["baseline"].parent / "phase1-receipt.json"
    receipt.write_text('{"qualified":true}')
    calls = []

    class RecordedPhase1:
        def verify(self, baseline):
            calls.append(baseline)
            return {"receipt_sha256": C.sha256_file(receipt)}

    inputs["phase1"] = RecordedPhase1()
    owner = materialize(inputs)
    checks = collaborators(owner)
    owner.verify(**checks)
    assert calls == [inputs["baseline"], inputs["baseline"]]
    receipt.write_text('{"qualified":false}')
    with pytest.raises(ValueError, match="Phase-1 qualification"):
        owner.verify(**checks)


@pytest.fixture
def frozen_phase1(inputs, monkeypatch):
    runs = inputs["baseline"].parent / "phase1-runs"
    run = runs / "qualified"
    files = (
        "environment.yaml",
        "qa_loop_summary.yaml",
        "freeze.json",
        "run_manifest.yaml",
        "grading_public/score_capsule.json",
        "grading_hidden/score_capsule.json",
    )
    for relative in files:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"synthetic_evidence": relative}))
    frozen = SimpleNamespace(
        run_id="qualified",
        run_dir=run,
        digest=inputs["baseline_sha256"],
        public_score={
            "n_passed": 1,
            "n_capsules": 2,
            "per_capsule": [{"capsule": "passing", "status": "pass"}, {"capsule": "known-gap", "status": "fail"}],
        },
        deviations=[],
    )
    calls = []

    def inspect(root, run_id, submission, *, waive):
        calls.append((root, run_id, submission, waive))
        return frozen

    monkeypatch.setattr(GI.PC, "inspect_functional_run", inspect)
    phase1 = GI.FrozenPhase1(runs, "qualified", inputs["baseline_sha256"], ("known-gap-waiver",), 1, 2, ("known-gap",))
    return phase1, frozen, calls, files


def test_frozen_phase1_preserves_exact_evidence_and_waivers(inputs, frozen_phase1):
    phase1, frozen, calls, files = frozen_phase1
    binding = phase1.verify(inputs["baseline"])
    assert calls == [(phase1.runs_root, phase1.run_id, phase1.submission_sha256, phase1.waiver_predicates)]
    assert binding["submission_sha256"] == inputs["baseline_sha256"]
    assert binding["known_functional_gap_ids"] == ["known-gap"]
    assert binding["waiver_predicates"] == ["known-gap-waiver"]
    assert binding["qualification_action"] == "read_existing_frozen_receipts_only"
    assert binding["evidence_sha256"] == {name: C.sha256_file(frozen.run_dir / name) for name in files}
    evidence = frozen.run_dir / "freeze.json"
    evidence.write_text('{"changed":true}')
    changed = phase1.verify(inputs["baseline"])
    assert changed != binding
    assert changed["evidence_sha256"]["freeze.json"] == C.sha256_file(evidence)


@pytest.mark.parametrize("mutation", ["baseline", "count", "gap", "gap_count"])
def test_frozen_phase1_refuses_compiler_or_exact_qualification_mismatch(inputs, frozen_phase1, mutation):
    phase1, frozen, _, _ = frozen_phase1
    if mutation == "baseline":
        (inputs["baseline"] / "compiler.py").write_text("changed")
    elif mutation == "count":
        frozen.public_score["n_passed"] = 0
    elif mutation == "gap":
        frozen.public_score["per_capsule"][1]["capsule"] = "different-gap"
    else:
        phase1 = replace(phase1, expected_public_passed=0)
        frozen.public_score["n_passed"] = 0
    with pytest.raises(ValueError, match="frozen qualification"):
        phase1.verify(inputs["baseline"])


def test_inputs_verify_detects_real_frozen_phase1_evidence_drift(inputs, frozen_phase1):
    phase1, frozen, _, _ = frozen_phase1
    inputs["phase1"] = phase1
    owner = materialize(inputs)
    checks = collaborators(owner)
    owner.verify(**checks)
    (frozen.run_dir / "grading_hidden/score_capsule.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Phase-1 qualification"):
        owner.verify(**checks)
