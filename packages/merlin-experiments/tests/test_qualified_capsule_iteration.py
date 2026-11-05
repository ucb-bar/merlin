"""Technical capsule admission is explicit, revalidated, and iteration only."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import capsule_qualification_handoff as H
from merlin_experiments.phase2 import capsule_iteration as CI
from merlin_experiments.phase2 import functional_inputs as FI
from test_capsule_certification import selected as selected
from test_capsule_qualification_handoff import qualified as qualified


def configuration(qualified):
    s, _, record = qualified
    return {
        "phase1_validation_scope": "capsules",
        "functional_run_id": s.run.name,
        "functional_submission_sha256": record["submission_sha256"],
        "capsule_qualification": record["receipt"],
        "waive_functional_gate": [],
    }


def admit(qualified, monkeypatch):
    s, _, _ = qualified
    calls = []

    def snapshot(base):
        calls.append(base)
        return base

    monkeypatch.setattr(FI, "_functional_input_snapshot", snapshot)
    base, owner = CI.selected_functional_run(s.run.parent, configuration(qualified), s.descriptor)
    assert calls == [base]  # The existing V4 input-snapshot admission is mandatory.
    return base, owner, owner.verify(base.submission_dir)


def test_real_technical_qualification_allows_only_iteration_without_author_publication(qualified, monkeypatch):
    s, _, record = qualified
    base, owner, binding = admit(qualified, monkeypatch)
    assert type(owner) is CI.FrozenQualifiedCapsulePhase1
    assert base.run_dir == s.run and base.submission_dir == Path(record["handoff"]["compiler"]["path"])
    assert base.public_capsules == 2 and base.hidden_capsules == 1 and not base.gate_clean
    assert not (s.run / "capsule_publication.json").exists()
    assert not (s.run / "qa_loop_summary.yaml").exists()
    assert binding["capsule_qualification"] == record["receipt"]
    assert binding["phase1_authoring_converged"] is False
    assert binding["final_results_qualified"] is binding["external_full_models_complete"] is False
    assert binding["waiver_predicates"] == []
    assert record["phase2_iteration_ready"] is record["authoring_converged"] is False
    CI.require_paid_iteration(SimpleNamespace(phase1=owner, phase1_binding=binding, baseline=base.submission_dir))
    CI.verify_checkpoint_scope({"phase1_qualification": binding, **CI.checkpoint_scope(binding)})
    with pytest.raises(ValueError, match="external full-model"):
        CI.require_final_results(binding, descriptor=s.descriptor)


def test_technical_selection_never_bypasses_missing_v4_input_ownership(qualified):
    s, _, _ = qualified
    with pytest.raises(FI.StageGateError, match="V4"):
        CI.selected_functional_run(s.run.parent, configuration(qualified), s.descriptor)


@pytest.mark.parametrize(
    "fault", ["scope", "waiver", "sha", "candidate", "run", "target", "ambiguous", "relative", "extra_key"]
)
def test_selection_rejects_mismatched_or_ambiguous_authority_before_input_admission(qualified, monkeypatch, fault):
    s, _, _ = qualified
    config = copy.deepcopy(configuration(qualified))
    descriptor = s.descriptor
    if fault == "scope":
        config["phase1_validation_scope"] = "full"
    elif fault == "waiver":
        config["waive_functional_gate"] = ["external_full_models_pending"]
    elif fault == "sha":
        config["capsule_qualification"]["sha256"] = "0" * 64
    elif fault == "candidate":
        config["functional_submission_sha256"] = "0" * 64
    elif fault == "run":
        config["functional_run_id"] = "another-run"
    elif fault == "target":
        descriptor = s.run / "other-descriptor.yaml"
        descriptor.write_text("target: other\n")
    elif fault == "ambiguous":
        config["capsule_publication_sha256"] = "0" * 64
    elif fault == "relative":
        config["capsule_qualification"]["path"] = H.RECEIPT
    else:
        config["capsule_qualification"]["passed"] = True
    monkeypatch.setattr(FI, "_functional_input_snapshot", lambda _: pytest.fail("unadmitted input reached V4"))
    with pytest.raises(ValueError):
        CI.selected_functional_run(s.run.parent, config, descriptor)


@pytest.mark.parametrize("fault", ["old_score", "resource", "export", "receipt", "other_producer"])
def test_iteration_reopens_all_evidence_and_refuses_other_producer_paths(qualified, monkeypatch, tmp_path, fault):
    s, out, record = qualified
    base, owner, binding = admit(qualified, monkeypatch)
    if fault == "other_producer":
        source = tmp_path / "same-bytes-other-producer.py"
        source.write_bytes(Path(H.__file__).read_bytes())
        monkeypatch.setattr(H, "__file__", str(source))
    else:
        path = {
            "old_score": s.run / "grading_hidden/score_capsule.json",
            "resource": Path(record["resource_producer_epoch"]["verdict"]["path"]),
            "export": out / "compiler/compiler.py",
            "receipt": out / H.RECEIPT,
        }[fault]
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        CI.require_paid_iteration(SimpleNamespace(phase1=owner, phase1_binding=binding, baseline=base.submission_dir))


@pytest.mark.parametrize("fault", ["no_owner", "fake_owner", "baseline", "binding_flag", "binding_schema"])
def test_paid_authoring_requires_exact_live_technical_owner(qualified, monkeypatch, tmp_path, fault):
    base, owner, binding = admit(qualified, monkeypatch)
    inputs = SimpleNamespace(phase1=owner, phase1_binding=copy.deepcopy(binding), baseline=base.submission_dir)
    if fault == "no_owner":
        inputs.phase1 = None
    elif fault == "fake_owner":
        inputs.phase1 = SimpleNamespace(verify=lambda _: binding)
    elif fault == "baseline":
        inputs.baseline = tmp_path / "other-compiler"
        inputs.baseline.mkdir()
        (inputs.baseline / "compiler.py").write_text("different\n")
    elif fault == "binding_flag":
        inputs.phase1_binding["phase1_authoring_converged"] = True
    else:
        inputs.phase1_binding["schema"] = "global_frozen_phase1_binding_v1"
    with pytest.raises(ValueError):
        CI.require_paid_iteration(inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_results_qualified", True),
        ("external_full_models_complete", True),
        ("phase1_authoring_converged", True),
        ("authoring_converged", True),
        ("final_release", True),
        ("qualified_model_results", True),
        ("launch_scope", "qualified_macro_experiment"),
    ],
)
def test_checkpoint_cannot_promote_technical_iteration(qualified, monkeypatch, field, value):
    _, _, binding = admit(qualified, monkeypatch)
    document = {"phase1_qualification": binding, **CI.checkpoint_scope(binding)}
    document[field] = value
    with pytest.raises(ValueError):
        CI.verify_checkpoint_scope(document)


def test_schema_rename_cannot_launder_technical_baseline_into_full_results(qualified, monkeypatch):
    _, _, binding = admit(qualified, monkeypatch)
    forged = {**binding, "schema": "global_frozen_phase1_binding_v1"}
    with pytest.raises(ValueError, match="lost its selected binding"):
        CI.verify_checkpoint_scope({"phase1_qualification": forged})
    with pytest.raises(ValueError, match="external full-model"):
        CI.require_final_results(forged)
