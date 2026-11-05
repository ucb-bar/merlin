"""Real capsule publication and Phase 2 admission, without agents or target runs."""

import copy
import hashlib
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import capsule_iteration as CI
from merlin_experiments.phase2 import contracts as D
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2.portfolio_authoring import PortfolioAuthoring
from test_capsule_certification import (
    publish_author,
    selected,  # noqa: F401 -- real capsule input fixture
)


@pytest.fixture
def admitted(selected):  # noqa: F811 -- imported shared pytest fixture
    _, publication = publish_author(selected)
    source = CI.FrozenCapsulePhase1(
        selected.run,
        selected.descriptor,
        publication["submission_sha256"],
        publication["receipt"]["sha256"],
    )
    baseline = source.compiler
    return SimpleNamespace(
        source=source, baseline=baseline, binding=source.verify(baseline), selected=selected, publication=publication
    )


def test_capsule_variant_admits_iteration_but_never_qualified_model_results(admitted):
    a = admitted
    inputs = SimpleNamespace(phase1=a.source, phase1_binding=a.binding, baseline=a.baseline)
    CI.require_paid_iteration(inputs)
    assert CI.launch_scope(a.binding) == "capsule_iteration"
    assert a.binding["final_results_qualified"] is a.binding["external_full_models_complete"] is False
    assert a.binding["waiver_predicates"] == []
    with pytest.raises(ValueError, match="external full-model qualification"):
        CI.require_final_results(a.binding)
    assert CI.launch_scope(None) == "development_readiness_only"
    assert CI.launch_scope({"schema": "global_frozen_phase1_binding_v1"}) == "qualified_macro_experiment"


@pytest.mark.parametrize("fault", ["baseline", "publication", "missing_owner", "changed_scope", "changed_flags"])
def test_paid_iteration_reopens_actual_owner_and_cannot_relabel_saved_flags(admitted, tmp_path, fault):
    a = admitted
    inputs = SimpleNamespace(phase1=a.source, phase1_binding=copy.deepcopy(a.binding), baseline=a.baseline)
    if fault == "baseline":
        inputs.baseline = tmp_path / "different"
        inputs.baseline.mkdir()
        (inputs.baseline / "compiler.py").write_text("different compiler\n")
    elif fault == "publication":
        path = a.selected.run / "capsule_publication.json"
        path.chmod(0o600)
        path.write_bytes(path.read_bytes() + b" ")
    elif fault == "missing_owner":
        inputs.phase1 = SimpleNamespace(verify=lambda _: a.binding)
    elif fault == "changed_scope":
        inputs.phase1_binding["schema"] = "global_frozen_phase1_binding_v1"
    else:
        inputs.phase1_binding["external_full_models_complete"] = True
    with pytest.raises(ValueError):
        CI.require_paid_iteration(inputs)


def test_real_paid_portfolio_route_accepts_capsules_then_preserves_edit_authority_gate(admitted, tmp_path):
    a = admitted
    root = tmp_path / "public-agent-inputs"
    root.mkdir()
    payload = b"public interface only"
    (root / "interface").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = root / "agent_input_manifest.json"
    manifest.write_bytes(
        D.canonical_json({"files": [{"path": "interface", "sha256": digest, "n_bytes": len(payload)}]})
    )
    aggregate = hashlib.sha256(f"interface\0{digest}\0{len(payload)}\n".encode()).hexdigest()
    public = AW.AgentInputSnapshot(root, manifest, D.sha256_file(manifest), aggregate, 1, len(payload))
    owner = PortfolioAuthoring.__new__(PortfolioAuthoring)
    for name in (
        "_sandbox_inputs",
        "target_experiment",
        "stage_root",
        "frozen_functional",
        "frozen_corpus_manifest",
        "model",
        "resolved_model",
        "effort",
        "codex_binary",
        "global_probe_provider",
        "global_semantic_provider",
        "global_context_provider",
        "global_paired_context_provider",
        "global_source_pair_provider",
    ):
        setattr(owner, name, None)
    owner.max_tool_calls = 1
    owner.agent_inputs = public
    owner.experiment = SimpleNamespace(
        inputs=SimpleNamespace(phase1=a.source, phase1_binding=a.binding, baseline=a.baseline),
        edit_authority=SimpleNamespace(contract=None),
    )
    with pytest.raises(ValueError, match="host-frozen compiler edit authority"):
        owner.run_round(a.baseline, round_index=0, round_timeout_s=1)
    owner.experiment.inputs.phase1_binding = None
    with pytest.raises(ValueError, match="existing Phase-1"):
        owner.run_round(a.baseline, round_index=0, round_timeout_s=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_results_qualified", True),
        ("external_full_models_complete", True),
        ("launch_scope", "qualified_macro_experiment"),
        ("qualification_scope", "final_release"),
        ("full_model_numerics_qualified", True),
        ("final_release", True),
    ],
)
def test_capsule_checkpoints_cannot_promote_flags_to_final_release(admitted, field, value):
    a = admitted
    document = {"phase1_qualification": a.binding, **CI.checkpoint_scope(a.binding)}
    CI.verify_checkpoint_scope(document)
    document[field] = value
    with pytest.raises(ValueError, match="cannot promote"):
        CI.verify_checkpoint_scope(document)


def test_capsule_checkpoint_cannot_hide_variant_by_renaming_schema(admitted):
    binding = copy.deepcopy(admitted.binding)
    binding["schema"] = "global_frozen_phase1_binding_v1"
    with pytest.raises(ValueError, match="lost its selected binding"):
        CI.verify_checkpoint_scope({"phase1_qualification": binding, "launch_scope": "qualified_macro_experiment"})
    with pytest.raises(ValueError, match="external full-model"):
        CI.require_final_results(binding)


def test_capsule_run_cannot_enter_full_frozen_api_or_waive_external_pending(admitted):
    from merlin_experiments.phase2 import campaign

    a = admitted
    with pytest.raises(campaign.CampaignGateError, match="explicit capsule_iteration"):
        campaign.inspect_functional_run(a.selected.run.parent, a.selected.run.name, a.binding["submission_sha256"])
    with pytest.raises(ValueError, match="explicit existing full qualification"):
        CI.require_final_results({"schema": "global_frozen_phase1_binding_v1"})


def test_worker_selection_preserves_existing_v4_snapshot_owner_and_exact_target(admitted, monkeypatch, tmp_path):
    a = admitted
    calls = []

    def snapshot(base):
        calls.append(base)
        return base  # External V4 snapshot owner has its own real transport tests.

    monkeypatch.setattr(FI, "_functional_input_snapshot", snapshot)
    config = {
        "phase1_validation_scope": "capsules",
        "waive_functional_gate": [],
        "functional_run_id": a.selected.run.name,
        "functional_submission_sha256": a.binding["submission_sha256"],
        "capsule_publication_sha256": a.publication["receipt"]["sha256"],
    }
    copied_descriptor = tmp_path / "descriptor.yaml"
    copied_descriptor.write_bytes(a.selected.descriptor.read_bytes())
    functional, owner = CI.selected_functional_run(a.selected.run.parent, config, copied_descriptor)
    assert calls == [functional]
    assert functional.submission_dir == a.baseline
    assert functional.public_capsules == 2 and functional.hidden_capsules == 1
    assert functional.gate_clean is False
    assert owner.verify(a.baseline) == a.binding
    config["waive_functional_gate"] = ["private_model_execution_incomplete"]
    with pytest.raises(ValueError, match="permits no functional waivers"):
        CI.selected_functional_run(a.selected.run.parent, config, copied_descriptor)


@pytest.mark.parametrize("fault", ["wrong_publication", "wrong_target", "wrong_candidate", "traversal"])
def test_worker_selection_rejects_stale_or_different_input_without_snapshot_admission(admitted, monkeypatch, fault):
    a = admitted
    config = {
        "phase1_validation_scope": "capsules",
        "waive_functional_gate": [],
        "functional_run_id": a.selected.run.name,
        "functional_submission_sha256": a.binding["submission_sha256"],
        "capsule_publication_sha256": a.publication["receipt"]["sha256"],
    }
    descriptor = a.selected.descriptor
    if fault == "wrong_publication":
        config["capsule_publication_sha256"] = "0" * 64
    elif fault == "wrong_target":
        descriptor = a.selected.run / "other-descriptor.yaml"
        descriptor.write_text("target: other\n")
    elif fault == "wrong_candidate":
        config["functional_submission_sha256"] = "0" * 64
    else:
        config["functional_run_id"] = "../run"
    monkeypatch.setattr(FI, "_functional_input_snapshot", lambda _: pytest.fail("unadmitted input reached V4 owner"))
    with pytest.raises(ValueError):
        CI.selected_functional_run(a.selected.run.parent, config, descriptor)
