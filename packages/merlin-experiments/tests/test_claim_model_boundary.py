"""Held-out evaluation models cannot enter a public Phase 0/1 candidate payload."""

import pytest
import yaml
from merlin_experiments.phase0.claim_boundary import assert_no_claim_capsules


@pytest.mark.parametrize(
    "entry",
    [
        {"cat": "model", "name": "SY_model_tiny_llama", "model": "tiny_llama"},
        {"cat": "model", "name": "other_name", "model": "resnet50_v1_5"},
        {"cat": "model", "name": "other_name", "loader": "workloads/resnet50_v1_5/loader.py"},
    ],
)
def test_claim_model_source_is_rejected_even_when_renamed(entry):
    with pytest.raises(ValueError, match="held-out claim model"):
        assert_no_claim_capsules([entry], ["resnet50", "tiny_llama"])


def test_derivation_model_with_overlapping_word_is_allowed():
    assert_no_claim_capsules(
        [{"cat": "model", "name": "M0_small_llama", "model": "small_llama"}],
        ["resnet50", "tiny_llama"],
    )


def test_manifest_records_count_only_owner_obligation(tmp_path):
    from merlin_experiments.phase0.provenance import update_provenance_manifest

    obligation = {
        "schema": "claim_model_evaluation_v1",
        "source": "workload_spec.models",
        "model_count": 2,
        "visibility": "owner_only_after_phase1_freeze",
        "public_capsules_emitted": 0,
        "status": "awaiting_phase1_freeze",
    }
    path = update_provenance_manifest(
        [],
        cap_root=tmp_path,
        target="sample_target",
        performance_record={"phase": {"category": "_perf"}},
        unbuilt_roster=[],
        claim_model_evaluation=obligation,
    )
    manifest = yaml.safe_load(path.read_text())
    assert manifest["claim_model_evaluation"]["sample_target"] == obligation
    assert manifest["phase_corpora"]["sample_target"]["phase1"]["generated_members"] == []
