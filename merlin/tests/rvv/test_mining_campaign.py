"""Generic compiler-mining campaign gates and holdout isolation."""
from __future__ import annotations

import pytest

from merlin.mining.campaign import Campaign, CandidateObservation, PartitionPolicy


def _capsules(policy, split, n=3):
    out = []
    i = 0
    while len(out) < n:
        name = f"capsule_{split}_{i}"
        if policy.split(name) == split:
            out.append(name)
        i += 1
    return out


def _rows(campaign, candidate="pass_x", speedup=1.1):
    ids = _capsules(campaign.partition, "train", 3) + _capsules(
        campaign.partition, "validation", 3)
    families = ["contraction", "layout", "reduction", "contraction", "layout", "reduction"]
    return [CandidateObservation(
        candidate=candidate, action_class="pass", capsule_id=capsule, family=family,
        workload="synthetic", baseline_ns=1100, candidate_ns=int(1100 / speedup),
        correctness_ok=True, baseline_code_digest=f"base{i}", candidate_code_digest=f"new{i}")
            for i, (capsule, family) in enumerate(zip(ids, families, strict=True))]


def test_candidate_must_transfer_to_validation_and_multiple_families():
    campaign = Campaign(excluded_models=frozenset({"paper_model"}))
    decision = campaign.decide(_rows(campaign))
    assert decision.accepted
    regressed = _rows(campaign, candidate="bad")
    first_validation = next(i for i, row in enumerate(regressed)
                            if campaign.partition.split(row.capsule_id) == "validation")
    row = regressed[first_validation]
    regressed[first_validation] = CandidateObservation(
        **{**row.__dict__, "candidate_ns": row.baseline_ns * 2})
    assert not campaign.decide(regressed).accepted


def test_paper_models_and_heldout_capsules_are_forbidden():
    campaign = Campaign(excluded_models=frozenset({"paper_model"}))
    rows = _rows(campaign)
    row = rows[0]
    rows[0] = CandidateObservation(**{**row.__dict__, "workload": "paper_model"})
    with pytest.raises(ValueError, match="leakage"):
        campaign.decide(rows)
    heldout = _capsules(campaign.partition, "heldout", 1)[0]
    rows = _rows(campaign)
    row = rows[0]
    rows[0] = CandidateObservation(**{**row.__dict__, "capsule_id": heldout})
    with pytest.raises(ValueError, match="heldout"):
        campaign.decide(rows)


def test_freeze_requires_two_empty_sweeps():
    campaign = Campaign(excluded_models=frozenset())
    rejected = campaign.decide(_rows(campaign, speedup=1.0))
    campaign.finish_sweep([rejected])
    with pytest.raises(ValueError, match="needs 2"):
        campaign.freeze(development_corpus_sha256="a", policy_sha256="b", runtime_sha256="c")
    campaign.finish_sweep([rejected])
    record = campaign.freeze(development_corpus_sha256="a", policy_sha256="b", runtime_sha256="c")
    assert record["status"] == "frozen" and record["convergence"]["observed_empty_sweeps"] == 2


def test_partition_is_stable_and_disjoint():
    policy = PartitionPolicy()
    assert policy.bucket("capsule") == policy.bucket("capsule")
    assert policy.split("capsule") in {"train", "validation", "heldout"}
