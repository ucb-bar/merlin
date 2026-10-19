"""Requirements keep member identities, refusals, repetition and physical gaps."""

import pytest
from merlin_experiments.phase0.development_test_audit import build


def member(ordinal, shape, *, source=True):
    return dict(
        ordinal=ordinal,
        request_sha256=str(ordinal) * 64,
        request=dict(
            basis="public_joint_shape_spectrum",
            declaration=dict(pattern="matmul", geometry=shape, arithmetic=dict(contract="same"), name=str(ordinal)),
            spectrum_member="same_point",
            occurrence_proxy=32,
            purposes=["lower_extent:M", "upper_extent:N"],
        ),
        loader_identity=dict(sha256="a" * 64) if source else {},
        status="source_frozen_reference_budget_denied" if source else "source_selection_budget_denied",
        reason="retained original refusal" if not source else None,
    )


def cohort(members):
    return dict(
        label="public",
        manifest_sha256="b" * 64,
        manifest=dict(scope="public_architecture_informed_development_not_validation", members=members),
    )


def test_coverage_counts_unique_occurrences_and_retains_duplicate_refusal():
    one = member(0, dict(M=1, K=64, N=16))
    two = member(1, dict(M=1, K=64, N=16), source=False)
    three = member(2, dict(M=16, K=65, N=16))
    report = build([cohort([one, two, three])], evidence={"public:0": dict(status="recorded_exact_outputs")})
    assert len(report["members"]) == 3
    assert report["members"][1]["original_reason"] == "retained original refusal"
    assert report["redundancy"]["duplicate_members_beyond_first"] == 1
    assert report["occurrence_sampling"]["occurrence_proxy_sum"] == 32
    rules = {r["requirement"]: r for r in report["requirements"]}
    assert rules["independent_k_tail"]["members"] == ["public:2"]
    assert rules["device_resident_rhs_and_mutation_alias_clobber_controls"]["status"] == "UNPROVED"
    assert rules["complete_reduced_decoder_model"]["members"] == []
    assert not report["adequate_for_full_model_performance_claim"]
    assert report["no_member_removed"]


def test_capacity_pairs_are_coarse_brackets_not_adjacent_or_joint_live_proofs():
    rows = [member(i, dict(M=16, N=16, K=(i + 1) * 16)) for i in range(3)]
    for row, extent in zip(rows, [40, 60, 110], strict=True):
        row["request"]["capacity"] = dict(
            family="knee", variant="joint", regime="declared", working_set=dict(operand_bytes=extent)
        )
    selected = cohort(rows)
    selected["manifest"]["resource_basis"] = dict(operand_store_bytes=100, accumulator_bytes=100)
    report = build([selected])
    assert len(report["capacity_brackets"]) == 2
    assert all(row["paired"] for row in report["capacity_brackets"])
    assert all(row["adjacent_legal_allocation_proven"] is False for row in report["capacity_brackets"])
    assert report["capacity_brackets"][0]["lower"]["member"] == "public:0"
    assert report["capacity_brackets"][0]["upper"]["member"] == "public:1"
    with pytest.raises(ValueError, match="unique"):
        build([selected, selected])
    selected["manifest"]["members"].pop(0)
    with pytest.raises(ValueError, match="complete original"):
        build([selected])
