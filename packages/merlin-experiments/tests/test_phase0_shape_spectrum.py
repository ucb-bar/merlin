"""Joint-shape planning preserves correlations and never admits validation inputs."""

import copy

import pytest
from merlin_experiments.phase0.shape_spectrum import SUMMARY_SCHEMA, plan


def summary(rows):
    return {"schema": SUMMARY_SCHEMA, "scope": "public_development", "source_sha256": ["a" * 64], "members": rows}


def row(m, k, n, *, count=1, contract="signed/raw", family="contraction"):
    return {"family": family, "contract": contract, "shape": {"M": m, "K": k, "N": n}, "occurrences": count}


def select(document, **kwargs):
    return plan(document, axis_granules={"M": 8, "K": 4, "N": 16}, resource_sha256="b" * 64, **kwargs)


def test_joint_distribution_preserves_tuples_and_bounds_without_cartesian_expansion():
    rows = [row(1, 120, 480, count=32), row(7, 60, 240), row(25, 24, 48), row(80, 12, 20)]
    result = select(summary(rows))
    original = {tuple(sorted(item["shape"].items())) for item in rows}
    assert result["status"] == "planned_complete" and result["missing_bins"] == 0
    for item in result["members"]:
        if not any(purpose.startswith("above_upper_extent:") for purpose in item["purposes"]):
            assert tuple(sorted(item["shape"].items())) in original
    for axis in ("M", "K", "N"):
        assert min(item["shape"][axis] for item in result["members"]) <= min(item["shape"][axis] for item in rows)
        assert max(item["shape"][axis] for item in result["members"]) > max(item["shape"][axis] for item in rows)
    assert len(result["members"]) <= len(rows) + 3
    assert sum(cell["occurrences"] for cell in result["occupied_bins"]) == 35
    assert all(value is False for value in result["claims"].values())


def test_semantics_layouts_and_morphologies_remain_separate():
    result = select(
        summary([row(1, 24, 32), row(1, 24, 32, contract="signed/grouped"), row(32, 24, 1), row(32, 1024, 32)])
    )
    bins = result["occupied_bins"]
    assert len(bins) == 4
    assert any("deep_k" in cell["morphology"] for cell in bins)
    assert any("gemv" in cell["morphology"] for cell in bins)
    assert {cell["contract"] for cell in bins} == {"signed/raw", "signed/grouped"}


def test_budget_exhaustion_retains_unrepresented_bins_and_envelope_requirements():
    result = select(summary([row(1, 12, 20), row(32, 128, 64)]), max_members=1, extra_density_points=0)
    assert result["status"] == "budget_incomplete"
    assert len(result["members"]) == 1 and result["deferred"] and result["missing_bins"] > 0
    assert any(any(p.startswith("above_upper_extent:") for p in item["purposes"]) for item in result["deferred"])
    assert result["unfulfilled_requirements"]


def test_deterministic_order_and_source_identity_are_preserved():
    document = summary([row(1, 24, 32), row(32, 48, 16)])
    before = copy.deepcopy(document)
    first = select(document)
    assert document == before
    reverse = {**document, "members": list(reversed(document["members"]))}
    second = select(reverse)
    assert first["members"] == second["members"] and first["occupied_bins"] == second["occupied_bins"]
    assert first["summary_sha256"] != second["summary_sha256"]


def test_granularity_and_density_are_explicit_policy():
    rows = [row(20 + i, 32, 48, count=1 if i != 3 else 99) for i in range(8)]
    coarse = select(summary(rows), extra_density_points=0)
    dense = select(summary(rows), extra_density_points=3)
    assert len(dense["members"]) >= len(coarse["members"])
    assert dense["policy"]["extra_density_points"] == 3
    assert dense["sha256"] != coarse["sha256"]
    assert any(item["shape"] == rows[3]["shape"] for item in dense["members"])


def test_repeated_geometry_aggregates_occurrences_before_density_selection():
    repeated = [row(23, 32, 48)] * 12
    others = [row(20 + i, 32, 48) for i in range(8) if i != 3]
    result = select(summary(repeated + others), extra_density_points=1)
    match = next(item for item in result["members"] if item["shape"] == repeated[0]["shape"])
    assert match["occurrence_proxy"] == 12
    assert sum(cell["occurrences"] for cell in result["occupied_bins"]) == 19


@pytest.mark.parametrize("scope", ["validation", "protected", "withheld_transfer", "sanitized_aggregate"])
def test_undisclosed_or_protected_metadata_cannot_enter_public_development_plan(scope):
    with pytest.raises(ValueError, match="public development"):
        select({**summary([row(1, 2, 3)]), "scope": scope})


@pytest.mark.parametrize("damage", ["zero", "bool", "oversize", "missing_granule", "payload", "unbounded"])
def test_invalid_geometry_payloads_and_budgets_refuse(damage):
    document = summary([row(1, 2, 3)])
    kwargs = {}
    if damage in {"zero", "bool", "oversize"}:
        document["members"][0]["shape"]["M"] = {"zero": 0, "bool": True, "oversize": 2**100}[damage]
    elif damage == "missing_granule":
        document["members"][0]["shape"]["heads"] = 3
    elif damage == "payload":
        document["members"][0]["golden_outputs"] = [1]
    else:
        kwargs["max_members"] = 2**30
    with pytest.raises(ValueError):
        select(document, **kwargs)
