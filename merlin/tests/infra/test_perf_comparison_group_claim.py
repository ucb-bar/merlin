"""A generated two-member comparison group must reach a cycle verdict without target knowledge."""
from __future__ import annotations

from merlin.perf import comparison_group_claim as P


REPLICATES = ("r000", "r001")


def _acceptance() -> dict:
    return {
        "schema_version": 1,
        "analyzer": P.ANALYZER,
        "program_arm": "candidate",
        "roles": ["island", "no_island"],
        "expected_faster": "no_island",
        "group_field": "comparison_group",
        "allowed_attribute_differences": ["comparison_role", "host_transform"],
        "replicates": {"exact_count": 2, "identities": list(REPLICATES)},
        "band": {"kind": "measured_replicate_dispersion", "declared_constant": None},
        "evidence": {"correctness_simulator": "functional", "correctness_tier": "L2",
                     "timing_simulator": "cycle_model", "timing_tier": "L3"},
    }


def _descriptors() -> list[dict]:
    rows = []
    for k in (32, 64):
        for role, transform in (("island", "xor_low_bit"), ("no_island", "none")):
            rows.append({
                "name": f"PB_{k}_{role}",
                "inputs": [
                    {"name": "A0", "role": "input", "shape": [16, k], "dtype": "i8"},
                    {"name": "W0", "role": "weight", "shape": [k, 16], "dtype": "i8"},
                    {"name": "W1", "role": "weight", "shape": [16, 16], "dtype": "i8"},
                ],
                "operation": {"op": "host_island_seam", "attributes": {
                    "M": 16, "K": k, "H": 16, "N": 16, "accelerator_contractions": 2,
                    "comparison_role": role, "host_transform": transform,
                }},
                "comparison_group": {"name": f"pb_k{k}", "role": role},
                "performance": {"family": "PB", "claim": "DIFFERENTIAL",
                                "acceptance": _acceptance(),
                                "falsifier": {"observation": "island_minus_no_island",
                                               "fires_when": "not_positive",
                                               "negative_control": "no_island"}},
            })
    return rows


def _results(*, reverse: bool = False) -> list[dict]:
    rows = []
    for k in (32, 64):
        costs = {"island": 120 + k, "no_island": 100 + k}
        if reverse:
            costs = {"island": 100 + k, "no_island": 120 + k}
        for role in ("island", "no_island"):
            for i, replicate in enumerate(REPLICATES):
                rows.append({"capsule": f"PB_{k}_{role}", "replicate": replicate,
                             "cycles": costs[role] + i, "simulator": "cycle_model", "tier": "L3",
                             "program_arm": "candidate", "artifact_sha256": "same-program",
                             "correct": True})
    return rows


def test_preflight_schedules_each_declared_member_once_per_lane_and_replicate() -> None:
    out = P.preflight_comparison_group_claim(_descriptors(), replicates=REPLICATES)
    assert out["status"] == "READY"
    assert len(out["cohort"]["groups"]) == 2
    assert len(out["expected_identities"]) == 4 * 2 * 2
    assert {row["program_arm"] for row in out["expected_identities"]} == {"candidate"}
    assert all("arm" not in row for row in out["expected_identities"])
    assert {row["comparison_role"] for row in out["expected_identities"]} == {
        "island", "no_island"}


def test_measured_group_direction_is_established_or_refuted() -> None:
    established = P.analyze_comparison_group_claim(_descriptors(), _results())
    assert established["verdict"] == "ESTABLISHED"
    assert all(row["delta_cycles"] > row["replicate_band"] for row in established["rows"])

    refuted = P.analyze_comparison_group_claim(_descriptors(), _results(reverse=True))
    assert refuted["verdict"] == "REFUTED"


def test_group_members_must_be_structurally_matched_except_declared_difference() -> None:
    descriptors = _descriptors()
    descriptors[1]["operation"]["attributes"]["accelerator_contractions"] = 1
    out = P.preflight_comparison_group_claim(descriptors, replicates=REPLICATES)
    assert out["status"] == "REFUSED"
    assert "differ outside" in out["refusal_reasons"][0]


def test_results_from_two_programs_cannot_be_mixed_into_one_group_verdict() -> None:
    results = _results()
    results[0]["artifact_sha256"] = "different-program"
    out = P.analyze_comparison_group_claim(_descriptors(), results)
    assert out["verdict"] == "REFUSED"
    assert "artifact" in out["reason"]


def test_results_from_the_wrong_program_arm_are_refused() -> None:
    results = _results()
    results[0]["program_arm"] = "baseline"
    out = P.analyze_comparison_group_claim(_descriptors(), results)
    assert out["verdict"] == "REFUSED"
    assert "program arms" in out["reason"]


def test_failed_member_grade_cannot_be_turned_into_a_cycle_verdict() -> None:
    results = _results()
    results[1]["correct"] = False
    out = P.analyze_comparison_group_claim(_descriptors(), results)
    assert out["verdict"] == "REFUSED"
    assert "passing correctness" in out["reason"]


def test_missing_member_grade_cannot_be_assumed_to_have_passed() -> None:
    results = _results()
    results[1].pop("correct")
    out = P.analyze_comparison_group_claim(_descriptors(), results)
    assert out["verdict"] == "REFUSED"
    assert "passing correctness" in out["reason"]
