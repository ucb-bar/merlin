"""A two-arm claim must be decidable from measured cycles, and refutable in a declared direction."""
from __future__ import annotations

import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root

sys.path.insert(0, str(merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"))

import perf_paired_claim as P  # noqa: E402

ROLES = ["resident", "spilling"]


def _contract(**over):
    base = {"schema_version": 1, "analyzer": P.ANALYZER, "roles": list(ROLES),
            "expected_faster": "resident", "negative_control_capsule": None,
            "band": {"kind": "measured_replicate_dispersion", "declared_constant": None}}
    base.update(over)
    return base


def _descriptors(contract, names=("A", "B")):
    return [{"name": n, "performance": {"family": "PR", "claim": "DIFFERENTIAL",
                                        "acceptance": contract}} for n in names]


def _rows(pairs, reps=2):
    out = []
    for name, (a, b) in pairs.items():
        for r in range(reps):
            out.append({"capsule": name, "arm": ROLES[0], "replicate": f"r{r:03d}", "cycles": a})
            out.append({"capsule": name, "arm": ROLES[1], "replicate": f"r{r:03d}", "cycles": b})
    return out


def test_the_predicted_arm_winning_everywhere_is_established():
    out = P.analyze_paired_claim(_descriptors(_contract()),
                                 _rows({"A": (100, 150), "B": (200, 260)}))
    assert out["verdict"] == P.ESTABLISHED
    assert len(out["rows"]) == 2


def test_the_predicted_arm_losing_is_refuted_not_refused():
    out = P.analyze_paired_claim(_descriptors(_contract()),
                                 _rows({"A": (150, 100), "B": (200, 260)}))
    assert out["verdict"] == P.REFUTED
    assert out["verdict"] != P.REFUSED, "a measured contradiction is evidence, not missing evidence"
    assert "A" in out["members"]


def test_arms_that_tie_within_the_band_refute_a_directional_claim():
    out = P.analyze_paired_claim(_descriptors(_contract()),
                                 _rows({"A": (100, 100), "B": (200, 200)}))
    assert out["verdict"] == P.REFUTED


def test_a_contract_with_no_predicted_direction_is_refused():
    out = P.analyze_paired_claim(_descriptors(_contract(expected_faster=None)),
                                 _rows({"A": (100, 150)}))
    assert out["verdict"] == P.REFUSED
    assert "could contradict it" in out["reason"]


def test_an_either_direction_family_needs_only_a_separation():
    out = P.analyze_paired_claim(_descriptors(_contract(expected_faster=P.EITHER)),
                                 _rows({"A": (150, 100), "B": (200, 260)}))
    assert out["verdict"] == P.ESTABLISHED


def test_an_either_family_is_refuted_when_the_arms_are_indistinguishable():
    out = P.analyze_paired_claim(_descriptors(_contract(expected_faster=P.EITHER)),
                                 _rows({"A": (100, 100)}))
    assert out["verdict"] == P.REFUTED


def test_a_negative_control_that_moves_refuses_the_whole_cohort():
    """If the same program measured twice differs, the instrument is measuring itself."""
    contract = _contract(negative_control_capsule="A")
    out = P.analyze_paired_claim(_descriptors(contract),
                                 _rows({"A": (100, 180), "B": (200, 260)}))
    assert out["verdict"] == P.REFUSED
    assert out["control_breaches"]
    assert "measuring itself" in out["reason"]


def test_a_member_missing_an_arm_is_refused():
    rows = [r for r in _rows({"A": (100, 150)}) if r["arm"] == ROLES[0]]
    out = P.analyze_paired_claim(_descriptors(_contract(), names=("A",)), rows)
    assert out["verdict"] == P.REFUSED and out["missing"]


@pytest.mark.parametrize("bad", [{"roles": ["only_one"]}, {"roles": "resident"},
                                 {"analyzer": "someone.else/v1"}])
def test_a_malformed_contract_is_refused(bad):
    out = P.analyze_paired_claim(_descriptors(_contract(**bad)), _rows({"A": (100, 150)}))
    assert out["verdict"] == P.REFUSED


def test_every_shipped_differential_family_predicts_a_direction():
    """The live profile must not carry a paired claim that no measurement could contradict."""
    profile = yaml.safe_load(
        (repo_root() / "merlin/contract/capsules/profiles/_perf.yaml").read_text(encoding="utf-8"))
    for sweep in profile["sweeps"]:
        performance = sweep["base"]["performance"]
        if performance.get("claim") != "DIFFERENTIAL":
            continue
        acceptance = performance.get("acceptance")
        assert isinstance(acceptance, dict), f"{sweep['id']} has no acceptance contract"
        predicted = acceptance.get("expected_faster")
        assert predicted, f"{sweep['id']} predicts no direction, so it cannot be wrong"
        roles = list(sweep.get("comparison_roles") or [])
        assert predicted == P.EITHER or predicted in roles, (
            f"{sweep['id']} predicts {predicted!r}, which is not one of its roles {roles}")
