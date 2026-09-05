"""A two-arm claim must be decidable from measured cycles, and refutable in a declared direction."""
from __future__ import annotations

import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root

sys.path.insert(0, str(merlin_dir() / "experiments" / "gemmini_perf_bench" / "scripts"))

import perf_paired_claim as P  # noqa: E402

from merlin.perf import barrier_arms as BA  # noqa: E402

ROLES = ["resident", "spilling"]


def _contract(**over):
    base = {"schema_version": 1, "analyzer": P.ANALYZER, "roles": list(ROLES),
            "expected_faster": "resident", "negative_control_capsule": None,
            "band": {"kind": "measured_replicate_dispersion", "declared_constant": None}}
    base.update(over)
    return base


#: A falsifier that asserts a DIRECTION. ``_validate_performance_block`` makes this a required field
#: on every real descriptor, so the fixture carries one too -- a descriptor without it is not a shape
#: the corpus can produce, and pretending otherwise would test a stricter input than exists.
DIRECTION_FALSIFIER = {"observation": "paired_cycle_difference",
                       "fires_when": "the_arms_cost_the_same_within_their_band",
                       "negative_control": "the_byte_identical_pair"}

#: A falsifier that asserts PER-UNIT GROWTH. Declaring this obliges the cohort to be decided by the
#: growth test as well as the direction test.
GROWTH_FALSIFIER = {"observation": BA.PER_UNIT_GROWTH_OBSERVATION,
                    "fires_when": "the_saving_does_not_grow_with_the_removed_barrier_count",
                    "negative_control": "the_single_job_member_whose_arms_are_byte_identical"}


def _descriptors(contract, names=("A", "B"), falsifier=None):
    return [{"name": n, "performance": {"family": "PR", "claim": "DIFFERENTIAL",
                                        "acceptance": contract,
                                        "falsifier": dict(falsifier or DIRECTION_FALSIFIER)}}
            for n in names]


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


def test_every_shipped_paired_family_predicts_a_direction():
    """A PAIRED claim must name which arm it expects to win, or no measurement can contradict it.

    Scoped to the families this analyzer actually decides. A differential is not always a two-arm
    direction: the residency family's claim is that a per-unit rate DIFFERS across a residency
    boundary, and its falsifier fires when the two rates AGREE -- so it is falsifiable without
    predicting a winner, and demanding one of it was this test over-generalising. Requiring a key
    the deciding analyzer does not read is not a stricter contract, it is drift: the residency
    family carried `expected_faster` for exactly that reason and its own preflight then refused it
    as "an acceptance contract this analyzer does not implement".
    """
    profile = yaml.safe_load(
        (repo_root() / "merlin/contract/capsules/profiles/_perf.yaml").read_text(encoding="utf-8"))
    checked = 0
    for sweep in profile["sweeps"]:
        performance = sweep["base"]["performance"]
        if performance.get("claim") != "DIFFERENTIAL":
            continue
        # FALSIFIABILITY AND WIRING ARE DIFFERENT PROPERTIES, and this conflated them. The invariant
        # this test is named for is that something could contradict the claim -- which a declared
        # falsifier observation provides. A frozen `acceptance` block is a separate thing: it names
        # the analyzer that DECIDES the claim, and a family can be perfectly falsifiable while its
        # decision procedure is not yet wired. The fusion family is exactly that: its falsifier is
        # `fused_cycles_minus_sum_of_parts`, a quantitative contradiction condition that needs no
        # `expected_faster` because the comparison is fused-against-the-sum-of-its-parts rather than
        # arm-against-arm. Demanding an acceptance block of it reported an unfalsifiable claim, which
        # is not what is wrong with it.
        acceptance = performance.get("acceptance")
        assert (performance.get("falsifier") or {}).get("observation"), (
            f"{sweep['id']} is DIFFERENTIAL and names no falsifier observation, so no measurement "
            f"could contradict it")
        if not isinstance(acceptance, dict):
            continue
        if str(acceptance.get("analyzer") or "").split(".")[0] != "perf_paired_claim":
            # Decided elsewhere; that analyzer's own contract test owns its shape. What must still
            # hold is that it declares a falsifier, i.e. that something could contradict it.
            assert (performance.get("falsifier") or {}).get("observation"), (
                f"{sweep['id']} is DIFFERENTIAL and names no falsifier observation")
            continue
        predicted = acceptance.get("expected_faster")
        assert predicted, f"{sweep['id']} predicts no direction, so it cannot be wrong"
        roles = list(sweep.get("comparison_roles") or [])
        assert predicted == P.EITHER or predicted in roles, (
            f"{sweep['id']} predicts {predicted!r}, which is not one of its roles {roles}")
        checked += 1
    assert checked >= 3, "the direction check matched almost nothing; it has stopped testing"


def test_a_family_may_not_declare_a_key_its_own_analyzer_does_not_implement():
    """The drift this cost a run: a bulk edit added `expected_faster` to six families at once,
    including one whose analyzer never reads it, and that family's preflight then refused every
    descriptor it had."""
    import importlib
    import sys

    profile = yaml.safe_load(
        (repo_root() / "merlin/contract/capsules/profiles/_perf.yaml").read_text(encoding="utf-8"))
    scripts = str(repo_root() / "merlin/experiments/gemmini_perf_bench/scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    for sweep in profile["sweeps"]:
        acceptance = (sweep["base"]["performance"].get("acceptance") or {})
        analyzer = str(acceptance.get("analyzer") or "")
        module_name = analyzer.split(".")[0]
        if not module_name:
            continue
        try:
            module = importlib.import_module(module_name)
            supported = module.supported_acceptance()
        except Exception:  # noqa: BLE001 - an unimportable analyzer is a separate finding
            continue
        extra = sorted(set(acceptance) - set(supported))
        assert not extra, (
            f"{sweep['id']} declares {extra} which {module_name} does not implement; its preflight "
            f"will refuse every descriptor in the family")


# ------------------------------------------------------------------ the per-unit growth falsifier

def test_a_growth_family_folds_the_per_unit_verdict_into_an_established_direction():
    """Direction and growth are two assertions and this family declares both."""
    out = P.analyze_paired_claim(
        _descriptors(_contract(), falsifier=GROWTH_FALSIFIER),
        _rows({"A": (100, 150), "B": (200, 500)}),
        per_unit={"A": 1, "B": 15})
    assert out["verdict"] == P.ESTABLISHED
    assert out["growth"]["verdict"] == BA.ESTABLISHED
    # (300 - 50) / (15 - 1)
    assert out["growth"]["measured"]["cycles_per_removed_barrier"] == pytest.approx(250 / 14)
    assert out["growth"]["measured"]["distinct_removed_counts"] == [1.0, 15.0]


def test_a_constant_saving_passes_the_direction_test_and_still_fails_the_growth_claim():
    """THE CASE THE DIRECTION TEST CANNOT SEE, and the reason this gate exists.

    Both members are cheaper on the predicted arm by the same 50 cycles, while one removed a single
    completion point and the other removed fifteen. A direction test calls that ESTABLISHED -- the
    predicted arm won everywhere. But a saving that does not grow with the count removed is a FIXED
    cost wearing a per-unit claim's clothes, and this family's falsifier is precisely that the saving
    must grow. Reporting ESTABLISHED here would be a verdict on an assertion nothing evaluated.
    """
    direction_only = P.analyze_paired_claim(
        _descriptors(_contract()), _rows({"A": (100, 150), "B": (200, 250)}))
    assert direction_only["verdict"] == P.ESTABLISHED, "the direction test is blind to this"

    out = P.analyze_paired_claim(
        _descriptors(_contract(), falsifier=GROWTH_FALSIFIER),
        _rows({"A": (100, 150), "B": (200, 250)}),
        per_unit={"A": 1, "B": 15})
    assert out["verdict"] == P.REFUTED
    assert out["growth"]["verdict"] == BA.REFUTED
    assert "GROWTH" in out["reason"]


def test_a_growth_family_with_no_per_unit_counts_refuses_rather_than_passing():
    """A check that could not run must not report success. Without the counts the growth claim is
    simply undecided, and the direction result does not stand in for it."""
    out = P.analyze_paired_claim(
        _descriptors(_contract(), falsifier=GROWTH_FALSIFIER),
        _rows({"A": (100, 150), "B": (200, 500)}))
    assert out["verdict"] == P.REFUSED
    assert out["direction_verdict"] == P.ESTABLISHED, "the direction result is reported, not lost"
    assert "no per-unit counts" in out["reason"]


def test_a_member_with_no_per_unit_count_refuses_the_cohort():
    out = P.analyze_paired_claim(
        _descriptors(_contract(), falsifier=GROWTH_FALSIFIER),
        _rows({"A": (100, 150), "B": (200, 500)}),
        per_unit={"A": 1})
    assert out["verdict"] == P.REFUSED and out["members"] == ["B"]


def test_one_distinct_removed_count_is_an_anecdote_not_a_growth_result():
    out = P.analyze_paired_claim(
        _descriptors(_contract(), falsifier=GROWTH_FALSIFIER),
        _rows({"A": (100, 150), "B": (200, 500)}),
        per_unit={"A": 4, "B": 4})
    assert out["verdict"] == P.REFUSED
    assert out["growth"]["verdict"] == BA.REFUSED


def test_a_direction_falsifier_ignores_per_unit_counts_entirely():
    """The gate is keyed on the DECLARED observation, so a family that never claimed growth is
    decided exactly as before even when counts happen to be available."""
    out = P.analyze_paired_claim(
        _descriptors(_contract()), _rows({"A": (100, 150), "B": (200, 250)}),
        per_unit={"A": 1, "B": 15})
    assert out["verdict"] == P.ESTABLISHED and "growth" not in out


def test_the_shipped_synchronization_family_declares_the_growth_falsifier():
    """Pins the wiring to the real declaration: whichever family claims a per-unit saving must be the
    one the growth gate fires on, so a rename cannot silently detach the check from its family."""
    profile = yaml.safe_load(
        (repo_root() / "merlin/contract/capsules/profiles/_perf.yaml").read_text(encoding="utf-8"))
    declaring = [s["id"] for s in profile["sweeps"]
                 if (s["base"]["performance"].get("falsifier") or {}).get("observation")
                 == BA.PER_UNIT_GROWTH_OBSERVATION]
    assert declaring, (
        "no shipped family declares the per-unit growth observation, so the growth gate is wired to "
        "nothing -- either a family lost its falsifier or the constant drifted")
