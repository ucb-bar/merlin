"""The family-agnostic affine PREDICTS procedure reads its bounds and never invents one."""
from __future__ import annotations

import copy
import sys

import pytest
import yaml

from merlin.common.paths import repo_root


SCRIPTS = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(SCRIPTS))
import perf_affine_claim as AF  # noqa: E402


METRIC = "gsim_L3_cycles"
REPLICATES = tuple(__import__("perf_pk_claim")._ACCEPTANCE_BASE["replicates"]["identities"])


def _contract() -> dict:
    """A frozen acceptance contract shaped exactly like the PM sweep's, at four points."""
    return {
        "schema_version": AF.SCHEMA_VERSION,
        "analyzer": AF.ANALYZER,
        "fit": {
            "form": "affine",
            "method": "ordinary_least_squares_all_L3_replicates",
            "independent_variable": "output_elements",
            "variable_source": {"kind": "output_elements", "lhs": "A0", "weight": "W"},
            "dependent_metric": METRIC,
        },
        "cohort": {
            "operation": "matmul",
            "fixed_fields": ["operation", "operand_dtype", "accum_dtype"],
            "exact_points": 4,
        },
        "replicates": {"exact_count": len(REPLICATES), "identities": list(REPLICATES)},
        "evidence": {
            "correctness_simulator": "spike", "correctness_tier": "L2",
            "timing_simulator": "gsim", "timing_tier": "L3",
            "spike_cycles_citable": False,
        },
        "thresholds": {
            "slope_min_exclusive": 0.0,
            "r_squared_min_inclusive": 0.95,
            "residual_bound": {
                "predicate": "abs_residual_le_max_of_floor_and_fraction_of_observed",
                "absolute_floor_cycles": 64,
                "observed_cycle_fraction": 0.1,
            },
        },
    }


def _descriptor(index: int, m: int) -> dict:
    """One PM-shaped member: a fixed reduction depth and N, a varying output extent."""
    return {
        "name": f"PM{index:02d}_m{m}n16",
        "kind": "model_slice",
        "label": "dev",
        "inputs": [
            {"name": "W", "role": "weight", "shape": [16, 16], "dtype": "i8"},
            {"name": "A0", "role": "input", "shape": [m, 16], "dtype": "i8"},
        ],
        "operation": {
            "op": "matmul",
            "attributes": {"lhs": "A0", "weight": "W", "out": "Y0", "epilogue": [],
                           "output_dtype": "i32"},
        },
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        "performance": {
            "level": "L1_tile",
            "family": "PM",
            "lever": "parallel_extents",
            "claim": "PREDICTS",
            "comparand": {"kind": "fitted_prediction",
                          "against": "measured_cycles_same_member"},
            "falsifier": {
                "observation": "residual_cycles_by_output_tile_count",
                "fires_when": "residuals_are_not_bounded_after_rate_and_intercept_fit",
                "negative_control": "fixed_K_across_all_M_and_N_points",
            },
            "acceptance": _contract(),
        },
    }


@pytest.fixture
def descriptors() -> list[dict]:
    return [_descriptor(index, m) for index, m in enumerate((16, 32, 48, 64))]


def _x(descriptor: dict) -> int:
    """The declared output extent M*N, read structurally the way the contract declares it."""
    return descriptor["inputs"][1]["shape"][0] * descriptor["inputs"][0]["shape"][1]


def _rows(descriptors: list[dict], cycles=lambda x: 3 * x + 200) -> list[dict]:
    return [
        {"capsule": descriptor["name"], "replicate": replicate, "tier": "L3",
         "simulator": "gsim", METRIC: cycles(_x(descriptor))}
        for descriptor in descriptors for replicate in REPLICATES
    ]


def _row(rows: list[dict], capsule: str, replicate: str = "r000") -> dict:
    return next(row for row in rows
                if row["capsule"] == capsule and row["replicate"] == replicate)


def _retune(descriptors: list[dict], **thresholds) -> None:
    """Re-freeze one threshold across every member, so the cohort still agrees."""
    for descriptor in descriptors:
        descriptor["performance"]["acceptance"]["thresholds"].update(thresholds)


# --------------------------------------------------------------------------------------
# ACCEPTED
# --------------------------------------------------------------------------------------

def test_exact_affine_evidence_inside_every_bound_is_accepted(descriptors):
    result = AF.analyze_affine_claim(descriptors, _rows(descriptors))
    assert result["verdict"] == AF.ACCEPTED
    assert result["family"] == "PM"
    assert "reason" not in result and "reasons" not in result
    measured = result["measured"]
    assert measured["slope_cycles_per_unit"] > 0
    assert measured["slope_cycles_per_unit"] == pytest.approx(3.0)
    assert measured["intercept_cycles"] == pytest.approx(200.0)
    assert measured["r_squared"] == pytest.approx(1.0)
    assert measured["n_observations"] == len(descriptors) * len(REPLICATES)
    assert measured["independent_variable"] == "output_elements"
    assert measured["distinct_x"] == [256.0, 512.0, 768.0, 1024.0]


def test_a_verdict_is_always_one_of_the_modules_own_three_constants(descriptors):
    """Guards the spelling itself: a test asserting literals once passed against a mismatch."""
    assert len({AF.ACCEPTED, AF.REFUTED, AF.REFUSED}) == 3
    for evidence in (_rows(descriptors),
                     _rows(descriptors, lambda x: 5000 - 2 * x),
                     _rows(descriptors)[:-1]):
        verdict = AF.analyze_affine_claim(descriptors, evidence)["verdict"]
        assert verdict in {AF.ACCEPTED, AF.REFUTED, AF.REFUSED}


def test_a_real_frozen_contract_from_the_profile_is_decided_by_this_analyzer():
    """The shipped PM contract (an ``output_elements`` axis) is evaluated, not just well-formed.

    This reads the LIVE profile rather than a stand-in, so a contract edit that made the family
    undecidable would fail here. It follows PM because the conv family PV, which formerly stood in
    this test, is now recorded blocked_unimplemented: the integer reference engine has no CONV2D
    definition, so no PV member can be captured at all.
    """
    profile = yaml.safe_load(
        (repo_root() / "merlin/contract/capsules/profiles/_perf.yaml").read_text())
    sweep = next(row for row in profile["sweeps"] if row["id"] == "PM")
    acceptance = sweep["base"]["performance"]["acceptance"]
    assert acceptance["analyzer"] == AF.ANALYZER
    assert acceptance["fit"]["variable_source"]["kind"] == "output_elements"

    tile = 16
    members, rows = [], []
    extents = [(m * tile, n * tile) for m in (1, 2, 3, 4) for n in (1, 2, 3, 4)]
    assert len(extents) == acceptance["cohort"]["exact_points"]
    for index, (m, n) in enumerate(extents):
        name = f"PM{index:02d}_m{m}n{n}"
        members.append({
            "name": name,
            "inputs": [
                {"name": "A0", "role": "input", "shape": [m, tile], "dtype": "i8"},
                {"name": "W", "role": "weight", "shape": [tile, n], "dtype": "i8"},
            ],
            "performance": {"family": "PM", "claim": "PREDICTS",
                            "acceptance": copy.deepcopy(acceptance)},
        })
        # cycles affine in the output extent, which is exactly what the contract fits
        rows += [{"capsule": name, "replicate": replicate,
                  acceptance["fit"]["dependent_metric"]: 3 * (m * n) + 200}
                 for replicate in acceptance["replicates"]["identities"]]

    result = AF.analyze_affine_claim(members, rows)
    assert result["verdict"] == AF.ACCEPTED
    assert result["measured"]["slope_cycles_per_unit"] == pytest.approx(3.0)


# --------------------------------------------------------------------------------------
# REFUTED -- complete evidence that misses a bound is a refutation, never a refusal
# --------------------------------------------------------------------------------------

def _assert_refuted_not_refused(result: dict) -> dict:
    assert result["verdict"] == AF.REFUTED
    assert result["verdict"] != AF.REFUSED, "a real refutation was downgraded to a refusal"
    assert "reason" not in result, "a refutation must not carry a refusal reason"
    assert result["reasons"], "a refutation must say which bound it missed"
    assert result["measured"]["n_observations"] == 4 * len(REPLICATES)
    return result


def test_slope_at_or_below_the_predeclared_floor_refutes(descriptors):
    result = _assert_refuted_not_refused(
        AF.analyze_affine_claim(descriptors, _rows(descriptors, lambda x: 5000 - 2 * x)))
    assert result["measured"]["slope_cycles_per_unit"] == pytest.approx(-2.0)
    assert any("slope" in reason for reason in result["reasons"])
    assert result["breaches"] == []


def test_flat_evidence_refutes_the_required_positive_slope(descriptors):
    result = _assert_refuted_not_refused(
        AF.analyze_affine_claim(descriptors, _rows(descriptors, lambda x: 900)))
    assert result["measured"]["slope_cycles_per_unit"] == pytest.approx(0.0)
    assert any("slope" in reason for reason in result["reasons"])


def test_r_squared_below_the_predeclared_minimum_refutes(descriptors):
    _retune(descriptors, r_squared_min_inclusive=0.999)
    rows = _rows(descriptors)
    for replicate in REPLICATES:
        _row(rows, "PM01_m32n16", replicate)[METRIC] += 100
    result = _assert_refuted_not_refused(AF.analyze_affine_claim(descriptors, rows))
    assert result["measured"]["slope_cycles_per_unit"] > 0
    assert result["measured"]["r_squared"] < 0.999
    assert any("r_squared" in reason for reason in result["reasons"])
    assert result["breaches"] == [], "this refutation is about the fit, not the residuals"


def test_a_residual_outside_the_predeclared_bound_refutes(descriptors):
    rows = _rows(descriptors)
    for replicate in REPLICATES:
        _row(rows, "PM01_m32n16", replicate)[METRIC] += 5000
    result = _assert_refuted_not_refused(AF.analyze_affine_claim(descriptors, rows))
    assert any("residual bound" in reason for reason in result["reasons"])
    assert result["breaches"]
    for breach in result["breaches"]:
        assert breach["residual"] > breach["allowed"]
        assert breach["allowed"] == max(64.0, 0.1 * breach["observed_cycles"])


# --------------------------------------------------------------------------------------
# REFUSED -- incomplete or malformed evidence is never scored
# --------------------------------------------------------------------------------------

def _drop_acceptance(ds, rows):
    ds[1]["performance"]["acceptance"] = None


def _disagree(ds, rows):
    ds[2]["performance"]["acceptance"]["thresholds"]["r_squared_min_inclusive"] = 0.5


def _other_analyzer(ds, rows):
    for descriptor in ds:
        descriptor["performance"]["acceptance"]["analyzer"] = "perf_pk_claim.analyze/v1"


def _non_positive_metric(ds, rows):
    _row(rows, "PM00_m16n16")[METRIC] = 0


def _duplicate_pair(ds, rows):
    rows[-1] = copy.deepcopy(rows[-2])


def _wrong_cohort_size(ds, rows):
    ds.pop()


def _no_variation(ds, rows):
    for descriptor in ds:
        descriptor["inputs"][1]["shape"][0] = 16


def _not_a_predicts_claim(ds, rows):
    for descriptor in ds:
        descriptor["performance"]["claim"] = "IMPROVES"


def _two_families(ds, rows):
    ds[3]["performance"]["family"] = "PV"


def _row_outside_the_cohort(ds, rows):
    rows.append({"capsule": "PM99_m99n16", "replicate": "r000", METRIC: 1})


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (_drop_acceptance, "no frozen acceptance contract"),
        (_disagree, "disagree about the frozen acceptance contract"),
        (_other_analyzer, "names a different analyzer"),
        (_non_positive_metric, "no positive value for the dependent metric"),
        (_duplicate_pair, "reported twice"),
        (_wrong_cohort_size, "not the exact predeclared size"),
        (_no_variation, "no slope is identifiable"),
        (_not_a_predicts_claim, "PREDICTS claims only"),
        (_two_families, "more than one family"),
        (_row_outside_the_cohort, "outside the cohort"),
    ],
)
def test_incomplete_or_malformed_evidence_is_refused(descriptors, mutate, reason):
    rows = _rows(descriptors)
    mutate(descriptors, rows)
    result = AF.analyze_affine_claim(descriptors, rows)
    assert result["verdict"] == AF.REFUSED
    assert reason in result["reason"]
    assert "measured" not in result, "a refusal must not report a fit"


@pytest.mark.parametrize(
    "evidence",
    [None, [], "not-a-sequence", {}],
)
def test_absent_descriptors_or_results_are_refused(descriptors, evidence):
    assert AF.analyze_affine_claim(evidence, _rows(descriptors))["verdict"] == AF.REFUSED
    assert AF.analyze_affine_claim(descriptors, evidence)["verdict"] == AF.REFUSED


def test_a_replicate_cohort_of_the_wrong_size_is_refused(descriptors):
    rows = _rows(descriptors)
    rows.append({"capsule": "PM00_m16n16", "replicate": "r003", METRIC: 968})
    result = AF.analyze_affine_claim(descriptors, rows)
    assert result["verdict"] == AF.REFUSED
    assert "exact predeclared (capsule, replicate) cohort" in result["reason"]
    # the surplus row is named, not merely counted, so the refusal is actionable
    assert result["n_unexpected"] == 1
    assert ("PM00_m16n16", "r003") in [tuple(pair) for pair in result["unexpected"]]


# --------------------------------------------------------------------------------------
# Thresholds are read, never defaulted
# --------------------------------------------------------------------------------------

def _pop_thresholds(contract):
    contract.pop("thresholds")


def _pop_slope_min(contract):
    contract["thresholds"].pop("slope_min_exclusive")


def _pop_r2_min(contract):
    contract["thresholds"].pop("r_squared_min_inclusive")


def _pop_residual_bound(contract):
    contract["thresholds"].pop("residual_bound")


def _pop_residual_floor(contract):
    contract["thresholds"]["residual_bound"].pop("absolute_floor_cycles")


def _pop_residual_fraction(contract):
    contract["thresholds"]["residual_bound"].pop("observed_cycle_fraction")


def _pop_variable_source(contract):
    contract["fit"].pop("variable_source")


def _pop_dependent_metric(contract):
    contract["fit"].pop("dependent_metric")


def _pop_exact_points(contract):
    contract["cohort"].pop("exact_points")


@pytest.mark.parametrize(
    "strip",
    [_pop_thresholds, _pop_slope_min, _pop_r2_min, _pop_residual_bound,
     _pop_residual_floor, _pop_residual_fraction, _pop_variable_source,
     _pop_dependent_metric, _pop_exact_points],
)
def test_a_contract_that_omits_a_bound_is_refused_not_scored_against_a_default(
        descriptors, strip):
    for descriptor in descriptors:
        strip(descriptor["performance"]["acceptance"])
    result = AF.analyze_affine_claim(descriptors, _rows(descriptors))
    assert result["verdict"] == AF.REFUSED, "a missing bound was silently defaulted"
    assert "measured" not in result


# --------------------------------------------------------------------------------------
# independent_value -- a structural read of declared shapes, or None
# --------------------------------------------------------------------------------------

CONV = {
    "name": "PV00_h8c4",
    "inputs": [
        {"name": "W", "role": "weight", "shape": [36, 8], "dtype": "i8"},
        {"name": "IFM", "role": "input", "shape": [1, 8, 8, 4], "dtype": "i8"},
        {"name": "A0", "role": "input", "shape": [32, 16], "dtype": "i8"},
        {"name": "SCALAR", "role": "input", "dtype": "i8"},
    ],
}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ({"kind": "output_elements", "lhs": "A0", "weight": "W"}, 32 * 8),
        ({"kind": "input_elements", "input": "IFM"}, 1 * 8 * 8 * 4),
        ({"kind": "input_elements", "input": "W"}, 36 * 8),
        ({"kind": "input_dim", "input": "IFM", "axis": 3}, 4),
        ({"kind": "input_dim", "input": "IFM", "axis": 1}, 8),
        ({"kind": "input_dim", "input": "A0", "axis": 0}, 32),
    ],
)
def test_independent_value_reads_each_kind_from_declared_operand_shapes(source, expected):
    assert AF.independent_value(CONV, source) == expected


def test_every_declared_variable_kind_has_a_read_covered_here():
    assert set(AF._VARIABLE_KINDS) == {"output_elements", "input_elements", "input_dim"}


@pytest.mark.parametrize(
    "source",
    [
        {"kind": "output_elements", "lhs": "A0", "weight": "MISSING"},
        {"kind": "output_elements", "lhs": "MISSING", "weight": "W"},
        {"kind": "output_elements", "lhs": "A0", "weight": "SCALAR"},
        {"kind": "output_elements", "lhs": "A0", "weight": "IFM_1D"},
        {"kind": "input_elements", "input": "MISSING"},
        {"kind": "input_elements", "input": "SCALAR"},
        {"kind": "input_dim", "input": "MISSING", "axis": 0},
        {"kind": "input_dim", "input": "SCALAR", "axis": 0},
        {"kind": "input_dim", "input": "IFM", "axis": 4},
        {"kind": "input_dim", "input": "IFM", "axis": -1},
        {"kind": "input_dim", "input": "IFM"},
        {"kind": "input_dim", "input": "IFM", "axis": "3"},
        {"kind": "an_axis_this_module_never_declared", "input": "IFM"},
        {},
    ],
)
def test_independent_value_returns_none_rather_than_guessing(source):
    descriptor = copy.deepcopy(CONV)
    descriptor["inputs"].append({"name": "IFM_1D", "role": "weight", "shape": [8]})
    assert AF.independent_value(descriptor, source) is None


def test_independent_value_returns_none_when_the_descriptor_declares_no_inputs():
    for descriptor in ({}, {"inputs": None}, {"inputs": [{"role": "weight"}]}):
        for source in ({"kind": "output_elements", "lhs": "A0", "weight": "W"},
                       {"kind": "input_elements", "input": "IFM"},
                       {"kind": "input_dim", "input": "IFM", "axis": 0}):
            assert AF.independent_value(descriptor, source) is None


def test_an_underivable_independent_variable_refuses_the_whole_claim(descriptors):
    descriptors[2]["inputs"][1].pop("shape")
    result = AF.analyze_affine_claim(descriptors, _rows(descriptors[:2] + descriptors[3:]))
    assert result["verdict"] == AF.REFUSED
    assert "not derivable" in result["reason"]
    assert result["capsule"] == "PM02_m48n16"


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------

def test_the_verdict_and_the_fit_do_not_depend_on_input_order(descriptors):
    rows = _rows(descriptors)
    forward = AF.analyze_affine_claim(descriptors, rows)
    reverse = AF.analyze_affine_claim(list(reversed(descriptors)), list(reversed(rows)))
    assert reverse["verdict"] == forward["verdict"] == AF.ACCEPTED
    assert reverse["measured"]["slope_cycles_per_unit"] == pytest.approx(
        forward["measured"]["slope_cycles_per_unit"])
    assert reverse["measured"]["distinct_x"] == forward["measured"]["distinct_x"]
    assert reverse == forward


def test_a_refutation_is_also_order_independent(descriptors):
    rows = _rows(descriptors)
    for replicate in REPLICATES:
        _row(rows, "PM01_m32n16", replicate)[METRIC] += 5000
    forward = AF.analyze_affine_claim(descriptors, rows)
    reverse = AF.analyze_affine_claim(list(reversed(descriptors)), list(reversed(rows)))
    assert reverse["verdict"] == forward["verdict"] == AF.REFUTED
    assert reverse["reasons"] == forward["reasons"]
    assert reverse["measured"]["slope_cycles_per_unit"] == pytest.approx(
        forward["measured"]["slope_cycles_per_unit"])


# --------------------------------------------------------------------------------------
# KNOWN DEFECT (this test is expected to FAIL; see the report accompanying this file)
#
# The replicate check compares a TOTAL row count, never per-member coverage, and never the
# replicate identities the contract freezes.  Evidence that covers only two of the four
# cohort members -- with six invented replicate ids each -- therefore reaches ACCEPTED on a
# two-point fit.  That is the module's own stated failure mode inverted: absence of proof
# reported as proof.
# --------------------------------------------------------------------------------------

def test_evidence_that_covers_only_part_of_the_cohort_must_not_be_accepted(descriptors):
    rows = [
        {"capsule": descriptor["name"], "replicate": f"r{index:03d}",
         METRIC: 3 * _x(descriptor) + 200}
        for descriptor in (descriptors[0], descriptors[3])
        for index in range(6)
    ]
    assert len(rows) == 12  # the exact predeclared total, from half the cohort
    result = AF.analyze_affine_claim(descriptors, rows)
    assert result["verdict"] == AF.REFUSED, (
        "two of four cohort members carry no evidence and the replicate identities are "
        f"undeclared, yet the claim was {result['verdict']}")
