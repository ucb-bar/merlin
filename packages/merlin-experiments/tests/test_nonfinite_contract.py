"""The versioned non-finite comparison contract (per_output.nonfinite.v1).

NaN/+-Inf expected outputs compare by class, but only for a capsule that declares the new schema
string. The default contract (per_output.v1) and every existing policy behave exactly as before.
"""

import copy
import math
import struct
from types import SimpleNamespace

import pytest
import yaml
from test_capsule_per_output_policy import written

from merlin.targetgen import capsule_golden as G
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import capsule_source as S
from merlin.targetgen.capture_refusals import M2MUnavailable
from merlin.targetgen.contract.schemas import ContractViolation, validate_capsule

NF = "merlin.numeric_policy.per_output.nonfinite.v1"
V1 = "merlin.numeric_policy.per_output.v1"
NAN, INF = math.nan, math.inf
BINDING = SimpleNamespace(atol=None, rtol=None)
ABI = [{"dtype": "f32", "shape": [4]}, {"dtype": "i32", "shape": [2]}]


def bits_to_float(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


NEG_QNAN = bits_to_float(0xFFF8000000000001)  # sign and payload differ from math.nan
SNAN = bits_to_float(0x7FF0000000000001)  # quiet bit clear: signaling


def policy(schema=NF, atol=0.0, rtol=0.0):
    return {
        "schema": schema,
        "compare": "per_output",
        "outputs": {
            "Y0": {"compare": "tolerance_float", "dtype": "f32", "atol": atol, "rtol": rtol},
            "Y1": {"compare": "exact_int", "dtype": "i32"},
        },
    }


def outputs(y0):
    return {"Y0": [y0], "Y1": [[1, 2]]}


def match(observed, expected, **kw):
    return G.match_per_output(observed, expected, policy(**kw))


# ---- class-wise comparison -------------------------------------------------------------------


def test_classes():
    assert G.nonfinite_class(1.5) is None and G.nonfinite_class(0) is None
    assert G.nonfinite_class(INF) == "+inf" and G.nonfinite_class(-INF) == "-inf"
    assert G.nonfinite_class(NAN) == G.nonfinite_class(NEG_QNAN) == "nan_quiet"
    assert G.nonfinite_class(SNAN) == "nan_signaling"


def test_identical_nonfinite_leaves_match():
    y = [NAN, INF, -INF, 1.0]
    assert match(outputs(y), outputs(list(y)))


def test_nan_matches_any_quiet_nan_ignoring_sign_and_payload():
    assert match(outputs([NEG_QNAN, 0.0]), outputs([NAN, 0.0]))


@pytest.mark.parametrize(
    "exp, obs",
    [
        (NAN, 1.0),
        (NAN, INF),
        (NAN, -INF),
        (NAN, SNAN),
        (SNAN, NAN),
        (INF, -INF),
        (INF, NAN),
        (INF, 1e38),
        (-INF, INF),
        (1.0, NAN),
        (1.0, INF),
        (0.0, -INF),
    ],
)
def test_class_mismatch_fails_without_raising(exp, obs):
    assert not match(outputs([exp]), outputs([obs]))
    rep = G.compare(outputs([exp]), outputs([obs]), policy())
    assert rep["status"] == "fail" and rep["mismatch_count"] == 1
    assert rep["first_mismatch"]["index"] == 0
    assert rep["per_output"]["Y0"]["mismatch_indices"] == [0]
    assert math.isfinite(rep["max_abs_error"]) and math.isfinite(rep["max_rel_error"])


def test_finite_leaves_keep_declared_tolerance_beside_nonfinite_leaves():
    expected = outputs([1.0, NAN, INF])
    assert match(outputs([1.01, NAN, INF]), expected, atol=0.02)
    assert not match(outputs([1.01, NAN, INF]), expected, atol=0.001)
    assert not match(outputs([1.0, NAN, 3.0]), expected, atol=100.0)  # tolerance never absorbs Inf


def test_exact_int_leaves_never_admit_nonfinite():
    for bad in (NAN, INF):
        assert not match({"Y0": [[1.0]], "Y1": [[bad]]}, {"Y0": [[1.0]], "Y1": [[1]]})
        assert not match({"Y0": [[1.0]], "Y1": [[1]]}, {"Y0": [[1.0]], "Y1": [[bad]]})


def test_report_counts_nonfinite_comparisons_only_under_the_contract():
    rep = G.compare(outputs([NAN, INF, 2.0]), outputs([NAN, INF, 2.0]), policy())
    assert rep["status"] == "pass" and rep["per_output"]["Y0"]["nonfinite_compared"] == 2
    finite = G.compare(outputs([1.0]), outputs([1.0]), policy(schema=V1))
    assert "nonfinite_compared" not in finite["per_output"]["Y0"]


def test_runner_dispatch_uses_the_contract():
    assert CR._match_by_policy(outputs([NAN, -INF]), outputs([NEG_QNAN, -INF]), policy())
    assert not CR._match_by_policy(outputs([NAN, -INF]), outputs([NEG_QNAN, INF]), policy())


# ---- default unchanged -----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [NAN, INF, -INF])
def test_v1_still_refuses_nonfinite_on_either_side(bad):
    v1 = policy(schema=V1)
    assert not G.match_per_output(outputs([bad]), outputs([bad]), v1)
    assert not G.match_per_output(outputs([1.0]), outputs([bad]), v1)
    assert not G.match_per_output(outputs([bad]), outputs([1.0]), v1)
    with pytest.raises(ValueError, match="finite"):
        G.compare(outputs([bad]), outputs([bad]), v1)


def test_v1_and_nf_agree_on_all_finite_data():
    expected, observed = outputs([1.0, 2.0]), outputs([1.0004, 2.5])
    for atol in (0.001, 1.0):
        assert match(observed, expected, atol=atol) == G.match_per_output(observed, expected, policy(V1, atol))
    a = G.compare(expected, observed, policy(V1, 0.001))
    b = G.compare(expected, observed, policy(NF, 0.001))
    b["per_output"]["Y0"].pop("nonfinite_compared")
    for key in ("status", "mismatch_count", "max_abs_error", "max_rel_error", "first_mismatch"):
        assert a[key] == b[key]


def test_flat_tolerance_policy_unchanged_for_nonfinite():
    flat = {"compare": "tolerance_float", "dtype": "f32", "atol": 0.1, "rtol": 0.0}
    assert G.compare({"Y": [1.0]}, {"Y": [1.0]}, flat)["status"] == "pass"
    # Pinned legacy behavior, deliberately not touched: a flat tolerance policy has no non-finite
    # contract and its `d > bound` test is False for NaN, so a NaN pair passes there. Only the
    # per-output nonfinite schema compares by class.
    assert G.compare({"Y": [NAN]}, {"Y": [NAN]}, flat)["status"] == "pass"
    assert G.compare({"Y": [NAN]}, {"Y": [NAN]}, flat) == G.compare({"Y": [NAN]}, {"Y": [1.0]}, flat)


def test_writer_default_and_v1_optin_still_refuse_nonfinite_goldens():
    for kw in ({}, {"output_names": ["Y0", "Y1"]}):
        with pytest.raises(M2MUnavailable):
            S._model_output_numeric_policy(ABI, [[NAN, 1.0, 1.0, 1.0], [1, 2]], BINDING, **kw)


# ---- mutation: the declaration is exact ------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema="merlin.numeric_policy.per_output.nonfinite.v2"),
        lambda p: p.update(schema=V1.replace("v1", "v2")),
        lambda p: p.update(extra=1),
        lambda p: p.pop("schema"),
        lambda p: p["outputs"].pop("Y1"),
        lambda p: p["outputs"].update(Z={"compare": "exact_int", "dtype": "i32"}),
        lambda p: p["outputs"]["Y0"].update(compare="exact_bits"),
        lambda p: p.update(compare="tolerance_float"),
    ],
)
def test_malformed_declaration_is_refused(mutate):
    p = policy()
    mutate(p)
    with pytest.raises(ValueError):
        G.match_per_output(outputs([1.0]), outputs([1.0]), p)


def test_observed_mutations_are_caught():
    expected_y0 = [NAN, INF, -INF, 1.0]
    expected = outputs(expected_y0)
    assert match(outputs(list(expected_y0)), expected)
    swaps = {NAN: INF, INF: -INF, -INF: NAN, 1.0: NAN}
    for i, value in enumerate(expected_y0):
        mutated = list(expected_y0)
        mutated[i] = swaps[value]
        assert not match(outputs(mutated), expected), i
    assert not match(outputs(expected_y0[:-1]), expected)  # length
    assert not match({"Y0": [expected_y0]}, expected)  # roster


def test_only_the_declared_schema_selects_the_contract():
    p = policy(schema=V1)
    assert not G.match_per_output(outputs([NAN]), outputs([NAN]), p)
    p["schema"] = NF
    assert G.match_per_output(outputs([NAN]), outputs([NAN]), p)


# ---- writer opt-in and round trip ------------------------------------------------------------


def test_writer_optin_publishes_the_nonfinite_policy():
    golden = [[NAN, INF, -INF, 1.0], [1, 2]]
    got = S._model_output_numeric_policy(
        ABI, golden, BINDING, output_names=["Y0", "Y1"], nonfinite_contract=True
    )
    assert got["schema"] == NF and got["compare"] == "per_output"
    assert got["outputs"]["Y1"] == {"compare": "exact_int", "dtype": "i32"}
    assert got["outputs"]["Y0"]["compare"] == "tolerance_float"


def test_optin_still_refuses_nonfinite_integer_results_and_incomplete_rosters():
    with pytest.raises(M2MUnavailable):
        S._model_output_numeric_policy(
            ABI, [[1.0] * 4, [1, NAN]], BINDING, output_names=["Y0", "Y1"], nonfinite_contract=True
        )
    with pytest.raises(M2MUnavailable, match="every unique"):
        S._model_output_numeric_policy(
            ABI, [[1.0] * 4, [1, 2]], BINDING, output_names=["Y0", "Y0"], nonfinite_contract=True
        )


def test_optin_single_dtype_capture_is_graded_under_the_contract():
    abi = [{"dtype": "f32", "shape": [2]}]
    got = S._model_output_numeric_policy(abi, [[NAN, 1.0]], BINDING, output_names=["Y0"], nonfinite_contract=True)
    assert got["schema"] == NF and set(got["outputs"]) == {"Y0"}


def test_yaml_round_trip_of_golden_and_policy_grades_a_matching_submission():
    golden = {"Y0": [[NAN, INF, -INF, 0.5]], "Y1": [[1, 2]]}
    pol = S._model_output_numeric_policy(
        ABI, [golden["Y0"][0], golden["Y1"][0]], BINDING, output_names=["Y0", "Y1"], nonfinite_contract=True
    )
    loaded_golden = yaml.safe_load(yaml.safe_dump(golden, sort_keys=False))
    loaded_policy = yaml.safe_load(yaml.safe_dump(pol, sort_keys=False))
    assert loaded_policy == pol
    assert math.isnan(loaded_golden["Y0"][0][0])
    assert G.match_per_output(copy.deepcopy(loaded_golden), loaded_golden, loaded_policy)
    wrong = copy.deepcopy(loaded_golden)
    wrong["Y0"][0][1] = -INF
    assert not G.match_per_output(wrong, loaded_golden, loaded_policy)


def test_ordinary_capsule_reader_accepts_the_declared_schema_only(tmp_path):
    directory, capsule = written(tmp_path)
    capsule["numeric_policy"]["schema"] = NF
    validate_capsule(capsule)
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    assert CR.load_capsule(directory)["numeric_policy"]["schema"] == NF
    capsule["numeric_policy"]["schema"] = NF + "x"
    with pytest.raises(ContractViolation):
        validate_capsule(capsule)
    capsule["numeric_policy"]["schema"] = NF
    capsule["operation"]["attributes"]["outs"] = ["Y0", "Y1"]  # roster check applies to the new schema too
    with pytest.raises(ContractViolation, match="roster"):
        validate_capsule(capsule)


def test_requirements_opt_in_exactly_the_four_coverage_labels():
    from merlin_experiments.phase0.requirements import NONFINITE_CONTRACT_LABELS

    assert NONFINITE_CONTRACT_LABELS == {
        "core_aten_coverage_affine",
        "core_aten_coverage_contraction",
        "core_aten_coverage_movement",
        "core_aten_coverage_reduce",
    }
