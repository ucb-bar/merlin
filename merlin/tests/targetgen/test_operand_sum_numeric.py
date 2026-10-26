import copy

import pytest

from merlin.targetgen.operand_sum_numeric import (
    PER_OPERAND_CONTRACT,
    audit_i8_operand_sum,
    audit_i8_per_operand_sum,
)

_FACET = {
    "rounding": "half_even",
    "clamp": [-128, 127],
    "accumulator_dtype": "i32",
    "operand_sum": {
        "operands": 2,
        "operand_dtype": "i8",
        "operand_rounding": "half_even",
        "operand_saturates": True,
        "scale_dtype": "f32",
    },
    "readouts": [{"selector": "i8", "applies": ["acc_scale", "relu"]}],
    "scale": {"dtype": "f32", "granularities": ["tensor"]},
}


def test_full_i8_domain_accepts_the_selected_gain_but_refuses_a_larger_gain():
    selected = audit_i8_operand_sum(
        lhs_scale=1.0626662443588681,
        rhs_scale=0.31430679816210955,
        bound_lsb=2,
        relu=True,
        facet=_FACET,
    )
    assert selected["status"] == "within_bound"
    assert selected["pairs_checked"] == 65536 and selected["max_error_lsb"] == 1
    assert selected["n_over_bound"] == 0 and len(selected["witness_sha256"]) == 64

    refused = audit_i8_operand_sum(lhs_scale=10, rhs_scale=5.1, bound_lsb=2, relu=True, facet=_FACET)
    assert refused["status"] == "exceeds_bound"
    assert refused["n_over_bound"] > 0 and refused["max_error_lsb"] == 5


def test_missing_numeric_fact_cannot_be_treated_as_success():
    assert audit_i8_operand_sum(lhs_scale=1, rhs_scale=1, bound_lsb=2, relu=True, facet={})["status"] == "unknown"


@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("scales,gain", [((0.5, 0.5), 1.0), ((1.0, 1.0), 0.5), ((0.37, 0.81), 1.7)])
def test_per_operand_exhaustive_screen_covers_ties_saturation_and_readout(relu, scales, gain):
    result = audit_i8_per_operand_sum(
        lhs_scale=scales[0], rhs_scale=scales[1], readout_scale=gain, relu=relu, facet=_FACET
    )
    assert result["status"] == "within_bound"
    assert result["pairs_checked"] == 65536
    assert result["max_error_lsb"] == result["n_over_bound"] == 0
    assert result["numerical_contract"] == PER_OPERAND_CONTRACT
    # Independently rounded halves differ from one RNE after addition.
    single = audit_i8_operand_sum(lhs_scale=0.5, rhs_scale=0.5, relu=relu, bound_lsb=0, facet=_FACET)
    assert single["status"] == "exceeds_bound"


@pytest.mark.parametrize("change", [dict(lhs_scale=1.0), dict(rhs_scale=1.0), dict(readout_scale=0.5), dict(relu=True)])
def test_per_operand_screen_refuses_changed_device_parameters(change):
    device = dict(lhs_scale=0.5, rhs_scale=0.5, readout_scale=1.0, relu=False)
    device.update(change)
    result = audit_i8_per_operand_sum(lhs_scale=0.5, rhs_scale=0.5, relu=False, facet=_FACET, device=device)
    assert result["status"] == "exceeds_bound" and result["n_over_bound"] > 0
    assert result["first_over_bound"] is not None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(lhs_scale=0),
        dict(rhs_scale=1.5),
        dict(lhs_scale=float("nan")),
        dict(readout_scale=float("inf")),
        dict(bound_lsb=1),
        dict(facet={}),
        dict(relu=1),
        dict(device={}),
        dict(facet=dict(_FACET, rounding="towards_zero")),
    ],
)
def test_per_operand_screen_fails_closed(overrides):
    parameters = dict(lhs_scale=0.5, rhs_scale=0.5, relu=False, facet=_FACET)
    parameters.update(overrides)
    assert audit_i8_per_operand_sum(**parameters)["status"] == "unknown"


def test_admission_never_uses_a_witness_for_the_other_rounding_contract(monkeypatch):
    from merlin.targetgen.operator_review import ACCEPT_ENV, PENDING
    from merlin.targetgen.software_spec import admit_operation

    witness = audit_i8_per_operand_sum(lhs_scale=0.5, rhs_scale=0.5, relu=False, facet=_FACET)
    witness["form"] = "fused_operand_sum"
    row = dict(
        id="per_operand",
        status=PENDING,
        numerical_contract=PER_OPERAND_CONTRACT,
        families=["elementwise_map"],
        placement="fused_accelerator",
        signature=dict(dtypes=["int8"], composed_with=["residual_add"]),
    )
    spec = dict(status="reviewed", operations=[row])
    signature = dict(
        family="elementwise_map", operand_dtype="i8", composed_with=["residual_add"], numeric_screen=witness
    )
    monkeypatch.setenv(ACCEPT_ENV, "0")
    assert admit_operation(spec, "relu", signature, "fused_accelerator")["status"] == "unknown"
    monkeypatch.setenv(ACCEPT_ENV, "1")
    decision = admit_operation(spec, "relu", signature, "fused_accelerator")
    assert decision["status"] == "admitted" and decision["review_status"] == PENDING
    for contract in ["operand_sum_exhaustive_i8_v1", None]:
        wrong = copy.deepcopy(signature)
        if contract:
            wrong["numeric_screen"]["numerical_contract"] = contract
        else:
            wrong["numeric_screen"].pop("numerical_contract")
        assert admit_operation(spec, "relu", wrong, "fused_accelerator")["status"] == "unknown"
    row["numerical_contract"] = "operand_sum_exhaustive_i8_v1"
    assert admit_operation(spec, "relu", signature, "fused_accelerator")["status"] == "unknown"


def test_retained_trace_screen_checks_every_distinct_source_and_device_gain():
    from merlin_experiments.phase0.fusion_execution import _screen_per_operand_case

    cap = dict(
        operation=dict(
            attributes=dict(
                N=17,
                quantization_scheme=dict(weight_scale_granularity="column_block", columns_per_scale=16),
                case=dict(
                    exit="requant",
                    readout_stages=["convert", "scale", "requant"],
                    accumulator_stages=[],
                    operand_multipliers=[0.5, 0.5],
                    multipliers=[1.0, 0.5],
                ),
            )
        )
    )
    proof = dict(
        operand_sum_rounding="per_operand",
        load_scale_bits=[0x3F000000, 0x3F000000],
        multiplier_bits=[0x3F800000] * 16 + [0x3F000000],
        relu=False,
    )
    screens = _screen_per_operand_case(cap, proof, _FACET)
    assert [s["column"] for s in screens] == [0, 16]
    assert all(s["pairs_checked"] == 65536 and s["status"] == "within_bound" for s in screens)
    proof["multiplier_bits"][-1] = 0x3F800000
    with pytest.raises(ValueError, match="exhaustive screen failed"):
        _screen_per_operand_case(cap, proof, _FACET)
    cap["operation"]["attributes"]["case"]["readout_stages"].append("pool")
    with pytest.raises(ValueError, match="does not cover this source tail"):
        _screen_per_operand_case(cap, proof, _FACET)
