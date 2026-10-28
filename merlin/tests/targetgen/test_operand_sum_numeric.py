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


def _legacy_audit_i8_operand_sum(*, lhs_scale, rhs_scale, bound_lsb, relu, facet):
    """A frozen copy of ``audit_i8_operand_sum`` before it learned a readout gain."""
    import math

    import numpy as np

    from merlin.targetgen import operand_sum_numeric as module

    f32 = module._f32
    if not module._establishes_i8_model(facet, relu):
        return {"status": "unknown", "reason": "selected facts do not establish the two-load i8 scaling model"}
    if type(bound_lsb) is not int or not 0 <= bound_lsb <= 255:
        return {"status": "unknown", "reason": "a finite i8 output-error bound is required"}
    try:
        lhs, rhs = f32(float(lhs_scale)), f32(float(rhs_scale))
    except (OverflowError, TypeError, ValueError):
        return {"status": "unknown", "reason": "operand scales are not finite f32 values"}
    if not (math.isfinite(lhs) and math.isfinite(rhs) and lhs > 0 and rhs > 0):
        return {"status": "unknown", "reason": "operand scales must be positive finite f32 values"}
    factor = f32(max(1.0, lhs, rhs))
    load_lhs, load_rhs = f32(lhs / factor), f32(rhs / factor)
    domain = np.arange(-128, 128, dtype=np.int16)
    a = domain[:, None].astype(np.float32)
    b = domain[None, :].astype(np.float32)
    reference = np.rint(a * np.float32(lhs) + b * np.float32(rhs))
    reference = np.clip(reference, 0 if relu else -128, 127).astype(np.int16)
    la = np.clip(np.rint(a * np.float32(load_lhs)), -128, 127).astype(np.int16)
    lb = np.clip(np.rint(b * np.float32(load_rhs)), -128, 127).astype(np.int16)
    summed = la + lb
    if relu:
        summed = np.maximum(summed, 0)
    actual = np.rint(summed.astype(np.float32) * np.float32(factor))
    actual = np.clip(actual, -128, 127).astype(np.int16)
    params = {
        "lhs_scale_f32": lhs,
        "rhs_scale_f32": rhs,
        "lhs_load_f32": load_lhs,
        "rhs_load_f32": load_rhs,
        "readout_scale_f32": factor,
        "bound_lsb": bound_lsb,
        "relu": relu,
        "operand_dtype": "i8",
        "output_dtype": "i8",
    }
    return module._screen_result(reference, actual, params)


@pytest.mark.parametrize("relu", [False, True])
@pytest.mark.parametrize("bound", [0, 1, 255])
@pytest.mark.parametrize(
    "scales",
    [(1.0, 1.0), (0.5, 0.25), (1 / 6, 0.1), (1.0626662443588681, 0.31430679816210955), (10.0, 5.1), (2.0**-7, 1.0)],
)
def test_without_a_readout_scale_the_operand_sum_audit_is_the_legacy_one(scales, bound, relu):
    kwargs = dict(lhs_scale=scales[0], rhs_scale=scales[1], bound_lsb=bound, relu=relu, facet=_FACET)
    assert audit_i8_operand_sum(**kwargs) == _legacy_audit_i8_operand_sum(**kwargs)
    assert audit_i8_operand_sum(**kwargs, readout_scale=None) == _legacy_audit_i8_operand_sum(**kwargs)


def test_legacy_refusals_are_unchanged_without_a_readout_scale():
    for kwargs in (
        dict(lhs_scale=0.0, rhs_scale=1.0, bound_lsb=0, relu=False, facet=_FACET),
        dict(lhs_scale=float("nan"), rhs_scale=1.0, bound_lsb=0, relu=False, facet=_FACET),
        dict(lhs_scale=1e300, rhs_scale=1.0, bound_lsb=0, relu=False, facet=_FACET),
        dict(lhs_scale=1.0, rhs_scale=1.0, bound_lsb=256, relu=False, facet=_FACET),
        dict(lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=False, facet={}),
    ):
        assert audit_i8_operand_sum(**kwargs) == _legacy_audit_i8_operand_sum(**kwargs)


@pytest.mark.parametrize("gain", [0.0, -1.0, float("nan"), float("inf"), 1e300, 1e38, "x", True, [1.0]])
def test_an_invalid_readout_scale_is_unknown(gain):
    result = audit_i8_operand_sum(
        lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=False, facet=_FACET, readout_scale=gain
    )
    assert result["status"] == "unknown" and result["reason"]


def test_a_readout_scale_at_the_range_limit_is_screened_and_one_beyond_is_not():
    import numpy as np

    limit = float(np.finfo(np.float32).max) / 256
    result = audit_i8_operand_sum(
        lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=False, facet=_FACET, readout_scale=limit
    )
    assert result["status"] == "within_bound" and result["parameters"]["readout_gain_f32"] == limit
    assert (
        audit_i8_operand_sum(
            lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=False, facet=_FACET, readout_scale=limit * 2
        )["status"]
        == "unknown"
    )


def test_a_missing_model_fact_still_refuses_with_a_readout_scale():
    assert (
        audit_i8_operand_sum(lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=False, facet={}, readout_scale=0.5)[
            "status"
        ]
        == "unknown"
    )


def test_a_readout_scale_changes_the_witness_and_a_unit_scale_is_exact():
    plain = audit_i8_operand_sum(lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=True, facet=_FACET)
    unit = audit_i8_operand_sum(lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=True, facet=_FACET, readout_scale=1.0)
    sixth = audit_i8_operand_sum(
        lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, relu=True, facet=_FACET, readout_scale=1 / 6
    )
    assert plain["witness_sha256"] != unit["witness_sha256"] != sixth["witness_sha256"]
    assert unit["parameters"]["readout_gain_f32"] == 1.0 and unit["parameters"]["readout_register_f32"] == 1.0
    assert sixth["status"] == "within_bound" and sixth["max_error_lsb"] == 0
    assert "readout_gain_f32" not in plain["parameters"]


def _unit_single_rounding_cap():
    return dict(
        operation=dict(
            attributes=dict(
                N=16,
                quantization_scheme=dict(
                    weight_scale_granularity="tensor", operand_sum_rounding="single_rounding"
                ),
                case=dict(
                    exit="requant",
                    readout_stages=["activation", "convert", "scale", "requant"],
                    accumulator_stages=[],
                    operand_multipliers=[1.0, 1.0],
                    multipliers=[0.1666666716337204],
                ),
            )
        )
    )


def _unit_proof(gain_bits=0x3E2AAAAB, n=16):
    return dict(
        operand_sum_rounding="single_rounding",
        load_scale_bits=[0x3F800000, 0x3F800000],
        multiplier_bits=[gain_bits] * n,
        relu=True,
    )


@pytest.mark.parametrize("n", [16, 1])
def test_realization_screens_a_unit_single_rounding_readout_register(n):
    from merlin_experiments.phase0.fusion_execution import _screen_per_operand_case

    screens = _screen_per_operand_case(_unit_single_rounding_cap(), _unit_proof(n=n), _FACET)
    assert [s["column"] for s in screens] == [0]
    assert screens[0]["status"] == "within_bound" and screens[0]["max_error_lsb"] == 0
    assert screens[0]["parameters"]["device"]["readout_scale"] == 0.1666666716337204


@pytest.mark.parametrize(
    "change, match",
    [
        (dict(multiplier_bits=[0x3F800000] * 16), "exhaustive screen failed"),  # a readout register of 1.0
        (dict(load_scale_bits=[0x3F000000, 0x3F800000]), "load both operands at scale 1.0"),
        (dict(operand_sum_rounding="per_operand"), "does not cover this source tail"),
        (dict(multiplier_bits=[0x3E2AAAAB] * 5), "do not match source columns"),
        (dict(relu=False), "exhaustive screen failed"),
    ],
)
def test_realization_refuses_a_wrong_unit_single_rounding_register(change, match):
    from merlin_experiments.phase0.fusion_execution import _screen_per_operand_case

    proof = {**_unit_proof(), **change}
    with pytest.raises(ValueError, match=match):
        _screen_per_operand_case(_unit_single_rounding_cap(), proof, _FACET)


def test_realization_never_screens_a_non_unit_single_rounding_case():
    from merlin_experiments.phase0.fusion_execution import _screen_per_operand_case

    cap = _unit_single_rounding_cap()
    cap["operation"]["attributes"]["case"]["operand_multipliers"] = [0.5, 1.0]
    with pytest.raises(ValueError, match="does not cover this source tail"):
        _screen_per_operand_case(cap, _unit_proof(), _FACET)


def test_one_readout_register_cannot_cover_several_source_gains():
    from merlin_experiments.phase0.fusion_execution import _screen_per_operand_case

    cap = _unit_single_rounding_cap()
    cap["operation"]["attributes"]["case"]["multipliers"] = [0.1666666716337204, 0.5]
    with pytest.raises(ValueError, match="several source gains"):
        _screen_per_operand_case(cap, _unit_proof(n=1), _FACET)


def test_the_realization_selects_unit_single_rounding_members():
    from merlin_experiments.phase0.fusion_execution import _unit_single_rounding

    cap = _unit_single_rounding_cap()
    assert _unit_single_rounding(cap)
    cap["operation"]["attributes"]["case"]["operand_multipliers"] = [1.0, 0.5]
    assert not _unit_single_rounding(cap)
    cap = _unit_single_rounding_cap()
    cap["operation"]["attributes"]["quantization_scheme"]["operand_sum_rounding"] = "per_operand"
    assert not _unit_single_rounding(cap)


@pytest.mark.parametrize(
    "kernel",
    [
        dict(symbol="k", status="proven"),
        dict(symbol="k", status="proven", operand_sum=dict(status="proven", load_scale_bits=[0x3F800000] * 2)),
        dict(symbol="k", status="pending", operand_sum=dict(status="proven", **{
            "load_scale_bits": [0x3F800000] * 2, "multiplier_bits": [0x3E2AAAAB],
            "relu": True, "operand_sum_rounding": "single_rounding"})),
        dict(symbol="other", status="proven"),
    ],
    ids=["no_operand_sum", "missing_registers", "kernel_not_proven", "no_kernel"],
)
def test_an_incomplete_provider_operand_sum_proof_is_pending_audit(kernel):
    from merlin_experiments.phase0.fusion_execution import _operand_sum_proof

    with pytest.raises(ValueError, match="pending_audit"):
        _operand_sum_proof(dict(kernels=[kernel]), "k")
