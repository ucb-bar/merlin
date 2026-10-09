"""Complete finite pointwise values, exact original storage and explicit choices."""

import copy
import math
from dataclasses import replace

import pytest

from merlin.targetgen.frontend_original_call import _literal
from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, prepare_original_reference
from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy, scalar_bound
from merlin.targetgen.original_pointwise_sources import FORM_SCHEMA, INTEGER_FORM_SCHEMA, pointwise_source
from merlin.targetgen.original_reference_values import TypedReferenceTensor as T


def policy(operation, dtype="float32"):
    floating = dtype == "float32"
    return OriginalPointwiseReferencePolicy(
        operation,
        (dtype,),
        (dtype,),
        dtype,
        "finite_f32" if floating else "bounded_exact",
        "not_applicable",
        "elementwise",
        "not_applicable",
        "rne" if floating else "exact_integer",
        True,
        False,
        "not_applicable",
        0.0,
        0.0,
        "preserve" if floating else "ignore",
    )


def form(operation, dtype="float32", *, rank=1, bounds=(None, None)):
    tensor = {
        "id": "input",
        "kind": "tensor",
        "dtype": dtype,
        "storage_dtype": dtype,
        "rank": rank,
        "layout": "torch.strided",
        "device": "cpu",
    }
    arguments = [
        {"name": "self", "type": "Tensor", "alias": None, "value": {"kind": "ssa", "node_id": "input", "value": tensor}}
    ]
    parameters = {}
    if operation == "aten.clamp.default":
        for name, value in zip(("min", "max"), bounds, strict=True):
            arguments.append({"name": name, "type": "Optional[number]", "alias": None, "value": _literal(value)})
            parameters[name] = value
    return {
        "form_schema": INTEGER_FORM_SCHEMA,
        "status": "supported",
        "target": operation,
        "arguments": arguments,
        "result_arity": 1,
        "schema_returns": [{"type": "Tensor", "alias": None}],
        "result_roster": [{**tensor, "id": "output"}],
        "rank": rank,
        "operand_dtypes": [dtype],
        "result_dtypes": [dtype],
        "parameters": parameters,
        "source_numerical_semantics": policy(operation, dtype).record(),
    }


def contract(operation, *, dtype="float32", count=10, rank=1, bounds=(None, None), selected=None, budget=None):
    original = form(operation, dtype, rank=rank, bounds=bounds)
    selected = policy(operation, dtype) if selected is None else selected
    original["source_numerical_semantics"] = selected.record()
    source = pointwise_source(original, extent=count, max_tensor_elements=10000)
    return prepare_original_reference(
        original,
        source,
        extent=count,
        policy=selected,
        budget=budget or OriginalReferenceBudget(10000, 100000, 100000, 20000),
        output_byteorder="little",
    )


def tensor(values, *, dtype="float32", name="X", shape=None):
    return T.from_values(name, dtype, [len(values)] if shape is None else shape, values, byteorder="little")


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("aten.relu.default", [0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 2.0**-149, 0.5, 1.5, 2.5]),
        ("aten.round.default", [-2.0, -2.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 2.0, 2.0]),
        ("aten.clamp.default", [-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 2.0**-149, 0.5, 1.0, 1.0]),
    ],
)
def test_complete_f32_readout_preserves_rne_ties_zero_sign_and_subnormals(operation, expected):
    selected = contract(operation, bounds=(-0.0, 1.0))
    inputs = (tensor([-2.5, -1.5, -0.5, -(2.0**-149), -0.0, 0.0, 2.0**-149, 0.5, 1.5, 2.5]),)
    wanted = tensor(expected, name="Y")
    actual = selected.evaluate(inputs)
    assert actual[0].data == wanted.data
    compared = selected.compare(inputs, (wanted,))
    assert compared["passed"] and compared["checked_elements"] == 10
    assert selected.verify()["scalar_products"] == 0


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ((None, 0.0), [-0.0, 0.0, -(2.0**-149), 0.0, -2.0, 0.0]),
        ((-0.0, None), [-0.0, 0.0, -0.0, 2.0**-149, -0.0, 2.0]),
        ((7, -3), [-3.0] * 6),
        ((1.0e-50, -0.0), [-0.0, 0.0, 0.0, -0.0, 0.0, -0.0]),
    ],
)
def test_original_optional_bounds_and_inverted_coerced_intervals(bounds, expected):
    selected = contract("aten.clamp.default", count=6, bounds=bounds)
    inputs = (tensor([-0.0, 0.0, -(2.0**-149), 2.0**-149, -2.0, 2.0]),)
    assert selected.source.metadata()["parameters"] == dict(zip(("min", "max"), bounds, strict=True))
    assert selected.evaluate(inputs)[0].data == tensor(expected, name="Y").data


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
@pytest.mark.parametrize("operation", ["aten.relu.default", "aten.round.default", "aten.clamp.default"])
def test_integer_pointwise_never_widens_or_wraps_original_readout(dtype, operation):
    values = [-7, -2, -1, 0, 1, 2, 7]
    expected = {
        "aten.relu.default": [0, 0, 0, 0, 1, 2, 7],
        "aten.round.default": values,
        "aten.clamp.default": [-2, -2, -1, 0, 1, 2, 3],
    }[operation]
    selected = contract(operation, dtype=dtype, count=len(values), bounds=(-2, 3))
    outputs = selected.evaluate((tensor(values, dtype=dtype),))
    assert outputs[0].dtype == dtype and outputs[0].data == tensor(expected, dtype=dtype, name="Y").data


def test_actual_pointwise_stress_uses_the_same_complete_reference_traversal():
    selected = contract("aten.round.default")
    inputs = (tensor([-2.5, -1.5, -0.5, -(2.0**-149), -0.0, 0.0, 2.0**-149, 0.5, 1.5, 2.5]),)
    trace = selected.observe_stress(inputs)
    assert trace["schema"] == "merlin.original_reference_stress.v2"
    assert trace["counts"]["pointwise_values"] == 10
    assert trace["counts"]["pointwise_rne_ties"] == 6
    assert trace["counts"]["negative_zero_inputs"] == 1 and trace["counts"]["negative_zero_outputs"] == 3
    assert trace["counts"]["subnormal_inputs"] == 2 and trace["counts"]["subnormal_outputs"] == 0
    assert trace["counts"]["products"] == trace["counts"]["additions"] == 0


def test_zero_sign_is_an_explicit_comparison_choice_and_last_element_is_checked():
    inputs = (tensor([-0.0, 1.5]),)
    observed = tensor([0.0, 2.0], name="Y")
    preserve = contract("aten.round.default", count=2)
    assert not preserve.compare(inputs, (observed,))["passed"]
    ignore = contract("aten.round.default", count=2, selected=replace(policy("aten.round.default"), zero_sign="ignore"))
    assert ignore.compare(inputs, (observed,))["passed"]
    changed = tensor([-0.0, 3.0], name="Y")
    verdict = preserve.compare(inputs, (changed,))
    assert not verdict["passed"] and verdict["mismatches"] == [
        {"slot": "Y", "index": 1, "expected": 2.0, "actual": 3.0}
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"arithmetic": "modular_wrap"},
        {"product_rounding": "accumulator_format"},
        {"reduction_cadence": "per_step"},
        {"subnormal_operand_flush": True},
        {"finite_only": False},
        {"atol": 0},
        {"readout_dtypes": ("float16",)},
    ],
)
def test_policy_refuses_unimplemented_or_implicit_pointwise_numerics(changes):
    with pytest.raises(ValueError):
        replace(policy("aten.round.default"), **changes).record()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float64"])
def test_unsupported_original_formats_do_not_become_f32(dtype):
    chosen = policy("aten.relu.default")
    chosen = replace(chosen, operand_dtypes=(dtype,), readout_dtypes=(dtype,), accumulator_dtype=dtype)
    with pytest.raises((ValueError, KeyError)):
        chosen.record()


@pytest.mark.parametrize("bounds", [(128, None), (None, -129), (0.5, 3)])
def test_integer_scalar_coercion_never_saturates_wraps_or_promotes(bounds):
    with pytest.raises(ValueError):
        contract("aten.clamp.default", dtype="int8", count=1, bounds=bounds)


def test_whole_pointwise_reference_work_precedes_typed_input_allocation():
    with pytest.raises(ValueError, match="before allocation"):
        contract("aten.relu.default", count=129, budget=OriginalReferenceBudget(10000, 100000, 1, 20000))


def test_original_scalar_rank_and_rectangular_rank_are_preserved():
    scalar = contract("aten.round.default", rank=0, count=129)
    assert scalar.verify()["inputs"][0]["shape"] == []
    assert scalar.evaluate((tensor([-0.5], shape=[]),))[0].data == tensor([-0.0], name="Y", shape=[]).data
    rectangular = contract("aten.relu.default", rank=2, count=3)
    assert rectangular.verify()["inputs"][0]["shape"] == [3, 4]


@pytest.mark.parametrize("dtype", ["int16", "int32", "int64"])
def test_new_integer_form_does_not_upgrade_legacy_factory_storage_scope(dtype):
    original = form("aten.relu.default", dtype)
    original["form_schema"] = FORM_SCHEMA
    with pytest.raises(ValueError):
        pointwise_source(original, extent=1, max_tensor_elements=100)
    original["form_schema"] = INTEGER_FORM_SCHEMA
    assert pointwise_source(original, extent=1, max_tensor_elements=100).metadata()["inputs"][0]["dtype"] == dtype


@pytest.mark.parametrize("operation", ["aten.relu.default", "aten.round.default", "aten.clamp.default"])
def test_signed64_extrema_and_scalar_bounds_never_pass_through_float(operation):
    values = [-(1 << 63), -(1 << 53) - 3, -1, 0, 1, (1 << 53) + 3, (1 << 63) - 1]
    expected = {
        "aten.relu.default": [0, 0, 0, 0, 1, (1 << 53) + 3, (1 << 63) - 1],
        "aten.round.default": values,
        "aten.clamp.default": [*values[:-1], (1 << 53) + 3],
    }[operation]
    selected = contract(operation, dtype="int64", count=len(values), bounds=(-(1 << 63), (1 << 53) + 3))
    actual = selected.evaluate((tensor(values, dtype="int64"),))
    assert actual[0].data == tensor(expected, dtype="int64", name="Y").data
    assert selected.compare((tensor(values, dtype="int64"),), actual)["checked_elements"] == len(values)


@pytest.mark.parametrize("power", [53, 60])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("odd", [False, True])
def test_original_integer_scalar_rounds_directly_to_f32_on_both_sides_of_ties(power, sign, odd):
    lower = 2.0**power + (2.0 ** (power - 23) if odd else 0.0)
    upper = lower + 2.0 ** (power - 23)
    midpoint = (1 << power) + (3 if odd else 1) * (1 << (power - 24))
    chosen = policy("aten.clamp.default")
    assert scalar_bound(sign * (midpoint - 1), chosen) == sign * lower
    assert scalar_bound(sign * midpoint, chosen) == sign * (upper if odd else lower)
    assert scalar_bound(sign * (midpoint + 1), chosen) == sign * upper


@pytest.mark.parametrize("bound,expected", [(-(1 << 63), -(2.0**63)), ((1 << 63) - 1, 2.0**63), (0, 0.0)])
def test_signed64_scalar_endpoints_and_zero_keep_the_original_f32_conversion(bound, expected):
    assert scalar_bound(bound, policy("aten.clamp.default")) == expected


@pytest.mark.parametrize("bound", [-(1 << 63) - 1, 1 << 63])
def test_original_scalar_overflow_is_refused_before_source_reference_allocation(bound):
    with pytest.raises(ValueError, match="overflow"):
        contract("aten.clamp.default", count=129, bounds=(bound, None))


@pytest.mark.parametrize("field", ["result_arity", "rank", "form_schema"])
def test_new_source_requires_exact_original_scalar_field_types(field):
    original = form("aten.relu.default")
    original[field] = [] if field == "form_schema" else True
    if field == "rank":
        original["result_roster"][0]["rank"] = True
    with pytest.raises(ValueError):
        pointwise_source(original, extent=1, max_tensor_elements=100)


def test_source_policy_and_argument_records_are_rederived_exactly():
    selected = contract("aten.clamp.default", bounds=(-0.0, 1.5))
    original = copy.deepcopy(selected.source.metadata())
    assert math.copysign(1, original["parameters"]["min"]) < 0
    damaged = replace(selected, source=replace(selected.source, loader=selected.source.loader.replace("-0.0", "0.0")))
    with pytest.raises(ValueError, match="binding changed"):
        damaged.verify()


@pytest.mark.parametrize(
    "field,bad",
    [("operation", []), ("arithmetic", 1), ("rounding", None), ("operand_dtypes", (True,)), ("zero_sign", False)],
)
def test_new_policy_refuses_scalar_type_aliases(field, bad):
    with pytest.raises(ValueError):
        replace(policy("aten.round.default"), **{field: bad}).verify()


def test_complete_clamp_coercion_refusal_precedes_typed_input_allocation():
    with pytest.raises(ValueError, match="overflow"):
        contract("aten.clamp.default", count=129, bounds=(None, 1.0e300))
    with pytest.raises(ValueError):
        contract("aten.clamp.default", count=129)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_finite_pointwise_policy_does_not_admit_nonfinite_runtime_storage(value):
    import struct

    selected = contract("aten.relu.default", count=1)
    original = T("X", "float32", (1,), struct.pack("<f", value), "little")
    with pytest.raises(ValueError, match="NaN|Inf|finite"):
        selected.evaluate((original,))
