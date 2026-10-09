"""Declared software contracts; fixtures are not independent phase authorities."""

import json
import struct
from dataclasses import replace

import pytest

from merlin.targetgen import original_operator_sources as S
from merlin.targetgen.original_operator_reference import (
    OriginalReferenceBudget,
    OriginalReferencePolicy,
    prepare_original_reference,
)
from merlin.targetgen.original_reference_values import TypedReferenceTensor as T
from merlin.targetgen.original_reference_values import integer_project, round_f32

BUDGET = OriginalReferenceBudget(10000, 40000, 100000, 30000)


def policy(operation="aten.matmul.default", dtype="float32", *, arithmetic=None, accumulator=None, bias=False):
    floating = dtype == "float32"
    return OriginalReferencePolicy(
        operation,
        (dtype,) * (3 if bias else 2),
        (dtype,),
        accumulator or ("float32" if floating else "int32"),
        arithmetic or ("finite_f32" if floating else "bounded_exact"),
        "accumulator_format",
        {
            "aten.matmul.default": "contracting_axis_sequential",
            "aten.add.Tensor": "elementwise",
            "aten.conv2d.default": "input_channel_kernel_row_kernel_column",
        }[operation],
        "per_step",
        "rne" if floating else "exact_integer",
        True,
        False,
        "after_reduction",
        0.0,
        0.0,
        "preserve" if floating else "ignore",
    )


def _argument(name, dtype, rank, index, *, argument_type="Tensor"):
    return {
        "name": name,
        "type": argument_type,
        "alias": None,
        "value": {
            "kind": "ssa",
            "node_id": str(index),
            "value": {
                "id": str(index),
                "kind": "tensor",
                "rank": rank,
                "dtype": dtype,
                "storage_dtype": dtype,
                "layout": "torch.strided",
                "device": "cpu",
            },
        },
    }


def _literal(value):
    if value is None:
        return {"kind": "none"}
    if type(value) is list:
        return {"kind": "list", "items": [_literal(item) for item in value]}
    return {"kind": "int", "value": value}


def form(selected):
    """Minimal original typed form for unit checks, not a schema/origin grant."""
    conv = selected.operation == "aten.conv2d.default"
    rank, dtype = (4 if conv else 2), selected.readout_dtypes[0]
    arguments = [
        _argument("input" if conv else "self", dtype, rank, 0),
        _argument("weight" if conv else "other", dtype, rank, 1),
    ]
    parameters = {}
    add = selected.operation == "aten.add.Tensor"
    if add:
        arguments.append({"name": "alpha", "type": "Scalar", "alias": None, "value": _literal(1)})
        parameters = {"alpha": 1, "broadcasting": "none"}
    if conv:
        bias = len(selected.operand_dtypes) == 3
        arguments += [
            _argument("bias", dtype, 1, 2, argument_type="Optional[Tensor]")
            if bias
            else {"name": "bias", "type": "Optional[Tensor]", "alias": None, "value": _literal(None)}
        ]
        parameters = {"stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "groups": 2, "bias": bias}
        for name, kind in (
            ("stride", "List[int]"),
            ("padding", "List[int]"),
            ("dilation", "List[int]"),
            ("groups", "int"),
        ):
            arguments.append({"name": name, "type": kind, "alias": None, "value": _literal(parameters[name])})
    return {
        "form_schema": S.FORM_SCHEMA if conv else S.ADD_FORM_SCHEMA if add else S.MATMUL_FORM_SCHEMA,
        "target": selected.operation,
        "status": "supported",
        "arguments": arguments,
        "schema_returns": [{"type": "Tensor", "alias": None}],
        "result_arity": 1,
        "result_roster": [
            {
                "id": "Y",
                "kind": "tensor",
                "rank": rank,
                "dtype": dtype,
                "storage_dtype": dtype,
                "layout": "torch.strided",
                "device": "cpu",
            }
        ],
        "operand_dtypes": list(selected.operand_dtypes),
        "result_dtypes": list(selected.readout_dtypes),
        "parameters": parameters,
        "source_numerical_semantics": selected.record(),
    }


def contract(selected=None, *, extent=2, budget=BUDGET):
    selected = selected or policy()
    original = form(selected)
    factory = {
        "aten.conv2d.default": S.conv2d_source,
        "aten.add.Tensor": S.add_source,
        "aten.matmul.default": S.matmul_source,
    }[selected.operation]
    source = factory(original, extent=extent, max_tensor_elements=budget.max_tensor_elements)
    return prepare_original_reference(
        original, source, extent=extent, policy=selected, budget=budget, output_byteorder="little"
    )


def inputs_for(checked, rows):
    return tuple(
        T.from_values(row["name"], row["dtype"], row["shape"], values, byteorder="little")
        for row, values in zip(checked.verify()["inputs"], rows, strict=True)
    )


def test_complete_nonzero_original_matmul_and_changed_last_element():
    checked = contract()
    inputs = inputs_for(checked, ([2, -3, 5, 7, -11, 13], [17, 19, -23, 29, 31, -37, 41, 43, 47, -53, 59, -61]))
    expected = T.from_values("Y", "float32", (2, 4), [176, -116, 126, -376, 389, -149, 155, -1063], byteorder="big")
    assert checked.evaluate(inputs)[0].values() == expected.values()
    assert checked.compare(inputs, (expected,))["checked_elements"] == 8
    changed = replace(expected, data=expected.data[:-4] + struct.pack(">f", -1062))
    result = checked.compare(inputs, (changed,))
    assert not result["passed"] and result["mismatches"] == [
        {"slot": "Y", "index": 7, "expected": -1063.0, "actual": -1062.0}
    ]
    assert "phase admission unproved" in result["scope"]


def test_original_grouped_conv_preserves_bias_and_every_scalar_argument():
    checked = contract(policy("aten.conv2d.default", bias=True), extent=1)
    inputs = inputs_for(checked, ([-3, 2, 5, -7], [2, 4, -1, 3], [1, -2]))
    assert checked.evaluate(inputs)[0].values() == (3.0, -28.0)
    assert checked.verify()["parameters"] == {
        "stride": [1, 1],
        "padding": [0, 0],
        "dilation": [1, 1],
        "groups": 2,
        "bias": True,
    }


def test_signed_integer_readout_stays_i8_and_wrap_is_only_explicit():
    checked = contract(policy(dtype="int8", arithmetic="modular_wrap"), extent=1)
    inputs = inputs_for(checked, ([127, 127], [2, 1, -1, 2, 1, -1]))
    result = checked.evaluate(inputs)[0]
    assert result.dtype == "int8" and result.values() == (-4, -2, 2) and len(result.data) == 3
    exact = contract(policy(dtype="int8"), extent=1)
    with pytest.raises(ValueError, match="readout overflow"):
        exact.evaluate(inputs)


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_add_keeps_original_scalar_alpha_and_storage(dtype):
    checked = contract(policy("aten.add.Tensor", dtype), extent=2)
    inputs = inputs_for(checked, ([-3, 2, 5, -7, 11, 13], [17, -19, 23, 29, -31, 37]))
    assert checked.evaluate(inputs)[0].values() == (14, -17, 28, 22, -20, 50)
    assert checked.verify()["parameters"] == {"alpha": 1, "broadcasting": "none"}
    changed = json.loads(checked.form_json)
    changed["arguments"][2]["value"] = _literal(2)
    with pytest.raises(ValueError, match="unit alpha"):
        prepare_original_reference(
            changed, checked.source, extent=2, policy=checked.policy, budget=BUDGET, output_byteorder="little"
        )


def test_bounded_integer_checks_product_and_reduction_prefix_before_cancellation():
    product = contract(policy(dtype="int8", accumulator="int8"), extent=1)
    with pytest.raises(ValueError, match="intermediate/readout overflow"):
        product.evaluate(inputs_for(product, ([100, -100], [2, 0, 0, 2, 0, 0])))
    prefix = contract(policy(dtype="int8", accumulator="int8"), extent=2)
    with pytest.raises(ValueError, match="intermediate/readout overflow"):
        prefix.evaluate(inputs_for(prefix, ([100, 100, -100, 0, 0, 0], [1, 0, 0, 0] * 3)))


def test_float_order_rounds_products_and_each_prefix_not_final_exact_sum():
    checked = contract()
    inputs = inputs_for(checked, ([16777216, 1, -16777216, 0, 0, 0], [1, 0, 0, 0] * 3))
    assert checked.evaluate(inputs)[0].values() == (0.0,) * 8
    reordered = inputs_for(checked, ([16777216, -16777216, 1, 0, 0, 0], [1, 0, 0, 0] * 3))
    assert checked.evaluate(reordered)[0].values()[0] == 1.0
    final_exact = T.from_values("Y", "float32", (2, 4), [1, 0, 0, 0, 0, 0, 0, 0], byteorder="little")
    assert not checked.compare(inputs, (final_exact,))["passed"]
    with pytest.raises(ValueError, match="sequential order"):
        contract(replace(policy(), reduction_order="arbitrary_framework"))


def test_product_rounding_is_distinct_from_fused_multiply_add():
    checked = contract(extent=1)
    # (1+2^-23)*(1-2^-23) rounds to 1 before the next product cancels it.
    inputs = inputs_for(checked, ([1 + 2**-23, -1], [1 - 2**-23, 0, 0, 1, 0, 0]))
    assert checked.evaluate(inputs)[0].values()[0] == 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_finite_policy_refuses_nonfinite_inputs_and_outputs(value):
    checked = contract(extent=1)
    inputs = inputs_for(checked, ([1, 2], [1, 2, 3, 4, 5, 6]))
    bad_input = replace(inputs[0], data=struct.pack("<ff", value, 2))
    with pytest.raises(ValueError, match="NaN/Inf"):
        checked.evaluate((bad_input, inputs[1]))
    output = checked.evaluate(inputs)[0]
    bad_output = replace(output, data=struct.pack("<fff", 9, 12, value))
    with pytest.raises(ValueError, match="NaN/Inf"):
        checked.compare(inputs, (bad_output,))


def test_finite_policy_refuses_intermediate_overflow_even_when_later_cancelled():
    checked = contract(extent=1)
    inputs = inputs_for(checked, ([3e38, -3e38], [2, 0, 0, 2, 0, 0]))
    with pytest.raises(ValueError, match="overflow"):
        checked.evaluate(inputs)


def test_signed_zero_comparison_and_explicit_tolerances_are_preserved():
    checked = contract(extent=1)
    inputs = inputs_for(checked, ([0, 0], [0] * 6))
    negative = T.from_values("Y", "float32", (1, 3), [-0.0, 0.0, 0.0], byteorder="little")
    assert len(checked.compare(inputs, (negative,))["mismatches"]) == 1
    ignores = contract(replace(policy(), zero_sign="ignore"), extent=1)
    assert ignores.compare(inputs, (negative,))["passed"]
    tolerant = contract(replace(policy(), atol=0.125, rtol=0.0), extent=1)
    near = T.from_values("Y", "float32", (1, 3), [0.125, 0, 0], byteorder="little")
    assert tolerant.compare(inputs, (near,))["passed"]
    assert not checked.compare(inputs, (near,))["passed"]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float64"])
def test_unimplemented_original_formats_refuse_without_widening(dtype):
    with pytest.raises(ValueError, match="does not implement format"):
        contract(replace(policy(), operand_dtypes=(dtype, dtype), readout_dtypes=(dtype,), accumulator_dtype=dtype))


@pytest.mark.parametrize("change", ["global_policy", "readout", "bool_tolerance", "loader", "metadata", "scalar"])
def test_changed_original_source_or_policy_cannot_reuse_reference(change):
    selected = policy("aten.conv2d.default", bias=True) if change == "scalar" else policy()
    original = form(selected)
    factory = S.conv2d_source if change == "scalar" else S.matmul_source
    source = factory(original, extent=2, max_tensor_elements=10000)
    if change == "global_policy":
        original["source_numerical_semantics"] = {"operand_dtype": "int8", "readout_dtype": "int32"}
    elif change == "readout":
        selected = replace(selected, readout_dtypes=("int32",))
    elif change == "bool_tolerance":
        original["source_numerical_semantics"]["atol"] = False
    elif change == "loader":
        source = replace(source, loader=source.loader.replace("aten.matmul.default", "aten.add.Tensor"))
    elif change == "metadata":
        metadata = source.metadata()
        metadata["outputs"][0]["dtype"] = "int32"
        source = replace(source, metadata_json=json.dumps(metadata, sort_keys=True))
    else:
        original["parameters"]["stride"] = [2, 1]
    with pytest.raises(ValueError):
        prepare_original_reference(
            original, source, extent=2, policy=selected, budget=BUDGET, output_byteorder="little"
        )


@pytest.mark.parametrize(
    "field", ["max_tensor_elements", "max_payload_bytes", "max_arithmetic_steps", "max_source_bytes"]
)
def test_cost_limits_refuse_before_tensor_decode_or_allocation(field, monkeypatch):
    checked = contract()
    selected = replace(checked, budget=replace(BUDGET, **{field: 1}))
    monkeypatch.setattr(T, "values", lambda *_: pytest.fail("budget must be checked before any tensor decode"))
    with pytest.raises(ValueError, match="budget"):
        selected.evaluate(())


@pytest.mark.parametrize("change", ["missing", "extra", "renamed", "shape", "dtype", "partial"])
def test_complete_original_output_roster_cannot_be_sampled_or_substituted(change):
    checked = contract(extent=1)
    inputs = inputs_for(checked, ([1, 2], [1, 2, 3, 4, 5, 6]))
    output = checked.evaluate(inputs)[0]
    altered = {
        "renamed": replace(output, name="X"),
        "shape": replace(output, shape=(3, 1)),
        "dtype": replace(output, dtype="int32"),
        "partial": replace(output, data=output.data[:-4]),
    }
    actual = () if change == "missing" else (output, output) if change == "extra" else (altered[change],)
    with pytest.raises(ValueError, match="roster|storage"):
        checked.compare(inputs, actual)


def test_contract_freezes_caller_form_and_reopens_source_bytes():
    selected = policy()
    original = form(selected)
    source = S.matmul_source(original, extent=1, max_tensor_elements=10000)
    checked = prepare_original_reference(
        original, source, extent=1, policy=selected, budget=BUDGET, output_byteorder="little"
    )
    original["parameters"]["unexpected"] = 7
    assert checked.verify()["parameters"] == {}
    with pytest.raises(ValueError, match="binding changed"):
        replace(checked, source=replace(source, loader=source.loader + "\n# changed\n")).verify()
    with pytest.raises(ValueError, match="descriptor changed"):
        replace(checked, formats_json=b"{}").verify()


def test_integer_input_storage_never_implicitly_casts_floating_values():
    with pytest.raises(ValueError, match="implicit floating casts"):
        T.from_values("X", "int8", (1,), [1.5], byteorder="little")
    with pytest.raises(ValueError, match="overflow"):
        T.from_values("X", "int8", (1,), [128], byteorder="little")


def test_f32_codec_retains_round_to_even_subnormals_and_negative_zero():
    assert round_f32(1 + 2**-24) == 1
    assert round_f32(1 + 2**-23 + 2**-24) == 1 + 2**-22
    assert round_f32(-1 - 2**-24) == -1
    assert T.from_values("X", "float32", (2,), [2**-149, -(2**-150)], byteorder="little").data == (
        b"\x01\x00\x00\x00\x00\x00\x00\x80"
    )


def test_integer_projection_never_reinterprets_floating_formats_or_values():
    with pytest.raises(ValueError, match="exact integer"):
        integer_project(1.0, "int8", "modular_wrap")
    with pytest.raises(ValueError, match="signed integer format"):
        integer_project(1, "float32", "modular_wrap")


def test_missing_add_source_is_explicitly_unavailable_not_a_matmul_substitute(monkeypatch):
    # The separate source owner adds original add; this consumer has no fallback.
    monkeypatch.delattr(S, "add_source", raising=False)
    original = form(policy())
    selected = policy("aten.add.Tensor")
    original.update(target=selected.operation, source_numerical_semantics=selected.record())
    source = S.matmul_source(form(policy()), extent=1, max_tensor_elements=10000)
    with pytest.raises(ValueError, match="supported original source factory"):
        prepare_original_reference(
            original, source, extent=1, policy=selected, budget=BUDGET, output_byteorder="little"
        )
