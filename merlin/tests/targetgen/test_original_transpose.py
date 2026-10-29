"""Original transpose preserves axes, storage bits and separate alias premises."""

import copy
from dataclasses import replace

import pytest

from merlin.targetgen import original_transpose_sources as S
from merlin.targetgen.frontend_original_call import _literal
from merlin.targetgen.original_operator_reference import OriginalReferenceBudget, prepare_original_reference
from merlin.targetgen.original_reference_values import TypedReferenceTensor as T
from merlin.targetgen.original_transpose_reference import OriginalTransposeReferencePolicy


def form(dtype="float32", axes=(-2, -1)):
    original = {
        "id": "input",
        "kind": "tensor",
        "dtype": dtype,
        "storage_dtype": dtype,
        "rank": 2,
        "layout": "torch.strided",
        "device": "cpu",
    }
    alias = {"before": ["a"], "after": ["a"], "write": False}
    arguments = [
        {"name": "self", "type": "Tensor", "alias": copy.deepcopy(alias), "value": {"kind": "ssa", "value": original}},
        *[
            {"name": name, "type": "int", "alias": None, "value": _literal(axis)}
            for name, axis in zip(("dim0", "dim1"), axes, strict=True)
        ],
    ]
    normalized = [axis + 2 if axis < 0 else axis for axis in axes]
    permutation = [0, 1]
    if all(0 <= axis < 2 for axis in normalized):
        permutation[normalized[0]], permutation[normalized[1]] = permutation[normalized[1]], permutation[normalized[0]]
    return {
        "form_schema": S.FORM_SCHEMA,
        "status": "supported",
        "target": S.TARGET,
        "arguments": arguments,
        "result_arity": 1,
        "schema_returns": [{"type": "Tensor", "alias": copy.deepcopy(alias)}],
        "result_roster": [{**original, "id": "output"}],
        "rank": 2,
        "operand_dtypes": [dtype],
        "result_dtypes": [dtype],
        "parameters": dict(zip(("dim0", "dim1"), axes, strict=True)),
        "permutation": permutation,
        "source_numerical_semantics": {"unqualified": True},
    }


def policy(dtype):
    return OriginalTransposeReferencePolicy(
        S.TARGET, (dtype,), (dtype,), "rank_two_axis_swap", True, "exact_element_storage_bits"
    )


def contract(dtype="float32", axes=(-2, -1), byteorder="little"):
    original = form(dtype, axes)
    selected = policy(dtype)
    original["source_numerical_semantics"] = selected.record()
    source = S.transpose_source(original, extent=2, max_tensor_elements=100)
    return prepare_original_reference(
        original,
        source,
        extent=2,
        policy=selected,
        budget=OriginalReferenceBudget(100, 1000, 100, 10000),
        output_byteorder=byteorder,
    )


@pytest.mark.parametrize("dtype", ["float32", "int8", "int16", "int32", "int64"])
@pytest.mark.parametrize("axes", [(-2, -1), (1, 0), (0, 0), (-1, 1)])
@pytest.mark.parametrize("byteorder", ["little", "big"])
def test_complete_byte_permutation_preserves_storage_and_endian(dtype, axes, byteorder):
    selected = contract(dtype, axes, byteorder)
    values = [-0.0, 0.0, -(2.0**-149), 2.0**-149, -1.5, 7.25] if dtype == "float32" else [-7, 2, -1, 0, 1, 7]
    if dtype == "int64":
        values = [-(1 << 63), (1 << 63) - 1, -(1 << 53) - 3, (1 << 53) + 3, -1, 7]
    original = T.from_values("X", dtype, (2, 3), values, byteorder="big" if byteorder == "little" else "little")
    output = selected.evaluate((original,))[0]
    expected = values if form(dtype, axes)["permutation"] == [0, 1] else [values[index] for index in (0, 3, 1, 4, 2, 5)]
    exact = T.from_values("Y", dtype, output.shape, expected, byteorder=byteorder)
    assert output == exact
    result = selected.compare((original,), (exact,))
    assert result["passed"] and result["checked_elements"] == 6
    assert "alias/effect" in result["scope"]


def test_signed_zero_is_an_exact_bit_obligation_not_numeric_equality():
    selected = contract()
    original = T.from_values("X", "float32", (2, 3), [-0.0, 1, 2, 3, 4, 5], byteorder="little")
    expected = selected.evaluate((original,))[0]
    actual = replace(expected, data=b"\0\0\0\0" + expected.data[4:])
    comparison = selected.compare((original,), (actual,))
    assert not comparison["passed"] and comparison["checked_elements"] == 6
    assert comparison["mismatches"] == [{"slot": "Y", "index": 0, "expected_hex": "00000080", "actual_hex": "00000000"}]


@pytest.mark.parametrize(
    "defect",
    [
        "axis_bool",
        "axis_float",
        "axis_high",
        "axis_low",
        "rank",
        "storage",
        "write",
        "alias",
        "result_count",
        "result_alias",
        "parameter",
        "permutation",
    ],
)
def test_original_scalar_schema_alias_and_complete_result_binding_are_required(defect):
    original = form()
    if defect in {"axis_bool", "axis_float", "axis_high", "axis_low"}:
        value = {"axis_bool": True, "axis_float": 1.0, "axis_high": 2, "axis_low": -3}[defect]
        original["arguments"][1]["value"] = _literal(value)
    elif defect == "rank":
        original["result_roster"][0]["rank"] = True
    elif defect == "storage":
        original["result_roster"][0]["storage_dtype"] = "int32"
    elif defect == "write":
        original["arguments"][0]["alias"]["write"] = True
    elif defect == "alias":
        original["arguments"][0]["alias"]["after"] = ["b"]
    elif defect == "result_count":
        original["result_arity"] = True
    elif defect == "result_alias":
        original["schema_returns"][0]["alias"] = None
    elif defect == "parameter":
        original["parameters"]["dim0"] = 0
    else:
        original["permutation"] = [True, 0]
    with pytest.raises(ValueError):
        S.transpose_source(original, extent=2, max_tensor_elements=100)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float64", "uint8"])
def test_unsupported_original_storage_stays_unavailable(dtype):
    with pytest.raises(ValueError):
        S.transpose_source(form(dtype), extent=2, max_tensor_elements=100)
    with pytest.raises(ValueError):
        policy(dtype).verify()


def test_complete_budgets_precede_allocation_and_source_policy_is_frozen():
    original = form()
    for extent, maximum in ((2, 11), (10**9, 100), (True, 100)):
        with pytest.raises(ValueError):
            S.transpose_source(original, extent=extent, max_tensor_elements=maximum)
    source = S.transpose_source(original, extent=2, max_tensor_elements=12)
    assert source.metadata()["tensor_elements"] == 12
    assert source.metadata()["inputs"][0]["shape"] == [2, 3]
    assert source.metadata()["outputs"][0]["shape"] == [3, 2]
    assert "transpose.int(X, -2, -1)" in source.loader
    original["source_numerical_semantics"]["unqualified"] = False
    assert source.metadata()["source_numerical_semantics"] == {"unqualified": True}
    selected = contract()
    for budget in (
        OriginalReferenceBudget(11, 1000, 100, 10000),
        OriginalReferenceBudget(100, 47, 100, 10000),
        OriginalReferenceBudget(100, 1000, 5, 10000),
        OriginalReferenceBudget(100, 1000, 100, 2),
    ):
        with pytest.raises(ValueError):
            replace(selected, budget=budget).verify()


def test_finite_domain_and_full_ordered_output_roster_remain_required():
    selected = contract()
    original = T("X", "float32", (2, 3), bytes.fromhex("0000807f") * 6, "little")
    with pytest.raises(ValueError, match="NaN/Inf"):
        selected.evaluate((original,))
    original = T.from_values("X", "float32", (2, 3), range(6), byteorder="little")
    output = selected.evaluate((original,))[0]
    for actual in ((), (replace(output, name="another"),), (replace(output, shape=(2, 3)),)):
        with pytest.raises(ValueError):
            selected.compare((original,), actual)
