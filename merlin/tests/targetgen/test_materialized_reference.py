"""Every result must have its original complete typed independently checked reference."""
import copy
import json

import pytest

from merlin.targetgen.materialized_reference import complete_float_outputs, complete_outputs, reference_members


def _source():
    meta = {"output_abi": [{"shape": [1, 2], "dtype": "f32"}, {"shape": [1], "dtype": "f32"}]}
    reference = {"schema": "merlin.capture.integer_reference.v1", "output_abi": meta["output_abi"],
                 "outputs": [[[1.25, -2.5]], [3.0]]}
    return meta, reference


def test_complete_outputs_and_expected_file_roles():
    meta, ref = _source()
    assert complete_float_outputs(meta, json.dumps(ref)) == ref["outputs"]
    assert reference_members({"golden.npy": {}}) == ("golden.npy",)
    assert reference_members({"golden.npy": {}, "integer-reference.json": {}}) == (
        "golden.npy", "integer-reference.json")


@pytest.mark.parametrize("change", ["missing", "shape", "dtype", "abi", "nonfinite", "lossy", "boolean",
                                    "mixed_boolean"])
def test_incomplete_or_converted_reference_refuses(change):
    meta, ref = _source()
    ref = copy.deepcopy(ref)
    if change == "missing":
        ref["outputs"].pop()
    if change == "shape":
        ref["outputs"][1] = [[3.0]]
    if change == "dtype":
        ref["output_abi"][1]["dtype"] = "i32"
    if change == "abi":
        ref["output_abi"].reverse()
    if change == "nonfinite":
        ref["outputs"][1] = [float("inf")]
    if change == "lossy":
        ref["outputs"][1] = [1.0000000000000002]
    if change == "boolean":
        ref["outputs"][1] = [True]
    if change == "mixed_boolean":
        ref["outputs"][0] = [[1.25, True]]
    with pytest.raises(ValueError):
        complete_float_outputs(meta, json.dumps(ref))


def test_complete_mixed_outputs_preserve_exact_boolean_and_large_integer():
    abi = [{"shape": [2], "dtype": "i1"}, {"shape": [2], "dtype": "i64"},
           {"shape": [1], "dtype": "f32"}]
    ref = {"schema": "merlin.capture.integer_reference.v1", "output_abi": abi,
           "outputs": [[True, False], [2**63 - 1, -(2**63)], [1.25]]}
    assert complete_outputs({"output_abi": abi}, json.dumps(ref)) == ref["outputs"]
    with pytest.raises(ValueError):
        complete_float_outputs({"output_abi": abi}, json.dumps(ref))


@pytest.mark.parametrize(("dtype", "value"), [("i1", 1), ("i1", 0.0), ("i8", True),
                                            ("i8", 1.0), ("i8", 128), ("i8", -129),
                                            ("i64", 2**63), ("i64", -(2**63)-1)])
def test_complete_integer_outputs_refuse_conversion_and_overflow(dtype, value):
    abi = [{"shape": [1], "dtype": dtype}]
    ref = {"schema": "merlin.capture.integer_reference.v1", "output_abi": abi, "outputs": [[value]]}
    with pytest.raises(ValueError):
        complete_outputs({"output_abi": abi}, json.dumps(ref))
