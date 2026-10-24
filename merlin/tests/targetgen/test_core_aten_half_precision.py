"""The half ABI preserves storage, mixed result types, and the suite's policy."""

import copy

import numpy as np
import pytest
from targetgen_fixtures import capture

from merlin.llvmlower.c_runtime import _bf16_bits, generate
from merlin.targetgen.core_aten_batch import build_core_aten_batch
from merlin.targetgen.core_aten_batch_grade import _grade_output
from merlin.targetgen.core_aten_semantics import validate_full


@pytest.mark.parametrize("dtype,logical", [("bf16", "bfloat16"), ("f16", "float16")])
def test_half_bundle_packs_real_16_bit_storage_and_keeps_f32_result(tmp_path, dtype, logical):
    captures = tmp_path / "captures"
    half = dict(dtype=dtype, shape=[4])
    full = dict(dtype="f32", shape=[4])
    layouts = [
        dict(dtype=dt, shape=[4], stride=[1], storage_offset=0, requires_grad=False) for dt in (logical, "float32")
    ]
    source = (
        f"builtin.module {{ func.func @forward(%a: tensor<4x{dtype}>, %b: tensor<4xf32>) "
        f"-> (tensor<4x{dtype}>, tensor<4xf32>) {{\n"
        f"func.return %a, %b : tensor<4x{dtype}>, tensor<4xf32>\n}} }}"
    )
    contract = dict(
        authority="fixture", inputs=layouts, results=[{**layout, "role": "user_output"} for layout in layouts]
    )
    values = [-0.0, 1.5, 2**-24, 2**-14]
    abi = [half, full]
    capture(captures / "fixture", source, [values, values], [values, values], abi, abi, result_contract=contract)
    arguments = dict(
        args=dict(kind="tuple", items=[dict(kind="tensor", **layout, values=values) for layout in layouts]), kwargs={}
    )
    case = dict(overload="fixture", arguments=arguments, expected=[])
    bundle = tmp_path / "bundle"
    report = build_core_aten_batch(dict(cases=[case]), captures, bundle)
    record = report["cases"][0]
    assert record["status"] == "bundled"
    assert record["output_abi"] == [half, full]
    generated = tmp_path / "generated"
    generate(bundle, generated, bundle / "inputs.npz")
    words = (
        _bf16_bits(np.array(values, np.float32)) if dtype == "bf16" else np.array(values, np.float16).view(np.uint16)
    )
    header = (generated / "model_io.h").read_text()
    for word in words:
        assert str(int(word)) in header
    assert "unsigned short" in header


def test_bfloat_grading_uses_float_values_and_existing_tolerances():
    values = np.array([1.0, -0.0, 2**-133, np.inf, np.nan], np.float32)
    expected = dict(kind="tensor", dtype="bfloat16", shape=[5], values=values.tolist())
    raw = _bf16_bits(values).tobytes()
    case = dict(comparison_parameters=dict(rtol=0, atol=0, equal_nan=True))
    assert _grade_output(raw, dict(dtype="bf16", shape=[5]), expected, case)["status"] == "pass"
    expected["values"][0] = 1.01
    assert _grade_output(raw, dict(dtype="bf16", shape=[5]), expected, case)["status"] == "mismatch"
    case["comparison_parameters"]["atol"] = 0.02
    assert _grade_output(raw, dict(dtype="bf16", shape=[5]), expected, case)["status"] == "pass"


def test_bfloat_full_boundary_decodes_post_values_and_mutation():
    values = [1.0, 2.0]
    tensor = dict(
        kind="tensor", dtype="bfloat16", shape=[2], stride=[1], storage_offset=0, requires_grad=False, values=values
    )
    arguments = dict(args=dict(kind="tuple", items=[tensor]), kwargs={})
    abi = dict(dtype="bf16", shape=[2])
    record = dict(
        input_indices=[0],
        output_indices=[0],
        output_abi=[abi],
        semantic_boundary=dict(post_indices=[1], post_abi=[abi]),
        case=dict(
            arguments=arguments,
            post_arguments=copy.deepcopy(arguments),
            expected=tensor,
            comparison="torch_close",
            comparison_parameters=dict(rtol=0, atol=0),
            mutated_arguments=[],
            output_input_aliases=["result->args[0]"],
        ),
    )
    raw = [_bf16_bits(np.array(values, np.float32)).tobytes()] * 2
    layout = dict(shape=[2], stride=[1], storage_offset=0, requires_grad=False)
    readback = dict(metadata=[layout, layout], aliases=[[1], [1]], pre_bytes={"0": raw[1].hex()})
    result = validate_full(record, raw, readback)
    assert result["status"] == "pass"
    assert result["observation"]["output"]["dtype"] == "bfloat16"
    assert result["observation"]["post_arguments"]["args"]["items"][0]["values"] == values
