"""Static storage/layout qualification and independent conversion negatives."""

import pytest
from mlir_fixtures import pointwise

from merlin.llvmlower.native_cpu_layout import joint_profile, profile
from merlin.targetgen.native_ieee_arithmetic_check import arithmetic_word
from merlin.targetgen.native_ieee_certificate import convert_word, verify_outputs
from merlin.targetgen.native_ieee_ir_check import conversion_function


@pytest.mark.parametrize(
    "shape,stride,row,prefix",
    [
        ([2, 3, 65], [234, 78, 1], 65, 64),
        ([2, 3, 65], [0, 65, 1], 195, 192),
        ([2, 3, 65], [390, 130, 2], 390, 0),
        ([1, 2, 3, 65], [390, 1, 130, 2], 390, 384),
    ],
)
def test_static_rows_keep_padding_broadcast_and_permuted_order(shape, stride, row, prefix):
    actual = profile(dict(shape=shape, stride=stride), block=32, grain=32768)
    assert actual["row"] == row
    assert actual["vector_prefix"] == prefix


@pytest.mark.parametrize(
    "layout", [dict(shape=[2, 3], stride=[1, 1]), dict(shape=[2, 3], stride=[3, -1]), dict(shape=[65536], stride=[1])]
)
def test_unknown_overlap_negative_stride_or_parallel_chunks_refuse(layout):
    assert profile(layout, block=32, grain=32768) is None


def test_joint_broadcast_scalar_loop():
    p = joint_profile(
        [dict(shape=[2, 1, 5], stride=[10, 10, 2]), dict(shape=[1, 3, 1], stride=[6, 2, 2])],
        [2, 3, 5],
        block=32,
        grain=32768,
    )
    assert p["row"] == 5
    assert p["vector_prefix"] == 0


@pytest.mark.parametrize("operation", ["add", "subtract"])
def test_wrong_nan_operand_priority_is_distinct(operation):
    a, b = 0x7E12, 0xFE34
    assert arithmetic_word(a, b, "f16", operation, "scalar") == a
    assert arithmetic_word(a, b, "f16", operation, "vector") == b


def test_binary32_widening_is_exact_and_quiets_nan_payloads():
    words = [0, 0x80000000, 1, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF]
    words += [0x7F800000, 0xFF800000, 0x7F800001, 0xFFBABCDE]
    actual = conversion_function("f32", "f64", "payload_quiet")(words)
    verify_outputs(words, actual, "f32", "f64", "payload_quiet")
    assert convert_word(1, "f32", "f64") == 0x36A0000000000000
    assert convert_word(0x80000000, "f32", "f64") == 1 << 63
    wrong = list(actual)
    wrong[-1] ^= 1 << 29
    with pytest.raises(ValueError, match="conversion proof refused"):
        verify_outputs(words, wrong, "f32", "f64", "payload_quiet")


@pytest.mark.parametrize("kind", ["nan_priority", "layout"])
def test_executable_checker_refuses_wrong_readout(kind):
    import os
    import subprocess

    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON names no CPU proof interpreter (torch is not in the test environment)")
    if kind == "nan_priority":
        script = """import torch
original = torch.add
torch.add = lambda a,b: original(b,a)
from merlin.targetgen.native_nan_priority_check import generate
try:
    generate()
except ValueError as e:
    assert 'NaN priority refused' in str(e)
else:
    raise AssertionError('wrong NaN priority was certified')
"""
    else:
        script = """import merlin.llvmlower.native_cpu_layout as layout
original = layout.profile
def wrong(*args, **kwargs):
    return dict(original(*args, **kwargs), vector_prefix=0)
layout.profile = wrong
from merlin.targetgen.native_layout_check import generate
try:
    generate()
except ValueError as e:
    assert 'CPU layout certificate refused' in str(e)
else:
    raise AssertionError('wrong vector/tail readout was certified')
"""
    result = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_proof_closes_a_folded_literal_nan_guard_without_mutating_source():
    from merlin.targetgen.native_typed_semantics import prove

    module = pointwise(
        ["tensor<4xf16>"],
        "tensor<4xf16>",
        """%v=arith.extf %a:f16 to f32
          %c=arith.constant 2.0:f32
          %p=arith.mulf %v,%c:f32
          %d=arith.constant -4194304:i32
          %default=arith.bitcast %d:i32 to f32
          %invalid=arith.cmpf uno,%p,%p:f32
          %base=arith.select %invalid,%default,%p:f32
          %bits=arith.bitcast %v:f32 to i32
          %mask=arith.constant 4194304:i32
          %qb=arith.ori %bits,%mask:i32
          %quiet=arith.bitcast %qb:i32 to f32
          %nan=arith.cmpf uno,%v,%v:f32
          %out=arith.select %nan,%quiet,%base:f32
          %false=arith.constant false
          %guard=arith.select %false,%c,%out:f32
          %result=arith.truncf %guard:f32 to f16""",
        yields="%result",
    )
    original = str(module)
    result = prove(module)
    assert result["status"] == "proven", result
    assert str(module) == original


def test_runtime_guard_is_not_folded_by_literal_closure():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.uniform_numeric_literals import fold_uniform_numeric_literals

    module = parse_mlir_text("""module {
      func.func @f(%condition:i1,%x:f32,%y:f32) -> f32 {
        %v=arith.select %condition,%x,%y:f32
        return %v:f32
      }}""")
    assert fold_uniform_numeric_literals(module, close_guards=True) == 0
    assert any(op.name == "arith.select" for op in module.walk())


def test_widening_does_not_license_binary64_arithmetic():
    from merlin.targetgen.native_typed_semantics import prove

    module = pointwise(
        ["tensor<4xf32>"],
        "tensor<4xf64>",
        "%v=arith.extf %a:f32 to f64\n%c=arith.constant 2.0:f64\n%result=arith.addf %v,%c:f64",
        yields="%result",
    )
    assert prove(module)["code"] == "native_binary64_arithmetic_form_unproved"


@pytest.mark.parametrize("storage,expected", [("f16", "proven"), ("f32", "refused")])
def test_new_guard_closure_stays_in_reduced_storage_family(storage, expected):
    from merlin.targetgen.native_typed_semantics import prove

    widen = "%a=arith.extf %lhs:f16 to f32\n%b=arith.extf %rhs:f16 to f32" if storage == "f16" else ""
    body = (
        widen
        + """
          %p=arith.addf %a,%b:f32
          %d=arith.constant -4194304:i32
          %default=arith.bitcast %d:i32 to f32
          %invalid=arith.cmpf uno,%p,%p:f32
          %base=arith.select %invalid,%default,%p:f32
          %mask=arith.constant 4194304:i32
          %br=arith.bitcast %b:f32 to i32
          %bq=arith.ori %br,%mask:i32
          %bquiet=arith.bitcast %bq:i32 to f32
          %bnan=arith.cmpf uno,%b,%b:f32
          %right=arith.select %bnan,%bquiet,%base:f32
          %false=arith.constant false
          %alias=arith.select %false,%a,%right:f32
          %ar=arith.bitcast %a:f32 to i32
          %aq=arith.ori %ar,%mask:i32
          %aquiet=arith.bitcast %aq:i32 to f32
          %anan=arith.cmpf uno,%a,%a:f32
          %out=arith.select %anan,%aquiet,%alias:f32"""
    )
    args = ("lhs", "rhs") if widen else "ab"
    result = prove(pointwise([f"tensor<4x{storage}>"] * 2, "tensor<4xf32>", body, yields="%out", args=args))
    assert result["status"] == expected, result
    if expected == "refused":
        assert result["code"] == "native_competing_nan_priority_unproved"
