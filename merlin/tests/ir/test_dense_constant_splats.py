"""Splat detection must preserve stored words before ordinary MLIR printing."""

from io import StringIO

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text

FORMATS = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "f8E4M3FN": 1, "f8E5M2": 1, "f8E4M3FNUZ": 1, "f8E5M2FNUZ": 1}


def constant(dtype, raw):
    return f'module {{func.func @f()->tensor<2x{dtype}> {{%v=arith.constant dense<"0x{raw.hex()}">:tensor<2x{dtype}> return %v:tensor<2x{dtype}>}}}}'


@pytest.mark.parametrize("dtype,size", FORMATS.items())
@pytest.mark.parametrize("reverse", [False, True])
def test_distinct_words_never_become_a_splat(dtype, size, reverse):
    from xdsl.printer import Printer

    raw = [bytes(size), (1 << (8 * size - 1)).to_bytes(size, "little")]
    raw = b"".join(reversed(raw) if reverse else raw)
    module = parse_mlir_text(constant(dtype, raw))
    value = next(op.value for op in module.walk() if op.name == "arith.constant")
    assert not value.is_splat()
    output = StringIO()
    Printer(stream=output, print_generic_format=True).print_op(module)
    restored = parse_mlir_text(output.getvalue())
    assert next(op.value.data.data for op in restored.walk() if op.name == "arith.constant") == raw


@pytest.mark.parametrize("dtype,size", FORMATS.items())
def test_true_zero_splats_remain_splats(dtype, size):
    if size == 1:
        pytest.importorskip("ml_dtypes")
    module = parse_mlir_text(constant(dtype, bytes(size) * 2))
    value = next(op.value for op in module.walk() if op.name == "arith.constant")
    assert value.is_splat()


@pytest.mark.parametrize("width", [1, 8, 16, 32, 64])
def test_integer_splats_are_unchanged(width):
    from xdsl.dialects import builtin

    dtype = builtin.IntegerType(width)
    shape = builtin.TensorType(dtype, [2])
    assert builtin.DenseIntOrFPElementsAttr.from_list(shape, [1, 1]).is_splat()
    assert not builtin.DenseIntOrFPElementsAttr.from_list(shape, [0, 1]).is_splat()
