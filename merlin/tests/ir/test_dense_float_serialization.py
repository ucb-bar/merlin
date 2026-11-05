"""Compiler IR readers and writers retain actual dense floating-point bits."""

import pytest
from xdsl.dialects import arith, builtin
from xdsl.utils.exceptions import ParseError

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_text, parse_ir

FORMATS = [
    (builtin.f16, [0, 0x8000, 0x7C00, 0xFC00, 0x7E21, 0x7C01, 1, 0x7BFF]),
    (builtin.bf16, [0, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x7F81, 1, 0x7F7F]),
    (builtin.f32, [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F800001, 1, 0x7F7FFFFF]),
    (
        builtin.f64,
        [
            0,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF8000000001234,
            0x7FF0000000000001,
            1,
            0x7FEFFFFFFFFFFFFF,
        ],
    ),
]


def constant(module):
    return module.body.block.first_op.value


@pytest.mark.parametrize("reader", [parse_mlir_text, parse_ir])
@pytest.mark.parametrize("typ,bits", FORMATS)
@pytest.mark.parametrize("splat", [False, True])
def test_unquoted_dense_hex_is_bits_in_the_declared_float_type(reader, typ, bits, splat):
    width = typ.compile_time_size
    if splat:
        literal = f"0x{bits[3]:X}"
        expected = bits[3].to_bytes(width, "little") * 8
    else:
        # Mixed literal forms must retain their respective bit/numeric meanings.
        literal = "[" + ", ".join(f"0x{value:X}" for value in bits) + ", 1.0]"
        expected = b"".join(value.to_bytes(width, "little") for value in bits) + typ.pack([1.0])
    count = 8 if splat else len(bits) + 1
    source = f"builtin.module {{ %c = arith.constant dense<{literal}> : tensor<{count}x{typ}> }}"
    module = reader(source)
    module.verify()
    assert constant(module).data.data == expected


@pytest.mark.parametrize("reader", [parse_mlir_text, parse_ir])
@pytest.mark.parametrize("typ,bits", FORMATS)
@pytest.mark.parametrize("splat", [False, True])
def test_shared_pass_generic_roundtrip_preserves_nan_payloads_and_infinities(reader, typ, bits, splat):
    if splat:
        bits = [bits[5]] * 8
    raw = b"".join(value.to_bytes(typ.compile_time_size, "little") for value in bits)
    value = builtin.DenseIntOrFPElementsAttr(builtin.TensorType(typ, [len(bits)]), builtin.BytesAttr(raw))
    module = builtin.ModuleOp([arith.ConstantOp(value)])
    for _ in range(3):
        module = reader(ir_text(module))
        module.verify()
        assert constant(module).data.data == raw


@pytest.mark.parametrize("reader", [parse_mlir_text, parse_ir])
def test_numeric_integer_scalar_float_and_quoted_raw_dense_values_keep_their_meaning(reader):
    source = """builtin.module {
      %i = arith.constant dense<[0xff, -1, 2]> : tensor<3xi16>
      %f = arith.constant 0xff800000 : f32
      %r = arith.constant dense<"0x0100807F00000080"> : tensor<2xf32>
      %e = arith.constant dense<> : tensor<0xf32>
    }"""
    module = reader(source)
    module.verify()
    integer, scalar, raw, empty = module.body.block.ops
    assert tuple(integer.value.iter_values()) == (255, -1, 2)
    assert scalar.value.value.data == float("-inf")
    assert raw.value.data.data == bytes.fromhex("0100807F00000080")
    assert empty.value.data.data == b""


@pytest.mark.parametrize("reader", [parse_mlir_text, parse_ir])
@pytest.mark.parametrize("literal", [
    "0x100000000", "-0xff800000", "- 0xff800000", "- // sign separated from literal\n0xff800000", "- 0x0",
])
def test_out_of_range_or_signed_float_bit_patterns_refuse(reader, literal):
    with pytest.raises(ParseError):
        reader(f"builtin.module {{ %c = arith.constant dense<{literal}> : tensor<2xf32> }}")


@pytest.mark.parametrize("payload", [b"", bytes(range(256)) * 3,
    (b"source/function/block/operation_1234 " * 2048) + b'"\\\n\0' + bytes(range(256))])
def test_literal_span_printing_is_byte_identical_to_xdsl(payload):
    from io import StringIO
    from xdsl.printer import Printer
    from merlin.common.dense_float_ir import DenseFloatPrinter

    reference, actual = StringIO(), StringIO()
    Printer(stream=reference).print_bytes_literal(payload)
    DenseFloatPrinter(stream=actual).print_bytes_literal(payload)
    assert actual.getvalue() == reference.getvalue()


def test_unicode_provenance_keeps_original_utf8_byte_escaping():
    from io import StringIO
    from xdsl.printer import Printer
    from merlin.common.dense_float_ir import DenseFloatPrinter

    reference, actual = StringIO(), StringIO()
    label = 'source "quoted" \\ λ 雪 🧭\n'
    Printer(stream=reference).print_string_literal(label)
    DenseFloatPrinter(stream=actual).print_string_literal(label)
    assert actual.getvalue() == reference.getvalue()
