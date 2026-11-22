"""Actual native min/max values retain the source NaN and zero policies."""

import ctypes
import math
import os
import struct

import pytest

from merlin.compiler.lower.model.codegen import build_host_shared
from merlin.compiler.lower.native.pipeline import lower_to_llvm_ir


@pytest.mark.skipif(not (os.environ.get("MERLIN_COMPILER_PYTHON") and os.environ.get("MERLIN_CLANG")),
                   reason="explicit native MLIR Python and host compiler required")
def test_signaling_nan_operands_and_zero_signs_survive_actual_native_minmax(tmp_path):
    source = '''module {
      func.func @numbers(%a:i16,%b:i16)->i64 attributes {llvm.emit_c_interface} {
        %x=arith.bitcast %a:i16 to f16
        %y=arith.bitcast %b:i16 to f16
        %max=arith.maxnumf %x,%y:f16
        %min=arith.minnumf %x,%y:f16
        %propmax=arith.maximumf %x,%y:f16
        %propmin=arith.minimumf %x,%y:f16
        %w0=arith.bitcast %max:f16 to i16
        %w1=arith.bitcast %min:f16 to i16
        %w2=arith.bitcast %propmax:f16 to i16
        %w3=arith.bitcast %propmin:f16 to i16
        %v0=arith.extui %w0:i16 to i64
        %v1=arith.extui %w1:i16 to i64
        %v2=arith.extui %w2:i16 to i64
        %v3=arith.extui %w3:i16 to i64
        %c16=arith.constant 16:i64
        %c32=arith.constant 32:i64
        %c48=arith.constant 48:i64
        %s1=arith.shli %v1,%c16:i64
        %s2=arith.shli %v2,%c32:i64
        %s3=arith.shli %v3,%c48:i64
        %r1=arith.ori %v0,%s1:i64
        %r2=arith.ori %r1,%s2:i64
        %r3=arith.ori %r2,%s3:i64
        return %r3:i64
      }}'''
    lowered = lower_to_llvm_ir(source, workdir=tmp_path / "lowering")
    path = tmp_path / "numbers.ll"
    path.write_text(lowered)
    image = build_host_shared(path, tmp_path / "numbers.so")
    function = ctypes.CDLL(str(image))._mlir_ciface_numbers
    function.argtypes, function.restype = [ctypes.c_uint16, ctypes.c_uint16], ctypes.c_uint64

    def value(word):
        return struct.unpack("<e", struct.pack("<H", word))[0]

    # Both NaN operand positions, both NaNs, the two zero orders and finite
    # extremes exercise one reusable compiled scalar function.
    for left, right in ((0x7C01, 0x3C00), (0xBC00, 0xFC01), (0x7E11, 0),
                        (0x8000, 0xFE22), (0x7C01, 0xFE22), (0, 0x8000),
                        (0x8000, 0), (0x7C00, 0xFC00), (1, 0x8001)):
        result = function(left, right)
        words = [(result >> shift) & 0xFFFF for shift in (0, 16, 32, 48)]
        a, b = value(left), value(right)
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(value(words[2])) and math.isnan(value(words[3]))
            if math.isnan(a) and math.isnan(b):
                assert all(math.isnan(value(word)) for word in words[:2])
            else:
                assert words[:2] == [right if math.isnan(a) else left] * 2
        elif a == b == 0:
            assert words[0] in (left, right) and words[1] in (left, right)
            assert words[2:] == [left & right, left | right]
        else:
            assert [value(word) for word in words] == [max(a, b), min(a, b)] * 2
