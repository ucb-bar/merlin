"""A pure FP8 tensor carrier coexists with exact scalar widening."""

import ctypes
import math
import os
import struct

import pytest

from merlin.compiler.lower.model.codegen import build_host_shared
from merlin.compiler.lower.native.pipeline import lower_to_llvm_ir


@pytest.mark.skipif(not (os.environ.get("MERLIN_COMPILER_PYTHON") and os.environ.get("MERLIN_CLANG")),
                   reason="explicit native MLIR Python and host compiler required")
def test_dynamic_fp8_tensor_element_keeps_widening_and_special_values(tmp_path):
    source = '''module {
      func.func @carrier(%first:i64,%second:i64,%position:i64)->i64 attributes {llvm.emit_c_interface} {
        %a=arith.trunci %first:i64 to i8
        %b=arith.trunci %second:i64 to i8
        %x=arith.bitcast %a:i8 to f8E4M3FNUZ
        %y=arith.bitcast %b:i8 to f8E4M3FNUZ
        %values=tensor.from_elements %x,%y:tensor<2xf8E4M3FNUZ>
        %two=arith.constant 2:i64
        %valid=arith.cmpi ult,%position,%two:i64
        cf.assert %valid,"carrier index outside source domain"
        %index=arith.index_cast %position:i64 to index
        %selected=tensor.extract %values[%index]:tensor<2xf8E4M3FNUZ>
        %wide=arith.extf %selected:f8E4M3FNUZ to f64
        %word=arith.bitcast %wide:f64 to i64
        return %word:i64
      }}'''
    lowered = lower_to_llvm_ir(source, workdir=tmp_path / "lowering",
        features=frozenset({"lower_fp8_widening"}))
    path = tmp_path / "carrier.ll"
    path.write_text(lowered)
    library = ctypes.CDLL(str(build_host_shared(path, tmp_path / "carrier.so")))
    function = library._mlir_ciface_carrier
    function.argtypes, function.restype = [ctypes.c_uint64] * 3, ctypes.c_uint64
    # FNUZ zero/unique NaN, minimum subnormal, maximum negative finite and
    # ordinary signed fractions. Two distinct values keep extraction dynamic.
    for first, second, expected in ((0, 128, (0.0, math.nan)),
                                     (1, 255, (2.0**-10, -240.0)),
                                     (56, 189, (0.5, -0.8125))):
        for position, value in enumerate(expected):
            observed = function(first, second, position)
            if math.isnan(value):
                assert math.isnan(struct.unpack("d", struct.pack("Q", observed))[0])
            else:
                assert observed == struct.unpack("Q", struct.pack("d", value))[0]
