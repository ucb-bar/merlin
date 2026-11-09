"""Actual stock mixed integer/FP8 roundtrips retain all cast semantics."""

import ctypes
import json
import subprocess

import pytest
from xdsl.dialects.arith import IntegerOverflowAttr, TruncIOp
from xdsl.dialects.builtin import StringAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import toolchain
from merlin.llvmlower.deployable_ir import checked_generic_text
from merlin.llvmlower.fp8_encode_fusion import fuse_clamped_encoders
from merlin.llvmlower.stock_arith_compat import unflagged_truncations

MIXED = """module {
func.func @forward(%i:tensor<4xi32>,%x:tensor<4xf32>)->(tensor<4xi8>,tensor<4xf8E4M3FN>){
 %be=tensor.empty():tensor<4xi8>
 %byte=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
 ins(%i:tensor<4xi32>) outs(%be:tensor<4xi8>){
 ^bb0(%v:i32,%old:i8):
  %b=arith.trunci %v:i32 to i8
  linalg.yield %b:i8
 }->tensor<4xi8>
 %e=tensor.empty():tensor<4xf32>
 %bounded=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
 ins(%x:tensor<4xf32>) outs(%e:tensor<4xf32>){
 ^bb0(%v:f32,%old:f32):
  %lo=arith.constant -448.0:f32
  %below=arith.cmpf olt,%v,%lo:f32
  %lower=arith.select %below,%lo,%v:f32
  %hi=arith.constant 448.0:f32
  %above=arith.cmpf ogt,%lower,%hi:f32
  %clamped=arith.select %above,%hi,%lower:f32
  linalg.yield %clamped:f32
 }->tensor<4xf32>
 %out=tensor.empty():tensor<4xf8E4M3FN>
 %encoded=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
 ins(%bounded:tensor<4xf32>) outs(%out:tensor<4xf8E4M3FN>){
 ^bb0(%v:f32,%old:f8E4M3FN):
  %f=arith.truncf %v:f32 to f8E4M3FN
  linalg.yield %f:f8E4M3FN
 }->tensor<4xf8E4M3FN>
 return %byte,%encoded:tensor<4xi8>,tensor<4xf8E4M3FN>
}}
"""


def stock(tmp_path, text):
    opt = toolchain.llvm_install() / "bin/mlir-opt"
    if not opt.is_file():
        pytest.skip("selected actual stock MLIR required")
    source, output = tmp_path / "source.mlir", tmp_path / "stock.mlir"
    source.write_text(text)
    subprocess.run(
        [str(opt), str(source), "--mlir-print-op-generic", "-o", str(output)], check=True, capture_output=True
    )
    return output


def test_actual_stock_default_none_retains_cast_objects_types_and_attributes(tmp_path):
    source = stock(tmp_path, "module {func.func @f(%x:i32)->i8{%r=arith.trunci %x:i32 to i8 return %r:i8}}")
    module = parse_mlir_text(source.read_text())
    op = next(op for op in module.walk() if isinstance(op, TruncIOp))
    assert op.properties["overflowFlags"] == IntegerOverflowAttr("none")
    op.attributes["custom.contract"] = StringAttr("preserved")
    operands, results, attrs = tuple(op.operands), tuple(op.results), dict(op.attributes)
    rows = unflagged_truncations(module)
    module.verify()
    assert rows[0]["source_property"] == {"overflowFlags": "#arith.overflow<none>"}
    assert tuple(op.operands) == operands and tuple(op.results) == results and op.attributes == attrs
    assert op in list(module.walk()) and not op.properties


@pytest.mark.parametrize("flags", ["nsw", "nuw", "nsw,nuw"])
def test_actual_stock_nondefault_flags_refuse_without_partial_mutation(tmp_path, flags):
    source = stock(
        tmp_path, "module {func.func @f(%x:i32)->i8{%r=arith.trunci %x overflow<" + flags + ">:i32 to i8 return %r:i8}}"
    )
    module = parse_mlir_text(source.read_text())
    op = next(op for op in module.walk() if isinstance(op, TruncIOp))
    before = dict(op.properties)
    with pytest.raises(ValueError, match="default-none"):
        unflagged_truncations(module)
    assert op.properties == before


def test_unknown_flag_kind_and_wrong_attribute_position_refuse(tmp_path):
    source = stock(tmp_path, "module {func.func @f(%x:i32)->i8{%r=arith.trunci %x:i32 to i8 return %r:i8}}")
    for mode in ("kind", "attribute"):
        module = parse_mlir_text(source.read_text())
        op = next(op for op in module.walk() if isinstance(op, TruncIOp))
        if mode == "kind":
            op.properties["overflowFlags"] = StringAttr("none")
        else:
            op.attributes["overflowFlags"] = IntegerOverflowAttr("none")
        with pytest.raises(ValueError):
            unflagged_truncations(module)


def test_no_property_changes_nothing():
    module = parse_mlir_text("module {func.func @f(%x:i32)->i8{%r=arith.trunci %x:i32 to i8 return %r:i8}}")
    before = checked_generic_text(module)
    assert not unflagged_truncations(module)
    assert checked_generic_text(module) == before


def test_mixed_full_function_keeps_integer_output_and_fuses_only_fp8(tmp_path):
    # The codec fixture supplies structural interfaces only. Real codec bytes
    # and source-domain numerical qualification remain separate requirements.
    from test_fp8_scalar_codec import codec

    source = tmp_path / "mixed.mlir"
    source.write_text(MIXED)
    policy = codec(tmp_path)
    result = fuse_clamped_encoders(source, tmp_path / "prepared", policy)
    parsed = parse_mlir_text(result.read_text())
    parsed.verify()
    assert source.read_text() == MIXED
    assert len([op for op in parsed.walk() if op.name == "arith.trunci"]) == 1
    assert len([op for op in parsed.walk() if op.name == "arith.truncf"]) == 1
    assert len([op for op in parsed.walk() if op.name == "linalg.generic"]) == 2
    report = json.loads((tmp_path / "prepared/report.json").read_text())
    assert len(report["stock_integer_compatibility"]) == len(report["pairs"]) == 1


def test_none_path_does_not_discover_source_or_make_output(tmp_path):
    source = tmp_path / "absent.mlir"
    output = tmp_path / "no-output"
    assert fuse_clamped_encoders(source, output, None) == source
    assert not output.exists()


def test_actual_compiler_modular_truncation_all_codes_and_extremes(tmp_path):
    source = stock(tmp_path, "module {func.func @f(%x:i32)->i8{%r=arith.trunci %x:i32 to i8 return %r:i8}}")
    module = parse_mlir_text(source.read_text())
    assert unflagged_truncations(module)
    clean = tmp_path / "clean.mlir"
    clean.write_text(checked_generic_text(module))
    llvmd, llvm, library = tmp_path / "llvm.mlir", tmp_path / "model.ll", tmp_path / "model.so"
    bindir = toolchain.llvm_install() / "bin"
    subprocess.run(
        [
            str(bindir / "mlir-opt"),
            str(clean),
            "--convert-arith-to-llvm",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            str(llvmd),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [str(bindir / "mlir-translate"), "--mlir-to-llvmir", str(llvmd), "-o", str(llvm)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [str(toolchain.clang()), "-shared", "-fPIC", str(llvm), "-o", str(library)], check=True, capture_output=True
    )
    function = ctypes.CDLL(str(library)).f
    function.argtypes, function.restype = [ctypes.c_int32], ctypes.c_int8
    for value in [*range(-65536, 65536), -(1 << 31), (1 << 31) - 1]:
        assert function(value) == (value + 128) % 256 - 128


def test_later_nondefault_cast_refuses_before_mutating_earlier_default(tmp_path):
    source = stock(
        tmp_path,
        "module {func.func @f(%x:i64)->i8{"
        "%a=arith.trunci %x:i64 to i32 "
        "%b=arith.trunci %a overflow<nsw>:i32 to i8 return %b:i8}}",
    )
    module = parse_mlir_text(source.read_text())
    casts = [op for op in module.walk() if isinstance(op, TruncIOp)]
    before = [dict(op.properties) for op in casts]
    with pytest.raises(ValueError, match="default-none"):
        unflagged_truncations(module)
    assert [op.properties for op in casts] == before


def test_helper_dependency_bytes_are_bound_and_cannot_change_during_fusion(tmp_path, monkeypatch):
    from test_fp8_scalar_codec import codec

    from merlin.llvmlower import stock_arith_compat

    dependency = tmp_path / "selected_compatibility.py"
    dependency.write_text("# immutable declared helper dependency\n")
    monkeypatch.setattr(stock_arith_compat, "__file__", str(dependency))
    actual = stock_arith_compat.unflagged_truncations

    def changed(module):
        result = actual(module)
        dependency.write_text("# changed dependency\n")
        return result

    monkeypatch.setattr(stock_arith_compat, "unflagged_truncations", changed)
    source = tmp_path / "mixed.mlir"
    source.write_text(MIXED)
    with pytest.raises(ValueError, match="changed"):
        fuse_clamped_encoders(source, tmp_path / "refused", codec(tmp_path))
    assert not (tmp_path / "refused/model.mlir").exists()
