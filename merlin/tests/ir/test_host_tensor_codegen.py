"""Shared host lowering executes with dense and caller-selected padded storage."""

import ctypes
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.host_builder import PTR, FnBuilder
from merlin.llvmlower.host_epilogue import HostEpilogueEmitter
from merlin.llvmlower.host_scalar_builder import FpBuilder
from merlin.llvmlower.host_tensor_codegen import HostLinalg, LoweringDeclined, row_pitch
from merlin.llvmlower.toolchain import clang
from merlin.tests.ir.test_linalg_pointwise_patterns import _typed_program


def _compiled(builder, tmp_path, stem):
    count = len(builder.entry.args)
    module = ModuleOp(
        [
            llvm.FuncOp(
                "host_kernel",
                llvm.LLVMFunctionType([PTR] * count),
                linkage=llvm.LinkageAttr("external"),
                body=builder.finish(),
            )
        ]
    )
    module.verify()
    mlir, ll, so = (tmp_path / (stem + suffix) for suffix in (".mlir", ".ll", ".so"))
    mlir.write_text(str(module))
    translate = clang().parent / "mlir-translate"
    subprocess.run(
        [str(translate), "--mlir-to-llvmir", str(mlir), "-o", str(ll)], check=True, capture_output=True, timeout=30
    )
    subprocess.run(
        [str(clang()), "-O3", "-ffp-contract=off", "-shared", "-fPIC", str(ll), "-o", str(so)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    kernel = ctypes.CDLL(str(so)).host_kernel
    kernel.argtypes = [ctypes.c_void_p] * count
    kernel.restype = None
    return kernel


@pytest.mark.parametrize("alignment", [1, 8])
@pytest.mark.parametrize("dtype,operation", [("i32", "arith.addi"), ("f32", "arith.addf")])
def test_cpu_output_all_elements_padding_and_repetition(tmp_path, alignment, dtype, operation):
    source = parse_mlir_text(_typed_program(operation, (dtype, dtype), dtype, shape="3x5"))
    function = next(op for op in source.walk() if op.name == "func.func")
    builder = FnBuilder([PTR] * 4)
    lower = HostLinalg(
        FpBuilder(builder), list(builder.entry.args[:3]), [builder.entry.args[3]], row_alignment=alignment
    )
    lower.run(function)
    # The suite reclaims case directories; dlopen caches each absolute filename.
    stem = f"host_{alignment}_{dtype}"
    kernel = _compiled(builder, tmp_path, stem)
    ty = np.int32 if dtype == "i32" else np.float32
    pitch = row_pitch(5, alignment=alignment)
    left = np.full((3, pitch), 37, dtype=ty)
    right, init = left.copy(), left.copy()
    left[:, :5] = np.arange(15, dtype=ty).reshape(3, 5) - 7
    right[:, :5] = np.arange(15, dtype=ty).reshape(3, 5) * 3
    if dtype == "i32":
        left[0, 0], right[0, 0] = 2147483647, 1
    else:
        left[0, 0], right[0, 0] = 16777216, 1
    expected = left[:, :5] + right[:, :5]
    for _ in range(2):
        guarded = np.full(3 * pitch + 16, 73, dtype=ty)
        output = guarded[8:-8].reshape(3, pitch)
        kernel(*(x.ctypes.data for x in (left, right, init, output)))
        np.testing.assert_array_equal(output[:, :5].view(np.uint32), expected.view(np.uint32))
        assert np.all(output[:, 5:] == 73)
        assert np.all(guarded[:8] == 73) and np.all(guarded[-8:] == 73)


def test_unknown_storage_and_excessive_straight_line_expansion_refuse():
    for alignment in (True, 0, -1):
        with pytest.raises(ValueError, match="alignment"):
            row_pitch(5, alignment=alignment)
    source = parse_mlir_text(_typed_program("arith.addf", ("f32", "f32"), "f32", shape="3x5"))
    function = next(op for op in source.walk() if op.name == "func.func")
    builder = FnBuilder([PTR] * 4)
    lower = HostLinalg(FpBuilder(builder), list(builder.entry.args[:3]), [builder.entry.args[3]], element_budget=4)
    with pytest.raises(LoweringDeclined, match="straight-line element"):
        lower.run(function)


@pytest.mark.parametrize("kind", ["transpose", "bias", "requant", "pool"])
def test_standalone_cpu_epilogues_have_explicit_storage_and_formats(tmp_path, kind):
    names = ["input", "bias", "output"] if kind == "bias" else ["input", "output"]
    builder = FnBuilder([PTR] * len(names))
    pointers = dict(zip(names, builder.entry.args))
    emitter = HostEpilogueEmitter(
        builder, pointers.__getitem__, FpBuilder(builder), row_alignment=8, operand_dtype="i8", pool_floor=-(1 << 31)
    )
    values = np.arange(12, dtype=np.int32).reshape(4, 3) * 61 - 300
    inputs = np.full((4, 8), 91, np.int32)
    inputs[:, :3] = values
    attrs = dict(rows=4, cols=3, dtype="i32")
    if kind == "transpose":
        emitter.emit_transpose(SimpleNamespace(attrs=attrs, bufs=names))
        expected = values.T
    elif kind == "bias":
        emitter.emit_bias_add(SimpleNamespace(attrs=attrs, bufs=names))
        expected = values + np.array([2, -7, 9], np.int32)
    else:
        attrs.update(
            stages=["maxpool"] if kind == "pool" else ["acc_scale"],
            stage_dtype="i32",
            out_dtype="i32" if kind == "pool" else "i8",
            src_row_elems=8,
            src_row_offset=0,
            dst_row_elems=8,
            dst_row_offset=0,
            acc_scale=0.5,
            pool_in_dims=(2, 2),
            pool_size=(2, 2),
            pool_stride=(2, 2),
            pool_padding=(0, 0, 0, 0),
            pool_pad_value=None,
        )
        emitter.emit_epilogue(SimpleNamespace(attrs=attrs, bufs=names))
        expected = (
            values.max(axis=0, keepdims=True)
            if kind == "pool"
            else np.clip(np.rint(values.astype(np.float32) * np.float32(0.5)), -128, 127).astype(np.int8)
        )
    kernel = _compiled(builder, tmp_path, kind)
    rows, cols = expected.shape
    guarded = np.full(rows * 8 + 16, 73, dtype=expected.dtype)
    output = guarded[8:-8].reshape(rows, 8)
    arguments = [inputs]
    if kind == "bias":
        arguments.append(np.array([2, -7, 9, 91, 91, 91, 91, 91], np.int32))
    arguments.append(output)
    kernel(*(x.ctypes.data for x in arguments))
    np.testing.assert_array_equal(output[:, :cols], expected)
    assert np.all(output[:, cols:] == 73)
    assert np.all(guarded[:8] == 73) and np.all(guarded[-8:] == 73)
