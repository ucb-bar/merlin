"""Execute generated host transport; no target arithmetic or simulator authority."""

import ctypes
import hashlib
import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i64
from xdsl.printer import Printer

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.llvmlower.convolution_packing import (
    ClippedSpatialHaloRows,
    emit_nhwc_halo_panel,
    emit_nhwc_window_panel,
)
from merlin.llvmlower.host_builder import PTR, FnBuilder


def emit(window, rows, *, disjoint=True, budget=65536):
    builder = FnBuilder([PTR, PTR, i64])
    receipt = emit_nhwc_window_panel(
        builder, window, *builder.entry.args, rows, source_destination_disjoint=disjoint, maximum_panel_bytes=budget
    )
    function = llvm.FuncOp(
        "pack",
        llvm.LLVMFunctionType([PTR, PTR, i64], llvm.LLVMVoidType()),
        linkage=llvm.LinkageAttr("external"),
        body=builder.finish(),
    )
    module = ModuleOp([function])
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue(), receipt


@pytest.mark.parametrize(
    "window,rows",
    [
        (Window2D((1, 5, 5, 3), (3, 3), padding=(1, 1, 1, 1)), 7),
        (Window2D((2, 6, 7, 2), (2, 3), stride=(2, 1), dilation=(2, 1), padding=(1, 2, 0, 1)), 5),
        (Window2D((1, 2, 3, 1), (1, 1)), 3),
    ],
)
def test_native_window_panel_copies_exactly_with_guards(tmp_path, window, rows):
    directory = Path(
        shutil.which("mlir-translate") or "/work/merlin/toolchains/llvm-a47bddcc/install/bin/mlir-translate"
    ).parent
    if not all((directory / name).is_file() for name in ("mlir-translate", "clang")):
        pytest.skip("native LLVM tools unavailable")
    text, receipt = emit(window, rows)
    source = tmp_path / "pack.mlir"
    source.write_text(text)
    ll = tmp_path / "pack.ll"
    library = tmp_path / ("pack-" + hashlib.sha256(text.encode()).hexdigest()[:16] + ".so")
    subprocess.run(
        [str(directory / "mlir-translate"), "--mlir-to-llvmir", str(source), "-o", str(ll)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [str(directory / "clang"), "-O1", "-shared", "-fPIC", str(ll), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    module = ctypes.CDLL(str(library))
    fn = module.pack
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    source_array = (
        ((np.arange(np.prod(window.input_shape)) * 7) % 251 - 125).astype(np.int8).reshape(window.input_shape)
    )
    kh, kw = window.kernel
    batch, height, width, channels = window.input_shape
    oh, ow = window.output_spatial
    count = batch * oh * ow
    for start in (0, 1, count - 1, count, (1 << 64) - 1):
        output = np.full(receipt["panel_bytes"] + 16, 97, dtype=np.int8)
        fn(source_array.ctypes.data, output.ctypes.data + 8, start)
        expected = []
        for row in range(start, start + rows):
            image, position = divmod(row, oh * ow)
            y, x = divmod(position, ow)
            for ky in range(kh):
                for kx in range(kw):
                    iy = y * window.stride[0] - window.padding[0] + ky * window.dilation[0]
                    ix = x * window.stride[1] - window.padding[1] + kx * window.dilation[1]
                    for channel in range(channels):
                        expected.append(
                            source_array[image, iy, ix, channel]
                            if row < count and 0 <= iy < height and 0 <= ix < width
                            else 0
                        )
        np.testing.assert_array_equal(output[8:-8], expected)
        assert np.all(output[:8] == 97) and np.all(output[-8:] == 97)
    assert "llvm.cond_br" in text and receipt["device_arithmetic"] is False


def test_storage_permission_and_capacity_are_explicit():
    window = Window2D((1, 5, 5, 3), (3, 3))
    with pytest.raises(ValueError, match="disjoint"):
        emit(window, 2, disjoint=False)
    with pytest.raises(ValueError, match="capacity"):
        emit(window, 2, budget=53)


@pytest.mark.parametrize("shape,padding", [
    ((2, 3, 5, 3), (1, 2, 0, 3)),
    ((1, 2, 3, 1), (0, 0, 0, 0)),
    ((2, 1, 4, 17), (2, 1, 3, 2)),
    ((1, 2, 7, 5), (0, 0, 1, 0)),
])
def test_complete_unit_window_uses_exact_copy_fill_spans(tmp_path, shape, padding):
    directory = Path("/work/merlin/toolchains/llvm-a47bddcc/install/bin")
    if not all((directory / name).is_file() for name in ("mlir-translate", "clang")):
        pytest.skip("native LLVM tools unavailable")
    window = Window2D(shape, (1, 1), padding=padding)
    count = shape[0] * np.prod(window.output_spatial)
    size = int(count * shape[-1])
    builder = FnBuilder([PTR, PTR])
    receipt = emit_nhwc_window_panel(
        builder, window, *builder.entry.args, builder.const(0), int(count),
        source_destination_disjoint=True, maximum_panel_bytes=size,
    )
    module = ModuleOp([llvm.FuncOp(
        "pack", llvm.LLVMFunctionType([PTR, PTR], llvm.LLVMVoidType()),
        linkage=llvm.LinkageAttr("external"), body=builder.finish(),
    )])
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    text = stream.getvalue()
    assert receipt["transport"] == "complete_unit_window_row_spans"
    assert "llvm.memcpy" in text and "llvm.udiv" not in text and "llvm.urem" not in text
    source = tmp_path / "unit.mlir"
    source.write_text(text)
    ll = tmp_path / "unit.ll"
    library = tmp_path / ("unit-" + hashlib.sha256(text.encode()).hexdigest()[:16] + ".so")
    subprocess.run([str(directory / "mlir-translate"), "--mlir-to-llvmir", str(source), "-o", str(ll)],
                   check=True, capture_output=True)
    subprocess.run([str(directory / "clang"), "-O1", "-shared", "-fPIC", str(ll), "-o", str(library)],
                   check=True, capture_output=True)
    fn = ctypes.CDLL(str(library)).pack
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    values = (np.arange(np.prod(shape)) * 7 % 251 - 125).astype(np.int8).reshape(shape)
    pt, pl, pb, pr = padding
    expected = np.pad(values, ((0, 0), (pt, pb), (pl, pr), (0, 0))).ravel()
    for source_alignment, destination_alignment in ((0, 0), (1, 2), (3, 7)):
        data = np.full(values.size + source_alignment + 16, 97, dtype=np.int8)
        data[source_alignment:source_alignment + values.size] = values.ravel()
        output = np.full(size + destination_alignment + 16, 97, dtype=np.int8)
        fn(data.ctypes.data + source_alignment, output.ctypes.data + destination_alignment)
        np.testing.assert_array_equal(output[destination_alignment:destination_alignment + size], expected)
        assert np.all(output[:destination_alignment] == 97) and np.all(output[destination_alignment + size:] == 97)
        assert np.all(data[source_alignment + values.size:] == 97)


@pytest.mark.parametrize("columns", [1, 3])
def test_native_clipped_halo_packing_crosses_scanlines_and_preserves_original_k(tmp_path, columns):
    directory = Path("/work/merlin/toolchains/llvm-a47bddcc/install/bin")
    if not all((directory / name).is_file() for name in ("mlir-translate", "clang")):
        pytest.skip("native LLVM tools unavailable")
    window = Window2D((2, 6, 7, 3), (3, 2), stride=(1, 1), dilation=(1, 2), padding=(1, 1, 1, 1))
    halo = ClippedSpatialHaloRows(SpatialMaxReadout(window.output_spatial, (3, 3), (2, 2), (1, 1, 1, 1), -128), columns)
    rows = 5
    builder = FnBuilder([PTR, PTR, i64, i64, i64, i64])
    receipt = emit_nhwc_halo_panel(
        builder,
        window,
        halo,
        *builder.entry.args,
        rows,
        source_destination_disjoint=True,
        maximum_panel_bytes=rows * 18,
    )
    module = ModuleOp(
        [
            llvm.FuncOp(
                "pack",
                llvm.LLVMFunctionType([PTR, PTR, i64, i64, i64, i64], llvm.LLVMVoidType()),
                linkage=llvm.LinkageAttr("external"),
                body=builder.finish(),
            )
        ]
    )
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    source = tmp_path / "halo.mlir"
    source.write_text(stream.getvalue())
    ll, library = tmp_path / "halo.ll", tmp_path / f"halo-{columns}.so"
    subprocess.run(
        [str(directory / "mlir-translate"), "--mlir-to-llvmir", str(source), "-o", str(ll)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [str(directory / "clang"), "-O1", "-shared", "-fPIC", str(ll), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    fn = ctypes.CDLL(str(library)).pack
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *([ctypes.c_uint64] * 4)]
    data = (np.arange(np.prod(window.input_shape)) % 127 - 63).astype(np.int8).reshape(window.input_shape)
    ph, pw = halo.spatial.output_hw
    cases = [
        (b, y, x, start)
        for b in range(2)
        for y in (0, ph - 1)
        for x in (0, pw - columns)
        for start in (0, halo.width - 1, halo.rows - 1, halo.rows)
    ]
    cases += [(2, 0, 0, 0), (0, ph, 0, 0), (0, 0, pw - columns + 1, 0), (0, 0, 0, (1 << 64) - 1)]
    for b, y, x, first in cases:
        output = np.full(receipt["panel_bytes"] + 16, 97, dtype=np.int8)
        fn(data.ctypes.data, output.ctypes.data + 8, b, y, x, first)
        expected = np.zeros((rows, 18), dtype=np.int8)
        for r in range(rows):
            local = first + r
            if not (b < 2 and y < ph and x <= pw - columns and local < halo.rows):
                continue
            hy, hx = divmod(local, halo.width)
            cy = min(max(y * 2 - 1 + hy, 0), 5)
            cx = min(max(x * 2 - 1 + hx, 0), 6)
            for ky in range(3):
                for kx in range(2):
                    iy, ix = cy - 1 + ky, cx - 1 + kx * 2
                    if 0 <= iy < 6 and 0 <= ix < 7:
                        expected[r, (ky * 2 + kx) * 3 : (ky * 2 + kx + 1) * 3] = data[b, iy, ix]
        np.testing.assert_array_equal(output[8:-8].reshape(rows, 18), expected, err_msg=str((b, y, x, first)))
        assert np.all(output[:8] == 97) and np.all(output[-8:] == 97)
