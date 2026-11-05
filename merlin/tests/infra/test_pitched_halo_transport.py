"""Execute exact halo storage for arbitrary original pointer congruences."""

import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i64

from merlin.compile.scheduling.window_transfer import Window2D
from merlin.llvmlower.host_builder import PTR, FnBuilder
from merlin.llvmlower.pitched_halo_transport import (
    PitchedNHWCHalo, emit_pitched_halo_address, emit_pitched_nhwc_halo,
)
from merlin.xdsl_dialects._common import text


@pytest.mark.parametrize("alignment", [1, 8, 16])
def test_all_original_pointer_phases_and_image_rows_are_congruent(alignment):
    layout = PitchedNHWCHalo(Window2D((2, 3, 5, 3), (1, 1), padding=(2, 1, 0, 3)), alignment)
    assert layout.row_pitch_bytes >= layout.logical_row_bytes
    assert layout.row_pitch_bytes % alignment == layout.source_row_bytes % alignment
    pt, pl, _, _ = layout.window.padding
    for src in range(alignment):
        for dst in range(alignment):
            for image in range(2):
                for row in range(3):
                    logical = image * layout.logical_image_bytes + (pt + row) * layout.logical_row_bytes + pl * 3
                    physical = layout.physical_offset(src, dst, logical)
                    assert (dst + physical) % alignment == (src + (image * 3 + row) * 15) % alignment
                    assert 0 <= physical < layout.private_bytes
    json.dumps(layout.summary())


def _module(layout):
    pack = FnBuilder([PTR, PTR])
    receipt = emit_pitched_nhwc_halo(pack, layout, *pack.entry.args,
        source_destination_disjoint=True, maximum_private_bytes=layout.private_bytes)
    address = FnBuilder([PTR, PTR, i64])
    result = emit_pitched_halo_address(address, layout, *address.entry.args)
    offset = address.add(llvm.SubOp(address.add(llvm.PtrToIntOp(result)).output,
                                   address.add(llvm.PtrToIntOp(address.entry.args[1])).output)).res
    address.add(llvm.ReturnOp(offset))
    module = ModuleOp([
        llvm.FuncOp("pack", llvm.LLVMFunctionType([PTR, PTR], llvm.LLVMVoidType()),
                    linkage=llvm.LinkageAttr("external"), body=pack.finish()),
        llvm.FuncOp("address", llvm.LLVMFunctionType([PTR, PTR, i64], i64),
                    linkage=llvm.LinkageAttr("external"), body=address.region),
    ])
    module.verify()
    return module, receipt


@pytest.mark.parametrize("window", [
    Window2D((1, 224, 224, 3), (1, 1), padding=(3, 3, 3, 3)),
    Window2D((1, 225, 227, 3), (1, 1), padding=(3, 3, 3, 3)),
    Window2D((2, 3, 5, 5), (1, 1), padding=(2, 1, 0, 3)),
])
def test_native_complete_source_halo_and_emitted_mapping_with_guards(tmp_path, window):
    tools = Path("/work/merlin/toolchains/llvm-a47bddcc/install/bin")
    layout = PitchedNHWCHalo(window, 8)
    module, receipt = _module(layout)
    source, ll, library = (tmp_path / name for name in ("copy.mlir", "copy.ll", "copy.so"))
    source.write_text(text(module))
    library = tmp_path / ("copy-" + hashlib.sha256(source.read_bytes()).hexdigest()[:16] + ".so")
    subprocess.run([str(tools / "mlir-translate"), "--mlir-to-llvmir", str(source), "-o", str(ll)],
                   check=True, capture_output=True)
    subprocess.run([str(tools / "clang"), "-O2", "-shared", "-fPIC", str(ll), "-o", str(library)],
                   check=True, capture_output=True)
    native = ctypes.CDLL(str(library))
    native.pack.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    native.address.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
    native.address.restype = ctypes.c_uint64
    count = int(np.prod(window.input_shape))
    pt, pl, pb, pr = window.padding
    values = ((np.arange(count) * 7 + 3) % 256).astype(np.uint8).reshape(window.input_shape)
    expected = np.pad(values, ((0, 0), (pt, pb), (pl, pr), (0, 0)))
    for source_phase, destination_phase in ((0, 0), (1, 3), (7, 2), (3, 7)):
        inputs = np.full(count + 32, 173, dtype=np.uint8)
        outputs = np.full(layout.private_bytes + 32, 197, dtype=np.uint8)
        src = inputs.ctypes.data + 8 + source_phase
        dst = outputs.ctypes.data + 8 + destination_phase
        inputs[8 + source_phase:8 + source_phase + count] = values.reshape(-1)
        original = inputs.copy()
        native.pack(src, dst)
        flat = expected.reshape(window.input_shape[0], window.output_spatial[0], layout.logical_row_bytes)
        for image in range(window.input_shape[0]):
            for row in range(window.output_spatial[0]):
                logical = image * layout.logical_image_bytes + row * layout.logical_row_bytes
                offset = layout.physical_offset(src, dst, logical)
                assert native.address(src, dst, logical) == offset
                start = 8 + destination_phase + offset
                np.testing.assert_array_equal(outputs[start:start + layout.logical_row_bytes], flat[image, row])
        np.testing.assert_array_equal(inputs, original)
        assert np.all(outputs[:8 + destination_phase] == 197)
        assert np.all(outputs[8 + destination_phase + layout.private_bytes:] == 197)
    assert receipt["source_overread_bytes"] == 0 and receipt["source_alignment_assumed"] is False


@pytest.mark.parametrize("case", ["alignment", "window", "alias", "capacity", "coordinate"])
def test_unknown_geometry_ownership_and_capacity_refuse(case):
    window = Window2D((1, 5, 7, 3), (1, 1), padding=(1, 2, 3, 4))
    if case == "alignment":
        with pytest.raises(ValueError):
            PitchedNHWCHalo(window, 3)
        return
    if case == "window":
        with pytest.raises(ValueError):
            PitchedNHWCHalo(Window2D((1, 5, 7, 3), (3, 3)), 8)
        return
    layout = PitchedNHWCHalo(window, 8)
    if case == "coordinate":
        with pytest.raises(ValueError):
            layout.physical_offset(0, 0, layout.logical_bytes)
        return
    builder = FnBuilder([PTR, PTR])
    with pytest.raises(ValueError):
        emit_pitched_nhwc_halo(builder, layout, *builder.entry.args,
            source_destination_disjoint=case != "alias",
            maximum_private_bytes=layout.private_bytes - (case == "capacity"))
