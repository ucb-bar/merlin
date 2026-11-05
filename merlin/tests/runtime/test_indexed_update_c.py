"""Guarded dynamic slice updates keep exact bytes and original state on errors."""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from merlin.compile.layout_c import emit_layout_c
from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def source_text(axis=1):
    shape = (2, 8, 4)
    update = list(shape)
    update[axis] = 1
    offsets = ["0"] * 3
    offsets[axis] = "-9223372036854775808"
    return f"""module {{
  func.func @forward(%cache: tensor<2x8x4xf32>,
      %update: tensor<{"x".join(map(str, update))}xf32>,
      %positions: tensor<1xi64>) -> tensor<2x8x4xf32> {{
    %zero_index = arith.constant 0 : index
    %position = tensor.extract %positions[%zero_index] : tensor<1xi64>
    %zero = arith.constant 0 : i64
    %limit = arith.constant {shape[axis]} : i64
    %low = arith.cmpi sge, %position, %zero : i64
    %high = arith.cmpi slt, %position, %limit : i64
    %valid = arith.andi %low, %high : i1
    cf.assert %valid, "index out of bounds"
    %at = arith.index_cast %position : i64 to index
    %result = "tensor.insert_slice"(%update,%cache,%at) <{{
      static_offsets = array<i64: {",".join(offsets)}>,
      static_sizes = array<i64: {",".join(map(str, update))}>,
      static_strides = array<i64: 1,1,1>, operandSegmentSizes = array<i32: 1,1,1,0,0>
    }}> : (tensor<{"x".join(map(str, update))}xf32>, tensor<2x8x4xf32>, index) -> tensor<2x8x4xf32>
    func.return %result : tensor<2x8x4xf32>
  }}
}}"""


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_dynamic_copy_c_and_independent_reference_agree(tmp_path, axis):
    source = translate_linalg_text(
        source_text(axis), entry="forward", target_identity="synthetic", lowering_policy="diagnostic"
    ).request
    operation = source.nodes[-1]
    assert operation.op == "indexed_insert_slice" and dict(operation.attrs)["axis"] == axis
    record, files = emit_layout_c(source, "update_cache")
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    compiler = shutil.which("cc")
    assert compiler
    library_name = f"copy-{axis}.so"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "update_cache.c",
            "-o",
            library_name,
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    entry = ctypes.CDLL(str(tmp_path / library_name)).update_cache
    entry.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    shape = (2, 8, 4)
    update_shape = list(shape)
    update_shape[axis] = 1
    patterns = np.array([0, 0x80000000, 0x7FC12345, 1, 0x3F800000, 0xBF000000], dtype="<u4")
    cache = np.resize(patterns, np.prod(shape)).reshape(shape)
    update = np.resize(patterns[::-1], np.prod(update_shape)).reshape(update_shape)
    a, b = ctypes.create_string_buffer(cache.tobytes()), ctypes.create_string_buffer(update.tobytes())
    for index in (0, shape[axis] - 1, -1, shape[axis], -(1 << 63), (1 << 63) - 1):
        # Deliberately unaligned index memory exercises the byte-buffer ABI.
        position = ctypes.create_string_buffer(b"\xcd" + int(index).to_bytes(8, "little", signed=True) + b"\xcd")
        out = ctypes.create_string_buffer(b"\xa7" * (cache.nbytes + 32))
        args = (ctypes.c_void_p * 3)(ctypes.addressof(b), ctypes.addressof(a), ctypes.addressof(position) + 1)
        outputs = (ctypes.c_void_p * 1)(ctypes.addressof(out) + 16)
        status = entry(None, args, outputs)
        assert a.raw[:-1] == cache.tobytes() and b.raw[:-1] == update.tobytes()
        if index < 0 or index >= shape[axis]:
            assert status == -2 and out.raw[:-1] == b"\xa7" * (cache.nbytes + 32)
            continue
        assert status == 0
        expected = cache.copy()
        window = [slice(None)] * 3
        window[axis] = slice(index, index + 1)
        expected[tuple(window)] = update
        assert out.raw[16 : 16 + cache.nbytes] == expected.tobytes()
        assert out.raw[:16] == b"\xa7" * 16 and out.raw[16 + cache.nbytes : -1] == b"\xa7" * 16
        values = {
            source.nodes[0].id: TensorValue(source.nodes[0].type, tuple(map(int, cache.reshape(-1)))),
            source.nodes[1].id: TensorValue(source.nodes[1].type, tuple(map(int, update.reshape(-1)))),
            source.nodes[2].id: TensorValue(source.nodes[2].type, (index,)),
        }
        reference = evaluate_graph(source, values)[0]
        assert reference.elements == tuple(map(int, expected.reshape(-1)))
    assert record["failure_writes_output"] is False


@pytest.mark.parametrize("fault", ["missing_guard", "wrong_limit", "unsigned_low", "wrong_extract"])
def test_changed_or_missing_index_guards_are_rejected(fault):
    source = source_text()
    if fault == "missing_guard":
        source = source.replace('    cf.assert %valid, "index out of bounds"\n', "")
    elif fault == "wrong_limit":
        source = source.replace("constant 8 : i64", "constant 7 : i64")
    elif fault == "unsigned_low":
        source = source.replace("cmpi sge", "cmpi uge")
    else:
        source = source.replace("constant 0 : index", "constant 1 : index")
    with pytest.raises(LinalgBridgeError):
        translate_linalg_text(source, entry="forward", target_identity="synthetic", lowering_policy="diagnostic")
