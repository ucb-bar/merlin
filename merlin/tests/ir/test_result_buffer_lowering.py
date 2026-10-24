"""General result allocation aliasing must never crash upstream lowering."""

import pytest

from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.toolchain import m2m_python


@pytest.mark.parametrize("extent", [0, 7])
def test_aliased_allocated_results_lower_without_erasing_twice(tmp_path, extent):
    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")
    source = f"""builtin.module {{
      func.func @forward() -> (tensor<{extent}xf32>, tensor<{extent}xf32>) {{
        %a = tensor.empty() : tensor<{extent}xf32>
        func.return %a, %a : tensor<{extent}xf32>, tensor<{extent}xf32>
      }}
    }}"""
    text = lower_to_llvm_ir(source, workdir=tmp_path)
    assert "define void @forward" in text


def test_repeated_borrowed_argument_is_not_an_allocation(tmp_path):
    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")
    source = """builtin.module {
      func.func @forward(%x: tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
        func.return %x, %x : tensor<3xi64>, tensor<3xi64>
      }
    }"""
    assert "define void @forward" in lower_to_llvm_ir(source, workdir=tmp_path)


def test_static_results_preserve_memrefs_when_observed(tmp_path):
    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")
    source = """builtin.module {
      func.func @forward(%x: tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
        func.return %x, %x : tensor<3xi64>, tensor<3xi64>
      }
    }"""
    text = lower_to_llvm_ir(source, workdir=tmp_path, preserve_result_descriptors=True)
    assert "define { { ptr, ptr, i64" in text
    assert "@forward" in text
    from merlin.llvmlower.result_buffers import descriptor_results

    pipeline = descriptor_results("buffer-results-to-out-params,finalize-memref-to-llvm", source, preserve=True)
    assert "buffer-results-to-out-params" not in pipeline
    assert "convert-bufferization-to-memref" in pipeline


def test_early_bufferization_conversion_requires_a_layout_contract():
    from merlin.llvmlower.result_buffers import descriptor_results
    from merlin.llvmlower.result_layout import module_attribute

    body = " { func.func @forward(%x: tensor<?xf32>) -> tensor<?xf32> { return %x : tensor<?xf32> } }"
    pipeline = (
        "one-shot-bufferize,buffer-results-to-out-params,bufferization-lower-deallocations,"
        "func.func(buffer-loop-hoisting),convert-linalg-to-loops,finalize-memref-to-llvm"
    )
    ordinary = descriptor_results(pipeline, "module" + body).split(",")
    assert ordinary.index("convert-bufferization-to-memref") + 1 == ordinary.index("finalize-memref-to-llvm")
    layout = descriptor_results(pipeline, "module" + module_attribute({}) + body).split(",")
    assert layout.index("bufferization-lower-deallocations") + 1 == layout.index("convert-bufferization-to-memref")


@pytest.mark.parametrize("layout", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_descriptor_results_name_missing_finalization_pass(layout, preserve):
    from merlin.llvmlower.result_buffers import descriptor_results
    from merlin.llvmlower.result_layout import module_attribute

    body = " { func.func @forward(%x: tensor<?xf32>) -> tensor<?xf32> { return %x : tensor<?xf32> } }"
    source = "module" + (module_attribute({}) if layout else "") + body
    with pytest.raises(ValueError, match="requires finalize-memref-to-llvm in the pipeline"):
        descriptor_results("one-shot-bufferize,buffer-results-to-out-params", source, preserve=preserve)


@pytest.mark.parametrize("extent", [0, 7])
def test_non_core_aten_dynamic_results_execute_with_original_pass_order(tmp_path, extent):
    import ctypes
    import subprocess

    import numpy as np

    from merlin.llvmlower.toolchain import clang, llvm_install

    source = """module {
      func.func @forward(%x: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>)
          attributes {llvm.emit_c_interface} {
        %c0 = arith.constant 0 : index
        %one = arith.constant 1.0 : f32
        %n = tensor.dim %x, %c0 : tensor<?xf32>
        %empty = tensor.empty(%n) : tensor<?xf32>
        %filled = linalg.fill ins(%one : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
        return %x, %filled : tensor<?xf32>, tensor<?xf32>
      }
    }"""
    text = lower_to_llvm_ir(source, workdir=tmp_path)
    path, library = tmp_path / "dynamic.ll", tmp_path / "dynamic.so"
    path.write_text(text)
    lib = llvm_install() / "lib"
    subprocess.run(
        [
            str(clang()),
            "-shared",
            "-fPIC",
            "-O1",
            "-Wl,-Bsymbolic",
            str(path),
            "-L" + str(lib),
            "-lmlir_c_runner_utils",
            "-Wl,-rpath," + str(lib),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )

    class Descriptor(ctypes.Structure):
        _fields_ = [
            ("allocated", ctypes.c_void_p),
            ("aligned", ctypes.c_void_p),
            ("offset", ctypes.c_int64),
            ("size", ctypes.c_int64),
            ("stride", ctypes.c_int64),
        ]

    class Pair(ctypes.Structure):
        _fields_ = [("original", Descriptor), ("filled", Descriptor)]

    values = np.arange(extent, dtype=np.float32)
    arg = Descriptor(values.ctypes.data, values.ctypes.data, 0, extent, 1)
    output = Pair()
    entry = ctypes.CDLL(str(library))._mlir_ciface_forward
    entry.argtypes = [ctypes.POINTER(Pair), ctypes.POINTER(Descriptor)]
    entry(ctypes.byref(output), ctypes.byref(arg))
    for descriptor, expected in [(output.original, values), (output.filled, np.ones(extent))]:
        assert descriptor.size == extent
        pointer = ctypes.cast(descriptor.aligned, ctypes.POINTER(ctypes.c_float))
        np.testing.assert_array_equal(
            [pointer[descriptor.offset + i * descriptor.stride] for i in range(extent)], expected
        )
    libc = ctypes.CDLL(None)
    for allocation in {output.original.allocated, output.filled.allocated} - {arg.allocated}:
        libc.free(ctypes.c_void_p(allocation))
