"""Real portable C layout products and rejection before artifact publication."""

import ctypes
import shutil
import subprocess

import pytest

from merlin import api
from merlin.compiler.backend import layout
from merlin.compiler.backend.cabi import CRegionABI


def program(operation, source="2x3xf32", result="6xf32"):
    return f"""builtin.module {{
      func.func @chosen_entry(%x: tensor<{source}>) -> tensor<{result}> {{
        %y = {operation}
        func.return %y : tensor<{result}>
      }}
    }}"""


@pytest.mark.parametrize("source", [
    program("tensor.collapse_shape %x [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>"),
    program("tensor.expand_shape %x [[0, 1]] output_shape [2, 3] : tensor<6xf32> into tensor<2x3xf32>",
            "6xf32", "2x3xf32"),
    program("tensor.cast %x : tensor<6xf32> to tensor<6xf32>", "6xf32", "6xf32"),
])
def test_public_layout_compilation_preserves_raw_float_storage(tmp_path, source):
    cc = shutil.which("gcc")
    if cc is None:
        pytest.skip("actual GCC required to execute the emitted C")
    result = api.compile(source, backend=layout, output=tmp_path / "compiled")
    abi = CRegionABI.from_compilation(result)
    library = tmp_path / "layout.so"
    subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    *(str(result.directory / name) for name in abi.link_inputs),
                    "-o", str(library)], check=True, capture_output=True, text=True)
    invoke = getattr(ctypes.CDLL(str(library)), abi.symbol)
    pointers = ctypes.POINTER(ctypes.c_void_p)
    invoke.argtypes = [ctypes.c_void_p, pointers, pointers]
    invoke.restype = ctypes.c_int
    # Signaling/quiet NaN payloads, signed zero, infinity and subnormal bits.
    words = (0x7F800001, 0x7FC12345, 0x80000000, 0x7F800000, 1, 0xFF800001)
    source_buffer = (ctypes.c_uint32 * 6)(*words)
    output_buffer = (ctypes.c_uint32 * 6)(*[0xDEADBEEF] * 6)
    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(source_buffer))
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output_buffer))
    assert invoke(None, inputs, outputs) == 0
    assert tuple(source_buffer) == tuple(output_buffer) == words
    assert invoke(None, inputs, inputs) != 0
    loaded = api.inspect(result.manifest)
    assert loaded["metadata"]["execution_observed"] is False
    assert loaded["source_sha256"] == api.Source(source).sha256
    assert {a.kind for a in result.artifacts} == {"c-source", "c-header", "json"}


@pytest.mark.parametrize("source,options", [
    (program("arith.addf %x, %x : tensor<2x3xf32>", result="2x3xf32"), None),
    (program("tensor.cast %x : tensor<?xf32> to tensor<?xf32>", "?xf32", "?xf32"), None),
    (program("tensor.cast %x : tensor<6xi1> to tensor<6xi1>", "6xi1", "6xi1"), None),
    (program("tensor.cast %x : tensor<6xf32> to tensor<6xf32>", "6xf32", "6xf32"),
     api.CompileOptions(platform="zephyr")),
])
def test_layout_rejects_arithmetic_dynamic_storage_and_image_options(tmp_path, source, options):
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        api.compile(source, backend=layout, output=output, options=options)
    assert not output.exists()
