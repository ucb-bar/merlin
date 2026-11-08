"""Check selected host BF16 dot scheduling with directed raw-bit fixtures."""
import ctypes
from dataclasses import replace
import shutil
import subprocess

import pytest

from merlin.compile.c_bf16_linear import POLICY, emit_c
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType


def request(k):
    type_ = lambda shape: TensorType(shape, "bf16", "raw-bit-pattern")
    maps = (IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
            IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0)),
            IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)))
    nodes = (SemanticNode("a", "input", (), type_((1, k)), effect="input"),
             SemanticNode("b", "input", (), type_((k, 1)), effect="input"),
             SemanticNode("seed", "input", (), type_((1, 1)), effect="input"),
             SemanticNode("out", "source_bf16_matmul_unqualified", ("a", "b", "seed"), type_((1, 1)),
                          (("numeric_semantics", "source-bf16-mul-add-unqualified-v1"), ("reduction_order", "unresolved")),
                          index_maps=(*maps, maps[-1])))
    return KernelRequest(nodes, ("out",), ("external",),
                         (("a", "external"), ("b", "external"), ("seed", "external")), "synthetic")


def compile_c(directory, k):
    record, files = emit_c(request(k), "dot")
    for name, text in files.items():
        (directory / name).write_text(text)
    (directory / "mode.c").write_text("""#include <fenv.h>
int mode(int downward) { return fesetround(downward?FE_DOWNWARD:FE_TONEAREST); }
""")
    # Passed tmp_path trees can be removed and their directory names reused.
    # The dynamic loader retains a path's first library, so bind it to shape.
    library_path = directory / f"dot-{k}.so"
    subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-frounding-math", "-ffp-contract=off", "-shared", "-fPIC",
                    str(directory / "dot.c"), str(directory / "mode.c"), "-lm", "-o", str(library_path)],
                   check=True, capture_output=True)
    library = ctypes.CDLL(str(library_path))
    library.dot.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    library.dot.restype = ctypes.c_int
    return record, library


def memory(words):
    return ctypes.create_string_buffer(b"guard" + b"".join(code.to_bytes(2, "little") for code in words) + b"end")


def call(library, arrays, output):
    inputs = (ctypes.c_void_p * 3)(*(ctypes.addressof(value) + 5 for value in arrays))
    return library.dot(None, inputs, (ctypes.c_void_p * 1)(ctypes.addressof(output) + 5))


@pytest.mark.parametrize("a,b,expected", [
    ([0x3f81], [0x3fc0], 0x3fc2),  # Exact BF16 product halfway rounds to even.
    ([0x3f80] * 4, [0x4b80, 0x3f80, 0xcb80, 0xbf80], 0),  # Adjacent FP32 tree, not serial accumulation.
    ([0x3f80] * 2, [0x4380, 0x4040], 0x4382),  # BF16 RNE after the FP32 tree.
    ([0x3f80] * 33, [0x3f80] + [0] * 31 + [0x3c40], 0x3f81),  # Tile accumulation chops.
    ([1], [0x3f80], 1),  # Preserve a BF16 subnormal through the declared operations.
    ([0x8000], [0x3f80], 0),  # The positive-zero seed determines this zero sign.
])
def test_directed_rounding_tree_tile_seam_and_raw_bits(tmp_path, a, b, expected):
    record, library = compile_c(tmp_path, len(a))
    arrays, output = [memory(a), memory(b), memory([0])], memory([0xdead])
    before = [value.raw for value in arrays]
    assert call(library, arrays, output) == 0
    assert output.raw == memory([expected]).raw
    assert [value.raw for value in arrays] == before
    assert record["numeric_semantics"] == POLICY
    assert record["placement"] == "host"


@pytest.mark.parametrize("k,expected", [(1, 0x3f80), (31, 0x41f8), (32, 0x4200), (33, 0x4204), (65, 0x4282)])
def test_partial_and_multiple_tiles(tmp_path, k, expected):
    _, library = compile_c(tmp_path, k)
    arrays, output = [memory([0x3f80] * k), memory([0x3f80] * k), memory([0])], memory([0xdead])
    assert call(library, arrays, output) == 0
    assert output.raw == memory([expected]).raw


def test_runtime_refusals_are_atomic(tmp_path):
    _, library = compile_c(tmp_path, 1)
    for a, b, seed in ((0x7fc0, 0x3f80, 0), (0x3f80, 0x7f80, 0), (0x7f7f, 0x7f7f, 0),
                       (0x3f80, 0x3f80, 0x8000)):
        output = memory([0xdead])
        assert call(library, [memory([a]), memory([b]), memory([seed])], output) == -2
        assert output.raw == memory([0xdead]).raw
    arrays, output = [memory([0x3f80]), memory([0x3f80]), memory([0])], memory([0xdead])
    assert library.mode(1) == 0
    try:
        assert call(library, arrays, output) == -1
    finally:
        assert library.mode(0) == 0
    assert output.raw == memory([0xdead]).raw
    assert call(library, arrays, arrays[0]) == -1


def test_source_contract_rejects_reinterpretation():
    source = request(1)
    changed = replace(source.nodes[-1], attrs=(("numeric_semantics", "ieee754-bf16-matmul-rne-v1"),))
    with pytest.raises(ValueError, match="source arithmetic"):
        emit_c(replace(source, nodes=(*source.nodes[:-1], changed)), "dot")
    operand = replace(source.nodes[0], attrs=(("hidden_scale", 2),))
    with pytest.raises(ValueError, match="source types or effects"):
        emit_c(replace(source, nodes=(operand, *source.nodes[1:])), "dot")
