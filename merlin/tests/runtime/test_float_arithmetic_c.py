"""Run generated binary64 multiply against independent rational IEEE rounding."""

import ctypes
import itertools
import os
import random
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.compile.c_float_arithmetic import emit_float_arithmetic_c
from merlin.semantic_compiler.model import (
    ConstantBinding,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph

_libraries = itertools.count()


def request(shape, shared=False):
    type_ = TensorType(shape, "f64", "raw-bit-pattern")
    a = SemanticNode("a", "input", (), type_, effect="input")
    b = replace(a, id="b")
    node = SemanticNode(
        "out",
        "f64_mul",
        ("a", "a" if shared else "b"),
        type_,
        (("numeric_semantics", "ieee754-f64-mul-rne-v1"),),
        index_maps=pointwise_identity_maps(len(shape), 2),
    )
    return KernelRequest(
        (a, node) if shared else (a, b, node),
        ("out",),
        ("external",),
        (("a", "external"),) if shared else (("a", "external"), ("b", "external")),
        "synthetic",
    )


def compile_c(directory, source):
    record, files = emit_float_arithmetic_c(source, "multiply")
    for name, text in files.items():
        (directory / name).write_text(text)
    (directory / "rounding.c").write_text("""#include <fenv.h>
    int mode(int i) { const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
      return fesetround(modes[i]); }
    """)
    output = directory / f"multiply_{next(_libraries)}.so"
    subprocess.run(
        [
            shutil.which("cc"),
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            str(directory / "multiply.c"),
            str(directory / "rounding.c"),
            "-lm",
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    if cross := os.environ.get("MERLIN_RISCV_GCC"):
        subprocess.run(
            [
                cross,
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-ffreestanding",
                "-march=rv64gcv",
                "-mabi=lp64d",
                "-mstrict-align",
                "-c",
                str(directory / "multiply.c"),
                "-o",
                str(directory / "multiply.rv64.o"),
            ],
            check=True,
            capture_output=True,
        )
    library = ctypes.CDLL(str(output))
    library.multiply.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    library.multiply.restype = ctypes.c_int
    return record, library


def memory(codes):
    return ctypes.create_string_buffer(b"guard" + b"".join(x.to_bytes(8, "little") for x in codes) + b"end!")


def invoke(library, operands, output):
    inputs = (ctypes.c_void_p * len(operands))(*(ctypes.addressof(a) + 5 for a in operands))
    return library.multiply(None, inputs, (ctypes.c_void_p * 1)(ctypes.addressof(output) + 5))


@pytest.mark.parametrize("shared", [False, True])
def test_boundaries_seeded_values_rounding_and_guards(tmp_path, shared):
    inf, sign = 0x7FF0000000000000, 1 << 63
    positive = [
        0,
        1,
        2,
        3,
        5,
        (1 << 52) - 1,
        1 << 52,
        0x3FEFFFFFFFFFFFFF,
        0x3FE0000000000000,
        0x3FF0000000000000,
        0x3FF0000000000001,
        0x3FF0000000000003,
        0x3FF8000000000000,
        0x4000000000000000,
        inf - 1,
        inf,
    ]
    edges = [n | s for s in (0, sign) for n in positive]
    rng = random.Random(20261020)
    pairs = list(itertools.product(edges, repeat=2))
    pairs += [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(4096)]
    pairs = [(a, a if shared else b) for a, b in pairs]
    pairs = [
        (a, b)
        for a, b in pairs
        if (a & (sign - 1)) <= inf
        and (b & (sign - 1)) <= inf
        and not ((a & (sign - 1)) == inf and not b & (sign - 1))
        and not ((b & (sign - 1)) == inf and not a & (sign - 1))
    ]
    source = request((len(pairs),), shared)
    record, library = compile_c(tmp_path, source)
    arrays = [tuple(a for a, _ in pairs)] + ([] if shared else [tuple(b for _, b in pairs)])
    arguments = {
        key: TensorValue(source.node(key).type, values) for key, values in zip(record["inputs"], arrays, strict=True)
    }
    (expected,) = evaluate_graph(source, arguments)
    expected_bytes = b"".join(x.to_bytes(8, "little") for x in expected.elements)
    inputs = [memory(values) for values in arrays]
    saved = [a.raw for a in inputs]
    try:
        for mode in range(4):
            assert library.mode(mode) == 0
            output = memory([0xCDCDCDCDCDCDCDCD] * len(pairs))
            assert invoke(library, inputs, output) == 0
            assert output.raw[5 : 5 + len(expected_bytes)] == expected_bytes
            assert output.raw[:5] == b"guard" and output.raw[-5:] == b"end!\0"
            assert [a.raw for a in inputs] == saved
    finally:
        library.mode(0)


@pytest.mark.parametrize(
    "bad_pair",
    [
        (0x7FF8000000000000, 0x3FF0000000000000),
        (0x3FF0000000000000, 0xFFF0000000000001),
        (0, 0x7FF0000000000000),
        (0xFFF0000000000000, 0x8000000000000000),
    ],
)
def test_late_invalid_lane_and_overlap_leave_output_untouched(tmp_path, bad_pair):
    source = request((3,))
    _, library = compile_c(tmp_path, source)
    one = 0x3FF0000000000000
    inputs = [memory([one, one, bad_pair[0]]), memory([one, one, bad_pair[1]])]
    output = memory([123, 456, 789])
    before = output.raw
    assert invoke(library, inputs, output) == -2 and output.raw == before
    before = inputs[0].raw
    assert invoke(library, inputs, inputs[0]) == -1 and inputs[0].raw == before
    assert library.multiply(None, None, None) == -1


def test_wrong_policy_types_and_index_maps_are_rejected():
    source = request((2, 3))
    node = source.nodes[-1]
    for wrong in (
        replace(node, attrs=()),
        replace(node, index_maps=()),
        replace(node, op="f64_add"),
        replace(node, type=TensorType((2, 3), "f32", "raw-bit-pattern")),
    ):
        with pytest.raises(ValueError):
            emit_float_arithmetic_c(replace(source, nodes=(*source.nodes[:-1], wrong)), "wrong")


def test_known_constant_zero_times_infinity_is_rejected_at_compile_time():
    source = request((1,))
    source = replace(
        source,
        nodes=tuple(replace(n, op="constant", effect="constant") if n.effect == "input" else n for n in source.nodes),
        input_storages=(),
        constants=(
            ConstantBinding("a", "f64-le-raw", "00" * 8),
            ConstantBinding("b", "f64-le-raw", (0x7FF0000000000000).to_bytes(8, "little").hex()),
        ),
    )
    with pytest.raises(ValueError, match="zero-times-infinity"):
        emit_float_arithmetic_c(source, "invalid_constants")
