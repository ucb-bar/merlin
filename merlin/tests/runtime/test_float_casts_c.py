"""Run source float casts with independent rounding, guard and ABI checks."""

import ctypes
import os
import random
import shutil
import subprocess
from dataclasses import replace
from itertools import count

import pytest

from merlin.compile.c_float_casts import emit_float_cast_c
from merlin.semantic_compiler.model import (
    ConstantBinding,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph

_TYPES = {
    "bf16_to_f32": ("bf16", "f32", 2, 4, "bf16-to-f32-exact-v1"),
    "f32_to_bf16": ("f32", "bf16", 4, 2, "f32-to-bf16-rne-v1"),
    "f32_to_f64": ("f32", "f64", 4, 8, "f32-to-f64-exact-v1"),
    "f64_to_f32": ("f64", "f32", 8, 4, "f64-to-f32-rne-v1"),
}
_libraries = count()


def request_for(operation, shape=(2, 3)):
    source, destination, _, _, policy = _TYPES[operation]
    arg = SemanticNode("x", "input", (), TensorType(shape, source, "raw-bit-pattern"), effect="input")
    node = SemanticNode("out", operation, (arg.id,), TensorType(shape, destination, "raw-bit-pattern"),
                        (("numeric_semantics", policy),), index_maps=pointwise_identity_maps(len(shape), 1))
    return KernelRequest((arg, node), (node.id,), ("external",), ((arg.id, "external"),), "synthetic")


def compile_c(directory, source):
    # Passed-test directories can be reused while dlopen retains old handles.
    library_path = directory / f"cast_{next(_libraries)}.so"
    record, files = emit_float_cast_c(source, "cast")
    for name, text in files.items():
        (directory / name).write_text(text)
    wrapper = directory / "rounding.c"
    wrapper.write_text("""#include <fenv.h>
    int set_rounding(int i) {
      const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
      return fesetround(modes[i]);
    }
    """)
    subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-shared", "-fPIC", "-fsanitize=undefined", "-fno-sanitize-recover=all",
                    str(directory / "cast.c"), str(wrapper), "-lm", "-o", str(library_path)],
                   check=True, capture_output=True)
    if cross := os.environ.get("MERLIN_RISCV_GCC"):
        subprocess.run([cross, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-ffreestanding",
                        "-march=rv64gcv", "-mabi=lp64d", "-mstrict-align", "-c", str(directory / "cast.c"),
                        "-o", str(directory / "cast.rv64.o")], check=True, capture_output=True)
    library = ctypes.CDLL(str(library_path))
    function = library.cast
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    function.restype = ctypes.c_int
    return record, library, function


def _non_nan(bits, width):
    fraction = {2: 7, 4: 23, 8: 52}[width]
    mask = (1 << (8 * width - 1)) - 1
    return bits & mask <= mask - ((1 << fraction) - 1)


@pytest.mark.parametrize("operation", tuple(_TYPES))
def test_casts_match_independent_source_reference_and_preserve_guards(tmp_path, operation):
    _, _, width, out_width, _ = _TYPES[operation]
    rng = random.Random(781)
    if operation == "bf16_to_f32":
        codes = [bits for bits in range(65536) if _non_nan(bits, width)]
        assert len(codes) == 65282  # Both infinities are qualified too.
    elif operation == "f32_to_bf16":
        # Every non-NaN upper half at the three sides of its BF16 tie.
        codes = [(high << 16) | low for high in range(65536) for low in (32767, 32768, 32769)
                 if _non_nan((high << 16) | low, width)]
        codes += [0, 0x80000000, 0x7F800000, 0xFF800000]
    else:
        codes = [rng.getrandbits(width * 8) for _ in range(4096)]
        if width == 8:
            # Exact ties around zero/min-subnormal, min-normal and overflow.
            for power in (-151, -150, -149, -127, -126, 127, 128):
                fractions = (0, 1, (1 << 28), (1 << 29), (1 << 52) - 1,
                             *((1 << 52) - (1 << shift) + delta for shift in (28, 29) for delta in (-1, 0, 1)))
                positive = [((power + 1023) << 52) | fraction for fraction in fractions]
                codes += [*positive, *(bits | (1 << 63) for bits in positive)]
        codes += [0, 1, (1 << (8 * width - 1)), (1 << (8 * width - 1)) | 1]
        fraction = 23 if width == 4 else 52
        infinity = ((1 << (8 * width - 1 - fraction)) - 1) << fraction
        codes += [infinity, infinity | (1 << (8 * width - 1))]
        codes = [bits for bits in codes if _non_nan(bits, width)]
    source = request_for(operation, (1, len(codes)))
    record, library, function = compile_c(tmp_path, source)
    expected = evaluate_graph(source, {"x": TensorValue(source.nodes[0].type, tuple(codes))})[0].elements
    payload = b"".join(bits.to_bytes(width, "little") for bits in codes)
    wanted = b"".join(bits.to_bytes(out_width, "little") for bits in expected)
    storage = ctypes.create_string_buffer(b"\xcd" + payload + b"\xcd")
    output = ctypes.create_string_buffer(b"\xa7" * (len(wanted) + 2))
    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(storage) + 1)
    outputs = (ctypes.c_void_p * 1)(ctypes.addressof(output) + 1)
    try:
        for mode in range(4):
            assert library.set_rounding(mode) == 0
            assert function(None, inputs, outputs) == 0
            assert output.raw[:len(wanted) + 2] == b"\xa7" + wanted + b"\xa7"
            assert storage.raw[:len(payload) + 2] == b"\xcd" + payload + b"\xcd"
    finally:
        assert library.set_rounding(0) == 0
    assert record["arithmetic"] == "exact-source-float-conversion"
    # A late NaN must fail the entire call without changing even an early output.
    fraction = {2: 7, 4: 23, 8: 52}[width]
    nan = (((1 << (8 * width - 1 - fraction)) - 1) << fraction) | 1
    ctypes.memmove(inputs[0] + len(payload) - width, nan.to_bytes(width, "little"), width)
    before = output.raw
    assert function(None, inputs, outputs) == -2
    assert output.raw == before
    assert function(None, inputs, inputs) == -1
    assert function(None, None, outputs) == -1


@pytest.mark.parametrize("operation", tuple(_TYPES))
def test_cast_refuses_changed_semantics_and_nan_constants(operation):
    source = request_for(operation)
    changed = replace(source.nodes[-1], attrs=())
    with pytest.raises(ValueError, match="rounding"):
        emit_float_cast_c(replace(source, nodes=(source.nodes[0], changed)), "cast")
    _, _, width, _, _ = _TYPES[operation]
    fraction = {2: 7, 4: 23, 8: 52}[width]
    nan = (((1 << (8 * width - 1 - fraction)) - 1) << fraction) | 1
    constant = replace(source.nodes[0], op="constant", effect="constant")
    binding = ConstantBinding("x", f"{constant.type.dtype}-le-raw", (nan.to_bytes(width, "little") * 6).hex())
    source = replace(source, nodes=(constant, source.nodes[-1]), input_storages=(), constants=(binding,))
    with pytest.raises(ValueError, match="NaN"):
        emit_float_cast_c(source, "cast")
