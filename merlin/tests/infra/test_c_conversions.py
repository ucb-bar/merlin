"""Generated source conversions preserve axis order and checked float encoding."""

import ctypes
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.compile.c_conversions import render_bf16_decode_c, render_f32_operand_c
from merlin.semantic_compiler.model import TensorType
from merlin.semantic_compiler.precision_intake import SourceValueRef
from merlin.semantic_compiler.precision_transform import BoundaryConversion
from merlin.semantic_compiler.precision_values import FloatBoundarySpec


def test_scaled_bf16_decoder_covers_every_code_and_supported_scale(tmp_path):
    compiler = shutil.which("cc")
    assert compiler, "C compiler required"
    # Enumerate all 65,536 codes at every exponent in the declared [-64,64]
    # domain, including normal/subnormal transitions, ties, signed zeros and NaNs.
    functions = [
        render_bf16_decode_c(
            FloatBoundarySpec("selected-result", "bf16", "decode", post_decode_scale_exponent=exponent),
            f"decode_{exponent + 64}",
        )
        for exponent in range(-64, 65)
    ]
    (tmp_path / "decode.c").write_text(
        "\n".join(functions) + "\n" + "void run(const uint16_t *in,uint32_t *out,unsigned scale) {\n"
        "for (unsigned i=0;i<65536;++i) { switch(scale) {\n"
        + "\n".join(f"case {index}: out[i]=decode_{index}(in[i]); break;" for index in range(129))
        + "\n} } }\n"
    )
    path = tmp_path / "scaled-decode.so"
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
            str(tmp_path / "decode.c"),
            "-o",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(path))
    library.run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    codes = np.arange(65536, dtype=np.uint16)
    bits = codes.astype(np.uint32) << 16
    finite = (bits & 0x7F800000) != 0x7F800000
    actual = np.empty(65536, dtype=np.uint32)
    for exponent in range(-64, 65):
        expected = bits.copy()
        with np.errstate(over="ignore", under="ignore"):
            expected[finite] = (
                np.ldexp(bits[finite].view(np.float32).astype(np.float64), exponent).astype(np.float32).view(np.uint32)
            )
        library.run(codes.ctypes.data, actual.ctypes.data, exponent + 64)
        assert np.array_equal(actual, expected), exponent
        spec = FloatBoundarySpec("selected-result", "bf16", "decode", post_decode_scale_exponent=exponent)
        assert np.array_equal(spec.decode(codes).view(np.uint32), expected), exponent


@pytest.mark.parametrize(
    "fields",
    [
        {"direction": "encode", "post_decode_scale_exponent": 1},
        {"direction": "decode", "post_decode_scale_exponent": True},
        {"direction": "decode", "post_decode_scale_exponent": -65},
        {"direction": "decode", "post_decode_scale_exponent": 65},
    ],
)
def test_result_scale_domain_is_explicit(fields):
    with pytest.raises(ValueError):
        FloatBoundarySpec("result", "bf16", **fields)


@pytest.mark.parametrize("axes,scale", [((0, 1), 0), ((1, 0), 2)])
def test_c_conversion_matches_logical_values_and_preserves_output_on_bad_input(tmp_path, axes, scale):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler unavailable")
    spec = FloatBoundarySpec(
        "selected-finite-conversion",
        "fp8_e4m3",
        "encode",
        max_normal_exponent_code=14,
        flush_below_normal=True,
        admitted_code_ranges=((0, 0), (8, 119), (136, 247)),
        canonicalize_negative_zero=True,
        pre_encode_scale_exponent=scale,
    )
    shape = (3, 7)
    conversion = BoundaryConversion(
        "operand",
        SourceValueRef("argument", 0),
        TensorType(shape, "f32", "raw-bit-pattern"),
        TensorType(tuple(shape[axis] for axis in axes), "fp8_e4m3", "raw-bit-pattern"),
        axes,
        spec.policy,
        spec,
    )
    header = (merlin_dir() / "runtime/c/finite_float_boundary.h").read_text()
    source = (
        header
        + render_f32_operand_c(conversion, "encode")
        + """
int run(const float *source, uint8_t *out) { return encode(source,out); }
"""
    )
    (tmp_path / "convert.c").write_text(source)
    path = tmp_path / f"convert-{axes[0]}-{scale}.so"
    subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "convert.c", "-o", str(path)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(path))
    library.run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    values = (np.arange(21, dtype=np.float32).reshape(shape) - 10) / 8
    values[0, 0] = -0.0
    output = ctypes.create_string_buffer(21)
    assert library.run(values.ctypes.data, output) == 0
    assert output.raw == conversion.convert_values(values).tobytes()
    for invalid in (np.nan, np.inf, -np.inf):
        values[-1, -1] = invalid
        ctypes.memset(output, 0xA5, 21)
        assert library.run(values.ctypes.data, output) != 0
        assert output.raw == bytes([0xA5]) * 21


@pytest.mark.parametrize("restricted,flush", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("axes", [(0, 1), (1, 0)])
def test_c_bf16_rounding_boundaries_and_declared_code_domain(tmp_path, restricted, flush, axes):
    compiler = shutil.which("cc")
    assert compiler, "C compiler required"
    spec = FloatBoundarySpec(
        "selected-bf16-cast",
        "bf16",
        "encode",
        min_normal_exponent_code=8 if flush else 1,
        max_normal_exponent_code=132 if flush else 254,
        flush_below_normal=flush,
        admitted_code_ranges=((0, 0), (0x0400, 0x4200), (0x8000, 0x8000), (0x8400, 0xC200)) if restricted else (),
    )
    # Every finite BF16 anchor, with binary32 values on both sides of its RNE
    # midpoint. This also covers signed zeros, subnormals and saturation.
    anchors = np.arange(65536, dtype=np.uint32)
    anchors = anchors[((anchors >> 7) & 255) != 255]
    bits = ((anchors[:, None] << 16) | np.array([0, 0x7FFF, 0x8000, 0x8001], dtype=np.uint32)).ravel()
    values = bits.view(np.float32)
    expected = replace(spec, admitted_code_ranges=()).encode(values)
    if restricted:
        admitted = np.zeros(len(expected), dtype=bool)
        for low, high in spec.admitted_code_ranges:
            admitted |= (expected >= low) & (expected <= high)
        values, expected = values[admitted], expected[admitted]
    shape = (2, len(values) // 2)
    conversion = BoundaryConversion(
        "operand",
        SourceValueRef("argument", 0),
        TensorType(shape, "f32", "raw-bit-pattern"),
        TensorType(tuple(shape[axis] for axis in axes), "bf16", "raw-bit-pattern"),
        axes,
        spec.policy,
        spec,
    )
    (tmp_path / "bf16.c").write_text(
        render_f32_operand_c(conversion, "encode")
        + """
int run(const float *in, uint8_t *out) { return encode(in,out); }
"""
    )
    library_path = tmp_path / f"bf16-{axes[0]}-{int(restricted)}-{int(flush)}.so"
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
            "bf16.c",
            "-o",
            str(library_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(library_path))
    library.run.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    output = ctypes.create_string_buffer(len(values) * 2)
    assert library.run(values.ctypes.data, output) == 0
    assert output.raw == expected.reshape(shape).transpose(axes).tobytes()
    for bad in (np.nan, np.inf, -np.inf, *([1e-37, 64.0] if restricted else [])):
        before = output.raw
        invalid = values.copy()
        invalid[-1] = bad
        assert library.run(invalid.ctypes.data, output) != 0
        assert output.raw == before
