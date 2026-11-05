"""C host conversion agrees with the selected Python precision boundary."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.semantic_compiler.precision_codegen import finite_f32_codebook
from merlin.semantic_compiler.precision_values import FloatBoundarySpec

_HEADER_DIR = Path(__file__).resolve().parents[3] / "merlin/runtime/c"


def _library(tmp_path: Path, spec: FloatBoundarySpec):
    table = finite_f32_codebook(spec)
    codes = ", ".join(f"{code}u" for code in table.positive_codes)
    transitions = ", ".join(f"0x{bits:08x}u" for bits in table.midpoint_bits)
    mask = ", ".join(f"0x{word:08x}u" for word in table.admitted_mask)
    source = tmp_path / "boundary.c"
    source.write_text(
        '#include "finite_float_boundary.h"\n'
        f"static const uint8_t codes[] = {{{codes}}};\n"
        f"static const uint32_t transitions[] = {{{transitions}}};\n"
        "static const merlin_f32_u8_codebook table = {\n"
        f"  codes, transitions, {len(table.positive_codes)}u, "
        f"0x{table.zero_gate_bits:08x}u, {{{mask}}}, "
        f"{int(table.canonicalize_negative_zero)}u\n"
        "};\n"
        "int encode(uint32_t bits, uint8_t *out) {\n"
        "  return merlin_f32_u8_encode_bits(bits, &table, out);\n"
        "}\n"
        "int pack(const float *source, uint8_t *out, size_t count) {\n"
        "  return merlin_pack_f32_u8(source, out, count, &table);\n"
        "}\n"
        "int unpack(const volatile uint8_t *source, float *out, size_t rows,\n"
        "           size_t columns, size_t first_columns) {\n"
        "  return merlin_unpack_split_bf16_f32(source, out, rows, columns, first_columns);\n"
        "}\n"
    )
    output = tmp_path / "boundary.so"
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fPIC",
            "-shared",
            "-I",
            str(_HEADER_DIR),
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    library = ctypes.CDLL(str(output))
    library.encode.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8)]
    library.encode.restype = ctypes.c_int
    library.pack.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    library.pack.restype = ctypes.c_int
    library.unpack.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    library.unpack.restype = ctypes.c_int
    return library, table


def _pack(library, values: np.ndarray, *, fill: int = 0xA5) -> tuple[int, bytes]:
    source = np.ascontiguousarray(values, dtype=np.float32)
    output = (ctypes.c_uint8 * len(source))(*([fill] * len(source)))
    status = library.pack(source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), output, len(source))
    return status, bytes(output)


def test_exact_rne_grid_clipping_flush_and_signed_zero(tmp_path: Path) -> None:
    spec = FloatBoundarySpec("fixture", "fp8_e4m3", "encode", max_normal_exponent_code=14, flush_below_normal=True)
    library, table = _library(tmp_path, spec)
    boundaries = np.asarray(table.midpoint_bits, dtype=np.uint32).view(np.float32)
    cases = [
        np.float32(0),
        np.float32(-0.0),
        np.float32(2.0**-7),
        np.float32(-(2.0**-7)),
        np.float32(2.0**-6),
        np.float32(240),
        np.float32(-240),
        np.float32(1000000),
        np.float32(-1000000),
    ]
    for midpoint in boundaries:
        cases.extend(
            (
                np.nextafter(midpoint, np.float32(0)),
                midpoint,
                np.nextafter(midpoint, np.float32(np.inf)),
                -np.nextafter(midpoint, np.float32(0)),
                -midpoint,
                -np.nextafter(midpoint, np.float32(np.inf)),
            )
        )
    rng = np.random.default_rng(0xEA4C3)
    raw = rng.integers(0, 1 << 32, size=2048, dtype=np.uint32)
    raw = raw[(raw & 0x7F800000) != 0x7F800000]
    values = np.concatenate((np.asarray(cases, dtype=np.float32), raw.view(np.float32)))
    status, observed = _pack(library, values)
    assert status == 0
    assert observed == spec.encode(values).tobytes()
    assert observed[0] == 0 and observed[1] == 0x80
    assert len(table.positive_codes) == 112


def test_explicit_zero_canonicalization_matches_host_and_preserves_nonzero_sign(tmp_path: Path) -> None:
    base = FloatBoundarySpec(
        "finite-normal-preserve-zero", "fp8_e4m3", "encode",
        max_normal_exponent_code=14, flush_below_normal=True,
        admitted_code_ranges=((0, 0), (8, 119), (136, 247)),
    )
    selected = FloatBoundarySpec(
        "finite-normal-positive-zero", "fp8_e4m3", "encode",
        max_normal_exponent_code=14, flush_below_normal=True,
        admitted_code_ranges=base.admitted_code_ranges,
        canonicalize_negative_zero=True,
    )
    with pytest.raises(ValueError, match="outside the selected target input domain"):
        base.encode(np.asarray([-0.0, -2.0**-20], dtype=np.float32))
    values = np.asarray([0.0, -0.0, -2.0**-20, 2.0**-20, -1.0, 1.0], dtype=np.float32)
    expected = selected.encode(values).tobytes()
    assert expected == bytes((0, 0, 0, 0, 0xB8, 0x38))
    assert selected.record()["canonicalize_negative_zero"] is True
    assert "canonicalize_negative_zero" not in base.record()
    library, table = _library(tmp_path, selected)
    assert table.canonicalize_negative_zero is True
    assert _pack(library, values) == (0, expected)
    output = ctypes.c_uint8(0xA5)
    assert library.encode(0x80000000, ctypes.byref(output)) == 0
    assert output.value == 0
    invalid = np.asarray([1.0, np.inf], dtype=np.float32)
    assert _pack(library, invalid) == (1, b"\xa5\xa5")


@pytest.mark.parametrize("exponent", [-5, 5])
def test_power_of_two_scaled_boundary_matches_compiled_host_encoder(tmp_path: Path, exponent: int) -> None:
    spec = FloatBoundarySpec(
        "scaled-fixture", "fp8_e4m3", "encode", max_normal_exponent_code=14,
        flush_below_normal=True, pre_encode_scale_exponent=exponent,
    )
    variant = tmp_path / f"scale-{exponent}"
    variant.mkdir()
    library, table = _library(variant, spec)
    thresholds = np.asarray(table.midpoint_bits, dtype=np.uint32).view(np.float32)
    cases = [np.float32(0), np.float32(-0.0), np.float32(0.5), np.float32(-0.5)]
    for midpoint in thresholds:
        cases.extend((np.nextafter(midpoint, np.float32(0)), midpoint,
                      np.nextafter(midpoint, np.float32(np.inf))))
    rng = np.random.default_rng(0x5CA1E + exponent)
    raw = rng.integers(0, 1 << 32, size=2048, dtype=np.uint32)
    raw = raw[(raw & 0x7F800000) != 0x7F800000]
    values = np.concatenate((np.asarray(cases, dtype=np.float32), raw.view(np.float32)))
    status, observed = _pack(library, values)
    assert status == 0
    assert observed == spec.encode(values).tobytes()


def test_domain_and_nonfinite_fail_without_partial_output(tmp_path: Path) -> None:
    spec = FloatBoundarySpec(
        "restricted",
        "fp8_e4m3",
        "encode",
        max_normal_exponent_code=14,
        flush_below_normal=True,
        admitted_code_ranges=((0, 0), (56, 56), (64, 64), (184, 184), (192, 192)),
    )
    library, _ = _library(tmp_path, spec)
    accepted = np.array([0, 1, 2, -1, -2], dtype=np.float32)
    assert _pack(library, accepted) == (0, spec.encode(accepted).tobytes())
    for source, expected_status in (
        ([0, 1, 0.5], 2),
        ([0, -0.0], 2),
        ([0, float("nan")], 1),
        ([0, float("inf")], 1),
    ):
        with np.errstate(invalid="ignore"):
            status, output = _pack(library, np.array(source, dtype=np.float32))
        assert status == expected_status
        assert output == b"\xa5" * len(source)
    out = ctypes.c_uint8(0xA5)
    assert library.encode(0x3F000000, ctypes.byref(out)) == 2
    assert out.value == 0xA5


def test_second_byte_format_uses_the_same_host_encoder(tmp_path: Path) -> None:
    spec = FloatBoundarySpec("e5m2-fixture", "fp8_e5m2", "encode", max_normal_exponent_code=30, flush_below_normal=True)
    library, table = _library(tmp_path, spec)
    rng = np.random.default_rng(0xE5A2)
    raw = rng.integers(0, 1 << 32, size=1024, dtype=np.uint32)
    values = raw[(raw & 0x7F800000) != 0x7F800000].view(np.float32)
    status, observed = _pack(library, values)
    assert status == 0
    assert observed == spec.encode(values).tobytes()
    assert len(table.positive_codes) == 120


def test_codebook_refuses_unimplemented_boundary(tmp_path: Path) -> None:
    spec = FloatBoundarySpec("no-flush", "fp8_e4m3", "encode")
    try:
        finite_f32_codebook(spec)
    except ValueError as exc:
        assert "flushing eight-bit" in str(exc)
    else:
        raise AssertionError("a different subnormal policy needs its own C lowering")


def test_split_bf16_readback_preserves_all_bits_and_rejects_overlap(tmp_path: Path) -> None:
    spec = FloatBoundarySpec("fixture", "fp8_e4m3", "encode", flush_below_normal=True)
    library, _ = _library(tmp_path, spec)
    rng = np.random.default_rng(0xBF16)
    for rows, columns, first_columns in ((3, 5, 2), (32, 32, 16), (256, 256, 128)):
        codes = (
            np.arange(1 << 16, dtype=np.uint16).reshape(rows, columns)
            if rows == 256
            else rng.integers(0, 1 << 16, size=(rows, columns), dtype=np.uint16)
        )
        if rows != 256:
            codes[0, :4] = [0, 0x8000, 0x7F80, 0x7FC1]
        physical = np.concatenate((codes[:, :first_columns].ravel(), codes[:, first_columns:].ravel())).astype("<u2")
        source = (ctypes.c_uint8 * physical.nbytes).from_buffer_copy(physical.tobytes())
        original = bytes(source)
        output = (ctypes.c_float * codes.size)()
        assert library.unpack(source, output, rows, columns, first_columns) == 0
        expected = (codes.astype("<u4") << 16).astype("<u4").tobytes()
        assert ctypes.string_at(ctypes.addressof(output), len(expected)) == expected
        assert bytes(source) == original
        assert library.unpack(source, output, rows, columns, columns) == 1
        assert (
            library.unpack(source, ctypes.cast(source, ctypes.POINTER(ctypes.c_float)), rows, columns, first_columns)
            == 1
        )
