"""Compiled mask expansion preserves affine indexing and rejects invalid rows."""

import ctypes
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.compile.c_softmax_mask import render_masked_softmax_inputs
from merlin.semantic_compiler.model import IndexMap
from merlin.semantic_compiler.precision_values import FloatBoundarySpec


@pytest.mark.parametrize('sentinel', [None, 0xff7f0000])
def test_all_bf16_score_codes_match_exact_widening_then_encoding(tmp_path, sentinel):
    region = SimpleNamespace(shape=(1, 1), mask_map=IndexMap(2, ((1, 0), (0, 1)), (0, 0)))
    spec = FloatBoundarySpec('test', 'bf16', 'encode',
        admitted_code_ranges=((0, 0), (0x0400, 0x4300), (0x8000, 0x8000), (0x8400, 0xc300)))
    helpers = ''.join(render_masked_softmax_inputs(region, (1, 1), spec, b'\x00\xc2', dtype,
        finite_sentinel_bits=sentinel, score_dtype=dtype) for dtype in ('f32', 'bf16'))
    (tmp_path/'codes.c').write_text(helpers + '''
int check_all(void) {
  float mask=0.0f;
  for (uint32_t code=0;code<65536u;++code) {
    uint32_t bits=code<<16; float wide; __builtin_memcpy(&wide,&bits,4u);
    uint8_t raw[2]={(uint8_t)code,(uint8_t)(code>>8)};
    uint8_t a[2]={0xa5,0xa5},b[2]={0xa5,0xa5},ka[2]={0xcd,0xcd},kb[2]={0xcd,0xcd};
    int old=f32_prepare(&wide,&mask,a,ka),now=bf16_prepare(raw,&mask,b,kb);
    if ((!old)!=(!now) || a[0]!=b[0] || a[1]!=b[1] || ka[0]!=kb[0] || ka[1]!=kb[1]) return 1;
    if (now && (b[0]!=0xa5 || b[1]!=0xa5 || kb[0]!=0xcd || kb[1]!=0xcd)) return 2;
  }
  return 0;
}
''')
    subprocess.run([shutil.which('cc'), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
        '-shared', '-fPIC', 'codes.c', '-o', 'codes.so'], cwd=tmp_path, check=True, capture_output=True)
    library = ctypes.CDLL(str(tmp_path/'codes.so'))
    assert library.check_all() == 0


@pytest.mark.parametrize("score_dtype", ["f32", "bf16"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("finite_sentinel_bits", [None, 0xff7f0000])
def test_mask_packing_checks_all_values_before_writing(tmp_path, reverse, finite_sentinel_bits, score_dtype):
    shape = (2, 3, 5)
    mapping = IndexMap(3, ((0, 1, 0), (0, 0, -1 if reverse else 1)), (0, 4 if reverse else 0))
    region = SimpleNamespace(shape=shape, mask_map=mapping)
    spec = FloatBoundarySpec("test", "bf16", "encode", admitted_code_ranges=((0, 0), (0x3E80, 0x4200)))
    helper = render_masked_softmax_inputs(region, (3, 5), spec, b"\x00\xc2", "mask",
                                          finite_sentinel_bits=finite_sentinel_bits, score_dtype=score_dtype)
    (tmp_path / "mask.c").write_text(
        helper
        + "\nint prepare(const void *a,const float *m,unsigned char *o,unsigned char *k) {\n"
        + "  return mask_prepare(a,m,o,k);\n}\n"
    )
    library_name = f"mask-{int(reverse)}-{finite_sentinel_bits}.so"
    compiler = shutil.which("cc")
    assert compiler
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "mask.c", "-o", library_name],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(tmp_path / library_name))
    library.prepare.argtypes = [ctypes.c_void_p] * 4
    scores = np.arange(30, dtype="<f4").reshape(shape) / 4
    mask = np.array(
        [[0.0, -np.inf, 0.0, -np.inf, 0.0], [-np.inf, -np.inf, 0.0, 0.0, -np.inf], [0.0, 0.0, -np.inf, 0.0, -np.inf]],
        dtype="<f4",
    )
    exclusion = -np.inf
    if finite_sentinel_bits is not None:
        exclusion = np.array(finite_sentinel_bits, dtype="<u4").view("<f4").item()
        mask[np.isneginf(mask)] = exclusion
        # These source arithmetic facts justify this explicit keep-mask route.
        edges = np.array([-32, 0, 32], dtype="<f4")
        assert np.all((edges + np.float32(exclusion)).view("<u4") == finite_sentinel_bits)
        assert np.all(np.exp(np.float32(exclusion) - edges, dtype=np.float32) == 0)
    encoded, keep = ctypes.create_string_buffer(60), ctypes.create_string_buffer(60)

    def call(a, m):
        if score_dtype == "bf16":
            a = (a.view("<u4") >> 16).astype("<u2")
        return library.prepare(a.ctypes.data, m.ctypes.data, encoded, keep)

    assert call(scores, mask) == 0
    selected = mask[:, ::-1] if reverse else mask
    active = np.broadcast_to(selected == 0, shape)
    bits = scores.view("<u4")
    expected = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
    expected[~active] = 0xC200
    assert encoded.raw == expected.tobytes()
    assert keep.raw == np.where(active, 0x3F80, 0).astype("<u2").tobytes()
    for fault in ("last_mask", "all_masked", "nan_score", "domain_score", "unselected_exclusion"):
        a, m = scores.copy(), mask.copy()
        if fault == "last_mask":
            m[-1, -1] = -1.0
        elif fault == "all_masked":
            m[-1] = exclusion
        elif fault == "nan_score":
            a.reshape(-1)[-1] = np.nan
        elif fault == "domain_score":
            a.reshape(-1)[-1] = 64.0
        else:
            m.reshape(-1)[-1] = (-np.inf if finite_sentinel_bits is not None else
                                 np.array(0xff7f0000, dtype="<u4").view("<f4").item())
        encoded.raw = b"\xa5" * len(encoded)
        keep.raw = b"\xcd" * len(keep)
        assert call(a, m) != 0
        assert encoded.raw == b"\xa5" * len(encoded)
        assert keep.raw == b"\xcd" * len(keep)
    with pytest.raises(ValueError, match="leaves its source"):
        render_masked_softmax_inputs(
            SimpleNamespace(shape=shape, mask_map=replace(mapping, offsets=(0, 5))), (3, 5), spec, b"\x00\xc2", "bad"
        )


@pytest.mark.parametrize("sentinel", [True, -1, 0, 0xff800000, 0xbf800000, 0x7f7f0000, 0xffc00000])
def test_finite_sentinel_requires_exact_addition_and_exp_underflow_proof(sentinel):
    region = SimpleNamespace(shape=(1, 3), mask_map=IndexMap(2, ((1, 0), (0, 1)), (0, 0)))
    spec = FloatBoundarySpec("test", "bf16", "encode", admitted_code_ranges=((0, 0), (0x3e80, 0x4200)))
    with pytest.raises(ValueError, match="sentinel"):
        render_masked_softmax_inputs(region, (1, 3), spec, b"\x00\xc2", "invalid",
                                      finite_sentinel_bits=sentinel)


@pytest.mark.parametrize("score_dtype", ["f32", "bf16"])
def test_guarded_boolean_mask_accepts_empty_rows_and_checks_canonical_bytes(tmp_path, score_dtype):
    from merlin.semantic_compiler.guarded_softmax_source import recognize_guarded_masked_row_softmax
    from merlin.tests.ir.test_guarded_softmax_source import guarded_graph

    region = recognize_guarded_masked_row_softmax(guarded_graph(), "guarded")
    spec = FloatBoundarySpec("test", "bf16", "encode", admitted_code_ranges=((0, 0), (0x3E80, 0x4200)))
    helper = render_masked_softmax_inputs(region, region.mask_shape, spec, b"\x00\xc2", "guarded", score_dtype=score_dtype)
    (tmp_path / "guarded.c").write_text(
        helper + "\nint prepare(const void *s,const void *m,"
        "unsigned char *o,unsigned char *k){return guarded_prepare(s,m,o,k);}\n"
    )
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
            "guarded.c",
            "-o",
            "guarded.so",
        ],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    library = ctypes.CDLL(str(tmp_path / "guarded.so"))
    library.prepare.argtypes = [ctypes.c_void_p] * 4
    scores = np.arange(30, dtype="<f4").reshape(2, 3, 5) / 4
    if score_dtype == "bf16":
        scores = (scores.view("<u4") >> 16).astype("<u2")
    mask = np.array([[1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 0, 0, 1, 0]], dtype="u1")
    encoded, keep = ctypes.create_string_buffer(60), ctypes.create_string_buffer(60)
    assert library.prepare(scores.ctypes.data, mask.ctypes.data, encoded, keep) == 0
    active = ~np.broadcast_to(mask.astype(bool), scores.shape)
    expected = scores.copy() if score_dtype == "bf16" else (scores.view("<u4") >> 16).astype("<u2")
    expected[~active] = 0xC200
    assert encoded.raw == expected.tobytes()
    assert keep.raw == np.where(active, 0x3F80, 0).astype("<u2").tobytes()
    for bad_mask in (True, False):
        a, m = scores.copy(), mask.copy()
        if bad_mask:
            m[-1, -1] = 2
        else:
            a[-1, -1, -1] = 0x7fc0 if score_dtype == "bf16" else np.nan
        encoded.raw, keep.raw = b"\xa5" * 60, b"\xcd" * 60
        assert library.prepare(a.ctypes.data, m.ctypes.data, encoded, keep) != 0
        assert encoded.raw == b"\xa5" * 60 and keep.raw == b"\xcd" * 60
