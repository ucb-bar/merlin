"""Target-neutral geometry tests for affine im2col span lowering."""

from __future__ import annotations

from itertools import product
import random

import pytest

from mlir_oot.codegen.llvm_emit import _universal_valid_x_span
from mlir_oot.lowering.plan import LoweringDeclined


def _naive_pack(src, *, ci, hi, wi, kh, kw, wo, sh, sw, dh, dw, pt, pl, oy):
    return [
        (src[channel][iy][ix] if 0 <= iy < hi and 0 <= ix < wi else 0)
        for channel in range(ci)
        for ky in range(kh)
        for kx in range(kw)
        for ox in range(wo)
        for iy, ix in [(oy * sh + ky * dh - pt, ox * sw + kx * dw - pl)]
    ]


def _span_pack(src, *, ci, hi, wi, kh, kw, wo, sh, sw, dh, dw, pt, pl, oy):
    """Independent executable statement of the split used by code generation."""
    lo, upper = _universal_valid_x_span(
        wi=wi, kw=kw, wo=wo, stride_w=sw, dilation_w=dw, pad_left=pl)
    packed = []
    for channel in range(ci):
        for ky in range(kh):
            iy = oy * sh + ky * dh - pt
            for kx in range(kw):
                for start, end, guarded in ((0, lo, True), (lo, upper, False),
                                            (upper, wo, True)):
                    for ox in range(start, end):
                        ix = ox * sw + kx * dw - pl
                        valid = 0 <= iy < hi and (not guarded or 0 <= ix < wi)
                        packed.append(src[channel][iy][ix] if valid else 0)
    return packed


def test_universal_span_is_the_exact_common_valid_interval() -> None:
    checked = 0
    for wi, kw, wo, sw, dw, pl in product(
            range(1, 10), range(1, 6), range(1, 12), range(1, 4), range(1, 4), range(6)):
        lo, upper = _universal_valid_x_span(
            wi=wi, kw=kw, wo=wo, stride_w=sw, dilation_w=dw, pad_left=pl)
        brute = [ox for ox in range(wo)
                 if all(0 <= ox * sw + kx * dw - pl < wi for kx in range(kw))]
        assert brute == list(range(lo, upper))
        checked += 1
    assert checked == 26_730


def test_segmented_packer_matches_naive_for_irregular_geometry() -> None:
    rng = random.Random(0xAFF1E)
    checked = 0
    for _ in range(500):
        ci, hi, wi = rng.randint(1, 4), rng.randint(1, 9), rng.randint(1, 9)
        kh, kw = rng.randint(1, 5), rng.randint(1, 5)
        sh, sw = rng.randint(1, 3), rng.randint(1, 3)
        dh, dw = rng.randint(1, 3), rng.randint(1, 3)
        pt, pl = rng.randint(0, 5), rng.randint(0, 5)
        wo, oy = rng.randint(1, 11), rng.randint(0, 10)
        src = [[[rng.randint(-128, 127) for _ in range(wi)]
                for _ in range(hi)] for _ in range(ci)]
        attrs = dict(ci=ci, hi=hi, wi=wi, kh=kh, kw=kw, wo=wo,
                     sh=sh, sw=sw, dh=dh, dw=dw, pt=pt, pl=pl, oy=oy)
        assert _span_pack(src, **attrs) == _naive_pack(src, **attrs)
        checked += 1
    assert checked == 500


@pytest.mark.parametrize("changes", [
    {"wi": 0}, {"kw": 0}, {"wo": 0}, {"stride_w": 0},
    {"dilation_w": 0}, {"pad_left": -1},
])
def test_span_classifier_fails_closed_on_invalid_geometry(changes) -> None:
    attrs = dict(wi=8, kw=3, wo=8, stride_w=1, dilation_w=1, pad_left=1)
    with pytest.raises(LoweringDeclined):
        _universal_valid_x_span(**(attrs | changes))
