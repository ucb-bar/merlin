"""The corpus-rigor gate: generated operands must make addressing / stride / transpose bugs VISIBLE.

An operand with duplicate rows, duplicate columns, or A==A^T hides whole bug classes (a wrong row stride
or a transposed load produces the same output). This pins that the format-derived synthesis is rigorous
for every shape/format, and that the checker catches degeneracy — so a corpus regeneration can never
silently weaken operands.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.corpus_operands import (
    derive_palette,
    e8m0_scale_codes,
    operand_values,
    rigor_findings,
    scale_rigor_findings,
)

_SHAPES = [(32, 32), (32, 64), (64, 32), (16, 16)]
_FORMATS = ["fp8_e4m3", "fp8_e5m2"]


@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("shape", _SHAPES)
def test_generated_operands_are_rigorous(shape, fmt):
    vals = operand_values(shape, fmt, salt=12345)
    assert len(vals) == shape[0] * shape[1]
    assert rigor_findings(vals, shape) == []            # distinct rows+cols, asymmetric, non-degenerate


@pytest.mark.parametrize("fmt", _FORMATS)
def test_palette_is_format_derived_ranged_and_signed(fmt):
    pal = derive_palette(fmt, 67)
    assert len(set(pal)) == 67                            # distinct
    assert any(v < 0 for v in pal) and any(v > 0 for v in pal)   # both signs
    assert max(pal) / min(v for v in pal if v > 0) > 8   # a genuine dynamic-range spread, not 11 tiny mags


def test_palettes_differ_between_formats():
    # the whole point: e4m3 and e5m2 are DIFFERENT -> their derived palettes are not the same set
    assert set(derive_palette("fp8_e4m3", 67)) != set(derive_palette("fp8_e5m2", 67))


def test_salt_changes_operands_but_keeps_them_rigorous():
    a = operand_values((32, 32), "fp8_e4m3", salt=1)
    b = operand_values((32, 32), "fp8_e4m3", salt=2)
    assert a != b                                        # per-capsule variation
    assert rigor_findings(a, (32, 32)) == [] and rigor_findings(b, (32, 32)) == []


def test_rigor_findings_flags_degeneracy():
    # a constant matrix, a row-identical matrix, and a symmetric matrix must each be flagged
    const = [1.0] * (4 * 4)
    assert rigor_findings(const, (4, 4))
    row_id = [1.0, 2.0, 3.0, 4.0] * 4                     # every row identical
    assert any("row" in f for f in rigor_findings(row_id, (4, 4)))
    sym = [1, 2, 3, 2, 5, 6, 3, 6, 9]                     # 3x3 symmetric (== its transpose)
    assert any("symmetric" in f for f in rigor_findings([float(x) for x in sym], (3, 3)))


# --- MX low-bit / SIMT formats: the small-alphabet + LUT-capped construction stays rigorous --------------
@pytest.mark.parametrize("fmt,shape,max_alpha", [
    ("fp4_e2m1", (32, 32), None),      # 14-value alphabet
    ("fp4_e2m1", (32, 64), None),
    ("fp6_e3m2", (32, 32), 16),        # LUT-capped to 16 values (the fp6 datapath limit)
    ("fp6_e3m2", (64, 32), 16),
    ("fp16", (32, 32), None),
    ("bf16", (32, 64), None),
])
def test_mx_and_simt_operands_are_rigorous(fmt, shape, max_alpha):
    vals = operand_values(shape, fmt, salt=99, max_alphabet=max_alpha, mag_cap=4.0)
    assert len(vals) == shape[0] * shape[1]
    assert rigor_findings(vals, shape) == []              # distinct rows+cols, asymmetric, non-degenerate


def test_fp6_lut_cap_keeps_alphabet_at_16():
    # the fp6 LUT holds only 16 codes -> the whole operand draws from <=16 distinct values, yet is rigorous
    vals = operand_values((32, 32), "fp6_e3m2", salt=1, max_alphabet=16, mag_cap=4.0)
    assert len({round(v, 7) for v in vals}) <= 16
    assert rigor_findings(vals, (32, 32)) == []


def test_e8m0_scale_stream_is_rigorous_and_bounded():
    for shape in [(1, 16), (1, 32), (2, 16)]:
        codes = e8m0_scale_codes(shape, salt=7)
        assert scale_rigor_findings(codes) == []          # non-constant, adjacent lanes differ
        flat = [c for row in codes for c in row]
        assert all(124 <= c <= 130 for c in flat)         # +/-3 around bias 127 -> no bf16 overflow


def test_scale_rigor_flags_constant_stream():
    assert scale_rigor_findings([[127] * 16])             # all-equal (inert) scale stream is flagged


# ------------------------------------------- a demand the alphabet cannot meet is not a finding

def test_a_single_row_operand_is_judged_against_what_its_alphabet_can_reach():
    """A `1 x N` operand's columns are single values, so at most `alphabet` of them can differ. Demanding
    N distinct columns is arithmetic, not rigor: no stimulus over that alphabet could pass. Measured on
    the corpus -- a `1 x 32` rmsnorm gamma over the shared 4-value stimulus was reported degenerate."""
    from merlin.runtime.tensor import Tensor

    vals = [float(v) for v in Tensor.deterministic("G", (1, 32), "f32").data]
    assert len({*vals}) > 1, "the fixture must not be constant, or this would pass for the wrong reason"
    assert rigor_findings(vals, (1, 32)) == []


def test_what_the_alphabet_cannot_reach_is_recorded_rather_than_dropped():
    """The limit is real even though it is not a finding: some column swaps in that gamma ARE invisible.
    Relaxing the gate without recording why would lose the fact, which is the failure mode the corpus
    rigor machinery exists to prevent."""
    from merlin.runtime.tensor import Tensor
    from merlin.targetgen.corpus_operands import rigor_limits

    vals = [float(v) for v in Tensor.deterministic("G", (1, 32), "f32").data]
    limits = rigor_limits(vals, (1, 32))
    assert limits and "4-value alphabet" in limits[0]
    # A wide operand with rows enough to carry distinct column tuples has no such limit.
    wide = [float(v) for v in Tensor.deterministic("A0", (16, 32), "i8").data]
    assert rigor_limits(wide, (16, 32)) == []


def test_an_operand_worse_than_its_alphabet_allows_is_still_flagged():
    """The relaxation must not become a licence. Two identical rows over an alphabet that could easily
    have made them differ is still a row-addressing blind spot."""
    dup = [0.0, 1.0, 2.0, 3.0] * 2          # 2x4, both rows identical
    assert any("row" in f for f in rigor_findings(dup, (2, 4)))
