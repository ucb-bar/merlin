"""The two per-target DATAPATH declarations a float corpus carries, and the guards on them.

Both exist because a reference model that disagrees with the hardware about the machine is not a
reference: it fails a correct compiler forever, with a number nobody can act on.

  * ``subnormal_operand_flush`` — a compute unit that admits only NORMAL operands sees zero wherever an
    operand's exponent field is zero. Measured on atlas and confirmed in its multiplier's own source
    (``aZero := aExp === 0.U``); declared per target, never assumed, and the subnormal test is DERIVED
    from the format descriptor rather than a byte layout.
  * ``inapplicable_oracle_tiers`` — a tier that cannot corroborate a result is reported skipped/N/A with
    its reason, not failed every round. Guarded so it can never silence a REQUIRED oracle.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from fractions import Fraction

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import corpus_spec as CS

_GEN = merlin_dir() / "contract" / "capsules"
if str(_GEN) not in sys.path:
    sys.path.insert(0, str(_GEN))

import generate_corpus as GC  # noqa: E402


@dataclass(frozen=True)
class _Fmt:
    """The shape of a refmodel float format: only the two fields the decoder derives from."""
    exp_bits: int
    mant_bits: int


class _D:
    """Stand-in for the refmodel's dtypes module — the decoder only calls decode_float_exact."""

    def __init__(self):
        self.calls = []

    def decode_float_exact(self, raw, fmt):
        self.calls.append(int(raw))
        return Fraction(int(raw))            # a sentinel value, not real arithmetic


# --------------------------------------------------------------------------------------------------
# operand decode
# --------------------------------------------------------------------------------------------------

def test_exact_decode_is_the_default_and_delegates_untouched():
    d = _D()
    dec = GC._operand_decoder(d, _Fmt(exp_bits=4, mant_bits=3), flush_subnormals=False)
    # 0x06 is subnormal in e4m3 (exponent field 0, mantissa 6); with no declaration it decodes normally.
    assert dec(0x06) == Fraction(6)
    assert d.calls == [0x06]


def test_declared_flush_sends_every_zero_exponent_code_to_zero():
    d = _D()
    dec = GC._operand_decoder(d, _Fmt(exp_bits=4, mant_bits=3), flush_subnormals=True)
    for code in (0x00, 0x06, 0x07, 0x80, 0x86, 0x87):     # exponent field zero, both signs
        assert dec(code) == 0, f"code {code:#04x} should flush"
    assert d.calls == [], "a flushed code must not reach the refmodel decode at all"


def test_declared_flush_leaves_normal_codes_alone():
    d = _D()
    dec = GC._operand_decoder(d, _Fmt(exp_bits=4, mant_bits=3), flush_subnormals=True)
    assert dec(0x08) == Fraction(0x08)       # smallest NORMAL e4m3 (exponent field 1)
    assert dec(0x7F) == Fraction(0x7F)
    assert d.calls == [0x08, 0x7F]


def test_the_subnormal_boundary_is_derived_from_the_format_not_a_byte_layout():
    """Same code, different exp/mant split -> different verdict. A hardcoded e4m3 mask would not move."""
    d = _D()
    e4m3 = GC._operand_decoder(d, _Fmt(exp_bits=4, mant_bits=3), flush_subnormals=True)
    e5m2 = GC._operand_decoder(d, _Fmt(exp_bits=5, mant_bits=2), flush_subnormals=True)
    # 0x08 = 0b0000_1000: exponent field is 1 under e4m3 (normal) but 2 under e5m2 (also normal).
    # 0x03 = 0b0000_0011: exponent field 0 under both -> subnormal under both.
    assert e4m3(0x08) != 0 and e5m2(0x08) != 0
    assert e4m3(0x03) == 0 and e5m2(0x03) == 0
    # 0x04 = 0b0000_0100: e4m3 exponent field 0 (subnormal); e5m2 exponent field 1 (NORMAL).
    assert e4m3(0x04) == 0
    assert e5m2(0x04) != 0


def test_binding_defaults_to_no_flush():
    """A target that declares nothing gets the format's own exact decode."""
    assert CS.CorpusBinding.__dataclass_fields__["subnormal_operand_flush"].default is False


# --------------------------------------------------------------------------------------------------
# inapplicable oracle tiers
# --------------------------------------------------------------------------------------------------

def test_inapplicable_tiers_are_carried_with_their_reason():
    got = CS._inapplicable_tiers({"inapplicable_oracle_tiers": {"L2": "  the model is a different machine  "}},
                                 ["L0", "L3"])
    assert got == {"L2": "the model is a different machine"}


def test_no_declaration_means_every_tier_runs():
    assert CS._inapplicable_tiers({}, ["L0", "L3"]) == {}


def test_a_tier_cannot_be_switched_off_without_saying_why():
    with pytest.raises(ValueError, match="no reason"):
        CS._inapplicable_tiers({"inapplicable_oracle_tiers": {"L2": ""}}, ["L3"])


def test_a_required_tier_can_never_be_declared_inapplicable():
    """The guard that keeps this from becoming a way to silence a failing mandatory oracle."""
    with pytest.raises(ValueError, match="BOTH required and inapplicable"):
        CS._inapplicable_tiers({"inapplicable_oracle_tiers": {"L3": "inconvenient"}}, ["L0", "L3"])


def test_a_malformed_declaration_raises_rather_than_being_ignored():
    with pytest.raises(ValueError, match="mapping"):
        CS._inapplicable_tiers({"inapplicable_oracle_tiers": ["L2"]}, ["L3"])
