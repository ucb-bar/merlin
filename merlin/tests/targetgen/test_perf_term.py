"""The minimal performance term, and the UNKNOWN semantics that are its whole point.

The property under test is negative and unusual: an UNKNOWN quantity must be *impossible* to read as
a number. Most of these tests therefore assert that something raises. That is deliberate -- the
failure mode being designed out is a caller who never checks, writes ``x or 0``, and publishes a
fabricated zero into a result someone later cites.
"""
from __future__ import annotations

import copy
import pickle

import pytest

from merlin.dse_guidance.evidence import EVIDENCE_TYPES, confidence_for, weakest_evidence
from merlin.perf.term import (UNKNOWN, UNKNOWN_TOKEN, Bounds, PerformanceTerm, Provenance,
                              UnknownValueError, Validity, combine_kinds, is_unknown,
                              known_or_raise)


def _prov(kind: str = "measured") -> Provenance:
    return Provenance(kind=kind, evidence=("a_declared_source",))


def _validity() -> Validity:
    return Validity(validated_regime="the one run this value came from",
                    expected_error="exact", weak_regime="anything else",
                    escalate_when="a different shape")


def _term(value, **kw) -> PerformanceTerm:
    return PerformanceTerm(name="t", value=value, unit="cycles", provenance=_prov(),
                           validity=_validity(), **kw)


# ---------------------------------------------------------------------------------------------
# UNKNOWN cannot be read as 0.0
# ---------------------------------------------------------------------------------------------


def test_unknown_is_not_zero_and_is_not_none():
    assert UNKNOWN is not None
    assert UNKNOWN != 0
    assert UNKNOWN != 0.0
    assert UNKNOWN != ""
    assert UNKNOWN != None            # noqa: E711 -- the identity of the comparison IS the test
    assert 0.0 != UNKNOWN             # reflected comparison must agree
    assert UNKNOWN == UNKNOWN
    assert is_unknown(UNKNOWN)
    assert is_unknown(UNKNOWN_TOKEN)
    assert not is_unknown(0.0)


@pytest.mark.parametrize("read", [
    pytest.param(lambda v: float(v), id="float"),
    pytest.param(lambda v: int(v), id="int"),
    pytest.param(lambda v: bool(v), id="bool"),
    pytest.param(lambda v: v + 3, id="add"),
    pytest.param(lambda v: 3 + v, id="radd"),
    pytest.param(lambda v: v - 3, id="sub"),
    pytest.param(lambda v: 3 - v, id="rsub"),
    pytest.param(lambda v: v * 2, id="mul"),
    pytest.param(lambda v: v / 2, id="truediv"),
    pytest.param(lambda v: 2 / v, id="rtruediv"),
    pytest.param(lambda v: -v, id="neg"),
    pytest.param(lambda v: abs(v), id="abs"),
    pytest.param(lambda v: round(v), id="round"),
    pytest.param(lambda v: v < 1, id="lt"),
    pytest.param(lambda v: v > 1, id="gt"),
    pytest.param(lambda v: sum([v, 1]), id="sum"),
    pytest.param(lambda v: max(v, 1), id="max"),
    # The exact idioms the design forbids -- each of them would otherwise publish a zero.
    pytest.param(lambda v: v or 0, id="or_zero"),
    pytest.param(lambda v: float(v or 0), id="float_or_zero"),
    pytest.param(lambda v: 0 if not v else v, id="not_guard"),
])
def test_unknown_refuses_every_numeric_read(read):
    """No arithmetic path yields a number from UNKNOWN -- it raises instead."""
    with pytest.raises(UnknownValueError):
        read(UNKNOWN)


def test_unknown_error_is_a_type_error():
    """So callers that only catch TypeError still fail loudly rather than coercing."""
    assert issubclass(UnknownValueError, TypeError)


def test_unknown_term_value_is_never_zero_in_arithmetic():
    t = PerformanceTerm.unknown("overlap_cycles", "cycles", _prov("structural_bound"), _validity(),
                                "the instrument partitions, so it cannot see concurrency")
    assert t.is_unknown
    with pytest.raises(UnknownValueError):
        _ = t.value + 0
    with pytest.raises(UnknownValueError):
        _ = float(t.value or 0)
    with pytest.raises(UnknownValueError):
        known_or_raise(t.value, "overlap_cycles")
    # ... and the only way to get a number out is to say so explicitly.
    assert t.to_dict()["value"] == UNKNOWN_TOKEN


def test_unknown_is_a_singleton_through_pickle_and_copy():
    """`value is UNKNOWN` must survive serialization, or the check silently stops firing."""
    assert pickle.loads(pickle.dumps(UNKNOWN)) is UNKNOWN
    assert copy.deepcopy(UNKNOWN) is UNKNOWN
    assert copy.copy(UNKNOWN) is UNKNOWN


def test_unknown_needs_a_reason():
    with pytest.raises(ValueError, match="no reason"):
        _term(UNKNOWN)
    with pytest.raises(ValueError, match="unknown_reason"):
        _term(12.0, unknown_reason="but it is known")


# ---------------------------------------------------------------------------------------------
# the evidence vocabulary is the existing one, and round-trips losslessly
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", EVIDENCE_TYPES)
def test_evidence_tag_round_trips_losslessly(kind):
    """A dse_guidance evidence tag survives a term round-trip unchanged, weight included."""
    term = PerformanceTerm(name="busy_cycles", value=158, unit="cycles",
                           provenance=Provenance(kind=kind, evidence=("src_a", "src_b")),
                           validity=Validity(validated_regime="one tile", expected_error="exact",
                                             weak_regime="chained accumulation",
                                             escalate_when="more than one compute per drain"),
                           bounds=Bounds(62, 1024))
    back = PerformanceTerm.from_dict(term.to_dict())
    assert back == term
    assert back.provenance.kind == kind
    assert back.provenance.evidence == ("src_a", "src_b")
    assert back.confidence == confidence_for(kind)
    assert back.validity == term.validity
    assert back.bounds == term.bounds
    # And the weight is the shared table's, not a private copy.
    assert term.provenance.confidence == confidence_for(kind)


def test_unknown_term_round_trips_losslessly():
    term = PerformanceTerm.unknown("predicted_busy_cycles", "cycles", _prov("structural_bound"),
                                   _validity(), "outside the validated regime",
                                   bounds=Bounds(62, UNKNOWN))
    back = PerformanceTerm.from_dict(term.to_dict())
    assert back == term
    assert back.value is UNKNOWN
    assert back.bounds.upper is UNKNOWN
    assert back.unknown_reason == "outside the validated regime"


def test_combined_provenance_takes_the_weakest_kind():
    combined = Provenance.combined([_prov("measured"), _prov("structural_bound")])
    assert combined.kind == weakest_evidence(["measured", "structural_bound"]) == "structural_bound"
    assert combine_kinds(["measured", "assumed"]) == "assumed"


def test_unknown_evidence_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown evidence kind"):
        Provenance(kind="vibes", evidence=("x",))


def test_a_term_must_cite_something():
    with pytest.raises(ValueError, match="carries no evidence"):
        Provenance(kind="measured", evidence=())


# ---------------------------------------------------------------------------------------------
# the rest of the minimal shape
# ---------------------------------------------------------------------------------------------


def test_bool_is_not_a_number():
    """`isinstance(True, int)` is True in Python; a bool priced as 1 cycle is a fabricated value."""
    with pytest.raises(TypeError, match="not a bool"):
        _term(True)


def test_nan_is_rejected_in_favour_of_unknown():
    with pytest.raises(ValueError, match="NaN"):
        _term(float("nan"))


def test_a_term_needs_a_unit_and_a_regime():
    with pytest.raises(ValueError, match="no unit"):
        PerformanceTerm(name="t", value=1, unit="", provenance=_prov(), validity=_validity())
    with pytest.raises(ValueError, match="validated_regime is required"):
        Validity(validated_regime="  ")


def test_bounds_unknown_end_is_not_an_infinity():
    """An underived bound must not silently pass every comparison."""
    b = Bounds(lower=0, upper=UNKNOWN)
    assert b.upper is UNKNOWN
    assert not b.known
    assert b.contains(10**9) is True          # no upper bound was derived, so nothing is refuted
    assert b.contains(-1) is False
    assert b.contains(UNKNOWN) is None        # the question is unanswerable, not answered "yes"
    with pytest.raises(UnknownValueError):
        _ = b.upper > 5


def test_value_outside_its_own_bounds_is_rejected():
    with pytest.raises(ValueError, match="outside its own bounds"):
        _term(500, bounds=Bounds(0, 100))
    with pytest.raises(ValueError, match="exceeds"):
        Bounds(10, 1)


def test_confidence_defaults_from_provenance_and_is_range_checked():
    assert _term(1).confidence == confidence_for("measured")
    assert _term(1, confidence=0.25).confidence == 0.25
    with pytest.raises(ValueError, match="outside"):
        _term(1, confidence=1.5)


def test_validity_mirrors_the_fidelity_contract_shape():
    """The two model the same thing; the conversion must be a field lift, not a translation."""

    class _Contract:                                   # duck-typed stand-in for FidelityContract
        validated_regime = "WS matmul, M/N/K multiples of 16"
        expected_error = "bit-exact"
        weak_regime = "unsupported CISC path"
        escalate_when = "unsupported_command"

    v = Validity.from_fidelity_contract(_Contract())
    assert v.validated_regime == _Contract.validated_regime
    assert v.expected_error == _Contract.expected_error
    assert v.weak_regime == _Contract.weak_regime
    assert v.escalate_when == _Contract.escalate_when
    # The full five-lattice unification is deliberately NOT attempted; only the shape matches.
    assert set(v.to_dict()) == {"validated_regime", "expected_error", "weak_regime", "escalate_when"}
