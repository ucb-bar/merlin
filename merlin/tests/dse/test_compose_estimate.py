"""A composed band must bound, and must stay silent when it cannot separate.

The band exists because the cycle oracle costs ~217 simulated cycles a second here, so the shapes that
most need optimising are the ones it cannot reach. What may honestly be composed from measurements of
smaller siblings is a RANGE, never a point -- and a range is only worth exposing if measurements
actually land inside it, which is what ``validate_composed_bands.py`` measures on held-out workloads.
These tests pin the properties that make that measurement meaningful.
"""
from __future__ import annotations

import pytest

from merlin.perf import compose_estimate as CE

PEAK = 256.0


def _buffer(k: int = 32, opcode: str = "MATMUL_RESIDENT"):
    return {
        "abi_version": "0.1", "target": "t", "version": "0.1", "params": {},
        "tensors": {"A0": {"role": "input", "shape": [16, k], "dtype": "i8"},
                    "W": {"role": "weight", "shape": [k, 16], "dtype": "i8"},
                    "Y0": {"role": "output", "shape": [16, 16], "dtype": "i32"}},
        "commands": [{"opcode": "RES_PACK", "operands": {"src": "W", "dst": "h"}, "attributes": {}},
                     {"opcode": opcode, "operands": {"lhs": "A0", "rhs": "h", "dst": "Y0"},
                      "attributes": {}},
                     {"opcode": "COMMIT", "operands": {"src": "Y0", "dst": "Y0"}, "attributes": {}}],
        "outputs": ["Y0"],
    }


def _band(k: int = 32, rate: float = 8.0):
    return CE.band(_buffer(k), target="t", peak_macs_per_cycle=PEAK, slowest_macs_per_cycle=rate)


def test_the_floor_is_the_work_over_the_structural_peak():
    out = _band(k=32)
    assert out["status"] == CE.DERIVED
    assert out["lower"] == pytest.approx(16 * 32 * 16 / PEAK)


def test_the_ceiling_is_the_work_over_the_slowest_measured_rate():
    out = _band(k=32, rate=8.0)
    assert out["upper"] == pytest.approx(16 * 32 * 16 / 8.0)


def test_both_ends_scale_with_the_work_rather_than_the_command_count():
    """THE PROPERTY THE FIRST CEILING LACKED, and it is why it was refuted 0/25.

    A per-command model prices a HISTOGRAM: one MATMUL is one command whether it contracts over 32 or
    over 256, so its ceiling was near-constant across workloads whose true cost spanned 269..3877 and
    every measurement sat above it. Doubling the reduction depth must double both ends.
    """
    narrow, wide = _band(k=32), _band(k=64)
    assert wide["lower"] == pytest.approx(2 * narrow["lower"])
    assert wide["upper"] == pytest.approx(2 * narrow["upper"])


def test_a_ceiling_without_a_measured_rate_refuses_rather_than_inventing_one():
    out = CE.band(_buffer(), target="t", peak_macs_per_cycle=PEAK, slowest_macs_per_cycle=None)
    assert out["status"] == CE.UNAVAILABLE and out["upper"] is None
    assert "not invented here" in out["reason"]


def test_a_floor_without_a_peak_refuses():
    out = CE.band(_buffer(), target="t", peak_macs_per_cycle=None, slowest_macs_per_cycle=8.0)
    assert out["status"] == CE.UNAVAILABLE and out["lower"] is None


def test_uncounted_work_refuses_the_CEILING_though_a_floor_would_still_hold():
    """The two ends are asymmetric on purpose. Work the counter did not see makes the true cost LARGER,
    so a floor built on partial work is still a floor -- while a ceiling built on it is not a ceiling."""
    buffer = _buffer()
    buffer["commands"].append({"opcode": "NOT_AN_ABI_OPCODE", "operands": {}, "attributes": {}})
    out = CE.band(buffer, target="t", peak_macs_per_cycle=PEAK, slowest_macs_per_cycle=8.0)
    assert out["status"] == CE.UNAVAILABLE
    assert out["floor"]["status"] == CE.DERIVED, "the floor survives partial work"
    assert out["ceiling"]["status"] == CE.UNAVAILABLE
    assert "cannot bound the cost from above" in out["ceiling"]["reason"]


def test_an_empty_interval_refuses_instead_of_being_reported():
    """A ceiling below a floor means one of the two is wrong. Returned as-is, `compare` would happily
    declare two empty intervals disjoint and eliminate a candidate on nothing."""
    out = CE.band(_buffer(), target="t", peak_macs_per_cycle=1.0, slowest_macs_per_cycle=1000.0)
    assert out["status"] == CE.UNAVAILABLE and "interval is empty" in out["reason"]


# ------------------------------------------------------------------------ compare: eliminate or stay quiet

def test_disjoint_bands_eliminate_in_the_right_direction():
    baseline, candidate = _band(k=2048), _band(k=32)
    assert baseline["lower"] > candidate["upper"], "the fixture must actually be disjoint"
    out = CE.compare(baseline, candidate)
    assert out["verdict"] == CE.ELIMINATE and out["faster"] == "candidate"
    assert out["separation_cycles"] > 0

    flipped = CE.compare(candidate, baseline)
    assert flipped["verdict"] == CE.ELIMINATE and flipped["faster"] == "baseline"


def test_a_doubling_of_work_does_NOT_separate_and_that_is_the_honest_limit():
    """HOW MUCH DIFFERENCE THIS SCREEN NEEDS, stated as a test rather than left to be discovered.

    The band's width is set by the spread between the slowest and fastest measured rate within a
    compute class -- measured at 18.9x median on this machine, down from 95.7x before the rates were
    split by class. Two bands that wide overlap unless the WORK differs by more than that spread, so a
    2x change is invisible here and only a change of roughly an order of magnitude separates.

    That is a real limit and it bounds what this screen is for: it will not rank two tilings of one
    shape, and it will separate a kernel issuing thousands of redundant synchronisations from one
    issuing two. Pinning it stops the band from being quoted as though it could do the former.
    """
    assert CE.compare(_band(k=64), _band(k=32))["verdict"] == CE.UNKNOWN


def test_identical_programs_are_UNKNOWN_and_never_a_win():
    """The negative control. Two bands for the same program overlap completely, and a ranker that
    reported a winner there would be reading noise."""
    out = CE.compare(_band(k=32), _band(k=32))
    assert out["verdict"] == CE.UNKNOWN


def test_overlapping_bands_are_UNKNOWN_even_when_one_is_plainly_lower():
    """A band may ELIMINATE and may never certify. Overlapping intervals have not shown a difference,
    however suggestive their midpoints are -- reporting the more likely one is the point estimate this
    module exists to avoid."""
    narrow, slightly_wider = _band(k=32), _band(k=33)
    assert narrow["upper"] > slightly_wider["lower"], "the fixture must actually overlap"
    assert CE.compare(narrow, slightly_wider)["verdict"] == CE.UNKNOWN


def test_a_band_that_did_not_derive_never_eliminates():
    undecidable = CE.band(_buffer(), target="t", peak_macs_per_cycle=None, slowest_macs_per_cycle=None)
    assert CE.compare(undecidable, _band())["verdict"] == CE.UNKNOWN
    assert CE.compare(_band(), undecidable)["verdict"] == CE.UNKNOWN


# ------------------------------------------------------------------------------- the compute class

@pytest.mark.parametrize("opcode,expected", [
    ("CONV2D", "CONV2D"), ("MATMUL", "MATMUL"), ("MATMUL_RESIDENT", "MATMUL_RESIDENT"),
    ("BATCHED_MATMUL", "BATCHED_MATMUL"), ("ATTENTION_QK", "ATTENTION_QK"),
])
def test_the_compute_class_is_read_off_the_program(opcode, expected):
    assert CE.compute_class(_buffer(opcode=opcode)) == expected


def test_a_program_issuing_no_compute_has_no_class():
    buffer = _buffer()
    buffer["commands"] = [buffer["commands"][0]]
    assert CE.compute_class(buffer) is None


def test_containment_is_the_acceptance_question_and_says_none_when_undecided():
    derived = _band(k=32, rate=8.0)
    assert CE.contains(derived, derived["lower"]) is True
    assert CE.contains(derived, derived["upper"]) is True
    assert CE.contains(derived, derived["lower"] - 1) is False
    assert CE.contains(derived, derived["upper"] + 1) is False
    assert CE.contains({"status": CE.UNAVAILABLE}, 100) is None
