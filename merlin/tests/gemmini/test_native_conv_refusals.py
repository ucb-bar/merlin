"""Every clause a convolution fails, not the first one.

A guard sequence returns on its first failing clause, so a census built from one counts what it
reached and is silent about what it did not. That silence has been acted on here before: a session
fixed the top clause, predicted admission, and got the clause underneath instead, and an earlier
"32 of 53 admitted" was an unsound collapse of a per-channel scale rather than an admission.
`perf.capability_refusal` carries the caveat in its payload as `first_refusal_only`.

The measured case these tests pin: on the recorded whole-model emission every one of the 53
convolutions reported ONE blocker and was failing THREE.
"""

from __future__ import annotations

import importlib
import json
from collections import Counter

import pytest

from merlin.common.paths import merlin_dir
from merlin.runtime.backends import base

#: The 53 CONV2D sites of the recorded whole-model emission -- attributes and operand shape/dtype
#: only, vendored from the artifact so this regression pin is unconditional. A test that skips when
#: an artifact tree is absent does not hold anything; this one has to run wherever the suite runs.
_SITES = merlin_dir() / "tests/data/recorded_conv_sites_resnet50.json"


@pytest.fixture(scope="module")
def lc():
    base.get_backend("gemmini")  # registers the out-of-tree package
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_loop_conv")


def _legal_conv():
    """An NHWC int8 3x3 stride-1 pad-1 convolution: nothing for a clause to catch."""
    attributes = {
        "kernel": [3, 3, 64, 64],
        "stride": [1, 1],
        "padding": [1, 1, 1, 1],
        "dilation": [1, 1],
        "layout": "nhwc",
        "output_dtype": "i8",
    }
    operands = {
        "ifm": {"shape": [1, 56, 56, 64], "dtype": "i8"},
        "weight": {"shape": [3, 3, 64, 64], "dtype": "i8"},
        "dst": {"shape": [1, 56, 56, 64], "dtype": "i8"},
    }
    return attributes, operands


def test_a_native_convolution_is_refused_by_nothing(lc):
    assert lc.native_conv_refusals(*_legal_conv(), operand_dtype="i8") == ()


def test_the_recorded_emission_fails_three_clauses_on_every_site(lc):
    """The measurement this exists for, re-run against the tracked predicate. The probe that produced
    it read a selector revision living only in an artifact tree; this pins the same answer to code."""
    doc = json.loads(_SITES.read_text())
    assert len(doc["sites"]) == 53, "the emission recorded 53 convolutions"
    depth, every = Counter(), Counter()
    for site in doc["sites"]:
        clauses = lc.native_conv_refusals(site["attributes"], site["operands"], operand_dtype="i8")
        depth[len(clauses)] += 1
        every.update(clauses)
    assert dict(depth) == {3: 53}, "every site fails three clauses, and none is admitted"
    assert every["loop_conv_store_is_narrow_only"] == 53
    assert every["native_layout_contract_not_satisfied"] == 53, "silent under a first-refusal census"
    assert every["buffer_shape_does_not_match_native_layout"] == 53, "silent under a first-refusal census"


def test_a_full_width_readout_and_a_foreign_layout_are_reported_together(lc):
    """The property a short-circuiting selector cannot have: two independent causes, both named."""
    attributes, operands = _legal_conv()
    attributes = {**attributes, "output_dtype": "i32", "layout": "nchw_streamed_row_im2col"}
    operands = {**operands, "dst": {"shape": [1, 64, 56, 56], "dtype": "i32"}}
    clauses = lc.native_conv_refusals(attributes, operands, operand_dtype="i8")
    assert "loop_conv_store_is_narrow_only" in clauses
    assert "native_layout_contract_not_satisfied" in clauses


def test_geometry_clauses_fire_independently(lc):
    attributes, operands = _legal_conv()
    assert "loop_conv_requires_uniform_2d_geometry" in lc.native_conv_refusals(
        {**attributes, "stride": [1, 2]}, operands, operand_dtype="i8"
    )
    assert "loop_conv_requires_uniform_padding" in lc.native_conv_refusals(
        {**attributes, "padding": [1, 1, 2, 1]}, operands, operand_dtype="i8"
    )
    assert "native_loop_conv_requires_0_le_padding_lt_kernel" in lc.native_conv_refusals(
        {**attributes, "padding": [3, 3, 3, 3]}, operands, operand_dtype="i8"
    )


class _Facet:
    def __init__(self, verdict):
        self._verdict = verdict

    def admits_granularity(self, granularity):
        return self._verdict


def test_a_granularity_the_readout_cannot_hold_is_refused(lc):
    attributes, operands = _legal_conv()
    clauses = lc.native_conv_refusals(
        attributes, operands, operand_dtype="i8", facet=_Facet(False), scale_granularity="column"
    )
    assert "native_loop_conv_scale_granularity_not_admitted" in clauses


def test_an_underivable_granularity_never_widens(lc):
    """The clause whose absence produced the unsound 32-of-53. Unknown is not admission."""
    attributes, operands = _legal_conv()
    clauses = lc.native_conv_refusals(
        attributes, operands, operand_dtype="i8", facet=_Facet(None), scale_granularity="column"
    )
    assert "native_loop_conv_scale_granularity_unknown" in clauses


def test_an_admitted_granularity_adds_nothing(lc):
    attributes, operands = _legal_conv()
    assert (
        lc.native_conv_refusals(
            attributes, operands, operand_dtype="i8", facet=_Facet(True), scale_granularity="tensor"
        )
        == ()
    )


def test_every_clause_returned_is_a_declared_one(lc):
    """Reason codes are data a census groups by; an undeclared one silently re-partitions a ledger."""
    attributes, operands = _legal_conv()
    broken = {**attributes, "output_dtype": "i32", "layout": "nchw", "stride": [1, 2], "padding": [9, 9, 9, 9]}
    for clause in lc.native_conv_refusals(broken, operands, operand_dtype="i8"):
        assert clause in lc.NATIVE_CONV_CLAUSES


def test_the_census_of_these_sites_reports_itself_complete(lc):
    """The two halves joined: a selector that returns every failing clause, and a census that reports
    the short-circuit caveat as a measured property. Read the SAME 53 sites both ways and the
    difference is the whole hazard -- one clause and "fix it" against three and "all of them".
    """
    from merlin.perf.capability_refusal import SELECTED, RefusalSite, census, unblocking_sequence

    doc = json.loads(_SITES.read_text())

    def rows(complete: bool):
        for i, site in enumerate(doc["sites"]):
            clauses = lc.native_conv_refusals(site["attributes"], site["operands"], operand_dtype="i8")
            if not clauses:
                yield RefusalSite(f"conv{i}", True, SELECTED)
            elif complete:
                yield RefusalSite(f"conv{i}", False, clauses[0], None, tuple(clauses))
            else:
                yield RefusalSite(f"conv{i}", False, clauses[0])

    partial = census("native_conv", rows(False))
    assert partial["first_refusal_only"] is True
    assert partial["clause_depth"] == {1: 53}
    assert len(partial["clauses"]) == 1, "a first-refusal census names ONE blocker for all 53"

    complete = census("native_conv", rows(True))
    assert complete["first_refusal_only"] is False
    assert complete["sites_with_complete_stack"] == 53
    assert complete["clause_depth"] == {3: 53}, "every site fails three, so the top clause admits none"
    assert {c["clause"]: c["sites"] for c in complete["clauses"]} == {
        "loop_conv_store_is_narrow_only": 53,
        "native_layout_contract_not_satisfied": 53,
        "buffer_shape_does_not_match_native_layout": 53,
    }
    steps = unblocking_sequence([complete])["steps"]
    assert steps and all(s["all_must_clear"] for s in steps), "nothing here is hidden under anything"
