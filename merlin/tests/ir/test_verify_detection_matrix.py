"""The detection matrix: what each verification layer actually catches, and what it costs.

These tests assert the *properties the matrix must have* to be evidence, not the specific cells —
cells are data and are allowed to move. The properties are:

1. No layer flags the unmutated program (no false positives).
2. Every seeded fault is caught by at least one layer (no blind spot in the corpus).
3. The static layer catches at least one fault NO other layer catches (non-redundancy).

Property 3 is the one that justifies the layer existing. If it ever fails, the static layer has
become a cost optimization rather than a coverage gain, and the write-up must say so.
"""
from __future__ import annotations

import pytest

from merlin.verify import HAS_XDSL, HAS_Z3
from merlin.verify.tools import find_filecheck, find_mlir_tool

pytestmark = pytest.mark.skipif(
    not (HAS_XDSL and HAS_Z3 and find_filecheck() and find_mlir_tool("mlir-translate")),
    reason="needs the verify extra and the in-tree LLVM tools")


@pytest.fixture(scope="module")
def matrix():
    from merlin.verify.evaluate import run_matrix
    return run_matrix(m=2, k=2, n=2, reuse=2, timeout_ms=60_000)


def test_no_layer_flags_the_unmutated_program(matrix):
    """A layer that rejects correct output is worse than no layer."""
    bad = [d for d in matrix["false_positives"] if d["detected"]]
    assert not bad, f"false positives: {bad}"


def test_every_fault_is_caught_by_something(matrix):
    caught = {d["fault"] for d in matrix["detections"] if d["detected"]}
    names = {f["name"] for f in matrix["faults"]}
    missed = names - caught
    assert not missed, f"no layer detects: {sorted(missed)} — an uncovered defect class"


def test_an_abstention_is_never_recorded_as_a_clean_miss(matrix):
    """The package's first invariant, asserted on the record rather than trusted in the prose.

    A solver timeout and a layer that ran and found nothing are both ``detected: False``. If the
    record cannot tell them apart, a coverage figure drawn from it can turn three timeouts into a
    claim about the layer below. Measured 2026-09-04: at 16x16x16 the three numeric faults abstain,
    and before this field existed they were indistinguishable from misses.
    """
    from merlin.verify.evaluate import OUTCOMES

    assert matrix["schema"].endswith("/v2"), "a pre-v2 record cannot express an abstention"
    assert "timeout_ms" in matrix, "a detection count is uninterpretable without its solver bound"
    for d in matrix["detections"] + matrix["false_positives"]:
        assert d.get("outcome") in OUTCOMES, f"attempt has no explicit outcome: {d}"
        assert d["detected"] == (d["outcome"] == "detected"), f"outcome disagrees with detected: {d}"


def test_static_layer_catches_something_nothing_else_does(matrix):
    """Non-redundancy, in the direction that justifies the cheapest layer.

    Residency-lifetime defects (a leaked weight, an evict before the last use) do not change the
    computed values, so neither the SMT refinement check nor a numeric golden can see them. Only the
    structural layer can.
    """
    # A fault only counts as static-only when every OTHER layer actually decided. If a layer
    # abstained, the exclusivity is an artifact of its budget, and counting it would let the cheap
    # layer's headline be inflated by solver timeouts rather than by coverage.
    abstained_on = {d["fault"] for d in matrix["detections"] if d.get("outcome") == "abstained"}
    by_fault: dict[str, set[str]] = {}
    for d in matrix["detections"]:
        if d["detected"]:
            by_fault.setdefault(d["fault"], set()).add(d["layer"])
    only_static = [f for f, layers in by_fault.items()
                   if layers == {"static"} and f not in abstained_on]
    assert only_static, (
        "no fault is caught by the static layer alone — it has become a cost optimization "
        "rather than a coverage gain, and the write-up must say so")


def test_render_reports_unmeasured_layers(matrix):
    """The RTL tiers must be reported as not measured, never silently omitted."""
    from merlin.verify.evaluate import render

    text = render(matrix)
    assert "not measured" in text and "rtl" in text
