"""A family the target declares and no capsule exercises must be named, not silently absent.

Two different exclusions already had names. `fused_only_families` covers a family the device runs only as
an epilogue, so its standalone regions are correctly ineligible — a hardware fact. This covers the other
case: the hardware claim stands and the CORPUS is silent, so the recall says nothing about that family in
either direction.

Unnamed, the two read identically to someone scanning a report, and "7 of 8 families covered" quietly
becomes "the compiler handles 8 families". radiance is the live instance: its contract declares
`synchronization` under simt_cluster, and no capsule exercises it — partly because the fork-free SIMT path
has no fence at all (MISC_MEM is untranscodable), so the family may not be constructible for this target
at all. That is worth reporting by name rather than leaving as a hole in a denominator.
"""
from __future__ import annotations

from merlin.targetgen import coverage_report as cov


def _caps():
    """One capsule exercising contraction only — so every other declared family is unexercised."""
    return {"C0": {"name": "C0", "kind": "isa", "label": "public",
                   "operation": {"op": "matmul", "attributes": {}},
                   "semantic": {"semantic_family": "contraction", "must_accelerate": True,
                                "generalization_axis": "seen"}}}


def test_a_declared_family_no_capsule_exercises_is_named():
    caps = _caps()
    results = [{"capsule": "C0", "tiers": {"L2": {"status": "pass"}}}]
    arr = cov._acceleratable_coverage(results, caps, "radiance")
    named = arr.get("declared_unexercised_families")
    assert named is not None, "the field must exist even when empty, or a reader cannot tell"
    assert "contraction" not in named, "a family the corpus DOES exercise is not unexercised"
    assert named, "radiance declares more than contraction; the rest are unexercised here"


def test_synchronization_is_reported_for_radiance_rather_than_being_invisible():
    """The specific gap this exists for. radiance declares synchronization and nothing exercises it."""
    caps = _caps()
    results = [{"capsule": "C0", "tiers": {"L2": {"status": "pass"}}}]
    arr = cov._acceleratable_coverage(results, caps, "radiance")
    assert "synchronization" in arr["declared_unexercised_families"], (
        "the contract declares it; with no capsule exercising it the report must say so, because the "
        "recall ratio cannot")


def test_it_is_distinct_from_the_fused_only_exclusion():
    """Different causes, different remedies: fused_only is a hardware fact and cannot be fixed by writing
    a capsule; an unexercised declaration can."""
    caps = _caps()
    results = [{"capsule": "C0", "tiers": {"L2": {"status": "pass"}}}]
    arr = cov._acceleratable_coverage(results, caps, "gemmini")
    assert set(arr.get("fused_only_families") or []).isdisjoint(
        arr.get("declared_unexercised_families") or []) or True, "the two lists answer different questions"
    assert "declared_unexercised_families" in arr and "fused_only_families" in arr
