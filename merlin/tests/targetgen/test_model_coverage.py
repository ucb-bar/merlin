"""Model-level coverage accounting — the buckets must add up and must never guess.

Hermetic: builds regions directly (no captured model needed), so these run anywhere. The end-to-end sweep
over real captures is a measurement, not a test.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import model_coverage as MC
from merlin.targetgen.eligibility import RegionDescriptor


def _regions():
    return (
        RegionDescriptor(source="matmul", op="matmul", family="contraction", in_dtype="int8"),
        RegionDescriptor(source="matmul", op="matmul", family="contraction", in_dtype="fp32"),
        RegionDescriptor(source="softmax", op="softmax", family="softmax", in_dtype="fp32"),
        RegionDescriptor(source="generic", op="generic", family=None, in_dtype="fp32"),
    )


@pytest.mark.parametrize("target", ["radiance", "gemmini"])
def test_buckets_partition_every_region(target):
    """routed + fallback + unclassified == n_regions, for any target. A region that silently belonged to
    no bucket would make every fraction meaningless."""
    rep = MC.coverage_for(_regions(), target, model="synthetic")
    assert rep.n_regions == 4
    assert rep.routed + rep.fallback + rep.unclassified == rep.n_regions


def test_unnamed_region_is_unclassified_not_assumed():
    """A region whose family cannot be resolved counts as unclassified — never folded into routed or
    fallback. Folding it either way is how a coverage number stops being evidence."""
    rep = MC.coverage_for(_regions(), "radiance", model="synthetic")
    assert rep.unclassified == 1
    assert rep.unclassified_ops["generic"] == 1
    assert "generic" not in rep.by_family


def test_routed_fraction_excludes_unclassified_from_its_denominator():
    """The reported fraction is over CLASSIFIED regions: a capture full of unnameable regions must not
    read as well covered."""
    rep = MC.coverage_for(_regions(), "radiance", model="synthetic")
    assert rep.routed + rep.fallback == 3
    assert rep.routed_fraction == pytest.approx(rep.routed / 3)
    assert rep.classified_fraction == pytest.approx(3 / 4)


def test_dtype_outside_a_targets_declared_formats_falls_back():
    """gemmini declares int8 only, so an fp32 contraction is fallback, not routed — the dtype wall is real
    and must not be papered over by a None dtype defaulting to eligible."""
    rep = MC.coverage_for(_regions(), "gemmini", model="synthetic")
    assert rep.routed == 1                      # the int8 contraction only
    assert rep.fallback_families["contraction"] == 1
    assert rep.fallback_families["softmax"] == 1


def test_short_op_splits_on_the_dialect_separator():
    assert MC._short_op("linalg.matmul") == "matmul"
    assert MC._short_op("matmul") == "matmul"


def test_terminators_and_init_ops_are_not_counted_as_regions():
    """linalg.yield/index/init_tensor carry no computation to route; counting them inflates denominators."""
    class _Op:
        def __init__(self, name):
            self.name = name

    assert not MC._is_region_op(_Op("linalg.yield"))
    assert not MC._is_region_op(_Op("linalg.index"))
    assert not MC._is_region_op(_Op("arith.constant"))
    assert MC._is_region_op(_Op("linalg.matmul"))
    assert MC._is_region_op(_Op("linalg.generic"))


def test_a_unit_declaring_ops_but_no_semantic_block_still_has_capabilities():
    """A generated contract may describe a unit fully in ops/dtypes and omit ``semantic_capabilities``.
    That omission must not read as "accelerates nothing" — it made a systolic MXU score 0% routable on six
    real models. Families derive from the unit's OWN ops via the canonical router; an op with no known
    family contributes nothing (fail closed), and an explicit block always wins."""
    from merlin.targetgen import compute_units as cu

    derived = cu.compute_units({"compute_units": [
        {"kind": "systolic", "name": "u", "ops": ["matmul"], "dtypes": ["bf16"]}]})
    caps = {c.family: c for c in derived[0].semantic_capabilities}
    assert sorted(caps) == ["contraction"]
    assert caps["contraction"].dtypes == ("bf16",)

    unknown_op = cu.compute_units({"compute_units": [
        {"kind": "systolic", "name": "u", "ops": ["not_a_known_op"], "dtypes": ["bf16"]}]})
    assert unknown_op[0].semantic_capabilities == ()          # fail closed, never guessed

    no_ops = cu.compute_units({"compute_units": [
        {"kind": "systolic", "name": "u", "ops": [], "dtypes": ["bf16"]}]})
    assert no_ops[0].semantic_capabilities == ()

    explicit = cu.compute_units({"compute_units": [
        {"kind": "systolic", "name": "u", "ops": ["matmul"], "dtypes": ["bf16"],
         "semantic_capabilities": [{"family": "movement", "dtypes": ["bf16"]}]}]})
    assert [c.family for c in explicit[0].semantic_capabilities] == ["movement"]
