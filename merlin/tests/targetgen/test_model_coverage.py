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
    """family_supported + family_unsupported + unclassified == n_regions, for any target. A region that
    silently belonged to no bucket would make every fraction meaningless."""
    rep = MC.coverage_for(_regions(), target, model="synthetic")
    assert rep.n_regions == 4
    assert rep.family_supported + rep.family_unsupported + rep.unclassified == rep.n_regions


def test_unnamed_region_is_unclassified_not_assumed():
    """A region whose family cannot be resolved counts as unclassified — never folded into supported or
    unsupported. Folding it either way is how a coverage number stops being evidence."""
    rep = MC.coverage_for(_regions(), "radiance", model="synthetic")
    assert rep.unclassified == 1
    assert rep.unclassified_ops["generic"] == 1
    assert "generic" not in rep.by_family


def test_family_fraction_excludes_unclassified_from_its_denominator():
    """The primary fraction is over CLASSIFIED regions: a capture full of unnameable regions must not read
    as well covered."""
    rep = MC.coverage_for(_regions(), "radiance", model="synthetic")
    assert rep.family_supported + rep.family_unsupported == 3
    assert rep.family_fraction == pytest.approx(rep.family_supported / 3)
    assert rep.classified_fraction == pytest.approx(3 / 4)


def test_family_coverage_is_dtype_agnostic():
    """gemmini declares contraction (int8 only), so an fp32 contraction is family-SUPPORTED and separately
    precision-blocked. Conflating the two is what made a dtype wall look like a missing family."""
    rep = MC.coverage_for(_regions(), "gemmini", model="synthetic")
    assert rep.family_supported == 2            # both contractions, whatever their dtype
    assert rep.unsupported_families["softmax"] == 1
    assert rep.dtype_ok == 1                    # only the int8 one clears the precision gate
    assert rep.dtype_blocked == 1


def test_unexpressed_precision_is_never_counted_as_accepted():
    """is_eligible treats a None dtype as not-applicable and returns eligible, so a region whose precision
    the capture never expressed must be excluded from the precision numbers entirely — otherwise missing
    metadata manufactures coverage. With nothing judged, the fraction is None, not 0% and not 100%."""
    regions = (RegionDescriptor(source="matmul", op="matmul", family="contraction", in_dtype=None),)
    rep = MC.coverage_for(regions, "gemmini", model="synthetic")
    assert rep.family_supported == 1
    assert rep.precision_known == 0
    assert rep.dtype_ok == 0 and rep.dtype_blocked == 0
    assert rep.precision_fraction is None


def test_manifest_precision_join_and_unknown_formats_are_dropped(tmp_path):
    """Precision comes from the weights manifest, joined on the region's OWNING module (the weight name
    minus its trailing component). A dtype the registry does not know is dropped, not mapped to a guess."""
    import json

    manifest = tmp_path / "m.safetensors.manifest.json"
    manifest.write_text(json.dumps({
        "0": {"weight": "m.enc.layer0.weight", "dtype": "int8"},
        "1": {"weight": "m.enc.layer1.weight", "dtype": "float8_e4m3fn"},
        "2": {"weight": "m.enc.layer2.weight", "dtype": "complex128"},
        "3": {"nonsense": True},
    }))
    got = MC.weight_precisions(manifest)
    assert got == {"m.enc.layer0": "int8", "m.enc.layer1": "fp8_e4m3"}


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
