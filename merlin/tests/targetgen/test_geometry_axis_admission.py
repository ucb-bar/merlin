"""The aspect-ratio axis: a member must land in the class it is named for, and a class with no
member must SAY SO.

The geometry axis exists because a `(family, dtype, alignment)` cell cannot state an aspect ratio, and
a corpus that is entirely square proves nothing about the shapes real models present. Two things have
to hold for it to mean anything, and neither was pinned:

* **the label must be derived, not asserted.** A member called ``..._squareish_gemm`` whose operands
  classify as something else is worse than no member: the requirement reads as covered and the grade
  measures a different geometry. So the class is re-derived here from the capsule's OWN declared
  operands (:mod:`merlin.perf.member_geometry`), never from the requirement row that produced it.
* **a class that got no member must be named as unsynthesizable.** The provenance list's whole
  contract is that "a class silently absent here reads downstream as a geometry the corpus covers" --
  and the synthesizer had two bare ``continue`` s that left no trace. Measured on this repo's gemmini
  census: the residual ``unknown`` class carries 36 regions and 4.6% of all contraction MAC work,
  more than two of the classes that DID get a capsule, and nothing recorded that it was dropped.

Nothing here names a target: the targets come from the conformance specs on disk and every fact comes
from the spec that target derived.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import merlin_dir


def _targets() -> list[str]:
    spec_dir = merlin_dir() / "contract" / "capsules" / "conformance"
    return sorted(p.stem for p in spec_dir.glob("*.yaml")) if spec_dir.is_dir() else []


def _geometry_members(target: str) -> list[tuple[str, dict]]:
    """``(class named in the member, capsule doc)`` for every minted geometry member of ``target``."""
    from merlin.targetgen.corpora import graded_capsule_roots
    from merlin.targetgen.corpus_synth import SYNTH_PREFIX

    prefix = f"{SYNTH_PREFIX}_geometry_"
    out: list[tuple[str, dict]] = []
    for root in graded_capsule_roots(target):
        for path in sorted(root.glob("*/capsule.yaml")):
            if not path.parent.name.startswith(prefix):
                continue
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            out.append((path.parent.name[len(prefix):], doc))
    return out


def _spec(target: str) -> dict:
    from merlin.verify.lattice import load_spec

    return load_spec(target)


@pytest.mark.parametrize("target", _targets())
def test_a_minted_geometry_member_lands_in_the_class_it_is_named_for(target):
    """The name is a claim about aspect ratio; the operands are the evidence for it.

    Derived from the capsule rather than from the requirement entry on purpose -- two derivations of
    one quantity eventually disagree, and this is the one the grader will actually run.
    """
    from merlin.perf.member_geometry import stamp_for

    members = _geometry_members(target)
    if not members:
        pytest.skip(f"{target} mints no geometry members")
    for klass, doc in members:
        stamp = stamp_for(doc, target=target)
        assert stamp is not None, f"{doc.get('name')}: its own operands declare no readable (M, K, N)"
        assert stamp["geometry_class"] == klass, (
            f"{doc.get('name')} is named for {klass!r} but its operands "
            f"({stamp['M']}x{stamp['K']}x{stamp['N']}) classify as {stamp['geometry_class']!r}")
        assert stamp["in_census"] is True, (
            f"{doc.get('name')} claims a class no capture presents; the axis exists to sample the "
            f"census, so an off-census member here is not a small member, it is the wrong one")


@pytest.mark.parametrize("target", _targets())
def test_the_bound_a_geometry_member_respects_is_its_largest_tensor(target):
    """⚠️ REGRESSION. Bounding the WRITTEN OUTPUT alone admitted an `11 x 2304` by `2304 x 23272`
    member: 255,992 output elements and a 53.6-MILLION-element weight, on which the generator held
    14.7 GB and produced nothing in 27 minutes. The golden has to materialize every operand, so the
    bound is the largest tensor -- and a skewed aspect ratio is exactly where the two diverge."""
    from merlin.perf.member_geometry import stamp_for
    from merlin.targetgen.conformance import _materialization_ceiling

    members = _geometry_members(target)
    if not members:
        pytest.skip(f"{target} mints no geometry members")
    ceiling = _materialization_ceiling(target)
    if not ceiling:
        pytest.skip(f"{target} derives no operand-store ceiling, so nothing here is scaled")
    for _klass, doc in members:
        stamp = stamp_for(doc, target=target)
        assert stamp is not None
        m, k, n = stamp["M"], stamp["K"], stamp["N"]
        largest = max(m * k, k * n, m * n)
        assert largest <= ceiling, (
            f"{doc.get('name')} carries a {largest}-element tensor against a {ceiling}-element "
            f"operand store; its output ({m * n}) is not the binding quantity")


# ------------------------------------------------------------- a class with no member must say so

def _minimal_spec(required: list[dict], **extra) -> dict:
    """The smallest conformance spec whose geometry axis runs: one contraction cell and a tile edge."""
    spec = {
        "target": "t",
        "cells": [{"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8",
                   "alignment": "aligned"}],
        "boundaries": {"tile_edge": 16,
                       "extent_probes": [{"boundary": "tile_edge", "edge": 16,
                                          "points": [1, 4, 8, 15, 16, 17, 32]}]},
        "shape_geometry": {"required": required},
    }
    spec.update(extra)
    return spec


def _synthesize(required: list[dict], **extra) -> dict:
    from merlin.targetgen import corpus_synth as CS

    return CS.synthesize(_minimal_spec(required, **extra),
                         workload_spec={"models": [], "precision_preference": ["int8"]})


def test_a_geometry_member_too_large_to_certify_rests_on_a_sibling_that_exists():
    """The tier follows the SIZE, and a capped member has to name a certified sibling that is in the
    corpus. Both halves matter: an uncapped member demands a certification nobody can run, and an
    `extends` pointing at nothing is an orphan claiming a functional guarantee no capsule provides.

    Asserted on the synthesizer rather than on the tracked corpus on purpose -- the corpus is a
    generated artifact whose freshness is a separate question, and pinning the RULE here means a
    regenerated corpus inherits it on every target at once."""
    from merlin.targetgen.corpus_synth import SYNTH_PREFIX

    out = _synthesize(
        [{"class": "squareish_gemm", "family": "contraction", "M": 256, "K": 768, "N": 192,
          "out_elements": 49152, "n_regions": 289, "mac_fraction": 0.094393}],
        cert_affordability={"max_elements": 875, "budget_s": 300.0,
                            "metric": "written_output_elements"},
        oracle_tiers=["L2", "L3"])
    names = {str(e.get("name")) for e in out["capsules"]}
    member = next(e for e in out["capsules"] if str(e.get("name", "")).endswith("_squareish_gemm"))
    assert member["M"] * member["N"] > 875, "the fixture has to be above the budget to test the cap"
    assert member.get("max_oracle_tier") == "L2", (
        f"a member writing {member['M'] * member['N']} elements against an affordable 875 must be "
        f"capped to the cheaper tier this target declares; it declares "
        f"{member.get('max_oracle_tier')!r}")
    sibling = str(member.get("extends") or "")
    assert sibling == f"{SYNTH_PREFIX}_contraction_i8_aligned" and sibling in names, (
        f"the capped member rests on {sibling!r}, which this synthesis did not emit: {sorted(names)}")
    assert "875" in member["source_reference"], (
        "the reason it was capped has to travel with the member, or its tier reads as a choice")


def test_a_member_inside_the_budget_is_not_capped():
    """The cap is a consequence of size, not a property of the axis. Capping every geometry member
    would quietly shrink the certified corpus -- which is the mirror-image failure and just as silent."""
    out = _synthesize(
        [{"class": "squareish_gemm", "family": "contraction", "M": 16, "K": 32, "N": 16,
          "out_elements": 256, "n_regions": 3, "mac_fraction": 0.01}],
        cert_affordability={"max_elements": 875, "budget_s": 300.0,
                            "metric": "written_output_elements"},
        oracle_tiers=["L2", "L3"])
    member = next(e for e in out["capsules"] if str(e.get("name", "")).endswith("_squareish_gemm"))
    assert member.get("max_oracle_tier") is None, (
        f"a 256-element member is inside the 875 the budget affords; capping it to "
        f"{member.get('max_oracle_tier')!r} would drop it out of the certified corpus for no reason")


def test_the_residual_class_is_reported_rather_than_dropped():
    """⚠️ REGRESSION. `unknown` is the taxonomy's fall-through, so it gets no member -- but it used to
    get no MENTION either, and the list it belongs in documents itself as the thing that stops a
    dropped class from reading as a covered one. On this repo's own gemmini census that silence hid
    4.6% of all contraction MAC work."""
    out = _synthesize([{"class": "unknown", "family": "contraction", "M": 256, "K": 192, "N": 768,
                        "out_elements": 196608, "n_regions": 36, "mac_fraction": 0.046123}])
    reported = out["provenance"]["geometry_classes_unsynthesizable"]
    named = [r for r in reported if r.startswith("unknown")]
    assert named, f"the residual class left no trace; reported: {reported}"
    assert "0.046123" in named[0] and "36" in named[0], (
        "the report has to carry the MASS, or a reader cannot tell a rounding error from a hole "
        f"bigger than the classes that did get members: {named[0]!r}")
    assert not [e for e in out["capsules"] if str(e.get("name", "")).endswith("_unknown")], (
        "a member named after the fall-through would claim a coverage the classifier never asserted")
    # NON-VACUITY. A list that named every class would satisfy the assertion above while saying
    # nothing, so a class that DID get a member must leave it empty.
    clean = _synthesize([{"class": "squareish_gemm", "family": "contraction", "M": 32, "K": 32,
                          "N": 32, "n_regions": 9, "mac_fraction": 0.1}])
    assert clean["provenance"]["geometry_classes_unsynthesizable"] == [], (
        "a class that got a member must not also be reported as unsynthesizable: "
        f"{clean['provenance']['geometry_classes_unsynthesizable']}")


def test_a_requirement_row_carrying_no_class_is_reported_rather_than_dropped():
    """The other bare `continue`. A malformed row is exactly the case where silence is most costly,
    because there is not even a name in the requirement to notice is missing."""
    out = _synthesize([{"family": "contraction", "M": 256, "K": 192, "N": 768, "n_regions": 4}])
    reported = out["provenance"]["geometry_classes_unsynthesizable"]
    assert any("carries no class" in r for r in reported), (
        f"an unnamed requirement row vanished without a trace; reported: {reported}")


def test_a_class_the_census_could_not_size_names_the_wall_that_stopped_it():
    """An unreachable class carries its reason INTO the corpus provenance, not just into the census.
    The two are read by different people: the census says what the models present, the provenance
    says what the corpus could do about it."""
    out = _synthesize([{"class": "wide_skinny", "family": "contraction", "M": None, "K": None,
                        "N": None, "n_regions": 2248, "mac_fraction": 0.821837,
                        "unreachable": "the heaviest shape carries a 589824000-element tensor"}])
    reported = out["provenance"]["geometry_classes_unsynthesizable"]
    named = [r for r in reported if r.startswith("wide_skinny")]
    assert named and "589824000-element tensor" in named[0], (
        f"the wall that stopped the class did not travel with the refusal: {reported}")
