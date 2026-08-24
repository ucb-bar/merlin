"""The capability auditor's SHAPE-AXIS findings — the half of an under-declaration nobody could see.

The evidence ladder decides FAMILIES ("can this target contract at all"), never the ranks, dtypes or
layouts it can contract OVER. That asymmetry is the module's own documented limit, and it has already
cost a real measurement: gemmini declared ``contraction ranks: [2]`` while its funct table carried a
conv loop nest, so every shipped rank-4 conv2d capsule scored INELIGIBLE and quietly left the ARR
denominator -- flattering recall by exactly the regions the mesh handles best. A human caught it; no
rung could have, and nothing in ``reconcile`` said a word.

These tests pin the two things ``reconcile`` now says instead of staying silent: an axis the evidence
shows and the contract omits (``missing_axis``), and an axis the contract claims that no rung could
confirm (``unaudited_axis``). The second is not an error -- it is the review obligation, written down.
"""
from __future__ import annotations

from merlin.targetgen import capability_derive as cd
from merlin.targetgen.compute_units import SemanticCapability


def _derived(**kw) -> cd.DerivedCapabilities:
    out = cd.DerivedCapabilities()
    cd._record(out, cd.FamilyEvidence(family="contraction", status="supported", source="rtl_facts",
                                      evidence="RTL array 'mesh' 16x16", **kw))
    return out


def _kinds(findings, kind):
    return [f for f in findings if f["kind"] == kind]


def test_an_evidenced_rank_the_contract_omits_is_reported_as_narrowing():
    """The gemmini shape: evidence shows a rank the declaration lacks, so regions of it leave the
    denominator. The finding must say which direction that moves recall, or it reads as cosmetic."""
    declared = {"contraction": SemanticCapability(family="contraction", dtypes=("int8",), ranks=(2,))}
    found = _kinds(cd.reconcile(declared, _derived(dtypes=("int8",), ranks=(2, 4))), "missing_axis")
    assert len(found) == 1
    assert found[0]["axis"] == "ranks" and "4" in found[0]["detail"]
    assert "RAISES recall" in found[0]["detail"], "the finding must name the direction of the error"


def test_a_declared_rank_no_rung_confirms_is_reported_as_unaudited():
    """Not an error -- the ladder cannot decide axes at all. But an unconfirmed claim that says nothing
    is indistinguishable from a confirmed one, which is how the last narrow axis survived review."""
    declared = {"contraction": SemanticCapability(family="contraction", dtypes=("int8",), ranks=(2, 4))}
    found = _kinds(cd.reconcile(declared, _derived(dtypes=("int8",), ranks=(2,))), "unaudited_axis")
    assert len(found) == 1 and found[0]["axis"] == "ranks" and "4" in found[0]["detail"]


def test_dtypes_are_compared_through_the_format_registry_not_as_strings():
    """The deriver reads the RTL datapath's spelling and the contract carries the capability
    vocabulary's. `i8` and `int8` are one format; reporting them as a mismatch would bury every real
    finding under noise on every target at once."""
    declared = {"contraction": SemanticCapability(family="contraction", dtypes=("int8",), ranks=(2,))}
    findings = cd.reconcile(declared, _derived(dtypes=("i8",), ranks=(2,)))
    assert not [f for f in findings if f.get("axis") == "dtypes"], \
        "an alias of a declared format is not an axis finding"


def test_an_axis_neither_declared_nor_evidenced_is_silent():
    """An unconstrained axis is not a finding. A gate that fires on every family with no rank data
    would be noise on every target, and noise is how a real finding gets scrolled past."""
    declared = {"contraction": SemanticCapability(family="contraction", dtypes=("int8",))}
    assert not [f for f in cd.reconcile(declared, _derived(dtypes=("int8",))) if "axis" in f]


def test_ranks_union_across_rungs_rather_than_first_wins():
    """Two rungs may each evidence a different rank. Keeping only the first rung's ranks drops an axis
    inside the deriver -- the same narrowing error, one layer earlier."""
    out = cd.DerivedCapabilities()
    cd._record(out, cd.FamilyEvidence(family="contraction", status="supported", source="rtl_facts",
                                      evidence="array", ranks=(2,)))
    cd._record(out, cd.FamilyEvidence(family="contraction", status="supported", source="unit_intent",
                                      evidence="unit", ranks=(4,)))
    assert out.supported["contraction"].ranks == (2, 4)
