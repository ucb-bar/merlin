"""What a target CAN accelerate and what its compiler's bindings actually route are different questions.

`coverage_for` answers the first, by semantic family and operand dtype. `route_model` answers the second:
`routing._legal_on` matches an op NAME against the unit's declared `ops`, so a region whose family and
dtype the target admits is still gapped to the scalar lane when its op is spelled something the unit does
not list.

Conflating them produced a wrong headline. An int8 capture routed 0 regions onto gemmini's mesh, which was
reported as a capture-pipeline gap. Measured properly: after the W8A8 rewrite that the compiler itself
performs, gemmini is family-eligible for 371 of 441 regions and 71 clear its dtype gate — but 0 route,
because every rewritten contraction is spelled `generic` and no unit declares `generic`. The zero is a
statement about op-name binding, not about the hardware or the capture.

The asymmetry is deliberate — compute_units documents the gap between declared capability and achieved
routing as the compiler deficiency the ARR number exists to surface. These tests exist so the two are
never again quoted as if they were one number.
"""
from __future__ import annotations

from merlin.targetgen import model_coverage as MC
from merlin.targetgen.eligibility import RegionDescriptor


def _int8_contraction_spelled(op: str):
    """One int8 contraction, varying only how the op is spelled."""
    return (RegionDescriptor(source=op, op=op, family="contraction", in_dtype="int8"),)


def test_capability_does_not_depend_on_how_the_op_is_spelled():
    """Family and dtype are properties of the work. Renaming the op cannot change what the hardware can do."""
    named = MC.coverage_for(_int8_contraction_spelled("matmul"), "gemmini", model="t")
    generic = MC.coverage_for(_int8_contraction_spelled("generic"), "gemmini", model="t")
    assert named.family_supported == generic.family_supported == 1
    assert named.dtype_ok == generic.dtype_ok == 1


def test_routing_does_depend_on_the_spelling_and_that_is_the_measurable_gap():
    """The same int8 contraction routes onto the mesh as `matmul` and does not as `generic`. That delta is
    a compiler-binding deficiency, and it is exactly what must NOT be reported as a hardware limit."""
    named = MC.route_model(_int8_contraction_spelled("matmul"), "gemmini")
    generic = MC.route_model(_int8_contraction_spelled("generic"), "gemmini")
    assert named["mesh"] == 1, "an int8 matmul is what a gemmini mesh is for"
    assert generic["mesh"] == 0, "same work, unlisted spelling — gapped to the scalar lane"


def test_a_zero_mesh_count_is_not_evidence_of_zero_capability():
    """The claim this file exists to prevent: reading route_model's mesh count as a capability verdict."""
    regions = _int8_contraction_spelled("generic")
    assert MC.route_model(regions, "gemmini")["mesh"] == 0
    rep = MC.coverage_for(regions, "gemmini", model="t")
    assert rep.family_supported == 1 and rep.dtype_ok == 1, (
        "zero routed while family-eligible and dtype-clear — the two numbers answer different questions")


def test_an_fp32_contraction_is_dtype_blocked_on_an_int8_mesh_whatever_the_spelling():
    """The other half: a capture that never offers int8 is genuinely blocked, and that IS about the data."""
    fp32 = (RegionDescriptor(source="matmul", op="matmul", family="contraction", in_dtype="fp32"),)
    rep = MC.coverage_for(fp32, "gemmini", model="t")
    assert rep.family_supported == 1, "the family is supported"
    assert rep.dtype_ok == 0 and rep.dtype_blocked == 1, "the precision is not"
