"""Evidence-aware comparison cannot turn missing observations into agreement."""

import pytest

from merlin.kernels.cca import CCA, ComputeFacet, DispatchFacet, MemoryFacet
from merlin.kernels.cca_compare import compare, compare_evidence
from merlin.kernels.reconstruction import goals_from_cca


def cca(*, op="matmul", scope="kernel", compute=None, memory=None, dispatch=None, facts=None, **provenance):
    return CCA(op=op, backend=[], scope=scope, compute=compute, memory=memory, dispatch=dispatch,
               provenance=dict(axis_evidence=facts or {}, **provenance))


def fact(status="observed", *, scope=None, evidence=("pinned-actual-artifact#observation",)):
    row = dict(status=status, evidence=evidence)
    if scope is not None:
        row["scope"] = scope
    return row


def test_empty_static_lifts_remain_unknown_including_both_absent_axes():
    reference = cca(op="unknown", level="static_disassembly")
    candidate = cca(op="unknown", level="static_disassembly")
    assert compare(reference, candidate) == []
    result = compare_evidence(reference, candidate)
    assert result.status == "unknown"
    assert result.denominator == 0
    assert result.divergences == result.compared_axes == ()
    unknowns = {row["axis"]: row for row in result.unknowns}
    assert unknowns["op"]["reason"] == "identity_unknown"
    assert unknowns["memory.onchip_resident"]["missing_sides"] == ("expert", "ours")
    assert unknowns["dispatch.n_dispatches"]["reason"] == "value_unavailable"
    assert result.to_dict()["denominator"] == 0


def test_unknown_register_block_suppresses_legacy_none_to_mr1_divergence():
    reference = cca(compute=ComputeFacet(register_block=(4, 8)),
                    facts={"compute.register_block": fact()})
    candidate = cca(compute=ComputeFacet(register_block=None),
                    facts={"compute.register_block": fact("unknown", evidence=())})
    assert any(row.axis == "compute.register_block" for row in compare(reference, candidate))
    result = compare_evidence(reference, candidate)
    assert result.divergences == ()
    assert "compute.register_block" not in result.compared_axes
    assert any(row["axis"] == "compute.register_block" and row["missing_sides"] == ("ours",)
               for row in result.unknowns)


@pytest.mark.parametrize("evidence", [{}, fact("unspecified"), fact("unknown"), fact("contradicted"),
                                    fact(evidence=()), fact(evidence=("",))])
def test_unsupported_axis_evidence_cannot_become_a_comparison(evidence):
    reference = cca(memory=MemoryFacet(onchip_resident="a"))
    # Legacy goal discovery intentionally permits an unspecified hypothesis.
    assert goals_from_cca(reference)["goals"] == ("reuse.lhs",)
    reference.provenance["axis_evidence"] = {"memory.onchip_resident": evidence}
    candidate = cca(memory=MemoryFacet(onchip_resident="b"),
                    facts={"memory.onchip_resident": fact(scope="complete_invocation_full_reduction")})
    result = compare_evidence(reference, candidate)
    assert result.status == "unknown" and not result.divergences
    assert any(row["axis"] == "memory.onchip_resident" and row["reason"] == "axis_evidence_unavailable"
               for row in result.unknowns)


@pytest.mark.parametrize("change", [{"op": "conv2d"}, {"scope": "program"}])
def test_known_operation_or_scope_mismatch_blocks_all_axes(change):
    reference = cca(compute=ComputeFacet(register_block=(4, 8)), facts={"compute.register_block": fact()})
    candidate = cca(compute=ComputeFacet(register_block=(2, 8)), facts={"compute.register_block": fact()}, **change)
    result = compare_evidence(reference, candidate)
    assert result.status == "mismatch"
    assert result.denominator == 0 and not result.divergences
    assert any(row["reason"] == "identity_mismatch" for row in result.unknowns)


def test_panel_and_whole_operand_residency_are_incompatible_even_for_same_value():
    axis = "memory.onchip_resident"
    reference = cca(memory=MemoryFacet(onchip_resident="a"),
                    facts={axis: fact(scope="complete_invocation_full_reduction")})
    candidate = cca(memory=MemoryFacet(onchip_resident="a"),
                    facts={axis: fact(scope="macro_block_panel_lifetime")})
    assert compare(reference, candidate) == []
    result = compare_evidence(reference, candidate)
    assert result.denominator == 0 and not result.divergences
    assert any(row["axis"] == axis and row["reason"] == "axis_scope_mismatch" for row in result.unknowns)


def test_legacy_residency_label_cannot_replace_per_axis_scope():
    axis = "memory.onchip_resident"
    reference = cca(memory=MemoryFacet(onchip_resident="a"), facts={axis: fact()}, residency_scope="panel")
    candidate = cca(memory=MemoryFacet(onchip_resident="a"), facts={axis: fact()}, residency_scope="full")
    result = compare_evidence(reference, candidate)
    assert result.denominator == 0
    assert any(row["axis"] == axis and row["reason"] == "axis_scope_unavailable" for row in result.unknowns)


def test_conflicting_global_residency_scope_still_blocks_axis():
    axis = "memory.onchip_resident"
    reference = cca(memory=MemoryFacet(onchip_resident="a"),
                    facts={axis: fact(scope="reduction")}, residency_scope="panel")
    candidate = cca(memory=MemoryFacet(onchip_resident="a"),
                    facts={axis: fact(scope="reduction")}, residency_scope="full")
    result = compare_evidence(reference, candidate)
    assert any(row["axis"] == axis and row["reason"] == "residency_scope_mismatch" for row in result.unknowns)


def test_static_sites_and_dynamic_target_dispatch_counts_are_not_comparable():
    axis = "dispatch.n_dispatches"
    reference = cca(dispatch=DispatchFacet(n_dispatches=10), facts={axis: fact(scope="listed_instruction_sites")})
    candidate = cca(dispatch=DispatchFacet(n_dispatches=100), facts={axis: fact(scope="target_issued_commands")})
    result = compare_evidence(reference, candidate)
    assert result.denominator == 0 and not result.divergences
    assert any(row["axis"] == axis and row["reason"] == "axis_scope_mismatch" for row in result.unknowns)


def test_partially_unknown_scope_record_cannot_close_residency_obligation():
    axis = "memory.onchip_resident"
    observed = cca(memory=MemoryFacet(onchip_resident="a"),
                   facts={axis: fact(scope={"lifetime": "reduction", "region": None})})
    result = compare_evidence(observed, observed)
    assert axis not in result.compared_axes
    assert any(row["axis"] == axis and row["reason"] == "axis_scope_unavailable" for row in result.unknowns)


def test_complete_matching_residency_scope_is_compared_without_an_agreement_claim():
    axis = "memory.onchip_resident"
    scope = {"lifetime": "complete_invocation_full_reduction", "logical_dimensions": [16, 32], "start": 0}
    reference = cca(memory=MemoryFacet(onchip_resident="a"), facts={axis: fact(scope=scope)})
    result = compare_evidence(reference, reference)
    assert result.compared_axes == (axis,)
    assert result.status == "compared" and result.denominator == 1 and not result.divergences
    assert result.unknowns


def test_scope_mismatch_blocks_only_the_affected_axis():
    reference = cca(compute=ComputeFacet(register_block=(2, 8)), dispatch=DispatchFacet(loop_offloaded=False),
                    facts={"compute.register_block": fact(scope="outer_loop"), "dispatch.loop_offloaded": fact()})
    candidate = cca(compute=ComputeFacet(register_block=(4, 8)), dispatch=DispatchFacet(loop_offloaded=False),
                    facts={"compute.register_block": fact(scope="inner_loop"), "dispatch.loop_offloaded": fact()})
    result = compare_evidence(reference, candidate)
    assert result.compared_axes == ("dispatch.loop_offloaded",)
    assert result.denominator == 1 and not result.divergences
    assert any(row["axis"] == "compute.register_block" and row["reason"] == "axis_scope_mismatch"
               for row in result.unknowns)


def test_supported_axes_reuse_exact_comparison_and_keep_actual_witnesses():
    axis = "compute.register_block"
    reference = cca(compute=ComputeFacet(register_block=(2, 8)),
                    facts={axis: fact(evidence=("reference#block",))})
    candidate = cca(compute=ComputeFacet(register_block=(4, 8)),
                    facts={axis: fact("inferred", evidence=("candidate#block",))})
    result = compare_evidence(reference, candidate)
    assert result.status == "compared" and result.denominator == 1
    assert result.compared_axes == (axis,)
    assert len(result.divergences) == 1
    divergence = result.divergences[0]
    assert (divergence.axis, divergence.expert, divergence.ours) == (axis, (2, 8), (4, 8))
    assert divergence.evidence == ["reference#block", "candidate#block"]
    assert "equality are not established" in result.advisory_scope
    assert any(row["axis"] == "memory.onchip_resident" for row in result.unknowns)


def test_zero_false_and_equal_values_count_as_compared_with_proper_evidence():
    facts = {"dispatch.n_dispatches": fact(scope="target_issued_commands"), "dispatch.loop_offloaded": fact()}
    reference = cca(dispatch=DispatchFacet(n_dispatches=0, loop_offloaded=False), facts=facts)
    result = compare_evidence(reference, reference)
    assert result.status == "compared"
    assert result.denominator == 2 and not result.divergences
