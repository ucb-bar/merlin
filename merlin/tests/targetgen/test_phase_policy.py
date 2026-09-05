"""A capsule's phase must be DERIVED, and a question nobody could answer must not read as an answer.

The defect these pin is the one this repo keeps re-encountering: a check that could not run reporting a
result. The first version of ``phase_policy`` folded an UNKNOWN certifiability verdict into "not
certifiable", so a target with no measured certification history reported ``both = 0`` -- which reads as
"no capsule serves both phases" and means "we cannot tell". Those are different findings with different
remedies, so ``undetermined`` is a distinct outcome and is pinned here.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import phase_policy as PP


def _capsule(**over):
    """A minimal rank-2 contraction capsule; each test overrides only what it is about."""
    base = {
        "name": "T0",
        "operation": {"op": "matmul", "attributes": {"lhs": "A", "weight": "W", "out": "Y"}},
        "inputs": [
            {"name": "A", "role": "input", "shape": [16, 16], "dtype": "i8"},
            {"name": "W", "role": "weight", "shape": [16, 16], "dtype": "i8"},
        ],
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        "semantic": {"semantic_family": "contraction"},
    }
    base.update(over)
    return base


# --------------------------------------------------------------- the tri-state must stay tri-state

def test_a_verdict_refuses_truthiness():
    """UNKNOWN would silently read as NO under ``if verdict:``, so the type refuses the question."""
    with pytest.raises(TypeError):
        bool(PP.Verdict(PP.UNKNOWN, "no history"))


def test_unknown_certifiability_is_undetermined_and_not_neither():
    """THE REGRESSION. With no measured history the size predicate cannot answer, and that must reach
    the caller as ``undetermined`` -- a statement about the evidence -- never as ``neither``, which is a
    statement about the capsule."""
    v = PP.phase_of(_capsule(), target="a_target_with_no_history", fit=None, budget_s=300.0)
    assert v.cert.value == PP.UNKNOWN
    assert v.phase == PP.UNDETERMINED, (
        "an unanswerable size question was folded into a verdict about the capsule; that is the "
        "'a check that could not run reported a result' failure this module exists to avoid"
    )
    assert v.phase != PP.NEITHER


def test_undetermined_is_counted_separately_in_a_report():
    rep = PP.split_report([_capsule()], target="t", fit=None, budget_s=300.0)
    assert rep["counts"][PP.UNDETERMINED] == 1
    assert rep["counts"][PP.NEITHER] == 0


# ------------------------------------------------------------------------------- the two predicates

def test_a_capsule_with_no_cycle_accurate_tier_cannot_be_certified():
    v = PP.certifiable(_capsule(required_oracle_tiers=["L0", "L1", "L2"]), target="t")
    assert v.value == PP.NO
    assert "cycle-accurate" in v.reason


def test_an_operand_past_the_measured_range_is_refused_for_unknown_cost_not_for_size():
    big = _capsule(inputs=[{"name": "A", "role": "input", "shape": [4096, 4096]},
                           {"name": "W", "role": "weight", "shape": [4096, 16]}])
    v = PP.certifiable(big, target="t")
    assert v.value == PP.NO
    assert "unknown, not merely large" in v.reason


def test_a_non_contracting_family_prices_at_zero_and_is_phase_one_only():
    """Zero is the TRUE work of a family that contracts nothing -- not a missing price. It still must
    keep such a member out of a performance corpus, because it has no utilization to improve."""
    mv = _capsule(semantic={"semantic_family": "movement"},
                  operation={"op": "movement", "attributes": {"out": "Y"}},
                  inputs=[{"name": "X", "role": "input", "shape": [16, 16]}])
    macs, why = PP.declared_macs(mv)
    assert macs == 0 and "contracts nothing" in why
    assert PP.priceable(mv).value == PP.NO


def test_an_unpriceable_member_says_why_it_costs_more_than_itself():
    odd = _capsule(operation={"op": "mystery", "attributes": {"out": "Y"}},
                   semantic={"semantic_family": "contraction"},
                   inputs=[{"name": "A", "role": "input", "shape": [3, 5]}])
    v = PP.priceable(odd)
    assert v.value == PP.NO
    assert "attainment stop condition" in v.reason


# --------------------------------------------------------------------------- the work derivation

def test_a_declared_contraction_is_priced_from_its_shared_reduction_axis():
    assert PP.declared_macs(_capsule())[0] == 16 * 16 * 16


def test_a_transposed_weight_is_priced_on_the_axis_it_actually_shares():
    """Assuming lhs[-1] == weight[0] by position prices a shape the capsule does not declare."""
    t = _capsule(inputs=[{"name": "A", "role": "input", "shape": [8, 32]},
                         {"name": "W", "role": "weight", "shape": [16, 32]}])
    assert PP.declared_macs(t)[0] == 8 * 32 * 16


def test_operands_that_share_no_axis_refuse_rather_than_guess():
    bad = _capsule(inputs=[{"name": "A", "role": "input", "shape": [8, 32]},
                           {"name": "W", "role": "weight", "shape": [7, 5]}])
    macs, why = PP.declared_macs(bad)
    assert macs is None and "share no reduction axis" in why


def test_a_weight_stationary_member_prices_every_activation_sharing_the_weight():
    """The reuse IS the point, so the work is the sum over activations, not one contraction."""
    rr = _capsule(operation={"op": "resident_reuse", "attributes": {"weight": "W"}},
                  inputs=[{"name": "W", "role": "weight", "shape": [16, 16]},
                          {"name": "A0", "role": "input", "shape": [16, 16]},
                          {"name": "A1", "role": "input", "shape": [16, 16]}])
    macs, why = PP.declared_macs(rr)
    assert macs == 2 * 16 * 16 * 16
    assert "sharing one weight" in why


def test_a_convolution_is_priced_from_its_declared_geometry():
    """Hand-checked: 10x10 in, 3x3 window, unit stride, no padding -> 8x8 positions; a [36, 16]
    im2col weight is a 36-tap window with 16 outputs. 8*8*36*16 = 36864."""
    conv = _capsule(
        operation={"op": "conv2d", "attributes": {
            "ifm": "IFM", "weight": "W", "out": "Y", "ci": 4, "kh": 3, "kw": 3,
            "stride": [1, 1], "padding": [0, 0, 0, 0], "dilation": [1, 1], "layout": "nhwc"}},
        inputs=[{"name": "W", "role": "weight", "shape": [36, 16]},
                {"name": "IFM", "role": "input", "shape": [1, 10, 10, 4]}])
    assert PP.declared_macs(conv)[0] == 36864


def test_padding_and_stride_change_the_priced_work():
    """MUTATION CONTROL. If the geometry axes were ignored -- which is how every conv capsule in this
    corpus came to declare the same default -- these three would price identically."""
    def conv(padding, stride):
        return _capsule(
            operation={"op": "conv2d", "attributes": {
                "ifm": "IFM", "weight": "W", "out": "Y", "ci": 4, "kh": 3, "kw": 3,
                "stride": stride, "padding": padding, "dilation": [1, 1], "layout": "nhwc"}},
            inputs=[{"name": "W", "role": "weight", "shape": [36, 16]},
                    {"name": "IFM", "role": "input", "shape": [1, 10, 10, 4]}])
    plain = PP.declared_macs(conv([0, 0, 0, 0], [1, 1]))[0]
    same = PP.declared_macs(conv([1, 1, 1, 1], [1, 1]))[0]
    strided = PP.declared_macs(conv([0, 0, 0, 0], [2, 2]))[0]
    assert plain != same != strided and plain != strided


def test_a_geometry_leaving_no_output_position_refuses():
    tiny = _capsule(
        operation={"op": "conv2d", "attributes": {
            "ifm": "IFM", "weight": "W", "out": "Y", "ci": 4, "kh": 9, "kw": 9,
            "stride": [1, 1], "padding": [0, 0, 0, 0], "dilation": [1, 1]}},
        inputs=[{"name": "W", "role": "weight", "shape": [324, 16]},
                {"name": "IFM", "role": "input", "shape": [1, 4, 4, 4]}])
    macs, why = PP.declared_macs(tiny)
    assert macs is None and "no output position" in why


def test_two_activations_sharing_an_axis_are_a_contraction_without_any_weight():
    """A scores block contracts Q against K and neither operand is a parameter."""
    qk = _capsule(operation={"op": "attention_qk", "attributes": {"q": "Q", "k": "K", "out": "Y"}},
                  inputs=[{"name": "Q", "role": "input", "shape": [16, 32]},
                          {"name": "K", "role": "input", "shape": [16, 32]}])
    assert PP.declared_macs(qk)[0] == 16 * 32 * 16


# ------------------------------------------------------------------------------------ the outcome

def test_both_is_reachable_and_is_the_anchor_state():
    class _Fit:  # a measured history that affords a generous size
        n_samples = 8
    from merlin.targetgen import cert_cost as CC

    v = PP.phase_of(_capsule(), target="t", fit=None, budget_s=None)
    assert v.phase == PP.BOTH, v.reason
