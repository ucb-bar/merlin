"""WS-C: the CCA-native fork proposer (action_catalog as the single source of truth for the beam)."""
from __future__ import annotations

from merlin.kernels.cca_compare import Divergence
from merlin.rvvgen.fork_from_action import action_to_fork, propose_forks_from_cca
from merlin.kernels.action_catalog import route

_KNOBS = {"op_match": [{"op": "matmul", "tile": [4, 8, 1], "vector": [4, 8, 1]}]}


def _div(axis, expert, ours):
    return Divergence(axis=axis, expert=expert, ours=ours, backend="rvv", evidence=["k1"])


def test_impr_feature_axis_maps_to_compiler_features_override():
    # contraction_form routes to a certified impr_features hook -> a compiler_features fork.
    p = action_to_fork(route(_div("compute.contraction_form", "fused_fma", "mul_add")), _KNOBS)
    assert p.forkable is True and p.lever == "feature"
    assert p.overrides == {"compiler_features": ["fused_vfmacc_contraction"]}


def test_int8_widening_maps_to_dtype_strategy_knob():
    p = action_to_fork(route(_div("compute.widening", True, False)), _KNOBS)
    assert p.forkable is True and p.lever == "knob"
    assert p.overrides == {"dtype_strategy": "int8_w8a8"}


def test_lmul_maps_to_wider_n_knob():
    p = action_to_fork(route(_div("vector.lmul", 4.0, 2.0)), _KNOBS)
    assert p.forkable is True and p.lever == "knob"
    # N (the second-to-last tile dim) is widened x2; only KNOWN knob keys are emitted.
    assert p.overrides["op_match"][0]["tile"] == [4, 16, 1]
    assert set(p.overrides) <= {"op_match", "contraction_strategy", "lowering_patterns", "dtype_strategy"}


def test_register_block_sets_mr_from_intended_expert():
    p = action_to_fork(route(_div("compute.register_block", (7, None), (1, None))), _KNOBS)
    assert p.forkable is True and p.lever == "knob"
    assert p.overrides["op_match"][0]["tile"][0] == 7   # MR set to the expert's MR


def test_deferred_pass_is_honest_work_item_not_a_faked_knob():
    # vl_strategy routes to a PASS with forkable_now=False -> recorded, empty overrides, forkable False.
    p = action_to_fork(route(_div("vector.vl_strategy", "vsetvl_loop", "vsetivli_fixed")), _KNOBS)
    assert p.forkable is False and p.lever == "work_item"
    assert p.overrides == {}


def test_propose_forks_from_cca_is_beam_compatible():
    divs = [_div("compute.contraction_form", "fused_fma", "mul_add"),
            _div("vector.vl_strategy", "vsetvl_loop", "vsetivli_fixed")]
    props = propose_forks_from_cca(divs, _KNOBS)
    assert len(props) == 2
    assert [p.forkable for p in props] == [True, False]   # one knob fork + one honest work-item
