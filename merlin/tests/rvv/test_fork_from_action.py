"""WS-C: the CCA-native fork proposer (action_catalog as the single source of truth for the beam)."""
from __future__ import annotations

from merlin.kernels.cca_compare import Divergence
from merlin.mining.fork_from_action import action_to_fork, propose_forks_from_cca
from merlin.kernels.action_catalog import route

_KNOBS = {"op_match": [{"op": "matmul", "tile": [4, 8, 1], "vector": [4, 8, 1]}]}


def _div(axis, expert, ours):
    return Divergence(axis=axis, expert=expert, ours=ours, backend="rvv", evidence=["k1"])


def test_impr_feature_axis_maps_to_compiler_features_override():
    # contraction_form routes to a certified impr_features hook -> a compiler_features fork.
    p = action_to_fork(route(_div("compute.contraction_form", "fused_fma", "mul_add")), _KNOBS)
    assert p.forkable is True and p.lever == "feature"
    assert p.overrides == {"compiler_features": ["fused_vfmacc_contraction"]}
    # the proposal carries its CompilerAction (intended_facet) so the beam can AUDIT the minted fork
    assert p.action is not None and p.action.intended_facet == {"compute.contraction_form": "fused_fma"}


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


def test_register_block_routes_to_per_op_mr_feature():
    # compute.register_block (expert MR > ours) now routes to the REAL whole-model-safe realization:
    # the per-op MR + M-pad-tail impr_features hook (accumulator_resident_wholemodel_vf_mrpad), which
    # emits the MR>1 register block on every matmul (padding the M-tail so M=1/M%MR!=0 no longer
    # scalar-fall-back) — not the op_match tile knob, whose high MR the transform codegen never realized
    # (and which could not lower the mixed-M VLA models).
    p = action_to_fork(route(_div("compute.register_block", (7, None), (1, None))), _KNOBS)
    assert p.forkable is True and p.lever == "feature"
    assert p.overrides == {"compiler_features": ["accumulator_resident_wholemodel_vf_mrpad"]}


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


def test_bb1a_previously_demoted_axes_now_fork():
    """BB1(a): four forkable_now=True axes the proposer used to silently demote to work-items now mint
    real forks — the HEURISTIC schedule-seams map to their registered features, the datapath axes to
    the int8 knob. This is the immediate 'propose more' win (catalog + features already existed)."""
    # HEURISTIC axes implemented by a registered impr_features feature.
    nr = action_to_fork(route(_div("compute.nr_is_vsetvlmax", True, False)), _KNOBS)
    assert nr.forkable is True and nr.overrides == {"compiler_features": ["fused_vfmacc_scalable"]}
    mr = action_to_fork(route(_div("compute.mr_adapts_to_m", True, False)), _KNOBS)
    assert mr.forkable is True and mr.overrides == {"compiler_features": ["accumulator_resident_mtail"]}
    # accumulate-width / element-width datapath axes reach the int8 datapath knob.
    for axis, exp, ours in (("compute.accumulator_dtype", "i32", "f32"), ("vector.sew", 8, 32)):
        p = action_to_fork(route(_div(axis, exp, ours)), _KNOBS)
        assert p.forkable is True and p.overrides == {"dtype_strategy": "int8_w8a8"}, axis


def test_bb1b_operand_packing_orphan_is_now_a_forkable_lever():
    """BB1(b): memory.access_pattern (the #1 expert data-movement gap-driver) was captured by
    decode.memory but orphaned (no route). It now routes to the vfmacc_packed feature and forks."""
    p = action_to_fork(route(_div("memory.access_pattern", "unit_stride", "strided")), _KNOBS)
    assert p.forkable is True and p.overrides == {"compiler_features": ["vfmacc_packed"]}
    # and it is no longer an orphan in the bijection ledger.
    from merlin.kernels.cca_contract import check_bijection, KNOWN_OPEN
    assert "memory.access_pattern" not in check_bijection("rvv").orphan_fields
    assert "memory.access_pattern" not in KNOWN_OPEN["rvv"]["orphan_fields"]
