"""The ENVELOPE facet: the code around the inner loop, and the blind spot it closes.

Every other facet describes the inner loop, so two kernels could agree on contraction form, vector
config, register block and access pattern -- look identical in CCA -- while one wrapped every tile
in a runtime call. Measured on K1 that blind spot WAS the whole f32 GEMM gap: our hot loop is better
per-FMA than XNNPACK's, but a per-tile ``memrefCopy`` cost ~79 instructions per output element.
These tests pin the discovery path end to end so it is found by the beam, not by hand.
"""
from __future__ import annotations

from pathlib import Path

from merlin.common.paths import merlin_dir
from merlin.kernels import action_catalog, cca, cca_compare
from merlin.kernels.cca_contract import check_bijection
from merlin.kernels.decode import rvv

EXPERT_ASM = merlin_dir() / "tests" / "data" / "cca_asm" / "xnnpack_f32_gemm_rvv.objdump"

# A tile epilogue that calls the rank-generic runtime copy AFTER storing the accumulators -- the
# shape our v3 codegen actually emitted (addresses/registers trimmed from the real K1 objdump).
OURS_ASM = """\
0000000000000000 <forward>:
   c8:\t855a      \tmv\ta0,s6
   ca:\t02056c07  \tvle32.v\tv24,(a0)
   ce:\tc004a787  \tflw\tfa5,-1024(s1)
   da:\tb387d457  \tvfmacc.vf\tv8,fa5,v24
   ea:\t20050513  \taddi\ta0,a0,512
   f4:\tfcf51be3  \tbne\ta0,a5,ca <forward+0xca>
   fa:\t0205e427  \tvse32.v\tv8,(a1)
  10c:\tfc050613  \taddi\ta2,a0,-64
  110:\t8132      \tmv\tsp,a2
  17a:\t00000097  \tauipc\tra,0x0
  17e:\t000080e7  \tjalr\tra
  18e:\t019ae063  \tbltu\ts5,s9,c8 <forward+0xc8>
"""


def _lift(asm_text: str, source: str, undefined):
    return cca.lift_asm(rvv.decode_text(asm_text), op="matmul", source=source,
                        undefined_symbols=undefined)


def test_envelope_facet_discriminates_expert_from_ours():
    expert = _lift(EXPERT_ASM.read_text(), "xnnpack", [])
    ours = _lift(OURS_ASM, "ours", ["memrefCopy", "memcpy", "malloc"])

    # The expert calls nothing from its compute envelope; we call a runtime helper per tile.
    assert expert.envelope.calls_in_loop == 0
    assert expert.envelope.runtime_calls == ()
    assert ours.envelope.calls_in_loop > 0
    assert "memrefCopy" in ours.envelope.runtime_calls

    # ...and this is exactly the blind spot: the INNER-LOOP facets agree, so without an `envelope` facet the
    # two kernels are indistinguishable in CCA despite a measured ~3.5x instruction-count gap.
    assert expert.compute.contraction_form == ours.compute.contraction_form == "fused_fma"


def test_a_new_facet_is_compared_without_being_hardcoded():
    """`compare` reflects CCA's facets. It used to hold a literal list, so a newly added facet was
    lifted and demanded a route yet never produced a divergence -- invisible to the beam."""
    assert "envelope" in cca_compare._facet_names()
    axes = {d.axis for d in cca_compare.compare(_lift(EXPERT_ASM.read_text(), "xnnpack", []),
                                                _lift(OURS_ASM, "ours", ["memrefCopy"]))}
    assert {"envelope.calls_in_loop", "envelope.runtime_calls"} <= axes


def test_envelope_divergences_route_to_compiler_actions():
    """A divergence the beam cannot act on is just a complaint. Both envelope axes must route, and the
    ladder must offer a forkable-now rung as well as the PASS that removes the call outright."""
    divs = [d for d in cca_compare.compare(_lift(EXPERT_ASM.read_text(), "xnnpack", []),
                                           _lift(OURS_ASM, "ours", ["memrefCopy"]))
            if d.axis.startswith("envelope.")]
    assert divs, "envelope divergences must survive to the router"
    by_axis = {d.axis: action_catalog.route(d) for d in divs}
    assert by_axis["envelope.runtime_calls"].action_class == "PASS"
    assert by_axis["envelope.runtime_calls"].forkable_now is True    # the rung that closes the gap
    # calls_in_loop is the symbol-free record of the same divergence, but no builder implements its
    # seam -- so it must NOT claim to be forkable-now; the proposer demotes it to a work-item.
    assert by_axis["envelope.calls_in_loop"].forkable_now is False


def test_envelope_axes_are_routed_in_the_bijection():
    """The contract ratchet is what FORCES a new axis to be actionable rather than decorative."""
    report = check_bijection("rvv")
    assert "envelope.runtime_calls" not in report.orphan_fields
    assert "envelope.calls_in_loop" not in report.orphan_fields


def test_no_symbol_table_is_unknown_not_clean():
    """Fail-closed: an unreadable symbol table must not be read as 'this kernel calls nothing'."""
    unknown = _lift(OURS_ASM, "ours", None)            # OURS_ASM contains a call
    assert unknown.envelope.runtime_calls is None      # unknown, NOT ()
    known_clean = _lift(OURS_ASM, "ours", ["some_unrelated_symbol"])
    assert known_clean.envelope.runtime_calls == ()


def test_zero_calls_is_soundly_clean_without_a_symbol_table():
    """A region with NO call instruction provably escapes nowhere, so () is sound even with no
    symbols. This is what lets the hand-written expert populate the axis at all -- without it BOTH
    sides stay None, the divergence never fires, and the beam cannot see the gap."""
    expert = _lift(EXPERT_ASM.read_text(), "xnnpack", None)
    assert expert.envelope.calls_in_loop == 0
    assert expert.envelope.runtime_calls == ()


def test_beam_reaches_the_fix_from_the_objdump_alone():
    """The whole point: discovery -> route -> FORK, with no human in the loop. Regression-guards the
    wiring that was missing (the beam lifted without a symbol table, so runtime_calls stayed None on
    both sides and the PASS that closes the gap was never proposed)."""
    from merlin.mining.fork_from_action import action_to_fork
    expert = _lift(EXPERT_ASM.read_text(), "xnnpack", None)          # no symbols needed
    ours = _lift(OURS_ASM, "ours", ["memrefCopy", "memcpy", "malloc"])
    div = next(d for d in cca_compare.compare(expert, ours) if d.axis == "envelope.runtime_calls")
    action = action_catalog.route(div)
    fork = action_to_fork(action, {"op_match": []})
    assert fork.forkable is True
    assert fork.overrides == {"compiler_features": ["erase_self_copy"]}
