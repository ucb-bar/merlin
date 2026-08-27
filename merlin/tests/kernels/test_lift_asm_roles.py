"""The role-indexed lifter — the generalization of `lift_asm` off RVV mnemonics.

`lift_asm` counts RVV mnemonic literals, so it recognizes exactly one target. This one consumes a
stream that some target's own decoder has role-tagged, and fills the facets those roles license. Which
engine-scoped facet gets filled follows the ENDPOINT's engine, never a target name — so a target cannot
end up with a lifted facet its declared silicon does not have.

Populates SpatialFacet for the first time in the tree, plus SimtFacet and a non-RVV VectorFacet.
"""
from __future__ import annotations

import pytest

from merlin.kernels import cca as C
from merlin.kernels import endpoints as EP


class _D:
    """A role-tagged instruction, the shape every decoder in kernels.decode emits."""

    def __init__(self, index, roles, addr=None):
        self.index, self.roles, self.addr = index, tuple(roles), index if addr is None else addr


def _stream(*rolesets):
    return [_D(i, r) for i, r in enumerate(rolesets)]


_CONTRACTION = ("config",), ("operand_load",), ("weight_load",), ("accumulate",), ("readout", "commit")


class TestTheEngineDecidesTheFacet:
    def test_an_array_endpoint_fills_the_spatial_facet(self):
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles(_stream(*_CONTRACTION), ep, op="matmul", source="t",
                             geometry={"pe_rows": 16, "pe_cols": 16, "dataflow": "ws"})
        assert c.spatial is not None and c.spatial.pe_rows == 16 and c.spatial.dataflow == "ws"
        assert c.simt is None and c.vector is None, "only the endpoint's own engine may be filled"

    def test_a_simt_endpoint_fills_the_simt_facet(self):
        ep = EP.load_endpoint("muon_simt")
        c = C.lift_asm_roles(_stream(("operand_load",), ("accumulate",), ("sync",), ("readout",)),
                             ep, op="matmul", source="t",
                             geometry={"threads_per_warp": 32, "warps": 8})
        assert c.simt is not None and c.simt.threads_per_warp == 32
        assert c.simt.barriers_in_loop == 1, "a sync role is the barrier count"
        assert c.spatial is None

    def test_a_lane_endpoint_fills_the_vector_facet(self):
        """The second VectorFacet instance, and the test that it generalizes off RVV."""
        ep = EP.load_endpoint("atlas_vpu")
        c = C.lift_asm_roles(_stream(("elementwise",)), ep, op="exp", source="t",
                             geometry={"sew": 8, "lmul": 1.0})
        assert c.vector is not None and c.vector.sew == 8
        assert c.spatial is None and c.simt is None

    def test_geometry_is_never_inferred_from_the_stream(self):
        """A stream using a 16-wide operand says nothing about how wide the array is. Geometry is
        IDENTITY the compiler tiles TO; absent, it stays None rather than being guessed."""
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles(_stream(*_CONTRACTION), ep, op="matmul", source="t")
        assert c.spatial.pe_rows is None and c.spatial.pe_cols is None


class TestDispatchIsFilledForEveryEndpoint:
    def test_the_dispatch_facet_is_engine_agnostic(self):
        # Every endpoint is driven by something, so dispatch is filled regardless of engine.
        for name in ("gemmini_rocc", "muon_simt", "atlas_vpu"):
            c = C.lift_asm_roles(_stream(("config",), ("accumulate",)), EP.load_endpoint(name),
                                 op="x", source="t")
            assert c.dispatch is not None and c.dispatch.n_dispatches == 2, name

    def test_a_loop_descriptor_reports_the_loop_as_offloaded(self):
        """Gemmini's biggest expert win is dispatch SHAPE, and this is the axis that expresses it."""
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles(_stream(("config",), ("loop_descriptor",)), ep, op="matmul", source="t")
        assert c.dispatch.loop_offloaded is True

    def test_a_hand_driven_stream_reports_no_offload(self):
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles(_stream(*_CONTRACTION), ep, op="matmul", source="t")
        assert c.dispatch.loop_offloaded is False

    def test_config_set_once_is_descriptor_reuse(self):
        ep = EP.load_endpoint("gemmini_rocc")
        reuse = C.lift_asm_roles(_stream(("config",), ("accumulate",), ("accumulate",)), ep,
                                 op="m", source="t")
        churn = C.lift_asm_roles(_stream(("config",), ("accumulate",), ("config",), ("accumulate",)),
                                 ep, op="m", source="t")
        assert reuse.dispatch.descriptor_reuse is True
        assert churn.dispatch.descriptor_reuse is False


class TestResidencyIsNeverGuessed:
    def test_one_static_accumulate_without_loop_spans_is_undecidable(self):
        """A looping reduction emits ONE accumulate statically, so "is there a readout between the
        first and the last" is vacuous for exactly the kernels that matter. Without resolved spans the
        honest answer is None — an unlinked object resolves every branch to its own address, finds no
        span, and would otherwise collapse every loop-scoped count to a confident zero."""
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles(_stream(("accumulate",), ("readout",)), ep, op="m", source="t")
        assert c.compute.accumulator_resident is None

    def test_an_unrolled_reduction_with_a_mid_readout_is_not_resident(self):
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles(_stream(("accumulate",), ("readout",), ("accumulate",)), ep,
                             op="m", source="t")
        assert c.compute.accumulator_resident is False

    def test_an_unrolled_reduction_committing_once_is_resident(self):
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles(_stream(("accumulate",), ("accumulate",), ("readout",)), ep,
                             op="m", source="t")
        assert c.compute.accumulator_resident is True

    def test_loop_spans_scope_residency_to_the_reduction(self):
        ep = EP.load_endpoint("gemmini_rocc")
        s = [_D(0, ("accumulate",), addr=100), _D(1, ("readout",), addr=500)]
        assert C.lift_asm_roles(s, ep, op="m", source="t",
                                loop_spans=[(90, 200)]).compute.accumulator_resident is True
        assert C.lift_asm_roles(s, ep, op="m", source="t",
                                loop_spans=[(90, 600)]).compute.accumulator_resident is False

    def test_an_undriven_engine_is_undecidable_not_resident(self):
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles(_stream(("config",)), ep, op="m", source="t")
        assert c.compute.accumulator_resident is None


class TestTheLiftReportsWhatItCouldNotSee:
    def test_a_contraction_missing_its_readout_is_low_confidence(self):
        """The measured failure: an audit counted accumulates and passed a kernel that never drained
        its accumulator."""
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles(_stream(("operand_load",), ("accumulate",)), ep, op="m", source="t")
        assert c.provenance["confidence"] == "low"
        assert "readout" in c.provenance["missing_contraction_roles"]

    def test_a_stream_with_no_recognized_roles_is_not_a_clean_lift(self):
        """Reporting high confidence for a stream where nothing was tagged is a false clean — it says
        the kernel is fine when it means the decoder saw nothing."""
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles([], ep, op="m", source="t")
        assert c.provenance["confidence"] == "low"
        assert c.provenance["no_roles_recognized"] is True

    def test_completeness_is_only_asked_of_a_contraction(self):
        """A lane engine's activation kernel is not a contraction, and judging it by contraction
        completeness would report every correct one as incomplete for lacking a weight load."""
        ep = EP.load_endpoint("atlas_vpu")
        c = C.lift_asm_roles(_stream(("elementwise",)), ep, op="exp", source="t")
        assert c.provenance["confidence"] == "high"
        assert c.provenance["missing_contraction_roles"] == []

    def test_the_role_census_rides_along_as_provenance(self):
        ep = EP.load_endpoint("gemmini_rocc")
        c = C.lift_asm_roles(_stream(*_CONTRACTION), ep, op="m", source="t")
        assert c.provenance["role_counts"]["accumulate"] == 1
        assert c.provenance["level"] == "asm" and c.provenance["engine"] == "spatial"


class TestTextIsaStreams:
    """A self-hosted-ISA corpus is hand-written text, not a disassembly, and the same lifter reads it."""

    def _atlas(self):
        from merlin.targetgen import isa_model as IM
        try:
            if IM.isa_model_for_target("atlas").is_empty():
                pytest.skip("atlas IsaModel unavailable")
        except Exception:                                  # noqa: BLE001
            pytest.skip("atlas IsaModel unavailable")

    def test_a_text_stream_lifts_like_a_binary_one(self):
        self._atlas()
        from merlin.kernels.decode import isa_text as T
        ep = EP.load_endpoint("atlas_isa")
        d = T.decode_text(["vmatpush.weight.mxu0 0, 1", "vmatmul.mxu0 0,0,0",
                           "vmatmul.mxu0 0,1,0", "vmatpop.mxu0 2"], "atlas", ep)
        c = C.lift_asm_roles(d, ep, op="matmul", source="t", geometry={"pe_rows": 32, "pe_cols": 32})
        assert c.spatial.pe_rows == 32
        assert c.provenance["role_counts"].get("accumulate") == 2

    def test_an_unresolvable_spelling_is_named_not_dropped(self):
        """Measured: every one of 137 corpus files contains a mnemonic the model does not have, and it
        is a SPELLING gap (VMATPUSH.W.MXU0 vs vmatpush.weight.mxu0), not a coverage gap. Mining the
        subset that happened to parse is the recorded silent-drop failure."""
        self._atlas()
        from merlin.kernels.decode import isa_text as T
        ep = EP.load_endpoint("atlas_isa")
        d = T.decode_text(["VMATPUSH.W.MXU0 0, 1"], "atlas", ep)
        assert "VMATPUSH.W.MXU0" in T.unresolved_mnemonics(d)

    def test_labels_directives_and_comments_are_not_instructions(self):
        self._atlas()
        from merlin.kernels.decode import isa_text as T
        d = T.decode_text(["loop:", "  // a comment", ".align 4", "", "nop"], "atlas", None)
        assert [x.mnemonic for x in d] == ["nop"]

    def test_two_endpoints_on_one_isa_do_not_claim_each_others_work(self):
        self._atlas()
        from merlin.kernels.decode import isa_text as T
        arr, vpu = EP.load_endpoint("atlas_isa"), EP.load_endpoint("atlas_vpu")
        line = ["vmatmul.mxu0 0,0,0"]
        assert T.role_histogram(T.decode_text(line, "atlas", arr)).get("accumulate") == 1
        assert T.role_histogram(T.decode_text(line, "atlas", vpu)) == {}
