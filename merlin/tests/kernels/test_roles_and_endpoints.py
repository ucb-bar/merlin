"""The role vocabulary and the endpoint bindings — Phase 4's contract.

`lift_asm` recognizes RVV by counting mnemonic literals, which works exactly once. On a derived-ISA
target there are no stable mnemonics: objdump prints `<unknown>`, atlas spells one instruction four
ways across its own corpus, and a Gemmini kernel looks IDENTICAL in C whether it lowers to the hardware
loop FSM or to a fine-grained preload/compute sequence. What every endpoint DOES have is a derived
encoding table and a role per entry, so roles are the only cross-target vocabulary.
"""
from __future__ import annotations

import pytest

from merlin.kernels import endpoints as EP
from merlin.kernels import roles as R


class TestTheVocabularyIsClosed:
    def test_an_unknown_role_is_refused(self):
        with pytest.raises(KeyError, match="vocabulary is closed"):
            R.check_roles(["accumulate", "teleport"])

    def test_every_declared_role_is_known(self):
        R.check_roles(R.ROLES)

    def test_roles_that_evidence_an_engine_are_a_strict_subset(self):
        assert set(R.ROLE_EVIDENCES_ENGINE) < set(R.ROLES)

    def test_ubiquitous_roles_evidence_nothing(self):
        # Every endpoint ever built configures, syncs, moves data and commits, so their presence
        # distinguishes nothing. A role that evidences an engine must be one not everything has.
        for role in ("config", "sync", "dma", "commit", "operand_load"):
            assert R.engine_of(role) is None, role

    def test_engine_facets_agree_with_the_engine_model(self):
        from merlin.kernels import engines as E
        assert set(R.ROLE_EVIDENCES_ENGINE.values()) <= E.ENGINE_FACETS

    def test_the_role_engine_map_agrees_with_the_isa_census(self):
        """Two modules read the same distinction off different inputs. A duplicate table that drifted
        once already let a target which merely pushes weights claim it could multiply."""
        from merlin.targetgen import capability_derive as cd
        for isa_role, facet in cd._ROLE_ENGINE.items():
            role = R.from_isa_role(isa_role)
            if role is None or role not in R.ROLE_EVIDENCES_ENGINE:
                continue
            assert R.ROLE_EVIDENCES_ENGINE[role] == facet, (
                f"{isa_role!r}: census says {facet}, role table says {R.ROLE_EVIDENCES_ENGINE[role]}")

    def test_a_scalar_isa_role_drives_no_endpoint(self):
        # Scalar code is the envelope AROUND the loop. Giving it a role would make every target look
        # like it drives an engine it does not have.
        assert R.from_isa_role("scalar") is None


class TestACompleteContraction:
    def test_a_stream_without_a_readout_is_incomplete(self):
        """A prior audit counted accumulates and reported success for a kernel that never drained its
        accumulator, because nothing asked whether the result came back out."""
        assert R.missing_contraction_roles(["operand_load", "accumulate"]) == ("readout",)

    def test_a_complete_stream_is_missing_nothing(self):
        assert R.missing_contraction_roles(["operand_load", "accumulate", "readout", "config"]) == ()


class TestEndpointsBindToTheirOwnDerivedTable:
    def test_every_declared_endpoint_resolves(self):
        assert EP.endpoint_names(), "no compute endpoints declared"
        for name in EP.endpoint_names():
            EP.load_endpoint(name)

    def test_a_declared_name_absent_from_the_derived_table_is_reported_missing(self):
        """Never silently dropped. A rename in the RTL must surface as a missing role, not as a decoder
        that quietly stops recognizing an instruction — the recorded rocc_decode failure shape."""
        ep = EP.load_endpoint("gemmini_rocc")
        if not ep.roles:
            pytest.skip("gemmini RTL facts unavailable")
        assert ep.missing == {}, ep.missing
        assert hasattr(ep, "missing"), "the missing channel must exist even when empty"

    def test_derived_names_carrying_no_role_are_reported_too(self):
        ep = EP.load_endpoint("gemmini_rocc")
        if not ep.roles:
            pytest.skip("gemmini RTL facts unavailable")
        assert isinstance(ep.unmapped, tuple)

    def test_one_instruction_may_carry_several_roles(self):
        """gemmini's MVOUT both drains the accumulator and makes the result visible. Recording only the
        first made a complete contraction report a missing readout."""
        ep = EP.load_endpoint("gemmini_rocc")
        if not ep.roles:
            pytest.skip("gemmini RTL facts unavailable")
        assert set(ep.roles_of("STORE_CMD")) == {"readout", "commit"}

    def test_two_endpoints_on_one_isa_are_split_by_engine(self):
        """atlas's array and lane engine share a self-hosted ISA. If both claimed every role, the
        attribution would say nothing — which is the state that lost 50 of its 137 expert kernels."""
        arr, lane = EP.load_endpoint("atlas_isa"), EP.load_endpoint("atlas_vpu")
        if not arr.roles or not lane.roles:
            pytest.skip("atlas IsaModel unavailable")
        assert "accumulate" in arr.roles and "accumulate" not in lane.roles
        assert "elementwise" in lane.roles and "elementwise" not in arr.roles

    def test_shared_roles_belong_to_both_endpoints(self):
        # operand_load/dma evidence no engine; assigning them to one endpoint would make the other
        # look inert.
        arr, lane = EP.load_endpoint("atlas_isa"), EP.load_endpoint("atlas_vpu")
        if not arr.roles or not lane.roles:
            pytest.skip("atlas IsaModel unavailable")
        assert "operand_load" in arr.roles and "operand_load" in lane.roles

    def test_an_endpoint_evidences_the_engine_its_roles_imply(self):
        assert EP.load_endpoint("atlas_isa").engines_evidenced() <= {"spatial"}
        assert EP.load_endpoint("atlas_vpu").engines_evidenced() <= {"vector"}

    def test_an_unavailable_derivation_is_not_reported_as_missing_roles(self):
        """An absent toolchain is not evidence about the hardware."""
        ep = EP.load_endpoint("saturn_opu")
        assert ep.roles, "saturn_opu roles come from matrix_units.yaml and are always available"
        assert set(ep.roles) == {"accumulate", "broadcast", "readout", "operand_load"}
