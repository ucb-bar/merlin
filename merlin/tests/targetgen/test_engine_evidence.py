"""Which compute ENGINES a target's own evidence reaches, and how that compares to what it declares.

A target's ``compute_units`` is a claim about hardware, and a claim nothing checks is an assertion.
The measured motivation: atlas declares ONE systolic unit whose ops are ``('matmul',)`` while 50 of its
137 hand-written expert kernels drive a vector engine EXCLUSIVELY. A target described only by its
declaration can be most of a machine short, and nothing compared the two.

The invariants below are mostly about the THREE-state discipline, because the two-state version of this
report is actively harmful: "the evidence contradicts your declaration" and "no instrument we have can
see that engine" want opposite responses, and a report that says the first when it means the second
tells a correct contract to delete a real engine.
"""
from __future__ import annotations

import pytest

from merlin.kernels import engines as E
from merlin.targetgen import capability_derive as cd


def _tax(roles: dict[str, list[str]]) -> dict:
    """A taxonomy stub in the shape `isa_taxonomy._classes_by_role` reads: by_class -> [{role}]."""
    return {"by_class": {name: [{"role": role}] for role, names in roles.items() for name in names}}


def _facts(arrays=()) -> dict:
    return {"facts": {"arrays": list(arrays)}}


class TestWhatTheEvidenceReaches:
    def test_array_plumbing_roles_evidence_an_array_engine(self):
        """The roles the FAMILY ladder deliberately discards are the ones that prove a grid exists.

        `_ROLE_FAMILY` omits weight_load / acc_seed / acc_readout as "contraction plumbing ... license
        nothing on their own", which is right about families and exactly wrong about engines: an
        instruction whose job is to push weights into a grid is the strongest evidence a grid is there.
        """
        d = cd.derive_engines("t", {}, taxonomy=_tax({"weight_load": ["WPush"], "scalar": ["ADD"]}))
        assert d.engines() == ["spatial"]
        assert d.evidenced["spatial"].source == "isa_role"
        assert "weight_load" in d.evidenced["spatial"].evidence

    def test_tensor_compute_roles_evidence_a_lane_engine(self):
        d = cd.derive_engines("t", {}, taxonomy=_tax({"tensor_compute_binary": ["VAdd"],
                                                      "matmul": ["MM"]}))
        assert d.engines() == ["spatial", "vector"], "a hybrid must report BOTH, never a primary"

    def test_a_mac_array_in_the_rtl_facts_evidences_an_array_engine(self):
        d = cd.derive_engines("t", {}, _facts([{"name": "mesh", "rows": 16, "cols": 16}]))
        assert d.engines() == ["spatial"]
        assert d.evidenced["spatial"].source == "rtl_facts"
        assert "16x16" in d.evidenced["spatial"].evidence, "the literal observation must be recorded"

    def test_the_declaration_is_not_evidence_for_itself(self):
        """`unit_intent` is a rung for FAMILIES and deliberately not one here.

        The contract's own compute_units is the thing being audited. Reading it back as evidence would
        make every target agree with itself and the report would never say anything.
        """
        contract = {"compute_units": [{"name": "mesh", "kind": "systolic", "ops": ["matmul"]}]}
        d = cd.derive_engines("t", contract)
        assert d.engines() == [], "declaring a unit must not evidence it"
        assert d.rungs == ()


class TestTheThreeStatesAreKeptApart:
    def test_a_scalar_only_census_does_not_count_as_a_compute_census(self):
        """Radiance's real shape: 6 scalar classes and nothing else.

        Counting that as "a rung ran" made the report accuse its contract of over-declaring an engine
        the census had never looked for. A census that read the ISA without covering compute is SILENT
        about engines, not negative about them.
        """
        d = cd.derive_engines("t", {}, taxonomy=_tax({"scalar": ["ADD", "SUB", "BEQ"]}))
        assert d.rungs == (), "a scalar-only census must not register as a rung"
        assert d.observable() == frozenset()

    def test_a_declared_engine_no_rung_can_see_is_unchecked_not_unevidenced(self):
        # muon's real shape: only rtl_facts ran, and that rung can never evidence simt.
        d = cd.derive_engines("t", {}, _facts([{"name": "mesh", "rows": 17, "cols": 17}]))
        drift = cd.reconcile_engines({"simt"}, d)
        assert any(x.startswith("unchecked_engine simt") for x in drift), drift
        assert not any(x.startswith("unevidenced_engine") for x in drift), (
            "a gap in our instruments must never be reported as an over-declaration")
        assert any("not a finding about the hardware" in x for x in drift)

    def test_a_declared_engine_a_capable_rung_missed_is_unevidenced(self):
        # The other side: isa_role CAN see a lane engine, so a lane declaration it did not find is a
        # real disagreement rather than a blind spot.
        d = cd.derive_engines("t", {}, taxonomy=_tax({"matmul": ["MM"]}))
        assert "vector" in d.observable()
        drift = cd.reconcile_engines({"spatial", "vector"}, d)
        assert any(x.startswith("unevidenced_engine vector") for x in drift), drift

    def test_evidence_beyond_the_declaration_is_reported_as_undeclared(self):
        d = cd.derive_engines("t", {}, taxonomy=_tax({"matmul": ["MM"],
                                                      "tensor_compute_unary": ["VExp"]}))
        drift = cd.reconcile_engines({"spatial"}, d)
        assert any(x.startswith("undeclared_engine vector") for x in drift), drift

    def test_agreement_produces_no_drift(self):
        """The control. A report that always complains is one nobody reads."""
        d = cd.derive_engines("t", {}, _facts([{"name": "mesh", "rows": 16, "cols": 16}]))
        assert cd.reconcile_engines({"spatial"}, d) == []


class TestTheStatedInstrumentGap:
    def test_no_rung_can_observe_a_simt_engine(self):
        """Pins the limit rather than leaving it implicit.

        What makes a SIMT engine SIMT — many threads of control over one instruction stream — is not a
        property of any single instruction's typed operands, so a role census cannot reach it, and an
        arrays fact says nothing about it either. Every SIMT declaration is therefore UNCHECKED today.
        When a rung that reads warp/barrier structure lands, this test is the one that should fail.
        """
        for rung, can_see in cd._RUNG_CAN_SEE.items():
            assert "simt" not in can_see, (
                f"rung {rung!r} now claims to observe simt — if that is real, delete this test and "
                f"pin what it observes instead")

    def test_every_rung_declares_what_it_can_see(self):
        # A rung that runs without an entry here would make its findings look total: everything it did
        # not find would read as "a capable rung missed it".
        d = cd.derive_engines("t", {}, _facts([{"name": "mesh", "rows": 8, "cols": 8}]),
                              taxonomy=_tax({"matmul": ["MM"]}))
        for rung in d.rungs:
            assert rung in cd._RUNG_CAN_SEE, f"rung {rung!r} declares no observability"

    def test_every_observable_engine_is_a_real_engine_facet(self):
        seen = {e for caps in cd._RUNG_CAN_SEE.values() for e in caps}
        assert seen <= E.ENGINE_FACETS, (
            f"rungs claim to observe {sorted(seen - E.ENGINE_FACETS)}, which are not engine facets")


class TestAgainstTheRealTargets:
    """The findings this exists to produce. Skips rather than fails where a checkout lacks a target."""

    def _report(self, target):
        from merlin.targetgen import target_registry as tr
        from merlin.targetgen.rtl import facts as F
        try:
            contract = tr.load_contract(target) or {}
        except Exception:                              # noqa: BLE001
            pytest.skip(f"{target} contract not resolvable in this checkout")
        d = cd.derive_engines(target, contract, F.load_facts(target) or {})
        return d, cd.reconcile_engines(E.facet_families_for(target), d)

    def test_a_target_whose_declaration_matches_its_rtl_reports_nothing(self):
        _, drift = self._report("gemmini")
        assert drift == [], f"gemmini's 16x16 mesh matches its declaration; got {drift}"

    def test_atlas_has_an_undeclared_lane_engine(self):
        """The finding the whole exercise is for.

        Atlas declares one systolic unit with ops ('matmul',). Its own ISA census reports tensor-compute
        roles, and 50 of its 137 expert kernels drive that engine exclusively.
        """
        d, drift = self._report("atlas")
        if not d.rungs:
            pytest.skip("atlas ISA taxonomy unavailable in this checkout")
        assert "vector" in d.engines(), d.engines()
        assert any(x.startswith("undeclared_engine vector") for x in drift), drift
