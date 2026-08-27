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


def _mesh(rows, cols, *, element="Tile", instances=None, container="Mesh", corroborated=True) -> dict:
    """A mesh fact shaped the way the extractor now emits one: geometry PLUS the corroboration."""
    return {"name": "mesh", "rows": rows, "cols": cols, "source": "mlc_discovery",
            "element": element, "instances": instances if instances is not None else rows * cols,
            "container": container, "element_variants": [[element, rows * cols]],
            "mac_idiom": {"muls": 1, "adds": 7, "regs": 3}, "corroborated": corroborated}


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
        d = cd.derive_engines("t", {}, _facts([_mesh(16, 16)]))
        assert d.engines() == ["spatial"]
        assert d.evidenced["spatial"].source == "rtl_facts"

    def test_the_evidence_quotes_the_observation_not_the_conclusion(self):
        """An earlier version of this test asserted `"16x16" in evidence` and called that "the literal
        observation". It is not: 16x16 is what was CONCLUDED. The observation is which element was found
        replicated, how many times, and where -- and only that can be refuted by a reader who knows the
        hardware. Checking the conclusion is how a 17x17 mesh derived from 289 flip-flops inside a
        divide/sqrt unit passed review looking exactly like a real one."""
        d = cd.derive_engines("t", {}, _facts([_mesh(16, 16, element="Tile", container="Mesh")]))
        ev = d.evidenced["spatial"].evidence
        assert "256 instances of 'Tile'" in ev, ev
        assert "in 'Mesh'" in ev, ev
        assert "mac idiom" in ev, ev

    def test_a_deduplicated_grid_reports_how_many_variants_were_summed(self):
        """CIRCT splits a mesh whose partial-sum bus widens down the column into several near-identical
        modules (measured: 128 + 80 + 32 + 16 = 256). The sum is the grid, but a reader must be able to
        see that a sum happened rather than trusting a bare count."""
        fact = _mesh(16, 16)
        fact["element_variants"] = [["Tile", 128], ["Tile_160", 80], ["Tile_128", 32], ["Tile_240", 16]]
        d = cd.derive_engines("t", {}, _facts([fact]))
        assert "across 4 structural variants" in d.evidenced["spatial"].evidence

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
        d = cd.derive_engines("t", {}, _facts([_mesh(16, 16)]))
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
        d = cd.derive_engines("t", {}, _facts([_mesh(16, 16)]))
        assert cd.reconcile_engines({"spatial"}, d) == []


class TestAnUncorroboratedReadingBlamesTheInstrument:
    """The fourth state. Every other outcome locates the disagreement in the CONTRACT; this one has to
    be able to say the extractor is wrong, because measured, it was.

    The case: a SIMT target's RTL facts declared a 17x17 mesh. There is no mesh. 289 D-flip-flops inside
    a floating-point divide/sqrt unit happened to be a perfect square, and the extractor's rule was "the
    largest perfect-square repeated instantiation is the grid" -- a rule with no floor, which therefore
    returned a mesh for every design ever handed to it. The audit reported `undeclared_engine spatial`
    with total confidence, and it would have been substantively right about that target for entirely the
    wrong reason, which is the worst way to be right.
    """

    def _bare(self, rows=17, cols=17):
        # What the extractor used to emit: geometry and nothing else.
        return _facts([{"name": "mesh", "rows": rows, "cols": cols, "source": "mlc_discovery"}])

    def test_geometry_without_corroboration_is_not_evidence(self):
        d = cd.derive_engines("t", {}, self._bare())
        assert d.engines() == [], "a bare rows x cols must not evidence an engine"
        assert "spatial" in d.suspect

    def test_an_uncorroborated_reading_does_not_register_as_a_rung(self):
        """Otherwise it poisons the OTHER states: a declared engine would come back `unevidenced`
        ("a rung that CAN see this ran and missed it"), telling a correct contract to delete a real
        engine on the strength of a reading we already know we cannot trust."""
        d = cd.derive_engines("t", {}, self._bare())
        assert d.rungs == ()
        drift = cd.reconcile_engines({"spatial"}, d)
        assert not any(x.startswith("unevidenced_engine") for x in drift), drift

    def test_it_is_reported_against_the_extractor_not_the_contract(self):
        d = cd.derive_engines("t", {}, self._bare())
        drift = cd.reconcile_engines({"simt"}, d)
        suspect = [x for x in drift if x.startswith("suspect_evidence spatial")]
        assert suspect, drift
        assert "OUR extractor" in suspect[0]
        assert not any(x.startswith("undeclared_engine") for x in drift), (
            "an uncorroborated reading must never be reported as the contract under-declaring")

    def test_the_suspect_observation_is_quoted_so_it_can_be_refuted(self):
        d = cd.derive_engines("t", {}, self._bare())
        assert "17x17" in d.suspect["spatial"]
        assert "NO multiply-accumulate idiom confirmed" in d.suspect["spatial"]

    def test_a_corroborated_reading_is_never_suspect(self):
        d = cd.derive_engines("t", {}, _facts([_mesh(16, 16)]))
        assert d.suspect == {}
        assert cd.reconcile_engines({"spatial"}, d) == []

    def test_the_audit_surfaces_suspicion_in_its_record(self):
        # It has to reach the manifest, not just the return value -- an instrument fault nobody reads
        # is the same as no instrument fault.
        d = cd.derive_engines("t", {}, self._bare())
        assert d.to_dict()["engines_suspect"] == [
            {"engine": "spatial", "observation": d.suspect["spatial"]}]


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
        d = cd.derive_engines("t", {}, _facts([_mesh(8, 8)]),
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
