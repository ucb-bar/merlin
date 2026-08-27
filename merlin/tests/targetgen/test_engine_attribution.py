"""Which ENGINE covers a region, not merely whether the target does.

`semantic_capability_map` folds every unit into one `family -> capability` map, and until the
attribution existed it dropped which unit contributed each entry. So the eligibility oracle knew a
target could run `elementwise_map`; it did not know that this happens on the VPU and not on the MXU.
That is the recorded atlas failure (`atlas-0of11-is-agent-not-tooling`): the MXU accumulate never
fired, and nothing in what the agent was handed said which engine owned which family.

The attribution is DERIVED from the declaring unit's kind, never authored, so a contract cannot drift
from its own compute_units.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import compute_units as cu
from merlin.targetgen import eligibility as el

_HYBRID = {"compute_units": [
    {"name": "cluster", "kind": "simt", "contains": ["mesh"], "dtypes": ["fp32"], "ops": ["elementwise"],
     "semantic_capabilities": [{"family": "elementwise_map", "dtypes": ["fp32"]},
                               {"family": "contraction", "dtypes": ["fp32"]}]},
    {"name": "mesh", "kind": "systolic", "dtypes": ["int8"], "ops": ["matmul"],
     "semantic_capabilities": [{"family": "contraction", "dtypes": ["int8"]}]},
]}


def _units(contract=None):
    return cu.compute_units(contract or _HYBRID)


class TestTheFoldNoLongerLosesTheUnit:
    def test_a_family_records_every_unit_that_declares_it(self):
        prov = cu.semantic_engine_map(_units())
        assert prov["contraction"] == (("cluster", "simt"), ("mesh", "systolic"))
        assert prov["elementwise_map"] == (("cluster", "simt"),)

    def test_the_capability_carries_the_engine_kinds(self):
        caps = cu.semantic_capability_map(_units())
        assert set(caps["contraction"].engines) == {"simt", "systolic"}
        assert caps["elementwise_map"].engines == ("simt",)

    def test_containment_does_not_relabel_the_contained_units_work(self):
        """`effective` folds a containing unit's capabilities with what it contains, so attribution
        taken from the OUTER unit would report the mesh's int8 contraction as SIMT work. Provenance
        reads each unit's own declaration instead."""
        caps = cu.semantic_capability_map(_units())
        assert "systolic" in caps["contraction"].engines, "the contained mesh lost its attribution"

    def test_merging_two_units_unions_their_engines(self):
        # Two ways to run a family is a capability, not a conflict -- the opposite of composed_with,
        # which intersects because it is a restriction.
        caps = cu.semantic_capability_map(_units())
        assert len(caps["contraction"].engines) == 2

    def test_authoring_the_attribution_is_refused(self):
        """Derived, never authored. A contract that could write its own attribution could declare that
        a `vector` unit's capability runs on a systolic array, and nothing downstream could contradict
        it -- the drift the derivation exists to prevent. Fail closed at parse time instead."""
        c = {"compute_units": [{"name": "u", "kind": "vector", "dtypes": ["fp32"], "ops": ["matmul"],
                                "semantic_capabilities": [{"family": "contraction", "dtypes": ["fp32"],
                                                           "engines": ["systolic"]}]}]}
        with pytest.raises(ValueError, match="derived from the unit's kind"):
            cu.compute_units(c)


class TestTheVerdictSaysWhereTheWorkLands:
    def test_an_eligible_region_names_its_engines_and_units(self):
        v = el.is_eligible(el.RegionDescriptor(op="matmul", in_dtype="int8"),
                           cu.semantic_capability_map(_units()),
                           providers=cu.semantic_engine_map(_units()))
        assert v.eligible and set(v.engines) == {"simt", "systolic"}
        assert set(v.units) == {"cluster", "mesh"}

    def test_units_are_optional_and_engines_are_not(self):
        # Naming the engine is the load-bearing half; naming the unit is a convenience for a report.
        v = el.is_eligible(el.RegionDescriptor(op="matmul", in_dtype="int8"),
                           cu.semantic_capability_map(_units()))
        assert v.engines and v.units == ()

    def test_an_ineligible_verdict_claims_no_engine(self):
        v = el.is_eligible(el.RegionDescriptor(op="matmul", in_dtype="mxfp4"),
                           cu.semantic_capability_map(_units()))
        assert not v.eligible and v.engines == ()


class TestTheEnginesAxis:
    def test_asking_the_wrong_engine_is_refused_with_the_right_one_named(self):
        """The atlas shape: a region the TARGET can run and this ENGINE cannot."""
        v = el.is_eligible(el.RegionDescriptor(op="gelu", in_dtype="fp32", engine="systolic"),
                           cu.semantic_capability_map(_units()))
        assert not v.eligible
        assert "does not provide" in v.reason and "simt" in v.reason

    def test_asking_the_right_engine_is_allowed(self):
        v = el.is_eligible(el.RegionDescriptor(op="gelu", in_dtype="fp32", engine="simt"),
                           cu.semantic_capability_map(_units()))
        assert v.eligible

    def test_not_asking_about_an_engine_constrains_nothing(self):
        v = el.is_eligible(el.RegionDescriptor(op="gelu", in_dtype="fp32"),
                           cu.semantic_capability_map(_units()))
        assert v.eligible, "the default question is 'can the TARGET run this', unchanged"

    def test_an_undeclared_attribution_admits_every_engine(self):
        """Fail-OPEN here, deliberately, and the audit table must agree. Narrowing an axis that
        currently excludes nothing is the mx_gemmini rank bug: it shrinks the ARR denominator and
        flatters recall."""
        assert el.empty_declaration_is_narrowing("engines") is False
        c = {"compute_units": [{"name": "u", "kind": "vector", "dtypes": ["fp32"], "ops": ["matmul"],
                                "semantic_capabilities": [{"family": "contraction", "dtypes": ["fp32"]}]}]}
        caps = cu.semantic_capability_map(cu.compute_units(c))
        caps["contraction"] = type(caps["contraction"])(family="contraction", dtypes=("fp32",))
        v = el.is_eligible(el.RegionDescriptor(op="matmul", in_dtype="fp32", engine="systolic"), caps)
        assert v.eligible, "an empty engines declaration must not exclude an engine"

    def test_the_axis_declares_its_empty_set_semantics(self):
        # The guard that makes adding an axis safe: it cannot be audited until someone writes this down.
        for axis in ("dtypes", "ranks", "layouts", "engines"):
            assert el.empty_declaration_is_narrowing(axis) in (True, False)
        with pytest.raises(KeyError, match="no declared empty-set semantics"):
            el.empty_declaration_is_narrowing("sparsity")


class TestAgainstTheRealTargets:
    def test_a_hybrid_in_the_tree_attributes_contraction_to_both_engines(self):
        try:
            prov = el.providers_for_target("radiance")
        except Exception:                                  # noqa: BLE001
            pytest.skip("radiance contract not resolvable in this checkout")
        kinds = {k for _, k in prov.get("contraction", ())}
        assert kinds == {"simt", "systolic"}, kinds

    def test_a_simt_only_family_is_refused_on_the_array(self):
        try:
            caps = el.capability_map_for_target("radiance")
        except Exception:                                  # noqa: BLE001
            pytest.skip("radiance contract not resolvable in this checkout")
        if "elementwise_map" not in caps:
            pytest.skip("radiance declares no elementwise_map")
        v = el.is_eligible(el.RegionDescriptor(op="gelu", in_dtype="fp32", engine="systolic"), caps)
        assert not v.eligible, "the MX array cannot run a standalone gelu"


class TestTheTargetsThatHadNoEngines:
    """saturn declared no compute_units at all and atlas declared one, so both were invisible to every
    consumer that asks what engines a target has."""

    def test_saturn_declares_both_of_its_engines(self):
        from merlin.targetgen import compute_units as _cu
        from merlin.targetgen import target_registry as _tr
        try:
            units = _cu.compute_units(_tr.load_contract("saturn"))
        except Exception:                                  # noqa: BLE001
            pytest.skip("saturn contract not resolvable")
        assert {u.kind for u in units} == {"vector", "spatial"}, [(u.name, u.kind) for u in units]

    def test_saturns_array_is_spatial_not_systolic(self):
        """The OPU reduces via rank-1 outer-product accumulate into a tile of accumulator cells, not a
        stationary-weight wavefront — the distinction KINDS draws, and why the two carry separate fact
        families. Calling it systolic would route it to the wrong fact extractor."""
        from merlin.targetgen import compute_units as _cu
        from merlin.targetgen import target_registry as _tr
        try:
            units = _cu.compute_units(_tr.load_contract("saturn"))
        except Exception:                                  # noqa: BLE001
            pytest.skip("saturn contract not resolvable")
        opu = next((u for u in units if u.kind == "spatial"), None)
        assert opu is not None and opu.exposure == "command_buffer", (
            "the OPU is command-buffer driven, not RoCC")

    def test_declaring_saturns_array_reclassifies_the_target(self):
        """Recorded deliberately rather than discovered later. saturn derived NO class while it declared
        no units; declaring them makes the precedence rule pick its accelerator datapath over its lane
        datapath. If that is ever judged wrong, the fix is the precedence rule in kernels.engines, not
        deleting a real engine from the contract."""
        from merlin.kernels import engines as E
        got = E.engines_for("saturn")
        if not got:
            pytest.skip("saturn contract not resolvable")
        assert E.target_class_for(got) == "npu"

    def test_an_undeclared_engine_the_evidence_reaches_is_synthesized(self):
        """The audit reported atlas's vector engine on every run; this is the half that acts on it."""
        from merlin.targetgen import capability_manifests as _cm
        from merlin.targetgen import target_registry as _tr
        from merlin.targetgen.rtl import facts as _F
        try:
            manifest = _tr.load_contract("atlas")
        except Exception:                                  # noqa: BLE001
            pytest.skip("atlas contract not resolvable")
        synth = _cm._derived_units_for_undeclared_engines("atlas", manifest, _F.load_facts("atlas") or {})
        if not synth:
            pytest.skip("atlas ISA taxonomy unavailable in this checkout")
        assert [u["kind"] for u in synth] == ["vector"]
        assert synth[0]["derived_from"].startswith("isa_role:"), "a derived unit must carry its evidence"

    def test_synthesis_never_touches_a_declared_unit(self):
        from merlin.targetgen import capability_manifests as _cm
        m = {"compute_units": [{"name": "vpu", "kind": "vector", "dtypes": ["fp32"], "ops": ["elementwise"],
                                "semantic_capabilities": [{"family": "elementwise_map", "dtypes": ["fp32"]}]}]}
        assert _cm._derived_units_for_undeclared_engines("t", m, {}) == [], (
            "a declared kind must never be synthesized a second time")

    def test_an_ambiguous_facet_synthesizes_nothing(self):
        """`spatial` maps to two kinds and a role census cannot tell them apart, so synthesizing either
        would assert a datapath nobody observed."""
        from merlin.targetgen import capability_manifests as _cm
        assert _cm._kind_for_facet("spatial") is None
        assert _cm._kind_for_facet("vector") == "vector"
        assert _cm._kind_for_facet("simt") == "simt"
