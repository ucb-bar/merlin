"""The engine model: a target is a SET of compute engines, and each engine picks a CCA facet.

The measurement that motivated this is worth restating, because the tests below are only interesting
if you believe it. Bucketing atlas's 180 hand-written expert kernels by the roles its DERIVED IsaModel
assigns: 26 drive only the matrix array, 50 drive only the vector engine, 35 drive both, 26 drive
neither. Filing atlas as "an NPU" and giving it one facet leaves 62% of its engine-driving corpus
undescribed. So the invariants here are about never collapsing the set, and about the three
vocabularies for "what kind of machine is this" being one relationship rather than three opinions.
"""
from __future__ import annotations

import pytest

from merlin.kernels import engines as E


class TestTheMappingIsTotalBothWays:
    """A kind with no facet describes nothing; a facet for a kind that cannot occur is dead."""

    def test_every_compute_unit_kind_has_a_facet_decision(self):
        from merlin.targetgen.compute_units import KINDS

        E.check_covers_kinds(KINDS)          # raises with the missing/stale names if not

    def test_a_new_kind_with_no_mapping_is_refused(self):
        # The failure mode this prevents: a kind is added to KINDS, nothing here mentions it, and it
        # silently becomes "an engine that no facet describes" rather than an error.
        with pytest.raises(KeyError, match="no engine->facet mapping"):
            E.check_covers_kinds(list(E.ENGINE_FACET) + ["quantum_mesh"])

    def test_a_mapping_for_a_kind_that_no_longer_exists_is_refused(self):
        with pytest.raises(KeyError, match="not compute-unit kinds any more"):
            E.check_covers_kinds([k for k in E.ENGINE_FACET if k != "simt"])

    def test_an_unknown_kind_raises_rather_than_answering_none(self):
        # "no facet" and "never heard of it" are different answers and only one is safe to act on.
        assert E.facet_for("scalar") is None, "scalar is a KNOWN non-engine"
        with pytest.raises(KeyError, match="unknown compute-unit kind"):
            E.facet_for("scalar_core")

    def test_the_two_array_datapaths_share_one_facet(self):
        # systolic and spatial are separate KINDS because they carry separate RTL fact families, but a
        # lifter asks the same question of both (tile edge, dataflow, accumulator residency). The
        # distinction survives as a VALUE of spatial.dataflow, not as a second facet.
        assert E.facet_for("systolic") == E.facet_for("spatial") == "spatial"

    def test_a_lane_engine_and_a_simt_engine_are_not_the_same_facet(self):
        # The collapse most worth refusing: element-parallelism within one thread of control has a VL
        # and no divergence; threads of control have barriers and no VL. vector.lmul has no SIMT
        # analogue and barrier placement has no lane analogue.
        assert E.facet_for("vector") != E.facet_for("simt")


class TestTheSetIsNeverCollapsed:
    def test_a_hybrid_reports_every_engine_it_has(self):
        from merlin.targetgen.compute_units import compute_units

        units = compute_units({"compute_units": [
            {"name": "cluster", "kind": "simt", "contains": ["mesh"]},
            {"name": "mesh", "kind": "systolic"},
        ]})
        assert E.engines_of_units(units) == frozenset({"simt", "systolic"})

    def test_composition_adds_engines_and_never_removes_them(self):
        # `contains` is a reason a target has MORE engines. A containing unit that hid what it embeds
        # would describe a hybrid as whichever half happened to be outermost.
        from merlin.targetgen.compute_units import compute_units

        units = compute_units({"compute_units": [
            {"name": "cluster", "kind": "simt", "contains": ["mesh"]},
            {"name": "mesh", "kind": "systolic"},
        ]})
        got = E.engines_of_units(units)
        assert "systolic" in got, "the contained mesh disappeared"
        # ...and the facet view follows the whole set, so the hybrid gets BOTH facets.
        assert {E.facet_for(k) for k in got} == {"simt", "spatial"}

    def test_a_target_that_declares_nothing_gets_an_empty_set_not_a_default(self):
        # An empty set means "nobody has said what silicon this is". Defaulting it would let an
        # undeclared accelerator be described by whatever facet happened to be first.
        assert E.engines_for("a-target-that-does-not-exist") == frozenset()
        assert E.target_class_for(frozenset()) is None

    def test_a_declared_hybrid_in_the_tree_really_is_one(self):
        # Guards the whole premise against a tree where every target is single-engine: if this ever
        # returns one engine, either the contract regressed or composition stopped being read.
        got = E.engines_for("radiance")
        if not got:
            pytest.skip("radiance contract not resolvable in this checkout")
        assert len(got) > 1, f"radiance should declare a SIMT cluster containing a mesh; got {sorted(got)}"


class TestTheThreeVocabulariesAgree:
    """`targetgen.families` used to claim in prose that KINDS is 'aligned with' TargetClass. Five
    tokens cannot align with three by inspection, so the claim is a check now."""

    def test_kinds_to_target_class_is_total_and_every_class_is_reachable(self):
        from merlin.runtime.backends.base import TargetClass
        from merlin.targetgen.compute_units import KINDS

        E.check_class_map_is_total(KINDS, [c.value for c in TargetClass])

    def test_a_class_no_engine_can_produce_is_refused(self):
        # The other direction, which matters as much: a class the runtime accepts but the compiler can
        # never derive is exactly the drift this map replaces.
        from merlin.targetgen.compute_units import KINDS

        with pytest.raises(KeyError, match="reachable from no compute-unit kind"):
            E.check_class_map_is_total(KINDS, ["cpu", "gpu", "npu", "dpu"])

    def test_every_kind_has_a_precedence_rank(self):
        # Without one, a hybrid's class would depend on set iteration order.
        from merlin.targetgen.compute_units import KINDS

        for kind in KINDS:
            assert E.target_class_for([kind]) is not None, kind

    def test_a_simt_cluster_containing_a_mesh_is_a_gpu_not_an_npu(self):
        # The one place a "primary engine" is the RIGHT question: which single class of device is this.
        # (It is the WRONG question for which facets or which RTL facts a target has -- that collapse
        # is what describes only half of a hybrid.)
        assert E.target_class_for({"simt", "systolic"}) == "gpu"
        assert E.target_class_for({"systolic"}) == "npu"
        assert E.target_class_for({"vector", "scalar"}) == "cpu"

    def test_the_runtime_derives_the_same_answer(self):
        from merlin.runtime.backends.base import TargetClass, target_class_for

        got = target_class_for("gemmini")
        if got is None:
            pytest.skip("gemmini contract not resolvable in this checkout")
        assert got is TargetClass.NPU


class TestFacetsFollowEngines:
    def test_an_engine_scoped_facet_exists_for_every_engine(self):
        for kind, facet in E.ENGINE_FACET.items():
            if facet is None:
                continue
            assert facet in E.ENGINE_FACETS, kind

    def test_every_engine_facet_is_a_real_cca_facet(self):
        # Catches a mapping that names a facet the schema does not have -- which would make
        # facet_families_for return something no lifter could ever populate.
        from merlin.kernels.cca_contract import FACET_CLASSES

        assert E.ENGINE_FACETS <= set(FACET_CLASSES), (
            f"engine facets {sorted(E.ENGINE_FACETS - set(FACET_CLASSES))} are not in the CCA schema")

    def test_every_cca_facet_is_declared_engine_scoped_or_agnostic(self):
        # The decision a new facet must not skip. Defaulting an unclassified facet to agnostic would
        # populate it for targets whose silicon cannot exhibit it.
        from merlin.kernels.cca_contract import FACET_CLASSES

        E.check_facets_are_classified(FACET_CLASSES)

    def test_a_new_unclassified_facet_is_refused(self):
        with pytest.raises(KeyError, match="neither engine-scoped nor agnostic"):
            E.check_facets_are_classified(list(E.ENGINE_FACETS | E.AGNOSTIC_FACETS) + ["tensorcore"])

    def test_a_facet_classified_here_but_absent_from_the_schema_is_refused(self):
        with pytest.raises(KeyError, match="not in the CCA schema"):
            E.check_facets_are_classified(E.AGNOSTIC_FACETS)

    def test_every_facet_named_family_tag_is_one_the_engine_model_knows(self):
        """The link between the bijection contract and the engine model.

        FIELD_REGISTRY tags a field either with a concrete backend ("rvv") or with a FAMILY, and a
        family tag is a facet name ("spatial", "simt", "compute"). A field tagged with a facet the
        engine model has never classified would be leverable for targets whose silicon cannot exhibit
        it -- the exact over-reach that gating family inheritance on `routed` exists to prevent.
        """
        from merlin.kernels.cca_contract import FACET_CLASSES, FIELD_REGISTRY

        tags = {b for spec in FIELD_REGISTRY.values() for b in spec.backends}
        classified = E.ENGINE_FACETS | E.AGNOSTIC_FACETS
        for tag in sorted(tags & set(FACET_CLASSES)):
            assert tag in classified, (
                f"{tag!r} is used as a FIELD_REGISTRY family tag and is a CCA facet, but the engine "
                f"model classifies it as neither engine-scoped nor agnostic")

    def test_the_lane_facet_is_still_target_tagged_not_family_tagged(self):
        # Records a KNOWN gap rather than asserting the end state. vector.* is an ENGINE facet but its
        # fields are tagged ("rvv",) -- a target name -- so an accelerator's own lane engine (atlas's
        # VPU) cannot pick them up the way a systolic target picks up spatial.*. Generalizing this is
        # what makes the second lane engine work; when it lands, this test flips to the assertion
        # above and this one goes away.
        from merlin.kernels.cca_contract import FIELD_REGISTRY

        vector_tags = {b for k, s in FIELD_REGISTRY.items() if s.axis.startswith("vector.")
                       for b in s.backends}
        assert "vector" not in vector_tags, (
            "vector.* is now family-tagged -- good: delete this test and rely on the one above")

    def test_scalar_is_not_an_engine_facet(self):
        # A scalar core is the code AROUND the loop (EnvelopeFacet's subject), and that facet is
        # engine-agnostic: every target has one. Mapping scalar to it would imply a target with a
        # scalar core has an engine-scoped facet that nothing else does.
        assert E.facet_for("scalar") is None
        assert "envelope" not in E.ENGINE_FACETS
