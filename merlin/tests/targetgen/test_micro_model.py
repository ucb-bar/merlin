"""The minimal whole-model inventory is DERIVED, and its shape is the point.

A whole-model capsule small enough to run at the cycle-accurate tier is the only way the end-to-end
claim ever gets checked against hardware. Which layers it must contain is not a matter of taste: a
hand-picked model covers whichever layers its author thought of, and this repo already shipped a corpus
whose only model capsule was a LLaMA decoder -- no tanh, no erf, no convolution, and nothing saying the
evidence was one architecture wide.
"""
from __future__ import annotations

from merlin.targetgen import boundary as BD
from merlin.targetgen import micro_model as MM


def _acc(fam, dtype="i8"):
    return MM.LayerRequirement(family=fam, dtype=dtype, side=MM.ACCELERATOR)


def _host(fam):
    return MM.LayerRequirement(family=fam, dtype=None, side=MM.HOST)


class TestInterleaving:
    def test_host_work_lands_between_accelerator_work(self):
        # A host PREFIX and a host SUFFIX compose as H->A->H, which the corpus already evidences. The
        # shape it evidences nowhere is A->H->A -- accelerator, host island, accelerator -- and that is
        # exactly the placement decision a whole-model compiler gets wrong.
        order = MM.interleave([_acc("contraction"), _acc("movement")], [_host("normalization")])
        assert [l.side for l in order] == [MM.ACCELERATOR, MM.HOST, MM.ACCELERATOR]

    def test_the_sequence_opens_and_closes_on_the_accelerator(self):
        order = MM.interleave([_acc("contraction"), _acc("movement")],
                              [_host("normalization"), _host("reduction"), _host("softmax")])
        assert order[0].side == MM.ACCELERATOR and order[-1].side == MM.ACCELERATOR

    def test_more_host_layers_than_gaps_still_keeps_every_one_an_island(self):
        order = MM.interleave([_acc("contraction"), _acc("movement")],
                              [_host(f"h{i}") for i in range(5)])
        assert order[0].side == MM.ACCELERATOR and order[-1].side == MM.ACCELERATOR
        assert sum(1 for l in order if l.side == MM.HOST) == 5

    def test_a_target_with_no_host_families_gets_no_manufactured_seam(self):
        # A target that admits everything the captures contain has no seam. Inventing one would test a
        # boundary that target does not have, which is worse than reporting there is nothing to test.
        order = MM.interleave([_acc("contraction"), _acc("movement")], [])
        assert all(l.side == MM.ACCELERATOR for l in order)

    def test_a_target_that_admits_nothing_is_all_host(self):
        order = MM.interleave([], [_host("normalization")])
        assert [l.side for l in order] == [MM.HOST]


class TestCompositionShape:
    def test_the_inventory_reports_its_own_composition_in_the_boundary_vocabulary(self):
        # One vocabulary for "how is this composed", shared with the coverage axis, so the model's
        # intended shape and the shape the corpus is measured on can never drift apart.
        s = MM.MicroModelSpec(layers=MM.interleave([_acc("contraction"), _acc("movement")],
                                                   [_host("normalization")]))
        assert s.composition() == BD.A_H_A

    def test_two_host_segments_between_three_accelerator_segments_is_routing(self):
        s = MM.MicroModelSpec(layers=MM.interleave(
            [_acc("contraction"), _acc("elementwise_map"), _acc("movement")],
            [_host("normalization"), _host("reduction")]))
        assert s.composition() == BD.ROUTING
        assert BD.A_H_A in BD.patterns_in_sequence(
            [BD.ACCEL if l.side == MM.ACCELERATOR else BD.HOST for l in s.layers])


class TestSpellingsComeFromCaptures:
    def test_the_spelling_is_the_one_real_models_use(self, tmp_path):
        # Which concrete op stands for a family is a question about real networks, and the captures
        # answer it. Choosing by taste is how a corpus ends up one architecture wide.
        class _R:
            def __init__(self, op, fam):
                self.op, self._f = op, fam

            def resolved_family(self):
                return self._f

        import merlin.targetgen.model_coverage as mc
        seen = {"a": [_R("matmul", "contraction")] * 3 + [_R("conv", "contraction")]}
        old_load, old_regions = mc.load_module, mc.regions_from_module
        try:
            mc.load_module = lambda p: str(p)
            mc.regions_from_module = lambda m: tuple(seen["a"])
            got = MM.observed_spellings({"a": tmp_path / "m.mlir"})
        finally:
            mc.load_module, mc.regions_from_module = old_load, old_regions
        assert got["contraction"].most_common(1)[0] == ("matmul", 3)

    def test_an_unreadable_capture_is_skipped_not_fatal(self, tmp_path):
        assert MM.observed_spellings({"missing": tmp_path / "nope.mlir"}) == {}


class TestSpecKeepsItsUncertainty:
    def test_an_unresolvable_target_reports_rather_than_inventing_layers(self):
        s = MM.spec("definitely_not_a_target", {})
        assert s.layers == []
        # No tile edge means extents cannot be sized against the target's own geometry, and the spec
        # must say so instead of quietly emitting a plausible number.
        assert s.extent is None or s.tile_edge is not None

    def test_a_dict_round_trips_with_the_composition_named(self):
        s = MM.MicroModelSpec(target="t", layers=[_acc("contraction"), _host("reduction"),
                                                  _acc("movement")])
        d = s.to_dict()
        assert d["composition"] == BD.A_H_A
        assert d["n_accelerator_layers"] == 2 and d["n_host_layers"] == 1
