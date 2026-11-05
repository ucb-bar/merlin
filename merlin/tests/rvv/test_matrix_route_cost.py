"""The geometry-derived routing cost model: what it derives, what it decides, and that it says why.

Two geometries are used because they are the two the package plugins declare today -- a 64-edge unit on a
256-bit datapath and a 32-edge one on a 128-bit datapath -- but nothing here depends on either: each
assertion is about a relation (a full tile routes, a padded one routes only when its estimate wins, the
swap is taken only when it is cheaper including its transposes), so a third geometry is a new row, not a
new rule.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import int8_contractions as IC
from merlin.llvmlower import matrix_route_cost as MC
from merlin.llvmlower import matrix_route_report as MR

WIDE, NARROW = MC.UnitGeometry(64, 32), MC.UnitGeometry(32, 16)


def _shape(*parallel, k):
    return SimpleNamespace(parallel=tuple(parallel), reduction=(k,))


class TestTheGeometryIsDerivedNotTabulated:
    @pytest.mark.parametrize("geom,beats", [(WIDE, 2), (NARROW, 2), (MC.UnitGeometry(64, 16), 4)])
    def test_beats_are_vector_length_over_datapath(self, geom, beats):
        assert geom.beats == beats
        assert geom.step_cycles == max(beats * beats, 2 * beats, 3)

    def test_an_alignment_wider_than_a_vector_is_one_beat(self):
        assert MC.UnitGeometry(32, 128).beats == 1

    def test_nonsense_is_refused(self):
        with pytest.raises(ValueError):
            MC.UnitGeometry(0, 16)

    def test_the_description_names_its_inputs(self):
        d = MC.CostModelSelector(WIDE).describe()
        assert d["policy"] == "cost_model" and d["tile_edge"] == 64 and d["datapath_bytes"] == 32
        assert d["derived"]["vlen_bits"] == 512 and d["derived"]["dlen_bits"] == 256


class TestWhatItDecides:
    @pytest.mark.parametrize("geom", [WIDE, NARROW])
    def test_whole_tiles_route_and_say_so(self, geom):
        e = geom.tile_edge
        d = MC.CostModelSelector(geom).decide(_shape(4 * e, 2 * e, k=256))
        assert d.route and d.orientation == MC.AS_IS
        assert d.chosen.utilisation == 1.0 and "fills whole tiles" in d.reason

    def test_a_tall_skinny_output_is_padded_and_the_padding_is_recorded(self):
        # The case the threshold rule refused: N=14 on a 64-wide tile, with a deep enough reduction.
        d = MC.CostModelSelector(WIDE).decide(_shape(2048, 14, k=256))
        assert d.route
        assert "padded N 14->64" in d.reason and "est. util 22%" in d.reason

    def test_too_little_work_stays_on_the_vector_unit(self):
        d = MC.CostModelSelector(WIDE).decide(_shape(8, 8, k=8))
        assert not d.route and d.reason.startswith("cost model")

    def test_more_reduction_never_makes_the_unit_cheaper_per_mac(self):
        sel = MC.CostModelSelector(NARROW)
        a, b = sel.decide(_shape(100, 20, k=64)).chosen, sel.decide(_shape(100, 20, k=4096)).chosen
        assert b.unit_cycles > a.unit_cycles and b.speedup >= a.speedup

    def test_the_swap_is_taken_only_when_it_pays_for_its_transposes(self):
        free = MC.decide(1, 4096, 8, 64, WIDE, layout_elements={MC.AS_IS: 0, MC.SWAPPED: 0})
        assert free.orientation == MC.SWAPPED  # far fewer rows to read out
        priced = MC.decide(1, 4096, 8, 64, WIDE, layout_elements={MC.AS_IS: 0, MC.SWAPPED: 4096 * 64 * 2})
        assert priced.orientation == MC.AS_IS
        assert {e.orientation for e in (priced.chosen, *priced.alternatives)} == {MC.AS_IS, MC.SWAPPED}

    def test_an_orientation_not_offered_is_not_priced(self):
        d = MC.decide(1, 4096, 8, 64, WIDE, layout_elements={MC.AS_IS: 0})
        assert d.orientation == MC.AS_IS and d.alternatives == ()
        with pytest.raises(ValueError):
            MC.decide(1, 64, 64, 64, WIDE, layout_elements={})

    def test_a_batch_scales_both_sides(self):
        one = MC.estimate(1, 64, 64, 64, WIDE)
        many = MC.estimate(8, 64, 64, 64, WIDE)
        assert many.unit_cycles == pytest.approx(8 * one.unit_cycles)
        assert many.vector_cycles == pytest.approx(8 * one.vector_cycles)


_TALL_SKINNY = """
builtin.module {
  func.func @forward(%a: tensor<2048x256xi8>, %b: tensor<256x14xi8>, %c: tensor<4x4xi8>, %d: tensor<4x4xi8>)
      -> (tensor<2048x14xi32>, tensor<4x4xi32>) {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<2048x14xi32>
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<2048x14xi32>) -> tensor<2048x14xi32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>,
                                          affine_map<(d0, d1, d2) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<2048x256xi8>, tensor<256x14xi8>) outs(%f : tensor<2048x14xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
    } -> tensor<2048x14xi32>
    %e2 = tensor.empty() : tensor<4x4xi32>
    %f2 = linalg.fill ins(%z : i32) outs(%e2 : tensor<4x4xi32>) -> tensor<4x4xi32>
    %r2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>,
                                           affine_map<(d0, d1, d2) -> (d0, d1)>],
                          iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%c, %d : tensor<4x4xi8>, tensor<4x4xi8>) outs(%f2 : tensor<4x4xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
    } -> tensor<4x4xi32>
    func.return %r, %r2 : tensor<2048x14xi32>, tensor<4x4xi32>
  }
}
"""


class TestTheDecisionShipsInTheSidecar:
    @pytest.fixture
    def rewrite(self):
        return IC.rewrite_contractions(
            parse_mlir_text(_TALL_SKINNY), select=MC.CostModelSelector(WIDE), tile_edge=64, symbol_prefix="cost_test"
        )

    def test_the_sub_tile_route_is_explained_not_warned(self, rewrite):
        (r,) = rewrite.routed
        assert (r.m, r.n) == (2048, 14) and r.decision["route"] is True
        assert rewrite.sub_tile() == (r,) and rewrite.unexplained_sub_tile() == ()

    def test_the_declined_one_carries_its_estimate(self, rewrite):
        (n,) = rewrite.not_routed
        assert n.why.startswith("cost model") and n.decision["estimate"]["vector_cycles"] > 0

    def test_the_report_groups_by_site_and_by_decision(self, rewrite):
        d = rewrite.to_dict()
        assert d["routing_rule"]["policy"]["policy"] == "cost_model"
        assert d["routing_rule"]["sub_tile_unexplained"] == 0
        sites = MR.by_site(d["routed"], d["not_routed"])
        assert sites["unknown"]["routed"] == 2048 * 14 * 256 and sites["unknown"]["not_routed"] == 4 * 4 * 4
        assert list(sites["unknown"]["not_routed_by_reason"]) == ["cost model"]
        priced = MR.priced_routes(d["routed"])
        assert priced["count"] == 1 and "padded N 14->64" in priced["by_decision"][0]["decision"]

    def test_the_historical_policy_still_refuses_it(self):
        sel = MC.TileFillingPolicy(IC.tile_filling_selector(64), 64)
        got = IC.rewrite_contractions(parse_mlir_text(_TALL_SKINNY), select=sel, tile_edge=64, symbol_prefix="t")
        assert got.count == 0 and all(r.why.startswith("smaller than one tile") for r in got.not_routed)
        assert got.to_dict()["routing_rule"]["policy"] == {"policy": "tile_filling", "tile_edge": 64}
