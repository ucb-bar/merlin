"""Candidate routing, the per-unit exposure axis, and the cost models that choose between them.

Two claims are load-bearing here and both are tested as properties rather than as expectations.

The first is that the refactor is inert: ``route`` must agree with the old first-legal-unit behaviour for
every demand, including gapped ones. It is written as a wrapper over ``select(route_candidates(...))``, so
the test checks the wrapper really does reproduce declaration order rather than checking a handful of
cases that happen to be in the suite.

The second is that a cost model has to be able to decline. An unmeasured unit that scored *well* would win
routing decisions on the strength of having no data, and an unmeasured unit that scored *badly* would be
ruled out for the same reason — so declining is a third outcome, and a demand whose only legal unit is
unmeasured still has to route somewhere and be honest that it did so unscored.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import compute_units as CU
from merlin.targetgen import routing as R

_MATMUL = ("matmul",)


def _unit(name, kind, *, dtypes=("int8",), ops=_MATMUL, exposure=None, contains=()):
    return CU.ComputeUnit(name=name, kind=kind, dtypes=tuple(dtypes), ops=tuple(ops),
                          accumulate=(CU.AccumRule("int8", "int8", "i32"),),
                          exposure=exposure, contains=tuple(contains))


#: A hybrid target: both units can legally run an int8 matmul, which is the case the old router cannot
#: express a preference about.
def _hybrid():
    return [_unit("vec", "vector"), _unit("mat", "spatial")]


def _demand(m=None, n=None, k=None, op="matmul", in_fmt="int8", weight_fmt="int8", site=""):
    return R.OpDemand(op=op, in_fmt=in_fmt, weight_fmt=weight_fmt, site=site, m=m, n=n, k=k)


class TestExposureIsPerUnit:
    def test_a_units_own_declaration_wins(self):
        u = _unit("mat", "spatial", exposure="inline_asm_insn")
        assert CU.resolve_exposure(u) == "inline_asm_insn"

    def test_the_target_endpoint_is_used_when_the_unit_is_silent(self):
        u = _unit("mat", "spatial")
        assert CU.resolve_exposure(u, target_endpoint_kind="external_backend") == "external_backend"

    def test_the_family_default_is_the_last_resort(self):
        # This is what makes the change inert: a unit that declares nothing, on a target that declares
        # nothing, resolves exactly as it did before the axis existed.
        from merlin.targetgen import families
        for kind in sorted(CU.KINDS):
            expected = families.family_profile(kind).endpoint_kind_default
            assert CU.resolve_exposure(_unit("u", kind)) == expected

    def test_a_unit_may_contradict_its_family(self):
        # The whole point: a spatial datapath defaults to command_buffer, but software drives THIS one
        # with vector instructions. A taxonomy that cannot express that cannot express the machine.
        from merlin.targetgen import families
        assert families.family_profile("spatial").endpoint_kind_default == "command_buffer"
        assert CU.resolve_exposure(_unit("mat", "spatial", exposure="inline_asm_insn")) == \
            "inline_asm_insn"

    def test_two_units_can_have_different_exposures_at_once(self):
        units = [_unit("vec", "vector", exposure="upstream_target"),
                 _unit("mat", "spatial", exposure="inline_asm_insn")]
        got = {c.unit: c.exposure
               for e in R.route_candidates([_demand()], units) for c in e.candidates}
        assert got == {"vec": "upstream_target", "mat": "inline_asm_insn"}

    def test_an_unknown_exposure_is_refused_at_parse_time(self):
        with pytest.raises(ValueError, match="exposure"):
            CU.compute_units({"compute_units": [
                {"name": "u", "kind": "spatial", "dtypes": ["int8"], "exposure": "telepathy"}]})

    def test_a_declared_exposure_survives_parsing(self):
        units = CU.compute_units({"compute_units": [
            {"name": "u", "kind": "spatial", "dtypes": ["int8"], "exposure": "inline_asm_insn"}]})
        assert units[0].exposure == "inline_asm_insn"

    def test_composition_does_not_inherit_a_childs_exposure(self):
        # Composition unions capability, not exposure. Inheriting it would silently retarget the parent.
        child = _unit("inner", "spatial", exposure="command_buffer")
        parent = _unit("outer", "simt", exposure="inline_asm_insn", contains=("inner",))
        got = CU.effective(parent, [parent, child])
        assert got.exposure == "inline_asm_insn"

    def test_composition_still_unions_capability(self):
        child = _unit("inner", "spatial", dtypes=("int8",), ops=("conv",))
        parent = _unit("outer", "simt", dtypes=("int8",), ops=("matmul",), contains=("inner",))
        got = CU.effective(parent, [parent, child])
        assert set(got.ops) == {"matmul", "conv"}


class TestCandidatesAreEnumerated:
    def test_every_legal_unit_is_returned_not_just_the_first(self):
        got = R.route_candidates([_demand()], _hybrid())
        assert [c.unit for c in got[0].candidates] == ["vec", "mat"]

    def test_declaration_order_is_preserved(self):
        got = R.route_candidates([_demand()], list(reversed(_hybrid())))
        assert [c.unit for c in got[0].candidates] == ["mat", "vec"]

    def test_an_unroutable_demand_is_gapped_with_a_reason(self):
        got = R.route_candidates([_demand(in_fmt="fp4", weight_fmt="fp4")], _hybrid())
        assert got[0].is_gapped and not got[0].candidates
        assert "no compute unit supports" in got[0].gap

    def test_the_accumulator_comes_from_the_matched_rule(self):
        got = R.route_candidates([_demand()], _hybrid())
        assert all(c.acc == "i32" for c in got[0].candidates)

    def test_a_shapeless_demand_is_still_routable(self):
        # Legality does not need extents; only the cost model does.
        assert not R.route_candidates([_demand()], _hybrid())[0].is_gapped


class TestTheWrapperIsInert:
    @pytest.mark.parametrize("units", [
        _hybrid(), list(reversed(_hybrid())), [_unit("only", "vector")],
        [_unit("v", "vector", dtypes=("int8",))],
    ])
    def test_route_picks_the_first_candidate(self, units):
        demands = [_demand(), _demand(m=1, n=64, k=64), _demand(in_fmt="fp4", weight_fmt="fp4")]
        cands = R.route_candidates(demands, units)
        for result, entry in zip(R.route(demands, units), cands, strict=True):
            if entry.is_gapped:
                assert result.unit is None and result.gap == entry.gap
            else:
                assert result.unit == entry.candidates[0].unit
                assert result.acc == entry.candidates[0].acc

    def test_route_still_reports_gaps_the_same_way(self):
        got = R.route([_demand(in_fmt="fp4", weight_fmt="fp4", site="w0")], _hybrid())
        assert got[0].gap is not None and "[w0]" in got[0].gap
        assert not R.is_fully_routed(got) and len(R.gaps(got)) == 1


class TestTheEagerBaseline:
    def test_it_prefers_the_matrix_unit_regardless_of_shape(self):
        for shape in [(1, 1, 64), (256, 256, 256)]:
            got = R.select(R.route_candidates([_demand(*shape)], _hybrid()), R.eager_cost)
            assert got[0].unit == "mat"

    def test_it_is_registered_so_an_ablation_can_name_it(self):
        assert R.COST_MODELS["eager"] is R.eager_cost
        assert R.COST_MODELS["first"] is R.first_candidate_cost


class TestTheMeasuredModel:
    def _model(self, **kw):
        base = dict(macs_per_cycle={"vec": 4.0, "mat": 256.0},
                    dispatch_cycles={"mat": 200.0, "vec": 0.0},
                    pack_cycles_per_element={"mat": 2.0},
                    requires_k_major=frozenset({"mat"}),
                    tile_edge={"mat": 32})
        base.update(kw)
        return R.MeasuredCost(**base)

    def test_a_large_square_contraction_routes_to_the_matrix_unit(self):
        got = R.select(R.route_candidates([_demand(256, 256, 256)], _hybrid()), self._model())
        assert got[0].unit == "mat"

    def test_a_rank_one_contraction_routes_away_from_the_matrix_unit(self):
        # M=N=1 occupies one cell of a 32x32 tile. The census found these in quantity and
        # arithmetically negligible, and a cost model is expected to decline them.
        got = R.select(R.route_candidates([_demand(1, 1, 64)], _hybrid()), self._model())
        assert got[0].unit == "vec"

    def test_a_partly_filled_tile_costs_a_full_one(self):
        # THE reason a narrow extent is expensive: M=1 and M=32 cost the same on a tiled unit, because
        # the tile is the unit of work. Without this, `macs/peak_rate` credits the unit with work it
        # never did and every narrow shape looks good on it.
        model = self._model(pack_cycles_per_element={}, requires_k_major=frozenset())
        cand = R.Candidate("mat", "spatial", "i32", "x")
        assert model(_demand(1, 8, 64), cand) == model(_demand(32, 32, 64), cand)

    def test_crossing_a_tile_boundary_costs_another_tile(self):
        # One extra row doubles the tiles. Dispatch is excluded here because it is a fixed additive term
        # and would dilute the factor being asserted.
        model = self._model(pack_cycles_per_element={}, requires_k_major=frozenset(),
                            dispatch_cycles={})
        cand = R.Candidate("mat", "spatial", "i32", "x")
        assert model(_demand(33, 32, 64), cand) == 2 * model(_demand(32, 32, 64), cand)

    def test_an_untiled_unit_is_costed_elementwise(self):
        # A vector unit sets its lanes per operation, so twice the work costs twice as much -- there is
        # no tile to leave half-empty. (vec has no dispatch overhead here, so cost is pure proportion.)
        model = self._model()
        cand = R.Candidate("vec", "vector", "i32", "x")
        assert model(_demand(1, 8, 64), cand) == 2 * model(_demand(1, 4, 64), cand)
        assert model(_demand(2, 8, 64), cand) == 2 * model(_demand(1, 8, 64), cand)

    def test_the_packing_cost_can_change_the_decision(self):
        # Same shape, same throughputs; only the layout charge differs. If packing were free the matrix
        # unit would win, which is how a routing decision comes out in favour of a unit that then spends
        # more time rearranging memory than computing.
        shape = _demand(64, 8, 512)
        cheap = R.select(R.route_candidates([shape], _hybrid()),
                         self._model(pack_cycles_per_element={"mat": 0.0}))
        dear = R.select(R.route_candidates([shape], _hybrid()),
                        self._model(pack_cycles_per_element={"mat": 40.0}))
        assert cheap[0].unit == "mat" and dear[0].unit == "vec"

    def test_an_unmeasured_unit_is_declined_not_scored(self):
        model = self._model(macs_per_cycle={"vec": 4.0})       # "mat" absent
        assert model(_demand(256, 256, 256), R.Candidate("mat", "spatial", "i32", "x")) is None

    def test_a_declined_unit_does_not_win_by_having_no_data(self):
        model = self._model(macs_per_cycle={"vec": 4.0})
        got = R.select(R.route_candidates([_demand(256, 256, 256)], _hybrid()), model)
        assert got[0].unit == "vec"

    def test_a_shapeless_demand_is_declined_rather_than_given_a_default_shape(self):
        assert self._model()(_demand(), R.Candidate("mat", "spatial", "i32", "x")) is None

    def test_a_nonpositive_rate_is_declined(self):
        model = self._model(macs_per_cycle={"mat": 0.0})
        assert model(_demand(8, 8, 8), R.Candidate("mat", "spatial", "i32", "x")) is None


class TestSelectionSemantics:
    def test_a_demand_whose_only_legal_unit_is_unmeasured_still_routes(self):
        # Dropping it would turn a missing measurement into a routing gap, i.e. report a capability the
        # target does not have.
        units = [_unit("mat", "spatial")]
        model = R.MeasuredCost(macs_per_cycle={})
        got = R.select(R.route_candidates([_demand(8, 8, 8)], units), model)
        assert got[0].unit == "mat" and got[0].gap is None

    def test_ties_fall_back_to_declaration_order(self):
        model = R.MeasuredCost(macs_per_cycle={"vec": 1.0, "mat": 1.0})
        got = R.select(R.route_candidates([_demand(8, 8, 8)], _hybrid()), model)
        assert got[0].unit == "vec"

    def test_a_gapped_demand_stays_gapped_under_every_model(self):
        d = [_demand(in_fmt="fp4", weight_fmt="fp4")]
        for model in (R.first_candidate_cost, R.eager_cost, R.MeasuredCost(macs_per_cycle={})):
            got = R.select(R.route_candidates(d, _hybrid()), model)
            assert got[0].unit is None and got[0].gap is not None

    def test_selection_preserves_demand_order_and_count(self):
        demands = [_demand(1, 1, 8, site="a"), _demand(in_fmt="fp4", weight_fmt="fp4", site="b"),
                   _demand(64, 64, 64, site="c")]
        got = R.select(R.route_candidates(demands, _hybrid()), R.eager_cost)
        assert [r.demand.site for r in got] == ["a", "b", "c"]


class TestExplain:
    def test_it_scores_every_candidate_so_a_decision_can_be_inspected(self):
        got = R.explain(R.route_candidates([_demand(1, 64, 64)], _hybrid()), R.eager_cost)
        assert len(got) == 1
        units = {c["unit"]: c["score"] for c in got[0]["candidates"]}
        assert set(units) == {"vec", "mat"} and units["mat"] < units["vec"]

    def test_a_declined_score_is_reported_as_none_not_omitted(self):
        model = R.MeasuredCost(macs_per_cycle={"vec": 4.0})
        got = R.explain(R.route_candidates([_demand(8, 8, 8)], _hybrid()), model)
        scores = {c["unit"]: c["score"] for c in got[0]["candidates"]}
        assert scores["mat"] is None and scores["vec"] is not None

    def test_a_gap_is_carried_through(self):
        got = R.explain(R.route_candidates([_demand(in_fmt="fp4", weight_fmt="fp4")], _hybrid()))
        assert got[0]["gap"] is not None and got[0]["candidates"] == []
