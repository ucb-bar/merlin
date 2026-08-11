"""What the e-graph prototype actually establishes, tested as such — and what it does not, asserted too.

The temptation with an e-graph is to report that it "chose better". It cannot, yet: with one e-class and
no rewrite rules, extraction minimises the same costs eager selection reads, so identical decisions are
the expected outcome and a disagreement would mean one of the two is misreading its own cost model.

So the tests here are about the MECHANISM, which is the part that is real:

* both implementations are present in the IR after construction, and
* re-costing an already-built graph changes which one is extracted.

The second is the falsifiable form of "persistent". If construction had committed to a unit, re-costing
could not change the outcome, and the graph would be a routing decision wearing an e-class.

The recorded hypothesis statuses are asserted to be UNPROVEN. That is deliberate: it means promoting one
to "established" requires editing this file, which is a much better place for that argument to happen than
a docstring.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import persistent_equivalence as PE
from merlin.targetgen import routing as R

_VEC = R.Candidate("vec", "vector", "i32", "upstream_target")
_MAT = R.Candidate("mat", "spatial", "i32", "inline_asm_insn")
_BOTH = (_VEC, _MAT)


def _model(**kw):
    base = dict(macs_per_cycle={"vec": 4.0, "mat": 256.0}, dispatch_cycles={"mat": 200.0},
                pack_cycles_per_element={"mat": 2.0}, requires_k_major=frozenset({"mat"}),
                tile_edge={"mat": 32})
    base.update(kw)
    return R.MeasuredCost(**base)


def _demand(m=64, n=64, k=64, site=""):
    return R.OpDemand("matmul", "int8", "int8", site=site, m=m, n=n, k=k)


class TestTheAlternativesAreRetained:
    def test_every_legal_implementation_is_in_the_graph(self):
        mod, names, _, _, _ = PE.build_egraph(_demand(), _BOTH, _model())
        assert names == ("vec", "mat")
        assert PE.alternatives_in(mod) == ("vec", "mat")

    def test_extraction_leaves_exactly_one(self):
        mod, *_ = PE.build_egraph(_demand(256, 256, 256), _BOTH, _model())
        assert len(PE.alternatives_in(mod)) == 2
        PE.run_extraction(mod)
        assert len(PE.alternatives_in(mod)) == 1

    def test_an_empty_candidate_set_is_refused(self):
        # An empty e-class extracts to nothing, which would read as a routing decision.
        with pytest.raises(ValueError, match="no alternatives"):
            PE.build_egraph(_demand(), (), _model())

    def test_a_declined_candidate_is_retained_but_uncosted(self):
        # It is still a legal implementation. Dropping it would erase a capability the target has;
        # costing it would let it win for lack of data.
        model = _model(macs_per_cycle={"vec": 4.0})
        mod, names, costs, unscored, _ = PE.build_egraph(_demand(), _BOTH, model)
        assert set(names) == {"vec", "mat"}
        assert unscored == ("mat",) and "mat" not in costs and "vec" in costs


class TestTheDecisionIsDeferred:
    """The falsifiable content of "persistent": the graph has not committed."""

    def test_recosting_an_existing_graph_changes_the_choice(self):
        mod, *_ = PE.build_egraph(_demand(64, 64, 64), _BOTH, _model())
        baseline, _ = PE.run_extraction(mod)

        again, *_ = PE.build_egraph(_demand(64, 64, 64), _BOTH, _model())
        # What a downstream pass would do: discover the vector path is far cheaper than it looked.
        assert PE.recost(again, {"vec": 1.0, "mat": 10_000_000.0}) == 2
        after, _ = PE.run_extraction(again)

        assert baseline == "mat"
        assert after == "vec", "re-costing did not change the extraction, so the graph had committed"

    def test_a_partial_recost_leaves_the_others_alone(self):
        mod, _, costs, _, _ = PE.build_egraph(_demand(1, 1, 64), _BOTH, _model())
        assert PE.recost(mod, {"mat": 0.0}) == 1
        assert PE.run_extraction(mod)[0] == "mat", "the untouched alternative should keep its cost"

    def test_recosting_a_unit_that_is_not_there_touches_nothing(self):
        mod, *_ = PE.build_egraph(_demand(), _BOTH, _model())
        assert PE.recost(mod, {"absent": 1.0}) == 0


class TestExtractionAgreesWithTheCostModel:
    @pytest.mark.parametrize("shape,expected", [
        ((256, 256, 256), "mat"),      # fills the tile
        ((1, 1, 64), "vec"),           # one cell of a 32x32 tile
    ])
    def test_the_cheaper_alternative_survives(self, shape, expected):
        got = PE.extract_choice(_demand(*shape), _BOTH, _model())
        assert got.chosen == expected and got.gap is None

    def test_the_choice_is_read_back_from_the_ir(self):
        # Not computed alongside: the surviving op carries the unit, and that is what is reported.
        got = PE.extract_choice(_demand(256, 256, 256), _BOTH, _model())
        assert got.chosen in got.alternatives

    def test_all_candidates_declined_fails_closed(self):
        got = PE.extract_choice(_demand(), _BOTH, R.MeasuredCost(macs_per_cycle={}))
        assert got.chosen is None
        assert "declined every alternative" in got.gap

    def test_it_agrees_with_eager_selection(self):
        # EXPECTED, not a win: same costs, same argmin. A disagreement would mean one of the two is not
        # reading the cost model it claims to.
        demands = [_demand(256, 256, 256, "a"), _demand(1, 1, 64, "b"), _demand(64, 8, 512, "c")]
        cands = R.route_candidates(demands, _hybrid_units())
        eager = R.select(cands, _model())
        extracted = PE.select_by_extraction(cands, _model())
        assert [r.unit for r in eager] == [r.unit for r in extracted]

    def test_a_gapped_demand_is_passed_through(self):
        d = [R.OpDemand("matmul", "fp4", "fp4", site="g", m=8, n=8, k=8)]
        cands = R.route_candidates(d, _hybrid_units())
        got = PE.select_by_extraction(cands, _model())
        assert got[0].unit is None and got[0].gap is not None

    def test_an_unscorable_demand_falls_back_to_declaration_order(self):
        # Same fallback as `select`, so the two are comparable rather than differing on missing data.
        cands = R.route_candidates([_demand(8, 8, 8)], _hybrid_units())
        got = PE.select_by_extraction(cands, R.MeasuredCost(macs_per_cycle={}))
        assert got[0].unit == cands[0].candidates[0].unit

    def test_the_accumulator_of_the_chosen_unit_is_carried(self):
        cands = R.route_candidates([_demand(256, 256, 256)], _hybrid_units())
        got = PE.select_by_extraction(cands, _model())
        assert got[0].acc == "i32"


class TestCostScaling:
    def test_costs_are_scaled_to_integers(self):
        _, _, costs, _, _ = PE.build_egraph(_demand(1, 1, 64), _BOTH, _model())
        assert all(isinstance(v, int) for v in costs.values())
        assert costs["vec"] == 16 * PE.COST_SCALE          # 64 macs / 4 per cycle

    def test_sub_cycle_differences_survive_scaling(self):
        # Without the scale factor these would round to the same integer and the ordering would be lost.
        model = R.MeasuredCost(macs_per_cycle={"vec": 1000.0, "mat": 1100.0})
        _, _, costs, _, _ = PE.build_egraph(_demand(10, 10, 10), _BOTH, model)
        assert costs["vec"] != costs["mat"]


class TestTheHypothesesStayHonest:
    def test_h_eq1_is_not_claimed_as_established(self):
        assert PE.hypothesis_status("H-EQ1") == "not_established"

    def test_h_eq2_is_not_claimed_as_exercised(self):
        # No rewrite rules are applied, so there is nothing to re-saturate.
        assert PE.hypothesis_status("H-EQ2") == "not_exercised"

    @pytest.mark.parametrize("name", ["H-EQ1", "H-EQ2"])
    def test_neither_hypothesis_reads_as_a_result(self, name):
        # Promoting one requires editing this test, which is where that argument should happen.
        assert PE.hypothesis_status(name) not in {"established", "confirmed", "true"}
        assert PE.HYPOTHESES[name]["why"], "a status without a reason is not a finding"

    def test_an_unknown_hypothesis_raises(self):
        # Returning "unknown" would let a typo read as a modest claim rather than a missing one.
        with pytest.raises(KeyError):
            PE.hypothesis_status("H-EQ9")

    def test_the_agreement_report_carries_the_hypotheses_and_the_cost(self):
        cands = R.route_candidates([_demand(256, 256, 256)], _hybrid_units())
        got = PE.agreement(cands, _model())
        assert got["agree"] is True and got["disagreements"] == []
        assert got["extraction_seconds"] >= 0 and got["slowdown"] is not None
        assert set(got["hypotheses"]) == {"H-EQ1", "H-EQ2"}

    def test_compile_time_is_reported_per_extraction(self):
        got = PE.extract_choice(_demand(256, 256, 256), _BOTH, _model())
        d = got.to_dict()
        assert d["build_seconds"] >= 0 and d["extract_seconds"] >= 0
        assert d["total_seconds"] == pytest.approx(d["build_seconds"] + d["extract_seconds"], abs=1e-6)


def _hybrid_units():
    from merlin.targetgen import compute_units as CU
    rule = (CU.AccumRule("int8", "int8", "i32"),)
    return [CU.ComputeUnit(name="vec", kind="vector", dtypes=("int8",), ops=("matmul",),
                           accumulate=rule),
            CU.ComputeUnit(name="mat", kind="spatial", dtypes=("int8",), ops=("matmul",),
                           accumulate=rule, exposure="inline_asm_insn")]
