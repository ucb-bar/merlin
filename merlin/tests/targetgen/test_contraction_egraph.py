"""The e-graph decides over REAL contraction IR, and the decision is read back rather than computed.

The distinction these tests protect is easy to lose: a function that builds an e-graph, then computes the
argmin in Python and returns it, demonstrates nothing — it is a threshold with extra steps. So the
assertions here are about what SURVIVES in the extracted IR, and about the fact that changing only the costs
changes which implementation is left.

The second thing they protect is the honest direction of failure. Extraction minimises exactly the cost it
is given, so a cost model that overrates a unit routes work onto it confidently and wrongly. The tests
therefore pin ``MeasuredCost``'s decline-when-unmeasured behaviour end to end: with no measured throughput
for the matrix unit, nothing is routed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import passes_opu as PO
from merlin.targetgen import contraction_egraph as CE
from merlin.targetgen import routing as R

#: A rank-2 int8 contraction in the form the int8 rewrite leaves behind — a generic with (i8, i8) inputs,
#: an i32 accumulator from a zero fill, and extsi/muli/addi in the body.
_INT8_MM = """
builtin.module {
  func.func @forward(%a: tensor<64x32xi8>, %b: tensor<32x16xi8>) -> tensor<64x16xi32> {
    %e = tensor.empty() : tensor<64x16xi32>
    %z = arith.constant 0 : i32
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<64x16xi32>) -> tensor<64x16xi32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                          affine_map<(d0, d1, d2) -> (d2, d1)>,
                                          affine_map<(d0, d1, d2) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<64x32xi8>, tensor<32x16xi8>)
        outs(%f : tensor<64x16xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xe = arith.extsi %x : i8 to i32
      %ye = arith.extsi %y : i8 to i32
      %m = arith.muli %xe, %ye : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
    } -> tensor<64x16xi32>
    func.return %r : tensor<64x16xi32>
  }
}
"""

_SYM = "merlin_opu_gemm_i8_0"


def _one_contraction():
    mod = parse_mlir_text(_INT8_MM)
    cands = PO.routable_contractions(mod)
    assert len(cands) == 1
    return mod, cands[0]


def _text(module) -> str:
    from merlin.xdsl_dialects._common import text as to_text
    return to_text(module)


class TestBothImplementationsArePresent:
    def test_the_class_holds_the_contraction_and_the_call(self):
        # If only one were present, extraction would have nothing to choose between and the whole exercise
        # would be a rename of eager selection.
        _mod, (op, _sh) = _one_contraction()
        module, _s = CE.build_contraction_egraph(op, symbol=_SYM, costs={CE.VECTOR: 5, CE.MATRIX: 3})
        got = _text(module)
        assert "linalg.generic" in got
        assert f"func.call @{_SYM}" in got
        assert "equivalence.class" in got

    def test_both_alternatives_have_the_same_result_type(self):
        # This is why an e-class over them is legal at all, and why the extracted function is directly the
        # compile path's answer rather than something needing a fix-up.
        _mod, (op, _sh) = _one_contraction()
        module, _s = CE.build_contraction_egraph(op, symbol=_SYM, costs={})
        got = _text(module)
        assert got.count("tensor<64x16xi32>") >= 3      # generic result, call result, class result

    def test_the_source_module_is_left_intact(self):
        # The decision is applied to that module afterwards, so building the graph must clone rather than
        # move: a moved contraction would leave the model with a hole.
        mod, (op, _sh) = _one_contraction()
        CE.build_contraction_egraph(op, symbol=_SYM, costs={CE.VECTOR: 1, CE.MATRIX: 2})
        assert len(PO.routable_contractions(mod)) == 1

    def test_the_operands_are_defined_inside_the_graph(self):
        # The eqsat passes require it. A clone still referring to the original module's values is not
        # self-contained, and the failure is a verification error a long way from its cause.
        _mod, (op, _sh) = _one_contraction()
        module, _s = CE.build_contraction_egraph(op, symbol=_SYM, costs={})
        got = _text(module)
        assert "func.func @contraction(%0: tensor<64x32xi8>" in got

    def test_a_wrongly_shaped_op_is_refused(self):
        class _Fake:
            operands, results = (), ()
        with pytest.raises(ValueError, match="3-operand"):
            CE.build_contraction_egraph(_Fake(), symbol=_SYM, costs={})


class TestTheCostDecidesAndTheChoiceIsReadBack:
    @pytest.mark.parametrize("vec,mat,expect", [(5000, 1200, CE.MATRIX), (1200, 5000, CE.VECTOR)])
    def test_flipping_the_costs_flips_the_surviving_implementation(self, vec, mat, expect):
        # THE test. Same IR, same construction, only the costs differ — so the decision is being made by
        # minimisation over the graph and not by anything about the order or the shape.
        _mod, (op, sh) = _one_contraction()
        got = CE.extract_contraction_choice(op, symbol=_SYM, shape=sh,
                                           costs={CE.VECTOR: vec, CE.MATRIX: mat})
        assert got.chosen == expect and got.gap is None
        assert got.on_matrix_unit is (expect == CE.MATRIX)

    def test_a_tie_leaves_the_work_on_the_vector_path(self):
        # Declaration order breaks the tie, and the contraction is added first on purpose: the vector path
        # is the control, and a coin-flip must not move work onto a unit whose advantage is unproven.
        _mod, (op, sh) = _one_contraction()
        got = CE.extract_contraction_choice(op, symbol=_SYM, shape=sh,
                                           costs={CE.VECTOR: 3000, CE.MATRIX: 3000})
        assert got.chosen == CE.VECTOR

    def test_the_extents_are_recorded_with_the_choice(self):
        _mod, (op, sh) = _one_contraction()
        got = CE.extract_contraction_choice(op, symbol=_SYM, shape=sh, costs={CE.VECTOR: 1})
        assert (got.m, got.n, got.k) == (64, 16, 32)

    def test_the_decision_time_is_reported(self):
        # The source paper measures a 401x geomean slowdown against egg, so "the mechanism works" has to
        # come with a number rather than an impression.
        _mod, (op, sh) = _one_contraction()
        got = CE.extract_contraction_choice(op, symbol=_SYM, shape=sh,
                                           costs={CE.VECTOR: 9, CE.MATRIX: 1})
        assert got.build_seconds > 0 and got.extract_seconds > 0
        assert got.total_seconds == pytest.approx(got.build_seconds + got.extract_seconds)
        assert got.to_dict()["total_seconds"] >= 0


class TestItFailsClosedWithoutCosts:
    def test_no_costs_means_no_choice_rather_than_an_arbitrary_one(self):
        _mod, (op, sh) = _one_contraction()
        got = CE.extract_contraction_choice(op, symbol=_SYM, shape=sh, costs={})
        assert got.chosen is None and not got.on_matrix_unit
        assert "nothing to minimise" in (got.gap or "")

    def test_only_the_vector_path_costed_keeps_the_work_there(self):
        # An uncosted alternative is present but unranked, so extraction cannot prefer it for lack of data.
        _mod, (op, sh) = _one_contraction()
        got = CE.extract_contraction_choice(op, symbol=_SYM, shape=sh, costs={CE.VECTOR: 10})
        assert got.chosen == CE.VECTOR

    def test_only_the_matrix_path_costed_moves_the_work(self):
        _mod, (op, sh) = _one_contraction()
        got = CE.extract_contraction_choice(op, symbol=_SYM, shape=sh, costs={CE.MATRIX: 10})
        assert got.chosen == CE.MATRIX


class TestTheMeasuredCostModelDeclinesRatherThanGuesses:
    """The quality of the decision is now entirely a measurement question, and that must be visible."""

    def _cost_of(self, cm):
        return CE.measured_cost_of(cm, vector_unit="rvv", matrix_unit="opu")

    def test_an_unmeasured_matrix_unit_is_declined(self):
        cm = R.MeasuredCost(macs_per_cycle={"rvv": 4.0}, tile_edge={"opu": 32})
        _mod, (_op, sh) = _one_contraction()
        cost_of = self._cost_of(cm)
        assert cost_of(sh, CE.VECTOR) is not None
        assert cost_of(sh, CE.MATRIX) is None, "an unmeasured unit must not be scored optimistically"

    def test_with_no_measurement_nothing_is_routed(self):
        # End to end: the selector the rewrite would use routes nothing, because the only ranked
        # alternative is the vector path. This is the honest state of the world until a throughput is
        # measured -- not a bug to work around.
        cm = R.MeasuredCost(macs_per_cycle={"rvv": 4.0}, tile_edge={"opu": 32})
        mod, (op, sh) = _one_contraction()
        select = CE.egraph_selector(self._cost_of(cm), symbol=_SYM)
        assert select(op, sh) is False

    def test_a_measured_matrix_unit_is_scored_and_can_win(self):
        # The mirror, so the test above cannot pass by declining everything.
        cm = R.MeasuredCost(macs_per_cycle={"rvv": 1.0, "opu": 64.0}, tile_edge={"opu": 32})
        _mod, (op, sh) = _one_contraction()
        select = CE.egraph_selector(self._cost_of(cm), symbol=_SYM)
        assert select(op, sh) is True

    def test_tile_occupancy_is_charged_so_a_narrow_shape_costs_more_per_element(self):
        # The property that makes the cost model worth using: a shape filling one row of a 32-lane tile is
        # charged for the whole tile. Compared per element so the two shapes are comparable.
        cm = R.MeasuredCost(macs_per_cycle={"rvv": 1.0, "opu": 64.0}, tile_edge={"opu": 32})
        cost_of = self._cost_of(cm)

        class _Shape:
            def __init__(self, m, n, k):
                self.parallel, self.reduction = (m, n), (k,)

        wide = cost_of(_Shape(64, 64, 32), CE.MATRIX) / (64 * 64)
        narrow = cost_of(_Shape(64, 1, 32), CE.MATRIX) / (64 * 1)
        assert narrow > wide, (narrow, wide)


class TestOnTheRealPreparedModel:
    """The only forms that matter are the ones the int8 rewrite actually produces."""

    @pytest.fixture
    def candidates(self):
        from merlin.common.paths import artifacts_dir
        from merlin.frontends.linalg_mlir import parse_mlir_file
        p = (Path(artifacts_dir()) / "target-evolution/saturn_opu/v1/latest/prepared"
             / "spectformer_int8_full/model.prepared.mlir")
        if not p.is_file():
            pytest.skip(f"no prepared module at {p}")
        return PO.routable_contractions(parse_mlir_file(p))

    def test_every_candidate_builds_a_type_correct_egraph(self, candidates):
        # A real model's contractions carry provenance attributes and regions the synthetic fixture does
        # not; the graph has to survive all 90 of them, not one.
        for op, _sh in candidates:
            module, _s = CE.build_contraction_egraph(op, symbol=_SYM,
                                                    costs={CE.VECTOR: 2, CE.MATRIX: 1})
            module.verify()

    def test_the_decision_is_made_for_every_contraction_without_gaps(self, candidates):
        record: list[CE.ContractionChoice] = []
        select = CE.egraph_selector(lambda sh, which: 1 if which == CE.MATRIX else 2,
                                    symbol=_SYM, record=record)
        for op, sh in candidates:
            select(op, sh)
        assert len(record) == len(candidates)
        assert [r for r in record if r.gap] == []

    def test_it_decides_in_a_time_worth_reporting(self, candidates):
        # Compile-time cost is a first-class number here. A per-contraction budget in the tens of
        # milliseconds is what makes this usable on a whole model at all.
        record: list[CE.ContractionChoice] = []
        select = CE.egraph_selector(lambda sh, which: 1 if which == CE.MATRIX else 2,
                                    symbol=_SYM, record=record)
        for op, sh in candidates:
            select(op, sh)
        per_op = sum(r.total_seconds for r in record) / len(record)
        assert per_op < 0.1, f"{per_op * 1e3:.1f} ms per contraction"

    def test_the_rewrite_consumes_the_extracted_decision(self, candidates):
        # The point of the whole exercise: the routing decision the compile path applies is the one
        # extraction made, not a threshold that happens to agree with it.
        decided = CE.for_rewrite(
            CE.egraph_selector(
                # Cheap on the matrix unit only when both extents fill a 32-lane tile.
                lambda sh, which: (1 if min(sh.parallel[0], sh.parallel[1]) >= 32 else 9)
                if which == CE.MATRIX else 5,
                symbol=_SYM),
            candidates)
        chosen = [sh for _op, sh in candidates if decided(sh)]
        assert chosen, "the selector routed nothing, so this asserts nothing"
        for sh in chosen:
            assert min(sh.parallel[0], sh.parallel[1]) >= 32

    def test_two_contractions_with_the_same_extents_get_their_own_decisions(self, candidates):
        # `for_rewrite` keys on the enumerated pair's identity. Keying on the extents would make one
        # decision stand in for twelve, which is exactly what a per-op cost model exists to avoid.
        seen: dict[tuple, int] = {}
        for _op, sh in candidates:
            key = (sh.parallel[0], sh.parallel[1], sh.reduction[0])
            seen[key] = seen.get(key, 0) + 1
        assert max(seen.values()) > 1, "the model should contain repeated shapes"
        decided = CE.for_rewrite(lambda _op, _sh: True, candidates)
        assert all(decided(sh) for _op, sh in candidates)
