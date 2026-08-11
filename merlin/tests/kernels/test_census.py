"""The census must never be more confident than its inputs allow.

Every test here defends one of three properties, each of which a plausible implementation gets wrong:

* an undecidable legality question reports ``unknown`` WITH its cause, and never ``illegal`` — the two
  read identically in a summary table and mean opposite things about the hardware;
* a verdict states the axes it covers, so "legal" cannot be read as "fits" when the capability
  contract expresses nothing about shape;
* the IR stage that was read is recorded, because an int8 model's element types are decided by a pass
  and a census of the capture answers a different question than a census of what gets compiled.
"""
from __future__ import annotations

import pytest

from merlin.kernels import census as cs
from merlin.kernels.microkernel import ContractionShape
from merlin.targetgen import compute_units as cu

# --- module fixtures: MLIR text, so the observer is exercised rather than stubbed ----------------

_MATMUL_I8 = """
module {
  func.func @forward(%a: tensor<8x64xi8>, %b: tensor<64x32xi8>) -> tensor<8x32xi32> {
    %e = tensor.empty() : tensor<8x32xi32>
    %0 = linalg.matmul {prov.fqn = "enc.l0", prov.op = "linear"}
         ins(%a, %b : tensor<8x64xi8>, tensor<64x32xi8>)
         outs(%e : tensor<8x32xi32>) -> tensor<8x32xi32>
    return %0 : tensor<8x32xi32>
  }
}
"""

_MATMUL_F32 = _MATMUL_I8.replace("xi8", "xf32").replace("xi32", "xf32")

#: What the int8 rewrite leaves behind: the contraction it split out of `enc.l0`, tagged with the role
#: that distinguishes it from the requant epilogue sharing that same fqn.
_MATMUL_I8_ROLED = _MATMUL_I8.replace('prov.fqn = "enc.l0"',
                                      'prov.fqn = "enc.l0", prov.role = "contraction"')


def _units(dtypes=("int8",), ops=("matmul",), accumulate=(("int8", "int8", "i32"),)):
    return [cu.ComputeUnit(name="tile", kind="spatial", dtypes=tuple(dtypes), ops=tuple(ops),
                           accumulate=tuple(cu.AccumRule(*a) for a in accumulate))]


class TestLegalityVerdicts:
    def test_a_matching_unit_is_legal_and_reports_the_accumulator(self):
        shape = ContractionShape("linalg.matmul", (8, 32), (64,), ("i8", "i8", "i32"))
        got = cs.legality_of(shape, _units())
        assert got.verdict == cs.LEGAL
        assert got.unit == "tile" and got.acc == "i32"

    def test_a_verdict_always_names_the_axes_it_covers(self):
        # The contract expresses op name and element types and nothing about tile geometry, so a
        # "legal" row must not be readable as "this shape fits the unit".
        got = cs.legality_of(ContractionShape("linalg.matmul", (8, 32), (64,), ("i8", "i8", "i32")),
                             _units())
        assert got.scope == cs.ROUTING_SCOPE
        assert "element_types" in got.scope and "op_name" in got.scope

    def test_a_dtype_the_unit_does_not_declare_is_illegal_with_the_demand_in_the_reason(self):
        shape = ContractionShape("linalg.matmul", (8, 32), (64,), ("f32", "f32", "f32"))
        got = cs.legality_of(shape, _units())
        assert got.verdict == cs.ILLEGAL
        assert "fp32" in got.reason, got.reason

    def test_an_undeclared_op_name_gaps_rather_than_being_assumed_equivalent(self):
        # A unit declaring `matmul` has not declared `batch_matmul`; fail closed.
        shape = ContractionShape("linalg.batch_matmul", (4, 8, 32), (64,), ("i8", "i8", "i32"))
        assert cs.legality_of(shape, _units()).verdict == cs.ILLEGAL

    def test_no_units_is_unknown_and_carries_the_supplied_cause(self):
        shape = ContractionShape("linalg.matmul", (8, 32), (64,), ("i8", "i8", "i32"))
        got = cs.legality_of(shape, [], undecidable="contract not found at /nowhere")
        assert got.verdict == cs.UNKNOWN
        assert "not found" in got.reason, "the row must state the real cause, not a generic one"

    def test_unobserved_element_types_are_unknown_not_illegal(self):
        # A shape with no dtypes says nothing about legality; calling that a refusal would credit the
        # census with a measurement it never made.
        got = cs.legality_of(ContractionShape("linalg.matmul", (8, 32), (64,)), _units())
        assert got.verdict == cs.UNKNOWN and got.reason

    def test_an_unregistered_element_token_is_unknown_and_names_the_token(self):
        shape = ContractionShape("linalg.matmul", (8, 32), (64,), ("i7", "i7", "i32"))
        got = cs.legality_of(shape, _units())
        assert got.verdict == cs.UNKNOWN
        assert "i7" in got.reason


class TestLegalityContract:
    def test_a_non_legal_verdict_cannot_be_constructed_without_a_reason(self):
        with pytest.raises(ValueError):
            cs.Legality(cs.ILLEGAL, cs.ROUTING_SCOPE)
        with pytest.raises(ValueError):
            cs.Legality(cs.UNKNOWN, cs.ROUTING_SCOPE)

    def test_an_unknown_verdict_spelling_is_rejected(self):
        with pytest.raises(ValueError):
            cs.Legality("probably", cs.ROUTING_SCOPE, reason="x")


class TestCensusRows:
    def test_reads_shape_dtype_and_provenance_off_one_module(self):
        got = cs.census(_MATMUL_I8, model="m", stage="test")
        assert len(got.rows) == 1
        row = got.rows[0]
        assert row.parallel == (8, 32) and row.reduction == (64,)
        assert row.dtypes == ("i8", "i8", "i32")
        assert row.key == "enc.l0", "the profile join key must come from prov.fqn"
        assert row.family == "linear"

    def test_work_counts_the_reduction_not_just_the_output(self):
        # 8*32*64 output-and-reduction points, times a multiply-accumulate. An output-element proxy
        # would report 8*32 and rank this contraction 64x too light.
        assert cs.census(_MATMUL_I8, model="m").rows[0].work == 8 * 32 * 64 * 2

    def test_records_the_stage_it_read_rather_than_inferring_one(self):
        assert cs.census(_MATMUL_I8, model="m", stage="prepared").stage == "prepared"

    def test_an_unparseable_module_yields_no_rows_instead_of_raising(self):
        got = cs.census("not mlir {{{", model="m")
        assert got.rows == ()

    def test_without_a_profile_the_ranking_says_it_is_not_a_measurement(self):
        got = cs.census(_MATMUL_I8, model="m")
        assert got.ranked_by == "work"
        assert any("NOT a cost measurement" in n for n in got.notes)

    def test_element_types_differ_between_stages_of_the_same_model(self):
        # This is the failure mode the stage label exists for: the same contraction is fp32 before the
        # quant rewrite and int8 after, so a legality verdict is only meaningful next to its stage.
        i8 = cs.census(_MATMUL_I8, model="m").rows[0]
        f32 = cs.census(_MATMUL_F32, model="m").rows[0]
        assert i8.dtypes != f32.dtypes
        assert cs.legality_of(ContractionShape(i8.op_class, i8.parallel, i8.reduction, i8.dtypes),
                              _units()).verdict == cs.LEGAL
        assert cs.legality_of(ContractionShape(f32.op_class, f32.parallel, f32.reduction, f32.dtypes),
                              _units()).verdict == cs.ILLEGAL


class TestTickJoin:
    #: One captured layer that a rewrite split in two: both pieces carry the layer's fqn, and only
    #: `prov.role` tells them apart.
    _TABLE = [{"id": 0, "fqn": "enc.l0", "role": "contraction", "mlir_op": "linalg.generic"},
              {"id": 1, "fqn": "enc.l0", "role": "requant", "mlir_op": "linalg.generic"},
              {"id": 2, "fqn": "enc.act", "mlir_op": "linalg.generic"}]
    _TICKS = {0: (700, 1), 1: (200, 1), 2: (100, 1)}

    def test_role_keeps_a_split_ops_pieces_apart(self):
        by_key, total = cs._ticks_by_key(self._TABLE, self._TICKS)
        assert total == 1000
        assert by_key["enc.l0"] == (900, 2), "the fqn bucket is the whole layer"
        assert by_key["enc.l0#contraction"] == (700, 1), "the role bucket is the contraction alone"

    def test_pct_model_denominator_is_every_profiled_op_not_only_contractions(self):
        _, total = cs._ticks_by_key(self._TABLE, self._TICKS)
        assert total == 1000, "the elementwise op's 100 ticks are part of the model"

    def test_no_profile_yields_no_ticks_and_no_denominator(self):
        assert cs._ticks_by_key(None, None) == ({}, None)
        assert cs._ticks_by_key(self._TABLE, None) == ({}, None)

    def test_a_row_joined_on_a_shared_key_is_flagged_as_an_upper_bound(self):
        # A contraction whose profiled op has no role falls back to the layer bucket, which covers two
        # ops; the census must say so rather than presenting the sum as the contraction's cost.
        table = [{"id": 0, "fqn": "enc.l0", "mlir_op": "linalg.matmul"},
                 {"id": 1, "fqn": "enc.l0", "mlir_op": "linalg.generic"}]
        got = cs.census(_MATMUL_I8, model="m", prof_table=table, prof_ticks={0: (700, 1), 1: (300, 1)})
        row = got.rows[0]
        assert row.ticks == 1000 and row.ticks_ops == 2
        assert any("upper bound" in n for n in got.notes)

    def test_a_joined_profile_switches_the_ranking_to_measured_cost(self):
        got = cs.census(_MATMUL_I8_ROLED, model="m", prof_table=self._TABLE, prof_ticks=self._TICKS)
        assert got.ranked_by == "ticks"
        assert got.rows[0].role == "contraction"
        assert got.rows[0].ticks == 700, "the contraction's own bucket, not the layer's"
        assert got.rows[0].ticks_ops == 1, "a role-qualified bucket covers exactly one op"
        assert got.rows[0].pct_model == pytest.approx(0.7)


class TestTargetResolution:
    def test_a_missing_target_contract_is_unknown_with_the_path_in_the_reason(self):
        got = cs.census(_MATMUL_I8, model="m", target="no_such_target_xyz")
        assert got.by_verdict(cs.UNKNOWN) == got.rows
        assert "no_such_target_xyz" in got.rows[0].legality.reason

    def test_no_target_is_undecided_rather_than_silently_legal(self):
        got = cs.census(_MATMUL_I8, model="m", target=None)
        assert got.rows[0].legality.verdict == cs.UNKNOWN

    def test_notes_record_what_the_units_declare(self):
        # A row gapped on a name mismatch is only diagnosable if the unit's vocabulary is reported.
        note = cs._unit_vocabulary(_units())
        assert "ops=[matmul]" in note and "dtypes=[int8]" in note


class TestReporting:
    def test_markdown_states_the_stage_the_scope_and_every_note(self):
        got = cs.census(_MATMUL_I8, model="m", stage="prepared", target="no_such_target_xyz")
        md = cs.to_markdown([got])
        assert "prepared" in md
        assert cs.ROUTING_SCOPE in md, "the summary may not present legality without its scope"
        for note in got.notes:
            assert note[:40] in md

    def test_a_truncated_table_says_how_many_rows_it_dropped(self):
        got = cs.census(_MATMUL_I8, model="m")
        md = cs.to_markdown([got], top=0)
        assert "1 more rows omitted" in md, "a bounded view must report what it left out"

    def test_a_model_with_no_contractions_says_so_instead_of_rendering_an_empty_table(self):
        md = cs.to_markdown([cs.census("not mlir {{{", model="m")])
        assert "no contractions observed" in md


class TestBundle:
    def test_a_missing_bundle_is_reported_rather_than_raising(self, tmp_path):
        got = cs.census_bundle(tmp_path / "absent")
        assert got.stage == "missing" and got.rows == ()
        assert any("no model.mlir" in n for n in got.notes)

    def test_reading_the_capture_says_the_element_types_precede_the_rewrite(self, tmp_path):
        bundle = tmp_path / "b"
        bundle.mkdir()
        (bundle / "model.mlir").write_text(_MATMUL_F32, encoding="utf-8")
        got = cs.census_bundle(bundle)
        assert got.stage == "capture"
        assert any("precede the quant rewrite" in n for n in got.notes)
