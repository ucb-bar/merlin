"""Infer mechanism questions from actual source/commands, never model labels."""

from __future__ import annotations

import pytest

from merlin.compile.analysis.schedule_structure import analyze_normalized_commands
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.kernels.mechanism_inference import (
    ReductionLanes,
    packing_profiles,
    repeated_read_mechanisms,
    source_mechanisms,
)


def reduction_source(*, extent=3, tap=7, address="d2", yielded="s"):
    return f"""module {{
      func.func @entry(%a: tensor<2x{tap}x{tap}x{extent}xi8>,
        %b: tensor<{tap}x{tap}x{extent}x5xi8>, %c: tensor<2x5xi32>) -> tensor<2x5xi32> {{
        %r = linalg.generic {{indexing_maps = [
          affine_map<(d0,d1,d2,d3,d4)->(d0,{address},d3,d4)>,
          affine_map<(d0,d1,d2,d3,d4)->(d2,d3,d4,d1)>,
          affine_map<(d0,d1,d2,d3,d4)->(d0,d1)>],
          iterator_types=["parallel","parallel","reduction","reduction","reduction"]}}
          ins(%a,%b: tensor<2x{tap}x{tap}x{extent}xi8>,tensor<{tap}x{tap}x{extent}x5xi8>)
          outs(%c: tensor<2x5xi32>) {{
          ^bb0(%x:i8,%y:i8,%z:i32):
            %xx = arith.extsi %x : i8 to i32
            %yy = arith.extsi %y : i8 to i32
            %p = arith.muli %xx,%yy : i32
            %s = arith.addi %p,%z : i32
            linalg.yield %{yielded} : i32
          }} -> tensor<2x5xi32>
        func.return %r : tensor<2x5xi32>
      }}
    }}"""


@pytest.mark.parametrize("tap,channels,width", [(7, 3, 16), (5, 2, 8), (3, 9, 4)])
def test_actual_multiaxis_reduction_exposes_conditional_lane_waste(tap, channels, width):
    module = parse_mlir_text(reduction_source(extent=channels, tap=tap))
    report = source_mechanisms(module, evidence_id="original-source:sha", lanes=ReductionLanes(width, "facts:sha#edge"))
    row, = report["reductions"]
    packing = row["packing"]
    useful = tap * tap * channels
    assert packing["useful_reduction_extent"] == useful
    separate, rows, flattened = packing["alternatives"]
    assert separate["pieces"] == tap * tap * (-(-channels // width))
    assert rows["pieces"] == tap * (-(-(tap * channels) // width))
    assert flattened["pieces"] == -(-useful // width)
    if (tap, channels, width) == (7, 3, 16):
        assert separate["physical_extent"] == 784
        assert rows["pieces"] == 14
        assert flattened["physical_extent"] == 160
    assert all(item["legality"] == item["emitted_application"] == "unknown" for item in packing["alternatives"])
    assert report["optimization_application"] == report["performance"] == "unknown"


def test_unselected_width_never_defaults_and_unsupported_source_is_unknown():
    report = source_mechanisms(parse_mlir_text(reduction_source()), evidence_id="source")
    assert all(row["physical_extent"] is None for row in report["reductions"][0]["packing"]["alternatives"])
    for source in (reduction_source(address="d2-1"), reduction_source(yielded="p")):
        unknown = source_mechanisms(parse_mlir_text(source), evidence_id="source")
        assert unknown["status"] == "partial" and not unknown["reductions"]
        assert unknown["unknowns_total"] == 1


def test_shared_input_is_ssa_identity_not_equal_shapes_or_grouping_labels():
    source = """module {func.func @entry(%a:tensor<2x3xi8>,%different:tensor<2x3xi8>,
      %b:tensor<3x4xi8>,%d:tensor<3x4xi8>,%c:tensor<2x4xi32>,%e:tensor<2x4xi32>)
      -> (tensor<2x4xi32>,tensor<2x4xi32>) {
      %r = linalg.matmul ins(%a,%b:tensor<2x3xi8>,tensor<3x4xi8>)
        outs(%c:tensor<2x4xi32>) -> tensor<2x4xi32>
      %s = linalg.matmul ins(%a,%d:tensor<2x3xi8>,tensor<3x4xi8>)
        outs(%e:tensor<2x4xi32>) -> tensor<2x4xi32>
      func.return %r,%s : tensor<2x4xi32>,tensor<2x4xi32>
    }}"""
    report = source_mechanisms(parse_mlir_text(source), evidence_id="source")
    shared, = report["shared_inputs"]
    assert shared["shape"] == [2, 3] and shared["consumers_total"] == 2
    assert shared["storage_reuse"] == "unknown"
    separate = source.replace("ins(%a,%d", "ins(%different,%d")
    assert source_mechanisms(parse_mlir_text(separate), evidence_id="source")["shared_inputs"] == []


def phase_reads(*, writes_input=False, write_unit="byte", complete=True, extent=100_352, phases=32):
    records = []
    for phase in range(phases):
        records.extend([
            dict(identity="copy_input", roles=["operand_load"], accesses=[dict(
                space="external", buffer="input", offset=0, extent=extent, mode="read", unit="byte")]),
            dict(identity="phase_write", roles=["commit"], accesses=[dict(
                space="external", buffer="input" if writes_input else "output", offset=phase,
                extent=1, mode="write", unit=write_unit)]),
        ])
    return analyze_normalized_commands(records, complete=complete, evidence_id="selected-effects")


@pytest.mark.parametrize("extent,phases", [(100_352, 32), (35, 7)])
def test_repeated_phase_reads_keep_capacity_and_retention_unproved(extent, phases):
    model = phase_reads(extent=extent, phases=phases)
    report = repeated_read_mechanisms(model)
    row, = report["findings"]
    assert row["status"] == "observed_invariant_read"
    assert row["total_read_units"] == extent * phases
    assert row["repeated_read_units"] == extent * (phases - 1)
    assert row["reuse_legality"] == row["optimization_application"] == "unknown"


def test_writes_unknown_units_and_incomplete_capture_prevent_invariant_conclusion():
    overwritten, = repeated_read_mechanisms(phase_reads(writes_input=True))["findings"]
    assert overwritten["status"] == "intervening_write_possible"
    wrong_unit, = repeated_read_mechanisms(phase_reads(writes_input=True, write_unit="row"))["findings"]
    assert wrong_unit["status"] == "unknown"
    partial, = repeated_read_mechanisms(phase_reads(complete=False))["findings"]
    assert partial["status"] == "unknown"


def test_all_inference_budgets_refuse_explicitly():
    source = source_mechanisms(parse_mlir_text(reduction_source()), evidence_id="source", max_operations=2)
    assert source["reason"] == "source_mechanism_operation_budget_exhausted"
    reuse = repeated_read_mechanisms(phase_reads(), max_comparisons=1)
    assert reuse["reason"] == "reuse_comparison_budget_exhausted" and reuse["findings"] == []
    with pytest.raises(ValueError, match="positive"):
        packing_profiles((0, 1))
