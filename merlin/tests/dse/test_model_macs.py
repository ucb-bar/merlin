from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.perf.model_macs import observe_model_macs
from merlin.perf.model_placement import contraction_placement


def convolution(*, spatial="5", address="d2 + d5", extra=""):
    return '''module {
      func.func @entry(%x: tensor<1x2xSPATIALx5xi8>, %w: tensor<3x2x3x3xi8>,
                       %out: tensor<1x3x3x3xi32>) -> tensor<1x3x3x3xi32> {
        %r = linalg.generic {indexing_maps = [
          affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d0,d4,ADDRESS,d3+d6)>,
          affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d1,d4,d5,d6)>,
          affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d0,d1,d2,d3)>],
          iterator_types = ["parallel","parallel","parallel","parallel",
                            "reduction","reduction","reduction"]}
          ins(%x, %w: tensor<1x2xSPATIALx5xi8>, tensor<3x2x3x3xi8>)
          outs(%out: tensor<1x3x3x3xi32>) attrs = {prov.region_id = "region_a"} {
            ^bb0(%a: i8, %b: i8, %c: i32):
              %aa = arith.extsi %a : i8 to i32
              %bb = arith.extsi %b : i8 to i32
              %p = arith.muli %aa, %bb : i32
              %s = arith.addi %p, %c : i32
              EXTRA
              linalg.yield %s : i32
          } -> tensor<1x3x3x3xi32>
        func.return %r : tensor<1x3x3x3xi32>
      }
    }'''.replace("SPATIAL", spatial).replace("ADDRESS", address).replace("EXTRA", extra)


def test_multiaxis_convolution_counts_all_reduction_axes():
    rows = observe_model_macs(convolution())
    assert len(rows) == 1
    shape = rows[0][1]
    assert shape.status == "derived", shape.reason
    assert shape.parallel == (1, 3, 3, 3)
    assert shape.reduction == (2, 3, 3)
    assert shape.dtypes == ("i8", "i8", "i32")
    assert shape.macs == 486


def test_affine_stride_has_same_mac_domain():
    shape = observe_model_macs(convolution(spatial="7", address="d2 * 2 + d5"))[0][1]
    assert shape.status == "derived", shape.reason
    assert shape.macs == 486


@pytest.mark.parametrize("spatial,address", [("?", "d2+d5"), ("4", "d2+d5"),
                                             ("5", "d2+d5-1")])
def test_unproved_bounds_are_unknown_not_zero(spatial, address):
    source = convolution(spatial=spatial, address=address)
    shape = observe_model_macs(source)[0][1]
    assert shape.status == "UNKNOWN"
    result = contraction_placement(source, [{"region": "region_a", "lane": "engine"}])
    assert result["status"] == "partial"
    assert result["total_contraction_macs"] is None
    assert result["unknown_mac_domain_count"] == 1
    assert result["mac_fraction_by_lane"] == {}


def test_named_matmul_retains_existing_work():
    source = '''module {func.func @entry(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>,
      %c: tensor<2x4xf32>) -> tensor<2x4xf32> {
      %r = linalg.matmul ins(%a, %b: tensor<2x3xf32>, tensor<3x4xf32>)
        outs(%c: tensor<2x4xf32>) -> tensor<2x4xf32>
      func.return %r : tensor<2x4xf32>
    }}'''
    rows = observe_model_macs(source)
    assert len(rows) == 1
    assert rows[0][1].macs == 24, rows[0][1].reason


def test_inherited_contraction_provenance_does_not_make_pointwise_a_mac():
    source = '''module {func.func @entry(%a: tensor<4xf32>, %b: tensor<4xf32>,
      %c: tensor<4xf32>) -> tensor<4xf32> {
      %r = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>,
        affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types=["parallel"]}
        ins(%a,%b: tensor<4xf32>,tensor<4xf32>)
        outs(%c: tensor<4xf32>) attrs = {prov.family="contraction"} {
        ^bb0(%x:f32,%y:f32,%z:f32):
          %p = arith.mulf %x,%y : f32
          linalg.yield %p : f32
        } -> tensor<4xf32>
      func.return %r : tensor<4xf32>
    }}'''
    assert observe_model_macs(source) == []


def test_product_bearing_unsupported_recurrence_remains_unknown():
    source = convolution().replace("linalg.yield %s", "linalg.yield %p")
    row = observe_model_macs(source)[0][1]
    assert row.status == "UNKNOWN"
    assert "recurrence" in row.reason


def test_multiple_entries_require_explicit_selection():
    source = convolution().replace("\n    }", "\n      func.func @other() {func.return}\n    }")
    assert observe_model_macs(source)[0][1].status == "UNKNOWN"
    assert observe_model_macs(source, entry="entry")[0][1].macs == 486
    assert observe_model_macs(source, entry="absent")[0][1].status == "UNKNOWN"


def test_source_call_is_not_counted_as_zero_work():
    source = '''module {func.func private @helper()
      func.func @entry() {func.call @helper() : () -> ()
      func.return}}'''
    rows = observe_model_macs(source, entry="entry")
    assert len(rows) == 1
    assert rows[0][1].status == "UNKNOWN"
    assert "multiplicity" in rows[0][1].reason


def test_full_operand_footprint_uses_each_actual_dtype(monkeypatch):
    from merlin.targetgen import memory_regime
    calls = []

    def size(shape, dtype):
        from math import prod
        calls.append(dtype)
        return prod(shape) * {"i8": 1, "i32": 4}[dtype]

    monkeypatch.setattr(memory_regime, "operand_store", lambda *args, **kwargs:
                        (SimpleNamespace(working_set_rows=size), 10000))
    result = contraction_placement(convolution(), [{"region": "region_a", "lane": "engine"}],
                                   target="mock_target")
    assert calls == ["i8", "i8", "i32"]
    assert result["contractions"][0]["working_set_rows"] == 50 + 54 + 108
    assert "not a tiled allocation" in result["memory_regime"]["note"]
