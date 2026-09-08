"""`merlin.perf.lane_cost` must refuse rather than complete a byte total with a default.

The failure this guards is the one this tree keeps re-finding: a metric that cannot fail reports
success. A byte total silently completed by a default element width reads exactly like a measurement,
and it is wrong in the flattering direction (an f64 or an fp8 tensor mis-sized as 4 bytes).
"""
from __future__ import annotations

from merlin.perf.lane_cost import dtype_bits, lane_cost, format_report


def test_dtype_bits_parses_structurally_not_from_a_table():
    for token, bits in (("i8", 8), ("i32", 32), ("f32", 32), ("f64", 64), ("bf16", 16),
                        ("i64", 64), ("f16", 16), ("int8", 8), ("uint16", 16), ("f8E4M3FN", 8)):
        assert dtype_bits(token) == bits, token


def test_dtype_bits_is_unknown_never_a_default():
    for token in ("nonsense", "", None, 32, "i", "f", "i0"):
        assert dtype_bits(token) is None, token


def _cb(tensors, placement=None):
    cb = {"tensors": tensors}
    if placement is not None:
        cb["params"] = {"lane_placement": placement}
    return cb


def test_traffic_counts_intermediates_twice_and_parameters_once():
    cost = lane_cost(_cb({
        "w": {"shape": [1024], "dtype": "i8", "role": "weight"},
        "t": {"shape": [1024], "dtype": "i32", "role": "intermediate"},
    }))
    assert not cost.is_lower_bound
    assert cost.footprint_bytes == 1024 + 4096
    # weight read once, intermediate written then read
    assert cost.traffic_bytes == 1024 + 2 * 4096


def test_an_unparseable_dtype_refuses_and_makes_the_total_a_lower_bound():
    cost = lane_cost(_cb({
        "good": {"shape": [8], "dtype": "i8", "role": "weight"},
        "bad": {"shape": [1000000], "dtype": "mystery", "role": "intermediate"},
    }))
    assert cost.is_lower_bound
    assert cost.footprint_bytes == 8, "the unsized tensor must be excluded, not defaulted"
    assert any("mystery" in why for why in cost.refusals)


def test_an_unparseable_shape_refuses_rather_than_counting_zero():
    cost = lane_cost(_cb({"t": {"shape": "8x8", "dtype": "i8", "role": "intermediate"}}))
    assert cost.is_lower_bound and cost.footprint_bytes == 0
    assert any("shape" in why for why in cost.refusals)


def test_missing_tensors_mapping_refuses():
    cost = lane_cost({"params": {}})
    assert cost.is_lower_bound and cost.traffic_bytes == 0 and cost.refusals


def test_accelerator_lanes_come_from_the_record_not_a_target_spelling():
    placement = [{"lane": "on_mesh", "family": "contraction"},
                 {"lane": "on_mesh", "family": "contraction"},
                 {"lane": "scalar_rvv_lane", "family": "quantize"}]
    cost = lane_cost(_cb({}, placement))
    assert cost.regions_by_lane_family[("on_mesh", "contraction")] == 2
    assert cost.host_lane_region_count == 1
    # A different target's spelling must not read as 100% host.
    other = lane_cost(_cb({}, [{"lane": "matrix_unit", "family": "contraction"}]))
    assert other.host_lane_region_count == 0


def test_absent_placement_reports_no_regions_rather_than_zero_host():
    cost = lane_cost(_cb({"w": {"shape": [4], "dtype": "i8", "role": "weight"}}))
    assert cost.regions_by_lane_family == {}
    assert not cost.is_lower_bound, "no placement record is not a byte-accounting refusal"


def test_format_report_marks_a_lower_bound_visibly():
    cost = lane_cost(_cb({"t": {"shape": [4], "dtype": "mystery", "role": "intermediate"}}))
    assert "LOWER BOUND" in format_report(cost)
