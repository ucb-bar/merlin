"""`merlin.perf.offload` must under-credit the accelerator, never over-credit it.

The metric decides whether a compiler change helped a fleet of workloads, so the direction of its
uncertainty is load-bearing: a command whose geometry cannot be read must make the routed total a
LOWER bound (and the offload fraction an UPPER bound), because silently crediting it would flatter
the unit. This is the same failure shape the tree keeps re-finding in the other direction — a metric
completed by a default reads exactly like a measurement.
"""
from __future__ import annotations

from merlin.perf.offload import format_report, offload_report


def _cb(tensors=None, commands=(), placement=None):
    cb = {"tensors": tensors or {}, "commands": list(commands)}
    if placement is not None:
        cb["params"] = {"lane_placement": placement}
    return cb


def test_convolution_macs_come_from_the_declared_geometry_not_operand_shapes():
    # A prepacked weight has had its shape rewritten, so only the declaration is trustworthy.
    cb = _cb({"ifm": {"shape": [1, 3, 8, 8], "dtype": "i8", "role": "intermediate"},
              "w": {"shape": [16, 27], "dtype": "i8", "role": "weight"},        # prepacked [Co,K]
              "out": {"shape": [1, 16, 8, 8], "dtype": "i32", "role": "intermediate"}},
             [{"opcode": "CONV2D", "operands": {"ifm": "ifm", "weight": "w", "dst": "out"},
               "attributes": {"kernel": [3, 3, 3, 16], "stride": [1, 1]}}])
    r = offload_report(cb)
    # every output element costs Ci*Kh*Kw MACs
    assert r.routed_macs == 1 * 16 * 8 * 8 * 3 * 3 * 3
    assert not r.routed_is_lower_bound


def test_a_contraction_writing_an_accumulator_is_resolved_through_its_commit():
    cb = _cb({"a": {"shape": [32, 49], "dtype": "i8", "role": "intermediate"},
              "c": {"shape": [32, 345], "dtype": "i32", "role": "intermediate"}},
             [{"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "a", "rhs": "b_res", "dst": "acc_c"}},
              {"opcode": "COMMIT", "operands": {"src": "acc_c", "dst": "c"}}])
    r = offload_report(cb)
    assert r.routed_macs == 32 * 345 * 49
    assert not r.routed_is_lower_bound


def test_an_unreadable_command_makes_the_total_a_lower_bound_never_zero():
    cb = _cb({"a": {"shape": [4, 4], "dtype": "i8", "role": "intermediate"},
              "c": {"shape": [4, 4], "dtype": "i32", "role": "intermediate"}},
             [{"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "a", "rhs": "r", "dst": "acc_c"},
               "attributes": {}},
              {"opcode": "COMMIT", "operands": {"src": "acc_c", "dst": "c"}},
              {"opcode": "MYSTERY_OP", "operands": {}, "attributes": {}}])
    r = offload_report(cb)
    assert r.routed_macs == 4 * 4 * 4, "the readable command must still be credited"
    assert r.routed_is_lower_bound
    assert any("MYSTERY_OP" in why for why in r.refusals)


def test_movement_commands_do_no_arithmetic_and_that_is_not_a_refusal():
    cb = _cb({}, [{"opcode": "RES_PACK", "operands": {"src": "w", "dst": "w_res"}},
                  {"opcode": "EVICT", "operands": {"handle": "w_res"}}])
    r = offload_report(cb)
    assert r.routed_macs == 0
    assert not r.routed_is_lower_bound, "a pack/evict is not an unreadable contraction"


def test_offload_fraction_counts_contractions_by_the_programs_own_lane_record():
    placement = [{"lane": "on_mesh", "family": "contraction"}] * 31 + \
                [{"lane": "scalar_rvv_lane", "family": "contraction"}] * 12 + \
                [{"lane": "scalar_rvv_lane", "family": "layout"}] * 131
    r = offload_report(_cb({}, (), placement))
    assert r.contractions_on_unit == 31 and r.contractions_off_unit == 12
    assert abs(r.contraction_offload_fraction - 31 / 43) < 1e-12
    assert r.unit_regions == 31 and r.host_regions == 143


def test_a_target_that_spells_its_unit_lane_differently_is_not_read_as_all_host():
    r = offload_report(_cb({}, (), [{"lane": "matrix_unit", "family": "contraction"}]))
    assert r.contractions_on_unit == 1 and r.contractions_off_unit == 0
    assert r.host_regions == 0


def test_no_declared_contraction_reports_no_fraction_rather_than_zero():
    r = offload_report(_cb({}, (), [{"lane": "scalar_rvv_lane", "family": "layout"}]))
    assert r.contraction_offload_fraction is None, "0/0 is not 0%"


def test_format_report_marks_a_lower_bound_visibly():
    cb = _cb({}, [{"opcode": "MYSTERY_OP", "operands": {}}])
    assert "LOWER bound" in format_report(offload_report(cb))
