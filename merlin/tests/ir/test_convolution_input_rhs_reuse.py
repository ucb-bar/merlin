"""Completed invariant inputs compose with independently admitted RHS writes."""

from collections import Counter
from dataclasses import replace

import pytest
import test_convolution_spatial_output_groups as spatial
from test_convolution_input_reuse import fixture, law
from test_counted_convolution import Integers

from merlin.compile.scheduling.convolution_exploration import emit_convolution_schedule
from merlin.compile.scheduling.convolution_input_reuse import InvariantConvolutionInput
from merlin.compile.scheduling.rolling_lookahead import RollingRhsFrontier, RollingRhsFrontierAdmission


def frontier_law():
    return RollingRhsFrontierAdmission("neutral", "source-i32", "TEST-ONLY-ordered-real-WAR-RAW",
                                     True, True, True, True, True)


@pytest.mark.parametrize("stride", [(1, 1), (2, 2)])
@pytest.mark.parametrize("pending_limit", [0, 2])
def test_original_products_traffic_and_drains_with_one_input_and_actual_future_rhs(monkeypatch, stride, pending_limit):
    args = fixture(stride=stride)
    original = spatial.emit_convolution_schedule
    monkeypatch.setattr(spatial, "emit_convolution_schedule",
        lambda *a, **kw: original(*a, **kw, invariant_input_law=law(args[0])))
    old_trace, new_trace = [], []
    old, old_fresh, _ = spatial.run(*args, pending_limit=pending_limit, trace=old_trace, grouped_transport=True)
    new, new_fresh, report = spatial.run(*args, pending_limit=pending_limit, trace=new_trace,
        grouped_transport=True, rhs_phase_prefetch_law=frontier_law())
    assert old_fresh == new_fresh
    assert Counter(old) == Counter(new)
    for boundary in ("input_ready", "input_last_use", "spatial_group_last_use", "rhs_phase_last_use",
                     "weight_preparation"):
        assert old.count(boundary) == new.count(boundary)
    # Exact original descriptors survive; only independent issue order changes.
    def effects(trace):
        return Counter((kind, tuple(sorted(a.items()))) for kind, a in trace
                       if kind in ("stage_input_rows", "fill_input_rows", "initialize_bias", "load_weight",
                                   "compute", "store"))
    assert effects(old_trace) == effects(new_trace)
    assert [a for kind, a in old_trace if kind == "compute"] == [a for kind, a in new_trace if kind == "compute"]
    assert old_trace != new_trace
    phases = report["phase_emissions"]
    assert phases[1]["rolling_transport"]["dynamic_stage_bytes"] == 0
    assert phases[1]["rolling_transport"]["rhs_dead_prefix_transfers"] > 0
    assert phases[-1]["rolling_transport"]["rhs_dead_prefix_transfers"] == 0
    assert report["rhs_phases"]["invariant_input"]["prepared_input_traversals"] == 1


@pytest.mark.parametrize("field", ["scratchpad_read_write_dependencies", "ordered_configuration",
    "finite_queue_progress", "immutable_weights", "full_consumer_transport_completion"])
def test_independent_rhs_target_admission_before_any_effect(field):
    s, rhs, rows, groups, array, readout = fixture()
    effects = []
    with pytest.raises(ValueError):
        emit_convolution_schedule(s, Integers(), lambda *a: effects.append(a), complete=lambda *a: effects.append(a),
            rhs=rhs, input_rows=rows, output_group=groups, array_law=array, readout_law=readout,
            grouped_transport=True, invariant_input_law=law(s),
            rhs_phase_prefetch_law=replace(frontier_law(), **{field: False}))
    assert effects == []


@pytest.mark.parametrize("mutation", ["input_overlap", "different_source", "wrong_columns", "final_phase"])
def test_future_rhs_map_cannot_change_retained_input_or_original_phase(mutation):
    s, rhs, rows, groups, _, _ = fixture()
    owner = InvariantConvolutionInput(s, rhs, rows, groups, law(s))
    phase = rhs.phases[0]
    frontier = RollingRhsFrontier(s.source.identity, phase.layout.rhs_base, s.source.plan.channel_tile,
        s.source.plan.output_channels, len(phase.column_intervals), rhs.phases[1].column_intervals)
    index = 0
    if mutation == "input_overlap":
        frontier = replace(frontier, rhs_base=rows.base)
    elif mutation == "different_source":
        frontier = replace(frontier, source_identity="0" * 64)
    elif mutation == "wrong_columns":
        frontier = replace(frontier, next_columns=rhs.phases[0].column_intervals)
    else:
        index = len(rhs.phases) - 1
    with pytest.raises(ValueError):
        owner.check_rhs_frontier(index, frontier, frontier_law())
