"""Completed full-image input views survive original column phases unchanged."""

from collections import Counter
from dataclasses import replace

import pytest
import test_convolution_spatial_output_groups as spatial
from test_convolution_rhs_residency import rhs_law
from test_counted_convolution import Integers

from merlin.compile.scheduling.convolution_exploration import (
    ConvolutionRhsPhaseLaw, emit_convolution_schedule, plan_convolution_rhs_phases, plan_convolution_schedule,
)
from merlin.compile.scheduling.convolution_input_reuse import (
    ConvolutionInputReuseAdmission, InvariantConvolutionInput, select_convolution_input_reuse,
)
from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group


def fixture(*, stride=(1, 1), batch=1, slots=None):
    shape = (batch, 7, 6, 5)
    slots = 9 if slots is None else slots
    s, _, rows, _, array, readout = spatial.selected(shape, slots=slots, stride=(stride[0], 1), max_output_rows=7)
    if stride[1] != 1:
        window = replace(s.source.plan.window, stride=stride)
        source = replace(s.source, plan=replace(s.source.plan, window=window, row_tile=min(4, window.output_spatial[1])))
        s = plan_convolution_schedule(source, s.choice, retention_law=s.retention_law,
                                      issue_law=s.issue_law, epoch_law=s.epoch_law)
        rows = replace(rows, window=window, source_identity=source.identity, compute_row_step=stride[1])
    phase = ConvolutionRhsPhaseLaw(s.source.identity, "TEST-ONLY-independent-original-exits", True, True, True, True)
    rhs = plan_convolution_rhs_phases(s, operand_rows=1024, law=rhs_law(), phase_law=phase,
                                    input_rows=rows, max_column_blocks=1)
    groups = plan_rolling_convolution_output_group(s, rhs, rows, max_output_rows=7)
    return s, rhs, rows, groups, array, readout


def law(s):
    return ConvolutionInputReuseAdmission("neutral", "source-i32", s.source.identity,
        "TEST-ONLY-preserving-original-effects", True, True, True, True, True)


@pytest.mark.parametrize("stride", [(1, 1), (2, 2)])
@pytest.mark.parametrize("pending_limit", [0, 2])
def test_ragged_original_products_stores_and_phase_drains_survive_one_input_preparation(monkeypatch, stride, pending_limit):
    args = fixture(stride=stride)
    control_trace, candidate_trace = [], []
    control, fresh, _ = spatial.run(*args, pending_limit=pending_limit, trace=control_trace, grouped_transport=True)
    original = spatial.emit_convolution_schedule
    monkeypatch.setattr(spatial, "emit_convolution_schedule",
        lambda *a, **kw: original(*a, **kw, invariant_input_law=law(args[0])))
    candidate, new_fresh, report = spatial.run(*args, pending_limit=pending_limit, trace=candidate_trace,
                                             grouped_transport=True)
    assert fresh == new_fresh
    for boundary in ["weight_preparation", "spatial_group_last_use", "input_last_use", "rhs_phase_last_use"]:
        assert candidate.count(boundary) == control.count(boundary) == 3
    assert candidate.count("input_ready") == 1 and control.count("input_ready") == 3
    def payload(trace):
        return Counter((kind, tuple(sorted(a.items()))) for kind, a in trace
                       if kind in ("stage_input_rows", "fill_input_rows"))
    assert payload(control_trace) == Counter({k: v * 3 for k, v in payload(candidate_trace).items()})
    # Each exact bias, RHS, compute and scalar store descriptor is unchanged.
    def other(trace):
        return [(kind, a) for kind, a in trace if kind in ("initialize_bias", "load_weight", "compute", "store")]
    assert other(control_trace) == other(candidate_trace)
    reuse = report["rhs_phases"]["invariant_input"]
    assert reuse["prepared_input_traversals"] == 1 and reuse["reused_input_traversals"] == 2
    assert all(p["rolling_transport"]["dynamic_stage_bytes"] == 0 for p in report["phase_emissions"][1:])


@pytest.mark.parametrize("field", ["immutable_source_input", "exclusive_private_input_rows",
    "other_effects_preserve_input_rows", "no_intermediate_observers", "real_preparation_and_phase_completion"])
def test_missing_effect_premise_refuses_before_any_target_preparation(field):
    s, rhs, rows, groups, array, readout = fixture()
    effects = []
    with pytest.raises(ValueError):
        emit_convolution_schedule(s, Integers(), lambda *a: effects.append(a), complete=lambda *a: effects.append(a),
            rhs=rhs, input_rows=rows, output_group=groups, array_law=array, readout_law=readout,
            grouped_transport=True, invariant_input_law=replace(law(s), **{field: False}))
    assert effects == []


@pytest.mark.parametrize("settings", [{"batch": 2}, {"slots": 3}])
def test_multiple_images_or_nonresident_full_union_refuse(settings):
    s, rhs, rows, groups, _, _ = fixture(**settings)
    with pytest.raises(ValueError):
        InvariantConvolutionInput(s, rhs, rows, groups, law(s))


@pytest.mark.parametrize("settings,eligible", [({}, True), ({"stride": (2, 2)}, True),
    ({"batch": 2}, False), ({"slots": 3}, False)])
def test_ordinary_source_selection_preserves_other_consumers(settings, eligible):
    s, rhs, rows, groups, _, _ = fixture(**settings)
    admission = law(s)
    result = select_convolution_input_reuse(s, rhs, rows, groups, admission=admission)
    assert (result is admission) == eligible
    assert result is None or result == admission
    with pytest.raises(ValueError):
        select_convolution_input_reuse(s, rhs, rows, groups, admission=replace(admission, immutable_source_input=False))


def test_no_input_view_or_completion_before_original_phase_order():
    s, rhs, rows, groups, array, readout = fixture()
    owner = InvariantConvolutionInput(s, rhs, rows, groups, law(s))
    with pytest.raises(ValueError):
        owner.emit_reused_group(rhs.phases[1], groups[1], Integers(), lambda *a: None, lambda *a: None,
            blocks=(), array_law=array, readout_law=readout, readout_transfer_law=None, ready_readout=False)
    with pytest.raises(ValueError):
        owner.complete_phase(1, lambda *a: None, {})
    with pytest.raises(ValueError):
        owner.summary()
    with pytest.raises(ValueError):
        owner.complete_phase(0, lambda *a: None, {})
    phase = rhs.phases[0]
    completed = []
    owner.complete_phase(0, lambda *a: completed.append(a),
        dict(phase=0, local_row=phase.layout.rhs_base, local_rows=phase.layout.rhs_rows))
    assert completed[0][0] == "rhs_phase_last_use"
    with pytest.raises(ValueError):
        owner.emit_reused_group(rhs.phases[1], groups[1], Integers(), lambda *a: None, lambda *a: None,
            blocks=(), array_law=array, readout_law=readout, readout_transfer_law=None, ready_readout=False)
