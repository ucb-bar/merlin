"""Producer/readout dependencies close at the original private block boundary."""
from dataclasses import replace

import pytest
from test_convolution_compute_epochs import plan
from test_counted_convolution import Integers

from merlin.compile.scheduling.convolution_exploration import emit_convolution_schedule
from merlin.compile.scheduling.window_transfer import (
    ConvolutionOutputTile,
    ConvolutionReadoutEpochLaw,
    emit_convolution_compute_epoch_column_block,
)


def readout_law():
    return ConvolutionReadoutEpochLaw("neutral", "source-i32", "original-store-source",
        "conditional-source-check", 3, 2, local_read_after_write=True,
        accumulator_result_visibility=True, ordered_configuration_capture=True,
        finite_queue_progress=True, complete_device_drain=True, exclusive_private_state=True)


@pytest.mark.parametrize("pairs", [2, 4])
@pytest.mark.parametrize("kernel", [(3, 2), (3, 1)])
def test_exact_original_effects_finite_stores_and_real_final_drain(pairs, kernel):
    schedule = plan(pairs, kernel=kernel)
    baseline, actual, pending, retired, drains = [], [], [], [], []
    emit_convolution_schedule(schedule, Integers(), lambda k, a: baseline.append((k, dict(a))),
        complete=lambda *_: None)
    final_producer_pending = False
    block_computes = kernel[0]*kernel[1]*2*3
    computes = stores = maximum = 0

    def issue(kind, fields):
        nonlocal final_producer_pending, computes, stores, maximum
        actual.append((kind, dict(fields)))
        if kind == "compute":
            computes += 1
            final_producer_pending = True
        elif kind == "store":
            # First readout is issued without the final producer's full drain.
            # The selected device must provide the real ACC RAW wait instead.
            if stores % 3 == 0:
                assert computes % block_computes == 0 and final_producer_pending
            if len(pending) == 2:
                retired.append(pending.pop(0))
            pending.append(dict(fields))
            maximum = max(maximum, len(pending))
            stores += 1

    def complete(kind, fields):
        nonlocal final_producer_pending
        if kind == "compute":
            assert (computes % block_computes) % pairs == 0 and computes % block_computes != 0
        else:
            assert kind == "store" and stores % 3 == 0
            retired.extend(pending)
            pending.clear()
        final_producer_pending = False
        drains.append(kind)

    result = emit_convolution_schedule(schedule, Integers(), issue, complete=complete,
        readout_law=readout_law())
    assert baseline == actual and not pending and not final_producer_pending
    assert maximum == 2 and len(retired) == stores
    assert drains.count("store") == stores//3
    assert drains.count("compute") == computes//block_computes*((block_computes-1)//pairs)
    assert result["readout_epoch"]["pending_readouts"] == 2
    assert result["qualification"] == "source/effect premises remain independently caller-owned"


@pytest.mark.parametrize("changes", [
    {"local_read_after_write": False}, {"accumulator_result_visibility": False},
    {"ordered_configuration_capture": False}, {"finite_queue_progress": False},
    {"complete_device_drain": False}, {"exclusive_private_state": False},
    {"max_issued_readouts": 2}, {"pending_readouts": 0}, {"pending_readouts": True},
    {"hardware_identity": "different"}, {"numerical_policy_identity": "different"},
])
def test_missing_premises_or_wrong_contract_refuses_before_effects(changes):
    effects = []
    with pytest.raises(ValueError):
        emit_convolution_schedule(plan(4), Integers(), lambda *a: effects.append(a),
            complete=lambda *a: effects.append(a), readout_law=replace(readout_law(), **changes))
    assert effects == []


def test_overlapping_private_rows_refuse_before_issue_and_drain():
    schedule = plan(4)
    tiles = (ConvolutionOutputTile(0, 4, 0), ConvolutionOutputTile(4, 4, 3))
    effects = []
    with pytest.raises(ValueError, match="disjoint"):
        emit_convolution_compute_epoch_column_block(schedule.source.plan, Integers(),
            lambda *a: effects.append(a), lambda *a: effects.append(a), output_row=0,
            rows=4, output_tiles=tiles, law=schedule.retention_law,
            issue_law=schedule.issue_law, epoch_law=schedule.epoch_law,
            compute_pairs=4, readout_law=readout_law())
    assert effects == []


def test_readout_law_cannot_bypass_issued_compute_epoch_contract():
    with pytest.raises(ValueError, match="issued compute epochs"):
        emit_convolution_schedule(plan(1), Integers(), lambda *_: None,
            complete=lambda *_: None, readout_law=readout_law())
