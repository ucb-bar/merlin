"""Validate selected convolution transport and output-group obligations."""

from __future__ import annotations

from merlin.compiler.schedule.convolution_choices import _law_digest as _law_digest
from merlin.compiler.schedule.convolution_choices import _rhs_intervals as _rhs_intervals
from merlin.compiler.schedule.window_transfer import (
    ConvolutionOutputTile,
    ConvolutionReadoutEpochLaw,
    ConvolutionReadoutTransferLaw,
    plan_convolution_readout_transfers,
)


def _verify_input_first_use(schedule, rows, group, blocks, grouped, law):
    if law is None:
        return
    from merlin.compiler.schedule.convolution_input_first_use import PendingConvolutionInput

    if not grouped or len(blocks) != 1:
        raise ValueError("first-use input needs one complete original grouped consumer")
    PendingConvolutionInput(schedule, group, rows, law, image=0)


def _verify_batch_input_prefetch(schedule, rows, group, blocks, grouped, law):
    if law is None:
        return
    from merlin.compiler.schedule.rolling_lookahead import RollingBatchFrontier, RollingBatchFrontierAdmission

    if (
        type(law) is not RollingBatchFrontierAdmission
        or not grouped
        or group is None
        or rows is None
        or schedule.epoch_law is None
        or len(blocks) != 1
    ):
        raise ValueError("batch input frontier requires its complete original consumer and target law")
    law.require()
    if (
        law.hardware_identity != schedule.epoch_law.hardware_identity
        or law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
    ):
        raise ValueError("batch input frontier differs from the original target/numerical scope")
    p = schedule.source.plan
    runs = p.window.kernel[0] * p.window.kernel[1] * ((p.window.input_shape[-1] + p.channel_tile - 1) // p.channel_tile)
    RollingBatchFrontier(schedule.source.identity, p.channel_tile).stages(rows.rolling_plan(), group.output_rows, runs)


def _verify_input_prefetch(schedule, rows, group, blocks, grouped, law):
    if law is None:
        return
    from merlin.compiler.schedule.rolling_lookahead import RollingLookaheadAdmission, _validate

    if (
        type(law) is not RollingLookaheadAdmission
        or not grouped
        or group is None
        or rows is None
        or schedule.epoch_law is None
        or len(blocks) != 1
    ):
        raise ValueError("input prefetch needs a complete grouped K-run consumer and selected target law")
    law.require()
    if not law.retained_consumer_continuations or law.hardware_identity != schedule.epoch_law.hardware_identity:
        raise ValueError("input prefetch lacks the original continuation/target admission")
    _validate(rows.rolling_plan(), group.output_rows)


def _phase_blocks(schedule, intervals):
    if intervals is None:
        return schedule.column_blocks
    columns = {column for column, _ in _rhs_intervals(schedule, intervals)}
    return tuple(
        part
        for block in schedule.column_blocks
        if (part := tuple(tile for tile in block if tile.output_column in columns))
    )


def _verify_readout_law(schedule, law, blocks):
    if law is None:
        return
    if (
        type(law) is not ConvolutionReadoutEpochLaw
        or not schedule.choice.load_issue
        or schedule.choice.compute_pairs <= 1
    ):
        raise ValueError("convolution readout epochs require a typed law and issued compute epochs")
    for block in blocks:
        law.require(schedule.source.plan, schedule.source.plan.row_tile, block)
    if (
        law.hardware_identity != schedule.epoch_law.hardware_identity
        or law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
    ):
        raise ValueError("convolution readout and compute epochs select different contracts")


def _verify_readout_transfers(schedule, law, readout_law, blocks):
    if law is None:
        return
    if (
        type(law) is not ConvolutionReadoutTransferLaw
        or readout_law is None
        or law.hardware_identity != readout_law.hardware_identity
        or law.numerical_policy_identity != readout_law.numerical_policy_identity
    ):
        raise ValueError("wide readout needs its separate map and original producer/readout epoch")
    for block in blocks:
        plan_convolution_readout_transfers(schedule.source.plan, block, law, rows=schedule.source.plan.row_tile)


def _verify_spatial_output_group(
    schedule,
    builder,
    lower,
    complete,
    *,
    rhs,
    input_rows,
    spatial_iteration,
    readout_law,
    readout_transfer_law,
    output_group,
    array_law,
    blocks,
):
    """Check active original column ownership and budgets before target effects."""
    p = schedule.source.plan
    group_identity = None
    array_identity = None
    if output_group is not None or array_law is not None:
        from merlin.compiler.schedule.convolution_output_groups import (
            ConvolutionOutputGroup,
            ConvolutionRowFragmentGroup,
            plan_rolling_convolution_output_group,
            plan_rolling_convolution_row_fragments,
        )
        from merlin.compiler.schedule.dense_array_retention import DenseArrayRetentionAdmission
        from merlin.compiler.schedule.window_transfer import _validate_convolution_load_issue

        if (
            type(array_law) is not DenseArrayRetentionAdmission
            or type(output_group) not in (ConvolutionOutputGroup, ConvolutionRowFragmentGroup)
            or spatial_iteration != "rows"
            or readout_law is None
        ):
            raise ValueError("spatial grouping needs separate array/readout premises")
        array_law.__post_init__()
        if (
            schedule.retention_law is None
            or not schedule.retention_law.independent_output_rows
            or not schedule.retention_law.no_intermediate_observers
        ):
            raise ValueError("spatial grouping needs original independent-row and observer permissions")
        planner = (
            plan_rolling_convolution_row_fragments
            if type(output_group) is ConvolutionRowFragmentGroup
            else plan_rolling_convolution_output_group
        )
        expected = planner(
            schedule, rhs, input_rows, max_row_blocks=output_group.row_blocks, max_output_rows=output_group.output_rows
        )
        if expected is None or output_group != expected:
            raise ValueError("spatial output group differs from original rolling/RHS ownership")
        if (
            array_law.hardware_identity != schedule.epoch_law.hardware_identity
            or array_law.numerical_policy_identity != schedule.epoch_law.numerical_policy_identity
        ):
            raise ValueError("spatial array retention selects a different original contract")
        for block in blocks:
            _validate_convolution_load_issue(
                p,
                builder,
                lower,
                complete,
                p.row_tile,
                schedule.issue_law,
                output_group.row_blocks * len(block) * int(p.bias),
            )
            if output_group.row_blocks * len(block) > readout_law.max_issued_readouts:
                raise ValueError("spatial group exceeds selected original readout work budget")
            for m in range(output_group.row_blocks):
                tiles = tuple(
                    ConvolutionOutputTile(
                        t.output_column,
                        t.columns,
                        p.accumulator_row + output_group.accumulator_row(m, n, active_column_blocks=len(block)),
                    )
                    for n, t in enumerate(block)
                )
                readout_law.require(p, p.row_tile, tiles)
                if readout_transfer_law is not None:
                    plan_convolution_readout_transfers(p, tiles, readout_transfer_law, rows=p.row_tile)
        group_identity, array_identity = _law_digest(output_group), _law_digest(array_law)
    return group_identity, array_identity


def _verify_grouped_transport(schedule, input_rows, output_group):
    from merlin.compiler.schedule.rolling_window import (
        materialize_grouped_rolling_window,
        verify_grouped_rolling_window,
    )

    if output_group is None or input_rows is None or schedule.issue_law is None or not schedule.choice.load_issue:
        raise ValueError("grouped transport needs checked spatial ownership and bounded target issue")
    ring = input_rows.rolling_plan()
    module = materialize_grouped_rolling_window(ring, output_rows=output_group.output_rows)
    report = verify_grouped_rolling_window(module, ring, output_rows=output_group.output_rows)
    commands = input_rows.stage_command_bound(report["max_staged_rows_per_epoch"])
    if commands > schedule.issue_law.load_issue_budget:
        raise ValueError("grouped transport exceeds selected actual transfer issue bound")
    return commands


# Preserve the original public lookup and pickle identities.
for _definition in (
    _verify_input_first_use,
    _verify_batch_input_prefetch,
    _verify_input_prefetch,
    _phase_blocks,
    _verify_readout_law,
    _verify_readout_transfers,
    _verify_spatial_output_group,
    _verify_grouped_transport,
):
    _definition.__module__ = "merlin.compiler.schedule.convolution_exploration"
del _definition
