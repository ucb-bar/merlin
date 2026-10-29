"""Actual ACC-window pairing, source arithmetic and missing-law rejection."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER
from test_dense_segmented_pipeline import chosen, inputs

from merlin.compile.scheduling.dense_panel_pipeline import DRAIN, DenseReadoutComputeAdmission, _verify_dense_epochs
from merlin.compile.scheduling.dense_readout_overlap import (
    propose_dense_readout_overlap,
    schedule_dense_readout_overlap,
)
from merlin.compile.scheduling.dense_segmented_pipeline import batch_dense_segmented_readouts
from merlin.sched.ir import Call, Loop, evaluate, instances


def selected(shape=(3, 4, 25)):
    p, r, s, b, admission = inputs(shape)
    original = chosen(p, r, s, b, admission)

    def store(component):
        return replace(
            component,
            max_issued_readouts=b.m * b.n,
            pending_readouts=2,
            readout_configuration="ordered_snapshot_before_store_issue",
        )

    admission = replace(admission, overlap=store(admission.overlap), reduction=store(admission.reduction))
    original = batch_dense_segmented_readouts(original, p, b, r, s, ADAPTER, admission)
    capability = DenseReadoutComputeAdmission(
        ADAPTER.configuration_identity,
        ADAPTER.numerical_policy_identity,
        "TEST-ONLY-separate-readout-compute-conjunction",
        "TEST-ONLY-ordered-store-snapshot",
        b.m * b.n,
        2,
    )
    return p, r, s, b, original, capability


def test_actual_counted_pair_odd_group_tails_and_full_source_proof():
    p, r, s, b, original, capability = selected()
    result = schedule_dense_readout_overlap(
        original, p, b, r, s, ADAPTER, capability, group_path=("outer_m", "outer_n")
    )
    assert result.realization["source"]["output_tiles"] == p.tiles[0] * p.tiles[2]
    assert result.realization["transformations"][0]["odd_tail"] == 1
    assert any(epoch["disjoint_readout_compute"] for epoch in result.realization["epochs"])
    assert result.realization["dynamic_epoch_drains"] < original.realization["dynamic_epoch_drains"]
    result.materialize_scf().verify()
    result.revalidate()
    for call, env in instances(result.kernel):
        if call.instr == "store":
            assert 0 <= evaluate(call.arg("acc"), env) < s.accumulator.capacity


def test_same_actual_scalar_source_outputs_with_odd_groups_and_primitive_tails():
    import numpy as np

    from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store

    p, r, storage, b, original, capability = selected()
    actual = schedule_dense_readout_overlap(
        original, p, b, r, storage, ADAPTER, capability, group_path=("outer_m", "outer_n")
    )
    lhs = (np.arange(p.m * p.k) % 7 - 3).reshape(p.m, p.k)
    rhs = (np.arange(p.k * p.n) % 5 - 2).reshape(p.k, p.n)
    output = np.full((p.m, p.n), -10000, dtype=np.int64)
    memories = {name: {} for name in ("lhs", "weight", "acc")}
    descriptor = None
    for call, env in instances(actual.kernel):
        if call.instr == DRAIN:
            continue
        op = ADAPTER.project(replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args)))
        if type(op) is Load:
            source = lhs if op.role == "lhs" else rhs
            for i in range(op.rows):
                for j in range(op.cols):
                    memories[op.role][(op.row + i, j)] = int(source[op.dram_row + i, op.dram_col + j])
        elif type(op) is Preload:
            descriptor = op
        elif type(op) is Compute:
            for i in range(op.rows):
                for j in range(descriptor.cols):
                    slot = (descriptor.accumulator_row + i, j)
                    value = memories["acc"][slot] if descriptor.accumulate else 0
                    for k in range(op.cols):
                        value += (
                            memories["lhs"][(op.input_row + i, k)] * memories["weight"][(descriptor.weight_row + k, j)]
                        )
                    memories["acc"][slot] = value
        else:
            assert type(op) is Store
            for i in range(op.rows):
                for j in range(op.cols):
                    output[op.dram_row + i, op.dram_col + j] = memories["acc"][(op.accumulator_row + i, j)]
    np.testing.assert_array_equal(output, lhs @ rhs)


def test_old_law_same_window_alias_and_unqualified_effects_refuse():
    p, r, s, b, original, capability = selected()
    actual, _ = propose_dense_readout_overlap(
        original.kernel, ADAPTER, s, group_path=("outer_m", "outer_n"), accumulator_rows=b.m * b.n * p.tile_m
    )
    with pytest.raises(ValueError, match="separate completion epoch"):
        _verify_dense_epochs(
            actual,
            p,
            r,
            s,
            ADAPTER,
            original.admission.overlap,
            static_budget=128,
            timeout_ms=5000,
            ordered_accumulator_commits=True,
        )
    # Move every relocated second-window accumulator descriptor back to the
    # first window; reject the ACTUAL mixed epoch even before scalar checking.
    rows = b.m * b.n * p.tile_m

    def corrupt(body, second=False):
        out = []
        for call in body:
            if type(call) is Loop:
                out.append(replace(call, body=corrupt(call.body, second or "readout_second" in call.var)))
            elif second and call.instr == "preload":
                out.append(replace(call, args=tuple((k, v + (-rows) if k == "acc" else v) for k, v in call.args)))
            else:
                out.append(call)
        return tuple(out)

    with pytest.raises(ValueError, match="overlapping in-flight"):
        _verify_dense_epochs(
            replace(actual, body=corrupt(actual.body)),
            p,
            r,
            s,
            ADAPTER,
            original.admission.overlap,
            static_budget=128,
            timeout_ms=5000,
            ordered_accumulator_commits=True,
            readout_compute_admission=capability,
        )
    # Build the alias by deliberately using an encoder that drops ACC shifts.
    rows = b.m * b.n * p.tile_m

    def encode(op):
        from merlin.compile.scheduling.block_schedule import Preload, Store

        if type(op) in (Preload, Store):
            op = replace(op, accumulator_row=op.accumulator_row + (-rows))
        return ADAPTER.encode(op)

    with pytest.raises(ValueError, match="roundtrip"):
        propose_dense_readout_overlap(
            original.kernel,
            replace(ADAPTER, encode=encode),
            s,
            group_path=("outer_m", "outer_n"),
            accumulator_rows=rows,
        )
    with pytest.raises(ValueError, match="two aligned"):
        propose_dense_readout_overlap(
            original.kernel,
            ADAPTER,
            replace(s, accumulator=replace(s.accumulator, capacity=rows)),
            group_path=("outer_m", "outer_n"),
            accumulator_rows=rows,
        )
    with pytest.raises(ValueError, match="separate snapshot"):
        schedule_dense_readout_overlap(
            original, p, b, r, s, ADAPTER, replace(capability, pending_readouts=3), group_path=("outer_m", "outer_n")
        )

    def unknown(body):
        return tuple(
            replace(x, body=unknown(x.body))
            if type(x) is Loop
            else Call("unqualified", ())
            if x.instr == "weight_load"
            else x
            for x in body
        )

    with pytest.raises(ValueError):
        propose_dense_readout_overlap(
            replace(original.kernel, body=unknown(original.kernel.body)),
            ADAPTER,
            s,
            group_path=("outer_m", "outer_n"),
            accumulator_rows=rows,
        )


def test_exact_checking_projection_rejects_changed_data_windows_and_unknown_expressions():
    from merlin.compile.scheduling.dense_readout_overlap import DenseReadoutCheckingProjection, _affine_window_bounds
    from merlin.sched.ir import Const

    p, r, storage, b, original, capability = selected()
    rows = b.m * b.n * p.tile_m
    path = ("outer_m", "outer_n")
    actual, _ = propose_dense_readout_overlap(original.kernel, ADAPTER, storage, group_path=path, accumulator_rows=rows)
    projection = DenseReadoutCheckingProjection(original.kernel, path, rows)
    canonical, receipt = projection.checking_kernel(actual, ADAPTER, storage)
    assert canonical == original.kernel
    assert receipt["paired_domains"][0]["odd_tail"] == 1

    def corrupt(body):
        return tuple(
            replace(s, body=corrupt(s.body))
            if type(s) is Loop
            else replace(s, args=tuple((k, v + 1 if k == "col" else v) for k, v in s.args))
            if s.instr == "store"
            else s
            for s in body
        )

    with pytest.raises(ValueError, match="all actual commands"):
        projection.checking_kernel(replace(actual, body=corrupt(actual.body)), ADAPTER, storage)

    def bad_window(body):
        return tuple(
            replace(s, body=bad_window(s.body))
            if type(s) is Loop
            else replace(s, args=tuple((k, v + 1 if k == "acc" else v) for k, v in s.args))
            if s.instr in ("preload", "store")
            else s
            for s in body
        )

    escaped = replace(original.kernel, body=bad_window(original.kernel.body))
    escaped_actual, _ = propose_dense_readout_overlap(escaped, ADAPTER, storage, group_path=path, accumulator_rows=rows)
    with pytest.raises(ValueError, match="escapes one injectively"):
        DenseReadoutCheckingProjection(escaped, path, rows).checking_kernel(escaped_actual, ADAPTER, storage)
    with pytest.raises(ValueError, match="unknown original ACC"):
        _affine_window_bounds(object(), {})
    # An arbitrary callable cannot furnish a replacement source kernel.
    with pytest.raises(ValueError, match="exact typed transformation"):
        _verify_dense_epochs(
            actual,
            p,
            r,
            storage,
            ADAPTER,
            original.admission.overlap,
            static_budget=128,
            timeout_ms=5000,
            ordered_accumulator_commits=True,
            readout_compute_admission=capability,
            readout_source_projection=lambda: original.kernel,
        )
    assert _affine_window_bounds(Const(3), {}) == (3, 3)


def test_wide_readout_owner_checks_every_actual_band_and_preserves_full_boundary():
    from merlin.compile.scheduling.dense_readout_overlap import admit_readout_overlap_tile_loads
    from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
    from merlin.compile.scheduling.tile_load_coalescing import (
        TileLoadAddressLaw,
        TileLoadMapping,
        _linear,
        propose_tile_load_coalescing,
    )
    from merlin.sched.ir import Const

    p, r, s, b, original, capability = selected((4, 6, 24))
    paired = schedule_dense_readout_overlap(
        original, p, b, r, s, ADAPTER, capability, group_path=("outer_m", "outer_n")
    )
    laws = {
        role: TileLoadAddressLaw(tile, rows, 4 * tile, 1, pitch, stride, "TEST-only-exact-wide-address-law")
        for role, tile, rows, pitch, stride in [
            ("lhs", p.tile_k, p.tile_m, p.k, p.tile_m),
            ("weight", p.tile_n, p.tile_k, p.n, p.tile_k),
        ]
    }

    def project(call):
        if call.instr not in ("lhs_load", "weight_load"):
            return None
        op = ADAPTER.project(call)
        view = s.lhs if op.role == "lhs" else s.rhs
        law = laws[op.role]
        return TileLoadMapping(
            op.role, view.backing, op.dram_row * law.source_pitch_bytes + op.dram_col, op.row, op.rows, op.cols, law
        )

    def encode(call, columns):
        return replace(call, args=tuple((k, Const(columns) if k == "cols" else v) for k, v in call.args))

    def band(call, mapping):
        offset, terms = _linear(mapping.source_byte)
        base, oldterms = _linear(project(call).source_byte)
        assert terms == oldterms
        columns = offset - base
        return replace(
            call,
            args=tuple(
                (
                    k,
                    mapping.local_row
                    if k == "dst"
                    else mapping.columns
                    if k == "cols"
                    else v + columns
                    if k == "src_col"
                    else v + columns // mapping.law.tile_columns
                    if k == "b"
                    else v,
                )
                for k, v in call.args
            ),
        )

    proposal, changes = propose_tile_load_coalescing(paired.kernel, project=project, encode=encode, max_tiles=4)
    assert changes["changes"]
    result = admit_readout_overlap_tile_loads(paired, proposal, p, b, r, s, ADAPTER, project=project, encode_band=band)
    assert (
        result.realization["tile_load_projection"]["expanded_logical_tile_bands"]
        > result.realization["tile_load_projection"]["physical_load_commands"]
    )
    assert result.realization["dynamic_epoch_drains"] == paired.realization["dynamic_epoch_drains"]
    assert result.realization["source"]["output_tiles"] == paired.realization["source"]["output_tiles"]
    assert result.realization["original_boundary"]["output_readout_identity"] is None
    result.materialize_scf().verify()
    result.revalidate()
    with pytest.raises(ValueError, match="original checked panel"):
        admit_panel_tile_loads(paired, proposal, p, b, r, s, ADAPTER, project=project, encode_band=band)

    def wrong_pitch(call):
        mapping = project(call)
        return (
            None
            if mapping is None
            else replace(mapping, law=replace(mapping.law, source_pitch_bytes=mapping.law.source_pitch_bytes + 1))
        )

    with pytest.raises(ValueError, match="source layout"):
        admit_readout_overlap_tile_loads(paired, proposal, p, b, r, s, ADAPTER, project=wrong_pitch, encode_band=band)

    def wrong_band(call, mapping):
        offset, _ = _linear(mapping.source_byte)
        base, _ = _linear(project(call).source_byte)
        if offset > base:
            mapping = replace(mapping, local_row=mapping.local_row + 1)
        return band(call, mapping)

    with pytest.raises(ValueError):
        admit_readout_overlap_tile_loads(paired, proposal, p, b, r, s, ADAPTER, project=project, encode_band=wrong_band)
