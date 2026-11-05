"""Biased full-K input overlap keeps original initialization completions."""

from dataclasses import replace

import numpy as np
import pytest
from test_compact_dense import mutate
from test_compact_dense_bias import ADAPTER
from test_dense_retained_reduction import inputs

from merlin.compile.scheduling.block_schedule import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import DenseBiasSource, emit_compact_dense
from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout, TiledIntegerStore
from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN, DenseReadoutConfigurationSnapshot, DenseReadyReadoutAdmission,
    dense_retained_compute_order, dense_retained_operand_hoists, schedule_dense_panel_pipeline,
)
from merlin.compile.scheduling.dense_ready_readout import dense_ready_input_choices, schedule_dense_ready_readout
from merlin.compile.scheduling.loop_schedule import propose_loop_transforms
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.sched.ir import Call, evaluate, instances


def prepared(shape):
    p, r, s, b, admission = inputs(shape)
    p = replace(p, bias=True)
    placement = next(c for c in dense_ready_input_choices(p, r, s)
                     if (c.block.m, c.block.n) == (b.m, b.n))
    b = placement.block
    bias = DenseBiasSource(p.n, "i32")
    readout = DenseTiledIntegerReadout.from_source_scales(
        [0.5 if (i // p.tile_n) % 2 else 0.25 for i in range(p.n)],
        columns=p.n, tile_columns=p.tile_n, relu=True, instruction_name="scaled_store")

    def encode(op):
        if type(op) is TiledIntegerStore:
            raw = ADAPTER.encode(Store(op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row))
            return replace(raw, instr="scaled_store", args=(*raw.args, ("tile", op.readout_tile)))
        return ADAPTER.encode(op)

    def project(call):
        if call.instr == "scaled_store":
            raw = ADAPTER.project(replace(call, instr="store", args=call.args[:-1]))
            return TiledIntegerStore(raw.dram_row, raw.dram_col, raw.rows, raw.cols,
                                     raw.accumulator_row, readout.identity, call.arg("tile"))
        return ADAPTER.project(call)

    adapter = replace(ADAPTER, encode=encode, project=project)
    admission = replace(admission, hardware_identity=adapter.configuration_identity,
        numerical_policy_identity=adapter.numerical_policy_identity,
        max_issued_readouts=b.m*b.n, pending_readouts=2,
        readout_configuration="ordered_snapshot_before_store_issue")
    original = emit_compact_dense(p, b, s, adapter, name="biased", dtypes=("i8", "i8", "i32"),
        outer_order=("n", "m"), output_readout=readout, bias_source=bias)
    hoists = dense_retained_operand_hoists(original, adapter, role="weight")
    hoisted = propose_loop_transforms(original, hoists)
    transforms = hoists + dense_retained_compute_order(hoisted, adapter, order=("k", "n", "m"))
    plan = make_dense_schedule_plan(p, r, s, adapter, name="biased", source_identity="b"*64,
        dtypes=("i8", "i8", "i32"), static_budget=256, block_tiles=(b.m, b.k, b.n),
        outer_order=("n", "m"), loop_transforms=transforms, pipeline_kind="retained_full_k",
        pipeline_prefetch=False, output_readout=readout, bias_source=bias)
    original = schedule_dense_panel_pipeline(p, r, s, adapter, admission, name="biased",
        source_identity="b"*64, dtypes=("i8", "i8", "i32"), schedule_plan=plan,
        external_buffers_disjoint=True, static_budget=256, output_readout=readout,
        bias_source=bias, readout_qualification_identity="TEST-ONLY-original-scaled-output")
    law = DenseReadyReadoutAdmission(adapter.configuration_identity, adapter.numerical_policy_identity,
        "TEST-ONLY-producer-RAW", "TEST-ONLY-ordered-store", b.m*b.n, 2,
        configuration_snapshot=DenseReadoutConfigurationSnapshot(adapter.configuration_identity,
            adapter.numerical_policy_identity, "TEST-ONLY-captured-request-scale", readout.identity),
        disjoint_load_qualification_identity="TEST-ONLY-disjoint-LD-EX-ST")
    return p, r, s, b, adapter, original, law, placement, readout


def execute(kernel, p, adapter, readout):
    """Execute actual addresses; delay stores until the next original drain."""
    lhs = (np.arange(p.m*p.k) % 7 - 3).reshape(p.m, p.k)
    rhs = (np.arange(p.k*p.n) % 5 - 2).reshape(p.k, p.n)
    bias = np.resize(np.array([2**31-1, -2**31, 0, 17, -23], dtype=np.int64), p.n)
    memory = {role: {} for role in ("lhs", "weight", "acc")}
    output = np.zeros((p.m, p.n), dtype=np.int8)
    writes = np.zeros(output.shape, dtype=np.int8)
    pending, descriptor = [], None

    def finish():
        while pending:
            op = pending.pop()
            for i in range(op.rows):
                for j in range(op.cols):
                    y, x = op.dram_row+i, op.dram_col+j
                    output[y, x] = readout.evaluate(memory["acc"][op.accumulator_row+i, j], x)
                    writes[y, x] += 1

    for call, env in instances(kernel):
        if call == Call(DRAIN, ()):
            finish()
            continue
        op = adapter.project(replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args)))
        if type(op) is InitializeAccumulator:
            assert not pending, "ACC initialization crossed a pending original readout"
            for i in range(op.rows):
                for j in range(op.cols):
                    memory["acc"][op.row+i, j] = int(bias[op.dram_col+i*op.source_row_stride+j*op.source_col_stride])
        elif type(op) is Load:
            source = lhs if op.role == "lhs" else rhs
            for i in range(op.rows):
                for j in range(op.cols):
                    memory[op.role][op.row+i, j] = int(source[op.dram_row+i, op.dram_col+j])
        elif type(op) is Preload:
            descriptor = op
        elif type(op) is Compute:
            for i in range(op.rows):
                for j in range(descriptor.cols):
                    key = descriptor.accumulator_row+i, j
                    total = memory["acc"][key] if descriptor.accumulate else 0
                    total += sum(memory["lhs"][op.input_row+i, k]*memory["weight"][descriptor.weight_row+k, j]
                                 for k in range(op.cols))
                    memory["acc"][key] = (total + 2**31) % 2**32 - 2**31
        else:
            assert type(op) is TiledIntegerStore
            pending.append(op)
    assert not pending
    raw = ((lhs @ rhs + bias + 2**31) % 2**32) - 2**31
    expected = np.array([[readout.evaluate(int(raw[y, x]), x) for x in range(p.n)]
                         for y in range(p.m)], dtype=np.int8)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(writes, 1)


@pytest.mark.parametrize("shape", [(21, 14, 25), (17, 13, 17)])
def test_bias_completion_and_original_varying_readouts_survive_pending_input(shape):
    p, r, s, b, adapter, original, law, placement, readout = prepared(shape)
    separate = schedule_dense_ready_readout(original, p, b, r, s, adapter, law, static_budget=256)
    result = schedule_dense_ready_readout(original, p, b, r, s, adapter, law,
        input_pipeline_placement=placement, static_budget=256)
    execute(result.kernel, p, adapter, readout)
    before = [tuple((k, evaluate(v, e)) for k, v in c.args)
              for c, e in instances(separate.kernel) if c.instr == "bias_init"]
    after = [tuple((k, evaluate(v, e)) for k, v in c.args)
             for c, e in instances(result.kernel) if c.instr == "bias_init"]
    assert before == after
    assert len(after) == p.tiles[0]*p.tiles[2]
    for epoch in result.realization["epochs"]:
        if epoch["initialization"]:
            assert not epoch["compute_pairs"] and not epoch["future_or_initial_loads"] and not epoch["readout"]
    assert all(row["initializer_scopes"] == 1 for row in result.realization["input_pipeline"])
    assert all(0 < row["preserved_initialization_drains"] <= row["groups"]*b.m*b.n
               for row in result.realization["input_pipeline"])
    assert all(row["remaining_drains"] == row["groups"]+1+row["preserved_initialization_drains"]
               for row in result.realization["input_pipeline"])
    assert result.realization["dynamic_epoch_drains"] < separate.realization["dynamic_epoch_drains"]
    result.materialize_scf().verify()
    result.revalidate()
    changed = mutate(result.kernel, "bias_init", "src", 1)
    with pytest.raises(ValueError, match="original source, actual commands"):
        replace(result, kernel=changed).revalidate()
