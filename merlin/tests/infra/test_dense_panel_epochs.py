"""Fact-backed dense panel epochs: fewer drains, the same data calls, numerics under the ordering law.

The ordering model below executes a counted dense program the way a target with the law's premises
may: every command is pending until a drain, loads complete in any order (here: newest first), a
compute waits only for the older loads into its rows (read after write), a load waits only for the
older computes that read its rows (write after read), computes are ordered among themselves only by
accumulator order, and a readout waits for the compute that wrote its accumulator rows. Dropping a
premise from the model shows the race the matching drain prevented.
"""

from dataclasses import replace

import numpy as np
import pytest
from test_compact_dense import ADAPTER
from test_dense_panel_pipeline import setup

from merlin.compile.scheduling.block_schedule import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.compile.scheduling.dense_panel_epochs import (
    EPOCH_FLAGS,
    DensePanelEpochLaw,
    DensePanelEpochResult,
    propose_dense_panel_epochs,
    schedule_dense_panel_epochs,
)
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN, _serial_projection, schedule_dense_panel_pipeline
from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_readouts
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.sched.ir import evaluate, instances
from merlin.transforms.dense_planning import dense_blocks

DTYPES = ("i8", "i8", "i32")


def law(adapter=ADAPTER, **unknown):
    flags = {flag: True for flag in EPOCH_FLAGS}
    flags.update(unknown)
    return DensePanelEpochLaw(
        adapter.configuration_identity, adapter.numerical_policy_identity, "test-only-ordering-premises", **flags
    )


def original(*, prefetch=True, issue_order="compute_then_prefetch", batch=False, biased=False, budget=64):
    p, r, s, b, a = setup()
    adapter, bias = ADAPTER, None
    if biased:
        from test_compact_dense_bias import ADAPTER as bias_adapter

        from merlin.compile.scheduling.compact_dense import DenseBiasSource

        p, adapter, bias = replace(p, bias=True), bias_adapter, DenseBiasSource(p.n, "i32")
        b = next(x for x in dense_blocks(p, replace(r, max_k_blocks=1)) if (x.m, x.n) == (2, 2))
        a = replace(
            a,
            hardware_identity=adapter.configuration_identity,
            numerical_policy_identity=adapter.numerical_policy_identity,
        )
    a = replace(a, max_issued_compute_pairs=budget, max_issued_loads=budget)
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        adapter,
        source_identity="a" * 64,
        dtypes=DTYPES,
        static_budget=128,
        block_tiles=(b.m, b.k, b.n),
        pipeline_prefetch=prefetch,
        pipeline_issue_order=issue_order,
        bias_source=bias,
    )
    result = schedule_dense_panel_pipeline(
        p,
        r,
        s,
        adapter,
        a,
        name="epochs",
        dtypes=DTYPES,
        schedule_plan=plan,
        source_identity="a" * 64,
        external_buffers_disjoint=True,
        bias_source=bias,
    )
    if batch:
        a = replace(
            a, max_issued_readouts=4, pending_readouts=1, readout_configuration="ordered_snapshot_before_store_issue"
        )
        result = batch_dense_panel_readouts(result, p, b, r, s, adapter, a)
    return p, r, s, b, a, adapter, result


def drains(kernel):
    return sum(1 for call, _ in instances(kernel) if call.instr == DRAIN)


def ordered_execution(kernel, adapter, p, s, *, raw=True, war=True, accumulator_order=True, visibility=True):
    """Run on host integers under the ordering premises; loads land newest first between drains."""
    lhs = (np.arange(p.m * p.k) % 7 - 3).reshape(p.m, p.k)
    rhs = (np.arange(p.k * p.n) % 5 - 2).reshape(p.k, p.n)
    bias = np.resize(np.array([2**31 - 1, -(2**31), 127, -128, 0], dtype=np.int64), p.n)
    output = np.full((p.m, p.n), -10000, dtype=np.int64)
    memory, nodes = {}, []
    writer, readers = {}, {}

    def rows(view, row, count):
        return [(view.backing, row + i) for i in range(count)]

    def add(kind, operation, pre=None):
        index, deps = len(nodes), set()
        if kind == "load":
            view = s.lhs if operation.role == "lhs" else s.rhs
            reads, writes = [], rows(view, operation.row, operation.rows)
        elif kind == "init":
            reads, writes = [], rows(s.accumulator, operation.row, operation.rows)
        elif kind == "pair":
            reads = rows(s.lhs, operation.input_row, operation.rows) + rows(s.rhs, pre.weight_row, pre.weight_rows)
            writes = rows(s.accumulator, pre.accumulator_row, pre.rows)
            if accumulator_order:
                previous = [n for n in range(index) if nodes[n][0] == "pair"]
                deps.update(previous[-1:])
        else:
            reads, writes = rows(s.accumulator, operation.accumulator_row, operation.rows), []
            deps.update(n for n in range(index) if nodes[n][0] == "store")
        for row in reads:
            source = writer.get(row)
            if source is not None:
                ordered = {"load": raw, "init": True, "pair": visibility if kind == "store" else accumulator_order}
                if ordered[nodes[source][0]]:
                    deps.add(source)
            readers.setdefault(row, []).append(index)
        for row in writes:
            for reader in readers.get(row, ()):
                # a load waits for older readers only under write-after-read; accumulator reuse after a
                # readout and accumulator order are separate premises of their own queues
                if nodes[reader][0] == "pair" and kind == "load" and war or nodes[reader][0] == "store":
                    deps.add(reader)
            source = writer.get(row)
            if source is not None and nodes[source][0] != "load" and accumulator_order:
                deps.add(source)
            writer[row], readers[row] = index, []
        nodes.append((kind, operation, pre, deps))

    def execute(kind, operation, pre):
        if kind == "load":
            view = s.lhs if operation.role == "lhs" else s.rhs
            source = lhs if operation.role == "lhs" else rhs
            for i in range(operation.rows):
                for j in range(operation.cols):
                    memory[view.backing, operation.row + i, j] = int(
                        source[operation.dram_row + i, operation.dram_col + j]
                    )
        elif kind == "init":
            for i in range(operation.rows):
                for j in range(operation.cols):
                    memory[s.accumulator.backing, operation.row + i, j] = int(bias[operation.dram_col + j])
        elif kind == "pair":
            for i in range(operation.rows):
                for j in range(pre.cols):
                    key = s.accumulator.backing, pre.accumulator_row + i, j
                    value = memory.get(key, 0) if pre.accumulate else 0
                    for k in range(operation.cols):
                        value += memory.get((s.lhs.backing, operation.input_row + i, k), 0) * memory.get(
                            (s.rhs.backing, pre.weight_row + k, j), 0
                        )
                    memory[key] = (value + 2**31) % 2**32 - 2**31
        else:
            for i in range(operation.rows):
                for j in range(operation.cols):
                    output[operation.dram_row + i, operation.dram_col + j] = memory.get(
                        (s.accumulator.backing, operation.accumulator_row + i, j), 0
                    )

    def drain():
        done = set()
        while len(done) < len(nodes):
            ready = [n for n in range(len(nodes)) if n not in done and nodes[n][3] <= done]
            chosen = max(ready)
            execute(*nodes[chosen][:3])
            done.add(chosen)
        nodes.clear()
        writer.clear()
        readers.clear()

    preload = None
    for call, env in instances(kernel):
        if call.instr == DRAIN:
            drain()
            continue
        operation = adapter.project(replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args)))
        if type(operation) is Preload:
            preload = operation
        elif type(operation) is Compute:
            add("pair", operation, preload)
            preload = None
        elif type(operation) is Load:
            add("load", operation)
        elif type(operation) is InitializeAccumulator:
            add("init", operation)
        else:
            assert isinstance(operation, Store)
            add("store", operation)
    drain()
    expected = (lhs @ rhs + (bias if p.bias else 0)).astype(np.int32).astype(np.int64)
    return output, expected


@pytest.mark.parametrize(
    "prefetch,issue_order",
    [(True, "compute_then_prefetch"), (False, "compute_then_prefetch"), (True, "prefetch_rhs_between_columns")],
)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("biased", [False, True])
def test_law_removes_k_step_drains_and_keeps_numerics_under_out_of_order_loads(prefetch, issue_order, batch, biased):
    p, r, s, b, a, adapter, result = original(prefetch=prefetch, issue_order=issue_order, batch=batch, biased=biased)
    checked = schedule_dense_panel_epochs(result, p, b, r, s, adapter, a, law(adapter))
    report = checked.realization["panel_epochs"]
    assert type(checked) is DensePanelEpochResult
    assert _serial_projection(checked.kernel) == _serial_projection(result.kernel)
    assert drains(checked.kernel) == report["dynamic_drains"] < report["input_dynamic_drains"] == drains(result.kernel)
    streams = report["streams"]
    assert streams and all(stream["readout_merged"] is batch for stream in streams)
    # one drain per output block (K stream with its batched readout) or two without batching
    blocks = sum(stream["instances"] for stream in streams)
    if batch and not biased:
        assert report["dynamic_drains"] == blocks
    output, expected = ordered_execution(checked.kernel, adapter, p, s)
    np.testing.assert_array_equal(output, expected)
    checked.materialize_scf().verify()


@pytest.mark.parametrize("missing", ["raw", "war"])
def test_the_ordering_model_detects_the_race_a_missing_premise_would_leave(missing):
    p, r, s, b, a, adapter, result = original(batch=True)
    checked = schedule_dense_panel_epochs(result, p, b, r, s, adapter, a, law(adapter))
    output, expected = ordered_execution(result.kernel, adapter, p, s, **{missing: False})
    np.testing.assert_array_equal(output, expected)  # the original drains order every K step
    output, expected = ordered_execution(checked.kernel, adapter, p, s, **{missing: False})
    assert not np.array_equal(output, expected)


@pytest.mark.parametrize("flag", [f for f in EPOCH_FLAGS if f != "accumulator_result_visibility"])
def test_an_unknown_stream_premise_keeps_every_drain(flag):
    p, r, s, b, a, adapter, result = original(batch=True)
    checked = schedule_dense_panel_epochs(result, p, b, r, s, adapter, a, law(adapter, **{flag: False}))
    assert checked.kernel == result.kernel
    assert checked.realization["panel_epochs"]["removed_dynamic_drains"] == 0
    assert flag in checked.realization["unknown_premises"]


def test_unknown_result_visibility_keeps_the_drain_before_the_readout():
    p, r, s, b, a, adapter, result = original(batch=True)
    full = schedule_dense_panel_epochs(result, p, b, r, s, adapter, a, law(adapter))
    kept = schedule_dense_panel_epochs(
        result, p, b, r, s, adapter, a, law(adapter, accumulator_result_visibility=False)
    )
    streams = kept.realization["panel_epochs"]["streams"]
    assert streams and not any(stream["readout_merged"] for stream in streams)
    assert drains(full.kernel) < drains(kept.kernel) < drains(result.kernel)
    assert drains(kept.kernel) - drains(full.kernel) == sum(stream["instances"] for stream in streams)
    output, expected = ordered_execution(kept.kernel, adapter, p, s, visibility=False)
    np.testing.assert_array_equal(output, expected)


def test_issued_budgets_bound_merged_epochs():
    small = original(batch=False, budget=4)
    p, r, s, b, a, adapter, result = small
    checked = schedule_dense_panel_epochs(result, p, b, r, s, adapter, a, law(adapter))
    assert checked.kernel == result.kernel  # one K step already fills the four-pair budget
    medium = original(batch=False, budget=8)
    p, r, s, b, a, adapter, result = medium
    checked = schedule_dense_panel_epochs(result, p, b, r, s, adapter, a, law(adapter))
    streams = checked.realization["panel_epochs"]["streams"]
    assert streams and all(stream["compute_pairs_per_epoch"] <= 8 for stream in streams)
    assert drains(checked.kernel) < drains(result.kernel)
    output, expected = ordered_execution(checked.kernel, adapter, p, s)
    np.testing.assert_array_equal(output, expected)


def test_deferred_consumption_refuses_instead_of_dropping_an_unordered_rewrite():
    p, r, s, b, a, adapter, result = original(prefetch=False, issue_order="prefetch_rhs_between_columns")
    with pytest.raises(ValueError, match="not read by the next step"):
        schedule_dense_panel_epochs(result, p, b, r, s, adapter, a, law(adapter))


def test_checked_epochs_refuse_changed_programs_laws_and_contracts():
    p, r, s, b, a, adapter, result = original(batch=True)
    checked = schedule_dense_panel_epochs(result, p, b, r, s, adapter, a, law(adapter))
    with pytest.raises(ValueError, match="changed"):
        replace(checked, kernel=result.kernel).revalidate()
    with pytest.raises(ValueError, match="changed"):
        replace(checked, law=law(adapter, accumulator_result_visibility=False)).revalidate()
    with pytest.raises(ValueError, match="different hardware"):
        schedule_dense_panel_epochs(result, p, b, r, s, adapter, a, replace(law(adapter), hardware_identity="other"))
    with pytest.raises(ValueError, match="original overlap admission"):
        schedule_dense_panel_epochs(result, p, b, r, s, adapter, replace(a, max_issued_loads=65), law(adapter))
    with pytest.raises(ValueError, match="typed identities"):
        DensePanelEpochLaw("h", "n", "q", local_read_after_write="yes")
    with pytest.raises(ValueError, match="typed epoch law"):
        propose_dense_panel_epochs(result.kernel, adapter, None, a)


def test_one_option_on_the_dense_entry_emits_only_the_remaining_drains():
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp

    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
    from merlin.llvmlower.host_builder import FnBuilder

    p, r, s, b, a = setup()
    a = replace(
        a,
        max_issued_compute_pairs=64,
        max_issued_loads=64,
        max_issued_readouts=4,
        pending_readouts=1,
        readout_configuration="ordered_snapshot_before_store_issue",
    )
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        ADAPTER,
        source_identity="a" * 64,
        dtypes=DTYPES,
        static_budget=128,
        block_tiles=(b.m, b.k, b.n),
        pipeline_prefetch=True,
    )
    common = dict(
        name="entry",
        dtypes=DTYPES,
        source_identity="a" * 64,
        external_buffers_disjoint=True,
        schedule_plan=plan,
        pipeline_admission=a,
        batch_readouts=True,
    )
    default = schedule_dense_pipeline(p, r, s, ADAPTER, **common)
    again = schedule_dense_pipeline(p, r, s, ADAPTER, **common, epoch_law=None)
    assert type(default) is type(again) and default.kernel.digest() == again.kernel.digest()
    selected = schedule_dense_pipeline(p, r, s, ADAPTER, **common, epoch_law=law())
    assert type(selected) is DensePanelEpochResult and selected.original.kernel == default.kernel
    for result in (default, selected):
        builder, effects, completions = FnBuilder([]), [], []

        def issue(tag, values, builder=builder, effects=effects):
            effects.append(tag)
            builder.add(llvm.InlineAsmOp("", "~{memory}", [], has_side_effects=True))

        def complete(builder=builder, completions=completions):
            completions.append(True)
            builder.add(llvm.InlineAsmOp("fence", "~{memory}", [], has_side_effects=True))

        receipt = result.emit_index_program(FnBuilderIndexAdapter(builder), issue, complete_epoch=complete)
        ModuleOp([llvm.FuncOp("entry", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())]).verify()
        assert len(completions) == receipt["completion_mapping"]["emitted_static_drains"]
        assert {"preload", "compute", "lhs_load", "weight_load", "store"} <= set(effects)
    assert sum(1 for c, _ in instances(selected.kernel) if c.instr == DRAIN) < sum(
        1 for c, _ in instances(default.kernel) if c.instr == DRAIN
    )
    with pytest.raises(ValueError, match="two-window"):
        schedule_dense_pipeline(
            p,
            r,
            s,
            ADAPTER,
            name="compact",
            dtypes=DTYPES,
            source_identity="a" * 64,
            external_buffers_disjoint=True,
            epoch_law=law(),
        )


def test_proposal_moves_only_completions():
    p, r, s, b, a, adapter, result = original()
    actual, records = propose_dense_panel_epochs(result.kernel, adapter, law(adapter), a)
    assert _serial_projection(actual) == _serial_projection(result.kernel)
    assert records and sum(r_["removed_dynamic_drains"] for r_ in records) == drains(result.kernel) - drains(actual)
