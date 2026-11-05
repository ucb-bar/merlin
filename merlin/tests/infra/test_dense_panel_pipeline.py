"""Actual two-window issue with independent source and in-flight hazards."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture

from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN,
    DensePanelOverlapAdmission,
    build_dense_panel_pipeline,
    verify_dense_panel_pipeline,
)
from merlin.sched.ir import Loop
from merlin.transforms.dense_planning import dense_blocks


def setup():
    p, r, s = fixture()
    block = next(b for b in dense_blocks(p, replace(r, max_k_blocks=1)) if (b.m, b.n) == (2, 2))
    admission = DensePanelOverlapAdmission(
        ADAPTER.configuration_identity,
        ADAPTER.numerical_policy_identity,
        "neutral-independent-epoch-interpreter-v1",
        4,
        4,
        2,
        2,
    )
    return p, r, s, block, admission


@pytest.mark.parametrize("prefetch", [False, True])
def test_typed_interleaving_changes_real_issue_order_preserves_exact_data_calls(prefetch):
    from collections import Counter

    from xdsl.dialects.builtin import StringAttr

    from merlin.compile.scheduling.dense_panel_pipeline import schedule_dense_panel_pipeline
    from merlin.compile.scheduling.schedule_ir import (
        dense_schedule_text,
        make_dense_schedule_plan,
        parse_dense_schedule_plan,
    )
    from merlin.sched.ir import evaluate, instances

    p, r, s, b, a = setup()
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        ADAPTER,
        source_identity="a" * 64,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        block_tiles=(b.m, b.k, b.n),
        pipeline_prefetch=prefetch,
        pipeline_issue_order="prefetch_rhs_between_columns",
    )
    plan = parse_dense_schedule_plan(dense_schedule_text(plan))
    result = schedule_dense_panel_pipeline(
        p,
        r,
        s,
        ADAPTER,
        a,
        name="pipeline",
        dtypes=("i8", "i8", "i32"),
        schedule_plan=plan,
        source_identity="a" * 64,
        external_buffers_disjoint=True,
    )

    def data(kernel):
        return [
            (call.instr, tuple((k, evaluate(v, env)) for k, v in call.args))
            for call, env in instances(kernel)
            if call.instr != DRAIN
        ]

    previous = build_dense_panel_pipeline(
        p, b, r, s, ADAPTER, name="pipeline", dtypes=("i8", "i8", "i32"), prefetch=prefetch
    )
    control = build_dense_panel_pipeline(
        p,
        b,
        r,
        s,
        ADAPTER,
        name="pipeline",
        dtypes=("i8", "i8", "i32"),
        prefetch=not prefetch,
        issue_order="prefetch_rhs_between_columns",
    )
    assert data(previous) != data(result.kernel)
    assert Counter(data(previous)) == Counter(data(result.kernel))
    assert data(result.kernel) == data(control)
    assert result.realization["schedule_ir"]["pipeline_issue_order"] == "prefetch_rhs_between_columns"
    if prefetch:
        mixed = [
            e["issue_order"]
            for e in result.realization["epochs"]
            if e["compute_pairs"] and e["future_or_initial_loads"]
        ]
        assert mixed and all(order[0] == "Load" and order[-1] == "Load" and "Compute" in order for order in mixed)
    result.materialize_scf().verify()
    pipeline = list(plan.body.block.first_op.body.block.ops)[2]
    pipeline.properties["issue_order"] = StringAttr("unknown_backend_answer")
    with pytest.raises(ValueError, match="unsupported executable panel issue order"):
        schedule_dense_panel_pipeline(
            p,
            r,
            s,
            ADAPTER,
            a,
            name="pipeline",
            dtypes=("i8", "i8", "i32"),
            schedule_plan=plan,
            source_identity="a" * 64,
            external_buffers_disjoint=True,
        )


def test_symbolic_epoch_can_issue_more_than_sixty_four_commands_with_small_pending_queues():
    p, r, s = fixture(shape=(65, 3, 1), tile=(1, 1, 1))
    s = replace(
        s,
        lhs=replace(s.lhs, capacity=130),
        rhs=replace(s.rhs, capacity=2),
        accumulator=replace(s.accumulator, capacity=65),
    )
    r = replace(r, max_m_blocks=65, max_n_blocks=1, max_k_blocks=1, operand_bytes=132, accumulator_bytes=260)
    b = next(b for b in dense_blocks(p, r) if b.m == 65)
    a = DensePanelOverlapAdmission(
        ADAPTER.configuration_identity,
        ADAPTER.numerical_policy_identity,
        "neutral-independent-epoch-interpreter-v1",
        65,
        66,
        2,
        2,
    )
    kernel = build_dense_panel_pipeline(p, b, r, s, ADAPTER, name="wide_issue", dtypes=("i8", "i8", "i32"))
    report = verify_dense_panel_pipeline(kernel, p, b, r, s, ADAPTER, a, static_budget=128, timeout_ms=5000)
    assert max(e["compute_pairs"] for e in report["epochs"]) == 65
    assert max(e["future_or_initial_loads"] for e in report["epochs"]) == 66
    assert report["source"]["output_tiles"] == 65


def test_typed_decision_source_custody_and_actual_scf_llvm(tmp_path):
    from test_dense_source import source
    from test_selected_dense_contracts import inputs
    from xdsl.dialects import llvm
    from xdsl.dialects.builtin import ModuleOp, StringAttr, i64

    from merlin.compile.scheduling.dense_panel_pipeline import schedule_source_dense_panel
    from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
    from merlin.compile.scheduling.scf_commands import emit_scf_commands
    from merlin.compile.scheduling.schedule_ir import (
        dense_schedule_text,
        make_dense_schedule_plan,
        parse_dense_schedule_plan,
    )
    from merlin.llvmlower.host_builder import PTR, FnBuilder
    from merlin.transforms.certificates import ir_digest

    selected, p, s, adapter = inputs(tmp_path)
    module, operation = source((p.m, p.k, p.n), selected.dense.dtypes)
    r, adapter = selected.dense.bind(
        p,
        s,
        adapter,
        dtypes=selected.dense.dtypes,
        hardware=selected.hardware_identity,
        policy=selected.numerical_policy,
    )
    admission = DensePanelOverlapAdmission(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        "neutral-independent-epoch-interpreter-v1",
        4,
        4,
        2,
        2,
    )
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        adapter,
        source_identity=ir_digest(module),
        dtypes=selected.dense.dtypes,
        static_budget=128,
        block_tiles=(2, 1, 2),
        pipeline_prefetch=True,
    )
    plan = parse_dense_schedule_plan(dense_schedule_text(plan))
    result = schedule_source_dense_panel(
        module,
        operation,
        selected_contracts=selected,
        storage=s,
        adapter=adapter,
        admission=admission,
        name="pipeline",
        schedule_plan=plan,
        external_buffers_disjoint=True,
    )
    assert result.realization["overlapped_compute_pairs"] > 0
    scf_module = result.materialize_scf()
    assert any(DRAIN in str(value) for op in scf_module.walk() for value in op.attributes.values())
    declarations = {}
    observed = []

    def lower(builder, instruction, operands):
        observed.append(instruction)
        declarations[instruction] = len(operands)
        builder.add(llvm.CallOp(instruction, *operands.values()))

    builder = FnBuilder([PTR] * 3)
    bridge = FnBuilderIndexAdapter(builder)
    emit_scf_commands(
        scf_module,
        bridge,
        lambda instruction, operands, metadata: lower(
            builder, instruction, {key: bridge.value(value) for key, value in operands.items()}
        ),
    )
    function = llvm.FuncOp("compiled", llvm.LLVMFunctionType([PTR] * 3, llvm.LLVMVoidType()), body=builder.finish())
    output = ModuleOp(
        [
            *(
                llvm.FuncOp(name, llvm.LLVMFunctionType([i64] * count, llvm.LLVMVoidType()))
                for name, count in declarations.items()
            ),
            function,
        ]
    )
    output.verify()
    assert DRAIN in observed and "compute" in observed and "lhs_load" in observed
    assert result.realization["schedule_ir"]["source_identity"] == ir_digest(module)
    module.attributes["changed"] = StringAttr("different original source")
    with pytest.raises(ValueError, match="source"):
        result.revalidate()


def test_old_synchronous_service_cannot_ignore_pipeline_decision():
    from merlin.compile.scheduling.compact_dense import schedule_compact_dense
    from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan

    p, r, s, b, a = setup()
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        ADAPTER,
        source_identity="a" * 64,
        dtypes=("i8", "i8", "i32"),
        static_budget=128,
        block_tiles=(b.m, b.k, b.n),
        pipeline_prefetch=True,
    )
    with pytest.raises(ValueError, match="panel overlap requires"):
        schedule_compact_dense(
            p,
            r,
            s,
            ADAPTER,
            name="pipeline",
            dtypes=("i8", "i8", "i32"),
            external_buffers_disjoint=True,
            schedule_plan=plan,
            source_identity="a" * 64,
            static_budget=128,
        )


@pytest.mark.parametrize("source_change", [None, "scalar", "observer"])
def test_original_typed_readout_preserves_scalar_and_all_external_observers(tmp_path, source_change):
    from test_dense_integer_readout import QUALIFICATION, readout_inputs, readout_source

    from merlin.compile.scheduling.dense_panel_pipeline import schedule_source_dense_panel
    from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
    from merlin.transforms.certificates import ir_digest

    selected, p, s, adapter = readout_inputs(tmp_path)
    module, op, output = readout_source(
        (p.m, p.k, p.n), selected.dense.dtypes, altered=source_change == "scalar", extra_use=source_change == "observer"
    )
    r, adapter = selected.dense.bind(
        p,
        s,
        adapter,
        dtypes=selected.dense.dtypes,
        hardware=selected.hardware_identity,
        policy=selected.numerical_policy,
        output_readout=selected.dense.output_readout,
    )
    a = DensePanelOverlapAdmission(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        "neutral-independent-epoch-interpreter-v1",
        4,
        4,
        2,
        2,
    )
    plan = make_dense_schedule_plan(
        p,
        r,
        s,
        adapter,
        source_identity=ir_digest(module),
        dtypes=selected.dense.dtypes,
        static_budget=128,
        block_tiles=(2, 1, 2),
        pipeline_prefetch=True,
        output_readout=selected.dense.output_readout,
    )

    def run():
        return schedule_source_dense_panel(
            module,
            op,
            selected_contracts=selected,
            storage=s,
            adapter=adapter,
            admission=a,
            name="narrow_pipeline",
            schedule_plan=plan,
            external_buffers_disjoint=True,
            output_operation=output,
            readout_qualification_identity=QUALIFICATION,
        )

    if source_change is not None:
        with pytest.raises(ValueError):
            run()
    else:
        result = run()
        assert result.kernel.args[-1].dtype == "i8"
        assert result.realization["source"]["requested_store_bytes"] == p.m * p.n
        assert all(
            not e["compute_pairs"] and not e["future_or_initial_loads"]
            for e in result.realization["epochs"]
            if e["readout"]
        )
        result.materialize_scf().verify()
        object.__setattr__(result, "readout_qualification_identity", "changed")
        with pytest.raises(ValueError, match="changed"):
            result.revalidate()


@pytest.mark.parametrize("prefetch", [False, True])
def test_complete_source_tail_cover_and_real_overlap_epochs(prefetch):
    p, r, s, b, a = setup()
    kernel = build_dense_panel_pipeline(
        p, b, r, s, ADAPTER, name="pipeline", dtypes=("i8", "i8", "i32"), prefetch=prefetch
    )
    report = verify_dense_panel_pipeline(kernel, p, b, r, s, ADAPTER, a, static_budget=128, timeout_ms=5000)
    assert report["source"]["output_tiles"] == p.tiles[0] * p.tiles[2]
    assert any(e["compute_pairs"] and e["future_or_initial_loads"] for e in report["epochs"]) is prefetch


def map_calls(body, rewrite):
    return tuple(replace(s, body=map_calls(s.body, rewrite)) if type(s) is Loop else rewrite(s) for s in body)


def test_real_epoch_guard_rejects_single_window_alias_even_when_serial_source_projection_passes():
    from merlin.compile.scheduling.compact_verification import verify_compact_dense
    from merlin.compile.scheduling.dense_panel_pipeline import _serial_projection
    from merlin.sched.ir import evaluate
    from merlin.sched.ir.expr import free_vars

    p, r, s, b, a = setup()
    kernel = build_dense_panel_pipeline(p, b, r, s, ADAPTER, name="pipeline", dtypes=("i8", "i8", "i32"))

    def collapse(call):
        role = {
            "lhs_load": ("dst", s.lhs, b.m * p.tile_m),
            "weight_load": ("dst", s.rhs, b.n * p.tile_k),
            "preload": ("weight", s.rhs, b.n * p.tile_k),
            "compute": ("lhs", s.lhs, b.m * p.tile_m),
        }.get(call.instr)
        if role is None:
            return call
        key, view, stride = role
        value = dict(call.args)[key]
        if evaluate(value, dict.fromkeys(free_vars(value), 0)) >= view.base + stride:
            return replace(call, args=tuple((k, v + (-stride) if k == key else v) for k, v in call.args))
        return call

    aliased = replace(kernel, body=map_calls(kernel.body, collapse))
    # In nominal sequential issue order each compute reads before the next load;
    # source cover alone cannot prove safe simultaneous in-flight execution.
    verify_compact_dense(_serial_projection(aliased), p, r, s, ADAPTER, static_budget=128, timeout_ms=5000)
    with pytest.raises(ValueError, match="overlapping in-flight"):
        verify_dense_panel_pipeline(aliased, p, b, r, s, ADAPTER, a, static_budget=128, timeout_ms=5000)


def test_missing_completion_and_original_issue_capacity_refuse():
    p, r, s, b, a = setup()
    kernel = build_dense_panel_pipeline(p, b, r, s, ADAPTER, name="pipeline", dtypes=("i8", "i8", "i32"))

    def missing(body):
        result = []
        for item in body:
            if type(item) is Loop:
                result.append(replace(item, body=missing(item.body)))
            elif item.instr != DRAIN:
                result.append(item)
        return tuple(result)

    with pytest.raises(ValueError, match="drained panel epoch|completion epochs"):
        verify_dense_panel_pipeline(
            replace(kernel, body=missing(kernel.body)), p, b, r, s, ADAPTER, a, static_budget=128, timeout_ms=5000
        )
    with pytest.raises(ValueError, match="in-flight command bounds"):
        verify_dense_panel_pipeline(
            kernel, p, b, r, s, ADAPTER, replace(a, max_issued_compute_pairs=3), static_budget=128, timeout_ms=5000
        )
    with pytest.raises(ValueError, match="two complete panel windows"):
        build_dense_panel_pipeline(
            p, b, r, replace(s, lhs=replace(s.lhs, capacity=4)), ADAPTER, name="pipeline", dtypes=("i8", "i8", "i32")
        )


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("biased,narrow", [(False, False), (True, False), (True, True), (False, "tiled")])
@pytest.mark.parametrize("issue_order", ["compute_then_prefetch", "prefetch_rhs_between_columns"])
def test_reversed_completion_with_finite_queue_backpressure_matches_every_output(prefetch, biased, narrow, issue_order, batch_readouts=False):
    import numpy as np

    from merlin.compile.scheduling.block_schedule import Compute, InitializeAccumulator, Load, Preload, Store
    from merlin.compile.scheduling.compact_dense import DenseBiasSource
    from merlin.compile.scheduling.dense_contracts import (
        DenseIntegerReadout,
        DenseTiledIntegerReadout,
        TiledIntegerStore,
    )
    from merlin.sched.ir import evaluate, instances

    p, r, s, b, a = setup()
    adapter = ADAPTER
    bias_source = None
    readout = None
    if biased:
        from test_compact_dense_bias import ADAPTER as bias_adapter

        p = replace(p, bias=True)
        bias_source = DenseBiasSource(p.n, "i32")
        adapter = bias_adapter
        b = next(block for block in dense_blocks(p, replace(r, max_k_blocks=1)) if (block.m, block.n) == (2, 2))
        a = replace(
            a,
            hardware_identity=adapter.configuration_identity,
            numerical_policy_identity=adapter.numerical_policy_identity,
        )
    if narrow:
        from test_dense_integer_readout import SCALAR_IR

        scales = [0.5] * 4 + [0.9900199770927429] * 4 + [1.5] * 4 + [2.0]
        readout = (
            DenseTiledIntegerReadout.from_column_scales(scales, tile_columns=p.tile_n)
            if narrow == "tiled"
            else DenseIntegerReadout(SCALAR_IR, "i8", "store_i8", ())
        )
        original = adapter

        def encode_narrow(op):
            if type(op) is TiledIntegerStore:
                base = original.encode(Store(op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row))
                return replace(
                    base, instr=readout.instruction_name, args=(*base.args, ("readout_tile", op.readout_tile))
                )
            return (
                replace(original.encode(replace(op, readout=None)), instr="store_i8")
                if type(op) is Store
                else original.encode(op)
            )

        def project_narrow(call):
            if narrow == "tiled" and call.instr == readout.instruction_name:
                op = original.project(
                    replace(call, instr="store", args=tuple((k, v) for k, v in call.args if k != "readout_tile"))
                )
                return TiledIntegerStore(
                    op.dram_row,
                    op.dram_col,
                    op.rows,
                    op.cols,
                    op.accumulator_row,
                    readout.identity,
                    call.arg("readout_tile"),
                )
            return (
                replace(original.project(replace(call, instr="store")), readout=readout.identity)
                if call.instr == "store_i8"
                else original.project(call)
            )

        adapter = replace(adapter, encode=encode_narrow, project=project_narrow)
    kernel = build_dense_panel_pipeline(
        p,
        b,
        r,
        s,
        adapter,
        name="pipeline",
        dtypes=("i8", "i8", "i32"),
        prefetch=prefetch,
        issue_order=issue_order,
        output_readout=readout,
        bias_source=bias_source,
    )
    if batch_readouts:
        from merlin.compile.scheduling.dense_panel_readout import _propose

        kernel, groups = _propose(kernel, adapter, 4)
        assert groups
        a = replace(a, max_issued_readouts=4, pending_readouts=1,
                    readout_configuration="ordered_snapshot_before_store_issue")
    verify_dense_panel_pipeline(
        kernel,
        p,
        b,
        r,
        s,
        adapter,
        a,
        static_budget=128,
        timeout_ms=5000,
        output_readout=readout,
        bias_source=bias_source,
    )
    lhs = (np.arange(p.m * p.k) % 7 - 3).reshape(p.m, p.k)
    rhs = (np.arange(p.k * p.n) % 5 - 2).reshape(p.k, p.n)
    bias_values = np.resize(np.array([2**31 - 1, -(2**31), 127, -128, 0], dtype=np.int64), p.n)
    output = np.full((p.m, p.n), -10000, dtype=np.int64)
    memory = {s.lhs.name: {}, s.rhs.name: {}, s.accumulator.name: {}}
    pending = []
    preload = None
    macs = 0
    peak = {"load": 0, "compute": 0, "store": 0, "initialize": 0}

    def complete(item):
        nonlocal macs
        kind, operation, pre = item
        if kind == "load":
            view = s.lhs if operation.role == "lhs" else s.rhs
            source = lhs if operation.role == "lhs" else rhs
            for i in range(operation.rows):
                for j in range(operation.cols):
                    memory[view.name][operation.row + i, j] = int(
                        source[operation.dram_row + i, operation.dram_col + j]
                    )
        elif kind == "initialize":
            for i in range(operation.rows):
                for j in range(operation.cols):
                    memory[s.accumulator.name][operation.row + i, j] = int(bias_values[operation.dram_col + j])
        elif kind == "compute":
            for i in range(operation.rows):
                for j in range(pre.cols):
                    key = pre.accumulator_row + i, j
                    value = memory[s.accumulator.name][key] if pre.accumulate else 0
                    for k in range(operation.cols):
                        value += (
                            memory[s.lhs.name][operation.input_row + i, k] * memory[s.rhs.name][pre.weight_row + k, j]
                        )
                        macs += 1
                    memory[s.accumulator.name][key] = (value + 2**31) % 2**32 - 2**31
        else:
            for i in range(operation.rows):
                for j in range(operation.cols):
                    value = memory[s.accumulator.name][operation.accumulator_row + i, j]
                    if narrow == "tiled":
                        value = int(
                            np.clip(
                                np.rint(
                                    np.float32(
                                        np.float32(value) * np.float32(scales[operation.readout_tile * p.tile_n])
                                    )
                                ),
                                -128,
                                127,
                            )
                        )
                    elif narrow:
                        value = min(127, max(0, value))
                    output[operation.dram_row + i, operation.dram_col + j] = value

    def issue(kind, operation, pre=None):
        bound = {"load": a.pending_loads, "compute": a.pending_compute_pairs, "store": 1, "initialize": 1}[kind]
        same = [index for index, item in enumerate(pending) if item[0] == kind]
        if len(same) >= bound:
            complete(pending.pop(same[0]))
        pending.append((kind, operation, pre))
        peak[kind] = max(peak[kind], sum(item[0] == kind for item in pending))

    for call, env in instances(kernel):
        if call.instr == DRAIN:
            # Future DMA can complete before the current compute has consumed
            # its operands. Separate private windows must make this safe.
            for item in reversed(pending):
                complete(item)
            pending.clear()
            continue
        operation = adapter.project(replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args)))
        if type(operation) is Load:
            issue("load", operation)
        elif type(operation) is Preload:
            preload = operation
        elif type(operation) is Compute:
            assert preload is not None
            issue("compute", operation, preload)
            preload = None
        elif type(operation) is InitializeAccumulator:
            issue("initialize", operation)
        else:
            assert isinstance(operation, Store)
            issue("store", operation)
    assert not pending and preload is None
    assert macs == p.m * p.k * p.n
    assert peak["compute"] <= a.pending_compute_pairs and peak["load"] <= a.pending_loads
    expected = (lhs @ rhs + (bias_values if biased else 0)).astype(np.int32).astype(np.int64)
    if narrow == "tiled":
        expected = np.clip(np.rint(expected.astype(np.float32) * np.array(scales, dtype=np.float32)), -128, 127)
    elif narrow:
        expected = np.clip(expected, 0, 127)
    np.testing.assert_array_equal(output, expected)
