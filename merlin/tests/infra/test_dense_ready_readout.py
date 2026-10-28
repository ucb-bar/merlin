"""Causal final-K readout uses actual counted domains and separate target law."""

import json
from dataclasses import replace
from math import prod

import numpy as np
import pytest
from test_compact_dense import ADAPTER
from test_dense_readout_overlap import selected
from xdsl.dialects import llvm, scf
from xdsl.dialects.builtin import ModuleOp, i64

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN,
    DenseReadoutConfigurationSnapshot,
    DenseReadyReadoutAdmission,
    _verify_dense_epochs,
)
from merlin.compile.scheduling.dense_ready_readout import (
    DenseReadyReadoutCheckingProjection,
    propose_dense_ready_readout,
    schedule_dense_ready_readout,
)
from merlin.llvmlower.host_builder import FnBuilder
from merlin.sched.ir import Call, Const, Loop, evaluate, instances


def setup(shape=(3, 4, 25)):
    p, r, s, b, original, disjoint = selected(shape)
    causal = DenseReadyReadoutAdmission(
        disjoint.hardware_identity, disjoint.numerical_policy_identity,
        "TEST-ONLY-RS-producer-retirement-and-ACC-pending-write-readiness",
        disjoint.store_qualification_identity, disjoint.max_issued_readouts, disjoint.pending_readouts,
    )
    return p, r, s, b, original, causal


def retained_setup(shape=(9, 14, 25)):
    from test_dense_retained_reduction import chosen, inputs

    from merlin.compile.scheduling.dense_panel_readout import batch_dense_panel_readouts

    p, r, s, b, admission = inputs(shape)
    original = chosen(p, r, s, b, admission, role="weight", order=("n", "m"), compute_order=("k", "n", "m"))
    admission = replace(admission, max_issued_readouts=b.m * b.n, pending_readouts=2,
                        readout_configuration="ordered_snapshot_before_store_issue")
    original = batch_dense_panel_readouts(original, p, b, r, s, ADAPTER, admission)
    law = DenseReadyReadoutAdmission(admission.hardware_identity, admission.numerical_policy_identity,
                                    "TEST-ONLY-original-ACC-producer-RAW-retirement",
                                    "TEST-ONLY-original-store-snapshot", b.m * b.n, 2)
    return p, r, s, b, original, law


def check(actual, p, r, s, original, causal, **kwargs):
    return _verify_dense_epochs(
        actual, p, r, s, ADAPTER, original.admission.overlap, static_budget=128, timeout_ms=5000,
        ordered_accumulator_commits=True, source_phase_fission=True,
        readout_compute_admission=causal, **kwargs,
    )


def execute(kernel, p):
    """Independent scalar products/readouts from every actual projected operand."""
    lhs = (np.arange(p.m * p.k) % 7 - 3).reshape(p.m, p.k)
    rhs = (np.arange(p.k * p.n) % 5 - 2).reshape(p.k, p.n)
    out = np.full((p.m, p.n), -(1 << 30), dtype=np.int64)
    memory = {name: {} for name in ("lhs", "weight", "acc")}
    descriptor = None
    counts = dict(products=0, stores=0)
    for call, env in instances(kernel):
        if call == Call(DRAIN, ()):
            continue
        op = ADAPTER.project(replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args)))
        if type(op) is Load:
            tensor = lhs if op.role == "lhs" else rhs
            for i in range(op.rows):
                for j in range(op.cols):
                    memory[op.role][op.row + i, j] = int(tensor[op.dram_row + i, op.dram_col + j])
        elif type(op) is Preload:
            descriptor = op
        elif type(op) is Compute:
            for i in range(op.rows):
                for j in range(descriptor.cols):
                    address = descriptor.accumulator_row + i, j
                    value = memory["acc"][address] if descriptor.accumulate else 0
                    for k in range(op.cols):
                        value += memory["lhs"][op.input_row + i, k] * memory["weight"][descriptor.weight_row + k, j]
                        counts["products"] += 1
                    memory["acc"][address] = value
        else:
            assert type(op) is Store
            for i in range(op.rows):
                for j in range(op.cols):
                    out[op.dram_row + i, op.dram_col + j] = memory["acc"][op.accumulator_row + i, j]
                    counts["stores"] += 1
    np.testing.assert_array_equal(out, lhs @ rhs)
    assert counts == dict(products=p.m * p.k * p.n, stores=p.m * p.n)
    return out


@pytest.mark.parametrize("shape", [(3, 4, 25), (8, 6, 16), (4, 7, 8)])
def test_actual_nested_frontier_preserves_each_original_scalar_and_full_acc_geometry(shape):
    p, r, s, b, original, law = setup(shape)
    result = schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law)
    np.testing.assert_array_equal(execute(result.kernel, p), execute(original.kernel, p))
    assert result.realization["dynamic_epoch_drains"] < original.realization["dynamic_epoch_drains"]
    assert any(e["causal_ready_readout_compute"] for e in result.realization["epochs"])
    assert all(not e["disjoint_readout_compute"] for e in result.realization["epochs"])
    for call, env in instances(result.kernel):
        if call.instr in ("preload", "store"):
            assert s.accumulator.base <= evaluate(call.arg("acc"), env) < s.accumulator.base + s.accumulator.capacity
    assert all(region["loop_transforms"] for region in result.realization["transformations"])
    result.revalidate()


def test_actual_scf_and_llvm_consume_frontier_not_conservative_source_projection():
    p, r, s, b, original, law = setup()
    result = schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law)
    module = result.materialize_scf()
    # A store follows an actual counted K loop in the SAME N block.
    assert any(type(op) is scf.ForOp and any(
        (attr := child.attributes.get("merlin.scf_commands.v1")) is not None
        and json.loads(attr.data)["instruction"] == "store" for child in op.body.block.ops
    ) and any(type(child) is scf.ForOp for child in op.body.block.ops) for op in module.walk())
    builder, declarations = FnBuilder([]), {}
    def lower(instruction, operands, metadata):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter

        bridge = FnBuilderIndexAdapter(builder)
        declarations[instruction] = len(operands)
        builder.add(llvm.CallOp(instruction, *(bridge.value(v) for v in operands.values())))
    result.emit_llvm(builder, lower)
    function = llvm.FuncOp("frontier", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())
    actual = ModuleOp([*(llvm.FuncOp(n, llvm.LLVMFunctionType([i64] * count, llvm.LLVMVoidType()))
                         for n, count in declarations.items()), function])
    actual.verify()
    assert {"preload", "compute", "store", DRAIN} <= declarations.keys()
    assert result.kernel.digest() != original.kernel.digest()
    projection = result.realization["readout_checking_projection"]
    assert projection["original_kernel_sha256"] == original.kernel.digest()
    assert projection["actual_kernel_sha256"] == result.kernel.digest()


def rewrite_frontier(kernel, action):
    changed = False
    def walk(body):
        nonlocal changed
        result = []
        for s in body:
            if type(s) is Loop:
                if (not changed and len(s.body) == 2 and type(s.body[0]) is Loop
                        and type(s.body[1]) is Call and s.body[1].instr == "store"):
                    s = replace(s, body=action(s.body))
                    changed = True
                else:
                    s = replace(s, body=walk(s.body))
            result.append(s)
        return tuple(result)
    actual = replace(kernel, body=walk(kernel.body))
    assert changed
    return actual


@pytest.mark.parametrize("action", [lambda body: body[::-1], lambda body: (*body, body[0])])
def test_early_readout_or_later_accumulator_reuse_refuses_before_source_projection(action):
    p, r, s, b, original, law = setup()
    actual, _ = propose_dense_ready_readout(original.kernel, p, ADAPTER)
    invalid = rewrite_frontier(actual, action)
    with pytest.raises(ValueError, match="producer must precede readout"):
        check(invalid, p, r, s, original, law,
              readout_source_projection=DenseReadyReadoutCheckingProjection(original.kernel, p))


def test_old_capability_alias_and_unknown_effects_do_not_acquire_causal_permission():
    p, r, s, b, original, law = setup()
    actual, _ = propose_dense_ready_readout(original.kernel, p, ADAPTER)
    with pytest.raises(ValueError, match="separate completion epoch"):
        check(actual, p, r, s, original, None)
    *_, disjoint = selected()
    with pytest.raises(ValueError, match="overlapping in-flight"):
        check(actual, p, r, s, original, disjoint)
    with pytest.raises(ValueError, match="snapshot and completion"):
        schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, replace(law, pending_readouts=3))
    with pytest.raises(ValueError, match="producer completion"):
        schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, replace(law, execution="ordered_issue_only"))
    # Changing local ACC mapping cannot use the deterministic inverse.
    invalid = rewrite_frontier(actual, lambda body: (body[0], replace(
        body[1], args=tuple((k, Const(s.accumulator.base) if k == "acc" else v) for k, v in body[1].args))))
    with pytest.raises(ValueError, match="producer must precede readout|exact original transformation"):
        check(invalid, p, r, s, original, law,
              readout_source_projection=DenseReadyReadoutCheckingProjection(original.kernel, p))
    invalid = rewrite_frontier(actual, lambda body: (*body, Call("unknown", ())))
    with pytest.raises(ValueError, match="unknown instruction"):
        check(invalid, p, r, s, original, law)


def test_strict_original_transform_transport_and_owner_revalidation():
    p, r, s, b, original, law = setup()
    result = schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law)
    inverse = DenseReadyReadoutCheckingProjection(original.kernel, p)
    invalid = rewrite_frontier(result.kernel, lambda body: (body[0], replace(
        body[1], args=tuple((k, v + 1 if k == "col" else v) for k, v in body[1].args))))
    with pytest.raises(ValueError, match="exact original transformation"):
        inverse.checking_kernel(invalid, ADAPTER, s)
    with pytest.raises(ValueError, match="original source, actual commands"):
        replace(result, kernel=invalid).revalidate()
    with pytest.raises(ValueError, match="original source, actual commands"):
        replace(result, admission=replace(law, qualification_identity="different")).revalidate()
    # Removing a result or reduction from the proposal never reconstructs source.
    assert prod(p.tiles) == sum(c.instr == "compute" for c, _ in instances(result.kernel))


@pytest.mark.parametrize("shape", [(21, 14, 25), (24, 12, 24), (17, 13, 17)])
def test_advancing_inputs_use_two_lhs_slots_one_rhs_and_original_accumulator(shape):
    p, r, s, b, original, law = retained_setup(shape)
    separate = schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law)
    law = replace(law, disjoint_load_qualification_identity="TEST-ONLY-disjoint-load-compute-ready-store")
    result = schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law,
        input_pipeline_row_offset=b.m * b.k * p.tile_m, static_budget=256)
    execute(result.kernel, p)
    assert result.realization["dynamic_epoch_drains"] < separate.realization["dynamic_epoch_drains"]
    assert all(row["rhs_slots"] == 1 and row["live_lhs_generations"] == 2
               for row in result.realization["input_pipeline"])
    def operands(kernel):
        return [ADAPTER.project(replace(c, args=tuple((k, evaluate(v, env)) for k, v in c.args)))
                for c, env in instances(kernel) if c != Call(DRAIN, ())]
    before, after = operands(separate.kernel), operands(result.kernel)
    assert [o for o in before if type(o) is Load and o.role == "weight"] == [
        o for o in after if type(o) is Load and o.role == "weight"]
    assert [o.accumulator_row for o in before if type(o) is Preload] == [
        o.accumulator_row for o in after if type(o) is Preload]
    assert sorted((o.dram_row, o.dram_col, o.rows, o.cols) for o in before if type(o) is Load) == sorted(
        (o.dram_row, o.dram_col, o.rows, o.cols) for o in after if type(o) is Load)
    result.revalidate()
    result.materialize_scf().verify()


def test_input_pipeline_refuses_missing_conjunction_alias_and_escaped_slot():
    p, r, s, b, original, law = retained_setup((21, 14, 25))
    offset = b.m * b.k * p.tile_m
    with pytest.raises(ValueError, match="separately selected"):
        schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law,
                                     input_pipeline_row_offset=offset)
    law = replace(law, disjoint_load_qualification_identity="TEST-ONLY-load-ready-conjunction")
    with pytest.raises(ValueError, match="overlapping in-flight"):
        schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law,
                                     input_pipeline_row_offset=1, static_budget=256)
    with pytest.raises(ValueError, match="assigned private view"):
        schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law,
                                     input_pipeline_row_offset=s.lhs.capacity, static_budget=256)


def test_single_pipeline_entry_consumes_input_slot_and_rejects_unused_routes():
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan

    p, r, s, b, original, law = retained_setup((21, 14, 25))
    law = replace(law, disjoint_load_qualification_identity="TEST-ONLY-load-ready-conjunction")
    options = dict(name="retained", dtypes=("i8", "i8", "i32"),
        source_identity=original.source_identity, external_buffers_disjoint=True, static_budget=128,
        schedule_plan=parse_dense_schedule_plan(original.realization["schedule_ir"]["text"]),
        pipeline_admission=original.admission, input_pipeline_row_offset=b.m * b.k * p.tile_m)
    result = schedule_dense_pipeline(p, r, s, ADAPTER, readout_mode="ready", readout_admission=law, **options)
    execute(result.kernel, p)
    assert result.input_pipeline_row_offset == options["input_pipeline_row_offset"]
    with pytest.raises(ValueError, match="explicit retained full-K ready"):
        schedule_dense_pipeline(p, r, s, ADAPTER, **options)
    with pytest.raises(ValueError, match="explicit retained full-K ready"):
        schedule_dense_pipeline(p, r, s, ADAPTER, **{**options, "schedule_plan": None})


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_final_wide_bands_keep_causal_order_original_source_and_strict_owner(retained, external):
    from merlin.compile.scheduling.dense_ready_readout import admit_ready_readout_tile_loads
    from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
    from merlin.compile.scheduling.tile_load_coalescing import (
        TileLoadAddressLaw,
        TileLoadMapping,
        _linear,
        propose_tile_load_coalescing,
    )

    p, r, s, b, original, law = retained_setup((4, 6, 24)) if retained else setup((4, 6, 24))
    ready = schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law)
    from merlin.compile.scheduling.panel_transfers import AffineMatrixSource

    external_sources = ({"weight": AffineMatrixSource(
        "weight", (p.k, p.n), 1, 5, 3 * p.n, 1, 0, 5 + (p.k - 1) * 3 * p.n + p.n,
    )} if external else None)
    laws = {role: TileLoadAddressLaw(tile, rows, 4 * tile, 1, pitch, stride, "TEST-ONLY-exact-band-addressing")
            for role, tile, rows, pitch, stride in (
                ("lhs", p.tile_k, p.tile_m, p.k, p.tile_m),
                ("weight", p.tile_n, p.tile_k, 3 * p.n if external else p.n, p.tile_k))}
    def project(call):
        if call.instr not in ("lhs_load", "weight_load"):
            return None
        op = ADAPTER.project(call)
        view = s.lhs if op.role == "lhs" else s.rhs
        law = laws[op.role]
        offset = 5 if external and op.role == "weight" else 0
        return TileLoadMapping(op.role, view.backing, offset + op.dram_row * law.source_pitch_bytes + op.dram_col,
                               op.row, op.rows, op.cols, law)
    def encode(call, columns):
        return replace(call, args=tuple((k, Const(columns) if k == "cols" else v) for k, v in call.args))
    def band(call, mapping):
        offset, terms = _linear(mapping.source_byte)
        base, oldterms = _linear(project(call).source_byte)
        assert terms == oldterms
        columns = offset - base
        return replace(call, args=tuple((k, mapping.local_row if k == "dst" else mapping.columns if k == "cols"
                                        else v + columns if k == "src_col"
                                        else v + columns // mapping.law.tile_columns if k == "b" else v)
                                       for k, v in call.args))
    proposal, _ = propose_tile_load_coalescing(ready.kernel, project=project, encode=encode, max_tiles=4)
    if external:
        with pytest.raises(ValueError, match="source layout"):
            admit_ready_readout_tile_loads(ready, proposal, p, b, r, s, ADAPTER, project=project, encode_band=band)
    result = admit_ready_readout_tile_loads(ready, proposal, p, b, r, s, ADAPTER,
                                          project=project, encode_band=band, external_sources=external_sources)
    assert result.realization["tile_load_projection"]["expanded_logical_tile_bands"] > (
        result.realization["tile_load_projection"]["physical_load_commands"])
    assert result.realization["dynamic_epoch_drains"] == ready.realization["dynamic_epoch_drains"]
    assert result.realization["source"]["output_tiles"] == p.tiles[0] * p.tiles[2]
    assert any(e["causal_ready_readout_compute"] for e in result.realization["epochs"])
    result.materialize_scf().verify()
    result.revalidate()
    with pytest.raises(ValueError, match="original checked panel"):
        admit_panel_tile_loads(ready, proposal, p, b, r, s, ADAPTER, project=project, encode_band=band)
    def wrong_band(call, mapping):
        if _linear(mapping.source_byte)[0] > _linear(project(call).source_byte)[0]:
            mapping = replace(mapping, local_row=mapping.local_row + 1)
        return band(call, mapping)
    with pytest.raises(ValueError):
        admit_ready_readout_tile_loads(ready, proposal, p, b, r, s, ADAPTER, project=project, encode_band=wrong_band)
    def wrong_pitch(call):
        m = project(call)
        return None if m is None else replace(m, law=replace(m.law, source_pitch_bytes=m.law.source_pitch_bytes + 1))
    with pytest.raises(ValueError, match="source layout"):
        admit_ready_readout_tile_loads(ready, proposal, p, b, r, s, ADAPTER, project=wrong_pitch, encode_band=band)


@pytest.mark.parametrize('shape', [(9, 14, 25), (8, 12, 24)])
def test_retained_full_k_frontier_preserves_every_product_and_minimal_rhs_loads(shape):
    p, r, s, b, original, law = retained_setup(shape)
    actual = schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law)
    np.testing.assert_array_equal(execute(actual.kernel, p), execute(original.kernel, p))
    assert actual.realization['dynamic_epoch_drains'] < original.realization['dynamic_epoch_drains']
    for role in ('lhs_loads', 'rhs_loads', 'output_tiles'):
        assert actual.realization['source'][role] == original.realization['source'][role]
    assert any(e['causal_ready_readout_compute'] for e in actual.realization['epochs'])
    assert actual.original is original
    actual.materialize_scf().verify()
    with pytest.raises(ValueError, match='full-K'):
        schedule_dense_ready_readout(original, p, replace(b, k=b.k-1), r, s, ADAPTER, law)


@pytest.mark.parametrize('uniform', [False, True])
@pytest.mark.parametrize('snapshot', [False, True])
def test_retained_tiled_readout_requires_original_coefficients_and_selected_snapshot(uniform, snapshot):
    from test_dense_retained_reduction import inputs

    from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout, TiledIntegerStore
    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan

    p, r, s, b, admission = inputs()
    coefficients = [0.5] * p.n
    if not uniform:
        coefficients[-1] = 0.25
    readout = DenseTiledIntegerReadout.from_column_scales(coefficients, tile_columns=p.tile_n)
    def encode(op):
        if type(op) is TiledIntegerStore:
            raw = ADAPTER.encode(Store(op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row))
            return replace(raw, instr=readout.instruction_name, args=(*raw.args, ('readout_tile', op.readout_tile)))
        return ADAPTER.encode(op)
    def project(call):
        if call.instr == readout.instruction_name:
            raw = ADAPTER.project(replace(call, instr='store',
                args=tuple((k, v) for k, v in call.args if k != 'readout_tile')))
            return TiledIntegerStore(raw.dram_row, raw.dram_col, raw.rows, raw.cols, raw.accumulator_row,
                                     readout.identity, call.arg('readout_tile'))
        return ADAPTER.project(call)
    adapter = replace(ADAPTER, encode=encode, project=project)
    plan = make_dense_schedule_plan(p, r, s, adapter, source_identity='a'*64, name='uniform_ready',
        dtypes=('i8', 'i8', 'i32'), static_budget=128, block_tiles=(b.m, b.k, b.n),
        output_readout=readout, pipeline_kind='retained_full_k', pipeline_prefetch=False)
    admission = replace(admission, max_issued_readouts=b.m*b.n, pending_readouts=2,
                        readout_configuration='ordered_snapshot_before_store_issue')
    law = DenseReadyReadoutAdmission(admission.hardware_identity, admission.numerical_policy_identity,
                                    'TEST-ONLY-producer', 'TEST-ONLY-store', b.m*b.n, 2)
    if snapshot:
        law = replace(law, configuration_snapshot=DenseReadoutConfigurationSnapshot(
            adapter.configuration_identity, adapter.numerical_policy_identity,
            'TEST-ONLY-configuration-consumed-in-order-and-captured-through-completion', readout.identity))
    options = dict(name='uniform_ready', dtypes=('i8', 'i8', 'i32'), source_identity='a'*64,
        external_buffers_disjoint=True, schedule_plan=plan, pipeline_admission=admission,
        batch_readouts=False, output_readout=readout, readout_qualification_identity='TEST-ONLY-original-readout')
    if not uniform and not snapshot:
        with pytest.raises(ValueError, match='mixed per-tile'):
            schedule_dense_pipeline(p, r, s, adapter, readout_mode='ready', readout_admission=law, **options)
        return
    original = schedule_dense_pipeline(p, r, s, adapter, **options)
    # The ordinary public pipeline service composes original full-K+uniform
    # readout+causal frontier; no unchecked reconstructed result is accepted.
    actual = schedule_dense_pipeline(p, r, s, adapter, readout_mode='ready', readout_admission=law, **options)
    assert actual.output_readout is readout
    assert actual.realization['source']['output_tiles'] == p.tiles[0]*p.tiles[2]
    original_stores = sorted(tuple(evaluate(v, env) for _, v in c.args)
        for c, env in instances(original.kernel) if c.instr == readout.instruction_name)
    actual_stores = sorted(tuple(evaluate(v, env) for _, v in c.args)
        for c, env in instances(actual.kernel) if c.instr == readout.instruction_name)
    assert actual_stores == original_stores
    _execute_tiled_with_deferred_readouts(actual.kernel, p, adapter, readout)
    actual.materialize_scf().verify()
    assert any(epoch['causal_ready_readout_compute'] for epoch in actual.realization['epochs'])
    if snapshot:
        for field in ('hardware_identity', 'numerical_policy_identity', 'readout_identity', 'execution'):
            changed = replace(law.configuration_snapshot, **{field: 'different'})
            with pytest.raises(ValueError, match='configuration snapshot'):
                schedule_dense_pipeline(p, r, s, adapter, readout_mode='ready',
                    readout_admission=replace(law, configuration_snapshot=changed), **options)
    actual.revalidate()


def _execute_tiled_with_deferred_readouts(kernel, problem, adapter, readout):
    """Independent scalar execution with a finite queue and delayed ACC reads.

    Each pending request captures its actual selected scale word. Completion
    deliberately happens after later configuration changes and reverses the
    pending requests at each real drain. Only real drains allow ACC reuse.
    """
    import struct

    from merlin.compile.scheduling.dense_contracts import TiledIntegerStore

    lhs = (np.arange(problem.m * problem.k) % 7 - 3).reshape(problem.m, problem.k)
    rhs = (np.arange(problem.k * problem.n) % 5 - 2).reshape(problem.k, problem.n)
    memory = {name: {} for name in ('lhs', 'weight', 'acc')}
    output = np.zeros((problem.m, problem.n), dtype=np.int8)
    written = np.zeros(output.shape, dtype=np.int8)
    descriptor, pending, configured = None, [], None

    def finish(request):
        operation, scale_word = request
        scale = np.float32(struct.unpack('<f', struct.pack('<I', scale_word))[0])
        for i in range(operation.rows):
            for j in range(operation.cols):
                raw = memory['acc'][operation.accumulator_row + i, j]
                raw = max(0, raw) if readout.relu else raw
                value = np.clip(np.rint(np.float32(raw) * scale), -128, 127).astype(np.int8)
                y, x = operation.dram_row + i, operation.dram_col + j
                output[y, x] = value
                written[y, x] += 1

    for call, environment in instances(kernel):
        if call == Call(DRAIN, ()):
            while pending:
                finish(pending.pop())
            continue
        concrete = replace(call, args=tuple((key, evaluate(value, environment)) for key, value in call.args))
        operation = adapter.project(concrete)
        if type(operation) is Load:
            source = lhs if operation.role == 'lhs' else rhs
            for i in range(operation.rows):
                for j in range(operation.cols):
                    memory[operation.role][operation.row + i, j] = int(
                        source[operation.dram_row + i, operation.dram_col + j])
        elif type(operation) is Preload:
            descriptor = operation
        elif type(operation) is Compute:
            for i in range(operation.rows):
                for j in range(descriptor.cols):
                    key = descriptor.accumulator_row + i, j
                    total = memory['acc'][key] if descriptor.accumulate else 0
                    total += sum(memory['lhs'][operation.input_row + i, k]
                                 * memory['weight'][descriptor.weight_row + k, j]
                                 for k in range(operation.cols))
                    memory['acc'][key] = total
        else:
            assert type(operation) is TiledIntegerStore
            configured = readout.table_words[operation.readout_tile]
            if len(pending) == 2:
                finish(pending.pop(0))
            pending.append((operation, configured))
    assert not pending
    expected = lhs @ rhs
    np.testing.assert_array_equal(written, 1)
    np.testing.assert_array_equal(output, np.array([
        [readout.evaluate(int(expected[y, x]), x) for x in range(problem.n)]
        for y in range(problem.m)
    ], dtype=np.int8))
