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


def test_final_wide_bands_keep_causal_order_original_source_and_strict_owner():
    from merlin.compile.scheduling.dense_ready_readout import admit_ready_readout_tile_loads
    from merlin.compile.scheduling.panel_tile_loads import admit_panel_tile_loads
    from merlin.compile.scheduling.tile_load_coalescing import (
        TileLoadAddressLaw,
        TileLoadMapping,
        _linear,
        propose_tile_load_coalescing,
    )

    p, r, s, b, original, law = setup((4, 6, 24))
    ready = schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law)
    laws = {role: TileLoadAddressLaw(tile, rows, 4 * tile, 1, pitch, stride, "TEST-ONLY-exact-band-addressing")
            for role, tile, rows, pitch, stride in (
                ("lhs", p.tile_k, p.tile_m, p.k, p.tile_m),
                ("weight", p.tile_n, p.tile_k, p.n, p.tile_k))}
    def project(call):
        if call.instr not in ("lhs_load", "weight_load"):
            return None
        op = ADAPTER.project(call)
        view = s.lhs if op.role == "lhs" else s.rhs
        law = laws[op.role]
        return TileLoadMapping(op.role, view.backing, op.dram_row * law.source_pitch_bytes + op.dram_col,
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
    result = admit_ready_readout_tile_loads(ready, proposal, p, b, r, s, ADAPTER, project=project, encode_band=band)
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
