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


@pytest.mark.parametrize("shape", [(24, 14, 24), (23, 13, 23)])
@pytest.mark.parametrize("limited_role", ["lhs_rows", "operand_bytes"])
def test_ready_input_selection_joins_live_footprints_before_explicit_plan(shape, limited_role, tmp_path):
    from test_dense_retained_reduction import chosen, inputs

    from merlin.compile.scheduling.dense_panel_pipeline import dense_retained_choices
    from merlin.compile.scheduling.dense_ready_readout import dense_ready_input_choices, schedule_dense_pipeline
    from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan

    p, r, s, _, admission = inputs(shape)
    if limited_role == "lhs_rows":
        s = replace(s, lhs=replace(s.lhs, capacity=30))
    else:
        r = replace(r, operand_bytes=180)
    ordinary = dense_retained_choices(p, r, s, roles=("weight",))[0]
    placements = dense_ready_input_choices(p, r, s)
    placement = placements[0]
    assert (ordinary.block.m, ordinary.block.n) == (2, 2)
    assert (placement.block.m, placement.block.n) == (1, 2)
    assert ordinary.block not in [choice.block for choice in placements]
    assert placement.lhs_slot_offset + placement.lhs_rows <= s.lhs.capacity
    assert ((placement.lhs_slot_offset + placement.lhs_rows) * s.lhs.unit_bytes
            + placement.rhs_rows * s.rhs.unit_bytes <= r.operand_bytes)

    def build(block, **input_options):
        law = replace(admission, max_issued_readouts=block.m * block.n, pending_readouts=2,
                      readout_configuration="ordered_snapshot_before_store_issue")
        original = chosen(p, r, s, block, law, role="weight", order=("n", "m"), compute_order=("k", "n", "m"))
        ready = DenseReadyReadoutAdmission(law.hardware_identity, law.numerical_policy_identity,
            "TEST-ONLY-producer", "TEST-ONLY-store", block.m * block.n, 2,
            disjoint_load_qualification_identity="TEST-ONLY-load-ready-conjunction")
        return schedule_dense_pipeline(p, r, s, ADAPTER, name="retained", dtypes=("i8", "i8", "i32"),
            source_identity=original.source_identity, external_buffers_disjoint=True,
            schedule_plan=parse_dense_schedule_plan(original.realization["schedule_ir"]["text"]),
            pipeline_admission=law, batch_readouts=True, readout_mode="ready", readout_admission=ready,
            static_budget=128, **input_options)

    # The old selected shape still refuses; the new option never silently
    # shrinks an explicit block or repairs somebody else's ownership.
    with pytest.raises(ValueError, match="assigned private view|operand byte budget"):
        build(ordinary.block, input_pipeline_row_offset=ordinary.block.m * ordinary.block.k * p.tile_m)
    with pytest.raises(ValueError, match="cannot replace the explicit selected block"):
        build(ordinary.block, input_pipeline_placement=placement)
    result = build(placement.block, input_pipeline_placement=placement)
    execute(result.kernel, p)
    result.materialize_scf().verify()
    assert result.input_pipeline_placement is placement
    assert result.realization["input_pipeline_placement"]["lhs_rows"] == placement.lhs_rows
    assert all(region["actual_role_windows"]["lhs_read"] == (
        s.lhs.base, s.lhs.base + placement.lhs_rows) for region in result.realization["input_pipeline"])
    manual = build(placement.block, input_pipeline_row_offset=placement.lhs_slot_offset)
    assert result.kernel == manual.kernel  # placement supplies no new schedule policy
    assert result.realization["dynamic_epoch_drains"] == manual.realization["dynamic_epoch_drains"]
    if limited_role == "lhs_rows":
        _execute_emitted_ready_c(tmp_path, result, p, s)
    with pytest.raises(ValueError, match="footprint"):
        replace(placement, lhs_rows=placement.lhs_rows - 1).validate_for(p, r, s)
    with pytest.raises(ValueError, match="footprint"):
        placement.validate_for(p, r, replace(s, lhs=replace(s.lhs, capacity=s.lhs.capacity + 1)))


def _execute_emitted_ready_c(directory, result, p, storage):
    """Native scalar effects execute the emitted counted control and row slots."""
    import ctypes
    import shutil
    import subprocess

    from merlin.compile.scheduling.dense_panel_pipeline import _calls
    from merlin.sched.codegen import emit_c_function
    from merlin.sched.isa import InstrDef, InstructionSet, Operand

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C compiler unavailable")
    definitions = {}
    for call in _calls(result.kernel.body):
        suffix = ", lhs" if call.instr == "lhs_load" else ", rhs" if call.instr == "weight_load" else (
            ", output" if call.instr == "store" else "")
        definitions[call.instr] = InstrDef(call.instr, tuple(Operand(k, "int") for k, _ in call.args),
            render_c=lambda args, n=call.instr, end=suffix: f"do_{n}({', '.join(args)}{end});")
    instruction_set = InstructionSet("neutral-ready-native", definitions,
        {"i8": "int8_t", "i32": "int32_t"}, "/* scalar synchronous effects */")
    views = (storage.lhs, storage.rhs, storage.accumulator)
    source = "#include <stdint.h>\n#include <string.h>\n"
    for name, view in zip(("left", "right", "sums"), views, strict=True):
        source += f"static unsigned char {name}[{(view.base + view.capacity + 2) * view.unit_bytes}];\n"
    source += ("static int64_t wr, ar, cont, nc; uint64_t products;\n"
               f"static int8_t resident[{p.tile_k}][{p.tile_n}];\n")
    for name, view, columns in (("lhs", storage.lhs, p.k), ("weight", storage.rhs, p.n)):
        memory = "left" if name == "lhs" else "right"
        source += f"""
static void do_{name}_load(int64_t a,int64_t b,int64_t r,int64_t c,int64_t nr,int64_t nc,
                          int64_t dst,int64_t step,int8_t *src) {{
  for(int i=0;i<nr;i++) for(int j=0;j<nc;j++)
    {memory}[(dst+i)*{view.unit_bytes}+j]=(unsigned char)src[(r+i*step)*{columns}+c+j];
}}
"""
    source += f"""
static void do_preload(int64_t w,int64_t k,int64_t n,int64_t a,int64_t c,
                       int64_t rows,int64_t cols,int64_t depth) {{wr=w;ar=a;cont=c;nc=cols;}}
static void do_keep_descriptor(int64_t a,int64_t c,int64_t rows,int64_t cols,int64_t depth)
  {{ar=a;cont=c;nc=cols;}}
static void pair(int64_t l,int64_t rows,int64_t depth,int fresh) {{
  if(fresh) for(int t=0;t<depth;t++) for(int j=0;j<nc;j++)
    resident[t][j]=(int8_t)right[(wr+t)*{storage.rhs.unit_bytes}+j];
  for(int i=0;i<rows;i++) for(int j=0;j<nc;j++) {{
    int32_t value=0;
    if(cont) memcpy(&value,sums+(ar+i)*{storage.accumulator.unit_bytes}+4*j,4);
    for(int t=0;t<depth;t++) {{
      value+=(int8_t)left[(l+i)*{storage.lhs.unit_bytes}+t]*resident[t][j]; products++;
    }}
    memcpy(sums+(ar+i)*{storage.accumulator.unit_bytes}+4*j,&value,4);
  }}
}}
static void do_compute(int64_t l,int64_t m,int64_t k,int64_t rows,int64_t depth) {{pair(l,rows,depth,1);}}
static void do_stay(int64_t l,int64_t m,int64_t k,int64_t rows,int64_t depth) {{pair(l,rows,depth,0);}}
static void do_store(int64_t r,int64_t c,int64_t rows,int64_t cols,int64_t a,int32_t *dst) {{
  for(int i=0;i<rows;i++) for(int j=0;j<cols;j++)
    memcpy(dst+(r+i)*{p.n}+c+j,sums+(a+i)*{storage.accumulator.unit_bytes}+4*j,4);
}}
static void do_{DRAIN}(void) {{}}
"""
    result.revalidate()
    source += emit_c_function(result.kernel, instruction_set, symbol="ready", index_bits=64)
    result.revalidate()
    source += "\nvoid run(int8_t *lhs,int8_t *rhs,int32_t *output) {ready(lhs,rhs,output);}\n"
    source += "\nvoid init(void) {products=0;memset(left,85,sizeof(left));"
    source += "memset(right,85,sizeof(right));memset(sums,85,sizeof(sums));}\nint guards(void) {"
    for name, view in zip(("left", "right", "sums"), views, strict=True):
        source += (f"for(int i=0;i<sizeof({name});i++) if((i<{view.base * view.unit_bytes}"
                   f"||i>={(view.base + view.capacity) * view.unit_bytes})&&{name}[i]!=85) return 0;")
    source += "return 1;}\n"
    # Passed pytest directories may be recycled while dlopen caches a handle
    # by absolute path. Distinct shapes must retain distinct library names.
    c_file = directory / "ready.c"
    library = directory / f"ready_{p.m}_{p.k}_{p.n}.so"
    c_file.write_text(source)
    subprocess.run([cc, "-std=c11", "-O1", "-shared", "-fPIC", str(c_file), "-o", str(library)],
                   check=True, capture_output=True)
    native = ctypes.CDLL(str(library))
    lhs = (np.arange(p.m * p.k) % 7 - 3).astype(np.int8).reshape(p.m, p.k)
    rhs = (np.arange(p.k * p.n) % 5 - 2).astype(np.int8).reshape(p.k, p.n)
    output = np.full(p.m * p.n + 16, -1234567, dtype=np.int32)
    native.init()
    native.run(ctypes.c_void_p(lhs.ctypes.data), ctypes.c_void_p(rhs.ctypes.data),
               ctypes.c_void_p(output.ctypes.data + 8 * output.itemsize))
    np.testing.assert_array_equal(output[8:-8].reshape(p.m, p.n), lhs.astype(np.int32) @ rhs.astype(np.int32))
    assert np.all(output[:8] == -1234567) and np.all(output[-8:] == -1234567)
    assert native.guards() == 1
    assert ctypes.c_uint64.in_dll(native, "products").value == p.m * p.k * p.n


def test_ready_input_placement_is_checked_against_actual_accesses_and_completion_law():
    from merlin.compile.scheduling.dense_ready_readout import (
        dense_ready_input_choices,
        propose_dense_ready_input_pipeline,
    )

    p, r, s, b, original, law = retained_setup((24, 14, 24))
    placement = next(c for c in dense_ready_input_choices(p, r, s) if c.block == b)
    ready, _ = propose_dense_ready_readout(original.kernel, p, ADAPTER)
    with pytest.raises(ValueError, match="actual ready input accesses"):
        propose_dense_ready_input_pipeline(ready, ADAPTER, s, row_offset=placement.lhs_slot_offset,
            placement=replace(placement, lhs_rows=placement.lhs_rows - 1))
    with pytest.raises(ValueError, match="separately selected"):
        schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law,
                                     input_pipeline_placement=placement)
    with pytest.raises(ValueError, match="one typed placement"):
        schedule_dense_ready_readout(original, p, b, r, s, ADAPTER, law,
            input_pipeline_placement=placement, input_pipeline_row_offset=placement.lhs_slot_offset)


@pytest.mark.parametrize("shape", [(24, 12, 24), (23, 13, 23)])
def test_counted_final_k_prefix_keeps_array_runs_and_emits_only_fresh_final_stores(shape, tmp_path):
    from test_dense_array_retention import ADAPTER as array_adapter
    from test_dense_array_retention import LAW
    from test_dense_array_retention import execute as array_execute

    from merlin.compile.scheduling.dense_ready_readout import schedule_dense_pipeline
    from merlin.compile.scheduling.schedule_ir import parse_dense_schedule_plan

    p, r, s, b, original, law = retained_setup(shape)
    law = replace(law, disjoint_load_qualification_identity="TEST-ONLY-load-ready-conjunction")
    result = schedule_dense_pipeline(p, r, s, array_adapter, name="retained", dtypes=("i8", "i8", "i32"),
        source_identity=original.source_identity, external_buffers_disjoint=True,
        schedule_plan=parse_dense_schedule_plan(original.realization["schedule_ir"]["text"]),
        pipeline_admission=original.admission, batch_readouts=True, readout_mode="ready", readout_admission=law,
        array_prefix_admission=LAW, input_pipeline_row_offset=b.m*b.k*p.tile_m, static_budget=128)
    array_execute(result.kernel, p)
    _execute_emitted_ready_c(tmp_path, result, p, s)
    assert result.realization["array_prefix_projection"]["fresh_read_kernel_sha256"] != result.kernel.digest()
    assert any(call.instr == "stay" for call, _ in instances(result.kernel))
    previous = None
    for call, env in instances(result.kernel):
        if call.instr == DRAIN:
            previous = None
            continue
        op = array_adapter.project(replace(call, args=tuple((k, evaluate(v, env)) for k, v in call.args)))
        if type(op) is Store:
            assert type(previous) is Compute and previous.fresh_weights
            assert previous.input_block[1] == p.tiles[1] - 1
        previous = op
    result.materialize_scf().verify()
    result.revalidate()
    projection = DenseReadyReadoutCheckingProjection(original.kernel, p,
        input_pipeline_row_offset=b.m*b.k*p.tile_m, final_k_prefix=True, array_prefix_admission=LAW)
    with pytest.raises(ValueError, match="exact pure retained-prefix"):
        projection.hazard_kernel(replace(result.kernel, name="changed"), array_adapter, s)
    with pytest.raises(ValueError, match="original selected adapter"):
        schedule_dense_ready_readout(original, p, b, r, s, array_adapter, law,
            array_prefix_admission=replace(LAW, hardware_identity="other"))
    with pytest.raises(ValueError, match="actual commands|capability or owner"):
        replace(result, array_prefix_admission=replace(LAW, qualification_identity="different")).revalidate()


def test_final_k_prefix_refuses_one_k_and_no_reusable_m_without_repair():
    from test_dense_array_retention import ADAPTER as array_adapter
    from test_dense_array_retention import LAW
    from test_dense_retained_reduction import chosen, inputs

    from merlin.transforms.dense_planning import dense_blocks

    p, r, s, b, original, law = retained_setup((24, 3, 24))
    with pytest.raises(ValueError, match="at least two K slices"):
        schedule_dense_ready_readout(original, p, b, r, s, array_adapter, law, array_prefix_admission=LAW)
    p, r, s, _, a = inputs((24, 12, 24))
    b = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (1, p.tiles[1], 2))
    original = chosen(p, r, s, b, a, role="weight", order=("n", "m"), compute_order=("k", "n", "m"))
    with pytest.raises(ValueError, match="at least one reusable M run"):
        propose_dense_ready_readout(original.kernel, p, array_adapter, final_k_prefix=True,
                                    array_prefix_admission=LAW)


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
