"""Isolated exported-source calls, with a neutral test codec and no target grant."""

from dataclasses import replace

from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i64

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import DenseLoopAdapter, DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_panel_pipeline import (
    DensePanelOverlapAdmission,
    DenseReadoutComputeAdmission,
    DenseRetainedReductionAdmission,
)
from merlin.compile.scheduling.dense_readout_overlap import schedule_dense_readout_overlap
from merlin.compile.scheduling.dense_segmented_pipeline import (
    DenseSegmentedAdmission,
    batch_dense_segmented_readouts,
    schedule_dense_segmented_pipeline,
)
from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
from merlin.compile.scheduling.scf_commands import (
    emit_scf_commands,
    materialize_scf_commands,
    unroll_small_command_loops,
    verify_scf_commands,
)
from merlin.compile.scheduling.schedule_ir import make_dense_schedule_plan
from merlin.llvmlower.host_builder import FnBuilder
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, as_expr
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks


def encode(op):
    if type(op) is Load:
        names = ("a", "b", "src_row", "src_col", "rows", "cols", "dst", "step")
        values = (*op.block, op.dram_row, op.dram_col, op.rows, op.cols, op.row, op.step)
        name = op.role + "_load"
    elif type(op) is Preload:
        names = ("weight", "k", "n", "acc", "continue", "rows", "cols", "depth")
        values = (op.weight_row, *op.weight_block, op.accumulator_row, op.accumulate,
                  op.rows, op.cols, op.weight_rows)
        name = "preload"
    elif type(op) is Compute:
        names = ("lhs", "m", "k", "rows", "depth")
        values = (op.input_row, *op.input_block, op.rows, op.cols)
        name = "compute"
    else:
        assert type(op) is Store
        names = ("row", "col", "rows", "cols", "acc")
        values = (op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row)
        name = "store"
    return Call(name, tuple(zip(names, map(as_expr, values))))


def project(call):
    a = dict(call.args)
    if call.instr in ("lhs_load", "weight_load"):
        return Load(call.instr.removesuffix("_load"), (a["a"], a["b"]),
                    a["src_row"], a["src_col"], a["rows"], a["cols"], a["dst"], a["step"])
    if call.instr == "preload":
        return Preload(a["weight"], (a["k"], a["n"]), a["acc"], a["continue"],
                       a["rows"], a["cols"], a["depth"])
    if call.instr == "compute":
        return Compute(a["lhs"], (a["m"], a["k"]), a["rows"], a["depth"], True)
    if call.instr == "store":
        return Store(a["row"], a["col"], a["rows"], a["cols"], a["acc"])
    raise ValueError("unknown neutral instruction")


def llvm_module(module, name, *, persistent=False):
    builder, declarations, annotations = FnBuilder([]), {}, []
    bridge = FnBuilderIndexAdapter(builder)

    def lower(instruction, operands, metadata):
        declarations[instruction] = len(operands)
        annotations.append(metadata)
        builder.add(llvm.CallOp(instruction, *(bridge.value(v) for v in operands.values())))

    report = emit_scf_commands(module, bridge, lower, allow_persistent_state=persistent)
    function = llvm.FuncOp(name, llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())
    output = ModuleOp([
        *(llvm.FuncOp(n, llvm.LLVMFunctionType([i64] * count, llvm.LLVMVoidType()))
          for n, count in declarations.items()), function,
    ])
    output.verify()
    assert any(type(op) is llvm.CallOp for op in function.walk())
    return output, report, annotations


def run():
    adapter = DenseLoopAdapter(encode, project, "neutral-test-config", "neutral-source-policy", True, True, 64)
    p = DenseProblem(3, 4, 25, 2, 3, 4, 1, 1, 4)
    storage = DenseLoopStorage(
        DenseMemoryView("lhs", "left_sram", 3, 2, 256),
        DenseMemoryView("rhs", "right_sram", 4, 3, 48),
        DenseMemoryView("acc", "acc_sram", 16, 1, 64),
    )
    resources = DenseResources(256 * 3 + 48 * 4, 64 * 16, 2, 2, 2,
                               partial_output_tiles=True, partial_output_blocks=True)
    block = next(b for b in dense_blocks(p, resources) if (b.m, b.k, b.n) == (2, 2, 2))
    args = (adapter.configuration_identity, adapter.numerical_policy_identity)
    overlap = DensePanelOverlapAdmission(*args, "TEST-ONLY-disjoint-input-epochs", 4096, 4096, 2, 2)
    reduction = DenseRetainedReductionAdmission(*args, "TEST-ONLY-ordered-reductions", 4096, 4096, 2, 2)
    law = DenseSegmentedAdmission(overlap, reduction, "TEST-ONLY-explicit-conjunction")
    plan = make_dense_schedule_plan(
        p, resources, storage, adapter, source_identity="b" * 64, dtypes=("i8", "i8", "i32"),
        static_budget=128, block_tiles=(2, 2, 2), pipeline_kind="retained_lhs_segmented_k",
        pipeline_prefetch=True,
    )
    original = schedule_dense_segmented_pipeline(
        p, resources, storage, adapter, law, name="exported_segmented", dtypes=("i8", "i8", "i32"),
        schedule_plan=plan, source_identity="b" * 64, external_buffers_disjoint=True,
    )
    def store(component):
        return replace(component, max_issued_readouts=4, pending_readouts=2,
                       readout_configuration="ordered_snapshot_before_store_issue")
    law = replace(law, overlap=store(law.overlap), reduction=store(law.reduction))
    original = batch_dense_segmented_readouts(original, p, block, resources, storage, adapter, law)
    capability = DenseReadoutComputeAdmission(
        *args, "TEST-ONLY-disjoint-ACC-completion", "TEST-ONLY-store-snapshot", 4, 2,
    )
    result = schedule_dense_readout_overlap(
        original, p, block, resources, storage, adapter, capability, group_path=("outer_m", "outer_n"),
    )
    assert result.realization["source"]["output_tiles"] == p.tiles[0] * p.tiles[2]
    assert result.realization["transformations"][0]["odd_tail"] == 1
    assert any(e["disjoint_readout_compute"] for e in result.realization["epochs"])
    assert result.realization["dynamic_epoch_drains"] < original.realization["dynamic_epoch_drains"]
    actual = result.materialize_scf()
    output, _, _ = llvm_module(actual, "readout_overlap")
    assert "merlin_dense_panel_drain" in str(output)
    result.revalidate()

    state_kernel = Kernel("persistent", (), (
        Call("configure", (), sets=(("mode", Const(-7)),)),
        Loop("m", 3, (Loop("n", 2, (
            Call("consume", (("index", Var("m") * 2 + Var("n")),), assumes=(("mode", Const(-7)),)),
        )),)),
    ))
    try:
        materialize_scf_commands(state_kernel)
    except ValueError as error:
        assert "persistent configuration" in str(error)
    else:
        raise AssertionError("persistent-state opt-in disappeared")
    state = materialize_scf_commands(state_kernel, allow_persistent_state=True)
    transformed = unroll_small_command_loops(state, allow_persistent_state=True)
    verify_scf_commands(transformed, allow_persistent_state=True)
    state_output, state_report, annotations = llvm_module(transformed, "persistent_state", persistent=True)
    assert any(a.sets for a in annotations) and any(a.assumes for a in annotations)
    assert state_report["declared_persistent_state"]["target_state_visibility_and_clobber_law"] == "NOT_ESTABLISHED"
    assert "llvm.call @consume" in str(state_output)
    return {
        "readout_source_products": p.m * p.k * p.n,
        "readout_actual_kernel_sha256": result.kernel.digest(),
        "readout_dynamic_drains_before": original.realization["dynamic_epoch_drains"],
        "readout_dynamic_drains_after": result.realization["dynamic_epoch_drains"],
        "readout_llvm_operations": sum(1 for _ in output.walk()),
        "persistent_llvm_operations": sum(1 for _ in state_output.walk()),
        "target_execution_qualification": "NOT_ESTABLISHED; neutral compiler closure probe only",
    }
