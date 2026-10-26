"""Actual whole-triple correspondence and compiler-valid adversarial changes."""

from __future__ import annotations

import copy
import io
import json
import os
import subprocess
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.ir import Block
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter

from merlin.common.paths import repo_root
from merlin.verify.lean_graph_loops_memory_scf import verify_memory_scf_files
from merlin.verify.lean_graph_loops_memory_scf_model import MEMORY_SCF_ROOTS, obligation
from merlin.verify.lean_graph_loops_memory_scf_selection import load_selection
from merlin.verify.lean_graph_loops_reuse import ALLOCATOR_MODE


@pytest.fixture(scope="module")
def context():
    selected = os.environ.get("MERLIN_GRAPH_MEMORY_SCF_SELECTION")
    if not selected:
        pytest.skip("explicit unaffected memory-SCF proof/source context unavailable")
    base = repo_root() / "out/artifacts/verification/graph-memory-scf"
    old = repo_root() / "out/artifacts/verification/graph-buffer-reuse"
    paths = dict(
        two_epochs=(
            old / "actual-cases-v3/two_epochs/before.mlir",
            old / "actual-cases-v3/two_epochs/after.mlir",
            old / "actual-neighbor-segment-v1/after.mlir",
            64,
        )
    )
    for row in json.loads((base / "independent-cases-v2/RUNS.json").read_bytes()):
        case = base / "independent-cases-v2" / row["name"]
        paths[row["name"]] = (case.with_suffix(".mlir"), case / "middle.mlir", case / "after.mlir", row["index_bits"])
    return load_selection(selected), paths


def check(context, tmp_path, *, name="two_epochs", texts=None, allocator=ALLOCATOR_MODE, max_nodes=5000):
    selection, cases = context
    before, middle, after, bits = cases[name]
    if texts is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        paths = [tmp_path / name for name in ("before.mlir", "middle.mlir", "after.mlir")]
        for path, text in zip(paths, texts, strict=True):
            path.write_text(text)
        before, middle, after = paths
    return verify_memory_scf_files(
        before,
        middle,
        after,
        selection=selection,
        output=tmp_path / "proof",
        index_bits=bits,
        max_nodes=max_nodes,
        allocator_model=allocator,
    )


def parse(text):
    ctx = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, scf.Scf, memref.MemRef, func.Func):
        ctx.load_dialect(dialect)
    module = Parser(ctx, text).parse_module()
    module.verify()
    return module


def text(module):
    output = io.StringIO()
    Printer(stream=output, print_generic_format=True).print_op(module)
    return output.getvalue()


@pytest.mark.parametrize(
    "name",
    (
        "two_epochs",
        "rank_one_tail",
        "rank_two_tail",
        "rank_two_one_row",
        "rank_two_prefix",
        "other_live_slot",
        "compact_small",
        "compact_huge",
    ),
)
def test_actual_complete_common_memory(context, tmp_path, name):
    result = check(context, tmp_path, name=name)
    assert result["lean_complete_composition_checked"] and result["returncode"] == 0
    assert set(result["transitive_axioms"]) == set(MEMORY_SCF_ROOTS)
    assert result["operation_level_initializer_bridge"] and result["same_semantics_exact_middle"]
    assert result["allocator_total_private_unobservable_is_premise"]
    assert not any(
        result[name]
        for name in (
            "categorized_failure_or_divergence_equivalence",
            "primitive_totality_inferred",
            "independent_review_complete",
            "pipeline_receipt_issued",
            "numerical_gates_discharged",
            "native_allocator_or_hardware_proved",
        )
    )


def test_huge_shape_no_proof_unrolling(context, tmp_path):
    small = check(context, tmp_path / "small", name="compact_small")
    huge = check(context, tmp_path / "huge", name="compact_huge")
    a = json.loads(Path(small["model"]["path"]).read_bytes())
    b = json.loads(Path(huge["model"]["path"]).read_bytes())
    assert a["structural_nodes"] == b["structural_nodes"]
    assert abs(small["query"]["bytes"] - huge["query"]["bytes"]) < 8192


def mutate(module, kind):
    function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
    block = function.body.block
    roots = [op for op in block.ops if isinstance(op, scf.ForOp)]
    second = roots[1]
    loops = [op for op in second.walk() if isinstance(op, scf.ForOp)]
    chunk, inner = loops[1:]
    store = next(op for op in second.walk() if isinstance(op, memref.StoreOp))
    one = next(
        op
        for op in block.ops
        if isinstance(op, arith.ConstantOp)
        and isinstance(op.result.type, builtin.IndexType)
        and op.value.value.data == 1
    )
    if kind == "second_value":
        store.operands = (block.args[0], *store.operands[1:])
    elif kind == "second_index":
        store.operands = (*store.operands[:-1], store.operands[-2])
    elif kind == "partial_coverage":
        inner.operands = (inner.lb, one.result, inner.step)
    elif kind == "inner_stride":
        chunk.operands = (chunk.lb, chunk.ub, second.step)
    elif kind == "tail_bound":
        minimum = next(op for op in second.walk() if isinstance(op, arith.MinUIOp))
        minimum.operands = (minimum.lhs, one.result)
    elif kind == "outer_stride":
        second.operands = (second.lb, second.ub, one.result)
    elif kind == "missing_second_initializer":
        block.erase_op(second)
    elif kind == "retained_capture":
        call = [op for op in block.ops if isinstance(op, func.CallOp)][-1]
        call.operands = (block.args[0], *call.operands[1:])
    elif kind == "untouched_observer":
        untouched = [op for op in module.walk() if isinstance(op, func.FuncOp)][-1]
        constant = next(op for op in untouched.walk() if isinstance(op, arith.ConstantOp))
        constant.properties["value"] = builtin.IntegerAttr(2, builtin.i32)
    elif kind in ("missing_final_free", "early_final_free", "double_free"):
        free = next(op for op in block.ops if isinstance(op, memref.DeallocOp))
        if kind == "missing_final_free":
            block.erase_op(free)
        elif kind == "early_final_free":
            block.detach_op(free)
            block.insert_op_before(free, second)
        else:
            block.insert_op_after(free.clone(), free)
    elif kind == "new_carry":
        body = Block(arg_types=[builtin.IndexType(), builtin.i32])
        mapper = {second.body.block.args[0]: body.args[0]}
        body.add_ops(op.clone(mapper) for op in list(second.body.block.ops)[:-1])
        body.add_op(scf.YieldOp(body.args[1]))
        replacement = scf.ForOp(second.lb, second.ub, second.step, [block.args[0]], body)
        Rewriter.replace_op(second, [replacement], [])
    elif kind == "added_alias":
        alloc = next(op for op in block.ops if isinstance(op, memref.AllocOp))
        block.insert_op_after(memref.CastOp.get(alloc.memref, alloc.memref.type), alloc)
    else:
        raise AssertionError(kind)
    module.verify()


@pytest.mark.parametrize(
    "kind",
    (
        "second_value",
        "second_index",
        "partial_coverage",
        "inner_stride",
        "tail_bound",
        "outer_stride",
        "missing_second_initializer",
        "retained_capture",
        "untouched_observer",
        "missing_final_free",
        "early_final_free",
        "double_free",
        "added_alias",
        "new_carry",
    ),
)
def test_compiler_valid_mutations_refuse(context, tmp_path, kind):
    _, paths = context
    buffers = [path.read_text() for path in paths["two_epochs"][:3]]
    module = parse(buffers[2])
    mutate(module, kind)
    buffers[2] = text(module)
    try:
        refused = check(context, tmp_path, texts=buffers)
    except ValueError:
        refused = json.loads((tmp_path / "proof/REFUSAL.json").read_bytes())
    else:
        assert refused["returncode"] > 0, "unexpected HOST crash disqualifies the context"
    assert not refused["lean_complete_composition_checked"]


def test_exact_middle_cannot_change(context, tmp_path):
    _, paths = context
    buffers = [path.read_text() for path in paths["two_epochs"][:3]]
    module = parse(buffers[1])
    untouched = [op for op in module.walk() if isinstance(op, func.FuncOp)][-1]
    constant = next(op for op in untouched.walk() if isinstance(op, arith.ConstantOp))
    constant.properties["value"] = builtin.IntegerAttr(2, builtin.i32)
    module.verify()
    buffers[1] = text(module)
    with pytest.raises(ValueError, match="retained"):
        check(context, tmp_path, texts=buffers)


@pytest.mark.parametrize("kind", ("slot", "inner_step", "scope", "missing_registry", "signed_bound", "context"))
def test_kernel_recomputes_facts(context, tmp_path, kind):
    selection, _ = context
    accepted = check(context, tmp_path / "valid")
    model = copy.deepcopy(json.loads(Path(accepted["model"]["path"]).read_bytes()))
    if kind == "slot":
        model["reuse"]["objects"][1]["slot"] = 99
    elif kind == "inner_step":
        model["configurations"][1]["segment"]["targetInnerStep"] = 2
    elif kind == "scope":
        model["configurations"][1]["program"]["leaf"]["indices"][0] += 100
    elif kind == "missing_registry":
        model["configurations"].pop()
    elif kind == "signed_bound":
        model["configurations"][1]["segment"]["bounds"]["upper"] = 2**63 - 1
    else:
        model["configurations"][1]["segment"]["targetContext"] += "changed"
    query = tmp_path / "Mutated.lean"
    query.write_text(obligation(model))
    before = selection.dependencies()
    env = dict(
        os.environ,
        LEAN_PATH=str(selection.proof.library),
        LEAN_SYSROOT=str(selection.proof.executable.parent.parent),
        LEAN_NUM_THREADS="2",
    )
    result = subprocess.run(
        [str(selection.proof.executable), str(query)], env=env, capture_output=True, text=True, check=False, timeout=120
    )
    (tmp_path / "stdout.log").write_text(result.stdout)
    (tmp_path / "stderr.log").write_text(result.stderr)
    assert result.returncode > 0, "unexpected HOST crash disqualifies this context; never retry to promote"
    assert before == selection.dependencies()


@pytest.mark.parametrize("allocator", ("finite_capacity", "observable_pointers", ""))
def test_allocator_counterexample_is_not_erased(context, tmp_path, allocator):
    with pytest.raises(ValueError, match="allocator"):
        check(context, tmp_path, allocator=allocator)


def test_complete_node_budget(context, tmp_path):
    with pytest.raises(ValueError, match="budget"):
        check(context, tmp_path, max_nodes=1)


def test_malformed_ir_is_separate(context, tmp_path):
    _, paths = context
    buffers = [path.read_text() for path in paths["two_epochs"][:3]]
    buffers[2] = buffers[2][:-13]
    with pytest.raises(ValueError, match="parsing/SSA"):
        check(context, tmp_path, texts=buffers)


def test_rank_three_remains_refused(context, tmp_path):
    from merlin.verify.lean_graph_loops_memory_scf_projection import initializer_program

    class UnusedDecoder:
        source_ops = {}

    with pytest.raises(ValueError, match="rank1/rank2"):
        initializer_program(UnusedDecoder(), {"shape": [2, 3, 4]})
