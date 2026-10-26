"""Whole actual shape boundaries and compiler-valid independent mutations."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser
from xdsl.printer import Printer

from merlin.common.paths import repo_root
from merlin.verify.lean_graph_loops_shape import verify_shape_files
from merlin.verify.lean_graph_loops_shape_model import SHAPE_ROOTS
from merlin.verify.lean_graph_loops_shape_selection import load_selection


def context_data():
    selected = os.environ.get("MERLIN_GRAPH_SHAPE_SELECTION")
    if not selected:
        pytest.skip("explicit unaffected shape proof/source selection unavailable")
    base = repo_root() / "out/artifacts/verification/graph-shape-normalization"
    cases = {
        name: (
            base / f"actual-cases-v2/{name}.mlir",
            base / f"actual-cases-v2/{name}/after.mlir",
            32 if name == "rank_one" else 64,
        )
        for name in ("rank_one", "rank_two", "rank_three", "nested_captures", "zero_metadata_only", "compact_huge")
    }
    cases["callee_dimensions"] = (
        base / "additional-cases-v1/callee_dimensions.mlir",
        base / "additional-cases-v1/callee_dimensions/after.mlir",
        64,
    )
    cases["shape_memory_chain"] = (
        base / "additional-cases-v2/shape_memory_chain.mlir",
        base / "additional-cases-v2/shape_memory_chain/after.mlir",
        64,
    )
    return load_selection(selected), cases


@pytest.fixture(scope="module")
def context():
    return context_data()


def parse(text):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, scf.Scf, memref.MemRef, func.Func):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    return module


def render(module):
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def check(context, output, *, name="rank_two", texts=None, policy="typed_static_identity_layout", max_nodes=5000):
    selection, cases = context
    before, after, bits = cases[name]
    if texts is not None:
        output.mkdir(parents=True, exist_ok=True)
        before, after = output / "source.mlir", output / "candidate.mlir"
        before.write_text(texts[0])
        after.write_text(texts[1])
    return verify_shape_files(
        before,
        after,
        selection=selection,
        output=output / "proof",
        index_bits=bits,
        max_nodes=max_nodes,
        descriptor_policy=policy,
    )


@pytest.mark.parametrize(
    "name",
    (
        "rank_one",
        "rank_two",
        "rank_three",
        "nested_captures",
        "zero_metadata_only",
        "compact_huge",
        "callee_dimensions",
        "shape_memory_chain",
    ),
)
def test_actual_whole_module(context, tmp_path, name):
    result = check(context, tmp_path, name=name)
    assert result["lean_complete_shape_checked"] and result["returncode"] == 0
    assert list(result["transitive_axioms"]) == list(SHAPE_ROOTS)
    assert result["returned_values_and_entire_world"] and result["private_ssa_stacks_related_not_equated"]
    assert result["retained_operations_may_be_partial_and_observable"]
    assert result["typed_runtime_descriptor_fidelity_is_premise"]
    assert not any(
        result[key]
        for key in (
            "ordinary_compiler_gating",
            "pipeline_receipt_issued",
            "old_epoch_model_adapter_proved",
            "independent_review_complete",
            "numerical_gates_discharged",
            "categorized_error_or_divergence_equivalence",
            "native_allocator_abi_hardware_conformance_proved",
        )
    )
    model = json.loads(Path(result["model"]["path"]).read_bytes())
    assert {site["kind"] for site in model["changed_sites"]} == {"cast", "dimension"}


def mutate(module, kind):
    function = next(op for op in module.walk() if isinstance(op, func.FuncOp))
    block = function.body.block
    constants = [op for op in block.ops if isinstance(op, arith.ConstantOp)]
    zero = next(op for op in constants if op.value.value.data == 0)
    one = next(op for op in constants if op.value.value.data == 1)
    loads = [op for op in function.walk() if isinstance(op, memref.LoadOp)]
    ret = next(op for op in block.ops if isinstance(op, func.ReturnOp))
    if kind == "wrong_buffer":
        loads[-1].operands = (block.args[1], *loads[-1].operands[1:])
    elif kind == "wrong_coordinate":
        loads[-1].operands = (*loads[-1].operands[:-1], one.result)
    elif kind in ("wrong_dimension", "wrong_other_axis"):
        dimension = constants[-2] if kind == "wrong_dimension" else constants[-1]
        dimension.properties["value"] = builtin.IntegerAttr.from_index_int_value(7)
    elif kind == "changed_return":
        ret.operands = (ret.operands[1], ret.operands[0], ret.operands[2])
    elif kind == "extra_effect":
        store = next(op for op in function.walk() if isinstance(op, memref.StoreOp))
        block.insert_op_before(store.clone(), ret)
    elif kind == "missed_cast":
        block.insert_op_before(memref.CastOp.get(block.args[0], block.args[0].type), ret)
    elif kind == "missed_dimension":
        dim = memref.DimOp(block.args[1], zero.result)
        block.insert_op_before(dim, ret)
    elif kind == "captured_index":
        loop = next(op for op in function.walk() if isinstance(op, scf.ForOp))
        load = next(op for op in loop.walk() if isinstance(op, memref.LoadOp))
        load.operands = (*load.operands[:-1], one.result)
    elif kind == "untouched_observer":
        other = [op for op in module.walk() if isinstance(op, func.FuncOp)][-1]
        result = other.body.block.last_op
        constant = arith.ConstantOp.from_int_and_width(0, 32)
        other.body.block.insert_op_before(constant, result)
        result.operands = (constant.result,)
    elif kind == "callee_return":
        callee = next(op for op in module.walk() if isinstance(op, func.FuncOp) and op.sym_name.data == "dimension")
        result = callee.body.block.last_op
        result.operands = (callee.body.block.first_op.results[0],)
    elif kind in ("missing_free", "double_free", "early_free"):
        free = next(op for op in block.ops if isinstance(op, memref.DeallocOp))
        if kind == "missing_free":
            block.erase_op(free)
        elif kind == "double_free":
            block.insert_op_after(free.clone(), free)
        else:
            allocation = next(op for op in block.ops if isinstance(op, memref.AllocOp))
            block.detach_op(free)
            block.insert_op_after(free, allocation)
    else:
        raise AssertionError(kind)
    module.verify()


@pytest.mark.parametrize(
    "kind,name",
    [
        (kind, "rank_two")
        for kind in (
            "wrong_buffer",
            "wrong_coordinate",
            "wrong_dimension",
            "wrong_other_axis",
            "changed_return",
            "extra_effect",
            "missed_cast",
            "missed_dimension",
            "untouched_observer",
        )
    ]
    + [("captured_index", "nested_captures"), ("callee_return", "callee_dimensions")]
    + [(kind, "shape_memory_chain") for kind in ("missing_free", "double_free", "early_free")],
)
def test_valid_changed_candidate_refused(context, tmp_path, kind, name):
    _, cases = context
    before, after, _ = cases[name]
    candidate = parse(after.read_text())
    mutate(candidate, kind)
    text = render(candidate)
    parse(text)  # The mutation's SSA/types verify before the validator is asked.
    if kind in ("wrong_dimension", "wrong_other_axis"):
        result = check(context, tmp_path, name=name, texts=(before.read_text(), text))
        assert not result["lean_complete_shape_checked"] and result["returncode"] > 0
        assert "Lean refused" in result["refusal"]
    else:
        with pytest.raises(ValueError):
            check(context, tmp_path, name=name, texts=(before.read_text(), text))
        refusal = json.loads((tmp_path / "proof/REFUSAL.json").read_bytes())
        assert not refusal["lean_complete_shape_checked"]
        assert not (tmp_path / "proof/Obligation.lean").exists()


@pytest.mark.parametrize("kind", ("dtype", "layout", "memory_space", "dynamic"))
def test_valid_unsupported_descriptor_boundary(context, tmp_path, kind):
    _, cases = context
    before, after, _ = cases["rank_two"]
    original, candidate = before.read_text(), after.read_text()
    if kind == "dtype":
        candidate = candidate.replace("i32", "i16")
    else:
        replacement = {
            "layout": "memref<2x3xi32, strided<[3, 1], offset: 1>>",
            "memory_space": "memref<2x3xi32, 1>",
            "dynamic": "memref<?x3xi32>",
        }[kind]
        original = original.replace("memref<2x3xi32>", replacement)
        candidate = candidate.replace("memref<2x3xi32>", replacement)
    parse(original)
    parse(candidate)
    with pytest.raises(ValueError):
        check(context, tmp_path, texts=(original, candidate))


def test_malformed_input_is_parse_refusal(context, tmp_path):
    with pytest.raises(ValueError, match="malformed IR"):
        check(context, tmp_path, texts=("not MLIR", "not MLIR"))
    assert not (tmp_path / "proof/Obligation.lean").exists()


def test_descriptor_contract_cannot_be_inferred(context, tmp_path):
    with pytest.raises(ValueError, match="descriptor policy"):
        check(context, tmp_path, policy="metadata_hashes_only")


def test_no_prefix_under_work_budget(context, tmp_path):
    with pytest.raises(ValueError, match="complete module structural node budget"):
        check(context, tmp_path, max_nodes=1)


def test_declined_identity_is_separate(context, tmp_path):
    _, cases = context
    normalized = cases["rank_two"][1].read_text()
    with pytest.raises(ValueError, match="identity/declined"):
        check(context, tmp_path, texts=(normalized, normalized))


def test_signed_dimension_limit_reaches_kernel(context, tmp_path):
    _, cases = context
    before, after, _ = cases["rank_one"]
    source = before.read_text().replace("memref<7xi8>", "memref<2147483648xi8>")
    candidate = after.read_text().replace("memref<7xi8>", "memref<2147483648xi8>")
    candidate = candidate.replace("value = 7 : index", "value = 2147483648 : index")
    parse(source)
    parse(candidate)
    result = check(context, tmp_path, name="rank_one", texts=(source, candidate))
    assert result["returncode"] > 0 and not result["lean_complete_shape_checked"]
    assert "Lean refused" in result["refusal"]


def test_compiler_valid_wrong_index_width(context, tmp_path):
    from xdsl.rewriter import Rewriter

    _, cases = context
    before, after, _ = cases["rank_two"]
    candidate = parse(after.read_text())
    function = candidate.body.block.first_op
    # Select the dimension used by the first returned value, independent of position.
    returned = function.body.block.last_op
    constant = returned.operands[0].owner
    replacement = arith.ConstantOp.from_int_and_width(constant.value.value.data, 64)
    Rewriter.replace_op(constant, replacement)
    function.properties["function_type"] = builtin.FunctionType.from_lists(
        function.function_type.inputs, [builtin.i64, builtin.IndexType(), builtin.i32]
    )
    candidate.verify()
    with pytest.raises(ValueError, match="signature"):
        check(context, tmp_path, texts=(before.read_text(), render(candidate)))
