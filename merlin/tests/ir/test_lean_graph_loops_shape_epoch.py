"""Actual changed four-buffer cases and compiler-valid adversarial boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from xdsl.dialects import arith, builtin, func, memref, scf

from merlin.tests.ir.test_lean_graph_loops_shape import parse, render
from merlin.verify.lean_graph_loops_shape_epoch import _parse, project_chain, verify_epoch_files
from merlin.verify.lean_graph_loops_shape_epoch_cases import CASES
from merlin.verify.lean_graph_loops_shape_epoch_selection import (
    ALLOCATOR_POLICY,
    DESCRIPTOR_POLICY,
    ENTRY_POLICY,
    REQUIRED_ROOTS,
    load_selection,
)


def case_paths(case):
    configured = os.environ.get("MERLIN_GRAPH_SHAPE_EPOCH_CASES")
    if not configured:
        pytest.skip("explicit retained actual epoch case roster unavailable")
    root = Path(configured) / case.name
    return tuple(root / name for name in ("source.mlir", "shape/after.mlir", "memory/middle.mlir", "memory/after.mlir"))


def selected():
    path = os.environ.get("MERLIN_GRAPH_SHAPE_EPOCH_SELECTION")
    if not path:
        pytest.skip("explicit unaffected epoch proof/source selection unavailable")
    return load_selection(path)


def fresh_exports(modules, output, bits):
    from merlin.verify.lean_graph_loops_shape_selection import load_selection as load_shape

    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"stage{index}.mlir" for index in range(4)]
    for path, module in zip(paths, modules, strict=True):
        path.write_text(render(module))
    if os.environ.get("MERLIN_GRAPH_SHAPE_EPOCH_SELECTION"):
        context = selected()
    else:
        path = os.environ.get("MERLIN_GRAPH_SHAPE_SELECTION")
        if not path:
            pytest.skip("explicit original parser/provider selection unavailable")
        context = load_shape(path)
    captures, _ = _parse(context, output.resolve(), tuple(path.resolve() for path in paths), bits)
    return tuple(data for _, data in captures)


def project(modules, output, bits=64):
    return project_chain(*fresh_exports(modules, output, bits), index_bits=bits, max_nodes=5000)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_actual_complete_chain(case, tmp_path):
    result = verify_epoch_files(
        *case_paths(case),
        selection=selected(),
        output=tmp_path / "proof",
        index_bits=case.index_bits,
        max_nodes=5000,
        allocator_policy=ALLOCATOR_POLICY,
        descriptor_policy=DESCRIPTOR_POLICY,
        entry_policy=ENTRY_POLICY,
    )
    assert result["returncode"] == 0 and result["lean_complete_chain_checked"]
    assert REQUIRED_ROOTS <= set(result["transitive_axioms"])
    assert result["same_interpretation_exact_middle"] and result["all_finite_call_depths"]
    assert result["complete_return_and_caller_partition_observations"]
    assert not any(
        result[key]
        for key in (
            "ordinary_compiler_gating",
            "pipeline_receipt_issued",
            "independent_review_complete",
            "numerical_gates_discharged",
            "native_abi_hardware_fidelity_proved",
            "categorized_error_or_divergence_equivalence",
        )
    )
    model = json.loads(Path(result["model"]["path"]).read_bytes())
    assert len(model["initializer_links"]) == 2
    assert len(model["step_binding"]["steps"]) + 1 == len(model["memory"]["reuse"]["instructions"])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_full_actual_projection(case, tmp_path):
    model = project([parse(path.read_text()) for path in case_paths(case)], tmp_path, case.index_bits)
    (tmp_path / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    assert model["step_binding"]["finalClock"] == 2
    assert len(model["shape"]["changed_sites"]) == 10


MUTATIONS = (
    (1, "wrong_dimension"),
    (1, "missed_dimension"),
    (1, "missed_cast"),
    (1, "wrong_scalar_operand"),
    (1, "wrong_private_coordinate"),
    (2, "wrong_scalar_operand"),
    (2, "wrong_return"),
    (2, "wrong_free"),
    (3, "wrong_scalar_operand"),
    (3, "wrong_return"),
    (3, "wrong_free"),
    (3, "captured_initializer_coordinate"),
    (3, "dropped_second_initializer"),
    (3, "changed_observer"),
    (3, "extra_observer_store"),
)


def mutate(module, kind):
    functions = [op for op in module.walk() if isinstance(op, func.FuncOp)]
    kernel, touch = functions[:2]
    block = kernel.body.block
    constants = [op for op in block.ops if isinstance(op, arith.ConstantOp)]
    zero, one = constants[:2]
    alloc = next(op for op in block.ops if isinstance(op, memref.AllocOp))
    returned = next(op for op in block.ops if isinstance(op, func.ReturnOp))
    if kind == "wrong_dimension":
        constants[-1].properties["value"] = builtin.IntegerAttr.from_index_int_value(7)
    elif kind == "missed_dimension":
        block.insert_op_before(memref.DimOp(alloc.memref, zero.result), returned)
    elif kind == "missed_cast":
        block.insert_op_before(memref.CastOp.get(alloc.memref, alloc.memref.type), returned)
    elif kind == "wrong_scalar_operand":
        operation = next(op for op in block.ops if isinstance(op, arith.AddiOp))
        operation.operands = (operation.operands[0], block.args[1])
    elif kind == "wrong_private_coordinate":
        operation = next(op for op in block.ops if isinstance(op, memref.LoadOp))
        operation.operands = (*operation.operands[:-1], one.result)
    elif kind == "wrong_return":
        returned.operands = (block.args[0], returned.operands[1])
    elif kind == "wrong_free":
        operation = next(op for op in block.ops if isinstance(op, memref.DeallocOp))
        operation.operands = (block.args[-1],)
    elif kind == "captured_initializer_coordinate":
        loop = next(op for op in block.ops if isinstance(op, scf.ForOp))
        store = next(op for op in loop.walk() if isinstance(op, memref.StoreOp))
        store.operands = (*store.operands[:2], zero.result, store.operands[-1])
    elif kind == "dropped_second_initializer":
        loops = [op for op in block.ops if isinstance(op, scf.ForOp)]
        block.erase_op(loops[-1])
    elif kind == "tail_bound":
        minimum = next(op for op in kernel.walk() if isinstance(op, arith.MinUIOp))
        minimum.operands = (minimum.operands[0], one.result)
    elif kind == "changed_observer":
        operation = next(op for op in touch.walk() if isinstance(op, arith.AddiOp))
        operation.operands = (operation.operands[0], operation.operands[0])
    elif kind == "extra_observer_store":
        store = next(op for op in touch.walk() if isinstance(op, memref.StoreOp))
        touch.body.block.insert_op_before(store.clone(), touch.body.block.last_op)
    else:
        raise ValueError(kind)
    module.verify()


@pytest.mark.parametrize("stage,kind", MUTATIONS, ids=[f"stage{stage}_{kind}" for stage, kind in MUTATIONS])
def test_valid_changed_boundary_refused(tmp_path, stage, kind):
    modules = [parse(path.read_text()) for path in case_paths(CASES[0])]
    mutate(modules[stage], kind)
    for index, module in enumerate(modules):
        (tmp_path / f"stage{index}.mlir").write_text(render(module))
    with pytest.raises(ValueError):
        project(modules, tmp_path)


def test_exact_middle_from_independent_case_refused(tmp_path):
    original = [parse(path.read_text()) for path in case_paths(CASES[0])]
    foreign = [parse(path.read_text()) for path in case_paths(CASES[1])]
    shape, _ = fresh_exports(original, tmp_path / "original", 64)
    _, memory = fresh_exports(foreign, tmp_path / "foreign", 64)
    with pytest.raises(ValueError, match="complete normalized"):
        project_chain(shape, memory, index_bits=64, max_nodes=5000)


def test_tail_bound_is_refused_by_kernel_checker(tmp_path):
    paths = case_paths(CASES[0])
    module = parse(paths[3].read_text())
    mutate(module, "tail_bound")
    candidate = tmp_path / "wrong_tail.mlir"
    candidate.write_text(render(module))
    result = verify_epoch_files(
        *paths[:3],
        candidate,
        selection=selected(),
        output=tmp_path / "proof",
        index_bits=64,
        max_nodes=5000,
        allocator_policy=ALLOCATOR_POLICY,
        descriptor_policy=DESCRIPTOR_POLICY,
        entry_policy=ENTRY_POLICY,
    )
    assert result["returncode"] != 0 and not result["lean_complete_chain_checked"]
    assert "Lean refused" in result["refusal"]
    assert not result.get("context_disqualified_unknown_native_failure", False)


@pytest.mark.parametrize("budget", (0, 1, 20))
def test_syntax_budget_cannot_drop_unchecked_suffix(budget, tmp_path):
    modules = [parse(path.read_text()) for path in case_paths(CASES[0])]
    shape, memory = fresh_exports(modules, tmp_path, 64)
    with pytest.raises(ValueError):
        project_chain(shape, memory, index_bits=64, max_nodes=budget)
