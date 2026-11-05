"""Typed source custody, independent arithmetic and corrupted actual effects."""

from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i32

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
from merlin.compile.scheduling.shared_input_schedule import SharedInputEffect, schedule_shared_input
from merlin.compile.scheduling.shared_input_verification import verify_shared_input
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.dense_planning import DenseResources


def source(m=5, k=11, widths=(7, 3, 9), *, different_lhs=False, gap=False, epilogue=False):
    arguments = [f"%a: tensor<{m}x{k}xi8>", f"%other: tensor<{m}x{k}xi8>"]
    arguments += [f"%w{i}: tensor<{k}x{n}xi8>" for i, n in enumerate(widths)]
    operations = []
    for i, n in enumerate(widths):
        ty = f"tensor<{m}x{n}xi32>"
        lhs = "%other" if different_lhs and i else "%a"
        if gap and i:
            operations.append(f"%gap{i} = arith.constant 3 : i32")
        operations.append(f"""%zero{i} = arith.constant 0 : i32
          %init{i} = "tensor.splat"(%zero{i}) : (i32) -> {ty}
          %r{i} = linalg.generic {{
            indexing_maps = [affine_map<(m,n,k)->(m,k)>, affine_map<(m,n,k)->(k,n)>,
                             affine_map<(m,n,k)->(m,n)>],
            iterator_types = ["parallel", "parallel", "reduction"]}}
            ins({lhs}, %w{i} : tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>) outs(%init{i} : {ty}) {{
          ^bb0(%x: i8, %y: i8, %s: i32):
            %xx = arith.extsi %x : i8 to i32
            %yy = arith.extsi %y : i8 to i32
            %p = arith.muli %xx, %yy : i32
            %v = arith.addi %s, %p : i32
            linalg.yield %v : i32
          }} -> {ty}""")
    names = [f"%r{i}" for i in range(len(widths))]
    if epilogue:
        ty = f"tensor<{m}x{widths[0]}xi32>"
        operations.append(f"%post = arith.addi %r0, %r0 : {ty}")
        names[0] = "%post"
    result_types = [f"tensor<{m}x{n}xi32>" for n in widths]
    return parse_mlir_text(
        "module { func.func @siblings("
        + ", ".join(arguments)
        + ") -> ("
        + ", ".join(result_types)
        + ") {\n"
        + "\n".join(operations)
        + "\nreturn "
        + ", ".join(names)
        + " : "
        + ", ".join(result_types)
        + "\n}}"
    )


def fixture(*, shape=(5, 11, (7, 3, 9)), tile=(2, 3, 4), padding=0, **source_options):
    module = source(*shape, **source_options)
    (group,) = find_shared_input_groups(module)
    tm, tk, tn = tile
    kt = (group.k + tk - 1) // tk
    storage = DenseLoopStorage(
        DenseMemoryView("input", "input_sram", tk + padding, 2, 3 * kt * tm),
        DenseMemoryView("weights", "weights_sram", tn + padding, 3, 12 * tk),
        DenseMemoryView("results", "result_sram", 4 * tn + padding, 1, 12 * tm),
    )
    resources = DenseResources(
        storage.lhs.capacity * storage.lhs.unit_bytes + storage.rhs.capacity * storage.rhs.unit_bytes,
        storage.accumulator.capacity * storage.accumulator.unit_bytes,
        max_m_blocks=3,
        max_n_blocks=3,
        max_k_blocks=2,
        partial_output_tiles=True,
        partial_output_blocks=True,
    )
    return module, group, tile, storage, resources


def schedule(f):
    module, group, tile, storage, resources = f
    return schedule_shared_input(module, group, primitive_tile=tile, storage=storage, resources=resources)


def verify(effects, f, **kwargs):
    module, group, tile, storage, resources = f
    return verify_shared_input(
        effects, module=module, group=group, primitive_tile=tile, storage=storage, resources=resources, **kwargs
    )


def execute(effects, f, lhs, weights):
    """Execute actual byte addresses, independently of source-product checker."""
    _, group, _, storage, _ = f
    views = {"lhs": storage.lhs, "weight": storage.rhs, "accumulator": storage.accumulator}
    memory = {}
    result = [np.full((group.m, n), -(1 << 31), dtype=np.int64) for n in group.widths]
    resident, descriptor = None, None

    def address(view, row, col, width):
        return view.backing, row * view.unit_bytes + col * width

    for effect in effects:
        member, op = effect.member, effect.operation
        if type(op) is Load:
            matrix = lhs if member is None else weights[member]
            view = views[op.role]
            for r in range(op.rows):
                for c in range(op.cols):
                    memory[address(view, op.row + r, c, 1)] = int(matrix[op.dram_row + r, op.dram_col + c])
        elif type(op) is Preload:
            descriptor = op
            if op.weight_row is not None:
                resident = [
                    [memory[address(storage.rhs, op.weight_row + k, n, 1)] for n in range(op.cols)]
                    for k in range(op.weight_rows)
                ]
        elif type(op) is Compute:
            for r in range(op.rows):
                for n in range(descriptor.cols):
                    dst = address(storage.accumulator, descriptor.accumulator_row + r, n, 4)
                    value = memory[dst] if descriptor.accumulate else 0
                    for k in range(op.cols):
                        value += memory[address(storage.lhs, op.input_row + r, k, 1)] * resident[k][n]
                        value = (value + (1 << 31)) % (1 << 32) - (1 << 31)
                    memory[dst] = value
        elif type(op) is Store:
            for r in range(op.rows):
                for c in range(op.cols):
                    result[member][op.dram_row + r, op.dram_col + c] = memory[
                        address(storage.accumulator, op.accumulator_row + r, c, 4)
                    ]
    return result


@pytest.mark.parametrize("tile", [(2, 3, 4), (3, 2, 5), (4, 4, 3)])
@pytest.mark.parametrize("shape", [(1, 1, (1, 2)), (5, 11, (7, 3, 9)), (8, 9, (4, 10)), (7, 6, (11, 1, 7))])
@pytest.mark.parametrize("padding", [0, 7])
def test_complete_independent_results_and_real_cross_sibling_reuse(tile, shape, padding):
    f = fixture(shape=shape, tile=tile, padding=padding)
    before = str(f[0])
    uses = {v: tuple(v.uses) for op in f[0].walk() for v in (*op.results, *op.operands)}
    result = schedule(f)
    m, k, widths = shape
    rng = np.random.default_rng(m + k + sum(widths))
    lhs = rng.integers(-128, 128, (m, k), dtype=np.int64)
    weights = [rng.integers(-128, 128, (k, n), dtype=np.int64) for n in widths]
    for actual, weight in zip(execute(result.operations, f, lhs, weights), weights):
        expected = (lhs @ weight).astype(np.int32).astype(np.int64)
        np.testing.assert_array_equal(actual, expected)
    report = result.realization
    lhs_ops = [e.operation for e in result.operations if type(e.operation) is Load and e.member is None]
    assert sum(op.rows * op.cols for op in lhs_ops) == m * k
    assert report["lhs_loads"] == ((m + tile[0] - 1) // tile[0]) * ((k + tile[1] - 1) // tile[1])
    # Each independent sibling needs at least one complete input read, even
    # with ideal intra-contraction retention. Sharing removes that repetition.
    assert m * k * len(widths) > sum(op.rows * op.cols for op in lhs_ops)
    assert report["products"] == m * k * sum(widths)
    assert report["output_elements"] == m * sum(widths)
    assert report["machine_emission"] == "not_established"
    assert str(f[0]) == before
    assert all(tuple(value.uses) == original for value, original in uses.items())


@pytest.mark.parametrize("block", [(1, 1, 1), (2, 1, 3), (3, 2, 2)])
def test_explicit_phase_tuning_preserves_products_and_final_epilogue_boundary(block):
    f = fixture(epilogue=True)
    module, group, tile, storage, resources = f
    result = schedule_shared_input(
        module, group, primitive_tile=tile, storage=storage, resources=resources, selected_block=block
    )
    assert (result.block.m, result.block.k, result.block.n) == block
    assert result.realization["products"] == group.m * group.k * sum(group.widths)
    assert list(next(op for op in module.walk() if op.name == "func.func").body.block.ops)[-2].name == "arith.addi"
    assert group.record()["result_boundary"].startswith("every original")


def mutate(operations, kind, **changes):
    result = list(operations)
    index = next(i for i, e in enumerate(result) if type(e.operation) is kind)
    if changes.pop("omit", False):
        result.pop(index)
    elif "member" in changes:
        result[index] = replace(result[index], member=changes["member"])
    else:
        result[index] = replace(result[index], operation=replace(result[index].operation, **changes))
    return tuple(result)


@pytest.mark.parametrize(
    "kind,changes",
    [
        (Load, {"omit": True}),
        (Load, {"dram_col": 1}),
        (Load, {"row": 10000}),
        (Load, {"member": 0}),
        (Preload, {"weight_row": None}),
        (Preload, {"accumulate": True}),
        (Preload, {"member": 1}),
        (Preload, {"accumulator_row": 2}),
        (Compute, {"fresh_weights": False}),
        (Compute, {"input_block": (1, 0)}),
        (Compute, {"cols": 1}),
        (Compute, {"omit": True}),
        (Store, {"member": 1}),
        (Store, {"dram_col": 1}),
        (Store, {"omit": True}),
        (Store, {"readout": "changed"}),
    ],
)
def test_corrupted_actual_effects_cannot_reuse_the_plan_report(kind, changes):
    f = fixture()
    with pytest.raises(ValueError):
        verify(mutate(schedule(f).operations, kind, **changes), f)


def test_clobbered_full_input_stripe_and_cross_member_weights_are_rejected():
    f = fixture()
    result = schedule(f)
    effects = list(result.operations)
    compute = next(i for i, e in enumerate(effects) if type(e.operation) is Preload)
    first = next(e for e in effects if type(e.operation) is Load and e.member is None)
    effects.insert(compute, replace(first, operation=replace(first.operation, dram_row=2)))
    with pytest.raises(ValueError, match="geometry"):
        verify(tuple(effects), f)
    weights = next(i for i, e in enumerate(result.operations) if type(e.operation) is Load and e.member is not None)
    effects = list(result.operations)
    effects[weights] = replace(effects[weights], member=1)
    with pytest.raises(ValueError, match="identity|original tensor"):
        verify(tuple(effects), f)


@pytest.mark.parametrize("option", [{"different_lhs": True}, {"gap": True}])
def test_equal_shapes_or_unknown_intervening_operations_do_not_establish_reuse(option):
    module = source(widths=(7, 3), **option)
    assert find_shared_input_groups(module) == ()


@pytest.mark.parametrize("change", ["constant", "unsigned", "effect", "group"])
def test_modified_source_semantics_or_forged_source_binding_requires_replanning(change):
    f = list(fixture())
    module, group = f[:2]
    if change == "constant":
        next(op for op in module.walk() if op.name == "arith.constant").properties["value"] = IntegerAttr(1, i32)
    elif change == "unsigned":
        f[0] = parse_mlir_text(str(module).replace("arith.extsi", "arith.extui"))
    elif change == "effect":
        next(op for op in module.walk() if op.name == "linalg.generic").attributes["unknown"] = StringAttr("write")
    else:
        f[1] = replace(group, widths=(1, 1))
    with pytest.raises(ValueError, match="current typed source"):
        schedule(f)


def test_missing_capacity_partial_support_and_actual_aliases_decline_or_refuse():
    f = list(fixture())
    f[3] = replace(f[3], lhs=replace(f[3].lhs, capacity=1))
    assert schedule(f) is None
    f = list(fixture())
    f[4] = replace(f[4], partial_output_tiles=False)
    assert schedule(f) is None
    f = list(fixture())
    good = schedule(f)
    f[3] = replace(f[3], rhs=replace(f[3].rhs, backing=f[3].lhs.backing))
    assert schedule(f) is None
    with pytest.raises(ValueError, match="aliases"):
        verify(good.operations, f)


def test_explicit_search_emission_and_checking_budgets_fail_closed():
    module, group, tile, storage, resources = f = fixture()
    arguments = dict(primitive_tile=tile, resources=resources, storage=storage)
    with pytest.raises(ValueError, match="candidate budget"):
        schedule_shared_input(module, group, **{**arguments, "resources": replace(resources, candidate_budget=1)})
    with pytest.raises(ValueError, match="command budget"):
        schedule_shared_input(module, group, **arguments, max_commands=1)
    with pytest.raises(ValueError, match="row-checking budget"):
        verify(schedule(f).operations, f, max_rows_visited=1)
    assert schedule_shared_input(module, group, **arguments, selected_block=(99, 1, 1)) is None
    with pytest.raises(ValueError, match="unknown sibling"):
        verify((*schedule(f).operations, SharedInputEffect(0, object())), f)


def test_signed_modular_overflow_preserves_each_original_result():
    f = fixture(shape=(1, 140_000, (1, 2)), tile=(1, 257, 1))
    result = schedule(f)
    lhs = np.full((1, 140_000), -128, dtype=np.int64)
    weights = [np.full((140_000, n), -128, dtype=np.int64) for n in (1, 2)]
    for actual, weight in zip(execute(result.operations, f, lhs, weights), weights):
        expected = (lhs @ weight).astype(np.int32).astype(np.int64)
        assert expected[0, 0] < 0  # The exact source recurrence wraps i32.
        np.testing.assert_array_equal(actual, expected)
