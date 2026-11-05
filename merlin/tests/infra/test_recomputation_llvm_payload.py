"""Typed submitted-LLVM payload witnesses for the neutral recomputation fixture.

These checks prove constant-loop scalar instruction payload and exact fixture
coordinates. They do not count native CPU, allocator, cache or bus traffic.
"""

import numpy as np
import pytest
from test_multi_output_source import CASES, OUTPUT_NAMES, source
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp, UnitAttr, UnregisteredOp, i1, i32, i64
from xdsl.ir import Block, Region

from merlin.common.digest import sha256_file
from merlin.perf.host_cfg_activity import _real_name, analyze_host_cfg_activity
from merlin.perf.host_cfg_index import prepare_host_cfg
from merlin.verify.host_call_alignment import import_host_entry, revalidate_host_entry
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer


def descriptor_roster(case):
    """Original tensor ABI, not LLVM SSA names or a selected-plan annotation."""
    entry = source(case).body.block.first_op
    types = (*entry.function_type.inputs, *entry.function_type.outputs)
    inputs = ("weight0", "weight1", "input", "prior_state", "prior_position")
    roles = (*inputs, *("echo_" + name if name in inputs else name for name in OUTPUT_NAMES))
    assert len(types) == len(roles) == 14
    rows, fields = {}, []
    for role, ty in zip(roles, types, strict=True):
        shape = tuple(ty.get_shape())
        start = len(fields)
        fields.extend((llvm.LLVMPointerType(), llvm.LLVMPointerType(), *([i64] * (1 + 2 * len(shape)))))
        rows[role] = dict(
            start=start,
            aligned=start + 1,
            offset=start + 2,
            sizes=tuple(range(start + 3, start + 3 + len(shape))),
            shape=shape,
            bits=ty.element_type.width.data,
        )
    return rows, tuple(fields)


def verify_buffer_activity(function, case, *, candidate):
    """A constrained fixture witness over actual submitted LLVM, not native bus traffic."""
    function.verify()
    cfg = prepare_host_cfg(function)
    activity = analyze_host_cfg_activity(function, prepared_cfg=cfg)
    assert activity["status"] == "derived", activity["problems"]
    roster, fields = descriptor_roster(case)
    entry = cfg.blocks[0]
    assert tuple(v.type for v in entry.args) == fields
    aliases = {row["aligned"]: role for role, row in roster.items()}
    elements, width = int(np.prod(case.shape)), case.width
    nbytes = elements * width
    repeated = 2 if candidate else 1
    expected = {
        "weight0": (repeated * nbytes, 0),
        "weight1": (2 * nbytes, 0),
        "input": (repeated * nbytes, 0),
        "prior_state": (repeated * nbytes, 0),
        "prior_position": (4 * (elements + 1), 0),
        "first": (nbytes, nbytes),
        "second": (nbytes, nbytes),
        "state": (0 if candidate else 2 * nbytes, nbytes if candidate else 2 * nbytes),
        "position": (0, 4),
    }
    actual = activity["buffer_payload"]
    assert set(actual) == {f"arg:{roster[role]['aligned']}" for role in expected}
    for role, (reads, writes) in expected.items():
        row = actual[f"arg:{roster[role]['aligned']}"]
        assert (row["load_payload_bytes"], row["store_payload_bytes"]) == (reads, writes)
        assert (row["load_operations"], row["store_operations"]) == (
            reads // (roster[role]["bits"] // 8),
            writes // (roster[role]["bits"] // 8),
        )

    loops = activity["loops"]
    axes = {}
    for row in loops:
        assert (row["initial"], row["step"], row["predicate"]) == (0, 1, "slt")
        assert row["bound"] in case.shape and row["trip_count"] == row["bound"]
        value = cfg.blocks[row["header_block"]].args[row["induction_argument_index"]]
        axes[value] = (case.shape.index(row["bound"]), {row["header_block"], *row["body_blocks"]})
    block_index = {block: i for i, block in enumerate(cfg.blocks)}

    def integer(value, block, depth=0):
        assert depth < 64
        constant = constant_integer(value)
        if constant is not None:
            return constant
        if value in axes:
            axis, blocks = axes[value]
            assert block_index[block] in blocks
            return ("axis", axis)
        if value in entry.args:
            return ("abi", value.index)
        op = value.owner
        assert not isinstance(op, Block) and op.name in ("llvm.add", "llvm.mul")
        left, right = (integer(v, block, depth + 1) for v in op.operands)
        if isinstance(left, int) and isinstance(right, int):
            return left + right if op.name == "llvm.add" else left * right
        if op.name == "llvm.mul":
            if left == 1:
                return right
            if right == 1:
                return left
        return (op.name, left, right)

    offset = ("llvm.add", ("llvm.mul", ("axis", 0), case.shape[1]), ("axis", 1))

    def pointer(value, block, *, copying=False):
        op = value.owner
        if value in entry.args:
            assert value.index in aliases
            role = aliases[value.index]
            assert roster[role]["shape"] == () and not copying
            return role
        assert not isinstance(op, Block) and op.name == "llvm.getelementptr" and len(op.operands) == 2
        base, index = op.operands
        assert base in entry.args and base.index in aliases
        role = aliases[base.index]
        assert str(op.properties["elem_type"]) == f"i{roster[role]['bits']}"
        assert tuple(op.properties["rawConstantIndices"].get_values()) == (llvm.GEP_USE_SSA_VAL,)
        expected_index = ("abi", roster[role]["offset"]) if copying else offset
        assert integer(index, block) == expected_index
        return role

    def scalar(value, block, depth=0):
        assert depth < 32
        constant = constant_integer(value)
        if constant is not None:
            return ("constant", value.type.width.data, constant)
        op = value.owner
        assert not isinstance(op, Block) and op.parent is block
        bits = value.type.width.data
        if op.name == "llvm.load":
            return ("load", bits, pointer(op.operands[0], block))
        if op.name in ("llvm.add", "llvm.mul"):
            assert op.properties["overflowFlags"].value.data == 0
            return (op.name, bits, *(scalar(v, block, depth + 1) for v in op.operands))
        if op.name == "llvm.trunc":
            assert not op.attributes and set(op.properties) <= {"overflowFlags"}
            flags = op.properties.get("overflowFlags")
            assert (
                flags is None
                or type(flags) is IntegerAttr
                and flags.value.data == 0
                or type(flags) is llvm.OverflowAttr
                and not flags.data
            )
            return ("llvm.trunc", bits, scalar(op.operands[0], block, depth + 1))
        raise AssertionError("unsupported scalar expression")

    bits = 8 * width

    def load(role):
        return ("load", roster[role]["bits"], role)

    p = ("llvm.add", bits, ("llvm.mul", bits, load("input"), load("weight0")), load("prior_state"))
    producer = p if candidate else load("state")
    expected_stores = ([] if candidate else [("state", p)]) + [
        ("first", ("llvm.add", bits, producer, load("weight1"))),
        ("second", ("llvm.mul", bits, producer, load("weight1"))),
        (
            "state",
            (
                "llvm.add",
                bits,
                ("llvm.add", bits, load("first"), load("second")),
                ("llvm.trunc", bits, load("prior_position")),
            ),
        ),
        ("position", ("llvm.add", 32, load("prior_position"), ("constant", 32, 1))),
    ]
    stores, copies, store_operations = [], [], []
    allowed = {
        "llvm.func",
        "llvm.mlir.constant",
        "llvm.br",
        "llvm.cond_br",
        "llvm.icmp",
        "llvm.add",
        "llvm.mul",
        "llvm.trunc",
        "llvm.getelementptr",
        "llvm.load",
        "llvm.store",
        "llvm.return",
        "llvm.intr.memcpy",
    }
    for op in function.walk():
        name = _real_name(op)
        assert name in allowed
        if name == "llvm.load":
            assert not op.attributes and set(op.properties) == {"alignment", "ordering"}
            assert op.properties["ordering"].value.data == 0
            role = pointer(op.operands[0], op.parent)
            assert op.results[0].type.width.data == roster[role]["bits"]
            alignment = op.properties["alignment"].value.data
            assert 0 < alignment <= roster[role]["bits"] // 8 and not alignment & (alignment - 1)
        elif name == "llvm.store":
            assert not op.attributes and set(op.properties) == {"alignment", "ordering"}
            assert op.properties["ordering"].value.data == 0
            alignment = op.properties["alignment"].value.data
            assert 0 < alignment <= op.operands[0].type.width.data // 8 and not alignment & (alignment - 1)
            stores.append((pointer(op.operands[1], op.parent), scalar(op.operands[0], op.parent)))
            store_operations.append(op)
        elif name == "llvm.intr.memcpy":
            assert len(op.operands) == 3 and op.properties["isVolatile"].value.data == 0
            dst, src = (pointer(value, op.parent, copying=True) for value in op.operands[:2])
            sizes = roster[src]["sizes"]
            length = roster[src]["bits"] // 8
            # Lowering multiplies dimensions before the element width.
            if sizes:
                length = ("abi", sizes[0])
                for size in sizes[1:]:
                    length = ("llvm.mul", length, ("abi", size))
                if roster[src]["bits"] != 8:
                    length = ("llvm.mul", length, roster[src]["bits"] // 8)
            assert integer(op.operands[2], op.parent) == length
            copies.append((dst, src))
    assert stores == expected_stores
    assert copies == [
        ("echo_" + role, role) for role in ("input", "weight0", "weight1", "prior_state", "prior_position")
    ]
    for previous, following in zip(store_operations, store_operations[1:]):
        outer = [
            row for row in loops if row["bound"] == case.shape[0] and block_index[previous.parent] in row["body_blocks"]
        ]
        assert len(outer) == 1
        nodes = {outer[0]["header_block"], *outer[0]["body_blocks"]}
        header = cfg.blocks[outer[0]["header_block"]]
        exits = [block for block in header.last_op.successors if block_index[block] not in nodes]
        assert len(exits) == 1 and cfg.dominance.dominates(exits[0], following.parent)
    return dict(
        activity=activity,
        descriptor_roster=roster,
        store_dags=stores,
        copies=copies,
        scope="typed submitted scalar LLVM, proven constant-trip loops, exact same-coordinate load/store "
        "DAGs and ordered complete loop phases; not post-optimization machine or bus traffic",
    )


def inspect_buffer_activity(result, case, *, candidate, diagnostics):
    model_object = result.host_so.with_suffix(".o")
    artifacts = {
        role: dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha256_file(path))
        for role, path in (("llvm_ir", result.ll_path), ("object", model_object))
    }
    imported, observation = import_host_entry(artifacts, diagnostics=diagnostics, max_ir_bytes=1 << 20)
    functions = [op for op in imported.body.block.ops if op.name == "llvm.func" and op.sym_name.data == "forward"]
    assert len(functions) == 1
    witness = verify_buffer_activity(functions[0], case, candidate=candidate)
    revalidate_host_entry(observation)
    return dict(**witness, import_observation=observation)


def activity_fixture(case, *, candidate):
    """Typed LLVM controls for this test witness; no compiler or native execution."""
    roster, fields = descriptor_roster(case)
    entry = Block(arg_types=fields)
    region = Region(entry)
    elem = IntegerType(8 * case.width)

    def constant(value, ty=i64):
        operation = llvm.ConstantOp(IntegerAttr(value, ty), ty)
        entry.add_op(operation)
        return operation.results[0]

    zero, one, m, n, width, scalar_one, four = (
        constant(value, ty)
        for value, ty in (
            (0, i64),
            (1, i64),
            (case.shape[0], i64),
            (case.shape[1], i64),
            (case.width, i64),
            (1, i32),
            (4, i64),
        )
    )

    def append(block, operation):
        block.add_op(operation)
        return operation.results[0]

    def address(block, role, index):
        return append(
            block,
            llvm.GEPOp.from_mixed_indices(
                entry.args[roster[role]["aligned"]], [index], IntegerType(roster[role]["bits"])
            ),
        )

    def load(block, role, index):
        pointer = address(block, role, index) if roster[role]["shape"] else entry.args[roster[role]["aligned"]]
        return append(block, llvm.LoadOp(pointer, IntegerType(roster[role]["bits"]), alignment=1))

    def produce(block, index):
        product = append(block, llvm.MulOp(load(block, "input", index), load(block, "weight0", index)))
        return append(block, llvm.AddOp(product, load(block, "prior_state", index)))

    def body(block, index, role):
        if role in ("first", "second"):
            p = produce(block, index) if candidate else load(block, "state", index)
            operation = llvm.AddOp if role == "first" else llvm.MulOp
            value = append(block, operation(p, load(block, "weight1", index)))
        elif role == "producer":
            role, value = "state", produce(block, index)
        else:
            total = append(block, llvm.AddOp(load(block, "first", index), load(block, "second", index)))
            position = append(block, llvm.TruncOp(load(block, "prior_position", index), elem))
            value = append(block, llvm.AddOp(total, position))
        block.add_op(llvm.StoreOp(value, address(block, role, index), alignment=1))

    start = entry
    for role in (() if candidate else ("producer",)) + ("first", "second", "state"):
        outer, pre, inner, work, latch, after = (
            Block(arg_types=[i64]),
            Block(),
            Block(arg_types=[i64]),
            Block(),
            Block(),
            Block(),
        )
        region.add_block([outer, pre, inner, work, latch, after])
        start.add_op(llvm.BrOp(outer, zero))
        test = append(outer, llvm.ICmpOp(outer.args[0], m, IntegerAttr(2, i64)))
        outer.add_op(llvm.CondBrOp(test, pre, [], after, []))
        pre.add_op(llvm.BrOp(inner, zero))
        test = append(inner, llvm.ICmpOp(inner.args[0], n, IntegerAttr(2, i64)))
        inner.add_op(llvm.CondBrOp(test, work, [], latch, []))
        product = append(work, llvm.MulOp(outer.args[0], n))
        index = append(work, llvm.AddOp(product, inner.args[0]))
        body(work, index, role)
        update = append(work, llvm.AddOp(inner.args[0], one))
        work.add_op(llvm.BrOp(inner, update))
        update = append(latch, llvm.AddOp(outer.args[0], one))
        latch.add_op(llvm.BrOp(outer, update))
        start = after
    position = append(start, llvm.AddOp(load(start, "prior_position", zero), scalar_one))
    start.add_op(llvm.StoreOp(position, entry.args[roster["position"]["aligned"]], alignment=1))
    for role in ("input", "weight0", "weight1", "prior_state", "prior_position"):
        src = address(start, role, entry.args[roster[role]["offset"]])
        dst = address(start, "echo_" + role, entry.args[roster["echo_" + role]["offset"]])
        length = width if roster[role]["shape"] else four
        if roster[role]["shape"]:
            sizes = [entry.args[i] for i in roster[role]["sizes"]]
            length = append(start, llvm.MulOp(sizes[0], sizes[1]))
            if case.width != 1:
                length = append(start, llvm.MulOp(length, width))
        start.add_op(
            UnregisteredOp.with_name("llvm.intr.memcpy").create(
                operands=[dst, src, length], properties={"isVolatile": IntegerAttr(0, i1)}
            )
        )
    start.add_op(llvm.ReturnOp())
    function = llvm.FuncOp("forward", llvm.LLVMFunctionType(fields), body=region)
    ModuleOp([function]).verify()
    return function


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("candidate", (False, True))
def test_typed_buffer_witness_checks_complete_matrix_and_source_dags(case, candidate):
    result = verify_buffer_activity(activity_fixture(case, candidate=candidate), case, candidate=candidate)
    assert result["activity"]["status"] == "derived" and len(result["copies"]) == 5


@pytest.mark.parametrize(
    "damage",
    (
        "extra_read",
        "extra_store",
        "wrong_root",
        "wrong_offset",
        "wrong_scalar",
        "unknown_control",
        "wrong_copy_length",
        "wrong_phase_order",
        "observing_store",
        "unqualified_alignment",
        "poison_narrowing",
    ),
)
def test_typed_buffer_witness_rejects_actual_llvm_mutations(damage):
    case = CASES[0]
    function = activity_fixture(case, candidate=False)
    verify_buffer_activity(function, case, candidate=False)
    operations = tuple(function.walk())
    load = next(op for op in operations if op.name == "llvm.load")
    store = next(op for op in operations if op.name == "llvm.store")
    if damage == "extra_read":
        load.parent.insert_op_before(llvm.LoadOp(load.operands[0], load.results[0].type), load)
    elif damage == "extra_store":
        store.parent.insert_op_before(llvm.StoreOp(*store.operands), store)
    elif damage == "wrong_root":
        gep = load.operands[0].owner
        roster, _ = descriptor_roster(case)
        gep.operands = (function.body.blocks[0].args[roster["weight1"]["aligned"]], *gep.operands[1:])
    elif damage == "wrong_offset":
        gep = load.operands[0].owner
        constant = next(
            op for op in operations if op.name == "llvm.mlir.constant" and constant_integer(op.results[0]) == 1
        )
        changed = llvm.AddOp(gep.operands[1], constant.results[0])
        gep.parent.insert_op_before(changed, gep)
        gep.operands = (gep.operands[0], changed.results[0])
    elif damage == "wrong_scalar":
        product = next(op for op in operations if op.name == "llvm.mul" and op.results[0].type == IntegerType(8))
        changed = llvm.AddOp(*product.operands)
        product.parent.insert_op_before(changed, product)
        product.results[0].replace_all_uses_with(changed.results[0])
        product.parent.erase_op(product)
    elif damage == "unknown_control":
        comparison = next(op for op in operations if op.name == "llvm.icmp")
        roster, _ = descriptor_roster(case)
        comparison.operands = (comparison.operands[0], function.body.blocks[0].args[roster["weight0"]["sizes"][0]])
    elif damage == "wrong_copy_length":
        copy = next(op for op in operations if _real_name(op) == "llvm.intr.memcpy")
        constant = next(
            op for op in operations if op.name == "llvm.mlir.constant" and constant_integer(op.results[0]) == 1
        )
        copy.operands = (*copy.operands[:2], constant.results[0])
    elif damage == "observing_store":
        store.properties["volatile_"] = UnitAttr()
    elif damage == "unqualified_alignment":
        load.properties["alignment"] = IntegerAttr(8, i64)
    elif damage == "poison_narrowing":
        narrowing = next(op for op in operations if op.name == "llvm.trunc")
        narrowing.properties["overflowFlags"] = llvm.OverflowAttr((llvm.OverflowFlag.NO_SIGNED_WRAP,))
    else:
        blocks = tuple(function.body.blocks)
        outer = [row for row in analyze_host_cfg_activity(function)["loops"] if row["bound"] == case.shape[0]]
        first, second, third = (blocks[row["header_block"]] for row in outer[:3])
        first_exit, second_exit = first.last_op.successors[1], second.last_op.successors[1]
        blocks[0].last_op.successors = (second,)
        second_exit.last_op.successors = (first,)
        first_exit.last_op.successors = (third,)
        assert analyze_host_cfg_activity(function)["status"] == "derived"
    with pytest.raises(AssertionError):
        verify_buffer_activity(function, case, candidate=False)
