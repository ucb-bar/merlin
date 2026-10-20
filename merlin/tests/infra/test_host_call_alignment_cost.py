"""Bounded pure-SSA alignment proofs; no compiler or target execution."""

from xdsl.dialects.builtin import IntegerAttr, i64

from merlin.targetgen.rocc.decode import _parse_module
from merlin.verify import host_call_alignment as H


def _dag(depth, *, fibonacci=False):
    rows = ["%scale = llvm.mlir.constant(64) : i64", "%v0 = llvm.mul %input, %scale : i64"]
    for index in range(1, depth + 1):
        right = index - 2 if fibonacci and index > 1 else index - 1
        rows.append(f"%v{index} = llvm.add %v{index - 1}, %v{right} : i64")
    module = _parse_module("module { llvm.func @test(%input: i64) {\n" + "\n".join(rows) + "\nllvm.return }}")
    values = [op.results[0] for op in module.walk() if op.name == "llvm.add"]
    return module, values[-1]


def test_repeated_identical_ssa_paths_share_a_proof(monkeypatch):
    _, value = _dag(64)
    original, calls = H.constant_integer, []

    def observed(item):
        calls.append(item)
        return original(item)

    monkeypatch.setattr(H, "constant_integer", observed)
    assert H.value_alignment(value) == 64
    assert len(calls) == len(set(calls)) <= 68


def test_distinct_dag_paths_exhaust_conservatively_without_repeated_constant_reads(monkeypatch):
    _, value = _dag(64, fibonacci=True)
    original, calls = H.constant_integer, []

    def observed(item):
        calls.append(item)
        return original(item)

    monkeypatch.setattr(H, "constant_integer", observed)
    assert H.value_alignment(value) == 1
    assert len(calls) == len(set(calls)) <= 68


def test_effort_exhaustion_discards_the_entire_proof(monkeypatch):
    module, value = _dag(5)
    function = next(op for op in module.walk() if op.name == "llvm.func")
    constant = next(op.results[0] for op in module.walk() if op.name == "llvm.mlir.constant")
    # This AND could otherwise retain 64 from its constant operand after an
    # incomplete subproof. Exhaustion must abandon the whole top-level proof.
    from xdsl.dialects.llvm import AndOp

    result = AndOp(constant, value)
    function.body.block.insert_op_before(result, function.body.block.last_op)
    monkeypatch.setattr(H, "_ALIGNMENT_STATE_LIMIT", 4)
    assert H.value_alignment(result.results[0]) == 1


def test_active_path_and_depth_limit_are_not_replaced_by_a_shallow_cached_proof():
    rows = ["%scale = llvm.mlir.constant(64) : i64", "%shared = llvm.and %input, %scale : i64"]
    previous = "%shared"
    for index in range(126):
        rows.append(f"%deep{index} = llvm.add {previous}, {previous} : i64")
        previous = f"%deep{index}"
    rows.append(f"%result = llvm.add %shared, {previous} : i64")
    module = _parse_module("module { llvm.func @test(%input: i64) {\n" + "\n".join(rows) + "\nllvm.return }}")
    result = [op.results[0] for op in module.walk() if op.name == "llvm.add"][-1]
    shared = next(op.results[0] for op in module.walk() if op.name == "llvm.and")
    assert H.value_alignment(shared) == 64
    assert H.value_alignment(result) == 1
    assert H.value_alignment(shared, frozenset({shared})) == 1


def test_memo_is_fresh_after_source_mutation_and_between_active_contexts():
    module, value = _dag(1)
    constant = next(op for op in module.walk() if op.name == "llvm.mlir.constant")
    assert H.value_alignment(value) == 64
    assert H.value_alignment(value, frozenset({value})) == 1
    constant.properties["value"] = IntegerAttr(3, i64)
    assert H.value_alignment(value) == 1


def test_cycles_and_constant_reader_recursion_fail_conservatively(monkeypatch):
    module, value = _dag(1)
    argument = next(op for op in module.walk() if op.name == "llvm.func").body.block.args[0]
    value.owner.operands = (value, argument)
    assert H.value_alignment(value) == 1

    def overflow(_value):
        raise RecursionError("synthetic deep source")

    monkeypatch.setattr(H, "constant_integer", overflow)
    assert H.value_alignment(value) == 1
