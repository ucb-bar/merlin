"""Exact ordered source recomputation and complete-module mutation controls."""

from dataclasses import replace

import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, tensor
from xdsl.parser import Parser

from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline_state import PipelineContext
from merlin.transforms.recomputation import (
    RecomputationBudgets,
    RecomputationPermissions,
    plan_source_recomputation,
)
from merlin.transforms.recomputation_check import verify_source_recomputation


def permissions(**changes):
    return RecomputationPermissions(
        "original-total-integer-region",
        **{
            "repeat_evaluation_no_faults": True,
            "repeat_evaluation_unobserved": True,
            "suppressed_materialization_no_faults": True,
            "suppressed_materialization_unobserved": True,
            "inputs_stable_during_region": True,
            **changes,
        },
    )


def budgets(**changes):
    return RecomputationBudgets(
        **{
            "max_candidates": 8,
            "max_source_elements": 1 << 16,
            "max_repeats": 2,
            "max_producer_scalar_ops": 16,
            "max_added_scalar_evaluations": 1 << 16,
            "max_constructed_operations": 1024,
            "max_scalar_ir_bytes": 65536,
            **changes,
        }
    )


def source(width=8, shape=(3, 7)):
    typ = f"tensor<{'x'.join(map(str, shape))}xi{width}>"
    scalar = f"i{width}"
    dims = ",".join(f"d{i}" for i in range(len(shape)))
    mapping = f"affine_map<({dims})->({dims})>"
    text = f"""builtin.module {{func.func @forward(%x:{typ},%state:{typ},%w0:{typ},%w1:{typ})
    ->({typ},{typ},{typ},{typ}) {{
    %pinit=tensor.empty():{typ}
    %e0init=tensor.empty():{typ}
    %e1init=tensor.empty():{typ}
    """
    for i, (inputs, body) in enumerate(
        (
            ("%x,%state", f"%s=arith.addi %a,%b:{scalar}\n%v=arith.muli %s,%a:{scalar}"),
            ("%p,%w0", f"%v=arith.subi %a,%b:{scalar}"),
            ("%p,%w1", f"%v=arith.addi %a,%b:{scalar}"),
        )
    ):
        out = "%pinit" if i == 0 else f"%e{i - 1}init"
        result = "%p" if i == 0 else f"%e{i - 1}"
        text += f"""{result}="linalg.generic"({inputs},{out}) <{{indexing_maps=[{mapping},{mapping},{mapping}],
        iterator_types=[{",".join(["#linalg.iterator_type<parallel>"] * len(shape))}],
        operandSegmentSizes=array<i32:2,1>}}> ({{^bb0(%a:{scalar},%b:{scalar},%init:{scalar}):
        {body}
        "linalg.yield"(%v):({scalar})->()}}):({typ},{typ},{typ})->{typ}\n"""
    text += f"func.return %e0,%e1,%x,%state:{typ},{typ},{typ},{typ} }} }}"
    context = Context()
    for dialect in (arith.Arith, builtin.Builtin, func.Func, linalg.Linalg, tensor.Tensor):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    return module


def generics(module):
    return tuple(op for op in module.walk() if op.name == "linalg.generic")


def proposal(module=None, **kwargs):
    return plan_source_recomputation(
        module if module is not None else source(),
        PipelineContext(numerical_policy="modular-integer"),
        permissions(),
        budgets(**kwargs),
    )


@pytest.mark.parametrize("width,shape", [(8, (3, 7)), (16, (2, 11)), (32, (1, 5))])
def test_checked_candidate_removes_p_and_counts_all_ordered_work(width, shape):
    module = source(width, shape)
    digest = ir_digest(module)
    plan = proposal(module)
    assert plan is not None
    assert plan.source is module and ir_digest(module) == digest
    assert len(generics(module)) == 3 and len(generics(plan.candidate)) == 2
    assert sum(op.name == "tensor.empty" for op in plan.candidate.walk()) == 2
    record = plan.record()
    n = shape[0] * shape[1]
    proof = record["conformance"]
    assert (proof["before_scalar_evaluations"], proof["after_scalar_evaluations"]) == (4 * n, 6 * n)
    assert (proof["before_producer_evaluations"], proof["after_producer_evaluations"]) == (n, 2 * n)
    assert proof["logical_producer_bytes_removed"] == n * width // 8
    assert record["constructed_operations"] > len(tuple(plan.candidate.walk()))
    assert record["status"] == "LEGAL_CANDIDATE" and record["cost"] == "UNKNOWN"
    assert plan.stamp.source_sha256 == digest and plan.stamp.ir_level == "linalg"
    assert plan.baseline_selection_sha256 != plan.candidate_selection_sha256
    plan.revalidate()


@pytest.mark.parametrize(
    "flag",
    [
        "repeat_evaluation_no_faults",
        "repeat_evaluation_unobserved",
        "suppressed_materialization_no_faults",
        "suppressed_materialization_unobserved",
        "inputs_stable_during_region",
    ],
)
def test_missing_original_permission_preserves_source(flag):
    module = source()
    before = ir_digest(module)
    assert plan_source_recomputation(module, PipelineContext(), permissions(**{flag: False}), budgets()) is None
    assert ir_digest(module) == before


@pytest.mark.parametrize(
    "change",
    [
        {"max_repeats": 1},
        {"max_added_scalar_evaluations": 1},
        {"max_constructed_operations": 10},
        {"max_source_elements": 1},
        {"max_producer_scalar_ops": 1},
    ],
)
def test_bounded_construction_declines_without_changing_original(change):
    module = source()
    before = ir_digest(module)
    assert proposal(module, **change) is None
    assert ir_digest(module) == before


def test_plan_ids_are_deterministic_across_reparsed_current_ssa():
    one, two = proposal(), proposal()
    assert one.baseline_selection_sha256 == two.baseline_selection_sha256
    assert one.candidate_selection_sha256 == two.candidate_selection_sha256


@pytest.mark.parametrize("which", ["source", "candidate"])
def test_late_equal_text_function_clone_is_stale(which):
    plan = proposal()
    module = getattr(plan, which)
    function = module.body.block.first_op
    module.body.block.insert_op_before(function.clone(), function)
    module.body.block.erase_op(function)
    assert ir_digest(module) == (
        plan.stamp.source_sha256 if which == "source" else plan.record.__self__._record["candidate_sha256"]
    )
    with pytest.raises(ValueError, match="SSA|source|candidate"):
        plan.revalidate()


@pytest.mark.parametrize("change", ["input", "wrap", "missing_output", "extra_work", "order"])
def test_independent_checker_refuses_actual_candidate_mutations(change):
    module = source()
    plan = proposal(module)
    candidate = plan.candidate.clone()
    first, second = generics(candidate)
    if change == "input":
        operands = list(first.operands)
        operands[0], operands[1] = operands[1], operands[0]
        first.operands = operands
    elif change == "wrap":
        mul = next(op for op in first.body.block.ops if type(op) is arith.MuliOp)
        mul.properties["overflowFlags"] = arith.IntegerOverflowAttr([arith.IntegerOverflowFlag.NSW])
    elif change == "missing_output":
        returned = next(op for op in candidate.walk() if op.name == "func.return")
        returned.operands = (second.results[0], *tuple(returned.operands)[1:])
    elif change == "extra_work":
        op = next(iter(first.body.block.ops))
        first.body.block.insert_op_before(op.clone(), first.body.block.last_op)
    else:
        body = first.body.block
        op = body.first_op
        # A dead repeated application still changes exact ordered multiplicity.
        body.insert_op_before(op.clone(), body.first_op)
    with pytest.raises(Exception):
        verify_source_recomputation(module, candidate, generics(module), permissions(), budgets())


def test_cloned_plan_is_not_issued_and_records_are_detached():
    plan = proposal()
    record = plan.record()
    record["conformance"]["after_scalar_evaluations"] = 0
    assert plan.record()["conformance"]["after_scalar_evaluations"] > 0
    with pytest.raises(ValueError, match="issued"):
        replace(plan).revalidate()


@pytest.mark.parametrize(
    "key,value",
    [("max_repeats", 3), ("max_candidates", True), ("max_added_scalar_evaluations", 0), ("max_scalar_ir_bytes", -1)],
)
def test_invalid_budgets_are_refused(key, value):
    with pytest.raises(ValueError):
        budgets(**{key: value})


def test_original_context_and_permissions_remain_current():
    module, context, original = source(), PipelineContext(numerical_policy="modular"), permissions()
    plan = plan_source_recomputation(module, context, original, budgets())
    object.__setattr__(context, "numerical_policy", "other")
    with pytest.raises(ValueError, match="context"):
        plan.revalidate()


def test_missing_required_permissions_are_not_default_true():
    with pytest.raises(TypeError):
        RecomputationPermissions("identity-only")
    with pytest.raises(ValueError):
        permissions(repeat_evaluation_unobserved=1)


def test_opaque_boundaries_and_observers_remain_complete():
    module = source()
    function = module.body.block.first_op
    barrier = func.FuncOp.external("opaque", [], [])
    module.body.block.add_op(barrier)
    module.body.block.add_op(func.FuncOp.external("different", [], []))
    call = func.CallOp("opaque", [], [])
    function.body.block.insert_op_before(call, function.body.block.last_op)
    module.verify()
    plan = proposal(module)
    assert plan is not None
    changed = plan.candidate.clone()
    observed = next(op for op in changed.walk() if type(op) is func.CallOp)
    observed.properties["callee"] = builtin.SymbolRefAttr("different")
    with pytest.raises(ValueError):
        verify_source_recomputation(module, changed, generics(module), permissions(), budgets())


def test_opaque_intervening_call_keeps_original():
    module = source()
    module.body.block.add_op(func.FuncOp.external("opaque", [], []))
    members = generics(module)
    members[0].parent.insert_op_before(func.CallOp("opaque", [], []), members[1])
    module.verify()
    assert proposal(module) is None


def test_observed_p_and_shared_destination_keep_original():
    module = source()
    members = generics(module)
    returned = module.body.block.first_op.body.block.last_op
    returned.operands = (*returned.operands, members[0].results[0])
    function = module.body.block.first_op
    function.properties["function_type"] = builtin.FunctionType.from_lists(
        function.function_type.inputs, [*function.function_type.outputs, members[0].results[0].type]
    )
    module.verify()
    assert proposal(module) is None
    module = source()
    members = generics(module)
    operands = list(members[1].operands)
    operands[-1] = members[0].outputs[0]
    members[1].operands = operands
    module.verify()
    assert proposal(module) is None


def test_unregistered_scalar_body_is_conservative():
    module = source()
    first = generics(module)[0]
    scalar = first.body.block.first_op
    scalar.attributes["extra.effect"] = builtin.UnitAttr()
    module.verify()
    assert proposal(module) is None


def test_same_code_replacement_of_checker_is_stale(monkeypatch):
    import merlin.transforms.recomputation_check as checker

    plan = proposal()
    original = checker._composed_graph
    monkeypatch.setattr(checker, "_composed_graph", lambda *args: None)
    with pytest.raises(ValueError, match="checker code"):
        plan.revalidate()
    monkeypatch.setattr(checker, "_composed_graph", original)
    plan.revalidate()


def test_stamp_and_own_permission_mutation_are_stale():
    plan = proposal()
    object.__setattr__(plan.stamp.context, "numerical_policy", "changed")
    with pytest.raises(ValueError):
        plan.revalidate()


def test_captured_constant_keeps_exact_order_and_all_work():
    module = source()
    members = generics(module)
    constant = arith.ConstantOp.from_int_and_width(3, 8)
    members[0].parent.insert_op_before(constant, members[0])
    add = members[0].body.block.first_op
    add.operands = (add.operands[0], constant.result)
    module.verify()
    plan = proposal(module)
    assert plan is not None
    plan.revalidate()


def test_late_external_use_of_original_p_invalidates_proposal():
    plan = proposal()
    p = generics(plan.source)[0].results[0]
    observer = func.ReturnOp(p)
    try:
        with pytest.raises(ValueError, match="SSA|source"):
            plan.revalidate()
    finally:
        observer.drop_all_references()
    plan.revalidate()


@pytest.mark.parametrize("field", ["baseline_selection_sha256", "candidate_selection_sha256", "_guard"])
def test_issued_result_binding_cannot_be_retargeted(field):
    plan = proposal()
    original = getattr(plan, field)
    object.__setattr__(plan, field, lambda: None if field == "_guard" else "changed")
    try:
        with pytest.raises(ValueError, match="issued"):
            plan.revalidate()
    finally:
        object.__setattr__(plan, field, original)
    plan.revalidate()


def test_independent_checker_enforces_actual_construction_ceiling():
    module = source()
    plan = proposal(module)
    with pytest.raises(ValueError):
        verify_source_recomputation(
            module,
            plan.candidate,
            generics(module),
            permissions(),
            budgets(max_constructed_operations=len(tuple(plan.candidate.walk()))),
        )


def test_same_ordered_graph_preserves_captured_const_shared_by_both_stages():
    module = source()
    members = generics(module)
    constant = arith.ConstantOp.from_int_and_width(3, 8)
    members[0].parent.insert_op_before(constant, members[0])
    for member in members[:2]:
        op = member.body.block.first_op
        op.operands = (op.operands[0], constant.result)
    module.verify()
    plan = proposal(module)
    assert plan is not None
    assert all(
        any(constant.result is value for scalar in op.body.block.ops for value in scalar.operands) is False
        for op in generics(plan.candidate)
    )
    # The copied module owns the same dominating capture, not the old module.
    captured = next(op for op in plan.candidate.body.block.first_op.body.block.ops if type(op) is arith.ConstantOp)
    assert len(tuple(captured.result.uses)) == 3
    plan.revalidate()


def test_scalar_capture_after_the_fanout_is_not_reordered():
    module = source()
    members = generics(module)
    constant = arith.ConstantOp.from_int_and_width(3, 8)
    members[0].parent.insert_op_before(constant, members[0].parent.last_op)
    add = members[0].body.block.first_op
    add.operands = (add.operands[0], constant.result)
    before = ir_digest(module)
    assert proposal(module) is None
    assert ir_digest(module) == before


def test_direct_layout_semantic_helper_replacement_is_stale(monkeypatch):
    from merlin.llvmlower import layout_conformance

    plan = proposal()
    monkeypatch.setattr(layout_conformance, "_shape", lambda value: None)
    with pytest.raises(ValueError, match="checker code"):
        plan.revalidate()


def test_candidate_search_bound_keeps_original_or_finds_next_legal_region():
    first, second = source(shape=(9, 9)), source(shape=(1, 3))
    second_function = second.body.block.first_op.clone()
    second_function.properties["sym_name"] = builtin.StringAttr("other")
    first.body.block.add_op(second_function)
    first.verify()
    assert proposal(first, max_candidates=1, max_added_scalar_evaluations=20) is None
    plan = proposal(first, max_candidates=2, max_added_scalar_evaluations=20)
    assert plan is not None
    assert plan.record()["conformance"]["shape"] == [1, 3]
    assert len(generics(plan.candidate)) == 5


def test_ordered_integer_execution_of_both_recomputed_consumers():
    # A tiny independent interpreter of actual candidate scalar operations,
    # with a Python reference that wraps after EACH original application.
    for width in (8, 16):
        plan = proposal(source(width=width, shape=(1, 9)))
        mask = (1 << width) - 1
        data = (
            [mask, 1, 1 << (width - 1), mask - 1, 4, 0, 7, 9, 11],
            [1, mask, 3, 5, mask, 2, 1, 4, 6],
            [2, 3, 4, 5, 6, 7, 8, 9, 10],
            [9, 8, 7, 6, 5, 4, 3, 2, 1],
        )
        before = tuple(tuple(row) for row in data)
        originals = tuple(plan.candidate.body.block.first_op.body.block.args)
        for member, op in enumerate(generics(plan.candidate)):
            actual = []
            for index in range(9):
                values = {
                    arg: data[originals.index(value)][index]
                    for arg, value in zip(tuple(op.body.block.args)[:-1], op.inputs, strict=True)
                }
                for scalar in tuple(op.body.block.ops)[:-1]:
                    a, b = (values[v] for v in scalar.operands)
                    value = a + b if type(scalar) is arith.AddiOp else a - b if type(scalar) is arith.SubiOp else a * b
                    values[scalar.results[0]] = value & mask
                actual.append(values[op.body.block.last_op.operands[0]])
            expected = []
            for x, state, w0, w1 in zip(*data, strict=True):
                p = (((x + state) & mask) * x) & mask
                expected.append((p - w0 if member == 0 else p + w1) & mask)
            assert actual == expected
        assert tuple(tuple(row) for row in data) == before


def test_same_issued_guard_function_code_replacement_is_refused():
    plan = proposal()
    original = plan._guard.__code__
    # The issued guard has closed-over state. Preserve the closure arity while
    # replacing its actual code; comparing only function identity is unsound.
    cells = len(plan._guard.__closure__ or ())
    namespace = {}
    text = "def outer():\n" + "".join(f"    x{i}=None\n" for i in range(cells))
    text += "    def noop():\n        return (" + ",".join(f"x{i}" for i in range(cells)) + ",)\n    return noop\n"
    exec(text, namespace)
    plan._guard.__code__ = namespace["outer"]().__code__
    try:
        with pytest.raises(ValueError, match="guard code"):
            plan.revalidate()
    finally:
        plan._guard.__code__ = original
    plan.revalidate()


def test_selection_ids_use_same_owner_bytes_after_path_relocation(tmp_path, monkeypatch):
    import merlin.transforms.recomputation as recompute

    first = proposal()
    expected = first.baseline_selection_sha256, first.candidate_selection_sha256
    paths = first.implementation_paths()
    replacements = []
    for i, path in enumerate(paths):
        copied = tmp_path / f"{i}-{path.name}"
        copied.write_bytes(path.read_bytes())
        replacements.append(copied)
    monkeypatch.setattr(recompute, "implementation_paths", lambda: tuple(replacements))
    second = proposal()
    assert (second.baseline_selection_sha256, second.candidate_selection_sha256) == expected
    assert second.record()["implementation_sources"] != first._record["implementation_sources"]
    second.revalidate()


def test_registry_does_not_retain_issued_plan_or_modules():
    import gc
    import weakref

    plan = proposal()
    refs = weakref.ref(plan), weakref.ref(plan.source), weakref.ref(plan.candidate)
    del plan
    gc.collect()
    assert all(reference() is None for reference in refs)
