"""Whole-source identity binding without target, timing or runtime qualification."""

import json
from dataclasses import replace

import pytest
from test_dense_source import contraction
from xdsl.dialects import builtin, func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.offload_plan import OffloadPlanUnavailable, bind_source, realize_offload_plan
from merlin.transforms.certificates import ir_text
from merlin.transforms.pipeline_state import PipelineContext
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, GlobalPlan, TransitionAlternative


def example(*, widths=("i8", "i8", "i32"), shape=(3, 7, 5)):
    m, k, n = shape
    x, y, z = widths
    module = parse_mlir_text(f"""module {{func.func @forward(%a:tensor<{m}x{k}x{x}>,%b:tensor<{k}x{n}x{y}>)
      ->(tensor<{m}x{n}x{z}>,tensor<{m}x{n}x{z}>) {{
      {contraction(shape, widths, output="first", tag="same")}
      {contraction(shape, widths, output="second", tag="same")}
      func.return %first,%second:tensor<{m}x{n}x{z}>,tensor<{m}x{n}x{z}>
    }}}}""")
    context = PipelineContext(
        "original hardware",
        "unchanged numerical policy",
        "physical resources",
        "qualified caller effects",
        "same shared library",
    )
    bound = bind_source(module, entry="forward", context=context)
    contractions = [index for index, operation in enumerate(bound.operations) if operation.name == "linalg.generic"]
    device = tuple(
        replace(
            bound.cpu_alternatives[index],
            id=f"device_{index}",
            implementation=f"original_kernel_{index}",
            placement="selected_endpoint",
        )
        for index in contractions
    )
    return bound, device, context


def choose(bound, device, *, indices=(1,)):
    chosen = {alternative.nodes[0]: alternative for position, alternative in enumerate(device) if position in indices}
    return GlobalPlan(
        tuple(chosen.get(index, alternative) for index, alternative in enumerate(bound.cpu_alternatives)),
        (),
        CycleInterval.unknown("no performance qualification from structural source binding"),
    )


@pytest.mark.parametrize("shape,widths", [((3, 7, 5), ("i8", "i8", "i32")), ((1, 9, 11), ("i16", "i16", "i32"))])
def test_complete_source_glue_and_exact_current_operations_are_bound(shape, widths):
    bound, device, context = example(shape=shape, widths=widths)
    before = ir_text(bound.module)
    plan = choose(bound, device)
    selected = realize_offload_plan(bound, plan, device_alternatives=device, context=context)
    assert ir_text(bound.module) == before
    assert len(bound.program.nodes) == len(bound.operations) == len(selected.covered_ordinals) == 8
    assert selected.selected_ops == (bound.operations[7],)
    assert selected.retained_cpu_ops == bound.operations[:7]
    current = parse_mlir_text(before)
    rebound = selected.select_for(current, context=context)
    assert len(rebound) == 1 and rebound[0] is not selected.selected_ops[0]
    assert tuple(current.walk())[selected.selected_ordinals[0]] is rebound[0]
    # Identical shapes and provenance do not cause both contractions to route.
    assert [op.name for op in rebound] == ["linalg.generic"]
    report = selected.record()
    assert report["selected_ordinals"] == list(selected.selected_ordinals)
    assert len(report["retained_cpu_ordinals"]) == 7
    assert [stage["name"] for stage in report["passes"]] == ["offload-source", "offload-plan", "offload-conformance"]
    assert not plan.cycles.resolved and "no runtime" in report["scope"]


def test_empty_device_roster_preserves_all_original_cpu_operations():
    bound, _, context = example()
    plan = GlobalPlan(bound.cpu_alternatives, (), CycleInterval.unknown("unpriced CPU baseline"))
    selected = realize_offload_plan(bound, plan, device_alternatives=(), context=context)
    assert selected.selected_ops == selected.selected_ordinals == ()
    assert selected.retained_cpu_ops == bound.operations
    assert selected.select_for(parse_mlir_text(ir_text(bound.module)), context=context) == ()


@pytest.mark.parametrize(
    "mutation",
    ["source", "graph", "context", "source_context", "selection_context", "plan", "alternatives", "membership"],
)
def test_original_custody_mutations_refuse_before_rebinding(mutation):
    bound, device, context = example()
    selected = realize_offload_plan(bound, choose(bound, device), device_alternatives=device, context=context)
    if mutation == "source":
        bound.operations[0].attributes["prov.mutation"] = builtin.UnitAttr()
    elif mutation == "graph":
        bound.program.nodes[-1].inputs.reverse()
    elif mutation == "context":
        object.__setattr__(context, "resources", "changed original pool")
    elif mutation == "source_context":
        object.__setattr__(bound._session.context, "numerical_policy", "changed source policy")
    elif mutation == "selection_context":
        object.__setattr__(selected._session.context, "execution", "changed completion")
    elif mutation == "plan":
        object.__setattr__(selected.plan, "cycles", CycleInterval.point(1, "changed original cost"))
    elif mutation == "alternatives":
        object.__setattr__(device[0], "implementation", "changed implementation")
    else:
        selected = replace(selected, selected_ops=(bound.operations[3],))
    with pytest.raises(ValueError, match="changed|stale|differs"):
        selected.revalidate(context=context)


@pytest.mark.parametrize("mutation", ["different_source", "same_shape_rewire", "other_entry", "changed_policy"])
def test_reparse_must_preserve_complete_original_source_and_context(mutation):
    bound, device, context = example(shape=(5, 5, 5))
    selected = realize_offload_plan(bound, choose(bound, device), device_alternatives=device, context=context)
    current = parse_mlir_text(ir_text(bound.module))
    if mutation == "different_source":
        current.attributes["prov.current"] = builtin.UnitAttr()
    elif mutation == "same_shape_rewire":
        operation = [op for op in current.walk() if op.name == "linalg.generic"][-1]
        operation.operands = (operation.operands[0], operation.operands[0], operation.operands[2])
        current = parse_mlir_text(ir_text(current))
    elif mutation == "other_entry":
        current.body.block.first_op.sym_name = builtin.StringAttr("different")
    else:
        context = replace(context, execution="different selected completion")
    current.verify()
    with pytest.raises((ValueError, OffloadPlanUnavailable)):
        selected.select_for(current, context=context)


@pytest.mark.parametrize(
    "mutation",
    [
        "overlap",
        "missing_glue",
        "fused",
        "cpu_impl",
        "cpu_placement",
        "device_impl",
        "abi",
        "resident",
        "scratch",
        "multiple_endpoints",
    ],
)
def test_unrealized_plan_changes_cannot_become_offload(mutation):
    bound, device, context = example()
    plan = choose(bound, device)
    rows = list(plan.selected)
    if mutation == "overlap":
        rows.append(replace(rows[-1], id="overlapping"))
    elif mutation == "missing_glue":
        rows.pop(0)
    elif mutation == "fused":
        rows[-1] = replace(rows[-1], nodes=(6, 7))
    elif mutation == "cpu_impl":
        rows[0] = replace(rows[0], implementation="different CPU computation")
    elif mutation == "cpu_placement":
        rows[0] = replace(rows[0], placement="selected_endpoint")
    elif mutation == "device_impl":
        rows[-1] = replace(rows[-1], implementation="not admitted")
    elif mutation == "abi":
        altered = replace(device[-1], inputs=())
        device = (*device[:-1], altered)
        rows[-1] = altered
    elif mutation == "resident":
        altered = replace(device[-1], resident_inputs=(device[-1].inputs[0].buffer,))
        device = (*device[:-1], altered)
        rows[-1] = altered
    elif mutation == "scratch":
        altered = replace(device[-1], scratch_bytes=8)
        device = (*device[:-1], altered)
        rows[-1] = altered
    else:
        device = (replace(device[0], placement="other_endpoint"), device[1])
    plan = replace(plan, selected=tuple(rows))
    with pytest.raises(ValueError, match="source offload plan"):
        realize_offload_plan(bound, plan, device_alternatives=device, context=context)


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "call", "nested", "placement", "unknown_attribute", "module_attribute", "function_attribute", "budget"],
)
def test_unknown_effects_control_and_explicit_placement_are_not_interpreted(mutation):
    bound, _, context = example()
    module = parse_mlir_text(ir_text(bound.module))
    function = module.body.block.first_op
    block = function.body.block
    if mutation == "unknown":
        block.insert_op_before(builtin.UnregisteredOp.with_name("opaque.effect").create(), block.last_op)
    elif mutation == "call":
        module.body.block.add_op(func.FuncOp.external("external", [], []))
        block.insert_op_before(func.CallOp("external", [], []), block.last_op)
    elif mutation == "nested":
        module = parse_mlir_text("""module {func.func @forward(%a:tensor<3xi8>)->tensor<3xi8> {
          %c0=arith.constant 0:index
          %c1=arith.constant 1:index
          scf.for %i=%c0 to %c1 step %c1 { scf.yield }
          func.return %a:tensor<3xi8>
        }}""")
    elif mutation == "placement":
        function.attributes["merlin.placement"] = builtin.StringAttr("host")
    elif mutation == "unknown_attribute":
        block.first_op.attributes["opaque_mode"] = builtin.IntegerAttr(0, 32)
    elif mutation == "module_attribute":
        module.attributes["opaque_mode"] = builtin.UnitAttr()
    elif mutation == "function_attribute":
        function.attributes["opaque_mode"] = builtin.UnitAttr()
    module.verify()
    with pytest.raises((ValueError, OffloadPlanUnavailable)):
        bind_source(module, entry="forward", context=context, max_operations=1 if mutation == "budget" else 4096)


def test_unrealized_transition_and_noncontraction_device_alternatives_refuse():
    bound, device, context = example()
    plan = choose(bound, device)
    value = device[-1].inputs[0]
    transition = TransitionAlternative(
        "unrealized",
        "movement",
        value.buffer,
        None,
        device[-1].id,
        value.representation,
        value.representation,
        CycleInterval.point(2, "synthetic screening value; no timing qualification"),
    )
    with pytest.raises(ValueError, match="physical realization"):
        realize_offload_plan(
            bound, replace(plan, transitions=(transition,)), device_alternatives=device, context=context
        )
    glue = replace(bound.cpu_alternatives[0], id="glue_device", placement="selected_endpoint")
    with pytest.raises(ValueError, match="signed contraction"):
        realize_offload_plan(bound, plan, device_alternatives=(*device, glue), context=context)


def test_derived_diagnostic_fields_cannot_replace_original_pipeline_binding():
    bound, device, context = example()
    selected = realize_offload_plan(bound, choose(bound, device), device_alternatives=device, context=context)
    substituted = replace(selected, plan=GlobalPlan(bound.cpu_alternatives, (), selected.plan.cycles))
    substituted = replace(
        substituted, _plan_record=json.dumps(substituted.plan.to_dict(), sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ValueError, match="changed|differs"):
        substituted.revalidate(context=context)
    graph = selected.record()
    graph["selected_ordinals"].clear()
    assert selected.record()["selected_ordinals"] == list(selected.selected_ordinals)
    assert selected.record()["device_alternatives"] == [alternative.to_dict() for alternative in device]


@pytest.mark.parametrize("max_operations", [False, 0, -1, 100001])
def test_source_binding_operation_budget_is_explicit_and_bounded(max_operations):
    bound, _, context = example()
    with pytest.raises(ValueError, match="bounded positive"):
        bind_source(bound.module, entry=bound.entry, context=context, max_operations=max_operations)


def test_original_read_only_aliases_and_unpruned_unused_glue_are_preserved():
    original, _, context = example(shape=(5, 5, 5))
    module = parse_mlir_text(ir_text(original.module))
    for operation in module.walk():
        if operation.name == "linalg.generic":
            operation.operands = (operation.operands[0], operation.operands[0], operation.operands[2])
    module.verify()
    bound = bind_source(module, entry="forward", context=context)
    index = 7
    device = (
        replace(
            bound.cpu_alternatives[index],
            id="original_alias_call",
            implementation="admitted_read_only_alias_kernel",
            placement="selected_endpoint",
        ),
    )
    selected = realize_offload_plan(
        bound, choose(bound, device, indices=(0,)), device_alternatives=device, context=context
    )
    (current,) = selected.select_for(parse_mlir_text(ir_text(module)), context=context)
    assert current.operands[0] is current.operands[1]
    assert len(selected.covered_ordinals) == 8
    assert len(selected.retained_cpu_ops) == 7


def test_unused_dynamic_argument_remains_outside_static_source_domain():
    bound, _, context = example()
    module = parse_mlir_text(ir_text(bound.module))
    function = module.body.block.first_op
    block = function.body.block
    unused = block.insert_arg(builtin.TensorType(builtin.i8, (builtin.DYNAMIC_INDEX, 7)), len(block.args))
    function.properties["function_type"] = builtin.FunctionType.from_lists(
        [value.type for value in block.args], function.function_type.outputs
    )
    module.verify()
    assert not unused.uses
    with pytest.raises((ValueError, OffloadPlanUnavailable), match="static"):
        bind_source(module, entry="forward", context=context)
