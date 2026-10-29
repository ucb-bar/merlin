"""Actual declared instruction IR uses the shared planner, with no target guess."""

from dataclasses import replace

import pytest
from test_selected_fusion_contracts import inputs
from xdsl.dialects import builtin

from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets
from merlin.llvmlower import target_schedule as T
from merlin.transforms.certificates import ir_text


def case(tmp_path, *, enabled=True, shape=(3, 7, 5, 4), width=1, separate=False):
    selected, kernel, objects, adapter, cap, codec, limits, options = inputs(
        tmp_path, shape=shape, width=width, separate=separate
    )
    selection = T.TargetScheduleSelection(
        selected,
        objects,
        adapter,
        codec,
        limits,
        parameters={
            **options,
            "budgets": ScheduleBudgets(max_fusions=8 if enabled else 0),
            "fusion_emitters": {cap.readout.selector: cap.emit},
            "fusion_qualification_identity": "independent-neutral-two-layer-v1",
        },
    )
    module = T.instruction_module(replace(kernel, name="target_program", args=(), attrs=()), selected)
    return selection, module


@pytest.mark.parametrize("shape,width,separate", [((3, 7, 5, 4), 1, False), ((2, 5, 11, 3), 2, True)])
def test_actual_instruction_ir_runs_existing_fusion_and_roundtrips_all_operands(tmp_path, shape, width, separate):
    selection, module = case(tmp_path, shape=shape, width=width, separate=separate)
    original = ir_text(module)
    result = selection.apply(selection.parse(original))
    assert ir_text(module) == original
    assert result.report["status"] == "checked"
    before = T.instruction_kernel(module, selection.selected_contracts)
    after = T.instruction_kernel(result.module, selection.selected_contracts)
    assert len(before.body) == 7 and len(after.body) == 5
    assert after == result.compilation.kernel
    assert T.instruction_kernel(selection.parse(ir_text(result.module)), selection.selected_contracts) == after
    result.revalidate()


def test_unknown_control_or_arithmetic_retains_complete_original_fragment(tmp_path):
    selection, _ = case(tmp_path)
    module = selection.parse('builtin.module { "foreign.opaque"() : () -> () }')
    result = selection.apply(module)
    assert result.compilation is None and result.report["status"] == "fallback"
    assert result.module.is_structurally_equivalent(module)
    result.revalidate()


@pytest.mark.parametrize("role", ["options", "derived_storage", "derived_fusion", "owner"])
def test_changed_original_controls_or_derived_contracts_refuse_before_planning(tmp_path, role):
    selection, module = case(tmp_path)
    selected = selection.selected_contracts
    if role == "options":
        selection.parameters["budgets"] = ScheduleBudgets(max_fusions=0)
    elif role == "owner":
        object.__setattr__(selection, "selected_contracts", replace(selected))
    else:
        object.__setattr__(selected, role.removeprefix("derived_"), None)
    with pytest.raises(ValueError, match="changed"):
        selection.apply(module)


@pytest.mark.parametrize("role", ["source", "selected", "report", "image"])
def test_result_rechecks_current_ir_report_and_actual_encoded_program(tmp_path, role):
    selection, module = case(tmp_path)
    result = selection.apply(module)
    if role == "source":
        result.source.body.block.erase_op(result.source.body.block.last_op)
    elif role == "selected":
        result.module.body.block.erase_op(result.module.body.block.last_op)
    elif role == "report":
        result.report["status"] = "forged"
    else:
        image = result.compilation.program.segments[0]
        object.__setattr__(result.compilation.program, "segments", (image + image,))
    with pytest.raises(ValueError, match="changed|lost|differs"):
        result.revalidate()


def test_owner_callback_code_cannot_change_after_original_selection(tmp_path):
    selection, module = case(tmp_path)
    callback = selection.adapter.project
    code = callback.__code__
    try:
        callback.__code__ = code.replace(co_name="mutated_project")
        with pytest.raises(ValueError, match="changed"):
            selection.apply(module)
    finally:
        callback.__code__ = code


def test_original_checker_replacement_cannot_bypass_source_pins(tmp_path, monkeypatch):
    selection, module = case(tmp_path)
    monkeypatch.setattr(T, "_check_callback", lambda *_: None)
    with pytest.raises(ValueError, match="checker changed"):
        selection.apply(module)


def test_annotations_are_not_dropped_by_the_finite_ir_writer(tmp_path):
    selection, module = case(tmp_path)
    kernel = T.instruction_kernel(module, selection.selected_contracts)
    bad = replace(kernel, body=(replace(kernel.body[0], unit="undeclared-engine"), *kernel.body[1:]))
    with pytest.raises(ValueError, match="annotations"):
        T.instruction_module(bad, selection.selected_contracts)
    decorated = module.clone()
    decorated.attributes["unproved"] = builtin.UnitAttr()
    assert selection.apply(decorated).report["status"] == "fallback"


@pytest.mark.parametrize(
    "binding",
    ["instruction_kernel", "instruction_module", "compile_schedule", "_call", "encode_instruction", "_functions"],
)
def test_live_implementation_alias_cannot_replace_retained_owner(tmp_path, monkeypatch, binding):
    selection, module = case(tmp_path)
    monkeypatch.setattr(T, binding, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="changed"):
        selection.apply(module)
