"""Fixed source/callee diagnostics; no device semantics or runtime authority."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.component_witness import REQUIRED_EXECUTION_EFFECTS
from merlin_experiments.phase2 import component_runtime_copy_controls as C
from merlin_experiments.phase2 import component_runtime_copy_support as S
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from test_component_runtime_support import prepared as prepared


def source(shape=(1, 1), dtype="i8"):
    value = "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">"
    return (
        f"module {{ func.func @main(%a: {value}, %b: {value}) -> ({value}, {value}) {{ "
        f"%d0=tensor.empty() : {value} "
        f"%r0=linalg.copy ins(%a : {value}) outs(%d0 : {value}) -> {value} "
        f"%d1=tensor.empty() : {value} "
        f"%r1=linalg.copy ins(%b : {value}) outs(%d1 : {value}) -> {value} "
        f"func.return %r0, %r1 : {value}, {value} }} }}"
    )


@pytest.mark.parametrize("shape,dtype", [((1, 1), "i8"), ((2, 3), "i16"), ((7,), "i32")])
def test_registered_copy_relation_preserves_shape_and_order_with_explicit_callee(shape, dtype):
    program = C.parse_copy(source(shape, dtype))
    emitted = C.emit_copy_llvm(program, entry_symbol="entry", callee_symbol="chosen_helper")
    proof = C.verify_copy_llvm(source(shape, dtype), emitted, entry_symbol="entry", callee_symbol="chosen_helper")
    assert proof["shape"] == list(shape) and proof["dtype"] == dtype
    assert proof["ordered_return_inputs"] == [0, 1]
    assert proof["helper_semantics"] == "UNKNOWN"
    assert "unqualified" in proof["scope"]
    # SSA spelling is immaterial to the exact ordered pointer relation.
    renamed = emitted.replace("%p", "%renamed")
    C.verify_copy_llvm(source(shape, dtype), renamed, entry_symbol="entry", callee_symbol="chosen_helper")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text: text.replace("chosen_helper", "another_helper"),
        lambda text: text.replace("(%p0, %p2)", "(%p1, %p2)"),
        lambda text: text.replace("(%p0, %p2)", "(%p2, %p0)"),
        lambda text: text.replace(
            "    llvm.return", "    llvm.call @chosen_helper(%p0, %p2) : (!llvm.ptr, !llvm.ptr) -> ()\n    llvm.return"
        ),
        lambda text: text.replace("llvm.return", "llvm.return %p0 : !llvm.ptr"),
    ],
)
def test_changed_callee_input_writer_extra_call_or_return_cannot_get_source_relation(mutation):
    emitted = C.emit_copy_llvm(C.parse_copy(source()), entry_symbol="entry", callee_symbol="chosen_helper")
    with pytest.raises(Exception):
        C.verify_copy_llvm(source(), mutation(emitted), entry_symbol="entry", callee_symbol="chosen_helper")


def test_original_copy_inputs_are_derived_from_registered_source_not_ordinals():
    reversed_source = source().replace("func.return %r0, %r1", "func.return %r1, %r0")
    assert C.parse_copy(reversed_source).outputs == (1, 0)
    emitted = C.emit_copy_llvm(C.parse_copy(source()), entry_symbol="entry", callee_symbol="chosen_helper")
    with pytest.raises(ValueError, match="pointer/callee relation"):
        C.verify_copy_llvm(reversed_source, emitted, entry_symbol="entry", callee_symbol="chosen_helper")


@pytest.mark.parametrize(
    "bad",
    [
        lambda text: text.replace("tensor<1x1xi8>", "tensor<?x1xi8>"),
        lambda text: text.replace("tensor<1x1xi8>", "tensor<1x1xf32>"),
        lambda text: text.replace("func.return %r0, %r1", "func.return %r0, %r0"),
        lambda text: text.replace("ins(%a", "ins(%d0"),
        lambda text: text.replace("func.return %r0, %r1", "func.return %a, %b"),
    ],
)
def test_unsupported_dynamic_float_alias_or_incomplete_source_refuses(bad):
    with pytest.raises(ValueError):
        C.parse_copy(bad(source()))


def test_support_metadata_cannot_replace_live_hardware_command_and_software(tmp_path):
    support = S.RuntimeCopyControlSupport(None, None, tmp_path / "helper.c", "chosen_helper", (1, 1), "i8", (), ())
    with pytest.raises(StageGateError, match="same live HW"):
        support.verify(hardware=SimpleNamespace(target="independent"), build=None, context_pins=())


def test_selected_copy_fixture_is_actual_source_sealed_and_keeps_unknown_stage(prepared, tmp_path):
    helper = tmp_path / "helper.c"
    helper.write_text("/* Unit source selection only; no device semantics. */\n")
    # Fixture bookkeeping is isolated by the shared unit fixture. None objects
    # cannot pass the live support verifier or qualify this synthetic selection.
    selected = S.RuntimeCopyControlSupport(
        None, None, helper, "chosen_helper", (2, 3), "i16", ((helper, sha256_file(helper)),), ()
    )
    context = replace(prepared, copy_control_support=selected)
    fixture = context.prepare_control("ownership_lifetime.positive", tmp_path / "control")
    text = (fixture.capsule_root / "source.mlir").read_text()
    program = C.parse_copy(text)
    assert program == C.CopyProgram((2, 3), "i16", 2, (0, 1))
    assert fixture.member["output_roster"] == ["first", "second"]
    command = (fixture.grade_arguments["package_dir"] / "manifest.yaml").read_text()
    assert "chosen_helper" in command
    assert fixture.required_effects == REQUIRED_EXECUTION_EFFECTS
    context.verify_control(fixture)
    lowered = tmp_path / "lowered.mlir"
    lowered.write_text(C.emit_copy_llvm(program, entry_symbol="control_entry", callee_symbol="chosen_helper"))
    relation = context._evaluate_source(fixture.capsule_root / "source.mlir", lowered, fixture)
    assert relation["status"] == "accepted" and relation["proof"]["helper_semantics"] == "UNKNOWN"
    result = tmp_path / "capsule_result.json"
    result.write_text('{"status":"pass","numeric":"pass"}')
    with pytest.raises(StageGateError, match="remain UNKNOWN"):
        context.stage_verifier(result_path=result)
    (fixture.capsule_root / "source.mlir").write_text(source())
    with pytest.raises(StageGateError, match="products changed"):
        context.verify_control(fixture)


def test_unavailable_original_defect_recipe_does_not_become_a_successful_negative(prepared, tmp_path):
    selected = S.RuntimeCopyControlSupport(None, None, tmp_path / "helper.c", "chosen_helper", (1, 1), "i8", (), ())
    fixture = SimpleNamespace(case_id="ownership_lifetime.negative")
    with pytest.raises(StageGateError, match="remains UNKNOWN"):
        selected.selected_build(fixture=fixture, build=prepared.build_service)


def test_foreign_support_cannot_execute_a_supplied_verifier_callback(prepared):
    class ForeignSupport:
        def verify(self, **_kwargs):
            raise AssertionError("foreign support callback was invoked before exact-type admission")

    context = replace(prepared, copy_control_support=ForeignSupport())
    with pytest.raises(StageGateError, match="fixed source-selection declaration"):
        context._copy_selection()
