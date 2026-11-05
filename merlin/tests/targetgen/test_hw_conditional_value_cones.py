"""Independent conditional source cones never evaluate state or memory history."""

import hashlib
import json
from dataclasses import replace

import pytest

from merlin.targetgen.rtl.hw_combinational import EvaluationLimits
from merlin.targetgen.rtl.hw_conditional_value_cones import ConditionalValueCut, prepare_conditional_value_cones
from merlin.targetgen.rtl.hw_value_bindings import OriginalValueSelection, ValueBindingLimits

BINDINGS = ValueBindingLimits(65536, 16, 512, 32, 256, 32, 256, 128, 8192, 16, 64, 65536)
EVALUATION = EvaluationLimits(65536, 256, 32, 32, 65536)


def source(*, coefficient=1, opaque=False, parameterized=False):
    cell = f""""hw.module"() ({{
      ^bb0(%data: i8, %choose: i1, %tick: !seq.clock):
        %offset = "hw.constant"() {{value = {coefficient} : i8}} : () -> i8
        %sum = "comb.add"(%data, %offset) : (i8, i8) -> i8
        %held = "seq.firreg"(%held, %tick) {{name = "stored"}} : (i8, !seq.clock) -> i8
        %selected = "comb.mux"(%choose, %sum, %held) : (i1, i8, i8) -> i8
        "sv.always"(%tick) ({{
          "sim.clocked_terminate"(%tick) {{success = false, message = "retained effect"}}
            : (!seq.clock) -> ()
        }}) : (!seq.clock) -> ()
        "hw.output"(%selected) : (i8) -> ()
      }}) {{sym_name = "Cell", parameters = [],
        module_type = !hw.modty<input data : i8, input choose : i1, input tick : !seq.clock,
          output observed : i8>}} : () -> ()"""
    if opaque:
        cell = """"hw.module.extern"() {sym_name = "Cell", parameters = [],
        module_type = !hw.modty<input data : i8, input choose : i1, input tick : !seq.clock,
          output observed : i8>} : () -> ()"""
    parameters = "[1 : i32]" if parameterized else "[]"
    return f"""builtin.module {{
      {cell}
      "hw.module"() ({{
      ^bb0(%word: i16, %choose: i1, %tick: !seq.clock):
        %lo = "comb.extract"(%word) {{lowBit = 0 : i32}} : (i16) -> i8
        %hi = "comb.extract"(%word) {{lowBit = 8 : i32}} : (i16) -> i8
        %join = "comb.concat"(%hi, %lo) : (i8, i8) -> i16
        %a = "hw.instance"(%lo, %choose, %tick) {{moduleName = @Cell, instanceName = "first",
          argNames = ["data", "choose", "tick"], resultNames = ["observed"], parameters = {parameters}}}
          : (i8, i1, !seq.clock) -> i8
        %b = "hw.instance"(%hi, %choose, %tick) {{moduleName = @Cell, instanceName = "second",
          argNames = ["data", "choose", "tick"], resultNames = ["observed"], parameters = []}}
          : (i8, i1, !seq.clock) -> i8
        "hw.output"(%a, %b, %join) : (i8, i8, i16) -> ()
      }}) {{sym_name = "Top", parameters = [],
        module_type = !hw.modty<input word : i16, input choose : i1, input tick : !seq.clock,
          output a : i8, output b : i8, output reconstructed : i16>}} : () -> ()
    }}"""


def selection(text, kind="module_output", ordinal=0, *, path=("Top",), module="Top", slot=0, typ="i8"):
    return OriginalValueSelection(hashlib.sha256(text.encode()).hexdigest(), path, module, kind, ordinal, slot, typ)


def state_cuts(text):
    return tuple(
        ConditionalValueCut(
            selection(text, "operation_result", 2, path=("Top", occurrence), module="Cell"), "state_result"
        )
        for occurrence in ("first", "second")
    )


def prepare(text, *, selections=None, cuts=None, binding_limits=BINDINGS, evaluation_limits=EVALUATION):
    if selections is None:
        selections = (selection(text), selection(text, ordinal=1), selection(text, ordinal=2, typ="i16"))
    return prepare_conditional_value_cones(
        text,
        root="Top",
        selections=selections,
        cuts=state_cuts(text) if cuts is None else cuts,
        binding_limits=binding_limits,
        evaluation_limits=evaluation_limits,
    )


def case(prepared, values):
    return {
        row.port.name: values[(row.selection.occurrence, row.selection.kind, row.selection.ordinal)]
        for row in prepared.inputs
    }


def inputs(word, choose, first, second):
    return {
        (("Top",), "module_input", 0): word,
        (("Top",), "module_input", 1): choose,
        (("Top", "first"), "operation_result", 2): first,
        (("Top", "second"), "operation_result", 2): second,
    }


@pytest.mark.parametrize(
    "word,choose,first,second", [(0, 0, 6, 19), (65535, 1, 3, 7), (0x83FC, 1, 31, 8), (0x12AB, 0, 255, 0)]
)
def test_hierarchical_complete_outputs_and_distinct_conditional_state_values(word, choose, first, second):
    prepared = prepare(source())
    assert prepared.evaluate((case(prepared, inputs(word, choose, first, second)),)) == (
        {
            "value_0": ((word & 255) + 1) % 256 if choose else first,
            "value_1": ((word >> 8) + 1) % 256 if choose else second,
            "value_2": word,
        },
    )
    assert {row.selection.occurrence for row in prepared.inputs if row.boundary == "state_result"} == {
        ("Top", "first"),
        ("Top", "second"),
    }


def test_same_type_coefficient_and_bit_order_come_from_actual_source():
    prepared = prepare(source(coefficient=7))
    result = prepared.evaluate((case(prepared, inputs(0xABFE, 1, 0, 0)),))[0]
    assert result == {"value_0": 5, "value_1": 178, "value_2": 0xABFE}
    changed = source().replace('"comb.concat"(%hi, %lo)', '"comb.concat"(%lo, %hi)')
    prepared = prepare(changed)
    assert prepared.evaluate((case(prepared, inputs(0x12AB, 0, 0, 0)),))[0]["value_2"] == 0xAB12
    with pytest.raises(ValueError, match="identity"):
        prepare(source(coefficient=7), selections=(selection(source()),), cuts=(state_cuts(source())[0],))


def test_complete_members_and_nested_effects_survive_without_execution_or_admission():
    prepared = prepare(source())
    record = prepared.record()
    assert record["admission_authority"] is False and record["source"]["admission_authority"] is False
    assert record["source"]["source_byte_binding"] == "exact supplied source bytes"
    cell = next(row for row in record["source"]["visited_definition_members"] if row["module"] == "Cell")
    assert cell["operation_count"] == 6 and cell["member_semantics"] == "UNKNOWN"
    assert cell["complete_operation_members"][2]["operands"][1]["type"] == "!seq.clock"
    assert cell["complete_operation_members"][4]["nested_operations"][0]["operation"] == "sim.clocked_terminate"
    assert "source_effect_execution_and_completion" in record["unknowns"]
    record["source"]["unknowns"].clear()
    assert prepared.record()["source"]["unknowns"]


def test_unselected_state_effect_and_unsupported_logic_remain_unknown_members():
    text = source().replace('"comb.add"(%data, %offset)', '"comb.mul"(%data, %offset)')
    prepared = prepare(text, selections=(selection(text, ordinal=2, typ="i16"),), cuts=())
    assert prepared.evaluate(({prepared.inputs[0].port.name: 0xABCD},)) == ({"value_0": 0xABCD},)
    assert len(prepared.record()["source"]["frames"]) == 3
    members = prepared.record()["source"]["visited_definition_members"][0]["complete_operation_members"]
    assert len(members) == 6 and members[3]["operation"] == "hw.instance"
    # The reader makes no claim about unvisited expression semantics.
    assert "unvisited_expression_semantics" in prepared.record()["source"]["unknowns"]


def test_unsupported_reachable_logic_cannot_be_hidden_by_declared_cut():
    text = source().replace('"comb.add"(%data, %offset)', '"comb.mul"(%data, %offset)')
    arbitrary = ConditionalValueCut(
        selection(text, "operation_result", 1, path=("Top", "first"), module="Cell"), "state_result"
    )
    with pytest.raises(ValueError, match="unsupported reachable"):
        prepare(text, cuts=(*state_cuts(text), arbitrary))


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "wrong_kind",
        "stale_source",
        "stale_path",
        "stale_module",
        "stale_type",
        "stale_slot",
        "stale_ordinal",
    ],
)
def test_conditional_cut_roster_is_exact_and_source_bound(change):
    text = source()
    cuts = state_cuts(text)
    alterations = {
        "wrong_kind": {"boundary": "memory_read_result"},
        "stale_source": {"selection": replace(cuts[0].selection, source_sha256="0" * 64)},
        "stale_path": {"selection": replace(cuts[0].selection, occurrence=("Top", "absent"))},
        "stale_module": {"selection": replace(cuts[0].selection, module="Top")},
        "stale_type": {"selection": replace(cuts[0].selection, type="i16")},
        "stale_slot": {"selection": replace(cuts[0].selection, slot=1)},
        "stale_ordinal": {"selection": replace(cuts[0].selection, ordinal=3)},
    }
    changed = (
        cuts[:1]
        if change == "missing"
        else (*cuts, ConditionalValueCut(selection(text, "operation_result", 0), "state_result"))
        if change == "extra"
        else (replace(cuts[0], **alterations[change]), cuts[1])
    )
    with pytest.raises(ValueError, match="cut|boundary|identity"):
        prepare(text, cuts=changed)


@pytest.mark.parametrize("boundary", ["unsupported_result", "clock_value", True, None])
def test_only_genuine_boundary_classes_are_declarable(boundary):
    with pytest.raises(ValueError, match="genuine boundary"):
        ConditionalValueCut(state_cuts(source())[0].selection, boundary)


def test_conditional_cut_cannot_substitute_module_input_or_output_alias():
    text = source()
    for kind in ("module_input", "module_output", "operation_operand"):
        with pytest.raises(ValueError, match="result identities"):
            ConditionalValueCut(selection(text, kind), "state_result")
    with pytest.raises(ValueError, match="rosters"):
        prepare(text, cuts=(*state_cuts(text), state_cuts(text)[0]))


@pytest.mark.parametrize("opaque,parameterized", [(True, False), (False, True)])
def test_opaque_result_is_explicit_independent_conditional_value(opaque, parameterized):
    text = source(opaque=opaque, parameterized=parameterized)
    selections = (selection(text),)
    cut = ConditionalValueCut(selection(text, "operation_result", 3), "opaque_instance_result")
    prepared = prepare(text, selections=selections, cuts=(cut,))
    assert prepared.evaluate(({prepared.inputs[0].port.name: 213},)) == ({"value_0": 213},)
    assert prepared.inputs[0].selection.occurrence == ("Top",)
    assert prepared.record()["source"]["frames"][1]["stop"] in {"external_body", "parameterized_body"}
    assert "memory_history_collision_and_opaque_implementation" in prepared.record()["unknowns"]


def test_memory_read_is_supplied_known_bits_not_an_executed_read():
    text = """builtin.module {
      "hw.module"() ({
      ^bb0(%address: i2, %tick: !seq.clock):
        %mem = "seq.firmem"() {readLatency = 1 : i32} : () -> !seq.firmem<4 x 8>
        %r = "seq.firmem.read_port"(%mem, %address, %tick)
          : (!seq.firmem<4 x 8>, i2, !seq.clock) -> i8
        "hw.output"(%r) : (i8) -> ()
      }) {sym_name = "Top", parameters = [],
        module_type = !hw.modty<input address : i2, input tick : !seq.clock, output r : i8>} : () -> ()
    }"""
    cut = ConditionalValueCut(selection(text, "operation_result", 1), "memory_read_result")
    prepared = prepare(text, selections=(selection(text),), cuts=(cut,))
    assert len(prepared.inputs) == 1 and prepared.inputs[0].boundary == "memory_read_result"
    assert prepared.evaluate(({prepared.inputs[0].port.name: 0xA7},)) == ({"value_0": 0xA7},)
    members = prepared.record()["source"]["visited_definition_members"][0]["complete_operation_members"]
    assert len(members) == 3 and members[0]["original_attributes"]["readLatency"] == "1 : i32"
    with pytest.raises(ValueError, match="signless integer|noninteger"):
        prepare(text, selections=(selection(text, "operation_result", 0, typ="!seq.firmem<4 x 8>"),), cuts=())


def test_clock_and_noninteger_values_are_not_conditional_known_bit_cuts():
    text = source()
    clock = selection(text, "module_input", 2, typ="!seq.clock")
    with pytest.raises(ValueError, match="signless integer"):
        prepare(text, selections=(clock,), cuts=())


@pytest.mark.parametrize("roster", [(), [], "missing"])
def test_absent_or_non_tuple_selection_roster_refuses(roster):
    with pytest.raises(ValueError, match="rosters"):
        prepare(source(), selections=roster)


@pytest.mark.parametrize(
    "field,value",
    [
        ("modules", 1),
        ("operations", 3),
        ("occurrences", 2),
        ("port_bindings", 1),
        ("nodes", 2),
        ("scalar_bits", 4),
        ("bit_work", 1),
        ("expression_depth", 1),
        ("metadata_bytes", 1),
        ("source_bytes", 16),
    ],
)
def test_complete_structural_bounds_refuse_before_conditional_composition(field, value):
    with pytest.raises(ValueError):
        prepare(source(), binding_limits=replace(BINDINGS, **{field: value}))


@pytest.mark.parametrize("field,value", [("nodes", 1), ("scalar_bits", 4), ("bit_work", 1), ("source_bytes", 16)])
def test_complete_evaluation_bounds_refuse_before_expression_allocation(field, value):
    with pytest.raises(ValueError):
        prepare(source(), evaluation_limits=replace(EVALUATION, **{field: value}))


def test_complete_case_and_bit_work_budgets_and_unsigned_known_bits():
    prepared = prepare(source())
    valid = case(prepared, inputs(0x1234, 1, 5, 9))
    for changed in (
        {},
        {**valid, "extra": 0},
        {**valid, next(iter(valid)): True},
        {**valid, next(iter(valid)): -1},
        {**valid, next(iter(valid)): 1 << prepared.inputs[0].port.width},
    ):
        with pytest.raises(ValueError):
            prepared.evaluate((changed,))
    tight = prepare(source(), evaluation_limits=replace(EVALUATION, cases=1))
    with pytest.raises(ValueError, match="case or bit-work"):
        tight.evaluate((valid, valid))
    work = prepared.expression.per_case_bit_work
    tight = prepare(source(), evaluation_limits=replace(EVALUATION, bit_work=work))
    with pytest.raises(ValueError, match="case or bit-work"):
        tight.evaluate((valid, valid))


def test_unknown_primitive_metadata_is_not_an_acceptable_conditional_cut():
    text = source().replace('"comb.add"(%data, %offset)', '"comb.add"(%data, %offset) {unproved = true}')
    with pytest.raises(ValueError, match="unsupported reachable"):
        prepare(text)


def test_changed_named_hierarchical_binding_refuses():
    text = source().replace('argNames = ["data", "choose", "tick"]', 'argNames = ["choose", "data", "tick"]')
    with pytest.raises(ValueError, match="binding"):
        prepare(text)


def test_original_next_and_reset_operands_are_values_without_a_state_transfer():
    text = """builtin.module {
      "hw.module"() ({
      ^bb0(%tick: !seq.clock, %reset: i1):
        %zero = "hw.constant"() {value = 0 : i4} : () -> i4
        %one = "hw.constant"() {value = 1 : i4} : () -> i4
        %next = "comb.add"(%held, %one) : (i4, i4) -> i4
        %held = "seq.firreg"(%next, %tick, %reset, %zero) {name = "state"}
          : (i4, !seq.clock, i1, i4) -> i4
        "hw.output"(%held) : (i4) -> ()
      }) {sym_name = "Top", parameters = [],
        module_type = !hw.modty<input tick : !seq.clock, input reset : i1, output stored : i4>} : () -> ()
    }"""
    sinks = tuple(
        selection(text, "operation_operand", 3, slot=slot, typ=typ) for slot, typ in ((0, "i4"), (2, "i1"), (3, "i4"))
    ) + (selection(text, typ="i4"),)
    cut = ConditionalValueCut(selection(text, "operation_result", 3, typ="i4"), "state_result")
    prepared = prepare(text, selections=sinks, cuts=(cut,))
    values = {(("Top",), "operation_result", 3): 15, (("Top",), "module_input", 1): 1}
    assert prepared.evaluate((case(prepared, values),)) == ({"value_0": 0, "value_1": 1, "value_2": 0, "value_3": 15},)
    assert prepared.inputs[0].boundary == "state_result"
    assert "initialization_reset_clock_events_and_state_reachability" in prepared.record()["unknowns"]


def test_preparse_source_width_budget_and_selected_evaluation_width_are_distinct():
    text = source().replace(
        '"hw.output"(%a, %b, %join)',
        '%unused = "hw.constant"() {value = 0 : i129} : () -> i129\n        "hw.output"(%a, %b, %join)',
    )
    with pytest.raises(ValueError):
        prepare(text)
    prepared = prepare(text, binding_limits=replace(BINDINGS, scalar_bits=129))
    assert max(row.port.width for row in prepared.inputs) == 16
    assert prepared.expression.limits.scalar_bits == 32
    members = prepared.record()["source"]["visited_definition_members"]
    assert any(
        member["result_types"] == ["i129"] for module in members for member in module["complete_operation_members"]
    )


def test_complete_returned_metadata_exact_boundary_includes_inputs_outputs_and_cuts():
    text = source()
    record = prepare(text).record()
    # Count the whole externally consumable JSON, not just attributes or the
    # structural subrecord. The declaration's own decimal length is included.
    limit = len(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
    for _ in range(8):
        record["source"]["limits"]["metadata_bytes"] = limit
        actual = len(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        if actual == limit:
            break
        limit = actual
    else:
        raise AssertionError("independent complete JSON boundary did not converge")
    prepared = prepare(text, binding_limits=replace(BINDINGS, metadata_bytes=limit))
    assert len(json.dumps(prepared.record(), sort_keys=True, separators=(",", ":")).encode()) == limit
    assert prepared.record()["source"]["cost"]["metadata_bytes"] < limit - 1
    with pytest.raises(ValueError, match="returned record.*metadata byte"):
        prepare(text, binding_limits=replace(BINDINGS, metadata_bytes=limit - 1))
