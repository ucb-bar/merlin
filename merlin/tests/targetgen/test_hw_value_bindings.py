"""Exact typed source connectivity does not classify getter/event semantics."""

import hashlib
from dataclasses import replace

import pytest

from merlin.targetgen.rtl.hw_value_bindings import (
    OriginalValueSelection,
    ValueBindingLimits,
    prepare_value_bindings,
)

LIMITS = ValueBindingLimits(65536, 16, 512, 32, 256, 16, 256, 128, 4096, 16, 64, 65536)


def source(*, parameterized=False, external=False):
    cell = """"hw.module"() ({
      ^bb0(%data: i8, %choose: i1, %tick: !seq.clock):
        %one = "hw.constant"() {value = 1 : i8} : () -> i8
        %sum = "comb.add"(%data, %one) : (i8, i8) -> i8
        %held = "seq.firreg"(%held, %tick) {name = "stored", firrtl.random_init_start = 0 : ui64}
          : (i8, !seq.clock) -> i8
        %selected = "comb.mux"(%choose, %sum, %held) : (i1, i8, i8) -> i8
        "sv.always"(%tick) ({
          "sv.if"(%choose) ({
            "sim.clocked_terminate"(%tick) {success = false, message = "original effect"}
              : (!seq.clock) -> ()
          }) : (i1) -> ()
        }) : (!seq.clock) -> ()
        "hw.output"(%selected) : (i8) -> ()
      }) {sym_name = "Cell", parameters = [],
          module_type = !hw.modty<input data : i8, input choose : i1, input tick : !seq.clock,
                                 output observed : i8>} : () -> ()"""
    if external:
        cell = """"hw.module.extern"() {sym_name = "Cell", parameters = [],
          module_type = !hw.modty<input data : i8, input choose : i1, input tick : !seq.clock,
                                 output observed : i8>} : () -> ()"""
    params = "[1 : i32]" if parameterized else "[]"
    return f"""builtin.module {{
      {cell}
      "hw.module"() ({{
      ^bb0(%word: i16, %choose: i1, %tick: !seq.clock):
        %lo = "comb.extract"(%word) {{lowBit = 0 : i32}} : (i16) -> i8
        %hi = "comb.extract"(%word) {{lowBit = 8 : i32}} : (i16) -> i8
        %join = "comb.concat"(%hi, %lo) : (i8, i8) -> i16
        %a = "hw.instance"(%lo, %choose, %tick) {{moduleName = @Cell, instanceName = "first",
          argNames = ["data", "choose", "tick"], resultNames = ["observed"], parameters = {params}}}
          : (i8, i1, !seq.clock) -> i8
        %b = "hw.instance"(%hi, %choose, %tick) {{moduleName = @Cell, instanceName = "second",
          argNames = ["data", "choose", "tick"], resultNames = ["observed"], parameters = []}}
          : (i8, i1, !seq.clock) -> i8
        "hw.output"(%a, %b, %join) : (i8, i8, i16) -> ()
      }}) {{sym_name = "Top", parameters = [],
          module_type = !hw.modty<input word : i16, input choose : i1, input tick : !seq.clock,
                                 output a : i8, output b : i8, output reconstructed : i16>}} : () -> ()
    }}"""


def selection(text, kind="module_output", ordinal=0, *, occurrence=("Top",), module="Top", slot=0, typ="i8"):
    return OriginalValueSelection(
        hashlib.sha256(text.encode()).hexdigest(), occurrence, module, kind, ordinal, slot, typ
    )


def observe(text, selections=None, limits=LIMITS):
    if selections is None:
        selections = (selection(text),)
    return prepare_value_bindings(text, root="Top", selections=selections, limits=limits)


def test_exact_instances_follow_combinational_values_but_keep_distinct_states():
    text = source()
    result = observe(text, (selection(text), selection(text, ordinal=1)))
    assert result["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert result["cost"]["occurrences"] == 3
    assert [row["path"] for row in result["frames"]] == [["Top"], ["Top", "first"], ["Top", "second"]]
    states = [row for row in result["expressions"] if row["kind"] == "state_result"]
    assert len(states) == 2 and states[0]["frame"] != states[1]["frame"]
    assert all(row["original_operand_values"][1]["type"] == "!seq.clock" for row in states)
    assert result["admission_authority"] is False
    assert "getter_return_sample_event_custody" in result["unknowns"]


def test_raw_word_extract_concat_path_is_structural_and_reuses_original_expression_reader():
    text = source()
    result = observe(text, (selection(text, ordinal=2, typ="i16"),))
    nodes = {row["id"]: row for row in result["expressions"]}
    sink = nodes[result["selections"][0]["value"]]
    assert sink["expression"] == "comb.concat"
    assert [nodes[index]["parameter"] for index in sink["operands"]] == [8, 0]
    assert [row["kind"] for row in result["expressions"]].count("root_input") == 1
    assert all("semantic_role" not in row for row in result["expressions"])


def test_complete_original_state_unused_effect_and_nested_metadata_membership_retained():
    text = source()
    result = observe(text)
    cell = next(row for row in result["visited_definition_members"] if row["module"] == "Cell")
    assert len(cell["complete_operation_members"]) == cell["operation_count"] == 6
    state = cell["complete_operation_members"][2]
    assert state["operation"] == "seq.firreg"
    assert len(state["operands"]) == 2 and state["result_types"] == ["i8"]
    effect = cell["complete_operation_members"][4]
    assert [row["operation"] for row in effect["nested_operations"]] == ["sv.if", "sim.clocked_terminate"]
    assert effect["nested_operations"][1]["original_attributes"]["message"] == '"original effect"'
    assert effect["nested_operations"][1]["path"] == [4, 0, 0, 0, 0, 0, 0]
    assert effect["nested_operations"][1]["operands"] == [{"kind": "module_input", "ordinal": 2, "type": "!seq.clock"}]
    assert cell["member_semantics"] == "UNKNOWN"


@pytest.mark.parametrize(
    "external,parameterized,stop", [(True, False, "external_body"), (False, True, "parameterized_body")]
)
def test_opaque_body_cut_retains_exact_incoming_metadata_and_types(external, parameterized, stop):
    text = source(external=external, parameterized=parameterized)
    result = observe(text)
    node = result["expressions"][0]
    assert node["kind"] == "opaque_instance_result" and node["stop"] == stop
    binding = result["frames"][1]["incoming_binding"]
    assert [row["type"] for row in binding["operands"]] == ["i8", "i1", "!seq.clock"]
    assert binding["result_types"] == ["i8"]
    assert binding["original_attributes"]["parameters"] == ("[1 : i32]" if parameterized else "[]")


def test_exact_original_operand_selection_retains_clock_cut():
    text = source()
    chosen = selection(
        text, "operation_operand", 2, occurrence=("Top", "first"), module="Cell", slot=1, typ="!seq.clock"
    )
    result = observe(text, (chosen,))
    assert result["expressions"][0]["kind"] == "clock_value"
    assert result["expressions"][0]["type"] == "!seq.clock"
    assert result["frames"][1]["incoming_binding"]["operands"][2]["ordinal"] == 2


@pytest.mark.parametrize("change", ["identity", "path", "module", "ordinal", "slot", "type"])
def test_changed_original_identity_or_endpoint_refuses(change):
    text = source()
    original = selection(text)
    modifications = {
        "identity": {"source_sha256": "0" * 64},
        "path": {"occurrence": ("Top", "absent")},
        "module": {"module": "Cell"},
        "ordinal": {"ordinal": 99},
        "slot": {"kind": "operation_result", "ordinal": 3, "slot": 1},
        "type": {"type": "i16"},
    }
    with pytest.raises(ValueError):
        observe(text, (replace(original, **modifications[change]),))


@pytest.mark.parametrize(
    "replacement",
    [
        ('argNames = ["data", "choose", "tick"]', 'argNames = ["choose", "data", "tick"]'),
        ('resultNames = ["observed"]', 'resultNames = ["different"]'),
        ('instanceName = "second"', 'instanceName = "first"'),
        ("moduleName = @Cell", "moduleName = @Missing"),
    ],
)
def test_complete_original_instance_membership_changes_refuse(replacement):
    text = source().replace(*replacement)
    with pytest.raises(ValueError):
        observe(text)


@pytest.mark.parametrize(
    "field,value",
    [
        ("operations", 2),
        ("modules", 1),
        ("occurrences", 2),
        ("port_bindings", 7),
        ("hierarchy_depth", 1),
        ("nodes", 1),
        ("scalar_bits", 4),
        ("bit_work", 1),
        ("expression_depth", 1),
        ("metadata_bytes", 1),
        ("source_bytes", 16),
    ],
)
def test_complete_original_bounds_refuse_without_partial_products(field, value):
    text = source()
    with pytest.raises(ValueError):
        observe(text, limits=replace(LIMITS, **{field: value}))


def test_unused_malformed_module_binding_is_not_skipped():
    text = source()
    unused = """"hw.module"() ({
      ^bb0(%x: i8):
        %r = "hw.instance"(%x) {moduleName = @Unavailable, instanceName = "unreachable",
          argNames = ["x"], resultNames = ["y"], parameters = []} : (i8) -> i8
        "hw.output"(%r) : (i8) -> ()
      }) {sym_name = "Unused", parameters = [], module_type = !hw.modty<input x : i8, output y : i8>}
      : () -> ()"""
    text = text[: text.rfind("}")] + unused + "}"
    with pytest.raises(ValueError, match="callee|binding"):
        observe(text)


def test_unknown_combinational_metadata_is_retained_as_cut():
    text = source().replace(
        '"comb.extract"(%word) {lowBit = 0 : i32}', '"comb.extract"(%word) {lowBit = 0 : i32, unavailable = true}'
    )
    result = observe(text, (selection(text, ordinal=2, typ="i16"),))
    cut = next(row for row in result["expressions"] if row["kind"] == "unsupported_result")
    assert cut["original_attributes"]["unavailable"] == "true"
    assert cut["original_operand_values"] == [{"kind": "module_input", "ordinal": 0, "type": "i16"}]


@pytest.mark.parametrize(
    "change", [{"source_sha256": True}, {"ordinal": True}, {"slot": False}, {"occurrence": ["Top"]}, {"kind": []}]
)
def test_selection_closed_types_refuse_boolean_substitution(change):
    with pytest.raises(ValueError):
        replace(selection(source()), **change)


def test_duplicate_or_empty_selection_roster_refuses():
    text = source()
    row = selection(text)
    for roster in ((), (row, row), [row]):
        with pytest.raises(ValueError):
            observe(text, roster)


def test_memory_read_remains_cut_with_all_original_operand_identities():
    text = """builtin.module {
      "hw.module"() ({
      ^bb0(%address: i2, %tick: !seq.clock):
        %mem = "seq.firmem"() {readLatency = 1 : i32} : () -> !seq.firmem<4 x 8>
        %r = "seq.firmem.read_port"(%mem, %address, %tick)
          : (!seq.firmem<4 x 8>, i2, !seq.clock) -> i8
        "hw.output"(%r) : (i8) -> ()
      }) {sym_name = "Top", parameters = [],
          module_type = !hw.modty<input address : i2, input tick : !seq.clock, output r : i8>}
          : () -> ()
    }"""
    result = observe(text)
    cut = result["expressions"][0]
    assert cut["kind"] == "memory_read_result"
    assert [v["type"] for v in cut["original_operand_values"]] == ["!seq.firmem<4 x 8>", "i2", "!seq.clock"]
    assert len(result["visited_definition_members"][0]["complete_operation_members"]) == 3


def test_recursive_unused_definition_refuses_before_roster_expansion():
    text = source()
    unused = """"hw.module"() ({
      ^bb0(%x: i8):
        %r = "hw.instance"(%x) {moduleName = @Unused, instanceName = "recursive",
          argNames = ["x"], resultNames = ["y"], parameters = []} : (i8) -> i8
        "hw.output"(%r) : (i8) -> ()
      }) {sym_name = "Unused", parameters = [], module_type = !hw.modty<input x : i8, output y : i8>}
      : () -> ()"""
    text = text[: text.rfind("}")] + unused + "}"
    with pytest.raises(ValueError, match="recursive"):
        observe(text)


def test_unused_duplicate_module_identity_refuses():
    text = source()
    copy = """"hw.module.extern"() {sym_name = "Cell", parameters = [],
      module_type = !hw.modty<input different : i8, output y : i8>} : () -> ()"""
    text = text[: text.rfind("}")] + copy + "}"
    with pytest.raises(ValueError, match="membership"):
        observe(text)
