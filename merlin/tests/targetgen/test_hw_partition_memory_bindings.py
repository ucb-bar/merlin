"""Original occurrence bit relations retain state, read and opaque boundaries."""

import dataclasses

import pytest
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr

from merlin.targetgen.rtl import hw_partition_memory_bindings as B
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits
from merlin.targetgen.rtl.hw_observations import _attribute, _name

LOCAL = MemoryPortLimits(16, 16, 32, 1024, 64, 65536, 64)
HIERARCHY = HierarchyBindingLimits(16, 16, 16, 32, 256, 2048, 64, 65536, 16, 64)
LIMITS = B.PartitionMemoryLimits(512, 128, 64, 128, 65536, 8192, 8192, 8192, 8192, 64)


def _unit(name, arguments, outputs, body):
    args = ",".join("%" + key + ":" + typ for key, typ in arguments)
    ports = ",".join(
        ["input " + key + ":" + typ for key, typ in arguments] + ["output " + key + ":" + typ for key, typ in outputs]
    )
    return (
        '"hw.module"() ({ ^bb0('
        + args
        + "):\n"
        + body
        + '\n}) {sym_name="'
        + name
        + '",module_type=!hw.modty<'
        + ports
        + ">,parameters=[]} : () -> ()\n"
    )


def _source(kind="direct", swapped=False):
    cell = '\n%lo = "comb.extract"(%word) {lowBit=0:i32} : (i8) -> i4\n'
    cell += '%hi = "comb.extract"(%word) {lowBit=4:i32} : (i8) -> i4\n'
    if swapped:
        cell += '%reorder = "comb.concat"(%lo,%hi) : (i4,i4) -> i8\n'
        cell += '%first = "comb.extract"(%reorder) {lowBit=0:i32} : (i8) -> i4\n'
        cell += '%second = "comb.extract"(%reorder) {lowBit=4:i32} : (i8) -> i4\n'
    for index, value in enumerate(("first", "second") if swapped else ("lo", "hi")):
        cell += (
            f'%m{index} = "seq.firmem"() {{name="m{index}",readLatency=0:i32,writeLatency=1:i32,'
            "ruw=0:i32,wuw=1:i32} : () -> !seq.firmem<3 x 4>\n"
            f'"seq.firmem.write_port"(%m{index},%index,%clock,%{value}) '
            "{operandSegmentSizes=array<i32:1,1,1,0,1,0>} : (!seq.firmem<3 x 4>,i2,!seq.clock,i4) -> ()\n"
            f'%r{index} = "seq.firmem.read_port"(%m{index},%index,%clock) '
            ": (!seq.firmem<3 x 4>,i2,!seq.clock) -> i4\n"
        )
    cell += '%seen = "comb.concat"(%r1,%r0) : (i4,i4) -> i8\n'
    cell += '"hw.output"(%seen) : (i8) -> ()'
    text = _unit("Cell", [("clock", "!seq.clock"), ("index", "i2"), ("word", "i8")], [("seen", "i8")], cell)
    if kind == "opaque":
        text += (
            '"hw.module.extern"() {sym_name="Unknown",module_type=!hw.modty<output word:i8>,parameters=[]} : () -> ()\n'
        )
    top = ""
    for name in ("x", "y"):
        for low in (0, 4):
            top += f'%{name}{low} = "comb.extract"(%{name}) {{lowBit={low}:i32}} : (i8) -> i4\n'
    if kind == "state":
        top += '%word = "seq.firreg"(%x,%clock) : (i8,!seq.clock) -> i8\n'
    elif kind == "mux":
        top += '%word = "comb.mux"(%choice,%x,%y) : (i1,i8,i8) -> i8\n'
    elif kind == "opaque":
        top += (
            '%word = "hw.instance"() {instanceName="different",moduleName=@Unknown,argNames=[],resultNames=["word"]} '
            ": () -> i8\n"
        )
    elif kind == "unsupported":
        top += '%word = "comb.divu"(%x,%y) : (i8,i8) -> i8\n'
    else:
        top += '%word = "comb.concat"(%x4,%x0) : (i4,i4) -> i8\n'
    for name, word in (("a", "word"), ("b", "a" if kind == "read" else "y")):
        top += (
            f'%{name} = "hw.instance"(%clock,%index,%{word}) '
            f'{{instanceName="{name}",moduleName=@Cell,argNames=["clock","index","word"],resultNames=["seen"]}} '
            ": (!seq.clock,i2,i8) -> i8\n"
        )
    top += '"hw.output"(%a,%b) : (i8,i8) -> ()'
    text += _unit(
        "Root",
        [("clock", "!seq.clock"), ("index", "i2"), ("choice", "i1"), ("x", "i8"), ("y", "i8")],
        [("first", "i8"), ("second", "i8")],
        top,
    )
    return "builtin.module {\n" + text + "}\n"


def _derive(parsed=None, **changes):
    return B.partition_memory_bindings(
        parse_generic_hw(_source()) if parsed is None else parsed,
        root="Root",
        local_limits=LOCAL,
        hierarchy_limits=HIERARCHY,
        limits=dataclasses.replace(LIMITS, **changes),
    )


def _cell_intervals(record):
    parts = record["local_partitions"]["partitions"]
    return [
        (row["frame"], piece["source_low_bit"], piece["destination_low_bit"], piece["width"])
        for row in record["ports"]
        for piece in row["intervals"]
        if parts[piece["partition"]]["module"] == "Cell"
    ]


def test_complete_port_roster_and_distinct_intermediate_occurrences():
    record = _derive()
    assert len(record["ports"]) == 8
    assert _cell_intervals(record) == [(1, 0, 0, 4), (1, 4, 0, 4), (2, 0, 0, 4), (2, 4, 0, 4)]
    assert sum(row["data_status"] == "no_write_data_operand" for row in record["ports"]) == 4
    assert {row["operation"] for row in record["ports"]} == {"seq.firmem.read_port", "seq.firmem.write_port"}
    assert all(memory["declaration"]["read_under_write"] == "undefined" for memory in record["hierarchy"]["memories"])
    assert all(memory["declaration"]["depth"] == 3 for memory in record["hierarchy"]["memories"])
    assert record["source_values_evaluated"] is False
    assert record["packing_mapping_admission"] is False
    assert record["command_capacity_axis_or_temporal_admission"] is False
    assert "physical_alias_lifetime_order_and_completion" in record["unknowns"]


def test_original_concat_order_changes_exact_intervals_without_default_widening():
    record = _derive(parse_generic_hw(_source(swapped=True)))
    assert _cell_intervals(record) == [(1, 4, 0, 4), (1, 0, 0, 4), (2, 4, 0, 4), (2, 0, 0, 4)]


def test_partial_source_interval_does_not_fill_unproved_destination_bits():
    source = (
        _source()
        .replace(
            '%m0 = "seq.firmem"()',
            '%piece = "comb.extract"(%lo) {lowBit=0:i32} : (i4) -> i2\n'
            '%zero = "hw.constant"() {value=0:i2} : () -> i2\n'
            '%mixed = "comb.concat"(%zero,%piece) : (i2,i2) -> i4\n'
            '%m0 = "seq.firmem"()',
        )
        .replace("%m0,%index,%clock,%lo)", "%m0,%index,%clock,%mixed)")
    )
    record = _derive(parse_generic_hw(source))
    assert _cell_intervals(record) == [(1, 0, 0, 2), (1, 4, 0, 4), (2, 0, 0, 2), (2, 4, 0, 4)]
    nodes = {row["id"]: row for row in record["hierarchy"]["expressions"]}
    assert any(
        nodes[identity].get("expression") == "hw.constant" for row in record["ports"] for identity in row["cuts"]
    )


def test_read_write_port_retains_mode_and_read_result_without_value_credit():
    source = (
        _source()
        .replace('%m0 = "seq.firmem"()', '%mode = "hw.constant"() {value=0:i1} : () -> i1\n%m0 = "seq.firmem"()')
        .replace(
            '"seq.firmem.write_port"(%m0,%index,%clock,%lo) '
            "{operandSegmentSizes=array<i32:1,1,1,0,1,0>} : (!seq.firmem<3 x 4>,i2,!seq.clock,i4) -> ()",
            '%rw = "seq.firmem.read_write_port"(%m0,%index,%clock,%lo,%mode) '
            "{operandSegmentSizes=array<i32:1,1,1,0,1,1,0>} : (!seq.firmem<3 x 4>,i2,!seq.clock,i4,i1) -> i4",
        )
    )
    record = _derive(parse_generic_hw(source))
    assert len(record["ports"]) == 8
    read_writes = [
        port
        for memory in record["hierarchy"]["memories"]
        for port in memory["ports"]
        if port["operation"] == "seq.firmem.read_write_port"
    ]
    assert len(read_writes) == 2
    assert all("mode" in port["bindings"] and port["result_types"] == ["i4"] for port in read_writes)
    assert _cell_intervals(record) == [(1, 0, 0, 4), (1, 4, 0, 4), (2, 0, 0, 4), (2, 4, 0, 4)]
    assert record["source_values_evaluated"] is False


@pytest.mark.parametrize(
    "kind,stop",
    [
        ("state", "state_result"),
        ("opaque", "opaque_instance_result"),
        ("read", "memory_read_result"),
        ("unsupported", "unsupported_result"),
        ("mux", "combinational"),
    ],
)
def test_intermediate_bits_remain_conditional_behind_original_stops(kind, stop):
    record = _derive(parse_generic_hw(_source(kind)))
    nodes = {row["id"]: row for row in record["hierarchy"]["expressions"]}
    assert len(_cell_intervals(record)) == 4
    assert stop in {nodes[identity]["kind"] for row in record["ports"] for identity in row["cuts"]}
    if kind == "mux":
        assert any(
            nodes[identity].get("expression") == "comb.mux" for row in record["ports"] for identity in row["cuts"]
        )
        # Local input partitions survive; root x/y dependency is not a value wire.
        parts = record["local_partitions"]["partitions"]
        first = next(row for row in record["ports"] if row["data_expression"] is not None)
        assert all(parts[piece["partition"]]["module"] == "Cell" for piece in first["intervals"])


def test_parameterized_body_is_retained_as_a_stop_without_memory_port_invention():
    source = _source().replace('sym_name="Cell",', 'sym_name="Cell",oldParameters={arbitrary=7:i32},')
    record = _derive(parse_generic_hw(source))
    assert len(record["hierarchy"]["frames"]) == 3
    assert sum(row["stop"] == "parameterized_body" for row in record["hierarchy"]["frames"]) == 2
    assert record["ports"] == []
    assert record["local_partitions"]["partitions"]
    assert "external_parameterized_and_unsupported_body_semantics" in record["unknowns"]
    assert record["packing_mapping_admission"] is False


@pytest.mark.parametrize("field", ["operations", "extractions", "local_trace_work", "expression_depth"])
def test_complete_wire_budgets_precede_local_partition_expansion(field, monkeypatch):
    monkeypatch.setattr(
        B, "equal_partitions", lambda *args: pytest.fail("denied original source reached local expansion")
    )
    source = _source(swapped=True)
    with pytest.raises(ValueError, match="budget"):
        _derive(parse_generic_hw(source), **{field: 1})


def test_small_shared_dag_cannot_expand_an_exponential_local_trace_tree(monkeypatch):
    body, previous = [], "lo"
    for index in range(16):
        body.extend(
            [
                f'%dup{index} = "comb.concat"(%{previous},%{previous}) : (i4,i4) -> i8',
                f'%tail{index} = "comb.extract"(%dup{index}) {{lowBit=0:i32}} : (i8) -> i4',
            ]
        )
        previous = "tail" + str(index)
    source = _source().replace('%m0 = "seq.firmem"()', "\n".join(body) + '\n%m0 = "seq.firmem"()')
    monkeypatch.setattr(
        B, "equal_partitions", lambda *args: pytest.fail("denied trace tree reached recursive expansion")
    )
    with pytest.raises(ValueError, match="pre-expansion local trace"):
        _derive(parse_generic_hw(source), local_trace_work=1000)


@pytest.mark.parametrize(
    "field", ["partitions", "slices", "root_match_work", "interval_pieces", "cut_memberships", "traversal_work"]
)
def test_complete_join_and_materialization_budgets_refuse(field):
    with pytest.raises(ValueError, match="budget"):
        _derive(**{field: 1})


@pytest.mark.parametrize("field", dataclasses.asdict(LIMITS))
def test_boolean_limits_cannot_replace_integer_budgets(field):
    with pytest.raises(ValueError, match="positive"):
        dataclasses.replace(LIMITS, **{field: True})


@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered", "type"])
def test_original_named_binding_and_type_membership_is_required(change):
    parsed = parse_generic_hw(_source())
    instance = next(op for op in parsed.walk() if _name(op) == "hw.instance")
    if change == "type":
        instance.results[0]._type = IntegerType(7)
    else:
        names = list(_attribute(instance, "argNames").data)
        if change == "missing":
            names.pop()
        elif change == "duplicate":
            names[-1] = names[-2]
        else:
            names.reverse()
        owner = instance.properties if "argNames" in instance.properties else instance.attributes
        owner["argNames"] = ArrayAttr(names)
    with pytest.raises(ValueError, match="binding|declared ports"):
        _derive(parsed)


@pytest.mark.parametrize("change", ["negative", "too_wide", "field_type", "signed", "unknown_field", "duplicate_field"])
def test_malformed_original_extract_cannot_issue_bit_intervals(change):
    parsed = parse_generic_hw(_source())
    op = next(op for op in parsed.walk() if _name(op) == "comb.extract")
    owner = op.properties if "lowBit" in op.properties else op.attributes
    if change == "negative":
        owner["lowBit"] = IntegerAttr(-1, 32)
    elif change == "too_wide":
        owner["lowBit"] = IntegerAttr(6, 32)
    elif change == "field_type":
        owner["lowBit"] = IntegerAttr(0, 64)
    elif change == "signed":
        from xdsl.dialects.builtin import Signedness

        op.results[0]._type = IntegerType(4, Signedness.SIGNED)
    elif change == "unknown_field":
        op.attributes["unselected"] = StringAttr("value")
    else:
        other = op.attributes if owner is op.properties else op.properties
        other["lowBit"] = owner["lowBit"]
    with pytest.raises(ValueError):
        _derive(parsed)


def test_fresh_derivation_observes_source_changes_without_record_upgrade():
    parsed = parse_generic_hw(_source())
    before = _derive(parsed)
    extracts = [op for op in parsed.walk() if _name(op) == "comb.extract"][:2]
    for op, low in zip(extracts, (4, 0), strict=True):
        owner = op.properties if "lowBit" in op.properties else op.attributes
        owner["lowBit"] = IntegerAttr(low, 32)
    after = _derive(parsed)
    assert _cell_intervals(before) != _cell_intervals(after)
    assert _cell_intervals(before) == [(1, 0, 0, 4), (1, 4, 0, 4), (2, 0, 0, 4), (2, 4, 0, 4)]
