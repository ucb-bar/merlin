"""Independent typed address registers and actual native Seq-to-SV branches."""

import copy
import dataclasses
import os
from pathlib import Path

import pytest
from merlin_experiments.phase0 import address_transition_intake as T
from merlin_experiments.phase0.hierarchical_memory_intake import issue_independent_hierarchical_memory_intake
from merlin_experiments.phase0.memory_port_intake import issue_independent_memory_port_intake
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal, issue_independent_hardware_intake
from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr, UnitAttr

from merlin.common import invocation_record as I
from merlin.targetgen.rtl.hw_address_transitions import AddressTransitionLimits, address_state_transitions
from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_hierarchy_bindings import HierarchyBindingLimits, _static_signature
from merlin.targetgen.rtl.hw_memory_ports import MemoryPortLimits
from merlin.targetgen.rtl.hw_observations import _attribute, _name
from merlin.targetgen.rtl.source_selection import produce_selection

LOCAL = MemoryPortLimits(16, 16, 32, 1024, 256, 65536, 64)
HIERARCHY = HierarchyBindingLimits(16, 16, 16, 32, 256, 1024, 256, 65536, 16, 128)
LIMITS = AddressTransitionLimits(32, 32, 128, 4096, 128, 1024, 256, 65536, 64)
NATIVE_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def _source(kind):
    text = "FIRRTL version 2.0.0\ncircuit Unit :\n"
    if kind == "instance":
        text += "  extmodule Route :\n    input value : UInt<2>\n    output result : UInt<2>\n    defname = Route\n"
    text += (
        "  module Leaf :\n    input clock : Clock\n    input reset : UInt<1>\n    input consent : UInt<1>\n"
        "    input index : UInt<2>\n    input data : UInt<8>\n    output seen : UInt<8>\n"
    )
    if kind == "no_reset":
        text += "    reg held : UInt<2>, clock\n"
    else:
        text += "    reg held : UInt<2>, clock with :\n      reset => (reset, UInt<2>(1))\n"
    text += (
        "    mem storage :\n      data-type => UInt<8>\n      depth => 3\n"
        "      read-latency => 1\n      write-latency => 1\n      reader => r\n      writer => w\n"
        "      read-under-write => undefined\n"
        "    storage.r.addr <= held\n    storage.r.en <= consent\n    storage.r.clk <= clock\n"
        "    storage.w.addr <= held\n    storage.w.en <= consent\n    storage.w.clk <= clock\n"
        "    storage.w.data <= data\n    storage.w.mask <= consent\n    seen <= storage.r.data\n"
    )
    value = "index"
    if kind == "memory":
        value = "bits(storage.r.data, 1, 0)"
    elif kind == "unsupported":
        value = "bits(div(data, pad(index, 8)), 1, 0)"
    elif kind == "instance":
        text += "    inst route of Route\n    route.value <= index\n"
        value = "route.result"
    elif kind == "state":
        text += "    reg other : UInt<2>, clock\n    other <= index\n"
        value = "bits(add(held, other), 1, 0)"
    arms = f"held, {value}" if kind == "hold_true" else f"{value}, held"
    text += f"    held <= mux(consent, {arms})\n"
    text += (
        "  module Unit : @[generators/test_unit/src/IndependentTransfer.scala 1:1]\n"
        "    input clock : Clock\n    input reset : UInt<1>\n    input consent : UInt<1>\n"
        "    input index : UInt<2>\n    input data : UInt<8>\n    output seen : UInt<8>\n"
    )
    for name in ("first", "second"):
        text += (
            f"    inst {name} of Leaf\n    {name}.clock <= clock\n    {name}.reset <= reset\n"
            f"    {name}.consent <= consent\n    {name}.index <= index\n    {name}.data <= data\n"
        )
    text += "    seen <= xor(first.seen, second.seen)\n"
    if kind == "implicit_enable":
        text = text.replace("storage.r.en <= consent", "storage.r.en <= UInt<1>(1)")
    return text


@pytest.fixture
def source(tmp_path, request):
    if not all(os.environ.get(key) for key in ("MERLIN_TEST_FIRTOOL", "MERLIN_TEST_CIRCT_OPT")):
        pytest.skip("transition controls require an explicitly selected coherent native CIRCT pair")
    fir = tmp_path / "minimal.fir"
    fir.write_text(_source(getattr(request, "param", "hold_false")))
    bundle = produce_selection(
        target="test_unit",
        firrtl=fir,
        generator="test_unit",
        config="IndependentTransfer",
        core_root="Unit",
        firtool=Path(os.environ["MERLIN_TEST_FIRTOOL"]),
        output=tmp_path / "original",
    )
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: test_unit\n")
    forbidden = (tmp_path / "excluded-answers",)
    hardware = issue_independent_hardware_intake(
        target="test_unit",
        descriptor=descriptor,
        source_bundle=bundle,
        forbidden_roots=forbidden,
        output=tmp_path / "hardware",
    )
    memory = issue_independent_memory_port_intake(
        hardware=hardware,
        circt_opt=Path(os.environ["MERLIN_TEST_CIRCT_OPT"]),
        source_bytes=1048576,
        limits=LOCAL,
        forbidden_roots=forbidden,
        output=tmp_path / "memory",
    )
    hierarchy = issue_independent_hierarchical_memory_intake(
        memory=memory,
        source_bytes=1048576,
        limits=HIERARCHY,
        forbidden_roots=forbidden,
        output=tmp_path / "hierarchy",
    )
    return hierarchy, forbidden


def _parsed(source):
    pin = next(pin for pin in source[0].memory.source_pins if pin.role == "generic-core-hw")
    return parse_generic_hw(Path(pin.path).read_text(), reject_dense_literals=True)


def _derive(parsed, limits=LIMITS):
    return address_state_transitions(
        parsed,
        root="Unit",
        local_limits=LOCAL,
        hierarchy_limits=HIERARCHY,
        limits=limits,
    )


def _issue(source, tmp_path):
    return T.issue_independent_address_transition_intake(
        hierarchy=source[0],
        source_bytes=1048576,
        limits=LIMITS,
        forbidden_roots=source[1],
        output=tmp_path / "transfers",
    )


@pytest.mark.parametrize(
    "source,reset,holds",
    [("hold_false", "synchronous", [False]), ("hold_true", "synchronous", [True]), ("no_reset", "absent", [False])],
    indirect=["source"],
)
def test_native_exact_typed_reset_and_same_ssa_hold_roster(source, reset, holds, tmp_path):
    before = source[0].record()
    facts = _issue(source, tmp_path).record()["facts"]
    assert facts["cost"]["address_bindings"] == 4
    assert facts["cost"]["state_occurrences"] == 2
    assert facts["cost"]["address_state_bindings"] == 4
    assert len({row["frame"] for row in facts["state_transfers"]}) == 2
    assert len(facts["source_frames"]) == 3
    assert all(row["declaration"]["read_under_write"] == "undefined" for row in facts["memory_declarations"])
    assert len(facts["address_stop_expressions"]) == 2
    for transfer in facts["state_transfers"]:
        assert transfer["reset_kind"] == reset
        assert transfer["operand_types"] == ["i2", "!seq.clock", *(["i1", "i2"] if reset == "synchronous" else [])]
        assert transfer["hold_mux"]["hold_when_condition"] == holds
        assert transfer["relation"]["evaluated"] is False
        assert transfer["initialization"] == "unproved"
    for address in facts["addresses"]:
        assert address["declared_depth"] == 3 and address["address_type"] == "i2"
        assert address["range_obligation"]["proved"] is False
        assert address["enable_expression"] is not None
    assert facts["clock_events_evaluated"] is False
    assert facts["temporal_or_command_capacity_axis_admission"] is False
    assert source[0].record() == before and not source[1][0].exists()


@pytest.mark.parametrize(
    "source,kind",
    [
        ("memory", "memory_read_result"),
        ("instance", "instance_result"),
        ("unsupported", "unsupported_result"),
        ("state", "state_result"),
    ],
    indirect=["source"],
)
def test_native_next_expression_preserves_semantic_stops(source, kind, tmp_path):
    facts = _issue(source, tmp_path).record()["facts"]
    assert kind in {row["kind"] for row in facts["expressions"]}
    assert all(row["relation"]["evaluated"] is False for row in facts["state_transfers"])
    assert "enabled_address_range_and_undefined_branch_validity" in facts["unknowns"]


@pytest.mark.parametrize("source", ["implicit_enable"], indirect=True)
def test_native_original_omitted_read_enable_preserves_primitive_default(source, tmp_path):
    facts = _issue(source, tmp_path).record()["facts"]
    read = [row for row in facts["addresses"] if row["operation"] == "seq.firmem.read_port"]
    assert len(read) == 2
    assert all(row["enable_expression"] is None and row["implicit_enable"] == "true" for row in read)
    assert all(row["range_obligation"]["proved"] is False for row in read)


@pytest.mark.parametrize("source", ["hold_false", "no_reset"], indirect=True)
def test_native_seq_to_sv_actual_reset_branch_and_hold_assignment(source, tmp_path):
    generic = next(pin for pin in source[0].memory.source_pins if pin.role == "generic-core-hw")
    lowered = tmp_path / "lowered.mlir"
    result = I.run(
        [
            os.environ["MERLIN_TEST_CIRCT_OPT"],
            generic.path,
            "--lower-seq-to-sv=disable-reg-randomization=true",
            "--mlir-print-op-generic",
            "-o",
            str(lowered),
        ],
        directory=tmp_path / "native-lowering",
        stage="independent_seq_transfer_lowering",
        inputs=(Path(generic.path),),
        outputs=(lowered,),
        cwd=tmp_path,
        env=NATIVE_ENV,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for record in (tmp_path / "native-lowering" / "invocations").glob("*/invocation.json"):
        I.require_environment(record, environment=NATIVE_ENV)
    parsed = parse_generic_hw(lowered.read_text(), reject_dense_literals=True)
    assert not any(_name(op) == "seq.firreg" for op in parsed.walk())
    regs = [op for op in parsed.walk() if _name(op) == "sv.reg" and _attribute(op, "name") == StringAttr("held")]
    assert len(regs) == 1
    assignments = [op for op in parsed.walk() if _name(op) == "sv.passign" and op.operands[0].owner is regs[0]]
    # The original shared definition lowers once; repeated occurrences stay separate in our record.
    assert len(assignments) in {1, 2}
    block = regs[0].parent
    inputs = dict(
        zip(
            (port.name for port in _static_signature(regs[0].parent_op()) if port.direction == "input"),
            block.args,
            strict=True,
        )
    )
    clock, consent, index = (inputs[key] for key in ("clock", "consent", "index"))
    always = next(op for op in block.ops if _name(op) == "sv.always")
    assert tuple(always.operands) == (clock,)
    update_assignment = next(op for op in assignments if op.operands[1] is index)
    update_condition = update_assignment.parent_op()
    assert _name(update_condition) == "sv.if" and tuple(update_condition.operands) == (consent,)
    assert update_assignment.parent is update_condition.regions[0].block
    assert not list(update_condition.regions[1].block.ops)
    facts = _issue(source, tmp_path).record()["facts"]
    reset_kind = facts["state_transfers"][0]["reset_kind"]
    if reset_kind == "synchronous":
        assert len(assignments) == 2
        reset_assignment = next(op for op in assignments if _name(op.operands[1].owner) == "hw.constant")
        assert _attribute(reset_assignment.operands[1].owner, "value") == IntegerAttr(1, IntegerType(2))
        assert _name(reset_assignment.parent_op()) == "sv.if"
        assert reset_assignment.parent is reset_assignment.parent_op().regions[0].block
        reset_condition = reset_assignment.parent_op()
        assert tuple(reset_condition.operands) == (inputs["reset"],)
        assert reset_condition.parent is always.regions[0].block
        assert update_condition.parent is reset_condition.regions[1].block
    else:
        assert len(assignments) == 1
        assert update_condition.parent is always.regions[0].block


@pytest.mark.parametrize(
    "change",
    [
        "missing_operand",
        "extra_operand",
        "next_width",
        "reset_width",
        "reset_value_width",
        "clock_width",
        "reset_annotation",
        "duplicate_name",
        "preset_width",
    ],
)
def test_malformed_original_firreg_roster_or_types_refuse(source, change, tmp_path):
    parsed = _parsed(source)
    op = next(op for op in parsed.walk() if _name(op) == "seq.firreg")
    values = list(op.operands)
    if change == "missing_operand":
        op.operands = values[:3]
    elif change == "extra_operand":
        op.operands = [*values, values[3]]
    elif change in {"next_width", "reset_width", "reset_value_width", "clock_width"}:
        slot = {"next_width": 0, "reset_width": 2, "reset_value_width": 3, "clock_width": 1}[change]
        values[slot]._type = IntegerType(7)
    elif change == "reset_annotation":
        op.attributes["isAsync"] = StringAttr("guessed")
    elif change == "duplicate_name":
        owner = op.properties if "name" in op.attributes else op.attributes
        owner["name"] = _attribute(op, "name")
    else:
        op.attributes["preset"] = IntegerAttr(1, IntegerType(7))
    with pytest.raises(ValueError):
        _derive(parsed)
    if change in {"next_width", "reset_width", "reset_value_width", "clock_width", "preset_width"}:
        malformed = tmp_path / "malformed.mlir"
        malformed.write_text(str(parsed))
        result = I.run(
            [
                os.environ["MERLIN_TEST_CIRCT_OPT"],
                str(malformed),
                "--canonicalize",
                "--verify-each",
                "--mlir-print-op-generic",
            ],
            directory=tmp_path / "native-refusal",
            stage="independent_malformed_seq_transfer_refusal",
            inputs=(malformed,),
            cwd=tmp_path,
            env=NATIVE_ENV,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and result.stderr


def test_async_and_preset_declarations_do_not_establish_events_or_initialization(source):
    parsed = _parsed(source)
    op = next(op for op in parsed.walk() if _name(op) == "seq.firreg")
    op.attributes["isAsync"] = UnitAttr()
    op.attributes["preset"] = IntegerAttr(2, IntegerType(2))
    facts = _derive(parsed)
    assert all(
        row["stop"] == "asynchronous_event_relation_unsupported" and row["relation"] is None
        for row in facts["state_transfers"]
    )
    assert all(row["initialization"] == "unproved" for row in facts["state_transfers"])


@pytest.mark.parametrize("source", ["no_reset"], indirect=True)
def test_async_annotation_without_original_reset_operands_refuses(source):
    parsed = _parsed(source)
    op = next(op for op in parsed.walk() if _name(op) == "seq.firreg")
    op.attributes["isAsync"] = UnitAttr()
    with pytest.raises(ValueError, match="exact reset operands"):
        _derive(parsed)


def test_other_original_state_primitive_is_retained_as_unsupported(source):
    parsed = _parsed(source)
    op = next(op for op in parsed.walk() if _name(op) == "seq.firreg")
    op.attributes["op_name__"] = StringAttr("seq.compreg")
    facts = _derive(parsed)
    assert facts["cost"]["address_bindings"] == 4 and facts["cost"]["state_occurrences"] == 2
    assert all(row["stop"] == "unsupported_state_primitive" for row in facts["state_transfers"])
    assert all(row["relation"] is None for row in facts["state_transfers"])


@pytest.mark.parametrize(
    "field",
    [
        "address_bindings",
        "state_occurrences",
        "address_state_bindings",
        "traversal_steps",
        "operand_bindings",
        "nodes",
        "bit_work",
        "expression_depth",
    ],
)
def test_complete_roster_and_local_expression_budgets_refuse(source, field):
    with pytest.raises(ValueError, match="budget"):
        _derive(_parsed(source), dataclasses.replace(LIMITS, **{field: 1}))


def test_local_transfer_width_requires_its_own_explicit_budget(source):
    with pytest.raises(ValueError, match="bounded scalar type"):
        _derive(_parsed(source), dataclasses.replace(LIMITS, scalar_bits=1))


def test_saved_fact_substitution_and_copy_cannot_mint_live_authority(source, tmp_path):
    intake = _issue(source, tmp_path)
    original = intake.record()
    replaced = copy.deepcopy(original)
    replaced["facts"]["clock_events_evaluated"] = 0
    with pytest.raises(RtlIntakeRefusal, match="complete original"):
        T.verify_record(replaced)
    replaced = copy.deepcopy(original)
    replaced["facts"]["cost"]["nodes"] = float(replaced["facts"]["cost"]["nodes"])
    with pytest.raises(RtlIntakeRefusal, match="complete original"):
        T.verify_record(replaced)
    replaced = copy.deepcopy(original)
    replaced["facts"]["addresses"].pop()
    with pytest.raises(RtlIntakeRefusal, match="complete original"):
        T.verify_record(replaced)
    with pytest.raises(RtlIntakeRefusal, match="live"):
        copy.copy(intake).verify()


def test_preparse_byte_admission_precedes_parser(source, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("byte admission must precede parsing")

    monkeypatch.setattr(T, "parse_generic_hw", forbidden)
    with pytest.raises(RtlIntakeRefusal, match="preparse"):
        T.issue_independent_address_transition_intake(
            hierarchy=source[0],
            source_bytes=1,
            limits=LIMITS,
            forbidden_roots=source[1],
            output=tmp_path / "denied",
        )
    assert not (tmp_path / "denied").exists()
