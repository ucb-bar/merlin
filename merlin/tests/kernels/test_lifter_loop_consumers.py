"""Loop nests, entry states and value domains from the register lifter reach CCA consumers."""

from __future__ import annotations

from merlin.compile.analysis.disassembly import RawInsn
from merlin.compile.analysis.schedule_structure import analyze_normalized_commands
from merlin.kernels import decoded_effects, machine_evidence
from merlin.kernels.decoded_effects import derive_effects
from merlin.kernels.machine_evidence import lifter_call, loop_nest, machine_projection
from merlin.targetgen.rocc.normalized_operands import BoundOperandObserver


def _observer():
    facts = dict(interfaces=[dict(name="register_bundle_layouts", bundles={
        "Packet": dict(width=64, fields={"extent": dict(offset=20, width=5), "slot": dict(offset=0, width=8)})})])
    protocol = dict(version=1, encoding=dict(semantic_class={4: "FETCH"}), roles=dict(classes={
        "FETCH": dict(fields=[dict(name="slot", operand="rs2", bundle="Packet", field="slot")])}))
    return BoundOperandObserver(target="array_fixture", protocol=protocol, facts=facts,
                                raw_isa=dict(CUSTOM_OPCODE=43, FUNCT3=3, FUNCT_CLASS={4: "RAW4"}), word_bits=64)


def _word(funct, rs2=6):
    return f"{funct << 25 | rs2 << 20 | 3 << 12 | 43:08x}"


def _rows():
    # 0: li a5,4   4: FETCH   8: addi a5,a5,-1   12: bnez a5,4   16: ret
    return [
        RawInsn(0, "addi", [], "00400793"),
        RawInsn(4, ".insn", [], _word(4)),
        RawInsn(8, "addi", [], "fff78793"),
        RawInsn(12, "bnez", [], "fe079ce3"),
        RawInsn(16, "ret", [], "00008067"),
    ]


LOOP = dict(header_pc=4, latch_pcs=[12], blocks=[4], nest_depth=1, parent_header_pc=None,
            induction=[dict(reg="a5", init=4, step=-1, status="proved")],
            trip_count=dict(status="proved", value=4))


def _decoder():
    return dict(semantics=_observer(), target="array_fixture",
                structure=dict(instruction_roles={"FETCH": ["operand_load"]}))


def _new_lifter(monkeypatch, *, seen):
    original_facts, original_counts = machine_evidence.register_facts, machine_evidence.finite_control_counts

    def register_facts(rows, *, max_steps=2_000_000, entry_state=None, call_summaries=None):
        seen.append(("facts", entry_state, call_summaries))
        states, analysis = original_facts(rows)
        return states, dict(analysis, loops=[LOOP], call_sites=[])

    def finite_control_counts(rows, *, custom_opcode, max_steps=2_000_000, entry_state=None, call_summaries=None):
        result = dict(original_counts(rows, custom_opcode=custom_opcode))
        result["status"] = "unknown"
        result["dynamic_command_count"] = dict(status="proved", value=4)
        return result

    monkeypatch.setattr(machine_evidence, "register_facts", register_facts)
    monkeypatch.setattr(machine_evidence, "finite_control_counts", finite_control_counts)


def _project(**options):
    model = analyze_normalized_commands([], source_kind="static_disassembly")
    return machine_projection(_rows(), model=model, evidence_id="loops", decoder=_decoder(), **options)


def test_lifter_call_passes_only_accepted_options():
    def old(rows, *, max_steps=1):
        return rows, max_steps

    def new(rows, *, entry_state=None):
        return entry_state

    assert lifter_call(old, [1], entry_state={"a0": 1}) == ([1], 1)
    assert lifter_call(new, [], entry_state={"a0": 1}) == {"a0": 1}


def test_lifter_without_loop_outputs_reports_loops_unknown(monkeypatch):
    original = machine_evidence.register_facts

    def register_facts(rows, *, max_steps=2_000_000):
        states, analysis = original(rows)
        return states, {key: value for key, value in analysis.items() if key not in ("loops", "call_sites")}

    monkeypatch.setattr(machine_evidence, "register_facts", register_facts)
    cca, report = _project(entry_state={"a0": 3})
    assert report["loop_nest"]["status"] == "unknown"
    assert report["entry_state"] is None  # the lifter could not consume it
    assert cca.envelope is None or cca.envelope.calls_in_loop is None
    effects = derive_effects(dict(mechanism_projection=report), {}, evidence_id="loops")
    assert effects["facts"]["loop_nest"]["status"] == "unknown"


def test_proved_loops_reach_cca_and_derived_effects(monkeypatch):
    seen = []
    _new_lifter(monkeypatch, seen=seen)
    cca, report = _project(entry_state={"a0": 3}, call_summaries={64: dict(status="proved")})
    assert seen[0][1] == {"a0": 3} and seen[0][2] == {64: dict(status="proved")}
    nest = report["loop_nest"]
    assert nest["status"] == "proved"
    (level,) = nest["value"]["levels"]
    assert level["header_pc"] == 4 and level["command_sites"] == 1
    assert level["role_sites"] == {"operand_load": 1}
    assert nest["value"]["dynamic_command_count"] == dict(status="proved", value=4)
    assert cca.dispatch.n_dispatches == 4
    assert cca.envelope.calls_in_loop == 0
    assert report["field_resolution"]["field_slots"] == 1
    effects = derive_effects(dict(mechanism_projection=report), {}, evidence_id="loops")
    fact = effects["facts"]["loop_nest"]
    assert fact["status"] == "proved"
    assert fact["value"]["depth"] == 1
    assert fact["value"]["per_level_trip_counts"] == [dict(status="proved", value=4)]
    assert fact["value"]["levels"][0]["induction_steps"] == [dict(reg="a5", step=-1, status="proved")]
    assert fact["value"]["levels"][0]["roles"] == {"operand_load": 1}
    assert fact["evidence"]


def test_loop_nest_places_roles_at_innermost_level():
    outer = dict(LOOP, header_pc=0, blocks=[0, 4], nest_depth=1, trip_count=dict(status="unknown"))
    inner = dict(LOOP, nest_depth=2, parent_header_pc=0)
    nest = loop_nest(dict(status="analyzed", loops=[outer, inner]), _rows(), sites={4: "FETCH"},
                     roles={"FETCH": ["operand_load"]}, evidence_id="e")
    assert nest["status"] == "partial"
    by_header = {level["header_pc"]: level for level in nest["value"]["levels"]}
    assert by_header[4]["command_sites"] == 1 and by_header[0]["command_sites"] == 0
    assert nest["value"]["depth"] == 2


def _structure():
    fields = [dict(name="addr", operand="rs2", bundle="Packet", field="addr")]
    protocol = dict(roles=dict(classes={"LOAD": dict(fields=fields)}, local_address_flags={}),
                    address_field=dict(bundle="Packet", field="addr"))
    return dict(instruction_roles={"LOAD": ["operand_load"]}, operand_protocol=protocol,
                field_layouts=[dict(bundle="Packet", name="addr", offset=0, width=16, status="derived")])


def test_address_windows_use_lifter_value_domains():
    affine = dict(base=16, terms=[dict(header_pc=4, step=8, count=4)])
    commands = [
        dict(index=0, pc=0, instruction="LOAD", fields={"addr": None}, status="decoded",
             operands=dict(rs2=dict(kind="unknown", raw=None, affine=affine))),
        dict(index=1, pc=4, instruction="LOAD", fields={"addr": None}, status="decoded",
             operands=dict(rs2=dict(kind="unknown", raw=None, value_set=[100, 200]))),
        dict(index=2, pc=8, instruction="LOAD", fields={"addr": 7}, status="decoded", operands={}),
    ]
    analysis = dict(mechanism_projection=dict(instruction_classes={"LOAD": 3}))
    fact = derive_effects(analysis, _structure(), commands=commands)["facts"]["local_address_windows"]
    assert fact["status"] == "observed"
    (window,) = fact["value"]["windows"]
    assert (window["min"], window["max"], window["sites"]) == (7, 200, 3)
    assert window["domain_basis"] == ["affine", "value_set"]
    assert fact["scope"]["value_basis"] == {"affine": 1, "exact_site_value": 1, "value_set": 1}


def test_absent_domain_stays_unresolved():
    commands = [dict(index=0, pc=0, instruction="LOAD", fields={"addr": None}, status="decoded",
                     operands=dict(rs2=dict(kind="unknown", raw=None)))]
    analysis = dict(mechanism_projection=dict(instruction_classes={"LOAD": 1}))
    fact = derive_effects(analysis, _structure(), commands=commands)["facts"]["local_address_windows"]
    assert fact["status"] == "unknown"
    assert fact["value"]["unresolved_sites"] == {"LOAD.addr": 1}


def test_decoded_commands_pass_entry_state(monkeypatch):
    seen = []
    from merlin.compile.analysis import riscv_machine

    original = riscv_machine.register_facts

    def register_facts(rows, *, max_steps=2_000_000, entry_state=None, call_summaries=None):
        seen.append(entry_state)
        return original(rows)

    monkeypatch.setattr(riscv_machine, "register_facts", register_facts)
    commands = decoded_effects.decoded_commands(_rows(), decoder=_decoder(), entry_state={"a1": 2})
    assert seen == [{"a1": 2}] and len(commands) == 1
