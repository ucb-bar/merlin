"""Proved issue-order command streams and the facts derived from them, on synthetic data."""

from __future__ import annotations

from merlin.compile.analysis.disassembly import RawInsn
from merlin.compile.analysis.riscv_machine import finite_control_counts
from merlin.kernels.decoded_effects import decode_stream, derive_effects, stream_facts

# A neutral synthetic target: class names, field names and widths are test data only.
ROLES = {
    "LD_A": ["operand_load"],
    "LD_B": ["operand_load"],
    "STAGE": ["weight_load"],
    "MAC": ["accumulate"],
    "CFG": ["config"],
    "SYNC": ["sync"],
    "OUT": ["readout", "commit"],
}
LOAD_FIELDS = [
    {"as": "operand", "name": "src", "operand": "rs1"},
    {"bundle": "LdRs2", "field": "nr", "name": "r", "operand": "rs2"},
    {"bundle": "LdRs2", "field": "nc", "name": "c", "operand": "rs2"},
    {"bundle": "LdRs2", "field": "laddr", "name": "at", "operand": "rs2"},
]
STRUCTURE = {
    "instruction_roles": ROLES,
    "operand_protocol": {
        "address_field": {"bundle": "LdRs2", "field": "laddr"},
        "roles": {
            "classes": {
                "LD_A": {"fields": LOAD_FIELDS},
                "LD_B": {"fields": LOAD_FIELDS},
                "OUT": {"fields": LOAD_FIELDS},
                "STAGE": {"fields": [{"bundle": "StRs", "field": "laddr", "name": "w", "operand": "rs1"}]},
                "MAC": {"fields": [{"bundle": "MRs", "field": "laddr", "name": "a", "operand": "rs1"}]},
            }
        },
    },
    "load_state_selection": {
        "transfer_bound": {"rows": {"bundle": "LdRs2", "name": "nr"}, "cols": {"bundle": "LdRs2", "name": "nc"}}
    },
}


def ptr(arg, offset):
    return {"kind": "unknown", "raw": None, "pointer_base": f"arg{arg}", "byte_offset": offset}


class Stream:
    def __init__(self):
        self.entries = []

    def add(self, klass, **fields):
        self.entries.append({"seq": len(self.entries), "class": klass, "roles": ROLES[klass], "fields": fields})

    def fence(self):
        self.entries.append({"seq": len(self.entries), "class": "FENCE", "roles": ["sync"], "fields": {}})

    def load(self, klass, arg, offset, at, rows=4, cols=8):
        self.add(klass, src=ptr(arg, offset), r=rows, c=cols, at=at)

    def compute(self, n):
        for _ in range(n):
            self.add("STAGE", w=0)
            self.add("MAC", a=0)


def double_buffered(interleaved, *, phases=4, loads=4, computes=8):
    """Weight tiles alternate between two windows; interleaved issue puts loads between computes."""
    s = Stream()
    for i in range(2):
        s.load("LD_B", 1, 64 * i, 900 + 4 * i, rows=4, cols=4)
    s.fence()
    for i in range(loads):
        s.load("LD_A", 0, 32 * i, 4 * i)
    s.fence()
    for phase in range(1, phases):
        window = 100 * (phase % 2)
        if interleaved:
            for i in range(loads):
                s.compute(computes // loads)
                s.load("LD_A", 0, 1000 * phase + 32 * i, window + 4 * i)
        else:
            s.compute(computes)
            for i in range(loads):
                s.load("LD_A", 0, 1000 * phase + 32 * i, window + 4 * i)
        s.fence()
    s.compute(computes)
    return s.entries


def facts(entries, **options):
    return derive_effects({}, STRUCTURE, stream=entries, **options)["facts"]


def test_interleaved_and_blocked_issue_differ_in_run_structure_and_overlap():
    inter = facts(double_buffered(True))["issue_overlap"]
    block = facts(double_buffered(False))["issue_overlap"]
    assert inter["status"] == block["status"] == "observed"
    assert inter["scope"] == "proved_dynamic_command_stream"
    assert inter["value"]["median_compute_run"] == 2 and inter["value"]["median_load_run"] == 1
    assert block["value"]["median_compute_run"] == 8 and block["value"]["median_load_run"] == 4
    assert inter["value"]["by_operand"]["a0"]["loads_with_compute_before_barrier"] == 0.5625
    assert block["value"]["by_operand"]["a0"]["loads_with_compute_before_barrier"] == 0.0
    assert block["value"]["by_operand"]["a1"]["loads_without_compute_before_barrier"] == 1.0
    assert inter["value"]["fences"] == block["value"]["fences"] == 5
    assert inter["value"]["computes"] == 32 and inter["value"]["computes_per_fence"] == 6.4
    assert block["value"]["runs_prefix"][:4] == [["load", 2], ["barrier", 1], ["load", 4], ["barrier", 1]]
    assert inter["value"]["loads_per_compute_run"] == 0.5
    assert inter["value"]["median_compute_run_between_load_bands"] == 2
    assert block["value"]["loads_per_compute_run"] == 0.5
    assert block["value"]["median_compute_run_between_load_bands"] == 8


def test_output_group_extent_per_barrier_block():
    s = Stream()
    for group in range(2):
        s.compute(4)
        for i in range(group + 2):
            s.add("OUT", src=ptr(2, 512 * group + 16 * i), r=4, c=4, at=i)
        s.fence()
    fact = facts(s.entries)["output_group_extent"]
    assert fact["status"] == "observed"
    first, second = fact["value"]["blocks_prefix"]
    assert first["readouts"] == 2 and first["column_extent_if_one_row_band"] == 8 and first["row_extent"] == 4
    assert second["readouts"] == 3 and second["column_extent_if_one_row_band"] == 12
    assert second["memory_byte_span"] == 32 and first["computes"] == 4
    assert fact["value"]["column_extents"] == {8: 1, 12: 1}
    assert facts(double_buffered(True))["output_group_extent"]["status"] == "unknown"


def test_two_alternating_windows_versus_one_reused_window():
    structure = dict(STRUCTURE, local_geometry={"uniform_row_elements": 8})

    def read(window):
        for i in range(4):
            stage = {"seq": 0, "class": "STAGE", "roles": ROLES["STAGE"], "fields": {"w": window + 4 * i}}
            yield stage
            yield {"seq": 0, "class": "MAC", "roles": ROLES["MAC"], "fields": {"a": 900}}

    s = Stream()
    for i in range(4):
        s.load("LD_A", 0, 32 * i, 4 * i)
    s.fence()
    for phase in range(1, 4):
        consumed, filled = 100 * ((phase - 1) % 2), 100 * (phase % 2)
        for i, (stage, mac) in enumerate(zip(*[iter(read(consumed))] * 2, strict=True)):
            s.entries += [stage, mac]
            s.load("LD_A", 0, 1000 * phase + 32 * i, filled + 4 * i)
            s.fence()  # fences between every pair: they must not shape the windows
    s.entries += list(read(100))
    two = derive_effects({}, structure, stream=s.entries)["facts"]["operand_buffering"]["value"]["operand_load"]["a0"]
    assert two["definition"] == "consumption_rounds_v2"
    assert (two["disjoint_windows"], two["alternating"]) == (2, True)
    assert two["window_sequence_prefix"] == [0, 1, 0, 1]
    s = Stream()
    for phase in range(3):
        for i in range(4):
            s.load("LD_A", 0, 1000 * phase + 32 * i, 4 * i)
        s.fence()
        s.entries += list(read(0))
    one = derive_effects({}, structure, stream=s.entries)["facts"]["operand_buffering"]["value"]["operand_load"]["a0"]
    assert (one["disjoint_windows"], one["alternating"]) == (1, False)
    assert one["distinct_base_addresses"] == 4 and one["fill_rounds"] == 3
    # Without derived local row geometry the footprints, and so the windows, stay unknown.
    unknown = facts(s.entries)["operand_buffering"]["value"]["operand_load"]["a0"]
    assert unknown["status"] == "unknown" and unknown["disjoint_windows"] is None


def test_per_role_and_per_operand_widths_keep_local_staging_separate():
    entries = double_buffered(False)
    width = facts(entries)["role_transfer_width"]
    value = width["value"]
    assert width["status"] == "observed"
    loads = value["data_movement"]["operand_load"]
    assert loads["transfers"] == 18 and loads["distinct_shapes"] == 2
    assert loads["by_operand"]["a0"]["shapes"] == [dict(rows=4, cols=8, count=16, elements=32)]
    assert loads["by_operand"]["a1"]["shapes"] == [dict(rows=4, cols=4, count=2, elements=16)]
    assert "weight_load" not in value["data_movement"]
    assert value["local_staging"] == {"weight_load": {"issues": 32, "classes": {"STAGE": 32}}}
    assert value["operand_basis"] == {"symbolic_argument_pointer": 18}


def test_exact_addresses_tie_to_entry_arguments_or_stay_unattributed():
    s = Stream()
    s.add("LD_A", src={"kind": "const", "raw": 0x2040}, r=4, c=8, at=0)
    s.add("LD_A", src={"kind": "const", "raw": 0x1010}, r=4, c=4, at=8)
    tied = facts(s.entries, arguments={"a0": 0x2000, "a1": 0x1000})["role_transfer_width"]["value"]
    assert set(tied["data_movement"]["operand_load"]["by_operand"]) == {"a0", "a1"}
    assert tied["operand_basis"] == {"nearest_entry_argument_pointer_at_or_below": 2}
    alone = facts(s.entries)["role_transfer_width"]["value"]
    assert set(alone["data_movement"]["operand_load"]["by_operand"]) == {"unattributed"}


def test_no_stream_leaves_each_fact_explicitly_unknown_and_truncation_is_partial():
    block = derive_effects({}, STRUCTURE)
    for name in ("role_transfer_width", "operand_buffering", "issue_overlap", "output_group_extent"):
        assert block["facts"][name]["status"] == "unknown"
        assert block["facts"][name]["reason"] == "no_proved_command_stream"
        assert {"fact": name, "reason": "no_proved_command_stream"} in block["unknowns"]
    assert stream_facts([], STRUCTURE)["issue_overlap"]["status"] == "unknown"
    partial = facts(double_buffered(True), stream_truncated=True)["issue_overlap"]
    assert partial["status"] == "partial" and partial["stream"]["truncated"] is True


class _Observer:
    """Synthetic observer: funct 1 is a load, 2 a compute; fields straight from the operands."""

    @staticmethod
    def isa_constants(target):
        return {"CUSTOM_OPCODE": 123}

    @staticmethod
    def decode_instruction(funct, rs1, rs2, isa):
        if funct == 1:
            return "LD_A", {"src": rs1, "r": 4, "c": 8, "at": rs2.get("raw")}
        return "MAC", {"a": rs1.get("raw")}


def _row(pc, word):
    return RawInsn(pc, "ignored_objdump_alias", [], f"{word:08x}")


def _addi(rd, rs, imm):
    return ((imm & 4095) << 20) | (rs << 15) | (rd << 7) | 0x13


def _custom(funct, rs1, rs2):
    return (funct << 25) | (rs2 << 20) | (rs1 << 15) | (3 << 12) | 123


def test_recorded_stream_is_ordered_exact_and_absent_by_default():
    rows = [
        _row(0, _addi(11, 0, 7)),
        _row(4, _custom(1, 10, 11)),
        _row(8, 0x0FF0000F),
        _row(12, _custom(2, 11, 0)),
        _row(16, _addi(10, 10, 64)),
        _row(20, _custom(1, 10, 11)),
        _row(24, 0x00008067),
    ]
    plain = finite_control_counts(rows, custom_opcode=123)
    recorded = finite_control_counts(rows, custom_opcode=123, record_commands=True)
    assert "command_stream" not in plain
    assert {k: v for k, v in recorded.items() if not k.startswith("command_stream")} == plain
    stream = recorded["command_stream"]
    assert recorded["command_stream_truncated"] is False
    assert [entry.get("kind") or entry["funct"] for entry in stream] == [1, "fence", 2, 1]
    assert stream[0]["rs2"] == 7 and stream[2]["rs1"] == 7
    assert stream[0]["rs1"]["pointer_base"] == "arg0" and stream[3]["rs1"]["byte_offset"] == 64
    assert [entry["seq"] for entry in stream] == [0, 1, 2, 3] and stream[1]["pc"] == 8
    assert [entry["host_instructions_since_prev"] for entry in stream] == [1, 0, 1, 1]
    cut = finite_control_counts(rows, custom_opcode=123, record_commands=True, max_recorded=2)
    assert len(cut["command_stream"]) == 2 and cut["command_stream_truncated"] is True
    decoded = decode_stream(stream, decoder={"semantics": _Observer(), "structure": {"instruction_roles": ROLES}})
    assert [e["class"] for e in decoded] == ["LD_A", "FENCE", "MAC", "LD_A"]
    assert decoded[1]["roles"] == ["sync"] and decoded[0]["roles"] == ["operand_load"]
    assert decoded[0]["fields"]["at"] == 7 and decoded[3]["host_instructions_since_prev"] == 1
    width = facts(decoded)["role_transfer_width"]["value"]["data_movement"]["operand_load"]
    assert width["by_operand"]["a0"]["transfers"] == 2
    unproved = finite_control_counts(rows[:3], custom_opcode=123, record_commands=True)
    assert unproved["status"] == "unknown" and "command_stream" not in unproved


def test_optional_stream_effects_module_facts_are_merged(monkeypatch):
    import sys
    import types

    module = types.ModuleType("merlin.kernels.stream_effects")
    module.derive = lambda stream, structure: {
        "extra_fact": {"status": "observed", "value": len(stream)},
        "issue_overlap": {"status": "observed", "value": "shadow"},
    }
    import merlin.kernels as kernels

    monkeypatch.setitem(sys.modules, "merlin.kernels.stream_effects", module)
    # A real stream_effects imported by an earlier test is also a package attribute; patch both.
    monkeypatch.setattr(kernels, "stream_effects", module, raising=False)
    merged = facts(double_buffered(True))
    assert merged["extra_fact"]["value"] == len(double_buffered(True))
    assert merged["issue_overlap"]["value"] != "shadow"
    assert "extra_fact" not in derive_effects({}, STRUCTURE)["facts"]


def _bne_zero(pc, target, rs1):
    imm = (target - pc) & 8191
    return (
        (imm >> 12) << 31
        | ((imm >> 5) & 63) << 25
        | rs1 << 15
        | 1 << 12
        | ((imm >> 1) & 15) << 8
        | ((imm >> 11) & 1) << 7
        | 0x63
    )


def _loop_rows(trips):
    """x11 = trips; loop: custom(rs1=a0, rs2=x12); a0 += 64; x12 += 1; x11 -= 1; bnez x11 loop; ret."""
    return [
        _row(0, _addi(11, 0, trips)),
        _row(4, _addi(12, 0, 5)),
        _row(8, _custom(1, 10, 12)),
        _row(12, _addi(10, 10, 64)),
        _row(16, _addi(12, 12, 1)),
        _row(20, _addi(11, 11, -1)),
        _row(24, _bne_zero(24, 8, 11)),
        _row(28, 0x00008067),
    ]


def test_affine_runs_compress_long_streams_and_expand_exactly():
    from merlin.compile.analysis.riscv_machine import expand_command_stream

    counted = finite_control_counts(_loop_rows(50), custom_opcode=123, record_commands=True, max_recorded=10)
    assert counted["status"] == "proved" and counted["command_stream_entries"] == 50
    assert len(counted["command_stream_runs"]) == 1 and counted["command_stream_runs_complete"] is True
    assert len(counted["command_stream"]) == 10 and counted["command_stream_truncated"] is True
    full = list(expand_command_stream(counted["command_stream_runs"]))
    assert [e["rs2"] for e in full] == list(range(5, 55))
    assert [e["rs1"]["byte_offset"] for e in full[:3]] == [0, 64, 128]
    assert full[0]["host_instructions_since_prev"] == 2 and full[1]["host_instructions_since_prev"] == 4
    assert full[:10] == counted["command_stream"]
    decoded = decode_stream(counted["command_stream_runs"], decoder={"semantics": _Observer()}, limit=20)
    assert len(decoded) == 20 and decoded[19]["fields"]["at"] == 24


def test_step_budget_exhaustion_is_an_explicit_unknown():
    counted = finite_control_counts(_loop_rows(50), custom_opcode=123, record_commands=True, max_steps=40)
    assert counted["status"] == "unknown" and "command_stream" not in counted
    assert counted["command_stream_status"] == {
        "status": "unknown",
        "reason": "step_budget_exhausted",
        "steps_reached": 40,
    }
    assert "command_stream_status" not in finite_control_counts(_loop_rows(50), custom_opcode=123, max_steps=40)


def test_calls_without_selected_words_may_be_assumed_to_return_only_on_request():
    jal_zero = ((0x40 >> 1) & 1023) << 21 | 0x6F  # j +0x40: a tail transfer out of the span
    rows = [_row(0, _custom(1, 10, 0)), _row(4, jal_zero)]
    summaries = {0x44: {"status": "unknown", "major_opcodes": [0x13]}}
    plain = finite_control_counts(rows, custom_opcode=123, call_summaries=summaries, record_commands=True)
    assert plain["status"] == "unknown" and plain["reason"] == "control_leaves_selected_span"
    assumed = finite_control_counts(
        rows, custom_opcode=123, call_summaries=summaries, record_commands=True, assume_returning_calls=True
    )
    assert assumed["status"] == "proved" and assumed["assumed_returning_calls"] == {0x44: 1}
    issuing = {0x44: {"status": "unknown", "major_opcodes": [123]}}
    refused = finite_control_counts(
        rows, custom_opcode=123, call_summaries=issuing, record_commands=True, assume_returning_calls=True
    )
    assert refused["status"] == "unknown"


def test_unlinked_objects_are_refused_with_an_actionable_reason(tmp_path):
    from merlin.compile.analysis.riscv_machine import linked_input_status

    rows = [_row(0, _addi(11, 0, 1)), _row(4, _bne_zero(4, 4, 11)), _row(8, 0x00008067)]
    counted = finite_control_counts(rows, custom_opcode=123, record_commands=True)
    assert counted["command_stream_status"]["reason"] == "unlinked_object_use_linked_elf"
    assert finite_control_counts(rows, custom_opcode=123, max_steps=50)["reason"] == (
        "finite_instruction_budget_exhausted"
    )

    def elf(path, e_type):
        path.write_bytes(b"\x7fELF\x02\x01\x01" + bytes(9) + e_type.to_bytes(2, "little") + bytes(46))

    elf(tmp_path / "kernel.o", 1)
    elf(tmp_path / "package_kernel.elf", 2)
    status = linked_input_status(tmp_path / "kernel.o")
    assert status["reason"] == "unlinked_object_use_linked_elf"
    assert status["linked_elf_beside"] == [str(tmp_path / "package_kernel.elf")]
    assert linked_input_status(tmp_path / "package_kernel.elf") is None


def _system(rd, rs1, funct3, csr=0x123):
    """A Zicsr word; the CSR number is test data, never a recognized register."""
    return (csr << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x73


def test_pure_csr_reads_leave_control_proved_and_their_value_unknown():
    read = [_row(0, _system(15, 0, 2)), _row(4, _custom(1, 10, 15)), _row(8, 0x00008067)]
    counted = finite_control_counts(read, custom_opcode=123, record_commands=True)
    assert counted["status"] == "proved" and counted["commands"] == 1
    assert counted["command_stream"][0]["rs2"] == {"kind": "unknown", "raw": None}
    immediate = [_row(0, _system(15, 0, 6)), *read[1:]]
    assert finite_control_counts(immediate, custom_opcode=123)["status"] == "proved"


def test_a_branch_on_a_csr_read_or_a_csr_write_still_refuses():
    branch = [_row(0, _system(15, 0, 2)), _row(4, _bne_zero(4, 12, 15)), _row(8, _custom(1, 10, 0)),
              _row(12, 0x00008067)]
    assert finite_control_counts(branch, custom_opcode=123)["reason"] == "data_dependent_or_unbound_control"
    for word in (_system(15, 11, 2), _system(0, 11, 1), _system(15, 3, 7)):
        rows = [_row(0, word), _row(4, _custom(1, 10, 0)), _row(8, 0x00008067)]
        assert finite_control_counts(rows, custom_opcode=123)["reason"] == "opaque_control_requires_capture"


def test_unproved_call_returns_become_one_symbol_per_dynamic_call():
    def jal_ra(pc, target):
        offset = target - pc  # positive and below 2 KiB: only imm[10:1] is set
        return ((offset >> 1) & 1023) << 21 | 1 << 7 | 0x6F

    def shift(rd, rs, funct3, amount):
        return amount << 20 | rs << 15 | funct3 << 12 | rd << 7 | 0x13

    rows = [
        _row(0, jal_ra(0, 0x100)),
        _row(4, 10 << 15 | 10 << 7 | 0x1B),  # sext.w a0, a0
        _row(8, _addi(11, 10, 16)),
        _row(12, shift(12, 11, 1, 32)),  # zero-extend a1 into a2
        _row(16, shift(12, 12, 5, 32)),
        _row(20, 0x400 << 20 | 12 << 15 | 6 << 12 | 13 << 7 | 0x13),  # ori a3, a2, 0x400
        _row(24, _custom(1, 12, 13)),
        _row(28, jal_ra(28, 0x100)),
        _row(32, _custom(1, 10, 10)),
        _row(36, 0x00008067),
    ]
    summaries = {0x100: {"status": "unknown", "major_opcodes": [0x13]}}
    options = dict(custom_opcode=123, call_summaries=summaries, record_commands=True, assume_returning_calls=True)
    plain = finite_control_counts(rows, **options)
    assert "symbolic_returns" not in plain and "window" not in plain["command_stream"][0]["rs1"]
    counted = finite_control_counts(rows, **options, symbolic_returns=True)
    assert counted["status"] == "proved" and len(counted["symbolic_returns"]) == 2
    first, second = (entry for entry in counted["command_stream"] if "funct" in entry)
    symbol = "call:0x100@0x0#0"
    assert first["rs1"]["window"] == dict(symbol=symbol, offset=16, at=0, width=32)
    assert first["rs1"]["known_bits"]["mask"] == ((1 << 32) - 1) << 32  # zero-extended: high bits known zero
    # An OR inside the window keeps only the bits below the first bit it sets.
    assert first["rs2"]["window"] == dict(symbol=symbol, offset=16, at=0, width=10)
    assert first["rs2"]["known_bits"]["value"] == 0x400
    # The same call site executed again returns a different, unrelated symbol.
    assert second["rs1"]["window"] == dict(symbol="call:0x100@0x1c#1", offset=0, at=0, width=64)
