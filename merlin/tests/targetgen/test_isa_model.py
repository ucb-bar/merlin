"""The derived-ISA substrate: the differential operand-field probe (isa_introspect) recovers each operand's
bit placement from a format's OWN encoder, and the consolidated IsaModel exposes it — both fully hermetic
(a SYNTHETIC instruction format, no real target, no model venv), so the machinery is proven target-agnostic.
"""
from __future__ import annotations

from merlin.targetgen.oracle_helpers import isa_introspect as II
from merlin.targetgen.isa_model import IsaModel, isa_model_for


# A synthetic RISC-V-ish format: opcode[0:7], rd[7:12], rs1[15:20], rs2[20:25]. object.__new__ bypasses
# __init__, so operand attrs carry class-level zero defaults (exactly what the probe relies on).
class _FakeMatMul:
    opcode = 0x2B
    rd = 0
    rs1 = 0
    rs2 = 0

    def to_bytecode(self) -> int:
        return ((self.opcode & 0x7F)
                | ((self.rd & 0x1F) << 7)
                | ((self.rs1 & 0x1F) << 15)
                | ((self.rs2 & 0x1F) << 20))


def _pack(fields: dict, **ops: int) -> int:
    """Scatter operand values into a word using ONLY the derived field map (the encoder's core operation)."""
    word = 0
    for attr, val in ops.items():
        for i, wb in enumerate(fields[attr]):
            if isinstance(wb, int) and wb >= 0 and (val >> i) & 1:
                word |= (1 << wb)
    return word


def _unpack(fields: dict, attr: str, word: int) -> int:
    """Gather an operand value back out of a word using the derived field map (the disassembler's core)."""
    val = 0
    for i, wb in enumerate(fields[attr]):
        if isinstance(wb, int) and wb >= 0 and (word >> wb) & 1:
            val |= (1 << i)
    return val


def test_operand_fields_recovers_bit_placement():
    base = II._base_word(_FakeMatMul)
    assert base == 0x2B                                   # all operands zero -> just the opcode
    fields = II._operand_fields(_FakeMatMul, base)
    # exactly the three operands the format uses, each mapped to its real word bits
    assert set(fields) == {"rd", "rs1", "rs2"}
    assert fields["rd"] == [7, 8, 9, 10, 11]
    assert fields["rs1"] == [15, 16, 17, 18, 19]
    assert fields["rs2"] == [20, 21, 22, 23, 24]


def test_fixed_signature_matches_valid_words_and_rejects_others():
    base = II._base_word(_FakeMatMul)
    fields = II._operand_fields(_FakeMatMul, base)
    mask, value = II._fixed_signature_from_fields(base, fields)
    # a word encoded by the format's OWN encoder decodes to this op...
    inst = _FakeMatMul(); inst.rd, inst.rs1, inst.rs2 = 5, 3, 7
    w = inst.to_bytecode()
    assert (w & mask) == value
    # ...and a word with a different opcode does not.
    assert ((w ^ 0x04) & mask) != value                  # flip an opcode bit (bit 2, outside every field)


def test_derived_fieldmap_round_trips_against_the_real_encoder():
    base = II._base_word(_FakeMatMul)
    fields = II._operand_fields(_FakeMatMul, base)
    inst = _FakeMatMul(); inst.rd, inst.rs1, inst.rs2 = 5, 3, 7
    truth = inst.to_bytecode()
    # PACK via the derived map alone reproduces the model's own encoding (minus the fixed bits)...
    packed = base | _pack(fields, rd=5, rs1=3, rs2=7)
    assert packed == truth
    # ...and UNPACK recovers each operand.
    assert _unpack(fields, "rd", truth) == 5
    assert _unpack(fields, "rs1", truth) == 3
    assert _unpack(fields, "rs2", truth) == 7


# --------------------------------------------------------------------------------------------
# IsaModel container (hand-built entry, no derivation needed)
# --------------------------------------------------------------------------------------------
def _fake_model() -> IsaModel:
    base = II._base_word(_FakeMatMul)
    fields = II._operand_fields(_FakeMatMul, base)
    mask, value = II._fixed_signature_from_fields(base, fields)
    by_mnem = {"FakeMatMul": {"class": "FakeMatMul", "role": "matmul", "opcode": 0x2B,
                              "fixed_mask": mask, "fixed_value": value, "fields": fields}}
    return IsaModel(target="fake", by_mnemonic=by_mnem,
                    asm_mnemonics={"MATMUL": "FakeMatMul"}, roles={"matmul": ["FakeMatMul"]},
                    dram_base=0x1000)


def test_isa_model_resolve_by_class_and_asm_alias():
    m = _fake_model()
    assert not m.is_empty()
    assert m.resolve("FakeMatMul")["role"] == "matmul"
    assert m.resolve("MATMUL")["class"] == "FakeMatMul"      # assembler alias
    assert m.resolve("matmul")["class"] == "FakeMatMul"      # case-insensitive alias
    assert m.resolve("NOPE") is None
    assert set(m.fields_of("MATMUL")) == {"rd", "rs1", "rs2"}


def test_isa_model_signatures_are_the_legality_oracle():
    m = _fake_model()
    sigs = m.signatures()
    assert len(sigs) == 1
    _cls, mask, value = sigs[0]
    inst = _FakeMatMul(); inst.rd = 9
    assert (inst.to_bytecode() & mask) == value             # a real word is legal
    assert (0xFFFFFFFF & mask) != value                     # all-ones matches nothing -> illegal


def test_isa_model_for_unknown_target_is_empty_not_a_guess():
    m = isa_model_for("a_target_that_does_not_exist")
    assert m.is_empty() and m.by_mnemonic == {}
