"""A shipped ISA encoder can declare an operand and then never pack it.

Measured on atlas: ``IType.to_bytecode`` assigns ``rd`` and immediately overwrites it with ``imm``. The
functional simulator hides that -- it reads the decoded instruction object, not the word -- but the RTL
decoder reads bits[11:7] and gets the immediate's low bits. Deriving merlin's assembler from the
unrepaired encoder produced an ``ADDI`` with no ``rd`` at all, so an agent could not emit the instruction
that sets up an address register: most of a kernel's scalar prologue was unwritable.

These tests drive the repair against synthetic encoders, so they need no model venv.
"""
from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import merlin_dir

_SRC = merlin_dir() / "python/merlin/targetgen/oracle_helpers/isa_introspect.py"


@pytest.fixture(scope="module")
def isa():
    spec = importlib.util.spec_from_file_location("isa_introspect_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mask(v, bits):
    return int(v) & ((1 << bits) - 1)


class _Healthy:
    """A U-type-alike: packs rd at bits 11:7, immediate above it."""
    rd: int = 0
    imm: int = 0
    opcode = 0x37

    def to_bytecode(self):
        return (_mask(self.imm, 20) << 12) | (_mask(self.rd, 5) << 7) | self.opcode


class _Clobbered:
    """An I-type-alike carrying the measured bug: rd is assigned, then overwritten by imm."""
    rd: int = 0
    rs1: int = 0
    imm: int = 0
    opcode = 0x13

    def to_bytecode(self):
        rd = self.rd
        rd = self.imm                       # the bug, verbatim in shape
        return ((_mask(self.imm, 12) << 20) | (_mask(self.rs1, 5) << 15)
                | (_mask(rd, 5) << 7) | self.opcode)


def _entries(isa, classes):
    out = []
    for name, cls in classes.items():
        base = isa._base_word(cls)
        fields, touched = isa._operand_fields(cls, base)
        mask, value = isa._fixed_signature_from_touched(base, touched)
        out.append({"mnemonic": name, "fields": fields, "fixed_mask": mask, "fixed_value": value})
    return out


def test_the_bug_is_reproduced_before_any_repair(isa):
    """Without the repair the clobbered format reports no rd and a non-linear immediate."""
    e = _entries(isa, {"CLOB": _Clobbered})[0]
    assert "rd" not in e["fields"], "the probe should find rd packed nowhere -- that is the bug"
    assert -1 in e["fields"]["imm"], "imm's low bits alias into rd's slot, so the packer must refuse them"


def test_a_dropped_operand_is_restored_where_the_rest_of_the_isa_puts_it(isa):
    ents = _entries(isa, {"HEALTHY": _Healthy, "CLOB": _Clobbered})
    isa._repair_dropped_operands(ents, {"HEALTHY": _Healthy, "CLOB": _Clobbered})
    clob = next(e for e in ents if e["mnemonic"] == "CLOB")
    assert clob["fields"]["rd"] == [7, 8, 9, 10, 11], "rd belongs where every other format puts it"
    assert clob["repaired"] == ["rd"], "the repair must be reported, not applied silently"
    assert all(b >= 0 for b in clob["fields"]["imm"]), \
        "with rd owning its own bits again the immediate is linear and packable"


def test_a_healthy_encoder_is_left_alone(isa):
    """Self-disabling: nothing to restore means nothing is touched."""
    ents = _entries(isa, {"HEALTHY": _Healthy})
    before = dict(ents[0]["fields"])
    isa._repair_dropped_operands(ents, {"HEALTHY": _Healthy})
    assert ents[0]["fields"] == before
    assert "repaired" not in ents[0]


def test_nothing_is_repaired_when_the_isa_disagrees_about_placement(isa):
    """Fail closed: if two formats place rd differently there is no evidence, so guess nothing."""
    class _Elsewhere(_Healthy):
        opcode = 0x17

        def to_bytecode(self):
            return (_mask(self.imm, 20) << 12) | (_mask(self.rd, 5) << 25) | self.opcode

    ents = _entries(isa, {"HEALTHY": _Healthy, "ELSEWHERE": _Elsewhere, "CLOB": _Clobbered})
    isa._repair_dropped_operands(ents, {"HEALTHY": _Healthy, "ELSEWHERE": _Elsewhere,
                                        "CLOB": _Clobbered})
    clob = next(e for e in ents if e["mnemonic"] == "CLOB")
    assert "rd" not in clob["fields"], "two placements is no placement -- refuse rather than pick one"


def test_the_repair_does_not_mutate_the_shared_isa_module(isa):
    """Other sessions import the same ISA classes; the probe must leave them exactly as it found them."""
    original = _Clobbered.to_bytecode
    ents = _entries(isa, {"HEALTHY": _Healthy, "CLOB": _Clobbered})
    isa._repair_dropped_operands(ents, {"HEALTHY": _Healthy, "CLOB": _Clobbered})
    assert _Clobbered.to_bytecode is original
