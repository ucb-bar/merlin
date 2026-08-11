"""An instruction encoding derived wrong does not fail loudly — it decodes as a different instruction.

So these tests are mostly about the ways a plausible reader gets a ChiselEnum ordinal wrong (each of
which shifts every later opcode by one and silently emits the neighbouring instruction), and about the
derivation refusing to hand back an encoding it could not fully ground.

The Chisel fixtures below are inline rather than read from a hardware checkout: the parsing rules are
what is under test, they must hold with no sibling tree present, and a fixture can state the awkward
declaration shapes a real file only sometimes contains.
"""
from __future__ import annotations

import os

import pytest

from merlin.targetgen.rtl import opu_isa as OI

_CONSTS = """
package hw.common

trait HasThings {
  def opcLoad   = "b0000111".U
  def opcVector = "b1010111".U
  def OPMVV = "b010".U(3.W)
  def OPMVX = "b110".U(3.W)
  def X = BitPat("b?")
}

object OPMFunct6 extends ChiselEnum {
  val a, b, c = Value          // 0, 1, 2
  val _, _ = Value             // 3, 4 reserved
  val d = Value                // 5
  val acc = Value              // 6
  val odd_neighbour = Value    // 7
  val movein = Value           // 8
  val illegal = Value(0x40.U)
}
"""

_INSNS = """
package hw.insns

class OPMVVInstruction(base: OPMInstruction) extends VectorInstruction {
  val props = base.props ++ Seq(F3(VectorConsts.OPMVV), ReadsVS1.Y)
}
object ACC    extends OPMInstruction { val props = Seq(F6(OPMFunct6.acc)   , ReadsVS1.Y, ReadsVS2.Y, WritesVD.N) }
object MOVEIN extends OPMInstruction { val props = Seq(F6(OPMFunct6.movein), ReadsVS1.N, ReadsVS2.Y, WritesVD.N) }
object OUT    extends OPMInstruction { val props = Seq(F6(OPMFunct6.d)     , ReadsVS1.N, ReadsVS2.N, WritesVD.Y) }
object UNUSED extends OPMInstruction { val props = Seq(F6(OPMFunct6.a)     , ReadsVS1.Y) }
"""

_PARAMS = """
  useUnit : Boolean = false,
) {
  def unitInsns = Seq(
    hw.insns.ACC.VV,
    hw.insns.MOVEIN.VX,
    hw.insns.OUT.VX)
  def supported = base ++ (if (useUnit) unitInsns else Nil)
}
"""

_HEADER = """
// opmvv. f6=b000110, f7=b0001101
#define ACCUM(md, vs2, vs1) \\
  asm volatile(".insn r 0x57, 0x2, 0xd, " md ", " vs1 ", " vs2);

// opmvx. f6=b001000, f7=b0010001
#define MOVE_IN(md, rs1, vs2) \\
  asm volatile(".insn r 0x57, 0x6, 0x11, " md ", %0, " vs2 : : "r"(rs1));

#define READ_OUT(vd, rs1, ms2) \\
  asm volatile(".insn r 0x57, 0x6, 0xb, " vd ", %0, " ms2 : : "r"(rs1));
"""


@pytest.fixture
def sources(tmp_path):
    (tmp_path / "Consts.scala").write_text(_CONSTS, encoding="utf-8")
    (tmp_path / "Instructions.scala").write_text(_INSNS, encoding="utf-8")
    (tmp_path / "Parameters.scala").write_text(_PARAMS, encoding="utf-8")
    (tmp_path / "hdr.h").write_text(_HEADER, encoding="utf-8")
    return tmp_path


def _derive(root):
    return OI.derive(consts=root / "Consts.scala", instructions=root / "Instructions.scala",
                     params=root / "Parameters.scala", funct6_enum="OPMFunct6",
                     consts_container="HasThings", insn_seq="unitInsns",
                     opcode_name="opcVector", form_funct3={"VV": "OPMVV", "VX": "OPMVX"})


class TestChiselEnumOrdinals:
    def test_a_multi_name_declaration_consumes_one_slot_per_name(self):
        got = OI.chisel_enum_ordinals(_CONSTS, "OPMFunct6")
        assert (got["a"], got["b"], got["c"]) == (0, 1, 2)

    def test_a_placeholder_advances_the_counter_without_being_named(self):
        # This is the whole reason funct6 cannot be counted by naming members: `val _, _ = Value`
        # occupies 3 and 4, so the next named member is 5, not 3.
        got = OI.chisel_enum_ordinals(_CONSTS, "OPMFunct6")
        assert got["d"] == 5
        assert "_" not in got

    def test_an_explicit_value_pins_that_member_and_resets_the_counter(self):
        got = OI.chisel_enum_ordinals(_CONSTS, "OPMFunct6")
        assert got["illegal"] == 0x40, "Value(0x40.U) is an explicit ordinal, not the next slot"

    def test_a_commented_out_slot_does_not_shift_later_members(self):
        # A stray `// val x = Value` in a comment must not consume an ordinal; if it did, every
        # instruction after it would encode as its neighbour.
        text = _CONSTS.replace("  val acc = Value", "  // val ghost = Value\n  val acc = Value")
        assert OI.chisel_enum_ordinals(text, "OPMFunct6")["acc"] == 6

    def test_an_absent_enum_yields_nothing_rather_than_raising(self):
        assert OI.chisel_enum_ordinals(_CONSTS, "NoSuchEnum") == {}

    def test_a_non_value_declaration_is_not_a_slot(self):
        text = _CONSTS.replace("  val acc = Value", "  val alias = something\n  val acc = Value")
        assert OI.chisel_enum_ordinals(text, "OPMFunct6")["acc"] == 6


class TestBitLiterals:
    def test_reads_binary_literals_with_and_without_a_width(self):
        got = OI.bit_literal_defs(_CONSTS, "HasThings")
        assert got["opcVector"] == 0x57
        assert got["OPMVV"] == 2 and got["OPMVX"] == 6

    def test_a_non_literal_def_is_skipped_rather_than_guessed(self):
        assert "X" not in OI.bit_literal_defs(_CONSTS, "HasThings")

    def test_an_absent_container_yields_nothing(self):
        assert OI.bit_literal_defs(_CONSTS, "NoSuchTrait") == {}


class TestInstructionProps:
    def test_reads_the_funct6_member_and_every_role_flag(self):
        got = OI.instruction_props(_INSNS)
        assert got["ACC"]["funct6_member"] == "acc"
        assert got["ACC"]["flags"] == {"ReadsVS1": True, "ReadsVS2": True, "WritesVD": False}

    def test_the_last_flag_before_the_closing_braces_is_not_dropped(self):
        # WritesVD is the final marker on the line and identifies the sole readout instruction; a
        # tokenizer that mistakes the closing `) }` for part of it loses exactly that fact.
        assert OI.instruction_props(_INSNS)["OUT"]["flags"]["WritesVD"] is True

    def test_only_one_instruction_writes_a_vector_register(self):
        props = OI.instruction_props(_INSNS)
        writers = [n for n, p in props.items() if p["flags"].get("WritesVD")]
        assert writers == ["OUT"]


class TestUnitInstructionSequence:
    def test_reads_each_object_with_its_instantiation_form(self):
        assert OI.unit_instruction_forms(_PARAMS, "unitInsns") == [
            ("ACC", "VV"), ("MOVEIN", "VX"), ("OUT", "VX")]

    def test_an_instruction_defined_but_not_in_the_sequence_is_not_the_units(self):
        # The instruction file defines the whole vector ISA; claiming all of it would credit the unit
        # with capability the hardware does not have.
        names = [n for n, _ in OI.unit_instruction_forms(_PARAMS, "unitInsns")]
        assert "UNUSED" in OI.instruction_props(_INSNS) and "UNUSED" not in names

    def test_an_absent_sequence_yields_nothing(self):
        assert OI.unit_instruction_forms(_PARAMS, "noSuchSeq") == []


class TestDerivation:
    def test_derives_every_field_of_every_instruction(self, sources):
        d = _derive(sources)
        assert d.gaps == ()
        acc = d.encodings["ACC"]
        assert (acc.opcode, acc.funct3, acc.funct6) == (0x57, 2, 6)
        assert d.encodings["MOVEIN"].funct3 == 6, "a VX form takes the OPMVX funct3"

    def test_funct7_is_the_funct6_shifted_with_the_unmasked_bit(self, sources):
        acc = _derive(sources).encodings["ACC"]
        assert acc.funct7 == (6 << 1) | 1 == 0xd

    def test_emits_an_insn_r_directive_because_no_assembler_knows_the_mnemonic(self, sources):
        got = _derive(sources).encodings["ACC"].insn_r("x1", "x5", "x4")
        assert got == ".insn r 0x57, 0x2, 0xd, x1, x5, x4"

    def test_a_missing_source_file_is_a_gap_not_an_exception(self, tmp_path):
        d = OI.derive(consts=tmp_path / "nope.scala", instructions=tmp_path / "nope.scala",
                      params=tmp_path / "nope.scala", funct6_enum="E", consts_container="C",
                      insn_seq="s", opcode_name="o", form_funct3={})
        assert not d.ok and d.gaps and d.encodings == {}

    def test_an_unmapped_form_gaps_that_instruction_rather_than_defaulting_a_funct3(self, sources):
        d = OI.derive(consts=sources / "Consts.scala", instructions=sources / "Instructions.scala",
                      params=sources / "Parameters.scala", funct6_enum="OPMFunct6",
                      consts_container="HasThings", insn_seq="unitInsns",
                      opcode_name="opcVector", form_funct3={"VV": "OPMVV"})   # VX omitted
        assert "MOVEIN" not in d.encodings
        assert any("VX" in g for g in d.gaps)
        assert not d.ok, "a partial derivation must never report itself usable"


class TestCrosscheck:
    def test_agreement_on_every_field_makes_the_derivation_usable(self, sources):
        d = OI.crosscheck(_derive(sources), sources / "hdr.h",
                          pairs={"ACC": "ACCUM", "MOVEIN": "MOVE_IN", "OUT": "READ_OUT"})
        assert d.ok
        assert all(c["agrees"] for c in d.crosschecks)

    def test_a_disagreement_fails_closed_and_names_the_field(self, sources):
        bad = _HEADER.replace("0x57, 0x2, 0xd", "0x57, 0x2, 0xf")   # funct6 6 -> 7, the odd neighbour
        (sources / "bad.h").write_text(bad, encoding="utf-8")
        d = OI.crosscheck(_derive(sources), sources / "bad.h",
                          pairs={"ACC": "ACCUM", "MOVEIN": "MOVE_IN", "OUT": "READ_OUT"})
        assert not d.ok, "disagreement must make the derivation unusable, not pick a side"
        rec = next(c for c in d.crosschecks if c["instruction"] == "ACC")
        assert "funct6" in rec["reason"] and "rtl=6" in rec["reason"]

    def test_an_instruction_with_no_counterpart_is_unchecked_not_agreeing(self, sources):
        d = OI.crosscheck(_derive(sources), sources / "hdr.h", pairs={"ACC": "ACCUM"})
        assert not d.ok
        assert any(c["instruction"] == "OUT" and not c["agrees"] for c in d.crosschecks)

    def test_records_the_macro_argument_order_because_it_differs_from_the_field_order(self, sources):
        # ACCUM(md, vs2, vs1) expands to `.insn r ..., md, vs1, vs2`: the macro's argument order is not
        # the encoding's field order, which is how an operand swap hides in a square-tile test.
        d = OI.crosscheck(_derive(sources), sources / "hdr.h",
                          pairs={"ACC": "ACCUM", "MOVEIN": "MOVE_IN", "OUT": "READ_OUT"})
        rec = next(c for c in d.crosschecks if c["instruction"] == "ACC")
        assert rec["macro_args"] == ["md", "vs2", "vs1"]

    def test_an_unreadable_header_fails_closed(self, sources):
        d = OI.crosscheck(_derive(sources), sources / "absent.h", pairs={"ACC": "ACCUM"})
        assert not d.ok


class TestInsnRMacros:
    def test_recovers_funct6_from_the_funct7_field(self):
        got = OI.insn_r_macros(_HEADER)
        assert got["ACCUM"]["funct7"] == 0xd
        assert got["ACCUM"]["funct6"] == 6 and got["ACCUM"]["vm"] == 1

    def test_reads_a_body_continued_across_a_line_continuation(self):
        assert "MOVE_IN" in OI.insn_r_macros(_HEADER), "the .insn is on the line after the #define"

    def test_a_define_without_an_insn_is_ignored(self):
        assert OI.insn_r_macros('#define m0 "x0"\n#define FOO(a) bar(a)\n') == {}


@pytest.mark.skipif(not os.environ.get("MERLIN_CHIPYARD"),
                    reason="needs the hardware checkout ($MERLIN_CHIPYARD)")
class TestAgainstTheRealHardware:
    """The same derivation against the actual RTL, so a drift upstream is caught here rather than in a
    silently mis-encoded kernel."""

    def _real(self):
        from pathlib import Path
        s = Path(os.environ["MERLIN_CHIPYARD"]) / "generators/saturn"
        if not s.is_dir():
            pytest.skip(f"no saturn generator under {s}")
        d = OI.derive(consts=s / "src/main/scala/common/Consts.scala",
                      instructions=s / "src/main/scala/insns/Instructions.scala",
                      params=s / "src/main/scala/common/Parameters.scala",
                      funct6_enum="OPMFunct6", consts_container="HasVectorConsts",
                      insn_seq="opuInsns", opcode_name="opcVector",
                      form_funct3={"VV": "OPMVV", "VX": "OPMVX"})
        return s, d

    def test_derives_and_crosschecks_clean(self):
        s, d = self._real()
        assert d.gaps == (), d.gaps
        checked = OI.crosscheck(d, s / "benchmarks/common/bme.h",
                                pairs={"OPMACC": "VOPACC", "OPMVIN": "VMV_RV",
                                       "OPMVINBCAST": "OPMVINBCAST", "OPMVOUT": "VMV_VR"})
        assert checked.ok, [c for c in checked.crosschecks if not c["agrees"]]

    def test_the_accumulate_reads_two_vector_operands_and_writes_none(self):
        _, d = self._real()
        acc = d.encodings["OPMACC"]
        assert acc.flags["ReadsVS1"] and acc.flags["ReadsVS2"]
        assert not acc.flags["WritesVD"], "it accumulates into matrix state, not a vector register"

    def test_exactly_one_instruction_is_the_readout(self):
        _, d = self._real()
        writers = sorted(n for n, e in d.encodings.items() if e.flags.get("WritesVD"))
        assert len(writers) == 1, f"expected a single readout path, got {writers}"

    def test_every_funct6_is_even(self):
        # The unit occupies the reserved EVEN slots; an odd one would be an existing vector op.
        _, d = self._real()
        assert all(e.funct6 % 2 == 0 for e in d.encodings.values())
