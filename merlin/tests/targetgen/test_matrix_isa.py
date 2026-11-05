"""MATCH/MASK derivation of a matrix extension's encodings, and the checks that make it fail closed.

The fixtures are synthetic Chisel/C written in the shapes real units use (a decode table keyed by props,
a host core's BitPat object, an author's ``.insn r`` header, a riscv-opcodes table, a SoC config with two
vector units of which one carries the matrix unit). Every value is invented; the tests are about the
derivation, not about any one unit.
"""

from __future__ import annotations

import pytest

from merlin.targetgen.rtl import matrix_isa as M

CONSTS = """
object FN6 extends ChiselEnum {
  val a, b = Value
  val _ = Value
  val mv = Value
  val packx = Value
}
trait Consts {
  def opcMain = "b1010111".U
  def opcAlt  = "b1110111".U // the alternate major opcode
  def FA = "b000".U(3.W)
  def FB = "b110".U(3.W)
}
"""

INSNS = """
// a: two-line props, a BitPat funct6, the alternate opcode
object X_MM extends VectorInstruction { val props = Seq(ALT.Y, F6(BitPat("b111100")), F3(Consts.FA), SZ(0.U(2.W)),
  ReadsA.Y, Kind(Kinds.MM.U)) }
object X_MOVE extends VectorInstruction { val props = Seq(F6(FN6.mv), F3(Consts.FB), R2(31.U(5.W)), WritesV.Y) }
object X_ZERO extends VectorInstruction { val props = Seq(F6(FN6.mv), F3(Consts.FB), R2(30.U(5.W)), R1(0.U(5.W))) }
object X_PACK extends VectorInstruction { val props = Seq(F6(FN6.packx), F3(Consts.FB)) }
"""

PARAMS = """
case class VP(
  vlen: Int = 128,
  useUnit : Boolean = false,
)
object VP {
  def baseParams = VP()
  def bigParams = baseParams.copy(vlen = 256)
  def unitParams = bigParams.copy(
    useUnit = true,
  )
}
trait HasP {
  def unitInsns = Seq(
    pkg.X_MM,
    pkg.X_MOVE,
    pkg.X_ZERO,
    pkg.X_PACK)
  def unitTypes: Seq[(Int, Int)] = if (useUnit) Seq((8, 4), (32, 1)) else Nil
}
"""

CORE = """
object CoreInsns {
  def SETN = BitPat("b1000100_00000_?????_111_?????_1010111")
  def SETK = BitPat("b1000100_00010_?????_111_?????_1010111")
}
object CoreConfig {
  def kmax(sew: Int, w: Int): Int = (sew, w) match {
    case (8, 4)  => 4
    case (32, 1) => 1
    case _       => 0
  }
}
"""

HEADER = r"""
// prose that mentions ".insn r" must not be read as an instruction
#define MM(t, a, b) asm volatile(".insn r 0x77, 0, 0x79, " T(t) ", " V(b) ", " V(a))
#define ZERO(t) asm volatile(".insn r 0x57, 6, 0x7, " T(t) ", x0, x30")
#define MOVE(vd, s) asm volatile(".insn r 0x57, 6, 0x7, " V(vd) ", %0, x31" : : "r"(s))
#define PACK(s, v) asm volatile(".insn r 0x57, 6, 0x9, x0, %0, " V(v) : : "r"(s))
static inline size_t setn(size_t n) {
  size_t r;
  if (n) {
    asm volatile(".insn r 0x57, 7, 0x44, %0, %1, x0" : "=r"(r) : "r"(n));
  }
  return r;
}
static inline size_t setk(size_t k) {
  size_t r;
  asm volatile(".insn r 0x57, 7, 0x44, %0, %1, x2" : "=r"(r) : "r"(k));
  return r;
}
"""

PAIRS = {"X_MM": "MM", "X_MOVE": "MOVE", "X_ZERO": "ZERO", "X_PACK": "PACK", "SETN": "setn", "SETK": "setk"}


@pytest.fixture
def tree(tmp_path):
    for name, text in {"consts.scala": CONSTS, "insns.scala": INSNS, "params.scala": PARAMS, "core.scala": CORE,
                       "unit.h": HEADER}.items():  # fmt: skip
        (tmp_path / name).write_text(text)
    return tmp_path


def _derive(tree, **over):
    kw = dict(
        consts=tree / "consts.scala",
        instructions=tree / "insns.scala",
        params=tree / "params.scala",
        insn_seq="unitInsns",
        consts_container="Consts",
        funct6_enum="FN6",
        opcode_name="opcMain",
        opcode_flags={"ALT": "opcAlt"},
        decode_fields={"F6": "funct6", "F3": "funct3", "R1": "rs1", "R2": "rs2"},
        vtype_fields=("SZ",),
        core_source=tree / "core.scala",
        core_container="CoreInsns",
        core_insns=["SETN", "SETK"],
    )
    kw.update(over)
    return M.derive(**kw)


def test_bitpat_parsing():
    assert M.parse_bitpat('BitPat("b1?0_1")') == (0b1001, 0b1011, 4)
    assert M.parse_bitpat('BitPat("h1f")') is None
    assert M.parse_bitpat("1.U") is None


def test_decode_table_and_bitpats_become_match_mask_pairs(tree):
    isa = _derive(tree)
    assert not isa.gaps, isa.gaps
    mm, move, zero = isa.encodings["X_MM"], isa.encodings["X_MOVE"], isa.encodings["X_ZERO"]
    # The alternate-opcode flag picked the other major opcode; the BitPat funct6 and the consts funct3 are read.
    assert (mm.opcode, mm.funct3, mm.funct6) == (0b1110111, 0, 0b111100)
    assert mm.vtype == {"SZ": 0} and mm.flags["Kind"] == "Kinds.MM.U"
    # The enum ordinal counts the placeholder slot: a, b, _, mv -> 3.
    assert move.funct6 == 3 and move.fixed("rs2") == 31 and move.fixed("rs1") is None
    assert zero.fixed("rs2") == 30 and zero.fixed("rs1") == 0
    # Same funct6 and funct3, told apart only by rs2: exactly what a triple-keyed table cannot do.
    assert (move.opcode, move.funct3, move.funct6) == (zero.opcode, zero.funct3, zero.funct6)
    setn = isa.encodings["SETN"]
    assert setn.mask == 0xFFF0707F and setn.fixed("rs2") == 0 and setn.funct7 == 0b1000100


def test_header_crosscheck_agrees_fills_and_spells(tree):
    isa = M.crosscheck_header(_derive(tree), tree / "unit.h", pairs=PAIRS)
    assert isa.ok, isa.to_dict()
    mm = isa.encodings["X_MM"]
    # The vm bit is not decoded by the RTL; the header states it and it is recorded as filled.
    assert mm.fill_mask == 1 << 25 and mm.funct7 == 0x79
    assert mm.insn_r("x1", "x16", "x0") == ".insn r 0x77, 0x0, 0x79, x1, x16, x0"
    # A field the encoding fixes is spelled from the encoding and may not be passed.
    zero = isa.encodings["X_ZERO"]
    assert zero.insn_r("x8") == ".insn r 0x57, 0x6, 0x7, x8, x0, x30"
    with pytest.raises(ValueError, match="fixed"):
        zero.insn_r("x8", "x1", None)
    with pytest.raises(ValueError, match="must be given"):
        isa.encodings["X_MOVE"].insn_r(None, "a0", None)
    # The header fixed rd=x0 for the pack, which the RTL leaves open: filled from the header.
    assert isa.encodings["X_PACK"].fixed("rd") == 0
    # The `if` block inside setn did not hide its enclosing function.
    assert "setn" in M.header_insn_sites(HEADER)


def test_a_disagreeing_header_fails_closed(tree):
    (tree / "bad.h").write_text(HEADER.replace("x0, x30", "x0, x29"))
    isa = M.crosscheck_header(_derive(tree), tree / "bad.h", pairs=PAIRS)
    assert not isa.ok
    bad = [c for c in isa.crosschecks if not c["agrees"]]
    assert [c["instruction"] for c in bad] == ["X_ZERO"] and "differ" in bad[0]["reason"]
    with pytest.raises(ValueError):
        # an unresolved emit field is refused rather than guessed
        M.MaskedEncoding("U", match=0x57, mask=0x7F, origin="t").insn_r("x1", "x2", "x3")


def test_an_unchecked_instruction_is_not_ok(tree):
    pairs = dict(PAIRS)
    pairs.pop("X_PACK")
    isa = M.crosscheck_header(_derive(tree), tree / "unit.h", pairs=pairs)
    assert not isa.ok
    assert M.MatrixIsa(encodings=_derive(tree).encodings).ok is False  # no cross-check at all


def test_missing_declarations_are_gaps_not_defaults(tree):
    assert _derive(tree, insn_seq="nope").gaps
    assert any("NOPE" in g for g in _derive(tree, core_insns=["NOPE"]).gaps)
    assert not _derive(tree, opcode_name="opcMissing").ok


def test_overlapping_derivations_are_a_gap(tree):
    (tree / "insns.scala").write_text(INSNS.replace("R2(30.U(5.W)), R1(0.U(5.W))", "R2(31.U(5.W))"))
    assert any("decode the same words" in g for g in _derive(tree).gaps)


def test_opcode_table_crosscheck(tree):
    table = "\n".join(
        [
            "#define MATCH_ZERO 0x43e06057",
            "#define MASK_ZERO 0xfffff0ff",
            "#define MATCH_SETN 0x88007057",
            "#define MASK_SETN 0xfff0707f",
        ]
    )
    (tree / "enc.h").write_text(table)
    isa = M.crosscheck_header(_derive(tree), tree / "unit.h", pairs=PAIRS)
    # Pin the derived X_ZERO funct6 to the table's slot so the fixture describes one consistent unit.
    got = M.crosscheck_opcode_table(isa, tree / "enc.h", pairs={"X_ZERO": "ZERO", "SETN": "SETN"})
    rec = {c["instruction"]: c for c in got.crosschecks if c.get("source") == str(tree / "enc.h")}
    assert rec["SETN"]["agrees"]
    # X_ZERO's funct6 is 3 in this fixture but 0x10 in the table: decoded bits differ.
    assert not rec["X_ZERO"]["agrees"]
    # Instructions with no table entry are failed checks, not silent passes.
    assert not rec["X_MM"]["agrees"] and not got.ok


CONFIG = """
class TwoUnitsConfig extends Config(
  new pkg.WithBVectorUnit(256, 256, VP.baseParams) ++
  new pkg.WithAVectorUnit(
    256,
    128,
    VP.unitParams
  ) ++
  new Other
)
"""
MIXINS = """
class WithAVectorUnit(vLen: Int = 128, dLen: Int = 64, params: VP = VP()) extends Config()
class WithBVectorUnit(vLen: Int = 128, dLen: Int = 64, params: VP = VP()) extends Config()
"""


def test_geometry_comes_from_the_vector_unit_that_enables_the_matrix_unit():
    got, notes = M.matrix_unit_params(
        CONFIG, "TwoUnitsConfig", mixin_text=MIXINS, params_text=PARAMS, params_object="VP", enable_flag="useUnit"
    )
    # The FIRST vector unit (B, dLen 256) has no matrix unit; reading it would give the wrong alignment.
    assert got == {"vLen": 256, "dLen": 128}
    assert any("WithBVectorUnit" in n and "false" in n for n in notes)
    none, why = M.matrix_unit_params(
        CONFIG, "TwoUnitsConfig", mixin_text=MIXINS, params_text=PARAMS, params_object="VP", enable_flag="useOther"
    )
    assert none == {} and why


def test_seq_and_match_tables():
    assert M.seq_int_tuples(PARAMS, "unitTypes") == [(8, 4), (32, 1)]
    assert M.match_case_table(CORE, "CoreConfig", "kmax") == {(8, 4): 4, (32, 1): 1}
    assert M.match_case_table(CORE, "Nope", "kmax") == {}
