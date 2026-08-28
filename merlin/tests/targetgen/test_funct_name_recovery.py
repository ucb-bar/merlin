"""A funct the SILICON decodes must not stay nameless because the ISA source spells it differently.

Measured on gemmini: the decoder's legal set contains funct 126; the ``// funct values`` block stops
at ``CONFIG_EX`` and never reaches ``val COUNTER_OP = 126.U(7.W)`` twenty lines later. The code was
therefore named ``"?"``, and a nameless instruction can never be given a role — so all 33 of its
occurrences across the built bareMetalC corpus sat in ``claimed_no_role`` permanently.

The recovery is deliberately narrow. GemminiISA.scala rebinds small numbers twice over (the
rs1-subfield group, and a "cisc-gemmini opcodes" group the file itself comments as ``// TODO the
numbers here overlap with the LOOP_WS commands``), so a whole-file scan would RENAME real functs.
Only an unambiguous binding is used; anything bound twice stays ``"?"``.
"""
from merlin.targetgen.rtl import circt_introspect as C


_ISA = """
  // funct values
  val CONFIG_CMD = 0.U
  val LOOP_WS_CONFIG_ADDRS_AB = 10.U
  val CONFIG_EX = 0.U
  val CONFIG_LOAD = 1.U

  // cisc-gemmini opcodes
  // TODO the numbers here overlap with the LOOP_WS commands
  val CISC_CONFIG  = 10.U(7.W)
  val COUNTER_OP   = 126.U(7.W)
  val GARBAGE_ADDR = "hffffffff".U(32.W)
"""


class TestOutsideBlockNames:
    def test_collects_every_binding_per_code_not_one(self):
        ob = C.outside_block_names(_ISA)
        assert ob["10"] == ["LOOP_WS_CONFIG_ADDRS_AB", "CISC_CONFIG"]
        assert ob["0"] == ["CONFIG_CMD", "CONFIG_EX"]

    def test_a_width_annotated_value_is_still_a_binding(self):
        """``126.U(7.W)`` is a real funct. The funct block excludes annotated values because its OWN
        rs1-subfields are annotated; applying that exclusion file-wide drops a live instruction."""
        assert C.outside_block_names(_ISA)["126"] == ["COUNTER_OP"]

    def test_a_non_numeric_literal_is_not_a_code(self):
        """NEGATIVE CASE: ``"hffffffff".U(32.W)`` binds a name to no funct code."""
        ob = C.outside_block_names(_ISA)
        assert not any("GARBAGE_ADDR" in v for v in ob.values())


class TestReconcileRecoversOnlyUnambiguousNames:
    @staticmethod
    def _reconcile(legal):
        decoder = {"name": "funct_decode_table", "legal_funct": list(legal), "names": {},
                   "evidence": "test"}
        header = C.extract_funct_table(_ISA)
        return C._reconcile_funct(decoder, header)

    def test_a_code_the_block_never_reached_gets_its_unambiguous_name(self):
        t = self._reconcile([0, 126])
        assert t["names"]["126"] == "COUNTER_OP"
        assert t["names_recovered_from_outside_block"] == {"126": "COUNTER_OP"}

    def test_an_ambiguously_bound_code_stays_unnamed(self):
        """NEGATIVE CASE: 99 is bound by nothing, so it must stay "?" rather than borrow a name."""
        t = self._reconcile([99])
        assert t["names"]["99"] == "?"
        assert "names_recovered_from_outside_block" not in t

    def test_recovery_never_overwrites_a_name_the_block_supplied(self):
        # 10 is named by the funct block; the cisc group rebinds it. The block wins.
        t = self._reconcile([10])
        assert t["names"]["10"] == "LOOP_WS_CONFIG_ADDRS_AB"

    def test_the_weaker_provenance_is_recorded_not_hidden(self):
        t = self._reconcile([126])
        assert "names_recovered_from_outside_block" in t, \
            "a name from outside the authoritative block must be visible as such"
