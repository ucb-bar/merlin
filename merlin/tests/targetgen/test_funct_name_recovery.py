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

import json

import pytest

from merlin.targetgen.rtl import circt_introspect as C
from merlin.targetgen.rtl import facts as F
from merlin.targetgen.rtl import introspect as I

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
_BLOCK = {"start_comment": "// funct values", "stop_declaration": "CONFIG_EX"}


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
        decoder = {"name": "funct_decode_table", "legal_funct": list(legal), "names": {}, "evidence": "test"}
        header = C.extract_funct_table(_ISA, **_BLOCK)
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
        assert "names_recovered_from_outside_block" in t, (
            "a name from outside the authoritative block must be visible as such"
        )


def test_declared_scala_span_is_target_owned_and_fails_closed(tmp_path, monkeypatch):
    contract = tmp_path / "target_contract.yaml"
    source = tmp_path / "isa.scala"
    source.write_text(
        "// tensor instruction codes\nval ISSUE = 9.U\nval SET_MODE = 0.U\nval PERF = 77.U(7.W)\n",
        encoding="utf-8",
    )
    contract.write_text(
        "rtl_extraction:\n"
        "  scala_funct_block:\n"
        "    start_comment: '// tensor instruction codes'\n"
        "    stop_declaration: SET_MODE\n"
        "  accumulator_memory:\n"
        "    module: LocalAcc\n"
        "    memory_name: accumulator\n"
        "    address_port: addr\n"
        "    data_prefix: lane_\n"
        "    data_suffix: _word\n"
        "    mask_prefix: mask_\n"
        "    firrtl_data_field: io.write.bits.data\n"
        "  boolean_feature_gate:\n"
        "    feature: zero_skip\n"
        "    module: LocalCompute\n"
        "    node: skip_is_enabled\n"
        "    dynamic_operand_contains: skip_requested\n"
        "    config_field: has_zero_skip\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(F, "target_contract_path", lambda target: contract)
    monkeypatch.setattr(C, "_declared_isa_headers", lambda target: [])
    header = C._funct_name_table("another_array", source)
    assert header is not None and header["legal_funct"] == [9]
    decoded = {"name": "funct_decode_table", "legal_funct": [9, 77], "names": {}, "evidence": "decoder"}
    reconciled = C._reconcile_funct(decoded, header)
    assert reconciled["names"] == {"9": "ISSUE", "77": "PERF"}
    assert reconciled["names_recovered_from_outside_block"] == {"77": "PERF"}
    hw = (
        "hw.module @LocalAcc(in %addr : i3, in %lane_0_word : i32, "
        "in %lane_1_word : i32, in %mask_0 : i1, in %mask_1 : i1, "
        "in %mask_2 : i1, in %mask_3 : i1, in %mask_4 : i1, "
        "in %mask_5 : i1, in %mask_6 : i1, in %mask_7 : i1)\n"
        "hw.instance @bank0 @LocalAcc\nhw.instance @bank1 @LocalAcc\n"
    )
    memory = C.extract_accumulator(hw, layout=C._accumulator_layout("another_array"))
    assert memory is not None and (memory["bytes"], memory["banks"], memory["row_bytes"]) == (128, 2, 8)
    fir = tmp_path / "another.fir"
    fir.write_text(
        "module LocalAcc :\n  output io : { write : { bits : { data : SInt<24>[1][2]}}}\n",
        encoding="utf-8",
    )
    hierarchy = tmp_path / "hierarchy.json"
    hierarchy.write_text(json.dumps({"module_name": "Top", "instances": []}), encoding="utf-8")
    legacy = I.extract_facts(fir, hierarchy, role_target="another_array")
    assert legacy["memories"][0]["elem_bits"] == 24
    assert legacy["datapaths"] == [{"name": "accumulator", "dtype": "i24", "evidence": "LocalAcc SInt<24>"}]
    assert I.extract_facts(fir, hierarchy)["memories"] == []
    contract.write_text(contract.read_text().replace("io.write.bits.data", "io.write.bits.missing"))
    unresolved = I.extract_facts(fir, hierarchy, role_target="another_array")
    assert unresolved["memories"][0]["elem_bits"] is None and unresolved["datapaths"] == []
    gate = C._boolean_feature_gate("another_array")
    feature_fir = "module LocalCompute :\n  node skip_is_enabled = and(skip_requested, UInt<1>(0h1))\n"
    assert C.extract_boolean_feature_gate(feature_fir, gate=gate)["value"] is True

    contract.write_text("name: another_array\n", encoding="utf-8")
    assert C._funct_name_table("another_array", source) is None
    assert C._accumulator_layout("another_array") is None
    assert C._boolean_feature_gate("another_array") is None
    assert C.extract_elaborated_rtl_features("another_array", {})["features"] == {}
    contract.write_text(
        "rtl_extraction:\n  scala_funct_block:\n    start_comment: '// missing'\n    stop_declaration: SET_MODE\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="start comment"):
        C._funct_name_table("another_array", source)
