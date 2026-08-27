"""The two new decoders: a RoCC command stream and a self-hosted-ISA stream.

Both decode the raw word and compare INTEGERS against a table derived from the target's own sources.
Neither file contains an opcode, a funct value, or (for muon) even a field position.
"""
from __future__ import annotations

import pytest

from merlin.kernels import endpoints as EP
from merlin.kernels.decode import derived_isa as MU
from merlin.kernels.decode import rocc as RC


class _I:
    """A minimal objdump RawInsn stand-in."""

    def __init__(self, hexcode, mnemonic="<unknown>", addr=0, operands=()):
        self.hexcode, self.mnemonic, self.addr, self.operands = hexcode, mnemonic, addr, operands


@pytest.fixture(scope="module")
def gemmini():
    table = RC.funct_table_for("gemmini")
    if not table.get("custom_opcode"):
        pytest.skip("gemmini RTL facts unavailable")
    ep = EP.load_endpoint("gemmini_rocc")
    op = int(table["custom_opcode"])
    funct = {v: int(k) for k, v in (table.get("names") or {}).items()}

    def word(name, xd=0, xs1=1, xs2=1):
        return f"{(funct[name] << 25) | (xd << 14) | (xs1 << 13) | (xs2 << 12) | op:08x}"

    return table, ep, word


class TestRoccDecodesAgainstDerivedFacts:
    def test_nothing_in_this_module_names_an_opcode(self):
        import inspect
        src = inspect.getsource(RC)
        assert "custom_opcode" in src, "the opcode must be READ from the derived table"
        assert "0x7b" not in src and "123" not in src, "a baked opcode value"

    def test_a_command_is_named_from_the_rtl_table(self, gemmini):
        _t, ep, word = gemmini
        a = RC.audit([_I(word("PRELOAD_CMD"))], "gemmini", ep)
        assert a.counts.get("PRELOAD_CMD") == 1 and a.endpoint_insns == 1

    def test_func3_varies_per_instruction_and_must_not_gate_identity(self, gemmini):
        """xd/xs1/xs2 say whether THIS instruction writes rd and reads rs1/rs2, so they differ between
        two uses of one command. A decoder that pinned them dropped every conformant instruction with
        a different operand shape."""
        _t, ep, word = gemmini
        insns = [_I(word("COMPUTE_AND_FLIP_CMD", xd=1, xs1=0, xs2=1)),
                 _I(word("COMPUTE_AND_FLIP_CMD", xd=0, xs1=1, xs2=0))]
        a = RC.audit(insns, "gemmini", ep)
        assert a.counts["COMPUTE_AND_FLIP_CMD"] == 2, "an operand-shape difference dropped an instruction"

    def test_a_word_on_our_opcode_the_table_does_not_claim_is_reported(self, gemmini):
        """Never counted as absent: a mis-encoded instruction looks exactly like this."""
        table, ep, _w = gemmini
        illegal = max(set(range(0x7F)) - set(table["legal_funct"]))
        a = RC.audit([_I(f"{(illegal << 25) | int(table['custom_opcode']):08x}")], "gemmini", ep)
        assert a.unaccounted and a.endpoint_insns == 0

    def test_a_base_isa_instruction_is_neither_claimed_nor_flagged(self, gemmini):
        _t, ep, _w = gemmini
        a = RC.audit([_I("00000013", mnemonic="nop")], "gemmini", ep)
        assert a.endpoint_insns == 0 and not a.unaccounted


class TestTheInstructionLevelIsReadOffTheStream:
    def test_a_fine_grained_stream_reports_fine_grained(self, gemmini):
        _t, ep, word = gemmini
        a = RC.audit([_I(word(n)) for n in
                      ("CONFIG_CMD", "LOAD_CMD", "PRELOAD_CMD", "COMPUTE_AND_FLIP_CMD", "STORE_CMD")],
                     "gemmini", ep)
        assert a.level == "fine_grained" and a.missing_roles == ()

    def test_an_fsm_stream_reports_fsm(self, gemmini):
        """The Phase 2 finding made operational: these two are indistinguishable in the C source —
        tiled_matmul_auto expands into the FSM inside the library — so the level is a property of the
        EMITTED stream and can only be established by disassembly."""
        _t, ep, word = gemmini
        a = RC.audit([_I(word(n)) for n in
                      ("CONFIG_CMD", "LOOP_WS_CONFIG_BOUNDS", "LOOP_WS_CONFIG_ADDRS_AB", "LOOP_WS")],
                     "gemmini", ep)
        assert a.level == "fsm"

    def test_a_mixed_stream_reports_both_rather_than_choosing(self, gemmini):
        # A kernel that offloads its inner loop but hand-drives an epilogue is a real and interesting
        # state, not an error to be resolved into one answer.
        _t, ep, word = gemmini
        a = RC.audit([_I(word(n)) for n in ("LOOP_WS", "PRELOAD_CMD", "COMPUTE_AND_STAY_CMD")],
                     "gemmini", ep)
        assert a.level == "both"

    def test_the_two_levels_do_not_hash_alike(self, gemmini):
        """The inert-lever guard hashes the MNEMONIC stream, and every command here disassembles to the
        same unknown text — so swapping an FSM offload for a hand-driven sequence would look inert."""
        _t, ep, word = gemmini
        fine = RC.audit([_I(word(n)) for n in ("PRELOAD_CMD", "COMPUTE_AND_FLIP_CMD")], "gemmini", ep)
        fsm = RC.audit([_I(word(n)) for n in ("LOOP_WS_CONFIG_BOUNDS", "LOOP_WS")], "gemmini", ep)
        assert fine.digest != fsm.digest


class TestMuonDecodesAgainstADerivedLayout:
    @pytest.fixture(scope="class")
    def enc(self):
        e = MU.encoding_for("muon")
        if not e.get("fields"):
            pytest.skip("muon derived encoding unavailable")
        return e

    def test_no_field_position_is_written_down(self):
        import inspect
        src = inspect.getsource(MU)
        assert "inst_width" in src and "fields" in src
        assert ">> 25" not in src and ">> 12" not in src, "a baked field position"

    def test_fields_come_from_the_derived_bit_ranges(self, enc):
        got = MU.fields_of(0, enc["fields"])
        assert set(got) == set(enc["fields"]), "every declared field must be extracted"

    def test_an_opcode_is_named_from_the_derived_table(self, enc):
        value = int((enc.get("opcodes") or {})["CUSTOM0"])
        d = MU.decode_stream([_I(f"{value:016x}")], enc, spaces=["custom_0"])
        assert d[0].space == "CUSTOM0"

    def test_the_endpoint_spans_more_than_one_opcode_space(self):
        """Measured: a real kernel's custom words split between the custom space and the REPURPOSED
        standard OP space. A decoder reading only the custom space sees a minority of the target."""
        block = (EP._spec()["endpoints"]["muon_simt"].get("encoding") or {})
        assert len(block.get("spaces") or []) > 1, block

    def test_a_word_narrower_than_the_declared_width_is_declined(self, enc):
        # Zero-extending a compressed word into a wide layout yields a confident decode of an
        # instruction that was never there.
        assert MU._word_of("0b", int(enc["inst_width"])) is None

    def test_a_word_nothing_can_place_is_unaccounted(self, enc):
        bad = MU.DerivedIsaInsn(index=0, addr=0, identity="<unknown>", space="", mnemonic="<unknown>")
        assert not MU.accountable(bad)

    def test_a_word_the_tool_named_is_accounted_even_without_a_space(self):
        """The distinction the 76%-unknown probe got wrong: 'the tool could not name this' is not the
        same as 'this is the endpoint's instruction'."""
        ok = MU.DerivedIsaInsn(index=0, addr=0, identity="addi", space="", mnemonic="addi")
        assert MU.accountable(ok)
