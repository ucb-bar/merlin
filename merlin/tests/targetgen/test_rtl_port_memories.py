"""A memory the cmem/smem census cannot see, derived from the BANKED WRITE PORTS below it.

Two separate failures meet here, and both were silent.

**FIRRTL 6 modules were invisible.** ``public module Foo :`` puts a qualifier before the keyword the
module reader matched on, so ``parse_ports`` returned an EMPTY dict for the whole design and every port
probe answered "this target exposes no such port" — a claim about the hardware, produced by a parse gap.

**A store instantiated above the elaboration's root has no census entry at all.** The census walks
``cmem``/``smem`` DECLARATIONS, so a memory wired in at a tile level a standalone repo cannot elaborate
produces no memories, and the target's whole memory-mapping axis then reports ``0 / 0`` cells — over a
store whose geometry the module below it declares in full on its own ports.

The identification carries no names: a bundle with a ``UInt<W>`` data field and a ``UInt<1>[N]`` vector
where ``N == W/8`` is a byte-enabled write to a W-bit row, and nothing else in a design has a Bool
vector whose length is exactly the byte count of a sibling field. What the port geometry gives is a
BOUND on the bank count (the index is n bits); what pins it is the design's own range check of a line
address against its elaborated limit. Where no such check exists the bound is published and the capacity
is withheld — a capacity wrong by the bank factor reads as a fit.
"""
from __future__ import annotations

import textwrap

from merlin.targetgen import address_space as AS
from merlin.targetgen.rtl import circt_introspect as CI
from merlin.targetgen.rtl import ports as P

# FIRRTL 6.0 spelling: the circuit's entry point is a `public module`, and the store is reached through
# a Valid wrapper (data and byte enable sit under `bits`, not at the port's top level). Trimmed from a
# real elaboration; the sibling `vecWrite` has NO byte enable and the sibling `read` no data at all.
_FIR6 = textwrap.dedent("""\
    FIRRTL version 6.0.0
    circuit Store :
      layer Verification, bind, "verification" :
      public module Store : @[a/b/Store.scala 55:7]
        input clock : Clock
        output io : { flip cmd : { valid : UInt<1>, bits : { op : UInt<2>, lineAddr : UInt<16>}}, read : { valid : UInt<1>, bits : { bankIdx : UInt<3>, bankAddr : UInt<13>}}, write : { valid : UInt<1>, bits : { bankIdx : UInt<3>, bankAddr : UInt<13>, data : UInt<256>, mask : UInt<1>[32]}}, vecWrite : { valid : UInt<1>, bits : { bankIdx : UInt<3>, bankAddr : UInt<13>, data : UInt<256>}}, busy : UInt<1>}
        node _T = geq(io.cmd.bits.byteAddr, UInt<21>(0h180000)) @[a/b/Store.scala 111:50]
        node _T_57 = lt(io.cmd.bits.lineAddr, UInt<16>(0hc000)) @[a/b/Store.scala 64:19]
        connect io.write.valid, _T_57
    """)

# The same design with the range check removed: geometry alone, so the bank count is bounded and not
# pinned.
_FIR6_UNPINNED = "\n".join(l for l in _FIR6.splitlines() if "_T_57" not in l) + "\n"


def _entry(fir_text: str) -> dict:
    recs = P.banked_store_ports(fir_text)
    assert len(recs) == 1, recs
    return recs[0]


class TestFirrtl6ModuleHeader:
    def test_a_public_module_is_read(self):
        # The bug this covers returned an EMPTY dict for the whole design, which reads downstream as
        # "the RTL has no such port" rather than as "the reader could not see the module".
        got = P.parse_ports(_FIR6)
        assert "Store" in got, "a FIRRTL 6 `public module` must be visible to the port reader"
        assert got["Store"].field_named("busy") is not None

    def test_the_plain_spelling_still_parses(self):
        assert "M" in P.parse_ports("circuit M :\n  module M :\n    output io : { busy : UInt<1>}\n")

    def test_a_statement_is_not_read_as_a_module_header(self):
        # The qualifier skip must not turn any second-position `module` token into a module: the name
        # that follows has to look like a declaration name.
        got = P.parse_ports("circuit M :\n  module M :\n    wire module : UInt<1>\n"
                            "    output io : { busy : UInt<1>}\n")
        assert set(got) == {"M"}, "a statement whose second token is a keyword is not a module header"


class TestBankedStoreGeometry:
    def test_the_byte_enable_identifies_the_write_port(self):
        rec = _entry(_FIR6)
        assert (rec["module"], rec["field"]) == ("Store", "write")

    def test_the_row_and_the_per_bank_depth_come_from_the_types(self):
        rec = _entry(_FIR6)
        assert rec["row_bits"] == 256 and rec["row_bytes"] == 32
        assert rec["row_addr_bits"] == 13 and rec["rows_per_bank"] == 8192

    def test_the_index_width_is_a_bound_on_the_bank_count(self):
        rec = _entry(_FIR6)
        assert rec["bank_id_bits"] == 3 and rec["banks_max"] == 8 and rec["banks_min"] == 1

    def test_the_range_check_pins_the_exact_bank_count(self):
        # 0xc000 = 49152 lines over 8192 rows per bank = 6 banks, which is FEWER than the 8 the index
        # could address: the bound alone would have overstated the store by a third.
        rec = _entry(_FIR6)
        assert rec["banks_exact"] is True
        assert rec["banks"] == 6 and rec["total_rows"] == 49152
        assert rec["bytes"] == 49152 * 32
        assert "0hc000" in rec["banks_evidence"], "the pinning literal must be quoted as evidence"

    def test_a_sibling_literal_that_is_not_a_line_limit_is_rejected(self):
        # 0x180000 is a whole multiple of the per-bank rows too (192 of them). It is refused because
        # 192 banks exceeds what a 3-bit index can address and because its declared 21-bit width does
        # not fit the 16-bit line address -- neither filter alone is the whole test.
        rec = _entry(_FIR6)
        assert rec["banks"] == 6, "the byte-capacity literal must not be read as the line limit"


class TestUnpinnedFailsClosed:
    def test_no_range_check_leaves_an_honest_bound(self):
        rec = _entry(_FIR6_UNPINNED)
        assert rec["banks"] is None and rec["banks_exact"] is False
        assert (rec["banks_min"], rec["banks_max"]) == (1, 8)
        assert rec["bytes"] is None and rec["total_rows"] is None
        assert "not pinned" in rec["banks_unknown"]

    def test_two_disagreeing_limits_are_ambiguous_not_averaged(self):
        # A module stating two admissible limits cannot tell this reader which one bounds the store.
        # Answering with either would be a guess; the bound is reported instead.
        two = _FIR6.replace("connect io.write.valid, _T_57",
                            "node _T_58 = lt(io.cmd.bits.lineAddr, UInt<16>(0h8000))")
        rec = _entry(two)
        assert rec["banks"] is None and "disagree" in rec["banks_unknown"]

    def test_the_geometry_survives_a_failed_pin(self):
        rec = _entry(_FIR6_UNPINNED)
        assert rec["row_bits"] == 256 and rec["rows_per_bank"] == 8192


class TestNotEveryBundleIsAMemory:
    def test_a_write_port_without_a_byte_enable_is_not_reported(self):
        # `vecWrite` carries the same 256-bit data and the same two address fields as `write` and no
        # mask. Reporting it would publish a second store the device does not have -- and the folding
        # in `memories_from_port_geometry` would not catch it, because its geometry differs from none.
        assert [r["field"] for r in P.banked_store_ports(_FIR6)] == ["write"]

    def test_a_mask_whose_length_is_not_the_byte_count_is_not_a_byte_enable(self):
        # A UInt<1>[8] beside a UInt<256> is some other per-something flag, not a byte enable. The
        # length must be exactly W/8 or the bundle says nothing about a row.
        wrong = _FIR6.replace("mask : UInt<1>[32]", "mask : UInt<1>[8]")
        assert P.banked_store_ports(wrong) == []

    def test_a_data_field_does_not_pair_with_a_sibling_bundles_mask(self):
        # Flattening every level into one namespace lets a `data` under one sub-bundle pair with a
        # `mask` under a DIFFERENT one and invent a memory neither port describes. Each level is its
        # own namespace, so this pairs with nothing.
        split = ("circuit M :\n  module M :\n"
                 "    output io : { a : { data : UInt<256>, addr : UInt<13>},"
                 " b : { mask : UInt<1>[32], addr : UInt<13>}}\n")
        assert P.banked_store_ports(split) == []

    def test_a_design_with_no_banked_port_yields_nothing(self):
        plain = "circuit M :\n  module M :\n    output io : { done : UInt<1>}\n"
        assert P.banked_store_ports(plain) == []


class TestMemoryFactShape:
    def _mems(self, tmp_path, text=_FIR6):
        f = tmp_path / "Store.fir"
        f.write_text(text)
        return CI.memories_from_port_geometry([f])

    def test_the_entry_matches_the_census_schema(self, tmp_path):
        mem = self._mems(tmp_path)[0]
        for key in ("name", "banks", "depth", "row_elems", "elem_bits", "row_bits_rtl", "bytes",
                    "source", "evidence"):
            assert key in mem, f"a consumer of census memories reads {key}"
        assert mem["banks"] == 6 and mem["depth"] == 8192 and mem["bytes"] == 49152 * 32
        assert mem["row_bits_rtl"] == 256

    def test_the_source_says_which_reading_produced_it(self, tmp_path):
        # A port-derived row is a WRITE GRANULARITY; a census row is the SRAM's declared element type.
        # A reader that cannot tell them apart cannot weigh them.
        mem = self._mems(tmp_path)[0]
        assert mem["source"] == "firrtl_port_geometry"
        assert "not a datapath element width" in mem["row_element_note"]

    def test_an_unpinned_store_publishes_no_capacity(self, tmp_path):
        mem = self._mems(tmp_path, _FIR6_UNPINNED)[0]
        assert mem["bytes"] is None and mem["banks"] is None
        assert (mem["banks_min"], mem["banks_max"]) == (1, 8)
        assert "bytes_unknown" in mem

    def test_an_unreadable_elaboration_contributes_nothing(self, tmp_path):
        assert CI.memories_from_port_geometry([tmp_path / "absent.fir"]) == []


class TestWiring:
    def test_a_census_result_is_never_overwritten(self, tmp_path):
        # This is the gap filler for an elaboration a census cannot see into, not a competing reading
        # of one it can.
        facts = {"memories": [{"name": "scratchpad", "bytes": 262144, "source": "firrtl_census"}]}
        before = [dict(m) for m in facts["memories"]]
        assert CI._memories_from_ports("no-such-target", facts) == []
        assert facts["memories"] == before

    def test_a_target_with_no_elaboration_gains_nothing(self):
        facts: dict = {}
        assert CI._memories_from_ports("no-such-target", facts) == []
        assert "memories" not in facts, ("an empty list would say the device has NO on-chip store, "
                                         "which is a claim; the absence of the key is the UNKNOWN")


class TestTheDerivedStoreIsAddressable:
    def _facts(self, tmp_path):
        f = tmp_path / "Store.fir"
        f.write_text(_FIR6)
        return {"memories": CI.memories_from_port_geometry([f])}

    def test_a_store_declaring_its_own_row_width_yields_a_capacity(self, tmp_path):
        # Without this the row width came only from `array cols x datapath element bits`, so a target
        # with no `datapaths` had no row at all and every memory-regime cell for it read `unknown`.
        space = AS.derive_address_space("t", facts=self._facts(tmp_path))
        (store,) = space.stores
        assert store.row_bytes == 32 and store.total_rows == 49152
        assert store.capacity_rows() == 49152
        assert "row_bits_rtl" in store.sources["row_bytes"]

    def test_the_bank_count_is_corroborated_by_an_independent_route(self, tmp_path):
        # `banks` here is bytes/row_bytes/depth -- arithmetic that never saw the range-check literal.
        # It landing on the same 6 is a cross-check, not a restatement.
        (store,) = AS.derive_address_space("t", facts=self._facts(tmp_path)).stores
        assert store.banks == 6 and store.bank_residue_rows == 0 and store.row_residue_bytes == 0

    def test_an_array_derived_row_width_still_wins_where_one_exists(self, tmp_path):
        # The fallback must not move any target whose row width already derives from its array.
        facts = self._facts(tmp_path)
        facts["arrays"] = [{"name": "mesh", "rows": 16, "cols": 16}]
        facts["datapaths"] = [{"name": facts["memories"][0]["name"], "dtype": "i8"}]
        (store,) = AS.derive_address_space("t", facts=facts).stores
        assert store.row_bytes == 16, "the array x datapath derivation is unchanged where it applies"
        assert any(u.quantity == "row_bytes" and "DISAGREE" in u.reason
                   for u in AS.derive_address_space("t", facts=facts).unknowns), \
            "a store whose two row widths disagree must SAY so rather than pick one silently"
