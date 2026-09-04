"""The compute element's datapath, read off its own ports — and refused when the RTL does not state it.

A target whose census produces no ``datapaths`` does not thereby have none: it has a contract that
DECLARES an operand format and an accumulate format from the generator's parameter class and admits, in
its own words, that the claim is not RTL-grounded. That declaration is the thing these tests exist to
replace, so what they pin is not "a number comes out" but "the number that comes out is the one the
elaboration states, and where the elaboration states none, nothing comes out".

Three properties, each of which failed silently in the family of readers this one joins:

* the WIDTHS come from the elaborated ports, never from a name. The cell's identifier may say ``E4M3``
  and its ports say 8 bits; only the second is a measurement;
* a NAME is what disambiguates a width, and only where the naming module also carries a port of that
  width. Eight bits is int8 or either OCP fp8 encoding, and a reader that picks by convention hands a
  bit-exact oracle the wrong decoder;
* a target that ALREADY has census datapaths is untouched. Those read the operand and accumulator SRAMs
  directly and are what every consumer has been calibrated against; a second opinion that silently
  displaced them would move numbers nobody asked to move.
"""
from __future__ import annotations

import textwrap

from merlin.targetgen import families
from merlin.targetgen.rtl import circt_introspect as CI
from merlin.targetgen.rtl import datapaths as DP

# A two-level compute cell, trimmed to the shape a real elaboration has: the cell declares its operand
# and accumulation widths as bare scalars in one bundle (the operands flipped INTO an output bundle),
# and the format names live one and two levels below it — the fused unit names the operand format, the
# rounding adder underneath it names the accumulate format. Nothing is named in the cell itself.
_MAC_CELL = textwrap.dedent("""\
    FIRRTL version 6.0.0
    circuit Grid :
      public module Grid :
        input clock : Clock
        output io : { flip go : UInt<1>}
        inst cell of Cell @[a/b/Grid.scala 10:7]
      module E4M3ProdAddBF16 :
        input clock : Clock
        output io : { flip prod13 : UInt<13>, flip addend16 : UInt<16>, out16 : UInt<16>}
      module E4M3Mul :
        input clock : Clock
        output io : { flip a : UInt<8>, flip b : UInt<8>, out : UInt<13>}
      module E4M3FMA :
        input clock : Clock
        output io : { flip a : UInt<8>, flip b : UInt<8>, flip addend16 : UInt<16>, out16 : UInt<16>}
        inst mul of E4M3Mul @[a/b/E4M3FMA.scala 24:19]
        inst addRound of E4M3ProdAddBF16 @[a/b/E4M3FMA.scala 28:24]
      module Cell :
        input clock : Clock
        output io : { flip act : UInt<8>, flip weight0 : UInt<8>, flip weight1 : UInt<8>, flip sel : UInt<1>, flip addend : UInt<16>, actQ : UInt<8>, mac : UInt<16>, selQ : UInt<1>}
        inst fma of E4M3FMA @[a/b/Cell.scala 101:23]
    """)

# The same cell with the arithmetic children renamed to say nothing about format. The geometry is
# identical, so a reader that answered from the width alone would answer identically here — which is
# exactly the failure this fixture exists to catch.
_UNNAMED_CELL = (_MAC_CELL
                 .replace("E4M3ProdAddBF16", "ProdAddRound")
                 .replace("E4M3Mul", "Mul")
                 .replace("E4M3FMA", "FusedMulAdd"))

_ARRAY_FACTS = {"arrays": [{"name": "mesh", "element": "Cell", "container": "Grid", "rows": 4,
                            "cols": 4}]}


def _fir(tmp_path, text: str, name: str = "Grid.fir"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


class TestWidthsComeFromThePorts:
    def test_the_cell_datapath_is_derived(self):
        rec, why = DP.cell_datapath(_MAC_CELL, "Cell")
        assert rec is not None, why
        # The accumulation chain is the width that both ENTERS and LEAVES the cell; the operand is the
        # widest inward field below it. Neither is read off a name.
        assert (rec.operand_bits, rec.accum_bits) == (8, 16)
        assert rec.operand_ports == ("act", "weight0", "weight1")
        assert rec.accum_in_ports == ("addend",) and rec.accum_out_ports == ("mac",)

    def test_a_one_bit_control_line_is_not_a_datapath(self):
        # `sel`/`selQ` are one bit and appear on both sides, so a reader taking any in/out width pair
        # could call them the accumulation chain of a cell that has a real one.
        rec, _ = DP.cell_datapath(_MAC_CELL, "Cell")
        assert 1 not in (rec.operand_bits, rec.accum_bits)

    def test_a_module_that_is_not_there_is_reported_not_assumed(self):
        rec, why = DP.cell_datapath(_MAC_CELL, "NoSuchCell")
        assert rec is None and "not declared" in why

    def test_a_cell_with_no_accumulation_chain_is_refused(self):
        # A pure feed-through: every width enters or leaves, none does both. It is not a MAC cell and
        # naming one of its widths an accumulator would invent a datapath.
        pipe = textwrap.dedent("""\
            FIRRTL version 6.0.0
            circuit P :
              public module P :
                output io : { flip a : UInt<8>, flip b : UInt<8>, out : UInt<32>}
            """)
        rec, why = DP.cell_datapath(pipe, "P")
        assert rec is None and "accumulation chain" in why


class TestACellThatCarriesARowPerPort:
    """A mesh cell may take a ROW of operands per port (``SInt<8>[N]``) instead of a scalar.

    That is the other real systolic spelling in this repo, and reading it as "no scalar data field"
    would report a reader's gap as a property of the design. The element width is the datapath's, and
    the vector length is how many of them arrive at once -- not a different width.
    """

    _ROW_CELL = textwrap.dedent("""\
        FIRRTL version 6.0.0
        circuit Grid :
          public module Grid :
            output io : { flip go : UInt<1>}
          module Cell :
            input clock : Clock
            output io : { flip in_a : SInt<8>[1], flip in_b : SInt<20>[1], flip in_d : SInt<20>[1], flip in_control : { dataflow : UInt<1>, propagate : UInt<1>}[1], flip in_id : UInt<3>[1], out_a : SInt<8>[1], out_c : SInt<20>[1], out_b : SInt<20>[1], out_id : UInt<3>[1]}
        """)

    def test_the_element_width_is_read_through_the_vector(self):
        rec, why = DP.cell_datapath(self._ROW_CELL, "Cell")
        assert rec is not None, why
        # The accumulation chain is 20 bits (in and out), the operand 8 -- the widths of the ELEMENTS,
        # not of the rows.
        assert (rec.operand_bits, rec.accum_bits) == (8, 20)

    def test_an_unnamed_integer_cell_still_refuses_a_dtype(self):
        # Nothing in this design names a format, so both widths stay UNKNOWN -- including the 20-bit
        # chain, which no registered format has at all.
        rec, _ = DP.cell_datapath(self._ROW_CELL, "Cell")
        assert rec.operand_dtype is None and rec.accum_dtype is None
        assert "no format NAME" in rec.accum_dtype_why


class TestTheNameDisambiguatesTheWidth:
    def test_the_instance_closure_names_both_formats(self):
        rec, _ = DP.cell_datapath(_MAC_CELL, "Cell")
        # The operand format is named one level down, the accumulate format two levels down: a reader
        # that stopped at the cell's own children would derive the first and refuse the second.
        assert rec.operand_dtype == "fp8_e4m3"
        assert rec.accum_dtype == "bf16"
        assert rec.naming[16][1] == "E4M3ProdAddBF16"

    def test_the_dtype_tokens_are_registry_names(self):
        from merlin.common import quant_formats as qf
        rec, _ = DP.cell_datapath(_MAC_CELL, "Cell")
        for token, bits in ((rec.operand_dtype, 8), (rec.accum_dtype, 16)):
            assert qf.get(token).element_bits == bits, "the token must resolve, and to that width"

    def test_a_longer_name_wins_over_a_substring_of_itself(self):
        # `bf16` CONTAINS `f16`, an alias of a different 16-bit format. A reader collecting every
        # substring hit would call this width ambiguous and refuse a fact the RTL states plainly.
        assert DP.format_tokens("E4M3ProdAddBF16") == ("fp8_e4m3", "bf16")

    def test_a_digit_free_word_is_not_a_format_name(self):
        # `half` and `double` are registry aliases AND ordinary English words. A module called
        # `HalfAdder` is describing a carry structure, and reading fp16 out of it would name a 16-bit
        # datapath on the strength of that. Every real format spelling states a number.
        assert DP.format_tokens("HalfAdder") == () and DP.format_tokens("DoubleBuffer") == ()
        assert DP.format_tokens("FP32Acc") == ("fp32",)

    def test_a_format_named_by_a_module_that_lacks_that_width_is_not_evidence(self):
        # The rounding adder's name says E4M3 too, but it carries no 8-bit port — so it is naming its
        # neighbour's format, not its own datapath, and must not be counted as evidence about 8 bits.
        rec, _ = DP.cell_datapath(_MAC_CELL, "Cell")
        assert rec.naming[8][1] == "E4M3FMA"


class TestAnAmbiguousWidthIsRefused:
    def test_an_unnamed_width_yields_no_dtype(self):
        rec, why = DP.cell_datapath(_UNNAMED_CELL, "Cell")
        assert rec is not None, why
        # The geometry is unchanged -- the widths are still measured.
        assert (rec.operand_bits, rec.accum_bits) == (8, 16)
        # ...and neither width is given a name, because nothing in the design states one.
        assert rec.operand_dtype is None and rec.accum_dtype is None

    def test_the_refusal_says_what_the_width_could_have_been(self):
        rec, _ = DP.cell_datapath(_UNNAMED_CELL, "Cell")
        # The candidate list is read out of the format registry, so it cannot drift from the vocabulary
        # the rest of the repo uses -- and a reader can see exactly what evidence would settle it.
        assert "int8" in rec.operand_dtype_why and "fp8_e4m3" in rec.operand_dtype_why
        assert "convention, not a measurement" in rec.operand_dtype_why

    def test_two_formats_at_one_width_fail_closed(self):
        # A design naming BOTH fp8 encodings at 8 bits: a vote between two encodings is not evidence.
        both = _MAC_CELL.replace("module E4M3Mul :", "module E5M2Mul :").replace(
            "inst mul of E4M3Mul", "inst mul of E5M2Mul")
        rec, _ = DP.cell_datapath(both, "Cell")
        assert rec.operand_dtype is None
        assert "fp8_e4m3" in rec.operand_dtype_why and "fp8_e5m2" in rec.operand_dtype_why

    def test_the_refusal_reaches_the_fact_entry(self):
        rec, _ = DP.cell_datapath(_UNNAMED_CELL, "Cell")
        entries = rec.to_facts()
        assert [e["name"] for e in entries] == [DP.OPERAND_ROLE, DP.ACCUM_ROLE]
        for e in entries:
            # UNKNOWN, carrying its reason -- never a width dressed up as a format.
            assert e["dtype"] is None and e["dtype_unknown"]
            assert isinstance(e["elem_bits"], int)


class TestTheElementIsFoundByKindNotByName:
    def test_every_known_kind_declares_how_its_element_is_located(self):
        for kind in families.known_kinds():
            assert families.family_profile(kind).compute_element in (
                "array_element", "lane_replication", "none")

    def test_an_array_kind_takes_the_element_from_the_discovered_array(self):
        found, notes = DP.compute_elements(("systolic",), _ARRAY_FACTS)
        assert found == ("Cell",) and not notes

    def test_a_lane_kind_takes_the_group_replicated_once_per_lane(self):
        facts = {"simt": {"lanes_per_warp": 16},
                 "replication_groups": [{"container": "Core", "element": "Lane", "instances": 16},
                                        {"container": "Core", "element": "Bank", "instances": 4}]}
        found, notes = DP.compute_elements(("simt",), facts)
        assert found == ("Lane",) and not notes

    def test_several_groups_replicated_per_lane_fail_closed(self):
        # Measured on a real SIMT elaboration: 23 distinct modules are instantiated once per lane and
        # all but one are per-lane INTERCONNECT. Taking the first would publish a bus monitor's port
        # widths as the machine's arithmetic, which is the exact shape of wrong number this whole
        # derivation exists to stop.
        facts = {"simt": {"lanes_per_warp": 16},
                 "replication_groups": [{"element": "Lane", "instances": 16},
                                        {"element": "LaneQueue", "instances": 16}]}
        found, notes = DP.compute_elements(("simt",), facts)
        assert found == ()
        assert notes and "UNKNOWN" in notes[0] and "interconnect" in notes[0]

    def test_a_replication_the_array_discovery_declined_is_not_the_element(self):
        # `geometry_unknown` is the array discovery saying "this is the widest sibling group and I will
        # not call it the compute array". Reading its element as the compute cell takes that back.
        facts = {"arrays": [{"name": "widest_replication", "element": "ComparePipe",
                             "geometry_unknown": "18 is not a perfect square"}]}
        found, notes = DP.compute_elements(("systolic",), facts)
        assert found == () and notes and "declined" in notes[0]

    def test_a_lane_kind_with_no_replication_census_says_so(self):
        found, notes = DP.compute_elements(("simt",), {"simt": {"lanes_per_warp": 16}})
        assert found == ()
        assert notes and "replication groups" in notes[0]

    def test_a_kind_with_no_replicated_element_says_so(self):
        found, notes = DP.compute_elements(("scalar",), _ARRAY_FACTS)
        # A scalar pipe has no cell to read even on a design that HAS an array fact: the routing is on
        # the kind, so it must not pick up the neighbouring array's element.
        assert found == () and notes and "no replicated compute element" in notes[0]

    def test_no_kind_at_all_is_reported_not_silent(self):
        found, notes = DP.compute_elements((), _ARRAY_FACTS)
        assert found == () and notes and "UNKNOWN" in notes[0]

    def test_an_unknown_kind_is_reported_not_guessed(self):
        found, notes = DP.compute_elements(("wishful",), _ARRAY_FACTS)
        assert found == () and notes and "not a known compute-unit kind" in notes[0]


class TestFactAssembly:
    def test_the_entries_are_census_shaped_and_tagged(self, tmp_path):
        dps, notes = DP.datapaths_from_compute_cells(
            _ARRAY_FACTS, [_fir(tmp_path, _MAC_CELL)], ("systolic",))
        assert not notes
        assert [(d["name"], d["dtype"], d["elem_bits"]) for d in dps] == [
            ("input", "fp8_e4m3", 8), ("accumulator", "bf16", 16)]
        for d in dps:
            # Same keys a census entry carries, so no consumer needs to know where it came from -- plus
            # a source tag, so a reader who cares CAN tell.
            assert d["source"] == DP.SOURCE != "firrtl_census"
            assert d["module"] == "Cell" and d["evidence"]

    def test_two_elaborations_that_disagree_publish_nothing(self, tmp_path):
        # The same cell, elaborated twice with different widths: the design has two configurations and
        # which one is under test is UNKNOWN. Publishing either would attribute a number to a device
        # that may not be the one being graded.
        other = _MAC_CELL.replace("flip addend : UInt<16>", "flip addend : UInt<32>").replace(
            "mac : UInt<16>", "mac : UInt<32>")
        dps, notes = DP.datapaths_from_compute_cells(
            _ARRAY_FACTS, [_fir(tmp_path, _MAC_CELL), _fir(tmp_path, other, "Grid2.fir")],
            ("systolic",))
        assert dps == []
        assert any("disagree" in n for n in notes)

    def test_an_unreadable_elaboration_contributes_a_reason(self, tmp_path):
        dps, notes = DP.datapaths_from_compute_cells(
            _ARRAY_FACTS, [tmp_path / "absent.fir"], ("systolic",))
        assert dps == [] and notes


class TestACensusTargetIsUntouched:
    """A target whose facts already carry datapaths must come out of the fact assembly bit-identical."""

    def _facts_with_census_datapaths(self) -> dict:
        return {"arrays": [{"name": "mesh", "element": "Cell"}],
                "datapaths": [{"name": "input", "dtype": "i8", "evidence": "operand smem UInt<8>"},
                              {"name": "accumulator", "dtype": "i32", "evidence": "accum SInt<32>"}]}

    def test_existing_datapaths_are_not_displaced(self, monkeypatch, tmp_path):
        facts = self._facts_with_census_datapaths()
        before = [dict(d) for d in facts["datapaths"]]
        # Point the elaboration walk at a cell that WOULD derive a different reading, so the no-op is
        # proven against a live alternative rather than against an empty directory.
        monkeypatch.setattr(CI, "elaborated_firrtl", lambda t: [_fir(tmp_path, _MAC_CELL)])
        sourced = CI._datapaths_from_cells("t", facts)
        assert sourced == []
        assert facts["datapaths"] == before
        assert "datapaths_undeterminable" not in facts

    def test_a_target_with_no_elaboration_records_the_gap(self, monkeypatch):
        facts = {"arrays": [{"name": "mesh", "element": "Cell"}]}
        monkeypatch.setattr(CI, "elaborated_firrtl", lambda t: [])
        assert CI._datapaths_from_cells("t", facts) == []
        assert "datapaths" not in facts, "an absent elaboration is UNKNOWN, never an empty datapath list"
        assert facts["datapaths_undeterminable"], "and the gap must be visible"
