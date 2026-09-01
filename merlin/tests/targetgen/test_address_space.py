"""The on-chip address space is a DERIVED artifact, and it is counted in ROWS.

A backend does not address bytes. It addresses rows of an SRAM whose width is the edge of the array it
feeds, and both of the memory failures this repo has measured were row-count failures that no artifact
stated:

  * ``vector::_M_range_check: __n (which is 16384) >= this->size() (which is 16384)`` -- a lowering that
    addressed all ``kt*nt`` weight tiles of a 512x512 layer as simultaneously resident, against a
    16384-row scratchpad;
  * ``__n (which is 1024) >= this->size() (which is 1024)`` -- four whole-model layers, (345,32)@(32,256)
    and (96,64)@(64,512), whose operands fit the 262144-byte scratchpad easily and whose OUTPUT overran
    the accumulator's 1024 rows.

Neither 16384 nor 1024 appears in the RTL facts: the artifact declares ``bytes`` and ``depth`` and stops.
These tests pin that the two numbers (and the bank counts, and whether the accumulator is a second
address space at all) are DERIVED from the target's own geometry and datapath widths, corroborated
against the SRAM widths mlc reads out of the RTL, and that everything not derivable stays UNKNOWN --
distinguishably from "declared absent", and never as a zero.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.address_space import (
    ABSENT,
    DERIVED,
    UNKNOWN,
    corroborate,
    derive_address_space,
    element_bits,
    working_set_rows,
)
from merlin.targetgen.rtl import facts as rtl_facts

TARGET = "gemmini"          # the one target whose RTL facts ship in-tree; the numbers below are its RTL


def _space(target: str = TARGET):
    if not rtl_facts.rtl_facts_path(target).is_file() and rtl_facts._committed_facts_path(target) is None:
        pytest.skip(f"no RTL-facts artifact for {target!r} in this checkout (extraction needs CIRCT/mlc)")
    sp = derive_address_space(target)
    if sp.stores_status != DERIVED:
        pytest.skip(f"{target} declares no on-chip stores here: {[u.to_dict() for u in sp.unknowns]}")
    return sp


# ------------------------------------------------------------------ the derivation, against real RTL

def test_the_row_width_is_derived_never_declared_and_both_routes_agree():
    """16 bytes is not a constant anyone typed, and it is derivable TWO independent ways.

    The two stores of one device share a geometry and differ by element width -- a 16-column array over
    an 8-bit operand datapath gives a 16-byte operand row, the same 16 columns over the declared 32-bit
    accumulate word give a 64-byte accumulator row. Separately, each store's representative SRAM bank
    declares its own word width in the RTL, and mlc discovery reads it.

    Both routes are kept. The array route is the only one available on a target whose extractor carries no
    per-memory word width; the SRAM route is the only one available on an archetype with no compute array
    to take an edge from (measured: the SIMT target's 64-bank, 4-byte-row shared memory). Where both
    exist they must AGREE -- that equality is the check that catches a row width wrong by the packing
    factor, and it is asserted here rather than assumed.
    """
    sp = _space()
    assert sp.array_cols and sp.array_rows, "no array geometry, no row width"
    for st in sp.stores:
        assert st.element_bits, f"{st.name}: element width must be linked to a declared datapath"
        assert st.row_bytes == sp.array_cols * (st.element_bits // 8), st.to_dict()
        prov = st.sources["row_bytes"]
        assert prov.startswith("array cols") or "RTL SRAM word width" in prov, \
            f"the provenance must name which derivation produced the row: {prov!r}"
        # Neither route may contradict the other: the value above IS the array product, so if the
        # provenance names the SRAM word, the SRAM word equals the array product.
        assert not [u for u in sp.unknowns
                    if u.quantity == "row_bytes" and u.store == st.name], \
            f"{st.name}: the two row-width derivations disagree -- {[u.to_dict() for u in sp.unknowns]}"


def test_the_row_totals_and_banks_are_the_numbers_the_simulator_aborted_on():
    """16384 operand rows in 4 banks, 1024 accumulator rows in 2 -- from bytes/depth, which is all the
    facts give. ``depth`` is the PER-BANK row count; reading it as the address limit under-addresses the
    store by exactly the bank factor."""
    sp = _space()
    for st in sp.stores:
        assert st.total_rows == st.nbytes // st.row_bytes
        assert st.banks == st.total_rows // st.depth
        assert st.row_residue_bytes == 0 and st.bank_residue_rows == 0, "an inexact fit is a wrong width"
        # the independent route: bytes per depth index must be a whole number of rows, and the multiple
        # IS the bank count. Two derivations, one answer -- a wrong row width breaks this equality.
        assert st.bytes_per_depth_entry == st.banks * st.row_bytes


def test_the_derivation_agrees_with_the_sram_widths_read_out_of_the_rtl():
    """Corroboration, matched by (bytes, depth) rather than by name: mlc reports RTL instance paths while
    the facts carry role names, and pairing those by spelling is how a conformant target gets dropped."""
    sp = _space()
    rep = corroborate(TARGET, sp)
    if not rep["available"]:
        pytest.skip(f"mlc discovery unavailable: {rep['reason']}")
    assert rep["agree"] is True, rep
    for row in rep["stores"]:
        assert row["derived_row_bytes"] == row["rtl_row_bytes"], row
        assert row["rtl_banks"] == sp.store(row["store"]).banks, row


def test_the_accumulator_is_a_separate_address_space():
    """Two stores whose rows are different sizes cannot be one flat space -- a row index means 16 bytes in
    one and 64 in the other. Writing an accumulator row index where an operand row index belongs is not a
    range-check abort, it is silently wrong data, which is why this has to be stated as a fact."""
    sp = _space()
    assert sp.separate_accumulator_space is True
    assert len(sp.row_widths) > 1, sp.row_widths


# ------------------------------------------------- absent vs undeterminable, and never a measured zero

def _artifact(memories, *, arrays=(("mesh", 16, 16),), datapaths=(("input", "i8", "scratchpad smem"),)):
    """A synthetic facts artifact in the shape ``load_facts`` returns, so the semantics below are pinned
    without depending on which target happens to have been extracted in this checkout."""
    body = {"arrays": [{"name": n, "rows": r, "cols": c} for n, r, c in arrays],
            "datapaths": [{"name": n, "dtype": d, "evidence": e} for n, d, e in datapaths]}
    if memories is not None:
        body["memories"] = memories
    return {"schema_version": "2.0", "inputs": {}, "facts": body}


def test_an_empty_store_list_and_a_missing_one_are_different_states():
    """The distinction this repo keeps re-learning: "the extractor ran and found no on-chip store" is a
    fact about the device; "the artifact carries no memory list" is a fact about our extraction. One is
    answerable by reading harder, the other is not, and collapsing them hides which."""
    absent = derive_address_space("t_absent", facts=_artifact([]))
    unknown = derive_address_space("t_unknown", facts=_artifact(None))
    assert absent.stores_status == ABSENT and unknown.stores_status == UNKNOWN
    assert absent.stores == () and unknown.stores == ()
    assert "no on-chip store" in absent.unknowns[0].reason
    assert "UNKNOWN" in unknown.unknowns[0].reason
    for sp in (absent, unknown):
        assert sp.separate_accumulator_space is None, "with no stores, 'no second space' is not a finding"


def test_a_target_with_no_facts_reports_unknown_and_never_zero():
    sp = derive_address_space("definitely_not_a_target")
    assert sp.stores_status == UNKNOWN and sp.stores == ()
    assert sp.separate_accumulator_space is None
    assert {"stores", "separate_accumulator_space"} <= set(sp.unknown_quantities())
    d = sp.to_dict()
    assert 0 not in (d["separate_accumulator_space"], d["array"]), "an unread quantity is not a zero"


def test_a_real_target_whose_banks_are_discovered_but_unclassified_stays_unknown():
    """mlc discovers 39 SRAM banks for this target and classifies none of them, so nothing says which
    bank the compute unit reads operands from and the capacity obligation is undecidable -- correctly.

    UNKNOWN here is a fact about our extraction, and the reason must say so: the banks ARE known, only
    their ROLE is not, which sends the reader to the classifier rather than to the extractor.
    """
    target = "atlas"
    if not rtl_facts.rtl_facts_path(target).is_file():
        pytest.skip(f"no RTL-facts artifact for {target!r} in this checkout")
    sp = derive_address_space(target)
    assert sp.stores_status == UNKNOWN, [u.to_dict() for u in sp.unknowns]
    assert sp.stores == () and sp.separate_accumulator_space is None


def test_the_simt_target_derives_a_store_without_any_compute_array():
    """This target was previously reported ABSENT -- "declared to have no on-chip store of its own" --
    which was a claim about hardware that has a 64-bank, 131072-byte shared memory.

    It read as absent because the row width was derived ONLY as the compute array's column edge times
    the datapath element width, and this machine has no array. The store's own RTL SRAM word width is the
    direct measurement, and with it a row is derivable on an archetype with no array at all. ABSENT
    remains covered where it belongs: over a synthetic artifact that really does declare ``memories: []``
    (see the test above), which is the only place that state can be asserted without claiming it of a
    real device.
    """
    target = "muon"
    if not rtl_facts.rtl_facts_path(target).is_file():
        pytest.skip(f"no RTL-facts artifact for {target!r} in this checkout")
    sp = derive_address_space(target)
    assert sp.stores_status == DERIVED, [u.to_dict() for u in sp.unknowns]
    assert sp.array_cols is None, "the point of this case is that there is no array to measure against"
    operand = min((s for s in sp.stores if s.row_bytes), key=lambda s: s.row_bytes)
    assert operand.row_bytes and operand.total_rows == operand.nbytes // operand.row_bytes
    assert "RTL SRAM word width" in operand.sources["row_bytes"], operand.sources


def test_an_unlinkable_element_width_leaves_the_row_unknown_not_assumed():
    """A store no datapath declares or evidences has no derivable element width, so no row width, so no
    row total. Assuming a byte would produce a full set of plausible numbers that are wrong by the
    accumulate factor -- measured elsewhere as a 4x too-generous accumulator bound."""
    sp = derive_address_space("t_unlinked", facts=_artifact([{"name": "buffer", "bytes": 4096, "depth": 64}]))
    st = sp.store("buffer")
    assert st.element_dtype is None and st.element_bits is None
    assert st.row_bytes is None and st.total_rows is None and st.banks is None
    assert "element_dtype" in sp.unknown_quantities() and "total_rows" in sp.unknown_quantities()
    assert st.working_set_rows((16, 16)) is None, "an uncomputable working set is not an empty one"


def test_an_ambiguous_array_leaves_the_row_unknown():
    """Two arrays with extents: which one a store feeds decides the row width, and choosing would be an
    assumption wearing a derivation's clothes."""
    sp = derive_address_space(
        "t_two_arrays",
        facts=_artifact([{"name": "scratchpad", "bytes": 4096, "depth": 64}],
                        arrays=(("mesh", 16, 16), ("other", 8, 8))))
    assert sp.array_cols is None
    assert sp.store("scratchpad").row_bytes is None
    assert any(u.quantity == "row_elems" and "2 arrays" in u.reason for u in sp.unknowns)


def test_a_residue_is_reported_rather_than_rounded_away():
    """A store whose rows do not divide into banks of the declared depth withholds the bank count and
    keeps the residue: rounding it would invent an addressing scheme the hardware does not have."""
    sp = derive_address_space(
        "t_residue", facts=_artifact([{"name": "scratchpad", "bytes": 1600, "depth": 64}]))
    st = sp.store("scratchpad")
    assert st.row_bytes == 16 and st.total_rows == 100
    assert st.banks is None and st.bank_residue_rows == 36
    assert any(u.quantity == "banks" and "36 rows over" in u.reason for u in sp.unknowns)


def test_a_sub_byte_datapath_will_not_guess_whether_the_row_packs():
    """Whether an SRAM packs two 4-bit elements into a byte or pads each to one is a wiring fact the
    facts do not carry, and the two answers differ by the packing factor. Refused, not halved."""
    sp = derive_address_space(
        "t_packed", facts=_artifact([{"name": "scratchpad", "bytes": 4096, "depth": 64}],
                                    datapaths=(("input", "mxfp4", "scratchpad smem"),)))
    st = sp.store("scratchpad")
    assert st.element_bits == 4, "the width itself is known; only the row layout is not"
    assert st.row_bytes is None and st.total_rows is None and st.banks is None
    assert any(u.quantity == "row_bytes" and "not byte-aligned" in u.reason for u in sp.unknowns)


# ---------------------------------------------------------------------------- rows, not element counts

def test_working_set_rows_is_the_row_the_backend_addresses():
    """The 512x512 weight tile the aborting lowering held resident is EXACTLY the whole store: 512 rows of
    512 int8 = 32 rows each = 16384 rows against 16384. That equality is the abort, stated as an
    obligation instead of as a simulator message."""
    sp = _space()
    st = min(sp.stores, key=lambda s: s.row_bytes)          # the narrow-row operand store
    assert st.working_set_rows((512, 512), "int8") == st.total_rows == 16384
    assert st.working_set_rows((16, 512), "int8") + st.working_set_rows((512, 512), "int8") > st.total_rows


def test_row_accounting_and_element_accounting_only_agree_on_whole_rows():
    """Why rows are worth deriving at all. The contract predicate counts the working set in ELEMENTS
    (bytes*8 / element bits), which equals the row count times the row width only when the innermost
    extent fills whole rows. At N=24 on a 16-element row the tensor spends 2 rows per K, not 1.5 -- the
    element count reports 75% of the residency the hardware actually spends, and reports it as a fit."""
    sp = _space()
    st = min(sp.stores, key=lambda s: s.row_bytes)
    per_row = st.elems_per_row("int8")
    assert st.working_set_rows((32, per_row * 2), "int8") == 32 * 2      # whole rows: the two agree
    ragged = st.working_set_rows((32, per_row + 1), "int8")
    assert ragged == 32 * 2, "a partial row is still a whole row"
    assert ragged * per_row > 32 * (per_row + 1), "row accounting must exceed the element count here"


def test_a_wider_element_takes_more_rows_in_the_same_store():
    """Rows are dtype-relative: the same store holds a quarter as many 32-bit words per row as bytes."""
    sp = _space()
    st = min(sp.stores, key=lambda s: s.row_bytes)
    assert st.elems_per_row("i32") * 4 == st.elems_per_row("int8")
    assert working_set_rows(st, (64, 64), "i32") == 4 * working_set_rows(st, (64, 64), "int8")


def test_a_sub_byte_format_is_counted_in_bits_not_rounded_up_to_a_byte():
    sp = _space()
    st = min(sp.stores, key=lambda s: s.row_bytes)
    assert st.elems_per_row("mxfp4") == 2 * st.elems_per_row("int8")


def test_zero_and_scalar_extents_are_not_confused():
    sp = _space()
    st = min(sp.stores, key=lambda s: s.row_bytes)
    assert st.working_set_rows(()) == 1, "a scalar still occupies a row"
    assert st.working_set_rows((0, 16)) == 0, "an empty tensor occupies none"
    with pytest.raises(ValueError):
        st.working_set_rows((-1, 16))


def test_element_bits_fails_closed_on_a_spelling_it_does_not_know():
    """The alternative -- defaulting to 8 -- is what makes an omitted accumulator dtype read a 65536-byte
    accumulator as 65536 elements instead of the 16384 its 32-bit accumulate word allows."""
    assert element_bits("i32") == 32 and element_bits("int8") == 8
    assert element_bits("fp8_e4m3") == 8, "a digit scrape reads this as 843"
    assert element_bits("mxfp4") == 4
    assert element_bits(None) is None and element_bits("not_a_dtype") is None
