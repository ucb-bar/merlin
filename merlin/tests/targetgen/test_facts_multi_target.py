"""RTL fact extraction on MORE THAN ONE target — the anti-overfit test for the FIRRTL reader.

The FIRRTL reader in :mod:`merlin.targetgen.rtl.introspect` was written against one design and matched
its constructs by name (a ``Scratchpad.scala`` ``smem``, an ``AccumulatorMem`` module, a ``Mesh``/
``Tile`` nesting). On any other elaboration every probe missed, and because the probes reported "found
nothing" the same way whether the design lacks the construct or spells it differently, the result was a
bundle with an EMPTY memory list — which downstream reads as "this device has no on-chip store". Two
targets sat in exactly that state with their 130 MB elaborations already on disk.

So the tests here are structured around the property that failed, not around the numbers of any one
device:

* the parsers are exercised on SYNTHETIC FIRRTL that names nothing this repo ships. A reader that can
  only be tested against a real checkout can only be tested on the designs somebody already built;
* WHERE each target's elaboration lives is asserted to come from that target's OWN declaration, and a
  target that declares none is asserted to RAISE rather than fall back to another target's config;
* the real elaborations, when present, are checked for INTERNAL CONSISTENCY (bytes = banks x row x
  depth; the census's own row width against the array-derived one) rather than against numbers copied
  into this file, which would just move the hardcode.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen.rtl import introspect

# A synthetic elaboration in FIRRTL's own surface syntax. Nothing here is any shipped target: the
# generator directory, the module names and the memory names are invented, which is the point — a
# reader that needs a real device to be exercised is a reader that cannot be trusted on a new one.
_SYNTHETIC_FIR = """\
FIRRTL version 3.3.0
circuit TestHarness :
  module Unrelated : @[generators/somethingelse/src/main/scala/Unrelated.scala 1:1]
    smem decoy : UInt<8>[4] [16] @[generators/somethingelse/src/main/scala/Unrelated.scala 2:2]
  module WidgetTop : @[generators/widget/src/main/scala/widget/WidgetTop.scala 10:7]
    input clock : Clock @[generators/widget/src/main/scala/widget/WidgetTop.scala 10:7]
    output io : { flip cmd : { flip ready : UInt<1>, valid : UInt<1>, bits : { inst : { funct : UInt<7>, rs2 : UInt<5>, opcode : UInt<7>}, rs1 : UInt<64>}}, busy : UInt<1>} @[x 1:1]
  module Bank : @[generators/widget/src/main/scala/widget/Bank.scala 20:3]
    smem cells_0 : UInt<8>[8] [256] @[generators/widget/src/main/scala/widget/Bank.scala 21:9]
    smem cells_1 : UInt<8>[8] [256] @[generators/widget/src/main/scala/widget/Bank.scala 21:9]
  module Accum : @[generators/widget/src/main/scala/widget/Accum.scala 5:3]
    smem wide : UInt<128> [64] @[generators/widget/src/main/scala/widget/Accum.scala 6:9]
  module Cell : @[generators/widget/src/main/scala/widget/Cell.scala 1:1]
    skip
"""

_SYNTHETIC_HIER = {
    "instance_name": "ChipTop", "module_name": "ChipTop",
    "instances": [
        {"instance_name": "junk", "module_name": "Unrelated", "instances": []},
        {"instance_name": "top", "module_name": "WidgetTop", "instances": [
            {"instance_name": "bank", "module_name": "Bank", "instances": []},
            {"instance_name": "acc", "module_name": "Accum", "instances": []},
        ] + [{"instance_name": f"cell_{i}", "module_name": "Cell", "instances": []}
             for i in range(10)]},
    ],
}


@pytest.fixture()
def synthetic(tmp_path):
    fir = tmp_path / "synthetic.fir"
    fir.write_text(_SYNTHETIC_FIR, encoding="utf-8")
    hier = tmp_path / "top_module_hierarchy.json"
    hier.write_text(json.dumps(_SYNTHETIC_HIER), encoding="utf-8")
    return fir, hier


# --------------------------------------------------------------------------- the parsers, alone
def test_memory_shape_keeps_a_flat_row_undecomposed():
    """``UInt<8>[32]`` declares 32 elements of 8 bits; ``UInt<256>`` declares ONE 256-bit word.

    Reading the second as "an element of 256 bits laid out like the first" is the bug this pins: the
    store's row width then comes out wrong by the packing factor, and a bank count derived from it is
    wrong by the same factor while looking perfectly derived.
    """
    assert introspect._mem_shape("UInt<8>[32] [8192]") == (8, 32, 8192)
    assert introspect._mem_shape("UInt<256> [64]") == (256, 1, 64)
    assert introspect._mem_shape("SInt<16>[4] [32]") == (16, 4, 32)
    # A bundle payload (a queue's ram) carries no element width at all -> None, never a guess.
    assert introspect._mem_shape("{ opcode : UInt<3>, data : UInt<64>} [2]") is None


def test_memory_declaration_is_split_by_the_language_not_by_a_pattern():
    kw, name, typ, site = introspect._mem_decl(
        "smem banks_0 : UInt<8>[32] [8192] @[generators/w/src/main/scala/w/V.scala 62:16]")
    assert (kw, name, typ) == ("smem", "banks_0", "UInt<8>[32] [8192]")
    assert site == "generators/w/src/main/scala/w/V.scala 62:16"
    assert introspect._mem_decl("wire foo : UInt<8>") is None


def test_store_name_is_derived_from_the_declaration_site():
    """The name is read off the RTL, so two targets differ exactly where their RTL differs."""
    assert introspect._store_name("generators/w/src/main/scala/w/VMEM.scala 62:16", "banks_3") \
        == "vmem.banks"
    assert introspect._store_name("a/b/RegisterFile.scala 61:16", "v0_mask") == "registerfile.v0_mask"


def test_host_command_port_is_classified_by_instruction_shape():
    """A decoupled port whose payload carries a RISC-V instruction bundle IS the co-processor handoff.

    Classified by the ISA's own instruction-format fields, not by a port name, so a design that calls
    it something else is still recognized and a design without one is honestly not.
    """
    rocc = ("output io : { flip cmd : { flip ready : UInt<1>, valid : UInt<1>, "
            "bits : { inst : { funct : UInt<7>, opcode : UInt<7>}, rs1 : UInt<64>}}}")
    assert introspect._host_command_port(rocc) == "cmd"
    plain = "output io : { flip req : { flip ready : UInt<1>, valid : UInt<1>, bits : UInt<32>}}"
    assert introspect._host_command_port(plain) is None


# ------------------------------------------------------------------ the census, on a synthetic design
def test_census_scopes_by_the_elaborations_own_provenance(synthetic):
    """Only modules the FIRRTL says were defined in THIS generator's tree are the target's.

    The decoy SRAM sits in another generator's module and must not be counted: an SoC elaboration is
    mostly somebody else's RTL, and a capacity that swept in the host's caches would over-declare the
    device by orders of magnitude.
    """
    fir, hier = synthetic
    out = introspect.census_facts(fir, hier, generator="widget")
    assert out["census"]["unit_root"] == "WidgetTop"
    assert out["census"]["units"] == 1
    names = {m["name"] for m in out["memories"]}
    assert names == {"bank.cells", "accum.wide"}
    assert "unrelated.decoy" not in names


def test_census_capacities_are_internally_consistent(synthetic):
    """bytes == banks x row_elems x elem_bits/8 x depth, for every store, from the RTL's own numbers."""
    fir, hier = synthetic
    out = introspect.census_facts(fir, hier, generator="widget")
    for mem in out["memories"]:
        assert mem["bytes"] == mem["banks"] * mem["row_elems"] * mem["elem_bits"] * mem["depth"] // 8
    cells = next(m for m in out["memories"] if m["name"] == "bank.cells")
    assert (cells["banks"], cells["row_elems"], cells["elem_bits"], cells["depth"]) == (2, 8, 8, 256)


def test_census_emits_a_datapath_only_where_the_rtl_declares_elements(synthetic):
    """A flat word row yields NO datapath rather than a fabricated element width."""
    fir, hier = synthetic
    out = introspect.census_facts(fir, hier, generator="widget")
    dtypes = {d["name"]: d["dtype"] for d in out["datapaths"]}
    assert dtypes == {"bank.cells": "u8"}          # accum.wide is `UInt<128>`: no element decomposition
    assert any(m["name"] == "accum.wide" for m in out["memories"])   # still reported as a store


def test_census_withholds_geometry_it_cannot_derive(synthetic):
    """10 siblings is not a square, so no row/column extent is claimed and the name says so."""
    fir, hier = synthetic
    array = introspect.census_facts(fir, hier, generator="widget")["arrays"][0]
    assert array["instances"] == 10 and array["element"] == "Cell"
    assert "rows" not in array and "cols" not in array
    assert array["name"] == "widest_replication"
    assert "geometry_unknown" in array


def test_census_reports_an_absent_unit_rather_than_an_empty_device(synthetic):
    """A generator with no module in this elaboration is UNKNOWN, not a device with no memory."""
    fir, hier = synthetic
    out = introspect.census_facts(fir, hier, generator="nothing_elaborated_here")
    assert out["census"]["unit_root"] is None and out["census"]["units"] == 0
    assert "note" in out["census"] and out["memories"] == []


def test_census_counts_per_unit_not_per_soc(tmp_path):
    """A dual-core SoC instantiates the unit twice; a compiler schedules into ONE of them.

    Summing across the design reports twice the capacity the device has, and does it as a derived
    number, which is the worst way to be wrong.
    """
    fir = tmp_path / "dual.fir"
    fir.write_text(
        "circuit T :\n"
        "  module Unit : @[generators/w/src/main/scala/w/Unit.scala 1:1]\n"
        "    skip\n"
        "  module Bank : @[generators/w/src/main/scala/w/Bank.scala 2:1]\n"
        "    smem cells : UInt<8>[4] [64] @[generators/w/src/main/scala/w/Bank.scala 3:1]\n",
        encoding="utf-8")
    hier = tmp_path / "h.json"
    hier.write_text(json.dumps({
        "instance_name": "top", "module_name": "Top", "instances": [
            {"instance_name": f"u{i}", "module_name": "Unit", "instances": [
                {"instance_name": "b", "module_name": "Bank", "instances": []}]}
            for i in range(2)]}), encoding="utf-8")
    out = introspect.census_facts(fir, hier, generator="w")
    assert out["census"]["units"] == 2
    cells = next(m for m in out["memories"] if m["name"] == "bank.cells")
    assert cells["banks"] == 1 and cells["bytes"] == 4 * 64      # one unit's worth, not two


# ------------------------------------------------------------- the target's own declaration of its RTL
@pytest.mark.parametrize("target", ["atlas", "saturn"])
def test_target_declares_its_own_elaboration(target):
    """WHERE the RTL is comes from the TARGET'S file, not from a table in the extractor.

    Both of these had an elaboration on disk and no way to reach it, and one of them lives in a
    DIFFERENT external checkout from the default — which is precisely the fact a table in shared code
    gets wrong and a per-target declaration cannot.
    """
    src = introspect.declared_rtl_source(target)
    assert src.target == target and src.config and src.generator
    assert src.origin.is_file()
    # The declaration is inside a file that belongs to this target, not in library code.
    assert target in src.origin.parts or src.origin.read_text(encoding="utf-8").find(target) != -1


def test_an_undeclared_target_raises_instead_of_borrowing_another_config():
    """Falling back to whichever config the module happens to name would attribute one device's
    structure to another — the failure the provenance convention exists to prevent."""
    with pytest.raises(introspect.RtlSourceUndeclared) as e:
        introspect.declared_rtl_source("no_such_target_declares_rtl")
    assert "declares an elaborated-RTL source" in str(e.value)


# --------------------------------------------------------------- the real elaborations, when present
def _artifacts_or_skip(target):
    try:
        arts = introspect.artifacts_for(target)
    except introspect.RtlSourceUndeclared as e:                    # pragma: no cover - env dependent
        pytest.skip(f"{target}: {e}")
    except KeyError as e:                                          # pragma: no cover - .env not present
        pytest.skip(f"{target}: external checkout not configured ({e})")
    if not arts["fir"].is_file() or not arts["hierarchy"].is_file():
        pytest.skip(f"{target}: declared elaboration not built at {arts['fir']}")
    return arts


@pytest.mark.parametrize("target", ["atlas", "saturn"])
def test_real_elaboration_yields_sized_stores(target):
    """Every store the census reports on a real design carries a capacity it can show its work for."""
    arts = _artifacts_or_skip(target)
    src = arts["source"]
    facts = introspect.extract_facts(arts["fir"], arts["hierarchy"], target=target,
                                     generator=src.generator, config=src.config)
    assert facts["memories"], f"{target}: the elaboration declares SRAMs and none were reported"
    for mem in facts["memories"]:
        assert mem["depth"] and mem["banks"]
        assert mem["bytes"] == mem["banks"] * mem["row_elems"] * mem["elem_bits"] * mem["depth"] // 8
        assert mem["evidence"].count("@") == 1                      # names the Chisel site it came from
    for dp in facts["datapaths"]:
        assert any(m["name"] == dp["name"] for m in facts["memories"])   # linkable to its store


@pytest.mark.parametrize("target", ["atlas", "saturn"])
def test_real_elaboration_corroborates_the_row_width_two_ways(target):
    """The array-derived row width must agree with the SRAM's own declared row.

    ``address_space`` derives a store's row as ``array cols x datapath element``. The FIRRTL declares
    the row directly. They are independent readings of the same wire, so a disagreement means one of
    the two derivations is wrong — and silently believing the derived one is how a backend gets handed
    a row count the hardware does not have.
    """
    from merlin.targetgen import address_space as AS

    _artifacts_or_skip(target)
    space = AS.derive_address_space(target)
    body = __import__("merlin.targetgen.rtl.facts", fromlist=["x"]).load_facts(target)["facts"]
    by_name = {m["name"]: m for m in body.get("memories") or [] if "row_bits_rtl" in m}
    checked = 0
    for store in space.stores:
        mem = by_name.get(store.name)
        if mem is None or store.row_bytes is None:
            continue                     # a store whose row width is UNKNOWN makes no claim to check
        assert store.row_bytes * 8 == mem["row_bits_rtl"], (
            f"{target}/{store.name}: address_space derives a {store.row_bytes}-byte row from the array "
            f"geometry, the RTL declares {mem['row_bits_rtl']} bits")
        checked += 1
    if checked:
        assert space.stores_status == AS.DERIVED
