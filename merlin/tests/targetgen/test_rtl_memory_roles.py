"""Declared memory roles name scoped-census stores without double counting."""

from merlin.targetgen.rtl.memory_roles import label_roles, separate_backings

_PROBE = {
    "name": "buffer",
    "banks": 4,
    "row_elems": 16,
    "depth": 64,
    "elem_bits": 8,
    "bytes": 4096,
    "evidence": "4x `smem mem : UInt<8>[16] [64]` @ Buffer.scala",
}
_CENSUS_BUFFER = {
    "name": "buffer.mem",
    "banks": 4,
    "depth": 64,
    "row_elems": 16,
    "elem_bits": 8,
    "row_bits_rtl": 128,
    "bytes": 4096,
    "source": "firrtl_census",
    "evidence": "4x `smem mem : UInt<8>[16] [64]` in ['Bank'] @ generators/accel/src/Buffer.scala 9:2 (per Accel unit)",
}
_CENSUS_BACKING = {
    "name": "syncmem.mem",
    "banks": 1,
    "copies": 2,
    "depth": 32,
    "row_elems": 64,
    "elem_bits": 8,
    "row_bits_rtl": 512,
    "bytes": 2048,
    "source": "firrtl_census",
    "evidence": "1x `smem mem : UInt<8>[64] [32]` in ['Sync'] @ generators/accel/src/SyncMem.scala 4:4",
}
_VIEW = {
    "name": "accumulator",
    "banks": 2,
    "depth": 32,
    "row_bytes": 64,
    "bytes": 4096,
    "source": "selected_hw_port_geometry",
    "view_kind": "address_space",
}


def test_census_store_takes_the_declared_role_name():
    rows = label_roles([_CENSUS_BUFFER, _CENSUS_BACKING], [_PROBE])
    assert [row["name"] for row in rows] == ["buffer", "syncmem.mem"]
    assert rows[0]["census_name"] == "buffer.mem" and rows[0]["view_kind"] == "address_space"
    assert rows[0]["source"] == "firrtl_census"


def test_role_probes_remain_the_stores_without_a_census():
    rows = label_roles([], [_PROBE, {**_PROBE, "name": "unsized", "bytes": None}])
    assert [(row["name"], row["source"]) for row in rows] == [("buffer", "selected_firrtl_role_probe")]


def test_geometry_or_site_mismatch_keeps_the_census_name():
    other_site = {**_PROBE, "evidence": "4x `smem mem : UInt<8>[16] [64]` @ Other.scala"}
    assert label_roles([_CENSUS_BUFFER], [other_site])[0]["name"] == "buffer.mem"
    assert label_roles([_CENSUS_BUFFER], [{**_PROBE, "depth": 32}])[0]["name"] == "buffer.mem"


def test_physical_backing_moves_to_the_inventory():
    facts = {"census": {"unit_root": "Accel"}}
    memories = label_roles([_CENSUS_BUFFER, _CENSUS_BACKING], [_PROBE]) + [_VIEW]
    kept = separate_backings(memories, memories, facts)
    assert [row["name"] for row in kept] == ["buffer", "accumulator"]
    inventory = facts["census"]["physical_memories"]
    assert [(row["name"], row.get("backs")) for row in inventory] == [("buffer", None), ("syncmem.mem", "accumulator")]


def test_unmatched_census_store_is_kept_and_unscoped_census_states_no_inventory():
    odd = {**_CENSUS_BACKING, "depth": 16}
    facts = {"census": {"unit_root": None}}
    assert [row["name"] for row in separate_backings([odd, _VIEW], [odd, _VIEW], facts)] == [
        "syncmem.mem",
        "accumulator",
    ]
    assert "physical_memories" not in facts["census"]
