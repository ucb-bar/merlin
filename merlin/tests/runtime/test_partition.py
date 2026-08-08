"""Carving one shared DRAM into per-image windows.

The failure this prevents is not a crash. On a chip where every core shares one DDR with no separate
banks, two independently-linked images default to the same weights and arena addresses and overwrite
each other — one model reads the other's activations and produces plausible WRONG numbers. So every
way a map could describe overlapping or out-of-region windows has to be refused, not repaired.
"""
from __future__ import annotations

import json

import pytest

from merlin.runtime.partition import (Partition, PartitionError, equal_partitions,
                                      load_partition_map, parse_size)

BASE, SIZE = 0x80000000, 512 * 1024 * 1024


def _write(tmp_path, doc) -> str:
    p = tmp_path / "map.json"
    p.write_text(json.dumps(doc))
    return str(p)


@pytest.mark.parametrize("text,want", [
    ("256M", 256 * 1024 ** 2),
    ("0x10000000", 0x10000000),
    ("512k", 512 * 1024),
    ("1G", 1024 ** 3),
    ("268435456", 268435456),
    (1024, 1024),
])
def test_sizes_are_read_the_way_a_human_writes_them(text, want):
    assert parse_size(text) == want


def test_suffixes_are_powers_of_two():
    """A "256M" that meant 256e6 would be 6% smaller than the linker script's 256 MiB, and the
    mismatch would land as an overlap at the far end of the region."""
    assert parse_size("256M") == 256 * 1024 * 1024 != 256_000_000


@pytest.mark.parametrize("bad", ["", "  ", "12X", "0xzz", "-8M", None, True, 3.5])
def test_an_unreadable_size_raises_rather_than_defaulting(bad):
    with pytest.raises(PartitionError):
        parse_size(bad)


def test_equal_split_produces_non_overlapping_aligned_windows():
    pm = equal_partitions(3, dram_base=BASE, dram_bytes=SIZE)
    assert len(pm.partitions) == 3
    for a, b in zip(pm.partitions, pm.partitions[1:]):
        assert a.end <= b.base, "windows must not overlap"
    for p in pm.partitions:
        assert p.base % (1 << 20) == 0
        assert p.base >= BASE and p.end <= BASE + SIZE
    # Equal-sized on purpose: a per-core image is then interchangeable between windows, and the
    # remainder is left unallocated rather than making one window bigger.
    assert len({p.size for p in pm.partitions}) == 1


def test_equal_split_refuses_more_windows_than_the_region_can_hold():
    with pytest.raises(PartitionError):
        equal_partitions(10_000, dram_base=BASE, dram_bytes=SIZE)


def test_a_partition_is_just_two_build_arguments():
    """The placing is done by the existing layout code, which already fails closed when a model does
    not fit; a partition only says WHERE. Keeping it to those two parameters is what makes this not a
    new code path."""
    pm = equal_partitions(2, dram_base=BASE, dram_bytes=SIZE)
    kw = pm.partitions[1].as_build_kwargs()
    assert set(kw) == {"dram_base", "dram_bytes"}
    assert kw["dram_base"] == pm.partitions[1].base
    assert kw["dram_bytes"] == pm.partitions[1].size


def test_sizes_pack_in_order_so_a_map_cannot_overlap_by_accident(tmp_path):
    path = _write(tmp_path, {"dram_base": hex(BASE), "dram_bytes": "512M", "partitions": [
        {"name": "big", "bytes": "256M"},
        {"name": "small", "bytes": "128M"},
        {"name": "spare", "bytes": "128M", "reserved": True},
    ]})
    pm = load_partition_map(path)
    assert [p.name for p in pm.partitions] == ["big", "small", "spare"]
    assert pm.get("big").base == BASE
    assert pm.get("small").base == BASE + 256 * 1024 ** 2
    assert [p.name for p in pm.usable()] == ["big", "small"]


def test_a_reserved_window_refuses_to_be_built_into(tmp_path):
    path = _write(tmp_path, {"dram_bytes": "512M", "partitions": [
        {"name": "mine", "bytes": "256M"},
        {"name": "theirs", "bytes": "256M", "reserved": True},
    ]})
    pm = load_partition_map(path)
    with pytest.raises(PartitionError, match="reserved"):
        pm.get("theirs").as_build_kwargs()


def test_explicit_bases_that_overlap_are_refused(tmp_path):
    """The whole point. An explicit base is CHECKED against its neighbours, never trusted."""
    path = _write(tmp_path, {"dram_bytes": "512M", "partitions": [
        {"name": "a", "bytes": "256M", "base": hex(BASE)},
        {"name": "b", "bytes": "256M", "base": hex(BASE + 128 * 1024 ** 2)},
    ]})
    with pytest.raises(PartitionError, match="OVERLAP"):
        load_partition_map(path)


def test_a_window_outside_the_region_is_refused(tmp_path):
    path = _write(tmp_path, {"dram_bytes": "128M", "partitions": [{"name": "a", "bytes": "256M"}]})
    with pytest.raises(PartitionError, match="outside the region"):
        load_partition_map(path)


def test_a_mistyped_key_is_refused_rather_than_silently_defaulted(tmp_path):
    """A typo'd size key would otherwise leave the partition at a default size — which is exactly the
    overlap this module exists to prevent, arriving via a spelling mistake."""
    path = _write(tmp_path, {"dram_bytes": "512M", "partitions": [{"name": "a", "byts": "256M"}]})
    with pytest.raises(PartitionError, match="unknown key"):
        load_partition_map(path)


def test_duplicate_names_are_refused(tmp_path):
    path = _write(tmp_path, {"dram_bytes": "512M", "partitions": [
        {"name": "a", "bytes": "128M"}, {"name": "a", "bytes": "128M"}]})
    with pytest.raises(PartitionError, match="duplicate"):
        load_partition_map(path)


@pytest.mark.parametrize("doc", [
    {"dram_bytes": "512M"},                                    # no partitions
    {"dram_bytes": "512M", "partitions": []},                  # empty
    {"partitions": [{"name": "a", "bytes": "1M"}]},            # no region size
    {"dram_bytes": "512M", "partitions": [{"name": "a"}]},     # no size
])
def test_an_incomplete_map_raises(tmp_path, doc):
    with pytest.raises(PartitionError):
        load_partition_map(_write(tmp_path, doc))


def test_malformed_json_and_missing_files_raise(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(PartitionError):
        load_partition_map(bad)
    with pytest.raises(PartitionError):
        load_partition_map(tmp_path / "nope.json")


def test_the_report_states_that_windows_are_not_isolation():
    """A partition map that reads as a safety guarantee is worse than none: nothing in the hardware
    stops one image writing another's window, and enforcement would need PMP."""
    pm = equal_partitions(2, dram_base=BASE, dram_bytes=SIZE)
    note = pm.to_dict()["note"]
    assert "not isolation" in note and "PMP" in note


def test_unallocated_space_is_reported_not_hidden():
    pm = equal_partitions(3, dram_base=BASE, dram_bytes=SIZE)
    d = pm.to_dict()
    assert d["unallocated_bytes"] == SIZE - sum(p.size for p in pm.partitions)
    assert d["unallocated_bytes"] >= 0


def test_an_unaligned_explicit_base_is_rounded_up_not_accepted_raw(tmp_path):
    """An unaligned window breaks the image's own alignment assumptions, so a base is aligned UP
    (never down, which would push it into the previous window)."""
    path = _write(tmp_path, {"dram_bytes": "512M", "partitions": [
        {"name": "a", "bytes": "1M", "base": hex(BASE + 1)}]})
    pm = load_partition_map(path)
    assert pm.get("a").base == BASE + (1 << 20)


def test_partition_end_is_base_plus_size():
    p = Partition(name="x", base=BASE, size=1 << 20)
    assert p.end == BASE + (1 << 20)
