"""Physical encodings are explicit address contracts, never volume-only waivers."""
from dataclasses import replace
from itertools import permutations, product

import pytest

from merlin.perf.storage_encoding import GroupedAxesStorage


def _layout():
    return GroupedAxesStorage((2, 3, 2, 2), "i8", ((0,), (1, 2, 3)), (2, 12), (16, 1), 32)


def test_flattened_padded_rows_use_original_axis_order():
    layout = _layout()
    assert layout.logical_strides_elements == (16, 4, 2, 1)
    data = bytes(range(24))
    packed = layout.pack_bytes(data, max_storage_bytes=32)
    assert packed == data[:12] + bytes(4) + data[12:] + bytes(4)
    assert layout.unpack_bytes(packed, max_storage_bytes=32) == data
    assert GroupedAxesStorage.from_dict(layout.to_dict()) == layout


def test_axis_permutation_is_not_silently_sorted_or_assumed():
    first = _layout()
    second = replace(first, axis_groups=((0,), (2, 3, 1)))
    assert second.logical_strides_elements == (16, 1, 6, 3)
    data = bytes(range(24))
    assert first.pack_bytes(data, max_storage_bytes=32) != second.pack_bytes(data, max_storage_bytes=32)
    assert second.unpack_bytes(second.pack_bytes(data, max_storage_bytes=32),
                               max_storage_bytes=32) == data


def test_transpose_mapping_matches_independent_nested_loop():
    layout = GroupedAxesStorage((3, 2), "i16", ((1,), (0,)), (2, 3), (5, 1), 10)
    data = bytes(range(12))
    expected = bytearray(20)
    for n in range(3):
        for k in range(2):
            expected[(k * 5 + n) * 2:(k * 5 + n + 1) * 2] = data[(n * 2 + k) * 2:(n * 2 + k + 1) * 2]
    assert layout.pack_bytes(data, max_storage_bytes=20) == expected


def test_scalar_storage_view_preserves_all_float_bits():
    layout = GroupedAxesStorage((), "f32", ((),), (1,), (1,), 1)
    for data in (bytes.fromhex("00000080"), bytes.fromhex("0100c07f"), bytes.fromhex("ffffffff")):
        assert layout.pack_bytes(data, max_storage_bytes=4) == data
        assert layout.unpack_bytes(data, max_storage_bytes=4) == data
    assert layout.element_offset(()) == 0


def test_all_permutations_have_unique_bounded_addresses():
    shape = (2, 3, 4)
    for order in permutations(range(3)):
        physical = tuple(shape[i] for i in order)
        strides = (physical[1] * physical[2], physical[2], 1)
        layout = GroupedAxesStorage(shape, "i8", tuple((i,) for i in order), physical, strides, 24)
        offsets = {layout.element_offset(indices) for indices in product(*(range(d) for d in shape))}
        assert offsets == set(range(24))


@pytest.mark.parametrize("changes", [
    {"axis_groups": ((0,), (1, 1, 3))}, {"axis_groups": ((0,), (1, 2))},
    {"axis_groups": ((0,), (1, 2, 4))}, {"axis_groups": ((False,), (1, 2, 3))},
    {"physical_shape": (3, 8)}, {"logical_shape": (2, 3, 0, 2)},
    {"strides_elements": (8, 1)}, {"strides_elements": (16, -1)},
    {"strides_elements": (16,)}, {"storage_elements": 27}, {"storage_elements": True},
    {"offset_elements": -1}, {"offset_elements": 5}, {"dtype": "i3"},
])
def test_invalid_encoding_refuses(changes):
    with pytest.raises(ValueError):
        replace(_layout(), **changes).validate()


@pytest.mark.parametrize("change", [
    {"schema": "future"}, {"unknown": 3}, {"axis_groups": "dense"},
    {"logical_shape": [2, True, 2, 2]}, {"physical_shape": None},
])
def test_malformed_contract_refuses(change):
    record = _layout().to_dict()
    record.update(change)
    with pytest.raises(ValueError):
        GroupedAxesStorage.from_dict(record)


def test_reference_copy_enforces_budget_and_exact_payload_before_allocating():
    layout = _layout()
    with pytest.raises(ValueError, match="budget"):
        layout.pack_bytes(bytes(24), max_storage_bytes=31)
    with pytest.raises(ValueError, match="source type"):
        layout.pack_bytes(bytes(32), max_storage_bytes=32)
    with pytest.raises(ValueError, match="allocated storage"):
        layout.unpack_bytes(bytes(24), max_storage_bytes=32)
    with pytest.raises(ValueError, match="outside"):
        layout.element_offset((0, 3, 0, 0))
