"""Source shape factors define one bit-preserving physical storage bijection."""

from dataclasses import replace
from itertools import permutations
from math import prod

import pytest

from merlin.compile.scheduling.tensor_storage import FactoredTensorStorage, column_panel_storage
from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime.storage_binding import StorageBinding, StoragePrepackRequired, resolve_storage_bindings


@pytest.mark.parametrize("shape,panel", [((7, 12), 3), ((17, 8), 4), ((1, 1), 1), ((5, 6), 1), ((3, 6), 6)])
@pytest.mark.parametrize("dtype,width", [("i8", 1), ("i32", 4), ("f32", 4)])
def test_exact_all_byte_words_and_inverse_map(shape, panel, dtype, width):
    k, n = shape
    storage = column_panel_storage(k, n, panel, dtype)
    logical = bytes((i * 37 + 193) % 256 for i in range(k * n * width))
    packed = storage.pack_bytes(logical, max_storage_bytes=len(logical))
    assert storage.unpack_bytes(packed, max_storage_bytes=len(logical)) == logical
    assert len({storage.element_offset((r, c)) for r in range(k) for c in range(n)}) == k * n
    for r in range(k):
        for c in range(n):
            address = (c // panel) * k * panel + r * panel + c % panel
            assert storage.element_offset((r, c)) == address
            assert packed[address * width:(address + 1) * width] == logical[(r * n + c) * width:(r * n + c + 1) * width]
    assert GroupedAxesStorage.from_dict(storage.to_dict()) == storage
    binding = StorageBinding(storage, "read", None)
    words = list(range(k * n))
    physical = binding.pack_words(words)
    assert all(physical[storage.element_offset((r, c))] == words[r * n + c] for r in range(k) for c in range(n))
    assert binding.requires_prepack == storage.reorders_logical_elements


@pytest.mark.parametrize("mutation", ["factor_product", "duplicate_axis", "missing_axis", "bool_factor",
    "unknown_dtype", "overflow"])
def test_malformed_storage_refuses(mutation):
    value = column_panel_storage(7, 12, 3, "i8")
    if mutation == "factor_product": value = replace(value, axis_factors=((7,), (2, 3)))
    elif mutation == "duplicate_axis": value = replace(value, physical_axis_order=(1, 1, 2))
    elif mutation == "missing_axis": value = replace(value, physical_axis_order=(1, 0))
    elif mutation == "bool_factor": value = replace(value, axis_factors=((7,), (4, True)))
    elif mutation == "unknown_dtype": value = replace(value, dtype="unknown")
    else: value = FactoredTensorStorage((1 << 62,), "i32", ((1 << 62,),), (0,))
    with pytest.raises(ValueError): value.validate()


def test_indivisible_source_keeps_original_route_and_copy_is_bounded():
    assert column_panel_storage(17, 11, 4, "i8") is None
    layout = column_panel_storage(7, 12, 3, "i8")
    for payload, budget in ((bytes(83), 84), (bytes(84), 83), (bytes(85), 85)):
        with pytest.raises(ValueError): layout.pack_bytes(payload, max_storage_bytes=budget)
    with pytest.raises(ValueError): layout.element_offset((7, 0))
    with pytest.raises(ValueError): layout.element_offset((0, 12))


def test_factored_input_still_requires_independent_host_prepack_authority():
    layout = column_panel_storage(7, 12, 3, "i8")
    out = GroupedAxesStorage((1,), "i8", ((0,),), (1,), (1,), 1)
    cb = dict(tensors=dict(W=dict(shape=list(layout.physical_shape), dtype="i8", role="weight"),
                           Y=dict(shape=[1], dtype="i8", role="output")),
        kernel_abi=dict(kind="whole_program", args=[dict(tensor="W", access="read"),
            dict(tensor="Y", access="write")], outputs=["Y"]),
        params=dict(storage_encodings=dict(W=layout.to_dict(), Y=out.to_dict())))
    with pytest.raises(StoragePrepackRequired):
        resolve_storage_bindings(cb, {"W": [[0] * 12 for _ in range(7)]}, max_storage_bytes=85)


@pytest.mark.parametrize("dtype,width", [("i8", 1), ("f32", 4), ("f64", 8)])
def test_factored_copies_match_tensor_transpose_for_all_axis_orders(dtype, width):
    import numpy as np

    factors = ((2, 1, 3), (2, 5))
    shape = tuple(v for group in factors for v in group)
    logical = bytes((i * 37 + 193) % 256 for i in range(prod(shape) * width))
    # Opaque words preserve arbitrary floating bit patterns, including NaNs.
    words = np.frombuffer(logical, dtype=f"V{width}").reshape(shape)
    for order in permutations(range(len(shape))):
        storage = FactoredTensorStorage((6, 10), dtype, factors, order)
        expected = words.transpose(order).tobytes()
        assert storage.pack_bytes(logical, max_storage_bytes=len(logical)) == expected
        assert storage.unpack_bytes(expected, max_storage_bytes=len(logical)) == logical
