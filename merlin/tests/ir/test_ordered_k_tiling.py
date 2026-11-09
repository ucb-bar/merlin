"""Exact ordered-reduction partitioning and runtime byte-window checks."""

from __future__ import annotations

from dataclasses import replace

import pytest

from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.ordered_k_tiling import (
    OrderedKTilingError,
    materialize_row_major_k_slices,
    partition_ordered_matmul_k,
)


def _source(*, k_extent: int = 6, policy: str = "ordered-rne") -> KernelRequest:
    maps = (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )
    return KernelRequest(
        (
            SemanticNode("a", "input", (), TensorType((2, k_extent), "i8", "raw-byte-copy"), effect="input"),
            SemanticNode("w", "input", (), TensorType((3, k_extent), "i8", "raw-byte-copy"), effect="input"),
            SemanticNode("result", "matmul", ("a", "w"), TensorType((2, 3), "i32", policy),
                         index_maps=maps),
        ),
        ("result",), ("external",), (("a", "external"), ("w", "external")),
        "synthetic-target", lowering_policy="diagnostic", source_identity="public-source",
    )


def test_exact_k_windows_preserve_order_and_all_source_bytes() -> None:
    tiling = partition_ordered_matmul_k(_source(), tile_k=2, qualified_policy="ordered-rne")
    assert tiling.tile_count == 3
    assert tiling.request.outputs == ("result",)
    assert tiling.request.source_identity == "public-source"
    assert [node.op for node in tiling.request.nodes] == ["input"] * 6 + [
        "matmul", "matmul_accumulate", "matmul_accumulate",
    ]
    assert tiling.request.nodes[-1].inputs[-1] == tiling.request.nodes[-2].id
    assert tiling.record()["tiled_request_digest"] == tiling.request.digest()

    original = {"a": bytes(range(12)), "w": bytes(range(30, 48))}
    tiles = materialize_row_major_k_slices(tiling, original, element_bytes=1)
    assert list(tiles) == ["a__k0", "w__k0", "a__k1", "w__k1", "a__k2", "w__k2"]
    for source, rows in (("a", 2), ("w", 3)):
        rebuilt = b"".join(
            b"".join(tiles[f"{source}__k{tile}"][2 * row:2 * row + 2] for tile in range(3))
            for row in range(rows)
        )
        assert rebuilt == original[source]
    expected = [
        sum(original["a"][6 * row + k] * original["w"][6 * col + k] for k in range(6))
        for row in range(2) for col in range(3)
    ]
    actual = [
        sum(tiles[f"a__k{tile}"][2 * row + k] * tiles[f"w__k{tile}"][2 * col + k]
            for tile in range(3) for k in range(2))
        for row in range(2) for col in range(3)
    ]
    assert actual == expected


def test_head_padded_k_tail_preserves_source_order() -> None:
    source = _source(k_extent=5)
    tiling = partition_ordered_matmul_k(
        source, tile_k=2, qualified_policy="ordered-rne",
        head_pad_element=b"\x00",
    )
    assert tiling.tile_count == 3
    assert tiling.record()["schema"] == "merlin.ordered_k_tiling.v2"
    assert tiling.record()["head_pad_element_hex"] == "00"
    assert [(item.k_start, item.source_count, item.destination_column_start)
            for item in tiling.source_slices[::2]] == [(0, 1, 1), (1, 2, 0), (3, 2, 0)]
    source_bytes = {"a": bytes(range(1, 11)), "w": bytes(range(11, 26))}
    tiles = materialize_row_major_k_slices(tiling, source_bytes, element_bytes=1)
    assert tiles["a__k0"] == b"\x00\x01\x00\x06"
    assert tiles["a__k1"] == b"\x02\x03\x07\x08"
    assert tiles["a__k2"] == b"\x04\x05\x09\x0a"
    for row in range(2):
        for col in range(3):
            original = sum(source_bytes["a"][row * 5 + k]
                           * source_bytes["w"][col * 5 + k] for k in range(5))
            padded = sum(
                tiles[f"a__k{tile}"][row * 2 + k]
                * tiles[f"w__k{tile}"][col * 2 + k]
                for tile in range(3) for k in range(2)
            )
            assert padded == original


def test_one_short_k_tile_keeps_one_source_result() -> None:
    tiling = partition_ordered_matmul_k(
        _source(k_extent=1), tile_k=2, qualified_policy="ordered-rne",
        head_pad_element=b"\x00",
    )
    assert tiling.tile_count == 1
    assert [node.op for node in tiling.request.nodes[-1:]] == ["matmul"]
    assert materialize_row_major_k_slices(
        tiling, {"a": b"\x03\x04", "w": b"\x05\x06\x07"}, element_bytes=1,
    ) == {"a__k0": b"\x00\x03\x00\x04",
          "w__k0": b"\x00\x05\x00\x06\x00\x07"}


def test_ordered_k_refuses_unqualified_tail_wrong_policy_and_changed_indexing() -> None:
    source = _source()
    with pytest.raises(OrderedKTilingError, match="qualified head pad"):
        partition_ordered_matmul_k(source, tile_k=4, qualified_policy="ordered-rne")
    with pytest.raises(OrderedKTilingError, match="policy differs"):
        partition_ordered_matmul_k(source, tile_k=2, qualified_policy="different-policy")
    changed = replace(source.nodes[-1], index_maps=source.nodes[-1].index_maps[::-1])
    with pytest.raises(OrderedKTilingError, match="maps"):
        partition_ordered_matmul_k(replace(source, nodes=(*source.nodes[:-1], changed)),
                                   tile_k=2, qualified_policy="ordered-rne")


def test_runtime_slice_refuses_missing_or_misaligned_source_bytes() -> None:
    tiling = partition_ordered_matmul_k(_source(), tile_k=2, qualified_policy="ordered-rne")
    changed = replace(tiling.source_slices[2], k_start=3)
    with pytest.raises(OrderedKTilingError, match="omit, overlap, or reorder"):
        replace(tiling, source_slices=(*tiling.source_slices[:2], changed, *tiling.source_slices[3:]))
    with pytest.raises(OrderedKTilingError, match="source inputs"):
        materialize_row_major_k_slices(tiling, {"a": bytes(12)}, element_bytes=1)
    with pytest.raises(OrderedKTilingError, match="source bytes"):
        materialize_row_major_k_slices(tiling, {"a": bytes(11), "w": bytes(18)}, element_bytes=1)
    with pytest.raises(OrderedKTilingError, match="element width"):
        materialize_row_major_k_slices(tiling, {"a": bytes(12), "w": bytes(18)}, element_bytes=0)


def test_two_byte_elements_keep_row_stride_and_element_units() -> None:
    tiling = partition_ordered_matmul_k(_source(), tile_k=2, qualified_policy="ordered-rne")
    original = {
        "a": b"".join(value.to_bytes(2, "little") for value in range(12)),
        "w": b"".join(value.to_bytes(2, "little") for value in range(30, 48)),
    }
    tiles = materialize_row_major_k_slices(tiling, original, element_bytes=2)
    assert tiles["a__k1"] == b"".join(value.to_bytes(2, "little") for value in (2, 3, 8, 9))
    assert tiles["w__k2"] == b"".join(value.to_bytes(2, "little") for value in (34, 35, 40, 41, 46, 47))
