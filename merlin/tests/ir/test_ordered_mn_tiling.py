"""Exact outer matrix tiling and row-major boundary placement."""

from __future__ import annotations

from dataclasses import replace

import pytest

from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.ordered_k_tiling import partition_ordered_matmul_k
from merlin.semantic_compiler.ordered_mn_tiling import (
    OrderedMNTilingError,
    assemble_row_major_mn_outputs,
    materialize_row_major_mn_inputs,
    partition_ordered_matmul_mn,
)


def _source(*, m: int = 4, n: int = 6, k: int = 4) -> KernelRequest:
    maps = (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )
    return KernelRequest(
        (
            SemanticNode("a", "input", (), TensorType((m, k), "i8", "raw-byte-copy"), effect="input"),
            SemanticNode("w", "input", (), TensorType((n, k), "i8", "raw-byte-copy"), effect="input"),
            SemanticNode("c", "matmul", ("a", "w"), TensorType((m, n), "i32", "ordered-rne"),
                         index_maps=maps),
        ),
        ("c",), ("external",), (("a", "external"), ("w", "external")),
        "synthetic-target", lowering_policy="diagnostic", source_identity="public-matrix",
    )


def test_mn_tiles_and_k_tiles_preserve_each_ordered_output() -> None:
    source = _source()
    plan = partition_ordered_matmul_mn(source, tile_m=2, tile_n=3,
                                       qualified_policy="ordered-rne")
    assert len(plan.tiles) == 4
    assert [(tile.m_start, tile.n_start) for tile in plan.tiles] == [
        (0, 0), (0, 3), (2, 0), (2, 3),
    ]
    assert plan.record()["source_request_digest"] == source.digest()
    assert all(tile.request.source_identity == source.source_identity for tile in plan.tiles)
    assert all(tile.request.nodes[-1].type.shape == (2, 3) for tile in plan.tiles)
    assert all(partition_ordered_matmul_k(
        tile.request, tile_k=2, qualified_policy="ordered-rne",
    ).tile_count == 2 for tile in plan.tiles)

    inputs = {"a": bytes(range(1, 17)), "w": bytes(range(17, 41))}
    packed = materialize_row_major_mn_inputs(plan, inputs, element_bytes=1)
    assert packed[1]["a"] == packed[0]["a"]
    assert packed[2]["w"] == packed[0]["w"]
    tile_outputs = []
    for tile_inputs in packed:
        tile_outputs.append(b"".join(
            sum(tile_inputs["a"][row * 4 + k] * tile_inputs["w"][col * 4 + k]
                for k in range(4)).to_bytes(2, "little")
            for row in range(2) for col in range(3)
        ))
    assembled = assemble_row_major_mn_outputs(plan, tile_outputs, element_bytes=2)
    expected = b"".join(
        sum(inputs["a"][row * 4 + k] * inputs["w"][col * 4 + k]
            for k in range(4)).to_bytes(2, "little")
        for row in range(4) for col in range(6)
    )
    assert assembled == expected


def test_mn_slice_uses_byte_width_and_preserves_row_stride() -> None:
    plan = partition_ordered_matmul_mn(
        _source(m=4, n=6, k=2), tile_m=2, tile_n=3, qualified_policy="ordered-rne",
    )
    inputs = {
        "a": b"".join(value.to_bytes(2, "little") for value in range(8)),
        "w": b"".join(value.to_bytes(2, "little") for value in range(12)),
    }
    packed = materialize_row_major_mn_inputs(plan, inputs, element_bytes=2)
    assert packed[2]["a"] == b"".join(value.to_bytes(2, "little") for value in range(4, 8))
    assert packed[1]["w"] == b"".join(value.to_bytes(2, "little") for value in range(6, 12))


def test_mn_tail_padding_preserves_valid_output_cells() -> None:
    source = _source(m=5, n=4, k=4)
    plan = partition_ordered_matmul_mn(
        source, tile_m=2, tile_n=3, qualified_policy="ordered-rne",
        tail_pad_element=b"\x00",
    )
    assert [(tile.m_start, tile.n_start) for tile in plan.tiles] == [
        (0, 0), (0, 3), (2, 0), (2, 3), (4, 0), (4, 3),
    ]
    assert plan.record()["schema"] == "merlin.ordered_mn_tiling.v2"
    assert plan.record()["tail_pad_element_hex"] == "00"
    assert plan.record()["tiles"][-1]["valid_m"] == 1
    assert plan.record()["tiles"][-1]["valid_n"] == 1

    inputs = {"a": bytes(range(1, 21)), "w": bytes(range(21, 37))}
    packed = materialize_row_major_mn_inputs(plan, inputs, element_bytes=1)
    assert packed[-1]["a"][-4:] == bytes(4)
    assert packed[-1]["w"][-8:] == bytes(8)
    tiles = [b"".join(
        sum(boundary["a"][row * 4 + k] * boundary["w"][col * 4 + k]
            for k in range(4)).to_bytes(2, "little")
        for row in range(2) for col in range(3)
    ) for boundary in packed]
    assert assemble_row_major_mn_outputs(plan, tiles, element_bytes=2) == b"".join(
        sum(inputs["a"][row * 4 + k] * inputs["w"][col * 4 + k]
            for k in range(4)).to_bytes(2, "little")
        for row in range(5) for col in range(4)
    )


def test_mn_single_partial_tile_keeps_one_result() -> None:
    plan = partition_ordered_matmul_mn(
        _source(m=1, n=1, k=2), tile_m=2, tile_n=3,
        qualified_policy="ordered-rne", tail_pad_element=b"\x00",
    )
    assert len(plan.tiles) == 1
    packed = materialize_row_major_mn_inputs(
        plan, {"a": b"\x02\x03", "w": b"\x05\x07"}, element_bytes=1,
    )
    assert packed == ({"a": b"\x02\x03\x00\x00",
                       "w": b"\x05\x07\x00\x00\x00\x00"},)
    assert assemble_row_major_mn_outputs(
        plan, ((31).to_bytes(2, "little") + bytes(10),), element_bytes=2,
    ) == (31).to_bytes(2, "little")


def test_mn_rejects_unqualified_tails_policy_and_reordered_maps() -> None:
    with pytest.raises(OrderedMNTilingError, match="qualified pad"):
        partition_ordered_matmul_mn(_source(m=5), tile_m=2, tile_n=3,
                                    qualified_policy="ordered-rne")
    with pytest.raises(OrderedMNTilingError, match="element width"):
        plan = partition_ordered_matmul_mn(
            _source(m=5), tile_m=2, tile_n=3,
            qualified_policy="ordered-rne", tail_pad_element=b"\x00\x00",
        )
        materialize_row_major_mn_inputs(
            plan, {"a": bytes(20), "w": bytes(24)}, element_bytes=1,
        )
    with pytest.raises(OrderedMNTilingError, match="policy differs"):
        partition_ordered_matmul_mn(_source(), tile_m=2, tile_n=3,
                                    qualified_policy="different-policy")
    source = _source()
    changed = replace(source.nodes[-1], index_maps=source.nodes[-1].index_maps[::-1])
    with pytest.raises(OrderedMNTilingError, match="maps"):
        partition_ordered_matmul_mn(replace(source, nodes=(*source.nodes[:-1], changed)),
                                    tile_m=2, tile_n=3, qualified_policy="ordered-rne")


def test_mn_record_and_runtime_reject_missing_or_changed_regions() -> None:
    plan = partition_ordered_matmul_mn(_source(), tile_m=2, tile_n=3,
                                       qualified_policy="ordered-rne")
    changed = replace(plan.tiles[1], n_start=0)
    with pytest.raises(OrderedMNTilingError, match="source region"):
        replace(plan, tiles=(plan.tiles[0], changed, *plan.tiles[2:]))
    with pytest.raises(OrderedMNTilingError, match="source inputs"):
        materialize_row_major_mn_inputs(plan, {"a": bytes(16)}, element_bytes=1)
    with pytest.raises(OrderedMNTilingError, match="source bytes"):
        materialize_row_major_mn_inputs(plan, {"a": bytes(15), "w": bytes(24)},
                                        element_bytes=1)
    with pytest.raises(OrderedMNTilingError, match="output tile bytes"):
        assemble_row_major_mn_outputs(plan, [bytes(12)] * 3, element_bytes=2)
    with pytest.raises(OrderedMNTilingError, match="output tile bytes"):
        assemble_row_major_mn_outputs(plan, [bytes(12)] * 3 + [bytes(11)],
                                      element_bytes=2)
