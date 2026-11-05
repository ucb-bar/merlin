"""M/N output partitioning composed with ordered K continuation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.ordered_k_tiling import OrderedKTilingError
from merlin.semantic_compiler.ordered_mn_tiling import OrderedMNTilingError
from merlin.semantic_compiler.ordered_mnk_tiling import (
    OrderedMNKTilingError,
    assemble_row_major_mnk_outputs,
    materialize_row_major_mnk_inputs,
    partition_ordered_matmul_mnk,
)


def _source(m: int, n: int, k: int) -> KernelRequest:
    maps = (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )
    return KernelRequest(
        (
            SemanticNode("a", "input", (), TensorType((m, k), "i8", "raw-byte"), effect="input"),
            SemanticNode("w", "input", (), TensorType((n, k), "i8", "raw-byte"), effect="input"),
            SemanticNode("c", "matmul", ("a", "w"), TensorType((m, n), "i32", "ordered-i32"), index_maps=maps),
        ),
        ("c",),
        ("external",),
        (("a", "external"), ("w", "external")),
        "synthetic-mnk",
        lowering_policy="diagnostic",
        source_identity="fresh-source",
    )


@pytest.mark.parametrize("m,n,k", [(32, 32, 32), (64, 64, 64), (33, 31, 65), (1, 65, 7)])
def test_composed_tiles_preserve_all_valid_outputs_and_k_order(m: int, n: int, k: int) -> None:
    source = _source(m, n, k)
    plan = partition_ordered_matmul_mnk(
        source,
        tile_m=32,
        tile_n=32,
        tile_k=32,
        qualified_policy="ordered-i32",
        mn_tail_pad_element=b"\x00" if m % 32 or n % 32 else None,
        k_head_pad_element=b"\x00" if k % 32 else None,
    )
    assert plan.record()["source_request_digest"] == source.digest()
    assert plan.record()["schema"] == "merlin.ordered_mnk_tiling.v1"
    assert len(plan.requests) == ((m + 31) // 32) * ((n + 31) // 32)
    assert all(request.source_identity == "fresh-source" for request in plan.requests)
    assert all(
        [node.op for node in request.nodes if node.op.startswith("matmul")]
        == ["matmul", *["matmul_accumulate"] * ((k + 31) // 32 - 1)]
        for request in plan.requests
    )

    a = bytes((row * 7 + index * 3) % 7 for row in range(m) for index in range(k))
    w = bytes((row * 5 + index * 2 + 1) % 6 for row in range(n) for index in range(k))
    packed = materialize_row_major_mnk_inputs(plan, {"a": a, "w": w}, element_bytes=1)
    outputs = []
    for reduction, inputs in zip(plan.k_tiles, packed):
        tile = bytearray()
        for row in range(32):
            for column in range(32):
                acc = 0
                for index in range(reduction.tile_count):
                    lhs = inputs[f"a__k{index}"]
                    rhs = inputs[f"w__k{index}"]
                    for lane in range(32):
                        acc += lhs[row * 32 + lane] * rhs[column * 32 + lane]
                tile.extend(acc.to_bytes(4, "little"))
        outputs.append(bytes(tile))
    result = assemble_row_major_mnk_outputs(plan, outputs, element_bytes=4)
    expected = b"".join(
        sum(a[row * k + index] * w[column * k + index] for index in range(k)).to_bytes(4, "little")
        for row in range(m)
        for column in range(n)
    )
    assert result == expected


def test_composed_plan_refuses_unqualified_padding_and_changed_k_plan() -> None:
    source = _source(33, 31, 65)
    with pytest.raises(OrderedMNTilingError, match="qualified pad"):
        partition_ordered_matmul_mnk(
            source,
            tile_m=32,
            tile_n=32,
            tile_k=32,
            qualified_policy="ordered-i32",
            k_head_pad_element=b"\x00",
        )
    with pytest.raises(OrderedKTilingError, match="qualified head pad"):
        partition_ordered_matmul_mnk(
            source,
            tile_m=32,
            tile_n=32,
            tile_k=32,
            qualified_policy="ordered-i32",
            mn_tail_pad_element=b"\x00",
        )
    plan = partition_ordered_matmul_mnk(
        source,
        tile_m=32,
        tile_n=32,
        tile_k=32,
        qualified_policy="ordered-i32",
        mn_tail_pad_element=b"\x00",
        k_head_pad_element=b"\x00",
    )
    with pytest.raises(OrderedMNKTilingError, match="K tile differs"):
        replace(plan, k_tiles=(replace(plan.k_tiles[0], head_pad_element=b"\x01"), *plan.k_tiles[1:]))
    with pytest.raises(OrderedMNTilingError, match="runtime source inputs"):
        materialize_row_major_mnk_inputs(plan, {"a": bytes(33 * 65)}, element_bytes=1)
    with pytest.raises(OrderedMNTilingError, match="output tile bytes"):
        assemble_row_major_mnk_outputs(plan, [bytes(32 * 32 * 4)], element_bytes=4)


@pytest.mark.parametrize(
    "shape,tiles",
    [((7, 11, 9), (3, 5, 4)), ((33, 31, 65), (32, 32, 32)), ((1, 65, 7), (32, 32, 32)), ((64, 64, 64), (32, 32, 32))],
)
def test_compact_loop_buffers_match_explicit_partition(shape, tiles):
    from merlin.semantic_compiler.ordered_mnk_tiling import (
        LoopedMNKTiling,
        assemble_looped_mnk_outputs,
        materialize_looped_mnk_inputs,
    )

    m, n, k = shape
    tm, tn, tk = tiles
    source = _source(m, n, k)
    pad_mn = b"\x00" if m % tm or n % tn else None
    pad_k = b"\x00" if k % tk else None
    compact = LoopedMNKTiling(source, tm, tn, tk, "ordered-i32", 1, 4, pad_mn, pad_k)
    explicit = partition_ordered_matmul_mnk(
        source,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        qualified_policy="ordered-i32",
        mn_tail_pad_element=pad_mn,
        k_head_pad_element=pad_k,
    )
    inputs = {"a": bytes((i * 7 + 3) % 251 for i in range(m * k)), "w": bytes((i * 13 + 2) % 251 for i in range(n * k))}
    a, w = materialize_looped_mnk_inputs(compact, inputs)
    expanded = materialize_row_major_mnk_inputs(explicit, inputs, element_bytes=1)
    mc, nc, kc = compact.counts
    ab, wb, ob = compact.tile_bytes
    assert (len(a), len(w), compact.buffer_bytes[2]) == compact.buffer_bytes
    assert compact.record()["source_request_digest"] == source.digest()
    for mi in range(mc):
        for ni in range(nc):
            for ki in range(kc):
                assert a[(mi * kc + ki) * ab : (mi * kc + ki + 1) * ab] == expanded[mi * nc + ni][f"a__k{ki}"]
                assert w[(ni * kc + ki) * wb : (ni * kc + ki + 1) * wb] == expanded[mi * nc + ni][f"w__k{ki}"]
    tile_outputs = [bytes((i * 5 + j) % 256 for j in range(ob)) for i in range(mc * nc)]
    assert assemble_looped_mnk_outputs(compact, b"".join(tile_outputs)) == assemble_row_major_mnk_outputs(
        explicit,
        tile_outputs,
        element_bytes=4,
    )
    with pytest.raises(OrderedMNKTilingError):
        materialize_looped_mnk_inputs(compact, {"a": inputs["a"]})
    with pytest.raises(OrderedMNKTilingError):
        assemble_looped_mnk_outputs(compact, b"")
