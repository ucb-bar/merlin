from __future__ import annotations

import pytest

from merlin.compiler.semantic.model import IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.compiler.transforms.contraction_mnk import (
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


@pytest.mark.parametrize("m,n,k", [(33, 31, 65)])
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
    assert len(plan.requests) == (m + 31) // 32 * ((n + 31) // 32)
    assert all((request.source_identity == "fresh-source" for request in plan.requests))
    assert all(
        (
            [node.op for node in request.nodes if node.op.startswith("matmul")]
            == ["matmul", *["matmul_accumulate"] * ((k + 31) // 32 - 1)]
            for request in plan.requests
        )
    )
    a = bytes(((row * 7 + index * 3) % 7 for row in range(m) for index in range(k)))
    w = bytes(((row * 5 + index * 2 + 1) % 6 for row in range(n) for index in range(k)))
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
        (
            sum((a[row * k + index] * w[column * k + index] for index in range(k))).to_bytes(4, "little")
            for row in range(m)
            for column in range(n)
        )
    )
    assert result == expected
