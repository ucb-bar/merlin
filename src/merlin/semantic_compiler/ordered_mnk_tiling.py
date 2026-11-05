"""Compose independent output tiles with ordered reduction tiles.

The target supplies the tile geometry, numerical policy, and separately
qualified pad bytes. Each resulting request retains the source's ordered K
sequence; this module only moves declared boundary bytes and output tiles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .model import KernelRequest
from .ordered_k_tiling import (
    OrderedKTiling,
    materialize_row_major_k_slices,
    partition_ordered_matmul_k,
)
from .ordered_mn_tiling import (
    OrderedMNTiling,
    assemble_row_major_mn_outputs,
    materialize_row_major_mn_inputs,
    partition_ordered_matmul_mn,
)


class OrderedMNKTilingError(ValueError):
    """An outer tile plan no longer describes the selected source exactly."""


@dataclass(frozen=True)
class OrderedMNKTiling:
    source: KernelRequest
    mn: OrderedMNTiling
    k_tiles: tuple[OrderedKTiling, ...]
    tile_k: int
    qualified_policy: str
    k_head_pad_element: bytes | None = None

    def __post_init__(self) -> None:
        if self.mn.source != self.source or len(self.k_tiles) != len(self.mn.tiles):
            raise OrderedMNKTilingError("M/N/K tiles do not cover the selected source")
        for output_tile, reduction in zip(self.mn.tiles, self.k_tiles):
            expected = partition_ordered_matmul_k(
                output_tile.request,
                tile_k=self.tile_k,
                qualified_policy=self.qualified_policy,
                head_pad_element=self.k_head_pad_element,
            )
            if reduction != expected:
                raise OrderedMNKTilingError("K tile differs from its output region")

    @property
    def requests(self) -> tuple[KernelRequest, ...]:
        """One separately compilable, ordered-reduction request per output tile."""
        return tuple(reduction.request for reduction in self.k_tiles)

    def record(self) -> dict[str, object]:
        return {
            "schema": "merlin.ordered_mnk_tiling.v1",
            "source_request_digest": self.source.digest(),
            "mn": self.mn.record(),
            "tile_k": self.tile_k,
            "qualified_policy": self.qualified_policy,
            "k_head_pad_element_hex": (None if self.k_head_pad_element is None else self.k_head_pad_element.hex()),
            "k": [reduction.record() for reduction in self.k_tiles],
        }


def partition_ordered_matmul_mnk(
    source: KernelRequest,
    *,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    qualified_policy: str,
    mn_tail_pad_element: bytes | None = None,
    k_head_pad_element: bytes | None = None,
) -> OrderedMNKTiling:
    """Partition one oriented contraction without changing valid-cell K order.

    The first short K tile is head padded, so real source terms still appear
    in increasing K order. The caller must qualify that pad as an exact reset
    identity. M/N padding is a separate caller premise for invalid output
    lanes, and output assembly drops those lanes.
    """
    mn = partition_ordered_matmul_mn(
        source,
        tile_m=tile_m,
        tile_n=tile_n,
        qualified_policy=qualified_policy,
        tail_pad_element=mn_tail_pad_element,
    )
    reductions = tuple(
        partition_ordered_matmul_k(
            output_tile.request,
            tile_k=tile_k,
            qualified_policy=qualified_policy,
            head_pad_element=k_head_pad_element,
        )
        for output_tile in mn.tiles
    )
    return OrderedMNKTiling(source, mn, reductions, tile_k, qualified_policy, k_head_pad_element)


def materialize_row_major_mnk_inputs(
    tiling: OrderedMNKTiling,
    source_bytes: Mapping[str, bytes],
    *,
    element_bytes: int,
) -> tuple[dict[str, bytes], ...]:
    """Pack each output tile's complete, ordered K boundary from source bytes."""
    mn_inputs = materialize_row_major_mn_inputs(
        tiling.mn,
        source_bytes,
        element_bytes=element_bytes,
    )
    return tuple(
        materialize_row_major_k_slices(reduction, inputs, element_bytes=element_bytes)
        for reduction, inputs in zip(tiling.k_tiles, mn_inputs)
    )


def assemble_row_major_mnk_outputs(
    tiling: OrderedMNKTiling,
    tile_outputs: Sequence[bytes],
    *,
    element_bytes: int,
) -> bytes:
    """Trim invalid M/N lanes and join final accumulated output tiles."""
    return assemble_row_major_mn_outputs(
        tiling.mn,
        tile_outputs,
        element_bytes=element_bytes,
    )


@dataclass(frozen=True)
class LoopedMNKTiling:
    """Compact loop geometry and packed buffers for one ordered contraction.

    Unlike the explicit partition above, this plan does not build a request
    for every output tile. The target supplies its geometry, numerical policy
    and qualified pad values. Runtime tensors remain outside compilation.
    """

    source: KernelRequest
    tile_m: int
    tile_n: int
    tile_k: int
    qualified_policy: str
    input_element_bytes: int
    output_element_bytes: int
    mn_tail_pad_element: bytes | None = None
    k_head_pad_element: bytes | None = None

    def __post_init__(self) -> None:
        from .ordered_k_tiling import _validated_oriented_source

        _validated_oriented_source(self.source, self.qualified_policy)
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.tile_m,
                self.tile_n,
                self.tile_k,
                self.input_element_bytes,
                self.output_element_bytes,
            )
        ):
            raise OrderedMNKTilingError("loop geometry and element widths must be positive integers")
        m, n, k = self.extents
        for needed, pad, label in (
            (bool(m % self.tile_m or n % self.tile_n), self.mn_tail_pad_element, "M/N"),
            (bool(k % self.tile_k), self.k_head_pad_element, "K"),
        ):
            if needed != (pad is not None) or (
                pad is not None and (type(pad) is not bytes or len(pad) != self.input_element_bytes)
            ):
                raise OrderedMNKTilingError(f"{label} tails need an explicit qualified pad of the input element width")

    @property
    def extents(self) -> tuple[int, int, int]:
        lhs, rhs, _ = self.source.nodes
        return lhs.type.shape[0], rhs.type.shape[0], lhs.type.shape[1]

    @property
    def counts(self) -> tuple[int, int, int]:
        return tuple(
            (extent + tile - 1) // tile
            for extent, tile in zip(
                self.extents,
                (self.tile_m, self.tile_n, self.tile_k),
                strict=True,
            )
        )

    @property
    def tile_bytes(self) -> tuple[int, int, int]:
        return (
            self.tile_m * self.tile_k * self.input_element_bytes,
            self.tile_n * self.tile_k * self.input_element_bytes,
            self.tile_m * self.tile_n * self.output_element_bytes,
        )

    @property
    def buffer_bytes(self) -> tuple[int, int, int]:
        m, n, k = self.counts
        a, w, out = self.tile_bytes
        return m * k * a, n * k * w, m * n * out

    def record(self) -> dict[str, object]:
        return {
            "schema": "merlin.looped_mnk_tiling.v1",
            "source_request_digest": self.source.digest(),
            "tile_shape": [self.tile_m, self.tile_n, self.tile_k],
            "loop_counts": list(self.counts),
            "loop_order": ["m", "n", "k"],
            "input_element_bytes": self.input_element_bytes,
            "output_element_bytes": self.output_element_bytes,
            "buffer_bytes": list(self.buffer_bytes),
            "input_tile_order": ["m,k", "n,k"],
            "output_tile_order": "m,n",
            "qualified_policy": self.qualified_policy,
            "mn_tail_pad_element_hex": None if self.mn_tail_pad_element is None else self.mn_tail_pad_element.hex(),
            "k_head_pad_element_hex": None if self.k_head_pad_element is None else self.k_head_pad_element.hex(),
        }


def materialize_looped_mnk_inputs(
    tiling: LoopedMNKTiling,
    source_bytes: Mapping[str, bytes],
) -> tuple[bytes, bytes]:
    """Pack each operand once, reusing A across N and W across M iterations.

    A is ordered by (M tile, K tile), W by (N tile, K tile). The short K
    segment is first and padded before its real elements. M/N tail rows
    occupy unobservable lanes and use their separately qualified pad value.
    """
    lhs, rhs, _ = tiling.source.nodes
    if set(source_bytes) != {lhs.id, rhs.id}:
        raise OrderedMNKTilingError("runtime source inputs differ from the looped plan")
    return tuple(materialize_looped_mnk_operand(tiling, node.id, source_bytes[node.id]) for node in (lhs, rhs))


def materialize_looped_mnk_operand(tiling: LoopedMNKTiling, source_id: str, payload: bytes) -> bytes:
    """Pack one logical operand, including an independently declared parameter."""
    lhs, rhs, _ = tiling.source.nodes
    if source_id not in (lhs.id, rhs.id):
        raise OrderedMNKTilingError("source operand is not an input of the looped plan")
    node, rows_per_tile = (lhs, tiling.tile_m) if source_id == lhs.id else (rhs, tiling.tile_n)
    width = tiling.input_element_bytes
    k_extent = tiling.extents[2]
    k_tail = k_extent % tiling.tile_k
    rows = node.type.shape[0]
    if type(payload) is not bytes or len(payload) != rows * k_extent * width:
        raise OrderedMNKTilingError("runtime tensor bytes differ from the looped source")
    packed = bytearray()
    for row_start in range(0, rows, rows_per_tile):
        for k_tile in range(tiling.counts[2]):
            count = k_tail if k_tail and k_tile == 0 else tiling.tile_k
            start = 0 if k_tile == 0 else (k_tail + (k_tile - 1) * tiling.tile_k if k_tail else k_tile * tiling.tile_k)
            pad = (tiling.k_head_pad_element or b"") * (tiling.tile_k - count)
            for row in range(row_start, row_start + rows_per_tile):
                if row >= rows:
                    assert tiling.mn_tail_pad_element is not None
                    packed.extend(tiling.mn_tail_pad_element * tiling.tile_k)
                else:
                    offset = (row * k_extent + start) * width
                    packed.extend(pad)
                    packed.extend(payload[offset : offset + count * width])
    return bytes(packed)


def assemble_looped_mnk_outputs(tiling: LoopedMNKTiling, output: bytes) -> bytes:
    """Trim row-major physical tiles into the source's logical M-by-N output.

    Target-specific lane/pair packing must be removed before calling this.
    """
    if type(output) is not bytes or len(output) != tiling.buffer_bytes[2]:
        raise OrderedMNKTilingError("looped output bytes differ from the declared tile buffer")
    m, n, _ = tiling.extents
    width = tiling.output_element_bytes
    result = bytearray(m * n * width)
    _, n_tiles, _ = tiling.counts
    for row in range(m):
        for n_tile in range(n_tiles):
            count = min(tiling.tile_n, n - n_tile * tiling.tile_n) * width
            tile = (row // tiling.tile_m) * n_tiles + n_tile
            start = tile * tiling.tile_bytes[2] + (row % tiling.tile_m) * tiling.tile_n * width
            dest = (row * n + n_tile * tiling.tile_n) * width
            result[dest : dest + count] = output[start : start + count]
    return bytes(result)
