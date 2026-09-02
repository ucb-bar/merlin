"""Target-independent static contraction tiling synthesized for a square mesh."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    m0: int
    n0: int
    k0: int
    rows: int
    cols: int
    depth: int


def tile_matmul(m: int, n: int, k: int, dim: int = 16) -> list[Tile]:
    if min(m, n, k, dim) <= 0:
        raise ValueError("tile extents must be positive")
    return [
        Tile(m0, n0, k0, min(dim, m - m0), min(dim, n - n0), min(dim, k - k0))
        for m0 in range(0, m, dim)
        for n0 in range(0, n, dim)
        for k0 in range(0, k, dim)
    ]

