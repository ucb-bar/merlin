"""Capability-derived property-test synthesis (Phase D1) — the harness derives the generalization
matrix from the target's declared capability closure, the agent never picks the tests.

Given a target's ``semantic_capabilities`` (the independent capability map), this deterministically
enumerates region probes that sample the *closure* of each declared family: shape corners (tiny,
tile-boundary ±1, prime, skinny, batched), each declared dtype, transpose and layout variants. Every
probe is a :class:`merlin.targetgen.eligibility.RegionDescriptor` tagged with the generalization axis it
exercises — so "matmul supported" stops meaning "one 16x32x16 fp16 GEMM passed" and starts meaning "the
declared closure is covered". Deterministic (no RNG): the same contract yields the same probe set.

By construction every probe is drawn from the declared capability, so every probe is *eligible* — a
self-consistency property the tests assert. The compiler-under-test is then scored on how many of these
derived probes it actually lowers (that is the recall the fuzzer and the grader measure).
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.targetgen import semantic_families as _sf
from merlin.targetgen.compute_units import SemanticCapability
from merlin.targetgen.eligibility import RegionDescriptor

# Fallback tile edge for a target whose geometry cannot be derived. NOT a hardware constant: it is the
# same software-tiling default `corpus_spec._tile_dim` falls back to for a target with no fixed mesh.
_FALLBACK_TILE = 16


def tile_edge(target: str | None = None) -> int:
    """The target's tile edge, DERIVED — the number every shape corner below is measured against.

    This used to be the literal ``16``, defended by a comment claiming the corners were "structural,
    not a specific mesh dim". They are not: a corner only means something RELATIVE to the tile it
    brackets. On a device whose mesh edge is 32, every corner built from 16 lands strictly INSIDE one
    tile -- ``tile_plus_1`` is 17, still tile 0 -- so the probe set could not cross a tile boundary in
    any dimension and could not, even in principle, detect a backend that handles one tile and no more.
    Measured: a submission that lowered exactly one 32x32 tile passed every derived probe and failed the
    first real holdout at 2 M-tiles.

    Derivation is delegated to :func:`corpus_spec._tile_dim`, which already resolves it from the
    manifest's declared mesh/tile geometry, else the CIRCT ``arrays[mesh].rows`` RTL fact -- so this
    honours the derive-never-hardcode rule through the existing path rather than a second one.
    """
    if not target:
        return _FALLBACK_TILE
    try:
        from merlin.targetgen.corpus_spec import _tile_dim
        from merlin.targetgen.target_experiment import load_capability_manifest
        return int(_tile_dim(target, load_capability_manifest(target).contract or {}))
    except Exception:  # noqa: BLE001 — underivable geometry -> the software-tiling default, never a guess
        return _FALLBACK_TILE


def shape_corners(tile: int) -> list[tuple[str, tuple[int, int, int], int]]:
    """Representative shape corners around ``tile``.

    The boundary ±1, a prime, a skinny vector, a batched rank-3 shape -- the cases that break naive
    tile-multiple codegen -- PLUS one corner per axis at TWO tiles.

    The multi-tile corners are the load-bearing addition and they are deliberately PER-AXIS. A backend
    that emits a loop over K and N but not over M generalizes over two of the three and fails the third,
    and a single "big" corner that grows all three at once cannot tell which. Measured on a real
    submission: unseen K-depth passed, unseen N-width passed, unseen M failed -- one axis, and naming it
    is the difference between "add a loop over M" and "the compiler does not generalize".
    """
    return [
        ("tiny", (1, 1, 1), 2),
        ("tile_minus_1", (tile - 1, tile - 1, tile - 1), 2),
        ("tile", (tile, tile, tile), 2),
        ("tile_plus_1", (tile + 1, tile + 1, tile + 1), 2),
        ("m_2tiles", (2 * tile, tile, tile), 2),
        ("k_2tiles", (tile, 2 * tile, tile), 2),
        ("n_2tiles", (tile, tile, 2 * tile), 2),
        ("prime", (17, 19, 23), 2),
        ("skinny_row", (1, 4096, 4096), 2),
        ("skinny_col", (4096, 4096, 1), 2),
        ("batched", (tile, tile, tile), 3),
    ]


@dataclass(frozen=True)
class Probe:
    name: str
    axis: str                 # the generalization axis this probe exercises (shape/dtype/layout)
    descriptor: RegionDescriptor


def _primary_shape(fam: str, tile: int):
    """A nominal in-closure shape for a family: contractions carry M/K/N, unary families a single dim."""
    if fam in ("contraction", "attention"):
        return (tile, tile, tile)
    return (tile, None, None)


def probes_for_family(fam: str, cap: SemanticCapability, *, tile: int | None = None) -> list[Probe]:
    """The derived probe set for one declared family capability.

    ``tile`` is the target's derived tile edge (see :func:`tile_edge`); omitted, the software-tiling
    fallback is used. Every shape corner is measured against it, so passing the WRONG one produces a
    probe set that cannot cross a tile boundary and silently proves nothing about multi-tile shapes.
    """
    tile = _FALLBACK_TILE if tile is None else int(tile)
    probes: list[Probe] = []
    dtypes = list(cap.dtypes) or [None]
    lead = dtypes[0]
    contractionish = fam in ("contraction", "attention")

    # shape corners on the lead dtype
    for corner, (m, k, n), rank in shape_corners(tile):
        if not contractionish and corner in ("prime", "skinny_col"):
            continue                                  # unary families: a couple of corners suffice
        if not contractionish and corner in ("k_2tiles", "n_2tiles"):
            continue                                  # a unary family has no K/N to tile over
        if rank == 3 and not cap.batch:
            continue                                  # skip batched when the unit declares batch=false
        axis = "shape" if corner != "batched" else "shape"
        d = RegionDescriptor(source=f"{fam}/{corner}", family=fam, in_dtype=lead,
                             weight_dtype=(lead if contractionish else None),
                             m=m, k=(k if contractionish else None),
                             n=(n if contractionish else None), rank=rank, batch=(2 if rank == 3 else 1))
        probes.append(Probe(name=f"{fam}.{corner}", axis=axis, descriptor=d))

    # one probe per additional declared dtype (dtype-generalization axis)
    m, k, n = _primary_shape(fam, tile)
    for dt in dtypes[1:]:
        d = RegionDescriptor(source=f"{fam}/dtype:{dt}", family=fam, in_dtype=dt,
                             weight_dtype=(dt if contractionish else None), m=m, k=k, n=n, rank=2)
        probes.append(Probe(name=f"{fam}.dtype_{dt}", axis="dtype", descriptor=d))

    # transpose + declared layout variants (layout-generalization axis)
    if cap.transpose and contractionish:
        d = RegionDescriptor(source=f"{fam}/transpose", family=fam, in_dtype=lead, weight_dtype=lead,
                             m=m, k=k, n=n, rank=2, layout="transposed")
        probes.append(Probe(name=f"{fam}.transpose", axis="layout", descriptor=d))
    for lay in cap.layouts:
        d = RegionDescriptor(source=f"{fam}/layout:{lay}", family=fam, in_dtype=lead,
                             weight_dtype=(lead if contractionish else None), m=m, k=k, n=n,
                             rank=2, layout=lay)
        probes.append(Probe(name=f"{fam}.layout_{lay}", axis="layout", descriptor=d))
    return probes


def synthesize(cap_map: dict[str, SemanticCapability], *, target: str | None = None) -> list[Probe]:
    """All derived probes for a target's declared capability map (families in canonical order).

    Pass ``target`` so the shape corners are measured against THAT device's derived tile edge. Without
    it the corners fall back to the software-tiling default, which on a wider mesh means every corner
    sits inside a single tile.
    """
    tile = tile_edge(target)
    out: list[Probe] = []
    order = [f for f in (*_sf.PRIMITIVES, *sorted(_sf.COMPOSITES)) if f in cap_map]
    for fam in order:
        out += probes_for_family(fam, cap_map[fam], tile=tile)
    return out
