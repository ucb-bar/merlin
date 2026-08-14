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

# Representative shape corners around a nominal tile edge (targets tile to a power-of-two mesh; we probe
# the boundary ±1, a prime, a skinny vector and a batched rank-3 shape — the cases that break naive
# tile-multiple codegen). Kept target-agnostic: these are structural corners, not a specific mesh dim.
_TILE = 16
_SHAPE_CORNERS: list[tuple[str, tuple[int, int, int], int]] = [
    ("tiny", (1, 1, 1), 2),
    ("tile_minus_1", (_TILE - 1, _TILE - 1, _TILE - 1), 2),
    ("tile", (_TILE, _TILE, _TILE), 2),
    ("tile_plus_1", (_TILE + 1, _TILE + 1, _TILE + 1), 2),
    ("prime", (17, 19, 23), 2),
    ("skinny_row", (1, 4096, 4096), 2),
    ("skinny_col", (4096, 4096, 1), 2),
    ("batched", (_TILE, _TILE, _TILE), 3),
]


@dataclass(frozen=True)
class Probe:
    name: str
    axis: str                 # the generalization axis this probe exercises (shape/dtype/layout)
    descriptor: RegionDescriptor


def _primary_shape(fam: str):
    """A nominal in-closure shape for a family: contractions carry M/K/N, unary families a single dim."""
    if fam in ("contraction", "attention"):
        return (_TILE, _TILE, _TILE)
    return (_TILE, None, None)


def probes_for_family(fam: str, cap: SemanticCapability) -> list[Probe]:
    """The derived probe set for one declared family capability."""
    probes: list[Probe] = []
    dtypes = list(cap.dtypes) or [None]
    lead = dtypes[0]
    contractionish = fam in ("contraction", "attention")

    # shape corners on the lead dtype
    for corner, (m, k, n), rank in _SHAPE_CORNERS:
        if not contractionish and corner in ("prime", "skinny_col"):
            continue                                  # unary families: a couple of corners suffice
        if rank == 3 and not cap.batch:
            continue                                  # skip batched when the unit declares batch=false
        axis = "shape" if corner != "batched" else "shape"
        d = RegionDescriptor(source=f"{fam}/{corner}", family=fam, in_dtype=lead,
                             weight_dtype=(lead if contractionish else None),
                             m=m, k=(k if contractionish else None),
                             n=(n if contractionish else None), rank=rank, batch=(2 if rank == 3 else 1))
        probes.append(Probe(name=f"{fam}.{corner}", axis=axis, descriptor=d))

    # one probe per additional declared dtype (dtype-generalization axis)
    m, k, n = _primary_shape(fam)
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


def synthesize(cap_map: dict[str, SemanticCapability]) -> list[Probe]:
    """All derived probes for a target's declared capability map (families in canonical order)."""
    out: list[Probe] = []
    order = [f for f in (*_sf.PRIMITIVES, *sorted(_sf.COMPOSITES)) if f in cap_map]
    for fam in order:
        out += probes_for_family(fam, cap_map[fam])
    return out
