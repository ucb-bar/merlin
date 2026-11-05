"""Coverage cells, origins and target-derived boundary data."""

from __future__ import annotations

from dataclasses import dataclass

OBSERVED = "observed"
DECLARED = "declared"
OBSERVED_VIA_PRIMITIVES = "observed_via_primitives"


@dataclass(frozen=True)
class Cell:
    """One coverage requirement, in ``cert_capsule_cover``'s vocabulary.

    ``alignment`` is ``"aligned"`` / ``"partial"`` / ``"sub_tile"``, or ``None`` when the target
    declares no tiling edge at all. All are required wherever an edge exists: a unit that only ever
    sees whole tiles has never exercised its tail path, and the tail is where tiling bugs live.

    ``sub_tile`` IS NOT ``partial`` UNDER ANOTHER NAME, and the difference was measured. ``partial``
    rags one axis by a single element, so the tile it produces is NEARLY FULL -- 15 of 16 lanes live
    on a 16-wide edge. A tile that is BARELY OCCUPIED is a different question for a tiling compiler:
    whether it still issues a full tile, whether it skips the pass, whether the tail predicate is
    even reached. Across the hand-authored corpus, extents of 2/4/7/8/9/10/12 against a 16-wide tile
    occur 45 times; across the derived corpus they occurred ZERO times, because the two alignment
    classes could not express one. That is the single largest structural gap between the two corpora,
    and by the measured certification cost law it is also the cheapest to close: a tile/4 capsule
    writes 16 elements and certifies in seconds, where the corpus median is 256 elements.
    """

    family: str
    dtype: str
    alignment: str | None

    def key(self) -> str:
        """The ``"/"``-joined spelling ``cert_capsule_cover`` reports, dropping absent axes."""
        return "/".join(x for x in (self.family, self.dtype, self.alignment) if x)


@dataclass
class CellOrigin:
    """Why a cell is required. Separate from :class:`Cell` so the cell stays hashable and comparable."""

    basis: str = OBSERVED  # OBSERVED | OBSERVED_VIA_PRIMITIVES | DECLARED
    observed_in: tuple[str, ...] = ()  # captures whose census contains this family
    admitted_by: tuple[str, ...] = ()  # compute units declaring the family+dtype
    citation: str = ""  # required when basis == DECLARED
    via_primitives: tuple[str, ...] = ()  # set when basis == OBSERVED_VIA_PRIMITIVES
    n_regions: int = 0  # how many regions across all captures carried it

    def to_dict(self) -> dict:
        out: dict = {"basis": self.basis}
        if self.observed_in:
            out["observed_in"] = list(self.observed_in)
        if self.admitted_by:
            out["admitted_by"] = list(self.admitted_by)
        if self.via_primitives:
            out["via_primitives"] = list(self.via_primitives)
        if self.n_regions:
            out["n_regions"] = self.n_regions
        if self.citation:
            out["citation"] = self.citation
        return out


@dataclass
class Boundaries:
    """The target's own edges, and where each came from.

    ``tile_edge`` is the value the corpus generator tiles against. On a target with no fixed hardware
    mesh it is a SOFTWARE default, and ``tile_edge_is_hardware_fact`` says so — reporting a software
    default as a hardware boundary is how a derived artifact starts lying.
    """

    tile_edge: int | None = None
    tile_edge_is_hardware_fact: bool = False
    tile_edge_source: str = ""
    block_scale_group: int | None = None  # MX E8M0 K-group, when the target has one
    block_scale_source: str = ""
    operand_store_bytes: int | None = None
    operand_store_source: str = ""

    def extent_probes(self) -> list[dict]:
        """Extents that straddle each real boundary — the derived edge cases.

        ``tile_edge`` catches the tiling tail; ``block_scale_group`` catches a K that does not divide
        the block-scale group, which is not a hypothetical: a captured LLaMA's own hidden size (344)
        is not a multiple of a 32-element group.

        SUB-BOUNDARY POINTS ARE A SEPARATE CLASS, and leaving them out was measured as the structural
        gap between this corpus and the hand-authored one. The points used to be
        ``[edge-1, edge, edge+1, edge*2]`` -- straddling the edge and doubling it, but never sampling
        WELL below it. Across the hand-authored capsules, extents of 2/4/7/8/9/10/12 against a 16-wide
        tile occur 45 times; across the derived corpus they occurred ZERO times, because no token in
        this list could produce one. A partial tile that is nearly full (``edge-1``) and one that is
        barely occupied (``edge//4``) are different cases for a tiling compiler: the first exercises
        the tail predicate, the second exercises whether a mostly-empty tile is issued at all.

        So each boundary now also contributes ``1`` (the degenerate extent -- a single row or element,
        which the hand set probes 26 times), ``edge//2`` and ``edge//4``. Deduplicated and ordered, so
        a small edge that collapses these onto each other simply yields fewer points rather than
        repeats.
        """
        out: list[dict] = []
        for name, edge, src in (
            ("tile_edge", self.tile_edge, self.tile_edge_source),
            ("block_scale_group", self.block_scale_group, self.block_scale_source),
        ):
            if not edge or edge < 2:
                continue
            e = int(edge)
            points = {1, e // 4, e // 2, e - 1, e, e + 1, e * 2}
            out.append({"boundary": name, "edge": e, "source": src, "points": sorted(x for x in points if x >= 1)})
        return out

    def to_dict(self) -> dict:
        return {
            "tile_edge": self.tile_edge,
            "tile_edge_is_hardware_fact": self.tile_edge_is_hardware_fact,
            "tile_edge_source": self.tile_edge_source,
            "block_scale_group": self.block_scale_group,
            "block_scale_source": self.block_scale_source,
            "operand_store_bytes": self.operand_store_bytes,
            "operand_store_source": self.operand_store_source,
            "extent_probes": self.extent_probes(),
        }
