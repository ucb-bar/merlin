"""Whether a contraction is cheaper on an outer-product matrix unit than on the vector unit, from geometry.

The historical rule routed a contraction only when ``min(M, N) >= tile_edge``. That is a statement about
utilisation, not about time: a 2048x14 output with K=14 fills 22% of a 64-wide tile, and the unit's
peak is two orders of magnitude above the vector unit's, so it is still the faster place for it. This
module replaces the threshold with an ESTIMATE in cycles for both places, derived only from the unit's
declared geometry and the contraction ABI, and records the estimate next to the decision so a reader can
check it rather than trust it.

Inputs, and where each comes from
---------------------------------
* ``tile_edge`` -- the unit's logical tile edge, VLMAX at the operand width (the provider's
  ``geometry(unit, config)[0]``). With the ABI's 8-bit operands this fixes the vector length:
  ``vlen_bits = tile_edge * OPERAND_BITS``.
* ``datapath_bytes`` -- the provider's ``geometry(...)[1]``: the operand alignment, which is the
  datapath width ``dLen / 8``. One datapath beat moves this many bytes.
* ``OPERAND_BITS`` / ``ACCUMULATOR_BITS`` -- the kernel ABI (i8 x i8 -> i32), not a hardware fact.

The model (unit-stride traffic moves a datapath beat per cycle; anything strided an element per cycle)
-----------------------------------------------------------------------------------------------------
Unit, per ``tile_edge x tile_edge`` output tile:

* each reduction step is one outer product. It is issued as ``beats**2`` datapath subtiles, where
  ``beats = vlen_bits / dLen`` (the unit's own sequencer iterates exactly these), and it cannot be
  faster than loading its two operand vectors (``2 * beats``) or than issuing its three instructions
  (row operand, column operand, accumulate -- the minimum any outer-product step issues);
* reading the tile out moves ``rows * tile_edge * 4`` bytes unit-stride, ``rows`` being its real rows.

Per call: packing the left operand K-major (a transpose: ``M*K`` elements, strided), and when ``N``
leaves a partial column tile, copying that tile's zero-padded right panel (``K * tile_edge`` bytes,
unit-stride).

Vector unit: ``M*N*K`` multiply-accumulates at ``dLen / ACCUMULATOR_BITS`` int32 lanes per cycle, plus
writing ``C``. This is its PEAK -- it ignores that a 14-wide N leaves vector lanes idle too -- so the
model leans towards leaving a contraction where it is.

A caller that has to move data to put a contraction in the unit's layout (a transpose of a stored
operand, an im2col gather, or the result's transpose when the roles are swapped) passes the ELEMENTS
those steps move, priced as strided traffic, so the decision prices the whole transformation and not
only the kernel. "One element per cycle" for strided traffic is the one modelling assumption here that
is not read off the geometry; it is the rate of a scalar copy loop, which is how a ``linalg.transpose``
lowers today, and it is what keeps the model from buying a swap with transposes it cannot afford.

Orientation
-----------
``C = A.B`` and ``C^T = B^T.A^T`` compute the same numbers. On a square tile they fill it equally, but
they read out, pack and pad different amounts: a 2048x14 output read out row by row is 2048 short rows,
its transpose is 14 long ones. Both are priced and the cheaper one is taken; the swap's extra
transposes are part of its price.

None of these numbers is a measurement. Spike is functional, so nothing here was calibrated against
time; what is guaranteed is that every term is derived from the declared geometry, that nothing names a
model or a shape, and that the estimate ships in the routing report beside each decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "ACCUMULATOR_BITS",
    "OPERAND_BITS",
    "CostModelSelector",
    "Estimate",
    "RouteDecision",
    "UnitGeometry",
    "decide",
    "estimate",
]

#: The kernel ABI's operand and accumulator widths (``i8 x i8 -> i32``). Not facts about any unit.
OPERAND_BITS = 8
ACCUMULATOR_BITS = 32
#: Instructions one outer-product reduction step issues at minimum: its two operands and the accumulate.
_STEP_ISSUE = 3
#: Cycles per element of strided data movement (a transpose, a gather): a scalar copy loop's rate.
STRIDED_ELEMENT_CYCLES = 1

#: How a decision is called in a report. ``as_is`` computes C = A.B; ``swapped`` computes C^T = B^T.A^T.
AS_IS, SWAPPED = "as_is", "swapped"


def _ceil(a: int, b: int) -> int:
    return -(-int(a) // int(b))


@dataclass(frozen=True)
class UnitGeometry:
    """``(tile_edge, datapath_bytes)`` exactly as a provider's ``geometry(unit=, config=)`` returns it."""

    tile_edge: int
    datapath_bytes: int

    def __post_init__(self) -> None:
        if int(self.tile_edge) < 1 or int(self.datapath_bytes) < 1:
            raise ValueError(f"geometry {self} is not a tile edge and a datapath width")

    @property
    def beats(self) -> int:
        """Datapath beats in one operand vector (``vLen / dLen``, rounded up; at least one).

        An alignment wider than one operand vector (a provider whose alignment is not its datapath)
        reads as one beat per vector: the vector cannot move in less than one.
        """
        return max(1, _ceil(int(self.tile_edge) * OPERAND_BITS, int(self.datapath_bytes) * 8))

    @property
    def step_cycles(self) -> int:
        """Cycles of one reduction step on one tile: the subtiles, the operand loads or the issue."""
        return max(self.beats * self.beats, 2 * self.beats, _STEP_ISSUE)

    @property
    def vector_macs_per_cycle(self) -> float:
        return (int(self.datapath_bytes) * 8) / ACCUMULATOR_BITS

    def to_dict(self) -> dict[str, int]:
        return {"tile_edge": int(self.tile_edge), "datapath_bytes": int(self.datapath_bytes)}


@dataclass(frozen=True)
class Estimate:
    """One orientation of one contraction, priced on both units."""

    orientation: str
    m: int
    n: int
    k: int
    batch: int
    unit_cycles: float
    vector_cycles: float
    utilisation: float
    m_padded: int
    n_padded: int

    @property
    def speedup(self) -> float:
        return self.vector_cycles / self.unit_cycles if self.unit_cycles else float("inf")

    def describe(self) -> str:
        pads = [
            f"padded {name} {real}->{padded}"
            for name, real, padded in (("M", self.m, self.m_padded), ("N", self.n, self.n_padded))
            if padded != real
        ]
        lead = ", ".join(pads) if pads else "fills whole tiles"
        swap = "roles swapped (C^T = B^T.A^T), " if self.orientation == SWAPPED else ""
        return f"{swap}{lead}, est. util {self.utilisation:.0%}, est. {self.speedup:.1f}x vs vector"


@dataclass(frozen=True)
class RouteDecision:
    route: bool
    chosen: Estimate
    alternatives: tuple[Estimate, ...]
    reason: str

    @property
    def orientation(self) -> str:
        return self.chosen.orientation

    def to_dict(self) -> dict[str, Any]:
        def est(e: Estimate) -> dict[str, Any]:
            d = asdict(e)
            d["unit_cycles"] = round(float(e.unit_cycles), 1)
            d["vector_cycles"] = round(float(e.vector_cycles), 1)
            d["utilisation"] = round(float(e.utilisation), 4)
            d["speedup"] = round(float(e.speedup), 3)
            return d

        return {
            "route": self.route,
            "orientation": self.orientation,
            "reason": self.reason,
            "estimate": est(self.chosen),
            "alternatives": [est(a) for a in self.alternatives],
        }


def estimate(
    batch: int, m: int, n: int, k: int, geom: UnitGeometry, *, orientation: str = AS_IS, layout_elements: int = 0
) -> Estimate:
    """Price ``batch`` independent ``(m, n, k)`` contractions on the unit and on the vector unit.

    ``m``, ``n`` are the extents AS THE KERNEL SEES THEM (already swapped for a swapped orientation).
    ``layout_elements`` is what the caller moves to put the operands and result in the kernel's layout.
    """
    e, dp = int(geom.tile_edge), int(geom.datapath_bytes)
    b, m, n, k = int(batch), int(m), int(n), int(k)
    tm, tn = _ceil(m, e), _ceil(n, e)
    acc_bytes = ACCUMULATOR_BITS // 8
    steps = tm * tn * k * geom.step_cycles
    readout = tn * m * (e * acc_bytes) / dp  # every real row of every column tile
    pack = m * k * STRIDED_ELEMENT_CYCLES  # a transpose: strided
    pad = (k * e / dp) if n % e else 0.0
    unit = b * (steps + readout + pack + pad) + layout_elements * STRIDED_ELEMENT_CYCLES
    vector = b * (m * n * k / geom.vector_macs_per_cycle + m * n * acc_bytes / dp)
    util = (m * n) / float(tm * e * tn * e)
    return Estimate(orientation, m, n, k, b, unit, vector, util, tm * e, tn * e)


def decide(
    batch: int,
    m: int,
    n: int,
    k: int,
    geom: UnitGeometry,
    *,
    layout_elements: dict[str, int] | None = None,
) -> RouteDecision:
    """Route iff the cheaper orientation's unit estimate beats the vector unit's.

    ``layout_elements`` maps an orientation to the elements its layout steps move; an orientation absent
    from it is not available (the caller could not express it) and is not priced.
    """
    layout = {AS_IS: 0} if layout_elements is None else dict(layout_elements)
    cands: list[Estimate] = []
    if AS_IS in layout:
        cands.append(estimate(batch, m, n, k, geom, orientation=AS_IS, layout_elements=layout[AS_IS]))
    if SWAPPED in layout:
        cands.append(estimate(batch, n, m, k, geom, orientation=SWAPPED, layout_elements=layout[SWAPPED]))
    if not cands:
        raise ValueError("no orientation is available to price")
    best = min(cands, key=lambda c: c.unit_cycles)
    others = tuple(c for c in cands if c is not best)
    if best.unit_cycles < best.vector_cycles:
        return RouteDecision(True, best, others, best.describe())
    return RouteDecision(
        False,
        best,
        others,
        f"cost model: vector unit is cheaper ({best.describe()}; "
        f"unit {best.unit_cycles:.0f} vs vector {best.vector_cycles:.0f} est. cycles)",
    )


class CostModelSelector:
    """A routing selector (``shape -> bool``) that prices instead of thresholding, and says why.

    ``int8_contractions`` calls :meth:`decide` when a selector has one, so the decision and its estimate
    are recorded per contraction; a plain ``shape -> bool`` caller still works through ``__call__``.
    """

    policy = "cost_model"

    def __init__(self, geometry: UnitGeometry) -> None:
        self.geometry = geometry
        self.tile_edge = int(geometry.tile_edge)

    def decide(self, shape, *, layout_elements: dict[str, int] | None = None) -> RouteDecision:
        *batch, m, n = (int(d) for d in shape.parallel)
        b = 1
        for d in batch:
            b *= d
        return decide(b, m, n, int(shape.reduction[0]), self.geometry, layout_elements=layout_elements)

    def __call__(self, shape) -> bool:
        return self.decide(shape).route

    def describe(self) -> dict[str, Any]:
        g = self.geometry
        return {
            "policy": self.policy,
            **g.to_dict(),
            "derived": {
                "vlen_bits": int(g.tile_edge) * OPERAND_BITS,
                "dlen_bits": int(g.datapath_bytes) * 8,
                "step_cycles": g.step_cycles,
                "unit_peak_macs_per_cycle": int(g.tile_edge) ** 2 / g.step_cycles,
                "vector_peak_macs_per_cycle": g.vector_macs_per_cycle,
            },
        }


class TileFillingPolicy:
    """The historical rule, kept selectable: canonical layouts only, ``min(M, N) >= tile_edge``.

    Wraps the provider's own ``selector(tile_edge)`` so a provider that customises it keeps its rule.
    ``layouts = False`` tells the rewrite not to transpose, collapse or gather anything.
    """

    policy = "tile_filling"
    layouts = False

    def __init__(self, select, tile_edge: int) -> None:
        self._select, self.tile_edge = select, int(tile_edge)

    def __call__(self, shape) -> bool:
        return bool(self._select(shape))

    def describe(self) -> dict[str, Any]:
        return {"policy": self.policy, "tile_edge": self.tile_edge}


#: The routing policies a build can name. ``cost_model`` is the default.
POLICIES = ("cost_model", "tile_filling")
#: Where a build's policy is named when the caller did not set ``MatrixRouting.policy``.
POLICY_ENV = "MERLIN_MATRIX_ROUTE_POLICY"


def routing_selector(routing):
    """The selector a ``MatrixRouting`` routes with: its ``policy``, else ``$MERLIN_MATRIX_ROUTE_POLICY``,
    else ``cost_model`` -- built from the provider's declared geometry for the routing's unit and config."""
    from ..common.paths import env

    name = getattr(routing, "policy", None) or env(POLICY_ENV) or "cost_model"
    if name not in POLICIES:
        raise ValueError(f"matrix routing policy {name!r} is not one of {POLICIES}")
    provider = routing.provider()
    edge, datapath = provider.geometry(unit=routing.unit, config=routing.config)
    if name == "tile_filling":
        return TileFillingPolicy(provider.selector(int(edge)), int(edge))
    return CostModelSelector(UnitGeometry(int(edge), int(datapath)))
