"""A cheap cost signal for an inner search loop: what a schedule ASKS THE DEVICE FOR.

WHY THIS EXISTS
---------------
An agentic performance loop needs a number it can evaluate thousands of times. It has two
instruments and neither can do that job. The functional model is blind to timing -- it will report a
schedule that moves twice the bytes as free. Elaborated RTL and FireSim are timing-true and cost
minutes to hours per point. The gap between them is where the search actually lives, and it was
empty.

An UNVALIDATED proxy is worse than no proxy. Measured on this tree: a block-DMA change scored a 23%
WIN under the functional model and measured 2.8% to 12.8% WORSE on hardware. So every claim this
module makes is checked against measured hardware points, the check is a test
(``tests/dse/test_schedule_proxy.py``, over the measured points recorded with the target that
produced them), and the correlation it reaches is stated as a number rather than asserted as a property.
:mod:`merlin.perf.group_headroom` prices a model group through :func:`schedule_cost`, and the
``cost_proxy`` rung of ``src/merlin/resources/contracts/measurement_ladder.yaml`` names this module as its
implementation: a ranking signal, never a cycle count.

THE ONE FACT THAT SHAPES THE DESIGN
-----------------------------------
Bytes moved are PROVABLY insufficient on their own. :mod:`merlin.perf.mesh_occupancy` measured a
whole model's array issue count moving 21,383,000 -> 16,050,000 cycles -- a 25% swing -- with the
DRAM bytes per operand role BIT-IDENTICAL and the launch count unchanged. A proxy built on traffic
alone cannot see that, and a proxy that scores those two configurations equally is vacuous. So the
cost here is a PAIR: what the array is asked to issue, and what the movement engine is asked to
move. :func:`schedule_cost` never returns one without the other, and the test holds the pair.

THE UNIT, AND WHY THE TWO TERMS ARE COMMENSURATE
------------------------------------------------
Both terms are counted in ARRAY-TILE TRANSACTIONS -- one command's worth of work at the granularity
the array itself is built at (``array_rows x array_cols``). A compute transaction is one issued tile,
priced at one command slot plus the array occupancy that tile buys (a tile streaming a full
``array_rows`` is one slot's worth; a short one proportionally less -- no free weight). A movement
transaction is one tile's worth of bytes crossing the DRAM boundary. The two are commensurate because
on a sequenced accelerator they are entries in the SAME command stream, and that -- not a roofline
over two independent resources -- is why they are ADDED. Their relative price is the one calibrated
number in the model (:class:`Pricing`), it is never defaulted silently, and the neutral value 1.0 is
labelled uncalibrated where it is used.

MEASURED, against 71 FireSim-measured ResNet-50 compute groups (the explicit-nest arm of
``per_group_fsm_vs_explicit.json``, bitstreams ``0af6f26e`` / ``a9a190b9``, queue jobs 759/760/761;
per-group cycles span 18,885 to 3,250,325 and FSM-vs-explicit ratios span 1.13x to 5.30x):

=========================================  ==============  ====================
signal                                     Spearman rho    pairwise accuracy
=========================================  ==============  ====================
issue transactions alone                           0.878                 83.0%
movement bytes alone                               0.402                 62.9%
BOTH, uncalibrated (exchange rate 1.0)             0.920                 89.4%
BOTH, calibrated (exchange rate 3.5)               0.978                 93.2%
=========================================  ==============  ====================

That table is the argument for the design: neither term alone ranks, and the pair does. Pairwise
accuracy is the honest figure for a search loop -- how often the proxy picks the more expensive of
two candidates -- and it is why rank matters here more than absolute error, which stays large
(Pearson 0.884; median absolute relative error 43%, worst 105%, with the slot cost fitted on these
same points -- so this predicts an ORDER, not a cycle count, and must not be quoted as one).

The exchange rate is fitted, so it is checked for fitting noise rather than asserted: over 200
random half-splits the rate chosen on one half has median 3.5 (10th-90th percentile 3.0-6.0) and
scores median rho 0.970 (10th percentile 0.941) on the half it was not fitted on.

WHY THERE IS A REGIME VERDICT AND NOT ONLY A COST
--------------------------------------------------
A scalar cost is not enough, and that is measured rather than argued. On pinned elaborated RTL, nine
scheduling policies over four shapes found two levers that are ANTI-CORRELATED: a hoisted move-in
burst costs runtime only where the LOAD side is critical, and a stationary-operand reload costs
runtime only where the MESH is critical. Two shapes were load-limited (burst 13-22%, reload free) and
two were mesh-limited (reload 18-33%, burst free); **no shape was limited by both**. A proxy that
answers with one number averages two populations that never co-occur and is wrong in a different
direction on each half of a model.

So :attr:`ScheduleCost.regime` is a separate verdict, computed from the BLOCKING and needing no
simulator -- see :func:`_regime` for the four measured points and the discriminant they select. It is
:data:`UNDECIDED_REGIME` until a caller supplies the measured boundary, because the two regimes take
opposite optimisations and a guessed verdict aims the loop at the lever that is free.

:attr:`ScheduleCost.reload_pressure` and :attr:`~ScheduleCost.burst_pressure` are reported beside it
as the two levers' arms, so a caller sees how much each lever has to work with rather than only which
one to pull.

ONE LEVER IS REFUTED AND IS DELIBERATELY NOT REWARDED HERE. Hoisting the load-configuration state
measured -0.4% to +0.5% across every shape: a load config completes on issue, so paying one per
operand switch costs nothing. Nothing in this model gives it credit, and nothing should start to.

WHAT THIS CANNOT SEE -- the blind axis, named because naming it is what stops the next
23%-win-that-is-a-12%-loss
------------------------------------------------------------------------------------------
1. **Re-reads from capacity spills.** The movement term is the COMPULSORY footprint -- each tensor
   crossing the boundary once. A schedule that spills and re-reads moves more, and this under-prices
   it by exactly that factor. The tiling is derived anyway (:func:`~merlin.perf.derived_bound.select_tiling`)
   and :attr:`ScheduleCost.spill` NAMES the operands whose slab does not fit, so the risk is reported
   rather than silently priced at zero. It is not folded into the cost because, confronted with the
   71 points, charging the re-reads made the ranking WORSE (rho 0.937 -> 0.604 on the term as first
   formulated): on this device the
   schedules kept their slabs resident and the charge was fiction.
2. **Anything that is not a transaction count.** Bank conflicts, DRAM row locality, refresh, the
   host's own scalar code between launches, cache behaviour on the control processor, and dependence
   stalls inside the array are all invisible. Two schedules with the same transaction count score
   equal here and can differ on hardware for any of those reasons.
3. **Absolute cycles, unless the slot cost was measured.** Without a measured
   ``cycles_per_transaction`` the result carries ``cycles = UNKNOWN`` on purpose; the rank signal is
   still available, because the loop needs to pick the better of two candidates rather than predict a
   cycle count.
4. **Cross-device transfer.** The exchange rate, the slot cost and the regime boundary are properties
   of one device and one runtime. Carrying any of them to another device, or to the same device at
   another clock, is not licensed by anything here.
5. **Where the regime boundary actually sits.** Four measured shapes bound it to the open interval
   (4, 7] in reload multiplicity and no further, and one of them carries a ~7% per-arm noise floor
   from binary composition. A caller that supplies a boundary inside that interval is supported by
   the evidence; one that supplies a boundary outside it is not, and nothing here will say so.

Nothing in this module names a target, an opcode or a capacity: the geometry, the element widths and
the store capacities all arrive through :func:`~merlin.perf.derived_bound.machine_from_facts` from
the target's own RTL facts, and a term whose input is not evidenced is UNKNOWN with the reason
attached rather than defaulted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .decompose import UNKNOWN, _Unknown, is_unknown
from .derived_bound import Gemm, Machine, Tiling, machine_from_facts, select_tiling
from .mesh_occupancy import tile_issue_cycles
from .optimization_ledger import arithmetic_intensity

__all__ = [
    "COMPUTE_TERM",
    "Contraction",
    "LOAD_CRITICAL",
    "MESH_CRITICAL",
    "MOVEMENT_TERM",
    "Pricing",
    "ScheduleCost",
    "UNDECIDED_REGIME",
    "schedule_cost",
]

#: The two regimes a contraction can be in, and the state for "the evidence does not separate them".
#: They are not two ends of one axis: measured on pinned elaborated RTL over four shapes, NO shape
#: was limited by both, and the lever that pays in one is free in the other.
MESH_CRITICAL = "mesh_critical"
LOAD_CRITICAL = "load_critical"
UNDECIDED_REGIME = "UNDECIDED"

#: Term names, so the report and the refusal map share one vocabulary with
#: :mod:`merlin.perf.derived_bound`.
COMPUTE_TERM = "compute"
MOVEMENT_TERM = "movement"


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-int(numerator) // int(denominator))


@dataclass(frozen=True)
class Contraction:
    """One contraction a schedule must issue, in extents plus the bytes it actually moves.

    ``m``/``n``/``k`` are the contraction's own extents. The three footprints are the bytes that
    cross the DRAM boundary once -- given rather than derived from the extents, because for a
    convolution they are NOT the same thing: its ``lhs`` is the image, while ``m * k`` counts the
    im2col matrix the image is never materialised as. Passing ``m * k * operand_bytes`` there would
    over-charge a 3x3 convolution ninefold.

    ``stream_run`` is the only schedule choice this model needs and it must be stated, because it is
    what separates two schedules of the same shape. It is the longest CONTIGUOUS run of the ``m``
    axis one issued tile may cover. For a plain contraction that is ``m`` itself. For a convolution
    whose output rows are not adjacent in memory it is the output row width -- and when that width is
    below ``array_rows`` every issued tile carries a short stream, so the same work costs more
    transactions. Measured: this is precisely what separates the four worst groups of a real model
    (7-wide output rows against a 16-row array) from the ones with the identical MAC count.

    ``0`` means "unconstrained", i.e. ``m``.
    """

    m: int
    n: int
    k: int
    lhs_bytes: int
    rhs_bytes: int
    result_bytes: int
    stream_run: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("m", "n", "k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"contraction extent {name} must be a positive int, got {value!r}")
        for name in ("lhs_bytes", "rhs_bytes", "result_bytes", "stream_run"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")

    @property
    def macs(self) -> int:
        return self.m * self.n * self.k

    @property
    def moved_bytes(self) -> int:
        """The compulsory footprint: every operand and the result, crossing the boundary once."""
        return self.lhs_bytes + self.rhs_bytes + self.result_bytes

    @property
    def gemm(self) -> Gemm:
        return Gemm(self.m, self.n, self.k)

    @property
    def name(self) -> str:
        return self.label or f"{self.m}x{self.n}x{self.k}"


@dataclass(frozen=True)
class Pricing:
    """What a movement transaction costs relative to a compute one, and what a slot costs in cycles.

    Both are MEASURED properties of a device and its runtime, so neither is invented here.
    :meth:`uncalibrated` returns the neutral pairing -- every transaction priced the same -- and says
    so in its own provenance, so a report built on it cannot read as a calibrated one.
    """

    movement_per_compute: float = 1.0
    cycles_per_transaction: float | _Unknown = UNKNOWN
    #: The reload multiplicity at or above which a shape is MESH-critical rather than LOAD-critical.
    #: A device property, measured by moving a lever and watching runtime -- never guessed, because
    #: the two regimes take OPPOSITE optimisations and a wrong verdict points the loop backwards.
    mesh_critical_reload_multiplicity: float | _Unknown = UNKNOWN
    provenance: str = ""

    @classmethod
    def uncalibrated(cls) -> Pricing:
        return cls(
            movement_per_compute=1.0,
            cycles_per_transaction=UNKNOWN,
            mesh_critical_reload_multiplicity=UNKNOWN,
            provenance=(
                "UNCALIBRATED: every transaction priced the same, no cycle value for a slot, and no "
                "measured regime boundary. The rank signal is usable; the absolute cycle count and "
                "the regime verdict are not derivable and are UNKNOWN"
            ),
        )

    @property
    def calibrated(self) -> bool:
        return bool(self.provenance) and not self.provenance.startswith("UNCALIBRATED")

    def to_dict(self) -> dict[str, Any]:
        return {
            "movement_per_compute": float(self.movement_per_compute),
            "cycles_per_transaction": "UNKNOWN"
            if is_unknown(self.cycles_per_transaction)
            else float(self.cycles_per_transaction),
            "mesh_critical_reload_multiplicity": "UNKNOWN"
            if is_unknown(self.mesh_critical_reload_multiplicity)
            else float(self.mesh_critical_reload_multiplicity),
            "calibrated": self.calibrated,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ItemCost:
    """One contraction's two terms, and the tiling and spill risk behind them."""

    label: str
    macs: int
    compute_transactions: int
    movement_transactions: float
    array_issue_cycles: int
    idle_slot_share: float | None
    moved_bytes: int
    stream_run: int
    #: How many streamed row-blocks share one stationary block -- i.e. how many times the schedule
    #: re-pushes that block through the array. It is the LEVER ARM of a reuse/reload fix, and it is
    #: read straight off the blocking, which is why the regime needs no simulator.
    reload_multiplicity: int = 1
    tiling: Tiling | None = None
    spill: tuple[str, ...] = ()

    @property
    def reload_pressure(self) -> float:
        """Slots a reuse fix could remove: the preloads that re-push a block already in the array."""
        if self.reload_multiplicity <= 1:
            return 0.0
        return self.compute_transactions * (1.0 - 1.0 / self.reload_multiplicity)

    @property
    def burst_pressure(self) -> float:
        """Slots a move-in ordering fix could reorder: every movement transaction is one."""
        return self.movement_transactions

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "macs": self.macs,
            "compute_transactions": self.compute_transactions,
            "movement_transactions": round(self.movement_transactions, 4),
            "array_issue_cycles": self.array_issue_cycles,
            "idle_slot_share": self.idle_slot_share,
            "moved_bytes": self.moved_bytes,
            "stream_run": self.stream_run,
            "reload_multiplicity": self.reload_multiplicity,
            "reload_pressure": round(self.reload_pressure, 4),
            "burst_pressure": round(self.burst_pressure, 4),
            "tiling": self.tiling.to_dict() if self.tiling is not None else None,
            "spill": list(self.spill),
        }


@dataclass(frozen=True)
class ScheduleCost:
    """The proxy's answer: a rank signal, its two terms, and everything it could not establish.

    :attr:`transactions` is what a search ranks by -- it is scale-free and needs no calibration.
    :attr:`cycles` is UNKNOWN unless the caller supplied a measured slot cost, because a cycle count
    built on a guessed slot cost is a fabricated hardware claim.
    """

    workload: str
    transactions: float
    terms: Mapping[str, float]
    cycles: float | _Unknown
    array_issue_cycles: int
    moved_bytes: int
    macs: int
    intensity: Mapping[str, Any]
    pricing: Pricing
    #: :data:`MESH_CRITICAL`, :data:`LOAD_CRITICAL` or :data:`UNDECIDED_REGIME`. NEVER collapsed into
    #: the scalar cost: the two regimes take opposite optimisations and never co-occur, so a single
    #: number averages two populations and is wrong in a different direction on each.
    regime: str = UNDECIDED_REGIME
    regime_basis: str = ""
    items: tuple[ItemCost, ...] = ()
    unresolved: tuple[str, ...] = ()
    reasons: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)
    blind_to: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        """Whether the RANK signal is usable.

        Deliberately not ``not self.unresolved``: an uncalibrated result lists ``cycles`` as
        unresolved and is still perfectly usable for ranking, which is what a search loop asks for.
        A result with no terms at all is the one that establishes nothing.
        """
        return bool(self.terms)

    @property
    def spill(self) -> tuple[str, ...]:
        """Every operand whose slab does not fit, i.e. where the single-pass assumption is at risk."""
        return tuple(f"{item.label}:{name}" for item in self.items for name in item.spill)

    @property
    def reload_pressure(self) -> float:
        """What a reuse/reload fix has to work with. Only pays where the regime is mesh-critical."""
        return sum(item.reload_pressure for item in self.items)

    @property
    def burst_pressure(self) -> float:
        """What a move-in ordering fix has to work with. Only pays where the regime is load-critical."""
        return sum(item.burst_pressure for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "merlin_schedule_proxy_v1",
            "workload": self.workload,
            "transactions": round(self.transactions, 4),
            "regime": self.regime,
            "regime_basis": self.regime_basis,
            "reload_pressure": round(self.reload_pressure, 4),
            "burst_pressure": round(self.burst_pressure, 4),
            "terms": {name: round(value, 4) for name, value in self.terms.items()},
            "cycles": "UNKNOWN" if is_unknown(self.cycles) else round(float(self.cycles), 1),
            "array_issue_cycles": self.array_issue_cycles,
            "moved_bytes": self.moved_bytes,
            "macs": self.macs,
            "intensity": dict(self.intensity),
            "pricing": self.pricing.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "unresolved": list(self.unresolved),
            "reasons": dict(self.reasons),
            "provenance": dict(self.provenance),
            "spill": list(self.spill),
            "blind_to": list(self.blind_to),
        }


#: What a transaction count is structurally incapable of separating. Carried on every result so the
#: blind axis travels with the number instead of living only in this docstring.
BLIND_TO = (
    "re-reads from a capacity spill: the movement term is the COMPULSORY footprint, so a schedule "
    "that spills is under-priced by exactly the re-read factor (the operands at risk are named in "
    "`spill`)",
    "everything that is not a transaction count: bank conflicts, DRAM row locality and refresh, "
    "dependence stalls inside the array, and the control processor's own code between launches",
    "absolute cycles without a measured slot cost; `cycles` is UNKNOWN rather than guessed",
    "any other device, or the same device at another clock: the pricing and the regime boundary are "
    "properties of the one that was measured",
    "where the regime boundary sits: four measured shapes bound it to (4, 7] in reload multiplicity "
    "and no further, so a boundary outside that interval is unsupported and nothing here says so",
)


def _unresolvable(workload: str, machine: Machine, reasons: Mapping[str, str]) -> ScheduleCost:
    return ScheduleCost(
        workload=workload,
        transactions=float("nan"),
        terms={},
        cycles=UNKNOWN,
        array_issue_cycles=0,
        moved_bytes=0,
        macs=0,
        intensity={"status": "unavailable", "reason": "the machine's own facts do not ground the terms"},
        pricing=Pricing.uncalibrated(),
        unresolved=tuple(sorted(reasons)),
        reasons=dict(reasons),
        provenance=dict(machine.provenance),
        blind_to=BLIND_TO,
    )


def _item_cost(work: Contraction, machine: Machine, rows: int, cols: int, tile_bytes: int) -> ItemCost:
    run = min(work.stream_run or work.m, work.m)
    row_block = max(1, min(rows, run))
    m_tiles = _ceil_div(work.m, row_block)
    n_tiles = _ceil_div(work.n, cols)
    k_tiles = _ceil_div(work.k, rows)
    compute_transactions = m_tiles * n_tiles * k_tiles

    # The array floor, through the same primitive the occupancy work is built on: one issued tile
    # streams `row_block` rows against a depth x cols stationary block, and a partial block is
    # charged whole. Called per DISTINCT tile shape rather than per tile -- an inner-loop cost model
    # cannot afford to materialise a million descriptors, and the aggregate is identical (held by
    # `test_schedule_proxy.py::test_array_floor_matches_mesh_occupancy`).
    issue = 0
    useful = 0
    for m_index in range(2):
        m_extent = row_block if m_index == 0 else work.m - (m_tiles - 1) * row_block
        m_count = m_tiles - 1 if m_index == 0 else 1
        if m_count < 1 or m_extent < 1:
            continue
        for k_index in range(2):
            k_extent = rows if k_index == 0 else work.k - (k_tiles - 1) * rows
            k_count = k_tiles - 1 if k_index == 0 else 1
            if k_count < 1 or k_extent < 1:
                continue
            for n_index in range(2):
                n_extent = cols if n_index == 0 else work.n - (n_tiles - 1) * cols
                n_count = n_tiles - 1 if n_index == 0 else 1
                if n_count < 1 or n_extent < 1:
                    continue
                multiplicity = m_count * k_count * n_count
                issue += multiplicity * tile_issue_cycles(
                    m_extent, k_extent, n_extent, array_rows=rows, array_cols=cols
                )
                useful += multiplicity * m_extent * k_extent * n_extent
    slots = issue * rows * cols
    idle = round(1.0 - useful / slots, 6) if slots else None

    found = select_tiling(
        work.gemm,
        machine,
        lhs_bytes=work.lhs_bytes,
        rhs_bytes=work.rhs_bytes,
        result_bytes=work.result_bytes,
    )
    tiling = None if isinstance(found, str) else found
    spill: list[str] = []
    if tiling is not None and not is_unknown(machine.operand_store_bytes):
        capacity = int(machine.operand_store_bytes)
        if tiling.j_trips > 1 and work.lhs_bytes > capacity:
            spill.append("lhs")
        if tiling.i_trips > 1 and work.rhs_bytes > capacity:
            spill.append("rhs")

    return ItemCost(
        label=work.name,
        macs=work.macs,
        compute_transactions=compute_transactions,
        movement_transactions=work.moved_bytes / float(tile_bytes),
        array_issue_cycles=issue,
        idle_slot_share=idle,
        moved_bytes=work.moved_bytes,
        stream_run=row_block,
        reload_multiplicity=m_tiles,
        tiling=tiling,
        spill=tuple(spill),
    )


def _regime(costs: Sequence[ItemCost], pricing: Pricing) -> tuple[str, str]:
    """Mesh-critical or load-critical, from the blocking alone -- or UNDECIDED, never a coin toss.

    MEASURED on pinned elaborated RTL (GSIM, every arm bit-exact against the vendor kernel in the
    same binary), four shapes, two levers that are ANTI-CORRELATED:

    ======================  ====================  ==========================  ==============
    shape                   move-in burst -> JIT  remove stationary reload    reload mult.
    ======================  ====================  ==========================  ==============
    3136 x 64 x 64                        -3.2%                     -24.6%              196
    49 x 512 x 1024, 7-row                -0.6%             -17.8% / -19.8%                7
    64 x 512 x 1024                      -22.3%                      -1.5%                4
    64 x 256 x 1024                      -12.7%                      +0.7%                4
    ======================  ====================  ==========================  ==============

    No shape was limited by both, and the lever that pays in one is free in the other -- which is why
    a single scalar cost is not enough: it averages two populations that never co-occur. Of the cheap
    discriminants available here, the reload multiplicity is the one that separates those four points
    (196 and 7 mesh-critical against 4 and 4 load-critical) AND is mechanistically the reload lever's
    own lever arm. The scalar compute/movement ratio does NOT separate them (3.96, 8.80 mesh against
    6.74, 6.10 load), so the regime is a separate verdict rather than a reading of the cost.

    Four points with a ~7% per-arm noise floor at the 7-row geometry bound the boundary to the open
    interval (4, 7] and no further, so the boundary is supplied as a measured device property and the
    verdict is UNDECIDED without one. A guessed boundary would point an optimisation loop at the
    wrong lever half the time, which is worse than pointing it at nothing.
    """
    boundary = pricing.mesh_critical_reload_multiplicity
    weight = float(sum(item.compute_transactions for item in costs)) or 1.0
    mean = sum(item.reload_multiplicity * item.compute_transactions for item in costs) / weight
    if is_unknown(boundary):
        return UNDECIDED_REGIME, (
            f"reload multiplicity {mean:.2f} (transaction-weighted), but no measured boundary was "
            "supplied, and the two regimes take OPPOSITE optimisations -- a guessed verdict points "
            "the loop at the lever that is free here"
        )
    verdict = MESH_CRITICAL if mean >= float(boundary) else LOAD_CRITICAL
    return verdict, (
        f"reload multiplicity {mean:.2f} (transaction-weighted) against a measured boundary of "
        f"{float(boundary):g}: {'at or above' if verdict == MESH_CRITICAL else 'below'} it, so the "
        f"{'stationary-operand reload' if verdict == MESH_CRITICAL else 'move-in ordering'} lever is "
        "the one that pays here"
    )


def schedule_cost(
    work: Contraction | Sequence[Contraction],
    *,
    target: str,
    facts: Mapping[str, Any] | None = None,
    machine: Machine | None = None,
    pricing: Pricing | None = None,
    machine_macs_per_byte: float | None = None,
    workload: str = "",
) -> ScheduleCost:
    """THE accessor: what this schedule asks the device for, cheap enough for an inner loop.

    ``work`` is one contraction or a whole program of them. ``target`` is a parameter, as everywhere
    in this package: the geometry, the element widths and the store capacities come from that
    target's own RTL facts. ``machine`` short-circuits the facts read for a caller that already holds
    one -- which is the difference between a microsecond and a facts load, and is how a search loop
    calls this thousands of times.

    Returns a :class:`ScheduleCost` whose :attr:`~ScheduleCost.transactions` is the rank signal. Every
    input the facts did not ground is listed in :attr:`~ScheduleCost.unresolved` with its reason, and
    the result is never partially fabricated: if the geometry or the element widths are UNKNOWN there
    is no cost at all, because a transaction has no size without them.
    """
    items = (work,) if isinstance(work, Contraction) else tuple(work)
    label = workload or (items[0].name if len(items) == 1 else f"{len(items)} contraction(s)")
    if not items:
        raise ValueError("a schedule with no contraction has no cost to report")

    resolved_machine = machine if machine is not None else machine_from_facts(target, facts=facts)
    needed = ("array_rows", "array_cols", "operand_bytes")
    missing = {
        name: resolved_machine.refusals.get(name, f"{name} is UNKNOWN")
        for name in needed
        if is_unknown(getattr(resolved_machine, name))
    }
    if missing:
        return _unresolvable(label, resolved_machine, missing)

    rows, cols = int(resolved_machine.array_rows), int(resolved_machine.array_cols)
    operand_bytes = int(resolved_machine.operand_bytes)
    tile_bytes = rows * cols * operand_bytes
    price = pricing if pricing is not None else Pricing.uncalibrated()

    costs = tuple(_item_cost(item, resolved_machine, rows, cols, tile_bytes) for item in items)
    # One issued tile costs a command SLOT plus the array occupancy it buys, both in slot units: a
    # tile that streams a full `array_rows` occupies one slot's worth, a short one proportionally
    # less. Neither half alone is enough. The slot count alone cannot tell two ORIENTATIONS of one
    # shape apart -- swapping m and n leaves the tile count identical while the array's idle-column
    # share moves, which is exactly the 21.4M -> 16.1M swing `mesh_occupancy` measured at
    # bit-identical DRAM bytes. The occupancy alone cannot tell a schedule whose tiles carry short
    # streams from one whose tiles are packed, which is what separates the worst real groups from
    # equal-MAC ones. There is no free weight between them: a full tile is one slot by construction.
    compute = float(sum(item.compute_transactions + item.array_issue_cycles / float(rows) for item in costs))
    movement = float(sum(item.movement_transactions for item in costs))
    transactions = compute + float(price.movement_per_compute) * movement
    issue_cycles = sum(item.array_issue_cycles for item in costs)
    moved = sum(item.moved_bytes for item in costs)
    macs = sum(item.macs for item in costs)

    unresolved: dict[str, str] = {}
    if is_unknown(price.cycles_per_transaction):
        cycles: float | _Unknown = UNKNOWN
        unresolved["cycles"] = (
            "no measured cycles-per-transaction was supplied, so the transaction count cannot be "
            "turned into a cycle count. The RANK signal (`transactions`) does not need one"
        )
    else:
        # max, not sum: the array's own issue loop and the command stream are different resources and
        # overlap, so the cost is the slower of the two floors. Neither is allowed to hide the other.
        cycles = max(float(issue_cycles), transactions * float(price.cycles_per_transaction))

    intensity = arithmetic_intensity(macs, moved, machine_macs_per_byte=machine_macs_per_byte)
    regime, regime_basis = _regime(costs, price)
    if regime == UNDECIDED_REGIME:
        unresolved["regime"] = regime_basis

    provenance = dict(resolved_machine.provenance)
    provenance["transaction"] = (
        f"one array-tile transaction is {rows}x{cols} elements of {operand_bytes} byte(s) = "
        f"{tile_bytes} bytes, from this target's own array geometry and datapath widths"
    )
    provenance["composition"] = (
        "the two terms are ADDED because both are entries in one sequenced command stream; the "
        "array's own issue loop is a separate resource and enters as a floor under max()"
    )
    provenance["pricing"] = price.provenance or "no provenance was stated for the pricing"

    return ScheduleCost(
        workload=label,
        transactions=transactions,
        terms={COMPUTE_TERM: compute, MOVEMENT_TERM: movement},
        cycles=cycles,
        array_issue_cycles=issue_cycles,
        moved_bytes=moved,
        macs=macs,
        intensity=intensity,
        pricing=price,
        regime=regime,
        regime_basis=regime_basis,
        items=costs,
        unresolved=tuple(sorted(unresolved)),
        reasons=unresolved,
        provenance=provenance,
        blind_to=BLIND_TO,
    )
