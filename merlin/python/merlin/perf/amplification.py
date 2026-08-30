"""Data-movement amplification: bytes actually moved vs bytes the computation needs.

A program that needs 4 KiB of operands but pulls 64 KiB across the bus is paying 16x for its data.
On a target where the movement engine is busy for most of the run that ratio, not overlap, is the
lever worth pulling -- and a performance bound built on *algorithmic* bytes rather than *moved* bytes
is optimistic by exactly this factor, which is the flattering direction.

**The ratio alone is a trap.** A fixed per-transfer granule -- a DMA command that always moves a
whole block whatever you asked for -- inflates the ratio on small tiles and amortizes away on large
ones. Reporting 16x without saying how much of it is that artifact invites someone to go chase a win
that proper tiling would have taken anyway. So every result here splits the ratio into two
multiplicative factors:

``granularity_factor``
    ``(transfers_min * block_bytes) / useful_bytes`` -- the fixed-block padding, where
    ``transfers_min = ceil(useful_bytes / block_bytes)`` is the fewest transfers that could carry the
    useful bytes. **This is the small-tile artifact.** It falls towards 1 as the tile grows past the
    block, so it is not a win that survives tiling.

``redundancy_factor``
    ``transfers / transfers_min`` -- bytes moved more than once: refetch, replication, re-reading an
    operand the program could have kept resident. **This is the part that survives amortization**,
    and it is the real target.

``ratio == granularity_factor * redundancy_factor`` exactly, so the split is a decomposition and not
an estimate.

Gated on the derived trait "explicit DMA / managed scratchpad": on a target whose data movement is
implicit (a hardware cache) there is no program-visible transfer to count and no useful/moved
distinction the compiler can act on, so the honest answer is
:class:`~merlin.perf.decompose.Unavailable`. The trait holding is not enough on its own -- the byte
evidence has to exist too, and a target with the trait but no measurement gets an ``Unavailable``
that names *evidence*, not *trait*, so the two situations stay distinguishable.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .decompose import UNKNOWN, Trait, Unavailable, _Unknown

__all__ = [
    "AmplificationSensitivity",
    "CorpusAmplification",
    "MovementObservation",
    "TensorOperand",
    "WorkloadAmplification",
    "amplification",
    "corpus_amplification",
    "moved_bytes_from_beats",
    "movement_trait",
    "useful_bytes",
]


@dataclass(frozen=True)
class TensorOperand:
    """One tensor a workload reads or writes, sized from its shape and dtype.

    ``broadcast`` marks an operand that materializes a single repeated value (a splat constant, an
    epsilon, a reciprocal-of-N). Its footprint is a lowering artifact rather than data the
    computation needs -- the program could synthesize it -- so it is excluded from the useful bytes
    and reported separately in ``broadcast_bytes``. Counting it would understate the amplification.
    """

    name: str
    elements: int
    element_bytes: float
    #: ``True`` for an operand the workload writes, ``False`` for one it reads.
    is_output: bool = False
    broadcast: bool = False

    @property
    def bytes(self) -> float:
        return self.elements * self.element_bytes


def useful_bytes(operands: Sequence[TensorOperand]) -> tuple[int, int]:
    """``(useful_bytes, broadcast_bytes)`` for a workload's declared operands.

    Useful bytes are the operands and results a computation genuinely consumes and produces, sized
    from shape and dtype. Broadcast splats are separated out rather than dropped silently.
    """
    useful = sum(o.bytes for o in operands if not o.broadcast)
    splat = sum(o.bytes for o in operands if o.broadcast)
    return int(round(useful)), int(round(splat))


def moved_bytes_from_beats(beats: int, beat_bytes: int) -> int:
    """Bytes on the bus for ``beats`` transfers of a ``beat_bytes``-wide data port.

    ``beat_bytes`` is an RTL fact about the target's data port and must be supplied by the caller
    from that target's derived facts. There is no default: a guessed bus width silently rescales
    every ratio in this module.
    """
    if beat_bytes <= 0:
        raise ValueError(f"beat_bytes must be positive, got {beat_bytes}")
    return int(beats) * int(beat_bytes)


@dataclass(frozen=True)
class MovementObservation:
    """What one workload actually moved, and what it needed."""

    workload: str
    #: Bytes that crossed the bus (from measured beats x the port width -- not from the shape).
    moved_bytes: int
    #: Bytes the computation needs: operands + results from shape and dtype.
    useful_bytes: int
    #: Number of movement commands the program issued. Needed to separate the fixed per-transfer
    #: granule from genuine refetch; ``None`` leaves the sensitivity UNKNOWN rather than guessed.
    transfers: int | None = None
    #: Per-command byte counts when they are individually known (heterogeneous descriptors). When
    #: given, the block is the largest command rather than the mean.
    transfer_bytes: tuple[int, ...] = ()
    broadcast_bytes: int = 0
    provenance: str = ""


@dataclass(frozen=True)
class WorkloadAmplification:
    """One workload's amplification, split into the artifact and the part that survives tiling."""

    workload: str
    moved_bytes: int
    useful_bytes: int
    ratio: float
    #: Bytes one movement command carries (the fixed granule), derived from the observation.
    block_bytes: float | _Unknown
    #: Fewest commands that could carry the useful bytes at that granule.
    transfers_min: int | _Unknown
    #: The small-tile fixed-block artifact. Falls to 1 as the tile grows past the block.
    granularity_factor: float | _Unknown
    #: Bytes moved more than once. Survives amortization; this is the real lever.
    redundancy_factor: float | _Unknown
    #: How full an average command is. Low fill == the ratio is mostly the artifact.
    fill_fraction: float | _Unknown
    provenance: str = ""

    @property
    def artifact_share(self) -> float | _Unknown:
        """Fraction of the ratio's magnitude attributable to the fixed granule, in log terms.

        ``log(granularity) / log(ratio)`` -- the two factors multiply, so their contributions add in
        logs. 1.0 means the whole ratio is the small-tile artifact; 0.0 means none of it is.
        """
        g, r = self.granularity_factor, self.ratio
        if g is UNKNOWN or r <= 1.0:
            return UNKNOWN
        return math.log(g) / math.log(r)


def movement_trait(manifest: Mapping[str, Any] | None = None,
                   facts: Mapping[str, Any] | None = None) -> Trait:
    """Does this target have explicit DMA or a software-managed scratchpad?

    Derived from the target's own description, never from its name:

    * a **declared movement capability** on any compute unit (the target says a unit moves data
      under program control), and/or
    * a **software-managed on-chip memory**: a resident/scratchpad memory model, or discovered
      on-chip memories in the RTL facts.

    Either establishes the trait, because either gives the program a transfer it chose to issue.
    A hardware cache gives neither, and on such a target this analysis has nothing to count.
    """
    evidence: list[str] = []
    body = (facts or {}).get("facts", facts or {})

    units = list((manifest or {}).get("compute_units") or ())
    units += list((manifest or {}).get("derived_compute_units") or ())
    for u in units:
        if not isinstance(u, Mapping):
            continue
        for cap in u.get("semantic_capabilities") or ():
            if cap.get("family") == "movement":
                evidence.append(f"compute unit {u.get('name')!r} declares a movement capability")
                break

    mem = (manifest or {}).get("memory_model") or {}
    managed = [k for k, v in mem.items() if v is True]
    if managed:
        evidence.append(f"memory_model declares {sorted(managed)} (software-managed on-chip state)")

    memories = [m for m in (body.get("memories") or ()) if isinstance(m, Mapping)]
    if memories:
        evidence.append("RTL facts discover on-chip memories "
                        f"{sorted(m.get('name', '?') for m in memories)}")

    if evidence:
        return Trait("explicit_movement", True, evidence="; ".join(evidence))
    return Trait("explicit_movement", None,
                 evidence="no declared movement capability, managed memory model or discovered "
                          "on-chip memory",
                 missing=("explicit DMA or a software-managed scratchpad",))


def _block_and_min(obs: MovementObservation) -> tuple[float | _Unknown, int | _Unknown]:
    """Derive the fixed per-command granule and the fewest commands that could carry the payload.

    Refuses on fewer than two observed commands: the granule is a fitted parameter and one transfer
    is one point. A block derived from a single command cannot be told apart from that command's
    payload, which would report every workload as 100% redundancy and 0% artifact.
    """
    if obs.transfer_bytes:
        if len(obs.transfer_bytes) < 2:
            return UNKNOWN, UNKNOWN
        block: float = float(max(obs.transfer_bytes))
    else:
        if obs.transfers is None or obs.transfers < 2:
            return UNKNOWN, UNKNOWN
        block = obs.moved_bytes / obs.transfers
    if block <= 0 or obs.useful_bytes <= 0:
        return UNKNOWN, UNKNOWN
    return block, max(1, math.ceil(obs.useful_bytes / block))


def amplification(obs: MovementObservation, *,
                  trait: Trait | None = None,
                  manifest: Mapping[str, Any] | None = None,
                  facts: Mapping[str, Any] | None = None) -> WorkloadAmplification | Unavailable:
    """Amplification for one workload, with the fixed-granule sensitivity attached.

    Returns :class:`Unavailable` when the movement trait is not established (the target has no
    program-visible transfers) or when the byte evidence is missing (it does, but nobody measured
    it). Those are different failures and the returned ``missing`` says which.
    """
    tr = trait if trait is not None else movement_trait(manifest, facts)
    if tr.satisfied is not True:
        return Unavailable("data-movement amplification", tr.missing or ("explicit data movement",),
                           tr.evidence)
    if obs.useful_bytes <= 0:
        return Unavailable("data-movement amplification",
                           ("the bytes the computation needs (operand shapes + dtypes)",),
                           f"{obs.workload}: useful_bytes={obs.useful_bytes}")
    if obs.moved_bytes <= 0:
        return Unavailable("data-movement amplification",
                           ("measured moved bytes (beats x the port width)",),
                           f"{obs.workload}: moved_bytes={obs.moved_bytes}")

    ratio = obs.moved_bytes / obs.useful_bytes
    block, tmin = _block_and_min(obs)
    if block is UNKNOWN:
        return WorkloadAmplification(
            workload=obs.workload, moved_bytes=obs.moved_bytes, useful_bytes=obs.useful_bytes,
            ratio=ratio, block_bytes=UNKNOWN, transfers_min=UNKNOWN, granularity_factor=UNKNOWN,
            redundancy_factor=UNKNOWN, fill_fraction=UNKNOWN, provenance=obs.provenance)

    n_transfers = len(obs.transfer_bytes) if obs.transfer_bytes else int(obs.transfers)
    granularity = (tmin * block) / obs.useful_bytes
    redundancy = n_transfers / tmin
    return WorkloadAmplification(
        workload=obs.workload, moved_bytes=obs.moved_bytes, useful_bytes=obs.useful_bytes,
        ratio=ratio, block_bytes=block, transfers_min=tmin, granularity_factor=granularity,
        redundancy_factor=redundancy,
        fill_fraction=(obs.useful_bytes / n_transfers) / block, provenance=obs.provenance)


@dataclass(frozen=True)
class AmplificationSensitivity:
    """How much of a corpus's amplification is the small-tile artifact, and how much survives.

    ``block_bytes_by_workload`` is kept per workload because a per-command granule is a *descriptor*
    choice: two workloads carrying the same tile can enqueue different transfer sizes, so a single
    corpus-wide block would be a fit to a bimodal population. When the blocks disagree the report
    says so instead of averaging them into a number nobody can act on.
    """

    n_points: int
    block_bytes_by_workload: dict[str, float]
    block_bytes_consistent: bool | None
    #: Geometric mean of the ratio that survives amortization, over the workloads that resolved.
    amortized_ratio: float | _Unknown
    #: Geometric mean of the small-tile fixed-block factor.
    granularity_factor: float | _Unknown
    #: Mean fraction (in logs) of the observed ratio explained by the fixed granule.
    artifact_share: float | _Unknown
    note: str = ""


def _geomean(values: Sequence[float]) -> float | _Unknown:
    vals = [v for v in values if v > 0]
    if not vals:
        return UNKNOWN
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


@dataclass(frozen=True)
class CorpusAmplification:
    """Amplification over a corpus, plus its sensitivity to the fixed per-transfer granule."""

    workloads: dict[str, WorkloadAmplification] = field(default_factory=dict)
    unavailable: dict[str, Unavailable] = field(default_factory=dict)
    sensitivity: AmplificationSensitivity | Unavailable | None = None

    @property
    def ratios(self) -> dict[str, float]:
        return {n: w.ratio for n, w in self.workloads.items()}


def corpus_amplification(observations: Iterable[MovementObservation], *,
                         trait: Trait | None = None,
                         manifest: Mapping[str, Any] | None = None,
                         facts: Mapping[str, Any] | None = None) -> CorpusAmplification:
    """:func:`amplification` over a corpus, with the corpus-level sensitivity.

    The sensitivity needs **at least two workloads that resolved a granule** -- one point cannot
    separate a fixed per-transfer cost from a scale-invariant one, and pretending otherwise is how a
    single measurement becomes a fitted law.
    """
    tr = trait if trait is not None else movement_trait(manifest, facts)
    ok: dict[str, WorkloadAmplification] = {}
    bad: dict[str, Unavailable] = {}
    for obs in observations:
        r = amplification(obs, trait=tr)
        if isinstance(r, Unavailable):
            bad[obs.workload] = r
        else:
            ok[obs.workload] = r

    resolved = {n: w for n, w in ok.items() if w.block_bytes is not UNKNOWN}
    if len(resolved) < 2:
        sens: AmplificationSensitivity | Unavailable = Unavailable(
            "amplification sensitivity",
            ("at least two workloads with >=2 movement commands each",),
            f"{len(resolved)} workload(s) resolved a per-transfer granule; a fixed per-transfer "
            f"cost cannot be separated from a scale-invariant one on one point")
        return CorpusAmplification(workloads=ok, unavailable=bad, sensitivity=sens)

    blocks = {n: float(w.block_bytes) for n, w in resolved.items()}
    consistent = max(blocks.values()) - min(blocks.values()) < 1e-9
    shares = [w.artifact_share for w in resolved.values() if w.artifact_share is not UNKNOWN]
    sens = AmplificationSensitivity(
        n_points=len(resolved),
        block_bytes_by_workload=blocks,
        block_bytes_consistent=consistent,
        amortized_ratio=_geomean([float(w.redundancy_factor) for w in resolved.values()]),
        granularity_factor=_geomean([float(w.granularity_factor) for w in resolved.values()]),
        artifact_share=(sum(shares) / len(shares)) if shares else UNKNOWN,
        note=("the per-command granule is uniform across the corpus" if consistent else
              "the per-command granule VARIES across the corpus -- it is a per-workload descriptor "
              "choice, so a single corpus-wide block would be a fit to a mixed population"))
    return CorpusAmplification(workloads=ok, unavailable=bad, sensitivity=sens)
