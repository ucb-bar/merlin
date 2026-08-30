"""What a search may spend, denominated in the unit that measurement says is scarce.

The obvious design rations *simulator queries* -- and on the target this layer was first measured
against that is wrong by two to three orders of magnitude. One cycle-accurate query is a fraction of
a second to a few seconds; producing the one program that query evaluates costs a median of ~36
minutes of agent wall clock and tens of dollars. The oracle is a fraction of a percent of the cost of
a datapoint, so machinery that rations it rations the cheap thing and lets the expensive thing run
unmetered.

The correction is NOT "synthesis calls are the scarce unit". It is that **which unit is scarce is a
measurement, and therefore a parameter**. On another target in this repo the deep simulator is ~115 s
per kernel and the deepest is tens of minutes; there, simulation genuinely is scarce, and the same
selection loop must ration it instead. So this module:

* prices each :class:`Channel` of spend from samples, a fitted cost law, or nothing at all;
* refuses to name a scarce unit while any channel is unpriced (:func:`scarce_unit` returns
  :class:`~merlin.perf.decompose.Unavailable`) -- an unpriced channel cannot be ruled out as the
  expensive one, and guessing here is the whole error being corrected;
* denominates a :class:`Budget` in whichever channel won, so "budget exhausted" means something
  measured rather than something assumed.

It also closes the calibration hole in :mod:`merlin.targetgen.tier_policy`: that module's cost table
is process-local (a module dict, no file I/O), so every process starts uncalibrated and re-pays the
probe. :func:`save_tier_costs` / :func:`load_tier_costs` persist it per ``(target, tier)`` under
``artifacts_dir()/capsule-bench/<target>/tier_policy/``, and :func:`calibration` reports the
uncalibrated case as a named state rather than letting a caller assume it away.

Nothing here knows a target. The target is a parameter, the tier names come from the caller's own
adapter map, and every price is read from evidence the caller supplies.
"""
from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common.paths import artifacts_dir
from .decompose import UNKNOWN, Unavailable, _Unknown, is_unknown
from merlin.targetgen import tier_policy

#: How a price was arrived at. Mirrors :class:`merlin.perf.oracle_cost.Provenance` deliberately --
#: the two vocabularies must not drift, because a price is only as good as its construction.
MEASURED = "MEASURED"
PROJECTED = "PROJECTED"
UNPRICED = "UNPRICED"


@dataclass(frozen=True)
class Channel:
    """One channel of spend, priced per *item*, with what one datapoint costs in it.

    A "datapoint" is one evaluated candidate. The two numbers are separate on purpose: a channel can
    be cheap per item and still dominate a datapoint (many items), or expensive per item and
    irrelevant (a fraction of an item). Comparing channels on per-item price alone is how a cheap
    oracle run thousands of times gets mistaken for free.

    ``seconds_per_item`` / ``dollars_per_item`` are ``None`` when nobody measured them. ``None`` is
    not zero: an unpriced channel blocks :func:`scarce_unit` rather than sorting to the bottom.
    """

    name: str
    seconds_per_item: float | None
    dollars_per_item: float | None = None
    #: How many items of this channel one evaluated candidate consumes.
    items_per_datapoint: float = 1.0
    n: int = 0
    provenance: str = UNPRICED
    evidence: str = ""
    #: Free-text warnings that travel with the price (extrapolation, lower bound, mixed concurrency).
    notes: tuple[str, ...] = ()

    @property
    def priced(self) -> bool:
        return self.seconds_per_item is not None and self.provenance != UNPRICED

    @property
    def seconds_per_datapoint(self) -> "float | _Unknown":
        if self.seconds_per_item is None:
            return UNKNOWN
        return self.seconds_per_item * self.items_per_datapoint

    @property
    def dollars_per_datapoint(self) -> "float | _Unknown":
        if self.dollars_per_item is None:
            return UNKNOWN
        return self.dollars_per_item * self.items_per_datapoint

    def to_dict(self) -> dict:
        def _s(v: object) -> object:
            return "UNKNOWN" if is_unknown(v) else v

        return {"name": self.name, "seconds_per_item": self.seconds_per_item,
                "dollars_per_item": self.dollars_per_item,
                "items_per_datapoint": self.items_per_datapoint,
                "seconds_per_datapoint": _s(self.seconds_per_datapoint),
                "dollars_per_datapoint": _s(self.dollars_per_datapoint),
                "n": self.n, "provenance": self.provenance, "evidence": self.evidence,
                "notes": list(self.notes)}


def unpriced_channel(name: str, *, missing: str, items_per_datapoint: float = 1.0) -> Channel:
    """A channel nobody has measured. Kept in the comparison so its absence is visible."""
    return Channel(name=name, seconds_per_item=None, dollars_per_item=None,
                   items_per_datapoint=items_per_datapoint, n=0, provenance=UNPRICED,
                   evidence=f"no price: {missing}")


def channel_from_samples(name: str, *, seconds: Sequence[float] = (),
                         dollars: Sequence[float] = (),
                         items_per_datapoint: float = 1.0,
                         evidence: str = "",
                         notes: Sequence[str] = ()) -> Channel:
    """Price a channel from observed per-item samples. Median, because these distributions are
    long-tailed: the runs this was measured on span 900-65,401 s and $0.51-$103.17, and a mean over
    that is a number no single item ever cost.

    An empty ``seconds`` yields an UNPRICED channel rather than a zero -- a channel with no
    observations has no price, and the two must not read alike.
    """
    secs = [float(s) for s in seconds if s is not None and float(s) >= 0]
    bucks = [float(d) for d in dollars if d is not None and float(d) >= 0]
    if not secs:
        return Channel(name=name, seconds_per_item=None,
                       dollars_per_item=statistics.median(bucks) if bucks else None,
                       items_per_datapoint=items_per_datapoint, n=len(bucks),
                       provenance=UNPRICED,
                       evidence=evidence or "no wall-clock samples supplied",
                       notes=tuple(notes))
    return Channel(name=name, seconds_per_item=statistics.median(secs),
                   dollars_per_item=statistics.median(bucks) if bucks else None,
                   items_per_datapoint=items_per_datapoint, n=len(secs), provenance=MEASURED,
                   evidence=evidence or f"median of {len(secs)} observed item(s)",
                   notes=tuple(notes))


def channel_from_cost_law(law, *, cycles: int, words: int, name: str | None = None,
                          items_per_datapoint: float = 1.0,
                          dollars_per_item: float | None = None) -> Channel:
    """Price an oracle channel by *projecting* a fitted :class:`~merlin.perf.oracle_cost.CostLaw`.

    The law's own honesty flags are carried through rather than flattened: a projection that excludes
    a term it could not fit is a LOWER bound, and one that reaches past the largest measured value on
    an axis is an extrapolation. Both become ``notes`` and downgrade the provenance to
    :data:`PROJECTED`, so a budget built on them cannot be mistaken for one built on direct
    observation of the query it is about to run.
    """
    est = law.estimate(cycles, words)
    notes: list[str] = []
    provenance = MEASURED
    if est.is_lower_bound:
        notes.append(f"LOWER BOUND -- the law could not supply {'+'.join(est.excluded)}")
        provenance = PROJECTED
    beyond = {a: f for a, f in est.extrapolation.items() if f > 1.0}
    if beyond:
        notes.append("EXTRAPOLATED beyond the measured domain: "
                     + ", ".join(f"{a} x{f:.3g}" for a, f in sorted(beyond.items())))
        provenance = PROJECTED
    if est.assumed:
        notes.append(f"assumed term(s): {'+'.join(sorted(set(est.assumed)))}")
    return Channel(
        name=name or est.substrate,
        seconds_per_item=float(est.seconds),
        dollars_per_item=dollars_per_item,
        items_per_datapoint=items_per_datapoint,
        n=int(getattr(law, "n_samples", 0) or 0),
        provenance=provenance,
        evidence=(f"projected from a fitted cost law at concurrency={est.concurrency} for "
                  f"{cycles} cycles / {words} words"),
        notes=tuple(notes))


def scarce_unit(channels: Sequence[Channel]) -> "Channel | Unavailable":
    """The channel that dominates the cost of one datapoint, or a refusal naming what is unpriced.

    Refusing on ANY unpriced channel is the point. The failure this replaces was not "we picked the
    wrong channel", it was "we never priced the other one" -- and a channel with no price cannot be
    ruled out as the expensive one. Ties break on name so the choice is deterministic.
    """
    if not channels:
        return Unavailable("the scarce budget unit", ("at least two priced channels of spend",),
                           "no channels supplied")
    unpriced = sorted(c.name for c in channels if not c.priced)
    if unpriced:
        return Unavailable(
            "the scarce budget unit",
            tuple(f"a measured per-item price for the {n!r} channel" for n in unpriced),
            f"{len(unpriced)} of {len(channels)} channel(s) unpriced ({', '.join(unpriced)}); an "
            "unpriced channel cannot be ruled out as the expensive one")
    if len(channels) < 2:
        return Unavailable("the scarce budget unit", ("a second priced channel to compare against",),
                           f"only {channels[0].name!r} was priced; scarcity is a comparison")
    return max(channels, key=lambda c: (float(c.seconds_per_datapoint), c.name))


@dataclass(frozen=True)
class UnitReport:
    """The scarcity comparison itself, so a result can cite why it rationed what it rationed."""

    unit: "Channel | Unavailable"
    channels: tuple[Channel, ...]

    @property
    def established(self) -> bool:
        return isinstance(self.unit, Channel)

    @property
    def ratios(self) -> dict[str, "float | _Unknown"]:
        """Each channel's share of one datapoint's seconds, against the scarce one."""
        if not self.established:
            return {c.name: UNKNOWN for c in self.channels}
        base = float(self.unit.seconds_per_datapoint)  # type: ignore[union-attr]
        out: dict[str, float | _Unknown] = {}
        for c in self.channels:
            s = c.seconds_per_datapoint
            out[c.name] = UNKNOWN if is_unknown(s) or base <= 0 else float(s) / base
        return out

    def to_dict(self) -> dict:
        return {"unit": self.unit.to_dict() if self.established else str(self.unit),
                "established": self.established,
                "channels": [c.to_dict() for c in self.channels],
                "share_of_datapoint": {k: ("UNKNOWN" if is_unknown(v) else v)
                                       for k, v in self.ratios.items()}}


def unit_report(channels: Sequence[Channel]) -> UnitReport:
    """:func:`scarce_unit` plus the comparison that produced it."""
    return UnitReport(unit=scarce_unit(list(channels)), channels=tuple(channels))


@dataclass
class Budget:
    """A ledger denominated in the measured scarce unit.

    ``limit_items`` is a count of scarce-unit items (synthesis calls where synthesis dominates,
    oracle queries where the oracle does). ``limit_seconds`` / ``limit_dollars`` are optional caps on
    the same spend expressed differently; whichever binds first exhausts the budget, and
    :attr:`exhausted_reason` says which.

    Unlimited by default, following :func:`merlin.targetgen.tier_policy.budget_seconds`: a budget
    that silently narrows what was explored is the quiet coverage loss this repo already had once.
    A caller opts in, and everything the budget then skips is reported by name.
    """

    unit: Channel
    limit_items: float | None = None
    limit_seconds: float | None = None
    limit_dollars: float | None = None
    #: Charged to :func:`merlin.targetgen.tier_policy.note_spend` as well, so the repo's existing
    #: certify-tier ledger sees this loop's spend. Empty means "do not charge the shared ledger".
    target: str = ""
    spent_items: float = 0.0
    spent_seconds: float = 0.0
    spent_dollars: float = 0.0
    #: One entry per charge: ``(items, seconds, dollars, label)``.
    ledger: list[tuple[float, float, float, str]] = field(default_factory=list)

    def charge(self, *, items: float = 1.0, seconds: float | None = None,
               dollars: float | None = None, label: str = "") -> None:
        """Charge one unit of work. Seconds and dollars default to the unit's measured price."""
        items = max(0.0, float(items))
        if seconds is None:
            s = self.unit.seconds_per_item
            seconds = 0.0 if s is None else s * items
        if dollars is None:
            d = self.unit.dollars_per_item
            dollars = 0.0 if d is None else d * items
        seconds = max(0.0, float(seconds))
        dollars = max(0.0, float(dollars))
        self.spent_items += items
        self.spent_seconds += seconds
        self.spent_dollars += dollars
        self.ledger.append((items, seconds, dollars, label))
        if self.target:
            tier_policy.note_spend(self.target, seconds)

    @property
    def exhausted_reason(self) -> str | None:
        """Which cap bound, or ``None`` while budget remains. Checked in a fixed order so the
        message is deterministic when two caps bind on the same charge."""
        if self.limit_items is not None and self.spent_items >= self.limit_items:
            return (f"{self.spent_items:g} of {self.limit_items:g} {self.unit.name} item(s) spent")
        if self.limit_seconds is not None and self.spent_seconds >= self.limit_seconds:
            return f"{self.spent_seconds:.0f}s of {self.limit_seconds:.0f}s spent"
        if self.limit_dollars is not None and self.spent_dollars >= self.limit_dollars:
            return f"${self.spent_dollars:.2f} of ${self.limit_dollars:.2f} spent"
        return None

    @property
    def exhausted(self) -> bool:
        return self.exhausted_reason is not None

    @property
    def remaining_items(self) -> "float | _Unknown":
        return UNKNOWN if self.limit_items is None else max(0.0, self.limit_items - self.spent_items)

    def can_afford(self, items: float = 1.0) -> tuple[bool, str | None]:
        """May the next charge of ``items`` be made? ``(True, None)`` or ``(False, reason)``.

        Mirrors :func:`merlin.targetgen.tier_policy.may_certify`'s shape: the refusal carries a
        reason a report can print, and it is explicitly NOT a verdict about the candidate -- it did
        not run.
        """
        why = self.exhausted_reason
        if why is not None:
            return False, (f"budget exhausted ({why}); NOT a verdict on this candidate -- it did "
                           f"not run")
        if self.limit_items is not None and self.spent_items + items > self.limit_items:
            return False, (f"the next charge of {items:g} {self.unit.name} item(s) would exceed the "
                           f"{self.limit_items:g}-item budget ({self.spent_items:g} already spent)")
        s = self.unit.seconds_per_item
        if self.limit_seconds is not None and s is not None and \
                self.spent_seconds + s * items > self.limit_seconds:
            return False, (f"the next charge would exceed the {self.limit_seconds:.0f}s budget "
                           f"({self.spent_seconds:.0f}s already spent)")
        d = self.unit.dollars_per_item
        if self.limit_dollars is not None and d is not None and \
                self.spent_dollars + d * items > self.limit_dollars:
            return False, (f"the next charge would exceed the ${self.limit_dollars:.2f} budget "
                           f"(${self.spent_dollars:.2f} already spent)")
        return True, None

    def to_dict(self) -> dict:
        return {"unit": self.unit.to_dict(), "limit_items": self.limit_items,
                "limit_seconds": self.limit_seconds, "limit_dollars": self.limit_dollars,
                "spent_items": self.spent_items, "spent_seconds": self.spent_seconds,
                "spent_dollars": self.spent_dollars, "exhausted": self.exhausted,
                "exhausted_reason": self.exhausted_reason,
                "remaining_items": ("UNKNOWN" if is_unknown(self.remaining_items)
                                    else self.remaining_items),
                "charges": len(self.ledger)}


def budget_from_channels(channels: Sequence[Channel], **limits) -> "Budget | Unavailable":
    """A :class:`Budget` denominated in whatever :func:`scarce_unit` says is scarce.

    Propagates the refusal: with the unit unestablished there is no denomination, and a budget with
    an assumed denomination is exactly the failure this module exists to prevent.

    With no ``limit_seconds`` given the wall-clock cap falls back to
    :func:`merlin.targetgen.tier_policy.budget_seconds`, so the env knob that already caps
    certify-tier spend in this repo caps this loop too rather than being silently a second,
    unrelated budget.
    """
    unit = scarce_unit(list(channels))
    if isinstance(unit, Unavailable):
        return unit
    limits.setdefault("limit_seconds", tier_policy.budget_seconds())
    return Budget(unit=unit, **limits)


# --- tier-cost persistence -------------------------------------------------------------------------

def tier_cost_path(target: str) -> Path:
    """Where a target's learned per-tier oracle prices live.

    Under the capsule-bench concern, keyed by target at folder level like every other product here.
    Deliberately NOT ``.oracle_timing.json``: that file is written by the readiness check for a
    different consumer (driver timeouts) and holds one tier for one config.
    """
    return artifacts_dir() / "capsule-bench" / target / "tier_policy" / "costs.json"


def save_tier_costs(target: str, tiers: Iterable[str]) -> Path:
    """Persist what this process learned about ``target``'s tiers, merged over what is on disk.

    Only the MEDIAN survives a round trip -- :mod:`~merlin.targetgen.tier_policy` exposes no raw
    sample list -- so a reloaded price is one observation's worth of evidence, and
    :func:`load_tier_costs` says so. That is a smaller claim than the process that measured it had,
    and it is the honest one.
    """
    path = tier_cost_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    on_disk: dict[str, float] = {}
    if path.is_file():
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            on_disk = {str(k): float(v) for k, v in (body.get("median_seconds") or {}).items()}
        except (ValueError, TypeError):
            on_disk = {}
    for tier in sorted(set(tiers)):
        seen = tier_policy.observed_cost(target, tier)
        if seen is not None:
            on_disk[tier] = float(seen)
    path.write_text(json.dumps(
        {"schema": 1, "median_seconds": dict(sorted(on_disk.items())),
         "note": "medians only; tier_policy exposes no raw sample list, so a reload is worth one "
                 "observation per tier"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_tier_costs(target: str) -> tuple[str, ...]:
    """Replay persisted prices into :mod:`~merlin.targetgen.tier_policy`. Returns the tiers primed.

    Closes the process-local hole: without this every process pays ``tier_order``'s
    unmeasured-tiers-first probe again, which is a fraction of a second on a fast oracle and minutes
    on a slow one.
    """
    path = tier_cost_path(target)
    if not path.is_file():
        return ()
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return ()
    primed: list[str] = []
    for tier, seconds in sorted((body.get("median_seconds") or {}).items()):
        if tier_policy.observed_cost(target, tier) is None:
            tier_policy.record_cost(target, str(tier), float(seconds))
            primed.append(str(tier))
    return tuple(primed)


@dataclass(frozen=True)
class Calibration:
    """Whether every tier of a target has a price yet, and which do not.

    Tri-state by construction: ``calibrated`` is ``False`` with ``unpriced`` naming the gap, never a
    silent ``True``. A caller that wants to proceed uncalibrated must say so; it cannot do it by
    forgetting to ask.
    """

    target: str
    calibrated: bool
    priced: tuple[str, ...]
    unpriced: tuple[str, ...]
    loaded_from_disk: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict:
        return {"target": self.target, "calibrated": self.calibrated, "priced": list(self.priced),
                "unpriced": list(self.unpriced),
                "loaded_from_disk": list(self.loaded_from_disk), "note": self.note}


def calibration(target: str, tiers: Iterable[str], *, load: bool = True) -> Calibration:
    """Report ``target``'s tier-price calibration, optionally priming it from disk first."""
    wanted = sorted(set(tiers))
    primed = load_tier_costs(target) if load else ()
    priced = [t for t in wanted if tier_policy.observed_cost(target, t) is not None]
    unpriced = [t for t in wanted if t not in set(priced)]
    if unpriced:
        note = (f"{len(unpriced)} tier(s) have no price on this target: {', '.join(unpriced)}. "
                "Ordering will try them FIRST (tier_policy.tier_order sorts unmeasured tiers ahead "
                "of measured ones so the ladder always learns), and any budget denominated in a "
                "tier price is missing theirs.")
    else:
        note = f"every requested tier of {target!r} has a price"
    return Calibration(target=target, calibrated=not unpriced, priced=tuple(priced),
                       unpriced=tuple(unpriced), loaded_from_disk=primed, note=note)


def tier_costs(target: str, tiers: Iterable[str]) -> dict[str, float]:
    """``{tier: median seconds}`` over the tiers that have a price. Unpriced tiers are ABSENT rather
    than defaulted, so a consumer (``oracle_schedule.schedule``'s ``cost_s``) sees the hole."""
    out: dict[str, float] = {}
    for tier in sorted(set(tiers)):
        seen = tier_policy.observed_cost(target, tier)
        if seen is not None:
            out[tier] = float(seen)
    return out


def channels_from_tiers(target: str, tiers: Iterable[str], *,
                        items_per_datapoint: Mapping[str, float] | None = None) -> list[Channel]:
    """One :class:`Channel` per oracle tier, priced from what the tier actually cost on ``target``.

    A tier nobody has run yields an UNPRICED channel, which is what blocks :func:`scarce_unit` --
    exactly the behaviour wanted, because on a target whose deep tier is tens of minutes the answer
    to "what is scarce" is decided by the tier that has not been timed yet.
    """
    per = dict(items_per_datapoint or {})
    out: list[Channel] = []
    for tier in sorted(set(tiers)):
        seen = tier_policy.observed_cost(target, tier)
        k = float(per.get(tier, 1.0))
        if seen is None:
            out.append(unpriced_channel(tier, missing=f"tier {tier!r} has never been timed on this "
                                                      f"target", items_per_datapoint=k))
        else:
            out.append(Channel(name=tier, seconds_per_item=float(seen), dollars_per_item=None,
                               items_per_datapoint=k, n=1, provenance=MEASURED,
                               evidence=f"median observed wall clock for tier {tier!r}"))
    return out
