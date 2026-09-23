"""Does an expert corpus and our own schedule AGREE on a capability axis, and over how many fields?

`merlin/contract/schedule_ir_coverage.yaml` declares thirteen capability axes and a gate computes each
one's status from the live tree. Eleven read EXPRESSED -- a primitive claims the axis, exports its
symbol, declares an obligation and has a live test -- and none reads EXERCISED, because EXERCISED is
the state that needs something the tree could not do: a resolvable expert corpus populating the axis'
CCA fields, and our own schedule agreeing with it on them. This module is that measurement.

THE NUMBER IS A TRIPLE, NEVER A RATIO. ``expressed / denominator / unexercised``. A pair of numbers
invites the reader to divide them, and division is precisely where this kind of measurement goes
wrong: the fields a comparator could not decide vanish from BOTH sides of the fraction and the
remaining ones -- the easy ones -- read as a high score. Here every field the axis names is in the
denominator whatever happened to it, so a measurement resting on two comparable fields out of nine
reads as ``2/9/7`` and not as ``100%``. :meth:`Triple.bounds` states the same thing as the interval
the evidence actually supports: thin evidence is a WIDE range, not a high number.

UNDECIDABLES ARE CHARGED TO THE DENOMINATOR, AND THE COMPARATOR IS ASKED RATHER THAN ASSUMED.
:func:`~merlin.kernels.cca.cca_agree` reports the fields it compared, and this module takes that list
as the denominator's decided part. Everything else the axis names is undecidable, with the reason
DERIVED from the two CCAs rather than from a list kept here:

    both sides None          neither the corpus nor the schedule determines it
    one side None            only one side determines it -- the comparator skips it silently
    both sides populated,    the comparator did not compare this field at all
    still not compared

That last case is real and is why the reason is derived. `cca_agree` iterates a hardcoded facet tuple
that omits two facets, so a field on one of them is invisible to it even when BOTH sides populate it,
and the report still says the two agree. `merlin/tests/kernels/test_cca_agree_blindness_is_declared.py`
declares that gap with a tripwire on the day it heals. Nothing here duplicates that declaration -- a
second copy of a known-limitation list is a second thing to forget -- and nothing here widens the
tuple, which would change what an existing backend's bijection suite asserts without anyone deciding
to. Detecting it by OBSERVATION means this measurement improves by itself when the tuple is widened,
and keeps charging the field honestly until then.

A CORPUS THAT DOES NOT RESOLVE REPORTS UNMEASURED AND NAMES THE MISSING INPUT. Never a zero
denominator (a measurement of nothing reads exactly like agreement on everything) and never a
denominator quietly shrunk to the fields that happened to be available. ``missing_input`` is the
environment variable, path or record that would make the corpus readable, taken from the corpus'
declared pin rather than spelled here.

A CORPUS ADMITS ONLY WHAT AN AUDIT SAYS A HUMAN WROTE. The SIMT corpus in this tree is 40%
compiler-generated, and its own provenance record's limitations say a `hand` verdict "CANNOT
distinguish source a human typed from agent output a human reviewed... never call them 'expert'".
Measuring our compiler against our compiler's output and reporting agreement would be the most
flattering number in this repo and would mean nothing, so a corpus with no eligibility record admits
nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "AGREE",
    "DISAGREE",
    "UNDECIDABLE",
    "STATUSES",
    "AxisMeasurement",
    "CorpusStatus",
    "FieldVerdict",
    "Triple",
    "corpus_status",
    "field_verdicts",
    "measure_axis",
    "measure_register",
    "report_triple",
    "totals",
    "triple_of",
]

#: What became of one field the axis names.
AGREE = "AGREE"
DISAGREE = "DISAGREE"
UNDECIDABLE = "UNDECIDABLE"

#: The states this measurement can put an axis in. They are the register's own vocabulary; this module
#: never invents a fourth, and never returns EXERCISED for an axis whose fields it could not decide.
STATUSES = ("EXERCISED", "EXPRESSED", "UNMEASURED")


class ExpressivenessError(ValueError):
    """The measurement was asked something it cannot answer without guessing."""


@dataclass(frozen=True)
class Triple:
    """``expressed / denominator / unexercised`` -- the whole result of one measurement.

    Three numbers rather than two because two invite a ratio, and the ratio is where the fields that
    could not be decided go missing. ``expressed + unexercised == denominator`` always holds, and the
    constructor refuses a triple where it does not: an inconsistent triple is a number whose parts
    were computed at different times against different field sets, which is how a denominator shrinks
    without anybody choosing to shrink it.
    """

    expressed: int
    denominator: int
    unexercised: int

    def __post_init__(self) -> None:
        for name in ("expressed", "denominator", "unexercised"):
            if getattr(self, name) < 0:
                raise ExpressivenessError(f"Triple.{name} is negative")
        if self.expressed + self.unexercised != self.denominator:
            raise ExpressivenessError(
                f"triple {self.text()} does not close: {self.expressed} + {self.unexercised} != "
                f"{self.denominator}. Every field the axis names belongs to the denominator whatever "
                "happened to it; a triple that does not close means some of them were dropped."
            )

    def text(self) -> str:
        return f"{self.expressed}/{self.denominator}/{self.unexercised}"

    def bounds(self) -> tuple[float, float]:
        """``(lower, upper)`` -- what the evidence supports if every undecidable went the worst, then
        the best, way. An empty denominator is ``(0.0, 1.0)``: nothing measured bounds nothing."""
        if not self.denominator:
            return (0.0, 1.0)
        return (self.expressed / self.denominator, (self.expressed + self.unexercised) / self.denominator)

    def __add__(self, other: "Triple") -> "Triple":
        return Triple(
            self.expressed + other.expressed,
            self.denominator + other.denominator,
            self.unexercised + other.unexercised,
        )


@dataclass(frozen=True)
class FieldVerdict:
    """One CCA axis, what became of it, and why -- the why in words a reader can act on."""

    axis: str
    state: str
    reason: str = ""
    expert: Any = None
    ours: Any = None


def _facet_value(cca, axis: str) -> tuple[Any, bool]:
    """``(value, the_facet_exists)`` for ``"facet.field"`` on ``cca``. Split structurally, no regex."""
    facet_name, _, field_name = str(axis).partition(".")
    if not field_name:
        raise ExpressivenessError(f"{axis!r} is not a 'facet.field' axis name")
    facet = getattr(cca, facet_name, None) if cca is not None else None
    if facet is None:
        return None, False
    return getattr(facet, field_name, None), True


def field_verdicts(expert, ours, axes: Sequence[str]) -> tuple[FieldVerdict, ...]:
    """What the comparator made of each ``axes`` entry, for one expert CCA and one of ours.

    The comparator is ASKED which fields it compared rather than predicted: what it looks at is a
    literal inside it, and a measurement that duplicated that literal would keep reporting the old
    answer after the literal changed.
    """
    from merlin.kernels.cca import cca_agree

    report = cca_agree(expert, ours)
    compared = set(report.compared_fields)

    out: list[FieldVerdict] = []
    for axis in axes:
        ve, have_e = _facet_value(expert, axis)
        vo, have_o = _facet_value(ours, axis)
        if axis in compared:
            state = AGREE if ve == vo else DISAGREE
            reason = "" if state == AGREE else "the corpus and the schedule report different values"
            out.append(FieldVerdict(axis, state, reason, ve, vo))
            continue
        if ve is None and vo is None:
            reason = (
                "neither the corpus nor the schedule determines it"
                if have_e and have_o
                else "neither side carries the facet this axis names"
            )
        elif ve is None:
            reason = "only the schedule determines it; the corpus does not, so the comparator skips it"
        elif vo is None:
            reason = "only the corpus determines it; the schedule does not, so the comparator skips it"
        else:
            reason = (
                "both sides determine it and the comparator still did not compare it -- its facet is "
                "outside what cca_agree iterates, so an agreement report says nothing about this axis"
            )
        out.append(FieldVerdict(axis, UNDECIDABLE, reason, ve, vo))
    return tuple(out)


def triple_of(verdicts: Iterable[FieldVerdict]) -> Triple:
    """The triple these verdicts add up to. Only AGREE is expressed; everything else is charged."""
    seq = list(verdicts)
    agreed = sum(1 for v in seq if v.state == AGREE)
    return Triple(expressed=agreed, denominator=len(seq), unexercised=len(seq) - agreed)


def report_triple(expert, ours) -> tuple[Triple, tuple[FieldVerdict, ...]]:
    """The measurement over EVERY axis either CCA touches, not only the ones a register row names.

    The denominator is the comparator's own ``compared_fields`` plus every axis one side populated and
    it did not compare -- the union, so widening the corpus can only widen the denominator. A row-level
    triple answers "is this capability exercised"; this one answers "how much of these two descriptions
    was comparable at all", which is the number that says whether the first one means anything.
    """
    from merlin.kernels.cca import cca_agree
    from merlin.kernels.cca_compare import uncomparable_axes

    compared = list(cca_agree(expert, ours).compared_fields)
    extra = [a for a, _ in uncomparable_axes(expert, ours) if "." in a]
    axes = list(dict.fromkeys([*compared, *extra]))
    verdicts = field_verdicts(expert, ours, axes)
    return triple_of(verdicts), verdicts


# ---- corpus resolution -------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusStatus:
    """Whether a declared corpus can stand as expert evidence in THIS checkout, and on what bytes."""

    name: str
    resolved: bool
    missing_input: str = ""
    root: Path | None = None
    pin: str = ""
    declared_commit: str = ""
    observed_commit: str = ""
    #: Ways the checkout differs from what the pin declares. Reported, never silently tolerated: a
    #: result attributed to the wrong revision is worse than no result, because it gets cited.
    drift: tuple[str, ...] = ()
    #: Kernels found, and how many of them an eligibility audit admits as expert-authored.
    found: int = 0
    admitted: int = 0
    #: Why the admitted count is what it is, in one line.
    eligibility: str = ""

    @property
    def usable(self) -> bool:
        """Resolvable AND carrying at least one kernel an audit admits. Drift does not clear this --
        drift changes WHICH bytes the result is about, and the result records that."""
        return self.resolved and self.admitted > 0


def _pin(name: str):
    from merlin.common import provenance

    try:
        return provenance.pin(name)
    except Exception:  # noqa: BLE001 -- an unreadable or absent pin is a missing input, not a crash
        return None


def _count(root: Path, unit: str) -> int:
    if unit == "directory":
        return sum(1 for p in root.iterdir() if p.is_dir())
    return sum(1 for p in root.rglob(f"*{unit}") if p.is_file())


def _eligibility(spec: dict, root: Path, unit: str, root_env: str) -> tuple[int, str]:
    """How many of the corpus' kernels an audit admits as expert-authored, and why that number.

    A corpus with no audit admits NOTHING. That is the conservative direction on purpose: "we did not
    check who wrote these" and "experts wrote these" are different statements, and only the second is
    the claim an expressiveness measurement would be making by using them.

    THE COUNT IS SCOPED AND DEDUPED, which is not fussiness. An audit may classify several checkouts
    in one file and name the same kernel in each; counting the rows gave 114 admitted kernels for a
    corpus holding 99 directories -- a number larger than the corpus it describes, i.e. a numerator
    that outran its own denominator. Rows are therefore scoped to THIS checkout's root variable when
    the record names one, deduplicated by kernel name, and intersected with what is on disk.
    """
    from merlin.targetgen.corpora import sched_corpus_record

    admits = {str(a) for a in (spec.get("admits") or [])}
    if not admits:
        return 0, (
            "it declares no `provenance_record` in merlin/contract/corpora.yaml, so nothing in it is "
            "admitted as expert-authored"
        )
    record = sched_corpus_record(spec)
    if record is None or not record.is_file():
        return 0, f"the declared provenance record is not present at {record}"

    from merlin.common.yaml import load_yaml

    doc = load_yaml(record) or {}
    rows = [r for r in (doc.get("kernels") or []) if isinstance(r, dict)]
    if not rows:
        return 0, f"the provenance record {record.name} classifies no kernel"

    # Scope to this checkout. The record spells a repository as its root VARIABLE, so the match is on
    # the variable name rather than on a resolved path (two variables can point at one directory).
    want = "${" + root_env + "}" if root_env else ""
    scoped = [r for r in rows if str(r.get("repo") or "") == want] if want else []
    scope_note = f"scoped to {want}" if scoped else "unscoped (the record names no matching checkout)"
    considered = scoped or rows

    present = {p.name for p in root.iterdir() if p.is_dir()} if unit == "directory" else set()
    names = {
        str(r.get("name") or "")
        for r in considered
        if str(r.get("verdict") or "") in admits and (not present or str(r.get("name") or "") in present)
    }
    names.discard("")
    limits = [str(x) for x in (doc.get("limitations") or [])]
    note = (
        f"{len(names)} of {len(considered)} classified kernels carry an admitted verdict "
        f"{sorted(admits)} and are on disk, {scope_note}"
    )
    if limits:
        note += f"; the record's own first limitation: {limits[0].split('.')[0].strip()}"
    return len(names), note


def corpus_status(name: str) -> CorpusStatus:
    """Resolve corpus ``name`` through the registry, the pin it declares and its eligibility audit.

    Every failure names the INPUT that would fix it -- an environment variable, a path, a record --
    because a measurement that reports "0" for a corpus nobody could find is indistinguishable from
    one that measured it and found nothing.
    """
    from merlin.targetgen.corpora import sched_corpora

    spec = sched_corpora().get(name)
    if spec is None:
        return CorpusStatus(
            name=name,
            resolved=False,
            missing_input=f"no corpus named {name!r} in merlin/contract/corpora.yaml (sched_corpora)",
        )

    pin_name = str(spec.get("pin") or "")
    pin = _pin(pin_name) if pin_name else None
    if pin is None:
        return CorpusStatus(
            name=name,
            resolved=False,
            pin=pin_name,
            missing_input=f"hardware pin {pin_name!r}, which merlin/contract/hardware_pins.yaml does not resolve",
        )

    checkout = pin.checkout()
    if checkout is None:
        return CorpusStatus(
            name=name,
            resolved=False,
            pin=pin_name,
            declared_commit=pin.commit,
            missing_input=f"${pin.root_env} (unset in the environment and in <repo>/.env)",
        )

    root = checkout / str(spec.get("kernels") or "")
    if not root.is_dir():
        return CorpusStatus(
            name=name,
            resolved=False,
            pin=pin_name,
            root=root,
            declared_commit=pin.commit,
            missing_input=f"{root} (${pin.root_env} resolves, but the corpus subtree is not there)",
        )

    from merlin.common import provenance

    try:
        verified = provenance.verify(pin_name)
        drift = tuple(verified.drift)
        observed = verified.observed.commit or ""
    except Exception as exc:  # noqa: BLE001 -- an unverifiable pin is drift we must report, not hide
        drift, observed = (f"the pin could not be verified ({exc})",), ""

    unit = str(spec.get("unit") or "directory")
    found = _count(root, unit)
    admitted, why = _eligibility(dict(spec), root, unit, pin.root_env)
    return CorpusStatus(
        name=name,
        resolved=True,
        root=root,
        pin=pin_name,
        declared_commit=pin.commit,
        observed_commit=observed,
        drift=drift,
        found=found,
        admitted=admitted,
        eligibility=why,
    )


# ---- the register-level measurement ------------------------------------------------------------


@dataclass(frozen=True)
class AxisMeasurement:
    """One register row, measured. ``status`` is this module's verdict, not the row's claim."""

    axis_id: str
    status: str
    triple: Triple
    verdicts: tuple[FieldVerdict, ...] = ()
    corpus: str = ""
    missing_input: str = ""
    measured_on: str = ""
    notes: tuple[str, ...] = ()

    def text(self) -> str:
        return f"{self.status:11s} {self.triple.text():>10s}  {self.axis_id}"


def measure_axis(row: dict, *, expert=None, ours=None, corpus: CorpusStatus | None = None) -> AxisMeasurement:
    """Measure one register row.

    EXERCISED requires all three things the register's header asks for and nothing less: the corpus
    resolves AND is admissible, every field the row names was comparable, and every one of them
    agreed. A row whose fields were comparable and disagreed is EXPRESSED, not EXERCISED -- the axis
    is in the vocabulary, the schedule just does not do what the corpus does. A row whose corpus could
    not be read is UNMEASURED with the input named, and its fields are still counted, so the
    denominator does not depend on whether the measurement succeeded.
    """
    axis_id = str(row.get("id") or "")
    axes = tuple((row.get("evidence") or {}).get("cca_axes") or ())
    corpus_name = str((row.get("evidence") or {}).get("corpus") or "")

    if not corpus_name:
        # No corpus declared: the row makes no agreement claim, so there is nothing to measure and
        # nothing to charge. The gate leaves such a row EXPRESSED on its primitive + test alone.
        return AxisMeasurement(axis_id, "EXPRESSED", Triple(0, len(axes), len(axes)), corpus="")

    status = corpus if corpus is not None else corpus_status(corpus_name)
    if not status.usable:
        missing = status.missing_input or (f"an expert-authored kernel in {status.root} -- {status.eligibility}")
        return AxisMeasurement(
            axis_id,
            "UNMEASURED",
            Triple(0, len(axes), len(axes)),
            verdicts=tuple(
                FieldVerdict(a, UNDECIDABLE, f"the corpus this row names is unavailable: {missing}") for a in axes
            ),
            corpus=corpus_name,
            missing_input=missing,
        )

    if expert is None or ours is None:
        missing = (
            f"a lifter that turns {corpus_name} into a CCA: the corpus resolves at {status.root} "
            f"({status.admitted} admissible kernels) but nothing in this checkout lifts it"
        )
        return AxisMeasurement(
            axis_id,
            "UNMEASURED",
            Triple(0, len(axes), len(axes)),
            verdicts=tuple(FieldVerdict(a, UNDECIDABLE, missing) for a in axes),
            corpus=corpus_name,
            missing_input=missing,
            measured_on=_stamp(status),
        )

    verdicts = field_verdicts(expert, ours, axes)
    triple = triple_of(verdicts)
    exercised = bool(axes) and triple.expressed == triple.denominator
    return AxisMeasurement(
        axis_id,
        "EXERCISED" if exercised else "EXPRESSED",
        triple,
        verdicts=verdicts,
        corpus=corpus_name,
        measured_on=_stamp(status),
        notes=tuple(status.drift),
    )


def _stamp(status: CorpusStatus) -> str:
    """Which bytes a result is about, as one citable string. Never the declared commit alone: a
    checkout that has moved off its pin still produces numbers, and they belong to what is there."""
    if not status.resolved:
        return ""
    seen = status.observed_commit[:12] or "unknown"
    if status.drift:
        return f"{status.name}@{seen} (OFF-PIN; {status.pin} declares {status.declared_commit[:12]})"
    return f"{status.name}@{seen} (pin {status.pin})"


def measure_register(rows: Sequence[dict], *, sides: dict | None = None) -> tuple[AxisMeasurement, ...]:
    """Measure every row. ``sides`` maps a corpus name to ``(expert_cca, ours_cca)`` where a lifter
    exists; a corpus absent from it measures as UNMEASURED naming the lifter as the missing input.

    Corpora are resolved ONCE and shared across the rows that name them, so a report cannot show one
    row's corpus as present and another's as absent within the same run.
    """
    sides = dict(sides or {})
    cache: dict[str, CorpusStatus] = {}
    out: list[AxisMeasurement] = []
    for row in rows:
        name = str((row.get("evidence") or {}).get("corpus") or "")
        status = None
        if name:
            status = cache.get(name) or cache.setdefault(name, corpus_status(name))
        expert, ours = sides.get(name, (None, None))
        out.append(measure_axis(dict(row), expert=expert, ours=ours, corpus=status))
    return tuple(out)


def totals(measurements: Iterable[AxisMeasurement]) -> Triple:
    """The triple over every measured row. Rows that could not be measured still contribute their
    fields, which is the whole point: a run that resolved nothing reports 0/N/N, not 0/0/0."""
    seq = list(measurements)
    if not seq:
        return Triple(0, 0, 0)
    out = Triple(0, 0, 0)
    for m in seq:
        out = out + m.triple
    return out
