"""Recover the timing evidence that functional runs already produce -- and refuse to over-claim it.

Every capsule this tree grades executes on a real oracle. Those runs cost hours of simulator time and
they emit a cycle count, and until now the cycle count was thrown away the moment the functional
verdict was written. This module recovers it, from artifacts already on disk as well as from live
runs, and hands it to the performance layer under three rules that are enforced here rather than
remembered.

WHY A HARVESTED LATENCY IS NOT A LATENCY
----------------------------------------
A number read out of a real program is **contended**. The unit was not alone: other engines were
moving data, the sequencer was arbitrating, and whatever the number is, the isolated cost of the
operation is no larger. So a harvested value is an **upper bound with a validity domain naming what
else was active**, and this module refuses to spell it any other way:

* its provenance kind is forced to :data:`HARVEST_KIND` (``trace_derived``) -- weaker than
  ``measured`` and weaker than ``calibrated``, so :func:`merlin.analysis.perf.contract.terms.combine_kinds` drags every
  composite built on it down to the same confidence;
* the occurrence **spread** travels with it (a value observed once at 1090 and once at 8889 is not a
  latency, it is a distribution, and the corpus contains exactly that case);
* :func:`promote` -- the only door from harvested to calibrated -- always raises. Calibration needs a
  dedicated experiment in which the thing being priced is the only thing running. There is no code
  path that turns a trace observation into a constant.

WHAT MAY CONTRIBUTE
-------------------
Only substrates the target's :class:`~merlin.analysis.perf.measurement.authority.MeasurementAuthority` declares
citable at the tier the number was reached at. A target that declares nothing contributes nothing:
an undeclared authority is a policy statement, not a licence to fall back to whichever substrate
happened to emit a ``cycles`` field. Measured on the corpus on disk this is not a formality -- on one
target it drops 929 functional-tier observations and keeps 360 RTL ones, and the functional tier is
the one that overcounts by ~3x.

AND WHAT IS NEVER FILLED IN
---------------------------
Harvest never turns an UNKNOWN into a number. Where a series is empty, or a pairing cannot be
established, or a program leaves the law's validated regime, the output is an UNKNOWN term with the
reason and a recorded refusal -- never a zero, never a silently dropped input.
:data:`merlin.analysis.perf.contract.terms.UNKNOWN` refuses ``__bool__`` and ``__float__``, and nothing here works
around that.

THE RULE REGISTRY
-----------------
The second half of this module reads ``src/merlin/resources/contracts/perf_rules/*.yaml``: *detected trait ->
required model term -> optimization family -> experiment family*, as data. The registry is data
because a rule expressed as code is a rule that gets a target baked into it; YAML also keeps the
name gate's scan roots clean.

A rule emits an experiment only when the evidence can actually carry it: **at least two observation
points per fitted parameter, and at least as many distinct x-levels as parameters**. That single
gate is what makes the registry's output honest rather than aspirational -- a rate fitted through one
point cannot separate a per-unit cost from a fixed overhead, and the rules whose axis the corpus
cannot observe come back as refusals naming what to buy.

Nothing here names a target, a unit, an opcode or a geometry. Bucket-to-family pairings are derived
from the evidence's own support sets and regression behaviour, and refuse when they are not unique.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from merlin.analysis.perf.contract.profile import TargetProfile
from merlin.analysis.perf.contract.record import (
    SUITE_SCHEMA,
    SuiteSchema,
    compose_unit_busy,
    derive_delay_mnemonic,
    derive_unit_roles,
    fill_cycles,
)
from merlin.analysis.perf.contract.terms import UNKNOWN, Bounds, PerformanceTerm, Provenance, Validity
from merlin.analysis.perf.measurement.authority import MeasurementAuthority, authority_for, citable
from merlin.analysis.perf.measurement.observations import TIMING_OBSERVATIONS_KEY as _TIMING_OBSERVATIONS_KEY
from merlin.common.paths import artifacts_dir, contract_dir, runs_dir
from merlin.common.record_formats import load_yaml

# ---- harvest -----------------------------------------------------------------------------------

HARVEST_KIND = "trace_derived"


_FIDELITY_TIER = {
    "elaborated_rtl": "rtl",
    "rtl_derived_model": "cycle_model",
    "functional_model": "functional",
}


_SCHEMA_VERSION = 1


class ContendedTermError(ValueError):
    """A harvested (contended) term was asked to become a calibrated constant."""


@dataclass(frozen=True)
class Refusal:
    """A source that could have contributed and did not, and why. Never silent, never a zero."""

    what: str
    reason: str
    where: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"what": self.what, "reason": self.reason, "where": self.where}


@dataclass(frozen=True)
class Observation:
    """One timing number recovered from a run that was executed for some other reason.

    ``submission`` is load-bearing and is never the workload name alone. Cycles are a property of the
    SUBMISSION: the same capsule has measured 1090 / 3078 / 8889 across three submissions of the same
    task, an 8.2x spread on identical inputs. Keying a series by workload name alone silently pools
    three different programs into one "latency".

    ``concurrent`` is what else was active while the number was taken. It becomes the term's validity
    domain, and it is the difference between "this operation costs N" and "this operation cost no
    more than N while the movement engine was also running".
    """

    submission: str
    workload: str
    stage: str  # the source's own stage/tier label, verbatim
    substrate: str  # what produced the number, in the producer's own words
    tier: str  # the shared measurement ladder tier this stage reached
    quantity: str
    value: float
    unit: str
    #: The VERDICT the stage reached. Kept because a cycle count from a stage that FAILED is timing
    #: evidence about a program that did not compute the declared operation -- worth harvesting (it
    #: is exactly the diagnostic the failing capsules never had) and never poolable with the passing
    #: ones. A measured example: one workload carries 2 cycles from a failing stage beside 6349 and
    #: 19928 from two passing submissions, and a series that pools them reports a 9964x "spread".
    status: str = ""
    concurrent: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission": self.submission,
            "workload": self.workload,
            "stage": self.stage,
            "substrate": self.substrate,
            "tier": self.tier,
            "quantity": self.quantity,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "concurrent": list(self.concurrent),
            "evidence": list(self.evidence),
        }


@dataclass
class Harvest:
    """Everything recovered for one target, plus everything that was refused."""

    target: str
    authority: MeasurementAuthority
    observations: tuple[Observation, ...] = ()
    refusals: tuple[Refusal, ...] = ()
    roots: tuple[str, ...] = ()

    def series(self, quantity: str, *, status: str | None = None) -> tuple[Observation, ...]:
        """Observations of one quantity, optionally restricted to one verdict.

        The filter is explicit rather than default: dropping the failing stages silently would hide
        the only cycle evidence the undiagnosable capsules have, and pooling them silently would
        report a failed program's cost as the workload's.
        """
        return tuple(o for o in self.observations if o.quantity == quantity and (status is None or o.status == status))

    def quantities(self) -> tuple[str, ...]:
        return tuple(sorted({o.quantity for o in self.observations}))

    def submissions(self) -> tuple[str, ...]:
        return tuple(sorted({o.submission for o in self.observations}))

    def by_workload(self, quantity: str, *, status: str | None = None) -> dict[str, tuple[Observation, ...]]:
        out: dict[str, list[Observation]] = {}
        for o in self.series(quantity, status=status):
            out.setdefault(o.workload, []).append(o)
        return {k: tuple(v) for k, v in sorted(out.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "target": self.target,
            "authority": self.authority.to_dict(),
            "roots": list(self.roots),
            "n_observations": len(self.observations),
            "submissions": list(self.submissions()),
            "quantities": list(self.quantities()),
            "observations": [o.to_dict() for o in self.observations],
            "refusals": [r.to_dict() for r in self.refusals],
        }


def spread(values: Sequence[float]) -> dict[str, Any]:
    """The occurrence spread of a harvested series. Empty in, empty out -- never a fabricated point.

    ``ratio`` is max/min, which is the number that says whether a "latency" is a latency at all: the
    corpus contains a workload whose ratio across submissions is 8.2.
    """
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0}
    lo, hi = min(vals), max(vals)
    return {"n": len(vals), "min": lo, "max": hi, "median": float(median(vals)), "ratio": (hi / lo if lo > 0 else None)}


def assert_contended(term: PerformanceTerm) -> PerformanceTerm:
    """Raise unless ``term`` is spelled as the contended observation it is."""
    if term.provenance.kind != HARVEST_KIND:
        raise ContendedTermError(
            f"harvested term {term.name!r} carries provenance kind {term.provenance.kind!r}. A value "
            f"read out of a running program is contended and may only be {HARVEST_KIND!r}: other "
            "engines were active, so it bounds the isolated cost from above and does not state it."
        )
    return term


def promote(term: PerformanceTerm, *, experiment: str = "") -> PerformanceTerm:
    """The door from harvested to calibrated. It is always shut.

    Kept as a named function so the refusal is discoverable at the place someone would reach for it,
    with the reason attached, instead of being a convention in a docstring.
    """
    raise ContendedTermError(
        f"refusing to promote {term.name!r} from {HARVEST_KIND!r} to a calibrated constant"
        + (f" on the strength of {experiment!r}" if experiment else "")
        + ". A harvested value is an upper bound taken while other units were running; calibration "
        "requires a dedicated experiment in which the priced thing is the only thing running, and "
        "no amount of trace evidence substitutes for one. Use the term as a prior, an uncertainty "
        "and a ranking input -- which is what it is good for."
    )


def harvested_term(
    name: str, observations: "Sequence[Observation]", *, unit: str, regime: str, escalate: str = ""
) -> PerformanceTerm:
    """One contended term from a series of occurrences, or UNKNOWN when the series is empty.

    The VALUE is the median occurrence -- a prior, not a constant -- and the BOUNDS are the observed
    minimum and maximum, so the spread cannot be read off without also reading the value. An empty
    series yields UNKNOWN with the reason, never a zero.
    """
    stats = spread([o.value for o in observations])
    statuses = sorted({o.status for o in observations if o.status})
    if len(statuses) > 1:
        raise ValueError(
            f"term {name!r} would pool observations with different verdicts {statuses}. A cycle "
            "count from a stage that did not pass is evidence about a program that did not compute "
            "the declared operation; it is worth keeping and it is not the same series. Filter with "
            "Harvest.series(..., status=...) and say which verdict the term is about."
        )
    concurrent = sorted({c for o in observations for c in o.concurrent})
    subs = sorted({o.submission for o in observations})
    evidence = tuple(sorted({e for o in observations for e in o.evidence})) or ("no citable observation was recovered",)
    prov = Provenance(kind=HARVEST_KIND, evidence=evidence)
    weak = (
        "observed inside running programs that were also doing: "
        + ("; ".join(concurrent) if concurrent else "nothing else this instrument can report")
        + ". Every occurrence is therefore an UPPER BOUND on the isolated cost"
    )
    validity = Validity(
        validated_regime=regime,
        expected_error=(
            f"observed {stats['n']} time(s) across {len(subs)} submission(s) at verdict "
            f"{statuses[0] if statuses else 'unstated'}; "
            f"spread {stats.get('min')}..{stats.get('max')}"
            if stats["n"]
            else "no occurrence"
        ),
        weak_regime=weak,
        escalate_when=(
            escalate
            or "a different submission, or any change to what else runs "
            "alongside; cycles are a property of the submission, not of the "
            "workload name"
        ),
    )
    if not stats["n"]:
        return assert_contended(
            PerformanceTerm.unknown(
                name,
                unit,
                prov,
                validity,
                "no citable observation was recovered for this quantity; the substrates that reported "
                "it are not the ones this target declares citable at the required tier, or nothing "
                "reported it at all. Absent, not zero",
            )
        )
    return assert_contended(
        PerformanceTerm(
            name=name,
            value=stats["median"],
            unit=unit,
            provenance=prov,
            validity=validity,
            bounds=Bounds(lower=stats["min"], upper=stats["max"]),
        )
    )


def _tier_of_record(record: Any) -> tuple[str, str]:
    """``(measurement-ladder tier, substrate)`` for one tier record. Fails closed to ``""``.

    A tier record in the bare-string form reports no fields, so it reaches no tier. A record that
    declares its own fidelity is believed over anything the tier NAME suggests; failing that, the two
    provenance flags the runner already carries place it, and a record carrying neither reaches no
    tier at all rather than a default one.
    """
    if not isinstance(record, Mapping):
        return "", ""
    substrate = str(record.get("evidence") or "") or str(record.get("fidelity") or "")
    fidelity = record.get("fidelity")
    if fidelity is not None:
        return _FIDELITY_TIER.get(str(fidelity), ""), substrate
    if record.get("derived_from_rtl"):
        return ("rtl" if record.get("cycle_accurate") else "cycle_model"), substrate
    if record.get("cycles") is not None:
        return "functional", substrate
    return "", substrate


def _submission_of(path: Path) -> str:
    """Which SUBMISSION a per-capsule artifact belongs to, derived from where it sits.

    The grade harness lays a submission out as ``<submission>/runs/<suite>/<capsule>/``, so the
    submission is the parent of the nearest enclosing ``runs`` directory. Derived structurally, and
    falling back to the capsule's grandparent rather than to the capsule NAME -- pooling two
    submissions of one capsule into a single series is the failure this field exists to prevent.
    """
    for parent in path.parents:
        if parent.name == "runs" and parent.parent is not None:
            return str(parent.parent)
    return str(path.parents[1]) if len(path.parents) > 1 else str(path.parent)


TIMING_OBSERVATIONS_KEY = _TIMING_OBSERVATIONS_KEY


def harvest_capsule_result(
    path: "str | Path", *, authority: MeasurementAuthority, submission: str | None = None
) -> tuple[list[Observation], list[Refusal]]:
    """Per-tier cycles -- and any finer per-unit timing the oracle chose to carry -- from one
    ``capsule_result.json``.

    The richest source on disk: every tier that ran records its own count next to what the oracle
    said it was. A tier that reports no count contributes nothing -- an adapter with no timing
    capability emits nothing, never a zero -- and a tier that carries a
    :data:`TIMING_OBSERVATIONS_KEY` block has each of its entries harvested under the same
    citability gate as the cycle count beside it.
    """
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 -- a real answer
        return [], [Refusal("capsule result", f"{type(exc).__name__}: {exc}", str(p))]
    workload = str(doc.get("capsule") or p.parent.name)
    sub = submission or _submission_of(p)
    shas = doc.get("toolchain_shas") or {}
    obs: list[Observation] = []
    refusals: list[Refusal] = []
    for stage, record in sorted((doc.get("tiers") or {}).items()):
        cycles = record.get("cycles") if isinstance(record, Mapping) else None
        if cycles is None:
            continue
        tier, substrate = _tier_of_record(record)
        if not tier:
            refusals.append(
                Refusal(
                    f"{workload}@{stage}",
                    "the tier record states neither a "
                    "fidelity nor a provenance flag, so the number reaches no tier "
                    "on the measurement ladder and cannot be placed",
                    str(p),
                )
            )
            continue
        if not citable(authority, tier):
            refusals.append(
                Refusal(
                    f"{workload}@{stage}",
                    f"reached tier {tier!r}; {authority.target} declares {authority.citable_tier!r} "
                    f"citable" + ("" if authority.declared else " (nothing is declared at all)"),
                    str(p),
                )
            )
            continue
        obs.append(
            Observation(
                submission=sub,
                workload=workload,
                stage=stage,
                substrate=substrate or "unnamed oracle",
                tier=tier,
                quantity="total_cycles",
                value=float(cycles),
                unit="cycles",
                status=str(record.get("status") or ""),
                concurrent=(
                    "the whole program: this substrate reports one number per run and no "
                    "per-unit decomposition, so every other unit the program uses was active "
                    "inside it",
                ),
                evidence=tuple(sorted({str(p)} | {f"{k}={v}" for k, v in shas.items()})),
            )
        )
        obs.extend(
            _fine_grained(
                record,
                path=p,
                workload=workload,
                stage=stage,
                tier=tier,
                substrate=substrate,
                submission=sub,
                shas=shas,
            )
        )
    return obs, refusals


def _fine_grained(
    record: Mapping[str, Any],
    *,
    path: Path,
    workload: str,
    stage: str,
    tier: str,
    substrate: str,
    submission: str,
    shas: Mapping[str, Any],
) -> list[Observation]:
    """The optional per-unit timing block, if this oracle carries one. Absent is not zero."""
    out: list[Observation] = []
    for entry in record.get(TIMING_OBSERVATIONS_KEY) or ():
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("value")
        quantity = entry.get("quantity")
        if value is None or not quantity:
            continue  # not reported: skipped, never recorded as a zero
        note = str(entry.get("note") or "")
        out.append(
            Observation(
                submission=submission,
                workload=workload,
                stage=stage,
                substrate=substrate or "unnamed oracle",
                tier=tier,
                quantity=str(quantity),
                value=float(value),
                unit=str(entry.get("unit") or "cycles"),
                status=str(record.get("status") or ""),
                concurrent=tuple(str(c) for c in (entry.get("concurrent") or ()))
                or ("the same program's other units, which this oracle does not enumerate",),
                evidence=tuple(
                    sorted(
                        {str(path), f"{TIMING_OBSERVATIONS_KEY}[{quantity}]"}
                        | ({note} if note else set())
                        | {f"{k}={v}" for k, v in shas.items()}
                    )
                ),
            )
        )
    return out


def harvest_score_file(
    path: "str | Path", *, authority: MeasurementAuthority
) -> tuple[list[Observation], list[Refusal]]:
    """Cycles from a graded ``score_capsule.json``'s ``cycles_diagnostic`` block.

    The block has two shapes on disk. The current one keys each capsule's counts BY TIER, which
    carries the provenance the ladder tier needs; the older one is a bare ``{capsule: cycles}`` and
    carries none. The older shape is refused rather than guessed at: attributing a count to a tier it
    does not name is exactly how a functional-model number gets quoted as a hardware result.
    """
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [], [Refusal("score file", f"{type(exc).__name__}: {exc}", str(p))]
    block = doc.get("cycles_diagnostic") or {}
    sub = str(doc.get("package") or _submission_of(p))
    obs: list[Observation] = []
    refusals: list[Refusal] = []
    for workload, entry in sorted(block.items()):
        if not isinstance(entry, Mapping):
            refusals.append(
                Refusal(
                    f"{workload} (score summary)",
                    "the summary records a bare cycle count with no tier, so which oracle produced it "
                    "is not recoverable from this file; the per-capsule results carry the tier and are "
                    "harvested instead",
                    str(p),
                )
            )
            continue
        for stage, cycles in sorted(entry.items()):
            if cycles is None:
                continue
            # A summary states the count and the tier NAME and nothing about the oracle, so the
            # ladder tier is not derivable here either -- but the per-capsule result beside it is.
            refusals.append(
                Refusal(
                    f"{workload}@{stage} (score summary)",
                    "the summary states a tier name but not the oracle's fidelity; the per-capsule "
                    "result carries both and is the harvested source",
                    str(p),
                )
            )
    return obs, refusals


def discover_roots(target: str) -> tuple[Path, ...]:
    """Where this target's already-executed graded runs live, by the layout convention.

    The target is a FOLDER COMPONENT under both roots (``out/runs/<target>/`` and
    ``out/artifacts/capsule-bench/<target>/``), so this composes a path from a parameter rather than
    knowing anything about any particular target.
    """
    return tuple(d for d in (runs_dir() / target, artifacts_dir() / "capsule-bench" / target) if d.is_dir())


def retro_mine(
    roots: "Sequence[str | Path] | None" = None,
    *,
    target: str,
    authority: MeasurementAuthority | None = None,
    descriptor: Mapping[str, Any] | None = None,
) -> Harvest:
    """Walk runs already on disk and recover every citable timing observation they left behind.

    This is the point of the module: the runs were paid for, they executed on real oracles, and their
    timing was discarded at the moment the functional verdict was written.
    """
    auth = (
        authority
        if authority is not None
        else authority_for(target, dict(descriptor) if descriptor is not None else None)
    )
    search = [Path(r) for r in (roots if roots is not None else discover_roots(target))]
    obs: list[Observation] = []
    refusals: list[Refusal] = []
    if not search:
        refusals.append(
            Refusal(
                f"{target} run roots",
                "no graded-run root exists on this host; nothing was executed here to harvest",
                "",
            )
        )
    for root in search:
        if not root.is_dir():
            refusals.append(Refusal(str(root), "not a directory on this host", str(root)))
            continue
        for p in sorted(root.rglob("capsule_result.json")):
            o, r = harvest_capsule_result(p, authority=auth)
            obs.extend(o)
            refusals.extend(r)
        for p in sorted(root.rglob("score_capsule*.json")):
            o, r = harvest_score_file(p, authority=auth)
            obs.extend(o)
            refusals.extend(r)
    if not auth.declared:
        refusals.append(
            Refusal(
                f"{target} measurement authority", "; ".join(auth.gaps()) or "nothing declared", "capability manifest"
            )
        )
    return Harvest(
        target=target,
        authority=auth,
        observations=tuple(obs),
        refusals=tuple(refusals),
        roots=tuple(str(r) for r in search),
    )


def harvest_op_stream(
    suite: Mapping[str, Any],
    *,
    target: str,
    authority: MeasurementAuthority | None = None,
    schema: SuiteSchema = SUITE_SCHEMA,
    submission: str = "",
) -> Harvest:
    """The scheduled inter-op delays a measured corpus's programs carry.

    Each op stream is a sequence of ``(family, mnemonic, immediate)``; exactly one mnemonic carries
    the delay immediate and that mnemonic is DERIVED from the corpus, never assumed
    (:func:`merlin.analysis.perf.contract.record.derive_delay_mnemonic`). The delay following an op is what the corpus
    scheduled behind it -- an observation about a program, not about a unit, and contended for the
    same reason as everything else here.
    """
    auth = authority if authority is not None else authority_for(target)
    kernels: Mapping[str, Any] = suite.get(schema.kernels_key) or {}
    streams = [k[schema.op_stream_key] for k in kernels.values() if k.get(schema.op_stream_key)]
    refusals: list[Refusal] = []
    if not streams:
        return Harvest(
            target=target, authority=auth, refusals=(Refusal("op streams", "the suite carries none", submission),)
        )
    try:
        delay_mnemonic = derive_delay_mnemonic(streams)
    except ValueError as exc:
        return Harvest(target=target, authority=auth, refusals=(Refusal("delay marker", str(exc), submission),))
    tier = auth.cycles_tier if auth.declared else ""
    if not tier or not citable(auth, tier):
        refusals.append(
            Refusal(
                "scheduled delays",
                f"the corpus's cycle tier {tier or 'UNKNOWN'!r} is not citable for "
                f"{target} (declared {auth.citable_tier!r})",
                submission,
            )
        )
        return Harvest(target=target, authority=auth, refusals=tuple(refusals))
    obs: list[Observation] = []
    for name in sorted(kernels):
        stream = kernels[name].get(schema.op_stream_key) or []
        for i, entry in enumerate(stream):
            fam, mnemonic, _imm = str(entry[0]), str(entry[1]), entry[2]
            if mnemonic == delay_mnemonic:
                continue
            if i + 1 >= len(stream) or str(stream[i + 1][1]) != delay_mnemonic:
                continue
            gap = stream[i + 1][2]
            if not gap:
                continue  # the program schedules no delay here: a fact, not a term
            others = sorted({str(e[0]) for e in stream if str(e[1]) != delay_mnemonic} - {fam})
            obs.append(
                Observation(
                    submission=submission or "the pinned measured corpus",
                    workload=name,
                    stage=fam,
                    substrate="the corpus's own program schedule",
                    tier=tier,
                    quantity=f"scheduled_delay.{mnemonic}",
                    value=float(gap),
                    unit="cycles",
                    status="pass",
                    concurrent=(
                        f"the same program's {', '.join(others)} ops"
                        if others
                        else "no other op family in this program",
                    ),
                    evidence=(
                        f"{schema.op_stream_key}[{name}]",
                        f"delay marker {delay_mnemonic!r} derived from the corpus",
                    ),
                )
            )
    return Harvest(target=target, authority=auth, observations=tuple(obs), refusals=tuple(refusals))


@dataclass(frozen=True)
class Point:
    """One (x, y) observation on an axis, labelled by where it came from."""

    x: float
    y: float
    label: str


@dataclass(frozen=True)
class AxisEvidence:
    """Every point an axis carries, and every observation the axis deliberately dropped."""

    axis: str
    x_name: str
    y_name: str
    y_unit: str
    points: tuple[Point, ...] = ()
    excluded: tuple[Refusal, ...] = ()
    note: str = ""

    @property
    def distinct_x(self) -> int:
        return len({p.x for p in self.points})

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "x_name": self.x_name,
            "y_name": self.y_name,
            "y_unit": self.y_unit,
            "n_points": len(self.points),
            "distinct_x": self.distinct_x,
            "note": self.note,
            "points": [{"x": p.x, "y": p.y, "label": p.label} for p in self.points],
            "excluded": [r.to_dict() for r in self.excluded],
        }


@dataclass(frozen=True)
class Fit:
    """A fitted structural equation, or the reason there was not enough evidence to fit one."""

    form: str
    parameters: dict[str, float] = field(default_factory=dict)
    n_points: int = 0
    distinct_x: int = 0
    residuals: dict[str, Any] = field(default_factory=dict)
    refusal: str = ""

    @property
    def ok(self) -> bool:
        return not self.refusal

    def to_dict(self) -> dict[str, Any]:
        return {
            "form": self.form,
            "parameters": dict(self.parameters),
            "n_points": self.n_points,
            "distinct_x": self.distinct_x,
            "residuals": dict(self.residuals),
            "refusal": self.refusal,
        }


FIT_FORMS: dict[str, tuple[str, ...]] = {
    "proportional": ("ratio",),  # y = ratio * x        -- one parameter, no intercept
    "affine": ("slope", "intercept"),  # y = slope * x + b    -- a rate AND a fixed overhead
}


def _residuals(points: "Sequence[Point]", predict) -> dict[str, Any]:
    res = [p.y - predict(p.x) for p in points]
    mean = sum(p.y for p in points) / len(points)
    ss_tot = sum((p.y - mean) ** 2 for p in points)
    ss_res = sum(r * r for r in res)
    return {
        "min": min(res),
        "max": max(res),
        "median": float(median(res)),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
    }


def fit_points(evidence: AxisEvidence, form: str) -> Fit:
    """Fit ``form`` to an axis, refusing below **two points per fitted parameter**.

    A single rate cannot price a unit whose cost is a rate PLUS a fixed overhead: the one point is
    consistent with every split between them. The distinct-x requirement is the same rule seen from
    the other side -- repeats at one x measure noise, they do not add a degree of freedom.
    """
    params = FIT_FORMS.get(form)
    if params is None:
        return Fit(
            form=form,
            refusal=f"no fit form named {form!r}; implemented "
            f"{sorted(FIT_FORMS)}. A structural equation is not guessed",
        )
    n, k = len(evidence.points), len(params)
    need_points, need_x = 2 * k, k
    if n < need_points or evidence.distinct_x < need_x:
        return Fit(
            form=form,
            n_points=n,
            distinct_x=evidence.distinct_x,
            refusal=(
                f"{form} fits {k} parameter(s) {list(params)} and needs >={need_points} "
                f"points at >={need_x} distinct {evidence.x_name} level(s); the "
                f"evidence carries {n} point(s) at {evidence.distinct_x} level(s). "
                "Refusing to fit -- an under-determined fit reports a number that the "
                "evidence does not contain"
            ),
        )
    pts = evidence.points
    if form == "proportional":
        sxx = sum(p.x * p.x for p in pts)
        if sxx == 0:
            return Fit(
                form=form,
                n_points=n,
                distinct_x=evidence.distinct_x,
                refusal=f"every {evidence.x_name} is zero; the ratio is not determined",
            )
        ratio = sum(p.x * p.y for p in pts) / sxx
        return Fit(
            form=form,
            parameters={"ratio": ratio},
            n_points=n,
            distinct_x=evidence.distinct_x,
            residuals=_residuals(pts, lambda x: ratio * x),
        )
    sx = sum(p.x for p in pts)
    sy = sum(p.y for p in pts)
    sxx = sum(p.x * p.x for p in pts)
    sxy = sum(p.x * p.y for p in pts)
    den = n * sxx - sx * sx
    if den == 0:
        return Fit(
            form=form,
            n_points=n,
            distinct_x=evidence.distinct_x,
            refusal=f"the {evidence.x_name} values are degenerate; the slope is not determined",
        )
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    return Fit(
        form=form,
        parameters={"slope": slope, "intercept": intercept},
        n_points=n,
        distinct_x=evidence.distinct_x,
        residuals=_residuals(pts, lambda x: slope * x + intercept),
    )


def invert_fill_law(law: str, fill: float, *, max_dimension: int = 4096) -> tuple[int | None, str]:
    """The structural dimension whose ``law`` fill is ``fill``, found by search over the law itself.

    The law is never re-expressed here in inverted form -- it is evaluated forwards until it matches,
    so a law added to :mod:`merlin.analysis.perf.contract.record` is invertible the moment it exists and no second,
    drifting copy of the algebra is created. Returns ``(None, reason)`` when nothing matches, which
    is a real answer: a measured fill that no dimension produces refutes the law for this unit.
    """
    target = round(float(fill))
    if abs(float(fill) - target) > 1e-6:
        return None, (
            f"the fitted fill {fill} is not an integer number of cycles, so no integer "
            f"dimension produces it under {law!r}"
        )
    for d in range(1, int(max_dimension) + 1):
        if fill_cycles(law, d) == target:
            return d, f"{law}({d}) == {target}"
    return None, (
        f"no dimension in 1..{max_dimension} gives a {law!r} fill of {target}; the law "
        "does not describe this unit, which is a finding rather than a fitting problem"
    )


def _support(values: Mapping[str, float]) -> frozenset[str]:
    return frozenset(k for k, v in values.items() if v)


def _pair_movement(
    kernels: Mapping[str, Any], buckets: "Sequence[str]", schema: SuiteSchema, policy: Mapping[str, Any]
) -> tuple[str | None, str, dict[str, float]]:
    """Which activity bucket IS the movement engine, decided by behaviour rather than by name.

    A movement engine's busy cycles are an affine function of the beat counters the same instrument
    reports; a compute unit's are not. So each bucket is regressed against total beats and the one
    that the beats explain is the movement one -- provided it explains it well AND no other bucket
    also looks linear, because two candidates mean the question was not answered.
    """
    min_r2 = float(policy.get("min_r2", 0.99))
    max_runner_up = float(policy.get("max_runner_up_r2", 0.9))
    scores: dict[str, float] = {}
    for bucket in buckets:
        pts = [
            Point(
                float(e[schema.activity_key][schema.read_beats_key] + e[schema.activity_key][schema.write_beats_key]),
                float(e[schema.activity_key][bucket]),
                name,
            )
            for name, e in kernels.items()
            if bucket in e[schema.activity_key]
        ]
        fit = fit_points(
            AxisEvidence(axis="movement_probe", x_name="beats", y_name=bucket, y_unit="cycles", points=tuple(pts)),
            "affine",
        )
        r2 = fit.residuals.get("r2") if fit.ok else None
        scores[bucket] = float(r2) if r2 is not None else float("-inf")
    if not scores:
        return None, "the activity block carries no per-unit buckets", scores
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_r2 = ranked[0]
    if best_r2 < min_r2:
        return (
            None,
            (
                f"no bucket's busy cycles are explained by the beat counters (best {best} at "
                f"r2={best_r2:.4f} < {min_r2}); this instrument does not identify a movement "
                "engine, so its terms stay unharvested"
            ),
            scores,
        )
    if len(ranked) > 1 and ranked[1][1] > max_runner_up:
        return (
            None,
            (
                f"two buckets look linear in the beat counters ({best} r2={best_r2:.4f}, "
                f"{ranked[1][0]} r2={ranked[1][1]:.4f} > {max_runner_up}); the pairing is "
                "ambiguous and is refused rather than picked"
            ),
            scores,
        )
    return best, f"{best} busy is affine in the beat counters (r2={best_r2:.6f})", scores


def _pair_compute(
    kernels: Mapping[str, Any], buckets: "Sequence[str]", schema: SuiteSchema, delay_mnemonic: str
) -> tuple[dict[str, str], list[Refusal]]:
    """Pair each remaining activity bucket with the op-stream family that drives it, by SUPPORT.

    A bucket is driven by a family when the bucket is busy in exactly the programs that issue that
    family's ops, and in no others. The pairing must be unique in both directions; where it is not,
    nothing is paired and the reason is recorded. Refusing here is what stops a unit's terms from
    being attributed to the wrong op stream -- and it does fire on real corpora, where a program
    issues a family's ops and the instrument still reports the bucket idle.
    """
    bucket_support = {b: _support({n: e[schema.activity_key].get(b, 0) for n, e in kernels.items()}) for b in buckets}
    fam_support: dict[str, set[str]] = {}
    for name, entry in kernels.items():
        for op in entry.get(schema.op_stream_key) or []:
            # The delay marker is a SCHEDULING pseudo-op, not an issued engine op. Counting it into
            # its family's support makes that family look like it is driven by whichever unit the
            # program happens to delay behind, which is how the scheduler gets mistaken for a unit.
            if str(op[1]) == delay_mnemonic:
                continue
            fam_support.setdefault(str(op[0]), set()).add(name)
    families = {f: frozenset(s) for f, s in fam_support.items()}
    paired: dict[str, str] = {}
    refusals: list[Refusal] = []
    for bucket, support in sorted(bucket_support.items()):
        if not support:
            refusals.append(
                Refusal(
                    bucket,
                    "the bucket is idle in every program in the corpus, so no op family's support can be matched to it",
                )
            )
            continue
        matches = sorted(f for f, s in families.items() if s == support)
        if len(matches) != 1:
            refusals.append(
                Refusal(
                    bucket,
                    f"{len(matches)} op family(ies) have exactly this bucket's support "
                    f"{sorted(support)[:3]}...; a bucket that is busy in programs that issue no "
                    "candidate family's ops (or in fewer of them) is not attributable, and "
                    "guessing the pairing attributes a unit's cycles to the wrong op stream",
                )
            )
            continue
        paired[bucket] = matches[0]
    for family, support in families.items():
        owners = [b for b, f in paired.items() if f == family]
        if len(owners) > 1:
            for b in owners:
                paired.pop(b, None)
            refusals.append(
                Refusal(family, f"family drives {len(owners)} buckets {owners} by support; the pairing is not unique")
            )
    return paired, refusals


__all__ = [
    "AxisEvidence",
    "ContendedTermError",
    "Deferral",
    "Experiment",
    "Fit",
    "HARVEST_KIND",
    "Harvest",
    "Observation",
    "Point",
    "Recovery",
    "Refusal",
    "Rule",
    "RuleRegistry",
    "TIMING_OBSERVATIONS_KEY",
    "assert_contended",
    "axes_from_suite",
    "detected_traits",
    "discover_roots",
    "fit_points",
    "harvest_capsule_result",
    "harvest_op_stream",
    "harvest_score_file",
    "harvested_term",
    "invert_fill_law",
    "load_registry",
    "main",
    "promote",
    "retro_mine",
    "spread",
]
