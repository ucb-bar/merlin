"""Which of two emitted candidates is better, decided by a NAMED instrument or not at all.

WHY THIS EXISTS OUTSIDE THE CAMPAIGN DRIVER. The phase-2 global driver is an authoring and
structural-validation loop: ``global_speedup_proven`` is written 22 times and set ``True`` zero
times, ``global_cost_validated`` 9 times and never ``True`` (asserted ``False`` at 3 further sites),
``full_model_simulation_allowed`` 10 times and never ``True`` (asserted at 7). *"It proved a global
speedup"* is not a state that code has, and ``probe_relevance`` says so in its own words -- *"this
function never promotes a local sample into a global performance verdict."* So a comparison verdict
cannot come from inside it, and forcing one there would mean inverting sixteen assertion sites.

THE TWO WAYS A COMPARISON HAS ALREADY LIED HERE, both of which this module refuses by name:

**Identical emitted bytes.** The last campaign's own receipt recorded
``command_buffer_identical: True`` and ``lowered_identical: True`` beside a performance hypothesis:
the candidate emitted the same program as its baseline. Any metric difference between two identical
programs is noise or instrument drift, and :func:`compare` reports ``IDENTICAL_EMISSION`` rather
than a ratio.

**A metric that cannot see the change.** A 1.642x improvement was measured on host instruction count
across arms whose traffic was byte-identical (169.52 MiB on every one). The number was real and it
answered a question nobody asked, because the axis under optimization was data movement. So a
verdict names the axes that MOVED, and an axis that did not move cannot carry the verdict.

WHAT IT WILL NEVER DO. It does not certify from a band. Two bands that overlap are ``UNKNOWN`` --
not a tie, not a small win -- because ``rate_table.holdout_containment``'s own licence is *"a band
may eliminate a candidate and may never certify one"*. And it does not rank on a proxy whose
degradation correlates with success: Spike prices every accelerator command at one cycle, so it
systematically flatters offloading (measured: a retarget that nearly doubled mesh instructions moved
the Spike metric 0.2%), which is why :class:`Metric` carries ``blind_to``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# ---- candidate_decision ------------------------------------------------------------------------

#: The axes a compiler change can move, declared so a verdict can say which one it is about. An
#: instrument blind to the axis under change cannot decide, however precise it is on its own axis.
AXES: tuple[str, ...] = (
    "emitted_program",  # the bytes themselves: opcodes, operands, attributes
    "offload",  # how much work reaches the unit at all
    "traffic",  # bytes moved
    "host_work",  # scalar-lane instructions or cycles
    "accelerator_time",  # cycles the unit was busy
    "wall_cycles",  # the whole measured window
)


@dataclass(frozen=True)
class Axis:
    """One axis, and whether the two candidates actually differ along it."""

    name: str
    moved: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if self.name not in AXES:
            raise ValueError(f"axis {self.name!r} is not one of {list(AXES)}")


@dataclass(frozen=True)
class Metric:
    """One instrument's reading of both arms, plus what it is blind to."""

    name: str
    instrument: str
    baseline: float
    candidate: float
    unit: str = ""
    lower_is_better: bool = True
    #: Axes this instrument cannot see. A metric blind to every axis that moved is REPORTED and
    #: excluded from the verdict rather than dropped: "the instrument could not see the change" and
    #: "the change did nothing" are different findings and only this field separates them.
    blind_to: tuple[str, ...] = ()
    #: A band reading is an INTERVAL, not a point. Supplied as (lo, hi) per arm when the instrument
    #: is a band; a verdict is then only reachable when the two intervals are disjoint.
    baseline_interval: tuple[float, float] | None = None
    candidate_interval: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for axis in self.blind_to:
            if axis not in AXES:
                raise ValueError(f"blind_to axis {axis!r} is not one of {list(AXES)}")

    @property
    def ratio(self) -> float | None:
        if self.baseline <= 0 or self.candidate <= 0:
            return None
        return (self.baseline / self.candidate) if self.lower_is_better else (self.candidate / self.baseline)

    @property
    def is_band(self) -> bool:
        return self.baseline_interval is not None and self.candidate_interval is not None

    def disjoint(self) -> bool | None:
        """For a band metric: are the two intervals disjoint? None when it is not a band."""
        if not self.is_band:
            return None
        b_lo, b_hi = self.baseline_interval  # type: ignore[misc]
        c_lo, c_hi = self.candidate_interval  # type: ignore[misc]
        return c_hi < b_lo or b_hi < c_lo

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instrument": self.instrument,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "unit": self.unit,
            "ratio": self.ratio,
            "lower_is_better": self.lower_is_better,
            "blind_to": list(self.blind_to),
            "is_band": self.is_band,
            "disjoint": self.disjoint(),
        }


@dataclass(frozen=True)
class CandidateFacts:
    """What is known about one arm's emitted program, independent of any comparison."""

    name: str
    #: Digest of the emitted command buffer. Two arms sharing it emitted the SAME program.
    command_buffer_sha256: str = ""
    lowered_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command_buffer_sha256": self.command_buffer_sha256,
            "lowered_sha256": self.lowered_sha256,
        }


#: The verdicts. Each names a DIFFERENT reason a comparison ended where it did.
BETTER = "BETTER"
WORSE = "WORSE"
NO_EFFECT = "NO_EFFECT"
UNKNOWN = "UNKNOWN"
IDENTICAL_EMISSION = "IDENTICAL_EMISSION"
BLIND = "BLIND"

VERDICTS = (BETTER, WORSE, NO_EFFECT, UNKNOWN, IDENTICAL_EMISSION, BLIND)


@dataclass
class Decision:
    """The verdict, the instrument that reached it, and everything that did not."""

    verdict: str
    why: str
    decided_by: str = ""
    axes: tuple[Axis, ...] = ()
    metrics: tuple[Metric, ...] = ()
    excluded: list[dict[str, Any]] = field(default_factory=list)

    @property
    def moved_axes(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.axes if a.moved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "merlin_candidate_decision_v1",
            "verdict": self.verdict,
            "why": self.why,
            "decided_by": self.decided_by,
            "moved_axes": list(self.moved_axes),
            "axes": [{"name": a.name, "moved": a.moved, "detail": a.detail} for a in self.axes],
            "metrics": [m.to_dict() for m in self.metrics],
            "excluded": list(self.excluded),
            "licence": (
                "a band may eliminate a candidate and may never certify one; an "
                "instrument blind to every axis that moved cannot decide, however "
                "precise it is on its own axis"
            ),
        }


def compare(
    baseline: CandidateFacts, candidate: CandidateFacts, *, axes: Sequence[Axis], metrics: Sequence[Metric]
) -> Decision:
    """Decide between two arms, or refuse and name what stopped the decision.

    The order of checks is the order in which a comparison goes wrong in practice, and each one
    returns a DIFFERENT verdict rather than collapsing into "no improvement".
    """
    axes = tuple(axes)
    metrics = tuple(metrics)
    seen = [a.name for a in axes]
    if len(set(seen)) != len(seen):
        raise ValueError(f"an axis is declared twice: {sorted({a for a in seen if seen.count(a) > 1})}")

    # 1. THE SAME PROGRAM. Checked first because every later number is then meaningless, and this
    #    is exactly what the last campaign shipped a performance hypothesis on top of.
    same_buffer = (
        baseline.command_buffer_sha256
        and candidate.command_buffer_sha256
        and baseline.command_buffer_sha256 == candidate.command_buffer_sha256
    )
    same_lowered = (
        baseline.lowered_sha256 and candidate.lowered_sha256 and baseline.lowered_sha256 == candidate.lowered_sha256
    )
    if same_buffer or same_lowered:
        which = " and ".join(n for n, y in (("command buffer", same_buffer), ("lowered module", same_lowered)) if y)
        return Decision(
            verdict=IDENTICAL_EMISSION,
            why=(
                f"the two arms share their {which} digest, so they are the same emitted program; "
                f"any metric difference between them is noise or instrument drift, not an effect"
            ),
            axes=axes,
            metrics=metrics,
        )

    moved = {a.name for a in axes if a.moved}
    if not moved:
        return Decision(
            verdict=NO_EFFECT,
            why=(
                "the arms differ in their emitted bytes but no declared axis moved, so there is "
                "nothing for a metric to be about"
            ),
            axes=axes,
            metrics=metrics,
        )

    # 2. AN INSTRUMENT THAT CANNOT SEE THE CHANGE. A metric blind to every moved axis is excluded
    #    with its reason -- this is the 1.642x-on-identical-traffic failure, made unrepeatable.
    usable: list[Metric] = []
    excluded: list[dict[str, Any]] = []
    for metric in metrics:
        blind = moved & set(metric.blind_to)
        if blind and blind == moved:
            excluded.append(
                {
                    "metric": metric.name,
                    "instrument": metric.instrument,
                    "reason": (
                        f"blind to every axis that moved ({sorted(moved)}), so its reading "
                        f"cannot be about this change however precise it is"
                    ),
                    "ratio": metric.ratio,
                }
            )
            continue
        if blind:
            excluded.append(
                {
                    "metric": metric.name,
                    "instrument": metric.instrument,
                    "reason": (
                        f"partially blind: cannot see {sorted(blind)} of the moved axes "
                        f"{sorted(moved)}; kept, but it under-reports this change"
                    ),
                    "ratio": metric.ratio,
                    "kept": True,
                }
            )
        usable.append(metric)

    if not usable:
        return Decision(
            verdict=BLIND,
            why=(
                f"axes {sorted(moved)} moved and every supplied instrument is blind to all of "
                f"them; this is an unmeasured change, NOT an ineffective one"
            ),
            axes=axes,
            metrics=metrics,
            excluded=excluded,
        )

    # 3. BANDS ELIMINATE, NEVER CERTIFY. An overlapping pair is UNKNOWN by licence.
    points = [m for m in usable if not m.is_band]
    bands = [m for m in usable if m.is_band]
    for band in bands:
        if not band.disjoint():
            excluded.append(
                {
                    "metric": band.name,
                    "instrument": band.instrument,
                    "reason": (
                        "the two bands overlap, and an overlapping pair is UNKNOWN rather than a "
                        "tie or a small win: a band may eliminate a candidate and may never "
                        "certify one"
                    ),
                    "baseline_interval": list(band.baseline_interval or ()),
                    "candidate_interval": list(band.candidate_interval or ()),
                }
            )

    decisive_bands = [b for b in bands if b.disjoint()]
    if not points and not decisive_bands:
        return Decision(
            verdict=UNKNOWN,
            why=("no point measurement was supplied and every band overlaps, so nothing separates the two arms"),
            axes=axes,
            metrics=metrics,
            excluded=excluded,
        )

    # 4. THE VERDICT. Prefer a point measurement; a disjoint band can only ELIMINATE, so it decides
    #    only WORSE (the candidate's whole interval sits above the baseline's).
    for band in decisive_bands:
        b_lo, b_hi = band.baseline_interval  # type: ignore[misc]
        c_lo, c_hi = band.candidate_interval  # type: ignore[misc]
        if c_lo > b_hi:
            return Decision(
                verdict=WORSE,
                decided_by=band.instrument,
                why=(
                    f"{band.name}: the candidate's entire band [{c_lo:g}, {c_hi:g}] lies above the "
                    f"baseline's [{b_lo:g}, {b_hi:g}], which eliminates it"
                ),
                axes=axes,
                metrics=metrics,
                excluded=excluded,
            )

    if not points:
        # A disjoint band favouring the candidate is NOT a win: certification is outside the licence.
        favouring = decisive_bands[0]
        excluded.append(
            {
                "metric": favouring.name,
                "instrument": favouring.instrument,
                "reason": (
                    "the bands are disjoint in the candidate's favour, which is an elimination "
                    "of the BASELINE and not a certification of the candidate; a band may never "
                    "certify one"
                ),
            }
        )
        return Decision(
            verdict=UNKNOWN,
            why=(
                "the only separating evidence is a band favouring the candidate, and a band may "
                "not certify a candidate -- only eliminate one"
            ),
            axes=axes,
            metrics=metrics,
            excluded=excluded,
        )

    ratios = [(m, m.ratio) for m in points if m.ratio is not None]
    if not ratios:
        return Decision(
            verdict=UNKNOWN,
            why="no point metric yielded a usable ratio (a non-positive reading has none)",
            axes=axes,
            metrics=metrics,
            excluded=excluded,
        )
    # The WEAKEST improving ratio decides, so a single flattering instrument cannot carry a verdict
    # that another usable one contradicts.
    worst, worst_ratio = min(ratios, key=lambda pair: pair[1])
    best, best_ratio = max(ratios, key=lambda pair: pair[1])
    if worst_ratio < 1.0 and best_ratio > 1.0:
        return Decision(
            verdict=UNKNOWN,
            decided_by="",
            why=(
                f"the instruments disagree: {best.name} reports {best_ratio:.3f}x better while "
                f"{worst.name} reports {worst_ratio:.3f}x. A verdict from either alone would be "
                f"the one that happened to be quoted"
            ),
            axes=axes,
            metrics=metrics,
            excluded=excluded,
        )
    if worst_ratio > 1.0:
        return Decision(
            verdict=BETTER,
            decided_by=worst.instrument,
            why=(
                f"every usable instrument improves; the weakest is {worst.name} at "
                f"{worst_ratio:.3f}x, and it is quoted rather than the best so a single flattering "
                f"reading cannot carry the verdict"
            ),
            axes=axes,
            metrics=metrics,
            excluded=excluded,
        )
    if best_ratio < 1.0:
        return Decision(
            verdict=WORSE,
            decided_by=best.instrument,
            why=(f"every usable instrument regresses; the mildest is {best.name} at {best_ratio:.3f}x"),
            axes=axes,
            metrics=metrics,
            excluded=excluded,
        )
    return Decision(
        verdict=NO_EFFECT,
        decided_by=worst.instrument,
        why=(
            "the axes moved and every usable instrument read the same on both arms, so the change "
            "is measured and ineffective -- which is a result, and is not the same as unmeasured"
        ),
        axes=axes,
        metrics=metrics,
        excluded=excluded,
    )


# ---- candidate_comparison ----------------------------------------------------------------------
# Give :mod:`merlin.analysis.optimization.candidates.decision` the inputs phase 2 actually has.
#
# WHY THIS EXISTS. ``candidate_decision.compare`` is the named instrument for "which of these two
# arms is better", and until this module it had **zero production callers**: the phase-2 loop emitted
# a baseline and a candidate, priced both, wrote ``command_buffer_identical`` and ``lowered_identical``
# beside the numbers -- and then had nothing that turned any of it into a verdict. The campaign driver
# cannot supply one (``global_speedup_proven`` / ``global_cost_validated`` /
# ``full_model_simulation_allowed`` are written dozens of times and set ``True`` at no site, and that
# is by design), so the verdict has to be derived from the evidence the host already owns. That
# derivation is this module; the decision itself stays in ``candidate_decision``, unchanged.
#
# WHAT IT DERIVES AND WHAT IT REFUSES TO DERIVE. An :class:`~.candidate_decision.Axis` is declared
# only when the host evidence can actually say whether the two arms differ along it. An axis nothing
# here can observe is NOT declared as ``moved=False`` -- "it did not move" and "nothing could see
# whether it moved" are different claims, and collapsing them is how an unmeasured change reads as an
# ineffective one. Undeclarable axes are returned in ``undeclared_axes`` with the reason, so the
# refusal is visible rather than absent.
#
# ``accelerator_time`` and ``wall_cycles`` are therefore NEVER declared from an emission pair: that
# analysis compiles a graph and does not execute it. Which of two legal orderings is faster is
# decidable only by measurement -- the phase-2 command-buffer analyzer carries the held-out evidence
# for that refusal (no command-buffer-readable ordering signal beat chance), and a verdict from here
# would contradict it. A verdict from this module is about the axes it NAMES, and about no others.
#
# INSTRUMENT BLIND SPOTS ARE DECLARED, NOT ASSUMED. Each metric carries the axes it cannot see, which
# is the whole mechanism that stops the measured 1.642x-on-byte-identical-traffic reading from being
# quoted as a data-movement win. A functional ISA simulator that prices every accelerator command at
# one cycle is blind to how much work reached the unit and to how long the unit was busy; callers
# pass :data:`FUNCTIONAL_SIMULATOR_BLIND_AXES` for such an engine so ``compare`` excludes it instead
# of ranking on it.

EMISSION_DECISION_SCHEMA = "candidate_emission_decision_v1"
MEASURED_DECISION_SCHEMA = "candidate_measured_decision_v1"

#: The two arms, named as the emission analysis names them.
BASELINE_ARM = "baseline"
CANDIDATE_ARM = "candidate"

#: Declared movement bytes read from a command buffer. It counts payload, so it cannot see host
#: scalar-lane work, how long the unit was busy, or the elapsed window.
MOVEMENT_BLIND_AXES: tuple[str, ...] = ("host_work", "accelerator_time", "wall_cycles")

#: Static machine instruction sites of the compiled lowered artifact. This is the instrument that
#: read 1.642x better across arms whose traffic was byte-identical: a static site count cannot see
#: how many bytes a loop moves, nor any dynamic time.
MACHINE_SITE_BLIND_AXES: tuple[str, ...] = ("traffic", "accelerator_time", "wall_cycles")

#: For a functional ISA simulator that prices every accelerator command at one cycle. Its
#: degradation correlates with success -- it systematically flatters offloading (measured: a
#: retarget that nearly doubled mesh instructions moved it 0.2%) -- so it may not rank an offload
#: or accelerator-time change, however precise its own reading is. ``wall_cycles`` is in the list
#: for the same measured reason: such an engine's cycle count PLATEAUS with workload size rather
#: than tracking the elapsed window, so its "cycles" are not that window and cannot order it.
FUNCTIONAL_SIMULATOR_BLIND_AXES: tuple[str, ...] = ("offload", "accelerator_time", "wall_cycles")

#: The issued-instruction classes the emitted-artifact lifter counts. A difference in any of them
#: means a different amount of work reached the unit.
ISSUED_FIELDS: tuple[str, ...] = (
    "movement_instructions",
    "compute_instructions",
    "configuration_instructions",
    "loop_descriptor_instructions",
    "synchronization_instructions",
    "dma_instructions",
)

_NOT_EXECUTED = (
    "this analysis emits and compares programs without executing either, so nothing here "
    "observes it; the measured path is what carries a timing verdict"
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> float | None:
    """A real number, or None. A bool is not a reading and neither is a non-finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _digest(value: Any) -> str:
    return value if isinstance(value, str) and value else ""


def _declared_movement(diagnostics: Mapping[str, Any], arm: str) -> tuple[float | None, str]:
    """Exact declared movement bytes for one arm, or None and why it is not available.

    A LOWER BOUND IS NOT A TOTAL. ``work_volume``/``movement_volume`` record a refusal for each
    command they cannot price, so ``is_lower_bound`` means "there is unpriced movement here";
    comparing two lower bounds would understate whichever arm has more unpriced commands, and it
    would do so in whichever direction happened to be flattering.
    """
    row = _mapping(_mapping(diagnostics.get("arms")).get(arm))
    if not row:
        return None, f"the {arm} arm carries no command-buffer analysis"
    if row.get("status") != "emitted":
        return None, f"the {arm} arm did not emit a command buffer (status {row.get('status')!r})"
    movement = _mapping(row.get("movement"))
    if movement.get("is_lower_bound") is not False:
        return None, f"the {arm} arm's declared movement is a lower bound, not a total"
    value = _finite(movement.get("exact_bytes"))
    if value is None:
        return None, f"the {arm} arm declares no exact movement byte total"
    return value, ""


def _machine_sites(diagnostics: Mapping[str, Any], emission: Mapping[str, Any], arm: str) -> tuple[float | None, str]:
    """Static machine instruction sites for one arm, bound to the artifact they were read from.

    The binding is the point: an audit whose ``source_sha256`` is not this arm's emitted artifact is
    a reading of some other program, and a stale one reads exactly like a fresh one.
    """
    machine = _mapping(_mapping(diagnostics.get("machine_artifact_activity")).get(arm))
    if not machine:
        return None, f"the {arm} arm carries no machine artifact audit"
    if machine.get("status") != "compiled":
        return None, f"the {arm} machine audit did not compile (status {machine.get('status')!r})"
    lowered = _digest(emission.get(f"{arm}_lowered_sha256"))
    if not lowered or machine.get("source_sha256") != lowered:
        return None, f"the {arm} machine audit is not bound to the artifact this analysis emitted"
    sites = _mapping(machine.get("instruction_sites"))
    # An older total excludes undecoded sites; two such totals are not commensurate with each other
    # or with a current one, so it is refused rather than silently compared.
    if sites.get("schema") != "encoded_instruction_sites_v1":
        return None, f"the {arm} machine audit reports an instruction-site total that is not commensurate"
    value = _finite(sites.get("total"))
    if value is None:
        return None, f"the {arm} machine audit reports no instruction-site total"
    return value, ""


def _issued(diagnostics: Mapping[str, Any], arm: str) -> tuple[Mapping[str, Any] | None, str]:
    activity = _mapping(_mapping(diagnostics.get("target_artifact_activity")).get(arm))
    if not activity:
        return None, f"the {arm} arm carries no emitted-artifact activity"
    if activity.get("status") != "decoded":
        return None, f"the {arm} emitted artifact was not decoded (status {activity.get('status')!r})"
    issued = _mapping(activity.get("issued"))
    if not issued:
        return None, f"the {arm} emitted artifact declares no issued instruction counts"
    return issued, ""


def _issued_delta(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[bool, str]:
    """Did any issued class differ, and which? Fields only one arm counts are NOT read as zero."""
    changed: list[str] = []
    unreadable: list[str] = []
    for field in ISSUED_FIELDS:
        left, right = _finite(baseline.get(field)), _finite(candidate.get(field))
        if left is None or right is None:
            unreadable.append(field)
            continue
        if left != right:
            changed.append(f"{field} {left:g} -> {right:g}")
    detail = "; ".join(changed) if changed else "every readable issued class is identical"
    if unreadable:
        detail += f" (unreadable on one or both arms: {', '.join(unreadable)})"
    return bool(changed), detail


def decide_emitted_pair(diagnostics: Mapping[str, Any], emission: Mapping[str, Any]) -> dict[str, Any]:
    """Decide between the two arms of one whole-model emission analysis, or say what stopped it.

    ``diagnostics`` is the host-owned analysis block (``arms``, ``target_artifact_activity``,
    ``machine_artifact_activity``); ``emission`` is the four digests plus the two identity flags the
    same analysis records. Nothing here reads a simulator, a golden or a holdout.
    """
    diagnostics = _mapping(diagnostics)
    emission = _mapping(emission)
    baseline = CandidateFacts(
        name="immutable_optimization_baseline",
        command_buffer_sha256=_digest(emission.get("baseline_command_buffer_sha256")),
        lowered_sha256=_digest(emission.get("baseline_lowered_sha256")),
    )
    candidate = CandidateFacts(
        name="live_phase2_candidate",
        command_buffer_sha256=_digest(emission.get("candidate_command_buffer_sha256")),
        lowered_sha256=_digest(emission.get("candidate_lowered_sha256")),
    )

    axes: list[Axis] = []
    metrics: list[Metric] = []
    undeclared: list[dict[str, str]] = []
    unavailable: list[dict[str, str]] = []

    # 1. THE EMITTED PROGRAM. Declarable whenever both arms recorded a digest of the same kind.
    pairs = (
        ("lowered module", baseline.lowered_sha256, candidate.lowered_sha256),
        ("command buffer", baseline.command_buffer_sha256, candidate.command_buffer_sha256),
    )
    comparable = [(label, left, right) for label, left, right in pairs if left and right]
    if not comparable:
        undeclared.append({"axis": "emitted_program", "reason": "neither arm recorded an emission digest to compare"})
    else:
        differing = [label for label, left, right in comparable if left != right]
        axes.append(
            Axis(
                "emitted_program",
                moved=bool(differing),
                detail=(
                    f"{' and '.join(differing)} digest(s) differ"
                    if differing
                    else "every recorded emission digest matches"
                ),
            )
        )

    # 2. TRAFFIC, and the instrument that reads it.
    base_bytes, base_why = _declared_movement(diagnostics, BASELINE_ARM)
    cand_bytes, cand_why = _declared_movement(diagnostics, CANDIDATE_ARM)
    if base_bytes is None or cand_bytes is None:
        reason = base_why or cand_why
        undeclared.append({"axis": "traffic", "reason": reason})
        unavailable.append(
            {"metric": "declared_movement_bytes", "instrument": "command_buffer_movement_volume", "reason": reason}
        )
    else:
        axes.append(
            Axis(
                "traffic",
                moved=base_bytes != cand_bytes,
                detail=f"declared movement {base_bytes:g} -> {cand_bytes:g} bytes",
            )
        )
        metrics.append(
            Metric(
                name="declared_movement_bytes",
                instrument="command_buffer_movement_volume",
                baseline=base_bytes,
                candidate=cand_bytes,
                unit="bytes",
                lower_is_better=True,
                blind_to=MOVEMENT_BLIND_AXES,
            )
        )

    # 3. HOST WORK, and the instrument that read 1.642x on byte-identical traffic.
    base_sites, base_site_why = _machine_sites(diagnostics, emission, BASELINE_ARM)
    cand_sites, cand_site_why = _machine_sites(diagnostics, emission, CANDIDATE_ARM)
    if base_sites is None or cand_sites is None:
        reason = base_site_why or cand_site_why
        undeclared.append({"axis": "host_work", "reason": reason})
        unavailable.append(
            {"metric": "machine_instruction_sites", "instrument": "machine_artifact_activity", "reason": reason}
        )
    else:
        axes.append(
            Axis(
                "host_work",
                moved=base_sites != cand_sites,
                detail=f"static machine instruction sites {base_sites:g} -> {cand_sites:g}",
            )
        )
        metrics.append(
            Metric(
                name="machine_instruction_sites",
                instrument="machine_artifact_activity",
                baseline=base_sites,
                candidate=cand_sites,
                unit="static instruction sites",
                lower_is_better=True,
                blind_to=MACHINE_SITE_BLIND_AXES,
            )
        )

    # 4. OFFLOAD. Observable as an axis; no instrument here prices it, because more work reaching
    #    the unit is neither better nor worse on its own.
    base_issued, base_issued_why = _issued(diagnostics, BASELINE_ARM)
    cand_issued, cand_issued_why = _issued(diagnostics, CANDIDATE_ARM)
    if base_issued is None or cand_issued is None:
        undeclared.append({"axis": "offload", "reason": base_issued_why or cand_issued_why})
    else:
        moved, detail = _issued_delta(base_issued, cand_issued)
        axes.append(Axis("offload", moved=moved, detail=detail))

    # 5. THE TWO TIMING AXES ARE NOT DECLARED HERE, EVER.
    for axis in ("accelerator_time", "wall_cycles"):
        undeclared.append({"axis": axis, "reason": _NOT_EXECUTED})

    scope = (
        "static emission evidence only -- no simulator, no golden, no holdout. A verdict here is "
        "about the axes it names and is never a cycle claim: which of two legal orderings is "
        "faster is decidable only by measurement."
    )
    if not axes:
        return {
            "schema": EMISSION_DECISION_SCHEMA,
            "verdict": UNKNOWN,
            "why": (
                "no axis could be derived from the host-owned emission evidence, so there is "
                "nothing a verdict could be about"
            ),
            "decided_by": "",
            "moved_axes": [],
            "scope": scope,
            "undeclared_axes": undeclared,
            "unavailable_instruments": unavailable,
            "decision": None,
        }

    decision = compare(baseline, candidate, axes=axes, metrics=metrics)
    return {
        "schema": EMISSION_DECISION_SCHEMA,
        "verdict": decision.verdict,
        "why": decision.why,
        "decided_by": decision.decided_by,
        "moved_axes": list(decision.moved_axes),
        "scope": scope,
        "undeclared_axes": undeclared,
        "unavailable_instruments": unavailable,
        "decision": decision.to_dict(),
    }


def decide_measured_totals(
    *,
    instrument: str,
    baseline_cycles: Any,
    candidate_cycles: Any,
    axis: str = "wall_cycles",
    blind_to: Sequence[str] = (),
    basis: str = "",
    baseline: CandidateFacts | None = None,
    candidate: CandidateFacts | None = None,
) -> dict[str, Any]:
    """Decide between two measured cycle totals with the same named instrument.

    The engine is the CALLER'S to declare, together with what it cannot see: pass
    :data:`FUNCTIONAL_SIMULATOR_BLIND_AXES` for an engine that prices every accelerator command at
    one cycle, and ``compare`` will exclude it from an offload or accelerator-time change rather
    than rank on it. An engine with no name is refused -- a number whose instrument is unrecorded
    cannot be attributed, and an unattributable result is not a weaker result, it is no result.
    """
    left, right = _finite(baseline_cycles), _finite(candidate_cycles)
    scope = basis or "measured cycle totals over one comparison cohort"
    if not str(instrument or "").strip():
        return {
            "schema": MEASURED_DECISION_SCHEMA,
            "verdict": UNKNOWN,
            "why": "the measurement names no engine, so its reading cannot be attributed to an instrument",
            "decided_by": "",
            "moved_axes": [],
            "scope": scope,
            "decision": None,
        }
    if left is None or right is None or left <= 0 or right <= 0:
        return {
            "schema": MEASURED_DECISION_SCHEMA,
            "verdict": UNKNOWN,
            "why": (
                "one or both arms carry no positive measured total, and an absent number is not a zero and not a tie"
            ),
            "decided_by": "",
            "moved_axes": [],
            "scope": scope,
            "decision": None,
        }
    base_facts = baseline or CandidateFacts(name="measured_baseline")
    cand_facts = candidate or CandidateFacts(name="measured_candidate")
    decision = compare(
        base_facts,
        cand_facts,
        axes=[Axis(axis, moved=left != right, detail=f"measured total {left:g} -> {right:g} cycles")],
        metrics=[
            Metric(
                name="measured_total_cycles",
                instrument=instrument,
                baseline=left,
                candidate=right,
                unit="cycles",
                lower_is_better=True,
                blind_to=tuple(blind_to),
            )
        ],
    )
    return {
        "schema": MEASURED_DECISION_SCHEMA,
        "verdict": decision.verdict,
        "why": decision.why,
        "decided_by": decision.decided_by,
        "moved_axes": list(decision.moved_axes),
        "scope": scope,
        "decision": decision.to_dict(),
    }


__all__ = [
    "AXES",
    "Axis",
    "BASELINE_ARM",
    "CANDIDATE_ARM",
    "CandidateFacts",
    "Decision",
    "EMISSION_DECISION_SCHEMA",
    "FUNCTIONAL_SIMULATOR_BLIND_AXES",
    "ISSUED_FIELDS",
    "MACHINE_SITE_BLIND_AXES",
    "MEASURED_DECISION_SCHEMA",
    "MOVEMENT_BLIND_AXES",
    "Metric",
    "compare",
    "decide_emitted_pair",
    "decide_measured_totals",
]
