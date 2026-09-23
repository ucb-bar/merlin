"""Give :mod:`merlin.perf.candidate_decision` the inputs phase 2 actually has.

WHY THIS EXISTS. ``candidate_decision.compare`` is the named instrument for "which of these two
arms is better", and until this module it had **zero production callers**: the phase-2 loop emitted
a baseline and a candidate, priced both, wrote ``command_buffer_identical`` and ``lowered_identical``
beside the numbers -- and then had nothing that turned any of it into a verdict. The campaign driver
cannot supply one (``global_speedup_proven`` / ``global_cost_validated`` /
``full_model_simulation_allowed`` are written dozens of times and set ``True`` at no site, and that
is by design), so the verdict has to be derived from the evidence the host already owns. That
derivation is this module; the decision itself stays in ``candidate_decision``, unchanged.

WHAT IT DERIVES AND WHAT IT REFUSES TO DERIVE. An :class:`~.candidate_decision.Axis` is declared
only when the host evidence can actually say whether the two arms differ along it. An axis nothing
here can observe is NOT declared as ``moved=False`` -- "it did not move" and "nothing could see
whether it moved" are different claims, and collapsing them is how an unmeasured change reads as an
ineffective one. Undeclarable axes are returned in ``undeclared_axes`` with the reason, so the
refusal is visible rather than absent.

``accelerator_time`` and ``wall_cycles`` are therefore NEVER declared from an emission pair: that
analysis compiles a graph and does not execute it. Which of two legal orderings is faster is
decidable only by measurement -- the phase-2 command-buffer analyzer carries the held-out evidence
for that refusal (no command-buffer-readable ordering signal beat chance), and a verdict from here
would contradict it. A verdict from this module is about the axes it NAMES, and about no others.

INSTRUMENT BLIND SPOTS ARE DECLARED, NOT ASSUMED. Each metric carries the axes it cannot see, which
is the whole mechanism that stops the measured 1.642x-on-byte-identical-traffic reading from being
quoted as a data-movement win. A functional ISA simulator that prices every accelerator command at
one cycle is blind to how much work reached the unit and to how long the unit was busy; callers
pass :data:`FUNCTIONAL_SIMULATOR_BLIND_AXES` for such an engine so ``compare`` excludes it instead
of ranking on it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .candidate_decision import UNKNOWN, Axis, CandidateFacts, Metric, compare

__all__ = [
    "BASELINE_ARM",
    "CANDIDATE_ARM",
    "EMISSION_DECISION_SCHEMA",
    "MEASURED_DECISION_SCHEMA",
    "FUNCTIONAL_SIMULATOR_BLIND_AXES",
    "MOVEMENT_BLIND_AXES",
    "MACHINE_SITE_BLIND_AXES",
    "ISSUED_FIELDS",
    "decide_emitted_pair",
    "decide_measured_totals",
]

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
