"""Score a candidate against a MEASURED reference point, so the loop has a destination.

An authoring loop that sees only its own last revision can tell whether it moved, never whether it
moved far enough or in the right direction. Measured: a campaign optimizing one collapsed host
operation count sealed a round at -2 operations -- an estimated +0.000% of the window -- while the
program it was optimizing sat far from a number the same compiler family had already reached by
hand. The number was known. It was simply not in the loop.

This module exposes the reference as a GAP: per-field structural distance plus an estimated cycle
distance. Two properties are deliberate.

**The gap is signed and named.** `mesh_regions: -12` means twelve regions the reference puts on the
matrix unit are somewhere else, and it says which field to look at. A scalar score would say only
"worse".

**A structural match is never reported as a cycle match.** Reaching the reference's command and
region census means the candidate emits the same SHAPE of program, which is evidence and not a
measurement -- the reference's own regression floor shares a byte-identical command buffer with it
and still ran 29% slower, because the spread lived in host-lane codegen. So `cycles` carries
`status: estimated` and the structural verdict is reported separately from it.

Target-neutral by construction: every fact comes from the ledger, which is data, and the design is
a parameter that must MATCH. Cycles are not comparable across designs (the same capsule reads 510
on one and 317 on another), so a mismatched design is refused rather than approximated.

**And cycles are not comparable across PROGRAMS either.** That refusal is the exact twin of the
design one and it was missing until 2026-09-19. Three programs in this project carry whole-model
numbers for one network on one device -- a static-calibrated recapture entry, the capsule's own
lowering of the same network, and a standalone C program with its parameters linked in -- and a
ratio between two of them was quoted for weeks as a compiler result. It is not one: no compiler
edit moves a candidate between identities, because the identity is a property of the lowering and
of the capture it came from. So :func:`structural_gap` and :func:`find_reference` REFUSE across
program identities, and :func:`estimate_cycles` returns a refusal record rather than raising,
because its result is embedded in a loop whose ``try/except`` would swallow a raise into silence.
See :mod:`merlin.perf.program_identity`.

**`design` is no longer sufficient to name a device.** Two registered bitstreams elaborate the same
configuration string onto the same board and are different devices. Every measured ledger entry now
states ``device`` -- the registry artifact name, which is unique per device -- and ``find_reference``
takes it as the stronger key. It is optional only so existing callers keep working, and a lookup
that settles for ``design`` says so in the record rather than silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import merlin_dir

from .program_identity import (
    UNKNOWN_IDENTITY,
    ProgramIdentity,
    ProgramIdentityError,
    comparability,
    identity_of_reference,
    require_comparable,
)

__all__ = [
    "REFERENCE_LEDGER_NAME",
    "ProgramIdentityError",
    "ReferenceError",
    "load_ledger",
    "load_references",
    "load_program_identities",
    "find_reference",
    "find_reference_for_capsule",
    "structural_gap",
    "estimate_cycles",
    "score_against_reference",
]

#: The tracked ledger of measured reference points, relative to the merlin package directory.
REFERENCE_LEDGER_NAME = "contract/perf_reference_targets.yaml"

#: Structural fields compared against a reference, in report order. Each is a count the emitted
#: command buffer declares, so the comparison needs no simulator.
_STRUCTURAL_FIELDS = ("commands", "mesh_regions", "host_lane_regions", "weight_prepack_recipes", "kernel_abi_args")


class ReferenceError(ValueError):
    """A reference could not be resolved, or was resolved against a design it does not describe."""


def load_ledger(path: Path | None = None) -> dict[str, Any]:
    """The whole ledger document -- ``references`` AND the ``program_identities`` roster.

    Separate from :func:`load_references` because the roster is not a reference and a caller that
    wants to resolve a candidate's identity needs it without pretending it is one.
    """
    ledger = Path(path) if path is not None else merlin_dir() / REFERENCE_LEDGER_NAME
    document = yaml.safe_load(ledger.read_text())
    if not isinstance(document, Mapping) or not isinstance(document.get("references"), Mapping):
        raise ReferenceError(f"{ledger} declares no `references` mapping")
    return dict(document)


def load_references(path: Path | None = None) -> dict[str, Any]:
    """The reference ledger, as data."""
    return dict(load_ledger(path)["references"])


def load_program_identities(path: Path | None = None) -> dict[str, Any]:
    """The declared program-identity roster, as data. Empty when the ledger declares none."""
    from .program_identity import declared_identities

    return declared_identities(load_ledger(path))


def find_reference(
    model: str,
    design: str,
    *,
    references: Mapping[str, Any] | None = None,
    status: str = "achieved",
    program_identity: str | None = None,
    device: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """The best measured reference for ``model`` on ``design`` -- the FEWEST cycles achieved.

    Refuses on a design mismatch instead of falling back to another design's number, because a
    cycle count from a different design is not a weaker answer, it is a wrong one. ``device`` and
    ``program_identity`` narrow the same way and refuse the same way:

    * ``device`` is the STRONGER form of ``design``. Two registered bitstreams can elaborate one
      configuration string onto one board and be different devices, so a ``design`` that survives
      to more than one ``device`` is refused rather than resolved by picking the fastest.
    * ``program_identity`` refuses a reference measured on a program the caller cannot produce.
      Passing it is how a caller says which program its candidate IS; omitting it leaves that
      unstated, which is weaker evidence and is reported by the caller's record, not here.
    """
    table = dict(references) if references is not None else load_references()
    candidates = {
        name: entry
        for name, entry in table.items()
        if isinstance(entry, Mapping)
        and entry.get("model") == model
        and entry.get("status") == status
        and isinstance((entry.get("measured") or {}).get("whole_model_cycles"), int)
    }
    if not candidates:
        known = sorted({str(entry.get("model")) for entry in table.values() if isinstance(entry, Mapping)})
        raise ReferenceError(
            f"no reference with status {status!r} carries a whole-model cycle count for model "
            f"{model!r}; the ledger describes {known}"
        )
    on_design = {name: entry for name, entry in candidates.items() if entry.get("design") == design}
    if not on_design:
        designs = sorted({str(entry.get("design")) for entry in candidates.values()})
        raise ReferenceError(
            f"model {model!r} has references on {designs} but none on design {design!r}; cycles "
            f"are not comparable across designs, so there is no reference to score against"
        )

    if device is not None:
        on_device = {name: entry for name, entry in on_design.items() if entry.get("device") == device}
        if not on_device:
            devices = sorted({str(entry.get("device")) for entry in on_design.values()})
            raise ReferenceError(
                f"design {design!r} has references on devices {devices} but none on {device!r}; a "
                "configuration name is not a device and cycles are not comparable across devices"
            )
        on_design = on_device
    else:
        devices = {str(entry.get("device") or "UNSTATED") for entry in on_design.values()}
        if len(devices) > 1:
            raise ReferenceError(
                f"design {design!r} does not pin a device -- its references span {sorted(devices)}; "
                "two bitstreams can elaborate one configuration string and be different devices, so "
                "pass `device=` rather than letting a design string select across them"
            )

    if program_identity is not None:
        on_identity = {
            name: entry for name, entry in on_design.items() if entry.get("program_identity") == program_identity
        }
        if not on_identity:
            identities = sorted({str(entry.get("program_identity") or "UNSTATED") for entry in on_design.values()})
            raise ReferenceError(
                f"model {model!r} has references of program identity {identities} but none of "
                f"{program_identity!r}; cycles are not comparable across program identities any more "
                "than across designs, and no compiler edit moves a candidate between them"
            )
        on_design = on_identity
    else:
        # The device rule, mirrored -- and this one has teeth. `min` over cycles is a race the most
        # different program wins: the surviving set here spans a callable compiler output and a
        # hand-written vendor benchmark that runs a BATCH OF FOUR and publishes it divided by four,
        # and the vendor entry is an order of magnitude smaller, so an unqualified lookup would
        # hand the authoring loop a destination whose batch size alone puts it out of reach. The
        # fastest number is exactly the one most likely to be a different program.
        identities = {str(entry.get("program_identity") or "UNSTATED") for entry in on_design.values()}
        if len(identities) > 1:
            raise ReferenceError(
                f"model {model!r} on device {device or '(any)'!r} has references of more than one "
                f"program identity ({sorted(identities)}); selecting the fewest cycles across them "
                "would pick whichever program is least like the candidate, so pass "
                "`program_identity=` rather than letting the ledger choose"
            )

    name = min(on_design, key=lambda key: on_design[key]["measured"]["whole_model_cycles"])
    return name, dict(on_design[name])


def find_reference_for_capsule(
    capsule: str, *, references: Mapping[str, Any] | None = None, status: str = "achieved"
) -> tuple[str, dict[str, Any]] | None:
    """The reference describing ``capsule``, or ``None`` when nothing measured describes it.

    ``None`` rather than a raise, and the caller is expected to RECORD it: an objective with no
    measured reference is the single most important thing a driver can say about itself, and the
    campaign this was written for was optimizing exactly such an objective for seventeen
    iterations without that fact appearing anywhere in the loop.
    """
    table = dict(references) if references is not None else load_references()
    matches = {
        name: dict(entry)
        for name, entry in table.items()
        if isinstance(entry, Mapping)
        and entry.get("status") == status
        and capsule in (entry.get("capsules") or ())
        and isinstance((entry.get("measured") or {}).get("whole_model_cycles"), int)
    }
    if not matches:
        return None
    devices = {str(entry.get("device") or "UNSTATED") for entry in matches.values()}
    if len(devices) > 1:
        raise ReferenceError(
            f"capsule {capsule!r} is named by references on devices {sorted(devices)}; picking the "
            "fastest would cross devices silently, which is the mistake `device` exists to stop"
        )
    name = min(matches, key=lambda key: matches[key]["measured"]["whole_model_cycles"])
    return name, matches[name]


def _candidate_identity(program_identity: str | None) -> ProgramIdentity:
    """A candidate's stated identity, or UNKNOWN with the reason it is unknown.

    A caller that states nothing gets UNKNOWN rather than a pass. That is the whole three-state
    rule: the candidate is the one thing the caller KNOWS about, so leaving it unstated is a
    decision not to say, not an absence of the fact.
    """
    name = str(program_identity or "").strip()
    if not name:
        return ProgramIdentity(
            name=None,
            facts={},
            confirmed_by=(),
            reason="the caller stated no `program_identity` for the candidate, so which program it "
            "is was never established; state it (the ledger's `program_identities` roster names "
            "the declared ones) rather than letting the comparison assume it",
        )
    if name == UNKNOWN_IDENTITY:
        return ProgramIdentity(
            name=None,
            facts={},
            confirmed_by=(),
            reason="the caller stated the candidate's program identity as UNKNOWN, which says the "
            "program is unrecorded -- it is not a program two entries can share",
        )
    return ProgramIdentity(name=name, facts={}, confirmed_by=("stated_by_the_caller",))


def _emitted_structure(command_buffer: Mapping[str, Any]) -> dict[str, int]:
    """The candidate's structural census, from its command buffer alone."""
    params = command_buffer.get("params")
    params = params if isinstance(params, Mapping) else {}
    abi = command_buffer.get("kernel_abi")
    abi = abi if isinstance(abi, Mapping) else {}

    def count(value: Any) -> int:
        return len(value) if isinstance(value, (list, tuple)) else 0

    return {
        "commands": count(command_buffer.get("commands")),
        "mesh_regions": count(params.get("mesh_regions")),
        "host_lane_regions": count(params.get("host_lane_regions")),
        "weight_prepack_recipes": count(params.get("weight_prepack_recipes")),
        "kernel_abi_args": count(abi.get("args")),
    }


def _opcode_census(command_buffer: Mapping[str, Any]) -> dict[str, int]:
    census: dict[str, int] = {}
    for command in command_buffer.get("commands") or ():
        if isinstance(command, Mapping):
            opcode = str(command.get("opcode"))
            census[opcode] = census.get(opcode, 0) + 1
    return census


def structural_gap(
    command_buffer: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    program_identity: str | None = None,
) -> dict[str, Any]:
    """Signed per-field distance from ``command_buffer`` to the reference's emitted structure.

    Negative means the candidate has FEWER than the reference. Only the reference's declared fields
    are compared -- a reference that records no structure yields no structural verdict rather than
    a vacuous pass.

    RAISES :class:`ProgramIdentityError` unless the candidate and the reference are the same
    program. This is the call site where a cross-identity comparison does its damage, and it did:
    scored against a reference whose lowering puts 54 contractions on the matrix unit, a candidate
    compiled from a capsule whose own interface is float reads ``mesh_regions: -53`` -- a number
    that looks exactly like a schedule the compiler failed to find, and is in fact the distance
    between two different programs, closable only by re-capturing the model. A signed per-field gap
    is the most quotable thing this module produces, so it is the one that must refuse rather than
    annotate: an UNKNOWN identity is refused too, because "we could not tell" is not "they match".
    """
    require_comparable(_candidate_identity(program_identity), identity_of_reference(reference))
    declared = reference.get("emitted_structure")
    if not isinstance(declared, Mapping):
        return {"status": "reference_declares_no_structure", "fields": {}, "matches": None}
    actual = _emitted_structure(command_buffer)
    fields = {
        name: {"candidate": actual[name], "reference": int(declared[name]), "gap": actual[name] - int(declared[name])}
        for name in _STRUCTURAL_FIELDS
        if isinstance(declared.get(name), int)
    }
    opcodes: dict[str, Any] = {}
    if isinstance(declared.get("opcodes"), Mapping):
        census = _opcode_census(command_buffer)
        for opcode, expected in declared["opcodes"].items():
            got = census.get(str(opcode), 0)
            opcodes[str(opcode)] = {"candidate": got, "reference": int(expected), "gap": got - int(expected)}
        for opcode, got in census.items():
            opcodes.setdefault(opcode, {"candidate": got, "reference": 0, "gap": got})
    unmatched = sorted(
        [name for name, row in fields.items() if row["gap"]]
        + [f"opcode:{name}" for name, row in opcodes.items() if row["gap"]]
    )
    return {
        "status": "derived",
        "fields": fields,
        "opcodes": opcodes,
        "matches": not unmatched,
        "unmatched": unmatched,
        "licence": (
            "a structural match means the candidate emits the same SHAPE of program, not "
            "that it runs as fast: the reference's own regression floor shares a "
            "byte-identical command buffer with it and ran 29% slower"
        ),
    }


def estimate_cycles(
    host_dynamic_operations: int,
    reference: Mapping[str, Any],
    *,
    baseline_host_dynamic_operations: int | None = None,
    program_identity: str | None = None,
) -> dict[str, Any]:
    """Estimated whole-model cycles for a candidate, anchored on the reference's measurement.

    The anchor is cycles-per-host-operation from a run that MEASURED both quantities, scaled by the
    host lane's share of the window. It prices every operation family alike, which is exactly why
    the returned record carries the caveat rather than only a number.

    RETURNS a refusal record -- it does not raise -- when the candidate and the reference are not
    the same program. Deliberately the opposite choice from :func:`structural_gap`, for a reason
    that is about where the result goes rather than about how bad the error is: this record is
    embedded in the feedback an authoring loop reads, and that loop wraps its analysis in
    ``try/except`` so nothing can stall a run. A raise there would become silence, and a broken
    comparison would be indistinguishable from a comparison nobody asked for. A loud record the
    agent READS is the stronger instrument; the raising form lives in :func:`structural_gap`.
    """
    verdict = comparability(_candidate_identity(program_identity), identity_of_reference(reference))
    if not verdict.ok:
        return {
            "status": "not_comparable_across_program_identities",
            "candidate_program_identity": verdict.get("candidate"),
            "reference_program_identity": verdict.get("reference"),
            "reason": verdict["reason"],
            "consequence": (
                "no cycle estimate is produced. Scaling this candidate's host operations onto that "
                "reference's measured window would price work the two programs do not share, and the "
                "resulting distance would be reported as an optimization gap the loop cannot close"
            ),
        }
    measured = reference.get("measured")
    measured = measured if isinstance(measured, Mapping) else {}
    anchor = measured.get("cycles_per_host_operation_anchor")
    share = measured.get("host_share_lower_bound")
    target = measured.get("whole_model_cycles")
    if not isinstance(anchor, (int, float)) or not isinstance(target, int):
        return {"status": "reference_carries_no_cycle_anchor"}
    if baseline_host_dynamic_operations:
        # A RELATIVE estimate against a measured point is the defensible form: the ratio of host
        # operations, applied to the host share of a measured window. It needs no claim that the
        # anchor prices this candidate's op mix correctly, only that the mix did not change much.
        share_value = float(share) if isinstance(share, (int, float)) else 0.93
        ratio = host_dynamic_operations / baseline_host_dynamic_operations
        estimated = target * (1.0 - share_value + share_value * ratio)
        basis = "relative_to_reference_measurement"
    else:
        estimated = float(anchor) * host_dynamic_operations
        basis = "absolute_from_anchor"
    return {
        "status": "estimated",
        "basis": basis,
        "estimated_whole_model_cycles": round(estimated),
        "reference_whole_model_cycles": target,
        "cycles_above_reference": round(estimated) - target,
        "fraction_of_reference": round(estimated / target, 6) if target else None,
        "licence": (
            "an estimate, never a measurement -- the anchor is a whole-window average and "
            "prices every host operation family identically, so a revision trading integer "
            "operations for floating-point ones can improve this number and lose cycles"
        ),
    }


def score_against_reference(
    command_buffer: Mapping[str, Any],
    *,
    model: str,
    design: str,
    host_dynamic_operations: int | None = None,
    baseline_host_dynamic_operations: int | None = None,
    references: Mapping[str, Any] | None = None,
    program_identity: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """The whole gap record for one candidate: which reference, structural distance, cycle distance.

    ``program_identity`` and ``device`` are the candidate's own; both are carried into the record so
    a reader can see what the score was licensed by, including when it settled for less.
    """
    name, reference = find_reference(
        model, design, references=references, program_identity=program_identity, device=device
    )
    record: dict[str, Any] = {
        "schema": "perf_reference_gap_v1",
        "reference": name,
        "model": model,
        "design": design,
        "device": device or reference.get("device") or "UNSTATED",
        "program_identity": {
            "candidate": program_identity or "UNSTATED",
            "reference": reference.get("program_identity") or "UNSTATED",
            "confirmed_by": "stated by the caller" if program_identity else "NOT stated by the caller",
        },
        "structural": structural_gap(command_buffer, reference, program_identity=program_identity),
    }
    if isinstance(host_dynamic_operations, int):
        record["cycles"] = estimate_cycles(
            host_dynamic_operations,
            reference,
            baseline_host_dynamic_operations=baseline_host_dynamic_operations,
            program_identity=program_identity,
        )
    for note in reference.get("caveats") or ():
        record.setdefault("caveats", []).append(str(note))
    return record
