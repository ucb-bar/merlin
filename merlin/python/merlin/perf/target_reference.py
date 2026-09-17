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
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import merlin_dir

__all__ = [
    "REFERENCE_LEDGER_NAME",
    "ReferenceError",
    "load_references",
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


def load_references(path: Path | None = None) -> dict[str, Any]:
    """The reference ledger, as data."""
    ledger = Path(path) if path is not None else merlin_dir() / REFERENCE_LEDGER_NAME
    document = yaml.safe_load(ledger.read_text())
    if not isinstance(document, Mapping) or not isinstance(document.get("references"), Mapping):
        raise ReferenceError(f"{ledger} declares no `references` mapping")
    return dict(document["references"])


def find_reference(
    model: str, design: str, *, references: Mapping[str, Any] | None = None, status: str = "achieved"
) -> tuple[str, dict[str, Any]]:
    """The best measured reference for ``model`` on ``design`` -- the FEWEST cycles achieved.

    Refuses on a design mismatch instead of falling back to another design's number, because a
    cycle count from a different design is not a weaker answer, it is a wrong one.
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
    name = min(matches, key=lambda key: matches[key]["measured"]["whole_model_cycles"])
    return name, matches[name]


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


def structural_gap(command_buffer: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Signed per-field distance from ``command_buffer`` to the reference's emitted structure.

    Negative means the candidate has FEWER than the reference. Only the reference's declared fields
    are compared -- a reference that records no structure yields no structural verdict rather than
    a vacuous pass.
    """
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
    host_dynamic_operations: int, reference: Mapping[str, Any], *, baseline_host_dynamic_operations: int | None = None
) -> dict[str, Any]:
    """Estimated whole-model cycles for a candidate, anchored on the reference's measurement.

    The anchor is cycles-per-host-operation from a run that MEASURED both quantities, scaled by the
    host lane's share of the window. It prices every operation family alike, which is exactly why
    the returned record carries the caveat rather than only a number.
    """
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
) -> dict[str, Any]:
    """The whole gap record for one candidate: which reference, structural distance, cycle distance."""
    name, reference = find_reference(model, design, references=references)
    record: dict[str, Any] = {
        "schema": "perf_reference_gap_v1",
        "reference": name,
        "model": model,
        "design": design,
        "structural": structural_gap(command_buffer, reference),
    }
    if isinstance(host_dynamic_operations, int):
        record["cycles"] = estimate_cycles(
            host_dynamic_operations, reference, baseline_host_dynamic_operations=baseline_host_dynamic_operations
        )
    for note in reference.get("caveats") or ():
        record.setdefault("caveats", []).append(str(note))
    return record
