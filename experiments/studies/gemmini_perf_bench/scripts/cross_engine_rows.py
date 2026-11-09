#!/usr/bin/env python3
"""Fold a measurement taken on ANOTHER engine into this bench, but only if a shared control licenses it.

A row measured in-model on an FPGA and a row measured in isolation on a cycle-accurate model are, by
default, not comparable: different device, different window. Printing them in one ``cycles`` column is
the failure this repo's hardware-provenance convention exists to prevent -- a result attributed to the
wrong device is worse than no result, because it gets cited.

There is one thing that can license it, and it is not an assertion: **a control measured on both
sides**. If the same stock-library kernel, on the same shape, lands within a declared tolerance across
the two engines and windows, then the two scales are calibrated against each other on exactly the work
in question, and a ratio taken through that control means something. This module computes that
agreement from the caller's OWN measured control rows and the foreign log, and REFUSES the row when
the agreement is outside tolerance. It never takes the licence on trust, and it never takes a summary:
the cycles come from the engine's own console log, and the log must show the run reaching its end.

The resulting row is kept in its own arm, carrying its engine, its design, its job identity, its
sealed/observed status, and the control it was calibrated through -- so nobody can mistake it for a
direct measurement on the bench's own device.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

#: A per-group console line: ``GM_GROUP <id> <op> <cycles> sum=<checksum>``.
GROUP_LINE = "GM_GROUP"
CHECKSUM_FIELD = "sum="
#: The line a completed run prints last. A log without it is a PREFIX, and a prefix reads better than
#: the truth -- the fast groups finish first.
COMPLETION_MARKER = "MERLIN_PROFILE measured end"
#: Retired 2026-09-19 from the emitter, which now prints the contracted ``METRIC cycles`` line. Kept
#: only because rows recorded before that date name it as their ``cycles_prefix``; nothing in this
#: module reads it, and a new consumer should not start.
TOTAL_PREFIX = "GROUP_MODEL_TOTAL cycles:"


class NotLicensed(ValueError):
    """The shared control does not calibrate the two engines well enough to put them in one table."""


class ConsoleUnreadable(ValueError):
    """A log announced a record this parser could not read.

    Distinct from :class:`NotLicensed`: that one says two runs cannot be COMPARED, this one says a log
    cannot be TRUSTED to say what it appears to say. Silently dropping the record would leave the rest
    of the log looking whole.
    """


@dataclass(frozen=True)
class ForeignRun:
    """One completed run on another engine: what it was, where it ran, and what it printed."""

    path: Path
    job_id: int | None
    label: str
    status: str
    observed_cycles: int | None
    oracle: str | None
    substrate: str
    completed: bool
    groups: dict[int, int]
    checksums: dict[int, str]

    @property
    def sealed(self) -> bool:
        return self.status == "sealed"


def parse_console(text: str) -> tuple[dict[int, int], dict[int, str], bool]:
    """``(cycles by group, checksum by group, reached_the_end)`` from an engine console log.

    Structural: split the line and read the fields by position, never a pattern match.

    A line that ANNOUNCES ITSELF as a group record and then cannot be read is an error, not something
    to skip. Skipping it silently drops one group's evidence while leaving the rest of the log looking
    complete, and the caller then compares a set of checksums with a hole in it -- which is how a run
    that lost a group to a garbled console reads as a run that agreed about it. A line that is not a
    group record at all is simply not ours and is passed over.
    """
    cycles: dict[int, int] = {}
    checksums: dict[int, str] = {}
    completed = False
    for line in text.splitlines():
        if COMPLETION_MARKER in line:
            completed = True
        parts = line.split()
        if not parts or parts[0] != GROUP_LINE:
            continue
        if len(parts) < 4 or not parts[1].isdecimal() or not parts[3].isdecimal():
            raise ConsoleUnreadable(
                f"a {GROUP_LINE} record could not be read, so this log is missing evidence it "
                f"claims to carry: {line.strip()!r}"
            )
        group = int(parts[1])
        cycles[group] = int(parts[3])
        for token in parts[4:]:
            if token.startswith(CHECKSUM_FIELD):
                checksums[group] = token[len(CHECKSUM_FIELD) :]
    return cycles, checksums, completed


def load_run(measurement_dir: Path) -> ForeignRun:
    """Read one measurement directory: its checkpoint record and the console log beside it."""
    measurement_dir = Path(measurement_dir)
    record_path = measurement_dir / "checkpoint_row.json"
    console = measurement_dir / "evidence" / "uartlog"
    if not record_path.is_file() or not console.is_file():
        raise FileNotFoundError(f"{measurement_dir} is not a measurement (need checkpoint_row.json + evidence/uartlog)")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    cycles, checksums, completed = parse_console(console.read_text(encoding="utf-8", errors="replace"))
    # The substrate is the folder the measurement lives under, which is how `new_measurement` lays it
    # out -- read rather than passed, so a run cannot be filed under a device it did not run on.
    substrate = measurement_dir.parent.parent.name
    return ForeignRun(
        path=measurement_dir,
        job_id=record.get("job_id"),
        label=str(record.get("label") or ""),
        status=str(record.get("status") or "unknown"),
        observed_cycles=record.get("observed_cycles"),
        oracle=record.get("oracle"),
        substrate=substrate,
        completed=completed,
        groups=cycles,
        checksums=checksums,
    )


def groups_by_shape(group_capsules: Mapping) -> dict[int, dict]:
    """Group id -> the convolution it computes, from the capsule inventory the route emitted.

    Only entries that carry a fused epilogue are used: the same group also appears with an empty
    epilogue (the raw contraction), and the two are different functions.
    """
    out: dict[int, dict] = {}
    for entry in group_capsules.get("entries", []):
        spec = entry.get("entry") or {}
        if not spec.get("epilogue"):
            continue
        for group in entry.get("groups") or []:
            out[int(group)] = spec
    return out


def matches_layer(group_spec: Mapping, layer: Mapping) -> bool:
    """Whether a group computes the bench's layer: same op, extents, window, stride and padding.

    Compared field by field against the layer the bench measured, so a group that merely resembles it
    (a different padding, say, or a raw contraction with no epilogue) does not match.
    """
    if group_spec.get("op") != layer.get("op") or group_spec.get("op") != "conv2d":
        return False
    padding = list(group_spec.get("padding") or [])
    stride = list(group_spec.get("stride") or [])
    return (
        int(group_spec.get("Himg", -1)) == int(layer["in_dim"])
        and int(group_spec.get("Wimg", -1)) == int(layer["in_dim"])
        and int(group_spec.get("ci", -1)) == int(layer["in_channels"])
        and int(group_spec.get("N", -1)) == int(layer["out_channels"])
        and int(group_spec.get("kh", -1)) == int(layer["kernel"])
        and int(group_spec.get("kw", -1)) == int(layer["kernel"])
        and stride == [int(layer["stride"])] * 2
        and padding == [int(layer["padding"])] * 4
    )


def control_agreement(foreign_control: int, own_control: int) -> float:
    """How far apart the two engines put the SAME control kernel, as a ratio >= 1."""
    if own_control <= 0 or foreign_control <= 0:
        raise ValueError("a control cycle count must be positive")
    return max(foreign_control, own_control) / min(foreign_control, own_control)


def calibrate(
    ours: ForeignRun,
    control: ForeignRun,
    *,
    group_capsules: Mapping,
    layer: Mapping,
    own_control_cycles: int,
    tolerance: float,
) -> dict:
    """One calibrated row for ``layer``, or raise :class:`NotLicensed` saying which check failed.

    Five things must hold, and each is checked here rather than asserted in prose:

    1. both foreign logs reached their end (a partial log is a prefix, and the fast groups finish first);
    2. the groups computing this layer are the same set in both foreign runs;
    3. those groups produced IDENTICAL output checksums in both runs, so the two arms computed the same
       function -- the foreign comparison is not of a faster-but-different kernel;
    4. the shared control agrees across engine and window within ``tolerance``;
    5. the tolerance itself is declared by the caller, and travels with the row.
    """
    if not ours.completed:
        raise NotLicensed(f"{ours.path.name}: console log never reached {COMPLETION_MARKER!r}; it is a prefix")
    if not control.completed:
        raise NotLicensed(f"{control.path.name}: console log never reached {COMPLETION_MARKER!r}; it is a prefix")
    shape_map = groups_by_shape(group_capsules)
    groups = sorted(g for g, spec in shape_map.items() if matches_layer(spec, layer))
    if not groups:
        raise NotLicensed("no compute group in the inventory computes this layer")
    missing = [g for g in groups if g not in ours.groups or g not in control.groups]
    if missing:
        raise NotLicensed(f"groups {missing} are not in both foreign logs")
    # A checksum that is ABSENT is not a checksum that agrees. Comparing two missing values with
    # ``!=`` reads ``None != None`` as False and licenses the ratio, so a pair of runs that printed no
    # checksum at all would be certified as computing the same function on the strength of neither
    # having said anything. Presence is therefore required before equality is consulted.
    unchecked = [g for g in groups if g not in ours.checksums or g not in control.checksums]
    if unchecked:
        raise NotLicensed(
            f"groups {unchecked} printed no output checksum in one or both foreign runs, so there is "
            "no evidence they computed the same function; a cycle ratio between them would be "
            "unlicensed. Absence of a checksum is not agreement."
        )
    differing = [g for g in groups if ours.checksums[g] != control.checksums[g]]
    if differing:
        raise NotLicensed(
            f"groups {differing} produced different outputs in the two foreign runs, so they did not "
            "compute the same function; a cycle ratio between them would be meaningless"
        )
    foreign_control = min(control.groups[g] for g in groups)
    agreement = control_agreement(foreign_control, own_control_cycles)
    if agreement > 1.0 + tolerance:
        raise NotLicensed(
            f"the shared control disagrees by {100 * (agreement - 1):.2f}% across the two engines "
            f"(foreign {foreign_control:,} vs own {own_control_cycles:,}), beyond the declared "
            f"{100 * tolerance:.2f}% tolerance; the two scales are not calibrated on this work"
        )
    per_group = {str(g): ours.groups[g] for g in groups}
    # The two sides share the contract's SHAPE (bias, per-tensor requant, activation) but may carry
    # different requant constants, because they were captured from the model separately. That does not
    # move cycles -- it is the same instruction either way -- but the arms are then not computing the
    # identical function, so the difference is reported rather than quietly absorbed.
    group_scale = shape_map[groups[0]].get("acc_scale")
    layer_scale = layer.get("scale")
    return {
        "arm": "compute_group",
        "epilogue": list(shape_map[groups[0]].get("epilogue") or []),
        "acc_scale": {
            "foreign": group_scale,
            "own": layer_scale,
            "identical": group_scale is not None and layer_scale is not None and group_scale == layer_scale,
        },
        "engine": ours.substrate,
        "measured_in": "the whole compute group, including its host sequencing",
        "job_id": ours.job_id,
        "label": ours.label,
        "status": ours.status,
        "sealed": ours.sealed,
        "oracle": ours.oracle,
        "evidence": str(ours.path),
        "groups": [str(g) for g in groups],
        "cycles_per_group": per_group,
        "cycles": min(ours.groups[g] for g in groups),
        "cycles_max": max(ours.groups[g] for g in groups),
        "calibration": {
            "through": "the stock-library kernel measured on both engines, on this same layer",
            "foreign_control_cycles": foreign_control,
            "own_control_cycles": own_control_cycles,
            "agreement_ratio": agreement,
            "tolerance": tolerance,
            "control_job_id": control.job_id,
            "control_evidence": str(control.path),
            "control_has_own_oracle": bool(control.oracle),
        },
        "same_output_as_control": True,
        "numerics": "exact" if ours.oracle else "not independently checked in this run",
    }
