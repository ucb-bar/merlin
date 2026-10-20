"""What a Phase 2 performance score may contain: accelerator work, never Merlin-supplied host compute.

Phase 2 tunes what decides ACCELERATOR performance: tiling and blocking, the dataflow choice, DMA
scheduling and double-buffering, scratchpad and accumulator reuse, epilogue fusion and command-queue
pressure. Host code counts only where it changes accelerator behaviour or codegen -- the layout and
packing that feed the accelerator, im2col, command issue. Pure host compute is Merlin-supplied
(``docs/design/merlin_supplied_host_placement.md``): Merlin's own host package compiles it, so it is
neither the candidate's work nor something a candidate may be scored on.

The split is the frozen Phase 0 placement (:mod:`merlin.targetgen.frozen_placement`), never the
candidate's output:

* operations Phase 0 placed on the **accelerator** are the candidate's scored work;
* operations placed on the **host** are Merlin-supplied and excluded;
* **support** operations (packing, residency, readout bookkeeping, shape plumbing) are deliberately
  unplaced by the Phase 0 screen. They are what the candidate emits around its device groups, so they
  count, as accelerator-affecting host code.

Two consequences for measurement:

1. A whole-program capsule whose frozen placement keeps compute on the host may not take the
   candidate-whole-program route, where the candidate would emit (and could tune) that host code. It
   takes the owned-host split instead, where Merlin compiles the host part.
2. A cycle count measured on such a program still contains Merlin's host stage. The absolute score is
   the measured count minus a measured Merlin host stage; without that measurement it is UNKNOWN. A
   paired delta is admissible without it only when both arms link byte-identical Merlin host code,
   because the host stage then cancels.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "merlin.phase2.metric_scope.v1"
ACCELERATOR_ONLY = "accelerator_only"
MERLIN_HOST_SPLIT = "merlin_host_split"
ACCELERATOR_AFFECTING_HOST = (
    "unplaced support operations the candidate emits around its device groups: layout and packing "
    "feeding the accelerator, im2col, residency and readout bookkeeping, command issue"
)


class MetricScopeRefusal(ValueError):
    """The scored scope cannot be established, so no score may be issued."""


def _placements(declaration: Mapping[str, Any]) -> Mapping[str, Any]:
    screen = declaration.get("software_screen")
    placements = screen.get("placements") if isinstance(screen, Mapping) else None
    if not isinstance(placements, Mapping):
        raise MetricScopeRefusal("the capsule carries no frozen Phase 0 placement, so its scored scope is unknown")
    for lane, count in placements.items():
        if type(count) is not int or count < 0:
            raise MetricScopeRefusal(f"the frozen placement count for lane {lane!r} is malformed")
    if not placements.get("accelerator"):
        raise MetricScopeRefusal("the frozen placement puts no operation on the accelerator; nothing is scorable")
    return placements


def declared_scope(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """The scope a declaration's frozen placement states, without reading program bytes.

    The route this selects re-binds the placement to the program bytes (:func:`bound_scope`, or the
    owned-host route's own ``frozen_placement.read``), which fails closed.
    """
    placements = _placements(declaration)
    host = int(placements.get("host") or 0)
    return {
        "schema": SCHEMA,
        "authority": "frozen Phase 0 software_screen",
        "kind": MERLIN_HOST_SPLIT if host else ACCELERATOR_ONLY,
        "placements": dict(sorted(placements.items())),
        "scored": "accelerator-placed operations plus " + ACCELERATOR_AFFECTING_HOST,
        "excluded": (
            "host-placed operations: Merlin-supplied host compute, compiled by Merlin's host package"
            if host
            else "none: the frozen placement keeps no compute on the host"
        ),
        "candidate_whole_program_admissible": not host,
    }


def bound_scope(declaration: Mapping[str, Any], directory: str | Path) -> dict[str, Any]:
    """:func:`declared_scope` bound to the capsule's program bytes, with per-operation lanes."""
    from merlin.targetgen import frozen_placement

    try:
        frozen = frozen_placement.read(declaration, directory)
    except frozen_placement.FrozenPlacementRefusal as exc:
        raise MetricScopeRefusal(str(exc)) from exc
    scope = declared_scope(declaration)
    scope.update(
        program_sha256=frozen["program_sha256"],
        scored_operations=frozen["accelerator_operations"],
        excluded_operations=frozen["host_operations"],
    )
    return scope


def scored_cycles(
    scope: Mapping[str, Any], measured_cycles: int, *, merlin_host_cycles: int | None = None
) -> dict[str, Any]:
    """The cycles a candidate is scored on, from a complete measured count.

    An accelerator-only program scores its whole measured count. A Merlin-host split scores the
    measured count minus a measured Merlin host stage, and is UNKNOWN without one: the whole-program
    count is never silently promoted to a candidate score.
    """
    if isinstance(measured_cycles, bool) or not isinstance(measured_cycles, int) or measured_cycles <= 0:
        raise MetricScopeRefusal("a scored count needs a positive measured cycle count")
    kind = scope.get("kind")
    if kind == ACCELERATOR_ONLY:
        if merlin_host_cycles not in (None, 0):
            raise MetricScopeRefusal("an accelerator-only program has no Merlin host stage to subtract")
        return {"status": "scored", "cycles": measured_cycles, "basis": "complete call; no Merlin host stage"}
    if kind != MERLIN_HOST_SPLIT:
        raise MetricScopeRefusal(f"unknown metric scope kind {kind!r}")
    if merlin_host_cycles is None:
        return {
            "status": "unknown",
            "cycles": None,
            "measured_cycles": measured_cycles,
            "basis": (
                "the measured count contains Merlin-supplied host compute and no Merlin host stage was "
                "measured; only a paired delta over byte-identical Merlin host code is admissible"
            ),
        }
    if isinstance(merlin_host_cycles, bool) or not isinstance(merlin_host_cycles, int) or merlin_host_cycles < 0:
        raise MetricScopeRefusal("a Merlin host stage must be a nonnegative measured cycle count")
    if merlin_host_cycles >= measured_cycles:
        raise MetricScopeRefusal("the Merlin host stage is not smaller than the complete measured count")
    return {
        "status": "scored",
        "cycles": measured_cycles - merlin_host_cycles,
        "measured_cycles": measured_cycles,
        "excluded_merlin_host_cycles": merlin_host_cycles,
        "basis": "complete call minus the measured Merlin-supplied host stage",
    }


def paired_delta(
    scope: Mapping[str, Any],
    *,
    candidate_cycles: int,
    baseline_cycles: int,
    candidate_host_sha256: str | None,
    baseline_host_sha256: str | None,
) -> dict[str, Any]:
    """Candidate minus baseline, admissible when Merlin's host stage provably cancels.

    For a Merlin-host split both arms must link the same Merlin host code, bound by digest; a differing
    or missing digest means the delta could contain host work and it is refused.
    """
    for value in (candidate_cycles, baseline_cycles):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MetricScopeRefusal("a paired delta needs positive measured counts for both arms")
    if scope.get("kind") == MERLIN_HOST_SPLIT:
        if not candidate_host_sha256 or candidate_host_sha256 != baseline_host_sha256:
            return {
                "status": "refused",
                "delta_cycles": None,
                "basis": "the arms do not provably link byte-identical Merlin host code",
            }
        basis = "Merlin host code is byte-identical in both arms, so its stage cancels"
    elif scope.get("kind") == ACCELERATOR_ONLY:
        basis = "accelerator-only program"
    else:
        raise MetricScopeRefusal(f"unknown metric scope kind {scope.get('kind')!r}")
    return {"status": "scored", "delta_cycles": candidate_cycles - baseline_cycles, "basis": basis}
