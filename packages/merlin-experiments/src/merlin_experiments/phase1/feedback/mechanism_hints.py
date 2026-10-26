"""Name catalogue mechanisms that address a test case's measured bottleneck; advice, never a verdict.

The optimization catalogue offered every mechanism as one long menu, so the author had to map its own
measurements onto it by hand and mostly wrote specialised schedules instead of reusing the library's
general passes. This module joins two things the author already has: the bottleneck its own checks
measured for a test case (the busiest modeled resource, or which roofline bound limits it) and the
mechanisms the architecture selection in ``OPTIMIZATION_SUGGESTIONS.json`` found applicable to this
target. Nothing here is curated per model, test case or target, and nothing ranks mechanisms by what is
known to win: the mapping below goes from the timing model's generic resource vocabulary to the
catalogue's own facets, and mechanisms keep catalogue order within a facet.

Resource names are ``<domain>.<kind>[.<detail>]`` (``accelerator.storage.scratchpad.0``,
``accelerator.issue.load``, ``cpu.issue``, ``memory.l1d.probe`` ...). The mapping:

* local storage busy (``storage``) -> storage, movement, overlap mechanisms;
* transfers busy (``memory.*``, or a ``load``/``store`` controller or issue lane) -> movement, overlap, storage;
* instruction issue busy (``cpu.issue``, ``cpu.rocc_dispatch``, an accelerator ``issue`` lane) -> issue,
  synchronization, scheduling;
* accelerator execution busy (``execute``) -> scheduling, overlap, fusion;
* other host work (``cpu.*``) -> host_compute, issue, packing.

Without resource feedback, the limiting roofline bound decides: request service -> movement, storage,
overlap; compute floor -> scheduling, issue, fusion. Hints take one mechanism per facet in turn.

A busy storage resource also names its unit (``accumulator`` in ``accelerator.storage.accumulator.all``).
A facet alone pointed an accumulator-bound case at tile capacity, copy coalescing and prefetch, which
change memory traffic and leave accumulator work alone. Mechanisms whose own catalogue entry (id,
requirements, mechanism, legality, evidence) names that unit therefore come first, most mentions first,
then those in the resource's facets; the facet hints fill the remaining places.

A case whose command stream drains the whole device after nearly every output tile (limiter
``FENCE_BOUND``, read by :mod:`.fence_overlap`) is held back by that ordering rather than by one unit:
overlap, synchronization, movement, mechanisms whose entry names a fence, barrier or readout first
(:func:`fence_hints`). Mechanisms of those facets that name them but whose capability the
selection could not confirm are listed apart with the unconfirmed requirements, never as available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SUGGESTIONS = "OPTIMIZATION_SUGGESTIONS.json"
#: Mechanisms named per test case.
PER_CASE = 4
_MAX_BYTES = 8 << 20

_STORAGE = ("storage", "movement", "overlap")
_TRANSFER = ("movement", "overlap", "storage")
_ISSUE = ("issue", "synchronization", "scheduling")
_EXECUTE = ("scheduling", "overlap", "fusion")
_HOST = ("host_compute", "issue", "packing")
#: The limiter of a stream fenced after nearly every output tile, and the facets that address it.
FENCE_BOUND = "fences"
_SYNC = ("overlap", "synchronization", "movement")
#: Words a catalogue entry uses for what a fence-bound stream does too often (stems match; ``drain``
#: is left out because its stem also matches ``dram``).
FENCE_WORDS = ("fence", "barrier", "readout")


def resource_facets(name) -> tuple[str, ...] | None:
    """Catalogue facets that address a busy resource, from its generic name segments."""
    if not isinstance(name, str) or not name:
        return None
    segments = name.split(".")
    domain, rest = segments[0], segments[1:]
    if "storage" in rest:
        return _STORAGE
    if domain == "memory" or "load" in rest or "store" in rest:
        return _TRANSFER
    if (domain == "cpu" and rest[:1] in (["issue"], ["rocc_dispatch"])) or (domain != "cpu" and "issue" in rest):
        return _ISSUE
    if "execute" in rest:
        return _EXECUTE
    if domain == "cpu":
        return _HOST
    return None


def storage_unit(name) -> str | None:
    """The storage unit a busy storage resource names (the segment after ``storage``), else None."""
    segments = name.split(".") if isinstance(name, str) else []
    if "storage" not in segments[1:]:
        return None
    after = segments[segments.index("storage", 1) + 1 :]
    return after[0].lower() if after and after[0].isalpha() else None


def _words(text) -> list[str]:
    return "".join(c if c.isalnum() else " " for c in str(text or "").lower()).split()


def _same_stem(word: str, unit: str) -> bool:
    """``accumulate``/``accumulation`` name an ``accumulator``: they share all but its last two letters."""
    stem = max(len(unit) - 2, 1)
    return len(word) >= stem and len(os.path.commonprefix([word, unit])) >= stem


def _mentions(row: dict, unit) -> int:
    """How many fields of a catalogue entry name ``unit`` (one word, or any of a tuple of words)."""
    units = (unit,) if isinstance(unit, str) else tuple(unit)
    fields = [
        row.get("id"),
        " ".join((row.get("requirements") or {}).keys()),
        row.get("mechanism"),
        row.get("legality"),
        row.get("evidence"),
    ]
    return sum(any(_same_stem(word, name) for word in _words(text) for name in units) for text in fields)


def bound_facets(bound_by) -> tuple[str, ...] | None:
    if not isinstance(bound_by, str):
        return None
    if "requests" in bound_by:
        return _TRANSFER
    if bound_by == "compute":
        return ("scheduling", "issue", "fusion")
    if bound_by == FENCE_BOUND:
        return _SYNC
    return None


def load_selection(workspace) -> dict | None:
    """The architecture selection staged for the author, or ``None`` when it was not staged."""
    path = Path(workspace) / SUGGESTIONS
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BYTES:
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) and isinstance(document.get("all_mechanisms"), list) else None


def _available(selection: dict) -> list[dict]:
    """Mechanisms whose capability and library requirements are all supported (workload may be pending)."""
    rows = []
    for row in selection.get("all_mechanisms") or ():
        requirements = row.get("requirements") if isinstance(row, dict) else None
        if not isinstance(requirements, dict) or row.get("status") == "unsupported":
            continue
        if all(state == "supported" for key, state in requirements.items() if not key.startswith("workload.")):
            rows.append(row)
    return rows


def fence_hints(selection: dict | None, *, why: str, limit: int = PER_CASE) -> tuple[list[dict], list[dict]]:
    """``(available, unconfirmed)`` mechanisms for a fence-bound stream (see the module docstring)."""
    if not selection:
        return [], []
    found = hints(selection, bound_by=FENCE_BOUND, limit=limit, words=FENCE_WORDS, why=why)
    chosen = {hint["id"] for hint in found}
    unconfirmed = []
    for row in selection.get("all_mechanisms") or ():
        requirements = row.get("requirements") if isinstance(row, dict) else None
        if (
            not isinstance(requirements, dict)
            or row.get("id") in chosen
            or row.get("status") == "unsupported"
            or row.get("family") not in _SYNC
            or not _mentions(row, FENCE_WORDS)
        ):
            continue
        unknown = sorted(
            key for key, state in requirements.items() if not key.startswith("workload.") and state != "supported"
        )
        if unknown:
            unconfirmed.append(
                {
                    "id": row.get("id"),
                    "api": f"{row.get('api')}:{row.get('symbol')}",
                    "facet": row.get("family"),
                    "capability_unconfirmed": unknown,
                }
            )
    return found, unconfirmed[:limit]


def hints(
    selection: dict | None, *, dominant=None, bound_by=None, limit: int = PER_CASE, words=None, why=None
) -> list[dict]:
    """Up to ``limit`` mechanisms for one test case's measured bottleneck (see the module docstring).

    ``words`` (with ``why``) replaces the busy storage unit as what an entry must name to come first.
    """
    if not selection:
        return []
    name = (dominant or {}).get("name") if isinstance(dominant, dict) else None
    facets = resource_facets(name)
    if facets is not None:
        why = f"busiest modeled resource {name} (activity {dominant.get('modeled_activity')})"
    else:
        facets = bound_facets(bound_by)
        if facets is not None and why is None:
            why = f"limited by its {bound_by} roofline bound"
    if facets is None:
        return []
    available = _available(selection)
    unit = (tuple(words) if words else storage_unit(name)) if why and (name or words) else None
    named = []
    if unit is not None:
        scored = [(_mentions(row, unit), index, row) for index, row in enumerate(available)]
        named = [
            row
            for count, _index, row in sorted(
                (item for item in scored if item[0]),
                key=lambda item: (-item[0], item[2].get("family") not in facets, item[1]),
            )
        ][: max(limit - 1, 1)]
    # Then one mechanism per facet in turn (catalogue order within a facet), facets the named mechanisms
    # did not cover first, so the hints span the ways to address the bottleneck.
    covered = {row.get("family") for row in named}
    order = sorted(facets, key=lambda facet: facet in covered)
    queues = [[row for row in available if row.get("family") == facet and row not in named] for facet in order]
    named_as = f"names the busy {unit} unit" if isinstance(unit, str) else f"names a {'/'.join(unit or ())}"
    chosen = [(row, f"{named_as}; {why}") for row in named]
    while len(chosen) < limit and any(queues):
        for queue in queues:
            if queue and len(chosen) < limit:
                chosen.append((queue.pop(0), why))
    result = []
    for row, row_why in chosen:
        workload = {key: state for key, state in (row.get("requirements") or {}).items() if key.startswith("workload.")}
        result.append(
            {
                "id": row.get("id"),
                "api": f"{row.get('api')}:{row.get('symbol')}",
                "facet": row.get("family"),
                "why": row_why,
                "evidence_to_request": row.get("evidence"),
                "workload_check": (
                    "satisfied"
                    if row.get("status") == "supported"
                    else "unchecked: run `python optimization_suggest.py --ir <source> --output <file>`"
                    f" for {sorted(k for k, v in workload.items() if v != 'supported')}"
                ),
            }
        )
    return result
