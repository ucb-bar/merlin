"""Why a capability the target offers was REFUSED, per site, aggregated by clause.

A capability selector is a sequence of guard clauses over target facts and operation semantics,
and it returns both a verdict and the clause that decided it. Compilers routinely keep the verdict
and discard the clause -- `eligible, _reason = select(...)` -- which throws away the only
machine-readable statement of what would have to change. Measured on a whole-model ResNet-50: all
53 convolutions were refused the device-side convolution sequencer, the selector named the deciding
clause every time, and every one of those 53 strings was dropped at the call site. Downstream, the
emitted program simply contains no such instruction, and the absence looks like a choice nobody
made rather than a refusal with a stated cause.

`isa_utilization` can say "this declared instruction is never emitted". Only the refusal ledger can
say WHY, and whether the cause is a contract this compiler chose (an output dtype, a tensor layout)
or a property of the operation it cannot change.

THE SHORT-CIRCUIT CAVEAT, which is the whole reason this is subtle. A guard sequence returns on its
FIRST failing clause, so a census counts first refusals and says nothing about the clauses never
reached. Removing the top clause does not admit those sites; it reveals the next clause. Measured
here: the sites refused for a full-width output dtype had their layout clause never evaluated, so
the dtype fix -- which was made, and did satisfy that clause -- exposed the layout requirement
underneath rather than admitting anything. A reader who treats the top clause as "the" blocker will
predict admission and be wrong. :func:`census` therefore reports `first_refusal_only: True` and
carries that caveat in the payload, and :func:`unblocking_sequence` reports the clauses as an
ORDERED CASCADE rather than a set of independent fixes.

Target-neutral by construction: clause names are opaque strings chosen by the selector, site labels
are opaque strings chosen by the caller. This module never interprets either, and never names a
target, capability, instruction or clause of its own.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

__all__ = ["RefusalSite", "census", "unblocking_sequence", "SELECTED"]

#: The conventional clause name a selector returns when it ADMITS a site. Callers may use any
#: string; passing it here keeps an admitted site out of the refusal census.
SELECTED = "selected"


class RefusalSite(NamedTuple):
    """One site's verdict for one capability.

    `site` and `clause` are opaque. `detail` carries whatever the caller wants a reader to see
    (geometry, dtypes, the tensor layouts) so a clause count can be turned back into the concrete
    thing that has to change.
    """

    site: str
    admitted: bool
    clause: str
    detail: Mapping[str, Any] | None = None


def _rows(sites: Iterable[RefusalSite]) -> list[RefusalSite]:
    out: list[RefusalSite] = []
    for row in sites:
        if isinstance(row, RefusalSite):
            out.append(row)
            continue
        # Accept the selector's own (bool, str) shape paired with a label, so a caller does not
        # have to restate what it already has.
        site, admitted, clause = row[0], bool(row[1]), str(row[2])
        detail = row[3] if len(row) > 3 else None
        out.append(RefusalSite(str(site), admitted, clause, detail))
    return out


def census(capability: str, sites: Iterable[RefusalSite]) -> dict[str, Any]:
    """Aggregate per-site verdicts for one capability into a clause census.

    The census is of FIRST refusals -- see the short-circuit caveat in the module docstring. It is
    reported in the payload rather than left to a reader's memory.
    """
    rows = _rows(sites)
    admitted = [r for r in rows if r.admitted]
    refused = [r for r in rows if not r.admitted]
    counts = Counter(r.clause for r in refused)
    # Rank by how many sites a clause decided, then by name so equal counts are stable.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "schema": "capability_refusal_census_v1",
        "capability": capability,
        "sites_total": len(rows),
        "sites_admitted": len(admitted),
        "sites_refused": len(refused),
        "admitted_sites": [r.site for r in admitted],
        "clauses": [
            {
                "clause": clause,
                "sites": count,
                "share_of_refused": round(count / len(refused), 6) if refused else 0.0,
                "example_sites": [r.site for r in refused if r.clause == clause][:5],
                "example_detail": next((dict(r.detail) for r in refused if r.clause == clause and r.detail), None),
            }
            for clause, count in ordered
        ],
        "first_refusal_only": True,
        "caveat": (
            "a guard sequence returns on its FIRST failing clause, so these counts are of "
            "first refusals and say nothing about clauses never reached. Removing the top "
            "clause reveals the next one; it does not admit these sites."
        ),
    }


def unblocking_sequence(censuses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The ordered cascade a caller would have to clear, across one or more capabilities.

    Deliberately NOT a set of independent fixes: the clauses are returned in refusal-count order
    with an explicit statement that each is only known to be the blocker while the ones above it
    stand.
    """
    steps: list[dict[str, Any]] = []
    for entry in censuses:
        clauses = entry.get("clauses") or ()
        for rank, row in enumerate(clauses):
            steps.append(
                {
                    "capability": entry.get("capability"),
                    "clause": row.get("clause"),
                    "sites": row.get("sites"),
                    "rank_within_capability": rank,
                    "known_blocker_only_while": [c.get("clause") for c in clauses[:rank]] or None,
                }
            )
    steps.sort(key=lambda s: (-(s["sites"] or 0), str(s["capability"]), str(s["clause"])))
    return {
        "schema": "capability_unblocking_sequence_v1",
        "steps": steps,
        "reading": (
            "each step is the deciding clause for that many sites GIVEN the clauses listed "
            "in `known_blocker_only_while` still stand. Clearing one re-runs the selector "
            "and may surface a clause that was never evaluated, so treat this as a cascade "
            "to walk, not a list to divide up."
        ),
    }
