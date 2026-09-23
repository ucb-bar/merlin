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
    #: EVERY clause this site fails, when the selector was asked for all of them instead of stopping
    #: at the first. ``None`` means the caller reported a first refusal only, which is the short-circuit
    #: shape the module docstring warns about. Supplying it is what lets :func:`census` report the
    #: caveat as a MEASURED property of its input rather than a blanket assumption — measured on the
    #: recorded ResNet-50 emission, all 53 convolutions reported one blocker and were failing three.
    clauses: tuple[str, ...] | None = None


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
        clauses = tuple(str(c) for c in row[4]) if len(row) > 4 and row[4] is not None else None
        out.append(RefusalSite(str(site), admitted, clause, detail, clauses))
    return out


def _blocking(row: RefusalSite) -> tuple[str, ...]:
    """Every clause this site is known to fail — its whole stack when it supplied one, else the single
    clause that decided it. A site that supplied a stack but omitted its own deciding clause still
    counts that clause, so the two shapes cannot disagree about what blocked it."""
    if row.clauses is None:
        return (row.clause,)
    return row.clauses if row.clause in row.clauses else (row.clause, *row.clauses)


def census(capability: str, sites: Iterable[RefusalSite]) -> dict[str, Any]:
    """Aggregate per-site verdicts for one capability into a clause census.

    Whether the census is of FIRST refusals is a property of what the caller supplied, not an
    assumption: a site that reports its whole failing stack (``RefusalSite.clauses``) is counted under
    every clause blocking it, and ``first_refusal_only`` is true only while some refused site did not.
    ``clause_depth`` carries the shape -- all-ones for a first-refusal census, and for a complete one
    the number that says how far clearing the top clause would actually get.
    """
    rows = _rows(sites)
    admitted = [r for r in rows if r.admitted]
    refused = [r for r in rows if not r.admitted]
    # MEASURED, not assumed: the caveat holds only while some refused site reported a first refusal
    # only. A census every one of whose sites supplied its whole stack is complete, and saying
    # otherwise would understate what is known — which is the opposite error, but still an error.
    partial = [r for r in refused if r.clauses is None]
    counts: Counter[str] = Counter()
    for r in refused:
        counts.update(set(_blocking(r)))
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
                "example_sites": [r.site for r in refused if clause in _blocking(r)][:5],
                "example_detail": next((dict(r.detail) for r in refused if clause in _blocking(r) and r.detail), None),
            }
            for clause, count in ordered
        ],
        "first_refusal_only": bool(partial),
        # How deep each refused site's KNOWN stack is. A census of first refusals is all-ones by
        # construction; anything else is the shape a complete stack has, and it is the number that
        # tells a reader how far "fix the top clause" would actually get.
        "clause_depth": dict(sorted(Counter(len(set(_blocking(r))) for r in refused).items())),
        "sites_with_complete_stack": len(refused) - len(partial),
        "caveat": (
            (
                "a guard sequence returns on its FIRST failing clause, so these counts are of "
                "first refusals and say nothing about clauses never reached. Removing the top "
                "clause reveals the next one; it does not admit these sites."
            )
            if partial
            else (
                "every refused site reported EVERY clause it fails, so these counts are complete: "
                "a site appears under each clause blocking it, the shares therefore sum above 1, "
                "and a clause cleared here does not reveal a hidden one underneath."
            )
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
        # A census whose sites each reported their WHOLE stack is not a cascade: nothing here is
        # conditional on a clause above it, and every listed clause has to be cleared before any of
        # those sites is admitted. Saying "walk this in order" about such a census would invite the
        # same wrong prediction from the other direction.
        cascade = bool(entry.get("first_refusal_only", True))
        for rank, row in enumerate(clauses):
            steps.append(
                {
                    "capability": entry.get("capability"),
                    "clause": row.get("clause"),
                    "sites": row.get("sites"),
                    "rank_within_capability": rank,
                    "known_blocker_only_while": (
                        ([c.get("clause") for c in clauses[:rank]] or None) if cascade else None
                    ),
                    "all_must_clear": not cascade,
                }
            )
    steps.sort(key=lambda s: (-(s["sites"] or 0), str(s["capability"]), str(s["clause"])))
    return {
        "schema": "capability_unblocking_sequence_v1",
        "steps": steps,
        "reading": (
            "a step with `known_blocker_only_while` is the deciding clause for that many sites GIVEN "
            "those clauses still stand: clearing it re-runs the selector and may surface a clause that "
            "was never evaluated, so that part is a cascade to walk, not a list to divide up. A step "
            "marked `all_must_clear` came from a census whose sites each reported every clause they "
            "fail, so nothing is hidden underneath it and every such clause must be cleared before any "
            "of its sites is admitted."
        ),
    }
