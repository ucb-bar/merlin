"""Route a family's frozen acceptance contract to the analyzer it names.

Three claim analyzers exist and each is reachable only by name from its own family's contract, so
adding a family meant adding a caller -- and nobody added one.  The consequence was silent: `PM`
and `PV` shipped declaring `PREDICTS` with a frozen contract that no code path ever evaluated, so
they produced measurements and no verdict, while reading in every artifact exactly like `PK`, which
is evaluated.  A claim nothing can decide is not a claim.

Dispatch is on the contract's own `analyzer` identity, never on the family name: the family is a
label, the analyzer identity is the thing the contract froze, and it already carries a version so a
retuned analyzer cannot silently score an old declaration.  An identity this registry does not know
is REFUSED and named -- never routed to a "closest" analyzer, and never quietly skipped, which is
the failure this module exists to end.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

REFUSED = "REFUSED"


def _registry() -> dict[str, Callable[..., dict]]:
    """Imported lazily so one broken analyzer cannot make the whole dispatcher unimportable."""
    table: dict[str, Callable[..., dict]] = {}
    try:
        import perf_pk_claim as PK
        table[PK._ACCEPTANCE_BASE["analyzer"]] = PK.analyze_pk_claim
    except Exception:  # noqa: BLE001 - an absent analyzer is reported at dispatch, not at import
        pass
    try:
        import perf_affine_claim as AF
        table[AF.ANALYZER] = AF.analyze_affine_claim
    except Exception:  # noqa: BLE001
        pass
    try:
        import perf_pr_claim as PR
        for name in ("ANALYZER", "_ANALYZER"):
            identity = getattr(PR, name, None)
            if isinstance(identity, str) and identity:
                table[identity] = PR.analyze_pr_claim
                break
    except Exception:  # noqa: BLE001
        pass
    return table


def declared_analyzer(descriptors: Sequence[Mapping[str, Any]]) -> str | None:
    """The single analyzer identity the cohort's contracts agree on, or None if they do not."""
    seen = set()
    for d in descriptors:
        if not isinstance(d, Mapping):
            return None
        acceptance = ((d.get("performance") or {}) if isinstance(d.get("performance"), Mapping)
                      else {}).get("acceptance")
        if not isinstance(acceptance, Mapping):
            return None
        seen.add(str(acceptance.get("analyzer")))
    return seen.pop() if len(seen) == 1 else None


def analyze(descriptors: object, results: object) -> dict[str, Any]:
    """Decide one family by handing it to the analyzer its own contract names."""
    if not isinstance(descriptors, Sequence) or not descriptors:
        return {"verdict": REFUSED, "reason": "no capsule descriptors were supplied"}
    identity = declared_analyzer(descriptors)
    if identity is None:
        return {"verdict": REFUSED,
                "reason": ("the cohort does not agree on one frozen analyzer identity, so there is "
                           "no single procedure that decides it")}
    table = _registry()
    if identity not in table:
        return {"verdict": REFUSED, "declared_analyzer": identity,
                "reason": (f"no analyzer registered for {identity!r}; the contract names a "
                           f"procedure this build cannot run, so the claim is undecided rather "
                           f"than assumed"),
                "registered": sorted(table)}
    verdict = table[identity](descriptors, results)
    if isinstance(verdict, Mapping):
        return {**verdict, "declared_analyzer": identity}
    return {"verdict": REFUSED, "declared_analyzer": identity,
            "reason": "the analyzer returned something other than a verdict mapping"}
