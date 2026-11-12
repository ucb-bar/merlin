"""Why a declared tier did not certify, and what a capsule may claim once one did not.

Two silences are closed here, and both were measured on the same graded run.

**A tier that did not run recorded no reason where anyone reads it.** The ladder writes a `reason`
onto every tier record it produces, and that text is genuinely good -- on one SIMT run it said the
elaborated-RTL model *"ran and neither failed nor completed OBSERVABLY: the console carries no
completion witness"*, which names the defect exactly. But the score projects a tier record down to its
bare status STRING, so the sentence stopped at the durable per-capsule file and the artifact that gets
cited carried `L3: "unavailable"` with `failure_plane: null` and `failure_detail: null`. 24 of 29 rows
looked like that. Uninterpretable, and indistinguishable from a tier nobody ever tried.

**A cert tier that ran and did not certify left the capsule reading `pass`.** Whether a tier blocks is
decided by ``required_oracle_tiers``; on that target no capsule declares its elaborated-RTL tier, so the
not-run-is-not-pass gate in :mod:`.capsule_runner` -- which walks the REQUIRED set only -- never looked at
it. Four rows carried ``L3: fail`` next to ``status: pass``, and 19 more carried ``L3: unavailable`` next
to ``status: pass``. An executed instrument that refused the program, and an instrument that produced no
answer at all, both read as a clean certification.

This module is the rule for both, and it is deliberately a pure function over a capsule result so it can
be tested by mutation rather than by running a simulator.

**Derived, not named.** Which tier certifies is NOT a name (``"L3"`` is one target's spelling and another
target's screen). A tier is treated as elaborated-RTL here when its OWN record says so --
``derived_from_rtl`` -- which is the same fact :mod:`.capsule_runner` stamps from ``cfg.rtl_tiers``, so no
target string, tier name, or engine name appears in this file.

**Fail closed.** Every branch that cannot establish a reason records an explicit ``UNKNOWN`` sentence
naming the tier and the status it came back with. Nothing here may return ``None`` for a tier that did
not pass, because ``None`` is exactly the state that made 24 rows unreadable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

#: Prefix for a reason this module had to synthesise because the record carried none. It is a SENTENCE,
#: not an empty string or ``None``: a reader (and a gate) must be able to tell "nobody wrote a reason"
#: apart from "the tier was never tried", and both apart from a real diagnosis.
UNKNOWN_PREFIX = "UNKNOWN"

#: The status a capsule takes when its elaborated-RTL tier was attempted and produced NO verdict. It
#: joins :data:`.capsule_common.NOT_MEASURED_STATUSES` for the same reason ``screened_only`` is there:
#: nothing about the submission was measured at that fidelity, so it belongs in neither the numerator
#: nor the denominator -- and must never be reported as a pass.
CERT_NOT_MEASURED = "cert_not_measured"

#: Tier statuses that are not a verdict on the submission at that fidelity.
NO_VERDICT_STATUSES = ("unavailable", "skipped")

#: The two KINDS of not-certified :func:`cert_tier_outcome` reports. They are not tier statuses: a tier
#: abandoned on the clock and a tier that refused the program are both recorded ``fail``, and only the
#: kind separates them.
REFUTED = "refuted"
NOT_MEASURED = "not_measured"

#: The one tier status that IS a verdict and IS negative -- subject to :func:`abandoned_on_budget`,
#: which is the difference between "the instrument refused this program" and "the clock ran out".
REFUTED_STATUS = "fail"

#: The vocabulary the two producers of an abandonment actually write: a subprocess timeout
#: (``Command '[...]' timed out after 1200 seconds``) and the broker's own budget wording. Substring
#: membership over a small closed set, NOT a pattern -- a too-narrow pattern silently reclassifies a
#: real abandonment as a defect, which is the failure this distinction exists to prevent, and this repo
#: gates ``import re`` in library code for exactly that reason. Kept in one place because the
#: consequence is asymmetric: measured on a 97-capsule batch, six capsules were abandoned at exactly
#: 900 s and counted against the arm's cert-clean headline as though their lowering were wrong.
BUDGET_MARKERS = ("timed out", "time budget", "budget exhausted", "out of budget")


def abandoned_on_budget(reason: object) -> bool:
    """Did this tier run out of TIME rather than answer wrongly?

    A cert the engine was killed on is not evidence of incorrectness. Treating it as one is how a
    harness limit gets written into the record as an agent defect -- a failure mode this repo has
    tracked eleven separate times.
    """
    if not isinstance(reason, str):
        return False
    low = reason.lower()
    return any(marker in low for marker in BUDGET_MARKERS)


def tier_field(record: object, key: str) -> object:
    """One field of a tier record, tolerating the bare-string form the score projects tiers into.

    A tier can reach a reader as ``{"status": "fail", "reason": ...}`` or as the string ``"fail"``.
    The string form reports no fields at all, which is a fact worth having rather than a crash.
    """
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def tier_status(record: object) -> str:
    """The status of a tier record in either carrier shape; ``"absent"`` when there is no record."""
    if record is None:
        return "absent"
    if isinstance(record, str):
        return record
    value = tier_field(record, "status")
    return str(value) if value is not None else "absent"


def is_rtl_tier(record: object) -> bool:
    """Does this record claim to come from elaborated RTL? Read off the record, never off a name."""
    return bool(tier_field(record, "derived_from_rtl"))


def reason_for(tier: str, record: object) -> str:
    """WHY this tier did not pass, as a sentence that is never empty and never ``None``.

    Order matters: the tier's own text leads whenever it has any, because that is the only half a
    submitter can act on. Everything below it is a synthesised statement of the state the record is
    actually in, and each one says which state that was.
    """
    status = tier_status(record)
    stated = tier_field(record, "reason")
    if isinstance(stated, str) and stated.strip():
        return stated.strip()
    if record is None:
        return (
            f"{UNKNOWN_PREFIX}: tier {tier} has no record in this result -- it was neither run nor "
            f"reported, and nothing states why"
        )
    if isinstance(record, str):
        return (
            f"{UNKNOWN_PREFIX}: tier {tier} reached this reader as the bare status {record!r}, "
            f"which carries no reason field"
        )
    if tier_field(record, "budget_deferred"):
        return (
            f"tier {tier} was not purchased: the certify budget stopped before it, so it is a "
            f"deliberate non-measurement rather than a verdict"
        )
    if tier_field(record, "not_applicable"):
        return f"tier {tier} is declared not applicable to this capsule and states no reason for it"
    if tier_field(record, "measured_now") is False:
        return f"tier {tier} carries a verdict earned by an earlier run and states no reason of its own"
    return (
        f"{UNKNOWN_PREFIX}: tier {tier} came back {status!r} and its record states no reason. An "
        f"instrument that produced no answer must say which one it was."
    )


def not_certified(tiers: Mapping | None, *, ladder: Iterable[str] = ()) -> dict[str, str]:
    """``{tier: why}`` for every tier that did not pass -- one entry per tier, never a ``None`` value.

    This is what travels with a verdict so ``unavailable`` stops being a bare word. ``ladder`` orders
    the result (tiers outside it follow in encounter order), so the cheap-to-expensive reading a target
    declares is preserved instead of re-sorted lexicographically.
    """
    found = {str(t): reason_for(str(t), r) for t, r in (tiers or {}).items() if tier_status(r) != "pass"}
    ordered = {t: found[t] for t in (str(x) for x in ladder) if t in found}
    ordered.update({t: why for t, why in found.items() if t not in ordered})
    return ordered


def cert_tier_outcome(tiers: Mapping | None) -> tuple[str, str, str, str] | None:
    """The worst elaborated-RTL tier outcome that is not a pass: ``(kind, tier, status, why)``.

    ``kind`` is :data:`REFUTED` or :data:`NOT_MEASURED` and is the load-bearing half. It is NOT
    recoverable from ``status``: a tier abandoned on the clock is recorded ``fail`` by its adapter and a
    tier that refused the program is recorded ``fail`` too, so a caller that re-derives the kind from the
    status string collapses the one distinction this function exists to make.

    ``None`` means every elaborated-RTL tier in this result passed, or there is no such tier at all --
    the two cases the caller must NOT act on, and which it distinguishes by looking for one itself.

    A record flagged ``not_applicable`` is exempt on purpose. That flag is how the ladder records both a
    tier a capsule's datapath cannot use and the advisory legality smoke that is documented never to
    block; turning either into a failure would break a deliberate escape hatch rather than close a
    silence. ``budget_deferred`` is exempt for the same reason: the ``screened_only`` verdict already
    covers a tier that was deliberately not bought, and reporting it twice would double-count it.
    """
    refuted: tuple[str, str, str, str] | None = None
    silent: tuple[str, str, str, str] | None = None
    for tier, record in sorted((tiers or {}).items(), key=lambda kv: str(kv[0])):
        if not is_rtl_tier(record):
            continue
        if tier_field(record, "not_applicable") or tier_field(record, "budget_deferred"):
            continue
        status = tier_status(record)
        why = reason_for(str(tier), record)
        if status == REFUTED_STATUS and abandoned_on_budget(tier_field(record, "reason")):
            # ABANDONED, not refuted. The tier is recorded `fail` by the adapter because its child
            # process did not return, but the clock is the thing that ended it, so it says nothing
            # about the program. It is a NON-measurement, and it must not become the agent's defect.
            if silent is None:
                silent = (NOT_MEASURED, str(tier), status, why)
        elif status == REFUTED_STATUS and refuted is None:
            refuted = (REFUTED, str(tier), status, why)
        elif status in NO_VERDICT_STATUSES and silent is None:
            silent = (NOT_MEASURED, str(tier), status, why)
    # A REFUTATION outranks a non-measurement: one of them is evidence about the submission.
    return refuted or silent


def qualify(result: Mapping) -> dict | None:
    """``{"status": ..., "failure": {...}}`` when ``result`` claims more than its cert tier certified.

    ``None`` when there is nothing to correct. Only a ``pass`` is ever rewritten -- a result that
    already carries a stronger, more specific failure keeps it, because the danger this closes is the
    flattering verdict, not the honest one.
    """
    if str(result.get("status")) != "pass":
        return None
    outcome = cert_tier_outcome(result.get("tiers"))
    if outcome is None:
        return None
    kind, tier, status, why = outcome
    if kind == REFUTED:
        return {
            "status": "fail",
            "failure": {
                "plane": "cert_tier",
                "category": "EXECUTED_CERT_TIER_REFUTED",
                "tier": tier,
                "tier_status": status,
                "tier_reason": why,
                "detail": (
                    f"the elaborated-RTL tier {tier} RAN and did not certify this "
                    f"capsule: {why}. A cheaper tier passing does not overturn it -- "
                    f"a screen may eliminate, it may never certify."
                ),
            },
        }
    return {
        "status": CERT_NOT_MEASURED,
        "failure": {
            "plane": "cert_not_measured",
            "category": "NOT_RUN_IS_NOT_PASS",
            "tier": tier,
            "tier_status": status,
            "tier_reason": why,
            "detail": (
                f"the elaborated-RTL tier {tier} produced NO verdict ({status}): "
                f"{why}. This is NOT a verdict on this capsule -- the cheaper tiers "
                f"that did pass are a screen, and a screen may never certify."
            ),
        },
    }


def apply(result: dict) -> dict:
    """Rewrite ``result`` in place when :func:`qualify` says its verdict outruns its evidence."""
    verdict = qualify(result)
    if verdict is None:
        return result
    result["status"] = verdict["status"]
    if not result.get("failure"):
        result["failure"] = verdict["failure"]
    return result
