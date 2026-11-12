"""Which groups of a whole-program buffer the SUBMISSION's own compiler answered, and which fell back.

Split out of :mod:`merlin.compiler.lower.model.whole_program` (which re-exports both names) to keep that module
under the repository's module-size limit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: WHAT A SPLIT MEASURED BEFORE 2026-09-24 IS WORTH. The splice bound a package's tensors to this
#: program's without checking their shapes whenever the role counts happened to agree, so a group
#: could be counted as "the submission's" while its kernel read a wrongly-ordered buffer. Measured on
#: a captured ResNet-50 the moment the classification existed: of 123 bindings, 2 were identical and
#: 121 were permutations -- so the 35-of-71 split reported all session described bindings that were
#: not correct, and the honest answer to "how much of this model does the submission author" was
#: UNKNOWN, not 35. Any earlier figure is void rather than superseded.
SPLIT_UNMEASURED_BEFORE = "2026-09-24: splits reported before the binding shape check are void"


def attribution(buffer: Mapping[str, Any] | None) -> dict[str, Any]:
    """Which groups of this program the SUBMISSION's own compiler answered, and which fell back.

    Evidence about the emission as it stands, for a reader who would otherwise have no way to tell.
    The distinction is invisible in the command list -- a reference command and a spliced one sit
    side by side and both ran -- so without this, work spent on a path the compiler is never asked
    about looks exactly like work spent on one it answers. Measured on a live campaign: an authoring
    agent's first edit went into the convolution lowering, every convolution group falls back, and
    the emission was correctly unchanged.

    States the attribution and nothing about what to do with it: fallback reasons are the package's
    OWN words, quoted, and the roll-up is a count rather than a recommendation.
    """
    # At call time: that module re-exports this function, so it imports this one.
    from merlin.compiler.lower.model.whole_program import FROM_REFERENCE, FROM_SUBMISSION

    route = (buffer or {}).get("whole_program") if isinstance(buffer, Mapping) else None
    if not isinstance(route, Mapping) or not isinstance(route.get("per_group"), list):
        return {
            "schema": "whole_program_attribution_v1",
            "status": "unavailable",
            "why": "this buffer carries no whole-program route record, so nothing says which compiler answered what",
        }
    rows = [row for row in route["per_group"] if isinstance(row, Mapping)]
    # A ROW THAT NAMES NO OPERATION IS NOT A ROW ABOUT "UNKNOWN". Measured on a live campaign: every
    # row bucketed to UNKNOWN and the per-operation split never appeared, because the buffer had been
    # RETAINED from an emission by an older harness that did not record the operation. Bucketing that
    # under a name reads as a measured fact about seventy-one groups; saying the field is absent says
    # what is true, and points at the retained buffer instead of at the model.
    by_op: dict[str, dict[str, int]] | None = {}
    if not any(row.get("op") for row in rows):
        by_op = None
    else:
        for row in rows:
            op = str(row.get("op") or "UNKNOWN")
            side = str(row.get("on") or "UNKNOWN")
            by_op.setdefault(op, {FROM_SUBMISSION: 0, FROM_REFERENCE: 0})
            by_op[op][side] = by_op[op].get(side, 0) + 1
    answered = sum(1 for row in rows if row.get("on") == FROM_SUBMISSION)
    # FALLBACKS GROUPED BY THE PACKAGE'S OWN REASON. The per-op roll-up says WHICH operations fall
    # back; this says WHY, and the two answer different questions. Measured on a live campaign, the
    # 36 fallbacks are three causes of very different kinds -- an operation the package has no
    # lowering for at all, a layout it states differently from this program, and a capacity argument
    # that is correct and should stay a refusal -- and a reader who saw only "36" would not be able
    # to tell the one worth closing from the one that must not be.
    reasons: dict[str, dict[str, Any]] = {}
    for row in rows:
        why = str(row.get("why") or "")
        if row.get("on") == FROM_SUBMISSION or not why:
            continue
        # KEYED ON THE CAUSE, NOT THE SENTENCE. The sentence carries the package's own shapes and
        # tensor names, so nineteen refusals of one cause read as nineteen causes of one -- measured,
        # and it buried the largest single cause beneath a smaller one whose text happened to repeat.
        key = str(row.get("cause") or why)
        seen = reasons.setdefault(
            key, {"cause": row.get("cause"), "groups": 0, "ops": {}, "examples": [], "why": why, "_sentences": {}}
        )
        seen["groups"] += 1
        seen["_sentences"][why] = seen["_sentences"].get(why, 0) + 1
        if row.get("op"):
            seen["ops"][str(row["op"])] = seen["ops"].get(str(row["op"]), 0) + 1
        if len(seen["examples"]) < 3:
            seen["examples"].append(row.get("group"))
    # THE QUOTED EXAMPLE IS THE ONE THE BUCKET IS MOSTLY MADE OF. A cause collects refusals whose
    # sentences differ, and quoting whichever arrived first states the wrong reason over the right
    # count: measured on a live campaign, a 17-group `package_declined` bucket quoted a stem
    # convolution's capacity argument while 16 of its 17 rows were an operation the package had no
    # lowering for at all -- so the one refusal that should STAY was presented as the whole bucket.
    # The full distribution travels beside it, because a bucket of several sentences is a different
    # fact from a bucket of one and a single quote cannot distinguish them.
    for detail in reasons.values():
        sentences = detail.pop("_sentences")
        detail["why"] = max(sentences.items(), key=lambda item: item[1])[0]
        detail["distinct_sentences"] = [
            {"count": count, "why": sentence}
            for sentence, count in sorted(sentences.items(), key=lambda item: -item[1])
        ]
    return {
        "schema": "whole_program_attribution_v1",
        "status": "derived",
        "groups": len(rows),
        "on_submission": answered,
        "on_reference": len(rows) - answered,
        "by_op": by_op,
        # `why` is ONE verbatim sentence -- the one MOST of the bucket's rows gave, never a paraphrase
        # and never a merge; `distinct_sentences` carries the rest with their counts.
        "fallback_by_reason": [
            dict(detail) for _key, detail in sorted(reasons.items(), key=lambda item: -item[1]["groups"])
        ],
        "grouping": (
            "by cause"
            if any(row.get("cause") for row in rows)
            else "by message text; these rows predate the cause field, so refusals differing only in "
            "their quoted shapes appear as separate causes"
        ),
        # The package's own words, unparaphrased, and absent rather than empty when the group was
        # answered -- "no reason given" and "a reason nobody recorded" are different facts.
        "per_group": [
            {
                "group": row.get("group"),
                "op": row.get("op"),
                "answered_by": row.get("on"),
                **({"why": row["why"]} if row.get("why") else {}),
            }
            for row in rows
        ],
        "means": (
            "a group answered by the submission was lowered by the compiler under test; one answered "
            "by the reference was stated by the harness because the submission did not lower it, for "
            "the quoted reason. Both run; the emission does not distinguish them."
        ),
    }
