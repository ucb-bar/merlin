"""Per-round artifact fingerprints — tell an INERT edit apart from a WRONG one.

In an iterative agentic compile loop a graded number that does not move has **four** distinct causes
that all present identically:

1. the agent changed nothing that reaches the graded submission;
2. it changed the submission, but nothing changed in what the emitter *produced* for a capsule;
3. it changed the emitted artifact and the result is equally wrong;
4. it changed the artifact and the **oracle is insensitive** for that capsule.

Without evidence, distinguishing these is guesswork. A measured case: an arm-4 run held 12 capsules at
byte-identical mismatch counts for six consecutive rounds while the agent edited its emitter every round
and self-reported substantive fixes. Three separate mechanisms were proposed from reading intermediate
artifacts by hand, and all three were wrong. The information needed to answer it in one line -- did these
bytes change? -- was never recorded.

This module records it. It compares **bytes**, never content: it does not parse an artifact, does not know
an opcode, an ISA, a stage name or a target name, so it cannot be too narrow and cannot go stale when a
backend changes what it emits. A target that emits three files and one that emits seven both work with no
edit here. Anything unreadable is recorded as ``UNKNOWN`` and surfaced -- never silently skipped.

It is **diagnostic, not a gate**: it reports movement, it never passes or fails a run. A threshold here
would create exactly the incentive this apparatus exists to defeat (make the number move by deleting the
hard case). It is also written **out of band** -- into the run dir, which the sandbox denies -- because
telling the agent "your edit was inert" is *feedback*, and feedback defines an arm. Surfacing it in the
agent's verdict would silently turn arm N into arm N+1 and invalidate cross-run comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from merlin.common.provenance import UNKNOWN, file_digest

# What a per-capsule or per-submission fingerprint set can say about one round versus the previous one.
NEW = "new"                # first round this artifact existed
CHANGED = "changed"        # bytes differ from the previous round
UNCHANGED = "unchanged"    # bytes identical to the previous round
ABSENT = "absent"          # existed before, does not now (emission stopped producing it)

# Statuses that are NOT the agent attempting this capsule and getting it wrong, so a frozen artifact
# under one of them is not actionable. A whole-model capsule deferred by its op-pass gate never emitted;
# naming it "failing and frozen" would point at work the agent cannot do yet. Anything NOT listed here
# counts as an attempt -- fail closed, so a status added later is surfaced rather than silently dropped.
NOT_AN_ATTEMPT = frozenset({"pass", "gated", "skipped", "incomplete",
                            "not_gradeable_no_oracle", None})

# The diagnosis for a round, derived from (submission moved?) x (emitted artifacts moved?).
NO_SUBMISSION_CHANGE = "no_submission_change"
EMIT_INSENSITIVE_TO_EDIT = "emit_insensitive_to_edit"
EMIT_MOVED = "emit_moved"
NOTHING_TO_COMPARE = "nothing_to_compare"


def _walk(root: Path) -> list[Path]:
    """Every regular file under *root*, sorted, with build/VCS scratch excluded.

    Scratch dirs are excluded because their churn is not the agent's edit: a ``__pycache__`` rewrite on
    every import would report CHANGED forever and make the signal useless.
    """
    skip = {"__pycache__", ".git", "build", ".pytest_cache", ".mypy_cache"}
    out: list[Path] = []
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in skip for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return out


def fingerprint_tree(root: "str | Path") -> dict[str, str]:
    """``{relative path: sha256}`` for every file under *root*.

    Relative keys, so the same tree copied to a different scratch dir each round fingerprints the same --
    the graded submission IS copied per round, and an absolute-path key would report CHANGED every time.
    """
    root = Path(root)
    return {str(p.relative_to(root)): file_digest(p) for p in _walk(root)}


def compare(now: Mapping[str, str], before: Mapping[str, str] | None) -> dict[str, str]:
    """Per-path verdict of *now* against *before* (``None`` == no previous round)."""
    if before is None:
        return {k: NEW for k in now}
    out: dict[str, str] = {}
    for k, v in now.items():
        if k not in before:
            out[k] = NEW
        elif before[k] == v and v != UNKNOWN:
            out[k] = UNCHANGED
        else:
            out[k] = CHANGED
    for k in before:
        if k not in now:
            out[k] = ABSENT
    return out


def _moved(verdicts: Mapping[str, str]) -> bool:
    return any(v in (CHANGED, NEW, ABSENT) for v in verdicts.values())


def build(*, submission_dir: "str | Path", emitted_roots: Mapping[str, "str | Path"],
          previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The ledger for one round.

    ``submission_dir`` is the exact tree that was graded (not the live workspace -- the graded copy, so
    the fingerprint is of what the oracle actually read). ``emitted_roots`` maps a capsule name to the
    directory holding whatever that capsule's emit path produced. ``previous`` is the prior round's
    ledger, or ``None``.

    The ``diagnosis`` separates causes 1-3 above. Cause 4 needs a perturbation probe, which costs oracle
    time and is deliberately not done here; ``emit_moved`` is the state in which to reach for it.
    """
    sub_now = fingerprint_tree(submission_dir)
    sub_prev = (previous or {}).get("submission_files")
    sub_verdicts = compare(sub_now, sub_prev)

    caps: dict[str, Any] = {}
    prev_caps = (previous or {}).get("capsules") or {}
    for name, root in sorted(emitted_roots.items()):
        now = fingerprint_tree(root)
        pv = (prev_caps.get(name) or {}).get("files")
        caps[name] = {"files": now, "verdicts": compare(now, pv), "root": str(root)}

    sub_moved = _moved(sub_verdicts) if sub_prev is not None else None
    any_emit_moved = any(_moved(c["verdicts"]) for c in caps.values()) if prev_caps else None

    if sub_prev is None:
        diagnosis = NOTHING_TO_COMPARE
    elif not sub_moved:
        diagnosis = NO_SUBMISSION_CHANGE
    elif not any_emit_moved:
        diagnosis = EMIT_INSENSITIVE_TO_EDIT
    else:
        diagnosis = EMIT_MOVED

    return {
        "submission_dir": str(submission_dir),
        "submission_files": sub_now,
        "submission_verdicts": sub_verdicts,
        "submission_moved": sub_moved,
        "capsules": caps,
        "emit_moved": any_emit_moved,
        "diagnosis": diagnosis,
        "n_unreadable": sum(1 for v in sub_now.values() if v == UNKNOWN)
                        + sum(1 for c in caps.values() for v in c["files"].values() if v == UNKNOWN),
        "note": ("Diagnostic only -- never gates a run. 'emit_insensitive_to_edit' means the graded "
                 "submission changed while every emitted artifact stayed byte-identical: the edit did "
                 "not reach emission, so no numeric verdict could have moved."),
    }


def capsules_that_did_not_move(ledger: Mapping[str, Any]) -> list[str]:
    """Capsules whose emitted artifacts are byte-identical to the previous round."""
    return sorted(name for name, c in (ledger.get("capsules") or {}).items()
                  if c.get("verdicts") and not _moved(c["verdicts"]))


def failing_and_frozen(ledger: Mapping[str, Any], verdict: Mapping[str, Any]) -> list[str]:
    """Capsules that FAILED this round **and** whose emitted bytes did not change.

    This is the actionable class, and the run-level ``diagnosis`` cannot express it. Replaying a real
    six-round plateau showed why: every round reported ``emit_moved`` because *some* capsule moved, while
    26 of 27 were frozen -- and in the last round the single capsule that moved was one that already
    passed. A run-level boolean says "something changed" and hides that nothing changed where it mattered.

    A capsule in this list cannot possibly produce a different number next round unless the emitter starts
    reaching it, so it is where the agent (or we) must look.
    """
    frozen = set(capsules_that_did_not_move(ledger))
    if not frozen:
        return []
    failing = {c.get("capsule") for c in (verdict.get("per_capsule") or [])
               if c.get("capsule") and c.get("status") not in NOT_AN_ATTEMPT}
    return sorted(frozen & failing)


def frozen_fraction(ledger: Mapping[str, Any]) -> float | None:
    """Share of capsules whose emission did not move (``None`` on the first round).

    Reported instead of a bare boolean so a plateau is visible as it forms rather than after six rounds.
    """
    caps = ledger.get("capsules") or {}
    comparable = [c for c in caps.values() if c.get("verdicts") and c.get("files")]
    if not comparable or ledger.get("emit_moved") is None:
        return None
    return len(capsules_that_did_not_move(ledger)) / len(comparable)


def summarize(ledger: Mapping[str, Any]) -> str:
    """One line for a log or a round banner."""
    caps = ledger.get("capsules") or {}
    still = capsules_that_did_not_move(ledger)
    frac = frozen_fraction(ledger)
    parts = [f"diagnosis={ledger.get('diagnosis')}",
             f"submission_moved={ledger.get('submission_moved')}",
             f"capsules={len(caps)}", f"unmoved={len(still)}"]
    if frac is not None:
        parts.append(f"frozen={frac:.0%}")
    if ledger.get("n_unreadable"):
        parts.append(f"unreadable={ledger['n_unreadable']}")
    return "stage_ledger: " + " ".join(parts)
