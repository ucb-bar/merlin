"""Which public capsules one self-check request grades.

``--capsules`` takes ``all``, comma-separated names, or a selector that names a set by what is already
known: ``failing`` is what the author's previous self-check did not pass, and ``changed`` is what an
edit changed, decided by exact-byte simulator certificates (:mod:`.selfcheck_changed`). A selector only
chooses capsules; it never relaxes a verdict and certifies nothing on its own.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class Selection:
    """What a request grades: ``want`` (``None`` = every public capsule), or the exit code of an answer
    the selector already gave (``answered``), in which case nothing is graded. A ``changed`` selection
    joins its own classification to the verdict of what it graded (:meth:`report`)."""

    want: set[str] | None
    answered: int | None = None
    changed: object | None = None

    def report(self, out: dict) -> dict:
        return out if self.changed is None else self.changed.report(out)


def _previously_failing() -> tuple:
    """``(set of capsule names that did not pass last time, why not)`` for ``--capsules failing``.

    THE CHEAP LOOP SHOULD BE THE EASY ONE. Re-checking only what is still broken was always possible
    -- the docstring above spells the comma-separated form -- but it required the agent to read a
    verdict, extract the names and paste them, so the reflex was `--capsules all`. Measured on one
    6.1 h run: 25 full sweeps, ~6 min each, while a typical edit leaves 82% of emitted programs
    byte-identical.

    Read from the LAST SELF-CHECK's own output, not from the harness verdict: this is the agent's own
    loop, the file is one it already owns, and taking the harness's grade would make a self-check
    depend on a grade that may be many minutes stale.

    Never guesses. If no previous self-check is on disk, the answer is empty with a reason -- returning
    "nothing is failing" would silently grade zero capsules and report success.
    """
    out = Path("selfcheck_out")
    latest, newest = None, -1.0
    try:
        completed = [
            Path(".qa_channel") / f"resp_{done.name[len('done_') :]}.json"
            for done in Path(".qa_channel").glob("done_*")
        ]
        for cand in list(out.glob("*.json")) + [Path("selfcheck.json")] + completed:
            if cand.is_file() and cand.stat().st_mtime > newest:
                latest, newest = cand, cand.stat().st_mtime
    except OSError:
        pass
    if latest is None:
        return set(), "no previous self-check output was found on disk"
    try:
        doc = json.loads(latest.read_text())
    except (OSError, ValueError) as exc:
        return set(), f"the previous self-check output could not be read ({type(exc).__name__})"
    rows = doc.get("per_capsule")
    if not isinstance(rows, list) or not rows:
        return set(), f"{latest} records no per-capsule rows"
    names = {r.get("capsule") for r in rows if isinstance(r, dict) and not r.get("pass") and r.get("capsule")}
    if not names:
        return set(), (
            f"every capsule in {latest} passed, so there is nothing to re-check; "
            "use --capsules all to confirm the whole suite"
        )
    return names, ""


def select_capsules(a, sub, context, capsules_root, contract, adapters, combined, target, barrier_tier) -> Selection:
    """The capsules ``a.capsules`` names (see the module docstring)."""
    if a.capsules.strip().lower() == "changed":
        from .selfcheck_changed import select_changed

        return select_changed(
            a,
            sub=sub,
            context=context,
            capsules_root=capsules_root,
            contract=contract,
            adapters=adapters,
            combined=combined,
            target=target,
            barrier_tier=barrier_tier,
        )
    if a.capsules.strip().lower() == "failing":
        want, why = _previously_failing()
        if not want:
            print(
                json.dumps(
                    {
                        "error": "--capsules failing: " + why,
                        "hint": "run once with --capsules all to establish what is failing",
                    },
                    indent=2,
                )
            )
            return Selection(None, answered=2)
        return Selection(want)
    return Selection(None if a.capsules == "all" else {s.strip() for s in a.capsules.split(",") if s.strip()})
