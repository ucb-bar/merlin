"""Has this run stopped making progress, and is what remains reachable at all?

WHY THIS EXISTS AS A SEPARATE, TESTABLE THING. A plateau detector already lived in the capsule-bench
driver, and it could not fire. It counted consecutive ROUNDS with no progress, and the default schedule
is ``continuous`` -- one long session, re-graded underneath the agent, which produces exactly one round.
Its threshold was therefore unreachable in the mode every real run uses, and it was opt-in on top
(``--plateau-rounds`` defaults to 0). Measured on ``merlincirct_g4p1_biasabi_20260906``: the score
reached 92/96 after 2.19 h and the run continued for 3.91 h more -- 64% of a 6.10 h run -- without the
score moving and without anything saying so.

So the unit here is a GRADE, which the continuous schedule does produce, and detection is separated
from termination. Detection is free and always available; stopping a run is a policy the operator opts
into. That split matters: a detector that can only speak by killing a run will be left switched off,
which is exactly how the previous one ended up unreachable.

**Two independent facts, and the second is the stronger one.** "No progress for N grades" is about
momentum and can be wrong -- an agent may be mid-way through a large refactor. "These capsules have
never passed in this run, in any grade" is about reachability, and on the run above it identified all
four remaining capsules: ``GN0_layernorm_host_only_bf16_pt``, ``M2_microvit_gemmini``,
``M3_host_island_seam_gemmini`` and ``SY_micro_model``, none of which has passed in any run on disk.
Both are reported; neither is presented as the other.

**Fails towards "not stuck".** Too little history, missing counts, an unparseable grade -- every one of
them yields ``stuck=False`` with a reason. A detector that declares a plateau it cannot evidence would
cut productive runs, and the first thing an operator does with a false positive is switch it off.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MIN_GRADES", "Plateau", "progress_key", "assess"]

#: Grades of history below which no plateau may be declared. One grade is not a trend, and the cost of
#: a false positive is a productive run cut short.
MIN_GRADES = 3

#: What a non-passing capsule with no numeric mismatch contributes to the residual. A structural
#: failure has nothing to count, and treating it as zero would make a structural stall read as solved.
_STRUCTURAL_RESIDUAL = 1_000_000


@dataclass
class Plateau:
    """What the grade history supports saying. ``stuck`` is never true without ``reason``."""

    stuck: bool = False
    reason: str = ""
    stalled_grades: int = 0
    n_grades: int = 0
    best_passed: int = 0
    latest_passed: int = 0
    n_capsules: int = 0
    #: Capsules that have not passed in ANY grade of this run -- reachability, not momentum.
    never_passed: tuple = ()
    #: Capsules that passed at some point and do not pass now: a regression the agent may not have seen.
    regressed: tuple = ()
    notes: list = field(default_factory=list)

    def sentence(self) -> str:
        """One plain statement for whoever reads it -- the agent, a log, or a run record."""
        if not self.n_grades:
            return "no grades yet, so nothing can be said about progress."
        head = (f"{self.latest_passed}/{self.n_capsules} passing after {self.n_grades} grade(s); "
                f"best so far {self.best_passed}.")
        if self.stalled_grades:
            head += f" No improvement for the last {self.stalled_grades} grade(s)."
        if self.never_passed:
            head += (f" {len(self.never_passed)} capsule(s) have NEVER passed in this run"
                     f" ({', '.join(self.never_passed[:4])}"
                     f"{', …' if len(self.never_passed) > 4 else ''})"
                     " — if repeated attempts have not moved them, the remaining work may not be"
                     " reachable by more of the same approach.")
        if self.regressed:
            head += (f" ⚠ {len(self.regressed)} capsule(s) passed earlier and do not now"
                     f" ({', '.join(self.regressed[:4])}) — check for a regression you introduced.")
        return head


def _statuses(grade) -> dict:
    """``{capsule: passed?}`` from either verdict shape, or ``{}``.

    Two shapes exist in this repo's verdicts -- a mapping of name to status string, and a list of
    per-capsule rows -- and a reader that understands only one silently sees an empty suite.
    """
    rows = (grade or {}).get("per_capsule")
    if isinstance(rows, dict):
        return {str(k): (v == "pass") for k, v in rows.items() if isinstance(k, str)}
    if isinstance(rows, list):
        out = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("capsule") or row.get("name")
            if not name:
                continue
            status = row.get("status")
            out[str(name)] = (row.get("pass") is True) if status is None else (status == "pass")
        return out
    return {}


def progress_key(grade) -> tuple:
    """``(#passed, -residual mismatch)`` for one grade; higher is better.

    The residual keeps a run that is reducing numeric error off the plateau even while its pass count
    is flat -- which is the shape of real progress on a hard capsule, and the reason a pass-count-only
    detector cuts productive runs.
    """
    grade = grade or {}
    residual = 0
    rows = grade.get("per_capsule")
    rows = list(rows.values()) if isinstance(rows, dict) else (rows if isinstance(rows, list) else [])
    for row in rows:
        if not isinstance(row, dict):
            residual += _STRUCTURAL_RESIDUAL          # a status string carries no mismatch to count
            continue
        if row.get("status") == "pass" or row.get("pass") is True:
            continue
        count = row.get("mismatch_count")
        residual += int(count) if isinstance(count, int) else _STRUCTURAL_RESIDUAL
    return (int(grade.get("n_passed") or 0), -residual)


def assess(grades, *, stall_threshold: int = 4) -> Plateau:
    """What ``grades`` (oldest first) supports saying about progress. Never raises.

    ``stall_threshold`` is how many consecutive grades without improvement constitute a plateau. It is
    a judgement about patience, not a measurement, so the caller owns it.
    """
    grades = [g for g in (grades or []) if isinstance(g, dict)]
    out = Plateau(n_grades=len(grades))
    if not grades:
        out.reason = "no grades have been recorded yet"
        return out

    latest = grades[-1]
    out.latest_passed = int(latest.get("n_passed") or 0)
    out.n_capsules = int(latest.get("n_capsules") or 0)
    out.best_passed = max(int(g.get("n_passed") or 0) for g in grades)

    best, stalled = None, 0
    for grade in grades:
        key = progress_key(grade)
        if best is None or key > best:
            best, stalled = key, 0
        else:
            stalled += 1
    out.stalled_grades = stalled

    ever, now = set(), _statuses(latest)
    for grade in grades:
        ever |= {name for name, passed in _statuses(grade).items() if passed}
    out.never_passed = tuple(sorted(name for name, passed in now.items()
                                    if not passed and name not in ever))
    out.regressed = tuple(sorted(name for name, passed in now.items()
                                 if not passed and name in ever))

    if len(grades) < MIN_GRADES:
        out.reason = (f"only {len(grades)} grade(s) of history; {MIN_GRADES} are needed before a "
                      f"plateau can be evidenced")
        return out
    if not out.n_capsules:
        out.reason = "the latest grade records no capsule count, so progress cannot be measured"
        return out
    if out.stalled_grades >= max(1, stall_threshold):
        out.stuck = True
        out.reason = (f"no improvement in pass count or residual mismatch for {out.stalled_grades} "
                      f"consecutive grades")
        return out
    out.reason = (f"progress within the last {max(1, stall_threshold)} grade(s); not a plateau")
    return out
