"""Plateau detection — ONE definition of "did this get better", used at both granularities.

The QA loop already stopped a run that made no progress for N consecutive ROUNDS. That terminator
could not fire on a run that takes ONE round, which is what a wall-capped continuous schedule
produces: the wall is checked between rounds, so a single round runs to completion and the round-level
stall counter never reaches 2. Measured across a four-arm ladder, every arm reached its final score in
the first ~50 minutes of a 90-136 minute round and then flat-lined; one arm spent 48 further minutes
and 13 further self-checks without moving, ~28% of its wall asleep polling for verdicts that never
changed. `--plateau-rounds 3` was set on that campaign and could not have fired.

So the same question has to be asked WITHIN a round, against the agent's own self-checks, which is the
only progress signal that exists before the round ends. Both callers import from here rather than
keeping their own copy: two detectors that disagree about what "progress" means is how one of them ends
up watching the wrong thing.
"""
from __future__ import annotations


def progress_key(verdict: dict) -> tuple:
    """Progress, higher is better: ``(#passed, -total residual numeric mismatch)``.

    A non-passing capsule with NO numeric mismatch (a structural fail — it never produced comparable
    output) counts as a large residual, so a structural stall never reads as "solved". Used only to
    detect a plateau; never to grade.
    """
    tot = 0
    for pc in (verdict.get("per_capsule") or []):
        if pc.get("status") == "pass" or pc.get("pass"):
            continue
        mc = pc.get("mismatch_count")
        tot += int(mc) if isinstance(mc, int) else 1_000_000
    return (verdict.get("n_passed") or 0, -tot)


def comparable(verdict: dict, capsules_arg: str) -> bool:
    """Whether this verdict may be compared against the previous one.

    Only a FULL-CORPUS check is comparable. A subset check reports a smaller denominator, so mixing the
    two makes a plateau appear (a subset that passes everything looks like no progress) or vanish (a
    growing subset looks like progress) depending only on what the agent happened to ask for. Also
    excludes the degenerate verdicts — a build failure or an empty result is not evidence either way,
    and counting it as "no progress" would stop a round on a transient broken build.
    """
    if str(capsules_arg or "") != "all":
        return False
    if verdict.get("error") or verdict.get("build_failed") or verdict.get("no_results"):
        return False
    return bool(verdict.get("n_capsules"))


class Detector:
    """Counts consecutive comparable observations that did not improve on the best seen.

    ``limit <= 0`` disables it: :meth:`stalled` is then always False, so the caller keeps its previous
    behaviour byte-for-byte.
    """

    def __init__(self, limit: int) -> None:
        self.limit = int(limit or 0)
        self.best: tuple | None = None
        self.stall = 0

    def observe(self, verdict: dict, capsules_arg: str = "all") -> bool:
        """Fold one verdict in; returns True when the caller should stop.

        An all-pass verdict never stalls: the run is converged, and convergence is the OTHER terminator.
        """
        if self.limit <= 0 or verdict.get("all_pass") or not comparable(verdict, capsules_arg):
            return False
        key = progress_key(verdict)
        if self.best is None or key > self.best:
            self.best, self.stall = key, 0
        else:
            self.stall += 1
        return self.stalled()

    def stalled(self) -> bool:
        return self.limit > 0 and self.stall >= self.limit

    def why(self, unit: str = "checks") -> str:
        return (f"no progress (pass count + residual mismatch) for {self.stall} consecutive "
                f"comparable {unit}; best seen {self.best}")
