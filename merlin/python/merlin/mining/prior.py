"""Turn a corpus of ATTEMPTS into a prior the search can rank by.

A search that starts every run from uniform ignorance re-learns what a previous campaign already
measured. The evidence exists: a transform ledger records, per attempt, which strategy was applied and
how it turned out (improved / regressed / correct-but-no-gain / incorrect / compile_error). The
per-strategy improvement rate is exactly the ``evidence_prior`` an action wants.

Three things this refuses to do, because each would make the prior look more informative than it is:

* **It does not guess the correspondence.** A ledger strategy is prose written for another campaign;
  a merlin ``action_family`` is a seam in this compiler. Nothing derives one from the other, so the
  mapping is DECLARED by the caller and an action with no declared strategy gets ``None`` — the
  unmeasured band — rather than the global base rate. A base rate applied to everything ranks nothing
  and reads as evidence.
* **It does not report a rate from a handful of attempts.** One improvement in two tries is not a 50%
  prior. Estimates below ``min_attempts`` come back ``None``, and every rate carries the ``n`` it was
  computed from so a reader can see what is behind it.
* **It does not silently treat a missing ledger as an empty one.** No ledger is UNKNOWN; a ledger
  that parsed to zero usable rows says so.

The ledger path is injected or read from the environment — never a literal, since the corpus lives
outside this repo and is pinned separately.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

#: Environment seam naming the transform ledger (JSONL, one attempt per line).
LEDGER_ENV = "MERLIN_TRANSFORM_LEDGER"

#: Outcome meaning the attempt made the kernel faster AND kept it correct. Everything else is not an
#: improvement -- including ``correct_no_gain``, which is a real negative result and must not be
#: folded in with the successes to flatter the rate.
IMPROVED = "improved"

#: An attempt that never produced a runnable kernel still consumed a candidate slot, so it belongs in
#: the denominator. Dropping compile errors would report the rate among attempts that happened to
#: build, which is not the rate a planner faces.
_COUNTED_OUTCOMES = ("improved", "regressed", "correct_no_gain", "incorrect", "compile_error")


@dataclass(frozen=True)
class StrategyEvidence:
    """What the corpus measured about one strategy."""
    strategy: str
    attempts: int
    improved: int
    outcomes: dict[str, int] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return (self.improved / self.attempts) if self.attempts else 0.0


@dataclass(frozen=True)
class OutcomePrior:
    """Per-strategy improvement rates plus the base rate they sit against.

    ``total_*`` describe the LABELLED population only — the attempts that name a strategy and are
    therefore the ones a per-strategy rate is computed from. The unlabelled attempts are counted
    separately rather than discarded, because whether they resemble the labelled ones is the question
    that decides how far these rates generalize, and a prior that hides its denominator invites a
    reader to apply it to the whole corpus.
    """
    by_strategy: dict[str, StrategyEvidence]
    total_attempts: int
    total_improved: int
    source: str
    unusable_rows: int = 0
    #: Attempts with a countable outcome but NO strategy label, and how many of them improved.
    unlabelled_attempts: int = 0
    unlabelled_improved: int = 0

    @property
    def base_rate(self) -> float | None:
        """Improvement rate among LABELLED attempts. Not the corpus rate — see ``corpus_base_rate``."""
        return (self.total_improved / self.total_attempts) if self.total_attempts else None

    @property
    def corpus_base_rate(self) -> float | None:
        """Improvement rate over every countable attempt, labelled or not."""
        n = self.total_attempts + self.unlabelled_attempts
        return ((self.total_improved + self.unlabelled_improved) / n) if n else None

    @property
    def unlabelled_rate(self) -> float | None:
        return ((self.unlabelled_improved / self.unlabelled_attempts)
                if self.unlabelled_attempts else None)

    def coverage_problems(self, *, tolerance: float = 0.25) -> tuple[str, ...]:
        """Why these rates may not describe the corpus they came from.

        The check that matters is whether the attempts WITHOUT a strategy label improve at a
        materially different rate from those with one. If they do, the labelled set is not a random
        sample and its rates must be quoted as conditional on being labelled — otherwise a planner
        reads a subset's behaviour as the corpus's.
        """
        out: list[str] = []
        total = self.total_attempts + self.unlabelled_attempts
        if self.unlabelled_attempts and total:
            share = self.unlabelled_attempts / total
            if share >= 0.05:
                out.append(f"{self.unlabelled_attempts}/{total} attempts ({share:.1%}) carry NO "
                           f"strategy label and contribute to no per-strategy rate")
        lab, unlab = self.base_rate, self.unlabelled_rate
        if lab is not None and unlab is not None and lab > 0:
            if abs(unlab - lab) / lab >= tolerance:
                out.append(f"labelled attempts improve at {lab:.1%} but unlabelled ones at "
                           f"{unlab:.1%} — the labelled set is NOT a random sample, so these rates "
                           f"are conditional on having a label, not properties of the corpus")
        return tuple(out)

    def rate_for(self, strategy: str | None, *, min_attempts: int = 5) -> float | None:
        """The measured improvement rate for ``strategy``, or None when nobody measured it enough.

        None is returned for an unknown strategy AND for one with too few attempts. Both are honestly
        "unmeasured"; neither is the base rate, which would assert a belief the corpus does not hold
        about this particular strategy.
        """
        if not strategy:
            return None
        ev = self.by_strategy.get(strategy)
        if ev is None or ev.attempts < min_attempts:
            return None
        return ev.rate

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "total_attempts": self.total_attempts,
                "total_improved": self.total_improved, "base_rate": self.base_rate,
                "corpus_base_rate": self.corpus_base_rate,
                "unlabelled_attempts": self.unlabelled_attempts,
                "unlabelled_improved": self.unlabelled_improved,
                "unlabelled_rate": self.unlabelled_rate,
                "coverage_problems": list(self.coverage_problems()),
                "unusable_rows": self.unusable_rows,
                "strategies": {k: {"attempts": v.attempts, "improved": v.improved,
                                   "rate": round(v.rate, 4), "outcomes": v.outcomes}
                               for k, v in sorted(self.by_strategy.items())}}


def ledger_path(explicit: str | Path | None = None) -> Path | None:
    """Explicit > ``$MERLIN_TRANSFORM_LEDGER``. None when neither names one -- which is UNKNOWN, and
    the caller must not turn it into an empty ledger."""
    if explicit:
        return Path(explicit)
    env = os.environ.get(LEDGER_ENV)
    return Path(env) if env else None


def load_outcome_prior(path: str | Path | None = None, *,
                       strategy_key: str = "strategy_num") -> OutcomePrior | None:
    """Read a JSONL transform ledger into an :class:`OutcomePrior`. None when no ledger is named or
    the named one does not exist -- a missing corpus is not a corpus of zero attempts."""
    p = ledger_path(path)
    if p is None or not p.is_file():
        return None
    counts: dict[str, dict[str, int]] = {}
    total = improved_total = unusable = 0
    unlabelled = unlabelled_improved = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            unusable += 1
            continue
        outcome = row.get("outcome")
        strategy = row.get(strategy_key)
        if outcome not in _COUNTED_OUTCOMES:
            unusable += 1
            continue
        if not strategy:
            # A countable attempt that names no strategy. It cannot feed a per-strategy rate, but it
            # is NOT unusable: it is the population against which the labelled set's representativity
            # is judged, and folding it into "unusable" would hide exactly that comparison.
            unlabelled += 1
            if outcome == IMPROVED:
                unlabelled_improved += 1
            continue
        bucket = counts.setdefault(str(strategy), {})
        bucket[outcome] = bucket.get(outcome, 0) + 1
        total += 1
        if outcome == IMPROVED:
            improved_total += 1
    by_strategy = {
        s: StrategyEvidence(strategy=s, attempts=sum(o.values()),
                            improved=o.get(IMPROVED, 0), outcomes=dict(o))
        for s, o in counts.items()}
    return OutcomePrior(by_strategy=by_strategy, total_attempts=total,
                        total_improved=improved_total, source=str(p), unusable_rows=unusable,
                        unlabelled_attempts=unlabelled, unlabelled_improved=unlabelled_improved)


def prior_fn_from(prior: OutcomePrior | None,
                  strategy_of: Callable[[Any], str | None],
                  *, min_attempts: int = 5) -> Callable[[Any], float | None]:
    """Adapt an :class:`OutcomePrior` into the ``prior_fn`` the beam's selector consumes.

    ``strategy_of(proposal) -> strategy label | None`` is the DECLARED correspondence between this
    compiler's actions and the ledger's strategies. It is a parameter because no derivation exists:
    inferring it from prose would produce a confident mapping nobody measured.
    """
    def fn(proposal: Any) -> float | None:
        if prior is None:
            return None
        return prior.rate_for(strategy_of(proposal), min_attempts=min_attempts)
    return fn


def family_strategy_map(pairs: Iterable[tuple[str, str]]) -> Callable[[Any], str | None]:
    """Build a ``strategy_of`` from explicit ``(action_family, strategy)`` pairs."""
    table = {str(f): str(s) for f, s in pairs}

    def strategy_of(proposal: Any) -> str | None:
        action = getattr(proposal, "action", None)
        fam = getattr(action, "action_family", "") if action is not None else ""
        return table.get(str(fam)) if fam else None
    return strategy_of
