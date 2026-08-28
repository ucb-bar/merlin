"""A signature-keyed store of generated kernels, and the ladder that decides whether to reuse one.

WHY THIS DECIDES THE STUDY'S FAIRNESS. The comparison is between paying per workload and paying once.
If the kernel-generation arm regenerates from scratch for every configuration it ever sees, it is
charged for work a competent practitioner would not repeat, the cost curve rises too steeply, and the
crossover arrives earlier than it should -- in favour of the compiler. So the kernel arm gets a
library, and reuse costs nothing when it genuinely applies.

KEYED BY SIGNATURE, NEVER BY CONFIGURATION. A config-keyed store is a lookup table: every shape gets
its own entry, every entry hits exactly once, and "reuse" measures nothing. Keying by the shape-free
signature is what makes the question -- does this kernel survive a shape it was not written for? --
answerable at all.

THE LADDER, AND WHY LEVEL 1 IS EXECUTED RATHER THAN ASSERTED:

    L0 EXACT       same signature, same config          no model call (still re-evaluated)
    L1 PARAMETRIC  same signature, different config     no model call, IF it actually passes
    L2 WARM_START  same family and regime, new signature  one delta call, seeded with the prior kernel
    L3 NEW         nothing matches                      full generation

A kernel that baked its dimensions in will link or mismatch against a different configuration. If L1
were granted on signature agreement alone, that kernel would be recorded as free reuse and its
failure would never appear -- inventing exactly the advantage this module exists to measure honestly.
So an L1 claim is a PROPOSAL that the caller must confirm by running the evaluator; an unconfirmed or
failed proposal demotes to L3 and is charged full price.

EVERY DECISION IS WRITTEN DOWN BEFORE THE MODEL IS CALLED. The ledger sums only rows whose
``llm_called`` is true, and the decision log is written first, so the two cannot disagree afterwards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Ladder levels, cheapest first. The names are part of the recorded decision.
EXACT, PARAMETRIC, WARM_START, NEW = "L0_EXACT", "L1_PARAMETRIC", "L2_WARM_START", "L3_NEW"

#: Levels that spend no model tokens *if confirmed*. L1's freedom is conditional on evaluation.
FREE_LEVELS = frozenset({EXACT, PARAMETRIC})


@dataclass(frozen=True)
class Entry:
    """One stored kernel: what it is for, what it was written against, what it may vary."""

    signature: str
    config_id: str
    kernel_path: str
    family: str = ""
    regime: str = ""
    #: Config axes the author claims this kernel handles without change. A CLAIM, not a guarantee --
    #: `propose` uses it to pick a candidate, and evaluation decides whether it was true.
    parametric_axes: tuple[str, ...] = ()
    cycles: int | None = None
    provenance: str = ""              # which arm/run produced it
    task_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    """What the library advised for one (signature, config), written BEFORE any model is called."""

    signature: str
    config_id: str
    level: str
    matched_entry: str | None
    rationale: str
    llm_called: bool
    #: True while an L1 proposal is unconfirmed. A decision may not be counted as free reuse until
    #: the evaluator has agreed, so this is the flag that keeps an optimistic claim out of the ledger.
    pending_confirmation: bool = False
    confirmed: bool | None = None
    demoted_from: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_free_reuse(self) -> bool:
        """Countable as zero-cost reuse: a free level, settled, and not demoted."""
        return (self.level in FREE_LEVELS and not self.pending_confirmation
                and self.confirmed is not False and not self.llm_called)


class KernelLibrary:
    """The store plus the ladder. Persisted as JSON so a run can be audited without this class."""

    def __init__(self, root: "str | Path | None" = None) -> None:
        self.root = Path(root) if root is not None else None
        self.entries: list[Entry] = []
        self.decisions: list[Decision] = []

    # -- store ---------------------------------------------------------------------------------

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)

    def find(self, signature: str, config_id: str | None = None) -> list[Entry]:
        out = [e for e in self.entries if e.signature == signature]
        if config_id is not None:
            out = [e for e in out if e.config_id == config_id]
        # Deterministic: cheapest known kernel first, then a stable tiebreak.
        return sorted(out, key=lambda e: (e.cycles if e.cycles is not None else 1 << 62,
                                          e.config_id, e.kernel_path))

    # -- ladder --------------------------------------------------------------------------------

    def propose(self, signature: str, config_id: str, *, family: str = "", regime: str = "",
                config_axes: Iterable[str] = ()) -> Decision:
        """Advise a level for this (signature, config). Records the decision; calls no model.

        An L1 proposal comes back `pending_confirmation`. The caller must run the evaluator and pass
        the verdict to `confirm`, which either settles it as free reuse or demotes it to L3.
        """
        exact = self.find(signature, config_id)
        if exact:
            d = Decision(signature, config_id, EXACT, exact[0].kernel_path,
                         "an entry exists for this exact signature and configuration",
                         llm_called=False)
            self.decisions.append(d)
            return d

        same_sig = self.find(signature)
        if same_sig:
            wanted = set(config_axes)
            for e in same_sig:
                # The entry must CLAIM every axis that differs. An entry claiming nothing is not a
                # candidate for a different config, which is the correct default: silence is not a
                # promise of generality.
                if wanted and wanted <= set(e.parametric_axes):
                    d = Decision(signature, config_id, PARAMETRIC, e.kernel_path,
                                 f"same signature; entry claims the differing axes "
                                 f"{sorted(wanted)}. UNCONFIRMED until it is evaluated against "
                                 f"this configuration.",
                                 llm_called=False, pending_confirmation=True)
                    self.decisions.append(d)
                    return d

        if family or regime:
            near = [e for e in self.entries
                    if e.family == family and e.regime == regime and e.signature != signature]
            if near:
                seed = sorted(near, key=lambda e: (e.cycles if e.cycles is not None else 1 << 62,
                                                   e.signature, e.kernel_path))[0]
                d = Decision(signature, config_id, WARM_START, seed.kernel_path,
                             f"no entry for this signature; seeding from a kernel in the same "
                             f"family and size regime ({family}/{regime})",
                             llm_called=True)
                self.decisions.append(d)
                return d

        d = Decision(signature, config_id, NEW, None,
                     "no entry shares this signature, family and regime", llm_called=True)
        self.decisions.append(d)
        return d

    def confirm(self, decision: Decision, *, passed: bool) -> Decision:
        """Settle a pending proposal against what the evaluator actually found.

        A failed L1 is DEMOTED to L3 and marked `llm_called`, because a kernel that did not survive
        the new configuration has to be regenerated and that costs tokens. Recording it as free reuse
        is the single change that would most flatter the kernel arm, so it is the one thing this
        method exists to prevent.
        """
        decision.pending_confirmation = False
        decision.confirmed = bool(passed)
        if not passed and decision.level in FREE_LEVELS:
            decision.demoted_from = decision.level
            decision.level = NEW
            decision.llm_called = True
            decision.rationale += " -- FAILED against this configuration; demoted to full generation"
        return decision

    # -- reporting -----------------------------------------------------------------------------

    def reuse_rate(self) -> float | None:
        """Fraction of settled decisions served without a model call."""
        settled = [d for d in self.decisions if not d.pending_confirmation]
        if not settled:
            return None
        return sum(1 for d in settled if d.is_free_reuse()) / float(len(settled))

    def generalization_depth(self) -> float | None:
        """Fraction of L1 proposals that actually PASSED.

        The metric that matters: compiling is not surviving, so this counts confirmations, not
        attempts. None when no L1 was ever proposed -- distinct from a depth of zero.
        """
        l1 = [d for d in self.decisions
              if d.level == PARAMETRIC or d.demoted_from == PARAMETRIC]
        if not l1:
            return None
        return sum(1 for d in l1 if d.confirmed) / float(len(l1))

    def matrix(self) -> dict[str, dict[str, str]]:
        """Rows = signature, cols = config, cell = the level that served it."""
        out: dict[str, dict[str, str]] = {}
        for d in self.decisions:
            out.setdefault(d.signature, {})[d.config_id] = d.level
        return out

    def summary(self) -> dict[str, Any]:
        settled = [d for d in self.decisions if not d.pending_confirmation]
        return {
            "entries": len(self.entries),
            "decisions": len(self.decisions),
            "pending_confirmation": sum(1 for d in self.decisions if d.pending_confirmation),
            "by_level": {lv: sum(1 for d in settled if d.level == lv)
                         for lv in (EXACT, PARAMETRIC, WARM_START, NEW)},
            "demoted": sum(1 for d in self.decisions if d.demoted_from),
            "llm_calls": sum(1 for d in settled if d.llm_called),
            "reuse_rate": self.reuse_rate(),
            "generalization_depth": self.generalization_depth(),
            "matrix": self.matrix(),
        }

    # -- persistence ---------------------------------------------------------------------------

    def write(self, root: "str | Path | None" = None) -> Path:
        base = Path(root) if root is not None else self.root
        if base is None:
            raise ValueError("no root given for the kernel library")
        base.mkdir(parents=True, exist_ok=True)
        (base / "entries.json").write_text(
            json.dumps([e.to_dict() for e in self.entries], indent=2, sort_keys=True))
        # The decision log is the ledger's counterpart: written whether or not a model was called.
        (base / "reuse_decisions.json").write_text(
            json.dumps([d.to_dict() for d in self.decisions], indent=2, sort_keys=True))
        (base / "reuse_summary.json").write_text(json.dumps(self.summary(), indent=2, sort_keys=True))
        return base

    @classmethod
    def read(cls, root: "str | Path") -> "KernelLibrary":
        base = Path(root)
        lib = cls(base)
        ep = base / "entries.json"
        if ep.is_file():
            lib.entries = [Entry(**d) for d in json.loads(ep.read_text())]
        dp = base / "reuse_decisions.json"
        if dp.is_file():
            lib.decisions = [Decision(**d) for d in json.loads(dp.read_text())]
        return lib


def audit_against_ledger(decisions: Sequence[Decision],
                         ledger_calls: int) -> dict[str, Any]:
    """Cross-check the decision log against what the accounting says was actually spent.

    They are produced independently -- the log before dispatch, the ledger from transcripts -- so
    disagreement means one of them is wrong, and silent disagreement is how a study reports reuse it
    never achieved. Surfaced rather than reconciled.
    """
    expected = sum(1 for d in decisions if d.llm_called)
    return {
        "decisions_expecting_a_call": expected,
        "ledger_calls": ledger_calls,
        "agrees": expected == ledger_calls,
        "discrepancy": ledger_calls - expected,
    }
