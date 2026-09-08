"""What a performance campaign DID, why, where and how -- assembled so it can be regenerated.

This is the performance lane's analogue of the functional lane's run report, and it answers a
different question. The functional lane asks *how many capsules pass over time*; a performance
campaign has no such curve, because its output is a compiler change and a measurement, and a round
that authors nothing still costs money. So this module reports three things instead:

**Where the budget went.** Brokered actions are the campaign's real cost, and they fall into three
tiers that differ by two orders of magnitude: an analysis action that is free by construction, a
compile at a few seconds, and a measurement at ~100 s. A campaign's shape is decided by how many of
each it ran, and by how many of them REFUSED -- a refusal costs the same wall time as a success and
buys nothing, so it belongs in the cost table rather than in a footnote.

**What was attempted, and on what evidence.** Every attempt carries the instrument that surfaced it,
the hypothesis it tested, the scope it acted at, and the measured delta that decided it
(:mod:`merlin.perf.optimization_ledger`). A campaign that only ever finds one scope's worth of work
is not finished, so the scope histogram is part of the verdict rather than a curiosity.

**What the campaign could not decide, and why.** Every figure is paired with a
:class:`~merlin.agentreport.availability.Status`, exactly as the functional lane's are: a zero plots
as a finding and a refusal plots as a gap, and confusing the two is what makes a stalled campaign
look like a converged one.

NOTHING HERE READS THE EXPERIMENT TREE. The caller walks the run roots and passes in what it found,
because the roots, the action vocabulary and the ledger's location are all launcher decisions -- a
module that hardcoded them would describe one campaign and mislabel the next.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from merlin.agentreport.availability import Availability, measured, unavailable

__all__ = ["ActionCost", "BudgetBreakdown", "AttemptSummary", "CampaignAnalysis",
           "summarize_broker_calls", "summarize_attempts", "REFUSAL_RETURNCODES",
           "COST_TIERS", "tier_of"]

#: Return codes a brokered action uses to REFUSE rather than to fail. Distinct from a non-zero exit
#: that means the action ran and the thing it did was wrong: a refusal means the action declined to
#: run at all, which costs the same wall time and produces no evidence. Measured across 1,296 calls:
#: 1,131 succeeded, 11 failed (rc 1), and 154 refused -- 11.9% of calls and 17.5% of brokered wall.
REFUSAL_RETURNCODES: frozenset[int] = frozenset({125, 126})

#: The three cost tiers a brokered action falls into, by measured mean wall time. These are BOUNDS
#: on a tier, not a classification of any particular action name -- the action vocabulary is derived
#: per run from the candidate's manifest, so naming actions here would describe one campaign.
COST_TIERS: tuple[tuple[str, float, str], ...] = (
    ("free", 1.0, "answered from artifacts already on disk; costs no measurement"),
    ("compile", 30.0, "a lowering or emission step; seconds, and the campaign runs many"),
    ("measure", float("inf"), "a simulation or profile; ~100 s, and it dominates the budget"),
)


def tier_of(mean_seconds: float) -> str:
    """Which cost tier a mean wall time falls in. Bounds, never an action-name table."""
    for name, upper, _ in COST_TIERS:
        if mean_seconds < upper:
            return name
    return COST_TIERS[-1][0]


@dataclass
class ActionCost:
    """One brokered action's measured cost and how much of it bought nothing."""

    action: str
    calls: int = 0
    wall_seconds: float = 0.0
    refused: int = 0
    failed: int = 0
    refused_seconds: float = 0.0

    @property
    def mean_seconds(self) -> float:
        return self.wall_seconds / self.calls if self.calls else 0.0

    @property
    def tier(self) -> str:
        return tier_of(self.mean_seconds)

    @property
    def refusal_rate(self) -> float:
        return self.refused / self.calls if self.calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "calls": self.calls,
                "wall_seconds": round(self.wall_seconds, 1),
                "mean_seconds": round(self.mean_seconds, 2), "tier": self.tier,
                "refused": self.refused, "failed": self.failed,
                "refused_seconds": round(self.refused_seconds, 1),
                "refusal_rate": round(self.refusal_rate, 4)}


@dataclass
class BudgetBreakdown:
    """Where a campaign's brokered wall time went, and how much of it refused."""

    actions: list[ActionCost] = field(default_factory=list)
    availability: Availability = field(default_factory=Availability)

    @property
    def calls(self) -> int:
        return sum(a.calls for a in self.actions)

    @property
    def wall_seconds(self) -> float:
        return sum(a.wall_seconds for a in self.actions)

    @property
    def refused(self) -> int:
        return sum(a.refused for a in self.actions)

    @property
    def refused_seconds(self) -> float:
        return sum(a.refused_seconds for a in self.actions)

    def by_tier(self) -> dict[str, dict[str, float]]:
        """Calls and wall time per cost tier. The shape of the campaign in one table."""
        out: dict[str, dict[str, float]] = {
            name: {"calls": 0, "wall_seconds": 0.0, "refused": 0} for name, _, _ in COST_TIERS}
        for action in self.actions:
            bucket = out[action.tier]
            bucket["calls"] += action.calls
            bucket["wall_seconds"] += action.wall_seconds
            bucket["refused"] += action.refused
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_phase2_budget_v1", "calls": self.calls,
                "wall_seconds": round(self.wall_seconds, 1),
                "refused": self.refused, "refused_seconds": round(self.refused_seconds, 1),
                "refused_call_fraction": round(self.refused / self.calls, 4) if self.calls else None,
                "refused_wall_fraction": (round(self.refused_seconds / self.wall_seconds, 4)
                                          if self.wall_seconds else None),
                "by_tier": {k: {"calls": int(v["calls"]),
                                "wall_seconds": round(v["wall_seconds"], 1),
                                "refused": int(v["refused"])}
                            for k, v in self.by_tier().items()},
                "actions": [a.to_dict() for a in
                            sorted(self.actions, key=lambda a: -a.wall_seconds)],
                "availability": {k: s.to_dict() for k, s in self.availability}}


def summarize_broker_calls(calls: Sequence[Any], *,
                           refusal_codes: frozenset[int] = REFUSAL_RETURNCODES
                           ) -> BudgetBreakdown:
    """Group brokered calls by action. ``calls`` are :class:`~merlin.agentreport.phase2.BrokerCall`.

    A call with no return code is counted but neither credited nor charged as a refusal, and the
    availability ledger says how many were in that state -- guessing would either inflate the
    success rate or invent refusals.
    """
    out = BudgetBreakdown()
    per: dict[str, ActionCost] = {}
    unknown_rc = 0
    for call in calls:
        action = str(getattr(call, "action", "") or "<unnamed>")
        cost = per.setdefault(action, ActionCost(action=action))
        seconds = float(getattr(call, "elapsed_s", 0.0) or 0.0)
        cost.calls += 1
        cost.wall_seconds += seconds
        code = getattr(call, "returncode", None)
        if code is None:
            unknown_rc += 1
        elif int(code) in refusal_codes:
            cost.refused += 1
            cost.refused_seconds += seconds
        elif int(code) != 0:
            cost.failed += 1
    out.actions = list(per.values())
    if not calls:
        out.availability.set("budget", unavailable(
            "no brokered receipts were found, so the campaign's cost cannot be attributed to "
            "actions; a zero here would read as a campaign that spent nothing"))
    else:
        out.availability.set("budget", measured(source="broker_receipts_jsonl"))
    if unknown_rc:
        out.availability.set("outcomes", unavailable(
            f"{unknown_rc} of {len(calls)} calls carry no return code, so their outcome is neither "
            f"a success nor a refusal and is excluded from both"))
    else:
        out.availability.set("outcomes", measured(source="broker_receipts_jsonl"))
    return out


@dataclass
class AttemptSummary:
    """What was attempted, grouped the three ways a reader of a campaign asks about it."""

    total: int = 0
    by_verdict: dict[str, int] = field(default_factory=dict)
    by_scope: dict[str, int] = field(default_factory=dict)
    instruments: list[str] = field(default_factory=list)
    #: Attempts whose verdict asserts a measurement, so their delta can be cited.
    measured_attempts: int = 0
    #: Refuted rows: the branches a later campaign must not re-walk.
    refuted: list[dict[str, Any]] = field(default_factory=list)
    #: Blocked rows still live, with what blocks each.
    live_blockers: list[dict[str, Any]] = field(default_factory=list)
    integrity_problems: list[dict[str, Any]] = field(default_factory=list)
    availability: Availability = field(default_factory=Availability)

    @property
    def scopes_reached(self) -> int:
        return len([s for s, n in self.by_scope.items() if n])

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_phase2_attempts_v1", "total": self.total,
                "by_verdict": dict(self.by_verdict), "by_scope": dict(self.by_scope),
                "scopes_reached": self.scopes_reached,
                "distinct_instruments": len(self.instruments),
                "instruments": list(self.instruments),
                "measured_attempts": self.measured_attempts,
                "n_refuted": len(self.refuted), "refuted": list(self.refuted),
                "n_live_blockers": len(self.live_blockers),
                "live_blockers": list(self.live_blockers),
                "integrity_problems": list(self.integrity_problems),
                "availability": {k: s.to_dict() for k, s in self.availability}}


def summarize_attempts(ledger: Any, *, asserts_measurement: frozenset[str] = frozenset(
        {"helped", "no_effect", "refuted"})) -> AttemptSummary:
    """Group a :class:`~merlin.perf.optimization_ledger.Ledger`'s rows for a campaign report.

    ``integrity_problems`` is reported rather than raised: a ledger row that does not stand up is a
    fact about the campaign's record-keeping, and dropping it would make the record look cleaner
    than it is.
    """
    out = AttemptSummary()
    attempts = list(getattr(ledger, "attempts", ()) or ())
    out.total = len(attempts)
    for attempt in attempts:
        verdict = str(getattr(attempt, "verdict", "") or "")
        scope = str(getattr(attempt, "scope", "") or "")
        out.by_verdict[verdict] = out.by_verdict.get(verdict, 0) + 1
        out.by_scope[scope] = out.by_scope.get(scope, 0) + 1
        if verdict in asserts_measurement:
            out.measured_attempts += 1
        if verdict == "refuted":
            out.refuted.append({
                "mechanism": str(getattr(attempt, "mechanism", "")),
                "found_by": str(getattr(attempt, "found_by", "")),
                "why_the_branch_is_dead": str(getattr(attempt, "hypothesis", "")),
                "deltas": [d.to_dict() for d in getattr(attempt, "deltas", ()) or ()],
                "evidence": str(getattr(attempt, "evidence", ""))})
        problems = tuple(getattr(attempt, "problems", lambda: ())() or ())
        if problems:
            out.integrity_problems.append({
                "mechanism": str(getattr(attempt, "mechanism", "")), "problems": list(problems)})
    out.instruments = sorted({str(getattr(a, "found_by", "")) for a in attempts
                              if getattr(a, "found_by", "")})
    for blocker in list(getattr(ledger, "live_blockers", lambda: ())() or ()):
        out.live_blockers.append({
            "mechanism": str(getattr(blocker, "mechanism", "")),
            "blocked_by": str(getattr(blocker, "blocked_by", "")),
            "found_by": str(getattr(blocker, "found_by", ""))})
    if not attempts:
        out.availability.set("attempts", unavailable(
            "the campaign ledger holds no attempts, so nothing can be said about what was tried; "
            "an empty table would read as a campaign that tried nothing rather than one whose "
            "record is missing"))
    else:
        out.availability.set("attempts", measured(source="optimization_ledger_json"))
    return out


@dataclass
class CampaignAnalysis:
    """The whole analysis: cost, attempts, outcome, and what none of it can decide."""

    target: str = ""
    stages: int = 0
    tool_spans: int = 0
    point_events: int = 0
    budget: BudgetBreakdown = field(default_factory=BudgetBreakdown)
    attempts: AttemptSummary = field(default_factory=AttemptSummary)
    #: Per-workload outcome rows the caller measured (binaries, frames, gates).
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    availability: Availability = field(default_factory=Availability)

    def to_dict(self) -> dict[str, Any]:
        merged = Availability()
        for name, status in self.budget.availability:
            merged.set(f"budget.{name}", status)
        for name, status in self.attempts.availability:
            merged.set(f"attempts.{name}", status)
        for name, status in self.availability:
            merged.set(name, status)
        return {"schema": "merlin_phase2_campaign_analysis_v1", "target": self.target,
                "stages": self.stages, "tool_spans": self.tool_spans,
                "point_events_excluded": self.point_events,
                "budget": self.budget.to_dict(), "attempts": self.attempts.to_dict(),
                "outcomes": list(self.outcomes),
                "availability": {k: s.to_dict() for k, s in merged},
                "availability_score": round(merged.score, 4)}


def build_analysis(*, target: str, stages: int, tool_spans: int, point_events: int,
                   broker_calls: Sequence[Any], ledger: Any,
                   outcomes: Sequence[Mapping[str, Any]] = ()) -> CampaignAnalysis:
    """Assemble the analysis from facts the caller read. No path is touched here."""
    analysis = CampaignAnalysis(target=str(target), stages=int(stages),
                                tool_spans=int(tool_spans), point_events=int(point_events))
    analysis.budget = summarize_broker_calls(broker_calls)
    analysis.attempts = summarize_attempts(ledger)
    analysis.outcomes = [dict(row) for row in outcomes]
    if not stages:
        analysis.availability.set("stages", unavailable(
            "no phase-2 stage directory carried telemetry, so the lane's shape is unknown"))
    else:
        analysis.availability.set("stages", measured(source="run_tree_walk"))
    if not tool_spans:
        analysis.availability.set("lane_shape", unavailable(
            "no tool spans were recorded, so whether the lane runs serially or in parallel cannot "
            "be established from this campaign"))
    else:
        analysis.availability.set("lane_shape", measured(source="agent_tools_jsonl"))
    if not analysis.outcomes:
        analysis.availability.set("outcomes", unavailable(
            "the caller supplied no per-workload outcome, so the campaign's product is unstated"))
    else:
        analysis.availability.set("outcomes", measured(source="caller_supplied_receipts"))
    return analysis
