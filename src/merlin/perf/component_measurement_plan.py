"""Bounded development measurement choices, without qualification authority.

The experiment owner must independently verify runtime, applicability and held
evidence before using these arithmetic choices. Nothing here issues that owner
or changes mandatory correctness, qualification or final measurement rosters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from merlin.common.digest import is_sha256
from merlin.perf.component_screen import ComponentOpportunity, cheapest_distinguishing_experiment
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


@dataclass(frozen=True)
class DevelopmentMeasurementDecision:
    id: str
    action: str
    reason: str


def plan_development_measurements(
    opportunities, *, applicability, owner_available, purpose, max_measurements, max_members=10000
):
    """Retain every member; choose bounded overlapping-interval experiments.

    ``owner_available`` and ``applicability`` are arithmetic input data, never
    capabilities. Production dispatch requires the experiment's live issuer.
    Disjoint predictions defer only development measurement, including when
    the candidate looks faster. They cannot promote or complete that candidate.
    """
    if type(purpose) is not str or purpose != "development_performance":
        raise ValueError("measurement pruning is unavailable for correctness, held or final evaluation")
    if (
        type(opportunities) is not tuple
        or not opportunities
        or type(max_members) is not int
        or not 0 < max_members <= 10000
        or len(opportunities) > max_members
        or type(max_measurements) is not int
        or not 0 < max_measurements <= max_members
        or type(owner_available) is not bool
        or type(applicability) is not dict
    ):
        raise ValueError("development measurement plan requires a closed bounded complete roster")
    ids = [row.id for row in opportunities if type(row) is ComponentOpportunity]
    if (
        len(ids) != len(opportunities)
        or len(set(ids)) != len(ids)
        or any(not is_sha256(value) for value in ids)
        or set(applicability) != set(ids)
        or any(type(value) is not str or value not in ("IN_DOMAIN", "UNKNOWN") for value in applicability.values())
    ):
        raise ValueError("development measurement membership or applicability is incomplete")
    decisions, pending = {}, []
    for row in opportunities:
        if (
            type(row.baseline) is not CycleInterval
            or type(row.candidate) is not CycleInterval
            or row.regime not in ("cold", "warm")
            or not row.family
            or any(value is not None and type(value) is not bool for value in (row.legal, row.correct))
        ):
            raise ValueError("development measurement evidence has an unsupported cost or gate")
        if row.legal is False or row.correct is False:
            action, reason = "REFUSAL", "original correctness or legality is refuted"
        elif not owner_available:
            action, reason = "UNAVAILABLE", "independent runtime, physical and held owners are unavailable"
        elif row.legal is not True or row.correct is not True:
            action, reason = "UNAVAILABLE", "complete original correctness and legality are not established"
        elif applicability[row.id] != "IN_DOMAIN" or not row.baseline.resolved or not row.candidate.resolved:
            action, reason = "REQUALIFY", "new or unsupported behavior requires independent requalification"
        elif row.candidate.hi < row.baseline.lo or row.baseline.hi < row.candidate.lo:
            action, reason = "DEFER", "qualified disjoint predictions defer development measurement only"
        else:
            seconds = row.evaluation_seconds
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                action, reason = "UNAVAILABLE", "bounded distinguishing measurement cost is unavailable"
            else:
                action, reason = "PENDING", "overlapping predictions require a bounded distinguishing measurement"
                pending.append({"id": row.id, "evaluation_seconds": seconds, "distinguishes": (row.id,)})
        decisions[row.id] = DevelopmentMeasurementDecision(row.id, action, reason)
    selected = []
    while pending and len(selected) < max_measurements:
        experiment = cheapest_distinguishing_experiment(pending, unresolved=tuple(row["id"] for row in pending))
        selected.append(experiment["id"])
        pending.remove(experiment)
        decisions[experiment["id"]] = DevelopmentMeasurementDecision(
            experiment["id"], "MEASURE", "overlapping predictions select a bounded distinguishing measurement"
        )
    return tuple(decisions[row.id] for row in opportunities), tuple(selected)
