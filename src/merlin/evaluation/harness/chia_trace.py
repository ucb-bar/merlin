"""Normalize public Chia trace metadata without owning prices or child ledgers.

Main's profiler is diagnostic, not an exhaustive usage ledger: failed calls can
lack completion metadata and a CLI dollar figure need not be a charge. Only
explicit billing declarations are passed to AET's canonical billing splitter.
"""

from __future__ import annotations

import json
import math


def record_trace(handle, events, *, accounting: str) -> None:
    """Record one owned collector snapshot; never rewrite identity or child spend.

    ``extra`` is Chia's public ``get_profiler().add_info`` transport. Main's
    built-in providers do not consistently declare billing_mode/cost_source;
    their metadata is retained as unknown, not priced or inferred here.
    """
    from aet.trajectory.billing import billing_mode_of, split

    if accounting not in {"trace", "child-ledgers"}:
        raise ValueError("unknown Chia accounting owner")
    if not isinstance(events, list):
        raise ValueError("Chia collector events must be a list")
    completed = {}
    observed = set()
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Chia trace event must be an object")
        call_id = event.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        observed.add(call_id)
        if event.get("type") not in {"complete", "local_end"}:
            continue
        extra = event.get("extra", {})
        if not isinstance(extra, dict):
            raise ValueError("Chia completion extra must be an object")
        # Dispatch/completion timestamps differ legitimately; duplicate usage
        # declarations do not. A conflicting duplicate must not select a winner.
        if call_id in completed and completed[call_id] != extra:
            raise ValueError(f"conflicting Chia completion metadata for {call_id}")
        completed[call_id] = extra

    rows = []
    classified = []
    for call_id in sorted(observed):
        extra = completed.get(call_id)
        row = {"call_id": call_id, "status": "unknown", "metadata": extra}
        if accounting == "child-ledgers":
            row["status"] = "child_ledger_owned"
        elif extra is not None and extra.get("billing_mode") and extra.get("cost_source") == "billed":
            cost = extra.get("cost_usd")
            if cost is not None and (
                isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0
            ):
                raise ValueError("Chia declared cost must be finite, nonnegative or null")
            # AET owns mode validation and splitting, never a second price table.
            billing_mode_of(extra)
            classified.append(extra)
            row["status"] = "declared" if cost is not None else "unpriced"
        rows.append(row)
    spend = split(classified)
    report = {
        "schema": "merlin.chia-trace-accounting.v1",
        "run_id": handle.run_id,
        "owner": accounting,
        "coverage": "observed_only_not_complete",
        "calls": rows,
        "metered_usd": spend.metered_usd if spend.metered_rows else None,
        "subscription_usd_notional": spend.subscription_usd if spend.subscription_rows else None,
        "unpriced_calls": spend.unpriced_rows,
        "unknown_calls": sum(row["status"] == "unknown" for row in rows),
    }
    path = handle.run_dir / "chia" / "accounting.json"
    # One collector belongs to one canonical run. Re-entry cannot charge it twice;
    # partial writer failures remain evidence, not permission to replay the charge.
    with path.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # Canonical child ledgers remain the only accounting owner for native tasks.
    if spend.metered_rows:
        handle.logger.log_cost(spend.metered_usd)
    if spend.subscription_rows:
        handle.logger.log_metric("chia.subscription.cost_equivalent_usd", spend.subscription_usd)
    handle.logger.log_event("chia.trace.accounting", report)
