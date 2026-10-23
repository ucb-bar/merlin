"""Bottleneck findings derived from one console snapshot.

Each rule compares numbers the sources already read and returns ``{code, severity, scope, title,
detail, evidence}``; findings sharing a code, scope and title merge into one with ``occurrences`` and
the distinct details as ``items``. Rules never grade, never promote and never treat a missing value as a pass: a
rule whose inputs were not recorded simply does not fire. Severity is ``error``, ``warn`` or ``info``.
"""

from __future__ import annotations

from typing import Any

SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}


def _sections(snapshot: dict[str, Any]) -> dict[str, Any]:
    sections = snapshot.get("sections") or {}
    return {k: v for k, v in sections.items() if isinstance(v, dict) and "error" not in v}


def _finding(code: str, severity: str, scope: str, title: str, detail: str = "", evidence: str | None = None):
    return {"code": code, "severity": severity, "scope": scope, "title": title, "detail": detail, "evidence": evidence}


def diagnose(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """All findings, most severe first."""
    s = _sections(snapshot)
    out: list[dict[str, Any]] = []
    for name, section in (snapshot.get("sections") or {}).items():
        if isinstance(section, dict) and section.get("error"):
            out.append(_finding("READER_ERROR", "warn", name, f"{name} reader failed", section["error"]))
    for rule in RULES:
        try:
            out.extend(rule(snapshot, s))
        except Exception as exc:  # noqa: BLE001 -- a broken rule is itself a finding, never a crash
            out.append(
                _finding("RULE_ERROR", "info", rule.__name__, "diagnostic rule failed", f"{type(exc).__name__}: {exc}")
            )
    # The same finding recorded against several retained snapshots is one finding seen N times.
    # A finding with one title and several details (one per failure plane, say) is one finding
    # whose details are its items.
    merged: dict[tuple, dict[str, Any]] = {}
    for finding in out:
        key = (finding["code"], finding["scope"], finding["title"])
        if key not in merged:
            merged[key] = {**finding, "occurrences": 1, "items": [finding["detail"]] if finding["detail"] else []}
            continue
        entry = merged[key]
        entry["occurrences"] += 1
        if finding["detail"] and finding["detail"] not in entry["items"]:
            entry["items"].append(finding["detail"])
    for entry in merged.values():
        if len(entry["items"]) > 1:
            entry["detail"] = f"{entry['items'][0]} (+{len(entry['items']) - 1} more)"
    return sorted(merged.values(), key=lambda f: (SEVERITY_ORDER.get(f["severity"], 3), -f["occurrences"], f["code"]))


def _progress_alerts(snapshot, s):
    for alert in (s.get("progress") or {}).get("alerts") or []:
        if isinstance(alert, dict) and alert.get("code"):
            sev = alert.get("severity") if alert.get("severity") in SEVERITY_ORDER else "warn"
            yield _finding(
                str(alert["code"]),
                sev,
                "progress",
                str(alert.get("title") or alert["code"]),
                str(alert.get("detail") or ""),
                alert.get("evidence"),
            )


def _queue_dominated(snapshot, s):
    sim = (s.get("jobs") or {}).get("sim") or {}
    wait, run = sim.get("queue_wait_p50"), sim.get("runtime_p50")
    if isinstance(wait, int | float) and isinstance(run, int | float) and run > 0 and wait > run:
        yield _finding(
            "QUEUE_DOMINATED",
            "warn",
            "jobs",
            "Simulation jobs wait longer than they run",
            f"median queue wait {wait:.1f}s vs median runtime {run:.1f}s",
        )


def _cpu_idle_with_backlog(snapshot, s):
    res = s.get("resources") or {}
    if res.get("mode") != "live":
        return
    busy, total = res.get("busy_cpus"), res.get("total_cpus")
    waiters = (res.get("queue") or {}).get("waiters")
    queued = ((s.get("jobs") or {}).get("sim") or {}).get("queued")
    backlog = (waiters or 0) + (queued or 0)
    if isinstance(busy, int | float) and isinstance(total, int) and total > 0 and backlog > 0 and busy < 0.25 * total:
        yield _finding(
            "CPU_IDLE_WITH_BACKLOG",
            "warn",
            "resources",
            "Reserved CPUs idle while work is queued",
            f"{busy:.1f}/{total} CPUs busy with {backlog} queued job(s)",
        )


def _slow_setup_stage(snapshot, s):
    stages = [r for r in (s.get("jobs") or {}).get("stages") or [] if isinstance(r, dict)]
    timed = [r for r in stages if isinstance(r.get("p95_s"), int | float) and (r.get("count") or 0) >= 3]
    if not timed:
        return
    worst = max(timed, key=lambda r: r["p95_s"])
    total = sum(r["p95_s"] for r in timed)
    if total > 0 and worst["p95_s"] >= 0.5 * total and worst["p95_s"] > 30:
        yield _finding(
            "SETUP_STAGE_SLOW",
            "info",
            "jobs",
            f"Stage '{worst.get('stage')}' dominates job time",
            f"p95 {worst['p95_s']:.1f}s over {worst.get('count')} jobs",
        )


def _agents(snapshot, s):
    agents = s.get("agents") or {}
    totals = agents.get("totals") or {}
    ceiling, working = totals.get("helper_ceiling"), totals.get("working")
    if isinstance(ceiling, int) and ceiling > 0 and isinstance(working, int) and working - 1 >= ceiling:
        yield _finding(
            "HELPERS_SATURATED",
            "info",
            "agents",
            "All helper slots are busy",
            f"{working - 1} helpers working against a ceiling of {ceiling}",
        )
    experiment = snapshot.get("experiment") or {}
    now = snapshot.get("generated_at")
    if experiment.get("state") in ("LIVE", "ACTIVE") and isinstance(now, int | float):
        last = max(
            (a.get("last_activity_at") or 0 for a in agents.get("agents") or [] if isinstance(a, dict)), default=0
        )
        if last and now - last > 900:
            yield _finding(
                "AGENT_IDLE",
                "warn",
                "agents",
                "No agent activity recorded recently",
                f"last agent event {int((now - last) // 60)} min ago",
            )


def _tokens_without_progress(snapshot, s):
    tokens = (s.get("tokens") or {}).get("series") or {}
    progress = (s.get("progress") or {}).get("series") or {}
    t, out = tokens.get("t") or [], tokens.get("output") or []
    pt, passed = progress.get("t") or [], progress.get("n_passed") or []
    if len(t) < 2 or not pt or not passed:
        return
    # Time of the last change in the passing count.
    last_change = pt[0]
    for i in range(1, len(passed)):
        if passed[i] != passed[i - 1]:
            last_change = pt[i]
    since = [(ti, oi) for ti, oi in zip(t, out) if isinstance(oi, int | float) and ti >= last_change]
    if len(since) >= 2:
        spent = since[-1][1] - since[0][1]
        hours = (since[-1][0] - since[0][0]) / 3600
        if spent > 200_000 and hours > 1:
            yield _finding(
                "TOKENS_WITHOUT_PROGRESS",
                "warn",
                "tokens",
                "Output tokens spent without a passing-count change",
                f"{spent:,.0f} output tokens over {hours:.1f} h since the count last changed",
            )


def _token_budget(snapshot, s):
    tokens = s.get("tokens") or {}
    budget = tokens.get("budget") or {}
    used = budget.get("used_pct")
    if isinstance(used, int | float) and used >= 90:
        severity = "error" if used >= 100 else "warn"
        yield _finding(
            "TOKEN_BUDGET",
            severity,
            "tokens",
            f"Author token budget {used:.0f}% used",
            f"{used:.1f}% of {budget.get('total') or 'the recorded'} tokens",
        )
    limit = (tokens.get("rate_limit") or {}).get("used_percent")
    if isinstance(limit, int | float) and limit >= 80:
        yield _finding(
            "RATE_LIMIT_PRESSURE",
            "warn",
            "tokens",
            f"Provider rate-limit window {limit:.0f}% used",
            "the agent may slow down or pause when the window fills",
        )


def _churn(snapshot, s):
    passes = [p for p in (s.get("passes") or {}).get("passes") or [] if isinstance(p, dict)]
    churn = sorted((p for p in passes if isinstance(p.get("churn"), int) and p["churn"] > 0), key=lambda p: -p["churn"])
    if churn and churn[0]["churn"] >= 100:
        top = churn[0]
        yield _finding(
            "INVALIDATION_CHURN",
            "info",
            "passes",
            f"'{top.get('name')}' invalidates cached results often",
            f"{top['churn']} recorded invalidations",
            top.get("source"),
        )


def _failure_cluster(snapshot, s):
    capsules = [c for c in (s.get("capsules") or {}).get("capsules") or [] if isinstance(c, dict)]
    clusters: dict[str, list[str]] = {}
    for capsule in capsules:
        failure = capsule.get("failure") or {}
        category = failure.get("category")
        if category and capsule.get("status") not in ("pass", "passed"):
            clusters.setdefault(str(category), []).append(str(capsule.get("name")))
    for category, names in sorted(clusters.items(), key=lambda kv: -len(kv[1]))[:3]:
        if len(names) >= 5:
            yield _finding(
                "FAILURE_CLUSTER",
                "warn",
                "capsules",
                f"{len(names)} capsules fail with {category}",
                ", ".join(names[:4]) + (" ..." if len(names) > 4 else ""),
            )


def _stage_blocked(snapshot, s):
    for stage in (s.get("stages") or {}).get("stages") or []:
        if isinstance(stage, dict) and stage.get("state") == "blocked":
            yield _finding(
                "STAGE_BLOCKED",
                "warn",
                "stages",
                f"Stage '{stage.get('title') or stage.get('id')}' is blocked",
                "; ".join(str(x) for x in (stage.get("items") or [])[:2]),
            )


def _perf(snapshot, s):
    perf = s.get("perf") or {}
    rows = [r for r in perf.get("rows") or [] if isinstance(r, dict)]
    eff = [r for r in rows if isinstance(r.get("efficiency_pct"), int | float)]
    if eff:
        worst = min(eff, key=lambda r: r["efficiency_pct"])
        if worst["efficiency_pct"] < 5:
            top = worst.get("top_resource") or {}
            detail = f"{worst['efficiency_pct']:.2f}% of the structural floor"
            if top.get("name"):
                detail += f"; busiest modeled resource {top['name']} {top.get('activity_pct', 0):.0f}%"
            yield _finding(
                "LOW_ROOFLINE_EFFICIENCY",
                "info",
                "perf",
                f"{worst.get('capsule')} runs far from its structural floor",
                detail,
                worst.get("source"),
            )
    gaps = [
        r
        for r in rows
        if r.get("comparable") and isinstance(r.get("divergence_pct"), int | float) and abs(r["divergence_pct"]) >= 25
    ]
    if gaps:
        worst = max(gaps, key=lambda r: abs(r["divergence_pct"]))
        yield _finding(
            "MODEL_GAP",
            "info",
            "perf",
            f"Spike model and gSIM disagree on {worst.get('capsule')}",
            f"{worst['divergence_pct']:+.0f}% (gSIM vs model estimate)",
            worst.get("source"),
        )
    unknown = (perf.get("summary") or {}).get("unknown")
    if isinstance(unknown, int) and unknown > 0 and not eff:
        yield _finding(
            "UNKNOWN_PERF",
            "info",
            "perf",
            f"{unknown} performance records are UNKNOWN",
            "no performance block passed its binding check",
        )


def _stale(snapshot, s):
    experiment = snapshot.get("experiment") or {}
    if experiment.get("state") == "STALLED":
        yield _finding(
            "RECORD_STALE",
            "error",
            "experiment",
            "Run is alive but its records stopped changing",
            str(experiment.get("state_basis") or ""),
        )


def _wall_time(snapshot, s):
    """Which activity the lead agent's wall time went to (the campaign's critical path)."""
    timeline = s.get("timeline") or {}
    verdict = timeline.get("verdict") or {}
    recorded = timeline.get("recorded") or {}
    window = (timeline.get("window") or {}).get("seconds")
    overlap = timeline.get("overlap") or {}
    bound, share = verdict.get("bound"), verdict.get("share_pct")
    if not bound or not isinstance(share, int | float) or not isinstance(window, int | float) or window < 600:
        return
    if bound == "agent_reasoning" and share >= 60:
        rate = recorded.get("output_tokens_per_think_s")
        detail = f"{share:.0f}% of the lead's time is model reasoning/generation ({verdict.get('basis')})"
        if isinstance(rate, int | float):
            detail += f"; {rate:.0f} output tok/s while thinking"
        if overlap.get("sim_while_think_s"):
            detail += f"; simulations ran {overlap['sim_while_think_s'] / 60:.0f} min in the background meanwhile"
        yield _finding("AGENT_REASONING_BOUND", "info", "timeline", "Agent reasoning is the critical path", detail)
    elif bound in ("tools", "tools_and_waits") and share >= 35:
        yield _finding(
            "TOOL_BOUND",
            "warn",
            "timeline",
            f"The lead waits on tool calls {share:.0f}% of the time",
            "builds, selfchecks and simulations sit on the critical path; see jobs stage p95 and the TIMELINE",
        )
    elif bound == "waits" and share >= 30:
        yield _finding(
            "WAIT_BOUND",
            "warn",
            "timeline",
            f"The lead is blocked on awaits {share:.0f}% of the time",
            "it waits for subagents or background jobs instead of working",
        )
    elif bound == "idle" and share >= 40:
        yield _finding(
            "LEAD_IDLE",
            "warn",
            "timeline",
            f"No lead turn is open {share:.0f}% of the window",
            "time goes to relaunches, grading gaps or setup rather than authoring",
        )
    elif bound == "setup" and share >= 40 and window >= 1800:
        yield _finding(
            "SETUP_DOMINATED",
            "warn",
            "timeline",
            f"{share:.0f}% of the window passed before the lead's first turn",
            "input preparation and tooling readiness delay authoring; see jobs stages",
        )
    blocked = overlap.get("lead_blocked_on_sim_s")
    if isinstance(blocked, int | float) and blocked / window >= 0.15:
        yield _finding(
            "SIM_ON_CRITICAL_PATH",
            "warn",
            "timeline",
            "The lead waits while simulations run",
            f"{blocked / 60:.0f} min ({blocked / window * 100:.0f}% of the window) blocked during L2/L3 jobs",
        )
    para = timeline.get("parallelism") or {}
    avg, ceiling, used = para.get("agents_avg"), para.get("helper_ceiling"), para.get("helpers_used")
    if (
        isinstance(avg, int | float)
        and isinstance(ceiling, int)
        and ceiling >= 4
        and window >= 3600
        and avg < max(1.5, ceiling / 8)
    ):
        yield _finding(
            "HELPERS_UNDERUSED",
            "info",
            "agents",
            f"{avg:.1f} agents busy on average against {ceiling} helper slots",
            f"{used or 0} helper(s) ever worked; the work runs mostly serially",
        )


def _orphaned_jobs(snapshot, s):
    experiment = snapshot.get("experiment") or {}
    running = (s.get("jobs") or {}).get("running") or []
    if running and experiment.get("state") not in ("LIVE", "STALLED", "ACTIVE"):
        yield _finding(
            "UNFINISHED_JOBS",
            "info",
            "jobs",
            f"{len(running)} job(s) never recorded an end",
            "the run stopped while they ran: " + ", ".join(str(r.get("stage")) for r in running[:3]),
        )


def _machine(snapshot, s):
    machine = s.get("machine") or {}
    cores = [c for c in machine.get("cores") or [] if isinstance(c, dict)]
    mine = [c for c in cores if c.get("mine") and isinstance(c.get("util_pct"), int | float)]
    if mine:
        busy = sum(c["util_pct"] for c in mine) / len(mine)
        waiters = ((s.get("resources") or {}).get("queue") or {}).get("waiters") or 0
        if busy >= 95 and waiters:
            yield _finding(
                "CPU_SATURATED",
                "warn",
                "machine",
                f"The experiment's {len(mine)} CPUs are saturated",
                f"{busy:.0f}% average with {waiters} waiter(s) queued for a CPU",
            )
    for alloc in machine.get("allocations") or []:
        if not isinstance(alloc, dict) or alloc.get("kind") != "reservation" or alloc.get("processes"):
            continue
        reserved = set(_cpu_list(alloc.get("cpus")))
        users = [a for a in machine.get("allocations") or [] if a.get("kind") == "run"]
        if not any(reserved & set(_cpu_list(a.get("cpus"))) for a in users):
            yield _finding(
                "RESERVATION_IDLE",
                "warn",
                "machine",
                f"CPUs {alloc.get('cpus')} reserved by {alloc.get('name')} with no experiment on them",
                "the general pool is smaller while the reservation is held; release it when no phase run needs it",
            )


def _cpu_list(text) -> list[int]:
    out: list[int] = []
    for item in str(text or "").split(","):
        a, sep, b = item.partition("-")
        if a.strip().isdigit() and (not sep or b.strip().isdigit()):
            out.extend(range(int(a), int(b if sep else a) + 1))
    return out


RULES = (
    _progress_alerts,
    _queue_dominated,
    _cpu_idle_with_backlog,
    _slow_setup_stage,
    _agents,
    _tokens_without_progress,
    _token_budget,
    _churn,
    _failure_cluster,
    _stage_blocked,
    _perf,
    _stale,
    _wall_time,
    _orphaned_jobs,
    _machine,
)

__all__ = ["RULES", "SEVERITY_ORDER", "diagnose"]
