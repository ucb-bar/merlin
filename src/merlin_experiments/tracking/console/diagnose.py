"""Bottleneck findings derived from one console snapshot.

Each rule compares numbers the sources already read and returns ``{code, severity, scope, title,
detail, evidence, action, impact_s}``: ``title`` is the finding in one line with its number, ``detail``
the evidence, ``action`` the next thing to look at or change, and ``impact_s`` the wall time at stake
(``None`` when the finding is not a time loss). Findings sharing a code, scope and title merge into one
with ``occurrences`` and the distinct details as ``items``; they sort by severity, then by impact.
Rules never grade, never promote and never treat a missing value as a pass: a rule whose inputs were
not recorded simply does not fire. Severity is ``error``, ``warn`` or ``info``.
"""

from __future__ import annotations

from typing import Any

SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}


def _sections(snapshot: dict[str, Any]) -> dict[str, Any]:
    sections = snapshot.get("sections") or {}
    return {k: v for k, v in sections.items() if isinstance(v, dict) and "error" not in v}


def _finding(
    code: str,
    severity: str,
    scope: str,
    title: str,
    detail: str = "",
    evidence: str | None = None,
    *,
    action: str | None = None,
    impact_s: float | None = None,
):
    return {
        "code": code,
        "severity": severity,
        "scope": scope,
        "title": title,
        "detail": detail,
        "evidence": evidence,
        "action": action,
        "impact_s": impact_s,
    }


def _dur(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{seconds / 3600:.1f}h"


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
    return sorted(
        merged.values(),
        key=lambda f: (SEVERITY_ORDER.get(f["severity"], 3), -(f.get("impact_s") or 0), -f["occurrences"], f["code"]),
    )


#: The next step for the grade-progress alerts (:mod:`..progress_health`).
ALERT_ACTIONS = {
    "GRADE_EXCEPTIONS": "the grader itself raised: inspect PROGRESS (Enter) for the request and its stage",
    "REPEATED_PRE_EXECUTION_FAILURE": "these never reach execution: open CAPSULES (sort status) and inspect one",
    "NO_PROGRESS": "no capsule changed state recently: check the agent and the grader in TIMELINE",
}


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
                action=ALERT_ACTIONS.get(str(alert["code"])),
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
    timed = [
        r
        for r in stages
        if isinstance(r.get("p95_s"), int | float) and (r.get("count") or 0) >= 3 and r.get("leaf") is not False
    ]
    if not timed:
        return
    worst = max(timed, key=lambda r: r.get("total_s") or 0)
    total = sum(r.get("total_s") or 0 for r in timed)
    if total > 0 and (worst.get("total_s") or 0) >= 0.5 * total and worst["p95_s"] > 30:
        yield _finding(
            "SETUP_STAGE_SLOW",
            "warn" if worst["p95_s"] >= 300 else "info",
            "jobs",
            f"Stage '{worst.get('stage')}' takes {_dur(worst.get('p50_s') or 0)} per job (median)",
            f"{100 * worst['total_s'] / total:.0f}% of all recorded job-stage time; p95 {_dur(worst['p95_s'])} over"
            f" {worst.get('count')} jobs; lane {worst.get('lane')}",
            action="Enter on its lane in TIMELINE lists the stages by total time; the capsule inspector shows each run",
            impact_s=worst.get("total_s"),
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
    explained = (s.get("lifecycle") or {}).get("phase") in ("ended", "startup")
    if experiment.get("state") in ("LIVE", "ACTIVE") and isinstance(now, int | float) and not explained:
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
        # A soft target (no managed admission) is advice; only an enforced budget can stop the agent.
        enforced = budget.get("enforced")
        soft = enforced is False or (enforced is None and budget.get("mode") == "soft")
        if soft:
            severity = "info"
            title = f"Author token budget {used:.0f}% used" + (
                ": soft target exceeded (not enforced)" if used >= 100 else " (soft target, not enforced)"
            )
        else:
            severity, title = ("error" if used >= 100 else "warn"), f"Author token budget {used:.0f}% used"
        yield _finding(
            "TOKEN_BUDGET",
            severity,
            "tokens",
            title,
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
                f"{len(names)} capsules fail the same way: {category}",
                ", ".join(names[:4]) + (" ..." if len(names) > 4 else ""),
                action=f"one fix may clear all {len(names)}: filter CAPSULES by '{category}' and inspect one",
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
    window = (timeline.get("window") or {}).get("seconds")
    overlap = timeline.get("overlap") or {}
    bound, share = verdict.get("bound"), verdict.get("share_pct")
    if not bound or not isinstance(share, int | float) or not isinstance(window, int | float) or window < 600:
        return
    if bound in ("agent_reasoning", "unclassified_agent_time", "command_lifetimes"):
        pass  # These records do not isolate deliberation or causal tool blocking.
    elif bound == "outer_tools" and share >= 35:
        yield _finding(
            "TOOL_BOUND",
            "warn",
            "timeline",
            f"The lead waits on tool calls {share:.0f}% of the time",
            "matched outer call intervals; the awaited service is not independently identified",
            action="see which tool lanes are long in TIMELINE; Enter on a lane lists its slowest stages",
            impact_s=share / 100 * window,
        )
    elif bound == "waits" and share >= 30:
        yield _finding(
            "WAIT_BOUND",
            "warn",
            "timeline",
            f"Recorded extension/wait intervals cover {share:.0f}% of the window",
            "the transport does not independently identify the awaited service",
            action="inspect the recorded call/return and job identities before attributing a cause",
            impact_s=share / 100 * window,
        )
    elif bound == "idle" and share >= 40:
        yield _finding(
            "LEAD_IDLE",
            "warn",
            "timeline",
            f"No agent turn is open {share:.0f}% of the run",
            "the author is not working: between turns, stopped, or waiting for an admission",
            action="see AUTHOR in TIMELINE for when turns ran and why they ended",
            impact_s=share / 100 * window,
        )
    elif bound == "setup" and share >= 30 and window >= 900:
        yield _finding(
            "SETUP_DOMINATED",
            "warn",
            "timeline",
            f"{share:.0f}% of the run went to preparing the author, not authoring",
            "input preparation and the host's per-turn startup (sandbox, overlays) hold the agent back",
            action="see the startup findings below and AUTHOR in TIMELINE",
            impact_s=share / 100 * window,
        )
    blocked = overlap.get("lead_blocked_on_sim_s")
    if (overlap.get("causal_wait_verified") is True
            and isinstance(blocked, int | float) and blocked / window >= 0.15):
        yield _finding(
            "SIM_ON_CRITICAL_PATH",
            "warn",
            "timeline",
            "The lead waits while simulations run",
            f"{blocked / 60:.0f} min ({blocked / window * 100:.0f}% of the window) blocked during L2/L3 jobs",
            action="run simulations in the background (promotion) or shorten them; see the L2/L3 lanes",
            impact_s=blocked,
        )
    para = timeline.get("parallelism") or {}
    avg, ceiling, used = para.get("helpers_avg"), para.get("helper_ceiling"), para.get("helpers_used")
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
            f"{avg:.1f} helpers active on average against {ceiling} helper slots",
            f"{used or 0} helper(s) observed; independent ready work and CPU availability are not established",
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


def _lifecycle(snapshot, s):
    """The author's turns: slow or starving startups, and an author that stopped while the run goes on."""
    life = s.get("lifecycle") or {}
    turns = [t for t in life.get("turns") or [] if isinstance(t, dict)]
    timed = [t for t in turns if isinstance(t.get("startup_s"), int | float)]
    fastest: dict[str, float] = {}
    for turn in timed:
        for stage in turn.get("stages") or []:
            wall = stage.get("wall_s")
            if isinstance(wall, int | float) and stage.get("depth", 0) >= 0:
                fastest[stage["stage"]] = min(fastest.get(stage["stage"], wall), wall)
    best = min((t["startup_s"] for t in timed), default=None)
    for turn in timed:
        startup = turn["startup_s"]
        if startup < 120:
            continue
        stages = sorted(
            (st for st in turn.get("stages") or [] if isinstance(st.get("wall_s"), int | float) and st.get("depth")),
            key=lambda st: -st["wall_s"],
        ) or sorted(
            (st for st in turn.get("stages") or [] if isinstance(st.get("wall_s"), int | float)),
            key=lambda st: -st["wall_s"],
        )
        parts = []
        for stage in stages[:3]:
            text = f"{stage['stage']} {_dur(stage['wall_s'])}"
            low = fastest.get(stage["stage"])
            if low and stage["wall_s"] >= 3 * low and len(timed) > 1:
                text += f" ({stage['wall_s'] / low:.0f}x its fastest)"
            parts.append(text)
        if turn.get("open_stage"):
            parts.insert(0, f"still in {turn['open_stage']['stage']}")
        name = f"Round {turn.get('round')} turn {turn.get('turn')}"
        agent = turn.get("agent_s")
        starved = turn.get("agent_end") is not None and (agent is None or agent < 60)
        live = (snapshot.get("experiment") or {}).get("state") in ("LIVE", "STALLED", "ACTIVE")
        if turn.get("open_stage") and not live:
            title = f"{name} never finished starting: {_dur(startup)} in startup when the run stopped"
            severity = "error" if startup >= 300 else "warn"
        elif turn.get("open_stage"):
            title = f"{name} has been starting for {_dur(startup)}; the agent cannot work until it finishes"
            severity = "error" if startup >= 300 else "warn"  # The author is blocked right now.
        elif starved:
            title = f"{name} spent {_dur(startup)} starting and the agent then worked {_dur(agent or 0)}"
            severity = "error"
        else:
            title = f"{name} startup took {_dur(startup)} before the agent could work"
            severity = "warn" if startup >= 300 else "info"
        action = "startup prepares the turn's sandbox (snapshot and library overlays); inspect the turn in TIMELINE"
        if turn.get("open_stage") and not live:
            parts[0] = f"stopped in {turn['open_stage']['stage']}"
        elif turn.get("open_stage"):
            open_stage = turn["open_stage"]
            now = snapshot.get("generated_at")
            if isinstance(now, int | float):
                parts[0] = f"now in {open_stage['stage']} for {_dur(now - open_stage['since'])}"
            io = ((s.get("resources") or {}).get("process") or {}).get("io") or {}
            busy = (s.get("machine") or {}).get("busy_cpus")
            facts = []
            if isinstance(io.get("read_bps"), int | float) and io["read_bps"] >= 20e6:
                facts.append(f"the run reads {io['read_bps'] / 1e6:.0f} MB/s")
            if isinstance(busy, int | float):
                facts.append(f"the machine is {busy:.1f} CPUs busy")
            if facts:
                action = "while it starts, " + " and ".join(facts) + "; " + action
        detail = "; ".join(parts)
        if best is not None and best < startup / 3:
            detail += f"; the fastest turn started in {_dur(best)}"
        if turn.get("end_reason") and starved:
            detail += f"; ended by {turn['end_reason']}"
        yield _finding(
            "AUTHOR_STARTUP_SLOW",
            severity,
            "lifecycle",
            title,
            detail,
            action=action,
            impact_s=startup - (best if best is not None and best < startup else 0),
        )
    experiment = snapshot.get("experiment") or {}
    now = snapshot.get("generated_at")
    if life.get("phase") == "ended" and experiment.get("state") in ("LIVE", "ACTIVE"):
        since = life.get("phase_since")
        ago = f" {_dur(now - since)} ago" if isinstance(since, int | float) and isinstance(now, int | float) else ""
        yield _finding(
            "AUTHOR_STOPPED",
            "warn",
            "lifecycle",
            f"Authoring stopped{ago}; the run continues without an agent",
            f"reason: {life.get('phase_detail') or 'not recorded'}",
            action="grading and simulations still run; the agent works again only if the host admits a new turn",
        )


def _activity(snapshot, s):
    """The run's own CPUs: mostly idle while held, or busy with processes that are not this experiment's."""
    act = s.get("activity") or {}
    hist = act.get("history") or {}
    cpus = act.get("cpus") or []
    totals = act.get("totals") or {}
    if act.get("mode") != "live" or not cpus or not isinstance(hist.get("dt"), int | float):
        return
    observed = (hist.get("n") or 0) * hist["dt"]
    if observed < 300:
        return
    busy = sum(v for k, v in totals.items() if k not in ("idle", "other"))
    other = totals.get("other") or 0
    capacity = observed * len(cpus)
    if capacity and busy / capacity < 0.15:
        top = sorted(((k, v) for k, v in totals.items() if k not in ("idle", "other")), key=lambda kv: -kv[1])[:3]
        mix = ", ".join(f"{k} {100 * v / busy:.0f}%" for k, v in top) if busy else "nothing"
        yield _finding(
            "CPUS_UNDERUSED",
            "info",
            "activity",
            f"The run's {len(cpus)} CPUs are {100 * busy / capacity:.0f}% busy",
            f"over the last {_dur(observed)}: {mix}",
            action="more parallel grading or simulation slots would use them; see CORES",
        )
    if capacity and other / capacity >= 0.2:
        yield _finding(
            "CPUS_SHARED",
            "warn",
            "activity",
            f"{100 * other / capacity:.0f}% of the run's CPU time goes to other processes",
            "work outside this experiment runs on its reserved CPUs",
            action="see 'other' in CORES; Enter on a core lists what ran there",
            impact_s=other / max(1, len(cpus)),
        )


GRADE_DELIVERY_GRACE_S = 300
PROGRESS_STALE_S = 1200
SELFCHECK_PENDING_S = 1800


def _agent_view(snapshot, s):
    """What the agent can see of its own feedback, against what the operator records say happened."""
    view = s.get("agent_view") or {}
    if not view.get("workspace"):
        return
    now = snapshot.get("generated_at") or 0
    live = view.get("live")
    qa = view["workspace"].rstrip("/") + "/qa/"
    verdict, latest = view.get("verdict") or {}, view.get("latest_completed") or {}
    progress, newest = view.get("progress") or {}, view.get("newest_grade") or {}
    # A grade that ran no case (every row deferred or gated) has nothing to deliver or show.
    executed = newest.get("executed", newest.get("rows"))
    if executed and not newest.get("delivered") and now - (newest.get("at") or now) >= GRADE_DELIVERY_GRACE_S:
        yield _finding(
            "AGENT_GRADE_NOT_DELIVERED",
            "warn",
            "agent",
            f"The newest grade ({newest['rows']} rows, {executed} executed, {_dur(now - newest['at'])} ago)"
            " reached no file the agent reads",
            "neither qa/verdict.json, qa/latest_completed_grade.json nor qa/progress.json carries it",
            newest.get("path"),
            action="the agent is steering by its own self-checks; check the grade publication rules",
        )
    if verdict.get("present") and not verdict.get("rows") and not latest.get("rows") and executed:
        yield _finding(
            "AGENT_VERDICT_EMPTY",
            "warn",
            "agent",
            "The agent's verdict has no per-test-case rows"
            + (" (model gate only)" if verdict.get("model_gate_only") else ""),
            "operator grades have rows, but none is visible to the agent",
            qa + "verdict.json",
        )
    if live and progress.get("present") and newest.get("at") and progress.get("at"):
        lag = (newest["at"] or 0) - progress["at"]
        if lag >= PROGRESS_STALE_S or (now - progress["at"] >= PROGRESS_STALE_S and lag > 0):
            yield _finding(
                "AGENT_FEEDBACK_STALE",
                "warn",
                "agent",
                f"The agent's progress is {_dur(now - progress['at'])} old",
                f"a newer grade finished {_dur(max(0, now - newest['at']))} ago",
                qa + "progress.json",
            )
    suggestions = view.get("suggestions") or {}
    if live and not suggestions.get("present"):
        yield _finding(
            "AGENT_SUGGESTIONS_MISSING",
            "info",
            "agent",
            "No OPTIMIZATION_SUGGESTIONS.json in the agent's workspace",
            "the architecture-selected mechanisms never reached this agent",
            view["workspace"],
        )
    checks = view.get("selfchecks") or {}
    oldest = checks.get("oldest_pending_at")
    if live and oldest and now - oldest >= SELFCHECK_PENDING_S:
        yield _finding(
            "AGENT_SELFCHECK_UNANSWERED",
            "warn",
            "agent",
            f"{checks.get('pending')} self-check(s) pending, the oldest for {_dur(now - oldest)}",
            "requests without a response in the agent's channel",
            view["workspace"].rstrip("/") + "/.qa_channel",
        )
    if (checks.get("recent_errors") or 0) >= 3:
        yield _finding(
            "AGENT_SELFCHECK_ERRORS",
            "warn",
            "agent",
            f"{checks['recent_errors']} of the newest self-check responses are errors",
            str(checks.get("last_error") or ""),
            view["workspace"].rstrip("/") + "/.qa_channel",
        )
    for label, doc in (("verdict.json", verdict), ("latest_completed_grade.json", latest)):
        for notice in doc.get("notices") or []:
            if notice != "operator_epoch_notice":
                yield _finding("AGENT_NOTICE", "info", "agent", f"{notice} is shown to the agent", "", qa + label)


def _volume(snapshot, s):
    volume = (s.get("machine") or {}).get("volume") or {}
    level = volume.get("level")
    if level in ("warn", "critical"):
        free = (volume.get("free_bytes") or 0) / (1 << 30)
        yield _finding(
            "DISK_LOW",
            "error" if level == "critical" else "warn",
            "machine",
            f"{free:.0f} GiB free ({volume.get('free_pct')}%) on the run's volume",
            "any run on this volume can fill it; a full volume stops grading",
            volume.get("path"),
            action="free space before the run's own writes fail",
        )


def _insights(snapshot, s):
    """Findings from the insight panels: unused performance tooling, undelivered grades, integrity."""
    tools = s.get("tools") or {}
    unused = tools.get("unused_perf_tools") or []
    commands = tools.get("commands") or 0
    if unused and commands >= 50:
        yield _finding(
            "PERF_TOOLS_UNUSED",
            "info",
            "tools",
            f"{len(unused)} performance tool(s) never used in {commands} agent commands",
            ", ".join(unused),
            action="the guidance names these tools; check AGENT TOOLS (v) for what the agent uses instead",
        )
    verdicts = (s.get("grader") or {}).get("verdicts") or {}
    n, delivered = verdicts.get("n") or 0, verdicts.get("delivered")
    if n >= 3 and delivered == 0:
        p50 = verdicts.get("p50_s")
        yield _finding(
            "GRADES_UNDELIVERED",
            "warn",
            "grader",
            f"none of {n} finished verdicts was delivered: every snapshot was superseded first",
            f"grade p50 {_dur(p50)}" if p50 is not None else "",
            action="the source changes faster than a grade finishes; see GRADER (v) for durations and closures",
            impact_s=p50,
        )
    log = ((s.get("integrity") or {}).get("log") or {}).get("counts") or {}
    bad = {k: v for k, v in log.items() if k in ("marker", "grade_error", "dropped", "enospc") and v}
    if bad:
        yield _finding(
            "RUN_INTEGRITY",
            "error" if bad.get("enospc") or bad.get("dropped") else "warn",
            "integrity",
            "the campaign log records " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in bad.items()),
            "a grade that hit these is not evidence",
            action="open INTEGRITY (v) and inspect the log signal",
        )


def _cpu_list(text) -> list[int]:
    out: list[int] = []
    for item in str(text or "").split(","):
        a, sep, b = item.partition("-")
        if a.strip().isdigit() and (not sep or b.strip().isdigit()):
            out.extend(range(int(a), int(b if sep else a) + 1))
    return out


RULES = (
    _lifecycle,
    _activity,
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
    _agent_view,
    _volume,
    _insights,
)

__all__ = ["RULES", "SEVERITY_ORDER", "diagnose"]
