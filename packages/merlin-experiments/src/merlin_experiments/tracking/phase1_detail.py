"""Phase 1 jobs, self-checks and grades on one timeline, with family pass rates and recorded cost and time.

Read, never written:

* ``selfcheck_log.jsonl`` -- one line per self-check the agent ran, offset from the self-check tool's T0
  (the authoring start; anchored here at ``qa_loop_state.yaml`` ``cumulative.started_at`` and labelled so);
* the self-check and simulation-job channel (``.qa_channel/``) the brokers keep in the live workspace
  (``environment.yaml`` ``workspace_path``) and that the run snapshots into ``agent_evidence_snapshot/``
  at its end: request files carry when a job was asked for, marker files (``simrun_``, ``simdone_``,
  ``simerr_``, ``done_``) say when it started and ended -- the latter by their file times, labelled so;
* ``qa_history/verdict_*.json`` (through :mod:`.records`), ``freeze.json`` and the operator seal in
  ``qa_loop_summary.yaml``;
* ``cost_time_toolcalls.yaml``, ``timing_detailed.json`` and ``qa_loop_state.yaml`` rounds.

A capsule's family comes from its own ``capsule.yaml`` (``semantic.semantic_family``, else the owner's
op -> family table) found under the corpus roots the run names; a capsule with no readable capsule.yaml
is in family "not recorded".
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import records as R

#: Most channel entries read per poll (a long run's channel can hold thousands of finished jobs).
CHANNEL_LIMIT = 4000


def _mtime(path: Path) -> float | None:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def _json_file(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _ns(value: Any) -> float | None:
    return value / 1e9 if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


# --------------------------------------------------------------------------- the run's own context
def environment(run_dir: Path, inventory: R.Inventory) -> dict[str, Any]:
    env = inventory.yaml("environment.yaml", Path(run_dir) / "environment.yaml") or {}
    return {
        "workspace": env.get("workspace_path"),
        "started_at": R.epoch(env.get("started_at")),
        "model": env.get("model"),
        "driver": env.get("driver"),
        "corpus": env.get("public_corpus_input"),
    }


def channel_dirs(run_dir: Path, workspace: str | None) -> list[Path]:
    """The job channel: the live workspace's, then the run's end-of-run snapshot."""
    out = []
    if workspace:
        out.append(Path(workspace) / ".qa_channel")
    out.append(Path(run_dir) / "agent_evidence_snapshot" / ".qa_channel")
    return [p for p in out if p.is_dir()]


def jobs(channels: Sequence[Path], inventory: R.Inventory) -> list[dict[str, Any]]:
    """Simulation jobs and self-check requests found in the channel, with their recorded times."""
    if not channels:
        inventory.note(".qa_channel (simulation jobs, self-checks)", Path("."), "absent")
        return []
    found: dict[str, dict[str, Any]] = {}
    for channel in channels:
        try:
            names = sorted(os.listdir(channel))[-CHANNEL_LIMIT * 4 :]
        except OSError:
            continue
        present = set(names)
        for name in names:
            if name.startswith("simreq_") and name.endswith(".json"):
                jid = name[len("simreq_") : -len(".json")]
                if jid in found:
                    continue
                request = _json_file(channel / name) or {}
                response = _json_file(channel / f"simresp_{jid}.json") if f"simresp_{jid}.json" in present else None
                done = f"simdone_{jid}" in present
                failed = f"simerr_{jid}" in present
                end = _mtime(channel / (f"simdone_{jid}" if done else f"simerr_{jid}")) if done or failed else None
                found[jid] = {
                    "kind": "simjob",
                    "id": jid,
                    "sim": str(request.get("sim") or "not recorded"),
                    "capsules": request.get("capsules"),
                    "promoted": bool(request.get("promoted")),
                    "tiers": request.get("tiers"),
                    "requested": R.epoch(request.get("submitted_at")),
                    "started": _mtime(channel / f"simrun_{jid}") if f"simrun_{jid}" in present else None,
                    "ended": end,
                    "state": "error"
                    if failed
                    else "done"
                    if done
                    else "running"
                    if f"simrun_{jid}" in present
                    else "queued",
                    "all_pass": (response or {}).get("all_pass"),
                    "error": (response or {}).get("error"),
                }
            elif name.startswith("req_") and name.endswith(".json"):
                rid = name[len("req_") : -len(".json")]
                key = f"selfcheck:{rid}"
                if key in found:
                    continue
                request = _json_file(channel / name) or {}
                progress = _json_file(channel / f"progress_{rid}.json") if f"progress_{rid}.json" in present else None
                response = _json_file(channel / f"resp_{rid}.json") if f"resp_{rid}.json" in present else None
                done_path = channel / f"done_{rid}"
                outcome = None
                if f"done_{rid}" in present:
                    try:
                        outcome = done_path.read_text(encoding="utf-8").strip() or "ok"
                    except OSError:
                        outcome = "ok"
                found[key] = {
                    "kind": "selfcheck",
                    "id": rid,
                    "sim": str(request.get("sim") or "not recorded"),
                    "capsules": request.get("capsules"),
                    "promoted": False,
                    "tiers": request.get("tiers"),
                    "requested": _ns(request.get("requested_at_unix_ns")),
                    "started": _ns((progress or {}).get("started_at_unix_ns")),
                    "ended": _mtime(done_path) if outcome else None,
                    "state": ("error" if outcome == "err" else "done")
                    if outcome
                    else "running"
                    if progress
                    else "queued",
                    "all_pass": (response or {}).get("all_pass"),
                    "n_passed": (response or {}).get("n_passed"),
                    "n_capsules": (response or {}).get("n_capsules"),
                    "error": (response or {}).get("error"),
                }
        inventory.note(".qa_channel (simulation jobs, self-checks)", channel, "read", f"{len(found)} jobs so far")
    rows = sorted(found.values(), key=lambda j: j["requested"] or j["started"] or 0.0)
    return rows[-CHANNEL_LIMIT:]


def selfcheck_log(run_dir: Path, anchor: float | None, inventory: R.Inventory, tails) -> dict[str, Any] | None:
    path = Path(run_dir) / "selfcheck_log.jsonl"
    rows: list[dict[str, Any]] = tails.state.setdefault(f"selfcheck:{path}", [])

    def add(row: Mapping[str, Any]) -> None:
        offset = row.get("wall_offset_s")
        rows.append(
            {
                "offset": offset if isinstance(offset, int | float) and not isinstance(offset, bool) else None,
                "sim": row.get("sim"),
                "capsules": row.get("capsules"),
                "n_passed": row.get("n_passed"),
                "n_capsules": row.get("n_capsules"),
                "all_pass": row.get("all_pass"),
                "build_failed": row.get("build_failed"),
                "failing": list(row.get("failing") or ())[:20],
            }
        )

    tail = tails.tail(path)
    tail.poll(add, on_reset=rows.clear)
    if not tail.present:
        inventory.note("selfcheck_log.jsonl", path, "absent")
        return None
    inventory.note("selfcheck_log.jsonl", path, "read", f"{len(rows)} self-checks")
    for row in rows:
        row["at"] = anchor + row["offset"] if anchor is not None and row["offset"] is not None else None
    return {"rows": rows, "anchored": anchor is not None}


# --------------------------------------------------------------------------- families
def _corpus_roots(run_dir: Path, env: Mapping[str, Any], extra: Sequence[Path]) -> list[Path]:
    roots = [Path(p) for p in extra]
    corpus = env.get("corpus")
    if isinstance(corpus, Mapping):
        for key in ("path", "root", "capsules_root", "public_root"):
            if corpus.get(key):
                roots.append(Path(str(corpus[key])))
    elif isinstance(corpus, str):
        roots.append(Path(corpus))
    if env.get("workspace"):
        ws = Path(str(env["workspace"]))
        roots += [ws / "capsules", ws / "public", ws / "corpus", ws / "public_capsules"]
    roots += [Path(run_dir) / "_frozen_corpus"]
    return [r for r in roots if r.is_dir()]


def capsule_families(roots: Sequence[Path], names: set[str]) -> dict[str, str]:
    """``{capsule: family}`` for the named capsules, from their own capsule.yaml (first root wins)."""
    from merlin.common.yaml import safe_load_text

    from .phase0 import _family_of

    out: dict[str, str] = {}
    for root in roots:
        for pattern in ("*/*/capsule.yaml", "*/capsule.yaml", "*/*/*/capsule.yaml"):
            for path in root.glob(pattern):
                name = path.parent.name
                if name not in names or name in out:
                    continue
                try:
                    cap = safe_load_text(path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 -- unreadable capsule: family stays not recorded
                    continue
                if isinstance(cap, Mapping):
                    family = _family_of(
                        (cap.get("operation") or {}).get("op"), (cap.get("semantic") or {}).get("semantic_family")
                    )
                    if family:
                        out[name] = family
    return out


def family_rates(grades: Sequence[Mapping[str, Any]], families: Mapping[str, str]) -> dict[str, list]:
    """Per family, ``[(graded_at, passed share, tip)]`` over the grades that graded it."""
    series: dict[str, list] = {}
    for grade in grades:
        if grade.get("at") is None:
            continue
        counts: dict[str, list[int]] = {}
        for capsule in grade.get("capsules") or ():
            family = families.get(capsule["capsule"], "not recorded")
            entry = counts.setdefault(family, [0, 0])
            entry[1] += 1
            entry[0] += capsule.get("status") == "pass"
        for family, (passed, total) in counts.items():
            series.setdefault(family, []).append(
                (
                    grade["at"],
                    passed / total,
                    f"{family}: {passed}/{total} pass\n{grade['name']} {R.stamp(grade['at'])}",
                )
            )
    return series


# --------------------------------------------------------------------------- cost and time
def cost_and_time(run_dir: Path, inventory: R.Inventory) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cost = inventory.yaml("cost_time_toolcalls.yaml", run_dir / "cost_time_toolcalls.yaml")
    timing = inventory.json("timing_detailed.json", run_dir / "timing_detailed.json")
    state = inventory.yaml("qa_loop_state.yaml", run_dir / "qa_loop_state.yaml")
    keys = (
        "model",
        "wall_time_seconds",
        "active_wall_s",
        "rate_limit_wait_s",
        "tokens_input",
        "tokens_cached",
        "tokens_output",
        "tokens_reasoning",
        "tokens_total",
        "tool_calls",
        "billing_mode",
        "subscription_notional_usd",
        "estimated_cost_usd",
        "usage_complete",
    )
    by_tool = (
        ((timing or {}).get("tools") or {}).get("by_tool") if isinstance((timing or {}).get("tools"), Mapping) else {}
    )
    rounds = []
    for row in (state or {}).get("rounds") or ():
        if isinstance(row, Mapping):
            rounds.append(
                {
                    k: row.get(k)
                    for k in (
                        "round",
                        "n_passed",
                        "n_capsules",
                        "tool_calls",
                        "tokens_input",
                        "tokens_output",
                        "tokens_total",
                        "agent_rc",
                        "all_pass",
                        "mode",
                    )
                }
            )
    cumulative = (state or {}).get("cumulative") if isinstance((state or {}).get("cumulative"), Mapping) else {}
    return {
        "cost": {k: cost.get(k) for k in keys} if cost else None,
        "timing": {
            k: timing.get(k)
            for k in (
                "method",
                "think_generate_s",
                "tool_and_wait_s",
                "think_pct",
                "measured_span_s",
                "sessions",
                "tool_calls_matched",
                "tool_calls_unterminated",
                "tool_concurrency_overlap_s",
            )
        }
        if timing
        else None,
        "by_tool": {
            str(name): {
                k: v.get(k)
                for k in (
                    "calls_started",
                    "calls_completed",
                    "errors",
                    "duration_p50_s",
                    "duration_p90_s",
                    "seconds_sum",
                )
                if k in v
            }
            for name, v in (by_tool or {}).items()
            if isinstance(v, Mapping)
        },
        "rounds": rounds,
        "started": R.epoch(cumulative.get("started_at")),
    }


# --------------------------------------------------------------------------- the timeline
def lanes(
    grades: Sequence[Mapping[str, Any]],
    jobs_rows: Sequence[Mapping[str, Any]],
    checks: Mapping[str, Any] | None,
    freeze: Mapping[str, Any] | None,
    seal: Mapping[str, Any] | None,
    now: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.append(
        {
            "lane": "grades",
            "points": [
                {
                    "at": g["at"],
                    "kind": "pass" if g.get("all_pass") else "grade",
                    "tip": f"{g['name']}\n{R.stamp(g['at'])}\npassed {g.get('n_passed')}/{g.get('n_capsules')}, "
                    f"highest tier {g.get('highest_tier')}",
                }
                for g in grades
                if g.get("at") is not None
            ],
        }
    )
    if checks:
        out.append(
            {
                "lane": "self-checks (log)",
                "points": [
                    {
                        "at": c["at"],
                        "kind": "fail" if c.get("build_failed") or c.get("all_pass") is False else "pass",
                        "tip": f"self-check {c.get('sim')} on {c.get('capsules')}\n{c.get('n_passed')}/"
                        f"{c.get('n_capsules')} pass"
                        + (" (build failed)" if c.get("build_failed") else "")
                        + "\n(time = authoring start + recorded offset)",
                    }
                    for c in checks["rows"]
                ],
            }
        )
    engines: dict[str, list[dict[str, Any]]] = {}
    for job in jobs_rows:
        start = job.get("started") or job.get("requested")
        if start is None:
            continue
        end = job.get("ended") if job["state"] in ("done", "error") else None
        if job["state"] in ("done", "error") and end is None:
            end = start
        kind = {"error": "fail", "done": "pass" if job.get("all_pass") is not False else "fail"}.get(
            job["state"], "sim"
        )
        label = f"{job['kind']} {job['sim']}" + (" (promoted)" if job.get("promoted") else "")
        engines.setdefault(f"{job['kind']}: {job['sim']}", []).append(
            {
                "start": start,
                "end": end,
                "kind": kind,
                "tip": f"{label} {job['id']}\ncapsules {job.get('capsules')}\nstate {job['state']}"
                + (f", all_pass {job.get('all_pass')}" if job.get("all_pass") is not None else "")
                + ("\nend = marker file time" if job.get("ended") else ""),
            }
        )
    out += [{"lane": name, "spans": spans} for name, spans in sorted(engines.items())]
    marks = []
    if freeze and freeze.get("at"):
        marks.append(
            {
                "at": freeze["at"],
                "kind": "freeze",
                "tip": f"frozen {R.stamp(freeze['at'])}\ncommit {str(freeze.get('frozen_commit') or '')[:12]}",
            }
        )
    if seal and seal.get("at"):
        marks.append(
            {
                "at": seal["at"],
                "kind": "freeze",
                "tip": f"operator seal requested {R.stamp(seal['at'])}\n{seal.get('reason') or ''}",
            }
        )
    out.append({"lane": "freeze / seal", "points": marks})
    return out


def seal(run_dir: Path) -> dict[str, Any] | None:
    """The operator seal ``qa_loop_summary.yaml`` records (already listed in the inventory by :mod:`.records`)."""
    summary = R.Inventory().yaml("qa_loop_summary.yaml", Path(run_dir) / "qa_loop_summary.yaml")
    operator = (summary or {}).get("operator_seal")
    if not isinstance(operator, Mapping):
        return None
    return {"at": R.epoch(operator.get("requested_at")), "reason": operator.get("reason")}


def read(
    run_dir: Path,
    p1: Mapping[str, Any],
    inventory: R.Inventory,
    tails,
    *,
    now: float,
    corpus: Sequence[Path] = (),
) -> dict[str, Any]:
    """Everything the extra Phase 1 sections draw, beside the existing phase-1 summary ``p1``."""
    env = environment(run_dir, inventory)
    cost = cost_and_time(run_dir, inventory)
    anchor = cost["started"] or env["started_at"]
    checks = selfcheck_log(run_dir, anchor, inventory, tails)
    channel = jobs(channel_dirs(run_dir, env["workspace"]), inventory)
    names = {c["capsule"] for g in p1.get("grades") or () for c in g.get("capsules") or ()}
    key = f"families:{run_dir}"
    cached = tails.state.get(key)
    if cached is None or not names <= set(cached[0]):
        roots = _corpus_roots(run_dir, env, corpus)
        families = capsule_families(roots, names) if names else {}
        cached = (sorted(names), families, [str(r) for r in roots])
        tails.state[key] = cached
    _, families, roots = cached
    operator_seal = seal(run_dir)
    return {
        "environment": env,
        "cost": cost,
        "selfchecks": checks,
        "jobs": channel,
        "families": families,
        "family_roots": roots,
        "family_rates": family_rates(p1.get("grades") or (), families),
        "seal": operator_seal,
        "lanes": lanes(p1.get("grades") or (), channel, checks, p1.get("freeze"), operator_seal, now),
    }


__all__ = ["capsule_families", "family_rates", "jobs", "lanes", "read"]
