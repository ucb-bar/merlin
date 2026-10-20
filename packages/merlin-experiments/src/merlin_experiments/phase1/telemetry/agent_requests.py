"""Which author thread wrote each self-check request, and which simulations each request caused.

Channel records are the harness's own: ``req_<id>.json`` (written by the in-sandbox client),
``resp_<id>.json`` (the broker's redacted verdict, with grade start/end), and ``simreq_<job>.json``
(promotions and agent simulator jobs), read from the live workspace or the run's retained
``agent_evidence_snapshot``. A request is charged to a thread by the ``caller_thread_id`` it carries
when the client forwarded one; otherwise by the one submitting ``agent_selfcheck.py`` command whose
interval holds ``requested_at_unix_ns`` with the same ``--capsules`` (nearest start wins when two
threads asked for the same list at once), and failing that by the only unclaimed command running at
that moment. Simulations come from the request's retained raw results (same capsule set, results
inside its grade window; a carried tier is not a new run), else from the response's tier statuses.
A promotion is charged to a request only when that request passed the capsule and published its
verdict just before the promotion was enqueued; otherwise it stays unattributed, never guessed.

Read-only; nothing here opens answer material or decides a grade.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from merlin_experiments.phase1.telemetry import agent_commands as AC

#: Slack around a command interval when matching a request it wrote (clock granularity, start-up).
MATCH_SLACK_S = 2.0
_RESULT_SLACK_S = 5.0
#: A self-check broker promotes within seconds of publishing the verdict that earned the promotion.
PROMOTION_AFTER_S = 60.0
_RUN_STATUSES = frozenset({"pass", "fail"})


def _iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, UTC).isoformat()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _ns(value) -> float | None:
    return value / 1e9 if isinstance(value, int) and not isinstance(value, bool) else None


def _channels(run_dir: Path, workspace: str | None) -> list[Path]:
    roots = []
    if workspace:
        roots.append(Path(workspace) / ".qa_channel")
    roots.append(Path(run_dir) / "agent_evidence_snapshot" / ".qa_channel")
    return [root for root in roots if root.is_dir()]


def _channel_records(channels: list[Path]) -> tuple[dict, dict]:
    requests: dict[str, dict] = {}
    jobs: dict[str, dict] = {}
    for channel in channels:
        for path in sorted(channel.glob("req_*.json")):
            rid = path.name[len("req_") : -len(".json")]
            if rid in requests:
                continue
            body = _read_json(path)
            if not isinstance(body, dict):
                continue
            response = _read_json(channel / f"resp_{rid}.json")
            requests[rid] = {"request": body, "response": response if isinstance(response, dict) else None}
        for path in sorted(channel.glob("simreq_*.json")):
            job = path.name[len("simreq_") : -len(".json")]
            body = _read_json(path)
            if job in jobs or not isinstance(body, dict):
                continue
            response = _read_json(channel / f"simresp_{job}.json")
            jobs[job] = {"request": body, "response": response if isinstance(response, dict) else None}
    return requests, jobs


def _raw_selfchecks(run_dir: Path) -> list[dict]:
    """Retained raw self-check runs: capsule set, result-time window and per-tier execution facts."""
    out = []
    for check in sorted((Path(run_dir) / "_qa_work" / "selfchecks").glob("*")):
        if not check.is_dir():
            continue
        results = []
        for path in sorted((check / "runs").glob("*/*/capsule_result.json")):
            data = _read_json(path)
            try:
                at = path.stat().st_mtime
            except OSError:
                continue
            if isinstance(data, dict):
                results.append((at, str(data.get("capsule") or path.parent.name), data.get("tiers") or {}))
        if results:
            times = [r[0] for r in results]
            out.append(
                {
                    "id": check.name,
                    "first": min(times),
                    "last": max(times),
                    "capsules": {r[1] for r in results},
                    "results": results,
                }
            )
    return out


def _sims_from_results(results) -> dict:
    by_engine: dict[str, int] = defaultdict(int)
    carried = 0
    for _at, _capsule, tiers in results:
        for tier in tiers.values() if isinstance(tiers, dict) else ():
            if not isinstance(tier, dict) or tier.get("status") not in _RUN_STATUSES:
                continue
            engine = tier.get("engine")
            if not engine:
                continue
            if tier.get("carried") or tier.get("measured_now") is False:
                carried += 1
            else:
                by_engine[str(engine)] += 1
    return {"by_engine": dict(sorted(by_engine.items())), "carried": carried, "source": "raw_results"}


def _sims_from_response(response: dict) -> dict:
    engines = response.get("tier_engines") if isinstance(response.get("tier_engines"), dict) else {}
    by_engine: dict[str, int] = defaultdict(int)
    for row in response.get("per_capsule") or []:
        for tier, status in (row.get("tiers") or {}).items() if isinstance(row, dict) else ():
            if status in _RUN_STATUSES and engines.get(tier):
                by_engine[str(engines[tier])] += 1
    return {
        "by_engine": dict(sorted(by_engine.items())),
        "carried": None,
        "source": "response_tier_statuses",
        "note": "redacted response statuses; reused (carried) tiers cannot be told apart here",
    }


def _capsule_names(value) -> list[str]:
    return [name.strip() for name in str(value or "").split(",") if name.strip()]


def _request_record(rid: str, entry: dict) -> dict:
    request, response = entry["request"], entry["response"] or {}
    requested = _ns(request.get("requested_at_unix_ns"))
    started = _ns(response.get("grade_started_at_unix_ns"))
    graded = _ns(response.get("graded_at_unix_ns"))
    labels = {
        k: request[k] for k in ("variant", "submission", "submission_path", "label") if isinstance(request.get(k), str)
    }
    probes = [
        k
        for k in (
            "compile_only",
            "scale_probe",
            "candidate_diagnostics",
            "shape_coverage",
            "offload_census",
            "model_layers",
            "dynamic_census",
        )
        if request.get(k)
    ]
    return {
        "request_id": rid,
        "requested_at": _iso(requested),
        "capsules": request.get("capsules", "all"),
        "n_capsules": None if request.get("capsules", "all") == "all" else len(_capsule_names(request.get("capsules"))),
        "sim": request.get("sim"),
        "tiers": request.get("tiers"),
        "mode": "+".join(probes) if probes else ("caller_layout" if request.get("caller_layout") else "check"),
        **({"request_labels": labels} if labels else {}),
        "caller_thread_id": request.get("caller_thread_id")
        if isinstance(request.get("caller_thread_id"), str)
        else None,
        "answered": bool(response),
        "grade_started_at": _iso(started),
        "graded_at": _iso(graded),
        "queue_s": round(started - requested, 3) if started is not None and requested is not None else None,
        "latency_s": round(graded - requested, 3) if graded is not None and requested is not None else None,
        "all_pass": response.get("all_pass"),
        "n_passed": response.get("n_passed"),
        "n_result_rows": response.get("n_result_rows"),
        "graded_submission_sha256": response.get("submission_sha256"),
        "_requested": requested,
        "_started": started,
        "_graded": graded,
        "_per_capsule": [str(row.get("capsule")) for row in response.get("per_capsule") or [] if isinstance(row, dict)],
        "_response": response,
    }


def _join_raw(requests: list[dict], raw: list[dict]) -> None:
    """Each answered request's retained raw run: same capsule set, results inside its grade window."""
    used: set[str] = set()
    for req in sorted((r for r in requests if r["_started"] is not None), key=lambda r: r["_started"]):
        wanted = set(req["_per_capsule"])
        lo = req["_started"] - _RESULT_SLACK_S
        hi = (req["_graded"] if req["_graded"] is not None else math.inf) + _RESULT_SLACK_S
        candidates = [
            d for d in raw if d["id"] not in used and d["capsules"] == wanted and lo <= d["first"] and d["last"] <= hi
        ]
        if candidates:
            chosen = min(candidates, key=lambda d: d["first"])
            used.add(chosen["id"])
            req["simulations"] = {
                **_sims_from_results(chosen["results"]),
                "raw_selfcheck": chosen["id"],
                "raw_candidates": len(candidates),
            }
        elif req["answered"]:
            req["simulations"] = _sims_from_response(req["_response"])
        else:
            req["simulations"] = {"by_engine": {}, "carried": None, "source": None, "note": "no response yet"}


def _passed(response: dict, capsule: str) -> bool:
    for row in response.get("per_capsule") or []:
        if isinstance(row, dict) and row.get("capsule") == capsule and row.get("pass") is True:
            return True
    return False


def _join_promotions(requests: list[dict], jobs: dict) -> list[dict]:
    """Promoted cert jobs, each charged to the self-check that earned it.

    The self-check broker enqueues a promotion right after it publishes a passing verdict, but the
    in-turn background grade promotes too, and the promotion's ``submission_digest`` uses a different
    hash than the request's. So a job is charged only to a request whose response passed that capsule
    and was graded at most ``PROMOTION_AFTER_S`` before the job was submitted; otherwise it stays
    unattributed (most likely a background-grade promotion), never guessed.
    """
    promotions = []
    for job, entry in sorted(jobs.items()):
        body = entry["request"]
        if not body.get("promoted"):
            continue
        submitted = body.get("submitted_at") if isinstance(body.get("submitted_at"), (int, float)) else None
        capsule = str(body.get("capsules") or "")
        owners = [
            r
            for r in requests
            if submitted is not None
            and r["_graded"] is not None
            and r["_graded"] - 1.0 <= submitted <= r["_graded"] + PROMOTION_AFTER_S
            and _passed(r["_response"], capsule)
        ]
        owner = max(owners, key=lambda r: r["_graded"]) if owners else None
        promotions.append(
            {
                "job_id": job,
                "request_id": owner["request_id"] if owner else None,
                "basis": "passed_capsule_then_promoted" if owner else None,
                "sim": body.get("sim"),
                "tiers": body.get("tiers"),
                "capsules": body.get("capsules"),
                "submitted_at": _iso(float(submitted)) if submitted is not None else None,
                "state": (entry["response"] or {}).get("state") if entry["response"] else "pending",
            }
        )
    return promotions


def _invocations(records: dict) -> list[dict]:
    calls = []
    for tid, rec in records.items():
        for command in rec["metrics"]["commands"]:
            for call in AC.client_calls(command["item"].get("command")):
                calls.append({**call, "thread_id": tid, "start": command["start"], "end": command["end"]})
    return calls


def _attribute(requests: list[dict], calls: list[dict], records: dict) -> None:
    """Charge each request to a thread: forwarded thread id first, else the command that wrote it."""
    submitting = [c for c in calls if c["client"] == AC.SELFCHECK and c["submits_request"]]
    for req in requests:
        req["thread_id"], req["basis"], req["candidates"] = None, None, 0
        if req["caller_thread_id"] in records:
            req["thread_id"], req["basis"] = req["caller_thread_id"], "caller_thread_id"
    pairs = []
    for i, req in enumerate(requests):
        if req["thread_id"] is not None or req["_requested"] is None:
            continue
        for j, call in enumerate(submitting):
            if call["start"] is None or call["end"] is None:
                continue
            if not call["start"] - MATCH_SLACK_S <= req["_requested"] <= call["end"] + MATCH_SLACK_S:
                continue
            if call["capsules"] != str(req["capsules"]):
                continue
            req["candidates"] += 1
            pairs.append((abs(req["_requested"] - call["start"]), i, j))
    taken_req, taken_call = set(), set()
    for _gap, i, j in sorted(pairs):
        if i in taken_req or j in taken_call:
            continue
        taken_req.add(i)
        taken_call.add(j)
        call = submitting[j]
        requests[i].update(
            thread_id=call["thread_id"], basis="command_window", submission=call["submission"], variant=call["variant"]
        )
    # A capsule list the parser could not read literally (a shell variable) still has one writer when
    # exactly one unclaimed submitting command was running at that moment.
    for i, req in enumerate(requests):
        if req["thread_id"] is not None or req["_requested"] is None:
            continue
        window = [
            j
            for j, call in enumerate(submitting)
            if j not in taken_call
            and call["start"] is not None
            and call["end"] is not None
            and call["start"] - MATCH_SLACK_S <= req["_requested"] <= call["end"] + MATCH_SLACK_S
        ]
        req["candidates"] = max(req["candidates"], len(window))
        if len(window) == 1:
            taken_call.add(window[0])
            call = submitting[window[0]]
            req.update(
                thread_id=call["thread_id"],
                basis="command_window_unmatched_capsules",
                submission=call["submission"],
                variant=call["variant"],
            )
    for req in requests:
        req.setdefault("submission", None)
        req.setdefault("variant", None)


def _attribute_simjobs(jobs: dict, calls: list[dict]) -> list[dict]:
    submits = [c for c in calls if c["client"] == AC.SIMJOB and c["submits_request"]]
    out = []
    for job, entry in sorted(jobs.items()):
        body = entry["request"]
        if body.get("promoted"):
            continue
        submitted = body.get("submitted_at") if isinstance(body.get("submitted_at"), (int, float)) else None
        owners = [
            c
            for c in submits
            if submitted is not None
            and c["start"] is not None
            and c["end"] is not None
            and c["start"] - MATCH_SLACK_S <= submitted <= c["end"] + MATCH_SLACK_S
            and c["capsules"] == str(body.get("capsules", "all"))
        ]
        owner = min(owners, key=lambda c: abs(submitted - c["start"])) if owners else None
        out.append(
            {
                "job_id": job,
                "thread_id": owner["thread_id"] if owner else None,
                "basis": "command_window" if owner else None,
                "candidates": len(owners),
                "sim": body.get("sim"),
                "capsules": body.get("capsules"),
                "submitted_at": _iso(float(submitted)) if submitted is not None else None,
            }
        )
    return out


def collect(run_dir: Path, workspace: str | None, records: dict) -> dict:
    """Requests, promotions and simulator jobs of one run, each with the thread it is charged to."""
    requests_raw, jobs = _channel_records(_channels(Path(run_dir), workspace))
    requests = [_request_record(rid, entry) for rid, entry in sorted(requests_raw.items())]
    calls = _invocations(records)
    _attribute(requests, calls, records)
    _join_raw(requests, _raw_selfchecks(Path(run_dir)))
    return {
        "requests": requests,
        "promotions": _join_promotions(requests, jobs),
        "simjobs": _attribute_simjobs(jobs, calls),
        "calls": calls,
    }
