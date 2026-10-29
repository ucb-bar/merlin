"""Read-only paper metrics from native telemetry and explicitly joined evidence.

No simulator, author, compiler, model fitting or significance test is started.
Missing evidence stays null. Tokens at a milestone cover completed provider
responses only: usage inside an unfinished response cannot be reconstructed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "merlin.study.tooling_metrics.v1"
CONTEXT = "study-metrics-context.json"
EVENTS = "study-metrics-events.jsonl"
TOKEN_KEYS = ("input_total", "fresh_input", "cache_read", "cache_write", "output", "reasoning", "total")
DEFINITIONS = {
    "experimental_unit": "one independent author run; cases and ordered hardware samples are not replicas",
    "tokens": "native per-thread ownership, fork-history exclusion and per-response deduplication; cached input included once",
    "milestone_tokens": "cumulative completed-response tokens observed by publication time; lower bound on in-flight usage",
    "first_correct_from_scratch": "first public numerical pass with a changed source and passing L2 execution in a natively admitted empty-backend run; no initial executable is assumed",
    "first_correct_changed_kernel": "public-case numerical pass, no explicit static/order failure, changed source and changed ELF versus its pinned initial-case ELF",
    "model_improvement": "first numerically passing changed source/case reducing complete-call modeled cycles >=5% or >=10%; native compare requires identical capsule, selection, model, domain and scope",
    "invalid_attempt": "one request/snapshot + source + public case; earliest explicit static, numeric or ordering failure; unavailable/incomplete is not failure",
    "repair_episode": "per-case consecutive failed attempts closed by a later passing changed source; failures after a success begin a new episode",
    "repeated_failure": "each additional identical structured failure signature inside a per-case repair episode",
    "coverage_auc": "step integral of latest published coherent snapshot's public numerical-pass fraction; zero published evidence before first grade; not a cross-source mosaic or claim about current ungraded edits",
    "observed_coverage_auc": "same published-evidence integral after the first valid snapshot only; the earlier interval remains explicitly unobserved",
    "checkpoints": "only retained source snapshots; requested times are not reconstructed from later source or equal performance",
    "mechanism": "explicit checked emitted-artifact receipt joined to exact source/case/artifact digest; counts receipt observations, not independent re-verification or physical speedup",
    "timings": "native request queue/latency, host stages and per-oracle wait samples; inclusive nested stage times are never summed",
    "statistics": "descriptive within-run metrics only; no p-values and no treating cases as independent author replicas",
}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _at(value):
    from merlin_experiments.phase1.telemetry.timeline import _stamp
    return float(value) if _number(value) else _stamp(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _read(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError("metrics input must be a JSON object")
    return value


def _lines(path):
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue  # live partial record is not an event
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _distribution(values):
    from merlin_experiments.phase1.telemetry.timeline import _percentile
    values = [float(v) for v in values if _number(v) and v >= 0]
    return dict(n=len(values), p50=_percentile(values, .5), p95=_percentile(values, .95),
                total_s=sum(values) if values else None)


def _usage_events(run_dir, decomposition):
    """Reuse native OWN-record and token arithmetic owners; do not parse forked provider history twice."""
    if "provider_usage_events" in decomposition:
        return decomposition["provider_usage_events"]
    from merlin_experiments.phase1.telemetry import author_agents
    found = author_agents.threads(Path(run_dir), codex_homes=decomposition.get("codex_homes", ()))
    rows, seen = [], set()
    for thread, record in found["records"].items():
        for raw in record["rollout"]["rows"]:
            payload = raw.get("payload") or {}
            if raw.get("type") != "token_usage_record" or not isinstance(payload.get("usage"), dict):
                continue
            response = payload.get("response_id") or raw.get("ordinal")
            key = (thread, response)
            if response is None or key in seen:
                continue
            seen.add(key)
            rows.append(dict(thread_id=thread, response_id=str(response), at=raw.get("timestamp"),
                             tokens=author_agents._tokens([payload["usage"]], None)))
    return rows


def _token_summary(decomposition, usage):
    agents = decomposition.get("author_agents") or {}
    totals = agents.get("totals") or {}
    known_agents = agents.get("agents") or []
    complete = (agents.get("available") is True and agents.get("complete") is True
                and decomposition.get("telemetry_integrity", {}).get("complete") is True
                and bool(known_agents) and all(row.get("tokens", {}).get("available") is True for row in known_agents))
    result = dict(provider_usage_complete=complete, observed_totals_are_lower_bounds=not complete,
                  threads=totals.get("threads"), helper_threads=totals.get("subagents"),
                  missing_leads=agents.get("missing_leads"), missing_helpers=agents.get("missing_subagents"))
    for label, key in (("all_agents", "tokens_all"), ("lead", "tokens_lead"), ("helpers", "tokens_subagents")):
        bucket = totals.get(key) or {}
        result[label] = {key: bucket.get(key) if bucket.get("available") else None for key in TOKEN_KEYS}
    unique, seen = [], set()
    for row in usage:
        key = row.get("thread_id"), row.get("response_id")
        if not all(isinstance(v, str) and v for v in key) or key in seen:
            continue
        seen.add(key)
        if _at(row.get("at")) is not None and all(_number(row.get("tokens", {}).get(k)) for k in TOKEN_KEYS):
            unique.append(row)
    observed = sum(row["tokens"]["total"] for row in unique) if unique else None
    expected = result["all_agents"]["total"]
    result["response_chronology_complete"] = bool(complete and observed is not None and observed == expected)
    result["response_events"] = len(unique)
    return result, sorted(unique, key=lambda row: _at(row["at"]))


def _milestone(event, context, usage, tokens):
    if event is None:
        return dict(at=None, elapsed_s=None, case=None, source_sha256=None,
                    completed_response_tokens=None, token_count_is_lower_bound=True)
    at, start = _at(event.get("at")), _at(context.get("started_at"))
    prior = [row for row in usage if at is not None and _at(row["at"]) <= at]
    return dict(at=event.get("at"), elapsed_s=at - start if at is not None and start is not None and at >= start else None,
        case=event.get("case"), source_sha256=event.get("source_sha256"), elf_sha256=event.get("elf_sha256"),
        completed_response_tokens={key: sum(row["tokens"][key] for row in prior) for key in TOKEN_KEYS} if prior else None,
        token_count_is_lower_bound=True, provider_usage_complete=tokens["provider_usage_complete"],
        response_chronology_complete=tokens["response_chronology_complete"])


def _public(event, context):
    roster = context.get("public_case_ids")
    return isinstance(roster, list) and event.get("case") in roster


def _passed(event):
    return (event.get("numeric_status") == "pass" or event.get("numeric_pass") is True) and not any(
        event.get(key) == "fail" for key in ("static_status", "ordering_status")) and _failure(event) is None


def _failure(event):
    for category, key in (("static", "static_status"), ("numeric", "numeric_status"), ("ordering", "ordering_status")):
        if event.get(key) == "fail":
            return category
    plane, category = event.get("failure_plane"), event.get("failure_category")
    if plane in {"static", "compiler", "package", "source", "structural", "ir_verification"}:
        return "static"
    if plane in {"numeric", "numerics", "comparison"} or category in {"numeric_mismatch", "output_mismatch"}:
        return "numeric"
    if plane in {"ordering", "race", "trace"} or category in {"race", "ordering_violation", "hazard"}:
        return "ordering"
    return "other" if event.get("status") == "fail" else None


def _attempts(events, context):
    selected = {}
    start, end = _at(context.get("started_at")), _at(context.get("observation_end_at"))
    for event in events:
        if event.get("kind") != "case_result" or not _public(event, context):
            continue
        at = _at(event.get("at"))
        if at is None or (start is not None and at < start) or (end is not None and at > end):
            continue
        key = event.get("attempt_id"), event.get("source_sha256"), event.get("case")
        if not all(isinstance(v, str) and v for v in key):
            continue
        # Repeated publication of one native attempt is not a new paid attempt.
        prior = selected.get(key)
        if prior is None or (_at(event.get("at")) or -1) > (_at(prior.get("at")) or -1):
            selected[key] = event
    return sorted(selected.values(), key=lambda row: _at(row.get("at")) or -1)


def _repairs(attempts):
    active, episodes, repeated = {}, [], 0
    for event in attempts:
        case, failure = event["case"], _failure(event)
        if failure:
            signature = _digest([failure, event.get("failure_plane"), event.get("failure_category"), event.get("failure_tier")])
            episode = active.setdefault(case, dict(case=case, started_at=event.get("at"), attempts=0,
                repeated_failures=0, signatures=[], first_source=event["source_sha256"], last_source=None))
            if signature in episode["signatures"]:
                episode["repeated_failures"] += 1
                repeated += 1
            episode["signatures"].append(signature)
            episode["attempts"] += 1
            episode["last_source"] = event["source_sha256"]
        elif _passed(event) and case in active:
            episode = active.pop(case)
            changed = event["source_sha256"] != episode["last_source"]
            begin, end = _at(episode["started_at"]), _at(event.get("at"))
            episodes.append({**episode, "closed_at": event.get("at"), "changed_source_repair": changed,
                             "elapsed_s": end - begin if begin is not None and end is not None and end >= begin else None})
    return dict(closed=len(episodes), open=len(active), repeated_failures=repeated,
                changed_source_repairs=sum(row["changed_source_repair"] for row in episodes),
                episodes=episodes, open_episodes=list(active.values()))


def _milestones(attempts, context, usage, tokens):
    from merlin_experiments.phase1.feedback.joint_performance_report import compare
    initial, elves = context.get("initial_source_sha256"), context.get("initial_case_elf_sha256") or {}
    correct = next((row for row in attempts if isinstance(initial, str) and initial
        and row["source_sha256"] != initial and _passed(row)
        and isinstance(elves.get(row["case"]), str) and elves[row["case"]]
        and isinstance(row.get("elf_sha256"), str) and row["elf_sha256"]
        and row["elf_sha256"] != elves[row["case"]]), None)
    baselines, improvements = {}, {"five_percent": None, "ten_percent": None}
    def model_join(row):
        value = row.get("performance")
        return (isinstance(value, dict) and isinstance(row.get("elf_sha256"), str)
                and bool(row["elf_sha256"]) and value.get("elf_sha256") == row["elf_sha256"])
    # A separately measured, source-bound time-zero baseline may precede author startup.
    for row in context.get("initial_model_observations") or []:
        if (initial and row.get("source_sha256") == initial and _public(row, context) and _passed(row)
            and model_join(row)):
            baselines.setdefault(row["case"], row["performance"])
    for event in attempts:
        performance = event.get("performance")
        if not model_join(event) or not _passed(event):
            continue
        if event["source_sha256"] == initial:
            baselines.setdefault(event["case"], performance)
            continue
        if not initial or event["case"] not in baselines:
            continue
        result = compare(baselines[event["case"]], performance)
        ratio = result.get("candidate_to_baseline_ratio")
        if result.get("status") != "PROVISIONAL" or not _number(ratio):
            continue
        for label, limit in (("five_percent", .95), ("ten_percent", .90)):
            if improvements[label] is None and ratio <= limit:
                improvements[label] = {**event, "ratio": ratio}
    def executed(row):
        tier = (row.get("tiers") or {}).get("L2")
        status = tier.get("status") if isinstance(tier, dict) else tier
        return row.get("numerical_execution_passed") is True or status == "pass"
    scratch = next((row for row in attempts
        if context.get("backend_origin") == "scratch" and isinstance(initial, str) and initial
        and row["source_sha256"] != initial and _passed(row) and executed(row)), None)
    result = {
        "first_correct_from_scratch": {**_milestone(scratch, context, usage, tokens),
            "qualification": "source-bound public numerical and L2 execution observation; no physical qualification"},
        "first_correct_changed_kernel": _milestone(correct, context, usage, tokens),
    }
    for key, event in improvements.items():
        result["first_model_improvement_" + key] = {**_milestone(event, context, usage, tokens),
            "candidate_to_baseline_ratio": event.get("ratio") if event else None,
            "timing_qualified": False, "qualification": "provisional model comparison only"}
    result["initial_source_known"] = bool(initial)
    result["initial_case_elfs_known"] = bool(elves)
    return result


def _coverage(events, context):
    roster = context.get("public_case_ids")
    start, end = _at(context.get("started_at")), _at(context.get("observation_end_at"))
    result = dict(public_cases=len(roster) if isinstance(roster, list) else None, snapshots=[], rejected_snapshots=0,
                  auc_fraction_seconds=None, mean_published_coverage=None, window_s=None,
                  unobserved_prefix_s=None, observed_window_s=None, observed_auc_fraction_seconds=None,
                  mean_observed_published_coverage=None)
    if not roster or len(set(roster)) != len(roster) or start is None or end is None or end <= start:
        return result
    seen = set()
    for event in events:
        if event.get("kind") != "grade_snapshot":
            continue
        source, at = event.get("source_sha256"), _at(event.get("at"))
        rows = [row for row in event.get("rows", []) if row.get("case") in roster]
        if not rows:
            continue  # a hidden-only or unrelated grade cannot replace public coverage
        if (not source or at is None or not start <= at <= end or event.get("coherent") is False
            or len({r["case"] for r in rows}) != len(rows)
            or any(r.get("source_sha256", source) != source for r in rows)):
            result["rejected_snapshots"] += 1
            continue
        key = (source, at, event.get("snapshot_id"))
        if key in seen:
            continue
        seen.add(key)
        passed = sum(row.get("numeric_status") == "pass" for row in rows)
        result["snapshots"].append(dict(at=at, source_sha256=source, passed=passed, fraction=passed / len(roster)))
    points = sorted(result["snapshots"], key=lambda row: row["at"])
    if points:
        auc, at, value = 0.0, start, 0.0
        for point in points:
            auc += value * (point["at"] - at)
            at, value = point["at"], point["fraction"]
        auc += value * (end - at)
        observed = end - points[0]["at"]
        result.update(auc_fraction_seconds=auc, mean_published_coverage=auc / (end - start), window_s=end - start,
                      unobserved_prefix_s=points[0]["at"] - start, observed_window_s=observed,
                      observed_auc_fraction_seconds=auc,
                      mean_observed_published_coverage=auc / observed if observed else None)
    return result


def progress_observations(progress_path, *, run_config):
    """Adapt the existing issued-result collector; never treat public echoes as grades.

    The caller passes the returned context/observations to extract_metrics. Usage
    stays with the native telemetry owner; this adapter never invents token times.
    """
    from evaluate_panels import _reopen, _tree

    document = _read(progress_path)
    if document.get("schema") != "private.kernel_abstractions.agent_progress.v1":
        raise ValueError("select an existing agent progress report")
    selected = _pin(run_config)
    rows = [row for row in document.get("reports", []) if row.get("config") == selected]
    if len(rows) != 1:
        raise ValueError("progress report must select exactly this run configuration")
    row, = rows
    config = _read(run_config)
    start = row.get("started_unix")
    duration = config.get("wall_s")
    end = start + duration if _number(start) and _number(duration) and duration > 0 else None
    baseline = _tree(Path(config["workspace"]) / "baseline")
    context = dict(workspace=config["workspace"], started_at=start, observation_end_at=end,
                   author_s=duration, public_case_ids=config["training_cases"],
                   initial_source_sha256=baseline["sha256"], initial_case_elf_sha256={},
                   backend_origin="seeded" if config.get("seed_continuation") or config.get("source_seed") else "UNKNOWN",
                   seed_history=copy.deepcopy(config.get("seed_continuation")),
                   seed_history_scope="historical provenance only; not current public-case coverage or author-time grades",
                   observation_window_kind="declared author clock; post-author results excluded from milestones",
                   provider_milestone_tokens="UNKNOWN unless supplied by native response chronology")
    events, issues = [], []
    for request in row.get("requests", []):
        rid, files = request.get("request_id"), request.get("files", {})
        try:
            for value in files.values():
                _reopen(value)
            response = _read(files["response"]["path"]) if "response" in files else {}
            at = response.get("graded_at_unix_ns")
            at = at / 1e9 if type(at) is int and at > 0 else None
            began, asked = response.get("grade_started_at_unix_ns"), response.get("requested_at_unix_ns")
            events.append(dict(kind="request_timing", request_id=rid,
                queue_s=(began - asked) / 1e9 if type(began) is int and type(asked) is int and began >= asked else None,
                latency_s=at - asked / 1e9 if at is not None and type(asked) is int and at >= asked / 1e9 else None))
            if request.get("action") == "reference-analysis":
                original = _read(files["request"]["path"])
                events.append(dict(kind="tool_observation", request_id=rid, at=at, tool="reference-analysis",
                    operation=(original.get("reference_analysis") or {}).get("operation"),
                    status=response.get("status"), reason=response.get("reason") or response.get("error")))
            if request.get("status") != "verified_private_result":
                continue
            if not {"request", "response", "done", "result"} <= set(files):
                raise ValueError("verified progress row lacks retained request/result publication")
            source = request["source"]
            _reopen(source, tree=True)
            cases = request.get("cases", [])
            if not cases or any(case.get("case") not in context["public_case_ids"] for case in cases):
                raise ValueError("progress row contains cases outside the public author roster")
            published = []
            for case in cases:
                failure, numeric = case.get("failure") or {}, case.get("numeric") or {}
                event = dict(kind="case_result", attempt_id=rid, source_sha256=source["sha256"], at=at,
                    case=case["case"], status=case.get("status"),
                    numeric_status="pass" if case.get("numeric_pass") is True else
                        "fail" if numeric.get("passed") is False else None,
                    failure_plane=failure.get("plane"), failure_category=failure.get("category"),
                    performance=case.get("performance"), elf_sha256=(case.get("performance") or {}).get("elf_sha256"))
                events.append(event)
                published.append(event)
                if source["sha256"] == baseline["sha256"] and event["numeric_status"] == "pass":
                    context["initial_case_elf_sha256"][case["case"]] = event["elf_sha256"]
            events.append(dict(kind="grade_snapshot", snapshot_id=rid, source_sha256=source["sha256"],
                               at=at, coherent=True, rows=published))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(dict(request_id=rid, error=str(exc)))
    context["initial_model_observations"] = [event for event in events if event.get("kind") == "case_result"
        and event.get("source_sha256") == baseline["sha256"] and _passed(event)
        and _at(event.get("at")) is not None and start is not None and end is not None
        and start <= _at(event["at"]) <= end]
    return dict(context=context, observations=events, issues=issues,
                sources=[_pin(progress_path), selected], qualification_issued=False)


def _checkpoints(context):
    """Project the native snapshot index or explicit retained operator observations."""
    from evaluate_panels import _reopen

    records = context.get("checkpoint_observations", [])
    index = context.get("source_snapshot_index")
    source, issue = None, None
    if index:
        try:
            source = _pin(index)
            document = _read(index)
            if (document.get("schema") != "merlin.study.snapshots.v1"
                    or document.get("started_at_epoch") != context.get("started_at")):
                raise ValueError("snapshot index does not bind this author clock")
            records = document.get("snapshots", [])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            records, issue = [], str(exc)
    duration = context.get("author_s")
    result = []
    for offset in (0, 1800, 3600, 7200, 14400):
        found = [row for row in records if row.get("offset_s") == offset]
        row = dict(offset_s=offset, status="not_applicable" if _number(duration) and offset > duration else "UNKNOWN",
                   source=None, observed_elapsed_s=None)
        if row["status"] == "not_applicable":
            result.append(row)
            continue
        if len(found) == 1:
            original = found[0]
            try:
                if original.get("status") != "frozen":
                    raise ValueError(original.get("reason", "snapshot unavailable"))
                if original.get("digest_basis") == "graded_source_hash":
                    from merlin_experiments.phase1.run_inputs import graded_source_hash
                    if graded_source_hash(Path(original["source"])) != original["compiler_sha256"]:
                        raise ValueError("retained source changed")
                    pinned = dict(path=original["source"], sha256=original["compiler_sha256"],
                                  digest_basis="graded_source_hash")
                else:
                    pinned = original["source"]
                    _reopen(pinned, tree=True)
                observed = original.get("observed_elapsed_s")
                row.update(status="retained", source=pinned, observed_elapsed_s=observed,
                           state_kind=original.get("state_kind"), collected_at=original.get("collected_at"),
                           exact_offset_observation=_number(observed) and observed == offset)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(status="UNKNOWN", reason=str(exc))
        elif found:
            row.update(status="UNKNOWN", reason="ambiguous checkpoint observations")
        result.append(row)
    return dict(source_index=source, checkpoints=result, interpolation=False, issue=issue)


def _mechanisms(events, context):
    counts, accepted, rejected, seen = Counter(), [], [], set()
    for event in events:
        if event.get("kind") != "mechanism" or not _public(event, context):
            continue
        try:
            artifact, proof_pin = event["artifact"], event["proof"]
            if _pin(artifact["path"]) != artifact or _pin(proof_pin["path"]) != proof_pin:
                raise ValueError("receipt bytes changed")
            proof = _read(proof_pin["path"])
            required = {"schema": "merlin.study.emitted_mechanism_proof.v1", "checked": True,
                "source_sha256": event["source_sha256"], "case": event["case"],
                "artifact_sha256": artifact["sha256"], "mechanism": event["mechanism"]}
            if any(proof.get(k) != v for k, v in required.items()) or proof.get("observation") not in {"emitted_reuse", "executed_fallback"}:
                raise ValueError("no exact emitted receipt join")
            key = event["source_sha256"], event["case"], artifact["sha256"], proof_pin["sha256"]
            if key in seen:
                continue
            seen.add(key)
            counts[proof["observation"]] += 1
            accepted.append({key: event[key] for key in ("source_sha256", "case", "mechanism", "artifact", "proof")})
        except (OSError, ValueError, KeyError, TypeError):
            rejected.append(event.get("event_id"))
    return dict(emitted_reuse_receipts=counts["emitted_reuse"] if accepted or rejected else None,
                executed_fallback_receipts=counts["executed_fallback"] if accepted or rejected else None,
                accepted=accepted, rejected_events=rejected, independent_reverification=False)


def _row_event(row, *, attempt, source, at):
    performance = row.get("performance") or {}
    origin = row.get("graded") or {}
    return dict(kind="case_result", attempt_id=attempt, source_sha256=origin.get("graded_source_sha256") or source,
        case=row.get("capsule"), at=origin.get("graded_at") or at, status=row.get("status"),
        numeric_status=row.get("numeric_status") or ("pass" if row.get("pass") is True else None),
        ordering_status=row.get("trace_status"), failure_plane=row.get("failure_plane"),
        failure_category=row.get("failure_category"), failure_tier=row.get("failure_tier"),
        elf_sha256=row.get("elf_sha256") or performance.get("elf_sha256"), performance=performance,
        tiers=row.get("tiers") or {}, numerical_execution_passed=row.get("numerical_execution_passed"),
        execution_digest=row.get("execution_digest"))


def _native_events(run_dir, context):
    """Read grade archives and native redacted channels, never latest-completed mosaic views."""
    from merlin_experiments.phase1.feedback.result_details import hydrate
    from merlin_experiments.phase1.telemetry import agent_requests, author_agents
    run_dir = Path(run_dir)
    workspace = context.get("workspace") or author_agents._environment(run_dir).get("workspace_path")
    events, pins, issues = [], [], []
    for path in sorted((run_dir / "qa_history").glob("verdict*.json")):
        try:
            doc = hydrate(_read(path), workspace=Path(workspace)) if workspace else hydrate(_read(path), document_path=path)
            source, at = doc.get("graded_source_sha256"), doc.get("graded_at")
            attempt = doc.get("graded_request_sha256") or path.stem
            rows = [_row_event(row, attempt=attempt, source=source, at=at) for row in doc.get("per_capsule", [])]
            events.extend(rows)
            events.append(dict(kind="grade_snapshot", source_sha256=source, at=at, snapshot_id=attempt, rows=rows,
                               coherent=doc.get("count_scope") in (None, "immutable_snapshot")))
            pins.append(_pin(path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(dict(path=str(path), error=type(exc).__name__))
    requests, _ = agent_requests._channel_records(agent_requests._channels(run_dir, workspace))
    for request_id, entry in requests.items():
        native = agent_requests._request_record(request_id, entry)
        response = entry.get("response") or {}
        try:
            response = hydrate(response, workspace=Path(workspace) if workspace else run_dir / "agent_evidence_snapshot")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(dict(request_id=request_id, error=type(exc).__name__))
            response = {}
        events.append(dict(kind="request_timing", request_id=request_id,
                           queue_s=native.get("queue_s"), latency_s=native.get("latency_s")))
        for row in response.get("per_capsule") or []:
            events.append(_row_event(row, attempt=request_id, source=response.get("submission_sha256"),
                                     at=native.get("graded_at")))
    for path in sorted((run_dir / "_qa_work").rglob("host_stage_timings.jsonl")):
        for row in _lines(path):
            events.append(dict(kind="stage_timing", event_id=_digest(row), stage=row.get("stage"), wall_s=row.get("wall_s")))
    for path in sorted((run_dir / "_qa_work").glob("**/setup-timing/*.json")):
        try:
            doc = _read(path)
            for row in doc.get("stages", []):
                if row.get("status") not in {"returned", "raised"}:
                    continue
                events.append(dict(kind="stage_timing", event_id=f"{doc.get('timeline_id')}:{row.get('stage_id')}",
                                   stage=row.get("stage"), wall_s=row.get("elapsed_s")))
        except (OSError, ValueError, TypeError):
            issues.append(dict(path=str(path), error="setup_timing_unreadable"))
    return events, pins, issues


def _timings(events, decomposition):
    queues, latency, stages, seen = [], [], defaultdict(list), set()
    for row in events:
        if row.get("kind") == "request_timing":
            key = "request", row.get("request_id")
            if not key[1] or key in seen:
                continue
            seen.add(key)
            queues.append(row.get("queue_s"))
            latency.append(row.get("latency_s"))
        elif row.get("kind") == "stage_timing":
            key = "stage", row.get("event_id")
            if not key[1] or key in seen or not row.get("stage"):
                continue
            seen.add(key)
            stages[row["stage"]].append(row.get("wall_s"))
    waits = [row.get("timing", {}).get("oracle_wait_s") for row in decomposition.get("oracle", {}).get("per_invocation", [])
             if isinstance(row.get("timing"), dict)]
    return dict(request_queue_s=_distribution(queues), request_latency_s=_distribution(latency),
                oracle_wait_s=_distribution(waits), host_stages={key: _distribution(values) for key, values in sorted(stages.items())})


def _tool_usage(events, context):
    start, end = _at(context.get("started_at")), _at(context.get("observation_end_at"))
    rows, seen = [], set()
    for row in events:
        if row.get("kind") != "tool_observation" or not row.get("request_id") or row["request_id"] in seen:
            continue
        at = _at(row.get("at"))
        if at is None or start is None or end is None or not start <= at <= end:
            continue
        seen.add(row["request_id"])
        rows.append(row)
    return dict(observations=rows, count=len(rows) if rows else None,
                operations=dict(Counter(row.get("operation") for row in rows)) if rows else None,
                statuses=dict(Counter(row.get("status") for row in rows)) if rows else None,
                failure_reasons=dict(Counter(row["reason"] for row in rows if row.get("reason"))) if rows else None,
                emitted_mechanism_admission=False, operator_subagents_included=False)


def _source_progress(attempts, context):
    """Separate inherited source checks from observed changed-source/ELF checks."""
    initial, elves = context.get("initial_source_sha256"), context.get("initial_case_elf_sha256") or {}
    rows = []
    for event in attempts:
        if not _passed(event):
            continue
        if not initial:
            change = "initial_source_UNKNOWN"
        elif event["source_sha256"] == initial:
            change = "inherited_source_recheck"
        elif not event.get("elf_sha256") or not elves.get(event["case"]):
            change = "changed_source_emitted_change_UNKNOWN"
        else:
            change = "changed_source_changed_elf" if event["elf_sha256"] != elves[event["case"]] else "changed_source_same_elf"
        rows.append(dict(case=event["case"], attempt_id=event["attempt_id"], source_sha256=event["source_sha256"],
                         elf_sha256=event.get("elf_sha256"), observation=change, at=event.get("at")))
    return dict(passing_observations=rows, counts=dict(Counter(row["observation"] for row in rows)),
                inherited_coverage_reconstructed=False, emitted_mechanism_admission=False)


def _panel(path):
    if path is None:
        return None
    document = _read(path)
    result = dict(source=_pin(path), schema=document.get("schema"), cells=None,
                  physical_timing_qualified=document.get("physical_timing_qualified"), scope="final frozen panel only; excluded from author milestone times")
    if document.get("schema") == "private.kernel_abstractions.evaluation_collection.v1":
        rows = document.get("cells") or []
        names = sorted({row["case"] for row in rows if isinstance(row.get("case"), str)})
        result.update(cells=len(rows), states=dict(Counter(row.get("state") for row in rows)),
                      cases=len(names), case_ids=names,
                      physical_measurements_imported=document.get("physical_measurements_imported"),
                      summaries=copy.deepcopy(document.get("summaries")))
    return result


def extract_metrics(run_dir, *, output=None, panel_results=None, mechanism_events=None, decomposition=None,
                    context=None, observations=None):
    """Pure extraction, apart from explicitly selected operator-side JSON/CSV output.

    `decomposition` accepts report.decompose_run output. Tests/callers may add
    provider_usage_events, already owned/deduplicated native usage rows, or
    codex_homes for lean runs without the standard round/environment layout. Explicit
    observations use kind case_result, grade_snapshot, request_timing or stage_timing;
    optional mechanism receipts are independently pinned, never inferred from imports.
    """
    from merlin_experiments.phase1.telemetry.report import decompose_run
    run_dir = Path(run_dir).resolve(strict=True)
    sources, issues = [], []
    if context is None:
        context = _read(run_dir / CONTEXT) if (run_dir / CONTEXT).is_file() else {}
        if (run_dir / CONTEXT).is_file():
            sources.append(_pin(run_dir / CONTEXT))
    if decomposition is None:
        try:
            decomposition = decompose_run(run_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            decomposition = {}
            issues.append(dict(stage="native_telemetry", error=type(exc).__name__))
    if observations is None:
        events, native_sources, native_issues = _native_events(run_dir, context)
        issues.extend(native_issues)
        sources.extend(native_sources)
        if (run_dir / EVENTS).is_file():
            events.extend(_lines(run_dir / EVENTS))
            sources.append(_pin(run_dir / EVENTS))
    else:
        events = list(observations)
    if mechanism_events is not None:
        events.extend(_lines(mechanism_events))
        sources.append(_pin(mechanism_events))
    try:
        usage_events = _usage_events(run_dir, decomposition)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        usage_events = []
        issues.append(dict(stage="native_provider_usage", error=type(exc).__name__))
    tokens, usage = _token_summary(decomposition, usage_events)
    attempts = _attempts(events, context)
    outcome_present = bool(attempts)
    failures = Counter(value for row in attempts if (value := _failure(row)))
    result = dict(schema=SCHEMA, run_dir=str(run_dir), operator_only=True, definitions=DEFINITIONS,
        context=context, tokens=tokens, milestones=_milestones(attempts, context, usage, tokens),
        invalid_attempts={key: failures[key] if outcome_present else None for key in ("static", "numeric", "ordering", "other")},
        attempts_observed=len(attempts) if outcome_present else None,
        repairs=_repairs(attempts) if outcome_present else None,
        public_coverage=_coverage(events, context), checkpoints=_checkpoints(context),
        changed_source_public_coverage=_coverage([event for event in events
            if context.get("initial_source_sha256") and event.get("source_sha256") != context["initial_source_sha256"]], context),
        source_progress=_source_progress(attempts, context),
        mechanisms=_mechanisms(events, context), tool_usage=_tool_usage(events, context),
        timings=_timings(events, decomposition), frozen_panel=_panel(panel_results), sources=sources,
        extraction_issues=issues, qualification_issued=False)
    if output is not None:
        destination = Path(output).resolve()
        workspace = Path(context["workspace"]).resolve() if context.get("workspace") else None
        if (destination == run_dir or run_dir in destination.parents
            or (workspace is not None and (destination == workspace or workspace in destination.parents))):
            raise ValueError("metrics output must stay outside the author run")
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "metrics.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        with (destination / "metrics.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["metric", "value"])
            def visit(value, prefix=""):
                if isinstance(value, dict):
                    for key, child in value.items():
                        visit(child, f"{prefix}.{key}" if prefix else key)
                elif not isinstance(value, list):
                    writer.writerow([prefix, "" if value is None else value])
            visit({k: v for k, v in result.items() if k not in {"definitions", "context", "sources", "extraction_issues"}})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-results", type=Path)
    parser.add_argument("--mechanism-events", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--decomposition", type=Path)
    args = parser.parse_args(argv)
    record = extract_metrics(args.run_dir, output=args.output, panel_results=args.panel_results,
        mechanism_events=args.mechanism_events, context=_read(args.context) if args.context else None,
        decomposition=_read(args.decomposition) if args.decomposition else None)
    print(json.dumps({"schema": record["schema"], "output": str(args.output), "issues": len(record["extraction_issues"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
