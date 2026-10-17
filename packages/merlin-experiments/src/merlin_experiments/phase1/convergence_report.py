"""Where a Phase 1 run's wall clock goes: bottlenecks, harness design, capsules or tooling.

A read-only accounting over one run directory (live or finished) and, optionally, the operator
directory that supervised it.  Five analyses, each from the run's own retained records:

1. **Author time** -- the provider event stream's arrival stamps.  A command occupies
   ``[item.started, item.completed]``; blocked time is the UNION of those intervals (parallel
   waits occupy one clock), split by what the author was waiting on.
2. **Grade cycle** -- per in-turn grade key: snapshot, first/last capsule result, verdict publish,
   the post-verdict tail (shape coverage and codegen scalability), and the idle gap before the
   next snapshot.  Supersession (``background_grade.status == deferred``) and gated counts say how
   much of each grade was thrown away.
3. **Capsule taxonomy** -- self-check and grade results joined per capsule: pass, genuine failure
   (by category and plane), tooling-incomplete (a tier that could not run here), gated.  Evidence
   is attributed to the candidate digest that produced it, and a join across digests is flagged.
4. **L3 cost** -- measured cert-tier timing split into build, simulator-active and queue wait, the
   simulated throughput, and a projected full-pass cost and wall floor.
5. **Capacity** -- the operator log's ``[resources]`` and ``[heartbeat]`` samples (load average
   against the CPU pool, pending simulator jobs).

Every analysis tolerates missing or partially written files: an absent input yields
``available: false`` with a reason, never a zero that reads as a measurement.  The report reads
status and timing fields only; it never opens answer material and never writes under the run or
operator directory.

    python -m merlin_experiments.phase1.convergence_report --run-dir RUN [--operator-dir DIR] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from merlin_experiments.phase1.telemetry import timeline

SCHEMA = "merlin.phase1_convergence_report.v1"

#: The 2-hour convergence budget this report checks a run against.
BUDGET = {
    "selfcheck_complete_s": 120.0,
    "grade_result_to_result_s": 600.0,
    "author_blocked_share": 0.30,
}

#: Thresholds behind the per-category verdict lines.  They are judgement calls, published with the
#: report so a reader can disagree with a verdict without re-deriving it.
THRESHOLDS = {
    "bottleneck_queue_share": 0.40,
    "bottleneck_pending_share": 0.50,
    "design_fully_gated_share": 0.25,
    "design_tail_s": 300.0,
    "tooling_incomplete_share": 0.25,
}

WINDOW_S = 1800
#: Gap the author spends re-issuing a wait (think time between two consecutive commands that is
#: shorter than this is reported as a second, re-poll-inclusive blocked figure).
REPOLL_GAP_S = 60.0

#: Waiting kinds, in classification priority.  Each command is charged to the first kind it matches.
KINDS = ("grade_wait", "model_check", "simjob", "selfcheck", "sleep_poll")
#: Workspace client scripts the harness provides, by the kind of wait they perform.
_TOOL_SCRIPTS = {
    "await_verdict": "grade_wait",
    "agent_model_check": "model_check",
    "model_check": "model_check",
    "simjob": "simjob",
    "agent_selfcheck": "selfcheck",
}
_POLL_WORDS = frozenset({"sleep", "while", "until", "watch"})
_SHELL_SEPARATORS = " \t\n;|&()<>'\"`={}[],"

#: Labels whose capsule names may be listed.  Any other label is counted, never named.
_NAMEABLE_LABELS = frozenset({"public", "dev"})

_TIMING_FIELDS = ("build_s", "sim_active_s", "oracle_wait_s", "adapter_wall_s")
#: Result classes that settle a capsule's correctness on the bytes that produced them.
_DECISIVE = frozenset({"pass", "genuine_fail"})


# --- small helpers ------------------------------------------------------------------------------


def _iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, UTC).isoformat()


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_json(path: Path):
    """Parsed JSON, or None for an absent, unreadable or partially written file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _round(value, digits: int = 1):
    return None if value is None else round(value, digits)


def _stats(values: list[float]) -> dict:
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return {"n": 0, "median": None, "p90": None, "max": None, "sum": None}
    return {
        "n": len(vals),
        "median": round(statistics.median(vals), 3),
        "p90": round(timeline._percentile(vals, 0.90), 3),
        "max": round(vals[-1], 3),
        "sum": round(sum(vals), 3),
    }


def _clip(intervals: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    return [(max(a, lo), min(b, hi)) for a, b in intervals if min(b, hi) > max(a, lo)]


def _bridged_union(intervals: list[tuple[float, float]], gap: float) -> float:
    """Union seconds after closing every gap of at most ``gap`` seconds between consecutive intervals."""
    total = 0.0
    cur_a = cur_b = None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b + gap:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        elif b > cur_b:
            cur_b = b
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def _turn_key_order(key: str) -> tuple:
    """``r0000_t000007`` -> (0, 7); anything else sorts after, lexicographically."""
    head, _, tail = key.partition("_")
    if head[:1] == "r" and tail[:1] == "t" and head[1:].isdigit() and tail[1:].isdigit():
        return (0, int(head[1:]), int(tail[1:]), key)
    return (1, 0, 0, key)


def _is_turn_key(key: str) -> bool:
    return _turn_key_order(key)[0] == 0


def _compact_stamp(value) -> float | None:
    """``20261017T004740Z`` -> POSIX seconds, or None."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


# --- 1. author time -----------------------------------------------------------------------------


def _words(command: str) -> list[str]:
    out: list[str] = []
    cur: list[str] = []
    for ch in command:
        if ch in _SHELL_SEPARATORS:
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _script_stem(word: str) -> str | None:
    base = word.rsplit("/", 1)[-1]
    for suffix in (".py", ".sh"):
        if base.endswith(suffix) and len(base) > len(suffix):
            return base[: -len(suffix)]
    return None


def _stem_parts(stem: str) -> set[str]:
    parts: set[str] = set()
    for chunk in stem.replace("-", "_").split("_"):
        if chunk:
            parts.add(chunk.lower())
    return parts


def classify_command(command: str) -> tuple[str | None, str | None]:
    """(kind, detail) for one shell command; kind None means the author was working, not waiting.

    Harness-provided clients are recognised by script name.  An author-written script whose name
    says it waits for a grade or verdict (``wait_current_grade.py``) is a grade wait too.
    """
    words = _words(command)
    found: dict[str, str] = {}
    for word in words:
        stem = _script_stem(word)
        if stem is None:
            continue
        if stem in _TOOL_SCRIPTS:
            found.setdefault(_TOOL_SCRIPTS[stem], stem)
            continue
        parts = _stem_parts(stem)
        if parts & {"wait", "await", "poll"} and parts & {"grade", "verdict"}:
            found.setdefault("grade_wait", f"author_script:{word}")
    if any(word in _POLL_WORDS for word in words):
        found.setdefault("sleep_poll", "shell_loop")
    for kind in KINDS:
        if kind in found:
            return kind, found[kind]
    return None, None


def _event_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "rounds").glob("round_*.codex_events.timestamped.jsonl"))


def author_time(run_dir: Path) -> dict:
    files = _event_files(run_dir)
    if not files:
        return {"available": False, "reason": "no rounds/round_*.codex_events.timestamped.jsonl"}
    events = timeline.read_events(files)
    stamps = [timeline._stamp(e.get("arrived_at")) for e in events]
    stamps = [s for s in stamps if s is not None]
    if not stamps:
        return {"available": False, "reason": "event stream carries no arrived_at stamps", "files": len(files)}
    start, end = min(stamps), max(stamps)

    open_items: dict[str, tuple[float, dict]] = {}
    calls: list[dict] = []
    for event in events:
        inner = event.get("event") if isinstance(event.get("event"), dict) else event
        item = inner.get("item")
        etype = inner.get("type")
        if not isinstance(item, dict) or item.get("type") not in ("command_execution", "file_change"):
            continue
        at = timeline._stamp(event.get("arrived_at"))
        if at is None:
            continue
        item_id = str(item.get("id") or "")
        if etype == "item.started":
            open_items[item_id] = (at, item)
        elif etype == "item.completed":
            begun = open_items.pop(item_id, None)
            calls.append({"start": begun[0] if begun else at, "end": at, "item": item, "terminated": True})
    for begun, item in open_items.values():
        # A command still running when the stream ends (a live run, or a killed session) occupies
        # the clock up to the last event; it is counted and marked rather than dropped.
        calls.append({"start": begun, "end": end, "item": item, "terminated": False})

    commands = [c for c in calls if c["item"].get("type") == "command_execution"]
    edits = [c for c in calls if c["item"].get("type") == "file_change"]
    edited_paths = sum(len(c["item"].get("changes") or []) for c in edits if c["terminated"])

    by_kind: dict[str, list[tuple[float, float]]] = defaultdict(list)
    kind_detail: dict[str, Counter] = defaultdict(Counter)
    all_cmd: list[tuple[float, float]] = []
    for call in commands:
        span = (call["start"], call["end"])
        all_cmd.append(span)
        kind, detail = classify_command(str(call["item"].get("command") or ""))
        if kind:
            by_kind[kind].append(span)
            kind_detail[kind][detail] += 1
    waiting = [span for spans in by_kind.values() for span in spans]
    wall = end - start

    windows = []
    lo = math.floor(start / WINDOW_S) * WINDOW_S
    while lo < end:
        a, b = max(lo, start), min(lo + WINDOW_S, end)
        span_s = b - a
        if span_s > 0:
            blocked = timeline._union_seconds(_clip(all_cmd, a, b))
            windows.append(
                {
                    "start": _iso(a),
                    "end": _iso(b),
                    "span_s": round(span_s, 1),
                    "blocked_s": round(blocked, 1),
                    "blocked_share": round(blocked / span_s, 3),
                    "by_kind_s": {k: round(timeline._union_seconds(_clip(by_kind[k], a, b)), 1) for k in KINDS},
                }
            )
        lo += WINDOW_S

    blocked_any = timeline._union_seconds(all_cmd)
    return {
        "available": True,
        "files": [str(p.relative_to(run_dir)) for p in files],
        "started_at": _iso(start),
        "last_event_at": _iso(end),
        "wall_s": round(wall, 1),
        "n_commands": len(commands),
        "n_edits": len(edits),
        "n_edited_paths": edited_paths,
        "n_unterminated": sum(1 for c in calls if not c["terminated"]),
        "blocked_s": round(blocked_any, 1),
        "blocked_share": round(blocked_any / wall, 3) if wall > 0 else None,
        "blocked_waiting_s": round(timeline._union_seconds(waiting), 1),
        "blocked_incl_repoll_gaps_s": round(_bridged_union(all_cmd, REPOLL_GAP_S), 1),
        "repoll_gap_s": REPOLL_GAP_S,
        "by_kind": {
            k: {
                "union_s": round(timeline._union_seconds(by_kind[k]), 1),
                "sum_s": round(sum(b - a for a, b in by_kind[k]), 1),
                "n": len(by_kind[k]),
                "detail": dict(kind_detail[k].most_common(8)),
            }
            for k in KINDS
        },
        "windows": windows,
        "note": (
            "blocked_s is the union of command intervals (parallel waits occupy one clock); "
            "blocked_incl_repoll_gaps_s also closes think gaps shorter than repoll_gap_s between commands"
        ),
    }


# --- 2. grade cycle -----------------------------------------------------------------------------


def _capsule_dirs(base: Path) -> list[Path]:
    """``<base>/runs/<suite>/<capsule>/`` directories of one grade or self-check request."""
    return sorted(p for p in (base / "runs").glob("*/*") if p.is_dir())


def _result_files(base: Path) -> list[Path]:
    return sorted((base / "runs").glob("*/*/capsule_result.json"))


def _commit_times(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    path = run_dir / "oot_commits.jsonl"
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("key"), str):
            oot = rec.get("oot") if isinstance(rec.get("oot"), dict) else {}
            out[rec["key"]] = {
                "committed_at": _compact_stamp(oot.get("committed_at")),
                "package_digest": oot.get("package_digest"),
            }
    return out


def _grade_keys(run_dir: Path) -> list[str]:
    keys: set[str] = set()
    work = run_dir / "_qa_work"
    for prefix in ("cand_", "runs_"):
        for path in work.glob(prefix + "*"):
            keys.add(path.name[len(prefix) :])
    for path in (run_dir / "qa_history").glob("verdict_inturn_*.json"):
        keys.add(path.name[len("verdict_inturn_") : -len(".json")])
    return sorted((k for k in keys if _is_turn_key(k)), key=_turn_key_order)


def grade_cycle(run_dir: Path) -> dict:
    keys = _grade_keys(run_dir)
    if not keys:
        return {"available": False, "reason": "no in-turn grade keys under _qa_work/ or qa_history/"}
    commits = _commit_times(run_dir)
    work, history = run_dir / "_qa_work", run_dir / "qa_history"
    grades = []
    for key in keys:
        cand_t = _mtime(work / f"cand_{key}")
        commit = commits.get(key, {})
        snapshot_t = cand_t if cand_t is not None else commit.get("committed_at")
        runs_dir = work / f"runs_{key}"
        capsule_dirs = _capsule_dirs(runs_dir)
        result_times = sorted(t for t in (_mtime(p) for p in _result_files(runs_dir)) if t is not None)
        verdict_path = history / f"verdict_inturn_{key}.json"
        verdict_t = _mtime(verdict_path)
        verdict = _read_json(verdict_path) if verdict_t is not None else None
        shape_t = _mtime(history / f"shape_coverage_round_{key}.json")
        scal_t = _mtime(history / f"codegen_scalability_round_{key}.json")
        tail_end = max((t for t in (shape_t, scal_t) if t is not None), default=None)
        rec = {
            "key": key,
            "snapshot_at": _iso(snapshot_t),
            "snapshot_source": "cand_dir_mtime" if cand_t is not None else ("oot_commit" if snapshot_t else None),
            "package_digest": commit.get("package_digest"),
            "n_capsule_dirs": len(capsule_dirs),
            "n_results": len(result_times),
            "first_result_at": _iso(result_times[0]) if result_times else None,
            "last_result_at": _iso(result_times[-1]) if result_times else None,
            "verdict_at": _iso(verdict_t),
            "shape_coverage_at": _iso(shape_t),
            "scalability_at": _iso(scal_t),
            "snapshot_to_first_result_s": (
                _round(result_times[0] - snapshot_t) if result_times and snapshot_t is not None else None
            ),
            "capsule_span_s": _round(result_times[-1] - result_times[0]) if result_times else None,
            "last_result_to_verdict_s": (
                _round(verdict_t - result_times[-1]) if result_times and verdict_t is not None else None
            ),
            "snapshot_to_verdict_s": (
                _round(verdict_t - snapshot_t) if verdict_t is not None and snapshot_t is not None else None
            ),
            "tail_after_verdict_s": _round(tail_end - verdict_t) if tail_end and verdict_t else None,
            # "unpublished" is a grade still running on a live run, or one abandoned when a run stopped.
            "state": "published" if verdict_t is not None else ("unpublished" if capsule_dirs else "pending"),
            "_snapshot_t": snapshot_t,
            "_verdict_t": verdict_t,
            "_tail_end_t": tail_end,
        }
        if isinstance(verdict, dict):
            per = [p for p in verdict.get("per_capsule") or [] if isinstance(p, dict)]
            statuses = Counter(str(p.get("status")) for p in per)
            background = verdict.get("background_grade") if isinstance(verdict.get("background_grade"), dict) else {}
            n_caps = verdict.get("n_capsules") if isinstance(verdict.get("n_capsules"), int) else len(per)
            gated = statuses.get("gated", 0)
            rec.update(
                {
                    "n_capsules": n_caps,
                    "n_passed": verdict.get("n_passed"),
                    "statuses": dict(sorted(statuses.items())),
                    "gated": gated,
                    "fully_gated": bool(n_caps) and gated == n_caps,
                    "background_status": background.get("status"),
                    "background_reason": background.get("reason"),
                    "superseded": background.get("status") == "deferred",
                    "graded_source_sha256": verdict.get("graded_source_sha256"),
                }
            )
        elif verdict_t is not None:
            rec["unreadable_verdict"] = True
        grades.append(rec)

    for prev, cur in zip(grades, grades[1:]):
        if prev["_verdict_t"] is not None and cur["_verdict_t"] is not None:
            cur["result_to_result_s"] = _round(cur["_verdict_t"] - prev["_verdict_t"])
        prev_end = prev["_tail_end_t"] or prev["_verdict_t"]
        if prev_end is not None and cur["_snapshot_t"] is not None:
            prev["gap_to_next_snapshot_s"] = _round(cur["_snapshot_t"] - prev_end)

    published = [g for g in grades if g["state"] == "published"]

    def col(name):
        return [g.get(name) for g in grades if isinstance(g.get(name), (int, float))]

    summary = {
        "n_keys": len(grades),
        "n_published": len(published),
        "n_unpublished": sum(1 for g in grades if g["state"] == "unpublished"),
        "n_fully_gated": sum(1 for g in published if g.get("fully_gated")),
        "n_superseded": sum(1 for g in published if g.get("superseded")),
        "result_to_result_s": _stats(col("result_to_result_s")),
        "snapshot_to_verdict_s": _stats(col("snapshot_to_verdict_s")),
        "snapshot_to_first_result_s": _stats(col("snapshot_to_first_result_s")),
        "capsule_span_s": _stats(col("capsule_span_s")),
        "tail_after_verdict_s": _stats(col("tail_after_verdict_s")),
        "gap_to_next_snapshot_s": _stats(col("gap_to_next_snapshot_s")),
    }
    for g in grades:
        for private in ("_snapshot_t", "_verdict_t", "_tail_end_t"):
            g.pop(private, None)
    return {"available": True, "summary": summary, "grades": grades}


# --- 3. capsule taxonomy ------------------------------------------------------------------------


def classify_result(status, failure) -> tuple[str, str | None]:
    """(class, detail) of one capsule result from its status and failure block alone."""
    failure = failure if isinstance(failure, dict) else {}
    category = failure.get("category")
    plane = failure.get("plane")
    detail = str(failure.get("detail") or "").lower()
    if status == "pass":
        return "pass", None
    if status == "gated" or category == "BACKGROUND_DEFERRED" or plane == "gate":
        return "gated", str(failure.get("detail") or "") or None
    if category == "NOT_RUN_IS_NOT_PASS":
        if "no adapter" in detail or "reachable tiers" in detail:
            sub = "no_adapter_for_tier"
        elif "not selected" in detail:
            sub = "tier_not_selected"
        elif "absent" in detail or "missing" in detail:
            sub = "missing_input"
        else:
            sub = "other"
        return "tooling_incomplete", f"{sub}:{failure.get('tier') or plane}"
    if status == "cert_not_measured" or category == "testbench_timeout":
        return "unmeasured_timeout", f"{category}/{plane}"
    if status == "fail":
        return "genuine_fail", f"{category}/{plane}"
    return f"other:{status}", f"{category}/{plane}" if category or plane else None


def _tier_combo(tiers) -> str:
    if not isinstance(tiers, dict):
        return ""
    parts = []
    for name in sorted(tiers):
        value = tiers[name]
        state = value.get("status") if isinstance(value, dict) else value
        parts.append(f"{name}={state}")
    return " ".join(parts)


def _evidence(run_dir: Path) -> tuple[list[dict], int]:
    work = run_dir / "_qa_work"
    sources: list[tuple[str, str, Path]] = []
    for runs in sorted(work.glob("runs_*")):
        key = runs.name[len("runs_") :]
        if _is_turn_key(key):
            sources.append(("grade", key, runs))
    for check in sorted((work / "selfchecks").glob("*")):
        if check.is_dir():
            sources.append(("selfcheck", check.name, check))
    records, unreadable = [], 0
    for kind, ident, base in sources:
        for path in _result_files(base):
            data = _read_json(path)
            at = _mtime(path)
            if not isinstance(data, dict) or at is None:
                unreadable += 1
                continue
            compiler = data.get("compiler_source") if isinstance(data.get("compiler_source"), dict) else {}
            cls, detail = classify_result(data.get("status"), data.get("failure"))
            tiers = data.get("tiers") if isinstance(data.get("tiers"), dict) else {}
            records.append(
                {
                    "capsule": str(data.get("capsule") or path.parent.name),
                    "label": data.get("label"),
                    "source": kind,
                    "source_id": ident,
                    "at": at,
                    "digest": compiler.get("candidate_sha256"),
                    "status": data.get("status"),
                    "class": cls,
                    "detail": detail,
                    "tier_combo": _tier_combo(tiers),
                    "tiers": tiers,
                    "path": path,
                }
            )
    return records, unreadable


def _latest_verdict(run_dir: Path) -> tuple[str | None, dict | None]:
    paths = sorted(
        (run_dir / "qa_history").glob("verdict_inturn_*.json"),
        key=lambda p: _turn_key_order(p.name[len("verdict_inturn_") : -len(".json")]),
    )
    for path in reversed(paths):
        data = _read_json(path)
        if isinstance(data, dict):
            return path.name[len("verdict_inturn_") : -len(".json")], data
    return None, None


def capsule_taxonomy(run_dir: Path, records: list[dict] | None = None, unreadable: int = 0) -> dict:
    if records is None:
        records, unreadable = _evidence(run_dir)
    latest_key, verdict = _latest_verdict(run_dir)
    suite: dict[str, dict] = {}
    if isinstance(verdict, dict):
        for p in verdict.get("per_capsule") or []:
            if isinstance(p, dict) and p.get("capsule"):
                suite[str(p["capsule"])] = p
    if not records and not suite:
        return {"available": False, "reason": "no capsule results and no in-turn verdict"}
    graded_digest = verdict.get("graded_source_sha256") if isinstance(verdict, dict) else None
    newest_digest = None
    if records:
        newest_digest = max((r for r in records if r["digest"]), key=lambda r: r["at"], default={}).get("digest")

    labels: dict[str, str | None] = {}
    by_capsule: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_capsule[rec["capsule"]].append(rec)
        labels.setdefault(rec["capsule"], rec["label"])
    for name, entry in suite.items():
        labels.setdefault(name, entry.get("label"))
    names = sorted(set(by_capsule) | set(suite))

    def nameable(name: str) -> bool:
        return labels.get(name) in _NAMEABLE_LABELS

    per_source: dict[str, dict] = {}
    for kind in ("selfcheck", "grade"):
        subset = [r for r in records if r["source"] == kind]
        per_source[kind] = {
            "n_results": len(subset),
            "n_capsules": len({r["capsule"] for r in subset}),
            "classes": dict(Counter(r["class"] for r in subset).most_common()),
            "genuine_fail": dict(Counter(r["detail"] for r in subset if r["class"] == "genuine_fail").most_common()),
            "tooling_incomplete": dict(
                Counter(r["detail"] for r in subset if r["class"] == "tooling_incomplete").most_common()
            ),
            "tier_combos": dict(Counter(r["tier_combo"] for r in subset).most_common(12)),
        }

    current: dict[str, Counter] = defaultdict(Counter)
    current_class: Counter = Counter()
    joins = Counter()
    fails: list[dict] = []
    gated_without_evidence: list[str] = []
    newer_partial = 0
    for name in names:
        evidence = sorted(by_capsule.get(name, []), key=lambda r: r["at"])
        decisive = [r for r in evidence if r["class"] in _DECISIVE]
        partial = [r for r in evidence if r["class"] not in _DECISIVE and r["class"] != "gated"]
        grade_status = suite.get(name, {}).get("status")
        best = decisive[-1] if decisive else (partial[-1] if partial else None)
        if best is not None:
            cls = best["class"]
            if decisive and partial and partial[-1]["at"] > best["at"]:
                newer_partial += 1
            if best["digest"] and graded_digest and best["digest"] == graded_digest:
                joins["same_as_latest_graded_source"] += 1
            elif best["digest"]:
                joins["different_source_version"] += 1
            else:
                joins["no_digest"] += 1
            current[cls][best["detail"]] += 1
            if cls == "genuine_fail" and nameable(name):
                fails.append(
                    {
                        "capsule": name,
                        "detail": best["detail"],
                        "source": f"{best['source']}:{best['source_id']}",
                        "at": _iso(best["at"]),
                        "same_source_as_latest_grade": bool(best["digest"] and best["digest"] == graded_digest),
                    }
                )
        else:
            cls = "gated" if grade_status == "gated" or evidence else "no_evidence"
            if cls == "gated":
                gated_without_evidence.append(name)
        current_class[cls] += 1

    selfchecked = {r["capsule"] for r in records if r["source"] == "selfcheck"}
    never = [n for n in names if n not in selfchecked]
    latest_status = Counter(str(p.get("status")) for p in suite.values())
    return {
        "available": True,
        "n_capsules_known": len(names),
        "n_unreadable_results": unreadable,
        "latest_grade": {
            "key": latest_key,
            "graded_source_sha256": graded_digest,
            "statuses": dict(sorted(latest_status.items())),
        },
        "newest_evidence_digest": newest_digest,
        "current": {
            "basis": (
                "per capsule, the newest decisive (pass / genuine_fail) result from any grade or self-check; "
                "else the newest tooling-incomplete one; else gated / no evidence"
            ),
            "classes": dict(current_class.most_common()),
            "genuine_fail": dict(current["genuine_fail"].most_common()),
            "tooling_incomplete": dict(current["tooling_incomplete"].most_common()),
            "evidence_source_version": dict(joins),
            "decisive_with_newer_partial_result": newer_partial,
            "genuine_fail_capsules": fails,
            "gated_without_other_evidence": {
                "n": len(gated_without_evidence),
                "capsules": [n for n in gated_without_evidence if nameable(n)],
            },
        },
        "by_source": per_source,
        "never_selfchecked": {
            "n": len(never),
            "capsules": [n for n in never if nameable(n)],
            "n_withheld": sum(1 for n in never if not nameable(n)),
        },
        "note": (
            "evidence from a different source version is a result for older candidate bytes; "
            "it says where a capsule stood, not where the current candidate stands"
        ),
    }


# --- 4. L3 cost ---------------------------------------------------------------------------------


def l3_cost(records: list[dict], suite_size: int | None, cpus: list[int]) -> dict:
    rows, partial = [], 0
    l3_required: dict[str, bool] = {}
    for rec in records:
        l3 = rec["tiers"].get("L3")
        if isinstance(l3, dict):
            l3_required[rec["capsule"]] = l3_required.get(rec["capsule"], False) or bool(l3.get("mandatory"))
        if not isinstance(l3, dict) or not l3.get("measured_now") or not isinstance(l3.get("timing"), dict):
            continue
        timing = l3["timing"]
        if timing.get("partial") or not all(isinstance(timing.get(f), (int, float)) for f in _TIMING_FIELDS):
            partial += 1
            continue
        engine = l3.get("engine")
        if engine is None and isinstance(l3.get("sim_provenance"), dict):
            engine = l3["sim_provenance"].get("engine")
        cycles = l3.get("cycles") if isinstance(l3.get("cycles"), (int, float)) else None
        rows.append(
            {
                "capsule": rec["capsule"],
                "label": rec["label"],
                "source": f"{rec['source']}:{rec['source_id']}",
                "engine": engine,
                "status": l3.get("status"),
                "cycles": cycles,
                **{f: timing[f] for f in _TIMING_FIELDS},
            }
        )
    if not rows:
        return {"available": False, "reason": "no measured L3 timing in any capsule result", "n_partial": partial}

    sums = {f: sum(r[f] for r in rows) for f in _TIMING_FIELDS}
    rates = [r["cycles"] / r["sim_active_s"] for r in rows if r["cycles"] and r["sim_active_s"] > 0]
    small = [r["sim_active_s"] for r in rows if r["cycles"] is not None and r["cycles"] <= 1000]
    fixed = statistics.median(small) if small else None
    # Throughput once the fixed startup is paid, from runs long enough for it to be measurable.
    marginal = [
        r["cycles"] / (r["sim_active_s"] - fixed)
        for r in rows
        if fixed is not None and r["cycles"] and r["cycles"] > 10000 and r["sim_active_s"] > 2 * fixed
    ]

    per_capsule: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        per_capsule[r["capsule"]].append(r["build_s"] + r["sim_active_s"])
    cost = {name: statistics.median(vals) for name, vals in per_capsule.items()}
    median_cost = statistics.median(cost.values())
    projection = None
    if suite_size:
        known = sum(cost.values())
        unknown = max(0, suite_size - len(cost))
        cpu_s = known + unknown * median_cost
        floor_s = max(cost.values())
        projection = {
            "suite_capsules": suite_size,
            "capsules_with_measured_cost": len(cost),
            "capsules_estimated_at_median": unknown,
            "median_capsule_cost_s": round(median_cost, 1),
            "cpu_minutes": round(cpu_s / 60, 1),
            "longest_capsule_floor_s": round(floor_s, 1),
            "wall_minutes_at": {str(n): round(max(cpu_s / n, floor_s) / 60, 1) for n in sorted(set(cpus)) if n > 0},
            "basis": (
                "per-capsule cost = median measured build_s + sim_active_s (queue wait excluded); "
                "unmeasured capsules priced at the median; wall = max(cpu / N, longest capsule)"
            ),
        }

    def show(r):
        name = r["capsule"] if r["label"] in _NAMEABLE_LABELS else "(withheld)"
        return {
            "capsule": name,
            "source": r["source"],
            "cycles": r["cycles"],
            **{f: round(r[f], 1) for f in _TIMING_FIELDS},
        }

    return {
        "available": True,
        "n_measured": len(rows),
        "n_partial": partial,
        "engines": dict(Counter(str(r["engine"]) for r in rows)),
        "timing_s": {f: _stats([r[f] for r in rows]) for f in _TIMING_FIELDS},
        "queue_vs_active": {
            "oracle_wait_share_of_adapter_wall": round(sums["oracle_wait_s"] / sums["adapter_wall_s"], 3)
            if sums["adapter_wall_s"]
            else None,
            "sim_active_share_of_adapter_wall": round(sums["sim_active_s"] / sums["adapter_wall_s"], 3)
            if sums["adapter_wall_s"]
            else None,
            "build_share_of_adapter_wall": round(sums["build_s"] / sums["adapter_wall_s"], 3)
            if sums["adapter_wall_s"]
            else None,
        },
        "cycles_per_s": {
            "basis": "recorded L3 cycles (the kernel window, not every simulated cycle) per sim_active_s second",
            **_stats(rates),
        },
        "small_capsule_sim_active_s": {
            "basis": (
                "runs recording at most 1000 cycles: their sim_active_s is the per-run fixed cost of the window "
                "between build and simulator return (identity checks, slot wait, simulator start and load)"
            ),
            **_stats(small),
        },
        "marginal_cycles_per_s": {
            "basis": "runs over 10000 cycles, after subtracting the small-capsule median as the fixed per-run cost",
            "median": _round(statistics.median(marginal), 1) if marginal else None,
            "n": len(marginal),
        },
        "slowest": [show(r) for r in sorted(rows, key=lambda r: -r["adapter_wall_s"])[:10]],
        "slowest_sim_active": [show(r) for r in sorted(rows, key=lambda r: -r["sim_active_s"])[:5]],
        "capsules_requiring_l3": sum(1 for v in l3_required.values() if v),
        "projection": projection,
    }


# --- 5. capacity --------------------------------------------------------------------------------


def _fields(text: str) -> dict:
    """``key=value`` fields of one operator log line; a value may contain spaces inside brackets."""
    out: dict[str, str] = {}
    key = None
    depth = 0
    for token in text.split(" "):
        if depth == 0 and "=" in token:
            head, _, value = token.partition("=")
            if head and (head.replace("_", "").isalnum()):
                key = head
                out[key] = value
                depth = value.count("[") + value.count("{") - value.count("]") - value.count("}")
                continue
        if key is not None:
            out[key] += " " + token
            depth += token.count("[") + token.count("{") - token.count("]") - token.count("}")
    return out


def _json_field(value):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _pool_size(operator_dir: Path, records: list[dict]) -> tuple[int | None, str | None]:
    health = _read_json(operator_dir / "health.json")
    leases = (health or {}).get("resources", {}).get("cpu_leases") if isinstance(health, dict) else None
    if isinstance(leases, list) and leases:
        return len(leases), "health.json resources.cpu_leases"
    preflight = _read_json(operator_dir / "preflight.json")
    slots = preflight.get("slots") if isinstance(preflight, dict) else None
    if isinstance(slots, list) and slots:
        return len(slots), "preflight.json slots"
    workers = Counter()
    for rec in records:
        for tier in rec["tiers"].values():
            if isinstance(tier, dict) and isinstance(tier.get("concurrency"), dict):
                w = tier["concurrency"].get("workers")
                if isinstance(w, int):
                    workers[w] += 1
    if workers:
        return workers.most_common(1)[0][0], "capsule result concurrency.workers"
    return None, None


def capacity(operator_dir: Path | None, records: list[dict]) -> dict:
    if operator_dir is None:
        return {"available": False, "reason": "no --operator-dir given"}
    log = operator_dir / "live.log"
    try:
        lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {"available": False, "reason": f"unreadable {log.name}"}
    pool, pool_source = _pool_size(operator_dir, records)
    loads: list[tuple[float, float]] = []
    pending: list[tuple[float, int]] = []
    native: list[tuple[float, int]] = []
    for line in lines:
        stamp, _, rest = line.partition(" ")
        t = timeline._stamp(stamp)
        if t is None:
            continue
        if rest.startswith("[resources] "):
            load = _json_field(_fields(rest[len("[resources] ") :]).get("load"))
            if isinstance(load, list) and load and isinstance(load[0], (int, float)):
                loads.append((t, float(load[0])))
        elif rest.startswith("[heartbeat] "):
            fields = _fields(rest[len("[heartbeat] ") :])
            jobs = _json_field(fields.get("sim_jobs"))
            if isinstance(jobs, dict):
                pending.append((t, int(jobs.get("pending") or 0)))
            active = _json_field(fields.get("native_active"))
            if isinstance(active, dict):
                native.append((t, sum(v for v in active.values() if isinstance(v, int))))
    if not loads and not pending:
        return {"available": False, "reason": "no [resources] or [heartbeat] samples", "pool_size": pool}

    windows = []
    stamps = [t for t, _ in loads] + [t for t, _ in pending]
    lo = math.floor(min(stamps) / WINDOW_S) * WINDOW_S
    hi = max(stamps)
    while lo <= hi:
        wl = [v for t, v in loads if lo <= t < lo + WINDOW_S]
        wp = [v for t, v in pending if lo <= t < lo + WINDOW_S]
        if wl or wp:
            windows.append(
                {
                    "start": _iso(lo),
                    "load1_median": _round(statistics.median(wl), 2) if wl else None,
                    "load1_max": _round(max(wl), 2) if wl else None,
                    "sim_jobs_pending_median": statistics.median(wp) if wp else None,
                    "sim_jobs_pending_max": max(wp) if wp else None,
                }
            )
        lo += WINDOW_S
    load_vals = [v for _, v in loads]
    pend_vals = [v for _, v in pending]
    # Pending work while the machine is NOT saturated means dispatch, not CPU, is what holds it:
    # each heartbeat is paired with the resource sample logged in the same observation (within 30 s).
    paired = []
    if pool and loads:
        load_by_time = sorted(loads)
        for t, n in pending:
            nearest = min(load_by_time, key=lambda s: abs(s[0] - t))
            if abs(nearest[0] - t) <= 30:
                paired.append((n, nearest[1]))
    return {
        "available": True,
        "pool_size": pool,
        "pool_size_source": pool_source,
        "n_resource_samples": len(loads),
        "n_heartbeats": len(pending),
        "load1": _stats(load_vals),
        "load1_over_pool_share": (
            round(sum(1 for v in load_vals if v > pool) / len(load_vals), 3) if pool and load_vals else None
        ),
        "sim_jobs_pending": _stats(pend_vals),
        "sim_jobs_pending_nonzero_share": (
            round(sum(1 for v in pend_vals if v > 0) / len(pend_vals), 3) if pend_vals else None
        ),
        "pending_while_load_below_pool_share": (
            round(sum(1 for n, load in paired if n > 0 and load < pool) / len(paired), 3) if paired else None
        ),
        "native_active": _stats([v for _, v in native]),
        "windows": windows,
    }


# --- self-check latency -------------------------------------------------------------------------


def selfcheck_latency(run_dir: Path) -> dict:
    root = run_dir / "_qa_work" / "selfchecks"
    if not root.is_dir():
        return {"available": False, "reason": "no _qa_work/selfchecks"}
    done, in_flight = [], 0
    for check in sorted(root.iterdir()):
        if not check.is_dir():
            continue
        starts = [t for t in (_mtime(check), _mtime(check / "runs")) if t is not None]
        dirs = _capsule_dirs(check)
        results = [t for t in (_mtime(p) for p in _result_files(check)) if t is not None]
        if not starts or not dirs:
            continue
        if len(results) < len(dirs):
            in_flight += 1
            continue
        done.append({"id": check.name, "capsules": len(dirs), "latency_s": max(results) - min(starts)})
    lat = [d["latency_s"] for d in done]
    return {
        "available": bool(done),
        "basis": "self-check directory creation to its last capsule_result.json (all capsules of the request)",
        "n_complete": len(done),
        "n_in_flight": in_flight,
        "latency_s": _stats(lat),
        "within_budget_share": round(sum(1 for v in lat if v <= BUDGET["selfcheck_complete_s"]) / len(lat), 3)
        if lat
        else None,
        "capsules_per_request": _stats([d["capsules"] for d in done]),
    }


# --- verdicts -----------------------------------------------------------------------------------


def _budget(author: dict, grades: dict, selfchecks: dict) -> dict:
    out = {}
    lat = selfchecks.get("latency_s", {}) if selfchecks.get("available") else {}
    out["selfcheck_complete_le_2min"] = {
        "limit_s": BUDGET["selfcheck_complete_s"],
        "median_s": lat.get("median"),
        "p90_s": lat.get("p90"),
        "met": (lat.get("p90") <= BUDGET["selfcheck_complete_s"]) if lat.get("p90") is not None else None,
        "rule": "p90 of complete self-check requests within the limit",
    }
    r2r = grades.get("summary", {}).get("result_to_result_s", {}) if grades.get("available") else {}
    out["grade_result_to_result_le_10min"] = {
        "limit_s": BUDGET["grade_result_to_result_s"],
        "median_s": r2r.get("median"),
        "max_s": r2r.get("max"),
        "met": (r2r.get("median") <= BUDGET["grade_result_to_result_s"]) if r2r.get("median") is not None else None,
        "rule": "median interval between consecutive published in-turn verdicts within the limit",
    }
    share = author.get("blocked_share") if author.get("available") else None
    out["author_blocked_share_lt_30pct"] = {
        "limit": BUDGET["author_blocked_share"],
        "share": share,
        "met": (share < BUDGET["author_blocked_share"]) if share is not None else None,
        "rule": "union of command intervals over the author's wall clock",
    }
    return out


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _mins(x) -> str:
    return "n/a" if x is None else f"{x / 60:.1f} min"


def _verdicts(author: dict, grades: dict, taxonomy: dict, cost: dict, cap: dict, selfchecks: dict) -> dict:
    out = {}

    # Bottleneck: compute capacity and simulator cost.
    queue = (
        (cost.get("queue_vs_active") or {}).get("oracle_wait_share_of_adapter_wall") if cost.get("available") else None
    )
    pend = cap.get("sim_jobs_pending_nonzero_share") if cap.get("available") else None
    small = (cost.get("small_capsule_sim_active_s") or {}).get("median") if cost.get("available") else None
    flag = (queue is not None and queue >= THRESHOLDS["bottleneck_queue_share"]) or (
        pend is not None and pend >= THRESHOLDS["bottleneck_pending_share"]
    )
    load = (cap.get("load1") or {}).get("median") if cap.get("available") else None
    below = cap.get("pending_while_load_below_pool_share") if cap.get("available") else None
    out["bottleneck"] = {
        "flag": flag if (queue is not None or pend is not None) else None,
        "line": (
            f"L3 queue wait {_pct(queue)} of adapter wall; sim jobs pending in {_pct(pend)} of heartbeats "
            f"({_pct(below)} with load1 below the pool); load1 median {load} vs pool {cap.get('pool_size')}; "
            f"fixed per-run L3 cost (<=1000-cycle sim_active median) {small} s"
        ),
    }

    # Design: the grade cycle the harness imposes.
    gs = grades.get("summary", {}) if grades.get("available") else {}
    published = gs.get("n_published") or 0
    fully = gs.get("n_fully_gated") or 0
    tail = (gs.get("tail_after_verdict_s") or {}).get("median")
    r2r = (gs.get("result_to_result_s") or {}).get("median")
    design_flag = None
    if published:
        design_flag = (
            fully / published >= THRESHOLDS["design_fully_gated_share"]
            or (tail is not None and tail >= THRESHOLDS["design_tail_s"])
            or (r2r is not None and r2r > BUDGET["grade_result_to_result_s"])
        )
    out["design"] = {
        "flag": design_flag,
        "line": (
            f"{fully} of {published} grades fully gated ({gs.get('n_superseded')} superseded); "
            f"result-to-result median {_mins(r2r)}; post-verdict tail median {_mins(tail)}; "
            f"gap to next snapshot median {_mins((gs.get('gap_to_next_snapshot_s') or {}).get('median'))}"
        ),
    }

    # Tooling: results that could not be produced here, and how long the author waited on tools.
    sc = (taxonomy.get("by_source") or {}).get("selfcheck", {}) if taxonomy.get("available") else {}
    n_sc = sc.get("n_results") or 0
    tooling = (sc.get("classes") or {}).get("tooling_incomplete", 0)
    timeouts = sum(
        (s.get("classes") or {}).get("unmeasured_timeout", 0) for s in (taxonomy.get("by_source") or {}).values()
    )
    lat = (selfchecks.get("latency_s") or {}).get("median") if selfchecks.get("available") else None
    tooling_flag = None
    if n_sc or lat is not None:
        tooling_flag = (n_sc and tooling / n_sc >= THRESHOLDS["tooling_incomplete_share"]) or (
            lat is not None and lat > BUDGET["selfcheck_complete_s"]
        )
    by_kind = author.get("by_kind", {}) if author.get("available") else {}
    waits = ", ".join(f"{k} {_mins(v.get('union_s'))}" for k, v in by_kind.items() if v.get("union_s"))
    out["tooling"] = {
        "flag": bool(tooling_flag) if tooling_flag is not None else None,
        "line": (
            f"{tooling} of {n_sc} self-check results tooling-incomplete; {timeouts} unmeasured timeouts; "
            f"self-check latency median {_mins(lat)}; author blocked {_pct(author.get('blocked_share'))} "
            f"({waits or 'no waits classified'})"
        ),
    }

    # Capsules: what genuinely fails on the newest evidence.
    cur = taxonomy.get("current", {}) if taxonomy.get("available") else {}
    classes = cur.get("classes") or {}
    fails = classes.get("genuine_fail", 0)
    older = (cur.get("evidence_source_version") or {}).get("different_source_version", 0)
    out["capsules"] = {
        "flag": bool(fails) if taxonomy.get("available") else None,
        "line": (
            f"{classes.get('pass', 0)} pass, {fails} genuine fail {dict(cur.get('genuine_fail') or {})}, "
            f"{classes.get('tooling_incomplete', 0)} tooling-incomplete, {classes.get('gated', 0)} gated with no "
            f"other evidence, {classes.get('no_evidence', 0)} never run ({older} judged on an older source "
            f"version); {(taxonomy.get('never_selfchecked') or {}).get('n')} never self-checked"
        ),
    }
    return out


# --- composition --------------------------------------------------------------------------------


def _run_state(run_dir: Path, author: dict, now: float) -> dict:
    newest = None
    for path in (run_dir / "rounds").glob("round_*.codex_events.timestamped.jsonl"):
        t = _mtime(path)
        newest = t if newest is None or (t and t > newest) else newest
    for path in (run_dir / "qa_history").glob("verdict_inturn_*.json"):
        t = _mtime(path)
        newest = t if newest is None or (t and t > newest) else newest
    age = None if newest is None else now - newest
    return {
        "last_activity_at": _iso(newest),
        "activity_age_s": _round(age),
        "looks_live": age is not None and age < 600,
        "unterminated_commands": author.get("n_unterminated") if author.get("available") else None,
    }


def build_report(
    run_dir: Path, operator_dir: Path | None = None, *, cpus: list[int] | None = None, now: float | None = None
) -> dict:
    run_dir = Path(run_dir)
    operator_dir = Path(operator_dir) if operator_dir is not None else None
    now = datetime.now(UTC).timestamp() if now is None else now
    records, unreadable = _evidence(run_dir)
    author = author_time(run_dir)
    grades = grade_cycle(run_dir)
    taxonomy = capsule_taxonomy(run_dir, records, unreadable)
    cap = capacity(operator_dir, records)
    suite = taxonomy.get("n_capsules_known") if taxonomy.get("available") else None
    cpu_list = sorted(set((cpus or [8, 16, 32]) + ([cap["pool_size"]] if cap.get("pool_size") else [])))
    cost = l3_cost(records, suite, cpu_list)
    selfchecks = selfcheck_latency(run_dir)
    report = {
        "schema": SCHEMA,
        "run_dir": str(run_dir),
        "operator_dir": str(operator_dir) if operator_dir else None,
        "generated_at": _iso(now),
        "run_state": _run_state(run_dir, author, now),
        "author": author,
        "grade_cycle": grades,
        "capsules": taxonomy,
        "l3_cost": cost,
        "capacity": cap,
        "selfcheck_latency": selfchecks,
        "budget": _budget(author, grades, selfchecks),
        "verdicts": _verdicts(author, grades, taxonomy, cost, cap, selfchecks),
        "thresholds": THRESHOLDS,
    }
    report["summary"] = summary_lines(report)
    return report


def summary_lines(report: dict) -> list[str]:
    lines = [f"Phase 1 convergence report: {Path(report['run_dir']).name}"]
    state = report["run_state"]
    lines.append(
        f"  state: last activity {state['last_activity_at']} ({'live' if state['looks_live'] else 'idle/finished'})"
    )
    a = report["author"]
    if a.get("available"):
        lines.append(
            f"  author: wall {_mins(a['wall_s'])}, {a['n_commands']} commands, {a['n_edits']} edits; blocked "
            f"{_mins(a['blocked_s'])} ({_pct(a['blocked_share'])}), incl. <{a['repoll_gap_s']:.0f}s re-poll gaps "
            f"{_mins(a['blocked_incl_repoll_gaps_s'])}"
        )
        lines.append(
            "  author windows: " + ", ".join(f"{w['start'][11:16]} {_pct(w['blocked_share'])}" for w in a["windows"])
        )
    else:
        lines.append(f"  author: unavailable ({a.get('reason')})")
    g = report["grade_cycle"]
    if g.get("available"):
        for rec in g["grades"]:
            lines.append(
                f"  grade {rec['key']}: {rec['state']}, results {rec['n_results']}/{rec['n_capsule_dirs']}, "
                f"snap->verdict {_mins(rec.get('snapshot_to_verdict_s'))}, "
                f"tail {_mins(rec.get('tail_after_verdict_s'))}, gap {_mins(rec.get('gap_to_next_snapshot_s'))}, "
                f"gated {rec.get('gated')}/{rec.get('n_capsules')}{' superseded' if rec.get('superseded') else ''}"
            )
    c = report["l3_cost"]
    if c.get("available"):
        t = c["timing_s"]
        p = c.get("projection") or {}
        lines.append(
            f"  L3: {c['n_measured']} measured; median build {t['build_s']['median']} s, active "
            f"{t['sim_active_s']['median']} s, queue {t['oracle_wait_s']['median']} s; cycles/s median "
            f"{c['cycles_per_s']['median']} (<=1000-cycle runs: {c['small_capsule_sim_active_s']['median']} s active; "
            f"marginal {c['marginal_cycles_per_s']['median']} cycles/s)"
        )
        lines.append(
            f"  L3 full pass: {p.get('cpu_minutes')} CPU-min; wall at N CPUs {p.get('wall_minutes_at')} min; "
            f"longest-capsule floor {p.get('longest_capsule_floor_s')} s"
        )
    for name, check in report["budget"].items():
        mark = {True: "MET", False: "MISSED", None: "n/a"}[check["met"]]
        values = ", ".join(f"{k} {v}" for k, v in check.items() if k not in ("met", "rule"))
        lines.append(f"  budget {name}: {mark} ({values})")
    for name, verdict in report["verdicts"].items():
        mark = {True: "YES", False: "no", None: "n/a"}[verdict["flag"]]
        lines.append(f"  {name.upper():10s} [{mark}] {verdict['line']}")
    return lines


def _inside(path: Path, parent: Path | None) -> bool:
    if parent is None:
        return False
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, type=Path, help="Phase 1 run directory (live or finished)")
    ap.add_argument("--operator-dir", type=Path, help="operator directory holding live.log and health.json")
    ap.add_argument("--out", type=Path, help="write the JSON report here (never inside the run/operator dir)")
    ap.add_argument("--json", action="store_true", help="print the JSON report instead of the summary")
    ap.add_argument("--cpus", default="8,16,32", help="comma-separated CPU counts for the full-pass wall projection")
    args = ap.parse_args(argv)
    if not args.run_dir.is_dir():
        ap.error(f"--run-dir {args.run_dir} is not a directory")
    try:
        cpus = [int(x) for x in args.cpus.split(",") if x.strip()]
    except ValueError:
        ap.error("--cpus must be comma-separated integers")
    if args.out and (_inside(args.out, args.run_dir) or _inside(args.out, args.operator_dir)):
        ap.error("--out must not be inside the run or operator directory (the report is read-only)")
    report = build_report(args.run_dir, args.operator_dir, cpus=cpus)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("\n".join(report["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
