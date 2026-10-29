"""Bounded operator-only agent and compiler-result metadata for terminal views.

Read existing records, never execute jobs or infer an optimization win from a
command's exit status. This view has no grading or promotion authority.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

from .tail import JsonlTail


SCHEMA = "merlin.terminal_metadata.v1"


def _object(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return value[:256] if isinstance(value, list) else []


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def capsule_metric(row, *, source_sha256=None, source=None):
    """Keep native estimates and structural floors bound to their exact domain."""
    perf = _object(row.get("performance"))
    if perf.get("schema") != "merlin.phase1.joint_performance.v1":
        perf = {}
    binding = _object(perf.get("structural_comparison_binding"))
    keys = ("capsule_sha256", "facts_sha256", "selection_sha256", "model_sha256",
            "comparison_domain_sha256", "scope_sha256")
    matched = all(perf.get(k) and binding.get(k) == perf[k] for k in keys)
    cycles = perf.get("estimated_whole_soc_cycles")
    usable = (matched and source_sha256 and perf.get("complete_event_accounting") is True
              and perf.get("status") in ("PROVISIONAL", "QUALIFIED") and _number(cycles))
    roof = _object(perf.get("structural_roofline")).get("structural_compute_floor_cycles")
    ratio = perf.get("structural_floor_to_estimate_ratio")
    ratio_ok = (usable and _number(roof) and roof <= cycles and _number(ratio)
                and math.isclose(ratio, roof / cycles, rel_tol=1e-9, abs_tol=1e-12))
    scoped = _object(perf.get("scoped_resource_feedback"))
    scoped_ok = (usable and scoped.get("schema") == "merlin.scoped_performance_feedback.v1"
                 and scoped.get("activity_status") == "modeled"
                 and scoped.get("estimated_cycles_same_window") == cycles
                 and all(scoped.get(k) == perf.get(k) for k in ("scope_sha256", "model_sha256")))
    resources = []
    if scoped_ok:
        for resource in _rows(scoped.get("resources")):
            ratio = _object(resource).get("capacity_normalized_activity")
            if type(ratio) in (int, float) and math.isfinite(ratio) and 0 <= ratio <= 1:
                resources.append({"name": resource.get("name"), "activity_pct": 100 * ratio})
        resources.sort(key=lambda r: -r["activity_pct"])
    return {
        "modeled_resource_activity": resources[:3],
        "modeled_resource_activity_status": "modeled" if scoped_ok else "UNKNOWN",
        "capsule": row.get("capsule") or row.get("name"),
        "source_sha256": source_sha256, "status": row.get("status", "UNKNOWN"),
        "numeric_status": _object(row.get("numeric")).get("status", "UNKNOWN"),
        "model_status": perf.get("status", "UNKNOWN"), "timing_mode": perf.get("timing_mode"),
        "model_sha256": perf.get("model_sha256"), "timing_epoch": perf.get("timing_epoch"),
        "timing_qualified": perf.get("timing_qualified") is True,
        "cycles": cycles if usable else None, "engine": "spike_model" if usable else None,
        "metric_kind": "estimated_whole_soc_cycles", "scope": perf.get("scope"),
        "scope_excludes": perf.get("scope_excludes", []),
        "scope_sha256": perf.get("scope_sha256"), "capsule_sha256": perf.get("capsule_sha256"),
        "comparison_domain_sha256": perf.get("comparison_domain_sha256"),
        "roofline_cycles": roof if ratio_ok else None,
        "roofline_efficiency_pct": 100 * ratio if ratio_ok else None,
        "roofline_gap_pct": 100 * (1 - ratio) if ratio_ok else None,
        "roofline_definition": "100 * structural compute floor / estimated complete-call cycles; "
                               "not measured utilization or an attainable whole-system roofline",
        "binding_status": "matched" if usable else "UNKNOWN",
        "source": source, "promotion_authority": False,
    }


class _Tail(JsonlTail):
    """The existing offset parser with a per-poll byte ceiling."""

    def bounded(self, limit):
        rows = []
        try:
            st = self.path.stat()
            identity = st.st_dev, st.st_ino
            reset = self.identity is not None and (identity != self.identity or st.st_size < self.offset)
            if reset:
                self.offset = 0
            self.identity = identity
            begin = self.offset
            with self.path.open("rb") as stream:
                stream.seek(begin)
                data = stream.read(limit)
            cut = data.rfind(b"\n")
            if cut >= 0:
                self._parse(data[:cut + 1], rows.append)
                self.offset += cut + 1
            elif len(data) == limit:
                # Do not silently jump over a giant record or parse a fragment.
                return [], len(data), reset, "record exceeds poll byte bound"
            self.bytes_read += len(data)
            return rows, len(data), reset, None
        except (OSError, ValueError):
            return [], 0, False, "unavailable"


class Metadata:
    """Read a bounded current-run view; all paths/details require operator mode."""

    def __init__(self, task: Path, operator_private=False, *, max_files=64, max_bytes=2 << 20):
        self.task = Path(task)
        self.operator_private = operator_private
        self.max_files, self.max_bytes = max_files, max_bytes
        if max_files < 1 or max_bytes < 1024:
            raise ValueError("metadata bounds must be positive and permit a JSON record")
        self._run = None
        self._tails = {}
        self._agents = {}
        self._rollout_meta = {}
        self._documents = {}
        from .terminal_capsules import Capsules
        self._capsule_counts = Capsules()

    def _read(self, path, *, yaml_document=False):
        path = Path(path)
        if self._files >= self.max_files:
            self._notices.add("file scan bound reached")
            return None, None
        self._files += 1
        try:
            if path.is_symlink():
                raise ValueError("indirect metadata refused")
            st = path.stat()
            signature = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            cached = self._documents.get(path)
            if cached and cached[0] == signature:
                return cached[1], cached[2]
            if st.st_size > self._remaining:
                self._notices.add("JSON record exceeds remaining byte bound")
                return None, None
            with path.open("rb") as stream:
                data = stream.read(self._remaining)
            self._remaining -= len(data)
            now = path.stat()
            if signature != (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns):
                raise ValueError("metadata changed during read")
            if yaml_document:
                import yaml
                try:
                    doc = yaml.load(data, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
                except yaml.YAMLError as exc:
                    raise ValueError("unreadable YAML") from exc
            else:
                doc = json.loads(data)
            if not isinstance(doc, dict):
                raise ValueError("metadata is not an object")
            pin = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                   "live": True, "meaning": "observed bytes, not immutable custody"}
            self._documents[path] = signature, doc, pin
            return doc, pin
        except (OSError, ValueError, UnicodeError):
            return None, None

    def _agent_rows(self, path, rows, reset):
        if reset:
            old = self._rollout_meta.pop(path, None)
            if old:
                self._agents.pop(old.get("id"), None)
        for row in rows:
            payload = row.get("payload") or {}
            if row.get("type") == "session_meta" and path not in self._rollout_meta:
                if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
                    continue
                self._rollout_meta[path] = payload
                origin = _object(payload.get("source"))
                spawn = _object(_object(origin.get("subagent")).get("thread_spawn"))
                parent = payload.get("parent_thread_id") or spawn.get("parent_thread_id")
                git = _object(payload.get("git"))
                self._agents[payload["id"]] = dict(
                    id=payload["id"], parent_id=parent,
                    name=payload.get("agent_nickname") or spawn.get("agent_nickname") or payload["id"],
                    agent_path=payload.get("agent_path") or spawn.get("agent_path"),
                    branch=git.get("branch"), status="observed", last_activity=payload.get("timestamp"),
                    source={"path": str(path), "live": True},
                    status_basis="own recorded lifecycle; absence is not death")
                continue
            meta = self._rollout_meta.get(path)
            if not meta or row.get("type") == "session_meta" or not isinstance(payload, dict):
                continue
            ordinal, start = row.get("ordinal"), meta.get("subagent_history_start_ordinal")
            if type(start) is int and (type(ordinal) is not int or ordinal < start):
                continue  # Forked parent history is not the helper's own work.
            if payload.get("thread_id", meta["id"]) != meta["id"]:
                continue
            agent = self._agents[meta["id"]]
            if row.get("type") == "event_msg":
                kind = payload.get("type")
                if kind in ("task_started", "task_complete", "task_aborted"):
                    agent["status"] = {"task_started": "working", "task_complete": "completed",
                                       "task_aborted": "aborted"}[kind]
                agent["last_activity"] = row.get("timestamp", agent["last_activity"])

    def _rollouts(self, run):
        roots = list(itertools.islice((run / "rounds").glob("round_*.codex_rollout_snapshot"), 8))
        for summary in itertools.islice((run / "rounds").glob("round_*.codex_summary.json"), 8):
            doc, _ = self._read(summary)
            home = _object(_object(doc).get("codex_home")).get("codex_home")
            if isinstance(home, str):
                roots.append(Path(home) / "sessions")
        env, _ = self._read(run / "environment.yaml", yaml_document=True)
        env = _object(env)
        policies = [_object(env.get("codex_runtime_policy")),
                    _object(_object(env.get("run_config")).get("codex_runtime"))]
        for policy in policies:
            home_root = policy.get("home_root")
            if isinstance(home_root, str) and Path(home_root).is_absolute():
                for home in itertools.islice(Path(home_root).glob(f"{run.name}_r*"), 16):
                    suffix = home.name[len(run.name) + 2:]
                    if suffix.isdigit():
                        roots.append(home / "sessions")
        return itertools.chain.from_iterable(root.glob("*/*/*/rollout-*.jsonl")
                                            for root in dict.fromkeys(roots))

    def sample(self, core_snapshot):
        self._files, self._remaining, self._notices = 0, self.max_bytes, set()
        output = dict(schema=SCHEMA, operator_private=self.operator_private, agents=[], attempts=[],
                      capsules=[], coverage=None, sources=[], notices=[], promotion_authority=False)
        if not self.operator_private:
            output["notices"] = ["Detailed agent, source and capsule records require operator-private mode."]
            return output
        campaign = _object(core_snapshot.get("campaign"))
        raw_run = campaign.get("run_dir")
        if not raw_run:
            output["notices"] = ["Current run directory not recorded."]
            return output
        run = Path(raw_run)
        if self._run != run:
            self._tails.clear(); self._agents.clear(); self._rollout_meta.clear(); self._documents.clear()
            self._run = run
        workspace = Path(campaign["workspace"]) if campaign.get("workspace") else None
        rollout_paths = campaign.get("rollout_paths")
        paths = map(Path, rollout_paths) if isinstance(rollout_paths, list) else self._rollouts(run)
        for path in itertools.islice(paths, max(0, self.max_files // 2)):
            if self._remaining <= 0 or self._files >= self.max_files:
                self._notices.add("JSONL byte bound reached"); break
            self._files += 1
            if path.is_symlink():
                continue
            tail = self._tails.setdefault(path, _Tail(path))
            rows, consumed, reset, error = tail.bounded(min(self._remaining, 512 << 10))
            self._remaining -= consumed
            self._agent_rows(path, rows, reset)
            if error:
                self._notices.add(error)
            output["sources"].append(dict(path=str(path), live=True, consumed_offset=tail.offset))
        output["agents"] = list(self._agents.values())
        if workspace:
            output["capsule_counts"] = self._capsule_counts.sample(run, workspace)
            progress, pin = self._read(workspace / "qa/progress.json")
            if progress and progress.get("schema") == "merlin.phase1.grade_progress.v1":
                output["coverage"] = {k: progress.get(k) for k in
                    ("state", "n_selected", "n_completed", "n_passed", "graded_source_sha256", "snapshot_current")}
                output["coverage"]["source"] = pin
                for row in _rows(progress.get("per_capsule")):
                    if isinstance(row, dict):
                        output["capsules"].append(capsule_metric(row,
                            source_sha256=progress.get("graded_source_sha256"), source=pin))
            channel = workspace / ".qa_channel"
            responses = itertools.chain(channel.glob("resp_*.json"), channel.glob("simresp_*.json"))
            recent = []
            for path in itertools.islice(responses, 4096):
                try:
                    recent.append((path.stat().st_mtime_ns, path))
                except OSError:
                    continue
            if len(recent) == 4096:
                self._notices.add("Response directory inventory capped; view may be incomplete.")
            for _, path in sorted(recent, reverse=True)[:max(0, self.max_files - self._files)]:
                doc, pin = self._read(path)
                if not doc:
                    continue
                rid = doc.get("selfcheck_request_id") or doc.get("job_id") or path.stem.partition("_")[2]
                source_hash = doc.get("submission_sha256") or doc.get("graded_source_sha256")
                output["attempts"].append(dict(
                    request_id=rid, time=doc.get("graded_at_unix_ns"), time_unit="unix_ns",
                    title="Native self-check" if path.name.startswith("resp_") else "Native simulation request",
                    status="pass" if doc.get("all_pass") is True else "fail" if doc.get("all_pass") is False else "UNKNOWN",
                    source_sha256=source_hash, requested_source_sha256=doc.get("requested_submission_sha256"),
                    hypothesis=None, pass_diff=None, source=pin,
                    outcome_basis="recorded native response; no optimization win inferred"))
                for row in _rows(doc.get("per_capsule") or doc.get("capsules")):
                    if isinstance(row, dict):
                        output["capsules"].append(capsule_metric(row, source_sha256=source_hash, source=pin))
        else:
            self._notices.add("Workspace not recorded; live QA feedback unavailable.")
        output["notices"] = sorted(self._notices)
        if not output["agents"]:
            output["notices"].append("Own rollout metadata unavailable; lead stream alone cannot identify all helpers.")
        return output
