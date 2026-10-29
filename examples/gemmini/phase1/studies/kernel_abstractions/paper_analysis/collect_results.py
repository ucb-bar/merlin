"""Read-only native campaign consolidation. No compiler, simulator or author is run.

The one CSV is an event table: select record_type before aggregating its nullable
columns. Provider usage uses the native ownership/deduplication implementation.
All timestamps are observations/publications, never guessed implementation times.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

FIELDS = """schema_version run_id experiment cohort framework_revision seed_origin
author_budget_s author_started_at collection_cutoff_at record_type record_id
observed_at elapsed_s phase within_first_4h publication_at publication_elapsed_s source_sha256
source_hash_algorithm attempt_id grade_id case_id case_family suite visibility
status numeric_status ordering_status l0_status l1_status l2_status l3_status
numerical_execution_passed barrier_tier barrier_status barrier_cycles requested_sim
failure_plane failure_category execution_status coherent_snapshot count_scope
n_cases n_passed n_numeric_passed n_l2_passed n_l3_passed n_failed n_pending n_gated n_refused
thread_id agent_role response_id tokens_input tokens_fresh_input tokens_cache_read
tokens_cache_write tokens_output tokens_reasoning tokens_total token_lower_bound
queue_s latency_s estimated_cycles fast_estimate_cycles measured_cycles
engine performance_status model_sha256 comparison_domain_sha256 scope_sha256
capsule_sha256 elf_sha256 complete_event_accounting scheduled_elapsed_s checkpoint_lag_s
wall_s exit_code tool_kind changed_paths_count metric value unit split shape_json
pair_id hardware_status source_path evidence_path evidence_sha256 evidence_locator notes""".split()
FIELDS += "subject sample_round sample_slot model_name program_name timestamp_source superseded numeric_credit timing_qualified details_json".split()
FIELDS += "evaluation_phase model_engine measured_engine".split()
SCHEMA = "merlin.paper_campaign_events.v1"
TOKEN_MAP = {"input_total": "tokens_input", "fresh_input": "tokens_fresh_input",
             "cache_read": "tokens_cache_read", "cache_write": "tokens_cache_write",
             "output": "tokens_output", "reasoning": "tokens_reasoning", "total": "tokens_total"}


def stamp(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        return float(v)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def iso(v):
    t = stamp(v)
    return datetime.fromtimestamp(t, timezone.utc).isoformat() if t is not None else None


def digest(b):
    return hashlib.sha256(b).hexdigest()


def stable_id(*values):
    return digest(json.dumps(values, sort_keys=True, default=str).encode())


def tier_status(tiers, name):
    v = (tiers or {}).get(name)
    return v.get("status") if isinstance(v, dict) else v


def family(case):
    c = (case or "").lower()
    if c.startswith("pg_") or "connected" in c:
        return "program"
    if c.startswith("jh_"):
        return "host"
    if "conv" in c or "pv0" in c:
        return "convolution"
    if "att_" in c or "kv_head" in c:
        return "attention"
    if c.startswith("ch_"):
        return "chain"
    if "residual" in c or "epilogue" in c or "bias" in c or "relu" in c:
        return "epilogue_residual"
    if "matmul" in c or "gemm" in c or "contraction" in c or "int_mm" in c or "_mm_" in c:
        return "dense"
    if c.startswith(("pm", "pc", "pq", "ps")):
        return "calibration"
    return "other"


def explicit_failure(row):
    # An unrun oracle is not a failed test. A failed structural trace is recorded
    # separately from dynamic completion-order checking and is not called a race.
    if row.get("status") == "fail" or row.get("numeric_status") == "fail":
        return True
    if any(tier_status(row.get("tiers"), t) == "fail" for t in ("L0", "L1", "L2", "L3")):
        return True
    failure = row.get("failure") or {}
    return row.get("barrier_status") == "fail" and failure.get("plane") not in ("oracle_unavailable", "unavailable")


class Collector:
    def __init__(self, task, out, cutoff):
        self.task, self.repo, self.out = task, task / "merlin", out
        self.cutoff = cutoff
        self.rows, self.sources, self.issues, self.runs = [], {}, [], []

    def read(self, path):
        path = Path(path)
        data = path.read_bytes()
        self.sources[str(path)] = {"path": str(path), "bytes": len(data), "sha256": digest(data),
                                   "read_at": iso(datetime.now(timezone.utc).timestamp()), "prefix": False}
        return json.loads(data)

    def add(self, meta, kind, at, evidence=None, locator=None, **values):
        t = stamp(at)
        publication = stamp(values.get("publication_at"))
        available = max(t, publication) if t is not None and publication is not None else t or publication
        # Post-cutoff live writes must not sneak into a frozen analysis release.
        if available is not None and available > self.cutoff:
            return
        elapsed = t - meta["started_epoch"] if t is not None and meta["started_epoch"] is not None else None
        visible_elapsed = available - meta["started_epoch"] if available is not None and meta["started_epoch"] is not None else None
        phase = ("pre_authoring" if elapsed < 0 else
                 "authoring" if elapsed <= meta["author_budget_s"] else
                 "drain" if elapsed <= meta["author_budget_s"] + meta["drain_s"] else "post_campaign") if elapsed is not None else "unknown"
        record = {"schema_version": SCHEMA, **{k: meta.get(k) for k in
            ("run_id", "experiment", "cohort", "framework_revision", "seed_origin", "author_budget_s", "author_started_at")},
            "collection_cutoff_at": iso(self.cutoff), "record_type": kind,
            "observed_at": iso(t), "elapsed_s": round(elapsed, 6) if elapsed is not None else None,
            "phase": phase, "within_first_4h": 0 <= visible_elapsed <= 14400 if visible_elapsed is not None else None,
            **values}
        if publication is not None:
            record["publication_elapsed_s"] = round(publication - meta["started_epoch"], 6)
        if evidence:
            record.update(evidence_path=str(evidence), evidence_sha256=self.sources.get(str(evidence), {}).get("sha256"),
                          evidence_locator=locator)
        record.setdefault("record_id", stable_id(meta["run_id"], kind, record))
        unknown = set(record) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown CSV columns: {unknown}")
        self.rows.append(record)

    def metadata(self, entry):
        run = self.repo / "out/runs/gemmini/phase1" / entry["run_id"]
        clock = self.read(run / "joint_campaign.json")
        policy = clock["policy"]
        t0 = clock["started_at_epoch"]
        # The immutable registry is authoritative for declared framework/source.
        meta = dict(run_id=entry["run_id"], experiment=entry["name"], cohort=entry["group"],
                    framework_revision=entry["source_revision"],
                    author_budget_s=policy["duration_s"] - policy.get("drain_s", 0),
                    drain_s=policy.get("drain_s", 0), started_epoch=t0, author_started_at=iso(t0),
                    seed_origin="seeded" if entry["name"] == "fresh10seed" else "empty_backend",
                    run_directory=str(run), model=entry.get("model"),
                    authoring_complete_at_cutoff=self.cutoff >= t0 + policy["duration_s"] - policy.get("drain_s", 0),
                    cpus=entry.get("cpus"), evaluation_policy="deferred" if entry["name"] in ("el1", "el2", "el3") else "continuous_promotion")
        ctx = run / "study-metrics-context.json"
        if ctx.exists():
            context = self.read(ctx)
            roster = context.get("public_cases_by_suite") or {}
            meta["case_suites"] = {c: suite for suite, cases in roster.items() if isinstance(cases, list) for c in cases}
            meta["public_roster"] = context.get("public_case_ids", [])
        else:
            # The tested run roster is the common frozen xs workload roster.
            context = self.read(self.repo / "out/runs/gemmini/phase1/xs-el123-20261028-r2-el1/study-metrics-context.json")
            meta["case_suites"] = {c: suite for suite, cases in context["public_cases_by_suite"].items() for c in cases}
            meta["public_roster"] = context["public_case_ids"]
            meta["roster_basis"] = "common xs roster; observed grade IDs independently cross-checked"
        # Avoid reading environment.yaml or any credentials for workspace lookup.
        meta["workspace"] = str(self.repo / "out/build/agent-workspaces/phase1/gemmini/merlin_assisted" / entry["run_id"] / "workspace")
        self.runs.append(meta)
        self.add(meta, "run", t0, run / "joint_campaign.json", status="observed",
                 notes=f"{meta['evaluation_policy']}; authoring unit is one independent run, not its cases")
        return meta

    def cases(self, meta, doc, path, channel, publication, grade_id):
        rows = doc.get("per_capsule") or []
        if not isinstance(rows, list):
            return
        top_source = doc.get("graded_source_sha256") or doc.get("submission_sha256")
        explicit_sources = {(r.get("graded") or {}).get("graded_source_sha256") for r in rows if isinstance(r, dict)} - {None}
        coherent = bool(top_source and explicit_sources <= {top_source} and doc.get("count_scope") in (None, "immutable_snapshot"))
        counts = Counter()
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            case, origin, perf = row.get("capsule"), row.get("graded") or {}, row.get("performance") or {}
            at = origin.get("graded_at") or publication
            source = origin.get("graded_source_sha256") or top_source
            if not source and isinstance(perf.get("feedback_brief"), dict):
                source = (perf["feedback_brief"].get("experiment_record") or {}).get("compiler_sha256")
            tiers = row.get("tiers") or {}
            l2, l3 = tier_status(tiers, "L2"), tier_status(tiers, "L3")
            # A redacted pass can mean a successful required refusal or a static
            # check. It is not proof of numerical execution.
            numeric = row.get("numeric_status") or ("pass" if row.get("numerical_execution_passed") is True else None)
            counts["numeric_passed"] += numeric == "pass"
            counts["l2_passed"] += l2 == "pass"
            counts["l3_passed"] += l3 == "pass"
            counts["failed"] += explicit_failure(row)
            counts["refused"] += (not explicit_failure(row) and
                (row.get("status") == "refused" or
                 (case == "SY_refuse_requant" and (row.get("status") == "pass" or row.get("pass") is True))))
            counts["gated"] += row.get("status") == "gated"
            counts["pending"] += row.get("status") in ("pending", "incomplete", "unavailable", "skipped", "gated")
            model_cycles = perf.get("estimated_whole_soc_cycles")
            measured = perf.get("measured_cycles")
            gsim = perf.get("gsim") or {}
            if isinstance(gsim, dict):
                measured = measured or gsim.get("cycles")
            complete = perf.get("complete_event_accounting")
            timing_status = perf.get("timing_status") or perf.get("status")
            failure = row.get("failure") or {}
            self.add(meta, channel + "_case", at, path, f"/per_capsule/{i}",
                     publication_at=iso(publication), source_sha256=source,
                     source_hash_algorithm=doc.get("source_hash_algorithm") or "native graded_source_sha256",
                     attempt_id=doc.get("graded_request_sha256") or grade_id, grade_id=grade_id,
                     case_id=case, case_family=family(case), suite=meta["case_suites"].get(case, "unknown"),
                     visibility="public" if case in meta["public_roster"] else "unclassified",
                     status=row.get("status"), numeric_status=numeric, ordering_status=row.get("trace_status"),
                     numerical_execution_passed=row.get("numerical_execution_passed"),
                     barrier_tier=row.get("barrier_tier"), barrier_status=row.get("barrier_status"),
                     barrier_cycles=row.get("barrier_cycles"), requested_sim=doc.get("sim"),
                     l0_status=tier_status(tiers, "L0"), l1_status=tier_status(tiers, "L1"), l2_status=l2, l3_status=l3,
                     failure_plane=row.get("failure_plane") or failure.get("plane"),
                     failure_category=row.get("failure_category") or failure.get("category"),
                     execution_status=row.get("execution_status"), coherent_snapshot=coherent,
                     count_scope=doc.get("count_scope"), estimated_cycles=model_cycles,
                     fast_estimate_cycles=(perf.get("fast_estimate") or {}).get("cycles"), measured_cycles=measured,
                     engine="spike_timing_model" if model_cycles is not None else None,
                     model_engine="spike_timing_model" if model_cycles is not None else None,
                     measured_engine=perf.get("measured_engine") or (gsim.get("engine") if isinstance(gsim, dict) else None),
                     performance_status=timing_status, model_sha256=perf.get("model_sha256"),
                     comparison_domain_sha256=perf.get("comparison_domain_sha256"), scope_sha256=perf.get("scope_sha256"),
                     capsule_sha256=perf.get("capsule_sha256"), elf_sha256=row.get("elf_sha256") or perf.get("elf_sha256"),
                     complete_event_accounting=complete,
                     notes="trace_status is structural trace verification, not automatically a dynamic race verdict")
        self.add(meta, channel + "_snapshot", publication, path, "/",
                 source_sha256=top_source, grade_id=grade_id,
                 attempt_id=doc.get("graded_request_sha256") or grade_id,
                 coherent_snapshot=coherent, count_scope=doc.get("count_scope"), status=doc.get("stage"),
                 n_cases=len(rows), n_passed=doc.get("n_passed"), n_numeric_passed=counts["numeric_passed"],
                 n_l2_passed=counts["l2_passed"], n_l3_passed=counts["l3_passed"], n_failed=counts["failed"],
                 n_pending=counts["pending"], n_gated=counts["gated"], n_refused=counts["refused"],
                 notes="Counts describe these published rows; partial selections and cross-source mosaics are not a final compiler grade")

    def grades(self, meta):
        from merlin_experiments.phase1.feedback.result_details import hydrate
        run, ws = Path(meta["run_directory"]), Path(meta["workspace"])
        for path in sorted((run / "qa_history").glob("verdict*.json")):
            try:
                doc = self.read(path)
                # Detail blobs carry additional timing/numerical fields. Native
                # hydration enforces their hashes and resolves only result refs.
                doc = hydrate(doc, workspace=ws)
                self.cases(meta, doc, path, "grade", doc.get("graded_at"), path.stem)
            except (OSError, ValueError, TypeError, KeyError) as e:
                self.issues.append({"path": str(path), "stage": "grade", "error": f"{type(e).__name__}: {e}"})

    def requests(self, meta):
        from merlin_experiments.phase1.telemetry import agent_requests as ar
        from merlin_experiments.phase1.feedback.result_details import hydrate
        run, ws = Path(meta["run_directory"]), Path(meta["workspace"])
        seen = set()
        for root in ar._channels(run, str(ws)):
            for path in sorted(root.glob("req_*.json")):
                rid = path.stem.removeprefix("req_")
                if rid in seen:
                    continue
                seen.add(rid)
                try:
                    req = self.read(path)
                    rp = root / f"resp_{rid}.json"
                    resp = self.read(rp) if rp.exists() else {}
                    native = ar._request_record(rid, {"request": req, "response": resp})
                    at = native.get("graded_at") or native.get("requested_at")
                    status = resp.get("status") or ("answered" if resp else "pending")
                    self.add(meta, "request", at, rp if resp else path,
                             attempt_id=rid, source_sha256=resp.get("submission_sha256"),
                             thread_id=native.get("caller_thread_id"), status=status,
                             queue_s=native.get("queue_s"), latency_s=native.get("latency_s"),
                             n_passed=native.get("n_passed"), n_cases=native.get("n_result_rows"),
                             failure_category=resp.get("failure_category"),
                             notes=f"mode={native.get('mode')}; sim={native.get('sim')}; error={str(resp.get('error') or resp.get('reason') or '')[:240]}")
                    if resp:
                        self.cases(meta, hydrate(resp, workspace=ws), rp, "selfcheck", at, rid)
                except (OSError, ValueError, TypeError, KeyError) as e:
                    self.issues.append({"path": str(path), "stage": "request", "error": f"{type(e).__name__}: {e}"})

    def tokens(self, meta):
        audit = self.task / ".tmp/paper-results-20261028/audits"
        name = meta["experiment"]
        token_file = (audit / "el123" / f"{name}-usage-events.jsonl" if name in ("el1", "el2", "el3") else
                      audit / "fresh10" / f"{name}-token-events.jsonl")
        if token_file.exists():
            data = token_file.read_bytes()
            self.sources[str(token_file)] = {"path": str(token_file), "bytes": len(data), "sha256": digest(data), "prefix": False}
            count, total = 0, 0
            seen = set()
            for i, line in enumerate(data.splitlines(), 1):
                row = json.loads(line)
                if row["run_id"] != meta["run_id"]:
                    raise ValueError("token audit belongs to another run")
                key = row["thread_id"], row["response_id"]
                if key in seen:
                    raise ValueError("duplicate native response in token audit")
                seen.add(key)
                tokens = row["tokens"]
                at = row.get("at_epoch")
                self.add(meta, "provider_usage", at, token_file, f"line:{i}",
                         thread_id=row["thread_id"], response_id=row["response_id"],
                         agent_role="helper" if row.get("parent_thread_id") else "lead",
                         token_lower_bound=True, notes="native evidence: " + str(row.get("evidence_path") or row.get("evidence")),
                         **{dest: tokens[key] for key, dest in TOKEN_MAP.items()})
                if at is not None and at <= self.cutoff:
                    count += 1
                    total += tokens["total"]
            meta["token_accounting"] = dict(response_events=count, completed_response_tokens=total,
                audit_path=str(token_file), basis="audited native own-thread response events; no cumulative lead-summary counters")
            return
        from merlin_experiments.phase1.telemetry import author_agents as aa
        reader = aa.UsageRows()
        found = aa.threads(Path(meta["run_directory"]), _read_rows=reader)
        seen = set()
        accounting = {"threads": len(found["records"]), "files_scanned": found["files_scanned"],
                      "missing_leads": found["missing_leads"], "missing_helpers": found["missing_subagents"],
                      "basis": "native own-thread records; fork-history exclusion; per-response deduplication; completed responses only"}
        total_events, total_tokens = 0, 0
        for tid, record in found["records"].items():
            rollout = record["rollout"]
            path = Path(rollout["path"])
            # Retain only accounting metadata in output. Never archive prompts or
            # command bodies. Pin the captured append-only prefix for provenance.
            raw = path.read_bytes()
            prefix = raw[:raw.rfind(b"\n") + 1]
            self.sources[str(path)] = {"path": str(path), "bytes": len(prefix), "sha256": digest(prefix), "prefix": True}
            for row in rollout["rows"]:
                payload = row.get("payload") or {}
                if row.get("type") != "token_usage_record" or not isinstance(payload.get("usage"), dict):
                    continue
                response = payload.get("response_id") or row.get("ordinal")
                key = (tid, str(response))
                if response is None or key in seen:
                    continue
                seen.add(key)
                at = stamp(row.get("timestamp"))
                if at is None:
                    self.issues.append({"stage": "tokens", "thread": tid, "error": "response timestamp absent"})
                    continue
                tokens = aa._tokens([payload["usage"]], None)
                self.add(meta, "provider_usage", at, path, f"thread={tid};response={response};ordinal={row.get('ordinal')}",
                         thread_id=tid, response_id=str(response), agent_role="helper" if rollout["parent_thread_id"] else "lead",
                         token_lower_bound=True, **{dest: tokens[key] for key, dest in TOKEN_MAP.items()})
                if at <= self.cutoff:
                    total_events += 1
                    total_tokens += tokens["total"]
        accounting.update(response_events=total_events, completed_response_tokens=total_tokens)
        meta["token_accounting"] = accounting

    def snapshots(self, meta):
        run = Path(meta["run_directory"])
        index = run / "study_snapshots/index.json"
        if index.exists():
            doc = self.read(index)
            for i, row in enumerate(doc.get("snapshots", [])):
                obs = row.get("observed_elapsed_s", row.get("offset_s"))
                self.add(meta, "source_snapshot", meta["started_epoch"] + obs, index, f"/snapshots/{i}",
                         source_sha256=row.get("compiler_sha256"), source_path=row.get("source"), status=row.get("status"),
                         scheduled_elapsed_s=row.get("offset_s"), checkpoint_lag_s=obs-row.get("offset_s", obs),
                         source_hash_algorithm=row.get("digest_basis"))
        if meta["experiment"] in ("fresh10cca", "fresh10seed"):
            for path in sorted((self.task / ".tmp/fresh10/snapshots").glob(meta["experiment"] + "-*/snapshot.json")):
                row = self.read(path)
                self.add(meta, "source_snapshot", row.get("snapshot_epoch"), path,
                         source_sha256=row.get("live_submission_sha256"), source_path=str(path.parent / "live-submission"),
                         scheduled_elapsed_s=row.get("offset_s"),
                         checkpoint_lag_s=stamp(row["snapshot_epoch"]) - meta["started_epoch"] - row["offset_s"],
                         status="frozen_live_source", notes=f"latest graded source differs={not row.get('live_equals_graded')}; live source has no implied grade")
                gp = path.parent / "latest_completed_grade.json"
                if gp.exists():
                    grade = self.read(gp)
                    # This is a cross-source historical view, not a paid attempt.
                    self.add(meta, "checkpoint_published_view", row.get("snapshot_epoch"), path,
                             source_sha256=row.get("graded_source_sha256"), scheduled_elapsed_s=row.get("offset_s"),
                             n_cases=row.get("n_capsules"), n_numeric_passed=(row.get("numerics") or {}).get("passing"),
                             n_passed=row.get("n_passed"), coherent_snapshot=False,
                             status="cross_source_mosaic", notes="latest published per-case evidence at checkpoint; not a coherent compiler/source score")
        if meta["experiment"].startswith("fresh10-tooling-"):
            path = self.task / ".tmp/fresh10-tooling-ablation-20261028/fresh10-tooling-20261028-r5/operator-checkpoints/index.json"
            doc = self.read(path)
            arm = meta["experiment"].removeprefix("fresh10-tooling-")
            for i, row in enumerate(doc.get("arms", {}).get(arm, {}).get("snapshots", [])):
                obs = row.get("observed_elapsed_s", row["offset_s"])
                self.add(meta, "source_snapshot", meta["started_epoch"] + obs, path, f"/arms/{arm}/snapshots/{i}",
                         source_sha256=row.get("compiler_sha256"), source_path=row.get("source"), status=row.get("status"),
                         scheduled_elapsed_s=row.get("offset_s"), checkpoint_lag_s=obs-row["offset_s"])

    def tools(self, meta):
        path = self.task / ".tmp/paper-results-20261028/audits/el123" / f"{meta['experiment']}-tool-outcomes.jsonl"
        if not path.exists():
            meta["command_outcomes_available"] = False
            return
        data = path.read_bytes()
        self.sources[str(path)] = {"path": str(path), "bytes": len(data), "sha256": digest(data), "prefix": False}
        for i, line in enumerate(data.splitlines(), 1):
            r = json.loads(line)
            self.add(meta, "author_tool", r.get("at_epoch"), path, f"line:{i}",
                     thread_id=r.get("thread_id"), agent_role="helper" if r.get("parent_thread_id") else "lead",
                     tool_kind=r.get("tool_kind"), wall_s=r.get("wall_s"), status=r.get("status"),
                     exit_code=r.get("exit_code"), changed_paths_count=r.get("changed_paths_count"),
                     notes="Operational proxy only; nonzero shell exits include non-test operations")
        meta["command_outcomes_available"] = True

    def jsonl(self, path):
        data = path.read_bytes()
        self.sources[str(path)] = {"path": str(path), "bytes": len(data), "sha256": digest(data), "prefix": False}
        return [(i, json.loads(line)) for i, line in enumerate(data.splitlines(), 1)]

    def model_health(self):
        root = self.task / ".tmp/paper-results-20261028/audits/model-health"
        paths = sorted(root.glob("model-health-*.jsonl"))
        if not paths:
            return
        path = paths[-1]  # One version per native gate/program, never sum exports.
        metadata = {m["run_id"]: m for m in self.runs}
        for line, row in self.jsonl(path):
            meta = metadata.get(row["run_id"])
            if meta is None:
                continue
            self.add(meta, "full_model_observation", row.get("gate_observed_at_utc"), path, f"line:{line}",
                     source_sha256=row.get("candidate_source_sha256"),
                     model_name=row.get("model"), program_name=row.get("program"),
                     status=row.get("native_program_status"), execution_status=row.get("observation_classification"),
                     superseded=row.get("superseded"), numeric_credit=row.get("numeric_credit"),
                     failure_category=row.get("shared_infrastructure_gap"), timestamp_source=row.get("timestamp_source"),
                     split=row.get("split"), wall_s=row.get("reported_total_seconds"),
                     notes="native gate: " + row["gate_receipt"]["path"],
                     details_json=json.dumps({k: row.get(k) for k in ("gate_passed", "gate_complete", "model_report_status", "model_returncode", "failure_metadata", "model_report")}, sort_keys=True))

    def small_studies(self):
        root = self.task / ".tmp/paper-results-20261028/audits/fresh10"
        path = root / "small-study-run-metadata.jsonl"
        if not path.exists():
            return
        metadata = {}
        for i, row in self.jsonl(path):
            meta = dict(run_id=row["run_id"], experiment="small-" + row["run_id"], cohort=row["cohort"],
                        framework_revision=Path(row["framework"]).name,
                        seed_origin="fresh6_with_declared_tool_migration", author_budget_s=row["wall_budget_s"],
                        started_epoch=row["author_start_unix"], author_started_at=iso(row["author_start_unix"]), drain_s=0,
                        authoring_complete_at_cutoff=True, model=row.get("model"),
                        helpers_allowed=row.get("helpers_allowed"), cpus=row.get("cpu_mask"),
                        cpu_equivalents=row.get("cpu_equivalents"),
                        evaluation_policy="30min family study; one independent author per arm",
                        initial_source_sha256=row.get("effective_initial_source_sha256"),
                        requested_seed_sha256=row.get("requested_initial_source_sha256"),
                        final_status=row.get("status"), pair_id=row.get("pair_id"))
            metadata[row["run_id"]] = meta
            self.runs.append(meta)
            self.add(meta, "run", row["author_start_unix"], path, f"line:{i}",
                     status=row["status"].get("status") if isinstance(row.get("status"), dict) else row.get("status"),
                     source_sha256=row.get("effective_initial_source_sha256"), pair_id=row.get("pair_id"),
                     details_json=json.dumps({k: row.get(k) for k in ("checkpoints", "interruptions", "cpu_equivalents", "helpers_allowed", "requested_initial_source_sha256")}, sort_keys=True),
                     notes="Actual 1800-second study, separate from four-hour EL/tooling campaigns; retained failure/capacity downtime")
        files = ["token-events", "author-events", "postfreeze-events", "case-metric-events", "metrics", "effort-events"]
        for suffix in files:
            path = root / f"small-study-{suffix}.jsonl"
            if not path.exists():
                continue
            for i, row in self.jsonl(path):
                key = row["run_id"]
                meta = metadata.get(key)
                if meta is None:
                    # Supplied expert comparison is a benchmark subject, never
                    # count it as another independent author run.
                    meta = dict(run_id=key, experiment="small-"+key+"-oracle", cohort="frozen_expert_comparison",
                                framework_revision=None, seed_origin="expert_reference", author_budget_s=0,
                                author_started_at=None, started_epoch=None, drain_s=0, independent_author_runs=0)
                    metadata[key] = meta
                    self.runs.append(meta)
                perf = row.get("performance") or {}
                failure = row.get("failure") or {}
                if not isinstance(failure, dict):
                    failure = {}
                kind = row.get("record_type") or row["event_type"]
                if kind == "response_tokens":
                    kind = "provider_usage"
                else:
                    kind = "small_" + kind
                values = {k: row.get(k) for k in ("source_sha256", "attempt_id", "case_id", "status", "engine", "metric", "unit", "split", "pair_id", "hardware_status", "thread_id", "response_id", "queue_s", "latency_s", "exit_code", "subject")}
                values.update(case_family=family(row.get("case_id")),
                              evaluation_phase=row.get("phase"), timestamp_source=row.get("publication_time_kind"),
                              shape_json=json.dumps(row.get("shape_json"), sort_keys=True) if row.get("shape_json") else None,
                              value=json.dumps(row.get("value")) if isinstance(row.get("value"), (dict, list)) else row.get("value"),
                              numeric_status=(row.get("numeric") or {}).get("status") or ("pass" if row.get("numeric_pass") is True else None),
                              failure_plane=failure.get("plane"), failure_category=failure.get("category"),
                              estimated_cycles=row.get("estimated_whole_soc_cycles") or perf.get("estimated_whole_soc_cycles"),
                              fast_estimate_cycles=row.get("fast_estimate_cycles") or (perf.get("fast_estimate") or {}).get("cycles"),
                              measured_cycles=row.get("cycles") if row.get("hardware") else row.get("median_cycles"),
                              model_engine="spike_model" if (row.get("estimated_whole_soc_cycles") or perf.get("estimated_whole_soc_cycles")) is not None else None,
                              measured_engine=row.get("engine") if row.get("hardware") else None,
                              model_sha256=perf.get("model_sha256"),
                              comparison_domain_sha256=row.get("comparison_domain_sha256") or perf.get("comparison_domain_sha256"),
                              scope_sha256=perf.get("scope_sha256"), capsule_sha256=perf.get("capsule_sha256"),
                              elf_sha256=row.get("elf_sha256") or perf.get("elf_sha256"),
                              complete_event_accounting=perf.get("complete_event_accounting"),
                              wall_s=row.get("duration_s") or row.get("evaluation_wall_s"),
                              changed_paths_count=row.get("changed_file_count"),
                              sample_round=row.get("round"), sample_slot=row.get("slot"),
                              timing_qualified=row.get("timing_qualified", row.get("physical_timing_qualified")),
                              notes="evidence="+str(row.get("evidence_path"))+"; phase="+str(row.get("phase"))+"; missing timestamps retained",
                              details_json=json.dumps({k: row.get(k) for k in ("model_comparison", "candidate_to_initial_ratio", "comparison_baseline_subject", "first_baseline_order_control_fraction", "within_one_percent_order_control", "reason", "qualification", "freeze", "tool", "operation", "aggregation") if row.get(k) is not None}, sort_keys=True))
                if kind == "provider_usage":
                    values.update({dest: row["tokens"][src] for src, dest in TOKEN_MAP.items()})
                    values.update(agent_role="helper" if row.get("parent_thread_id") else "lead", token_lower_bound=True)
                self.add(meta, kind, row.get("at"), path, f"line:{i}", **values)

    def write(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows.sort(key=lambda r: (r["run_id"], stamp(r["observed_at"]) or 0, r["record_type"], r["record_id"]))
        ids = [r["record_id"] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate event identities")
        path = self.out / "all_results.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        for name, data in [("runs.json", self.runs), ("sources.json", list(self.sources.values())), ("extraction_issues.json", self.issues)]:
            (self.out / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        receipt = dict(schema=SCHEMA, collection_cutoff_at=iso(self.cutoff),
                       collected_at=iso(datetime.now(timezone.utc).timestamp()),
                       rows=len(self.rows), by_record_type=dict(Counter(r["record_type"] for r in self.rows)),
                       by_run=dict(Counter(r["experiment"] for r in self.rows)),
                       csv_sha256=digest(path.read_bytes()), sources=len(self.sources), issues=len(self.issues),
                       authoring_or_simulation_started=False)
        (self.out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, sort_keys=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cutoff", help="fixed UTC analysis cutoff; defaults to process start")
    ap.add_argument("--experiments", nargs="*")
    args = ap.parse_args()
    task, out = args.task.resolve(), args.out.resolve()
    if not out.is_relative_to(task):
        raise ValueError("analysis output must remain inside the task")
    cutoff = stamp(args.cutoff) if args.cutoff else datetime.now(timezone.utc).timestamp()
    if cutoff is None:
        raise ValueError("invalid cutoff timestamp")
    c = Collector(task, out, cutoff)
    registry = c.read(c.repo / "examples/gemmini/phase1/studies/kernel_abstractions/configurations/20261028/registry.json")
    for entry in registry["records"]:
        if args.experiments and entry["name"] not in args.experiments:
            continue
        meta = c.metadata(entry)
        print("collecting " + meta["experiment"], flush=True)
        c.grades(meta)
        c.requests(meta)
        c.tokens(meta)
        c.snapshots(meta)
        c.tools(meta)
    if not args.experiments:
        c.model_health()
        c.small_studies()
    c.write()


if __name__ == "__main__":
    main()
