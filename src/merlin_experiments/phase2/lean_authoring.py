"""Explicit unqualified compiler exploration using the existing author transport.

No historical qualification, pass factory, source-effect or timing capability
is constructed here. The operator supplies the numerical/count evaluator. Its
complete outputs and estimates remain exploratory until separate qualification.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.target_experiment import load_target_experiment

from . import agent_workspace as AW
from . import broker as PB
from . import codex_author_runtime as CLIENT
from . import normal_resource_launch as NR
from .broker_policy import ActionAdmission
from .contracts import StageGateError, canonical_json, mapping_file, write_json
from .edit_authority import FrozenEditAuthority
from .retained_revisions import RetainedRevisionStore, executing_revisions, source_inventory


def _codex_binary(config):
    """Resolve an explicit executable or the worker's PATH before runtime pinning."""
    selected = config.get("codex_binary", "codex")
    if not isinstance(selected, (str, Path)) or not str(selected).strip():
        raise StageGateError("codex_binary must name an executable")
    resolved = shutil.which(str(selected))
    if resolved is None:
        raise StageGateError("Codex executable unavailable; set codex_binary or install codex on PATH")
    return Path(resolved).resolve(strict=True)

PROTOCOL = "merlin.phase2.lean_native_exploration.v1"
WORKFLOW = "lean-native-exploration-v1"
DISCLAIMER = dict(qualification="UNQUALIFIED", source_effects="PENDING", formal_admission=False,
                  promotion_authority="NONE", roofline_convergence=False)


def _epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _source_view(members, destination):
    """Expose only operator-selected public IR and shape metadata, never capsules."""
    destination.mkdir()
    rows, public = [], []
    for member in members:
        name = member["name"]
        if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
            raise StageGateError("lean exploration needs distinct simple public member names")
        path = Path(member["public_source_interface"])
        if path.is_symlink() or path.resolve(strict=True) != path or path.stat().st_size > 8 << 20:
            raise StageGateError("lean public IR must be a bounded canonical source leaf")
        data = path.read_bytes()
        relative = name + ".mlir"
        (destination / relative).write_bytes(data)
        rows.append(dict(path=relative, sha256=hashlib.sha256(data).hexdigest(), n_bytes=len(data)))
        public.append(dict(name=name, source=relative, source_sha256=rows[-1]["sha256"],
                           compute_floor_cycles=member.get("compute_floor_cycles"),
                           logical_payload=member.get("logical_payload")))
    if len({r["name"] for r in public}) != len(public) or not public:
        raise StageGateError("lean exploration requires a nonempty unique public roster")
    manifest = canonical_json(dict(schema=PROTOCOL, members=public, **DISCLAIMER))
    (destination / "members.json").write_bytes(manifest)
    rows.append(dict(path="members.json", sha256=hashlib.sha256(manifest).hexdigest(), n_bytes=len(manifest)))
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update((row["path"] + "\0" + row["sha256"] + "\0" + str(row["n_bytes"]) + "\n").encode())
    content = aggregate.hexdigest()
    document = dict(schema_version=1, files=rows, content_sha256=content,
                    n_files=len(rows), n_bytes=sum(r["n_bytes"] for r in rows), **DISCLAIMER)
    marker = destination / "agent_input_manifest.json"
    payload = canonical_json(document)
    marker.write_bytes(payload)
    for path in destination.iterdir():
        path.chmod(0o444)
    destination.chmod(0o555)
    return AW.AgentInputSnapshot(destination, marker, hashlib.sha256(payload).hexdigest(),
                                 content, len(rows), document["n_bytes"])


def _public_arm(arm, floor, expected_model):
    """Publish aggregate counts, never private readback, native pins or operands."""
    arm = arm if type(arm) is dict else {}
    counts = arm.get("count_observation") or {}
    value = dict(status=arm.get("status", "UNKNOWN"), complete_outputs_checked=False,
                 estimated_cycles=None, attainment_fraction=None, count_status=counts.get("status", "UNKNOWN"))
    numerical = arm.get("status") == "numeric_pass" and arm.get("complete_outputs_checked") is True
    value["complete_outputs_checked"] = numerical
    for key in ("failure_category", "public_failure_summary"):
        if isinstance(arm.get(key), str):
            value[key] = arm[key][:2000]
    cycles = counts.get("estimated_cycles")
    value["raw_estimated_cycles"] = cycles
    valid = (numerical and counts.get("status") == "observed"
             and counts.get("native_outputs_complete") is True and counts.get("feature_complete") is True
             and counts.get("complete_counts") is True and counts.get("timing_complete") is True
             and counts.get("unresolved_events") == [] and counts.get("model_sha256") == expected_model
             and type(cycles) in (int, float) and math.isfinite(cycles) and cycles > 0
             and type(floor) in (int, float) and math.isfinite(floor) and 0 < floor <= cycles
             and type(counts.get("attainment_fraction")) in (int, float)
             and math.isclose(counts["attainment_fraction"], floor / cycles, rel_tol=1e-12)
             and counts.get("comparison_consistent") is not False
             and isinstance(counts.get("scope_sha256"), str) and len(counts["scope_sha256"]) == 64)
    if valid:
        value.update(estimated_cycles=cycles, attainment_fraction=floor / cycles,
                     model_sha256=counts["model_sha256"], scope_sha256=counts["scope_sha256"])
    for key in ("feature_counts", "complete_counts", "timing_complete", "unresolved_events"):
        if key in counts:
            value[key] = counts[key]
    nested = counts.get("count_observation")
    if "feature_counts" not in value and type(nested) is dict and type(nested.get("feature_counts")) is dict:
        value["feature_counts"] = nested["feature_counts"]
    return {**value, **DISCLAIMER}


class LeanWorkflow:
    """Bounded asynchronous jobs and retained sources; no qualification authority."""

    workflow_id = WORKFLOW

    def __init__(self, config, *, candidate, target, receipt, store, baseline, evaluate_candidate):
        self.config, self.candidate, self.target_experiment, self.receipt_path = config, candidate, target, receipt
        self.store, self.baseline, self.evaluate_candidate = store, baseline, evaluate_candidate
        self.members = {m["name"]: dict(m, id=m.get("id", m["name"]),
            interface_mlir=m.get("interface_mlir", m.get("source_interface")),
            target_descriptor=m.get("target_descriptor", m.get("target_experiment")),
            floor_cycles=m.get("floor_cycles", m.get("compute_floor_cycles")),
            rtl_facts=m.get("rtl_facts", config.get("rtl_facts")),
            board_config=m.get("board_config", config.get("board_config")),
            target_fraction=m.get("target_fraction", config.get("target_fraction", .3))) for m in config["members"]}
        self.root = Path(config["output"]) / "jobs"
        self.root.mkdir()
        self.beam = config.get("beam", config.get("beam_width", 3))
        if type(self.beam) is not int or not 1 <= self.beam <= 3:
            raise StageGateError("lean exploration beam must be between one and three")
        self.executor = ThreadPoolExecutor(max_workers=self.beam, thread_name_prefix="lean-native")
        self.lock, self.jobs, self.latest, self.finished = threading.RLock(), {}, baseline, False
        self.closed = False
        self.evaluation_pool = self.search_coordinator = self
        self._compiler_pass_selection = None

    @staticmethod
    def actions():
        declarations = (
            ("freeze", ("parent_revision", "hypothesis"), "Retain compiler source bytes before evaluation"),
            ("evaluate-selected", ("revision_id", "members"), "Queue complete numerical and fixed-count checks"),
            ("status", ("job_id",), "Read jobs and lowest comparable exploratory roofline priorities; job_id=all"),
            ("finish", ("revision_id",), "Drain and retain chosen source only after host stopping condition"),
        )
        return tuple(PB.BrokerAction(name, ("__lean_host_action__", *["{" + p + "}" for p in args]),
                                    args, purpose, False) for name, args, purpose in declarations)

    def validate_actions(self, actions):
        if set(actions) != {a.name for a in self.actions()}:
            raise StageGateError("lean author broker action roster changed")

    def admission(self, _name):
        return ActionAdmission(None, True, "lean action budget exhausted", "lean deadline exhausted")

    def _collect(self):
        for job in self.jobs.values():
            future = job["future"]
            if job["state"] != "running" or not future.done():
                continue
            try:
                result = future.result()
                if type(result) is not dict or type(result.get("members")) is not list:
                    raise StageGateError("lean evaluator omitted its paired member rows")
                if [r.get("id") for r in result["members"]] != job["members"]:
                    raise StageGateError("lean evaluator changed the selected member roster")
                public = []
                for row in result["members"]:
                    member = self.members[row["id"]]
                    public.append(dict(id=row["id"], **{
                        arm: _public_arm(row.get(arm), member.get("compute_floor_cycles"),
                                         self.config["frozen_model_sha256"])
                        for arm in ("baseline", "candidate")
                    }))
                write_json(job["root"] / "private-result.json", result)
                job.update(state="completed", result=dict(members=public, **DISCLAIMER))
            except Exception as error:
                job.update(state="refused", error_type=type(error).__name__)
                write_json(job["root"] / "private-refusal.json", dict(type=type(error).__name__, detail=str(error)))
            job["elapsed_seconds"] = time.monotonic() - job["started"]
            write_json(job["root"] / "public-result.json", self._job_view(job))

    @staticmethod
    def _job_view(job):
        return {k: v for k, v in job.items() if k not in {"future", "root", "started"}}

    def _priorities(self):
        current = {name: [] for name in self.members}
        baseline = {name: [] for name in self.members}
        for job in self.jobs.values():
            if job["state"] != "completed":
                continue
            for row in job["result"]["members"]:
                if row["baseline"].get("attainment_fraction") is not None:
                    baseline[row["id"]].append(row["baseline"]["attainment_fraction"])
                if (job["revision_id"] == self.latest.revision_id
                        and row["candidate"].get("attainment_fraction") is not None):
                    current[row["id"]].append(row["candidate"]["attainment_fraction"])
        rows = []
        for name in self.members:
            values = current[name] or baseline[name]
            rows.append(dict(id=name, attainment_fraction=min(values) if values else None,
                             status="estimated" if values else "UNKNOWN",
                             observation_origin="current_candidate" if current[name] else
                             "original_control" if baseline[name] else "UNKNOWN"))
        return sorted(rows, key=lambda r: (r["attainment_fraction"] is None, r["attainment_fraction"] or 0, r["id"]))

    def _target_revision(self):
        for revision in {j["revision_id"] for j in self.jobs.values()}:
            rows = {name: [] for name in self.members}
            for job in self.jobs.values():
                if job["state"] == "completed" and job["revision_id"] == revision:
                    for row in job["result"]["members"]:
                        rows[row["id"]].append(row["candidate"]["attainment_fraction"])
            if all(values and all(v is not None and v >= self.config.get("target_fraction", .3) for v in values)
                   for values in rows.values()):
                return revision
        return None

    def stop_reason(self):
        if time.time() >= _epoch(self.config["deadline_utc"]):
            return "original campaign deadline reached"
        if time.time() >= _epoch(self.config["admission_cutoff_utc"]):
            return "original admission cutoff reached"
        if self._target_revision() is not None:
            return "estimated_structural_target_met; qualification remains deferred"
        return None

    def status(self, job_id="all"):
        with self.lock:
            self._collect()
            if job_id != "all":
                if job_id not in self.jobs:
                    raise StageGateError("unknown lean job identity")
                return self._job_view(self.jobs[job_id])
            return dict(schema=PROTOCOL, jobs=[self._job_view(j) for j in self.jobs.values()],
                        latest_revision=self.latest.public_record(), priorities=self._priorities(),
                        stop_reason=self.stop_reason(), target_revision=self._target_revision(), **DISCLAIMER)

    def _evaluate(self, revision, selected, root, timeout):
        with executing_revisions(revision, self.baseline):
            revision.verify()
            result = self.evaluate_candidate(
                candidate_source=revision.source, baseline_source=self.baseline.source,
                selected_members=tuple(self.members[n] for n in selected), work_root=root / "evaluation",
                timeout_s=timeout, baseline_cache_root=Path(self.config["output"]) / "baseline-cache",
            )
            revision.verify()
            return result

    def execute(self, _request, action_name, bindings, _index, _timeout, started):
        with self.lock:
            self._collect()
            if action_name == "freeze":
                parent = bindings["parent_revision"]
                self.latest = self.store.freeze(self.candidate, parent_revision=None if parent == "none" else parent,
                                                options=dict(hypothesis=bindings["hypothesis"]))
                result = self.latest.public_record()
            elif action_name == "evaluate-selected":
                if self.stop_reason() is not None:
                    raise StageGateError("lean exploration admission is closed")
                if len(self.jobs) >= self.config.get("max_jobs", 128):
                    raise StageGateError("lean exploration exhausted its declared job budget")
                if sum(j["state"] == "running" for j in self.jobs.values()) >= self.beam:
                    result = dict(status="BUSY", reason="all bounded evaluation slots are active")
                else:
                    selected = bindings["members"].split(",")
                    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(self.members):
                        raise StageGateError("lean evaluation requires explicit original public member names")
                    revision = self.store.get(bindings["revision_id"])
                    identity = uuid.uuid4().hex
                    root = self.root / identity
                    root.mkdir()
                    timeout = min(self.config.get("job_timeout_seconds", 2400),
                                  _epoch(self.config["deadline_utc"]) - time.time() - 10)
                    job = dict(job_id=identity, revision_id=revision.revision_id, source_sha256=revision.source_sha256,
                               members=selected, state="running", root=root, started=time.monotonic(), result=None)
                    job["future"] = self.executor.submit(self._evaluate, revision, selected, root, timeout)
                    self.jobs[identity] = job
                    result = self._job_view(job)
            elif action_name == "status":
                result = self.status(bindings["job_id"])
            elif action_name == "finish":
                if self.stop_reason() is None:
                    raise StageGateError("lean exploration remains open; continue the automatic search")
                self.latest = self.store.get(bindings["revision_id"])
                self.finished = True
                result = dict(selected=self.latest.public_record(), stop_reason=self.stop_reason(), **DISCLAIMER)
            else:
                raise StageGateError("lean broker refuses an undeclared action")
        return dict(returncode=0, stdout=canonical_json(result).decode() + "\n", stderr="",
                    elapsed_s=time.monotonic() - started), result

    def close(self):
        if self.closed:
            return
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            self._collect()
            self.closed = True

    def selection(self):
        """Choose exact retained bytes after drain; partial evidence stays partial."""
        target = self._target_revision()
        if self.finished or target is not None:
            revision = self.latest if self.finished else self.store.get(target)
            return revision, "author selected" if self.finished else "estimated_structural_target_met"
        scored, numerically_failed = {}, set()
        for job in self.jobs.values():
            if job["state"] != "completed":
                continue
            for row in job["result"]["members"]:
                candidate = row["candidate"]
                if candidate["complete_outputs_checked"]:
                    member_values = scored.setdefault(job["revision_id"], {})
                    member_values.setdefault(row["id"], []).append(candidate["attainment_fraction"])
                else:
                    numerically_failed.add(job["revision_id"])
        for identity in numerically_failed:
            scored.pop(identity, None)
        if not scored:
            return self.baseline, "no completed candidate evidence; original control retained"
        ranked = []
        for identity, members in scored.items():
            estimates = [min(v for v in values if v is not None) for values in members.values()
                         if values and all(v is not None for v in values)]
            ranked.append((len(members) == len(self.members), len(estimates), min(estimates, default=0), identity))
        return self.store.get(max(ranked)[-1]), "partial exploratory selection; unmeasured obligations remain pending"


def run_lean_authoring(config, *, evaluate_candidate, sandbox_inputs=None):
    """Start one existing continuous Codex session without qualification replay."""
    from merlin.common.cpu_pool import current_thread_owns_cpu_lease, grading_cpus, shared_compute_enabled
    from merlin_experiments.phase1.providers import codex_agent

    if type(config) is not dict or config.get("schema") != PROTOCOL or not callable(evaluate_candidate):
        raise StageGateError("lean authoring requires its explicit operator configuration and evaluator")
    selected_cpus = config.get("maximum_cpu_ids", list(range(24, 32)))
    if (not shared_compute_enabled() or not set(selected_cpus) <= set(range(24, 32))
            or not set(selected_cpus) <= set(grading_cpus()) or current_thread_owns_cpu_lease()):
        raise StageGateError("lean author launch needs shared assigned CPUs without a parent metadata lease")
    if _epoch(config["admission_cutoff_utc"]) >= _epoch(config["deadline_utc"]):
        raise StageGateError("lean admission cutoff must precede the original deadline")
    if time.time() >= _epoch(config["admission_cutoff_utc"]):
        raise StageGateError("lean authoring cannot start after admission closes")
    output = Path(config["output"])
    stage = output / "author"
    stage.mkdir()
    workspace = stage / "workspace"
    workspace.mkdir()
    candidate = workspace / "submission"
    shutil.copytree(Path(config["baseline"]), candidate)
    for path in (candidate, *candidate.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644 | (path.stat().st_mode & 0o111))
    authority_root = stage / "edit-authority"
    authority_root.mkdir()
    authority = FrozenEditAuthority(authority_root, guidance_contract=Path(config["contract_root"]))
    authority.freeze(candidate, mapping_file(Path(config["edit_contract"])), has_iterations=False)
    store = RetainedRevisionStore(stage / "retained", edit_authority=authority, max_bytes=32 << 20,
                                  max_files=8192, max_total_bytes=256 << 20, max_revisions=224)
    baseline = store.freeze(candidate, options=dict(hypothesis="immutable original control"))
    agent_inputs = _source_view(config["members"], stage / "public-inputs")
    descriptor = config.get("target_descriptor") or config["members"][0]["target_experiment"]
    target = load_target_experiment(descriptor)
    control = stage / "control"
    control.mkdir()
    workflow = LeanWorkflow(config, candidate=candidate, target=target, receipt=control / "receipts.jsonl",
                            store=store, baseline=baseline, evaluate_candidate=evaluate_candidate)
    inner = AW.inner_execution_policy(target, candidate, agent_inputs, functional_base=baseline.source,
                                      inputs=sandbox_inputs)
    maximum = config.get("maximum_tool_calls", 224)
    if type(maximum) is not int or not 1 <= maximum <= 224:
        raise StageGateError("lean broker calls must retain the bounded file transport limit")
    deadline = time.monotonic() + (_epoch(config["deadline_utc"]) - time.time())
    broker = PB.Broker(inner, target, candidate, workflow.actions(), workflow.receipt_path,
                       deadline=deadline, workflow=workflow, max_calls=maximum, max_tool_seconds=120)
    prompt = (
        "Implement reusable compiler optimizations in submission. This is explicitly UNQUALIFIED exploration. "
        "Historical qualification, capability bindings and source effects are deferred; never claim them PASS. "
        "Use python3 /perf-control/perf_tool.py ACTION NAME=VALUE. Actions: freeze parent_revision=ID|none "
        "hypothesis=TEXT; evaluate-selected revision_id=ID members=NAME[,NAME]; status job_id=all|ID; "
        "finish revision_id=ID. Freeze before evaluation. Jobs are asynchronous: edit while they run, "
        "and poll status for complete full-output numerical checks, aggregate event counts, fixed-model cycles "
        "and partial ideal-compute roofline attainment. Prioritize the lowest comparable attainment; missing "
        "or refused counts remain UNKNOWN. Preserve full K, scale-one signed-i8 saturation, raw-i32 controls, "
        "tails/capacity/connected-pattern guards and no hardware FSM commands. Generalize across all source IR "
        "in /perf-corpus/members.json. Keep working until host cutoff/deadline or a complete exploratory target; "
        "finish before that is refused. Final qualification and FireSim follow a compiler freeze. "
        f"Initial retained control: {baseline.revision_id}.\n" + config.get("guidance", "")
    )
    (workspace / "TASK.md").write_text(prompt)
    (stage / "prompt.txt").write_text(prompt)
    binary = _codex_binary(config)

    def policy(ws, binds=()):
        original = AW.outer_codex_policy(ws, agent_inputs, binds, target, functional_base=baseline.source,
                                         control_dir=control, inputs=sandbox_inputs)
        extra = []
        private = [output, *(Path(m["capsule_dir"]) for m in config["members"])]
        public_sources = {Path(m["public_source_interface"]).parent for m in config["members"]}
        for declared in config.get("public_code_roots", []):
            path = Path(declared)
            if (path.resolve(strict=True) != path or path.is_symlink()
                    or path not in public_sources
                    and any(path == p or path in p.parents or p in path.parents for p in private)):
                raise StageGateError("lean public code grant overlaps private evaluator or output state")
            extra.extend(["--ro-bind", str(path), str(path)])
        return replace(original, argv=tuple([*original.argv, *extra]))

    started, code, transcript, error = time.monotonic(), None, None, None
    scratch = ExitStack()
    try:
        runtime = CLIENT.select_runtime(binary=binary, home_root=stage / "codex-homes", workspace=workspace,
                                        policy=policy(workspace))

        def sandbox(command, ws, _bundle, extra_binds=None):
            selected = policy(ws, extra_binds or ())
            CLIENT.verify_policy(runtime, workspace=ws, policy=policy(ws))
            argv = scratch.enter_context(NR.bounded_normal_scratch(selected.argv, owner=stage / "author-scratch"))
            return BW.compose_command(list(argv), " bash -c '" + command.replace("'", "'\\''") + "'", ws)

        def stop():
            if len(broker.calls) >= maximum:
                return "lean broker call budget exhausted"
            with workflow.lock:
                workflow._collect()
                return workflow.stop_reason()

        write_json(stage / "status.json", dict(stage="launching_actual_codex", **DISCLAIMER))
        with broker.serving_author(transport="files.v1", workspace=workspace, control_dir=control,
                                   tool_timeout_s=120, actions=workflow.actions()):
            code, transcript = codex_agent.run_round(
                workspace, stage, config.get("model", "gpt-6.1-sol"), {}, target, "bwrap", 0,
                max(1, int(deadline - time.monotonic())), effort=config.get("effort", "high"),
                prompt=prompt, effective_model=config.get("model", "gpt-6.1-sol"), sandbox_command=sandbox,
                continue_session=True, continuation_stop=stop,
                continuation_prompt="Continue the same compiler search. Read status, use lowest estimated attainment, "
                                    "author/freeze/evaluate reusable revisions. Full qualification remains PENDING.",
                max_concurrent_subagents=1, **runtime.round_kwargs(),
            )
    except BaseException as caught:
        error = caught
    finally:
        try:
            workflow.close()
        finally:
            scratch.close()
    selected, selection_reason = workflow.selection()
    selected.verify()
    frozen = stage / "frozen-compiler"
    shutil.copytree(selected.source, frozen)
    actual = source_inventory(frozen, max_bytes=32 << 20, max_files=8192, readonly=True)
    if actual["sha256"] != selected.source_sha256:
        raise StageGateError("lean final compiler differs from its exact retained source")
    result = dict(schema=PROTOCOL, returncode=code, transcript=str(transcript),
                  elapsed_seconds=time.monotonic() - started, final_status=workflow.status(),
                  selected_revision=selected.public_record(), frozen_compiler=str(frozen),
                  selection_reason=selection_reason, error_type=type(error).__name__ if error else None, **DISCLAIMER)
    write_json(stage / "result.json", result)
    if error is not None:
        raise error
    return result
