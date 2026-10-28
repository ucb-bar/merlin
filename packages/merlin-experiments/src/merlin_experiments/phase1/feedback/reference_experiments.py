"""Explicit operator-selected reference experiments in ordinary author feedback.

Only the pinned host invocation selects a packet. Workspace files, candidate
requests, environment variables and claimed task completion carry no selection
authority. Advice never changes grading, promotion, or reference access policy.

A selection may withhold its prior tasks (``tasks=False``): the selection still
grants the read-only reference analysis endpoint (:mod:`.reference_analysis`)
and its TASK.md guidance, but no ``REFERENCE_EXPERIMENTS.json`` is staged and no
prior task, hypothesis or request reaches feedback or the idle dispatcher -- for
a campaign whose packet was derived against another compiler.
"""

from __future__ import annotations

import argparse
import contextvars
import copy
import fcntl
import hashlib
import json
import os
import shlex
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from merlin.verify.lean_contract import document_digest

SCHEMA = "merlin.reference_experiment_selection.v1"
PACKET_SCHEMA = "merlin.reference_experiment_packet.v1"
SHARDS_SCHEMA = "merlin.reference_experiment_shards.v1"
FEEDBACK_SCHEMA = "merlin.reference_experiment_feedback.v1"
KEY = "reference_experiments"
_ACTIVE = contextvars.ContextVar("operator_reference_experiment_selection", default=None)
_MAX_BYTES = 2 << 20
_MAX_SHARDS = 128
_MAX_SHARD_BYTES = 64 << 20
_CONTEXT_FIELDS = ("compiler_sha256", "source_sha256", "environment_sha256", "reference_sha256")


@dataclass(frozen=True)
class _Selection:
    path: Path
    sha256: str
    process: int
    tasks: bool = True


def _sha(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _current():
    selected = _ACTIVE.get()
    if selected is not None and selected.process != os.getpid():
        raise ValueError("reference feedback needs an explicit selection in this process")
    return selected


@contextmanager
def invocation(path=None, sha256=None, *, tasks=True):
    """Select immutable operator input; absent arguments explicitly keep the default inert.

    ``tasks=False`` keeps the selection's analysis grant and withholds its prior tasks."""
    if (path is None) != (sha256 is None):
        raise ValueError("reference experiments require an explicit selection path and SHA256")
    if path is not None and (not Path(path).is_absolute() or not _sha(sha256)):
        raise ValueError("reference experiment selection requires an absolute path and exact SHA256")
    if type(tasks) is not bool or (path is None and not tasks):
        raise ValueError("withheld reference tasks require an explicit selection")
    selected = None if path is None else _Selection(Path(path), sha256, os.getpid(), tasks)
    if selected is not None:
        _load(selected)
    prior = _ACTIVE.set(selected)
    try:
        yield
    except BaseException:
        # Preserve the original execution failure; secondary custody failures
        # must not conceal it. Successful runs recheck their selected input.
        raise
    else:
        if selected is not None:
            _load(selected)
    finally:
        _ACTIVE.reset(prior)


def child_arguments():
    """Private host-child CLI transport; never constructed from author request fields."""
    selected = _current()
    if selected is None:
        return ()
    return ("--reference-experiments-selection", str(selected.path),
            "--reference-experiments-selection-sha256", selected.sha256,
            *(() if selected.tasks else ("--reference-experiments-selection-tasks", "off")))


def selected_child(function):
    """Consume only the explicit original host child's selected packet arguments."""
    @wraps(function)
    def call(argv=None, *args, **kwargs):
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("--reference-experiments-selection", type=Path)
        parser.add_argument("--reference-experiments-selection-sha256")
        parser.add_argument("--reference-experiments-selection-tasks", choices=("off",))
        values, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
        path, sha = values.reference_experiments_selection, values.reference_experiments_selection_sha256
        tasks = values.reference_experiments_selection_tasks is None
        if (path is None) != (sha is None) or (path is None and not tasks):
            raise ValueError("reference feedback child requires its exact path and SHA256")
        active = _current()
        if path is None:
            return function(argv, *args, **kwargs)
        if active is not None:
            if (path, sha, tasks) != (active.path, active.sha256, active.tasks):
                raise ValueError("reference feedback child cannot replace the selected operator input")
            return function(remaining, *args, **kwargs)
        with invocation(path, sha, tasks=tasks):
            return function(remaining, *args, **kwargs)
    return call


def background_target(function):
    """Carry only this original operator selection into an existing worker thread."""
    selected = _current()
    if selected is None:
        return function, None
    failures = []
    def body():
        try:
            with invocation(selected.path, selected.sha256, tasks=selected.tasks):
                return function()
        except BaseException as error:
            failures.append(error)
    def joined():
        _current()
        _load(selected)
        if failures:
            raise RuntimeError("reference experiment background owner failed") from failures[0]
    return body, joined


def _read(path, sha, *, limit=_MAX_BYTES):
    path = Path(path)
    if (not _sha(sha) or not path.is_absolute() or path.is_symlink() or path.resolve() != path
            or any(parent.is_symlink() for parent in path.parents) or not path.is_file()
            or path.stat().st_size > limit):
        raise ValueError("selected_input_unavailable")
    raw = path.read_bytes()
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("selected_input_changed")
    def pairs(values):
        record = {}
        for key, value in values:
            if key in record:
                raise ValueError("duplicate_packet_field")
            record[key] = value
        return record
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite_packet_field")))


def _load_single(selected, selection=None):
    selection = _read(selected.path, selected.sha256) if selection is None else selection
    if (not isinstance(selection, dict) or selection.get("schema") != SCHEMA
            or not {"packet", "current_context"} <= set(selection)):
        raise ValueError("unsupported_reference_selection")
    pin, context = selection["packet"], selection["current_context"]
    if (not isinstance(pin, dict) or set(pin) != {"path", "sha256"} or not isinstance(context, dict)
            or any(not _sha(context.get(key)) for key in _CONTEXT_FIELDS)):
        raise ValueError("incomplete_reference_context")
    evaluator = context.get("evaluator")
    if (not isinstance(evaluator, dict) or not evaluator.get("identity") or not _sha(evaluator.get("sha256"))):
        raise ValueError("incomplete_evaluator_context")
    packet = _read(pin["path"], pin["sha256"])
    if (not isinstance(packet, dict) or packet.get("schema") != PACKET_SCHEMA
            or packet.get("identity") != document_digest(
                {key: value for key, value in packet.items() if key != "identity"})
            or document_digest(packet.get("context")) != document_digest(context)):
        raise ValueError("packet_identity_or_context_changed")
    if context.get("visibility") not in ("reference_guided", "public"):
        raise ValueError("packet_not_selected_for_author_visibility")
    execution = copy.deepcopy(context)
    execution["evaluator"].pop("trusted_result_sha256", None)
    if packet.get("execution_context_identity") != document_digest(execution):
        raise ValueError("execution_context_identity_changed")
    for item in selection.get("context_pins", ()):
        if not isinstance(item, dict):
            raise ValueError("context_pin_unavailable")
        # Context leaves need not be JSON. Byte custody is rechecked before any
        # selected task text can reach the author.
        path = Path(item["path"])
        if (not path.is_absolute() or path.is_symlink() or path.resolve() != path or not path.is_file()
                or path.stat().st_size > 256 << 20):
            raise ValueError("context_input_changed")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError("context_input_changed")
    tasks = packet.get("tasks")
    if not isinstance(tasks, list) or len(tasks) > 16 or any(not isinstance(task, dict) for task in tasks):
        raise ValueError("invalid_reference_task_roster")
    task_ids, request_ids = set(), set()
    for task in tasks:
        task_id = task.get("task_id")
        if (not isinstance(task_id, str) or task_id in task_ids or not isinstance(task.get("acceptance"), dict)
                or task.get("status") not in ("ready", "blocked", "unranked", "rejected", "resolved")
                or not isinstance(task.get("goals", []), list)
                or any(not isinstance(goal, str) for goal in task.get("goals", []))):
            raise ValueError("invalid_reference_task_contract")
        task_ids.add(task_id)
        for field in ("evidence", "hypotheses", "prior_attempts"):
            if (not isinstance(task.get(field, []), list)
                    or any(not isinstance(row, dict) for row in task.get(field, []))):
                raise ValueError("invalid_reference_task_evidence")
        requests = task.get("requests", [])
        if not isinstance(requests, list):
            raise ValueError("invalid_reference_requests")
        for request in requests:
            if (not isinstance(request, dict) or not isinstance(request.get("request_id"), str)
                    or request["request_id"] in request_ids
                    or request.get("context_identity") != packet["execution_context_identity"]
                    or not isinstance(request.get("settings"), dict)
                    or request.get("settings_sha256") != document_digest(request["settings"])
                    or not isinstance(request.get("argv"), list) or not 1 <= len(request["argv"]) <= 64
                    or any(not isinstance(value, str) for value in request["argv"])):
                raise ValueError("invalid_reference_request_contract")
            request_ids.add(request["request_id"])
            if len(request_ids) > 32:
                raise ValueError("reference_request_budget_exceeded")
    return packet, context


def _load(selected):
    """Read one selector or the existing pinned, complete bounded-packet index."""
    selection = _read(selected.path, selected.sha256)
    if not isinstance(selection, dict) or selection.get("schema") != SHARDS_SCHEMA:
        return _load_single(selected, selection)
    shards, total = selection.get("shards"), selection.get("tasks")
    if (selection.get("complete") is not True or type(total) is not int or not 0 <= total <= 16 * _MAX_SHARDS
            or not isinstance(shards, list) or not 1 <= len(shards) <= _MAX_SHARDS):
        raise ValueError("incomplete_reference_shard_index")
    packets, tasks, origins = [], [], {}
    packet_ids, task_ids, request_ids, selectors = set(), set(), set(), set()
    context, offset, byte_count = None, 0, 0
    for shard in shards:
        if (not isinstance(shard, dict) or set(shard) != {"packet", "selection", "selection_sha256"}
                or not isinstance(shard["selection"], str) or not isinstance(shard["packet"], str)):
            raise ValueError("invalid_reference_shard_pin")
        child = _Selection(Path(shard["selection"]), shard["selection_sha256"], selected.process)
        if child.path in selectors:
            raise ValueError("duplicate_reference_shard_selection")
        selectors.add(child.path)
        single = _read(child.path, child.sha256)
        packet, current = _load_single(child, single)
        if single["packet"]["path"] != shard["packet"]:
            raise ValueError("reference_shard_packet_pin_changed")
        if context is not None and document_digest(current) != document_digest(context):
            raise ValueError("reference_shard_context_changed")
        context = current
        byte_count += child.path.stat().st_size + Path(shard["packet"]).stat().st_size
        if byte_count > _MAX_SHARD_BYTES:
            raise ValueError("reference_shard_byte_budget_exceeded")
        identity = packet["identity"]
        if identity in packet_ids:
            raise ValueError("duplicate_reference_shard_packet")
        packet_ids.add(identity)
        coverage = packet.get("coverage")
        end = offset + len(packet["tasks"])
        if ((not packet["tasks"] and (total != 0 or len(shards) != 1))
                or not isinstance(coverage, dict) or any(type(coverage.get(key)) is not int for key in
                                                ("selected_start", "selected_end", "total_diagnoses"))
                or coverage["selected_start"] != offset or coverage["selected_end"] != end
                or coverage["total_diagnoses"] != total or end > total
                or coverage.get("next_offset") != (end if end < total else None)):
            raise ValueError("reference_shard_coverage_incomplete")
        offset = end
        for task in packet["tasks"]:
            if task["task_id"] in task_ids:
                raise ValueError("duplicate_reference_shard_task")
            task_ids.add(task["task_id"])
            origins[task["task_id"]] = identity
            for request in task.get("requests", []):
                if request["request_id"] in request_ids:
                    raise ValueError("duplicate_reference_shard_request")
                request_ids.add(request["request_id"])
            tasks.append(task)
        packets.append(identity)
    if offset != total:
        raise ValueError("reference_shard_coverage_incomplete")
    # Host-only view, authenticated transitively by the pinned index and every
    # original bounded packet. No producer packet or task identity is rewritten.
    aggregate = dict(schema=SHARDS_SCHEMA, context=context, tasks=tasks,
                     execution_context_identity=packet["execution_context_identity"],
                     selected_index_sha256=selected.sha256, source_packet_identities=packets,
                     task_packet_identities=origins)
    aggregate["identity"] = document_digest(aggregate)
    return aggregate, context


def _task_packet_identity(packet, task_id):
    return packet.get("task_packet_identities", {}).get(task_id, packet["identity"])


def _request_limit(packet):
    return 32 * len(packet.get("source_packet_identities", [packet["identity"]]))


def _journal_limit(packet):
    return 16 << 20 if packet.get("schema") == SHARDS_SCHEMA else _MAX_BYTES


def _unknown(reason):
    return dict(schema=FEEDBACK_SCHEMA, status="unknown", advisory_only=True, qualification_issued=False,
                reason=reason, next_action=dict(route="acquire_new_evidence",
                    action="Rebuild the reference experiment packet from the current trusted compiler and evidence."),
                lines=["reference experiment (advisory): context unavailable; acquire current evidence.",
                       "No experiment is marked tested or complete from this packet."])


@contextmanager
def _journal(selected):
    """One host-private sibling of the original selector, never workspace output."""
    if selected.process != os.getpid():
        raise ValueError("foreign_reference_observation_owner")
    _load(selected)
    path = selected.path.parent / (".reference-experiments-" + selected.sha256 + ".json")
    lock = path.with_suffix(".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as stream:
        owner = os.fstat(stream.fileno())
        if owner.st_uid != os.geteuid() or owner.st_mode & 0o077:
            raise ValueError("reference_observation_store_not_private")
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield path


def _read_store(selected, packet, path):
    if not path.exists() and not path.is_symlink():
        return dict(schema="merlin.reference_experiment_observations.v1", selection_sha256=selected.sha256,
                    packet_identity=packet["identity"], context_identity=packet["execution_context_identity"],
                    outcomes={})
    if (path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077
            or path.stat().st_uid != os.geteuid() or path.stat().st_size > _journal_limit(packet)):
        raise ValueError("reference_observation_store_unavailable")
    raw = path.read_bytes()
    document = _read(path, hashlib.sha256(raw).hexdigest(), limit=_journal_limit(packet))
    if (not isinstance(document, dict) or document.get("schema") != "merlin.reference_experiment_observations.v1"
            or document.get("selection_sha256") != selected.sha256
            or document.get("packet_identity") != packet["identity"]
            or document.get("context_identity") != packet["execution_context_identity"]
            or document.get("identity") != document_digest({key: value for key, value in document.items()
                                                            if key != "identity"})):
        raise ValueError("reference_observation_store_changed")
    return document


def _read_outcomes(selected, packet, path):
    document = _read_store(selected, packet, path)
    outcomes = document.get("outcomes")
    if not isinstance(outcomes, dict) or len(outcomes) > _request_limit(packet):
        raise ValueError("reference_observation_roster_invalid")
    requests = {request["request_id"]: (task["task_id"], request)
                for task in packet["tasks"] for request in task.get("requests", [])}
    for key, outcome in outcomes.items():
        if not isinstance(outcome, dict) or key not in requests:
            raise ValueError("reference_observation_request_unknown")
        task_id, request = requests[key]
        expected = dict(request_id=key, task_id=task_id, packet_identity=_task_packet_identity(packet, task_id),
                        context_identity=packet["execution_context_identity"],
                        settings_sha256=request["settings_sha256"],
                        base_compiler_sha256=packet["context"]["compiler_sha256"])
        if (any(outcome.get(field) != value for field, value in expected.items())
                or outcome.get("task_complete") is not False or outcome.get("qualification_issued") is not False
                or outcome.get("mechanism_status") != "unknown" or not _sha(outcome.get("executed_compiler_sha256"))):
            raise ValueError("reference_observation_join_changed")
    return outcomes


def _outcomes(selected, packet):
    with _journal(selected) as path:
        return _read_outcomes(selected, packet, path)


def _record_outcome(selected, packet, result):
    with _journal(selected) as path:
        document = _read_store(selected, packet, path)
        outcomes = _read_outcomes(selected, packet, path)
        prior = outcomes.get(result["request_id"])
        if prior is not None and prior != result:
            raise ValueError("reference_observation_request_already_recorded")
        outcomes[result["request_id"]] = copy.deepcopy(result)
        document["outcomes"] = outcomes
        _write_store(path, document)


def _write_private(path, document):
    descriptor, temporary = tempfile.mkstemp(prefix=".reference-observations-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_store(path, document):
    document = {key: value for key, value in document.items() if key != "identity"}
    document["identity"] = document_digest(document)
    _write_private(path, document)


def _effective(selected, compiler_sha256):
    """Reopen a host-generated current packet without unwithholding prior tasks."""
    packet, context = _load(selected)
    with _journal(selected) as path:
        active = _read_store(selected, packet, path).get("current_analysis")
    if not active:
        return selected, packet, context
    if active.get("compiler_sha256") != compiler_sha256:
        return selected, packet, context
    pin = active.get("selection", {})
    current_path = Path(pin.get("path", ""))
    if (current_path.parent != selected.path.parent or not current_path.name.startswith(
            ".reference-analysis-" + selected.sha256 + "-") or not _sha(pin.get("sha256"))):
        raise ValueError("current_reference_analysis_selection_invalid")
    current = _Selection(current_path, pin["sha256"], os.getpid())
    generated, fresh = _load(current)
    if (fresh.get("compiler_sha256") != compiler_sha256
            or active.get("compiler_sha256") != compiler_sha256):
        return selected, packet, context
    if any(fresh.get(key) != value for key, value in context.items() if key not in ("compiler_sha256", "tuning")):
        raise ValueError("current_reference_analysis_context_changed")
    return current, generated, fresh


def _analysis_pins(report, workspace):
    """Reopen actual admitted analysis leaves, including candidate artifact inputs."""
    pins = {}

    def visit(value):
        if isinstance(value, dict):
            label, sha = value.get("path"), value.get("sha256")
            if isinstance(label, str) and _sha(sha) and "#" not in label:
                path = workspace / label
                if path.is_file():
                    if path.is_symlink() or path.resolve() != path or not path.is_relative_to(workspace):
                        raise ValueError("current_reference_analysis_input_invalid")
                    with path.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                            raise ValueError("current_reference_analysis_input_changed")
                    pins[str(path)] = dict(path=str(path), sha256=sha)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(report)
    if len(pins) > 64:
        raise ValueError("current_reference_analysis_pin_budget_exceeded")
    return list(pins.values())


def record_analysis(report, *, workspace, compiler_sha256):
    """Publish fresh broker analysis through the existing packet/observation owners.

    This entry is host-only. It refreshes compiler and declared tuning evidence,
    retaining the selected evaluator, public applicability and reference inputs.
    No reported setting, artifact claim or numerical result admits a mechanism.
    """
    selected = _current()
    if selected is None or not _sha(compiler_sha256):
        return None
    from merlin.compile.tuning_knobs import DECLARATION
    from merlin.mining.reconstruction_experiments import build_experiment_packet, workspace_tuning

    from ..brokers.selfcheck import _submission_digest

    packet, context = _load(selected)
    workspace = Path(workspace).absolute()
    submission = workspace / context.get("baseline", {}).get("submission", "submission")
    if _submission_digest(submission) != compiler_sha256:
        return dict(status="prior_only", reason="current_compiler_context_requires_rebinding", requests=[])
    fresh = copy.deepcopy(context)
    fresh["compiler_sha256"] = compiler_sha256
    if (submission / DECLARATION).is_file():
        fresh["tuning"] = workspace_tuning(submission)
    else:
        fresh["tuning"] = dict(current={}, domains={})
    generated = build_experiment_packet(report, context=fresh)
    if not generated["tasks"]:
        return generated
    pins = _analysis_pins(report, workspace)
    pins.extend((fresh.get("tuning") or {}).get("artifacts", []))
    original = _read(selected.path, selected.sha256)
    prefix = ".reference-analysis-" + selected.sha256 + "-" + document_digest(
        dict(packet=generated["identity"], inputs=pins))
    generated_path = selected.path.parent / (prefix + ".packet.json")
    selection_path = selected.path.parent / (prefix + ".selection.json")
    generated_sha = _write_private(generated_path, generated)
    selection_sha = _write_private(selection_path, dict(schema=SCHEMA,
        packet=dict(path=str(generated_path), sha256=generated_sha), current_context=fresh,
        context_pins=[*original.get("context_pins", []), *pins]))
    current = _Selection(selection_path, selection_sha, os.getpid())
    _load(current)
    with _journal(selected) as path:
        document = _read_store(selected, packet, path)
        document["current_analysis"] = dict(selection=dict(path=str(selection_path), sha256=selection_sha),
            compiler_sha256=compiler_sha256, report_identity=document_digest(report))
        _write_store(path, document)
    return generated


def _text(value, limit=220):
    return " ".join(str(value).split())[:limit]


def _attempt_text(value):
    if isinstance(value, dict) and value.get("schema") == "merlin.reference_experiment_observation.v1":
        outcome = {"pass": "passed", "fail": "failed"}.get(value.get("numerical_status"), "unknown")
        return (_text(value.get("request_id")) + ": numerical check " + outcome
                + "; emitted mechanism remains unresolved")
    return _text(value)


def _compact_prior_task(task):
    """Bounded author guidance; omit raw evidence values, settings, requests and outcomes."""
    def identifier(value, limit):
        # Identity fields remain exact or absent; shortening them could create
        # a different goal or a false provenance pointer.
        return value if isinstance(value, str) and len(value.encode()) <= limit else ""
    evidence = []
    for row in task.get("evidence", [])[:3]:
        status = row.get("status")
        item = dict(scope="prior_reference_context", current_status="unknown",
                    prior_status=status if status in ("observed", "inferred", "unknown") else "unknown",
                    pointer=identifier(row.get("pointer", ""), 240))
        if row.get("pointer") and not item["pointer"]:
            item["omitted_pointer_sha256"] = document_digest(row["pointer"])
        if _sha(row.get("sha256")):
            item["sha256"] = row["sha256"]
        evidence.append(item)
    hypotheses = [dict(scope="prior_reference_context", current_status="unknown", basis="inferred",
                       summary=_text(row.get("summary", "")),
                       hypothesis_id=identifier(row.get("hypothesis_id", ""), 160))
                  for row in task.get("hypotheses", [])[:2]]
    action = task.get("next_action") or {}
    if not isinstance(action, dict):
        action = {}
    goals = [goal for goal in task.get("goals", [])[:8] if identifier(goal, 160)]
    return dict(task_id=identifier(task["task_id"], 160), source_task_id_sha256=document_digest(task["task_id"]),
                scope="prior_reference_context", current_status="unverified", prior_task_status=task["status"],
                goals=goals, omitted_goal_count=len(task.get("goals", [])) - len(goals),
                summary=_text(task.get("summary", task["task_id"])), evidence=evidence, hypotheses=hypotheses,
                prior_action=dict(route=_text(action.get("route", ""), 160), action=_text(action.get("action", ""))))


def _advisory_tasks(packet):
    """Diverse construction guidance only; never reorder executable requests."""
    pending = [task for task in packet["tasks"] if task["status"] != "resolved"]
    resolved = [task for task in packet["tasks"] if task["status"] == "resolved"]
    if packet.get("schema") != SHARDS_SCHEMA:
        return pending + resolved
    result = []
    for pool in (pending, resolved):
        ordered = sorted(pool, key=lambda task: 0 if task.get("kind") == "adapter_gap"
                         else 1 if task.get("kind") == "representation_gap" else 2)
        seen, unique, repeated = set(), [], []
        for task in ordered:
            action = task.get("next_action") or {}
            if not isinstance(action, dict):
                action = {}
            signature = document_digest(dict(goals=sorted(task.get("goals", [])),
                route=action.get("route"), action=action.get("action")))
            (repeated if signature in seen else unique).append(task)
            seen.add(signature)
        result.extend([*unique, *repeated])
    return result


def _prior_guidance(packet, context, current_compiler_sha256):
    """Keep immutable reference intent useful without rebasing any execution evidence."""
    tasks = packet["tasks"]
    ranked = _advisory_tasks(packet)
    prior = [_compact_prior_task(task) for task in ranked[:4]]
    goals = list(dict.fromkeys(goal for task in prior for goal in task["goals"]))[:16]
    workflow = [dict(kind=kind, scope="current_compiler", status="required_new_evidence",
                     argv_template=["python3", "agent_selfcheck.py", "--submission", "submission",
                                    "--capsules", "{public_capsule}", *flags])
                for kind, flags in (("candidate_diagnostics", ["--candidate-diagnostics"]),
                                    ("compile_only", ["--compile-only"]), ("selfcheck", []))]
    lines = ["reference guidance (prior only; current evidence unknown): "
             + (prior[0]["summary"] if prior else "selected reference goals remain advisory.")]
    if goals:
        lines[0] += "; goals: " + ", ".join(_text(goal, 90) for goal in goals[:3])
    lines.append("required current evidence: use current library/scheduling APIs; run agent_selfcheck.py "
                 "--candidate-diagnostics, then --compile-only and a selfcheck for a declared public capsule; "
                 "use --reference-analysis inventory/lift/candidate to reanalyze visible inputs and new emitted bytes.")
    if prior and prior[0]["hypotheses"]:
        lines.append("prior hypothesis (unverified for current compiler): " + prior[0]["hypotheses"][0]["summary"])
    if prior and prior[0]["evidence"]:
        item = prior[0]["evidence"][0]
        lines.append("prior evidence: " + item["pointer"] + " (" + _text(item["prior_status"], 40)
                     + "; selected packet " + packet["identity"][:12] + ").")
    if prior and prior[0]["prior_action"]["action"]:
        lines.append("prior action to reassess with current APIs: " + prior[0]["prior_action"]["action"])
    lines.append("Current checks must establish source/fusion equivalence and emitted mechanisms; "
                 "performance requires fresh paired complete-region cycles and held-out shapes.")
    actual = current_compiler_sha256 if _sha(current_compiler_sha256) else None
    return dict(schema=FEEDBACK_SCHEMA, status="prior_reference_guidance", advisory_only=True,
                reason="current_compiler_differs_from_reference_context" if actual else "current_compiler_not_observed",
                current_evidence_status="unknown", current_compiler_sha256=actual,
                packet_identity=packet["identity"],
                prior_execution_context_identity=packet["execution_context_identity"],
                prior_context={key: context[key] for key in _CONTEXT_FIELDS},
                prior_tasks=prior, pending_goals=goals, total_prior_tasks=len(tasks),
                omitted_prior_tasks=len(tasks) - len(prior),
                observed=[], observed_requests=[], prior_attempts=[], required_selfchecks=[],
                mechanism_status="unknown", performance_status="unknown", inherited_results_applied=False,
                executable_requests_allowed=False, outcome_join_allowed=False, task_complete=False,
                qualification_issued=False, diagnostic_workflow=workflow,
                library_guidance=dict(inventory="OPTIMIZATION_SUGGESTIONS.json",
                    skill=".agents/skills/pass-library/SKILL.md",
                    scheduling_api="merlin.compile.scheduling.schedule_ir",
                    reference_analysis="python agent_selfcheck.py --reference-analysis inventory; "
                        "use lift with reference-kernels/FILE or candidate with "
                        "submission/build/CASE/kernel.dis (build with inspect-kernel --out first)",
                    action="Inspect current supported transformation APIs and express decisions through the scheduling "
                           "dialect; inspect the emitted kernel before claiming that a mechanism was applied."),
                next_action=dict(route="acquire_current_candidate_evidence",
                    action="Use the prior goals to guide current library transformations, resolve a capsule from the "
                           "current public inventory, then collect diagnostics, compilation and numerical selfcheck "
                           "evidence for the current compiler.",
                    capsule_binding="current declared public inventory; no inherited experiment request"), lines=lines)


def project(document=None, *, current_compiler_sha256=None):
    """One next experiment, only for the independently observed current compiler."""
    selected = _current()
    if selected is None:
        return None
    if current_compiler_sha256 is None and isinstance(document, dict):
        # Broker/generated-source identities, never author requested hashes,
        # experiment echoes, notes, or completion claims.
        current_compiler_sha256 = document.get("graded_source_sha256") or document.get("submission_sha256")
        variant = document.get("variant")
        if isinstance(variant, dict) and variant.get("applied") is True:
            current_compiler_sha256 = variant.get("sha256")
    try:
        selected, packet, context = _effective(selected, current_compiler_sha256)
        if not selected.tasks:
            return None
        outcomes = _outcomes(selected, packet)
    except (OSError, ValueError, TypeError, KeyError):
        return _unknown("selected_reference_input_stale_or_unavailable")
    if not _sha(current_compiler_sha256) or current_compiler_sha256 != context["compiler_sha256"]:
        return _prior_guidance(packet, context, current_compiler_sha256)
    tasks = packet["tasks"]
    task = next((row for row in tasks if row.get("status") in ("ready", "blocked", "unranked", "rejected")), None)
    if task is None:
        return dict(schema=FEEDBACK_SCHEMA, status="no_pending_experiment", packet_identity=packet["identity"],
                    advisory_only=True, qualification_issued=False, lines=[])
    if not isinstance(task.get("task_id"), str) or not isinstance(task.get("acceptance"), dict):
        return _unknown("selected_task_contract_incomplete")
    requests = task.get("requests") or []
    if (not isinstance(requests, list) or len(requests) > 32 or any(not isinstance(row, dict) for row in requests)
            or any(not isinstance(row.get("argv"), list) or len(row["argv"]) > 64
                   or any(type(value) is not str for value in row["argv"]) for row in requests)):
        return _unknown("selected_experiment_request_incomplete")
    hypotheses = task.get("hypotheses") or []
    attempts = task.get("prior_attempts") or []
    observed_requests = [outcomes[row["request_id"]] for row in requests if row["request_id"] in outcomes]
    requests = [row for row in requests if row["request_id"] not in outcomes]
    attempts = attempts + observed_requests
    evidence = task.get("evidence") or []
    lines = ["reference experiment (advisory; verify the emitted result): "
             + _text(task.get("summary", task["task_id"]))]
    if evidence:
        lines.append("observed/inferred: " + _text(evidence[0]))
    if hypotheses:
        lines.append("hypothesis: " + _text(hypotheses[0].get("summary", hypotheses[0])))
    if attempts:
        lines.append("previous trial: " + _attempt_text(attempts[-1]))
    if requests:
        command = shlex.join(requests[0]["argv"])
        lines.append("required selfcheck: " + (command if len(command.encode()) <= 600
                     else "use the complete required_selfchecks[0].argv in the response JSON"))
    elif observed_requests:
        lines.append("required evidence: inspect the executed mechanism and paired complete-region performance.")
    else:
        action = (task.get("next_action") or {}).get("action", "acquire new evidence")
        lines.append("required evidence: " + _text(action))
    lines.append("acceptance: executable mechanism checks, unchanged numerics, comparable complete-region cycles, "
                 "held-out shapes.")
    return dict(schema=FEEDBACK_SCHEMA, status="pending_experiment",
                packet_identity=_task_packet_identity(packet, task["task_id"]),
                execution_context_identity=packet.get("execution_context_identity"), task_id=task["task_id"],
                task_status=task.get("status"), observed=copy.deepcopy(evidence), hypotheses=copy.deepcopy(hypotheses),
                prior_attempts=copy.deepcopy(attempts), required_selfchecks=copy.deepcopy(requests),
                experiment_status="observed_mechanism_unresolved" if observed_requests else "pending",
                observed_requests=copy.deepcopy(observed_requests),
                acceptance=copy.deepcopy(task["acceptance"]), next_action=copy.deepcopy(task.get("next_action")),
                advisory_only=True, qualification_issued=False, completion_authority="trusted evaluator closure only",
                lines=lines)


def attach(document, *, current_compiler_sha256=None):
    """Append one selected experiment beside existing priorities without changing grades."""
    try:
        found = project(document, current_compiler_sha256=current_compiler_sha256)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        found = _unknown("selected_experiment_projection_unavailable")
    if found is None:
        return None
    document[KEY] = found
    prior = document.get("next_steps")
    if not isinstance(prior, dict):
        prior = dict(schema="merlin.selfcheck_next_steps.v1", advisory_only=True, lines=[], steps=[])
        document["next_steps"] = prior
    previous = prior.get("reference_experiment_lines", ())
    existing = [line for line in prior.get("lines", []) if line not in previous]
    # Two concise lines fit the existing in-sandbox client's bounded printer;
    # full observations, hypotheses and acceptance remain structured fields.
    selected_lines = found.get("lines", [])
    summary = selected_lines[0] if selected_lines else ""
    request = next((line for line in selected_lines if line.startswith("required ")), "")
    selected_lines = [summary, request] if request else [summary] if summary else []
    prior["lines"] = selected_lines + existing
    prior["reference_experiment_lines"] = selected_lines
    prior[KEY] = {key: value for key, value in found.items() if key != "lines"}
    return found


def pending_requests(compiler_sha256, *, include_observed=False):
    """Trusted ready requests for the existing experiment dispatcher; no execution here."""
    try:
        selected = _current()
        if selected is None or not _sha(compiler_sha256):
            return ()
        selected, packet, context = _effective(selected, compiler_sha256)
        if not selected.tasks:
            return ()
        if compiler_sha256 != context["compiler_sha256"]:
            return ()
        outcomes = _outcomes(selected, packet)
        requests = []
        for task in packet["tasks"]:
            action = task.get("next_action") or {}
            states = ("ready", "unranked", "rejected", "resolved") if include_observed else ("ready",)
            if task.get("status") not in states or action.get("depends_on") or action.get("gaps"):
                continue
            for request in task.get("requests") or ():
                if request.get("request_id") in outcomes and not include_observed:
                    continue
                if (not isinstance(request, dict)
                        or request.get("context_identity") != packet["execution_context_identity"]
                        or not isinstance(request.get("request_id"), str)
                        or request.get("settings_sha256") != document_digest(request.get("settings"))):
                    return ()
                entry = {**copy.deepcopy(request), "task_id": task["task_id"],
                         "packet_identity": _task_packet_identity(packet, task["task_id"])}
                if include_observed:
                    entry["owner_observed"] = request["request_id"] in outcomes
                requests.append(entry)
                if len(requests) == 32:
                    return tuple(requests)
        return tuple(requests)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return ()


def observe_result(request, document, *, base_compiler_sha256):
    """Attribute an executed broker result to one exact planned request.

    Candidate echoes identify a *proposed* join only. Actual source custody,
    applied knob state, selected cases and engine must independently agree.
    A numerical result never establishes the requested executable mechanism.
    """
    try:
        metadata = (request.get("tuning_experiment") or {}).get("reference_experiment")
        selected_id = request.get("reference_experiment_request_id")
        if isinstance(metadata, dict):
            request_id = metadata.get("request_id")
            if selected_id is not None and selected_id != request_id:
                return None
        elif metadata is None and isinstance(selected_id, str):
            request_id = selected_id
        else:
            return None
        planned = next((row for row in pending_requests(base_compiler_sha256)
                        if row["request_id"] == request_id), None)
        if planned is not None and metadata is None:
            # A pasted ID only proposes the lookup. All authority and the full
            # context come from the original selected packet, never its echo.
            metadata = {key: planned[key] for key in ("task_id", "packet_identity", "context_identity")}
        if planned is None or any(metadata.get(key) != planned.get(key) for key in
                                  ("task_id", "packet_identity", "context_identity")):
            return None
        if (request.get("submission", "submission") != planned.get("submission", "submission")
                or planned.get("baseline_compiler_sha256", base_compiler_sha256) != base_compiler_sha256):
            return None
        def flag(name):
            argv = planned.get("argv", [])
            positions = [index for index, value in enumerate(argv) if value == name]
            return argv[positions[0] + 1] if len(positions) == 1 and positions[0] + 1 < len(argv) else None
        capsule = planned.get("capsule")
        selector = request.get("capsules")
        cases = selector.split(",") if isinstance(selector, str) else selector
        if not isinstance(capsule, str) or cases != [capsule]:
            return None
        expected_sim = planned.get("sim") or flag("--sim")
        actual_sim = document.get("sim") or request.get("sim")
        if not isinstance(expected_sim, str) or actual_sim != expected_sim:
            return None
        if request.get("suite", "core") != (planned.get("suite") or flag("--suite") or "core"):
            return None
        expected_tiers = planned.get("tiers") or flag("--tiers")
        if expected_tiers is not None and request.get("tiers") != expected_tiers:
            return None
        selected, packet, context = _effective(_current(), base_compiler_sha256)
        current_settings = (context.get("tuning") or {}).get("current")
        if not isinstance(current_settings, dict):
            return None
        expected = planned.get("tuning_overrides")
        if expected is None:
            expected = {key: value for key, value in planned["settings"].items() if current_settings.get(key) != value}
        variant = document.get("variant") or {}
        if expected:
            if (variant.get("applied") is not True
                    or document_digest(variant.get("tuning_overrides")) != document_digest(expected)
                    or not _sha(variant.get("sha256"))):
                return None
            executed_source = variant["sha256"]
        else:
            if variant.get("applied") or variant.get("tuning_overrides"):
                return None
            executed_source = base_compiler_sha256
        if document.get("submission_sha256") != base_compiler_sha256:
            return None
        observations = [dict(capsule=row.get("capsule", row.get("name")), status=row.get("status"),
                             failure_plane=row.get("failure_plane"), cycles=row.get("cycles"))
                        for row in document.get("per_capsule", document.get("cases", ())) if isinstance(row, dict)]
        if any(row["capsule"] != capsule for row in observations) or not observations:
            return None
        from ..construction_actions import report_projection

        numerical = report_projection(document)["requested_numerical_screen"]
        result = dict(schema="merlin.reference_experiment_observation.v1", task_id=planned["task_id"],
                      request_id=planned["request_id"], packet_identity=planned["packet_identity"],
                      context_identity=packet["execution_context_identity"],
                      role=planned.get("role"), status="observed", task_complete=False,
                      mechanism_status="unknown", base_compiler_sha256=base_compiler_sha256,
                      executed_compiler_sha256=executed_source, settings_sha256=planned["settings_sha256"],
                      capsule=capsule, sim=actual_sim, observations=observations,
                      all_pass=document.get("all_pass"), numerical_status=numerical, qualification_issued=False,
                      required_next="Check emitted mechanism and paired complete-region performance "
                                    "before accepting the hypothesis.")
        result["selfcheck_request_id"] = document.get("selfcheck_request_id")
        result["graded_at_unix_ns"] = document.get("graded_at_unix_ns")
        _record_outcome(selected, packet, result)
        document["reference_experiment_result"] = result
        feedback = document.get(KEY)
        if isinstance(feedback, dict):
            feedback["latest_observation"] = copy.deepcopy(result)
            feedback["experiment_status"] = "observed_mechanism_unresolved"
            feedback["completion_authority"] = "trusted evaluator closure and executable mechanism proof required"
            feedback["next_action"] = dict(route="inspect_emitted_mechanism",
                action="Inspect the executed kernel and compare a paired baseline before accepting the hypothesis.")
            steps = document.get("next_steps")
            if isinstance(steps, dict):
                outcome = {"pass": "passed", "fail": "failed"}.get(numerical, "unresolved")
                line = "observed reference trial: " + _text(planned["request_id"], 100)
                line += "; numerical check " + outcome + "; emitted mechanism remains unresolved."
                previous = steps.get("reference_experiment_lines", [])
                retained = [value for value in steps.get("lines", []) if value not in previous]
                shown = [previous[0], line] if previous else [line]
                steps["lines"] = shown + retained
                steps["reference_experiment_lines"] = shown
                steps[KEY] = {key: value for key, value in feedback.items() if key != "lines"}
        return result
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None


def brief_lines(workspace):
    """Refresh the current source identity through the existing broker digest."""
    if _current() is None:
        return []
    from ..brokers.selfcheck import _submission_digest
    try:
        compiler = _submission_digest(Path(workspace) / "submission")
    except (OSError, ValueError):
        compiler = None
    try:
        found = project(current_compiler_sha256=compiler)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError):
        found = _unknown("selected_experiment_projection_unavailable")
    if not found or not found.get("lines"):
        return []
    return ["## Selected reference experiment", "", *found["lines"], ""]


_OBJECTIVE_TEXT = (
    "Primary optimization objective: approach the supplied expert kernels' performance through compiler rules "
    "that generalize. Treat supplied expert measurements as trusted performance targets; retain their stated "
    "measurement conditions and distinguish them from our model estimates. For every representative, record "
    "expert versus emitted candidate evidence for fusion, operand retention/traffic, transfer shapes, tiling, "
    "overlap and completion. Prioritize the largest supported mechanism and performance gaps; a slower valid "
    "kernel remains unfinished optimization work. Missing observations require a tool/evidence repair task. "
    "Exact instruction matching is unnecessary; demonstrate equivalent useful mechanisms or a measured better "
    "alternative. Derive schedules from source semantics, dimensions, tails, layouts, capacities and target laws. "
    "Reference names and copied per-case schedules must not select the implementation. "
)
_ANALYSIS_TEXT = (
    "Use references during construction, before a case passes. Choose a declared public case and one mapped "
    "region; align its source graph, fusion/live-outs, shapes, dtypes, layouts and quantization first. "
    "Inspect with `python agent_selfcheck.py --reference-analysis lift --reference-entry ID` (or "
    "`--reference-path reference-kernels/FILE --reference-symbol SYMBOL`). Inventory and bounded pages use "
    "`--reference-analysis inventory` and `--reference-offset N --reference-limit 8`; ZIPs accept "
    "`--reference-member MEMBER`. Missing operands/ISA correspondence are gaps, not inferred zeroes. "
    "After compilation, rebuild returned lowered IR with the existing "
    "`.agents/skills/inspect-kernel/scripts/build_kernel.py LOWERED_IR --out variants/NAME/build/CASE`, "
    "For a mapped machine-code view, run `python agent_selfcheck.py --reference-analysis compare --reference-entry ID "
    "--candidate-path variants/NAME/build/CASE/kernel.dis --candidate-symbol SYMBOL`. "
    "Select one region on each side; do not silently compare whole-model orchestration to a leaf kernel. "
    "A zero comparison denominator is missing evidence. Declared semantic contracts can be compared with "
    "format `semantic_region`; equal declarations still require binding to actual compilation products. "
)
_TOOLS_TEXT = (
    "For each mechanism, keep a baseline plus alternative explanations/schedules until evidence separates them. "
    "Use the selected transformation catalog and pass-library/ab-library-pass APIs; construction feedback "
    "names applicable APIs and missing source/effect adapters. Recompute applicability with "
    "`python optimization_suggest.py --ir SOURCE --output suggestions.json` when staged. "
    "Express choices through schedule.plan and consume it in emission. Inspect an actual plan using "
    "`--reference-analysis candidate --reference-format schedule_mlir --reference-path variants/NAME/plan.mlir`; "
    "its decisions alone never prove realization. Keep the consumer receipt and emitted-artifact pins, "
    "then run existing diagnostics, compile checks and numerical selfchecks. If composition cannot express a "
    "supported expert mechanism, implement the missing pass, lowering, schedule consumer or general scheduling "
    "rule on permitted compiler surfaces, with explicit legality and fallback; evaluate it through the same loop. "
)
_CLOSING_TEXT = (
    "Use declared knobs or ordinary authored variants for bounded "
    "alternatives, then compare compatible complete-region timing and check unseen shapes. "
    "Classify unresolved source semantics, ISA, parser, library, effect and schedule-consumer bindings as "
    "explicit compiler/tooling tasks; continue independent construction without waiting "
    "for manual per-kernel decisions. Maintain the experiment's `OPTIMIZATION_RULES.json` as a mechanism-by-operation "
    "family/shape index: legality/resource guards, pass order/settings, positive and negative results, emitted "
    "witnesses, trusted reference targets, timing domain, exact receipt paths and remaining gaps. Reuse "
    "`merlin.perf.optimization_ledger` and pinned `merlin.mining.reconstruction_history` records for outcomes; "
    "link those receipts from the rules and `ITERATION_NOTES.md`. Authored rules remain "
    "hypotheses until checked. "
    "Test baseline, alternatives and ablations on neighboring shapes, tails, capacity boundaries and related "
    "operations; freeze candidates before independent held-out evaluation. Promote rules through existing "
    "`merlin.transforms.shape_policy` proposal/assessment, preserving fallback outside validated guards. "
    "This guidance grants no additional source access, numerical, mechanism or performance credit.\n"
)
_MAP_TEXT = (
    "Read `REFERENCE_MAP.json` for expert reference regions linked to operation kinds and shapes, and lift a "
    "mapped region (its `file` and `locator`) with the analysis below to read its instructions, loops and "
    "data movement. "
)


def guidance_block(*, tasks: bool, reference_map: bool) -> str:
    """The TASK.md section of a selection: prior tasks when served, else the map and the analysis endpoint."""
    head = "\n\n## Use selected kernel reference evidence\n\n"
    if tasks:
        head += "Read `REFERENCE_EXPERIMENTS.json` for bounded goals, hypotheses and evidence provenance. "
    if reference_map:
        head += _MAP_TEXT
    if tasks:
        use = ("Use these tasks from the first turn, including missing fusion, representation and adapter work. "
               "They describe prior evidence; establish applicability from current source semantics, shapes, "
               "layouts, numerics and resource facts. ")
        after = ("After edits, retain prior goals and collect current evidence; never replay an old exact request "
                 "against different source bytes. ")
    else:
        use = ("Establish a reference technique's applicability from current source semantics, shapes, "
               "layouts, numerics and resource facts. ")
        after = ""
    return head + _OBJECTIVE_TEXT + _ANALYSIS_TEXT + use + _TOOLS_TEXT + after + _CLOSING_TEXT


def stage_guidance(workspace, run_dir):
    """Serve compact prior tasks before the ordinary treatment seal, only at fresh setup.

    A selection that withholds its tasks stages no ``REFERENCE_EXPERIMENTS.json``; its section points at
    the staged ``REFERENCE_MAP.json`` (when present) and the analysis endpoint instead."""
    selected = _current()
    if selected is None:
        return
    roots = (Path(workspace), Path(run_dir))
    mapped = (roots[0] / "REFERENCE_MAP.json").is_file()
    if not selected.tasks:
        _load(selected)
        for root in roots:
            path = root / "REFERENCE_EXPERIMENTS.json"
            if path.exists() or path.is_symlink():
                raise ValueError("withheld reference tasks cannot be served from a staged file")
        task = (roots[0] / "TASK.md").read_text()
        block = guidance_block(tasks=False, reference_map=mapped)
        for root in roots:
            (root / "TASK.md").write_text(task + block)
        return
    packet, context = _load(selected)
    prior = _advisory_tasks(packet) if packet.get("schema") == SHARDS_SCHEMA else packet["tasks"]
    guidance = dict(schema="merlin.reference_experiment_guidance.v1", packet_identity=packet["identity"],
                    prior_execution_context_identity=packet["execution_context_identity"],
                    prior_compiler_sha256=context["compiler_sha256"],
                    scope="prior_reference_context", current_evidence_status="unverified",
                    advisory_only=True, qualification_issued=False, executable_requests_allowed=False,
                    tasks=[_compact_prior_task(task) for task in prior[:16]])
    if packet.get("schema") == SHARDS_SCHEMA:
        guidance.update(total_prior_tasks=len(packet["tasks"]), omitted_prior_tasks=max(0, len(packet["tasks"]) - 16),
                        task_roster_identity=document_digest([task["task_id"] for task in packet["tasks"]]),
                        source_packet_count=len(packet["source_packet_identities"]))
    data = json.dumps(guidance, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if len(data.encode()) > 64 << 10:
        raise ValueError("reference author guidance exceeds 64 KiB")
    for root in roots:
        path = root / "REFERENCE_EXPERIMENTS.json"
        if path.exists() or path.is_symlink():
            raise ValueError("reference guidance is already staged; resume must verify original bytes")
    block = guidance_block(tasks=True, reference_map=mapped)
    task = (roots[0] / "TASK.md").read_text()
    for root in roots:
        path = root / "REFERENCE_EXPERIMENTS.json"
        path.write_text(data)
        path.chmod(0o444)
        (root / "TASK.md").write_text(task + block)
