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

from merlin.verification.lean.contract import document_digest

# ---- reference_experiments ---------------------------------------------------------------------

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
    return (
        "--reference-experiments-selection",
        str(selected.path),
        "--reference-experiments-selection-sha256",
        selected.sha256,
        *(() if selected.tasks else ("--reference-experiments-selection-tasks", "off")),
    )


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
    if (
        not _sha(sha)
        or not path.is_absolute()
        or path.is_symlink()
        or path.resolve() != path
        or any(parent.is_symlink() for parent in path.parents)
        or not path.is_file()
        or path.stat().st_size > limit
    ):
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

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite_packet_field")),
    )


def _load_single(selected, selection=None):
    selection = _read(selected.path, selected.sha256) if selection is None else selection
    if (
        not isinstance(selection, dict)
        or selection.get("schema") != SCHEMA
        or not {"packet", "current_context"} <= set(selection)
    ):
        raise ValueError("unsupported_reference_selection")
    pin, context = selection["packet"], selection["current_context"]
    if (
        not isinstance(pin, dict)
        or set(pin) != {"path", "sha256"}
        or not isinstance(context, dict)
        or any(not _sha(context.get(key)) for key in _CONTEXT_FIELDS)
    ):
        raise ValueError("incomplete_reference_context")
    evaluator = context.get("evaluator")
    if not isinstance(evaluator, dict) or not evaluator.get("identity") or not _sha(evaluator.get("sha256")):
        raise ValueError("incomplete_evaluator_context")
    packet = _read(pin["path"], pin["sha256"])
    if (
        not isinstance(packet, dict)
        or packet.get("schema") != PACKET_SCHEMA
        or packet.get("identity") != document_digest({key: value for key, value in packet.items() if key != "identity"})
        or document_digest(packet.get("context")) != document_digest(context)
    ):
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
        if (
            not path.is_absolute()
            or path.is_symlink()
            or path.resolve() != path
            or not path.is_file()
            or path.stat().st_size > 256 << 20
        ):
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
        if (
            not isinstance(task_id, str)
            or task_id in task_ids
            or not isinstance(task.get("acceptance"), dict)
            or task.get("status") not in ("ready", "blocked", "unranked", "rejected", "resolved")
            or not isinstance(task.get("goals", []), list)
            or any(not isinstance(goal, str) for goal in task.get("goals", []))
        ):
            raise ValueError("invalid_reference_task_contract")
        task_ids.add(task_id)
        for field in ("evidence", "hypotheses", "prior_attempts"):
            if not isinstance(task.get(field, []), list) or any(
                not isinstance(row, dict) for row in task.get(field, [])
            ):
                raise ValueError("invalid_reference_task_evidence")
        requests = task.get("requests", [])
        if not isinstance(requests, list):
            raise ValueError("invalid_reference_requests")
        for request in requests:
            if (
                not isinstance(request, dict)
                or not isinstance(request.get("request_id"), str)
                or request["request_id"] in request_ids
                or request.get("context_identity") != packet["execution_context_identity"]
                or not isinstance(request.get("settings"), dict)
                or request.get("settings_sha256") != document_digest(request["settings"])
                or not isinstance(request.get("argv"), list)
                or not 1 <= len(request["argv"]) <= 64
                or any(not isinstance(value, str) for value in request["argv"])
            ):
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
    if (
        selection.get("complete") is not True
        or type(total) is not int
        or not 0 <= total <= 16 * _MAX_SHARDS
        or not isinstance(shards, list)
        or not 1 <= len(shards) <= _MAX_SHARDS
    ):
        raise ValueError("incomplete_reference_shard_index")
    packets, tasks, origins = [], [], {}
    packet_ids, task_ids, request_ids, selectors = set(), set(), set(), set()
    context, offset, byte_count = None, 0, 0
    for shard in shards:
        if (
            not isinstance(shard, dict)
            or set(shard) != {"packet", "selection", "selection_sha256"}
            or not isinstance(shard["selection"], str)
            or not isinstance(shard["packet"], str)
        ):
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
        if (
            (not packet["tasks"] and (total != 0 or len(shards) != 1))
            or not isinstance(coverage, dict)
            or any(type(coverage.get(key)) is not int for key in ("selected_start", "selected_end", "total_diagnoses"))
            or coverage["selected_start"] != offset
            or coverage["selected_end"] != end
            or coverage["total_diagnoses"] != total
            or end > total
            or coverage.get("next_offset") != (end if end < total else None)
        ):
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
    aggregate = dict(
        schema=SHARDS_SCHEMA,
        context=context,
        tasks=tasks,
        execution_context_identity=packet["execution_context_identity"],
        selected_index_sha256=selected.sha256,
        source_packet_identities=packets,
        task_packet_identities=origins,
    )
    aggregate["identity"] = document_digest(aggregate)
    return aggregate, context


def _task_packet_identity(packet, task_id):
    return packet.get("task_packet_identities", {}).get(task_id, packet["identity"])


def _request_limit(packet):
    return 32 * len(
        (packet["source_packet_identities"] if "source_packet_identities" in packet else [packet["identity"]])
    )


def _journal_limit(packet):
    return 16 << 20 if packet.get("schema") == SHARDS_SCHEMA else _MAX_BYTES


def _unknown(reason):
    return dict(
        schema=FEEDBACK_SCHEMA,
        status="unknown",
        advisory_only=True,
        qualification_issued=False,
        reason=reason,
        next_action=dict(
            route="acquire_new_evidence",
            action="Rebuild the reference experiment packet from the current trusted compiler and evidence.",
        ),
        lines=[
            "reference experiment (advisory): context unavailable; acquire current evidence.",
            "No experiment is marked tested or complete from this packet.",
        ],
    )


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
        return dict(
            schema="merlin.reference_experiment_observations.v1",
            selection_sha256=selected.sha256,
            packet_identity=packet["identity"],
            context_identity=packet["execution_context_identity"],
            outcomes={},
        )
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_mode & 0o077
        or path.stat().st_uid != os.geteuid()
        or path.stat().st_size > _journal_limit(packet)
    ):
        raise ValueError("reference_observation_store_unavailable")
    raw = path.read_bytes()
    document = _read(path, hashlib.sha256(raw).hexdigest(), limit=_journal_limit(packet))
    if (
        not isinstance(document, dict)
        or document.get("schema") != "merlin.reference_experiment_observations.v1"
        or document.get("selection_sha256") != selected.sha256
        or document.get("packet_identity") != packet["identity"]
        or document.get("context_identity") != packet["execution_context_identity"]
        or document.get("identity")
        != document_digest({key: value for key, value in document.items() if key != "identity"})
    ):
        raise ValueError("reference_observation_store_changed")
    return document


def _read_outcomes(selected, packet, path):
    document = _read_store(selected, packet, path)
    outcomes = document.get("outcomes")
    if not isinstance(outcomes, dict) or len(outcomes) > _request_limit(packet):
        raise ValueError("reference_observation_roster_invalid")
    requests = {
        request["request_id"]: (task["task_id"], request)
        for task in packet["tasks"]
        for request in task.get("requests", [])
    }
    for key, outcome in outcomes.items():
        if not isinstance(outcome, dict) or key not in requests:
            raise ValueError("reference_observation_request_unknown")
        task_id, request = requests[key]
        expected = dict(
            request_id=key,
            task_id=task_id,
            packet_identity=_task_packet_identity(packet, task_id),
            context_identity=packet["execution_context_identity"],
            settings_sha256=request["settings_sha256"],
            base_compiler_sha256=packet["context"]["compiler_sha256"],
        )
        if (
            any(outcome.get(field) != value for field, value in expected.items())
            or outcome.get("task_complete") is not False
            or outcome.get("qualification_issued") is not False
            or outcome.get("mechanism_status") != "unknown"
            or not _sha(outcome.get("executed_compiler_sha256"))
        ):
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
    if (
        current_path.parent != selected.path.parent
        or not current_path.name.startswith(".reference-analysis-" + selected.sha256 + "-")
        or not _sha(pin.get("sha256"))
    ):
        raise ValueError("current_reference_analysis_selection_invalid")
    current = _Selection(current_path, pin["sha256"], os.getpid())
    generated, fresh = _load(current)
    if fresh.get("compiler_sha256") != compiler_sha256 or active.get("compiler_sha256") != compiler_sha256:
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


DEPENDENCY_UNAVAILABLE = "analysis_experiment_dependency_unavailable"


def record_analysis(report, *, workspace, compiler_sha256):
    """Publish fresh broker analysis through the existing packet/observation owners.

    This entry is host-only. It refreshes compiler and declared tuning evidence,
    retaining the selected evaluator, public applicability and reference inputs.
    No reported setting, artifact claim or numerical result admits a mechanism.
    """
    selected = _current()
    if selected is None or not _sha(compiler_sha256):
        return None
    from merlin.compiler.selection.knobs import DECLARATION

    try:
        from merlin.analysis.mining.reconstruction.diagnoses import workspace_tuning
        from merlin.analysis.mining.reconstruction.experiment_packets import build_experiment_packet
    except ImportError:
        # One optional dependency missing from the broker environment degrades
        # the experiment binding; the analysis itself still answers.
        return dict(status="unknown", reason=DEPENDENCY_UNAVAILABLE, tasks=[], requests=[])
    from merlin.development.backend.brokers.selfcheck_dispatch import _submission_digest

    packet, context = _load(selected)
    workspace = Path(workspace).absolute()
    # Omitted author workspace selection names the fixed submission directory;
    # its actual current digest is still required immediately below.
    baseline = context.get("baseline") or {}
    submission_name = baseline["submission"] if "submission" in baseline else "submission"
    submission = workspace / submission_name
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
    prefix = (
        ".reference-analysis-"
        + selected.sha256
        + "-"
        + document_digest(dict(packet=generated["identity"], inputs=pins))
    )
    generated_path = selected.path.parent / (prefix + ".packet.json")
    selection_path = selected.path.parent / (prefix + ".selection.json")
    generated_sha = _write_private(generated_path, generated)
    selection_sha = _write_private(
        selection_path,
        dict(
            schema=SCHEMA,
            packet=dict(path=str(generated_path), sha256=generated_sha),
            current_context=fresh,
            context_pins=[*original.get("context_pins", []), *pins],
        ),
    )
    current = _Selection(selection_path, selection_sha, os.getpid())
    _load(current)
    with _journal(selected) as path:
        document = _read_store(selected, packet, path)
        document["current_analysis"] = dict(
            selection=dict(path=str(selection_path), sha256=selection_sha),
            compiler_sha256=compiler_sha256,
            report_identity=document_digest(report),
        )
        _write_store(path, document)
    return generated


def _text(value, limit=220):
    return " ".join(str(value).split())[:limit]


def _emitted_program(document, *, capsule, compiler_sha256):
    """Read only the evaluator's complete scoped program observation, never knob claims."""
    rows = document.get("per_capsule", document.get("cases", ()))
    rows = [row for row in rows if isinstance(row, dict) and row.get("capsule") == capsule]
    if len(rows) != 1 or document.get("candidate_sha256", compiler_sha256) != compiler_sha256:
        return None
    value = rows[0].get("performance") or {}
    fields = (
        "capsule_sha256",
        "selection_sha256",
        "model_sha256",
        "comparison_domain_sha256",
        "scope_sha256",
        "elf_sha256",
    )
    if (
        not isinstance(value, dict)
        or value.get("schema") != "merlin.phase1.joint_performance.v1"
        or value.get("complete_event_accounting") is not True
        or any(not _sha(value.get(key)) for key in fields)
    ):
        return None
    return dict(
        compiler_sha256=compiler_sha256,
        **{key: value[key] for key in fields},
        counts_sha256=document_digest(value.get("feature_counts", {})),
    )


def _emitted_effects(task, outcomes):
    """An unchanged matched ELF is negative effect evidence, never mechanism admission."""
    requests = {row["request_id"]: row for row in task.get("requests", [])}
    effects = []
    for request in requests.values():
        if request.get("role") != "variant" or request["request_id"] not in outcomes:
            continue
        variant = outcomes[request["request_id"]]
        baseline = outcomes.get(request.get("baseline_request_id"), {})
        before, after = baseline.get("emitted_program"), variant.get("emitted_program")
        status, reason = "unknown", "paired_scoped_emitted_program_witness_missing"
        if isinstance(before, dict) and isinstance(after, dict):
            keys = ("capsule_sha256", "selection_sha256", "model_sha256", "comparison_domain_sha256", "scope_sha256")
            matched = (
                baseline.get("role") == "baseline"
                and baseline.get("task_id") == variant.get("task_id")
                and all(
                    baseline.get(k) == variant.get(k)
                    for k in ("context_identity", "base_compiler_sha256", "capsule", "sim", "suite", "tiers")
                )
                and baseline.get("settings_sha256") != variant.get("settings_sha256")
                and all(_sha(before.get(k)) and before[k] == after.get(k) for k in keys)
                and before.get("compiler_sha256") == baseline.get("executed_compiler_sha256")
                and after.get("compiler_sha256") == variant.get("executed_compiler_sha256")
                and before.get("compiler_sha256") != after.get("compiler_sha256")
                and all(_sha(row.get("elf_sha256")) for row in (before, after))
            )
            if matched:
                same = before["elf_sha256"] == after["elf_sha256"]
                if not same or before.get("counts_sha256") == after.get("counts_sha256"):
                    status = "requested_but_no_emitted_change" if same else "emitted_program_changed"
                    reason = "identical_matched_emitted_program" if same else "goal_specific_mechanism_still_unknown"
        effects.append(
            dict(
                task_id=task["task_id"],
                request_id=request["request_id"],
                baseline_request_id=request.get("baseline_request_id"),
                capsule=request.get("capsule"),
                status=status,
                reason=reason,
                goals=task.get("goals", []),
                application_status=(
                    "optimization_did_not_apply_emitted_fallback"
                    if status == "requested_but_no_emitted_change"
                    else "emitted_change_requires_mechanism_check"
                    if status == "emitted_program_changed"
                    else "unknown"
                ),
                application_scope="requested_variant_relative_to_matched_baseline",
                fallback_scope="retained_baseline_program" if status == "requested_but_no_emitted_change" else None,
                baseline=copy.deepcopy(before),
                candidate=copy.deepcopy(after),
                advisory_only=True,
                mechanism_status="unknown",
                qualification_issued=False,
            )
        )
    return effects


def _construction_action(task):
    return dict(
        route="construction_gap",
        action="inspect_applicability_and_library_route",
        task_id=task["task_id"],
        gaps=["construction.requested_but_no_emitted_change"],
        library_routes=copy.deepcopy((task.get("next_action") or {}).get("library_routes", [])),
    )


def _attempt_text(value):
    if isinstance(value, dict) and value.get("schema") == "merlin.reference_experiment_observation.v1":
        outcome = {"pass": "passed", "fail": "failed"}.get(value.get("numerical_status"), "unknown")
        return (
            _text(value.get("request_id")) + ": numerical check " + outcome + "; emitted mechanism remains unresolved"
        )
    return _text(value)
