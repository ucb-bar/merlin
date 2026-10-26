"""Trusted accepted models for new requests; existing requests retain their pins.

The campaign's initial selection and clock never change. Only the host owner can
publish an accepted calibration result. This module has no author-stop hook and
does not grant measurements, fitting, broader coverage or source revisions.
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
from functools import wraps

from merlin.runtime import _native_files as F
from .feedback import joint_performance as PERF

REQUEST_ENV = "MERLIN_JOINT_PERFORMANCE_REQUEST"
SCHEMA = "merlin.joint_accepted_model.v1"
REQUEST_SCHEMA = "merlin.joint_request_model.v1"
_GRADE = ContextVar("joint_grade_model_admission", default=None)


def _root(state_path):
    return F.canonical(Path(state_path)).parent / "joint_model_activation"


def _state(state_path):
    path = F.canonical(Path(state_path))
    pin = F.pin(path, max_bytes=16 << 20)
    return PERF._read(pin)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)
    return F.pin(path, max_bytes=16 << 20)


@contextlib.contextmanager
def _lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "owner.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _head(root, state):
    """Verify the bounded committed chain; a bad head cannot stop authoring.

    A committed marker is the durable acceptance point. Uncommitted record
    files are never selected by recovery. The mutable head is a cache only.
    A damaged later record preserves the last valid prefix, with a warning.
    """
    previous_pin, previous, warning = None, None, None
    markers = sorted(root.glob("committed-*.json"))[:65]
    if len(markers) > 64:
        warning = "accepted model history exceeds the bounded review roster"
        markers = markers[:64]
    for sequence, marker in enumerate(markers, 1):
        try:
            if marker != root / f"committed-{sequence:04d}.json":
                raise ValueError("accepted model committed sequence is not contiguous")
            committed = PERF._read(F.pin(marker, max_bytes=16 << 20))
            record_pin = committed["record"]
            if Path(record_pin["path"]) != root / f"accepted-{sequence:04d}.json":
                raise ValueError("accepted model record is outside its original campaign")
            record = PERF._read(record_pin)
            expected_incumbent = previous["selection"] if previous else state.get("performance_selection")
            if (record.get("schema") != SCHEMA or record.get("sequence") != sequence
                    or record.get("previous") != previous_pin or record.get("incumbent") != expected_incumbent
                    or record.get("policy_sha256") != state["policy_sha256"]
                    or record.get("initial_selection") != state.get("performance_selection")
                    or not record.get("review_id")):
                raise ValueError("accepted model predecessor/campaign ownership changed")
            PERF._read(record["selection"])
            PERF._read(record["calibration_evidence"])
            previous_pin, previous = record_pin, record
        except (OSError, ValueError, KeyError, TypeError):
            warning = "accepted model history is invalid; retaining its last verified selection"
            break
    try:
        path = root / "accepted.json"
        if path.exists():
            cached = PERF._read(F.pin(path, max_bytes=16 << 20))
            if cached != {"record": previous_pin}:
                warning = warning or "accepted model pointer is stale; retaining verified committed selection"
        elif previous_pin is not None:
            warning = warning or "accepted model pointer is missing; retaining verified committed selection"
    except (OSError, ValueError, KeyError, TypeError):
        warning = warning or "accepted model pointer is invalid; retaining verified committed selection"
    return previous_pin, previous, warning


def current(state_path):
    """Read the host pointer only at admission/round start, never during a request."""
    state = _state(state_path)
    _, record, _ = _head(_root(state_path), state)
    return record["selection"] if record else state.get("performance_selection")


def _accepted_epoch(candidate_pin, incumbent_pin, state):
    """Reopen the existing producer's strict local admission, not a caller claim."""
    from . import joint_calibration as C
    from .joint_calibration_admission import assert_allowed
    from .joint_calibration_eligibility import require
    from .joint_runtime_revision import validate

    old, new = PERF._read(incumbent_pin), PERF._read(candidate_pin)
    before, after = old.get("timing_epochs", []), new.get("timing_epochs", [])
    if (not isinstance(after, list) or len(after) != len(before) + 1
            or after[:-1] != before
            or {k:v for k,v in old.items() if k != "timing_epochs"}
                != {k:v for k,v in new.items() if k != "timing_epochs"}):
        raise ValueError("acceptance requires one appended calibrated epoch and unchanged source/runtime selection")
    epoch = after[-1]
    if type(epoch.get("epoch")) is not int or epoch["epoch"] <= max((e["epoch"] for e in before), default=0):
        raise ValueError("accepted epoch does not advance the prior model sequence")
    # Reuse the public coverage/schema verifier; no new qualification is minted.
    from .feedback.joint_timing_selection import choose
    coverage = epoch.get("coverage") or {}
    capsules = coverage.get("capsule_sha256") or []
    if not capsules or choose(new, capsules[0]) != epoch:
        raise ValueError("accepted epoch has no selected coverage")
    evidence = PERF._read(epoch["calibration_evidence"])
    if (epoch.get("qualification") != "limited_calibrated_provisional"
            or evidence.get("status") != "limited_calibrated_provisional"
            or evidence.get("admitted") is not True
            or evidence.get("design", {}).get("identifiable") is not True
            or evidence.get("selected_runtime") != epoch["runtime"]
            or evidence.get("qualification_profile") != C.GATES["qualification_profile"]
            or evidence.get("all_declared_cases_evaluated") is not True
            or any(type(evidence.get(k)) not in (int,float) or not math.isfinite(evidence[k])
                   or not low <= evidence[k] <= high for k,low,high in (
                       ("maximum_case_error",0,C.GATES["max_case"]),
                       ("p95",0,C.GATES["max_p95"]),
                       ("rank_agreement",C.GATES["min_rank_agreement"],1)))):
        raise ValueError("calibration producer did not admit this runtime")
    plan = PERF._read(evidence["plan"])
    profile = PERF._read(state["policy"]["native_calibration"])
    if (plan.get("schema") != C.SCHEMA or plan.get("gates") != C.GATES
            or plan.get("selection") != incumbent_pin
            or plan.get("parameters") != profile["parameters"]):
        raise ValueError("accepted model uses a stale incumbent or different calibration admission")
    validate(PERF._read(profile["performance_selection"]), old)
    assert_allowed(plan)
    groups = {}
    ids = set()
    covered = set()
    for case in plan["cases"]:
        cap = C._capsule(case["capsule"])
        name, group, split = cap["name"], case["source_group_sha256"], case["split"]
        if (case["id"] in ids or split not in {"train", "held"}
                or (group in groups and groups[group] != split)
                or cap.get("label") not in {"public", "dev"} or cap.get("kind") == "model"
                or name not in state["policy"]["gsim"]["witness_capsules"]
                or group != profile["members"].get(name, {}).get("source_group_sha256")):
            raise ValueError("accepted round has invalid independent public groups")
        ids.add(case["id"]);groups[group] = split
        source = PERF.selected_capsule(old, cap)
        source_hash = PERF.capsule_identity(source)
        if old.get("members", {}).get(name, {}).get("capsule_sha256") != source_hash:
            raise ValueError("accepted calibration capsule differs from original public source")
        covered.add(source_hash)
        require(PERF._read(case["result"]), cap, compiler_sha256=case["compiler_sha256"])
    if any(sum(v == split for v in groups.values()) < 2 for split in ("train", "held")):
        raise ValueError("accepted round lacks independent training/held groups")
    held_count = sum(case["split"] == "held" for case in plan["cases"])
    if (set(capsules) != covered or evidence.get("declared_held_cases") != held_count
            or evidence.get("evaluated_held_cases") != held_count):
        raise ValueError("accepted epoch coverage or held count differs from its original frozen cases")
    runtime = PERF._runtime(epoch["runtime"], facts_pin=new["facts"])
    runtime.verify()
    incumbent = PERF._runtime(plan["runtime"], facts_pin=new["facts"])
    expected_model = json.loads(incumbent.model_json)
    actual_model = json.loads(runtime.model_json)
    for parameter in plan["parameters"]:
        value = C._at(actual_model, parameter["path"])
        if value not in parameter["values"]:
            raise ValueError("accepted model price lies outside the frozen reviewed probes")
        C._set(expected_model, parameter["path"], value)
    if expected_model != actual_model:
        raise ValueError("accepted calibration changes more than the reviewed parameter prices")
    return epoch


def accept(*, state_path, selection_pin, incumbent_pin, review_id):
    """Publish one accepted producer outcome for future requests only (CAS)."""
    if not isinstance(review_id, str) or not review_id.strip() or len(review_id) > 1024:
        raise ValueError("model acceptance needs a bounded trusted review identity")
    state = _state(state_path)
    if state.get("started_at_epoch") is None:
        raise ValueError("model acceptance requires the already-started campaign")
    root = _root(state_path)
    with _lock(root):
        previous_pin, previous, warning = _head(root, state)
        selected = previous["selection"] if previous else state.get("performance_selection")
        if selected == selection_pin:
            return dict(status="already_accepted_for_future_requests", record=previous_pin, selection=selected,
                        activation_warning=warning)
        if selected != incumbent_pin:
            raise ValueError("accepted model incumbent was superseded; rebase before activation")
        epoch = _accepted_epoch(selection_pin, incumbent_pin, state)
        sequence = 1 + (previous["sequence"] if previous else 0)
        if sequence > 64:
            raise ValueError("accepted model history requires a new bounded host review")
        record = dict(schema=SCHEMA, sequence=sequence, previous=previous_pin,
            policy_sha256=state["policy_sha256"], initial_selection=state["performance_selection"],
            incumbent=incumbent_pin, selection=selection_pin, review_id=review_id,
            calibration_evidence=epoch["calibration_evidence"],
            qualification="limited_calibrated_provisional", applies_to="future_request_admission_only")
        destination = root / f"accepted-{sequence:04d}.json"
        if destination.exists():
            if PERF._read(F.pin(destination, max_bytes=16 << 20)) != record:
                raise ValueError("model acceptance record already has different ownership")
            pin = F.pin(destination, max_bytes=16 << 20)
        else:
            pin = _write(destination, record)
        committed = root / f"committed-{sequence:04d}.json"
        if committed.exists():
            if PERF._read(F.pin(committed, max_bytes=16 << 20)) != {"record":pin}:
                raise ValueError("model acceptance committed marker changed")
        else:
            _write(committed, {"record":pin})
        temporary = root / (".accepted-" + secrets.token_hex(8) + ".json")
        _write(temporary, {"record": pin})
        temporary.replace(root / "accepted.json")
        return dict(status="accepted_for_future_requests", record=pin, selection=selection_pin)


def request_environment(environment, *, channel, request_id, request_bytes):
    """Journal one request selection and return a fresh child env; no global mutation."""
    from . import joint_control
    env = dict(environment)
    state_path = env.get(joint_control.ENV)
    if not state_path:
        return env
    if (channel not in {"selfcheck", "simjob", "model_check", "grade"}
            or not isinstance(request_id,str) or not 1 <= len(request_id) <= 240
            or any(not (c.isascii() and (c.isalnum() or c in "_.-")) for c in request_id)):
        raise ValueError("request model pin requires a bounded native channel identity")
    state = _state(state_path)
    root = _root(state_path)
    path = root / "requests" / channel / (request_id + ".json")
    identity = dict(schema=REQUEST_SCHEMA, policy_sha256=state["policy_sha256"], channel=channel,
        campaign_state=str(F.canonical(Path(state_path))), initial_selection=state.get("performance_selection"),
        request_id=request_id, request_sha256=hashlib.sha256(request_bytes).hexdigest())
    with _lock(root):
        if path.exists():
            pin = F.pin(path, max_bytes=16 << 20)
            record = PERF._read(pin)
            if any(record.get(k) != v for k,v in identity.items()):
                raise ValueError("request identity changed after model admission")
        else:
            accepted_pin, accepted, warning = _head(root, state)
            record = dict(identity, selection=accepted["selection"] if accepted else state.get("performance_selection"),
                          accepted_record=accepted_pin, activation_warning=warning)
            pin = _write(path, record)
    if record["selection"]:
        PERF._read(record["selection"])
        env[PERF.ENV] = json.dumps(record["selection"], sort_keys=True)
    else:
        env.pop(PERF.ENV, None)
    env[REQUEST_ENV] = json.dumps(pin, sort_keys=True)
    return env


def request_binding(environment=None):
    """Public pin-only provenance; no accepted-pointer lookup in an active request."""
    env = os.environ if environment is None else environment
    raw = env.get(REQUEST_ENV)
    if not raw:
        return None
    pin = json.loads(raw)
    record = PERF._read(pin)
    selection = json.loads(env[PERF.ENV]) if env.get(PERF.ENV) else None
    if record.get("schema") != REQUEST_SCHEMA or record.get("selection") != selection:
        raise ValueError("worker request and selected model pins differ")
    return {k:record.get(k) for k in ("channel", "request_id", "request_sha256", "selection", "accepted_record", "activation_warning")}


def grade_model_scope(function):
    """Lifetime for a grade's one frozen-snapshot admission; nesting preserves it."""
    @wraps(function)
    def run(*args, **kwargs):
        if _GRADE.get() is not None:
            return function(*args, **kwargs)
        token = _GRADE.set({"environment":None, "source_sha256":None})
        try:
            with PERF.environment_scope(PERF.environment()):
                return function(*args, **kwargs)
        finally:
            _GRADE.reset(token)
    return run


def admit_grade(*, candidate, source_sha256, output, kind, identity):
    """Bind once after immutable source publication, before any evaluation work.

    Completion owners receive the returned environment explicitly because
    scheduler callbacks need not inherit their initiating thread's context.
    Pure grade_offload functions take explicit arguments and do not consume
    this evaluator; actual timing observers receive a frozen runtime directly.
    """
    frame = _GRADE.get()
    if frame is None:
        raise ValueError("grade model admission requires its original scoped owner")
    if frame["environment"] is not None:
        if frame["source_sha256"] != source_sha256:
            raise ValueError("nested grade changed the original frozen candidate")
        return dict(frame["environment"])
    record = dict(schema="merlin.grade_model_admission.v1", candidate=str(Path(candidate).resolve(strict=True)),
        source_sha256=source_sha256, output=str(Path(output).absolute()), kind=kind, identity=identity)
    raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
    env = request_environment(PERF.environment(), channel="grade",
        request_id="grade-" + hashlib.sha256(raw).hexdigest(), request_bytes=raw)
    frame["environment"], frame["source_sha256"] = env, source_sha256
    # The decorator restores the preceding thread context even on failure.
    PERF._ENVIRONMENT.set(dict(env))
    return dict(env)
