"""Conformance summaries used by the Phase 1 authoring continuation."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path


def conformance_ever(rounds) -> dict:
    """Fold the run-level gate: true if ever satisfied, None if always inapplicable.

    Resume rebuilds this same view from persisted round records.
    """
    ever: dict = {}
    for entry in rounds or []:
        checks = ((entry or {}).get("conformance") or {}).get("checks") or {}
        for key, value in checks.items():
            if value is None:
                ever.setdefault(key, None)
            else:
                ever[key] = bool(ever.get(key)) or bool(value)
    return ever


def workflow_conformance(
    tpath: Path, submission_dir: Path, arm: str, endpoint_kind: str, resolved_tools
) -> tuple[dict, bool]:
    """Recompute current workflow evidence from resolved tools; checker errors fail closed."""
    try:
        from merlin_experiments.phase1 import conformance as _CONF

        conf = _CONF.compute(tpath, submission_dir, arm, endpoint_kind, resolved_tools=resolved_tools)
    except Exception as exc:  # noqa: BLE001 — conformance is a formal gate; unavailable means false
        conf = {"conformant": False, "error": f"{type(exc).__name__}: {exc}"}
    return conf, conf.get("conformant") is True


def verify_live_selection(options, environment: dict) -> None:
    """This opt-in author feedback selection is immutable across continuation/resume."""
    recorded = environment.get("run_config") or {}
    prior = recorded.get("live_conformance_feedback", False)
    if type(prior) is not bool or prior != options.live_conformance_feedback:
        raise RuntimeError("resume refused: live workflow conformance feedback selection changed")


def _public_projection(conf: dict, submission: Path, *, running: bool) -> dict:
    """Publish no transcript, raw tool results, private paths or exception strings."""
    checks = conf.get("checks")
    available = isinstance(checks, dict) and not any(key in conf for key in ("error", "regex_scan_error"))
    public_checks = {
        key: value
        for key, value in (checks.items() if isinstance(checks, dict) else ())
        if isinstance(key, str) and (type(value) is bool or value is None)
    }
    hits = []
    for hit in conf.get("regex_hits") or []:
        if not isinstance(hit, dict) or not isinstance(hit.get("file"), str):
            continue
        path, line = Path(hit["file"]), hit.get("line")
        if path.is_absolute() or ".." in path.parts or type(line) is not int or line < 1:
            continue
        if not (submission / path).resolve().is_relative_to(submission.resolve()):
            continue
        hits.append({"file": str(path), "line": line})
    return {
        "schema_version": 1,
        "advisory": True,
        "available": available,
        "conformant": available and conf.get("conformant") is True,
        "checks": public_checks,
        "regex_hits": hits,
        "discovery_source_attribution": {
            key: value for key, value in (conf.get("discovery_source_attribution") or {}).items()
            if isinstance(key, str) and type(value) is int
        },
        "running": running,
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _publish(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".workflow-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def live_feedback(
    workspace: Path,
    run_dir: Path,
    round_index: int,
    arm: str,
    endpoint_kind: str,
    resolved_tools,
    *,
    interval_s: float = 5,
):
    """Observe the same actual checker during one launch; never run a capsule grade.

    This view is advisory and current-round only. Formal completion independently
    recomputes its existing gate. Missing startup evidence remains false while work
    is in progress, and a discovery-order failure cannot be repaired by late calls.
    """
    stop = threading.Event()
    transcript = run_dir / "rounds" / f"round_{round_index:02d}.transcript.jsonl"
    submission = workspace / "submission"
    destination = workspace / "qa" / "workflow_conformance.json"
    last_error = None

    def private_error(value: str) -> None:
        nonlocal last_error
        if value == last_error:
            return
        last_error = value
        try:
            descriptor = os.open(
                run_dir / "workflow_conformance_errors.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            with os.fdopen(descriptor, "a") as stream:
                stream.write(json.dumps({"round": round_index, "error": value}) + "\n")
        except OSError:
            pass  # Advisory publication must not replace the formal completion gate.

    def refresh(*, running: bool) -> None:
        try:
            conf, _ = workflow_conformance(transcript, submission, arm, endpoint_kind, resolved_tools())
            for key in ("error", "regex_scan_error"):
                if key in conf:
                    private_error(str(conf[key]))
            _publish(destination, _public_projection(conf, submission, running=running))
        except Exception as exc:  # noqa: BLE001 — errors remain private; final gate is independent
            private_error(f"{type(exc).__name__}: {exc}")
            try:
                _publish(destination, _public_projection({}, submission, running=running))
            except OSError:
                pass

    def observe() -> None:
        while not stop.wait(interval_s):
            refresh(running=True)

    refresh(running=True)
    thread = threading.Thread(target=observe, name="workflow-conformance-feedback", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        refresh(running=False)


def live_launch(launch, enabled: bool, workspace: Path, run_dir: Path, arm: str, endpoint_kind: str, resolved_tools):
    """Wrap the shared author launch once, including normal, continuous and L3 repair paths."""
    if not enabled:
        return launch

    @wraps(launch)
    def invoke(*args, **kwargs):
        with live_feedback(workspace, run_dir, args[6], arm, endpoint_kind, resolved_tools):
            return launch(*args, **kwargs)

    return invoke
