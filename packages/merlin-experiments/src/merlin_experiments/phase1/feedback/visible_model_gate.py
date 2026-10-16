"""Visible full-model feedback on immutable agent source snapshots."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from merlin.common.tree_hash import hash_tree
from merlin_experiments.phase1.brokers.model_check import PROGRAMS, execute_visible
from merlin_experiments.phase1.run_inputs import strip_build_state


def _rows(source_sha256: str | None) -> dict:
    return {f"{model}/{program}": {"model": model, "program": program,
                                   "status": "pending", "census": None, "metrics": None,
                                   "cycles": None, "first_bad_group": None}
            for model, programs in PROGRAMS.items() for program in programs}


def _combine(doc: dict) -> dict:
    gate = doc.get("model_gate")
    if not isinstance(gate, dict):
        return doc
    rows = gate.get("programs") or {}
    statuses = [row.get("status") for row in rows.values()]
    complete = len(statuses) == 6 and all(status in {"pass", "fail"} for status in statuses)
    passed = complete and all(status == "pass" for status in statuses)
    gate["complete"], gate["passed"] = complete, passed
    capsule = doc.get("capsule_all_pass")
    same_source = gate.get("source_sha256") == doc.get("graded_source_sha256")
    if capsule is False or (complete and not passed):
        doc["all_pass"] = False
    elif capsule is True and same_source and passed:
        doc["all_pass"] = True
    else:
        doc["all_pass"] = None
    return doc


def publish(path: Path, update: dict) -> dict:
    """Atomically merge capsule and model results without either writer erasing the other."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / ".verdict.lock"
    with lock.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            if "model_gate" in update:
                result = {**previous, **update}
            else:
                result = dict(update)
                if "model_gate" in previous:
                    result["model_gate"] = previous["model_gate"]
                result["capsule_all_pass"] = update.get("all_pass")
            result = _combine(result)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
            return result
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _program_row(model: str, program: str, result: dict) -> dict:
    report = result.get("report") or {}
    programs = report.get("programs") if isinstance(report, dict) else None
    item = next((row for row in programs if isinstance(row, dict) and row.get("program") == program), None) \
        if isinstance(programs, list) else None
    present = isinstance(item, dict)
    if not present:
        item = {}
    status = result.get("status")
    if not present:
        status = "unavailable"
    elif status not in {"pass", "fail"}:
        status = "unavailable" if status not in {"pending", "timeout", "cancelled"} else status
    elif status == "pass":
        census = item.get("census")
        metrics = report.get("model_metrics")
        if (not isinstance(census, dict) or not isinstance(census.get("denominator"), int)
                or not isinstance(census.get("device"), int) or not isinstance(metrics, dict)
                or not isinstance(metrics.get("worst"), dict)):
            status = "unavailable"
    runs = item.get("runs") or []
    cycles = [run.get("cycles_functional") for run in runs if isinstance(run, dict)
              and run.get("cycles_functional") is not None]
    failures = report.get("failures") or []
    first_bad = next((row.get("first_bad_group") for row in failures if isinstance(row, dict)
                      and row.get("program") == program and row.get("first_bad_group") is not None), None)
    return {"model": model, "program": program, "status": status,
            "build": item.get("build"), "census": item.get("census"),
            "metrics": report.get("model_metrics"), "cycles": cycles,
            "first_bad_group": first_bad}


def check_snapshot(ws: Path, run_dir: Path, snapshot: Path, source_sha256: str, *,
                   stop: threading.Event, timeout_s: int = 540,
                   source_current: Callable[[], bool] | None = None) -> dict:
    gate = {"source_sha256": source_sha256, "split": "visible", "programs": _rows(source_sha256)}
    verdict = ws / "qa" / "verdict.json"
    publish(verdict, {"model_gate": gate})
    artifact = run_dir / "qa_history" / f"model_gate_{source_sha256[:16]}"
    artifact.mkdir(parents=True, exist_ok=True)
    last_probe = 0.0
    current = True

    def fresh_current() -> bool:
        if source_current is None:
            return True
        try:
            return bool(source_current())
        except (OSError, ValueError):
            return False

    def stopped() -> bool:
        nonlocal last_probe, current
        if stop.is_set():
            return True
        now = time.monotonic()
        if source_current is not None and now - last_probe >= 2:
            last_probe = now
            current = fresh_current()
        return not current

    for model, programs in PROGRAMS.items():
        if stopped():
            break
        result = execute_visible(ws, {"action": "check", "model": model, "split": "visible"},
                                 snapshot_source=snapshot, timeout_s=timeout_s, stop=stopped)
        if result.get("source_sha256") != source_sha256 or hash_tree(snapshot)["sha256"] != source_sha256:
            result = {"status": "error", "model": model,
                      "error": "model check did not bind the selected snapshot"}
        (artifact / f"{model}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        for program in programs:
            gate["programs"][f"{model}/{program}"] = _program_row(model, program, result)
        publish(verdict, {"model_gate": gate})
        if stopped() or not fresh_current():
            break
    gate["superseded"] = not fresh_current()
    if gate["superseded"]:
        publish(verdict, {"model_gate": gate})
    (artifact / "gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return gate


def start(ws: Path, run_dir: Path, *, interval_s: int = 120, timeout_s: int = 540):
    stop = threading.Event()
    publish(ws / "qa" / "verdict.json", {"model_gate": {
        "source_sha256": None, "split": "visible", "programs": _rows(None),
    }})

    def work():
        last_source = None
        while not stop.is_set():
            source = ws / "submission"
            if (source / "manifest.yaml").is_file():
                import tempfile

                with tempfile.TemporaryDirectory(prefix="visible-model-gate-", dir=run_dir / "_qa_work") as scratch:
                    snapshot = Path(scratch) / "submission"
                    try:
                        if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
                            raise ValueError("submission snapshot contains a symlink")
                        before = hash_tree(source)["sha256"]
                        shutil.copytree(source, snapshot,
                                        ignore=shutil.ignore_patterns("build", "__pycache__", ".git"))
                        strip_build_state(snapshot)
                        digest = hash_tree(snapshot)["sha256"]
                        if hash_tree(source)["sha256"] != before:
                            raise ValueError("submission changed while taking model snapshot")
                        if digest != last_source:
                            def source_current():
                                try:
                                    return hash_tree(source)["sha256"] == before
                                except (OSError, ValueError):
                                    return False
                            gate = check_snapshot(ws, run_dir, snapshot, digest, stop=stop, timeout_s=timeout_s,
                                                  source_current=source_current)
                            if source_current():
                                # A stable numeric failure is a result for these bytes; an
                                # infrastructure failure must be retried at the next interval.
                                if all(row["status"] in {"pass", "fail"}
                                       for row in gate["programs"].values()):
                                    last_source = digest
                            else:
                                stop.wait(1)
                                continue
                    except (OSError, ValueError) as exc:
                        print(f"[model-gate] snapshot unavailable: {type(exc).__name__}: {exc}", flush=True)
            stop.wait(max(30, interval_s))

    (run_dir / "_qa_work").mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=work, name="visible-model-gate", daemon=True)
    thread.start()
    return thread, stop


def stop_gate(handle):
    if handle is None:
        return
    thread, stop = handle
    stop.set()
    thread.join(timeout=10)
