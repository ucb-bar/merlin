"""Bounded process-group inspection and shutdown for frozen experiment plans."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from yaml import YAMLError

from merlin.common.digest import sha256_file as _sha
from merlin.development.backend.session.inputs.implementation_inventory import fingerprint
from merlin.experiments.definition import (
    ExperimentSpec,
    SpecError,
    backend_launch_inputs,
    engine_definitions,
    read_yaml,
    require_plan_selection,
    selected_engines,
)
from merlin.experiments.definition import default_run_dir as default_run_dir
from merlin.experiments.runs.adapters import (
    ADAPTERS,
    PHASE0_MODULE,
    PHASE1_MODULE,
    phase0_sources,
    phase0_startup_inputs,
    phase1_entrypoint,
)

# ---- runner_process ----------------------------------------------------------------------------


def _process_active(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A reaped-by-another-supervisor zombie no longer owns any working process.
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().rpartition(") ")[2].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _process_group_active(group: int) -> bool:
    """A driver exit is not proof that its workers exited. Unknown stays active."""
    processes = Path("/proc")
    if not processes.is_dir():
        return True
    try:
        for path in processes.iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path / "stat").read_text().rpartition(") ")[2].split()
                if int(fields[2]) == group and fields[0] != "Z":
                    return True
            except FileNotFoundError:
                continue  # a process exited during inspection
            except (OSError, IndexError, ValueError):
                return True
    except OSError:
        return True
    return False


def _stop_process_group(process: subprocess.Popen) -> None:
    """Best-effort shutdown, never proof that an abnormal run left no descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # Workers may outlive their driver; always signal the whole private group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


# ---- runner_records ----------------------------------------------------------------------------
# Atomic process-plan records and structural JSON validation.


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpecError(f"invalid run record {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise SpecError(f"invalid run record {path}: expected an object")
    return document


# ---- runner ------------------------------------------------------------------------------------
# Freeze and execute process plans while leaving experiment evidence with its engine.


def resume(run_dir: Path) -> int:
    from merlin.experiments.runs.plans import status

    root = run_dir.expanduser().resolve()
    status(root)  # Validate frozen plan binding before using any stored argv.
    plan = _read_json(root / "resolved-plan.json")
    record = _read_json(root / "orchestration.json")
    return _execute(root, plan, record)


def _verify_inputs(plan: dict) -> None:
    for command in plan["phases"].values():
        if command["adapter"] not in ADAPTERS:
            raise SpecError(f"experiment adapter {command['adapter']!r} is unavailable; create a current definition")
    from merlin.experiments.runs.plan_sources import (
        _verify_corpus_closures,
        _verify_phase0_sources,
        _verify_phase1_sources,
    )

    _verify_corpus_closures(plan)
    _verify_phase0_sources(plan)
    _verify_phase1_sources(plan)

    for name, pin in plan["inputs"].items():
        if fingerprint(pin["path"]) != pin["sha256"]:
            raise SpecError(f"frozen input changed: {name} ({pin['path']}); create a new experiment definition/run")


_PHASE1_COMPLETION_MEMBERS = (
    "submission",
    "run_manifest.yaml",
    "qa_loop_summary.yaml",
    "timing_detailed.json",
)


def _phase1_completion_inputs(command: dict) -> dict[str, dict[str, str]]:
    """Bind the compiler and its four final handoff members, excluding mutable workspaces/caches."""
    root = Path(command["engine_output"])
    if root.resolve() != root or root.is_symlink():
        raise SpecError("installed Phase 1 handoff output changed location")
    result = {}
    for name in _PHASE1_COMPLETION_MEMBERS:
        path = root / name
        if path.is_symlink() or not path.exists():
            raise SpecError(f"installed Phase 1 completion member is missing or linked: {name}")
        if name == "submission" and not path.is_dir():
            raise SpecError("installed Phase 1 compiler submission is not a directory")
        if name == "submission" and any(member.is_symlink() for member in path.rglob("*")):
            raise SpecError("installed Phase 1 compiler submission contains linked members")
        if name != "submission" and not path.is_file():
            raise SpecError(f"installed Phase 1 completion member is not a file: {name}")
        result[name] = {"path": str(path), "sha256": fingerprint(path)}
    return result


def _verify_completed_phase1(command: dict, attempt: dict) -> None:
    expected = attempt.get("completion_inputs")
    if not isinstance(expected, dict) or set(expected) != set(_PHASE1_COMPLETION_MEMBERS):
        raise SpecError("completed installed Phase 1 lacks bound handoff outputs; create a new experiment run")
    try:
        observed = _phase1_completion_inputs(command)
    except (OSError, SpecError) as exc:
        raise SpecError(f"completed installed Phase 1 handoff changed: {exc}") from exc
    if observed != expected:
        raise SpecError("completed installed Phase 1 handoff bytes changed; create a new experiment run")


@contextmanager
def _engine_ownership(root: Path, plan: dict):
    """Retain ownership after ANY ambiguous post-spawn exit, even after killpg.

    Process groups cannot prove that a descendant did not detach. Only explicit
    operator acknowledgement may release that abnormal ownership. Never let a
    generic exception-closing context manager declare this output abandoned.
    """
    from merlin.common.paths import out_dir
    from merlin.common.storage.retention import acquire, inventory

    frozen_root = plan.get("storage_root")
    if not frozen_root or Path(frozen_root).resolve() != out_dir().resolve():
        raise SpecError("frozen storage root differs from MERLIN_OUT_ROOT; restore the original output root to resume")
    for command in plan["phases"].values():
        child_root = command.get("env", {}).get("MERLIN_OUT_ROOT")
        if not child_root or Path(child_root).resolve() != Path(frozen_root):
            raise SpecError("engine output root differs from the frozen storage ownership root")
    state = inventory()
    if state["status"] != "known":
        raise SpecError("storage ownership cannot be established")
    for row in state["paths"]:
        owned = Path(row["path"])
        if row["leases"] and (owned == root or owned in root.parents or root in owned.parents):
            raise SpecError(
                "unresolved engine ownership; verify all descendants stopped before acknowledging abandonment"
            )
    held = acquire(root, owner="merlin-experiments")
    ownership = {"unresolved": False, "failed": False}
    try:
        yield ownership
    except BaseException:
        if not ownership["unresolved"]:
            held.close("failed")
        raise
    else:
        if not ownership["unresolved"]:
            held.close("failed" if ownership["failed"] else "completed")


def _execute(root: Path, plan: dict, record: dict) -> int:
    from merlin.experiments.runs.plan_sources import _now
    from merlin.experiments.runs.plans import _exclusive

    with _exclusive(root), _engine_ownership(root, plan) as ownership:
        # Re-read after acquiring the lock: a preceding orchestrator may have finished.
        record = _read_json(root / "orchestration.json")
        for attempt in record["attempts"]:
            if attempt["state"] == "running" and _process_active(attempt.get("pid")):
                raise SpecError(f"previous engine process {attempt['pid']} is still running")
        _verify_inputs(plan)
        for number, command in plan["phases"].items():
            previous = [entry for entry in record["attempts"] if entry["phase"] == number]
            if previous and previous[-1]["state"] == "execution_succeeded":
                if command["adapter"] == "capsule_derivation":
                    expected = previous[-1].get("output_sha256")
                    try:
                        observed = fingerprint(command["engine_output"])
                    except (OSError, SpecError) as exc:
                        raise SpecError("phase-0 output identity changed; create a new experiment run") from exc
                    if not expected or observed != expected:
                        raise SpecError("phase-0 output identity changed; create a new experiment run")
                if command["adapter"] == "capsule_bench" and command.get("module") == PHASE1_MODULE:
                    _verify_completed_phase1(command, previous[-1])
                continue
            argv = list(command["argv"])
            if previous and command["resume_policy"] == "native_flag":
                argv.append("--resume")
            engine_output = command["engine_output"]
            _verify_inputs(plan)
            log = root / f"phase{number}-attempt-{len(previous) + 1:04d}.log"
            entry = {
                "phase": number,
                "adapter": command["adapter"],
                "state": "running",
                "started_at": _now(),
                "argv": argv,
                "log": str(log),
                "engine_output": engine_output,
            }
            record["attempts"].append(entry)
            record["state"] = "running"
            _write_json(root / "orchestration.json", record)
            process = None
            try:
                with log.open("w", encoding="utf-8") as output:
                    from merlin.experiments.runs.execution import execution_environment

                    process = subprocess.Popen(
                        command.get("frozen_launch", argv),
                        cwd=command["cwd"],
                        env=execution_environment(command),
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    ownership["unresolved"] = True
                    entry["pid"] = process.pid
                    _write_json(root / "orchestration.json", record)
                    try:
                        returncode = process.wait()
                    except KeyboardInterrupt:
                        _stop_process_group(process)
                        entry["ownership"] = "retained_until_descendants_verified"
                        returncode = 130
                    else:
                        if _process_group_active(process.pid):
                            raise SpecError("engine driver exited while workers remain; storage lease retained")
                        ownership["unresolved"] = False
            except OSError as exc:
                if process is not None:
                    _stop_process_group(process)
                    raise SpecError(
                        "post-spawn bookkeeping failed; storage lease retained for descendant verification"
                    ) from exc
                entry["error"] = str(exc)
                returncode = 127
            except BaseException:
                if process is not None and ownership["unresolved"]:
                    _stop_process_group(process)
                raise
            entry.update(
                returncode=returncode,
                ended_at=_now(),
                state="execution_succeeded" if returncode == 0 else "execution_failed",
            )
            if returncode == 0 and command["adapter"] == "capsule_derivation":
                try:
                    _verify_inputs(plan)
                    entry["output_sha256"] = fingerprint(engine_output)
                except (OSError, SpecError) as exc:
                    entry.update(
                        returncode=1,
                        state="execution_failed",
                        error=f"phase0 output identity not established: {exc}",
                    )
                    returncode = 1
            elif returncode == 0 and command["adapter"] == "capsule_bench":
                try:
                    _verify_inputs(plan)
                    if command.get("module") == PHASE1_MODULE:
                        entry["completion_inputs"] = _phase1_completion_inputs(command)
                except (OSError, SpecError) as exc:
                    entry.update(
                        returncode=1,
                        engine_returncode=0,
                        state="execution_failed",
                        error=f"phase1 handoff identity not established; engine evidence unchanged: {exc}",
                    )
                    returncode = 1
            record["state"] = entry["state"]
            _write_json(root / "orchestration.json", record)
            if returncode != 0:
                ownership["failed"] = True
                return returncode if returncode > 0 else 1
        record["state"] = "execution_succeeded"
        _write_json(root / "orchestration.json", record)
    return 0
