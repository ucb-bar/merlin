"""Own the capsule controller clock; external model validation has a separate owner.

Only a successful strict capsule-certification reopening admits the frozen
handoff. This supervisor never starts model validation or interprets its result.
"""

from __future__ import annotations

import datetime as dt
import json
import signal
import time
from pathlib import Path


class CapsuleStopped(BaseException):
    """Interrupt slow certification as well as the ordinary polling loop."""


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_handoff(run_dir: Path, descriptor: Path) -> dict:
    from merlin_experiments.phase1.feedback import capsule_certification

    return capsule_certification.verify_phase2_handoff(run_dir, descriptor=descriptor)


def run(launch, prep: Path, preparation_started: float, *, memory, processes_factory, env, poll_s: float) -> int:
    document = launch.document
    if document.get("validation_scope") != "capsules":
        raise ValueError("capsule supervisor requires explicit capsule scope")
    external = document["external_validation"]
    if (external.get("start_after") != "capsule_owner_terminal"
            or external.get("submission") != "verified_frozen_capsule_handoff"
            or external.get("budget_s", 0) <= 0):
        raise ValueError("invalid independent external-validation policy")
    total = document["budgets"]["supervisor_backstop_s"]
    limit = int(launch.variables["SUPERVISOR_PSS_MAX_BYTES"])
    headroom = int(launch.variables["SUPERVISOR_HOST_HEADROOM_BYTES"])
    processes = processes_factory()
    run_dir = Path(launch.variables["RUN"])
    state = {
        "schema": "merlin.phase1.capsule_supervision.v1", "validation_scope": "capsules",
        "status": "native_controller_starting", "capsule_complete": False,
        "external_full_models_complete": False, "phase1_complete": False,
        "external_validation": {**external, "status": "not_started"}, "models": {},
    }

    def publish(message, **fields):
        at = dt.datetime.now(dt.UTC).isoformat()
        record = {"at": at, "message": message, **fields}
        with (prep / "live.log").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        state.update(fields, updated_at=at)
        save(prep / "status.json", state)

    def interrupt(signum, _frame):
        raise CapsuleStopped("capsule_deadline" if signum == signal.SIGALRM else f"signal_{signum}")

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)
    previous = {signum: signal.signal(signum, interrupt) for signum in signals}
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    state.update(
        preparation_wall_s=round(started - preparation_started, 3), capsule_budget_s=total,
        capsule_started_at=dt.datetime.now(dt.UTC).isoformat(),
        absolute_deadline=(dt.datetime.now(dt.UTC) + dt.timedelta(seconds=total)).isoformat(),
    )
    signal.setitimer(signal.ITIMER_REAL, min(total, old_timer[0]) if old_timer[0] else total)
    native = None
    result = 2
    try:
        repo = launch.path(document["inputs"]["repo"])
        native = processes.start(
            launch.argv, prep / "phase1-native.log", role="native",
            cwd=Path(launch.expand(document["launcher"].get("cwd", str(repo)))), env=env,
            cpus=set(launch.manager_cpus),
        )["process"]
        save(prep / "controller.json", {
            "pid": native.pid, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "start_ticks": processes.known.get(native.pid, (None, None))[0],
            "started_at": state["capsule_started_at"], "capsule_budget_s": total,
        })
        publish("Fresh capsule controller started", status="authoring_or_capsule_grading", controller_pid=native.pid)
        last_update = 0
        while True:
            elapsed = time.monotonic() - started
            records = processes.owned()
            charged, available = memory(records)
            with (prep / "memory.jsonl").open("a") as stream:
                stream.write(json.dumps({"elapsed_s": round(elapsed, 1), "pss_bytes": charged,
                                         "host_available_bytes": available, "processes": len(records)}) + "\n")
            if elapsed >= total or charged > limit or available < headroom:
                raise CapsuleStopped("capsule_deadline" if elapsed >= total else "capsule_memory_guard")
            code = native.poll()
            if code is not None:
                state["controller_exit_code"] = code
                result = 1
                if code == 0:
                    publish("Reopening strict capsule certification", status="verifying_capsule_handoff")
                    verified = verify_handoff(run_dir, launch.path(document["inputs"]["descriptor"]))
                    state.update(capsule_certification=verified, capsule_complete=True)
                    result = 0
                publish("Capsule controller finished", status="capsules_complete" if result == 0 else "capsules_failed")
                break
            if elapsed - last_update >= 30:
                publish("Capsule progress", elapsed_s=round(elapsed), remaining_s=max(0, round(total - elapsed)),
                        pss_bytes=charged, host_available_bytes=available)
                last_update = elapsed
            time.sleep(poll_s)
    except CapsuleStopped as error:
        result = 2
        publish("Capsule owner stopped", status="stopped_incomplete", termination_reason=str(error))
    except BaseException as error:
        publish("Capsule owner failed", status="stopped_incomplete", termination_reason=type(error).__name__,
                error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            processes.close()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            if old_timer[0]:
                remaining = max(0.001, old_timer[0] - (time.monotonic() - started))
                signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])
            state.update(elapsed_s=round(time.monotonic() - started, 3), native_running=False, running_models=[])
            save(prep / "status.json", state)
            save(prep / "controller-result.json", {
                "exit_code": None if native is None else native.poll(), "status": state["status"],
                "capsule_complete": state["capsule_complete"], "external_full_models_complete": False,
                "phase1_complete": False, "capsule_elapsed_s": state["elapsed_s"],
                "preparation_wall_s": state["preparation_wall_s"],
            })
    return result
