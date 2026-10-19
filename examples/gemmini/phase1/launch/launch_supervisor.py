"""Bound the controller and parallel full-model checks as one operator campaign."""

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import full_model_verification as F


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_table() -> dict[int, tuple[int, int]]:
    result = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                result[int(path.name)] = (int(fields[1]), int(fields[19]))
            except (OSError, ValueError, IndexError):
                pass
    return result


class Processes:
    """Remember start ticks and ownership even after an observed child is reparented."""

    def __init__(self):
        self.children = []
        self.known = {}

    def start(self, argv, path: Path, *, role: str, cwd: Path, env: dict, cpus: set[int]):
        stream = path.open("xb")
        try:
            child = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=lambda: os.sched_setaffinity(0, cpus),
            )
        except BaseException:
            stream.close()
            raise
        row = {"process": child, "stream": stream, "role": role, "argv": argv}
        self.children.append(row)
        record = process_table().get(child.pid)
        if record is not None:
            self.known[child.pid] = (record[1], role)
        return row

    def owned(self, role: str | None = None) -> dict[int, tuple[int, int]]:
        current = process_table()
        selected = {
            pid: owner for pid, (ticks, owner) in self.known.items() if pid in current and current[pid][1] == ticks
        }
        while True:
            extra = {
                pid: selected[parent]
                for pid, (parent, _) in current.items()
                if parent in selected and pid not in selected
            }
            if not extra:
                break
            selected.update(extra)
        self.known.update({pid: (current[pid][1], owner) for pid, owner in selected.items()})
        return {pid: current[pid] for pid, owner in selected.items() if role is None or owner == role}

    def stop(self, role: str | None = None) -> None:
        records = self.owned(role)
        for signum in (signal.SIGTERM, signal.SIGKILL):
            current = process_table()
            for pid, (_, ticks) in records.items():
                if pid in current and current[pid][1] == ticks:
                    try:
                        os.kill(pid, signum)
                    except ProcessLookupError:
                        pass
            if signum == signal.SIGTERM and records:
                time.sleep(0.25)
                records.update(self.owned(role))
        for row in self.children:
            if role is None or row["role"] == role:
                try:
                    row["process"].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    row["process"].kill()
                    row["process"].wait()

    def close(self):
        self.stop()
        for row in self.children:
            row["stream"].close()


def run(launch, prep: Path, started: float, *, memory, poll_s: float = 5) -> int:
    document = launch.document
    total = document["budgets"]["supervisor_backstop_s"]
    reserve = int(launch.variables["SUPERVISOR_FINAL_RESERVE_S"])
    limit = int(launch.variables["SUPERVISOR_PSS_MAX_BYTES"])
    headroom = int(launch.variables["SUPERVISOR_HOST_HEADROOM_BYTES"])
    if not 0 < reserve < total or min(limit, headroom) <= 0:
        raise ValueError("invalid supervisor time or memory policy")
    if time.monotonic() - started >= total - reserve:
        raise ValueError("preparation exhausted the authoring allocation")
    from merlin_experiments.phase1.launch_config import Prepend

    env = {key: value for key, value in os.environ.items() if not key.startswith("MERLIN_")}
    for name, value in {**launch.environment, **launch.launch_environment}.items():
        env[name] = value.prefix + (":" + env[name] if env.get(name) else "") if isinstance(value, Prepend) else value
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    repo = launch.path(document["inputs"]["repo"])
    run_dir = Path(launch.variables["RUN"])
    workspace = (
        repo
        / "out/build/agent-workspaces/phase1"
        / document["identity"]["target"]
        / document["identity"]["arm"]
        / run_dir.name
        / "workspace"
    )
    assigned = (
        set(document["cpu"]["author"]["cpus"])
        | set(document["cpu"]["compute_pool"]["cpus"])
        | {int(cpu) for cpu in launch.variables["REFERENCE_CPUS"].split(",")}
    )
    policy = document.get("full_model_verification")
    selected = F.selection(launch) if policy else None
    identity = F.selected_identity(launch) if policy else None
    processes = Processes()
    state = {
        "status": "native_controller_starting",
        "phase1_complete": False,
        "models": {},
        "snapshot": None,
        "full_model_execution_complete": False,
        "formal_certification": "native evidence required separately",
        "absolute_deadline": (
            dt.datetime.now(dt.UTC) + dt.timedelta(seconds=total - (time.monotonic() - started))
        ).isoformat(),
    }
    attempts = 0
    stable = None
    last_update = 0
    current = None
    stopping = None

    def publish(message: str, **fields):
        record = {"at": dt.datetime.now(dt.UTC).isoformat(), "message": message, **fields}
        with (prep / "live.log").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        state.update(fields)
        state["updated_at"] = record["at"]
        save(prep / "status.json", state)

    def interrupt(signum, _frame):
        raise RuntimeError(f"campaign supervisor signal {signum}")

    previous = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        native = processes.start(
            launch.argv,
            prep / "phase1-native.log",
            role="native",
            cwd=Path(launch.expand(document["launcher"].get("cwd", str(repo)))),
            env=env,
            cpus=set(launch.manager_cpus),
        )
        native_process = native["process"]
        save(
            prep / "controller.json",
            {
                "pid": native_process.pid,
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "start_ticks": processes.known.get(native_process.pid, (None, None))[0],
                "started_at": dt.datetime.now(dt.UTC).isoformat(),
            },
        )
        publish(
            "Fresh empty-origin controller started",
            controller_pid=native_process.pid,
            status="authoring_or_native_preparation",
        )
        while True:
            waiting_snapshot = False
            elapsed = time.monotonic() - started
            state["remaining_s"] = max(0, round(total - elapsed))
            records = processes.owned()
            charged, available = memory(records)
            with (prep / "memory.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "elapsed_s": round(elapsed, 1),
                            "pss_bytes": charged,
                            "host_available_bytes": available,
                            "processes": len(records),
                        }
                    )
                    + "\n"
                )
            if elapsed >= total or charged > limit or available < headroom:
                stopping = "campaign_deadline" if elapsed >= total else "campaign_memory_guard"
                break
            native_records = processes.owned("native")
            if elapsed >= total - reserve:
                for pid, (_, ticks) in native_records.items():
                    try:
                        if (Path("/proc") / str(pid) / "comm").read_text().strip() == "codex" and process_table().get(
                            pid, (None, None)
                        )[1] == ticks:
                            os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            if policy:
                try:
                    author_digest = F.hash_tree(workspace / "submission").get("sha256")
                except (OSError, ValueError):
                    author_digest = None
                if current and author_digest != current["package"]["hash_tree_sha256"]:
                    processes.stop("model")
                    publish(
                        "Author changed compiler; old model checks superseded", status="awaiting_new_author_snapshot"
                    )
                    current, stable = None, None
                    state.update(models={}, full_model_execution_complete=False, correspondence={})
                snapshot = (
                    F.native_snapshot(run_dir, workspace / "submission")
                    if F.terminal_author(run_dir, native_records)
                    else None
                )
                waiting_snapshot = bool(snapshot and snapshot != stable and not current)
                if (
                    snapshot
                    and snapshot == stable
                    and (not current or current["snapshot"]["sha256"] != snapshot["sha256"])
                ):
                    remaining = int(total - elapsed) - 10
                    if remaining > 60:
                        attempts += 1
                        folder = prep / f"full-model-checks-{attempts:02d}"
                        folder.mkdir()
                        package = F.copy_snapshot(run_dir, snapshot, folder / "compiler")
                        children = {}
                        for name in selected["names"]:
                            report = folder / f"{name}.json"
                            work = repo / "out/build/model_check/phase1-launch" / run_dir.name / str(attempts) / name
                            argv = F.model_command(launch, name, Path(package["path"]), report, work, remaining)
                            child = processes.start(
                                argv, folder / f"{name}.log", role="model", cwd=repo, env=env, cpus=assigned
                            )
                            child["report"] = report
                            children[name] = child
                            save(
                                folder / f"{name}-pin.json",
                                {
                                    "selected_identity": identity,
                                    "submission": package,
                                    "argv": argv,
                                    "work": str(work),
                                    "timeout_s": remaining,
                                    "pid": child["process"].pid,
                                },
                            )
                        current = {"snapshot": snapshot, "package": package, "children": children}
                        save(
                            folder / "native-package-pin.json",
                            {
                                **snapshot,
                                "submission": package,
                                "native_ledger": "pending",
                                "native_final_publication": "pending",
                            },
                        )
                        publish(
                            "Full-model checks started in parallel with native grading",
                            status="native_grading_and_full_models",
                            snapshot={**snapshot, "submission": package},
                            reports_dir=str(folder),
                        )
                stable = snapshot
                if current:
                    rows = {}
                    for name, child in current["children"].items():
                        try:
                            report = F.read(child["report"]) if child["report"].is_file() else {}
                        except (OSError, ValueError):
                            report = {}  # The ordinary report writer may still be publishing its final JSON.
                        complete = child["process"].poll() == 0 and F.report_complete(
                            report, selected["models"].model(name), policy, current["package"]
                        )
                        rows[name] = {
                            "pid": child["process"].pid,
                            "exit_code": child["process"].poll(),
                            "report": str(child["report"]),
                            "full_chain_complete": complete,
                            "gates": report.get("gates", {}),
                        }
                    correspondence = F.final_correspondence(run_dir, current["snapshot"], current["package"])
                    state.update(models=rows, correspondence=correspondence)
                    state["full_model_execution_complete"] = all(
                        row["full_chain_complete"] for row in rows.values()
                    ) and correspondence == {
                        "native_oot_commit_verified": True,
                        "native_ledger": "exact",
                        "native_final_publication": "exact",
                        "frozen_copy_unchanged": True,
                    }
            native_code = native_process.poll()
            model_running = current and any(row["process"].poll() is None for row in current["children"].values())
            if native_code is not None and not model_running and not waiting_snapshot:
                state["controller_exit_code"] = native_code
                break
            if elapsed - last_update >= 30:
                publish(
                    "Campaign progress",
                    elapsed_s=round(elapsed),
                    native_running=native_code is None,
                    running_models=[
                        name
                        for name, row in (current["children"].items() if current else ())
                        if row["process"].poll() is None
                    ],
                    pss_bytes=charged,
                    host_available_bytes=available,
                )
                last_update = elapsed
            time.sleep(poll_s)
        code = native_process.poll()
        save(
            prep / "controller-result.json",
            {
                "exit_code": code,
                "status": "native_controller_finished" if code is not None else "campaign_stopped",
                "phase1_complete": False,
            },
        )
        publish(
            "Campaign supervisor ended",
            status="stopped_incomplete" if stopping else "campaign_processes_finished",
            termination_reason=stopping,
            elapsed_s=round(time.monotonic() - started),
            native_running=False,
            running_models=[],
        )
        if stopping:
            return 2
        return int(code or (1 if policy and not state["full_model_execution_complete"] else 0))
    except BaseException as error:
        processes.stop()
        publish(
            "Campaign supervisor interrupted",
            status="stopped_incomplete",
            termination_reason=type(error).__name__,
            native_running=False,
            running_models=[],
            full_model_execution_complete=False,
        )
        raise
    finally:
        processes.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
