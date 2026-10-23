"""Standalone public author action runner; ordinary commands and self-check IPC only.

The automatic author and offline study execute this exact file inside their existing
sandbox. It grants no test, ISA, compilation, correctness or performance authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "merlin.construction_actions.v1"
BOUNDARIES = ("graph", "numeric_layout", "schedule_storage", "isa_lowering", "host_runtime", "integration")


def validate(plan):
    if not isinstance(plan, dict) or set(plan) != {"schema", "actions"} or plan.get("schema") != SCHEMA:
        raise ValueError("expected construction action schema and actions only")
    actions = plan["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= 256:
        raise ValueError("expected 1..256 explicit actions")
    seen = set()
    for row in actions:
        if not isinstance(row, dict) or set(row) - {"id", "capability", "boundary", "kind", "argv", "after"}:
            raise ValueError("unknown action fields")
        name = row.get("id")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 80
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
            or name in seen
        ):
            raise ValueError("action IDs must be unique safe names")
        if row.get("boundary") not in BOUNDARIES or not isinstance(row.get("capability"), str) or not row["capability"]:
            raise ValueError("each action needs a semantic capability and compiler boundary")
        if row.get("kind") not in {"command", "selfcheck"}:
            raise ValueError("only ordinary command and selfcheck actions are supported")
        argv = row.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or "\0" in v for v in argv):
            raise ValueError("action argv must be a nonempty string list")
        if row["kind"] == "selfcheck" and any(
            v in {"--attach", "--list"} or v.startswith(("--attach=", "--list=")) for v in argv
        ):
            raise ValueError("selfcheck actions submit checks; collection is automatic")
        after = row.get("after", [])
        if not isinstance(after, list) or any(not isinstance(v, str) or v not in seen for v in after):
            raise ValueError("after must reference earlier actions; no cycles or implicit global barriers")
        seen.add(name)
    return plan


def load(path):
    data = Path(path).read_bytes()
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("action plan exceeds 2 MiB; use small public scripts")
    return validate(json.loads(data)), hashlib.sha256(data).hexdigest()


def command(row, *, python=sys.executable):
    """The exact same argv in automatic and offline invocation; no mode switch."""
    if row["kind"] == "command":
        return list(row["argv"])
    return [python, "agent_selfcheck.py", *row["argv"], "--wait-budget", "1"]


def _response(stdout):
    # The ordinary shim prints JSON (possibly preceded by progress text).
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char == "{":
            try:
                value, _ = decoder.raw_decode(stdout[index:])
                if isinstance(value, dict) and ("request_id" in value or "selfcheck_request_id" in value):
                    return value
            except ValueError:
                pass
    return {}


def execute(plan, *, output, timeout_s, plan_sha256, command_workers=2):
    """Run independent construction while submitted checks remain in flight.

    Explicit local dependencies gate only their dependent action. A failed capsule
    never establishes a global construction barrier. Numerical verdicts remain the
    broker's output; command return codes are execution status, not capsule passes.
    """
    validate(plan)
    if not 1 <= command_workers <= 16:
        raise ValueError("command_workers must be 1..16")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + timeout_s
    states = {}
    pending = {}
    running = {}
    events = root / "events.jsonl"

    def record(row, state, **details):
        states[row["id"]] = state
        value = {
            "action_id": row["id"],
            "capability": row["capability"],
            "boundary": row["boundary"],
            "kind": row["kind"],
            "state": state,
            "time_ns": time.time_ns(),
            "argv_sha256": hashlib.sha256(json.dumps(command(row)).encode()).hexdigest(),
            **details,
        }
        with events.open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")

    def stop(process):
        # Own descendants of each scripted action too. Detached evaluator jobs
        # belong to the ordinary broker, outside this client process group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def start(row, argv, suffix):
        out = (root / f"{row['id']}.{suffix}.stdout").open("w")
        err = (root / f"{row['id']}.{suffix}.stderr").open("w")
        try:
            process = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
        except OSError as error:
            err.write(str(error) + "\n")
            out.close()
            err.close()
            pending.pop(row["id"], None)
            record(row, "failed", reason="command could not start", error=str(error))
            return
        running[row["id"]] = (row, process, out, err, suffix)
        if suffix == "submit":
            record(row, "running")

    def complete(row, process, out, err, suffix):
        out.close()
        err.close()
        stop(process)
        response = _response((root / f"{row['id']}.{suffix}.stdout").read_text()) if row["kind"] == "selfcheck" else {}
        request = response.get("request_id") or response.get("selfcheck_request_id")
        if process.returncode == 3 and request:
            pending[row["id"]] = (row, request)
            record(row, "pending", request_id=request, returncode=process.returncode)
        else:
            pending.pop(row["id"], None)
            record(
                row,
                "completed" if process.returncode == 0 else "failed",
                returncode=process.returncode,
                request_id=request,
                feedback_file=f"{row['id']}.{suffix}.stdout",
                feedback={
                    key: response[key]
                    for key in (
                        "submission_sha256",
                        "requested_submission_sha256",
                        "all_pass",
                        "counts",
                        "n_pass",
                        "n_capsules",
                        "graded_newer_bytes",
                    )
                    if key in response
                },
            )

    remaining = list(plan["actions"])
    iteration = 0
    try:
        while remaining or pending or running:
            if time.monotonic() >= deadline:
                raise TimeoutError("action study wall budget exhausted")
            progress = False
            # No blocking completion callbacks: inspect every child, then refill.
            for name, work in list(running.items()):
                if work[1].poll() is not None:
                    complete(*work)
                    del running[name]
                    progress = True
            for row in list(remaining):
                deps = [states.get(key) for key in row.get("after", [])]
                if any(state in {"failed", "deferred"} for state in deps):
                    record(row, "deferred", reason="explicit local prerequisite failed")
                    remaining.remove(row)
                    progress = True
                elif all(state == "completed" for state in deps) and len(running) < command_workers:
                    start(row, command(row), "submit")
                    remaining.remove(row)
                    progress = True
            # Keep new construction/test admission ahead of attach calls. An
            # attach is also asynchronous and cannot block another free slot.
            for row, request in list(pending.values()):
                if row["id"] not in running and len(running) < command_workers:
                    iteration += 1
                    start(
                        row,
                        [sys.executable, "agent_selfcheck.py", "--attach", request, "--wait-budget", "1"],
                        f"collect-{iteration}",
                    )
                    progress = True
            if not progress:
                if not running and not pending:
                    raise RuntimeError("unresolved action dependencies")
                time.sleep(0.02)
    except TimeoutError as error:
        unfinished = {row["id"]: row for row in remaining}
        unfinished.update({name: value[0] for name, value in pending.items()})
        unfinished.update({name: value[0] for name, value in running.items()})
        for row in unfinished.values():
            record(row, "unfinished", reason=str(error))
    finally:
        for _, process, out, err, _ in running.values():
            stop(process)
            out.close()
            err.close()
    result = {
        "schema": SCHEMA,
        "plan_sha256": plan_sha256,
        "actions": states,
        "completed": sum(v == "completed" for v in states.values()),
        "capsule_passes": None,
        "qualification_authority": False,
        "note": "Execution status only. Inspect original broker numerical/tier/model results separately.",
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument(
        "--command-workers",
        type=int,
        default=2,
        choices=range(1, 17),
        help="concurrent ordinary commands/client calls; all inherit the same shared CPU reservation",
    )
    args = ap.parse_args(argv)
    if args.timeout <= 0:
        ap.error("--timeout must be positive")
    plan, digest = load(args.plan)

    def interrupted(_signum, _frame):
        raise TimeoutError("construction action runner interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    result = execute(
        plan, output=args.out, timeout_s=args.timeout, plan_sha256=digest, command_workers=args.command_workers
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if all(v == "completed" for v in result["actions"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
