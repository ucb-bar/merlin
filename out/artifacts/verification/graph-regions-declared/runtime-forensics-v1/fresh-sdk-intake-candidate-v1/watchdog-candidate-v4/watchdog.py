"""Own-child Python supervisor. Complete child lifetime, including cleanup.

Only the Popen object created here can be signaled. No supplied PID/group or
external command is accepted. OS scheduling/signal/reap behavior remains trust.
"""
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import fdio
from policy import absolute, require


def budget(value, maximum):
    require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= maximum, "finite watchdog budget")


def supervise(command, workspace, *, seconds, terminate_seconds, reap_seconds):
    budget(seconds, 1800)
    budget(terminate_seconds, 5)
    budget(reap_seconds, 5)
    require(terminate_seconds + reap_seconds <= 5, "at most five seconds termination/reap grace")
    require(type(command) is list and len(command) >= 5 and command[:4] == [sys.executable, "-I", "-S", "-B"]
            and all(type(x) is str for x in command), "one isolated own Python child only")
    absolute(command[4])
    path = absolute(str(workspace))
    parent = fdio.directory(path.parent)
    root = stdout = stderr = None
    process = None
    start = time.monotonic()
    result = {"command": command, "budget_seconds": seconds, "terminate_grace_seconds": terminate_seconds,
              "kill_reap_grace_seconds": reap_seconds, "retry": False, "native_authorized": False,
              "old_holds_cleared": False, "signals": [], "status": "PREPARING"}
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)
        root = os.open(path.name, fdio.DIR, dir_fd=parent)
        fdio.save(root, "PARENT-ATTEMPT.json", {**result, "attempt_consumed": True})
        os.fsync(parent)
        stdout = fdio.new_file(root, "child.stdout.txt")
        stderr = fdio.new_file(root, "child.stderr.txt")
        process = subprocess.Popen(command, cwd=str(path), env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, close_fds=True)
        result["owned_pid"] = process.pid
        fdio.save(root, "OWNED-CHILD.json", {"pid": process.pid, "source": "Popen returned by this one launch", "command": command})
        try:
            code = process.wait(timeout=max(0, seconds - (time.monotonic() - start)))
            result.update(status="CHILD_COMPLETED" if code == 0 else "CHILD_FAILED", child_exit_code=code,
                          child_reaped=True, forced_stop=False)
            if code != 0:
                fdio.save(root, "FIRST-PARENT-FAILURE.json", {**result, "error": "owned child returned nonzero"})
        except subprocess.TimeoutExpired:
            result.update(status="COMPLETE_INTAKE_BUDGET_EXCEEDED", forced_stop=True,
                          error="owned complete intake, including cleanup, exceeded parent budget",
                          in_memory_evidence="Unavailable child buffers/first exception may be lost on forced stop; retained files are not deleted")
            # Preserve parent first error BEFORE signaling; a failed record never permits acceptance.
            try:
                fdio.save(root, "FIRST-PARENT-FAILURE.json", result)
            except BaseException as error:
                result["first_error_preservation_failure"] = fdio.error_details(error)
            stop_owned(process, result, terminate_seconds, reap_seconds)
        result["elapsed_seconds"] = time.monotonic() - start
        os.fsync(stdout)
        os.fsync(stderr)
        fdio.save(root, "PARENT-RESULT.json", result)
        return result
    except BaseException as error:
        result.update(status="PARENT_FIRST_FAILURE", **fdio.error_details(error),
                      in_memory_evidence="Child in-memory evidence may be unavailable; existing files preserved")
        if root is not None:
            try:
                fdio.save(root, "FIRST-PARENT-FAILURE.json", result)
            except BaseException as preservation:
                result["first_error_preservation_failure"] = fdio.error_details(preservation)
        if process is not None and not result.get("stop_attempted") and process.poll() is None:
            stop_owned(process, result, terminate_seconds, reap_seconds)
        if root is not None:
            try:
                fdio.save(root, "PARENT-ERROR-RESULT.json", result)
            except BaseException as preservation:
                result["result_preservation_failure"] = fdio.error_details(preservation)
        raise fdio.IntakeStop(result) from error
    finally:
        for fd in (stderr, stdout, root, parent):
            if fd is not None:
                os.close(fd)


def stop_owned(process, result, terminate_seconds, reap_seconds):
    """Bounded escalation using ONLY this launch's Popen handle; no PID input."""
    result["stop_attempted"] = True
    try:
        process.terminate()
        result["signals"].append({"signal": "SIGTERM", "owned_pid": process.pid})
    except ProcessLookupError:
        pass
    except OSError as error:
        result["terminate_error"] = str(error)
    try:
        result["child_exit_code"] = process.wait(timeout=terminate_seconds)
        result["child_reaped"] = True
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
        result["signals"].append({"signal": "SIGKILL", "owned_pid": process.pid})
    except ProcessLookupError:
        pass
    except OSError as error:
        result["kill_error"] = str(error)
    try:
        result["child_exit_code"] = process.wait(timeout=reap_seconds)
        result["child_reaped"] = True
    except subprocess.TimeoutExpired:
        result.update(child_reaped=False, status="OWNED_CHILD_UNREAPED_HOLD",
                      live_owned_pid=process.pid, no_retry=True)
