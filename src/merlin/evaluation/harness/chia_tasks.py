"""Caller-owned Chia task references, with bounded public-Ray cancellation observation.

This owns only returned ObjectRefs, never a borrowed cluster or native process tree.
Task cancellation acknowledgement does NOT prove subprocess descendants have stopped.
No retries, private Chia fields, forced worker kills or upstream PID-registry hooks.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from contextlib import contextmanager


class TaskCleanupIncomplete(RuntimeError):
    """A dispatch outcome is unknown or cancellation was not acknowledged in time."""


class _Tasks:
    def __init__(self, run, timeout):
        import ray

        self._ray, self._run, self._timeout = ray, run, timeout
        self._pending = {}
        self._rows = []
        self._errors = []
        self._path = run.run_dir / "chia/tasks.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._document = {
            "schema": "merlin.chia-task-ownership.v1",
            "native_descendants": "not_verified",
            "tasks": self._rows,
            "cleanup_errors": self._errors,
            "cleanup_complete": False,
        }
        # A run owns one task group. Never overwrite a previous scope's evidence.
        with self._path.open("x") as stream:
            json.dump(self._document, stream, indent=2)

    def _write(self):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=self._path.parent, delete=False) as stream:
                temporary = stream.name
                json.dump(self._document, stream, indent=2)
                stream.write("\n")
            os.replace(temporary, self._path)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)

    def submit(self, launch, /, *args, **kwargs):
        """Dispatch once and immediately own its raw ObjectRef; never retry work."""
        row = {"submission": len(self._rows), "state": "dispatching"}
        self._rows.append(row)
        self._write()
        primary = None
        try:
            ref = launch(*args, **kwargs)
            if not isinstance(ref, self._ray.ObjectRef):
                raise TypeError("Chia task ownership requires a raw public Ray ObjectRef")
            if any(previous.get("task_ref") == ref.hex() for previous in self._rows[:-1]):
                raise ValueError("task reference is already owned by this group")
            self._pending[ref] = row
            row.update(task_ref=ref.hex(), state="submitted")
        except BaseException as error:
            # A raising dispatcher may already have submitted work without returning its handle.
            row.update(state="dispatch_unknown", error=type(error).__name__)
            primary = error
            raise
        finally:
            self._persist(primary)
        return ref

    def _persist(self, primary):
        try:
            self._write()
        except BaseException as error:
            self._run.mark_failed()
            if primary is None:
                raise
            primary.add_note(f"Chia task receipt also failed: {type(error).__name__}: {error}")

    def _terminal(self, ref, *, value=None, error=None):
        row = self._pending.pop(ref)
        row["state"] = "failed" if error is not None else "returned"
        if error is not None:
            row["error"] = type(error).__name__
            self._run.mark_failed()
        elif isinstance(value, dict) and isinstance(value.get("returncode"), int):
            row["returncode"] = value["returncode"]

    def get(self, ref):
        """Keep Chia's public unwrapping/profiler semantics and the original result/error."""
        from chia.base.ChiaFunction import get

        if ref not in self._pending:
            raise ValueError("task reference is not pending in this caller-owned group")
        primary = None
        try:
            value = get(ref)
        except BaseException as error:
            primary = error
            # A driver interruption/timeout is not a terminal task acknowledgement.
            try:
                ready, _ = self._ray.wait([ref], num_returns=1, timeout=0)
                if ref in ready:
                    self._terminal(ref, error=error)
            except BaseException as observation_error:
                error.add_note(f"Chia task-state observation failed: {type(observation_error).__name__}")
            raise
        else:
            self._terminal(ref, value=value)
            return value
        finally:
            self._persist(primary)

    def _observe(self, timeout):
        if not self._pending:
            return
        ready, _ = self._ray.wait(list(self._pending), num_returns=len(self._pending), timeout=timeout)
        for ref in ready:
            try:
                value = self._ray.get(ref, timeout=0)
            except Exception as error:
                self._terminal(ref, error=error)
            else:
                # Cleanup observes readiness only. It never decodes private Chia profiler wrappers
                # or feeds a raw wrapped value back to normal callers.
                self._terminal(ref, value=value)

    def _finish(self):
        deadline = time.monotonic() + self._timeout
        if self._pending:
            self._run.mark_failed()  # No implicit successful fire-and-forget work.
        try:
            self._observe(0)
        except BaseException as error:
            self._errors.append({"operation": "observe", "error": type(error).__name__})
        for ref, row in list(self._pending.items()):
            self._run.mark_failed()
            row["cancel_requested"] = True
            try:
                self._ray.cancel(ref, force=False, recursive=True)
            except BaseException as error:
                self._errors.append({"operation": "cancel", "task_ref": ref.hex(), "error": type(error).__name__})
        try:
            self._observe(max(0.0, deadline - time.monotonic()))
        except BaseException as error:
            self._errors.append({"operation": "drain", "error": type(error).__name__})
        for row in self._pending.values():
            row["state"] = "cancel_unacknowledged"
        unknown = any(row["state"] == "dispatch_unknown" for row in self._rows)
        complete = not (self._pending or unknown or self._errors)
        self._document["cleanup_complete"] = complete
        self._write()
        if not complete:
            self._run.mark_failed()
            raise TaskCleanupIncomplete(
                f"Chia task cleanup not acknowledged or dispatch outcome unknown; inspect {self._path}. "
                "Native descendant cleanup is not verified."
            )


@contextmanager
def chia_tasks(run, *, teardown_timeout_s=10.0):
    """Own a caller's task refs inside chia_run; drain before its collector is stopped.

    The shared deadline bounds public Ray wait/get observation, not task execution.
    Cancellation requests are cooperative. Non-acknowledgement is durable failed evidence,
    not a claim that remote work stopped. Borrowed peers are never enumerated or cancelled.
    """
    timeout = float(teardown_timeout_s)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("teardown timeout must be positive finite seconds")
    tasks = _Tasks(run, timeout)
    primary = None
    try:
        yield tasks
    except BaseException as error:
        primary = error
        run.mark_failed()
        raise
    finally:
        try:
            tasks._finish()
        except BaseException as error:
            run.mark_failed()
            if primary is None:
                raise
            primary.add_note(f"Chia task cleanup also failed: {type(error).__name__}: {error}")
