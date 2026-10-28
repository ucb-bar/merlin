"""Run proof checks beside a compile and stop the compile, with evidence, if one fails.

The compile proceeds as if every check will pass. Each submitted check runs in its own owned worker
process (``workers`` at a time, default one, low priority); a check that is not ``accepted`` records a failure, prints
which check failed, why, and where its exact input and output are, and interrupts the compile.
Nothing here decides whether a result is correct: it only schedules checks, never treats a missing
verdict as a pass, and never leaves a worker running after ``close``.

Verdict kinds: ``accepted``; ``unsupported`` (outside the modelled domain, nothing was proved);
``not_established`` (the checker ran and did not accept: an optimizer or projection bug);
``unavailable`` (the checker could not run, crashed, or ran out of budget).
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.owned_group import close_owned_pipes, start_owned, stop_owned

from . import gate_worker

WORKER_MODULE = gate_worker.__name__

VERDICTS = ("accepted", "unsupported", "not_established", "unavailable")
_LOG_TAIL = 3000


class GateFailure(RuntimeError):
    """A speculative check did not accept; ``report`` holds the evidence."""

    def __init__(self, report: dict[str, Any]):
        super().__init__(format_failure(report))
        self.report = report


@dataclass(frozen=True)
class Job:
    id: str
    directory: Path
    check: str
    label: str
    params: dict[str, Any]


def first_difference(before: str, after: str) -> dict[str, Any] | None:
    """The first line at which two texts differ, or None when they are equal."""
    for number, (left, right) in enumerate(itertools.zip_longest(before.splitlines(), after.splitlines()), start=1):
        if left != right:
            return {"line": number, "before": left, "after": right}
    return None


def format_failure(report: dict[str, Any]) -> str:
    lines = [
        f"[merlin-gate] speculative check failed: {report['check']} ({report.get('label') or 'unlabelled'})",
        f"  verdict : {report['verdict']}",
        f"  reason  : {report['reason']}",
    ]
    if report.get("before"):
        lines.append(f"  before: {report['before']}")
    if report.get("after"):
        lines.append(f"  after:  {report['after']}")
    difference = report.get("first_difference")
    if difference:
        lines += [
            f"  first textual change at line {difference['line']} (the rewrite, not necessarily the error):",
            f"    - {difference['before']}",
            f"    + {difference['after']}",
        ]
    lines.append(f"  evidence: {report['directory']} (request.json, verdict.json, worker.log)")
    if report.get("log_tail"):
        lines += ["  worker log (tail):", *("    " + line for line in report["log_tail"].splitlines()[-12:])]
    return "\n".join(lines)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text)[:40] or "job"


class SpeculativeGate:
    """Owned-process runner for gate checks, ``workers`` at a time (default one: a shared host).
    Not reusable after a failure or ``close``."""

    def __init__(
        self,
        directory: Path,
        *,
        interrupt: bool = False,
        job_timeout: float | None = None,
        worker_path: Sequence[Path] = (),
        nice: int = 19,
        aliases: dict[str, str] | None = None,
        workers: int = 1,
    ):
        if type(workers) is not int or workers < 1:
            raise ValueError("a speculative gate needs at least one worker")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.verdicts: list[dict[str, Any]] = []
        self._interrupt = interrupt
        self._job_timeout = job_timeout
        self._worker_path = [str(p) for p in worker_path]
        self._nice = nice
        self._aliases = dict(aliases or {})
        self._workers = workers
        self._queue: queue.Queue[Job | None] = queue.Queue()
        self._condition = threading.Condition()
        self._threads: list[threading.Thread] = []
        self._processes: set[subprocess.Popen] = set()
        self._pending = 0
        self._submitted = 0
        self._counter = itertools.count(1)
        self._failure: GateFailure | None = None
        self._closed = False
        self._guarding = 0
        self._main_guards = 0  # only the main thread can be interrupted by a signal
        self._interrupt_scheduled = False
        self._interrupt_sent = threading.Event()

    # -- submission ---------------------------------------------------------------------------
    def submit(
        self,
        check: str,
        *,
        before: str | None = None,
        after: str | None = None,
        params: dict[str, Any] | None = None,
        label: str = "",
    ) -> Job:
        """Queue one check; its exact input and output texts are written beside the request."""
        self.poll()
        check = self._aliases.get(check, check)
        with self._condition:
            if self._closed:
                raise RuntimeError("the speculative gate is closed")
            job_id = f"{next(self._counter):04d}-{_slug(label or check)}"
            self._pending += 1
            self._submitted += 1
        directory = self.directory / job_id
        directory.mkdir(parents=True)
        request: dict[str, Any] = {
            "check": check,
            "label": label,
            "params": params or {},
            "directory": str(directory),
            "before": None,
            "after": None,
        }
        for key, text in (("before", before), ("after", after)):
            if text is not None:
                (directory / f"{key}.mlir").write_text(text)
                request[key] = str(directory / f"{key}.mlir")
        (directory / "request.json").write_text(json.dumps(request, indent=2, sort_keys=True))
        job = Job(job_id, directory, check, label, request["params"])
        self._queue.put(job)
        self._start()
        return job

    def _start(self) -> None:
        with self._condition:
            if not self._threads:
                self._threads = [
                    threading.Thread(target=self._run, name=f"merlin-gate-{i}", daemon=True)
                    for i in range(self._workers)
                ]
                for thread in self._threads:
                    thread.start()

    # -- results ------------------------------------------------------------------------------
    def poll(self) -> None:
        """Raise the recorded failure, if a finished check has failed."""
        if self._failure is not None:
            raise self._failure

    def join(self, timeout: float | None = None) -> None:
        """Wait for every submitted check; raise if any did not accept."""
        with self._condition:
            done = self._condition.wait_for(lambda: self._pending == 0 or self._failure is not None, timeout)
        self.poll()
        if not done:
            raise TimeoutError("speculative gate checks did not finish in time")
        with self._condition:
            missing = self._submitted - len(self.verdicts)
        if missing and not self._closed:
            # Success needs a recorded verdict for every submission; a dropped job is not a pass.
            self._fail(
                {
                    "verdict": "unavailable",
                    "check": "gate",
                    "label": "scheduler",
                    "reason": f"no verdict was recorded for {missing} submitted check(s)",
                    "detail": {},
                    "directory": str(self.directory),
                    "before": None,
                    "after": None,
                    "first_difference": None,
                    "log_tail": "",
                }
            )
            self.poll()

    @contextmanager
    def guard(self) -> Iterator[SpeculativeGate]:
        """Wrap the compile: a failure interrupts it, pending checks are joined, workers never outlive it."""
        on_main = threading.current_thread() is threading.main_thread()
        with self._condition:
            self._guarding += 1
            self._main_guards += on_main
        try:
            try:
                yield self
                self.join()
            except KeyboardInterrupt:
                if self._failure is not None:
                    raise self._failure from None
                raise
        finally:
            with self._condition:
                self._guarding -= 1
                self._main_guards -= on_main
            self._absorb_interrupt()
            self.close()

    def _absorb_interrupt(self) -> None:
        """Take an interrupt scheduled for this guard before the guard returns, so it cannot land later."""
        if not self._interrupt_scheduled:
            return
        try:
            self._interrupt_sent.wait(timeout=10)
            time.sleep(0.05)  # a signal already on its way lands here, where it is ours to drop
        except KeyboardInterrupt:
            pass

    @property
    def guarding(self) -> bool:
        """Whether a ``guard()`` is active, so a failed check will be turned into a stop."""
        with self._condition:
            return self._guarding > 0 and not self._closed

    def close(self) -> None:
        """Cancel queued work and kill the running worker's whole process group. Idempotent."""
        with self._condition:
            self._closed = True
            processes, threads = list(self._processes), list(self._threads)
        for process in processes:
            if process.poll() is None:
                stop_owned(process)
        for _ in range(max(len(threads), 1)):
            self._queue.put(None)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=15)

    # -- worker thread ------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                if self._failure is None and not self._closed:
                    try:
                        verdict = self._execute(job)
                    except Exception as error:  # noqa: BLE001 -- a scheduler fault is a failed check, never a pass
                        reason = f"the gate could not run the check: {type(error).__name__}: {error}"
                        verdict = {"verdict": "unavailable", "reason": reason, "detail": {}}
                    if verdict is not None:
                        self._record(job, verdict)
            finally:
                with self._condition:
                    self._pending -= 1
                    self._condition.notify_all()

    def _execute(self, job: Job) -> dict[str, Any] | None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [*self._worker_path, *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
        )
        command = [sys.executable, "-m", WORKER_MODULE, str(job.directory)]
        if self._nice and shutil.which("nice"):
            command = ["nice", "-n", str(self._nice), *command]
        parent_death = shutil.which("setpriv")
        if sys.platform == "linux" and parent_death is None:
            raise RuntimeError("Linux gate workers require util-linux setpriv for parent-death cleanup")
        if parent_death is not None:
            # A compile killed by a signal runs no cleanup; the worker must not outlive it. The
            # worker turns this SIGTERM into a kill of its own group, checker processes included.
            command = [parent_death, "--pdeathsig", "TERM", "--", *command]
            env["MERLIN_GATE_PARENT_PID"] = str(os.getpid())
        verdict_path = job.directory / "verdict.json"
        with (job.directory / "worker.log").open("wb") as log:
            process = start_owned(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            with self._condition:
                self._processes.add(process)
            deadline = None if self._job_timeout is None else time.monotonic() + self._job_timeout
            try:
                while process.poll() is None:
                    if self._closed:
                        stop_owned(process)
                        return None
                    if deadline is not None and time.monotonic() > deadline:
                        stop_owned(process)
                        return {
                            "verdict": "unavailable",
                            "reason": f"check exceeded its budget of {self._job_timeout:g}s and was killed",
                            "detail": {},
                        }
                    time.sleep(0.05)
            finally:
                # A worker crash or SIGKILL can leave its checker children alive even
                # after the leader exited. Reap the whole specifically owned group.
                stop_owned(process)
                close_owned_pipes(process)
                with self._condition:
                    self._processes.discard(process)
        if self._closed:
            return None  # cancelled: a worker killed by close() is not a failed check
        try:
            verdict = json.loads(verdict_path.read_text())
            if verdict.get("verdict") not in VERDICTS:
                raise ValueError(f"unknown verdict {verdict.get('verdict')!r}")
        except (OSError, ValueError) as error:
            tail = (job.directory / "worker.log").read_text(errors="replace")[-_LOG_TAIL:]
            return {
                "verdict": "unavailable",
                "reason": f"worker exited {process.returncode} without a usable verdict ({error}): {tail[-400:]}",
                "detail": {},
            }
        if verdict["verdict"] == "accepted" and process.returncode != 0:
            return {
                "verdict": "unavailable",
                "reason": f"worker exited {process.returncode} after accepting",
                "detail": {},
            }
        return verdict

    def _record(self, job: Job, verdict: dict[str, Any]) -> None:
        with self._condition:
            self.verdicts.append({**verdict, "job": job.id, "check": job.check, "label": job.label})
        if verdict["verdict"] == "accepted":
            return
        try:
            report = self._report(job, verdict)
        except Exception as error:  # noqa: BLE001 -- losing the evidence must not lose the failure
            report = {
                "verdict": verdict["verdict"],
                "check": job.check,
                "label": job.label,
                "reason": f"{verdict.get('reason', '')} (evidence unreadable: {error})",
                "detail": verdict.get("detail", {}),
                "directory": str(job.directory),
                "before": None,
                "after": None,
                "first_difference": None,
                "log_tail": "",
            }
        self._fail(report)

    @staticmethod
    def _report(job: Job, verdict: dict[str, Any]) -> dict[str, Any]:
        request = json.loads((job.directory / "request.json").read_text())
        texts = {k: Path(request[k]).read_text() for k in ("before", "after") if request.get(k)}
        log = job.directory / "worker.log"
        return {
            "verdict": verdict["verdict"],
            "check": job.check,
            "label": job.label,
            "reason": verdict.get("reason", ""),
            "detail": verdict.get("detail", {}),
            "directory": str(job.directory),
            "before": request.get("before"),
            "after": request.get("after"),
            "first_difference": first_difference(texts["before"], texts["after"]) if len(texts) == 2 else None,
            "log_tail": log.read_text(errors="replace")[-_LOG_TAIL:] if log.exists() else "",
        }

    def _fail(self, report: dict[str, Any]) -> None:
        """Record the first failure, print its evidence, and interrupt the compile if guarded."""
        failure = GateFailure(report)
        try:
            (Path(report["directory"]) / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True, default=str)
            )
        except OSError:
            pass
        with self._condition:
            if self._failure is not None:
                return
            self._failure = failure
            # A guard owned by another thread learns of the failure when it joins on exit.
            interrupt = self._interrupt and self._main_guards > 0 and not self._closed
            self._interrupt_scheduled = interrupt
            self._condition.notify_all()
        print(str(failure), file=sys.stderr, flush=True)
        if interrupt:
            try:
                self._interrupt_main()
            finally:
                self._interrupt_sent.set()

    @staticmethod
    def _interrupt_main() -> None:
        # Only the default handler turns this into KeyboardInterrupt; never replace a caller's own.
        if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
