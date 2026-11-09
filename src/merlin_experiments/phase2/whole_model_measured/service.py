"""Asynchronous whole-model measurements, keyed by the digest of the exact package bytes measured.

WHY ASYNCHRONOUS.  A whole model takes minutes on a board and hours on an elaborated-RTL simulator.
A loop that waited for each one would spend its agent's session idle; so a measurement is a JOB:
requested, run in a detached process (:mod:`.worker`), and read back whenever it lands.

WHY KEYED BY DIGEST.  A result belongs to the bytes that earned it, never to "the candidate" as it
happens to be now.  A request snapshots the package into the job directory FIRST, hashes the
snapshot, and refuses if it does not hash to the digest the request named -- so an agent editing its
workspace mid-measurement cannot change what is being measured.  A second request for bytes already
measured (or being measured) returns the existing job instead of paying again, and a candidate whose
PROGRAM digest matches an earlier job is aliased to it (documentation edits are not measurements).

WHY DETACHED PROCESSES.  Each job runs as its own session leader writing only into its own directory,
so the process that owns the agent session can exit, crash or be restarted and the measurement still
completes; nothing a result depends on is held in memory.

WHAT REOPENS A JOB.  A job's content-addressed key says "these bytes were tried"; it never says "the
harness that tried them is still correct forever".  A SCREEN_FAILED job whose failure was
infrastructure's and was recorded under a different screen spec is re-screened under this request's
spec; a DONE job whose builder identity changed, or whose infra-caused refusal predates the current
spec or builder, is re-measured -- its earlier attempt archived, never deleted -- and carries THIS
request's builder record forward, so a later request does not reopen it again for nothing.

WHAT HOLDS A BUILD.  A whole-model build writes gigabytes under its job directory and its workers'
TMPDIR; one that dies of ENOSPC halfway costs its time and leaves a refusal that reads like the
candidate's.  Below :data:`MIN_BUILD_FREE_BYTES` (or the section's ``min_build_free_bytes``) on either
filesystem, no worker is started: the store records the hold (``disk_hold.json``, closed into
``disk_holds.jsonl`` when space returns) and each waiting job says why it waits -- deferred, never
refused.  The board path has its own floor (:func:`.machines.disk_refusal`).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_verdict as V

from . import attempts as A
from . import batch as B
from . import gates as G
from . import jobs as J
from . import retention as RET
from .identity import (
    builder_identity,
    job_key,
    key_of,
    load_builder,
    locked,
    now,
    package_digest,
    program_digest,
    read_json,
    sha256_file,
    write_json_atomic,
)

WORKER_MODULE = "merlin_experiments.phase2.whole_model_measured"
#: A job whose worker died without writing a result lost its MEASUREMENT, not its verdict: re-queued
#: this many times, then recorded as lost -- never as a verdict on the candidate.
WORKER_LOSS_REQUEUES = 1

#: Below this many free bytes on the store's or the workers' TMPDIR filesystem, no build starts.
MIN_BUILD_FREE_BYTES = 8 * 1024**3
DISK_HOLD = "disk_hold.json"
DISK_HOLDS = "disk_holds.jsonl"
DISK_LOW = "infra_disk_low"


def build_free_bytes(path: Path) -> int:
    """Free bytes on the filesystem that holds (or will hold) ``path``: its nearest existing ancestor, so a
    TMPDIR not created yet is measured where it will be (the single seam tests replace)."""
    path = Path(path).absolute()
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free


def build_disk_refusal(paths: Mapping[str, Path], *, minimum: int) -> str | None:
    """Why no build may start for lack of disk, or None.  The reason names the filesystem, its free
    bytes and the floor, so it can be read against the threshold it missed."""
    for label, path in paths.items():
        free = build_free_bytes(Path(path))
        if free < minimum:
            return (
                f"{DISK_LOW}: the {label} filesystem at {path} has only {free} byte(s) free, below the "
                f"{minimum} minimum; builds wait for space (not a verdict)"
            )
    return None


def spawn(argv: list[str], **kwargs: Any) -> subprocess.Popen:
    """Start one detached worker or batch runner (the single seam tests replace)."""
    return subprocess.Popen(argv, **kwargs)


def descendant_pids(root: int) -> set[int]:
    """Every descendant of ``root``, by parent PID read from /proc (no name matching)."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
        except OSError:
            continue
        children.setdefault(int(fields[1]), []).append(int(entry.name))
    found, stack = set(), [root]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def alive(pid: Any, owner: Path) -> bool:
    """Liveness by /proc, and only for a process whose command line names ``owner`` (pids are reused)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return str(owner).encode() in cmdline


def _worker_exit(pid: Any) -> str:
    return f"worker pid {pid} is gone; its exit status is not observable from this process (not its parent)"


class MeasurementService:
    """Request, schedule and read whole-model measurements under one store directory.

    ``slots`` bounds how many run at once; ``max_pending`` bounds the queue.  When the queue is full
    the OLDEST not-started request is marked ``superseded`` -- recorded with the digest that displaced
    it, never silently dropped."""

    def __init__(
        self,
        root: Path,
        *,
        target: str,
        builder: str,
        builder_sha256: str | None,
        machine: Mapping[str, Any],
        slots: int = 1,
        max_pending: int = 4,
        timeout_seconds: float = 6 * 3600,
        python: str | None = None,
        environment: Mapping[str, str] | None = None,
        build_options: Mapping[str, Any] | None = None,
        reference: Path | None = None,
        excuse_reference_failures: bool = False,
        pre_measure_check: Mapping[str, Any] | None = None,
        retain: Any = None,
        certifier_root: Path | None = None,
        instruction_policy: Mapping[str, Any] | None = None,
        min_build_free_bytes: int | None = None,
        machine_capabilities: Mapping[str, Any] | None = None,
        exactness: Mapping[str, Any] | None = None,
    ) -> None:
        if slots < 1 or max_pending < 1:
            raise J.ServiceError("slots and max_pending must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.target = target
        # Resolve the builder once at construction, so a bad spec or a wrong pin fails the LAUNCH.
        load_builder(builder, expected_sha256=builder_sha256)
        self.builder = {
            "spec": builder,
            "sha256": builder_sha256,
            "module_identity": builder_identity(builder, builder_sha256),
        }
        self.machine = dict(machine)
        self.slots = int(slots)
        self.max_pending = int(max_pending)
        self.timeout_seconds = float(timeout_seconds)
        self.python = python or sys.executable
        self.environment = dict(environment or {})
        self.build_options = dict(build_options or {})
        self.excuse_reference_failures = bool(excuse_reference_failures)
        self.pre_measure_check = dict(pre_measure_check) if pre_measure_check else None
        self.retain = [str(p) for p in retain] if retain else None
        #: Set by the objective: the package-authored coverage a candidate must keep to reach the board.
        self.coverage_gate: dict[str, Any] | None = None
        #: Set by the objective: computes the gate afresh at every candidate request (see :meth:`request`).
        self.coverage_gate_provider: Callable[[], Mapping[str, Any] | None] | None = None
        self.certifier_root = str(certifier_root) if certifier_root else None
        #: The sealed Phase 0 instruction policy every candidate job carries, so the instruction gate
        #: holds a program to the rule Phase 0 sealed rather than to whatever the roles derive to today.
        self.instruction_policy = dict(instruction_policy) if instruction_policy else None
        self.min_build_free_bytes = int(
            min_build_free_bytes if min_build_free_bytes is not None else MIN_BUILD_FREE_BYTES
        )
        # The reference is bound PER JOB, at request time, to the bytes the file holds then.
        self.reference_path = Path(reference) if reference is not None else None
        #: What this service's machine can do and lacks (:func:`.capabilities.compact`); every job, and so
        #: every result, carries it.
        self.machine_capabilities = dict(machine_capabilities) if machine_capabilities else None
        #: The exactness contract (by value) and the launch's group forms every verdict is graded under
        #: (:func:`merlin.perf.exactness.apply_to_verdict`); None is the default, every form exact.
        self.exactness = dict(exactness) if exactness else None

    def _reference_binding(self) -> dict[str, Any] | None:
        if self.reference_path is None:
            return None
        if not self.reference_path.is_file():
            return {"path": str(self.reference_path), "sha256": None, "state": "not yet available at request time"}
        return {"path": str(self.reference_path), "sha256": sha256_file(self.reference_path)}

    def derive_coverage_gate(self) -> dict[str, Any] | None:
        """The coverage gate the objective would arm, from this store alone: the earliest-requested
        candidate with a build is the seed, priced by the bound reference's own per-group cycles
        (:func:`.feedback.package_authored`, the objective's own pricing).  None while the store has no
        built seed or no reference result -- a gate is never guessed."""
        from . import feedback as F

        if self.reference_path is None or not self.reference_path.is_file():
            return None
        reference = read_json(self.reference_path) or {}
        seeds = sorted(
            (j for j in self.jobs() if not j.get("replicate") and j.get("role", J.ROLE_CANDIDATE) == J.ROLE_CANDIDATE),
            key=lambda j: float(j.get("requested_epoch") or 0),
        )
        for job in seeds:
            found = self.result_by_key(key_of(job))
            if not found or not (found.get("build") or {}).get("groups"):
                continue
            mine = F.package_authored(found, reference)
            if mine.get("priced_share") is None:
                return None
            price = {str(r["group"]): int(r.get("cycles") or 0) for r in V.group_table(reference.get("verdict") or {})}
            return {
                "floor_package_sha256": job["package_sha256"],
                "floor_share": mine["priced_share"],
                "floor_groups": list(mine["groups"]),
                "price": price,
                "derived_by": "store",
            }
        return None

    # ---- requests
    def _reopen_stale(self, job_dir: Path, existing: dict[str, Any], *, exempt: bool) -> dict[str, Any]:
        """Reopen a job whose recorded outcome predates this request's harness (see the module doc)."""
        if existing.get("state") == J.SCREEN_FAILED and existing.get("pre_measure_check") != self.pre_measure_check:
            stale = G.screen_was_infra(read_json(job_dir / "pre_measure_check_result.json"))
        else:
            stale = False
        if existing.get("state") == J.SCREEN_FAILED and (exempt or stale):
            # The screen's refusal is set aside as an attempt of its own (moved whole, never renamed over).
            J.archive_attempt(
                job_dir, "screen_failed_attempt_", why="re-screened: " + ("seed" if exempt else "infra-caused screen")
            )
            existing.update(
                state=J.PENDING,
                screen_exempt=bool(exempt),
                reopened_at=now(),
                builder=dict(self.builder),
                **(
                    {"reopened_infra_snapshot_mismatch": True, "pre_measure_check": self.pre_measure_check}
                    if stale and not exempt
                    else {}
                ),
            )
            write_json_atomic(job_dir / "job.json", existing)
            return existing
        if existing.get("state") != J.DONE:
            return existing
        document = A.annotated(read_json(job_dir / "result.json"), job_dir) or {}
        recorded = ((document.get("builder") or {}).get("module_identity") or {}).get("sha256")
        current = (self.builder.get("module_identity") or {}).get("sha256")
        builder_changed = recorded != current
        # An operator's infra mark (store_admin mark-infra) counts the same as a recognised infra refusal.
        infra = G.is_infra_refusal(document.get("refusal")) or bool(document.get("infra_marked"))
        stale_infra = infra and (existing.get("pre_measure_check") != self.pre_measure_check or builder_changed)
        if not (builder_changed or stale_infra):
            return existing
        attempt = J.archive_attempt(
            job_dir,
            "reopened_attempt_",
            why="re-measured: "
            + ("the builder changed" if builder_changed else "an infra-caused refusal predates this harness"),
        )
        RET.prune_archived_attempt(attempt, existing.get("retain"))
        existing.update(
            state=J.PENDING,
            reopened_at=now(),
            reopened_builder_changed=bool(builder_changed),
            reopened_infra_build_regression=bool(stale_infra),
            pre_measure_check=self.pre_measure_check if existing.get("role") != J.ROLE_REFERENCE else None,
            # CARRY THIS REQUEST'S BUILDER RECORD FORWARD: left stale, the very next request would see the
            # original record as "changed" again and reopen a second time for nothing.
            builder=dict(self.builder),
        )
        for stale_field in ("worker_pid", "dispatched_at", "started_at", "finished_at", "timing_status"):
            existing.pop(stale_field, None)
        write_json_atomic(job_dir / "job.json", existing)
        return existing

    def request(
        self,
        package: Path,
        *,
        label: str = "",
        priority: int = 0,
        role: str = J.ROLE_CANDIDATE,
        replicate: int = 0,
        solo: bool = False,
        screen_exempt: bool = False,
        attribution: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Snapshot ``package`` and enqueue it, or return the job that already covers its bytes.

        ``replicate`` > 0 asks for ANOTHER run of the same bytes, under its own key: a machine whose
        failures are non-deterministic is only evidence of correctness over repeated runs.
        ``attribution`` (``state``, ``round``, ``why``) records who asked, on whichever job covers the
        bytes, BEFORE any poll can promote it (:func:`.jobs.merge_attribution`)."""
        package = Path(package)
        digest = package_digest(package)
        if replicate < 0:
            raise J.ServiceError("a replicate index is non-negative")
        key = job_key(digest, replicate)
        job_dir = self.root / key
        program = program_digest(package)
        if role == J.ROLE_CANDIDATE:
            # ARMED HERE, not on a later poll: the request is what carries the gate into the job. With no
            # objective attached (a one-off or external request, e.g. a cell win transplanted to the
            # whole model), the store derives the same gate from its own seed and bound reference.
            gate = self.coverage_gate_provider() if self.coverage_gate_provider is not None else None
            if gate is None and self.coverage_gate is None:
                gate = self.derive_coverage_gate()
            if gate is not None:
                self.coverage_gate = dict(gate)
        with locked(self.root):
            existing = read_json(job_dir / "job.json")
            # THE FIRST CANDIDATE OF A STORE IS THE SEED, run whatever its screen says: without one run
            # there is no per-group table to attribute from.
            first = not any(
                job.get("role", J.ROLE_CANDIDATE) == J.ROLE_CANDIDATE and key_of(job) != key for job in self.jobs()
            )
            exempt = screen_exempt or (first and role == J.ROLE_CANDIDATE and replicate == 0)
            if existing is not None:
                existing = self._reopen_stale(job_dir, existing, exempt=exempt)
            if existing is not None and existing.get("state") != J.SUPERSEDED:
                if attribution is not None:
                    self.attribute(key, attribution)
                return existing
            if replicate == 0 and role == J.ROLE_CANDIDATE and not solo:
                same = next(
                    (
                        job
                        for job in self.jobs()
                        if job.get("program_sha256") == program
                        and int(job.get("replicate") or 0) == 0
                        and job.get("state") != J.SUPERSEDED
                        and job.get("role", J.ROLE_CANDIDATE) == J.ROLE_CANDIDATE
                    ),
                    None,
                )
                if same is not None:
                    aliases = self.root / "aliases"
                    aliases.mkdir(exist_ok=True)
                    write_json_atomic(
                        aliases / f"{digest}.json",
                        {
                            "package_sha256": digest,
                            "program_sha256": program,
                            "same_program_as": key_of(same),
                            "at": now(),
                        },
                    )
                    if attribution is not None:
                        self.attribute(key_of(same), attribution)
                    return {**same, "aliased_from": digest}
            staging = self.root / f".staging_{key}_{os.getpid()}"
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(package, staging / "package", symlinks=True)
            snapshot = package_digest(staging / "package")
            if snapshot != digest:
                shutil.rmtree(staging)
                raise J.ServiceError(
                    f"{package} changed while it was being snapshotted ({digest} -> {snapshot}); request again"
                )
            prior = read_json(job_dir / J.ATTRIBUTION_FILE)  # who earned these bytes outlives a supersede
            if job_dir.exists():
                # A superseded job's directory is replaced by the new request -- but never its history: its
                # own result is set aside as an attempt, and every attempt moves into the new directory.
                if (job_dir / J.RESULT_FILE).is_file():
                    J.archive_attempt(job_dir, "superseded_attempt_", why="the job was requested again")
                if (job_dir / J.ATTEMPTS_DIR).is_dir():
                    shutil.move(str(job_dir / J.ATTEMPTS_DIR), str(staging / J.ATTEMPTS_DIR))
                shutil.rmtree(job_dir)
            staging.rename(job_dir)
            record = (
                J.merge_attribution(prior, attribution, at=now(), created=prior is None)
                if attribution is not None
                else prior
            )
            if record is not None:
                write_json_atomic(job_dir / J.ATTRIBUTION_FILE, record)
            job = {
                "schema": J.JOB_SCHEMA,
                "job_key": key,
                "replicate": int(replicate),
                "package_sha256": digest,
                "program_sha256": program,
                "source": str(package),
                "label": label,
                "priority": int(priority),
                "requested_at": now(),
                "requested_epoch": time.time(),
                "state": J.SCREENING if (self.pre_measure_check and role != J.ROLE_REFERENCE) else J.PENDING,
                "target": self.target,
                "builder": dict(self.builder),
                "machine": dict(self.machine),
                "build_options": dict(self.build_options),
                "reference": self._reference_binding(),
                "excuse_reference_failures": self.excuse_reference_failures,
                "pre_measure_check": self.pre_measure_check if role != J.ROLE_REFERENCE else None,
                "role": role,
                "solo": bool(solo),
                "retain": self.retain,
                "coverage_gate": self.coverage_gate if role == J.ROLE_CANDIDATE else None,
                "certifier_root": self.certifier_root,
                "timeout_seconds": self.timeout_seconds,
                "instruction_policy": self.instruction_policy if role != J.ROLE_REFERENCE else None,
            }
            job["machine_capabilities"] = self.machine_capabilities
            job["exactness"] = self.exactness
            write_json_atomic(job_dir / "job.json", job)
        if job["state"] == J.SCREENING:
            job = self._screen(job_dir, job, exempt=exempt)
        with locked(self.root):
            self._bound_queue(protect=key)
        self.poll()
        return read_json(job_dir / "job.json") or job

    def attribution(self, key: str) -> dict[str, Any] | None:
        """Who earned the bytes of job ``key`` (:data:`.jobs.ATTRIBUTION_FILE`), or None (the harness)."""
        return read_json(self.root / key / J.ATTRIBUTION_FILE)

    def attributable(self, key: str) -> bool:
        return J.attributable(self.attribution(key))

    def attribute(self, key: str, change: Mapping[str, Any]) -> dict[str, Any]:
        """Record one attribution ``change`` on the EXISTING job ``key`` and return its record."""
        job_dir = self.root / key
        if not (job_dir / "job.json").is_file():
            raise J.ServiceError(f"no job {key} to attribute")
        with locked(job_dir):
            record = J.merge_attribution(self.attribution(key), change, at=now(), created=False)
            write_json_atomic(job_dir / J.ATTRIBUTION_FILE, record)
        return record

    def _screen(self, job_dir: Path, job: dict[str, Any], *, exempt: bool) -> dict[str, Any]:
        """The declared capsule screen, at request time, on the snapshot."""
        try:
            check = G.pre_measure_check(job, job_dir / "package", job_dir)
        except Exception as exc:  # noqa: BLE001 -- an unrunnable screen is recorded, never a pass
            check = {
                "label": "pre-measure check",
                "required": True,
                "passed": False,
                "summary": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        write_json_atomic(job_dir / "pre_measure_check_result.json", check or {})
        finalize = False
        with locked(job_dir):
            fresh = read_json(job_dir / "job.json") or job
            fresh["screen"] = {
                k: (check or {}).get(k) for k in ("label", "required", "passed", "summary", "wall_seconds")
            }
            failed = bool(check) and check.get("required") and not check.get("passed")
            if failed and not exempt:
                fresh.update(state=J.SCREEN_FAILED, finished_at=now(), timing_status=V.TIMING_REFUSED)
                finalize = True
                J.write_result(
                    job_dir,
                    J.refused(
                        fresh,
                        f"screen_failed: the required capsule screen failed ({(check or {}).get('summary')}); "
                        "these bytes were not built or run",
                        screen_failed=True,
                        pre_measure_check=check,
                    ),
                )
            else:
                fresh.update(state=J.PENDING, screen_exempt=bool(failed and exempt))
            write_json_atomic(job_dir / "job.json", fresh)
        if finalize:
            RET.finalize_if_allowed(self.root, job_dir)
        return fresh

    def _bound_queue(self, *, protect: str) -> None:
        pending = [job for job in self.jobs() if job.get("state") == J.PENDING]
        excess = len(pending) - self.max_pending
        if excess <= 0:
            return
        pending.sort(key=lambda job: (int(job.get("priority") or 0), float(job.get("requested_epoch") or 0)))
        for job in pending:
            if excess <= 0:
                break
            if key_of(job) == protect or int(job.get("replicate") or 0) > 0:
                continue
            job.update(state=J.SUPERSEDED, superseded_at=now(), superseded_by=protect)
            write_json_atomic(self.root / key_of(job) / "job.json", job)
            excess -= 1

    # ---- scheduling
    def _account_running(self, job: dict[str, Any]) -> bool:
        """Settle one RUNNING job whose process may be gone.  True when it still holds a slot."""
        job_dir = self.root / key_of(job)
        if job.get("batch") and alive(job.get("worker_pid"), self.root):
            return False  # on the board inside a live batch; it holds no worker slot
        runner_losses = list(job.get("batch_runner_losses") or [])
        if (
            job.get("batch")
            and not (job_dir / "result.json").is_file()
            and len(runner_losses) < B.BATCH_RUNNER_REQUEUES
            and not B.unlinkable(job_dir)
        ):
            # THE BATCH RUNNER EXITED, NOT THE BUILD: its board objects are intact, so it goes back to
            # the board queue to be linked again, not rebuilt from nothing.
            job.update(
                state=J.BOARD,
                board_ready_epoch=time.time(),
                batch_runner_losses=[
                    *runner_losses,
                    {"at": now(), "batch": job.get("batch"), "exit": _worker_exit(job.get("worker_pid"))},
                ],
                notice="infra_batch_runner_lost: the board run's controller exited before it finished; "
                "re-queued for the board (not a verdict)",
            )
            for stale in ("batch", "worker_pid"):
                job.pop(stale, None)
            write_json_atomic(job_dir / "job.json", job)
            return False
        if alive(job.get("worker_pid"), job_dir):
            return True
        if (job_dir / "result.json").is_file():
            job.update(state=J.DONE, finished_at=job.get("finished_at") or now())
        else:
            loss = {"at": now(), "worker_pid": job.get("worker_pid"), "exit": _worker_exit(job.get("worker_pid"))}
            losses = list(job.get("worker_losses") or [])
            if len(losses) < WORKER_LOSS_REQUEUES:
                attempt = J.archive_attempt(job_dir, "lost_attempt_", why="its worker exited without a result")
                RET.prune_archived_attempt(attempt, job.get("retain"))
                loss["moved_to"] = str(attempt)
                job.update(
                    state=J.PENDING,
                    worker_losses=[*losses, loss],
                    requeued_at=now(),
                    # THE SPEC A REQUEUED JOB RUNS UNDER IS THIS SERVICE'S OWN, never the dead attempt's:
                    # the screen spec names snapshot-specific paths and is a request-time input.
                    pre_measure_check=self.pre_measure_check if job.get("role") != J.ROLE_REFERENCE else None,
                    notice=f"{J.INFRA_WORKER_LOST}: measurement lost to host pressure (its worker exited "
                    "without a result), re-queued",
                )
                for stale in ("worker_pid", "dispatched_at", "started_at"):
                    job.pop(stale, None)
            else:
                job.update(
                    state=J.FAILED,
                    failed_at=now(),
                    worker_losses=[*losses, loss],
                    failure=f"{J.INFRA_WORKER_LOST}: measurement lost to host pressure {len(losses) + 1} "
                    "times (its worker exited without a result); not a verdict on these bytes",
                )
                J.write_result(
                    job_dir,
                    J.refused(job, job["failure"], infra_worker_lost=True, worker_losses=job["worker_losses"]),
                )
        write_json_atomic(job_dir / "job.json", job)
        return False

    def poll(self) -> list[str]:
        """Start pending jobs while slots are free; settle a dead worker's job.  Returns started keys."""
        started: list[str] = []
        with locked(self.root):
            jobs = self.jobs()
            running = sum(1 for job in jobs if job.get("state") == J.RUNNING and self._account_running(job))
            pending = sorted(
                (job for job in jobs if job.get("state") == J.PENDING),
                key=lambda job: (-int(job.get("priority") or 0), float(job.get("requested_epoch") or 0)),
            )
            held = self._disk_hold(pending) if pending and running < self.slots else None
            for job in pending:
                if running >= self.slots or held:
                    break
                job_dir = self.root / key_of(job)
                if str(job.get("notice") or "").startswith(DISK_LOW):
                    job.pop("notice")
                with (job_dir / "worker.log").open("ab") as log:
                    process = spawn(
                        [self.python, "-m", WORKER_MODULE, "work", str(job_dir)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                        env={**os.environ, **self.environment},
                        cwd=str(job_dir),
                    )
                job.update(state=J.RUNNING, worker_pid=process.pid, dispatched_at=now())
                write_json_atomic(job_dir / "job.json", job)
                running += 1
                started.append(key_of(job))
            if self.machine.get("kind") == "batched":
                self._dispatch_batch()
        return started

    def _worker_tmpdir(self) -> Path:
        return Path(self.environment.get("TMPDIR") or os.environ.get("TMPDIR") or tempfile.gettempdir())

    def _disk_hold(self, pending: list[dict[str, Any]]) -> str | None:
        """The reason builds are held for lack of disk, recorded on the store and on each waiting job;
        None (closing any open hold) when both filesystems have room."""
        hold_path = self.root / DISK_HOLD
        reason = build_disk_refusal(
            {"store": self.root, "worker TMPDIR": self._worker_tmpdir()}, minimum=self.min_build_free_bytes
        )
        if reason is None:
            hold = read_json(hold_path)
            if hold is not None:
                with (self.root / DISK_HOLDS).open("a", encoding="utf-8") as log:
                    log.write(json.dumps({**hold, "closed_at": now()}, sort_keys=True) + "\n")
                hold_path.unlink(missing_ok=True)
            return None
        hold = read_json(hold_path) or {"opened_at": now(), "minimum_free_bytes": self.min_build_free_bytes}
        write_json_atomic(hold_path, {**hold, "reason": reason, "checked_at": now()})
        for job in pending:
            if job.get("notice") != reason:
                job["notice"] = reason
                write_json_atomic(self.root / key_of(job) / "job.json", job)
        return reason

    def _dispatch_batch(self) -> None:
        """Start one batch runner when none is running and the waiting jobs are worth a board job."""
        marker = self.root / "batch_runner.json"
        current = read_json(marker) or {}
        if alive(current.get("pid"), self.root):
            return
        if not B.worth_starting(self.root, self.machine):
            return
        with (self.root / "batch_runner.log").open("ab") as log:
            process = spawn(
                [self.python, "-m", WORKER_MODULE, "batch", str(self.root)],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, **self.environment},
                cwd=str(self.root),
            )
        write_json_atomic(marker, {"pid": process.pid, "started_at": now()})

    def supersede(
        self, key: str, *, reason: str, stop_running: bool, outcome: str = J.SUPERSEDED
    ) -> dict[str, Any] | None:
        """End one job as SUPERSEDED, with ``reason`` in its result.  A RUNNING one is stopped only when
        ``stop_running`` -- its worker and that worker's descendants, by exact PID resolved from /proc,
        each PID recorded.  Terminal jobs are untouched.  Never a board job's queue submission: a started
        board job is never cancelled."""
        job_dir = self.root / key
        with locked(self.root):
            job = read_json(job_dir / "job.json")
            if job is None or job.get("state") in J.TERMINAL:
                return None
            stopped: list[int] = []
            if job.get("state") == J.RUNNING:
                if not stop_running or job.get("batch"):
                    return None
                pid = job.get("worker_pid")
                if isinstance(pid, int) and alive(pid, job_dir):
                    stopped = sorted({pid, *descendant_pids(pid)})
                    for target in stopped:
                        with contextlib.suppress(ProcessLookupError):
                            os.kill(target, signal.SIGTERM)
            job.update(
                state=J.SUPERSEDED,
                superseded_at=now(),
                superseded_reason=reason,
                stopped_pids=stopped,
                ended_as=outcome,
            )
            write_json_atomic(job_dir / "job.json", job)
            ended = J.refused(job, f"{outcome}: {reason}", superseded=True, ended_as=outcome, stopped_pids=stopped)
            try:
                J.write_result(job_dir, ended)
            except J.ResultExists:
                # Its worker finished first: that result stands, and the supersede is kept beside it.
                J.preserve_result(job_dir, ended, why="a supersede that arrived after the job's own result")
        return job

    # ---- reading
    def jobs(self) -> list[dict[str, Any]]:
        return [job for job in (read_json(p) for p in sorted(self.root.glob("*/job.json"))) if job is not None]

    def result(self, digest: str) -> dict[str, Any] | None:
        """The FIRST run's result for these bytes (replicate 0): the one that stands across its attempts
        (:func:`.attempts.effective_result`) -- an infra outcome never hides an earlier verdict."""
        return A.effective_result(self.root / digest)

    def result_by_key(self, key: str | None) -> dict[str, Any] | None:
        return A.effective_result(self.root / key) if key else None

    def alias_of(self, digest: str) -> str | None:
        alias = read_json(self.root / "aliases" / f"{digest}.json")
        return str(alias["same_program_as"]) if alias and alias.get("same_program_as") else None

    def measurement_for(self, digest: str) -> dict[str, Any]:
        """What the loop knows about these exact bytes: a result, or the job's state."""
        if not (self.root / digest).is_dir():
            target = self.alias_of(digest)
            if target is not None:
                return {**self.measurement_for(target), "aliased_from": digest, "same_program_as": target}
        found = self.result(digest)
        if found is not None:
            if found.get("from_attempt"):
                job = read_json(self.root / digest / "job.json") or {}
                if job.get("state") not in J.TERMINAL:
                    found = {
                        **found,
                        "job_state": job.get("state"),
                        "notice": f"re-measuring (attempt {found['from_attempt']['attempt']} is the standing result)",
                    }
            return found
        job = read_json(self.root / digest / "job.json")
        if job is None:
            return {"schema": J.RESULT_SCHEMA, "package_sha256": digest, "timing_status": "UNMEASURED"}
        document = {
            "schema": J.RESULT_SCHEMA,
            "package_sha256": digest,
            "timing_status": J.TIMING_PENDING,
            "job_state": job.get("state"),
            "requested_at": job.get("requested_at"),
            "label": job.get("label"),
            "superseded_by": job.get("superseded_by"),
            "screen": job.get("screen"),
            "notice": job.get("notice"),
        }
        early = self.early_correctness(digest)
        if early:
            document["correctness_first"] = early
        return document

    def early_correctness(self, digest: str) -> dict[str, Any] | None:
        """The capsule screen (seconds) and the functional model's whole-model grade (minutes), each as
        soon as it exists, before the timing lands."""
        job_dir = self.root / digest
        check = read_json(job_dir / "pre_measure_check.json")
        local = read_json(job_dir / J.LOCAL_VERDICT)
        if check is None and local is None:
            return None
        return {"capsule_screen": check, "whole_model_functional": local}

    def history(self) -> list[dict[str, Any]]:
        """Every measurement in request order, compact: what the agent reads as search memory."""
        rows = []
        for job in sorted(self.jobs(), key=lambda job: float(job.get("requested_epoch") or 0)):
            found = self.result_by_key(key_of(job)) or {}
            verdict = found.get("verdict") or {}
            rows.append(
                {
                    "package_sha256": job["package_sha256"],
                    "replicate": int(job.get("replicate") or 0),
                    "label": job.get("label"),
                    "attribution": J.attribution_state(self.attribution(key_of(job))),
                    "requested_at": job.get("requested_at"),
                    "state": job.get("state"),
                    "timing_status": found.get("timing_status")
                    or (J.TIMING_PENDING if job.get("state") in (J.PENDING, J.RUNNING) else None),
                    "objective_cycles": found.get("objective_cycles"),
                    "whole_window_cycles": verdict.get("whole_window_cycles"),
                    "groups_failed": (verdict.get("correctness") or {}).get("groups_failed"),
                    "groups_unverifiable": len((verdict.get("correctness") or {}).get("groups_unverifiable") or []),
                    "vendor_also_fails_count": verdict.get("vendor_also_fails_count"),
                    "argmax": (verdict.get("correctness") or {}).get("argmax"),
                    "refusal": (found.get("refusal") or verdict.get("refusal") or "")[:200] or None,
                    "notice": job.get("notice") if job.get("state") in (J.PENDING, J.RUNNING, J.BOARD) else None,
                    "infra_worker_lost": bool(found.get("infra_worker_lost")),
                    "attempts": A.attempt_count(self.root / key_of(job)),
                    "from_attempt": (found.get("from_attempt") or {}).get("attempt"),
                    "invalid_reason": verdict.get("invalid_reason"),
                    "failing": J.failing_summary(found),
                }
            )
        return rows

    def best(self) -> dict[str, Any] | None:
        """The lowest-cycle MEASURED (valid), ATTRIBUTABLE result.  An invalid run, and bytes no authored
        round earned, are never a candidate for this."""
        best: dict[str, Any] | None = None
        for job in self.jobs():
            if int(job.get("replicate") or 0) > 0 or not self.attributable(key_of(job)):
                continue
            found = self.result(job["package_sha256"])
            cycles = V.objective_cycles((found or {}).get("verdict"))
            if cycles is None or (found or {}).get("timing_status") != V.TIMING_MEASURED:
                continue
            if best is None or cycles < int(best["objective_cycles"]):
                best = found
        return best


__all__ = ["MeasurementService", "WORKER_MODULE", "alive", "descendant_pids"]
