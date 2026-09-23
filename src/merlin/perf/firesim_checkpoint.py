"""Submit queue-owned FireSim jobs and seal them as one checkpoint.

`firesim_receipt` is the boundary AFTER a run: it seals one finished job from its three logs.
This module is the half before it, and the loop around it:

1. build the one submission the shared queue permits (`runworkload-full`, never a bare FireSim
   phase), always naming the bitstream it flashes;
2. run it, and copy the client log, the daemon's job log and the UART into an evidence directory
   that outlives the queue's own job directory;
3. seal that evidence with the fail-closed receipt;
4. repeat over a manifest, so "did this workstream move the hardware number" is one command whose
   output is a table of sealed receipts rather than numbers read off a terminal.

Every host and design fact is an input: the queue executable, where the queue keeps job state, the
chipyard root, the workload, the hwdb entry. The queue's install location and its state directory
are separate inputs because they are separate things (a reinstall moved one and not the other).

The queue flashes the FPGA once per job, and a job boots one binary. Running several programs per
flash therefore means several programs in ONE boot binary; invoking FireSim phases from inside a
job is the thing the queue exists to prevent, so this module never offers it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.perf.execution_policy import FIRESIM_QUEUE_OPERATION, FireSimQueuePreflight
from merlin.perf.firesim_receipt import (
    FireSimReceiptError,
    parse_queued_firesim_receipt,
    write_queued_firesim_receipt,
)

MANIFEST_SCHEMA = "merlin_firesim_checkpoint_manifest_v1"
CHECKPOINT_SCHEMA = "merlin_firesim_checkpoint_v1"
#: An ``observe`` key an entry sets, before the run, when its program may be cut off after it has
#: closed its measured window (one that drains large outputs over a simulated UART, for one).
JOB_MAY_NOT_FINISH = "job_may_not_finish"
#: A measured window read from a job that did not finish. Never ``observed``, never ``sealed``.
OBSERVED_INCOMPLETE = "observed_job_incomplete"

#: Our own record of what a submission STAGED, written at submit time and keyed by queue job id.
#: The queue's per-job record carries no digest of the staged binary, and for a job submitted by the
#: same user as the daemon it keeps no copy of it either, so ``deploy_overlay/workloads/`` is simply
#: absent -- which is why reading a finished job used to fail for every job it was pointed at.
STAGED_BINARY_SCHEMA = "merlin_firesim_staged_binary_v1"
#: Ledger namespace under ``out/artifacts/perf-studies/``. A ledger, not a cache: it is the only
#: evidence tying a queue job id to the bytes that ran, and a purge of it would silently turn every
#: later reading of those jobs into "cannot be bound to an entry".
STAGED_BINARY_LEDGER = "firesim_staged_binaries"
#: How a result is bound to the bytes that produced it, recorded on the row so a reader can see
#: which evidence was available rather than assuming the strongest.
BOUND_BY_OVERLAY = "queue_deploy_overlay"
BOUND_BY_SUBMIT_RECORD = "submit_time_staged_binary_record"

_CLIENT_PREFIX = "[firesim-queue]"
_LIFECYCLE_DECLARATION = "kill -> infrasetup -> runworkload -> kill"

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class CheckpointError(ValueError):
    """A checkpoint step could not be completed honestly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_file(path: Path, role: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise CheckpointError(f"{role} must be an absolute plain file: {path}")
    return path


@dataclass(frozen=True)
class QueueHost:
    """Where this host keeps its queue and its FireSim manager. Policy, never derived.

    ``queue_executable`` is the raw client (not a PATH symlink to it): the receipt binds the
    executable's bytes, and a symlink's target can change under it. ``queue_state_root`` is the
    directory the DAEMON writes ``jobs/<id>/stdout.log`` under.
    """

    queue_executable: Path
    queue_state_root: Path
    chipyard: Path

    def __post_init__(self) -> None:
        _plain_file(self.queue_executable, "queue executable")
        if not os.access(self.queue_executable, os.X_OK):
            raise CheckpointError(f"queue executable is not executable: {self.queue_executable}")
        for role, directory in (("queue state root", self.queue_state_root), ("chipyard root", self.chipyard)):
            if not directory.is_absolute() or not directory.is_dir():
                raise CheckpointError(f"{role} must be an absolute directory: {directory}")


@dataclass(frozen=True)
class QueueSubmission:
    """One `runworkload-full` job. ``hw_config`` is required, not defaulted.

    Left unset, the queue inherits the shared runtime template's default design. Two users wanting
    different bitstreams then race, and the loser runs the other's hardware with plausible output.
    """

    host: QueueHost
    workload: str
    bootbinary: str
    elf: Path
    hw_config: str
    hwdb_config_artifact: Path | None = None
    timeout_s: int = 1800
    priority: int = 5
    project: str = "merlin-checkpoint"

    def __post_init__(self) -> None:
        for role, value in (
            ("workload", self.workload),
            ("bootbinary", self.bootbinary),
            ("hw_config", self.hw_config),
            ("project", self.project),
        ):
            if not value.strip() or value != value.strip() or " " in value:
                raise CheckpointError(f"{role} must be one nonempty token, got {value!r}")
        _plain_file(self.elf, "staged ELF")
        if self.hwdb_config_artifact is not None:
            _plain_file(self.hwdb_config_artifact, "hwdb config artifact")
        if self.timeout_s < 1:
            raise CheckpointError("a FireSim job needs a positive RUNNING-phase timeout")
        if self.priority not in (0, 5, 10):
            raise CheckpointError("queue priority is one of 0, 5, 10")
        self.preflight()

    def argv(self) -> tuple[str, ...]:
        argv = [
            str(self.host.queue_executable),
            FIRESIM_QUEUE_OPERATION,
            "--chipyard",
            str(self.host.chipyard),
            "--workload",
            self.workload,
            "--bootbinary",
            self.bootbinary,
            "--stage-from",
            str(self.elf),
            "--hw-config",
            self.hw_config,
        ]
        if self.hwdb_config_artifact is not None:
            argv += ["--hwdb-config-artifact", str(self.hwdb_config_artifact)]
        argv += ["--priority", str(self.priority), "--project", self.project, "--timeout", str(self.timeout_s)]
        return tuple(argv)

    def preflight(self) -> FireSimQueuePreflight:
        return FireSimQueuePreflight(str(self.host.queue_executable), self.argv())


def inspect_queue_contract(
    host: QueueHost, submission: QueueSubmission, *, runner: Runner = subprocess.run
) -> dict[str, Any]:
    """Prove the INSTALLED queue owns the lifecycle and accepts every flag we are about to pass.

    A queue that silently ignores `--hw-config` would flash the shared default. Asking the
    installed client what it accepts is the only way to learn that before the FPGA is touched.
    """
    texts: list[str] = []
    for arguments in (("--help",), (FIRESIM_QUEUE_OPERATION, "--help")):
        completed = runner([str(host.queue_executable), *arguments], capture_output=True, text=True, check=False)
        if completed.returncode:
            raise CheckpointError(f"the queue could not describe {' '.join(arguments)}")
        texts.append(completed.stdout + completed.stderr)
    described = " ".join(" ".join(texts).split())
    if _LIFECYCLE_DECLARATION not in described:
        raise CheckpointError(
            f"the queue does not declare that {FIRESIM_QUEUE_OPERATION} owns '{_LIFECYCLE_DECLARATION}'"
        )
    words = set(described.replace("[", " ").replace("]", " ").replace(",", " ").split())
    flags = [token for token in submission.argv() if token.startswith("--")]
    missing = [flag for flag in flags if flag not in words]
    if missing:
        raise CheckpointError(f"the installed queue does not accept {missing}")
    return {
        "queue_operation": FIRESIM_QUEUE_OPERATION,
        "accepted_flags": flags,
        "contract_help_sha256": hashlib.sha256(described.encode("utf-8")).hexdigest(),
    }


def client_environment(
    *, path_prefix: Sequence[Path] = (), drop: Sequence[str] = (), base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The queue client's environment under this host's policy.

    A daemon run by another account cannot write into the submitter's checkout, so the queue gives
    each job a private deploy overlay. Two things defeat that overlay and are host policy to undo:
    the FireSim manager resolves its deploy root from the path it was found on (``path_prefix``
    puts a launcher that preserves the daemon's cwd first), and the client forwards the submitter's
    identity variables, which point the daemon at a home it cannot read (``drop`` omits them).
    """
    env = dict(os.environ if base is None else base)
    for name in drop:
        env.pop(name, None)
    for directory in path_prefix:
        if not directory.is_absolute() or not directory.is_dir():
            raise CheckpointError(f"client PATH prefix must be an absolute directory: {directory}")
    if path_prefix:
        inherited = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([*map(str, path_prefix), *([inherited] if inherited else [])])
    return env


def _client_records(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        tokens = raw_line.strip().split()
        if not tokens or tokens[0] != _CLIENT_PREFIX:
            continue
        values: dict[str, str] = {}
        for token in tokens[1:]:
            key, separator, value = token.partition("=")
            if separator and key and value:
                values.setdefault(key, value)
        values["_terminal"] = "yes" if "terminal" in tokens else ""
        records.append(values)
    return records


def active_jobs(host: QueueHost, *, runner: Runner = subprocess.run) -> list[int]:
    """Job ids the queue reports as queued or running (terminal jobs are not listed)."""
    completed = runner([str(host.queue_executable), "status"], capture_output=True, text=True, check=False)
    if completed.returncode:
        raise CheckpointError("the queue could not report its status")
    jobs: list[int] = []
    for record in _client_records(completed.stdout + completed.stderr):
        if "job_id" in record:
            try:
                jobs.append(int(record["job_id"]))
            except ValueError as exc:
                raise CheckpointError("queue status reported a non-integer job id") from exc
    return jobs


def _submitted_job_id(client_text: str) -> int:
    submitted = [
        record
        for record in _client_records(client_text)
        if record.get("kind") == FIRESIM_QUEUE_OPERATION and record.get("state") == "QUEUED"
    ]
    if len(submitted) != 1:
        raise CheckpointError("the queue client did not report exactly one submitted job")
    try:
        return int(submitted[0]["job_id"])
    except (KeyError, ValueError) as exc:
        raise CheckpointError("the queue client reported no integer job id") from exc


def _find_uart(host: QueueHost, workload: str, job_id: int) -> Path:
    """The per-job UART: the daemon's overlay for a cross-user job, else the native deploy tree."""
    roots = (
        host.queue_state_root / "jobs" / str(job_id) / "deploy_overlay" / "results-workload",
        host.chipyard / "sims" / "firesim" / "deploy" / "results-workload",
    )
    for root in roots:
        runs = sorted(root.glob(f"*-{workload}-q{job_id}"), key=lambda path: path.stat().st_mtime, reverse=True)
        for run in runs:
            uart = run / f"{workload}0" / "uartlog"
            if uart.is_file() and not uart.is_symlink():
                return uart
    raise CheckpointError(f"no per-job UART log exists for queue job {job_id}")


class JobIncomplete(CheckpointError):
    """A job reached a terminal state that is not ``DONE`` with exit code 0.

    It carries what the row needs to say so: the job, its state, and where the evidence that could
    still be collected was put. A job cut off by its time limit may have printed a complete measured
    window first; whether that window may be READ is the entry's declaration to make, not this
    harness's.
    """

    def __init__(self, message: str, *, job_id: int, state: str, evidence_dir: Path):
        super().__init__(message)
        self.job_id, self.state, self.evidence_dir = job_id, state, evidence_dir


def _find_partial_uart(host: QueueHost, workload: str, job_id: int) -> Path | None:
    """The UART of a job that did not finish: where a finished job leaves it, else the live slot."""
    try:
        return _find_uart(host, workload, job_id)
    except CheckpointError:
        pass
    slots = sorted((host.queue_state_root / "jobs" / str(job_id) / "simulation").glob("sim_slot_*/uartlog"))
    plain = [path for path in slots if path.is_file() and not path.is_symlink()]
    return plain[0] if len(plain) == 1 else None


@dataclass(frozen=True)
class SubmissionEvidence:
    job_id: int
    submission_json: Path
    client_log: Path
    daemon_log: Path
    uart_log: Path
    wall_s: float
    #: sha256 of the file this submission handed to ``--stage-from``, hashed BEFORE the queue was
    #: asked to run it. A result is bound to its bytes by this, not by the path it was read from.
    stage_from_sha256: str = ""


def staged_binary_ledger() -> Path:
    """Where the submit-time record of a staged binary lives: one JSON per queue job id."""
    from merlin.common.paths import artifacts_dir

    return Path(artifacts_dir()) / "perf-studies" / STAGED_BINARY_LEDGER


def record_staged_binary(submission: QueueSubmission, job_id: int, digest: str, *, ledger: Path) -> Path:
    """Record WHICH BYTES this job was submitted with, so a later reading can bind its result.

    The shape is borrowed from :func:`merlin.kernels.opu_cert.provenance_stamp` and the check the
    corpus report makes with it: the identity of what was built is recorded at build time, and the
    reader COMPARES rather than assumes. There a bare-metal image prints its own stamp and the
    report refuses a mismatch as "a stale binary ran"; here the ELF cannot print anything, so the
    stamp is written beside the submission instead and compared the same way.

    Writing is best-effort ONLY in the sense that it never blocks a submission that has already been
    accepted by the queue; a missing record makes a later reading refuse, which is the fail-closed
    direction.
    """
    root = Path(ledger)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"job{int(job_id)}.json"
    destination.write_text(
        json.dumps(
            {
                "schema": STAGED_BINARY_SCHEMA,
                "job_id": int(job_id),
                "workload": submission.workload,
                "bootbinary": submission.bootbinary,
                "hw_config": submission.hw_config,
                "stage_from": str(submission.elf),
                "stage_from_sha256": digest,
                "stamp": f"job={int(job_id)} elf={digest[:12]} hw={submission.hw_config}",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def staged_binary_record(job_id: int, *, ledger: Path) -> dict[str, Any] | None:
    """This repository's own record of what job ``job_id`` staged, or None when none was kept."""
    path = Path(ledger) / f"job{int(job_id)}.json"
    if path.is_symlink() or not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"the staged-binary record for job {job_id} is not readable JSON: {path}") from exc
    if not isinstance(record, Mapping) or record.get("schema") != STAGED_BINARY_SCHEMA:
        raise CheckpointError(f"the staged-binary record for job {job_id} is not a {STAGED_BINARY_SCHEMA} document")
    return dict(record)


def _write_staged_binary(
    submission: QueueSubmission, job_id: int, digest: str, evidence_dir: Path, ledger: Path | None
) -> None:
    """Beside the evidence always; into the job-id ledger when the caller named one.

    The copy beside the evidence travels with the run.  The ledger is what a LATER reading has: it
    is keyed by queue job id, which is the only handle ``--from-job`` is given.
    """
    beside = evidence_dir / "staged-binary.json"
    if ledger is None:
        record_staged_binary(submission, job_id, digest, ledger=evidence_dir)
        (evidence_dir / f"job{int(job_id)}.json").replace(beside)
        return
    shutil.copyfile(record_staged_binary(submission, job_id, digest, ledger=ledger), beside)


def submit(
    submission: QueueSubmission,
    evidence_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
    runner: Runner = subprocess.run,
    clock: Callable[[], float] | None = None,
    ledger: Path | None = None,
) -> SubmissionEvidence:
    """Run one job to its terminal state and copy its evidence out of the queue's own tree.

    The daemon log keeps ``jobs/<id>/`` in its copied path: the receipt binds a daemon log to its
    job by that path component, and the queue is free to purge the original.

    The bytes handed to ``--stage-from`` are HASHED BEFORE the queue is asked to run them, and the
    digest is written both beside the evidence and into the job-id ledger. Nothing else records it:
    the queue's own per-job record carries the path and not the content, and for a job submitted by
    the daemon's own user it keeps no copy of the binary either -- so without this, a result read
    back later cannot be shown to belong to the program that was built for it.
    """
    now = clock or time.monotonic
    if not evidence_dir.is_absolute():
        raise CheckpointError("the evidence directory must be absolute")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    argv = submission.argv()
    submission_json = evidence_dir / "submission.json"
    submission_json.write_text(json.dumps(list(argv), indent=2) + "\n", encoding="utf-8")
    stage_from_sha256 = sha256_file(submission.elf)

    started = now()
    # The client prints its phase and terminal lines on stderr; merge them or a finished job reads
    # as one that never reported.
    completed = runner(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=submission.timeout_s + 1800,
        env=dict(env) if env is not None else None,
    )
    wall_s = round(now() - started, 3)
    client_log = evidence_dir / "queue-client.log"
    client_log.write_text(completed.stdout or "", encoding="utf-8")
    job_id = _submitted_job_id(completed.stdout or "")
    terminal = [record for record in _client_records(completed.stdout or "") if record["_terminal"]]
    if (
        completed.returncode
        or len(terminal) != 1
        or terminal[0].get("state") != "DONE"
        or terminal[0].get("exit_code") != "0"
    ):
        # Keep what the job did print. A program can close its measured window and then be cut off
        # while it drains its outputs over a simulated UART; without the log that is
        # indistinguishable from a job that measured nothing.
        partial = _find_partial_uart(submission.host, submission.workload, job_id)
        if partial is not None:
            shutil.copyfile(partial, evidence_dir / "uartlog")
        # RECORDED HERE TOO, and this is the case that needs it: a job cut off after closing its
        # measured window is exactly the one someone reads back later with `--from-job`, and without
        # the digest that reading has nothing to bind the result to.
        if job_id > 0:
            _write_staged_binary(submission, job_id, stage_from_sha256, evidence_dir, ledger)
        state = str(terminal[0].get("state")) if len(terminal) == 1 else "UNKNOWN"
        raise JobIncomplete(
            f"queue job {job_id} did not finish DONE with exit_code=0 (state {state}); see {client_log}",
            job_id=job_id,
            state=state,
            evidence_dir=evidence_dir,
        )

    source_daemon = host_daemon_log(submission.host, job_id)
    daemon_log = evidence_dir / "queue" / "jobs" / str(job_id) / "stdout.log"
    daemon_log.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_daemon, daemon_log)
    uart_log = evidence_dir / "uartlog"
    shutil.copyfile(_find_uart(submission.host, submission.workload, job_id), uart_log)
    _write_staged_binary(submission, job_id, stage_from_sha256, evidence_dir, ledger)
    return SubmissionEvidence(
        job_id=job_id,
        submission_json=submission_json,
        client_log=client_log,
        daemon_log=daemon_log,
        uart_log=uart_log,
        wall_s=wall_s,
        stage_from_sha256=stage_from_sha256,
    )


def host_daemon_log(host: QueueHost, job_id: int) -> Path:
    path = host.queue_state_root / "jobs" / str(job_id) / "stdout.log"
    if path.is_symlink() or not path.is_file():
        raise CheckpointError(f"queue job {job_id} left no plain daemon log at {path}")
    return path


def seal(submission: QueueSubmission, evidence: SubmissionEvidence, validation_policy: Path) -> Path:
    """Seal one job. Raises `FireSimReceiptError` unless every proof agrees."""
    receipt = parse_queued_firesim_receipt(
        queue_client_log=evidence.client_log,
        queue_daemon_log=evidence.daemon_log,
        uart_log=evidence.uart_log,
        expected_queue_executable=submission.host.queue_executable,
        expected_submission_json=evidence.submission_json,
        expected_job_id=evidence.job_id,
        expected_workload=submission.workload,
        validation_policy_json=validation_policy,
    )
    return write_queued_firesim_receipt(receipt, evidence.submission_json.parent / "firesim-receipt.json")


@dataclass(frozen=True)
class CheckpointEntry:
    """One program of a checkpoint: what it is, the ELF, and how its UART proves correctness."""

    label: str
    model: str
    experiment: str
    elf: Path
    validation_policy: Path
    timeout_s: int = 1800
    #: For a program built before the sealed UART protocol: how to READ its cycle count when the
    #: receipt cannot be sealed. ``{"cycles_prefix": ..., "after": ..., "before": ...,
    #: "markers": [...]}``. A row read this way is ``observed``, never ``sealed``: it is a
    #: denominator on the same design, stated as unsealed, and ``compare`` keeps the two apart.
    observe: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for role, value in (("label", self.label), ("model", self.model), ("experiment", self.experiment)):
            if not value or "/" in value or value != value.strip():
                raise CheckpointError(f"checkpoint entry {role} must be one path-safe token")
        _plain_file(self.elf, f"ELF of {self.label}")
        _plain_file(self.validation_policy, f"validation policy of {self.label}")


@dataclass(frozen=True)
class CheckpointPlan:
    name: str
    host: QueueHost
    workload: str
    bootbinary: str
    hw_config: str
    substrate: str
    entries: tuple[CheckpointEntry, ...]
    hwdb_config_artifact: Path | None = None
    priority: int = 5
    design_pin: str | None = None
    require_idle_queue: bool = True
    client_env: Mapping[str, str] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not self.entries:
            raise CheckpointError("a checkpoint needs at least one entry")
        labels = [entry.label for entry in self.entries]
        if len(set(labels)) != len(labels):
            raise CheckpointError("checkpoint entry labels must be unique")
        for entry in self.entries:  # a malformed plan fails at load, not at its third job
            self.submission_for(entry)

    def submission_for(self, entry: CheckpointEntry) -> QueueSubmission:
        return QueueSubmission(
            host=self.host,
            workload=self.workload,
            bootbinary=self.bootbinary,
            elf=entry.elf,
            hw_config=self.hw_config,
            hwdb_config_artifact=self.hwdb_config_artifact,
            timeout_s=entry.timeout_s,
            priority=self.priority,
            project=self.name,
        )


def _client_env(policy: Mapping[str, Any] | None, resolve_path: Callable[[str], Path]) -> dict[str, str] | None:
    if policy is None:
        return None
    unknown = set(policy) - {"path_prefix", "drop"}
    if unknown:
        raise CheckpointError(f"unknown client_env policy keys {sorted(unknown)}")
    return client_environment(
        path_prefix=[resolve_path(item) for item in policy.get("path_prefix", ())], drop=list(policy.get("drop", ()))
    )


def load_plan(document: Mapping[str, Any], *, resolve_path: Callable[[str], Path] = Path) -> CheckpointPlan:
    """Build a plan from a manifest mapping. ``resolve_path`` lets a caller expand host keys."""
    if document.get("schema") != MANIFEST_SCHEMA:
        raise CheckpointError(f"checkpoint manifest schema must be {MANIFEST_SCHEMA}")
    try:
        host_doc, design, workload = document["host"], document["design"], document["workload"]
        host = QueueHost(
            queue_executable=resolve_path(host_doc["queue_executable"]),
            queue_state_root=resolve_path(host_doc["queue_state_root"]),
            chipyard=resolve_path(host_doc["chipyard"]),
        )
        artifact = design.get("hwdb_config_artifact")
        entries = tuple(
            CheckpointEntry(
                label=row["label"],
                model=row["model"],
                experiment=row["experiment"],
                elf=resolve_path(row["elf"]),
                validation_policy=resolve_path(row["validation_policy"]),
                observe=(dict(row["observe"]) if isinstance(row.get("observe"), Mapping) else None),
                timeout_s=int(row.get("timeout_s", 1800)),
            )
            for row in document["entries"]
        )
        return CheckpointPlan(
            name=document["checkpoint"],
            host=host,
            workload=workload["name"],
            bootbinary=workload["bootbinary"],
            hw_config=design["hw_config"],
            substrate=design["substrate"],
            entries=entries,
            hwdb_config_artifact=resolve_path(artifact) if artifact else None,
            priority=int(document.get("priority", 5)),
            design_pin=design.get("pin"),
            require_idle_queue=bool(document.get("require_idle_queue", True)),
            client_env=_client_env(document.get("client_env"), resolve_path),
        )
    except KeyError as exc:
        raise CheckpointError(f"checkpoint manifest is missing {exc}") from exc


def run_checkpoint(
    plan: CheckpointPlan,
    *,
    evidence_root: Callable[[CheckpointEntry], Path],
    runner: Runner = subprocess.run,
    ledger: Path | None = None,
) -> dict[str, Any]:
    """Run every entry as its own queue job, in order, and return the checkpoint table.

    Identical ELFs are measured once: two labels whose bytes agree would only measure the FPGA's
    run-to-run spread under two names. A failed entry is recorded and the batch continues, so one
    broken program does not cost the others their slot; the table says which rows are sealed.
    """
    first = plan.submission_for(plan.entries[0])
    contract = inspect_queue_contract(plan.host, first, runner=runner)
    if plan.require_idle_queue:
        busy = active_jobs(plan.host, runner=runner)
        if busy:
            raise CheckpointError(f"the shared queue is busy with jobs {busy}; not submitting")

    rows: list[dict[str, Any]] = []
    measured: dict[str, dict[str, Any]] = {}
    for entry in plan.entries:
        digest = sha256_file(entry.elf)
        row: dict[str, Any] = {
            "label": entry.label,
            "model": entry.model,
            "experiment": entry.experiment,
            "elf": str(entry.elf),
            "elf_sha256": digest,
        }
        if digest in measured:
            row.update(status="duplicate_elf", same_as=measured[digest]["label"])
            rows.append(row)
            continue
        try:
            submission = plan.submission_for(entry)
            evidence = submit(submission, evidence_root(entry), env=plan.client_env, runner=runner, ledger=ledger)
            row.update(
                job_id=evidence.job_id,
                queue_wall_s=evidence.wall_s,
                evidence_dir=str(evidence.submission_json.parent),
                # The digest of the bytes ACTUALLY handed to --stage-from, taken at submit time. It
                # equals `elf_sha256` on a healthy run; recording both is what makes an ELF rebuilt
                # between planning and submission visible instead of silent.
                stage_from_sha256=evidence.stage_from_sha256,
            )
            receipt_path = seal(submission, evidence, entry.validation_policy)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            cycles = receipt["queue_receipt"]["warm_profile"]["total_compute_cycles"]
            row.update(
                status="sealed", cycles=cycles, receipt=str(receipt_path), receipt_sha256=sha256_file(receipt_path)
            )
            measured[digest] = row
        except (CheckpointError, FireSimReceiptError, subprocess.TimeoutExpired) as exc:
            row.update(status="not_sealed", reason=str(exc))
            incomplete = isinstance(exc, JobIncomplete)
            if incomplete:
                row.update(job_id=exc.job_id, job_state=exc.state, evidence_dir=str(exc.evidence_dir))
            uart = Path(str(row.get("evidence_dir") or "")) / "uartlog"
            # A window read from a job that did not finish is admitted only where the entry said,
            # BEFORE the run, that its program may not finish (``job_may_not_finish``), and it is
            # its own status: weaker than ``observed``, which is weaker than ``sealed``.
            allowed = entry.observe and (not incomplete or entry.observe.get(JOB_MAY_NOT_FINISH) is True)
            if allowed and uart.is_file():
                try:
                    row.update(
                        status=OBSERVED_INCOMPLETE if incomplete else "observed",
                        observed_cycles=observe_cycles(
                            uart.read_text(encoding="utf-8", errors="replace"), entry.observe
                        ),
                        observe=dict(entry.observe),
                    )
                    measured[digest] = row
                except CheckpointError as unreadable:
                    row["observe_error"] = str(unreadable)
        rows.append(row)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint": plan.name,
        "substrate": plan.substrate,
        "hw_config": plan.hw_config,
        "design_pin": plan.design_pin,
        "hwdb_config_artifact_sha256": (sha256_file(plan.hwdb_config_artifact) if plan.hwdb_config_artifact else None),
        "queue_contract": contract,
        "entries": rows,
        "sealed": sum(row["status"] == "sealed" for row in rows),
        "total": len(rows),
    }


def observe_finished_job(
    plan: Any, entry: CheckpointEntry, job_id: int, evidence_dir: Path, *, ledger: Path | None = None
) -> dict[str, Any]:
    """A row for a job that ALREADY ran, read from the queue's own record of it. Submits nothing.

    For a job that closed its measured window and was then cut off, rerunning buys a second cut-off
    and hours of a shared FPGA. The job is bound to the entry by content: the queue's record must
    name this design (hardware config and the digest of its hwdb entry), and the bytes the queue RAN
    must be the entry's, byte for byte.

    TWO WAYS TO ESTABLISH THOSE BYTES, and the stronger one is usually not available. The queue
    stages a copy under ``jobs/<id>/deploy_overlay/workloads/`` only when the submitter is not the
    daemon's own user; for a same-user job that directory does not exist, and this function used to
    require it unconditionally -- so the glob returned nothing and EVERY job it was pointed at was
    refused with "did not stage this entry's executable", whatever it had really run. The fallback
    is this repository's own submit-time record (:func:`record_staged_binary`): the digest of the
    file handed to ``--stage-from``, hashed before the job ran and keyed by job id. With neither,
    the function refuses; it does not fall back to "the entry names an ELF, so that must be it".
    """
    host = plan.host
    job = host.queue_state_root / "jobs" / str(job_id)
    record_path = job / "runworkload-full.json"
    if record_path.is_symlink() or not record_path.is_file():
        raise CheckpointError(f"queue job {job_id} left no plain record at {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "hw_config": plan.hw_config,
        "hwdb_config_artifact_sha256": sha256_file(plan.hwdb_config_artifact) if plan.hwdb_config_artifact else None,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise CheckpointError(f"queue job {job_id} ran {key}={record.get(key)!r}, not this plan's {value!r}")
    staged = sorted((job / "deploy_overlay" / "workloads").glob(f"*/{record.get('bootbinary')}"))
    staged = [path for path in staged if path.is_file() and not path.is_symlink()]
    digest = sha256_file(entry.elf)
    if staged:
        if len(staged) != 1 or sha256_file(staged[0]) != digest:
            raise CheckpointError(f"queue job {job_id} did not stage this entry's executable ({digest[:12]})")
        bound_by = BOUND_BY_OVERLAY
    else:
        submitted = staged_binary_record(job_id, ledger=ledger or staged_binary_ledger())
        if submitted is None:
            raise CheckpointError(
                f"queue job {job_id} kept no staged copy of its bootbinary and this repository holds "
                f"no submit-time record of what it staged, so its result cannot be bound to this "
                f"entry's bytes ({digest[:12]}); a result attributed to the wrong binary is worse "
                "than no result"
            )
        if submitted.get("stage_from_sha256") != digest:
            raise CheckpointError(
                f"queue job {job_id} was submitted with {str(submitted.get('stage_from_sha256'))[:12]} "
                f"and this entry is {digest[:12]}; a stale binary ran"
            )
        bound_by = BOUND_BY_SUBMIT_RECORD
    if not entry.observe:
        raise CheckpointError("a finished job is read through the entry's observe block; it declares none")
    uart = _find_partial_uart(host, str(record.get("workload")), job_id)
    if uart is None:
        raise CheckpointError(f"queue job {job_id} left no UART log")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(uart, evidence_dir / "uartlog")
    shutil.copyfile(record_path, evidence_dir / "runworkload-full.json")
    # The overlay is corroborating evidence WHEN THE QUEUE KEPT ONE. Requiring it unconditionally
    # had the same defect as the staging glob above: a same-user job has no `deploy_overlay/` at all,
    # so every such job read as "did not finish" however cleanly it had finished.
    overlay = job / "deploy_overlay"
    finished = _finished(host, str(record.get("workload")), job_id) and (
        not overlay.is_dir() or (overlay / "results-workload").is_dir()
    )
    if not finished and entry.observe.get(JOB_MAY_NOT_FINISH) is not True:
        raise CheckpointError(
            f"queue job {job_id} did not finish, and the entry does not declare `{JOB_MAY_NOT_FINISH}`"
        )
    return {
        "label": entry.label,
        "model": entry.model,
        "experiment": entry.experiment,
        "elf": str(entry.elf),
        "elf_sha256": digest,
        "stage_from_sha256": digest,
        "binary_bound_by": bound_by,
        "job_id": job_id,
        "evidence_dir": str(evidence_dir),
        "status": "observed" if finished else OBSERVED_INCOMPLETE,
        "observed_cycles": observe_cycles(
            (evidence_dir / "uartlog").read_text(encoding="utf-8", errors="replace"), entry.observe
        ),
        "observe": dict(entry.observe),
        "read_from_existing_job": True,
    }


def _finished(host: QueueHost, workload: str, job_id: int) -> bool:
    try:
        _find_uart(host, workload, job_id)
    except CheckpointError:
        return False
    return True


def observe_cycles(uart_text: str, how: Mapping[str, Any]) -> int:
    """The cycle count of a program that predates the sealed protocol, read as its entry declares.

    Exactly one line starting with ``cycles_prefix`` must lie between the unique ``after`` and
    ``before`` lines, every ``markers`` line must appear exactly once, and the number must be a
    positive integer. Anything else raises: an ambiguous read is not a denominator.
    """
    lines = [line.strip() for line in uart_text.splitlines()]

    def unique(text: str) -> int:
        found = [i for i, line in enumerate(lines) if line == text]
        if len(found) != 1:
            raise CheckpointError(f"UART must contain exactly one {text!r} line, found {len(found)}")
        return found[0]

    prefix = str(how.get("cycles_prefix") or "")
    if not prefix:
        raise CheckpointError("an observe block needs a cycles_prefix")
    start, end = unique(str(how.get("after") or "")), unique(str(how.get("before") or ""))
    window = lines[start + 1 : end]
    for marker in how.get("markers") or ():
        # Inside the measured window: a program that warms up first prints its pass line twice.
        count = sum(1 for line in window if line == str(marker))
        if count != 1:
            raise CheckpointError(f"the measured window must contain exactly one {marker!r} line, found {count}")
    inside = [line for line in window if line.startswith(prefix)]
    if len(inside) != 1:
        raise CheckpointError(f"expected one {prefix!r} line in the measured window, found {len(inside)}")
    token = inside[0][len(prefix) :].split()[0] if inside[0][len(prefix) :].split() else ""
    if not token.isdigit() or int(token) <= 0:
        raise CheckpointError(f"{inside[0]!r} carries no positive integer cycle count")
    return int(token)


def compare(current: Mapping[str, Any], previous: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Row-by-row change against an earlier checkpoint ON THE SAME DESIGN, or a refusal."""
    for key in ("substrate", "hw_config", "hwdb_config_artifact_sha256"):
        if current.get(key) != previous.get(key):
            raise CheckpointError(f"checkpoints differ in {key}; cycles from different designs are not comparable")
    before = {row["label"]: row for row in previous["entries"] if row.get("status") == "sealed"}
    table: list[dict[str, Any]] = []
    for row in current["entries"]:
        if row.get("status") != "sealed":
            continue
        old = before.get(row["label"])
        change = {"label": row["label"], "cycles": row["cycles"], "previous_cycles": old["cycles"] if old else None}
        if old:
            change["ratio"] = round(row["cycles"] / old["cycles"], 6)
            change["same_elf"] = row["elf_sha256"] == old["elf_sha256"]
        table.append(change)
    return table


def _resolve_host_path(value: str) -> Path:
    """`ext:<key>[/sub/path]` reads the host's `.env`; anything else is a literal path."""
    from merlin.common.paths import ext_path

    if not value.startswith("ext:"):
        return Path(value)
    key, _separator, rest = value[len("ext:") :].partition("/")
    return ext_path(key) / rest if rest else ext_path(key)


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    from merlin.common.artifacts import new_measurement

    parser = argparse.ArgumentParser(
        description="Run a manifest of ELFs as queue-owned FireSim jobs and seal the checkpoint."
    )
    parser.add_argument("manifest", help="checkpoint manifest (YAML)")
    parser.add_argument("--previous", help="an earlier checkpoint.json on the same design")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate the manifest and the queue contract; submit nothing"
    )
    parser.add_argument(
        "--from-job",
        action="append",
        default=[],
        metavar="LABEL=JOB_ID",
        help="read this entry from a queue job that already ran instead of submitting it; the job is "
        "bound to the entry by design and executable digest. With this flag nothing is submitted, "
        "and only the named entries are recorded.",
    )
    arguments = parser.parse_args(argv)

    plan = load_plan(
        yaml.safe_load(Path(arguments.manifest).read_text(encoding="utf-8")), resolve_path=_resolve_host_path
    )
    if arguments.dry_run:
        contract = inspect_queue_contract(plan.host, plan.submission_for(plan.entries[0]))
        print(
            json.dumps(
                {
                    "checkpoint": plan.name,
                    "entries": len(plan.entries),
                    "queue_contract": contract,
                    "active_jobs": active_jobs(plan.host),
                },
                indent=2,
            )
        )
        return 0

    directories: dict[str, Any] = {}

    def evidence_root(entry: CheckpointEntry) -> Path:
        measurement = new_measurement(
            plan.substrate, entry.model, entry.experiment, notes=f"checkpoint {plan.name} / {entry.label}"
        )
        directories[entry.label] = measurement
        return measurement.path / "evidence"

    if arguments.from_job:
        wanted = dict(item.partition("=")[::2] for item in arguments.from_job)
        by_label = {entry.label: entry for entry in plan.entries}
        unknown = sorted(set(wanted) - set(by_label))
        if unknown:
            raise SystemExit(f"--from-job names entries the manifest does not have: {unknown}")
        rows = []
        for label, job in wanted.items():
            entry = by_label[label]
            try:
                rows.append(
                    observe_finished_job(plan, entry, int(job), evidence_root(entry), ledger=staged_binary_ledger())
                )
            except (CheckpointError, ValueError) as refusal:
                rows.append({"label": label, "status": "not_sealed", "reason": str(refusal), "job_id": job})
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "checkpoint": plan.name,
            "substrate": plan.substrate,
            "hw_config": plan.hw_config,
            "design_pin": plan.design_pin,
            "hwdb_config_artifact_sha256": (
                sha256_file(plan.hwdb_config_artifact) if plan.hwdb_config_artifact else None
            ),
            "queue_contract": None,
            "entries": rows,
            "sealed": 0,
            "total": len(rows),
            "submitted_nothing": True,
        }
    else:
        checkpoint = run_checkpoint(plan, evidence_root=evidence_root, ledger=staged_binary_ledger())
    if arguments.previous:
        checkpoint["compared_with"] = str(Path(arguments.previous).resolve())
        checkpoint["comparison"] = compare(checkpoint, json.loads(Path(arguments.previous).read_text(encoding="utf-8")))
    for row in checkpoint["entries"]:
        measurement = directories.get(row["label"])
        if measurement is None:
            continue
        (measurement.path / "checkpoint_row.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        measurement.write_manifest()
    summary = new_measurement(plan.substrate, "_checkpoints", plan.name, notes="checkpoint table")
    output = summary.path / "checkpoint.json"
    output.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary.write_manifest()
    print(output)
    return 0 if checkpoint["sealed"] == checkpoint["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
