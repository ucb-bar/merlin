"""The machines a whole-model program is run on, each able to say which device it is.

A cycle count means nothing without its device: two bitstreams, or a bitstream and an elaborated-RTL
emulator, are different designs, so a number from one is never a divisor for a number from the other.
Every machine here therefore produces an IDENTITY before it produces a number, resolved from the pin
registry (``merlin/contract/hardware_pins.yaml``) by the bytes it is about to run -- never by a name --
and refuses to run when that identity cannot be established.

Two rules this module enforces before a simulator cycle is spent:

* **The device is registered.**  An unregistered binary is refused: a result attributed to bytes the
  registry does not describe is exactly the wrong-device hazard the registry exists to stop.
* **The program's ABI is the device's.**  The build record states the parameter header the program
  was compiled against; the device's artifact states the header its elaboration takes.  They must be
  equal, and an UNKNOWN on either side refuses rather than assumes.

``run`` is BLOCKING and is meant to be called from a detached worker process (:mod:`.worker`),
because a whole model takes tens of minutes to hours on an elaborated-RTL simulator and nothing that
owns an agent session should wait on it.  Which machine a run uses, and how, is target DATA -- the
machine registry (:mod:`.registry`) -- never a literal here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.common import provenance
from merlin.perf import emulator_capture
from merlin.perf.execution_policy import FIRESIM_QUEUE_OPERATION

UNKNOWN = "UNKNOWN"

#: The emulator's own end-of-run line, as it prints it on its diagnostic stream.  A run that stopped
#: at its cycle budget prints the same line with ``done=0`` and still exits 0, so the exit status is
#: NOT evidence of completion; this line's ``done`` field is.
FINISHED_PREFIX = "[gsim-emu] FINISHED:"
#: The emulator's statement of how the program image reached memory.  Recorded, not required: it is
#: the difference between a run that loads in seconds and one that spends ~170M cycles loading, and a
#: slow run is diagnosable only if the log says which path it took.
LOAD_PATH_PREFIX = "[tsi] load path:"


class MachineRefusal(RuntimeError):
    """This machine cannot produce an attributable measurement of this program."""


#: The FireSim deploy tree's own layout (the tool's, not any target's): where a workload's inputs live.
DEPLOY_RELATIVE = ("sims", "firesim", "deploy")
WORKLOAD_INPUTS = "workloads"


def deploy_load_path(chipyard: Path, workload: str, bootbinary: str) -> Path:
    if not Path(chipyard).is_absolute():
        raise MachineRefusal("the chipyard root must be an absolute path")
    if not str(workload).strip() or not str(bootbinary).strip():
        raise MachineRefusal("the load path needs both a workload and a bootbinary name")
    return Path(chipyard).joinpath(*DEPLOY_RELATIVE, WORKLOAD_INPUTS, workload, bootbinary)


def _key_values(tokens: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if sep and key and value:
            values[key] = value
    return values


def finished_line(diagnostics: str) -> dict[str, Any] | None:
    """The emulator's FINISHED line as fields, or None when it never printed one (it crashed)."""
    found: dict[str, Any] | None = None
    for line in diagnostics.splitlines():
        stripped = line.strip()
        if not stripped.startswith(FINISHED_PREFIX):
            continue
        values = _key_values(stripped[len(FINISHED_PREFIX) :].split())
        record: dict[str, Any] = {"line": stripped}
        for key in ("cycles", "done", "exit_code"):
            try:
                record[key] = int(values[key]) if key in values else None
            except ValueError:
                record[key] = None
        found = record
    return found


def load_path_line(diagnostics: str) -> str | None:
    for line in diagnostics.splitlines():
        stripped = line.strip()
        if stripped.startswith(LOAD_PATH_PREFIX):
            return stripped[len(LOAD_PATH_PREFIX) :].strip()
    return None


@dataclass(frozen=True)
class DeviceIdentity:
    """Which hardware a number is about, resolved from the registry by content."""

    machine: str
    target: str
    binary: str
    binary_sha256: str
    artifact: str
    config: str
    built_from: tuple[str, ...]
    abi_header_sha256: str
    rung: str
    registry_verification: Mapping[str, Any] = field(default_factory=dict)
    engine_citation: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine": self.machine,
            "target": self.target,
            "binary": self.binary,
            "binary_sha256": self.binary_sha256,
            "artifact": self.artifact,
            "config": self.config,
            "built_from": list(self.built_from),
            "abi_header_sha256": self.abi_header_sha256,
            "rung": self.rung,
            "registry_verification": dict(self.registry_verification),
            "engine_citation": dict(self.engine_citation),
        }


def registered_artifact_for(digest: str, *, target: str, role: str) -> provenance.Artifact:
    """The ONE registered artifact of ``role`` for ``target`` whose declared digest is ``digest``."""
    matches = [
        artifact
        for artifact in provenance.load_artifacts().values()
        if artifact.digest == digest and artifact.target == target and artifact.role == role
    ]
    if not matches:
        raise MachineRefusal(
            f"no {role} artifact for target {target!r} declares digest {digest[:16]}; the bytes about to "
            "run are not a registered device, so no result from them could be attributed"
        )
    if len(matches) > 1:
        raise MachineRefusal(f"{len(matches)} artifacts declare {digest[:16]}; one device is one entry")
    return matches[0]


def _abi_along(artifact: provenance.Artifact, artifacts: Mapping[str, provenance.Artifact]) -> str:
    """The ABI header the artifact (or, failing that, the elaboration it was built from) declares."""
    if artifact.abi_header_sha256:
        return artifact.abi_header_sha256
    for parent in artifact.built_from:
        upstream = artifacts.get(parent)
        if upstream is not None and upstream.abi_header_sha256:
            return upstream.abi_header_sha256
    return UNKNOWN


class GsimMachine:
    """The target's installed elaborated-RTL emulator, run locally as one process per program.

    ``environment`` is what the caller declares the emulator needs (for example, the switch that
    selects its backdoor memory load); it is recorded in the run record verbatim, because it changes
    how long a run takes and must therefore be visible in any account of one.
    """

    kind = "gsim"
    role = "gsim_binary"
    rung = "elaborated_rtl"

    def __init__(
        self,
        target: str,
        *,
        max_cycles: int,
        environment: Mapping[str, str] | None = None,
        emulator: Path | None = None,
    ) -> None:
        if not target.strip():
            raise MachineRefusal("a machine needs its target")
        if max_cycles <= 0:
            raise MachineRefusal("a run needs a positive cycle budget")
        self.target = target
        self.max_cycles = int(max_cycles)
        self.environment = dict(environment or {})
        self._emulator = Path(emulator) if emulator is not None else None

    def spec(self) -> dict[str, Any]:
        """What a detached worker needs to rebuild this machine, as plain data."""
        return {
            "kind": self.kind,
            "target": self.target,
            "max_cycles": self.max_cycles,
            "environment": dict(self.environment),
            "emulator": str(self._emulator) if self._emulator is not None else None,
        }

    def identity(self) -> DeviceIdentity:
        from merlin.targetgen import gsim_emulator

        citation = gsim_emulator.citation(self.target)
        if self._emulator is not None:
            binary = self._emulator
            digest = provenance.file_digest(binary)
        else:
            if not citation.get("available"):
                raise MachineRefusal(f"the {self.target} GSIM emulator is unavailable: {citation.get('reason')}")
            binary = Path(str(citation["path"]))
            digest = str(citation.get("binary_sha256") or provenance.file_digest(binary))
        if digest == UNKNOWN or len(digest) != 64:
            raise MachineRefusal(f"the emulator at {binary} could not be hashed")
        artifact = registered_artifact_for(digest, target=self.target, role=self.role)
        artifacts = provenance.load_artifacts()
        return DeviceIdentity(
            machine=self.kind,
            target=self.target,
            binary=str(binary),
            binary_sha256=digest,
            artifact=artifact.name,
            config=artifact.config,
            built_from=artifact.built_from,
            abi_header_sha256=_abi_along(artifact, artifacts),
            rung=self.rung,
            registry_verification={
                name: provenance.verify_artifact(name).to_dict()
                for name in (artifact.name, *artifact.built_from)
                if name in artifacts
            },
            engine_citation=citation,
        )

    def admit(self, identity: DeviceIdentity, *, program_header_sha256: str | None) -> None:
        """Refuse a program compiled against an ABI this device does not take."""
        if identity.abi_header_sha256 == UNKNOWN:
            raise MachineRefusal(
                f"the device {identity.artifact} declares no ABI header, so a program cannot be shown to "
                "have been compiled for it"
            )
        if not program_header_sha256 or program_header_sha256 == UNKNOWN:
            raise MachineRefusal("the build record states no parameter header; its ABI is UNKNOWN")
        if program_header_sha256 != identity.abi_header_sha256:
            raise MachineRefusal(
                f"the program was compiled against header {program_header_sha256[:8]} but the device "
                f"{identity.artifact} takes {identity.abi_header_sha256[:8]}; an ABI mismatch runs and is "
                "wrong, and reads as a schedule regression"
            )

    def run(self, elf: Path, workdir: Path, *, timeout_s: float) -> dict[str, Any]:
        """Run one program to completion; the program's own output goes to ``uart.log``.

        The emulator writes the program's console to stdout and its own diagnostics to stderr, so
        the two go to separate files and simulator chatter can never split a protocol line.
        """
        identity = self.identity()
        workdir.mkdir(parents=True, exist_ok=True)
        uart, diagnostics = workdir / "uart.log", workdir / "sim.log"
        argv = [identity.binary, str(elf), f"+loadmem={elf}", f"+max-cycles={self.max_cycles}"]
        started = time.time()
        with uart.open("wb") as out:
            # The diagnostics are bounded on disk and the run ends at its first hardware assertion
            # (merlin.perf.emulator_capture: one failing design wrote 8.5 GB of assertions per run).
            capture = emulator_capture.run_capped(
                argv,
                stdout=out,
                stderr_path=diagnostics,
                cwd=workdir,
                env={**os.environ, **self.environment},
                timeout_s=timeout_s,
                on_start=lambda pid: (workdir / "sim.pid").write_text(str(pid), encoding="utf-8"),
            )
        returncode, timed_out = capture.returncode, capture.timed_out
        diag_text = diagnostics.read_text(encoding="utf-8", errors="replace")
        finished = finished_line(diag_text)
        assertion = capture.assertion
        completed = bool(finished and finished.get("done") == 1 and not timed_out and assertion is None)
        return {
            "machine": self.spec(),
            "device": identity.to_dict(),
            "argv": argv,
            "returncode": returncode,
            "timed_out": timed_out,
            "wall_seconds": round(time.time() - started, 3),
            "finished": finished,
            "completed": completed,
            "load_path": load_path_line(diag_text),
            "uart_log": str(uart),
            "uart_log_sha256": provenance.file_digest(uart),
            "diagnostics_log": str(diagnostics),
            "diagnostics_capture": {k: v for k, v in capture.to_dict().items() if k not in ("returncode", "timed_out")},
            "incomplete_reason": None
            if completed
            else (
                f"hardware_assertion: {assertion.get('message') or 'unnamed'}"
                + (f" at {assertion['location']}" if assertion.get("location") else "")
                if assertion is not None
                else "the wall-clock budget ended the run"
                if timed_out
                else "the emulator never printed its FINISHED line"
                if finished is None
                else f"the run stopped at its cycle budget ({finished.get('cycles')} cycles, done=0)"
                if finished.get("done") == 0
                else "the emulator finished without reporting completion"
            ),
        }


#: The queue's terminal job states, as its ``status`` table prints them.
QUEUE_TERMINAL = ("DONE", "FAILED", "CANCELLED", "TIMEOUT")
#: The session recorder's header, the log's OWN claim about when it began (a copy resets mtime).
SESSION_START_PREFIX = "Script started on "


def _sha(path: Path) -> str:
    return provenance.file_digest(path)


def _queue_state(table: str, job_id: int) -> str | None:
    """The state column of ``job_id``'s row in the queue's status table, parsed by columns."""
    for line in table.splitlines():
        tokens = line.split()
        if len(tokens) >= 4 and tokens[0] == str(job_id):
            return tokens[3]
    return None


def _queue_timeout_at_least(arguments: Sequence[str], timeout_s: float) -> list[str]:
    """The declared submit arguments with any ``--timeout N`` raised to this run's own budget: a batch
    of several programs in one ELF runs for longer than the one-program default, and a queue that
    kills it at the default leaves every variant of the batch incomplete."""
    out = [str(a) for a in arguments]
    for index, token in enumerate(out[:-1]):
        if token == "--timeout":
            try:
                out[index + 1] = str(max(int(float(out[index + 1])), int(timeout_s)))
            except ValueError:
                pass
    return out


#: The queue writes one banner per lifecycle phase into a job's stdout.log
#: (``=== [firesim-queue] phase=<NAME> job_id=<N> ===``); ``RUNNING`` is the phase in which the workload
#: itself runs on the board. A job that ended without reaching it never ran the program: whatever went
#: wrong (the FPGA absent, a failed flash or infrasetup) is the BOARD's, and says nothing about the bytes.
QUEUE_BANNER = ("===", "[firesim-queue]")
QUEUE_WORKLOAD_PHASE = "RUNNING"
INFRA_BOARD_UNAVAILABLE = "infra_board_unavailable"

#: Below this on the filesystem holding the machine's chipyard, a submission is refused before it reaches
#: the queue: measured 2026-10-01, board jobs each burned a full queue slot and failed at INFRASETUP with
#: ENOSPC under the chipyard's own results-build tree -- a fact a disk-usage call catches for free.
MIN_DEPLOY_FREE_BYTES = 20 * 1024**3
#: Below this on the HOST's root filesystem, the queue's own per-job child processes fail staging into
#: /tmp during infrasetup/kill, whatever this process's TMPDIR says.  Measured 2026-10-01: twelve
#: consecutive board jobs (about four hours of slots) failed with "No space left on device" under /tmp
#: while the deploy tree had 201 GB free -- nothing had looked at the host root itself.
MIN_HOST_ROOT_FREE_BYTES = 5 * 1024**3


def deploy_free_bytes(chipyard: Path) -> int:
    """Free bytes on the filesystem holding the FireSim deploy tree, derived from the machine's own
    ``chipyard`` path (never a literal filesystem name)."""
    return shutil.disk_usage(chipyard).free


def host_root_free_bytes() -> int:
    """Free bytes on the host's root filesystem, where the queue's child processes stage into ``/tmp``."""
    return shutil.disk_usage("/").free


def disk_refusal(chipyard: Path) -> str | None:
    """Why a submission must not reach the queue for lack of disk, or None.  Both checks are exact byte
    counts in the reason, so a refusal can be read against the threshold it missed."""
    free = deploy_free_bytes(chipyard)
    if free < MIN_DEPLOY_FREE_BYTES:
        return (
            f"{INFRA_BOARD_UNAVAILABLE}: the FireSim deploy filesystem at {chipyard} has only {free} byte(s) "
            f"free, below the {MIN_DEPLOY_FREE_BYTES} minimum; refusing before a board slot is spent"
        )
    root_free = host_root_free_bytes()
    if root_free < MIN_HOST_ROOT_FREE_BYTES:
        return (
            f"{INFRA_BOARD_UNAVAILABLE}: the host root filesystem has only {root_free} byte(s) free, below "
            f"the {MIN_HOST_ROOT_FREE_BYTES} minimum; the FireSim queue stages into /tmp there during "
            "infrasetup/kill whatever this process's TMPDIR says; refusing before a board slot is spent"
        )
    return None


def queue_phases(stdout_log: Path) -> list[str]:
    """The lifecycle phases a queue job's own log says it entered, in order (parsed by tokens)."""
    try:
        text = Path(stdout_log).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    phases = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) >= 3 and tuple(tokens[:2]) == QUEUE_BANNER and tokens[2].startswith("phase="):
            phases.append(tokens[2].partition("=")[2])
    return phases


def _job_id(output: str) -> int | None:
    for token in output.split():
        key, sep, value = token.partition("=")
        if sep and key == "job_id" and value.isdigit():
            return int(value)
    return None


def _session_start(uart_text: str) -> float | None:
    from datetime import datetime

    for line in uart_text.splitlines():
        if line.startswith(SESSION_START_PREFIX):
            stamp = line[len(SESSION_START_PREFIX) :].partition(" [")[0].strip()
            try:
                return datetime.fromisoformat(stamp).timestamp()
            except ValueError:
                return None
    return None


#: The submitter identity a cross-user queue daemon must not receive: with any of these forwarded it
#: resolves a home it cannot read and the job dies at ~35 s behind a misleading SSH error.
IDENTITY_VARIABLES = ("HOME", "USER", "LOGNAME")


def identity_dropped(queue_command: Sequence[str]) -> bool | list[str]:
    """True when ``queue_command`` begins with an ``env`` invocation that unsets every
    :data:`IDENTITY_VARIABLES`; otherwise the names it does not unset (parsed by tokens)."""
    tokens = [str(t) for t in queue_command]
    if not tokens or Path(tokens[0]).name != "env":
        return list(IDENTITY_VARIABLES)
    unset: set[str] = set()
    index = 1
    while index < len(tokens) - 1 and tokens[index] in ("-u", "--unset"):
        unset.add(tokens[index + 1])
        index += 2
    missing = [name for name in IDENTITY_VARIABLES if name not in unset]
    return True if not missing else missing


class FiresimMachine:
    """The shared FireSim board, driven one job at a time through its queue.

    Everything this learned the hard way is enforced here rather than remembered:

    * **The simulation reads the REAL deploy tree**, not the queue's per-job overlay (the launcher
      re-enters its own directory), so the ELF is copied to ``load_path`` and its digest is checked
      before submission AND after the job ends.  A load path whose digest moved during the job means
      another session restaged it; that run is refused, because nothing can say which bytes ran.
    * **The log is the job's own**, read at ``jobs_root/<id>/<uart>`` -- never through the queue's tail
      helper, which resolves a log by newest mtime across every job and user.  The log's own
      declared start must postdate the staging.
    * **The job's own record must name this ELF** as its stage source; a job id parsed from someone
      else's output is refused.
    * **One job at a time from this process tree** (a lock file), because the load path is one file.

    ``program_header_sha256`` is the ABI the operator DECLARES programs for this board are compiled
    against.  The registry records this bitstream's ABI as UNKNOWN, so a declared header is admitted
    and recorded as ``operator_declared`` -- never upgraded to verified.
    """

    kind = "firesim"
    role = "firesim_bitstream"
    rung = "fpga_firesim"

    def __init__(
        self,
        target: str,
        *,
        hw_config: str,
        chipyard: Path,
        workload: str,
        bootbinary: str,
        queue_command: tuple[str, ...],
        jobs_root: Path,
        uart_relative: str,
        submit_arguments: tuple[str, ...] = (),
        prepare_command: Sequence[Any] = (),
        program_header_sha256: str = "",
        lock_path: Path | None = None,
        poll_seconds: float = 20.0,
        runner: Any = None,
    ) -> None:
        if not target.strip() or not hw_config.strip() or not workload.strip() or not bootbinary.strip():
            raise MachineRefusal("a FireSim machine needs its target, hw-config, workload and bootbinary")
        if not queue_command:
            raise MachineRefusal("a FireSim machine needs the queue command")
        dropped = identity_dropped(queue_command)
        if dropped is not True:
            raise MachineRefusal(
                f"the queue command must drop the submitter's identity ({', '.join(IDENTITY_VARIABLES)}) "
                f"before invoking the queue, e.g. `env -u HOME -u USER -u LOGNAME <queue>`; missing: {dropped}"
            )
        self.target = target
        self.hw_config = hw_config
        self.chipyard = Path(chipyard)
        self.workload = workload
        self.bootbinary = bootbinary
        self.queue_command = tuple(queue_command)
        self.jobs_root = Path(jobs_root)
        self.uart_relative = uart_relative
        self.submit_arguments = tuple(submit_arguments)
        # One argv, or a list of argvs run in order. Each is a declared host step, logged in full.
        steps = list(prepare_command or ())
        if steps and all(isinstance(token, str) for token in steps):
            steps = [steps]
        self.prepare_command = tuple(tuple(str(token) for token in argv) for argv in steps)
        self.program_header_sha256 = program_header_sha256
        self.lock_path = (
            Path(lock_path)
            if lock_path
            else Path(os.environ.get("TMPDIR", "/tmp")) / (f"merlin_firesim_{os.getuid()}.lock")
        )
        self.poll_seconds = float(poll_seconds)
        self._run = runner or (lambda argv, **kw: subprocess.run(argv, capture_output=True, text=True, **kw))

    @property
    def load_path(self) -> Path:
        """The file the simulation reads, which is NOT necessarily the file the queue staged: the
        launcher re-enters its own deploy tree before resolving the workload input, so a loop that
        verified only the queue's staged copy would verify a file no simulation ever opened."""
        return deploy_load_path(self.chipyard, self.workload, self.bootbinary)

    def spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "hw_config": self.hw_config,
            "chipyard": str(self.chipyard),
            "workload": self.workload,
            "bootbinary": self.bootbinary,
            "queue_command": list(self.queue_command),
            "jobs_root": str(self.jobs_root),
            "uart_relative": self.uart_relative,
            "submit_arguments": list(self.submit_arguments),
            "prepare_command": [list(argv) for argv in self.prepare_command],
            "program_header_sha256": self.program_header_sha256,
            "lock_path": str(self.lock_path),
            "poll_seconds": self.poll_seconds,
        }

    def identity(self) -> DeviceIdentity:
        matches = [
            artifact
            for artifact in provenance.load_artifacts().values()
            if self.hw_config in artifact.hw_configs and artifact.role == self.role and artifact.target == self.target
        ]
        if len(matches) != 1:
            raise MachineRefusal(
                f"{len(matches)} registered {self.role} artifacts for {self.target!r} name hw-config "
                f"{self.hw_config!r}; exactly one device must answer to a submission name"
            )
        artifact = matches[0]
        check = provenance.verify_artifact(artifact.name)
        artifacts = provenance.load_artifacts()
        return DeviceIdentity(
            machine=self.kind,
            target=self.target,
            binary=check.path,
            binary_sha256=artifact.digest,
            artifact=artifact.name,
            config=artifact.config,
            built_from=artifact.built_from,
            abi_header_sha256=_abi_along(artifact, artifacts),
            rung=self.rung,
            registry_verification={artifact.name: check.to_dict()},
            engine_citation={"hw_config": self.hw_config, "queue": list(self.queue_command)},
        )

    def admit(self, identity: DeviceIdentity, *, program_header_sha256: str | None) -> None:
        if not program_header_sha256:
            raise MachineRefusal("the build record states no parameter header; its ABI is UNKNOWN")
        if identity.abi_header_sha256 != UNKNOWN:
            if program_header_sha256 != identity.abi_header_sha256:
                raise MachineRefusal(
                    f"the program's header {program_header_sha256[:8]} is not the device's "
                    f"{identity.abi_header_sha256[:8]}"
                )
            return
        if not self.program_header_sha256 or program_header_sha256 != self.program_header_sha256:
            raise MachineRefusal(
                f"the registry records no ABI for {identity.artifact} and the program's header "
                f"{program_header_sha256[:8]} is not the operator-declared {self.program_header_sha256[:8] or 'NONE'}"
            )

    def run(self, elf: Path, workdir: Path, *, timeout_s: float) -> dict[str, Any]:
        import fcntl

        identity = self.identity()
        workdir.mkdir(parents=True, exist_ok=True)
        elf = Path(elf)
        elf_sha = _sha(elf)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "machine": self.spec(),
            "device": identity.to_dict(),
            "abi_status": "registry" if identity.abi_header_sha256 != UNKNOWN else "operator_declared",
            "elf_sha256": elf_sha,
            "load_path": str(self.load_path),
        }
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if self.prepare_command:
                    # THE HOST PREPARATION, explicit and logged: every step's argv, exit status and
                    # output go beside the measurement, so a job that later fails in infrasetup can be
                    # read against what was (or was not) prepared for it.
                    log_lines, codes = [], []
                    for argv in self.prepare_command:
                        prepared = self._run(list(argv))
                        codes.append(prepared.returncode)
                        log_lines.append(
                            f"$ {' '.join(argv)}\nrc={prepared.returncode}\n"
                            f"{prepared.stdout or ''}{prepared.stderr or ''}"
                        )
                    (workdir / "prepare.log").write_text("\n".join(log_lines), encoding="utf-8")
                    record["prepare_returncodes"] = codes
                    record["prepare_log"] = str(workdir / "prepare.log")
                load = self.load_path
                record["load_sha256_prior"] = _sha(load)
                shutil.copyfile(elf, load)
                with contextlib_suppress(OSError):
                    load.chmod(0o666)
                staged_epoch = time.time()
                staged = _sha(load)
                record["load_sha256_staged"] = staged
                if staged != elf_sha:
                    return {**record, "completed": False, "incomplete_reason": f"staging wrote {staged}, not {elf_sha}"}
                # REFUSED HERE, before the queue sees it: a full disk fails at INFRASETUP every time, but
                # only after a real queue slot and the wall time to discover it there.
                low_disk = disk_refusal(self.chipyard)
                if low_disk is not None:
                    return {**record, "completed": False, INFRA_BOARD_UNAVAILABLE: True, "incomplete_reason": low_disk}
                submitted = self._run(
                    [
                        *self.queue_command,
                        FIRESIM_QUEUE_OPERATION,
                        "--chipyard",
                        str(self.chipyard),
                        "--workload",
                        self.workload,
                        "--stage-from",
                        str(elf),
                        "--hw-config",
                        self.hw_config,
                        *_queue_timeout_at_least(self.submit_arguments, timeout_s),
                        "--background",
                    ],
                    timeout=120,
                )
                job_id = _job_id((submitted.stdout or "") + " " + (submitted.stderr or ""))
                record["submit_returncode"] = submitted.returncode
                if job_id is None:
                    return {
                        **record,
                        "completed": False,
                        "incomplete_reason": "the queue printed no job id: "
                        + (submitted.stdout or submitted.stderr or "")[-400:],
                    }
                record["job_id"] = job_id
                deadline = time.time() + timeout_s
                state = None
                while time.time() < deadline:
                    table = self._run([*self.queue_command, "status", "--all"], timeout=60)
                    state = _queue_state(table.stdout or "", job_id)
                    if state in QUEUE_TERMINAL:
                        break
                    time.sleep(self.poll_seconds)
                record["job_state"] = state
                after = _sha(load)
                record["load_sha256_after"] = after
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        job_dir = self.jobs_root / str(job_id)
        job_record_path = job_dir / "runworkload-full.json"
        try:
            job_record = json.loads(job_record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            job_record = {}
        record["job_record"] = job_record
        uart_source = job_dir / self.uart_relative
        reasons = []
        if state not in QUEUE_TERMINAL:
            reasons.append(f"job {job_id} did not finish within {timeout_s:.0f}s (state {state})")
        elif state != "DONE":
            reasons.append(f"job {job_id} ended {state}")
        if after != elf_sha:
            reasons.append(
                f"the load path changed during job {job_id} ({elf_sha[:12]} -> {after[:12]}); another session "
                "restaged it, so which bytes ran cannot be said"
            )
        if str(job_record.get("stage_from") or "") != str(elf):
            reasons.append(
                f"job {job_id}'s own record names stage source {job_record.get('stage_from')!r}, not this ELF"
            )
        if job_record.get("hw_config") not in (None, self.hw_config):
            reasons.append(f"job {job_id} ran under hw-config {job_record.get('hw_config')!r}")
        phases = queue_phases(job_dir / "stdout.log")
        record["queue_phases"] = phases
        if state in QUEUE_TERMINAL and state != "DONE" and QUEUE_WORKLOAD_PHASE not in phases:
            # THE WORKLOAD NEVER RAN: the board was not available to it (absent FPGA, failed flash or
            # infrasetup). Not an incomplete RUN of these bytes -- nothing of them executed.
            tail = ""
            with contextlib_suppress(OSError):
                tail = (job_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")[-300:].strip()
            return {
                **record,
                "completed": False,
                INFRA_BOARD_UNAVAILABLE: True,
                "incomplete_reason": f"{INFRA_BOARD_UNAVAILABLE}: job {job_id} ended {state} before its workload "
                f"ran (phases {phases or 'none recorded'}){'; ' + tail if tail else ''}",
            }
        uart = workdir / "uart.log"
        if uart_source.is_file() and not uart_source.is_symlink():
            shutil.copyfile(uart_source, uart)
            started = _session_start(uart.read_text(encoding="utf-8", errors="replace"))
            record["uart_session_start_epoch"] = started
            if started is None or started < staged_epoch - 5:
                reasons.append("the uart log's own declared start does not postdate this staging")
        else:
            reasons.append(f"job {job_id} has no uart log at {uart_source}")
        return {
            **record,
            "uart_source": str(uart_source),
            "uart_log": str(uart),
            "uart_log_sha256": _sha(uart) if uart.is_file() else UNKNOWN,
            "completed": not reasons,
            "incomplete_reason": "; ".join(reasons) or None,
        }


class SpikeMachine:
    """A functional model run locally (no board, no queue): the machine a LOCAL grade is taken on.

    Its cycle count is not a measurement of anything and is never read. What it contributes is a
    complete run of the program -- every group's local grade, the classification, and one byte digest
    per group -- in minutes on the host, in parallel with the board. Identity is the digest of every
    file the caller declares it runs (the simulator and its extension library), because a functional
    model built for another parameter set returns wrong numbers rather than failing.
    """

    kind = "spike"
    rung = "functional_model"

    def __init__(
        self,
        target: str,
        *,
        command: Sequence[str],
        environment: Mapping[str, str] | None = None,
        identity_files: Sequence[str] = (),
    ) -> None:
        if not command:
            raise MachineRefusal("a functional-model machine needs its command")
        self.target = target
        self.command = tuple(str(c) for c in command)
        self.environment = dict(environment or {})
        self.identity_files = tuple(str(p) for p in identity_files)

    def spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "command": list(self.command),
            "environment": dict(self.environment),
            "identity_files": list(self.identity_files),
        }

    def identity(self) -> DeviceIdentity:
        digests = {path: _sha(Path(path)) for path in (self.command[0], *self.identity_files)}
        if any(value == UNKNOWN for value in digests.values()):
            raise MachineRefusal(f"a functional-model file cannot be hashed: {digests}")
        return DeviceIdentity(
            machine=self.kind,
            target=self.target,
            binary=self.command[0],
            binary_sha256=digests[self.command[0]],
            artifact=UNKNOWN,
            config="",
            built_from=(),
            abi_header_sha256=UNKNOWN,
            rung=self.rung,
            registry_verification={},
            engine_citation={"files": digests},
        )

    def admit(self, identity: DeviceIdentity, *, program_header_sha256: str | None) -> None:
        return None

    def run(self, elf: Path, workdir: Path, *, timeout_s: float) -> dict[str, Any]:
        identity = self.identity()
        workdir.mkdir(parents=True, exist_ok=True)
        uart, diagnostics = workdir / "uart.log", workdir / "sim.log"
        argv = [*self.command, str(elf)]
        started = time.time()
        with uart.open("wb") as out, diagnostics.open("wb") as err:
            process = subprocess.Popen(
                argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL, env={**os.environ, **self.environment}
            )
            try:
                returncode, timed_out = process.wait(timeout=timeout_s), False
            except subprocess.TimeoutExpired:
                process.kill()
                returncode, timed_out = process.wait(), True
        completed = returncode == 0 and not timed_out
        return {
            "machine": self.spec(),
            "device": identity.to_dict(),
            "argv": argv,
            "returncode": returncode,
            "timed_out": timed_out,
            "wall_seconds": round(time.time() - started, 3),
            "completed": completed,
            "uart_log": str(uart),
            "uart_log_sha256": provenance.file_digest(uart),
            "diagnostics_log": str(diagnostics),
            "incomplete_reason": None
            if completed
            else ("the wall-clock budget ended the run" if timed_out else f"the model exited {returncode}"),
        }


def contextlib_suppress(*exceptions):
    import contextlib

    return contextlib.suppress(*exceptions)


#: Machine kinds that compose other machines or delegate to another owner; :func:`machine_from_spec`
#: never builds one of these directly (see :mod:`.worker`).
COMPOSITE_KINDS = ("paired", "batched", "contract", "cell")


def machine_from_spec(spec: Mapping[str, Any]) -> GsimMachine | FiresimMachine | SpikeMachine:
    """Rebuild a machine from its ``spec()`` -- the one form a worker process receives."""
    kind = spec.get("kind")
    if kind == GsimMachine.kind:
        emulator = spec.get("emulator")
        return GsimMachine(
            str(spec["target"]),
            max_cycles=int(spec["max_cycles"]),
            environment=dict(spec.get("environment") or {}),
            emulator=Path(str(emulator)) if emulator else None,
        )
    if kind == FiresimMachine.kind:
        return FiresimMachine(
            str(spec["target"]),
            hw_config=str(spec["hw_config"]),
            chipyard=Path(str(spec["chipyard"])),
            workload=str(spec["workload"]),
            bootbinary=str(spec["bootbinary"]),
            queue_command=tuple(spec["queue_command"]),
            jobs_root=Path(str(spec["jobs_root"])),
            uart_relative=str(spec["uart_relative"]),
            submit_arguments=tuple(spec.get("submit_arguments") or ()),
            prepare_command=list(spec.get("prepare_command") or ()),
            program_header_sha256=str(spec.get("program_header_sha256") or ""),
            lock_path=Path(str(spec["lock_path"])) if spec.get("lock_path") else None,
            poll_seconds=float(spec.get("poll_seconds") or 20.0),
        )
    if kind == SpikeMachine.kind:
        return SpikeMachine(
            str(spec["target"]),
            command=list(spec["command"]),
            environment=dict(spec.get("environment") or {}),
            identity_files=list(spec.get("identity_files") or ()),
        )
    if kind in COMPOSITE_KINDS:
        raise MachineRefusal(f"a {kind!r} machine composes others; it is not run directly")
    raise MachineRefusal(f"no whole-model machine of kind {kind!r}")


__all__ = [
    "COMPOSITE_KINDS",
    "DeviceIdentity",
    "FiresimMachine",
    "SpikeMachine",
    "deploy_load_path",
    "GsimMachine",
    "MachineRefusal",
    "finished_line",
    "load_path_line",
    "machine_from_spec",
    "registered_artifact_for",
]
