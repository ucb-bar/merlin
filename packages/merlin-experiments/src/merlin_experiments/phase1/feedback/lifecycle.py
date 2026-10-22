"""Host-side feedback lifetimes, scheduling and channel-delivery accounting.

Brokers live inside an agent launch; the background grader surrounds that launch.
Keep the separate stop boundaries: stopping a thread is an unbounded single-flight
handoff, not cancellation or a timed join. Numerical grading is supplied explicitly.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from merlin.common.paths import module_source_path
from merlin.targetgen import tool_registry as _TR

from ..compiler_library import join_background, library_thread
from ..context import InvocationContext, context_argv

FIRST_BACKGROUND_TICK = 900
FIRST_GRADE_POLL_S = 30
BROKER_GRACE_SECONDS = 15
BROKER_REAP_SECONDS = 5


@dataclass(frozen=True)
class BrokerConfig:
    context: InvocationContext
    tools: tuple[str, ...]
    timing_file: Path
    sim_max_jobs: int = 0
    capsules_root: Path | None = None
    policy_root: Path | None = None
    contract: Path | None = None
    selected_rtl_facts: Path | None = None
    compiler_bundle_manifest: Path | None = None
    sim_per_capsule_timeout: int = 0
    raw_evidence_root: Path | None = None

    def __post_init__(self):
        from ..options import MAX_SIM_PER_CAPSULE_TIMEOUT_S

        if (
            type(self.sim_per_capsule_timeout) is not int
            or not 0 <= self.sim_per_capsule_timeout <= MAX_SIM_PER_CAPSULE_TIMEOUT_S
        ):
            raise ValueError("sim per-capsule timeout must be an explicit integer in 0..86400")


@dataclass(frozen=True)
class GradeCadence:
    qa_timeout: int
    no_oracle: bool
    grade_interval: int


class IntervalGrade(Protocol):
    def __call__(
        self,
        ws: Path,
        run_dir: Path,
        tick: int,
        no_oracle: bool,
        timeout: int,
        *,
        label: str,
        scratch_key: str,
        previous_scratch_key: str | None,
    ) -> dict: ...


class FastGrade(Protocol):
    def __call__(self, ws: Path, run_dir: Path, tick: int, timeout: int) -> dict: ...


def stage_client(ws: Path, source: Path, dst_name: str) -> None:
    """Copy an in-box shim into the ws, replacing any bound symlink (copy-onto-symlink would clobber
    the real script — learned the hard way)."""
    dst = ws / dst_name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy(source, dst)


def start_brokers(ws: Path, config: BrokerConfig):
    """Stage the in-box shims (sync self-check + async simjob) and launch BOTH driver-side brokers, which
    run the redacted grade OUTSIDE the sandbox so the oracle never enters the box. Returns a list of
    Popens. The async simjob broker lets the agent run slow verilator per-capsule without blocking a turn."""
    capsules_root, policy_root, contract = config.capsules_root, config.policy_root, config.contract

    ch = ws / ".qa_channel"
    ch.mkdir(parents=True, exist_ok=True)
    (ch / "STOP").unlink(missing_ok=True)
    # Blocking wait on the harness's OWN grade. Not a broker and not a channel: it reads only
    # qa/verdict.json, which the agent may already read, and imports nothing from merlin. It exists
    # because every other wait in the toolbox blocks and this one did not, so the only way to notice a
    # new grade was to look again -- measured at 89 polling commands in one 6.1 h run, each one a model
    # round trip.
    for client_module, staged_as in _TR.COMMON_CLIENTS:
        stage_client(ws, module_source_path(client_module), staged_as)
    broker_specs = [
        (spec.module, spec.log, spec.channel)
        for spec in _TR.COMMON_BROKERS
        if spec.channel != ".model_check_channel" or config.context.target == "gemmini"  # target-ok: full-model
    ]
    # Brokered TOOLS (the ISA assembler/disassembler/linter; the two mandated CCA calls) are part of the
    # arm's TREATMENT, so which ones start is read from the bundle's resolved tool set rather than from
    # the arm's name. That distinction matters twice over: the CIRCT arm runs under _ARM ==
    # "merlin_assisted" with a swapped bundle, so a name test cannot see it; and an ablation cell differs
    # from its rung only in that tool set. Each broker gets its own channel dir + STOP, like the
    # self-check. All of them are oracle-free and read no golden.
    for bs in _TR.brokers_for(config.tools):
        ch_dir = ws / bs.channel
        ch_dir.mkdir(parents=True, exist_ok=True)
        (ch_dir / "STOP").unlink(missing_ok=True)
        for shim_src, staged_as in bs.shims:
            stage_client(ws, module_source_path(shim_src), staged_as)
        broker_specs.append((bs.module, bs.log, bs.channel))
    brokers = []
    owned_channels = []
    try:
        for name, log, channel in broker_specs:
            command = ["-m", name]
            argv = [sys.executable, *command, "--ws", str(ws)]
            if name in {
                "merlin_experiments.phase1.brokers.isa_tools",
                "merlin_experiments.phase1.brokers.model_check",
                "merlin_experiments.phase1.brokers.selfcheck",
                "merlin_experiments.phase1.brokers.simjob",
            }:
                if name == "merlin_experiments.phase1.brokers.isa_tools":
                    # ISA inspection does not execute or grade output buffers.
                    argv += ["--descriptor", str(config.context.descriptor), "--repo", str(config.context.repo)]
                else:
                    argv += context_argv(config.context)
            if config.selected_rtl_facts is not None and name in {
                "merlin_experiments.phase1.brokers.isa_tools",
                "merlin_experiments.phase1.brokers.cca",
                "merlin_experiments.phase1.brokers.selfcheck",
                "merlin_experiments.phase1.brokers.simjob",
            }:
                argv += ["--rtl-facts", str(config.selected_rtl_facts)]
            if name == "merlin_experiments.phase1.brokers.simjob":
                argv += ["--timing-file", str(config.timing_file)]
            if config.compiler_bundle_manifest is not None and name in {
                "merlin_experiments.phase1.brokers.model_check",
                "merlin_experiments.phase1.brokers.selfcheck",
                "merlin_experiments.phase1.brokers.simjob",
            }:
                argv += ["--compiler-bundle-manifest", str(config.compiler_bundle_manifest)]
            if name in {"merlin_experiments.phase1.brokers.selfcheck", "merlin_experiments.phase1.brokers.simjob"}:
                if capsules_root is not None:
                    argv += ["--capsules-root", str(capsules_root)]
                if policy_root is not None:
                    argv += ["--policy-capsules-root", str(policy_root)]
                if contract is not None:
                    argv += ["--contract", str(contract)]
            # HOW MANY CERT JOBS MAY RUN AT ONCE is an operator decision about THIS machine, and until now
            # it could not be made: the broker was launched with only --ws, so it always fell back to its
            # own default of 4 no matter what engine was certifying. That default was set when the cert tier
            # meant Verilator, which the broker separately caps at 2 global slots because one instance eats a
            # core for ~45 min. GSIM is a different animal -- measured on this host: 10.4 MB RSS, one thread,
            # ~12 s per capsule, and 24 concurrent instances returned bit-identical cycle counts while total
            # throughput rose 13.4x. Capping that at 4 leaves the cert tier running at a sixth of what the
            # machine will give. Forwarded ONLY to the broker that has the flag (the tool brokers and the
            # self-check broker take --ws alone), and only when the operator set it, so the default path is
            # byte-identical to before.
            if config.sim_max_jobs and name == "merlin_experiments.phase1.brokers.simjob":
                argv += ["--max-jobs", str(config.sim_max_jobs)]
            if config.sim_per_capsule_timeout and name == "merlin_experiments.phase1.brokers.simjob":
                argv += ["--per-capsule-timeout", str(config.sim_per_capsule_timeout)]
            if name in {"merlin_experiments.phase1.brokers.selfcheck", "merlin_experiments.phase1.brokers.simjob",
                        "merlin_experiments.phase1.brokers.model_check"}:
                from ..compiler_library import child_arguments

                argv += child_arguments()
            channel_dir = ws / channel
            channel_dir.mkdir(parents=True, exist_ok=True)
            (channel_dir / "STOP").unlink(missing_ok=True)
            output = open(channel_dir / log, "w")
            launch_error = None
            try:
                from merlin_experiments.frozen_python import inherited_python_command

                launch_kwargs = {}
                if name in {
                    "merlin_experiments.phase1.brokers.selfcheck",
                    "merlin_experiments.phase1.brokers.simjob",
                }:
                    from .raw_selfcheck_evidence import broker_environment

                    launch_kwargs["env"] = broker_environment(config.raw_evidence_root, ws)
                brokers.append(
                    subprocess.Popen(
                        inherited_python_command(argv), stdout=output, stderr=subprocess.STDOUT, **launch_kwargs
                    )
                )
                owned_channels.append(channel)
            except BaseException as error:
                launch_error = error
                raise
            finally:
                try:
                    output.close()
                except BaseException as close_error:
                    if launch_error is None:
                        raise
                    launch_error.add_note(f"broker log close failed: {type(close_error).__name__}: {close_error}")
    except BaseException as error:
        _stop_brokers(ws, brokers, primary_error=error, channels=tuple(dict.fromkeys(owned_channels)))
        raise
    return brokers


class BrokerCleanupError(RuntimeError):
    """All owned children were attempted, but one or more cleanup steps failed."""

    def __init__(self, failures: list[str], unreaped_pids: list[int | None]):
        self.failures = tuple(failures)
        self.unreaped_pids = tuple(unreaped_pids)
        detail = "; ".join(failures)
        super().__init__(f"broker cleanup failed: {detail}; reap unconfirmed for PIDs {unreaped_pids}")


def stop_brokers(ws: Path, brokers, *, primary_error: BaseException | None = None) -> None:
    """Signal, then wait or kill/reap every owned broker (not detached descendants).

    A successfully reaped forced stop is not a cleanup failure. Actual signal,
    kill or reap errors are reported after all children have been attempted. A
    propagating provider/launch exception remains primary; callers pass it from
    their finally block. Cleanup details are attached to it, never substituted.
    """
    _stop_brokers(
        ws,
        brokers,
        primary_error=primary_error,
        channels=(".qa_channel", ".model_check_channel", ".isa_channel", ".cca_channel"),
    )


def _stop_brokers(ws: Path, brokers, *, primary_error: BaseException | None, channels: tuple[str, ...]) -> None:
    # Startup rollback may signal only channels with a successfully created child.
    # Normal shutdown retains the historical workspace-wide STOP-channel policy.
    if not brokers:
        return
    failures: list[str] = []
    unreaped: list[int | None] = []
    interrupt: BaseException | None = None

    def failed(action: str, error: BaseException) -> None:
        nonlocal interrupt
        failures.append(f"{action}: {type(error).__name__}: {error}")
        if not isinstance(error, Exception) and interrupt is None:
            interrupt = error

    for channel in channels:
        try:
            directory = ws / channel
            if channel == ".qa_channel" or directory.is_dir():
                (directory / "STOP").write_text("stop")
        except BaseException as error:
            failed(f"signal {channel}", error)
    for broker in brokers if isinstance(brokers, list) else [brokers]:
        pid = getattr(broker, "pid", None)
        try:
            broker.wait(timeout=BROKER_GRACE_SECONDS)
            continue
        except BaseException as error:
            if not isinstance(error, Exception):
                failed(f"graceful wait PID {pid}", error)
        try:
            broker.kill()
        except BaseException as error:
            failed(f"kill PID {pid}", error)
        try:
            broker.wait(timeout=BROKER_REAP_SECONDS)
        except BaseException as error:
            unreaped.append(pid)
            failed(f"reap PID {pid}", error)
    if failures:
        cleanup = BrokerCleanupError(failures, unreaped)
        if primary_error is not None:
            primary_error.add_note(str(cleanup))
        elif interrupt is not None:
            interrupt.add_note(str(cleanup))
            raise interrupt
        else:
            raise cleanup


def channel_health(ws: Path, *, now: float | None = None, live: bool = False) -> dict:
    """Durable end-of-run accounting for the synchronous self-check channel.

    Expired requests are abandoned clients, not model failures, but they still prove the promised
    synchronous feedback was not delivered. Any request without an atomic response+done pair makes the
    channel unhealthy; ``expired`` versus ``stranded`` distinguishes a past timeout from live work.

    ``live`` is the account taken WHILE the brokers still serve: a request that is still inside its own
    deadline is ``in_flight``, not stranded, and does not fail the channel. Without it the first grade's
    pool lock-out raised "channel UNHEALTHY ... 3 stranded" at minute four of a run for three requests
    that were merely queued and were all answered. At the end of a run (the default) nobody will answer
    an incomplete request, so it stays stranded there.
    """
    ch = Path(ws) / ".qa_channel"
    now = time.time() if now is None else now
    if not ch.is_dir():
        return {
            "protocol": 3,
            "requests": 0,
            "completed": 0,
            "expired": 0,
            "stranded": 0,
            **({"in_flight": 0} if live else {}),
            "orphan_responses": 0,
            "done_without_response": 0,
            "replayed": 0,
            "max_queue_depth": 0,
            "healthy": True,
        }
    requests = {p.stem[len("req_") :]: p for p in ch.glob("req_*.json")}
    responses = {p.stem[len("resp_") :]: p for p in ch.glob("resp_*.json")}
    done = {p.name[len("done_") :] for p in ch.glob("done_*")}
    # THE DENOMINATOR IS NOT A DIRECTORY LISTING. `requests` globs the request files still on disk, and
    # the completed set was a SUBSET of it, so a request whose req_ file is gone left the numerator and
    # the denominator together and became invisible. Measured on a live run: feedback_health.json read
    # "requests: 59, completed: 59" while the channel held 67 req_, 72 resp_ and 72 done_ -- five
    # exchanges the ratio could not see, and a perfect score that was an artifact of what had been
    # deleted. A response or a done proves a request existed, so the universe is their union.
    universe = set(requests) | set(responses) | done
    completed = {rid for rid in universe if rid in responses and rid in done}
    expired: set[str] = set()
    in_flight: set[str] = set()
    from merlin_experiments.phase1.brokers.selfcheck import _request_deadline

    for rid, req in requests.items():
        if rid not in completed:
            try:
                if now > _request_deadline(req):
                    expired.add(rid)
                elif live:
                    in_flight.add(rid)
            except (FileNotFoundError, OSError, ValueError):
                pass
    # An incomplete rid whose req_ file is gone cannot have a deadline read, so it can never be shown to
    # have expired; it is stranded, which is the reportable half either way.
    stranded = universe - completed - expired - in_flight
    broker_stats = {}
    health_file = ch / "broker_health.json"
    if health_file.is_file():
        try:
            broker_stats = json.loads(health_file.read_text(encoding="utf-8"))
        except Exception:
            broker_stats = {"health_record_error": "unreadable"}
    orphan_responses = set(responses) - set(requests)
    done_without_response = done - set(responses)
    # A response and a done together ARE the delivery this gate exists to check, so an orphan that
    # completed is an accounting curiosity, not an undelivered exchange; it stays reported (something
    # removed a request file and that is worth seeing) but no longer fails the channel on its own. An
    # orphan that did NOT complete is already counted in `stranded` above.
    completed_without_request = orphan_responses & completed
    healthy = not expired and not stranded and not done_without_response
    return {
        "protocol": 3,
        "requests": len(universe),
        "requests_on_disk": len(requests),
        "completed": len(completed),
        "completed_without_request": len(completed_without_request),
        "expired": len(expired),
        "stranded": len(stranded),
        **({"in_flight": len(in_flight)} if live else {}),
        "orphan_responses": len(orphan_responses),
        "done_without_response": len(done_without_response),
        "replayed": int(broker_stats.get("replayed", 0)),
        "max_queue_depth": int(broker_stats.get("max_queue_depth", 0)),
        "broker_starts": int(broker_stats.get("broker_starts", 0)),
        "healthy": healthy,
    }


GRADE_FAILURE_ALERT_ENV = "MERLIN_GRADE_FAILURE_ALERT_AFTER"
GRADE_FAILURE_ALERT_DEFAULT = 3


def grade_failure_alert_threshold() -> int:
    """Consecutive identical in-turn grade failures that raise the alert (``0`` records but never alerts)."""
    raw = os.environ.get(GRADE_FAILURE_ALERT_ENV, str(GRADE_FAILURE_ALERT_DEFAULT))
    if not raw.isascii() or not raw.isdecimal() or not 0 <= int(raw) <= 1000:
        raise ValueError(f"{GRADE_FAILURE_ALERT_ENV} must be an integer between 0 and 1000")
    return int(raw)


def _failure_class(error: BaseException) -> str:
    """A failure's identity for repeat counting: type and message, with every run of digits folded.

    Ticks, PIDs and scratch keys differ between failures that are otherwise the same failure.
    """
    folded, digits = [], False
    for character in f"{type(error).__name__}: {error}"[:500]:
        if not (character.isdigit() and digits):
            folded.append("#" if character.isdigit() else character)
        digits = character.isdigit()
    return "".join(folded)


class GradeFailureAlert:
    """Operator-side alarm for in-turn grades that keep failing the same way. Never raises.

    One failed grade is ordinary (a tree caught mid-write). The same failure again and again is not:
    measured, 75 consecutive in-turn grades died with one identical ``StageGateError`` over 4.8 h, each
    printed as an ordinary "skipped" line, and nothing said the agent had no grade feedback for the
    whole run. At ``threshold`` repeats, and every ``threshold`` after, this prints ``[grade-alert]`` and
    ``grade_health.json`` (run directory only, never the agent's workspace: feedback defines an arm)
    carries ``alert: true`` until a grade completes.
    """

    SCHEMA = "merlin.phase1.grade_health.v1"

    def __init__(self, run_dir: Path, *, threshold: int):
        self.path = Path(run_dir) / "grade_health.json"
        self.threshold = threshold
        self.signature: str | None = None
        self.error: str | None = None
        self.count = 0
        self.first_failed_at: str | None = None
        self.last_failed_at: str | None = None
        self.last_completed_at: str | None = None
        self.alerts_raised = 0

    @property
    def alert(self) -> bool:
        return self.threshold > 0 and self.count >= self.threshold

    def failed(self, error: BaseException) -> None:
        try:
            now = _utc_now()
            signature = _failure_class(error)
            if signature != self.signature:
                self.signature, self.count, self.first_failed_at = signature, 0, now
            self.count += 1
            self.error, self.last_failed_at = f"{type(error).__name__}: {error}"[:500], now
            if self.alert and self.count % self.threshold == 0:
                self.alerts_raised += 1
                print(
                    f"[grade-alert] {self.count} consecutive in-turn grades failed with the same error since "
                    f"{self.first_failed_at}: {self.error[:300]} — the agent is getting no grade feedback; "
                    f"inspect the grader, not the submission",
                    flush=True,
                )
            self._write()
        except Exception:  # noqa: BLE001 -- diagnostics may never fail a grade
            return

    def succeeded(self) -> None:
        try:
            self.signature, self.error, self.count = None, None, 0
            self.last_completed_at = _utc_now()
            self._write()
        except Exception:  # noqa: BLE001 -- diagnostics may never fail a grade
            return

    def _write(self) -> None:
        record = {
            "schema": self.SCHEMA,
            "updated_at": _utc_now(),
            "alert": self.alert,
            "consecutive_identical_failures": self.count,
            "threshold": self.threshold,
            "error": self.error,
            "error_class": self.signature,
            "first_failed_at": self.first_failed_at if self.count else None,
            "last_failed_at": self.last_failed_at if self.count else None,
            "last_completed_at": self.last_completed_at,
            "alerts_raised": self.alerts_raised,
            "scope": "in-turn interval grades; operator-side record, never shown to the agent",
        }
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


def _utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.UTC).isoformat()


def grade_key(round_index: int, tick: int) -> str:
    """A run-wide scratch identity for one in-turn grade.

    The human-facing numeric tick deliberately remains in its historical band; only filesystem and
    archive identities need the agent-round coordinate that prevents a later round from reopening a
    frozen snapshot.
    """
    if round_index < 0 or tick < 1:
        raise ValueError("in-turn grade coordinates must be a non-negative round and positive tick")
    return f"r{round_index:04d}_t{tick:06d}"


def wait_for_stable_source(source_identity, stop, *, quiet_s=30, max_wait_s=120):
    """Debounce superseded advisory work without holding a CPU lease.

    This is scheduling only: the grade still captures/verifies its own source.
    Continuous edits cannot postpone feedback beyond max_wait_s; explicit
    self-checks and final grades never pass through this helper.
    """
    started = time.monotonic()
    previous = source_identity()
    stable_since = time.monotonic()
    while not stop.is_set():
        now = time.monotonic()
        if now - started >= max_wait_s:
            return previous
        if previous is not None and now - stable_since >= quiet_s:
            return previous
        if stop.wait(min(5, max_wait_s - (now - started))):
            break
        current = source_identity()
        if current != previous or current is None:
            previous, stable_since = current, time.monotonic()
    return None


def start_background(
    ws: Path,
    run_dir: Path,
    cadence: GradeCadence,
    *,
    interval_grades: bool,
    round_index: int = 0,
    on_tick=None,
    grade_callback: IntervalGrade,
    fast_grade_callback: FastGrade,
    source_identity=None,
):
    """Start the in-turn grader for ONE agent turn; returns a handle for `stop_background`.

    ``on_tick`` is called after each interval grade with that tick's index. It exists so telemetry can
    be sunk on this cadence: under ``--schedule continuous`` this is the ONLY loop that runs while the
    agent works, so anything that must survive a kill has to hang off it and not off the end of the run.

    Phase 1 (always): if the agent has no verdict at all, land one — the fast loop-tier grade above —
    as soon as there is a submission to grade. Phase 2 (long turns only): keep re-grading on
    --grade-interval with the FULL mandatory ladder, which is the grade that can converge the run.
    Neither phase touches the driver's own `verdict`: the authoritative round verdict is still the
    post-turn `qa_grade`, and neither phase checkpoints (a background tick is not a round number).

    Both phases lease shared compute at the GRADE class: a concurrent author self-check (AUTHOR)
    is served first CPU by CPU, but a cadence is never skipped while one runs.
    """
    import threading

    from merlin.common.cpu_pool import AUXILIARY, GRADE, cpu_priority, preparation_cpus, selected_pool

    stop = threading.Event()
    start_cadence = os.environ.get("MERLIN_GRADE_START_TO_START", "0")
    if start_cadence not in {"0", "1"}:
        raise ValueError("MERLIN_GRADE_START_TO_START must be 0 or 1")
    if source_identity is None and start_cadence == "1":
        from .background_admission import source_digest

        def source_identity():
            return source_digest(ws)

    failures = GradeFailureAlert(run_dir, threshold=grade_failure_alert_threshold())

    def _grade_body() -> None:
        from .joint_test_scope import focused

        concurrent_snapshots = focused()
        first_snapshot_obsolete = False
        verdict_path = ws / "qa" / "verdict.json"
        try:
            previous = json.loads(verdict_path.read_text())
            readiness_pending = (previous.get("package_readiness") or {}).get("status") == "not_ready"
        except (OSError, ValueError, AttributeError):
            readiness_pending = False
        if not verdict_path.exists() or readiness_pending:
            landed, readiness_published, tries = False, False, 0
            while not stop.is_set():
                tries += 1
                try:
                    v = fast_grade_callback(ws, run_dir, FIRST_BACKGROUND_TICK, min(cadence.qa_timeout, 900))
                except Exception as e:  # noqa: BLE001 — no submission yet, or a mid-write tree: retry
                    # Said out loud the first time and then periodically: a first grade that never lands
                    # means the agent IS working blind, and that has to be visible in the log rather
                    # than inferred later from a missing qa/ directory.
                    if tries == 1 or tries % 10 == 0:
                        print(
                            f"[first-grade] not yet ({type(e).__name__}: {str(e)[:120]}); "
                            f"retrying every {FIRST_GRADE_POLL_S}s",
                            flush=True,
                        )
                    if stop.wait(FIRST_GRADE_POLL_S):
                        break
                    continue
                if (v.get("package_readiness") or {}).get("status") == "not_ready":
                    readiness_published = True
                    if tries == 1 or tries % 10 == 0:
                        print(
                            f"[first-grade] declared tool not ready; no capsules executed; "
                            f"retrying every {FIRST_GRADE_POLL_S}s",
                            flush=True,
                        )
                    if stop.wait(FIRST_GRADE_POLL_S):
                        break
                    continue
                landed = True
                first_snapshot_obsolete = v.get("snapshot_current") is False
                print(
                    f"[first-grade] loop-tier verdict published: {v.get('n_passed')}/"
                    f"{v.get('n_capsules')} at {v.get('tiers_graded')} "
                    f"(all_pass={v.get('all_pass')}; NOT run: {v.get('tiers_not_run')})",
                    flush=True,
                )
                break
            if not landed:
                print(
                    "[first-grade] only package-readiness feedback landed; no capsules executed"
                    if readiness_published
                    else "[first-grade] NO verdict landed during this turn — the agent ran without "
                    "feedback; treat the turn's result accordingly",
                    flush=True,
                )
                return
        if not interval_grades or int(cadence.grade_interval) <= 0:
            return
        t = 0
        previous_key = None
        interval = max(30, int(cadence.grade_interval))
        # A buildable first snapshot still finishes under its original custody.
        # Bounded joint smoke completes concurrently with edits; only legacy
        # broad grades wait briefly for stable source after supersession.
        delay, completed_source = (0 if first_snapshot_obsolete else interval), None
        debounce = first_snapshot_obsolete
        while not stop.wait(delay):
            delay = interval
            if debounce and source_identity is not None and not concurrent_snapshots:
                print("[in-turn grade] source superseded; waiting up to 120s for 30s quiet", flush=True)
                before = wait_for_stable_source(source_identity, stop)
                debounce = False
            else:
                before = source_identity() if source_identity is not None else None
            if stop.is_set():
                return  # Handoff may have arrived while source metadata was being verified.
            if source_identity is not None and (before is None or before == completed_source):
                delay = min(30, interval)
                continue
            t += 1
            scratch_key = grade_key(round_index, t)
            started = time.monotonic()
            try:
                v = grade_callback(
                    ws,
                    run_dir,
                    FIRST_BACKGROUND_TICK + t,
                    cadence.no_oracle,
                    cadence.qa_timeout,
                    label="inturn",
                    scratch_key=scratch_key,
                    previous_scratch_key=previous_key,
                )
            except Exception as e:  # noqa: BLE001 — a mid-write submission must never kill the run
                print(f"[in-turn grade {t}] skipped: {type(e).__name__}: {e}", flush=True)
                failures.failed(e)
                delay = min(30, interval)
                continue
            failures.succeeded()
            if start_cadence == "1":
                delay = max(0, interval - (time.monotonic() - started))
            if source_identity is not None:
                after = source_identity()
                curriculum = (v.get("joint_test_selection") or {}).get("curriculum") or {}
                if (before == after and v.get("snapshot_current") is True
                        and not v.get("package_failure") and not curriculum.get("pending")):
                    completed_source = before
                elif after is not None and after != before:
                    delay, debounce = 0, not concurrent_snapshots
            previous_key = scratch_key
            print(
                f"[in-turn grade {t}] {v.get('n_passed')}/{v.get('n_capsules')} all_pass={v.get('all_pass')}",
                flush=True,
            )
            if on_tick is not None:
                try:
                    with cpu_priority(AUXILIARY):  # Telemetry yields to every grade and author request.
                        on_tick(t)
                except Exception as e:  # noqa: BLE001 - a tick hook must never kill the grader
                    print(f"[in-turn grade {t}] on_tick skipped: {type(e).__name__}: {e}", flush=True)

    def _body() -> None:
        # A new thread inherits the manager's CPU pool. Establish the grading
        # base before callbacks create executors so nested build/sim leases
        # restore preparation CPUs, including between-capsule provenance work.
        # A new thread also starts without a lease class: grade at GRADE, below
        # author-requested work and above preparation and probes.
        with selected_pool(preparation_cpus()), cpu_priority(GRADE):
            _grade_body()

    th = library_thread(_body, name="in-turn-grader", daemon=True)
    th.start()
    return th, stop


def stop_background(handle) -> None:
    """Drain the in-turn owner before reusing its receipt or starting a post-turn grade.

    ``Thread.join(timeout=...)`` is not cancellation.  If the thread is already inside ``qa_grade``, a
    timed join merely returns with that full-suite grade still running and the caller immediately starts
    a second, authoritative full-suite grade.  Apart from racing the verdict files, two Atlas grades were
    observed contending across 67 threads for hours.  Set the interruptible interval event, then perform
    the single-flight hand-off: the authoritative grade may start only after the in-turn grade exits.
    """
    if not handle:
        return
    th, stop = handle
    stop.set()
    join_background(th)


def record_channel_health(ws: Path, run_dir: Path) -> None:
    """Check the self-check channel's health DURING the run and record it operator-side. Never raises.

    `channel_health` already computes exactly the right thing -- expired requests, stranded ones,
    orphan responses, broker restarts, whether the channel is healthy at all. It was only ever called
    ONCE, in the end-of-run summary, where it gates whether the official grade counts as complete. By
    then the run has spent its whole budget.

    Measured on merlincirct_atlas_feedback_v3_20260906, evaluated by hand while it was still running:
    `healthy: false`, 16 expired requests, 3 stranded, 6 broker starts -- and the last completed
    self-check 5.5 h earlier while the agent kept editing. Every one of those facts was computable
    hours before the run ended, by a function already written. A post-mortem that could have been a
    warning is the same defect as a check that cannot fail: the information existed and nobody was
    listening.

    Operator-side only, for the reason `_write_stage_ledger` gives: telling the agent its feedback
    channel is broken is feedback, and feedback defines an arm.

    While the brokers serve (no ``STOP`` in the channel) a request still inside its deadline is in
    flight, not stranded; once they are stopped nothing will answer it, so the end-of-run account
    applies.
    """
    try:
        health = channel_health(ws, live=not (Path(ws) / ".qa_channel" / "STOP").exists())
        (run_dir / "feedback_health.json").write_text(json.dumps(health, indent=2))
        if not health.get("healthy", True):
            print(
                f"[feedback] channel UNHEALTHY: {health.get('expired', 0)} expired, "
                f"{health.get('stranded', 0)} stranded, "
                f"{health.get('orphan_responses', 0)} orphan response(s), "
                f"{health.get('broker_starts', 0)} broker start(s) — the agent may be iterating "
                f"without feedback",
                flush=True,
            )
    except Exception:  # noqa: BLE001 -- diagnostics may never fail a grade
        return
