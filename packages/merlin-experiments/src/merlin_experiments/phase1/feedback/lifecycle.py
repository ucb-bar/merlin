"""Host-side feedback lifetimes, scheduling and channel-delivery accounting.

Brokers live inside an agent launch; the background grader surrounds that launch.
Keep the separate stop boundaries: stopping a thread is an unbounded single-flight
handoff, not cancellation or a timed join. Numerical grading is supplied explicitly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from merlin.common.paths import module_source_path
from merlin.targetgen import tool_registry as _TR

from ..context import InvocationContext

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
    broker_specs = [(spec.module, spec.log, spec.channel) for spec in _TR.COMMON_BROKERS]
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
                "merlin_experiments.phase1.brokers.selfcheck",
                "merlin_experiments.phase1.brokers.simjob",
            }:
                argv += ["--descriptor", str(config.context.descriptor), "--repo", str(config.context.repo)]
            if name == "merlin_experiments.phase1.brokers.simjob":
                argv += ["--timing-file", str(config.timing_file)]
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
            output = open(ch / log, "w")
            launch_error = None
            try:
                from merlin_experiments.frozen_python import inherited_python_command

                brokers.append(
                    subprocess.Popen(inherited_python_command(argv), stdout=output, stderr=subprocess.STDOUT)
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
    _stop_brokers(ws, brokers, primary_error=primary_error, channels=(".qa_channel", ".isa_channel", ".cca_channel"))


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


def channel_health(ws: Path, *, now: float | None = None) -> dict:
    """Durable end-of-run accounting for the synchronous self-check channel.

    Expired requests are abandoned clients, not model failures, but they still prove the promised
    synchronous feedback was not delivered. Any request without an atomic response+done pair makes the
    channel unhealthy; ``expired`` versus ``stranded`` distinguishes a past timeout from live work.
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
    from merlin_experiments.phase1.brokers.selfcheck import _request_deadline

    for rid, req in requests.items():
        if rid not in completed:
            try:
                if now > _request_deadline(req):
                    expired.add(rid)
            except (FileNotFoundError, OSError, ValueError):
                pass
    # An incomplete rid whose req_ file is gone cannot have a deadline read, so it can never be shown to
    # have expired; it is stranded, which is the reportable half either way.
    stranded = universe - completed - expired
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
        "orphan_responses": len(orphan_responses),
        "done_without_response": len(done_without_response),
        "replayed": int(broker_stats.get("replayed", 0)),
        "max_queue_depth": int(broker_stats.get("max_queue_depth", 0)),
        "broker_starts": int(broker_stats.get("broker_starts", 0)),
        "healthy": healthy,
    }


def grade_key(round_index: int, tick: int) -> str:
    """A run-wide scratch identity for one in-turn grade.

    The human-facing numeric tick deliberately remains in its historical band; only filesystem and
    archive identities need the agent-round coordinate that prevents a later round from reopening a
    frozen snapshot.
    """
    if round_index < 0 or tick < 1:
        raise ValueError("in-turn grade coordinates must be a non-negative round and positive tick")
    return f"r{round_index:04d}_t{tick:06d}"


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
    """
    import threading

    stop = threading.Event()

    def _body() -> None:
        if not (ws / "qa" / "verdict.json").exists():
            landed, tries = False, 0
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
                landed = True
                print(
                    f"[first-grade] loop-tier verdict published: {v.get('n_passed')}/"
                    f"{v.get('n_capsules')} at {v.get('tiers_graded')} "
                    f"(all_pass={v.get('all_pass')}; NOT run: {v.get('tiers_not_run')})",
                    flush=True,
                )
                break
            if not landed:
                print(
                    "[first-grade] NO verdict landed during this turn — the agent ran without "
                    "feedback; treat the turn's result accordingly",
                    flush=True,
                )
                return
        if not interval_grades or int(cadence.grade_interval) <= 0:
            return
        t = 0
        previous_key = None
        while not stop.wait(max(30, int(cadence.grade_interval))):
            t += 1
            scratch_key = grade_key(round_index, t)
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
                continue
            previous_key = scratch_key
            print(
                f"[in-turn grade {t}] {v.get('n_passed')}/{v.get('n_capsules')} all_pass={v.get('all_pass')}",
                flush=True,
            )
            if on_tick is not None:
                try:
                    on_tick(t)
                except Exception as e:  # noqa: BLE001 - a tick hook must never kill the grader
                    print(f"[in-turn grade {t}] on_tick skipped: {type(e).__name__}: {e}", flush=True)

    th = threading.Thread(target=_body, name="in-turn-grader", daemon=True)
    th.start()
    return th, stop


def stop_background(handle) -> None:
    """Stop the in-turn grader before the post-turn authoritative grade runs.

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
    th.join()


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
    """
    try:
        health = channel_health(ws)
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
