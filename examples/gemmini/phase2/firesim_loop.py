"""The loop-facing submission layer for the shared FPGA queue: the guards the queue does not have.

WHY THIS MODULE EXISTS.  The board is ground truth and it is CHEAP per candidate.  Measured over
four real queue jobs (N = 4, 13, 13, 31 windows), an ordinary least squares fit with R^2 ~ 1.00
gives ``wall = 159 s + 11.5 s/window``: the fixed term is the bitstream flash, compilation happens
outside the job, and the simulation itself is nearly free.  The largest batch actually run measured
31 windows in 515 s, every window digest-exact, with a 242 ppm order effect.  That is 17 s per
candidate against a 331 s mean (117-887 s) for the cycle-accurate software simulator, so the
crossover is at N >= 1: there is no batch size at which the simulator is the cheaper way to rank.

WHY IT NEEDS GUARDS.  The queue daemon's ``_pick_next_job`` does a priority tier and then a weighted
round-robin BY USER.  Fair share only engages when another user has something queued; against an idle
queue -- and the board is idle 90-97% of the time -- there is no per-user cap and no rate limit at
all, so a runaway automated loop takes 100% of the only FPGA on the host.  Nothing in the queue will
stop it.  This module is where the loop stops itself:

1. :class:`FpgaBudget` + :class:`BudgetLedger` -- self-imposed FPGA-seconds per day and per run,
   charged against a persistent ledger so restarting the loop cannot reset the day.
2. :data:`LOOP_PRIORITY` -- an automated submission goes in at priority 0, BELOW the queue's default
   of 5, so any human submitting normally jumps ahead without having to ask.  A non-zero priority is
   refused here rather than merely discouraged.
3. :func:`contending_jobs` -- the queue status is read before every submission and the loop yields
   while another user has work QUEUED.
4. :class:`FailureTripwire` -- jobs 707-721 were fifteen consecutive FAILED jobs of at most 40 s each.  A
   blind retry loop would have repeated them indefinitely.  After a declared number of consecutive
   failures this halts, and only an explicit written diagnosis clears it.

AND THE SHAPE THAT MAKES IT ECONOMICAL.  Because the cost is ``159 + 11.5N``, a generation of N
candidates measured one-per-job costs ``N * 170.5 s`` and the same generation measured as ONE job of
N windows costs ``159 + 11.5N`` -- at N = 31 that is 5.3 kiloseconds against 515 s.  So the entry
point here, :func:`measure_generation`, takes a whole generation and never a single candidate, and
   it links that generation through :func:`merlin.perf.firesim_batch.link_batch`, which adds the
order-effect control window that makes a batched number comparable to a solo one.

THE SIMULATOR IS A CORRECTNESS PREFLIGHT AND NOTHING ELSE.  Its error is arm-correlated: at most
1.7% mean for hand-written, library, and third-party published kernels, but -16.74% mean and -21.55%
worst on this repository's own generated-package arm, and it errs in the FLATTERING direction --
simulated numbers put that arm 1.48x behind a published baseline at 512^3 where the hardware says
1.82x.  Ranking candidates on it therefore optimizes a bias, not a program.  :class:`CorrectnessPreflight`
carries the simulator's verdict with its cycle counts STRUCTURALLY ABSENT, and :func:`rank_candidates`
refuses any measurement not sourced from hardware, so a caller cannot rank on the simulator by
accident.

HOW A SUBMISSION MUST BE INVOKED (for whoever runs this):

* Prefix a shell submission with ``env -u HOME -u USER -u LOGNAME``.  The queue client forwards
  those to a daemon running as another user, which then resolves a home it cannot read and the job
  dies at ~35 s behind a misleading SSH error.  Proven from the queue's own database: jobs 722-724
  DONE with HOME absent, 725 FAILED with it set, nothing else different.  In-process, that is
  :func:`loop_client_environment`, and :func:`assert_identity_dropped` refuses an environment that
  still carries them.
* Only ``firesim-queue runworkload-full`` may be used -- never ``infrasetup``, ``runworkload`` or
  ``kill`` directly, never a second queue, one job at a time.  That is enforced by
  :class:`~merlin.perf.execution_policy.FireSimQueuePreflight`, which every submission built here
  passes through.

This example module submits nothing by itself: :func:`measure_generation` takes the submitter as an argument
(defaulting to :func:`merlin.perf.firesim_checkpoint.submit`) so the whole guard layer is testable
with the queue mocked and no hardware touched.
"""

from __future__ import annotations

import getpass
import json
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.perf.execution_policy import FIRESIM_QUEUE_OPERATION

# --------------------------------------------------------------------- measured cost model
#: Fixed per-job wall: the bitstream flash.  Seconds.
FIXED_JOB_SECONDS = 159.0
#: Marginal wall per measured window.  Seconds.
SECONDS_PER_WINDOW = 11.5
#: Where the two numbers above come from, carried beside them so a reader can date them.
COST_MODEL_PROVENANCE = (
    "OLS over 4 queue jobs of N = 4, 13, 13, 31 windows, R^2 ~ 1.00; the largest observed batch "
    "measured 31 windows in 515 s, digest-exact, order effect 242 ppm"
)
#: Mean per-candidate wall of the cycle-accurate software simulator, for the comparison only.
#: Observed range 117-887 s.  It is NEVER a ranking input; see :func:`rank_candidates`.
SIMULATOR_SECONDS_PER_CANDIDATE_MEAN = 331.0
#: The measured, arm-correlated simulator error that makes ranking on it invalid.
SIMULATOR_RANKING_BIAS = (
    "the software simulator's error is arm-correlated: <=1.7% mean on hand-written, library and "
    "published third-party arms but -16.74% mean / -21.55% worst on this repository's own generated "
    "arm, and it errs in the flattering direction (1.48x behind a published baseline simulated, "
    "1.82x measured); ranking on it optimizes the bias"
)

# --------------------------------------------------------------------- queue policy
#: An automated loop submits BELOW the queue's default tier, so a human never waits on it.
LOOP_PRIORITY = 0
#: What the queue client uses when a human does not pass ``--priority``.
QUEUE_DEFAULT_PRIORITY = 5
#: Queue states that mean another user is waiting for the board.
CONTENDING_STATES = ("QUEUED",)
#: Identity variables a cross-user daemon must not inherit.
IDENTITY_VARIABLES = ("HOME", "USER", "LOGNAME")
#: The shell prefix a submission must carry when it is invoked from a command line.
SUBMISSION_COMMAND_PREFIX = ("env", "-u", "HOME", "-u", "USER", "-u", "LOGNAME")

#: A conservative standing ceiling: 14 400 FPGA-seconds is 16.7% of a day on a board measured at
#: 3.4% (24 h) / 3.5% (7 d) / 9.8% (30 d) utilization, and buys ~27 jobs of 31 windows, ~840
#: candidates.  A run gets a quarter of that.  Both are POLICY, stated so they can be argued with.
DEFAULT_DAILY_FPGA_SECONDS = 14_400.0
DEFAULT_RUN_FPGA_SECONDS = 3_600.0
_DAY_SECONDS = 86_400.0

#: What :meth:`LoopGovernor.authorize` may answer.
ACTION_SUBMIT = "submit"
ACTION_YIELD = "yield"
ACTION_HALT = "halt"

#: Where a cycle count may have come from.  Only one of these may be ranked.
SOURCE_HARDWARE = "firesim"
SOURCE_SIMULATOR = "software_simulator"
MEASUREMENT_SOURCES = (SOURCE_HARDWARE, SOURCE_SIMULATOR)

#: Ledger namespace under ``out/artifacts/perf-studies/``.  A LEDGER, not a cache: it is the only
#: record of what this loop has already spent, and losing it silently re-grants the day's budget.
BUDGET_LEDGER = "firesim_loop_budget"
LEDGER_SCHEMA = "merlin_firesim_loop_spend_v1"
TRIPWIRE_SCHEMA = "merlin_firesim_loop_tripwire_v1"


class LoopRefusal(RuntimeError):
    """The loop refused to spend a queue slot.  Carries the verdict that explains why."""

    def __init__(self, verdict: Verdict):
        super().__init__(verdict.reason)
        self.verdict = verdict


class BudgetExhausted(LoopRefusal):
    """This submission would exceed the loop's self-imposed FPGA-seconds budget."""


class PriorityRefused(LoopRefusal):
    """An automated submission asked for a priority other than 0."""


class QueueContended(LoopRefusal):
    """Another user has work queued; the loop yields the board rather than racing for it."""


class RepeatedFailureHalt(LoopRefusal):
    """Consecutive failures reached the tripwire limit.  Diagnose; do not retry."""


class QueueStatusUnreadable(LoopRefusal):
    """The queue status could not be read structurally, so contention is UNKNOWN."""


class SimulatorRankingRefused(RuntimeError):
    """Something tried to rank candidates on simulated cycles."""


class GenerationFailed(RuntimeError):
    """One generation's queue job did not produce evidence.  Says whether the loop may continue."""

    def __init__(self, message: str, *, consecutive: int, halted: bool, job_id: int | None = None):
        super().__init__(message)
        self.consecutive, self.halted, self.job_id = consecutive, halted, job_id


# --------------------------------------------------------------------- cost
def estimate_job_seconds(window_count: int) -> float:
    """What one queue job of ``window_count`` measured windows costs the board, in seconds."""
    if not isinstance(window_count, int) or isinstance(window_count, bool) or window_count < 1:
        raise ValueError("a queue job measures at least one window")
    return FIXED_JOB_SECONDS + SECONDS_PER_WINDOW * window_count


def generation_window_count(candidate_count: int) -> int:
    """Windows a generation of ``candidate_count`` candidates declares, including the order control.

    :func:`~merlin.perf.firesim_batch.link_batch` appends a repeat of window 0, and that repeat is a
    declared window that costs a window's wall time like any other.
    """
    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool) or candidate_count < 1:
        raise ValueError("a generation needs at least one candidate")
    return candidate_count + 1


def affordable_window_count(seconds: float) -> int:
    """How many windows fit in ``seconds`` of board time.  Zero when not even the flash fits."""
    if seconds < FIXED_JOB_SECONDS + SECONDS_PER_WINDOW:
        return 0
    return int((seconds - FIXED_JOB_SECONDS) // SECONDS_PER_WINDOW)


def serial_vs_batched_seconds(candidate_count: int) -> tuple[float, float]:
    """``(one job per candidate, one job for the generation)`` -- the reason this module batches."""
    serial = candidate_count * estimate_job_seconds(1)
    batched = estimate_job_seconds(generation_window_count(candidate_count))
    return serial, batched


# --------------------------------------------------------------------- reading the queue
@dataclass(frozen=True)
class QueueJob:
    """One row of ``firesim-queue status``."""

    job_id: int
    user: str
    priority: int
    state: str
    phase: str
    wall_s: float
    exit_code: int | None
    command: str


@dataclass(frozen=True)
class QueueStatus:
    daemon_alive: bool
    jobs: tuple[QueueJob, ...]

    def in_states(self, states: Sequence[str]) -> tuple[QueueJob, ...]:
        wanted = {state.upper() for state in states}
        return tuple(job for job in self.jobs if job.state.upper() in wanted)


def _int_or_none(token: str) -> int | None:
    return None if token == "-" else int(token)


def _tabular_job(tokens: Sequence[str], continuation: Sequence[str] = ()) -> QueueJob:
    """``id user prio state phase wall_s rc cmd...`` -- split on whitespace, never pattern-matched."""
    job_id, user, priority, state, phase, wall, code = tokens[:7]
    command = "\n".join([" ".join(tokens[7:]), *continuation]).strip()
    return QueueJob(
        job_id=int(job_id),
        user=user,
        priority=int(priority),
        state=state,
        phase=phase,
        wall_s=float(wall),
        exit_code=_int_or_none(code),
        command=command,
    )


def _record_job(tokens: Sequence[str]) -> QueueJob | None:
    """A ``[firesim-queue] key=value ...`` client record, the other shape the client has printed."""
    values: dict[str, str] = {}
    for token in tokens[1:]:
        key, separator, value = token.partition("=")
        if separator and key and value:
            values.setdefault(key, value)
    if "job_id" not in values:
        return None
    return QueueJob(
        job_id=int(values["job_id"]),
        user=values.get("user", ""),
        priority=int(values.get("priority", QUEUE_DEFAULT_PRIORITY)),
        state=values.get("state", ""),
        phase=values.get("phase", "-"),
        wall_s=float(values.get("wall_s", 0.0)),
        exit_code=_int_or_none(values.get("exit_code", "-")),
        command=values.get("kind", ""),
    )


def _is_tabular_row(tokens: Sequence[str]) -> bool:
    """Whether these tokens open a status ROW rather than continue the previous row's command.

    A job's command is printed raw, so a multi-line command wraps onto lines that carry no job at
    all.  A row is recognised by the SHAPE of its fixed columns -- integer id, integer priority, an
    all-uppercase state word, a numeric wall -- and never by matching the state against a list of
    the states known today, because a state spelled differently tomorrow must not silently become a
    command continuation and disappear from the contention check.
    """
    if len(tokens) < 7:
        return False
    try:
        int(tokens[0])
        int(tokens[2])
        float(tokens[5])
    except ValueError:
        return False
    return tokens[3].isalpha() and tokens[3].isupper()


def parse_queue_status(text: str) -> QueueStatus:
    """Read ``firesim-queue status`` structurally.  An unclassifiable line RAISES; none is skipped.

    A line this cannot classify might be another user's queued job, and a contention guard that
    silently drops what it does not recognise is a guard that reports an idle queue exactly when it
    matters.  Four things are known not to open a job row: the daemon banner and the client's other
    ``[firesim-queue]`` notices, the column header, blank lines, and a continuation of the preceding
    row's printed command.
    """
    daemon_alive = False
    rows: list[tuple[tuple[str, ...], list[str]]] = []
    jobs: list[QueueJob] = []
    for raw_line in text.splitlines():
        tokens = raw_line.split()
        if not tokens:
            continue
        if tokens[0] == "[firesim-queue]":
            if "daemon:" in tokens:
                daemon_alive = "ALIVE" in tokens
                continue
            try:
                record = _record_job(tokens)
            except ValueError as exc:
                raise QueueStatusUnreadable(
                    Verdict(ACTION_YIELD, "queue_status_unreadable", f"unreadable queue record {raw_line!r}")
                ) from exc
            if record is not None:
                jobs.append(record)
            continue
        if tokens[0] == "id" and len(tokens) > 1 and tokens[1] == "user":
            continue
        if _is_tabular_row(tokens):
            rows.append((tuple(tokens), []))
            continue
        if rows:  # a wrapped line of the previous row's command
            rows[-1][1].append(raw_line.strip())
            continue
        raise QueueStatusUnreadable(
            Verdict(ACTION_YIELD, "queue_status_unreadable", f"unreadable queue status line {raw_line!r}")
        )
    for tokens, continuation in rows:
        try:
            jobs.append(_tabular_job(tokens, continuation))
        except ValueError as exc:
            raise QueueStatusUnreadable(
                Verdict(ACTION_YIELD, "queue_status_unreadable", f"unreadable queue status row {tokens!r}")
            ) from exc
    return QueueStatus(daemon_alive=daemon_alive, jobs=tuple(jobs))


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def read_queue_status(queue_executable: str | Path, *, runner: Runner = subprocess.run) -> QueueStatus:
    """Ask the installed queue what it is doing.  ``status`` is read-only and submits nothing."""
    completed = runner([str(queue_executable), "status"], capture_output=True, text=True, check=False)
    if completed.returncode:
        raise QueueStatusUnreadable(
            Verdict(ACTION_YIELD, "queue_status_unreadable", "the queue could not report its status")
        )
    return parse_queue_status((completed.stdout or "") + (completed.stderr or ""))


def submitting_user() -> str:
    """Whose submissions this loop's are.  Used only to tell our own jobs from everyone else's."""
    return getpass.getuser()


def contending_jobs(status: QueueStatus, *, me: str, states: Sequence[str] = CONTENDING_STATES) -> tuple[QueueJob, ...]:
    """Another user's jobs in a state that means they are waiting for the board."""
    if not me:
        raise ValueError("the loop must know whose jobs are its own before it can yield to anyone")
    return tuple(job for job in status.in_states(states) if job.user != me)


# --------------------------------------------------------------------- budget
@dataclass(frozen=True)
class FpgaBudget:
    """FPGA-seconds this loop allows itself.  The queue enforces neither of these."""

    seconds_per_day: float = DEFAULT_DAILY_FPGA_SECONDS
    seconds_per_run: float = DEFAULT_RUN_FPGA_SECONDS

    def __post_init__(self) -> None:
        for role, value in (("seconds_per_day", self.seconds_per_day), ("seconds_per_run", self.seconds_per_run)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{role} must be a positive number of FPGA-seconds")
        if self.seconds_per_run > self.seconds_per_day:
            raise ValueError("a single run may not be allowed more board time than a whole day")


def budget_ledger_path() -> Path:
    """The shared spend ledger.  One file, so a second loop process cannot re-grant the same day."""
    from merlin.common.paths import artifacts_dir

    return Path(artifacts_dir()) / "perf-studies" / BUDGET_LEDGER / "spend.jsonl"


class BudgetLedger:
    """Append-only record of the board time this loop has spent, keyed by run and by wall clock.

    The estimate is RESERVED before the job is submitted and reconciled to the observed wall
    afterwards, because a loop that charged itself only on success would be unbounded exactly in the
    case that matters: a job that fails, still flashes the bitstream, and is retried.  A process that
    dies mid-job leaves its reservation standing, which over-counts -- the fail-closed direction.
    """

    def __init__(self, path: Path, *, run_id: str):
        if not str(run_id).strip():
            raise ValueError("a spend ledger must name the run it is charging")
        self.path, self.run_id = Path(path), str(run_id).strip()

    def rows(self) -> list[dict[str, Any]]:
        if self.path.is_symlink() or not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping) or row.get("schema") != LEDGER_SCHEMA:
                raise ValueError(f"{self.path} carries a row that is not a {LEDGER_SCHEMA} record")
            rows.append(dict(row))
        return rows

    def _append(self, row: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"schema": LEDGER_SCHEMA, **dict(row)}, sort_keys=True) + "\n")

    def spent_since(self, cutoff: float) -> float:
        """Board seconds charged at or after ``cutoff``, by ANY run.  Never negative."""
        return max(0.0, sum(float(row["seconds"]) for row in self.rows() if float(row["ts"]) >= cutoff))

    def spent_this_run(self) -> float:
        return max(0.0, sum(float(row["seconds"]) for row in self.rows() if row.get("run_id") == self.run_id))

    def reserve(self, seconds: float, *, windows: int, now: float, note: str = "") -> str:
        """Charge the estimate BEFORE the board is asked for anything.  Returns the reservation id."""
        if seconds <= 0:
            raise ValueError("a reservation must charge a positive number of seconds")
        taken = sum(1 for row in self.rows() if row.get("run_id") == self.run_id and row.get("kind") == "reserve")
        reservation = f"{self.run_id}-{taken + 1}"
        self._append(
            {
                "kind": "reserve",
                "reservation": reservation,
                "run_id": self.run_id,
                "ts": float(now),
                "seconds": float(seconds),
                "windows": int(windows),
                "note": note,
            }
        )
        return reservation

    def settle(self, reservation: str, observed_seconds: float, *, now: float, job_id: int | None = None) -> float:
        """Reconcile a reservation to the wall the job actually took.  Returns the adjustment."""
        rows = self.rows()
        reserved = [row for row in rows if row.get("reservation") == reservation and row.get("kind") == "reserve"]
        if len(reserved) != 1:
            raise ValueError(f"no single reservation {reservation!r} to settle in {self.path}")
        if any(row.get("reservation") == reservation and row.get("kind") == "settle" for row in rows):
            raise ValueError(f"reservation {reservation!r} is already settled; settling twice miscounts the day")
        delta = float(observed_seconds) - float(reserved[0]["seconds"])
        self._append(
            {
                "kind": "settle",
                "reservation": reservation,
                "run_id": self.run_id,
                "ts": float(now),
                "seconds": delta,
                "observed_seconds": float(observed_seconds),
                "job_id": job_id,
                "windows": int(reserved[0].get("windows", 0)),
            }
        )
        return delta


# --------------------------------------------------------------------- failure tripwire
@dataclass(frozen=True)
class TripwireState:
    consecutive: int
    limit: int
    halted: bool
    reasons: tuple[str, ...]
    diagnosis: str | None = None


class FailureTripwire:
    """Counts CONSECUTIVE failed jobs and halts the loop at a declared limit.

    Jobs 707-721 are fifteen consecutive FAILED jobs of at most 40 s each: a loop that retried on failure
    without looking would have repeated them until someone noticed.  The counter is persisted, so a
    loop cannot clear it by restarting, and clearing it requires a written diagnosis -- the point is
    diagnose-or-stop, and an untyped ``reset()`` is just a slower blind retry.
    """

    def __init__(self, path: Path, *, limit: int = 3):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("the consecutive-failure limit must be a positive integer")
        self.path, self.limit = Path(path), limit

    def state(self) -> TripwireState:
        if self.path.is_symlink() or not self.path.is_file():
            return TripwireState(0, self.limit, False, ())
        document = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or document.get("schema") != TRIPWIRE_SCHEMA:
            raise ValueError(f"{self.path} is not a {TRIPWIRE_SCHEMA} document")
        consecutive = int(document.get("consecutive", 0))
        return TripwireState(
            consecutive=consecutive,
            limit=self.limit,
            halted=consecutive >= self.limit,
            reasons=tuple(str(item) for item in document.get("reasons", ())),
            diagnosis=document.get("diagnosis"),
        )

    def _write(self, consecutive: int, reasons: Sequence[str], diagnosis: str | None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "schema": TRIPWIRE_SCHEMA,
                    "consecutive": int(consecutive),
                    "limit": self.limit,
                    "reasons": list(reasons),
                    "diagnosis": diagnosis,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def record_failure(self, reason: str) -> TripwireState:
        current = self.state()
        reasons = (*current.reasons, str(reason))[-self.limit :]
        self._write(current.consecutive + 1, reasons, current.diagnosis)
        return self.state()

    def record_success(self) -> TripwireState:
        self._write(0, (), None)
        return self.state()

    def clear(self, *, diagnosis: str) -> TripwireState:
        """Re-arm the loop after a HALT.  Refuses an empty diagnosis: that is a blind retry."""
        if not isinstance(diagnosis, str) or not diagnosis.strip():
            raise ValueError(
                "a halted loop is cleared by a written diagnosis of why the jobs failed, never by "
                "an unexplained reset; fifteen consecutive failures is what an unexplained reset buys"
            )
        self._write(0, (), diagnosis.strip())
        return self.state()


# --------------------------------------------------------------------- the governor
@dataclass(frozen=True)
class Verdict:
    """What the loop may do next, and the one reason it may say so."""

    action: str
    refusal: str
    reason: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in (ACTION_SUBMIT, ACTION_YIELD, ACTION_HALT):
            raise ValueError(f"unknown loop action {self.action!r}")
        if self.action != ACTION_SUBMIT and not self.reason.strip():
            raise ValueError("a refusal must state its reason")

    @property
    def allowed(self) -> bool:
        return self.action == ACTION_SUBMIT

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "refusal": self.refusal, "reason": self.reason, "detail": dict(self.detail)}


_REFUSAL_EXCEPTIONS: dict[str, type[LoopRefusal]] = {
    "budget_exhausted_day": BudgetExhausted,
    "budget_exhausted_run": BudgetExhausted,
    "priority_not_zero": PriorityRefused,
    "queue_contended": QueueContended,
    "queue_daemon_down": QueueContended,
    "queue_status_unreadable": QueueStatusUnreadable,
    "repeated_failures": RepeatedFailureHalt,
}


class LoopGovernor:
    """Holds the four guards and answers one question: may this generation be submitted now?

    Every guard is checked on every submission, in a fixed order, and the FIRST refusal is the
    verdict.  The order is deliberate: the tripwire outranks everything (a halted loop must not
    submit even with budget and an idle queue), then priority (a bug in the caller, not a condition
    to wait out), then the budget, then queue contention -- which is the only one that means "come
    back later" rather than "stop".
    """

    def __init__(
        self,
        *,
        budget: FpgaBudget,
        ledger: BudgetLedger,
        tripwire: FailureTripwire,
        user: str | None = None,
        clock: Callable[[], float] = time.time,
        contending_states: Sequence[str] = CONTENDING_STATES,
    ):
        self.budget, self.ledger, self.tripwire = budget, ledger, tripwire
        self.user = user or submitting_user()
        self.clock, self.contending_states = clock, tuple(contending_states)

    # -- guards ------------------------------------------------------------
    def authorize(self, *, window_count: int, priority: int, status: QueueStatus) -> Verdict:
        """The verdict for one prospective job of ``window_count`` windows."""
        estimate = estimate_job_seconds(window_count)
        now = self.clock()

        tripwire = self.tripwire.state()
        if tripwire.halted:
            return Verdict(
                ACTION_HALT,
                "repeated_failures",
                f"{tripwire.consecutive} consecutive failed jobs reached the limit of {tripwire.limit}; "
                "diagnose them and clear the tripwire with a written diagnosis -- do not retry",
                {"consecutive": tripwire.consecutive, "limit": tripwire.limit, "reasons": list(tripwire.reasons)},
            )

        if priority != LOOP_PRIORITY:
            return Verdict(
                ACTION_HALT,
                "priority_not_zero",
                f"an automated loop submits at priority {LOOP_PRIORITY}, below the human default of "
                f"{QUEUE_DEFAULT_PRIORITY}; priority {priority} was requested",
                {"requested_priority": priority},
            )

        day = self.ledger.spent_since(now - _DAY_SECONDS)
        if day + estimate > self.budget.seconds_per_day:
            return Verdict(
                ACTION_HALT,
                "budget_exhausted_day",
                f"this job would take {estimate:.1f} FPGA-s and {day:.1f} of the "
                f"{self.budget.seconds_per_day:.1f} s daily budget is already spent; the queue has no "
                "per-user cap, so this budget is the only thing that stops the loop",
                {"estimate_seconds": estimate, "spent_seconds": day, "budget_seconds": self.budget.seconds_per_day},
            )
        run = self.ledger.spent_this_run()
        if run + estimate > self.budget.seconds_per_run:
            return Verdict(
                ACTION_HALT,
                "budget_exhausted_run",
                f"this job would take {estimate:.1f} FPGA-s and this run has spent {run:.1f} of its "
                f"{self.budget.seconds_per_run:.1f} s allowance",
                {"estimate_seconds": estimate, "spent_seconds": run, "budget_seconds": self.budget.seconds_per_run},
            )

        if not status.daemon_alive:
            return Verdict(
                ACTION_YIELD,
                "queue_daemon_down",
                "the queue daemon is not reporting ALIVE; a submission now would sit unserved",
            )
        waiting = contending_jobs(status, me=self.user, states=self.contending_states)
        if waiting:
            owners = sorted({job.user for job in waiting})
            return Verdict(
                ACTION_YIELD,
                "queue_contended",
                f"{len(waiting)} job(s) from {owners} are waiting for the board; the loop yields "
                "rather than competing for a shared FPGA",
                {"jobs": [job.job_id for job in waiting], "users": owners},
            )

        return Verdict(ACTION_SUBMIT, "", "", {"estimate_seconds": estimate, "windows": window_count})

    def require(self, *, window_count: int, priority: int, status: QueueStatus) -> Verdict:
        """:meth:`authorize`, raising the matching :class:`LoopRefusal` unless the answer is submit."""
        verdict = self.authorize(window_count=window_count, priority=priority, status=status)
        if not verdict.allowed:
            raise _REFUSAL_EXCEPTIONS.get(verdict.refusal, LoopRefusal)(verdict)
        return verdict

    # -- accounting --------------------------------------------------------
    def reserve(self, window_count: int, *, note: str = "") -> str:
        return self.ledger.reserve(
            estimate_job_seconds(window_count), windows=window_count, now=self.clock(), note=note
        )

    def settle_success(self, reservation: str, observed_seconds: float, *, job_id: int | None = None) -> TripwireState:
        self.ledger.settle(reservation, observed_seconds, now=self.clock(), job_id=job_id)
        return self.tripwire.record_success()

    def settle_failure(
        self, reservation: str, observed_seconds: float, *, reason: str, job_id: int | None = None
    ) -> TripwireState:
        """A failed job still flashed the bitstream, so it is charged like any other."""
        self.ledger.settle(reservation, observed_seconds, now=self.clock(), job_id=job_id)
        return self.tripwire.record_failure(reason)


# --------------------------------------------------------------------- the simulator's place
@dataclass(frozen=True)
class CorrectnessPreflight:
    """The software simulator's verdict on one PROGRAM, with its cycle counts structurally absent.

    Running the candidate program on the cycle-accurate simulator first is worth its wall time for
    one reason: it proves the program emits its window frame, that every window's output digest
    matches the oracle, and that it terminates -- the three ways a submission wastes a queue slot
    without saying why.  It is NOT a performance filter.  This type therefore has no cycles field
    and no constructor that accepts one, so no caller can rank on it by reaching through the
    preflight result: see :data:`SIMULATOR_RANKING_BIAS` for what ranking on it would optimize.
    """

    program_sha256: str
    completed: bool
    labels: tuple[str, ...]
    failures: tuple[tuple[str, str], ...] = ()
    cycles_withheld: str = SIMULATOR_RANKING_BIAS

    def __post_init__(self) -> None:
        if not isinstance(self.program_sha256, str) or len(self.program_sha256) != 64:
            raise ValueError("a preflight must name the program it proved, by sha256 of its bytes")
        if not self.labels:
            raise ValueError("a preflight must name the windows it checked")

    @property
    def passed(self) -> bool:
        return self.completed and not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_sha256": self.program_sha256,
            "completed": self.completed,
            "labels": list(self.labels),
            "failures": [list(item) for item in self.failures],
            "passed": self.passed,
            "cycles_withheld": self.cycles_withheld,
        }

    @classmethod
    def from_simulator_run(
        cls,
        run: Mapping[str, Any],
        *,
        program_sha256: str,
        expected_digests: Mapping[str, int],
    ) -> CorrectnessPreflight:
        """Read a ``run_on_gsim``-shaped document, keeping the correctness half and dropping cycles.

        ``expected_digests`` is required: a preflight that checked only that windows printed is the
        shape a published-and-wrong result already walked through once, where a marker was
        byte-identical to a correct run's and the number under it was not.  A window whose record
        carries no digest is recorded as an UNKNOWN failure, never passed over.
        """
        if not expected_digests:
            raise ValueError("a preflight without expected digests proves nothing; supply the oracle digests")
        records = {str(row.get("label")): row for row in run.get("records") or ()}
        failures: list[tuple[str, str]] = []
        for label, expected in expected_digests.items():
            row = records.get(label)
            if row is None:
                failures.append((label, "no window record in the simulator run"))
                continue
            fields = row.get("fields") if isinstance(row.get("fields"), Mapping) else row
            observed = fields.get("digest")
            if observed is None:
                failures.append((label, "UNKNOWN(the window record carries no digest field)"))
                continue
            try:
                same = int(str(observed), 0) == int(expected)
            except ValueError:
                failures.append((label, f"UNKNOWN(digest {observed!r} is not an integer)"))
                continue
            if not same:
                failures.append((label, f"digest {observed} does not match the oracle {expected}"))
        return cls(
            program_sha256=program_sha256,
            completed=bool(run.get("completed")),
            labels=tuple(expected_digests),
            failures=tuple(failures),
        )


@dataclass(frozen=True)
class CandidateMeasurement:
    """One candidate's measured cycles, carrying WHERE the number came from."""

    label: str
    cycles: int
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("a measurement must name the candidate it measured")
        if isinstance(self.cycles, bool) or not isinstance(self.cycles, int) or self.cycles <= 0:
            raise ValueError(f"candidate {self.label!r} carries no positive cycle count")
        if self.source not in MEASUREMENT_SOURCES:
            raise ValueError(f"a measurement's source must be one of {MEASUREMENT_SOURCES}, got {self.source!r}")


def rank_candidates(measurements: Iterable[CandidateMeasurement]) -> tuple[CandidateMeasurement, ...]:
    """Order candidates cheapest-first -- on HARDWARE cycles only.

    A simulated cycle count is refused here rather than warned about, because the error that makes
    it unusable is arm-correlated and flattering: it is largest exactly on this repository's own
    generated arm, which is the arm a search loop is optimizing.  A loop ranking on it would measure
    its own bias getting smaller.
    """
    ranked = list(measurements)
    simulated = [item.label for item in ranked if item.source != SOURCE_HARDWARE]
    if simulated:
        raise SimulatorRankingRefused(
            f"candidates {sorted(simulated)} carry simulated cycles and may not be ranked: {SIMULATOR_RANKING_BIAS}"
        )
    return tuple(sorted(ranked, key=lambda item: (item.cycles, item.label)))


# --------------------------------------------------------------------- submission environment
def assert_identity_dropped(env: Mapping[str, str] | None) -> None:
    """Refuse an environment that still forwards the submitter's identity to a cross-user daemon."""
    if env is None:
        raise ValueError(
            "a loop submission must pass an explicit client environment; inheriting this process's "
            f"environment forwards {IDENTITY_VARIABLES} and the job dies at ~35 s behind an SSH error"
        )
    present = [name for name in IDENTITY_VARIABLES if name in env]
    if present:
        raise ValueError(
            f"the client environment still carries {present}; drop them "
            f"({' '.join(SUBMISSION_COMMAND_PREFIX)} at a shell) or the job dies at ~35 s behind a "
            "misleading SSH error"
        )


def loop_client_environment(
    *, path_prefix: Sequence[Path] = (), base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The submitter's environment minus its identity.  The in-process form of the ``env -u`` prefix."""
    from merlin.perf.firesim_checkpoint import client_environment

    env = client_environment(path_prefix=path_prefix, drop=IDENTITY_VARIABLES, base=base)
    assert_identity_dropped(env)
    return env


# --------------------------------------------------------------------- the generation entry point
@dataclass(frozen=True)
class GenerationRequest:
    """ONE generation of candidates, to be measured as ONE queue job of N windows.

    There is deliberately no single-candidate entry point.  At ``159 + 11.5N`` seconds a job, N
    candidates submitted one per job cost ``N * 170.5 s`` and the same N measured as one job's
    windows cost ``159 + 11.5N`` -- an order of magnitude at the batch sizes this loop runs.
    """

    generation_id: str
    members: tuple[Any, ...]
    elf: Path
    validation_policy: Path
    workload: str
    bootbinary: str
    hw_config: str
    host: Any
    preflight: CorrectnessPreflight
    project: str = "merlin-loop"
    hwdb_config_artifact: Path | None = None
    timeout_s: int = 1800
    order_effect_bound_ppm: int = 10_000

    def __post_init__(self) -> None:
        from merlin.perf.firesim_checkpoint import sha256_file

        if not str(self.generation_id).strip():
            raise ValueError("a generation must name itself; an unattributable batch is not a measurement")
        if not self.members:
            raise ValueError("a generation needs at least one candidate")
        if not isinstance(self.preflight, CorrectnessPreflight):
            raise ValueError(
                "a generation is preflighted on the software simulator before it spends a queue slot; "
                "the preflight is a correctness gate and carries no cycle counts"
            )
        if not self.preflight.passed:
            raise ValueError(
                f"the correctness preflight for {self.generation_id!r} did not pass "
                f"({list(self.preflight.failures)}); fix the program, do not spend a queue slot on it"
            )
        digest = sha256_file(Path(self.elf))
        if digest != self.preflight.program_sha256:
            raise ValueError(
                "the preflight proved a different program than the one being submitted "
                f"({self.preflight.program_sha256[:12]} vs {digest[:12]})"
            )
        labels = {getattr(member, "label", None) for member in self.members}
        missing = sorted(label for label in labels if label not in set(self.preflight.labels))
        if missing:
            raise ValueError(f"the preflight does not cover candidate windows {missing}")


@dataclass(frozen=True)
class GenerationOutcome:
    """What one generation's queue job cost and where its evidence went."""

    generation_id: str
    job_id: int
    windows: tuple[str, ...]
    wall_seconds: float
    estimated_seconds: float
    evidence_dir: Path
    batch: Mapping[str, Any]
    verdict: Verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "job_id": self.job_id,
            "windows": list(self.windows),
            "wall_seconds": self.wall_seconds,
            "estimated_seconds": self.estimated_seconds,
            "evidence_dir": str(self.evidence_dir),
            "batch": dict(self.batch),
            "verdict": self.verdict.to_dict(),
        }


class Settlement:
    """What one guarded submission turned out to cost, filled in by the caller as it learns it.

    It starts at the ESTIMATE, so a submission that dies without reporting a wall is still charged
    for the board time it took: the direction that over-counts is the safe one.
    """

    def __init__(self, estimate_seconds: float, verdict: Verdict):
        self.seconds = float(estimate_seconds)
        self.verdict = verdict
        self.job_id: int | None = None
        self.failure_reason = ""

    def observed(self, seconds: float, *, job_id: int | None = None) -> None:
        self.seconds = float(seconds)
        if job_id is not None:
            self.job_id = int(job_id)

    def failed(self, reason: str, *, job_id: int | None = None, seconds: float | None = None) -> None:
        self.failure_reason = str(reason).strip() or "the job failed without stating a reason"
        if seconds is not None:
            self.seconds = float(seconds)
        if job_id is not None:
            self.job_id = int(job_id)


@contextmanager
def guarded_submission(
    governor: LoopGovernor,
    *,
    queue_executable: str | Path,
    window_count: int,
    priority: int = LOOP_PRIORITY,
    runner: Runner = subprocess.run,
    note: str = "",
) -> Iterator[Settlement]:
    """Put ONE queue submission behind all four guards, and charge it whatever it ends up costing.

    This is the only place the guards are applied, so there is one copy of the order they are
    applied in: authorize (which polls the live queue), reserve the estimated board time, run the
    caller's submission, then reconcile the reservation and move the failure tripwire.  A caller
    that yields, refuses or halts never reaches its body.

    Anything that escapes the body is a failure -- a caller does not have to remember to report one
    -- and a caller that catches its own error says so with :meth:`Settlement.failed` rather than
    letting the spend go unrecorded.
    """
    status = read_queue_status(queue_executable, runner=runner)
    verdict = governor.require(window_count=window_count, priority=priority, status=status)
    settlement = Settlement(estimate_job_seconds(window_count), verdict)
    reservation = governor.reserve(window_count, note=note)
    try:
        yield settlement
    except BaseException as escaped:
        governor.settle_failure(reservation, settlement.seconds, reason=str(escaped), job_id=settlement.job_id)
        raise
    if settlement.failure_reason:
        governor.settle_failure(
            reservation, settlement.seconds, reason=settlement.failure_reason, job_id=settlement.job_id
        )
    else:
        governor.settle_success(reservation, settlement.seconds, job_id=settlement.job_id)


def default_governor(
    *, run_id: str, budget: FpgaBudget | None = None, root: Path | None = None, limit: int = 3
) -> LoopGovernor:
    """A governor on the repository's own spend ledger, for a caller that has no opinion on where.

    The ledger and the tripwire live together under ``out/artifacts/perf-studies/``, shared by every
    run, because a per-run budget file is a budget a restart can re-grant.
    """
    base = Path(root) if root is not None else budget_ledger_path().parent
    return LoopGovernor(
        budget=budget or FpgaBudget(),
        ledger=BudgetLedger(base / "spend.jsonl", run_id=run_id),
        tripwire=FailureTripwire(base / "tripwire.json", limit=limit),
    )


def measure_generation(
    request: GenerationRequest,
    governor: LoopGovernor,
    *,
    evidence_dir: Path,
    env: Mapping[str, str] | None = None,
    runner: Runner = subprocess.run,
    submitter: Callable[..., Any] | None = None,
) -> GenerationOutcome:
    """Measure a whole generation in ONE queue job, under every guard, or refuse to submit at all.

    The order of operations is the contract: link the batch (which appends the order-effect control
    and refuses a batch that cannot fit its wall limit), read the live queue, ask the governor,
    reserve the estimated board time, submit at priority 0 with the identity variables dropped, then
    reconcile the reservation against the observed wall and move the failure tripwire.

    ``submitter`` defaults to :func:`merlin.perf.firesim_checkpoint.submit`, which is the only thing
    here that touches hardware; passing a stand-in is how this whole layer is tested without a board.
    """
    from merlin.perf.firesim_batch import link_batch
    from merlin.perf.firesim_checkpoint import CheckpointError, QueueSubmission

    if submitter is None:
        from merlin.perf.firesim_checkpoint import submit

        submitter = submit

    batch = link_batch(
        list(request.members),
        batch_id=request.generation_id,
        queue_wall_limit_seconds=float(request.timeout_s),
        order_effect_bound_ppm=request.order_effect_bound_ppm,
    )
    window_count = len(batch.labels)

    submission = QueueSubmission(
        host=request.host,
        workload=request.workload,
        bootbinary=request.bootbinary,
        elf=Path(request.elf),
        hw_config=request.hw_config,
        hwdb_config_artifact=request.hwdb_config_artifact,
        timeout_s=request.timeout_s,
        priority=LOOP_PRIORITY,
        project=request.project,
    )
    argv = submission.argv()
    if argv[1] != FIRESIM_QUEUE_OPERATION:
        raise ValueError(f"a loop submission is exactly one {FIRESIM_QUEUE_OPERATION}, never a bare FireSim phase")
    if list(argv)[argv.index("--priority") + 1] != str(LOOP_PRIORITY):
        raise PriorityRefused(
            Verdict(ACTION_HALT, "priority_not_zero", f"the built submission does not carry --priority {LOOP_PRIORITY}")
        )
    client_env = dict(env) if env is not None else loop_client_environment()
    assert_identity_dropped(client_env)

    estimate = estimate_job_seconds(window_count)
    evidence = None
    refusal: Exception | None = None
    with guarded_submission(
        governor,
        queue_executable=request.host.queue_executable,
        window_count=window_count,
        runner=runner,
        note=request.generation_id,
    ) as settlement:
        verdict = settlement.verdict
        try:
            evidence = submitter(submission, Path(evidence_dir), env=client_env, runner=runner)
        except (CheckpointError, subprocess.SubprocessError) as failure:
            refusal = failure
            settlement.failed(str(failure), job_id=getattr(failure, "job_id", None))
        else:
            settlement.observed(float(evidence.wall_s), job_id=int(evidence.job_id))
    if refusal is not None or evidence is None:
        state = governor.tripwire.state()
        raise GenerationFailed(
            f"generation {request.generation_id!r} produced no evidence: {refusal}",
            consecutive=state.consecutive,
            halted=state.halted,
            job_id=getattr(refusal, "job_id", None),
        ) from refusal
    return GenerationOutcome(
        generation_id=request.generation_id,
        job_id=int(evidence.job_id),
        windows=tuple(batch.labels),
        wall_seconds=float(evidence.wall_s),
        estimated_seconds=estimate,
        evidence_dir=Path(evidence_dir),
        batch=batch.to_dict(),
        verdict=verdict,
    )
