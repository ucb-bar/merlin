"""Execution contracts for fast iteration and citable milestone measurements.

Search and certification use reduced witnesses and may not schedule a simulator execution longer
than ten minutes.  Full-size model execution is an optional, separate validation tier rather than a
Phase-2 prerequisite; when FireSim is available and explicitly used, it is admitted only through an
external queue.  This module validates requests and receipts; it deliberately does not shell out,
so no library caller can bypass the queue by importing a convenient runner.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ITERATION_MAX_SECONDS = 600.0
# Agent authoring and host-only whole-graph analysis do not execute a model.  Large global compiler
# changes may need more than the reduced-witness simulation ceiling, so keep their bound separate.
GLOBAL_AUTHORING_ROUND_MAX_SECONDS = 1200.0
# Full-graph compilation and static accounting never execute the model.  Keep their host deadline
# separate from the reduced-witness simulation limit.  This ceiling covers one ordered portfolio,
# not one member: four cold full-model lowers can exceed twenty minutes when the resource guard
# serializes workers, even though the normal two-worker wall time remains near ten minutes.
FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS = 2400.0
STATIC_PLANNER_MAX_SECONDS = 300.0


def _running_phase(trace: tuple[tuple[str, str], ...]) -> list[str | None]:
    """The phase banner in force at each trace entry, so commands can be scoped to it."""
    current: str | None = None
    scopes: list[str | None] = []
    for kind, value in trace:
        if kind == "phase":
            current = value
        scopes.append(current)
    return scopes


# The queue daemon's own phase/command interleaving, in the order it prints it, as observed
# byte-identically in jobs 535 and 610 (/scratch/firesim_queue/jobs/<id>/stdout.log; the phase
# and command lines are vendored verbatim in merlin/tests/data/firesim_queue/).  A "phase" entry
# is a `=== [firesim-queue] phase=<NAME> job_id=<id> ===` banner; a "command" entry is the FireSim
# manager's `Running: <subcommand>` line.  Note the daemon opens an INFRASETUP banner covering the
# whole kill+infrasetup group and then re-banners each half, so INFRASETUP appears twice and the
# leading kill is scoped to LEADING_KILL rather than to INFRASETUP.
#
# Every phase, lifecycle and command-scoping expectation in this module and in
# `merlin.perf.firesim_receipt` derives from this one sequence, so a daemon change cannot leave
# one copy stale while another still passes.
FIRESIM_QUEUE_TRACE = (
    ("phase", "STAGING"),
    ("phase", "INFRASETUP"),
    ("phase", "LEADING_KILL"),
    ("command", "kill"),
    ("phase", "INFRASETUP"),
    ("command", "infrasetup"),
    ("phase", "RUNNING"),
    ("command", "runworkload"),
    ("phase", "TEARDOWN"),
    ("command", "kill"),
)


def _collapse_adjacent(names: tuple[str, ...]) -> tuple[str, ...]:
    collapsed: list[str] = []
    for name in names:
        if not collapsed or collapsed[-1] != name:
            collapsed.append(name)
    return tuple(collapsed)


FIRESIM_LIFECYCLE = tuple(("firesim", value) for kind, value in FIRESIM_QUEUE_TRACE if kind == "command")
FIRESIM_QUEUE_OPERATION = "runworkload-full"
FIRESIM_QUEUE_PHASES = _collapse_adjacent(tuple(value for kind, value in FIRESIM_QUEUE_TRACE if kind == "phase"))
# Each lifecycle command paired with the phase whose banner was open when the daemon ran it.
FIRESIM_QUEUE_SCOPED_COMMANDS = tuple(
    (phase, value)
    for phase, (kind, value) in zip(_running_phase(FIRESIM_QUEUE_TRACE), FIRESIM_QUEUE_TRACE)
    if kind == "command"
)
_PROFILE_METRICS = frozenset(
    {
        "total_compute_cycles",
        "resource_busy_cycles",
        "movement_bytes",
        "movement_commands",
        "encoding_transitions",
        "movement_compute_overlap_cycles",
        "overlap_available_cycles",
        "idle_cycles",
        "critical_path_cycles",
    }
)


def require_probe_execution(descriptor: Mapping[str, Any]) -> None:
    """Exclude complete model captures from the search oracle, even if they would run quickly.

    The model is compiled and accounted globally. Only a separate mechanism-equivalent probe may
    be executed for calibration. Passing this exclusion check is not proof of probe equivalence;
    the experiment must additionally bind the probe's mechanism and validity domain.
    """
    operation = descriptor.get("operation")
    operation = operation if isinstance(operation, Mapping) else {}
    performance = descriptor.get("performance")
    performance = performance if isinstance(performance, Mapping) else {}
    semantic = descriptor.get("semantic")
    semantic = semantic if isinstance(semantic, Mapping) else {}
    if (
        descriptor.get("kind") == "model"
        or operation.get("op") == "model"
        or semantic.get("generalization_axis") == "model"
        or performance.get("global_objective") is True
        or performance.get("measurement_scope") in ("full_model", "full_layer")
    ):
        raise ValueError(
            "the full model/layer is a compile-only search objective; calibrate with a separate "
            "mechanism-equivalent probe, even when full execution would fit the time budget"
        )


#: The two ways a descriptor may reach a measurement, named so a caller states which it wants.
#: These are ROUTES, not permissions: :data:`ROUTE_BATCHED_MEASUREMENT` says a whole-model claim
#: must arrive as one window of a queue-owned batch, and nothing here makes it a search probe.
ROUTE_IN_LOOP_PROBE = "in_loop_probe"
ROUTE_BATCHED_MEASUREMENT = "batched_measurement"


def execution_route(descriptor: Mapping[str, Any]) -> str:
    """Which route this descriptor's execution may take -- ADDED BESIDE the probe exclusion, not
    instead of it.

    There is exactly ONE predicate for "this is a whole-model/global claim": the five conditions
    :func:`require_probe_execution` raises on.  This function asks that same function rather than
    restating them, so the route and the refusal cannot drift apart -- a second copy of a
    five-clause condition is how a relaxation gets introduced while the test that pins the refusal
    still passes.

    A descriptor the probe check REFUSES is not thereby admitted to anything: it is routed to the
    batched tier, which has its own refusals in :func:`require_batch_measurement`.
    """
    try:
        require_probe_execution(descriptor)
    except ValueError:
        return ROUTE_BATCHED_MEASUREMENT
    return ROUTE_IN_LOOP_PROBE


def require_batch_measurement(descriptor: Mapping[str, Any], *, batch_id: str, queue_owned: bool) -> None:
    """Admit ONLY a whole-model measurement that arrives as one window of a queue-owned batch.

    The mirror image of :func:`require_probe_execution`, and deliberately not its inverse: that
    function refuses the model everywhere in the search loop and still does, permanently.  This one
    refuses everything that is NOT whole-model-via-batch, so a per-op capsule cannot borrow the
    batched route to buy itself an FPGA slot, and a whole-model claim cannot be made outside the
    queue's ownership.

    ``batch_id`` names the round-group the window accumulates into.  It is required and nonempty
    because an unattributable window is the failure this whole mechanism exists to prevent: a cycle
    count that cannot be tied to the batch it was measured in is a number with no denominator.
    """
    route = execution_route(descriptor)
    if route != ROUTE_BATCHED_MEASUREMENT:
        raise ValueError(
            "the batched hardware tier measures whole-model/global objectives only; this "
            f"descriptor routes to {route!r} and must use the reduced-witness iteration tier"
        )
    if not queue_owned:
        raise ValueError(
            "a batched hardware measurement must be owned by the FireSim queue; a directly "
            "launched session cannot produce a sealed receipt"
        )
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError(
            "a batched hardware measurement must name the round-group it accumulates into; an "
            "unattributable window is not a measurement"
        )


@dataclass(frozen=True)
class SimulationBudget:
    timeout_seconds: float
    reference_timeout_seconds: float
    tier: str = "iteration"

    def __post_init__(self) -> None:
        if self.tier != "iteration":
            raise ValueError("SimulationBudget is only for the reduced-witness iteration tier")
        for name, value in (
            ("timeout_seconds", self.timeout_seconds),
            ("reference_timeout_seconds", self.reference_timeout_seconds),
        ):
            if value <= 0 or value > ITERATION_MAX_SECONDS:
                raise ValueError(
                    f"{name} must be in (0, {ITERATION_MAX_SECONDS:g}] seconds for iteration; "
                    "reduce the witness shape; optional full-size validation is a separate tier"
                )


@dataclass(frozen=True)
class SimulationAdmission:
    admitted: bool
    estimated_seconds: float | None
    reason: str = ""


def admit_reduced_witness(
    *,
    estimated_cycles: int,
    measured_cycles_per_second: float | None,
    startup_seconds: float = 0.0,
    budget: SimulationBudget,
) -> SimulationAdmission:
    """Admit only a witness whose evidence-backed runtime fits the bounded inner loop."""
    if estimated_cycles < 0 or startup_seconds < 0:
        raise ValueError("estimated cycles and startup seconds must be non-negative")
    if measured_cycles_per_second is None or measured_cycles_per_second <= 0:
        return SimulationAdmission(
            False, None, "simulator throughput is UNKNOWN; measure it before admitting a witness"
        )
    estimate = startup_seconds + estimated_cycles / measured_cycles_per_second
    if estimate > budget.timeout_seconds:
        return SimulationAdmission(
            False,
            estimate,
            f"estimated {estimate:.3f}s exceeds the {budget.timeout_seconds:g}s per-execution "
            "iteration budget; reduce the witness shape",
        )
    return SimulationAdmission(True, estimate)


@dataclass(frozen=True)
class WarmProfileContract:
    warmup_runs: int = 1
    measured_runs: int = 1
    captured_metrics: frozenset[str] = frozenset({"total_compute_cycles"})

    def __post_init__(self) -> None:
        if self.warmup_runs < 1:
            raise ValueError("a performance profile requires at least one unmeasured warm run")
        if self.measured_runs != 1:
            raise ValueError("the iteration receipt must contain exactly one measured run after warmup")
        if "total_compute_cycles" not in self.captured_metrics:
            raise ValueError("total_compute_cycles is the required primary metric")
        extra = sorted(self.captured_metrics - _PROFILE_METRICS)
        if extra:
            raise ValueError(
                f"non-minimal profile metrics {extra}; capture compute cycles and only the "
                "resource/movement counters needed to explain them"
            )


@dataclass(frozen=True)
class WarmComputeReceipt:
    workload: str
    total_compute_cycles: int
    contract: WarmProfileContract
    provenance: str
    resource_busy_cycles: tuple[tuple[str, int], ...] = ()
    movement_bytes: int | None = None
    movement_commands: int | None = None
    encoding_transitions: int | None = None
    movement_compute_overlap_cycles: int | None = None
    overlap_available_cycles: int | None = None
    idle_cycles: int | None = None
    critical_path_cycles: int | None = None

    def __post_init__(self) -> None:
        if not self.workload.strip() or not self.provenance.strip():
            raise ValueError("a warm compute receipt must name its workload and provenance")
        if self.total_compute_cycles < 0:
            raise ValueError("total compute cycles cannot be negative")
        if tuple(sorted(self.resource_busy_cycles)) != self.resource_busy_cycles:
            raise ValueError("resource busy counters must be sorted for stable receipts")
        if any(value < 0 for _, value in self.resource_busy_cycles):
            raise ValueError("resource busy counters cannot be negative")
        optional = (
            self.movement_bytes,
            self.movement_commands,
            self.encoding_transitions,
            self.movement_compute_overlap_cycles,
            self.overlap_available_cycles,
            self.idle_cycles,
            self.critical_path_cycles,
        )
        if any(value is not None and value < 0 for value in optional):
            raise ValueError("movement, encoding, and occupancy counters cannot be negative")
        bounded = (self.movement_compute_overlap_cycles, self.idle_cycles, self.critical_path_cycles)
        if any(value is not None and value > self.total_compute_cycles for value in bounded):
            raise ValueError("overlap, idle, and critical-path cycles cannot exceed the measured run")
        if (
            self.movement_compute_overlap_cycles is not None
            and self.overlap_available_cycles is not None
            and self.movement_compute_overlap_cycles > self.overlap_available_cycles
        ):
            raise ValueError("realized overlap cannot exceed overlap available cycles")

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "profile": {
                "warmup_runs": self.contract.warmup_runs,
                "measured_runs": self.contract.measured_runs,
                "captured_metrics": sorted(self.contract.captured_metrics),
            },
            "total_compute_cycles": self.total_compute_cycles,
            "resource_busy_cycles": dict(self.resource_busy_cycles),
            "movement_bytes": self.movement_bytes,
            "movement_commands": self.movement_commands,
            "encoding_transitions": self.encoding_transitions,
            "movement_compute_overlap_cycles": self.movement_compute_overlap_cycles,
            "overlap_available_cycles": self.overlap_available_cycles,
            "idle_cycles": self.idle_cycles,
            "critical_path_cycles": self.critical_path_cycles,
            "provenance": self.provenance,
        }


def occupancy_from_warm_receipt(receipt: WarmComputeReceipt, resource_kinds: Mapping[str, object]):
    """Build planner occupancy from the minimal warm receipt, preserving every unknown.

    Resource roles come from the target adapter, never from counter spellings.  Aggregate busy
    counts alone do not establish overlap, idle time, or a critical path, so absent optional fields
    remain ``None`` and are named in ``missing``; downstream whole-model gates then refuse instead of
    treating an unmeasured quantity as zero.
    """
    from merlin.perf.decompose import ResourceKind
    from merlin.perf.global_planner import OccupancySummary

    busy = dict(receipt.resource_busy_cycles)
    unknown = sorted(set(busy) - set(resource_kinds))
    if unknown:
        raise ValueError(f"resource kinds are absent for warm counters {unknown}")

    def kind(value: object) -> ResourceKind:
        try:
            return value if isinstance(value, ResourceKind) else ResourceKind(str(value))
        except ValueError as exc:
            raise ValueError(f"invalid resource kind {value!r}") from exc

    # Keep all adapter-declared resources in scope. Dropping an unobserved
    # engine changes the utilization denominator and can make a partial trace
    # look fully occupied. A known inactive engine needs an explicit zero.
    resolved = {name: kind(value) for name, value in resource_kinds.items()}
    compute = tuple(sorted(name for name, value in resolved.items() if value is ResourceKind.COMPUTE))
    movement = tuple(sorted(name for name, value in resolved.items() if value is ResourceKind.MOVEMENT))
    missing: list[str] = []
    for name, value in (
        ("at least one compute resource", compute or None),
        ("movement/compute overlap cycles", receipt.movement_compute_overlap_cycles),
        ("overlap available cycles", receipt.overlap_available_cycles),
        ("idle cycles", receipt.idle_cycles),
        ("critical path cycles", receipt.critical_path_cycles),
        ("physical movement bytes", receipt.movement_bytes),
        ("movement command count", receipt.movement_commands),
        ("executed encoding transition count", receipt.encoding_transitions),
    ):
        if value is None:
            missing.append(name)
    return OccupancySummary(
        total_cycles=float(receipt.total_compute_cycles),
        busy_cycles=tuple((name, float(value)) for name, value in receipt.resource_busy_cycles),
        compute_resources=compute,
        movement_resources=movement,
        overlap_cycles=(
            float(receipt.movement_compute_overlap_cycles)
            if receipt.movement_compute_overlap_cycles is not None
            else None
        ),
        overlap_available_cycles=(
            float(receipt.overlap_available_cycles) if receipt.overlap_available_cycles is not None else None
        ),
        idle_cycles=float(receipt.idle_cycles) if receipt.idle_cycles is not None else None,
        critical_path_cycles=(
            float(receipt.critical_path_cycles) if receipt.critical_path_cycles is not None else None
        ),
        movement_bytes=float(receipt.movement_bytes) if receipt.movement_bytes is not None else None,
        movement_commands=receipt.movement_commands,
        encoding_transitions=receipt.encoding_transitions,
        provenance=(receipt.provenance,),
        missing=tuple(missing),
    )


@dataclass(frozen=True)
class FireSimQueuePreflight:
    """A single, directly-executed queue submission pinned by its raw executable path.

    ``expected_queue_executable`` is an explicit host policy input.  Keeping it outside this
    target-neutral module lets another installation pin its own queue while still making path drift
    part of the receipt.
    """

    expected_queue_executable: str
    submission: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = Path(self.expected_queue_executable)
        if not expected.is_absolute() or not self.expected_queue_executable.strip():
            raise ValueError("the FireSim queue executable policy must be an absolute raw path")
        if len(self.submission) < 2:
            raise ValueError("FireSim requires one complete firesim-queue submission")
        if self.submission[:2] != (self.expected_queue_executable, FIRESIM_QUEUE_OPERATION):
            raise ValueError(
                "direct FireSim invocation is forbidden; submit exactly one "
                f"{self.expected_queue_executable} {FIRESIM_QUEUE_OPERATION} command"
            )
        if any(not isinstance(token, str) or not token for token in self.submission):
            raise ValueError("the FireSim queue submission must contain nonempty string arguments")
        shell_controls = frozenset({";", "&&", "||", "|", "&"})
        if any(token in shell_controls or "\n" in token or "\0" in token for token in self.submission):
            raise ValueError("the FireSim queue submission cannot contain shell control syntax")
        if any(Path(token).name == "firesim" for token in self.submission):
            raise ValueError("direct firesim commands cannot be nested in the queue submission")


@dataclass(frozen=True)
class QueueLogEvidence:
    """Content-bound plain-file evidence captured for one queue job."""

    role: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("queue log evidence must have a role")
        self.verified_bytes()

    def verified_bytes(self) -> bytes:
        """Read the still-plain artifact and refuse content drift."""
        artifact = Path(self.path)
        if not artifact.is_absolute() or artifact.is_symlink() or not artifact.is_file():
            raise ValueError("queue log evidence must name an absolute plain file")
        payload = artifact.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if self.sha256 != observed:
            raise ValueError(f"queue log evidence hash mismatch for {self.role}")
        return payload


@dataclass(frozen=True)
class QueuedFireSimReceipt:
    queue_job_id: int
    queue_owned: bool
    preflight: FireSimQueuePreflight
    queue_phases: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]
    logs: tuple[QueueLogEvidence, ...]
    warm_profile: WarmComputeReceipt
    #: The windows measured AFTER ``warm_profile`` inside the SAME ``runworkload-full``.
    #:
    #: One queue job runs exactly one ``runworkload``, so N candidates cannot be N held sessions --
    #: they are N measured windows of ONE linked bootbinary.  ``warm_profile`` stays window 0 so a
    #: receipt with no additional windows is byte-identical to the single-window receipts already
    #: sealed; the declared window list is ``measured_windows``.
    additional_windows: tuple[WarmComputeReceipt, ...] = ()

    @property
    def measured_windows(self) -> tuple[WarmComputeReceipt, ...]:
        """Every declared window, in declared order, window 0 first."""
        return (self.warm_profile,) + tuple(self.additional_windows)

    def __post_init__(self) -> None:
        if not self.queue_owned or self.queue_job_id < 1:
            raise ValueError("FireSim milestone execution must be owned by the FireSim queue")
        if self.queue_phases != FIRESIM_QUEUE_PHASES:
            raise ValueError(f"FireSim queue phases must be exactly {FIRESIM_QUEUE_PHASES}")
        prefixes = tuple(tuple(command[:2]) for command in self.commands)
        if prefixes != FIRESIM_LIFECYCLE:
            rendered = tuple(" ".join(item) for item in prefixes)
            expected = tuple(" ".join(item) for item in FIRESIM_LIFECYCLE)
            raise ValueError(f"FireSim lifecycle must be exactly {expected}, observed {rendered}")
        if any(len(command) < 2 for command in self.commands):
            raise ValueError("every FireSim lifecycle command must name its subcommand")
        roles = tuple(evidence.role for evidence in self.logs)
        required_roles = ("queue_client", "queue_daemon", "uart")
        if roles != required_roles:
            raise ValueError(f"FireSim receipt logs must be exactly {required_roles}, observed {roles}")
        if str(self.queue_job_id) not in Path(self.logs[1].path).parts:
            raise ValueError("queue daemon log is not bound to the recorded queue job id")
        client_text = self.logs[0].verified_bytes().decode("utf-8", errors="replace")
        if f"job_id={self.queue_job_id}" not in client_text or "terminal state=DONE" not in client_text:
            raise ValueError("queue client log does not prove this job id completed DONE")
        daemon_text = self.logs[1].verified_bytes().decode("utf-8", errors="replace")
        daemon_markers = tuple(
            f"=== [firesim-queue] phase={value} job_id={self.queue_job_id}" if kind == "phase" else f"Running: {value}"
            for kind, value in FIRESIM_QUEUE_TRACE
        )
        cursor = 0
        for marker in daemon_markers:
            position = daemon_text.find(marker, cursor)
            if position < 0:
                raise ValueError(f"queue daemon log does not prove ordered lifecycle marker {marker!r}")
            cursor = position + len(marker)
        windows = self.measured_windows
        labels = tuple(window.workload for window in windows)
        if len(set(labels)) != len(labels):
            raise ValueError("every declared FireSim window must name a distinct workload label")
        for window in windows:
            if window.contract.captured_metrics != frozenset({"total_compute_cycles"}):
                raise ValueError("final FireSim receipt may capture only measured compute cycles")
        uart_text = self.logs[2].verified_bytes().decode("utf-8", errors="replace")
        metric_lines = [line.split() for line in uart_text.splitlines() if line.startswith("METRIC ")]
        # EXACTLY ONE `METRIC cycles` PER DECLARED WINDOW, IN DECLARED ORDER, AND NO EXTRAS. A stray
        # metric line means the linked ELF ran a window nobody declared, so the batch measured
        # something the plan cannot name; that invalidates every window in it, not just the extra
        # one, because the unnamed work shares the run's cache, DRAM and FPGA state with the rest.
        if len(metric_lines) != len(windows):
            raise ValueError(
                f"final FireSim UART must contain exactly one metric line per declared window "
                f"({len(windows)}), observed {len(metric_lines)}"
            )
        for window, metric in zip(windows, metric_lines, strict=True):
            if len(metric) != 3 or metric[:2] != ["METRIC", "cycles"]:
                raise ValueError("final FireSim UART may capture only measured compute cycles")
            try:
                observed_cycles = int(metric[2])
            except ValueError as exc:
                raise ValueError("final FireSim UART cycle metric must be an integer") from exc
            if observed_cycles != window.total_compute_cycles:
                raise ValueError("FireSim UART cycles do not match the warm profile receipt")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete queue, lifecycle, log, and measured-window evidence.

        ``measured_windows`` appears only for a batch (two or more windows), so a single-window
        receipt serialises byte-identically to the ones already sealed."""
        for evidence in self.logs:
            evidence.verified_bytes()
        extra: dict[str, Any] = {}
        if self.additional_windows:
            extra["measured_windows"] = [window.to_dict() for window in self.measured_windows]
        return {
            "queue_job_id": self.queue_job_id,
            "queue_owned": self.queue_owned,
            "queue_executable": self.preflight.expected_queue_executable,
            "queue_submission": list(self.preflight.submission),
            "queue_operation": FIRESIM_QUEUE_OPERATION,
            "queue_phases": list(self.queue_phases),
            "firesim_lifecycle": [list(command) for command in self.commands],
            "logs": {evidence.role: {"path": evidence.path, "sha256": evidence.sha256} for evidence in self.logs},
            "warm_profile": self.warm_profile.to_dict(),
            **extra,
        }
