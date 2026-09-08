"""Search stays bounded; citable full models stay queue-owned and warm."""
import hashlib

import pytest

from merlin.perf.execution_policy import (
    FIRESIM_QUEUE_PHASES,
    FireSimQueuePreflight,
    QueueLogEvidence,
    QueuedFireSimReceipt,
    SimulationBudget,
    WarmComputeReceipt,
    WarmProfileContract,
    admit_reduced_witness,
    occupancy_from_warm_receipt,
    require_probe_execution,
)


_LOCAL_QUEUE = "/scratch2/agustin/firesim_queue/bin/firesim-queue"


def _warm() -> WarmComputeReceipt:
    return WarmComputeReceipt(
        "whole_model", 1234,
        WarmProfileContract(captured_metrics=frozenset({
            "total_compute_cycles", "resource_busy_cycles", "movement_bytes"})),
        "same-process post-warm counter",
        resource_busy_cycles=(("array", 1000), ("dma", 300)),
        movement_bytes=4096,
    )


def _cycle_only_warm() -> WarmComputeReceipt:
    return WarmComputeReceipt(
        "whole_model", 1234, WarmProfileContract(), "same-process post-warm counter")


def _daemon_lifecycle_log() -> str:
    return "\n".join((
        "=== [firesim-queue] phase=STAGING job_id=41 ===",
        "=== [firesim-queue] phase=INFRASETUP job_id=41 ===",
        "Running: kill",
        "Running: infrasetup",
        "=== [firesim-queue] phase=RUNNING job_id=41 ===",
        "Running: runworkload",
        "=== [firesim-queue] phase=TEARDOWN job_id=41 ===",
        "Running: kill",
    ))


def _log(tmp_path, role: str, relative: str, content: str | None = None) -> QueueLogEvidence:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content is not None else f"{role}\n", encoding="utf-8")
    return QueueLogEvidence(role, str(path), hashlib.sha256(path.read_bytes()).hexdigest())


def _queued_receipt(tmp_path, **changes) -> QueuedFireSimReceipt:
    preflight = FireSimQueuePreflight(
        _LOCAL_QUEUE,
        (_LOCAL_QUEUE, "runworkload-full", "--stage-from", "/artifacts/model.elf"),
    )
    fields = {
        "queue_job_id": 41,
        "queue_owned": True,
        "preflight": preflight,
        "queue_phases": FIRESIM_QUEUE_PHASES,
        "commands": (
            ("firesim", "kill"),
            ("firesim", "infrasetup"),
            ("firesim", "runworkload", "--workload", "model.json"),
            ("firesim", "kill"),
        ),
        "logs": (
            _log(
                tmp_path, "queue_client", "receipts/client.log",
                "job_id=41\nterminal state=DONE\n",
            ),
            _log(
                tmp_path, "queue_daemon", "queue/jobs/41/stdout.log",
                _daemon_lifecycle_log(),
            ),
            _log(
                tmp_path, "uart", "receipts/uartlog",
                "METRIC cycles 1234\nDONE\n",
            ),
        ),
        "warm_profile": _cycle_only_warm(),
    }
    fields.update(changes)
    return QueuedFireSimReceipt(**fields)


def test_rejects_the_old_seven_hour_reference_timeout() -> None:
    with pytest.raises(ValueError, match="600"):
        SimulationBudget(timeout_seconds=600, reference_timeout_seconds=25200)


def test_static_analysis_ceiling_does_not_relax_reduced_witness_simulation() -> None:
    from merlin.perf.execution_policy import FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS

    assert FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS == 2400
    with pytest.raises(ValueError, match="600"):
        SimulationBudget(timeout_seconds=FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS,
                         reference_timeout_seconds=600)


def test_reduced_witness_is_admitted_from_measured_simulator_throughput() -> None:
    admission = admit_reduced_witness(
        estimated_cycles=50_000, measured_cycles_per_second=193,
        startup_seconds=4, budget=SimulationBudget(600, 600))

    assert admission.admitted
    assert admission.estimated_seconds == pytest.approx(263.0673575)


def test_full_layer_is_refused_from_inner_loop_when_it_would_take_hours() -> None:
    admission = admit_reduced_witness(
        estimated_cycles=2_900_000, measured_cycles_per_second=193,
        budget=SimulationBudget(600, 600))

    assert not admission.admitted
    assert admission.estimated_seconds > 4 * 60 * 60
    assert "reduce the witness shape" in admission.reason


@pytest.mark.parametrize("descriptor", [
    {"kind": "model"}, {"operation": {"op": "model"}},
    {"performance": {"global_objective": True}},
    {"performance": {"measurement_scope": "full_layer"}},
    {"semantic": {"generalization_axis": "model"}},
])
def test_model_search_objective_is_never_admitted_as_a_fast_probe(descriptor) -> None:
    with pytest.raises(ValueError, match="compile-only search objective"):
        require_probe_execution(descriptor)


def test_small_mechanism_probe_can_pass_model_execution_exclusion() -> None:
    require_probe_execution({"kind": "layer", "operation": {"op": "matmul"},
                             "performance": {"measurement_scope": "mechanism_probe"}})


def test_unknown_throughput_never_reads_as_fast_enough() -> None:
    admission = admit_reduced_witness(
        estimated_cycles=1, measured_cycles_per_second=None,
        budget=SimulationBudget(600, 600))

    assert not admission.admitted
    assert admission.estimated_seconds is None
    assert "UNKNOWN" in admission.reason


def test_profile_is_warm_and_rejects_counters_outside_the_minimal_contract() -> None:
    with pytest.raises(ValueError, match="warm run"):
        WarmProfileContract(warmup_runs=0)
    with pytest.raises(ValueError, match="non-minimal"):
        WarmProfileContract(captured_metrics=frozenset({
            "total_compute_cycles", "wall_time", "every_pc_sample"}))


def test_firesim_receipt_requires_queue_and_exact_lifecycle_order(tmp_path) -> None:
    receipt = _queued_receipt(tmp_path)
    assert receipt.queue_owned
    assert receipt.preflight.submission[:2] == (_LOCAL_QUEUE, "runworkload-full")

    with pytest.raises(ValueError, match="exactly"):
        _queued_receipt(
            tmp_path,
            commands=(("firesim", "infrasetup"), ("firesim", "kill"),
                      ("firesim", "runworkload"), ("firesim", "kill")),
        )


@pytest.mark.parametrize("submission", [
    ("firesim", "runworkload"),
    ("/another/queue/bin/firesim-queue", "runworkload-full"),
    (_LOCAL_QUEUE, "status"),
    (_LOCAL_QUEUE, "runworkload-full", ";", "firesim", "kill"),
])
def test_firesim_preflight_rejects_direct_or_unpinned_execution(submission) -> None:
    with pytest.raises(ValueError, match="direct FireSim|shell control"):
        FireSimQueuePreflight(_LOCAL_QUEUE, submission)


def test_firesim_receipt_binds_queue_job_and_all_logs(tmp_path) -> None:
    receipt = _queued_receipt(tmp_path)

    assert receipt.queue_job_id == 41
    assert tuple(log.role for log in receipt.logs) == ("queue_client", "queue_daemon", "uart")
    document = receipt.to_dict()
    assert document["queue_submission"][:2] == [_LOCAL_QUEUE, "runworkload-full"]
    assert document["queue_job_id"] == 41
    assert tuple(document["logs"]) == ("queue_client", "queue_daemon", "uart")
    assert document["warm_profile"]["profile"]["captured_metrics"] == [
        "total_compute_cycles"]

    wrong_job_log = _log(
        tmp_path, "queue_daemon", "queue/jobs/99/stdout.log",
        _daemon_lifecycle_log(),
    )
    with pytest.raises(ValueError, match="recorded queue job id"):
        _queued_receipt(
            tmp_path,
            logs=(receipt.logs[0], wrong_job_log, receipt.logs[2]),
        )


def test_firesim_log_evidence_refuses_tampering(tmp_path) -> None:
    evidence = _log(tmp_path, "queue_client", "client.log")
    path = tmp_path / "client.log"
    path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        QueueLogEvidence(evidence.role, evidence.path, evidence.sha256)


def test_firesim_receipt_serialization_refuses_log_drift(tmp_path) -> None:
    receipt = _queued_receipt(tmp_path)
    (tmp_path / "receipts/client.log").write_text(
        "job_id=41\nterminal state=DONE\nchanged\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        receipt.to_dict()


def test_firesim_receipt_refuses_unproven_daemon_phase_order(tmp_path) -> None:
    receipt = _queued_receipt(tmp_path)
    reversed_log = _log(
        tmp_path, "queue_daemon", "queue/jobs/41/reversed.log",
        "\n".join(
            f"=== [firesim-queue] phase={phase}" for phase in reversed(FIRESIM_QUEUE_PHASES)),
    )

    with pytest.raises(ValueError, match="ordered lifecycle marker"):
        _queued_receipt(tmp_path, logs=(receipt.logs[0], reversed_log, receipt.logs[2]))


def test_firesim_receipt_refuses_unproven_command_order_inside_phases(tmp_path) -> None:
    receipt = _queued_receipt(tmp_path)
    wrong_commands = _log(
        tmp_path, "queue_daemon", "queue/jobs/41/wrong-commands.log",
        _daemon_lifecycle_log().replace(
            "Running: kill\nRunning: infrasetup",
            "Running: infrasetup\nRunning: kill",
            1,
        ),
    )

    with pytest.raises(ValueError, match="Running: infrasetup"):
        _queued_receipt(tmp_path, logs=(receipt.logs[0], wrong_commands, receipt.logs[2]))


def test_firesim_receipt_refuses_client_job_id_mismatch(tmp_path) -> None:
    receipt = _queued_receipt(tmp_path)
    wrong_client = _log(
        tmp_path, "queue_client", "receipts/wrong-client.log",
        "job_id=99\nterminal state=DONE\n",
    )

    with pytest.raises(ValueError, match="this job id completed DONE"):
        _queued_receipt(tmp_path, logs=(wrong_client, receipt.logs[1], receipt.logs[2]))


def test_final_firesim_receipt_rejects_nonminimal_profile(tmp_path) -> None:
    with pytest.raises(ValueError, match="only measured compute cycles"):
        _queued_receipt(tmp_path, warm_profile=_warm())


@pytest.mark.parametrize("uart", [
    "METRIC cycles 1234\nMETRIC idle_cycles 0\nDONE\n",
    "METRIC cycles 999\nDONE\n",
    "DONE\n",
])
def test_final_firesim_receipt_refuses_nonexact_uart_metric(tmp_path, uart) -> None:
    receipt = _queued_receipt(tmp_path)
    uart_log = _log(tmp_path, "uart", "receipts/other-uartlog", uart)

    with pytest.raises(ValueError, match="exactly one metric|only measured|do not match"):
        _queued_receipt(tmp_path, logs=(receipt.logs[0], receipt.logs[1], uart_log))


def test_warm_profile_becomes_target_neutral_occupancy_without_guessing_roles() -> None:
    receipt = WarmComputeReceipt(
        "reduced_complete_model", 100,
        WarmProfileContract(captured_metrics=frozenset({
            "total_compute_cycles", "resource_busy_cycles", "movement_bytes",
            "movement_commands", "encoding_transitions",
            "movement_compute_overlap_cycles", "overlap_available_cycles", "idle_cycles",
            "critical_path_cycles"})),
        "same-window hardware counters",
        resource_busy_cycles=(("engine_a", 80), ("engine_b", 40)),
        movement_bytes=1024, movement_commands=4, encoding_transitions=1,
        movement_compute_overlap_cycles=30, overlap_available_cycles=40,
        idle_cycles=10, critical_path_cycles=90)

    occupancy = occupancy_from_warm_receipt(
        receipt, {"engine_a": "compute", "engine_b": "movement"})

    assert occupancy.compute_utilization == pytest.approx(0.8)
    assert occupancy.latency_hiding_efficiency == pytest.approx(0.75)
    assert occupancy.movement_bytes == 1024
    assert occupancy.encoding_transitions == 1
    assert occupancy.missing == ()


def test_partial_warm_profile_keeps_latency_hiding_unknown(tmp_path) -> None:
    occupancy = occupancy_from_warm_receipt(
        _warm(), {"array": "compute", "dma": "movement"})

    assert occupancy.compute_utilization == pytest.approx(1000 / 1234)
    assert occupancy.latency_hiding_efficiency is None
    assert occupancy.encoding_transitions is None
    assert "movement/compute overlap cycles" in occupancy.missing
    with pytest.raises(ValueError, match="owned by the FireSim queue"):
        _queued_receipt(tmp_path, queue_owned=False)


def test_warm_conversion_preserves_declared_but_unobserved_engine():
    occupancy = occupancy_from_warm_receipt(
        _warm(), {"array": "compute", "dma": "movement", "second_engine": "compute"})
    assert occupancy.compute_resources == ("array", "second_engine")
    assert occupancy.compute_busy_cycles is None
    assert occupancy.compute_utilization is None
    assert occupancy.busy == {"array": 1000, "dma": 300}
    assert "busy cycles for declared resource second_engine" in occupancy.missing


def test_warm_conversion_preserves_missing_movement_counter_without_guessing_zero():
    occupancy = occupancy_from_warm_receipt(
        _warm(), {"array": "compute", "dma": "movement", "store_engine": "movement"})
    assert occupancy.movement_resources == ("dma", "store_engine")
    assert "store_engine" not in occupancy.busy
    assert "busy cycles for declared resource store_engine" in occupancy.missing
    assert occupancy.compute_utilization == pytest.approx(1000 / 1234)
