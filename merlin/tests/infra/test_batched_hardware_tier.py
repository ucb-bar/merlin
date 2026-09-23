"""The batched tier's three seams: the N-window receipt, the model tier block, and the schema word.

These are the places a batched hardware verdict could quietly become a pass:

* :class:`~merlin.perf.execution_policy.QueuedFireSimReceipt` counting cycle metrics;
* ``capsule_runner._model_tier_map``, which synthesises a model capsule's tier block because a model
  capsule never enters the tier ladder at all;
* the capsule schema's availability words, which had no consumer anywhere in ``merlin/python`` or
  ``build_tools/scripts`` and so could contradict ``required_oracle_tiers`` unnoticed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from merlin.perf.execution_policy import (
    FIRESIM_LIFECYCLE,
    FIRESIM_QUEUE_OPERATION,
    FIRESIM_QUEUE_PHASES,
    FireSimQueuePreflight,
    QueuedFireSimReceipt,
    QueueLogEvidence,
    WarmComputeReceipt,
    WarmProfileContract,
)
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.contract import schemas

# The mechanism is target-neutral; the DECLARATION it reads is not, and the tests that check the
# declaration read gemmini's own descriptor -- the one target whose deepest oracle is an FPGA behind
# a shared queue. The batch, receipt and policy tests below name no target at all.
pytestmark = pytest.mark.target("gemmini")

_JOB_ID = 535


def _daemon_text() -> str:
    from merlin.common.paths import merlin_dir

    return (merlin_dir() / "tests/data/firesim_queue/job535_daemon_phase_skeleton.log").read_text(encoding="utf-8")


def _evidence(path: Path, text: str, role: str) -> QueueLogEvidence:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return QueueLogEvidence(role=role, path=str(path), sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())


def _window(label: str, cycles: int) -> WarmComputeReceipt:
    return WarmComputeReceipt(
        workload=label,
        total_compute_cycles=cycles,
        contract=WarmProfileContract(),
        provenance="one queue-owned post-warm FireSim compute-cycle window",
    )


def _receipt(
    tmp_path: Path, *, cycles: list[int], uart_cycles: list[int] | None = None, labels: list[str] | None = None
):
    queue = tmp_path / "bin" / "firesim-queue"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text("#!/bin/sh\n", encoding="utf-8")
    preflight = FireSimQueuePreflight(
        str(queue), (str(queue), FIRESIM_QUEUE_OPERATION, "--workload", "merlin-checkpoint")
    )
    client = (
        "\n".join(
            (
                f"[firesim-queue] job_id={_JOB_ID} kind={FIRESIM_QUEUE_OPERATION} state=QUEUED",
                f"[firesim-queue] job_id={_JOB_ID} terminal state=DONE exit_code=0 wall=1.0s",
            )
        )
        + "\n"
    )
    uart = "\n".join(f"METRIC cycles {value}" for value in (uart_cycles if uart_cycles is not None else cycles)) + "\n"
    names = labels or [f"w{index}" for index in range(len(cycles))]
    windows = [_window(name, value) for name, value in zip(names, cycles, strict=True)]
    return QueuedFireSimReceipt(
        queue_job_id=_JOB_ID,
        queue_owned=True,
        preflight=preflight,
        queue_phases=FIRESIM_QUEUE_PHASES,
        commands=FIRESIM_LIFECYCLE,
        logs=(
            _evidence(tmp_path / "queue-client.log", client, "queue_client"),
            _evidence(tmp_path / "jobs" / str(_JOB_ID) / "stdout.log", _daemon_text(), "queue_daemon"),
            _evidence(tmp_path / "uartlog", uart, "uart"),
        ),
        warm_profile=windows[0],
        additional_windows=tuple(windows[1:]),
    )


# ----------------------------------------------------------------- the N-window receipt
def test_one_window_still_seals_exactly_as_before(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, cycles=[987_654])

    assert receipt.measured_windows == (receipt.warm_profile,)
    # The serialised receipt is unchanged for a solo run: `measured_windows` appears only for a batch.
    assert "measured_windows" not in receipt.to_dict()


def test_n_windows_seal_when_each_publishes_its_own_metric(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, cycles=[10, 20, 30])

    assert [w.total_compute_cycles for w in receipt.measured_windows] == [10, 20, 30]
    assert len(receipt.to_dict()["measured_windows"]) == 3
    # The queue contract is untouched by the generalisation.
    assert receipt.to_dict()["queue_operation"] == FIRESIM_QUEUE_OPERATION
    assert tuple(receipt.queue_phases) == FIRESIM_QUEUE_PHASES


def test_a_stray_metric_line_refuses_the_receipt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one metric line per declared window"):
        _receipt(tmp_path, cycles=[10, 20], uart_cycles=[10, 20, 30])


def test_a_missing_metric_line_refuses_the_receipt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one metric line per declared window"):
        _receipt(tmp_path, cycles=[10, 20, 30], uart_cycles=[10, 20])


def test_windows_out_of_declared_order_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not match the warm profile receipt"):
        _receipt(tmp_path, cycles=[10, 20, 30], uart_cycles=[10, 30, 20])


def test_two_windows_cannot_share_a_label(tmp_path: Path) -> None:
    """Two windows with one label make a cycle count unattributable to the candidate that earned it."""
    with pytest.raises(ValueError, match="distinct workload label"):
        _receipt(tmp_path, cycles=[10, 20], labels=["same", "same"])


# ----------------------------------------------------------------- the model tier block
_DECLARED = ["L0", "L1", "L2", "L3", "L5"]
_ON_MESH = {"matmul_layers_on_mesh": 15, "matmul_layers_host_fallback": 0}


def test_a_batched_tier_is_unavailable_until_a_measurement_is_handed_in(monkeypatch) -> None:
    monkeypatch.setattr(CR, "_batched_tiers_of", lambda target: frozenset({"L5"}))

    tiers = CR._model_tier_map(_DECLARED, "gemmini", _ON_MESH)

    assert tiers["L5"].status == "unavailable"
    assert tiers["L5"].mandatory is True
    assert tiers["L5"].to_dict()["not_run_is_not_pass"] is True
    assert "round-group" in tiers["L5"].reason
    # A batched tier NEVER borrows the model's own layer counters: those describe a simulator run.
    assert tiers["L5"].cycles is None


def test_declaring_a_batched_tier_does_not_displace_the_simulator_tier(monkeypatch) -> None:
    """The bug this guards: `model_citable_rtl_tier` returns the LAST RTL tier, so declaring L5
    would have replaced L3's counter-derived verdict with an unavailable one and left L3 absent."""
    monkeypatch.setattr(CR, "_batched_tiers_of", lambda target: frozenset({"L5"}))

    tiers = CR._model_tier_map(_DECLARED, "gemmini", _ON_MESH)

    assert tiers["L3"].status == "pass"
    assert "executed on the accelerator" in tiers["L3"].reason
    assert set(tiers) >= {"L0", "L1", "L3", "L5"}


def test_a_handed_in_measurement_becomes_the_batched_tier_verdict(monkeypatch) -> None:
    monkeypatch.setattr(CR, "_batched_tiers_of", lambda target: frozenset({"L5"}))

    tiers = CR._model_tier_map(
        _DECLARED,
        "gemmini",
        _ON_MESH,
        measurement={
            "L5": {
                "status": "fail",
                "cycles": 23_787_829,
                "reason": "group 1 checksum 5652929 != 5663048",
                "cycle_accurate": True,
                "evidence": "uartlog",
            }
        },
    )

    assert (tiers["L5"].status, tiers["L5"].cycles) == ("fail", 23_787_829)
    assert tiers["L5"].cycle_accurate is True
    assert "5652929" in tiers["L5"].reason


def test_a_target_declaring_no_batched_tier_is_byte_identical(monkeypatch) -> None:
    monkeypatch.setattr(CR, "_batched_tiers_of", lambda target: frozenset())

    tiers = CR._model_tier_map(["L0", "L1", "L2", "L3"], "gemmini", _ON_MESH)

    assert set(tiers) == {"L0", "L1", "L3"}
    assert tiers["L3"].status == "pass"


def test_the_batched_declaration_is_read_off_the_targets_own_descriptor() -> None:
    """Item 7: no tier or simulator name lives in the runner -- the target's descriptor says."""
    from merlin.targetgen.target_experiment import batched_oracle_tiers

    assert batched_oracle_tiers("gemmini") == frozenset({"L5"})
    assert batched_oracle_tiers(None) == frozenset()
    assert batched_oracle_tiers("no-such-target-exists") == frozenset()


def test_the_descriptor_states_the_resource_and_its_wall() -> None:
    from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment

    experiment = load_target_experiment(descriptor_for("gemmini"))

    assert experiment.batched_oracle_tiers == frozenset({"L5"})
    assert experiment.exclusive_resource("L5") == "firesim_fpga_queue"
    assert experiment.queue_wall_limit_seconds("L5") == 43200.0
    # A tier that contends for nothing says so, rather than resolving to a default.
    assert experiment.exclusive_resource("L3") is None
    assert experiment.queue_wall_limit_seconds("L3") is None


# ----------------------------------------------------------------- the schema word
def _capsule(**overrides) -> dict:
    capsule = {
        "name": "SY_probe",
        "kind": "isa",
        "source_role": "derived_sweep",
        "label": "public",
        "operation": {"op": "matmul", "attributes": {"lhs": "A0", "weight": "W", "out": "Y0"}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "expected": {"instruction_classes": ["MVIN"]},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
    }
    capsule.update(overrides)
    return capsule


def test_requiring_a_tier_whose_availability_says_unavailable_is_refused() -> None:
    capsule = _capsule(required_oracle_tiers=["L0", "L1", "L2", "L3", "L5"], firesim="unavailable")

    with pytest.raises(schemas.ContractViolation, match="oracle it declares cannot run"):
        schemas.validate_capsule(capsule)


def test_the_same_capsule_with_the_tier_available_validates() -> None:
    schemas.validate_capsule(_capsule(required_oracle_tiers=["L0", "L1", "L2", "L3", "L5"], firesim="optional"))


def test_unavailable_beside_a_tier_the_capsule_does_not_require_is_fine() -> None:
    """`firesim: unavailable` is only a contradiction against a tier the capsule DEMANDS."""
    schemas.validate_capsule(_capsule(firesim="unavailable"))


def test_the_pairing_is_read_from_the_schema_not_written_in_code() -> None:
    """The check knows no tier and no simulator name: both come from `x-oracle-tier`."""
    properties = schemas.load_schema("capsule")["properties"]

    annotated = {
        name: spec["x-oracle-tier"]
        for name, spec in properties.items()
        if isinstance(spec, dict) and "x-oracle-tier" in spec
    }
    assert annotated == {"vcs": "L4", "firesim": "L5", "verilator": "L3"}
