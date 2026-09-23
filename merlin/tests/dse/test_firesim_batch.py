"""What the batched hardware tier ADMITS and what it REFUSES, from fixture UART text only.

No hardware, no queue, no subprocess.  Every test here is a string and a dataclass, which is the
point: the admission rule has to be falsifiable without an FPGA slot, or it can only be tested by
the runs it is supposed to be protecting.

The centre of the file is :func:`test_a_correct_argmax_with_one_wrong_group_checksum_is_a_fail` --
the job-730 mutation.  That run returned 23,787,829 cycles, cleared both declared exit criteria,
and was wrong; its UART argmax marker printed identically to a correct run's, and only group 1's
checksum (5,652,929 against an oracle 5,663,048) showed it.  Everything else in this module exists
so that the mutation has a rule to fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.perf.execution_policy import (
    ROUTE_BATCHED_MEASUREMENT,
    ROUTE_IN_LOOP_PROBE,
    execution_route,
    require_batch_measurement,
    require_probe_execution,
)
from merlin.perf.firesim_batch import (
    VALIDATION_POLICY_SCHEMA_V2,
    BatchError,
    BatchMember,
    BatchValidationPolicy,
    admit_batch,
    admit_window,
    link_batch,
    load_batch_validation_policy,
    submission_spec,
)

_BATCH = "round-group-41"
_WEIGHTS = "a" * 64
_ARGMAX = "GM_ARGMAX got=21 want=21 agrees=1"
# The oracle checksums.  Group 1's is the number job 730 should have printed.
_ORACLE = {"1": 5_663_048, "2": 4_410_117}
# What job 730 actually printed for group 1, beside a byte-identical argmax marker.
_JOB_730_GROUP_1 = 5_652_929


def _members(n: int = 2, *, weights: list[str] | None = None, seconds: float = 100.0) -> list[BatchMember]:
    digests = weights or [_WEIGHTS] * n
    return [
        BatchMember(
            label=f"cand{index}",
            program_sha256=f"{index:064d}",
            weights_sha256=digests[index],
            observed_window_seconds=seconds,
        )
        for index in range(n)
    ]


def _policy_document(labels, *, checksums=None, markers=(_ARGMAX,)) -> dict:
    return {
        "schema": VALIDATION_POLICY_SCHEMA_V2,
        "policy_id": "synthetic/group-checksums-v2",
        "workload": "merlin-checkpoint",
        # The real spelling the gemmini group-model harness prints:
        # `GM_GROUP <group> <kind> <cycles> sum=<checksum>`.
        "checksum_line": {"prefix": "GM_GROUP", "value_key": "sum", "group_token_offset": 1},
        "per_window": [
            {"label": label, "markers": list(markers), "checksums": dict(_ORACLE if checksums is None else checksums)}
            for label in labels
        ],
    }


def _window_text(
    label: str, *, cycles: int | None, checksums: dict[str, int], markers=(_ARGMAX,), close: bool = True
) -> list[str]:
    body = [
        f"MERLIN_WINDOW begin label={label}",
        "MERLIN_PROFILE warmup begin",
        "MERLIN_PROFILE warmup end rc=0",
        "MERLIN_PROFILE measured begin",
        *markers,
        *(f"GM_GROUP {group} matmul 1234 sum={value}" for group, value in checksums.items()),
    ]
    if cycles is not None:
        body.append(f"METRIC cycles {cycles}")
    body.append("MERLIN_PROFILE measured end rc=0")
    if close:
        body.append(f"MERLIN_WINDOW end label={label}")
    return body


def _uart(batch, *, per_window, extra_lines=()) -> str:
    lines = [f"MERLIN_BATCH begin id={batch.batch_id} windows={len(batch.labels)}"]
    for label in batch.labels:
        lines.extend(_window_text(label, **per_window[label]))
    lines.extend(extra_lines)
    lines.append(f"MERLIN_BATCH end id={batch.batch_id}")
    return "\n".join(lines) + "\n"


def _linked(n: int = 2, **kwargs):
    return link_batch(_members(n), batch_id=_BATCH, queue_wall_limit_seconds=43200.0, **kwargs)


def _healthy(batch, *, cycles=1_000_000):
    """Every declared window correct, and the order control reproducing window 0 exactly."""
    return {label: {"cycles": cycles, "checksums": dict(_ORACLE)} for label in batch.labels}


# ----------------------------------------------------------------- the job-730 mutation
def test_a_correct_argmax_with_one_wrong_group_checksum_is_a_fail() -> None:
    """THE reason the admission rule exists.  Argmax clean, one checksum wrong -> fail, not incomplete."""
    batch = _linked(1)
    policy = BatchValidationPolicy.from_json(_policy_document(batch.labels))
    mutated = dict(_ORACLE) | {"1": _JOB_730_GROUP_1}
    per_window = _healthy(batch)
    per_window[batch.control_label] = {"cycles": 23_787_829, "checksums": mutated}

    verdict = admit_window(
        _uart(batch, per_window=per_window), policy.window(batch.control_label), policy.checksum_line
    )

    assert verdict.status == "fail", "a run that RAN and was wrong is not an absence"
    assert verdict.cycles == 23_787_829, "the wrong number is recorded, not discarded"
    assert verdict.checksum_mismatches == (("1", _JOB_730_GROUP_1, _ORACLE["1"]),)
    assert verdict.missing_markers == (), "the argmax marker printed exactly as a correct run's"
    assert "RAN and was wrong" in verdict.reason


def test_the_same_uart_without_the_mutation_passes() -> None:
    """The control for the test above: nothing else in that UART is what rejects it."""
    batch = _linked(1)
    policy = BatchValidationPolicy.from_json(_policy_document(batch.labels))
    per_window = _healthy(batch)
    per_window[batch.control_label] = {"cycles": 23_787_829, "checksums": dict(_ORACLE)}

    verdict = admit_window(
        _uart(batch, per_window=per_window), policy.window(batch.control_label), policy.checksum_line
    )

    assert verdict.status == "pass"
    assert verdict.cycles == 23_787_829


# ----------------------------------------------------------------- the v1 policy
def test_a_v1_policy_is_refused_for_a_cycle_claim(tmp_path: Path) -> None:
    """The shape job 730 defeated: an argmax and a cosine, and no per-group checksum at all."""
    v1 = {
        "schema": "merlin_firesim_uart_validation_policy_v1",
        "policy_id": "resnet50_group_model_correctness_v2",
        "workload": "merlin-checkpoint",
        "success_markers": [_ARGMAX, "GM_COSINE_PPM 997981"],
    }
    path = tmp_path / "validation-policy.json"
    path.write_text(json.dumps(v1, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(BatchError, match="per-window group checksums"):
        load_batch_validation_policy(path)


def test_a_v2_window_declaring_no_checksum_is_refused() -> None:
    """v2 without checksums would be v1 wearing a new schema string."""
    with pytest.raises(BatchError, match="declares no per-group checksum"):
        BatchValidationPolicy.from_json(_policy_document(("cand0",), checksums={}))


# ----------------------------------------------------------------- metric-line accounting
def test_a_stray_metric_line_invalidates_the_whole_batch() -> None:
    """An undeclared window ran; nothing can say what shared the FPGA with the declared ones."""
    batch = _linked(2)
    policy = BatchValidationPolicy.from_json(_policy_document(batch.labels))
    text = _uart(batch, per_window=_healthy(batch), extra_lines=("METRIC cycles 777",))

    verdict = admit_batch(text, batch, policy)

    assert verdict.status == "incomplete"
    assert "nobody declared" in verdict.reason
    assert "1 METRIC line(s) lie outside" in verdict.reason


def test_a_missing_metric_line_is_incomplete_and_the_capsule_is_not_dropped() -> None:
    """Not run is not pass -- and it is not a deletion either."""
    batch = _linked(2)
    policy = BatchValidationPolicy.from_json(_policy_document(batch.labels))
    per_window = _healthy(batch)
    absent = batch.labels[1]
    per_window[absent] = {"cycles": None, "checksums": dict(_ORACLE)}

    verdict = admit_batch(_uart(batch, per_window=per_window), batch, policy)

    missing = next(w for w in verdict.windows if w.label == absent)
    assert missing.status == "incomplete"
    assert missing.reason == "window_not_run"
    assert missing.cycles is None
    assert missing.to_dict()["not_run_is_not_pass"] is True
    assert [w.label for w in verdict.windows] == list(batch.labels), "an unmeasured window stays in the denominator"


def test_a_window_whose_frame_never_opened_is_incomplete_not_absent() -> None:
    batch = _linked(1)
    policy = BatchValidationPolicy.from_json(_policy_document(batch.labels))

    verdict = admit_window("boot noise\nDONE\n", policy.window(batch.control_label), policy.checksum_line)

    assert (verdict.status, verdict.reason) == ("incomplete", "window_not_run")


# ----------------------------------------------------------------- the order-effect control
def test_the_batch_appends_a_repeat_of_window_zero() -> None:
    batch = _linked(3)

    assert batch.labels[-1] == batch.repeat_label
    assert batch.repeat_label.startswith(batch.control_label)
    assert batch.control_label == "cand0"
    # The repeat costs one more window's observed wall, and the estimate says so.
    assert batch.estimated_wall_seconds == pytest.approx(400.0)


def test_an_order_effect_beyond_the_declared_bound_invalidates_the_batch() -> None:
    batch = _linked(2, order_effect_bound_ppm=10_000)
    policy = BatchValidationPolicy.from_json(_policy_document(batch.labels))
    per_window = _healthy(batch, cycles=1_000_000)
    per_window[batch.repeat_label] = {"cycles": 1_050_000, "checksums": dict(_ORACLE)}

    verdict = admit_batch(_uart(batch, per_window=per_window), batch, policy)

    assert verdict.status == "incomplete", "the instrument was disproved, not any candidate"
    assert verdict.order_effect_ppm == 50_000
    assert "position-contaminated" in verdict.reason
    # Each window still carries its own verdict: the batch is unusable, not unexamined.
    assert all(w.status == "pass" for w in verdict.windows)


def test_an_order_effect_within_the_bound_admits_the_batch() -> None:
    batch = _linked(2, order_effect_bound_ppm=10_000)
    policy = BatchValidationPolicy.from_json(_policy_document(batch.labels))
    per_window = _healthy(batch, cycles=1_000_000)
    per_window[batch.repeat_label] = {"cycles": 1_001_000, "checksums": dict(_ORACLE)}

    verdict = admit_batch(_uart(batch, per_window=per_window), batch, policy)

    assert verdict.status == "pass"
    assert verdict.order_effect_ppm == 1_000


# ----------------------------------------------------------------- linking
def test_members_disagreeing_on_the_weight_blob_are_refused() -> None:
    with pytest.raises(BatchError, match="ONE shared weights blob"):
        link_batch(_members(2, weights=[_WEIGHTS, "b" * 64]), batch_id=_BATCH, queue_wall_limit_seconds=43200.0)


def test_a_member_with_no_observed_wall_cannot_be_priced() -> None:
    with pytest.raises(BatchError, match="OBSERVED window wall time"):
        BatchMember(label="cand0", program_sha256="0" * 64, weights_sha256=_WEIGHTS, observed_window_seconds=0.0)


def test_a_batch_that_does_not_fit_the_queue_wall_is_refused_not_truncated() -> None:
    with pytest.raises(BatchError, match="drop members rather than truncate"):
        link_batch(_members(4, seconds=3_000.0), batch_id=_BATCH, queue_wall_limit_seconds=10_000.0)


def test_an_unnamed_round_group_is_refused() -> None:
    with pytest.raises(BatchError, match="round-group"):
        link_batch(_members(1), batch_id="  ", queue_wall_limit_seconds=43200.0)


def test_a_uart_from_another_batch_is_not_evidence_about_this_one() -> None:
    batch = _linked(1)
    policy = BatchValidationPolicy.from_json(_policy_document(batch.labels))
    text = _uart(batch, per_window=_healthy(batch)).replace(f"id={batch.batch_id}", "id=round-group-9")

    with pytest.raises(BatchError, match="not 'round-group-41'"):
        admit_batch(text, batch, policy)


# ----------------------------------------------------------------- submission is argv, never a run
def test_submission_spec_returns_one_queue_submission_and_runs_nothing(tmp_path: Path) -> None:
    batch = _linked(2)
    queue = tmp_path / "bin" / "firesim-queue"
    queue.parent.mkdir(parents=True)
    queue.write_text("#!/bin/sh\n", encoding="utf-8")

    preflight = submission_spec(
        batch,
        queue_executable=queue,
        workload="merlin-checkpoint",
        bootbinary=tmp_path / "batch.elf",
        validation_policy=tmp_path / "policy.json",
    )

    assert preflight.submission[:2] == (str(queue), "runworkload-full")
    assert "--batch-id" in preflight.submission
    assert batch.batch_id in preflight.submission
    # FireSimQueuePreflight is what refuses a nested direct firesim command; the argv never
    # contains one, and nothing here executes anything.
    assert not any(Path(token).name == "firesim" for token in preflight.submission)


def test_a_nested_direct_firesim_command_cannot_ride_a_batch_submission(tmp_path: Path) -> None:
    batch = _linked(1)
    queue = tmp_path / "bin" / "firesim-queue"
    queue.parent.mkdir(parents=True)
    queue.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(ValueError, match="nested in the queue submission"):
        submission_spec(
            batch,
            queue_executable=queue,
            workload="merlin-checkpoint",
            bootbinary=tmp_path / "firesim",
            validation_policy=tmp_path / "p.json",
        )


# ----------------------------------------------------------------- the policy route
@pytest.mark.parametrize(
    "descriptor",
    [
        {"kind": "model"},
        {"operation": {"op": "model"}},
        {"performance": {"global_objective": True}},
        {"performance": {"measurement_scope": "full_layer"}},
        {"semantic": {"generalization_axis": "model"}},
    ],
)
def test_the_five_probe_refusals_still_raise_and_route_to_the_batch(descriptor) -> None:
    """The route was ADDED beside the exclusion; the exclusion itself is untouched."""
    with pytest.raises(ValueError, match="compile-only search objective"):
        require_probe_execution(descriptor)
    assert execution_route(descriptor) == ROUTE_BATCHED_MEASUREMENT


def test_a_mechanism_probe_still_routes_to_the_in_loop_tier() -> None:
    probe = {"kind": "layer", "operation": {"op": "matmul"}, "performance": {"measurement_scope": "mechanism_probe"}}

    assert execution_route(probe) == ROUTE_IN_LOOP_PROBE
    with pytest.raises(ValueError, match="reduced-witness iteration tier"):
        require_batch_measurement(probe, batch_id=_BATCH, queue_owned=True)


@pytest.mark.parametrize(
    "batch_id,queue_owned,expected",
    [
        (_BATCH, False, "owned by the FireSim queue"),
        ("", True, "round-group it accumulates into"),
        ("   ", True, "round-group it accumulates into"),
    ],
)
def test_require_batch_measurement_refuses_an_unowned_or_unattributable_window(batch_id, queue_owned, expected) -> None:
    with pytest.raises(ValueError, match=expected):
        require_batch_measurement({"kind": "model"}, batch_id=batch_id, queue_owned=queue_owned)


def test_link_batch_refuses_a_non_model_descriptor() -> None:
    with pytest.raises(ValueError, match="whole-model/global objectives only"):
        link_batch(
            _members(1),
            batch_id=_BATCH,
            queue_wall_limit_seconds=43200.0,
            descriptor={"kind": "layer", "operation": {"op": "matmul"}},
        )
