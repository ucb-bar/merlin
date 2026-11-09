"""Validate staged worker shared-clock windows without profiling the controller.

Raw clock ticks stay distinct from local cycles/retirement and physical busy.
All aligned tracks use one explicitly declared shared-clock epoch. Frequency,
source clock coherence and numerical equivalence require separate evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from merlin.common.jsonio import canonical_sha256
from merlin.common.strict_json import loads
from merlin.runtime.staged_dispatch import StagedRoles

_WINDOWS = ("arm_ready_tick", "compute_begin_tick", "compute_end_tick", "final_ack_tick")
_STATES = {0: "disabled", 1: "valid", 2: "read_error", 3: "nonmonotonic", 4: "incomplete"}


def _uint(value, name):
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise ValueError(f"{name} requires an unsigned 64-bit integer")
    return value


def _status(value, name):
    if type(value) is not int or not -(1 << 31) <= value < 1 << 31:
        raise ValueError(f"{name} requires a signed 32-bit status")
    return value


def _flag(value, name):
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} requires an integer zero/one flag")
    return bool(value)


def analyze_worker_windows(
    record: dict,
    roles: StagedRoles,
    *,
    expected_sequence: int,
    clock_domain: str,
    counter_semantics: str,
) -> tuple[dict, dict]:
    """Return a strict timing report and aligned raw-tick Perfetto coordinates.

    Explicit read errors, incomplete windows and clock regressions yield null
    durations and a failed profile gate; contradictory/malformed records refuse.
    The supplied role policy is independent of the observed record.
    """
    if not isinstance(record, dict) or record.get("abi_version") != 2:
        raise ValueError("staged worker window ABI 2 required")
    if type(record.get("abi_version")) is not int:
        raise ValueError("ABI version must be an integer")
    for value in (clock_domain, counter_semantics):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("explicit clock domain and counter semantics required")
    sequence = _uint(record.get("sequence"), "sequence")
    if type(expected_sequence) is not int or sequence != expected_sequence or sequence == 0:
        raise ValueError("stage sequence differs from independently selected sequence")
    if _uint(record.get("global_clock_hart"), "clock hart") != roles.compute_harts[0]:
        raise ValueError("global clock reader must be the declared compute leader")
    if not _flag(record.get("worker_clock_windows_enabled"), "window selection"):
        raise ValueError("worker windows were not explicitly enabled")
    status = _status(record.get("status"), "stage status")
    quiescent = _flag(record.get("quiescent"), "quiescence")
    claimed_all_valid = _flag(record.get("worker_clock_windows_valid"), "aggregate validity")
    global_valid = _flag(record.get("global_interval_valid"), "global validity")
    observed = record.get("workers")
    if not isinstance(observed, list):
        raise ValueError("explicit worker roster required")
    if not quiescent:
        if observed or claimed_all_valid or global_valid or status == 0:
            raise ValueError("nonquiescent record may publish only stable failure metadata")
    elif len(observed) != len(roles.compute_harts):
        raise ValueError("complete exact compute-worker roster required")

    rows, events, seen = [], [], set()
    for worker in observed:
        if not isinstance(worker, dict):
            raise ValueError("worker observations must be objects")
        hart = _uint(worker.get("hart"), "worker hart")
        if hart not in roles.compute_harts or hart in seen:
            raise ValueError("unexpected, controller or duplicated worker hart")
        seen.add(hart)
        compute_status = _status(worker.get("status"), "compute status")
        wait_status = _status(worker.get("wait_status"), "wait status")
        mask = _uint(worker.get("clock_valid_mask"), "clock mask")
        clock_status = _uint(worker.get("clock_status"), "clock status")
        reader_status = _status(worker.get("clock_reader_status"), "clock reader status")
        if mask > 15 or clock_status not in _STATES or clock_status == 0:
            raise ValueError("invalid or disabled clock windows")
        ticks = []
        for index, name in enumerate(_WINDOWS):
            raw = worker.get(name)
            ticks.append(_uint(raw, name) if mask & (1 << index) else None)
            if raw is not None:
                _uint(raw, name)
        monotonic = mask == 15 and all(ticks[i] <= ticks[i + 1] for i in range(3))
        expected_status = 2 if reader_status else 4 if mask != 15 else 1 if monotonic else 3
        if clock_status != expected_status:
            raise ValueError("clock status contradicts its raw mask, reader status or endpoints")
        valid = clock_status == 1 and status == 0 and compute_status == 0 and wait_status == 0
        endpoints = {
            key: _uint(worker.get(key), key) for key in ("begin_cycle", "end_cycle", "begin_instret", "end_instret")
        }
        local_valid = (
            endpoints["end_cycle"] >= endpoints["begin_cycle"]
            and endpoints["end_instret"] >= endpoints["begin_instret"]
        )
        local_observed = mask & 6 == 6
        durations = [ticks[i + 1] - ticks[i] for i in range(3)] if valid else [None] * 3
        inclusive = ticks[3] - ticks[0] if valid else None
        row = {
            "hart": hart,
            "status": compute_status,
            "wait_status": wait_status,
            "clock_status": _STATES[clock_status],
            "clock_reader_status": reader_status,
            "clock_valid_mask": mask,
            **dict(zip(_WINDOWS, ticks, strict=True)),
            **endpoints,
            "local_counters_monotonic": local_valid,
            "local_counter_interval_observed": local_observed,
            "compute_cycles": endpoints["end_cycle"] - endpoints["begin_cycle"]
            if local_valid and local_observed
            else None,
            "compute_retired": endpoints["end_instret"] - endpoints["begin_instret"]
            if local_valid and local_observed
            else None,
            "pre_compute_wait_ticks": durations[0],
            "compute_window_ticks": durations[1],
            "post_compute_wait_ticks": durations[2],
            "inclusive_worker_ticks": inclusive,
            "compute_window_fraction": durations[1] / inclusive if valid and inclusive else None,
            "window_valid": valid,
            "clock_window_valid": clock_status == 1,
            "physical_busy_fraction": None,
            "elapsed_seconds": None,
        }
        rows.append(row)
        if valid:
            for index, name in enumerate(("pre_compute_wait", "compute_callback_window", "post_compute_wait")):
                events.append(
                    {
                        "hart": hart,
                        "name": name,
                        "begin_tick": ticks[index],
                        "end_tick": ticks[index + 1],
                        "duration_ticks": durations[index],
                    }
                )

    publication_order_checked = bool(rows) and all(row["clock_window_valid"] for row in rows)
    if publication_order_checked and (
        max(row["arm_ready_tick"] for row in rows) > min(row["compute_begin_tick"] for row in rows)
        or max(row["compute_end_tick"] for row in rows) > min(row["final_ack_tick"] for row in rows)
    ):
        raise ValueError("shared-clock windows contradict ARM or finish publication order")
    all_valid = quiescent and status == 0 and bool(rows) and all(r["window_valid"] for r in rows)
    if claimed_all_valid != all_valid:
        raise ValueError("aggregate window validity contradicts complete worker observations")
    leader = next((row for row in rows if row["hart"] == roles.compute_harts[0]), None)
    global_begin = record.get("global_begin")
    global_end = record.get("global_end")
    if global_begin is not None:
        _uint(global_begin, "global_begin")
    if global_end is not None:
        _uint(global_end, "global_end")
    if global_valid and (
        leader is None
        or not leader["clock_window_valid"]
        or global_begin != leader["arm_ready_tick"]
        or global_end != leader["final_ack_tick"]
    ):
        raise ValueError("leader compatibility interval differs from observed worker windows")
    if all_valid and not global_valid:
        raise ValueError("complete valid windows require the leader compatibility interval")
    origins = [e["begin_tick"] for e in events]
    origin = min(origins) if origins else 0
    trace_events = [
        {
            "name": event["name"] + " [raw ticks]",
            "cat": "observed_staged_caller_window",
            "ph": "X",
            "pid": 0,
            "tid": event["hart"],
            "ts": event["begin_tick"] - origin,
            "dur": event["duration_ticks"],
            "args": {
                **event,
                "clock_domain": clock_domain,
                "coordinate_is_elapsed_time": False,
                "physical_busy": False,
            },
        }
        for event in events
    ]
    report = {
        "schema": "merlin.staged_worker_windows.v1",
        "pass": all_valid and all(row["local_counters_monotonic"] for row in rows),
        "source_record_canonical_sha256": canonical_sha256(record),
        "roles": roles.to_dict(),
        "stage_status": status,
        "quiescent": quiescent,
        "sequence": sequence,
        "clock_domain": clock_domain,
        "clock_hz": None,
        "counter_semantics": counter_semantics,
        "global_begin": global_begin,
        "global_end": global_end,
        "global_interval_valid": global_valid,
        "inclusive_stage_ticks": global_end - global_begin if global_valid and status == 0 else None,
        "workers": rows,
        "aligned_observed_windows": events,
        "scope": "observed software windows; waits include preparation/dispatch/finish and bookkeeping",
        "clock_source_qualified": False,
        "publication_order_checked": publication_order_checked,
        "numerical_gate_performed": False,
        "controller_profile_tracks": False,
        "physical_unit_busy": None,
        "causal_stall_attribution": None,
        "memory_bandwidth": None,
        "elapsed_seconds": None,
        "timing_accuracy_certified": False,
    }
    trace = {
        "traceEvents": trace_events,
        "metadata": {
            "clock_domain": clock_domain,
            "common_raw_tick_origin": origin,
            "coordinate_unit": "raw_shared_clock_ticks",
            "clock_hz": None,
            "coordinate_is_elapsed_time": False,
            "warning": "One raw tick uses one display-microsecond coordinate; this is not elapsed microseconds.",
            "scope": "Aligned observed caller windows on compute harts only; no controller or physical-busy tracks.",
        },
    }
    return report, trace


def write_worker_windows(directory: str | Path, report: dict, trace: dict) -> None:
    """Write a fresh immutable result directory; never change existing outputs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (directory / "trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False) + "\n")
    with (directory / "workers.csv").open("w", newline="") as stream:
        rows = report["workers"]
        fields = list(rows[0]) if rows else ["hart", "clock_status", "window_valid"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--clock-domain", required=True)
    parser.add_argument("--counter-semantics", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.input.open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    with args.roles.open("rb") as stream:
        policy_raw = stream.read((1 << 20) + 1)
    if len(raw) > 1 << 20 or len(policy_raw) > 1 << 20:
        raise ValueError("staged clock records exceed the one-MiB input budget")
    policy = loads(policy_raw)
    for name in ("cpu_hart_ids", "compute_harts", "profile_harts"):
        if not isinstance(policy.get(name), list):
            raise ValueError("role policy requires explicit JSON hart lists")
        policy[name] = tuple(policy[name])
    report, trace = analyze_worker_windows(
        loads(raw),
        StagedRoles(**policy),
        expected_sequence=args.sequence,
        clock_domain=args.clock_domain,
        counter_semantics=args.counter_semantics,
    )
    report["input_sha256"] = {
        str(args.input.resolve()): hashlib.sha256(raw).hexdigest(),
        str(args.roles.resolve()): hashlib.sha256(policy_raw).hexdigest(),
    }
    write_worker_windows(args.output, report, trace)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
