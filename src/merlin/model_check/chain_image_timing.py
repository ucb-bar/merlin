"""Decode a continuous chain diagnostic that carries no numerical verdict.

The generated guest retains scalar counters until all forwards finish. Output
arrays are neither copied nor printed, and no reference is selected. A separate
complete-output execution remains necessary for numerical qualification.
"""

from __future__ import annotations


def _ticks(value):
    result = int(value)
    if str(result) != value or not 0 <= result < 1 << 64:
        raise ValueError("noncanonical chain timing counter")
    return result


def decode_timing(image, console):
    if image["transport"] != "timing_only_v1" or image.get("group_profile", {}).get("enabled", False):
        raise ValueError("continuous chain timing cannot contain group profile transport")
    if b"OUT_BIN_" in console or b"CHAIN_PROFILE_" in console:
        raise ValueError("timing-only transport cannot carry arrays or group profiles")
    rows = [line.split() for line in console.decode("ascii").splitlines()
            if line.startswith("CHAIN_") or line == "DONE"]
    schedule = image["plan"]["schedule"]
    if len(rows) != len(schedule) + 3 or image["chain_steps"] != len(schedule):
        raise ValueError("incomplete continuous chain timing roster")
    steps = []
    for expected, row in zip(schedule, rows, strict=False):
        if (len(row) != 6 or row[:2] != ["CHAIN_STEP", "v1"]
                or [_ticks(v) for v in row[2:4]] != [expected["program"], expected["step"]]):
            raise ValueError("missing/duplicate/reordered chain timing step")
        steps.append({
            "program": image["plan"]["programs"][expected["program"]]["name"],
            "step": expected["step"], "cycles": _ticks(row[4]),
            "memref_rank_mismatches": _ticks(row[5]),
        })
    terminal, timing, done = rows[-3:]
    if terminal != ["CHAIN_END", "v1", str(len(schedule)), "0", "0"] or done != ["DONE"]:
        raise ValueError("timing-only terminal cannot claim numerical comparisons")
    if len(timing) != 5 or timing[:2] != ["CHAIN_TIMING", "v1"]:
        raise ValueError("missing continuous whole-chain counter")
    total, forwards, readback = map(_ticks, timing[2:])
    if forwards != sum(row["cycles"] for row in steps) or forwards > total:
        raise ValueError("whole-chain interval does not enclose its forward intervals")
    return {
        "schema": "merlin.chain_image.readback.v1", "transport": "timing_only_v1",
        "complete_chain_steps": len(steps), "output_array_count": 0,
        "declared_output_array_count": len(image["output_roster"]), "comparisons": [],
        "step_cycles": steps,
        "timing": {
            "units": "selected_counter_ticks", "counter": image.get("counter"),
            "whole_chain": total, "forward_sum": forwards,
            "session_overhead": total - forwards, "readback_transport": readback,
            "whole_chain_scope": (
                "input/state transfers, arena reset, descriptors, forwards and scalar timing bookkeeping"
            ),
            "excluded": "initial boot/validation, array readback/comparison, final reporting and termination",
            "readback_scope": "buffered CHAIN_STEP and CHAIN_END reporting; excludes CHAIN_TIMING and DONE",
            "physical_cycles_verified": False,
        },
        "runtime_status": "pass" if all(not row["memref_rank_mismatches"] for row in steps) else "fail",
        "numerical_status": "not_evaluated_timing_only", "execution_origin_verified": False,
        "scientific_qualification": False,
        "scope": "Diagnostic continuous session; no numerical, placement or hardware qualification transferred.",
    }
