"""Observe compact guest transport without certifying execution or numerics.

This standard-library-only reader derives completion from actual UART records.
It does not read references, authenticate the UART, or qualify placement.
"""

from __future__ import annotations

import hashlib
import math
import struct

SCHEMA = "merlin.post_freeze.chain_observation.v1"
_WIDTHS = {"f32": 4, "f64": 8, "bf16": 2, "f16": 2, "i64": 8, "i32": 4, "i16": 2, "i8": 1, "i1": 1}


def _integer(token):
    value = int(token)
    if str(value) != token or not 0 <= value < 1 << 64:
        raise ValueError("noncanonical chain counter/extent")
    return value


def _hex(token):
    if len(token) != 16 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("noncanonical chain bit-pattern/digest")
    return int(token, 16)


def _double(token):
    value = struct.unpack("<d", struct.pack("<Q", _hex(token)))[0]
    if math.isnan(value) or value < 0:
        raise ValueError("invalid nonnegative chain error summary")
    return value if math.isfinite(value) else None


def _plan(model):
    stages, roster = model["stages"], model["output_roster"]
    if not isinstance(stages, dict) or not stages or not isinstance(roster, list) or not roster:
        raise ValueError("compact observation needs stages and a complete nonempty output roster")
    order = []
    for row in roster:
        stage = row["stage"]
        if not isinstance(stage, str) or stage not in stages:
            raise ValueError("output roster has an unknown stage")
        if stage not in order:
            order.append(stage)
        if any(type(row[key]) is not int or row[key] < 0 for key in ("step", "index")):
            raise ValueError("output roster has invalid step/index")
    if set(order) != set(stages):
        raise ValueError("zero-output or omitted stages are ambiguous")
    expected, steps = [], []
    for program, stage in enumerate(order):
        policy = stages[stage]
        count, outputs = policy["steps"], policy["outputs"]
        if type(count) is not int or not 0 < count < 1 << 64 or not isinstance(outputs, list) or not outputs:
            raise ValueError("zero-output steps or invalid stage counts are ambiguous")
        for output in outputs:
            shape = output["shape"]
            if (
                not isinstance(output["dtype"], str)
                or output["dtype"] not in _WIDTHS
                or not isinstance(shape, list)
                or any(type(d) is not int or d <= 0 for d in shape)
                or math.prod(shape) >= 1 << 64
            ):
                raise ValueError("unsupported compact output storage ABI")
        # The supplied roster bounds expansion even for malformed step declarations.
        if count * len(outputs) > len(roster):
            raise ValueError("stage counts exceed the supplied output roster")
        for step in range(count):
            rows = [
                {"stage": stage, "step": step, "index": i, "dtype": output["dtype"], "shape": output["shape"]}
                for i, output in enumerate(outputs)
            ]
            expected.extend(rows)
            steps.append((program, stage, step, rows))
    if (
        roster != expected
        or type(model["chain_steps"]) is not int
        or model["chain_steps"] != len(steps)
        or (
            "declared_output_arrays" in model
            and (type(model["declared_output_arrays"]) is not int or model["declared_output_arrays"] != len(roster))
        )
    ):
        raise ValueError("compact roster differs from the complete declared stages")
    return steps


def observe_compact(model: dict, console: bytes) -> dict:
    """Validate an exact ordered compact transcript, retaining observed failures.

    Program indices follow first stage appearance in the selected output roster.
    Reference and actual digests are syntax-checked only: valid float tolerance
    and signed-zero comparisons need not have equal byte digests.
    """
    steps = _plan(model)
    if not isinstance(console, bytes) or b"OUT_BIN_" in console:
        raise ValueError("compact observation needs unmixed UART bytes")
    rows = []
    for line in console.decode("ascii").splitlines():
        tokens = line.split()
        if tokens and (tokens[0].startswith("CHAIN_") or tokens[0] == "DONE"):
            if not line.startswith(tokens[0]):
                raise ValueError("noncanonical prefixed chain record")
            rows.append(tokens)
    at, comparisons, cycles, mismatches = 0, [], [], 0
    for program, stage, step, outputs in steps:
        if at >= len(rows):
            raise ValueError("truncated compact step transport")
        row = rows[at]
        at += 1
        if len(row) != 6 or row[:2] != ["CHAIN_STEP", "v1"] or list(map(_integer, row[2:4])) != [program, step]:
            raise ValueError("missing/duplicate/reordered chain step")
        cycles.append(
            {
                "stage": stage,
                "program": program,
                "step": step,
                "cycles": _integer(row[4]),
                "memref_rank_mismatches": _integer(row[5]),
            }
        )
        for output in outputs:
            if at >= len(rows):
                raise ValueError("truncated compact array transport")
            row = rows[at]
            at += 1
            elements = math.prod(output["shape"])
            if (
                len(row) < 13
                or row[:2] != ["CHAIN_ARRAY", "v1"]
                or list(map(_integer, row[2:6])) != [program, step, output["index"], elements]
            ):
                raise ValueError("missing/duplicate/reordered compact array")
            failed, finite, samples = _integer(row[6]), _integer(row[7]), _integer(row[12])
            if (
                failed > elements
                or finite not in (0, 1)
                or (not finite and not failed)
                or samples != min(failed, 4)
                or len(row) != 13 + 3 * samples
            ):
                raise ValueError("inconsistent compact comparison summary")
            locations = list(map(_integer, row[13::3]))
            if len(set(locations)) != len(locations) or any(i >= elements for i in locations):
                raise ValueError("invalid compact mismatch sample positions")
            values = [(_hex(row[i]), _hex(row[i + 1])) for i in range(14, len(row), 3)]
            if any(max(pair) >= 1 << (8 * _WIDTHS[output["dtype"]]) for pair in values):
                raise ValueError("compact sample exceeds declared storage width")
            maximum, square = _double(row[8]), _double(row[9])
            _hex(row[10])
            _hex(row[11])
            comparisons.append(
                {
                    **output,
                    "elements": elements,
                    "mismatched": failed,
                    "finite": bool(finite),
                    "max_abs_error": maximum if finite else None,
                    "rms_error": math.sqrt(square / elements) if finite and square is not None else None,
                    "actual_fnv1a64": row[10],
                    "reference_fnv1a64": row[11],
                    "samples": [
                        {"index": i, "actual_bits": a, "reference_bits": b}
                        for i, (a, b) in zip(locations, values, strict=True)
                    ],
                }
            )
            mismatches += failed
    if (
        len(rows) != at + 2
        or rows[at] != ["CHAIN_END", "v1", str(len(cycles)), str(len(comparisons)), str(mismatches)]
        or rows[-1] != ["DONE"]
    ):
        raise ValueError("compact terminal counts are absent or inconsistent")
    runtime_ok = all(row["memref_rank_mismatches"] == 0 for row in cycles)
    return {
        "schema": SCHEMA,
        "transport": "all_element_comparison_v1",
        "uart": {"sha256": hashlib.sha256(console).hexdigest(), "bytes": len(console)},
        "completed_chain_steps": len(cycles),
        "output_array_count": len(comparisons),
        "mismatched_elements": mismatches,
        "step_cycles": cycles,
        "comparisons": comparisons,
        "runtime_status": "pass" if runtime_ok else "fail",
        "comparison_status": "pass" if not mismatches else "fail",
        "status": "observed_complete" if runtime_ok and not mismatches else "observed_failure",
        "execution_origin_verified": False,
        "numerical_qualification": False,
        "placement_qualified": False,
        "scientific_qualification": False,
        "scope": "UART transport observation only; no reference, execution, placement or numerical qualification.",
    }
