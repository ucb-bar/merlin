"""Parse the simulator-independent bare-metal model console protocol."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np


def parse_model_console(console: str, *, error_type: type[RuntimeError]) -> dict[str, Any]:
    """Parse the shared bare-metal model protocol, independent of its simulator.

    OUT/OUT_I64/OUT_I1 is a bounded prefix (at most 4096 values), not evidence of a complete
    larger tensor. Process success and hardware provenance belong to the caller.
    Duplicate, truncated, or malformed output must never qualify a run.
    """
    lines = console.splitlines()
    out_lines = [line for line in lines if line.startswith(("OUT ", "OUT_I64 ", "OUT_I1 "))]
    done_lines = [line for line in lines if line.strip() == "DONE"]
    if len(out_lines) != 1 or len(done_lines) != 1:
        raise error_type(f"run requires exactly one OUT and DONE:\n{console[-2000:]}")
    if lines.index(out_lines[0]) >= lines.index(done_lines[0]):
        raise error_type("DONE preceded model output")
    parts = out_lines[0].split()
    integer = parts[0] == "OUT_I64"
    boolean = parts[0] == "OUT_I1"
    try:
        n = int(parts[1])
        bits = [int(x) for x in parts[2:]]
    except (IndexError, ValueError) as exc:
        raise error_type("malformed OUT count or raw output words") from exc
    maximum_word = 1 if boolean else 0xFFFFFFFF
    if not 0 <= n <= 4096 or len(bits) != n * (2 if integer else 1) or any(not 0 <= b <= maximum_word for b in bits):
        raise error_type("OUT count or raw output words do not match the bare-metal protocol")
    raw = (
        np.asarray([lo | (hi << 32) for lo, hi in zip(bits[::2], bits[1::2], strict=True)], dtype=np.uint64)
        if integer
        else np.asarray(bits, dtype=np.uint8 if boolean else np.uint32)
    )
    flat = raw.view(np.int64 if integer else np.bool_ if boolean else np.float32)  # exact prefix (≤4096)
    metrics = {}
    argmax = None
    sumval = None
    for line in console.splitlines():
        if line.startswith("METRIC "):
            parts = line.split()
            if len(parts) != 3 or parts[1] in metrics:
                raise error_type("malformed or duplicate model METRIC")
            _, k, v = parts
            # Not every metric is a number: `build_hash` is a hex digest. Keeping the string beats
            # crashing the parse of an otherwise complete run.
            try:
                metrics[k] = int(v)
            except ValueError:
                metrics[k] = v
        elif line.startswith("ARGMAX "):
            if integer or boolean:
                raise error_type("integer/Boolean model output cannot carry a floating ARGMAX digest")
            p = line.split()
            try:
                count = int(p[1])
                values = [int(x) for x in p[2:]]
                if argmax is not None or count < 0 or len(values) != count:
                    raise ValueError("duplicate or truncated ARGMAX")
                argmax = np.array(values, dtype=np.int64)
            except (IndexError, ValueError, OverflowError) as exc:
                raise error_type("malformed or duplicate model ARGMAX") from exc
        elif line.startswith("SUM "):
            if integer or boolean:
                raise error_type("integer/Boolean model output cannot carry a floating SUM digest")
            p = line.split()
            try:
                bits = int(p[1])
                if sumval is not None or len(p) != 2 or not 0 <= bits <= 0xFFFFFFFF:
                    raise ValueError("duplicate or malformed SUM")
                sumval = struct.unpack("<f", struct.pack("<I", bits))[0]
            except (IndexError, ValueError) as exc:
                raise error_type("malformed or duplicate model SUM") from exc
    return {
        "outputs": flat,
        "prefix": flat,
        "output_dtype": "i64" if integer else "i1" if boolean else "f32",
        "raw_output_bits": raw,
        "argmax": argmax,
        "sum": sumval,
        "metrics": metrics,
        "console": console,
    }
