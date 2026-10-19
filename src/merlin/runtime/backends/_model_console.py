"""Parse the simulator-independent bare-metal model console protocol."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np


def parse_model_console(console: str, *, error_type: type[RuntimeError],
                        expected_output_elements: int | None = None) -> dict[str, Any]:
    """Parse the shared bare-metal model protocol, independent of its simulator.

    OUT/OUT_I64/OUT_I1 is a bounded prefix (at most 4096 values). Large
    outputs use an OUT_BEGIN declaration and contiguous OUT_CHUNK records.
    A prefix is never evidence of a complete larger tensor; the caller must
    compare the observed count with its independently captured output shape.
    Process success and hardware provenance belong to the caller.
    Duplicate, truncated, or malformed output must never qualify a run.
    An explicit caller-derived output count permits larger complete tensors
    and requires exact equality. The default resource bound stays unchanged.
    """
    if expected_output_elements is not None and (
            type(expected_output_elements) is not int or
            not 1 <= expected_output_elements <= 2**31 - 1):
        raise error_type("expected output count must be a positive signed 32-bit integer")
    lines = console.splitlines()
    if any(line.startswith(("FATAL:", "FATAL ", "FAIL ")) or "*** FAILED" in line
           for line in lines):
        raise error_type("guest reported failure in model console")
    out_lines = [(i, line) for i, line in enumerate(lines)
                 if line.startswith(("OUT ", "OUT_I64 ", "OUT_I1 "))]
    begin_lines = [(i, line) for i, line in enumerate(lines) if line.startswith("OUT_BEGIN ")]
    chunk_lines = [(i, line) for i, line in enumerate(lines) if line.startswith("OUT_CHUNK ")]
    done_lines = [i for i, line in enumerate(lines) if line.strip() == "DONE"]
    if len(done_lines) != 1 or bool(out_lines) == bool(begin_lines):
        raise error_type(f"run requires one output stream and DONE:\n{console[-2000:]}")
    if begin_lines:
        if len(begin_lines) != 1 or not chunk_lines or begin_lines[0][0] >= done_lines[0]:
            raise error_type("malformed or incomplete chunked model output")
        head = begin_lines[0][1].split()
        try:
            dtype, n = head[1], int(head[2])
        except (IndexError, ValueError) as exc:
            raise error_type("malformed OUT_BEGIN") from exc
        limit = expected_output_elements if expected_output_elements is not None else 1_048_576
        if len(head) != 3 or dtype not in {"f32", "i64", "i1"} or not 4096 < n <= limit:
            raise error_type("invalid chunked model output type or count")
        if expected_output_elements is not None and n != expected_output_elements:
            raise error_type("model output count differs from the expected complete tensor")
        integer, boolean = dtype == "i64", dtype == "i1"
        bits = []
        offset = 0
        for line_index, line in chunk_lines:
            parts = line.split()
            try:
                start, count = int(parts[2]), int(parts[3])
                words = [int(word) for word in parts[4:]]
            except (IndexError, ValueError) as exc:
                raise error_type("malformed OUT_CHUNK") from exc
            if (len(parts) < 5 or parts[1] != dtype or start != offset or
                    not 1 <= count <= 1024 or count > n - offset or
                    len(words) != count * (2 if integer else 1) or
                    not begin_lines[0][0] < line_index < done_lines[0]):
                raise error_type("missing, repeated, or out-of-order OUT_CHUNK")
            bits.extend(words)
            offset += count
        if offset != n:
            raise error_type("chunked model output is incomplete")
    else:
        if chunk_lines:
            raise error_type("OUT_CHUNK has no OUT_BEGIN")
        if len(out_lines) != 1 or out_lines[0][0] >= done_lines[0]:
            raise error_type("DONE preceded model output")
        parts = out_lines[0][1].split()
        integer = parts[0] == "OUT_I64"
        boolean = parts[0] == "OUT_I1"
        try:
            n = int(parts[1])
            bits = [int(x) for x in parts[2:]]
        except (IndexError, ValueError) as exc:
            raise error_type("malformed OUT count or raw output words") from exc
        if not 0 <= n <= 4096:
            raise error_type("legacy OUT count exceeds its bounded prefix")
        if expected_output_elements is not None and n != expected_output_elements:
            raise error_type("model output count differs from the expected complete tensor")
    maximum_word = 1 if boolean else 0xFFFFFFFF
    if len(bits) != n * (2 if integer else 1) or any(not 0 <= b <= maximum_word for b in bits):
        raise error_type("OUT count or raw output words do not match the bare-metal protocol")
    raw = (
        np.asarray([lo | (hi << 32) for lo, hi in zip(bits[::2], bits[1::2], strict=True)], dtype=np.uint64)
        if integer
        else np.asarray(bits, dtype=np.uint8 if boolean else np.uint32)
    )
    flat = raw.view(np.int64 if integer else np.bool_ if boolean else np.float32)
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
