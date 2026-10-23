"""Lossless ``OUT_BYTES`` result readback for the bare-metal model console protocol.

Each result is emitted as ordered storage bytes in MLIR result order, optionally with
``OUT_SHAPE`` extents and the strict semantic frames parsed by
:mod:`merlin.runtime.semantic_readback`. The prefix protocol in
:func:`merlin.runtime.backends.spike_model.parse_console` stays the default; it
dispatches here only when the console carries byte frames.
"""

from __future__ import annotations

from typing import Any


def parse_all_output_bytes(console: str) -> list[bytes]:
    """Decode ordered result frames without interpreting their storage dtype.

    This frame decoder also accepts an unterminated fragment for diagnostics.
    The execution parser separately requires a single terminal DONE.
    """
    from merlin.runtime.backends.spike_model import SpikeModelError

    outputs = []
    for line in console.splitlines():
        parts = line.split()
        if not parts or parts[0] != "OUT_BYTES":
            continue
        try:
            index, count = int(parts[1]), int(parts[2])
            values = [int(value) for value in parts[3:]]
            if index != len(outputs) or count < 0 or len(values) != count:
                raise ValueError("result index or byte count disagrees with frame")
            outputs.append(bytes(values))
        except (IndexError, ValueError, OverflowError) as exc:
            raise SpikeModelError("malformed OUT_BYTES result index, count or bytes") from exc
    return outputs


def parse_bytes_console(console: str) -> dict[str, Any]:
    """Parse an ``OUT_BYTES`` console: ordered frames, optional shapes and semantics, one DONE."""
    from merlin.runtime.backends.spike_model import SpikeModelError

    lines = console.splitlines()
    done = [i for i, line in enumerate(lines) if line.strip() == "DONE"]
    frames = [i for i, line in enumerate(lines) if line.split()[:1] == ["OUT_BYTES"]]
    if len(done) != 1 or not frames or frames[-1] >= done[0]:
        raise SpikeModelError("run requires ordered OUT_BYTES followed by exactly one DONE")
    if any(line.startswith(("OUT ", "OUT_I64 ", "OUT_I1 ")) for line in lines):
        raise SpikeModelError("mixed model output protocols")
    metrics = {}
    for line in lines:
        if line.startswith("METRIC "):
            parts = line.split()
            if len(parts) != 3 or parts[1] in metrics:
                raise SpikeModelError("malformed or duplicate model METRIC")
            try:
                metrics[parts[1]] = parts[2] if parts[1] == "build_hash" else int(parts[2])
            except ValueError:
                metrics[parts[1]] = parts[2]
    output_bytes = parse_all_output_bytes(console)
    shapes = []
    for at, line in enumerate(lines):
        parts = line.split()
        if not parts or parts[0] != "OUT_SHAPE":
            continue
        try:
            index, rank, *dims = [int(p) for p in parts[1:]]
            if index != len(shapes) or rank < 0 or len(dims) != rank or any(d < 0 for d in dims) or at >= done[0]:
                raise ValueError("invalid shape frame")
            shapes.append(dims)
        except ValueError as exc:
            raise SpikeModelError("malformed OUT_SHAPE result index, rank or extents") from exc
    if shapes and len(shapes) != len(output_bytes):
        raise SpikeModelError("result shape/byte frame count differs")
    from merlin.runtime.semantic_readback import parse_semantic_readback

    semantic = parse_semantic_readback(console, len(output_bytes))
    return {
        "output_bytes": output_bytes,
        "output_shapes": shapes,
        "metrics": metrics,
        "console": console,
        **({"semantic_readback": semantic} if semantic is not None else {}),
    }
