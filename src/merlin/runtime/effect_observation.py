"""Decode opt-in source stdio and invocation outcomes separately from results.

Raw diagnostic bytes live inside a checked binary frame, so source strings
cannot impersonate driver markers. Nonzero exit alone never proves an assert.
"""

from .out_bin import binary_console_diagnostics, parse_binary_console_details


def decode(raw, returncode, *, max_bytes):
    if type(returncode) is not int or returncode not in (0, 134):
        raise ValueError("effect process did not return or stop through the selected source assertion runtime")
    if type(max_bytes) is not int or not 0 < max_bytes <= 4 << 20:
        raise ValueError("effect observation needs a bounded diagnostic capacity")
    returning = returncode == 0
    outputs, metrics, frames = parse_binary_console_details(raw, require_done=returning)
    if metrics.get("memref_rank_mismatch", 0):
        raise ValueError("effect invocation reported a runtime descriptor rank mismatch")
    lines = binary_console_diagnostics(raw, require_done=returning).splitlines()
    begin = b"MERLIN_EFFECT_BEGIN v1"
    end = b"MERLIN_EFFECT_END v1 " + (b"return" if returning else b"assertion_failure")
    if (not lines or lines[0] != begin or lines.count(begin) != 1 or lines.count(end) != 1
            or any(line.startswith(b"MERLIN_EFFECT_") and line not in (begin, end) for line in lines)):
        raise ValueError("effect outcome lacks its exact source invocation phase")
    boundary = lines.index(end)
    # Before END, only the single diagnostic frame may follow BEGIN. No
    # tensor result or caller observation may stand in for a completed call.
    if (boundary != 3 or not lines[1].startswith(b"OUT_BIN_BEGIN v1 source_stdout ")
            or not lines[2].startswith(b"OUT_BIN_END v1 ")
            or any(not (line.startswith((b"OUT_BIN_BEGIN v1 ", b"OUT_BIN_END v1 ", b"METRIC "))
                        or line == b"DONE") for line in lines[boundary + 1:])
            or not returning and (b"DONE" in lines or metrics)):
        raise ValueError("effect observation has incomplete, premature or unexpected driver output")
    frame = frames.pop("source_stdout", None)
    values = outputs.pop("source_stdout", None)
    if (frame is None or (frame.rows, frame.word_bytes, frame.signed) != (1, 1, False)
            or frame.cols > max_bytes or values is None):
        raise ValueError("effect outcome lacks its complete bounded source stdout frame")
    if frame.cols:
        if len(values) != 1 or len(values[0]) != frame.cols:
            raise ValueError("effect stdout byte count differs from its declared frame")
        stdout = bytes(values[0])
    else:
        if values != []:
            raise ValueError("empty source stdout has unexpected payload")
        stdout = b""
    if not returning and any(not name.startswith("readonly_arg_") for name in outputs):
        raise ValueError("an aborted source invocation cannot publish tensor results")
    return {"termination": "return" if returning else "assertion_failure", "stdout": stdout,
            "stderr": b"", "outputs": outputs, "frames": frames, "metrics": metrics}
