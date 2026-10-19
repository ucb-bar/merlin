"""Read digest-checked output dumps from a compiled multi-program session."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .session_bundle import MultiProgramSession

_DTYPE_BYTES = {"f32": 4, "i64": 8, "i1": 1, "bf16": 2,
                "f16": 2, "i32": 4}


def observed_output_dump_bytes(log: str, session: MultiProgramSession,
                               selected: list[dict], digests: list[dict], *,
                               byte_limit: int) -> dict[tuple[int, int, int], bytes]:
    """Read complete native-session hex taps without writing observed artifacts.

    The caller first validates the full digest stream against its typed session
    ABI. This reader binds every selected step's complete bytes to that digest,
    refusing duplicates, missing taps and the caller's explicit size limit.
    It establishes byte transport only, never numerical or runtime equivalence.
    """
    if type(byte_limit) is not int or byte_limit <= 0:
        raise ValueError("invalid session output dump byte limit")
    wanted = {(row["program"], row["output"]) for row in selected}
    if len(wanted) != len(selected) or any(
            type(program) is not int or not 0 <= program < len(session.programs)
            or type(output) is not int or output < 0 for program, output in wanted):
        raise ValueError("selected output dumps differ from the session ABI")
    rows = [row for row in digests if (row["program"], row["output"]) in wanted]
    expected = {(row["program"], row["step"], row["output"]): row for row in rows}
    keys = {(program, step, output) for program, output in wanted
            for step in range(session.programs[program].steps)}
    if len(expected) != len(rows) or set(expected) != keys:
        raise ValueError("selected output dumps differ from the session ABI")
    values = {}
    for line in log.splitlines():
        if not line.startswith("SESSION_HEX "):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError("malformed session output dump")
        try:
            program, step, output, byte_count = map(int, parts[1:5])
        except ValueError as error:
            raise ValueError("malformed session output dump") from error
        key = (program, step, output)
        if (key not in expected or key in values or byte_count <= 0
                or byte_count != expected[key]["bytes"] or byte_count > byte_limit
                or len(parts[5]) != byte_count * 2):
            raise ValueError("session output dump differs from its capture ABI")
        try:
            data = bytes.fromhex(parts[5])
        except ValueError as error:
            raise ValueError("malformed session output dump") from error
        if len(data) != byte_count or hashlib.sha256(data).hexdigest() != expected[key]["sha256"]:
            raise ValueError("session output dump differs from its guest digest")
        values[key] = data
    if set(values) != set(expected):
        raise ValueError("complete session lacks a selected output dump")
    return values


def assembled_output_bytes(session: MultiProgramSession, run_root: Path,
                           dumps: list[dict], *, program: str,
                           output: int) -> tuple[bytes, ...]:
    """Return each step's output after checking its ABI, path, and digest."""
    if program not in session.program_names:
        raise ValueError(f"unknown session program {program!r}")
    program_index = session.program_names.index(program)
    selected = session.programs[program_index]
    if not isinstance(output, int) or output < 0 or output >= len(selected.outputs):
        raise ValueError("unknown session output")
    shape, dtype = selected.outputs[output]
    if dtype not in _DTYPE_BYTES or any(type(dim) is not int or dim <= 0
                                        for dim in shape):
        raise ValueError("unsupported session output ABI")
    byte_count = _DTYPE_BYTES[dtype]
    for dim in shape:
        byte_count *= dim
    root = Path(run_root).resolve(strict=True)
    by_step: dict[int, bytes] = {}
    for row in dumps:
        if row.get("program") != program_index or row.get("output") != output:
            continue
        step = row.get("step")
        relative = Path(str(row.get("path", "")))
        if (type(step) is not int or not 0 <= step < selected.steps or
                step in by_step or row.get("bytes") != byte_count or
                relative.is_absolute() or not relative.parts or
                ".." in relative.parts):
            raise ValueError("session output dump differs from its ABI")
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("session output dump must be a regular file")
        path = candidate.resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("session output dump escapes its run directory")
        data = path.read_bytes()
        if (len(data) != byte_count or
                hashlib.sha256(data).hexdigest() != row.get("sha256")):
            raise ValueError("session output dump differs from its digest")
        by_step[step] = data
    if set(by_step) != set(range(selected.steps)):
        raise ValueError("session output dumps do not cover every step")
    return tuple(by_step[step] for step in range(selected.steps))
