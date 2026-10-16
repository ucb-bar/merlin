"""Read digest-checked output dumps from a compiled multi-program session."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .session_bundle import MultiProgramSession


_DTYPE_BYTES = {"f32": 4, "i64": 8, "i1": 1, "bf16": 2,
                "f16": 2, "i32": 4}


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
