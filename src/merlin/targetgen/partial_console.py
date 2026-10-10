"""What survives of a simulator console when the run is stopped before it finishes.

An elaborated-RTL run of a full-shape layer can spend its whole budget streaming the result tensor
over the simulated serial port, and the harness prints ``METRIC cycles`` BEFORE that frame. When the
run is stopped (wall-clock timeout or the engine's own cycle cap), the cycle count of the kernel
itself has therefore usually been printed already. Discarding the partial console loses a measured
number and leaves only "UNMEASURED". This module reads it back, tolerantly: no ``DONE`` is required,
and only complete ``METRIC <name> <int>`` lines count.
"""

from __future__ import annotations

from pathlib import Path

#: Where :func:`merlin.targetgen.contract.compile.run_on_oracle` leaves the transcript of an execution
#: (``.bin`` for the binary readback transport), including a partial one from a stopped run.
CONSOLE_NAMES = ("oracle_console.log", "oracle_console.bin")


def recover(workdir: str | Path) -> str | bytes | None:
    """The console a stopped run left in ``workdir``, or None when there is none."""
    for name in CONSOLE_NAMES:
        path = Path(workdir) / name
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            return data if name.endswith(".bin") else data.decode("utf-8", errors="replace")
    return None


def metrics(console: str | bytes | None) -> dict[str, int]:
    """Every complete ``METRIC <name> <int>`` line in ``console`` (text, or the text part of a binary
    frame), first occurrence wins; nothing is inferred from a truncated line."""
    if console is None:
        return {}
    text = console.decode("utf-8", errors="replace") if isinstance(console, bytes) else str(console)
    lines = text.split("\n")
    if not text.endswith("\n"):
        lines = lines[:-1]  # the last line may have been cut mid-print
    found: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[0] == "METRIC" and parts[1] not in found:
            try:
                found[parts[1]] = int(parts[2])
            except ValueError:
                continue
    return found
