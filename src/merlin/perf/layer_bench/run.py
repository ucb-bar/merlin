"""Run a layer program on the target's pinned GSIM engine and return what it printed.

The command comes from the backend's ``prepare_gsim_command``, which pins the ELF bytes and the engine
binary/receipt and refuses if either moved. It is revalidated immediately before and after the run, as
its contract asks. This module owns the one thing that contract leaves to the caller: a hard
wall-clock deadline.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .console import EngineFinish, LayerRecord, parse_engine_finish, parse_layer_records


class EngineUnavailable(RuntimeError):
    """The target declares no GSIM engine, or its engine is not usable here."""


@dataclass(frozen=True)
class EngineRun:
    records: tuple[LayerRecord, ...]
    finish: EngineFinish | None
    wall_seconds: float
    returncode: int
    stdout_sha256: str
    engine: dict
    command_sha256: str
    stdout_tail: str
    stderr_tail: str
    #: "backdoor" (the target's declared load backdoor was applied) or "serial" (engine default loader).
    load_path: str = "serial"

    @property
    def completed(self) -> bool:
        return self.returncode == 0 and self.finish is not None and self.finish.done


def run_on_gsim(
    elf: Path, *, target: str, max_cycles: int, timeout_s: float, backdoor: bool = True, stdout_path: Path | None = None
) -> EngineRun:
    """``backdoor`` applies the target's declared load backdoor (``gsim_backdoor_env``) when it has one,
    so image size stops costing simulated cycles. A target without one runs on the engine's default
    loader, and the run records which path it took.

    ``stdout_path`` also writes the program's COMPLETE stdout there (the returned run keeps only a tail
    plus the full text's sha256), for programs whose own report is longer than the tail."""
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import gsim_emulator

    backend = backends.get_backend(target)
    prepare = getattr(backend, "prepare_gsim_command", None)
    if prepare is None:
        raise EngineUnavailable(f"target {target!r} declares no GSIM command")
    engine = gsim_emulator.citation(target, env_var=getattr(backend, "GSIM_EMU_ENV", None))
    if not engine.get("available") or engine.get("refused"):
        raise EngineUnavailable(f"GSIM engine for {target!r} unavailable: {engine.get('why') or engine}")
    elf = Path(elf).absolute()
    elf_sha = hashlib.sha256(elf.read_bytes()).hexdigest()
    command = prepare(elf, expected_elf_sha256=elf_sha, expected_engine_provenance=engine, max_cycles=int(max_cycles))
    before = command.revalidate()
    env = dict(os.environ)
    env.update(dict(command.env_overrides))
    declared = getattr(backend, "gsim_backdoor_env", None)
    load_path = "serial"
    if backdoor and declared is not None:
        env.update(declared())
        load_path = "backdoor"
    start = time.monotonic()
    try:
        proc = subprocess.run(
            list(command.argv), capture_output=True, text=True, env=env, timeout=timeout_s, cwd=str(elf.parent)
        )
        returncode, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as expired:
        returncode = -1
        out = expired.stdout.decode(errors="replace") if isinstance(expired.stdout, bytes) else (expired.stdout or "")
        err = expired.stderr.decode(errors="replace") if isinstance(expired.stderr, bytes) else (expired.stderr or "")
    wall = time.monotonic() - start
    if stdout_path is not None:
        Path(stdout_path).write_text(out, encoding="utf-8")
    command.revalidate()
    return EngineRun(
        records=tuple(parse_layer_records(out)),
        finish=parse_engine_finish(err),
        wall_seconds=wall,
        returncode=returncode,
        stdout_sha256=hashlib.sha256(out.encode()).hexdigest(),
        engine=engine,
        command_sha256=before["command_sha256"],
        stdout_tail=out[-2000:],
        stderr_tail=err[-2000:],
        load_path=load_path,
    )
