"""Atlas-owned hooks for the generic inline-assembly preflight protocol."""
from __future__ import annotations

import importlib.util

from merlin.common.paths import ext_path
from merlin.targetgen.program_oracle import run_raw_program


def assemble(*, source: str, fixture: dict, **_kwargs) -> dict:
    """Assemble with the target tool selected by the descriptor fixture."""
    root = ext_path(str(fixture["assembler_root"]))
    path = root / str(fixture["assembler_path"])
    if not path.is_file():
        raise RuntimeError(f"declared target assembler is absent: {path}")
    spec = importlib.util.spec_from_file_location(
        f"_target_preflight_assembler_{abs(hash(path.resolve())):x}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import declared target assembler: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hook = getattr(module, str(fixture.get("assembler_callable") or "assemble"), None)
    if not callable(hook):
        raise RuntimeError(f"declared assembler callable is absent in {path}")
    return {"words": list(hook(source))}


def run(*, te, words: list[int], preload: list[tuple[int, bytes]],
        readback: list[tuple[int, int]], max_cycles: int, **_kwargs) -> dict:
    """Run assembled words on the descriptor target's discovered Arc program backend."""
    result = run_raw_program(te.target, words=words, preload=preload, max_cycles=max_cycles)
    return {
        "halted": bool(result.halted),
        "cycles": int(result.cycles),
        "reads": int(result.reads),
        "writes": int(result.writes),
        "memory": {
            address: bytes(result.slave.captured(address, size))
            for address, size in readback
        },
    }
