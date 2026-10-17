"""Record the exact before/after IR of each Merlin compilation boundary (design §8.1).

Off unless ``MERLIN_LEAN_STAGES`` names a directory. Each recorded boundary becomes
``<dir>/<seq>_<boundary>/{before.mlir,after.mlir,meta.json}`` with SHA-256s of both texts; sequence
numbers are reserved atomically and stages are published by rename, so concurrent recorders sharing a
directory neither collide nor expose partial stages (read them with ``completed``). Texts are
GENERIC-form prints of the module (the form MLIR parses back without custom-printer skew). Recording
only prints the module; it never changes it, so builds are byte-identical with recording on or off.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ENV = "MERLIN_LEAN_STAGES"
# One id per recording process, written into every stage it records: chains recorded concurrently into
# one directory stay distinguishable.
RUN = uuid.uuid4().hex


def enabled() -> bool:
    return bool(os.environ.get(ENV))


def _dir() -> Path:
    return Path(os.environ[ENV])


def generic_text(module) -> str:
    from merlin.xdsl_dialects._common import text

    return text(module, generic=True)


@contextmanager
def boundary(name: str, pass_entry: str, module) -> Iterator[None]:
    """Record ``module`` before and after the block as boundary ``name`` when recording is on; the
    block runs unchanged either way."""
    before = generic_text(module) if enabled() else None
    yield
    if before is not None:
        record_boundary(name, pass_entry, before, generic_text(module))


def run(name: str, fn, module, *args, **kwargs):
    """``fn(module, *args, **kwargs)``, recording ``module`` around the call as boundary ``name``."""
    with boundary(name, f"{fn.__module__}.{fn.__qualname__}", module):
        return fn(module, *args, **kwargs)


def parse_recorded(name: str, path):
    """Parse the MLIR file ``path`` with Merlin's reader, recording the file's text against the parsed
    module as boundary ``name``."""
    from merlin.frontends.linalg_mlir import parse_mlir_file

    module = parse_mlir_file(path)
    if enabled():
        record_boundary(
            name, "merlin.frontends.linalg_mlir.parse_mlir_file", Path(path).read_text(), generic_text(module)
        )
    return module


def write_recorded(name: str, path: Path, text: str, module) -> None:
    """Write ``text``, a print of ``module``, to ``path``, recording the module against it."""
    path.write_text(text)
    if enabled():
        record_boundary(name, "merlin.xdsl_dialects._common.text", generic_text(module), text)


def record_handoff(path: Path) -> None:
    """Record the file the upstream lowering reads, so a chain can be checked to end at those exact
    bytes (``<dir>/handoff.json``)."""
    if enabled():
        data = Path(path).read_bytes()
        root = _dir()
        root.mkdir(parents=True, exist_ok=True)
        (root / "handoff.json").write_text(
            json.dumps(
                {
                    "schema": "merlin.lean.handoff.v1",
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                },
                indent=2,
                sort_keys=True,
            )
        )


def _reserve(root: Path) -> int:
    """Claim the next sequence number atomically (``mkdir`` fails for every other claimant), so
    concurrent recorders sharing a directory never write the same boundary directory."""
    reserved = root / ".reserved"
    reserved.mkdir(parents=True, exist_ok=True)
    seq = sum(1 for _ in reserved.iterdir())
    while True:
        try:
            (reserved / f"{seq:06d}").mkdir()
            return seq
        except FileExistsError:
            seq += 1


def record_boundary(boundary: str, pass_entry: str, before: str, after: str, *, extra: dict | None = None) -> Path:
    """Write one boundary's before/after texts; returns the boundary directory. The directory is
    written under a hidden name and renamed into place once complete, so a reader never sees a
    partial stage (``completed`` lists only published ones)."""
    root = _dir()
    root.mkdir(parents=True, exist_ok=True)
    seq = _reserve(root)
    final = root / f"{seq:04d}_{boundary}"
    tmp = root / f".writing-{seq:04d}_{boundary}"
    tmp.mkdir()
    (tmp / "before.mlir").write_text(before)
    (tmp / "after.mlir").write_text(after)
    meta = {
        "schema": "merlin.lean.stage.v1",
        "boundary": boundary,
        "pass": pass_entry,
        "run": RUN,
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "changed": before != after,
        **({"extra": extra} if extra else {}),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    tmp.rename(final)
    return final


def completed(root: Path) -> list[Path]:
    """The published boundary directories under ``root``, in recording order (never a reservation or
    a stage still being written)."""
    return sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "meta.json").is_file()
    )
