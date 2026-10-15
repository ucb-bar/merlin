"""Record the exact before/after IR of each Merlin compilation boundary (design §8.1).

Off unless ``MERLIN_LEAN_STAGES`` names a directory. Each recorded boundary becomes
``<dir>/<seq>_<boundary>/{before.mlir,after.mlir,meta.json}`` with SHA-256s of both texts. Texts are
GENERIC-form prints of the module (the form MLIR parses back without custom-printer skew). Recording
only prints the module; it never changes it, so builds are byte-identical with recording on or off.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

ENV = "MERLIN_LEAN_STAGES"


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


def record_boundary(boundary: str, pass_entry: str, before: str, after: str, *, extra: dict | None = None) -> Path:
    """Write one boundary's before/after texts; returns the boundary directory."""
    root = _dir()
    root.mkdir(parents=True, exist_ok=True)
    seq = sum(1 for p in root.iterdir() if p.is_dir())
    d = root / f"{seq:04d}_{boundary}"
    d.mkdir()
    (d / "before.mlir").write_text(before)
    (d / "after.mlir").write_text(after)
    meta = {
        "schema": "merlin.lean.stage.v1",
        "boundary": boundary,
        "pass": pass_entry,
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "changed": before != after,
        **({"extra": extra} if extra else {}),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return d
