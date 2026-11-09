"""Where an author keeps its cross-round notes: the workspace root, outside the graded compiler.

The notes used to live at ``submission/docs/iteration_notes.md``. Everything under ``submission/``
is the compiler's identity (only ``build``, ``__pycache__`` and ``.git`` are ignored,
:data:`.run_inputs.SNAPSHOT_IGNORED`), so every note edit -- which every brief asks for, every round --
moved the graded source digest: each reconciled grade row became an "older version" and the verdict's
``snapshot_current`` went false until the next grade finished.

The notes now live at ``ITERATION_NOTES.md`` in the workspace root, next to the experiment's other
author records. Excluding ``submission/docs/`` from the identity instead was rejected: compiler code can
open files there (the docs directory also holds agent-written JSON proofs), so excluding it would let
bytes a graded command can read escape the identity, and it would change every existing digest.

Readers fall back to the legacy location, so a seeded or older workspace keeps its notes.
"""

from __future__ import annotations

from pathlib import Path

#: The notes file, relative to the workspace root.
NOTES_NAME = "ITERATION_NOTES.md"
#: Where notes lived before; read only as a fallback.
LEGACY_NOTES = Path("submission") / "docs" / "iteration_notes.md"


def notes_path(workspace) -> Path:
    """The file an author writes its notes to."""
    return Path(workspace) / NOTES_NAME


def existing_notes(workspace) -> Path | None:
    """The notes file to read: the root file, else the legacy one; ``None`` when neither exists."""
    for path in (notes_path(workspace), Path(workspace) / LEGACY_NOTES):
        if path.is_file():
            return path
    return None


def read_notes(workspace) -> str:
    """The author's notes text (``""`` when there are none)."""
    path = existing_notes(workspace)
    return path.read_text(encoding="utf-8", errors="replace") if path is not None else ""
