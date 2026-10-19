"""Write one corpus member in a staging directory and move it into the corpus only when it succeeds.

A writer that fails part-way used to leave its half-written member in the corpus: a ``capsule.yaml``
and a ``golden.yaml`` that MANIFEST.yaml does not list, which the next ``rglob`` picks up as though
it had been built. Writing under a private staging root and renaming each finished member into place
makes a failed (or killed) write leave nothing at the member's corpus path.

The stage is a view of the corpus. Every member already written is linked into it, so a writer
that consults its siblings -- a qualified model reuses a per-group capsule another model already
wrote -- sees and reuses the real one exactly as before, and only directories it newly creates are
promoted. The member being written is not linked: a rerun into an existing corpus writes it afresh
and replaces the previous copy only once the new one is complete.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

STAGING_PREFIX = ".staging-"


def _corpus_members(root: Path) -> list[Path]:
    out = []
    for category in sorted(root.iterdir()):
        if not category.is_dir() or category.is_symlink() or category.name.startswith(STAGING_PREFIX):
            continue
        out += [m for m in sorted(category.iterdir()) if m.is_dir() and not m.is_symlink()]
    return out


def _link_existing(stage: Path, out_root: Path, own: str | None) -> None:
    for member in _corpus_members(out_root):
        relative = member.relative_to(out_root)
        if relative.as_posix() == own:
            continue
        link = stage / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(member.resolve(), target_is_directory=True)


def promote(stage: Path, out_root: Path) -> dict[Path, Path]:
    """Move every member the write created (not the linked ones) into ``out_root``."""
    moved: dict[Path, Path] = {}
    retired = stage / ".replaced"
    for member in _corpus_members(stage):
        destination = out_root / member.relative_to(stage)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            # A rerun's fresh copy of this member: the old one is set aside, then dropped with the stage.
            aside = retired / member.relative_to(stage)
            aside.parent.mkdir(parents=True, exist_ok=True)
            destination.rename(aside)
        member.rename(destination)
        moved[member] = destination
    return moved


def write_staged(
    write: Callable[[Path], object], out_root: Path | str, *, member: str | None = None, operator_exclusion=None
):
    """Run ``write(stage_root)`` and promote what it created; a failure leaves the corpus untouched.

    ``member`` is the ``<category>/<name>`` being written, which is not linked into the stage.
    Returns what ``write`` returned, with a staged member path mapped to its final corpus path.
    """
    out_root = Path(out_root)
    from .operator_exclusion import check_publication, require_selection

    require_selection(operator_exclusion, out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=out_root))
    try:
        _link_existing(stage, out_root, member)
        written = write(stage)
        for created in _corpus_members(stage):
            check_publication(operator_exclusion, created)
        moved = promote(stage, out_root)
        if isinstance(written, (str, Path)) and written and Path(written).is_relative_to(stage):
            return moved.get(Path(written), out_root / Path(written).relative_to(stage))
        return written
    finally:
        shutil.rmtree(stage, ignore_errors=True)
