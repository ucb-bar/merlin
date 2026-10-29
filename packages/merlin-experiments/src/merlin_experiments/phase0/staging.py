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

import contextvars
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

STAGING_PREFIX = ".staging-"
_PUBLICATION_LOCK = threading.Lock()


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
        with _PUBLICATION_LOCK:
            _link_existing(stage, out_root, member)
        written = write(stage)
        for created in _corpus_members(stage):
            check_publication(operator_exclusion, created)
        with _PUBLICATION_LOCK:
            moved = promote(stage, out_root)
        if isinstance(written, (str, Path)) and written and Path(written).is_relative_to(stage):
            return moved.get(Path(written), out_root / Path(written).relative_to(stage))
        return written
    finally:
        shutil.rmtree(stage, ignore_errors=True)


class PreparedWrite:
    """One unpublished source write, finished once by the ordered generation caller."""

    def __init__(self, stage, out_root, member, operator_exclusion, written, finish):
        self.stage, self.out_root, self.member = stage, out_root, member
        self.operator_exclusion, self.written, self._finish = operator_exclusion, written, finish
        self._context, self._pid, self._closed = contextvars.copy_context(), os.getpid(), False

    def close(self):
        """Discard only this private attempt; completed or failed publications are untouched."""
        if os.getpid() != self._pid:
            raise ValueError("prepared corpus write belongs to another process")
        if not self._closed:
            self._closed = True
            shutil.rmtree(self.stage, ignore_errors=True)

    def finish(self):
        """Refresh siblings, recheck, qualify and publish after all earlier model writes."""
        from .operator_exclusion import check_publication, require_selection

        if os.getpid() != self._pid or self._closed:
            raise ValueError("prepared corpus write is closed or belongs to another process")
        try:
            require_selection(self.operator_exclusion, self.out_root)
            with _PUBLICATION_LOCK:
                _link_existing(self.stage, self.out_root, self.member)
            written = self._context.run(self._finish, self.written, self.stage)
            for created in _corpus_members(self.stage):
                check_publication(self.operator_exclusion, created)
            with _PUBLICATION_LOCK:
                moved = promote(self.stage, self.out_root)
            if isinstance(written, (str, Path)) and written and Path(written).is_relative_to(self.stage):
                return moved.get(Path(written), self.out_root / Path(written).relative_to(self.stage))
            return written
        finally:
            self.close()


def prepare_staged(write, finish, out_root, *, member, operator_exclusion=None) -> PreparedWrite:
    """Capture into a private stage without selecting a stale shared-sibling view.

    The source callback may write only its own member. The unchanged finishing
    callback runs later, under this exact observed context, on the ordered caller.
    Nothing becomes a corpus member until finishing and publication both succeed.
    """
    from .operator_exclusion import require_selection

    out_root = Path(out_root)
    require_selection(operator_exclusion, out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=out_root))
    try:
        written = write(stage)
        if any(created.relative_to(stage).as_posix() != member for created in _corpus_members(stage)):
            raise ValueError("a prepared source write created a sibling member")
        return PreparedWrite(stage, out_root, member, operator_exclusion, written, finish)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
