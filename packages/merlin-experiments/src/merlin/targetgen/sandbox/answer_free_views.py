"""Answer-free hard-link mirrors that serve frozen grants without their exposed answer surfaces.

:func:`merlin.targetgen.sandbox.bwrap.apply_final_answer_masks` calls :func:`answer_free_views` before
the mask pass, so a release corpus with an answer file in every capsule directory does not cost one
bwrap overlay per surface. The mirrors live beside the bundle snapshot and are removed with it
(:func:`merlin.targetgen.sandbox.bwrap.remove_bundle_snapshot`). Each view records the frozen source it
stands in for beside itself, so the mask pass and the coverage proof name the answers it serves from
the argv alone (:func:`view_twins`).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
from merlin.targetgen.sandbox.bwrap import (
    _SNAPSHOT_COMPLETE,
    _VIEW_DIR,
    _make_snapshot_writable,
    _MountVisibility,
    bundle_snapshot_root,
)

from .mount_visibility import _DEVNULL, _EXPOSE_OPS, _HIDE_DEST_OPS, _mounts, _skipped_try, _UnsettledPath, path_kind

_VIEW_RECORD_SUFFIX = ".source"


def answer_free_views(argv: list[str], surfaces: list[AnswerSurface], ws: Path) -> list[str]:
    """Serve each frozen grant that holds exposed answer surfaces from a mirror without them.

    One overlay per surface does not scale: bwrap caps its whole argument list, ``--args`` files
    included, at 9,000 (``MAX_ARGS``), and a release corpus carries an answer file in every capsule
    directory beside the files the agent must read. The snapshot is host-owned, so the grant is
    instead served from a hard-link mirror of its frozen tree that omits those surfaces: they are
    absent from the agent's view rather than overlaid, at no argument per surface.

    Returns the rewritten argv. The mask pass and the coverage proof read each view's source back from
    it (:func:`view_twins`), so every surface below a mirrored source is still checked where the view
    serves it. A surface that a later mount needs as its mount point stays in the mirror and keeps its
    overlay. A source covered by a surface, or holding one that declares a public subtree, is never
    mirrored: its projection depends on the source path itself. A bind that reaches view storage in
    any other way is hidden or refused (:func:`_hide_host_storage`).
    """
    storage = ws.parent / _VIEW_DIR
    return _hide_host_storage(_answer_free_argv(argv, surfaces, ws, storage), storage, bundle_snapshot_root(ws))


def _answer_free_argv(argv: list[str], surfaces: list[AnswerSurface], ws: Path, storage: Path) -> list[str]:
    root = bundle_snapshot_root(ws).absolute()
    marker = root / _SNAPSHOT_COMPLETE
    if marker.is_symlink() or not marker.is_file():
        return argv
    mounts = _mounts(argv)
    visibility = _MountVisibility(argv, mounts)
    created = [argv[i + 1] for i in range(len(argv) - 1) if argv[i] in ("--dir", "--chdir")]
    mount_points = {
        path for dest in [*created, *(d for _, _, d in mounts)] for path in (Path(dest), *Path(dest).parents)
    }
    destinations: dict[Path, list[Path]] = {}
    for state, src, dest in mounts:
        if state == "expose":
            destinations.setdefault(Path(src), []).append(Path(dest))
    # The agent names of frozen surfaces: the snapshot holds no symlinks, so its binds project
    # lexically. A surface spelled any other way is not mirrored and keeps its overlay.
    projected = list(surfaces)
    for surface in surfaces:
        for source in surface.path.parents:
            if source in destinations and source.is_relative_to(root):
                relative = surface.path.relative_to(source)
                projected += [replace(surface, path=dest / relative) for dest in destinations[source]]
    # Only a directory surface can cover a source or hold a public subtree; check both spellings.
    spelled = [
        (surface, path)
        for surface in projected
        if surface.kind == "dir"
        for path in {surface.path.absolute(), surface.path.resolve()}
    ]
    covering = {path for _, path in spelled}
    fixed = {ancestor for surface, path in spelled if surface.grantable for ancestor in (path, *path.parents)}
    blocked: dict[Path, bool] = {}
    omitted: dict[Path, set[Path]] = {}
    for surface in projected:
        if surface.grantable:
            continue
        try:
            served = visibility.serving(surface.path)
        except _UnsettledPath:
            continue
        if served is None or served[0] == served[1] or not served[0].is_relative_to(root):
            continue
        source, frozen = served
        if source not in blocked:
            spellings = {source, source.resolve()}
            blocked[source] = any(
                spelling in fixed or any(path in covering for path in (spelling, *spelling.parents))
                for spelling in spellings
            )
        if blocked[source]:
            continue
        relative = frozen.relative_to(source)
        if path_kind(frozen) == "missing" or any(dest / relative in mount_points for dest in destinations[source]):
            continue
        omitted.setdefault(source, set()).add(relative)
    if not omitted:
        return argv
    salt = hashlib.sha256(marker.read_bytes()).hexdigest()
    views = {source: _answer_free_view(source, names, storage, salt) for source, names in omitted.items()}
    out = list(argv)
    i = 0
    while i < len(out):  # the same walk as _mounts, so exactly the parsed binds are rewritten
        if out[i] in _EXPOSE_OPS and i + 2 < len(out):
            if out[i + 1] != _DEVNULL and not _skipped_try(out[i], out[i + 1]) and Path(out[i + 1]) in views:
                out[i + 1] = str(views[Path(out[i + 1])])
            i += 3
        elif (out[i] in _HIDE_DEST_OPS or out[i] in ("--dev", "--proc")) and i + 1 < len(out):
            i += 2
        else:
            i += 1
    return out


def _view_sources(mounts: list[tuple[str, str, str]]) -> dict[Path, Path]:
    """The answer-free views these binds serve, each with the frozen source it stands in for.

    A view records what it was built from beside itself, never inside, so the agent does not see the
    record. Masks and proofs read it back from the argv alone: a caller holding only the frozen surfaces
    still reaches the bytes a rewritten bind serves, wherever that bind puts them. A record counts only
    when it reproduces the view's name; a view without one is not recorded.
    """
    sources: dict[Path, Path] = {}
    for state, src, _ in mounts:
        view = Path(src).absolute()
        if state != "expose" or view.parent.name != _VIEW_DIR:
            continue
        record = view.parent / f"{view.name}{_VIEW_RECORD_SUFFIX}"
        if record.is_symlink() or not record.is_file():
            continue
        try:
            built = json.loads(record.read_text())
            if _view_name(built["salt"], built["source"], built["omitted"]) == view.name:
                sources[view] = Path(built["source"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sources


def _view_name(salt: str, source: str, omitted: list[str]) -> str:
    return hashlib.sha256("\0".join([salt, source, *sorted(omitted)]).encode()).hexdigest()[:32]


def view_twins(mounts: list[tuple[str, str, str]], surfaces: list[AnswerSurface]) -> list[AnswerSurface]:
    """Every surface below a recorded view's source, named inside the view that serves the source's bytes.

    A twin projects through the view's bind to the same agent path the frozen bind exposed, so an
    answer a view keeps (a later mount point) or one that reaches it later is still masked and proven.
    A directory surface that is or holds the source denies the whole view, less the subtrees it
    declares public there. Both spellings of a surface and of a source are compared: one surface
    spelled through a symlink is not omitted from the view, so it is masked through its twin instead.
    """
    sources = {view: {source, source.resolve()} for view, source in _view_sources(mounts).items()}
    if not sources:
        return []
    twins: dict[Path, AnswerSurface] = {}
    for surface in surfaces:
        for path in {surface.path.absolute(), surface.path.resolve()}:
            for view, spellings in sources.items():
                for source in spellings:
                    if path.is_relative_to(source) and path != source:
                        twin = replace(surface, path=view / path.relative_to(source))
                    elif surface.kind == "dir" and source.is_relative_to(path):
                        inner = source.relative_to(path)
                        public = [Path(sub) for sub in surface.grantable if sub]
                        if any(inner.is_relative_to(sub) for sub in public):
                            continue  # the source lies in a subtree this surface declares public
                        kept = tuple(str(sub.relative_to(inner)) for sub in public if sub.is_relative_to(inner))
                        twin = replace(surface, path=view, grantable=kept)
                    else:
                        continue
                    twins.setdefault(twin.path, twin)
    return list(twins.values())


def _hide_host_storage(argv: list[str], storage: Path, snapshot: Path) -> list[str]:
    """Hide the run's host storage wherever a bind of a directory above it serves it.

    A provider may alias the run directory, which serves the frozen snapshot and view storage beside
    the workspace. Storage holds views of other commands, kept for mount points this argv does not
    have, and staging trees, which no surface names; a snapshot served whole there brings back one
    overlay per answer, past bwrap's argument cap. Each such bind is followed at once by a tmpfs over
    the storage it serves: bwrap serves a path from the last operation on it or an ancestor, so a
    later bind there still takes effect. A bind of view storage itself is hidden the same way; a bind
    of anything inside it but a whole recorded view is refused.
    """
    recorded = {spelling for view in _view_sources(_mounts(argv)) for spelling in (view, view.resolve())}
    stores = [(storage, False), *([(snapshot, True)] if snapshot.is_dir() else [])]  # a tmpfs needs its mount point
    roots = [(root, kept) for path, kept in stores for root in {path.absolute(), path.resolve()}]
    out: list[str] = []
    hidden = False
    i = 0
    while i < len(argv):  # the same walk as _mounts, so exactly the parsed binds are followed
        if argv[i] in _EXPOSE_OPS and i + 2 < len(argv):
            out += argv[i : i + 3]
            if argv[i + 1] != _DEVNULL:  # a skipped *-try bind serves nothing, but may not name storage either
                served = not _skipped_try(argv[i], argv[i + 1])
                source = Path(argv[i + 1]).absolute()
                aliases: list[str] = []
                for spelling in {source, source.resolve()}:
                    for root, kept in roots:
                        if served and root.is_relative_to(spelling) and not (kept and root == spelling):
                            aliases.append(str(Path(argv[i + 2]) / root.relative_to(spelling)))
                        elif not kept and spelling.is_relative_to(root) and spelling not in recorded:
                            raise RuntimeError(f"bind {argv[i + 1]} reaches inside answer-free view storage {storage}")
                for alias in dict.fromkeys(aliases):
                    out += ["--tmpfs", alias]
                    hidden = True
            i += 3
        elif (argv[i] in _HIDE_DEST_OPS or argv[i] in ("--dev", "--proc")) and i + 1 < len(argv):
            out += argv[i : i + 2]
            i += 2
        else:
            out.append(argv[i])
            i += 1
    if hidden:
        storage.mkdir(mode=0o700, exist_ok=True)  # a tmpfs needs its mount point; views may come later
    return out


def _answer_free_view(source: Path, omitted: set[Path], views: Path, salt: str) -> Path:
    """A read-only hard-link mirror of one frozen directory without ``omitted`` (relative paths).

    The name binds the snapshot marker, the source and the omitted set, so a view is built once per
    run and policy; a link the filesystem refuses becomes a read-only copy of the same bytes. The
    source is recorded beside the view before the view is published (:func:`_view_sources`).
    """
    built = {"salt": salt, "source": str(source.absolute()), "omitted": sorted(p.as_posix() for p in omitted)}
    view = views / _view_name(built["salt"], built["source"], built["omitted"])
    record = views / f"{view.name}{_VIEW_RECORD_SUFFIX}"
    if view.is_dir() and not view.is_symlink() and record.is_file() and not record.is_symlink():
        return view
    views.mkdir(mode=0o700, exist_ok=True)
    # A private staging directory per build, beside the view so publishing is a same-directory rename:
    # concurrent builders never share or remove each other's tree or record.
    pending = Path(tempfile.mkdtemp(prefix=f".{view.name}.", suffix=".pending", dir=views))
    staged_record = pending.with_suffix(".record")
    modes: list[tuple[Path, int]] = []

    def fail(error: OSError) -> None:
        raise error

    try:
        for directory, dirs, files in os.walk(source, onerror=fail):
            here = Path(directory)
            relative = here.relative_to(source)
            target = pending / relative
            if relative.parts:  # mkdtemp made the root
                target.mkdir(mode=0o700)
            modes.append((target, here.stat().st_mode & 0o7555))
            dirs[:] = [name for name in dirs if relative / name not in omitted]
            for name in [*dirs, *files]:
                if (here / name).is_symlink():
                    raise RuntimeError(f"bundle input snapshot contains a symlink: {here / name}")
            for name in files:
                if relative / name in omitted:
                    continue
                try:
                    os.link(here / name, target / name)
                except OSError:
                    shutil.copy2(here / name, target / name)
                    (target / name).chmod((target / name).stat().st_mode & ~0o222)
        for path, mode in reversed(modes):
            path.chmod(mode)
        staged_record.write_text(json.dumps(built))
        os.replace(staged_record, record)
        if not (view.is_dir() and not view.is_symlink()):
            pending.rename(view)  # same parent: a read-only directory can be renamed in place
    except BaseException:
        if not (view.is_dir() and not view.is_symlink() and record.is_file()):
            raise
        # a concurrent caller published the same view first
    finally:
        staged_record.unlink(missing_ok=True)
        if pending.exists():
            _make_snapshot_writable(pending)
            shutil.rmtree(pending)
    return view
