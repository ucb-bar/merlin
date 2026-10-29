"""Infer a variant tree's own change against a seed default and apply it.

Variants are full-tree copies of the default taken at some fork time, with
file modification times preserved by the copy.  A variant's *own* change is
the set of files it wrote after its fork: added files, and differing files
whose modification time is after the fork.  A differing file older than the
fork is the default's past version that the default has since moved on from,
so it is not part of the change.

The fork time is the variant directory's birth time (``stat %W``).  When the
filesystem does not record it, the change is ``UNKNOWN`` and the variant can
only be listed as a candidate.  Applying a change to the default is clean when
the default's file is unchanged since the fork; otherwise a three-way merge
(``diff3 -m``) against the newest copy of that file older than the fork is
attempted, and a conflicting merge is refused.
"""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

IGNORED_PARTS = frozenset({"__pycache__", "docs", "evidence", "selfcheck_out"})
IGNORED_NAMES = frozenset({"REPORT.md"})


def _tracked_files(tree: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in tree.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(tree)
        if IGNORED_PARTS.intersection(rel.parts) or rel.name in IGNORED_NAMES:
            continue
        if rel.suffix == ".pyc":
            continue
        out[rel.as_posix()] = path
    return out


def birth_time(path: Path) -> float | None:
    """Birth time of ``path`` from ``stat``; ``None`` when the filesystem lacks it."""
    try:
        proc = subprocess.run(["stat", "-c", "%W", str(path)], capture_output=True,
                              text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    value = proc.stdout.strip()
    if proc.returncode != 0 or not value.isdigit() or int(value) == 0:
        return None
    return float(value)


@dataclass
class FileChange:
    path: str
    kind: str  # added | edited | merged | conflict | deleted_unapplied
    note: str = ""


@dataclass
class VariantChange:
    name: str
    root: str
    fork_time: float | None
    changes: list[FileChange] = field(default_factory=list)
    status: str = "UNKNOWN"  # no_change | applicable | conflict | UNKNOWN

    @property
    def files(self) -> list[str]:
        return [c.path for c in self.changes if c.kind in ("added", "edited", "merged")]

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "root": self.root,
            "fork_time": self.fork_time,
            "status": self.status,
            "changes": [c.__dict__ for c in self.changes],
        }


class BaseIndex:
    """Every copy of each relative file across a set of trees, by modification time."""

    def __init__(self, trees: list[Path]):
        self._copies: dict[str, list[tuple[float, Path]]] = {}
        for tree in trees:
            for rel, path in _tracked_files(tree).items():
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                self._copies.setdefault(rel, []).append((mtime, path))

    def newest_before(self, rel: str, when: float) -> Path | None:
        best: tuple[float, Path] | None = None
        for mtime, path in self._copies.get(rel, []):
            if mtime <= when and (best is None or mtime > best[0]):
                best = (mtime, path)
        return best[1] if best else None


def _merge3(mine: Path, base: Path, theirs: Path) -> str | None:
    """``diff3 -m`` merge text, or ``None`` on conflict or failure."""
    try:
        proc = subprocess.run(["diff3", "-m", str(mine), str(base), str(theirs)],
                              capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


_MISSING = object()


def _merge_values(mine, base, theirs):
    """Key-level three-way merge of JSON values; raises ``ValueError`` on conflict."""
    if theirs == base:
        return mine
    if mine == base or mine == theirs:
        return theirs
    if all(isinstance(v, dict) for v in (mine, base, theirs)):
        out = dict(mine)
        for key in set(base) | set(theirs):
            m, b, t = mine.get(key, _MISSING), base.get(key, _MISSING), theirs.get(key, _MISSING)
            value = _merge_values(m, b, t)
            if value is _MISSING:
                out.pop(key, None)
            else:
                out[key] = value
        return out
    raise ValueError("conflict")


def _merge_json(mine: Path, base: Path, theirs: Path) -> str | None:
    try:
        values = [json.loads(p.read_text(encoding="utf-8")) for p in (mine, base, theirs)]
        merged = _merge_values(*values)
    except (OSError, ValueError):
        return None
    return json.dumps(merged, indent=2) + "\n"


def infer_change(variant: Path, default: Path, index: BaseIndex) -> tuple[VariantChange, dict[str, str]]:
    """The variant's own change and the merged text for files needing a three-way merge."""
    fork = birth_time(variant)
    result = VariantChange(name=variant.name, root=str(variant), fork_time=fork)
    merged: dict[str, str] = {}
    if fork is None:
        return result, merged
    vfiles = _tracked_files(variant)
    dfiles = _tracked_files(default)
    for rel, vpath in sorted(vfiles.items()):
        vtime = vpath.stat().st_mtime
        dpath = dfiles.get(rel)
        if dpath is None:
            if vtime > fork:
                result.changes.append(FileChange(rel, "added"))
            continue
        if filecmp.cmp(vpath, dpath, shallow=False) or vtime <= fork:
            continue
        base = index.newest_before(rel, fork)
        if base is None:
            kind = "edited" if dpath.stat().st_mtime <= fork else "conflict"
            result.changes.append(FileChange(rel, kind, "no copy older than the fork"))
            continue
        if filecmp.cmp(dpath, base, shallow=False):
            result.changes.append(FileChange(rel, "edited", f"base {base}"))
            continue
        merger = _merge_json if rel.endswith(".json") else _merge3
        text = merger(dpath, base, vpath)
        if text is None:
            result.changes.append(FileChange(rel, "conflict", "default and variant both edited it"))
        else:
            merged[rel] = text
            result.changes.append(FileChange(rel, "merged", f"base {base}"))
    for rel, dpath in sorted(dfiles.items()):
        if rel not in vfiles and dpath.stat().st_mtime <= fork:
            result.changes.append(FileChange(rel, "deleted_unapplied", "variant removed it"))
    if any(c.kind == "conflict" for c in result.changes):
        result.status = "conflict"
    elif result.files:
        result.status = "applicable"
    else:
        result.status = "no_change"
    return result, merged


def _writable(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            continue
        try:
            os.chmod(path, path.stat().st_mode | 0o200 | (0o100 if path.is_dir() else 0))
        except OSError:
            pass


def materialize(default: Path, out: Path, changes: list[tuple[VariantChange, dict[str, str]]]) -> list[str]:
    """Copy ``default`` to ``out`` and apply each change in order.

    Returns the names of the changes that applied.  A change that touches a
    file an earlier applied change already wrote is skipped rather than merged
    blindly, so the result never mixes two edits of one file unchecked.
    """
    if out.exists():
        _writable(out)
        shutil.rmtree(out)
    shutil.copytree(default, out, symlinks=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
    # Frozen trees are often read-only; the promoted copy is a new, editable tree.
    _writable(out)
    written: set[str] = set()
    applied: list[str] = []
    for change, merged in changes:
        files = set(change.files)
        if change.status != "applicable" or files & written:
            continue
        for rel in sorted(files):
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            if rel in merged:
                target.write_text(merged[rel], encoding="utf-8")
            else:
                shutil.copy2(Path(change.root) / rel, target)
                os.chmod(target, target.stat().st_mode | 0o200)
        written |= files
        applied.append(change.name)
    return applied


def variant_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                  and (p / "manifest.yaml").is_file())


def discover(variants_root: Path, default: Path, extra_bases: list[Path]) -> list[tuple[VariantChange, dict[str, str]]]:
    trees = variant_dirs(variants_root)
    index = BaseIndex([default, *trees, *extra_bases])
    return [infer_change(tree, default, index) for tree in trees]


def owned_files_changed(change: VariantChange) -> list[str]:
    """Top-level author module stems the change touches (for family selection)."""
    stems: list[str] = []
    for rel in change.files:
        head = rel.partition("/")[0]
        stems.append(os.path.splitext(head)[0])
    return stems
