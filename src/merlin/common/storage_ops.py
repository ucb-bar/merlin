"""Disk operations on trees the generated-output convention does not own: move, link, census.

``merlin-storage report``/``prune``/``dedup`` reason about ``out/``. A host also carries checkouts,
worktrees, scratch trees and second disks, and reclaiming space there was done with throwaway shell
scripts whose safety rested on whoever ran them remembering every check. This module is those
operations with the checks built in:

* :func:`move` -- copy a tree with ``rsync``, verify the copy by CHECKSUM, then replace the source
  with a RELATIVE symlink to the copy, so every path that quoted the old location still resolves.
* :func:`link_duplicates` -- hard-link byte-identical files to one another (``merlin-storage dedup
  --peers``), for trees that cannot reach the content store because they live on another
  filesystem, or that the store should not own.
* :func:`worktrees` -- classify each worktree of a repository: clean or dirty, merged into a base
  ref or not, commits ahead, size, locked/prunable, and whether a live process holds it.

Every operation is a dry run unless the caller says ``apply``; none follows a symlink (walks use
``lstat``, ``rsync`` runs without ``-L``/``-K``/``--copy-links``); and the two that change the disk
refuse a path some live process holds open. That check is :func:`open_file_checker` -- ``lsof`` when
it is installed, else ``fuser`` over the walked entries -- and when neither tool exists the
operation is refused rather than assumed safe, unless the caller explicitly opts out.

A deny-list (``deny=``) names trees an operation must not read or touch at all. It is empty here and
configured by the caller: which trees belong to someone else is a property of the host, not of this
code.
"""

from __future__ import annotations

import bisect
import hashlib
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

__all__ = [
    "OpenFileCheckUnavailable",
    "checker_for",
    "denied",
    "link_duplicates",
    "move",
    "open_file_checker",
    "worktrees",
]

_CHUNK = 8 * 1024 * 1024
#: Arguments per ``fuser`` invocation when it stands in for ``lsof``.
_FUSER_BATCH = 256


class OpenFileCheckUnavailable(RuntimeError):
    """No tool on this host can say whether a live process holds a path open."""


# --- the open-file check ---------------------------------------------------------------------------


def _real(path: Path) -> str:
    """The path as ``lsof`` reports it: parents resolved, the last component NOT dereferenced."""
    path = Path(path)
    return os.path.join(os.path.realpath(path.parent), path.name)


class _LsofIndex:
    """Every open file on the host, from ONE ``lsof`` call, queried by path prefix.

    One listing instead of one ``lsof +D`` per path: a ``+D`` walk costs seconds each, and a
    de-duplication pass asks about thousands of files. An unprivileged user cannot see another
    account's open files anyway, so the listing is limited to this user's processes unless running as
    root -- measured on a busy host, 1.7 s instead of 22 s for the same answer.
    """

    tool = "lsof"

    def __init__(self, executable: str) -> None:
        mine = ["-u", str(os.geteuid())] if os.geteuid() != 0 else []
        try:
            done = subprocess.run(
                [executable, "-w", "-n", "-P", "-l", *mine, "-F", "pn"], capture_output=True, text=True, timeout=900
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OpenFileCheckUnavailable(f"lsof could not run: {exc}") from exc
        pid = None
        rows: dict[str, set[int]] = {}
        for line in done.stdout.splitlines():
            if line.startswith("p") and line[1:].isdigit():
                pid = int(line[1:])
            elif line.startswith("n") and pid is not None:
                rows.setdefault(line[1:], set()).add(pid)
        # lsof exits 1 for warnings it was told to suppress and for "nothing matched"; an empty
        # listing with a failing status is the case that means it did not run at all.
        if not rows and done.returncode != 0:
            raise OpenFileCheckUnavailable(f"lsof failed ({done.returncode}): {done.stderr.strip()[:200]}")
        self._names = sorted(rows)
        self._rows = rows

    def holders(self, path: Path) -> list[int]:
        real = _real(path)
        prefix = real.rstrip(os.sep) + os.sep
        pids: set[int] = set(self._rows.get(real, ()))
        start = bisect.bisect_left(self._names, prefix)
        for name in self._names[start:]:
            if not name.startswith(prefix):
                break
            pids |= self._rows[name]
        return sorted(pids)


class _FuserCheck:
    """``fuser`` over every entry of a tree, walked without following symlinks."""

    tool = "fuser"

    def __init__(self, executable: str) -> None:
        self._executable = executable

    def holders(self, path: Path) -> list[int]:
        # Absolute names: fuser takes no "--", so a relative name starting with "-" would be an option.
        names = [os.path.abspath(p) for p in _walk_entries(Path(path))]
        pids: set[int] = set()
        for start in range(0, len(names), _FUSER_BATCH):
            try:
                done = subprocess.run(
                    [self._executable, *names[start : start + _FUSER_BATCH]],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise OpenFileCheckUnavailable(f"fuser could not run: {exc}") from exc
            # fuser exits 0 when it found a holder and 1 when it found none; anything else is a failure,
            # and a failed check must not read as "nothing holds this".
            if done.returncode not in (0, 1):
                raise OpenFileCheckUnavailable(f"fuser exited {done.returncode}: {done.stderr.strip()[:300]}")
            # fuser prints the pids on stdout (access letters, if any, go to stderr with the names).
            pids |= {int(token) for token in done.stdout.split() if token.isdigit()}
        return sorted(pids)


def open_file_checker(*, which: Callable[[str], str | None] | None = None):
    """An object whose ``holders(path)`` lists the pids holding ``path`` or anything under it.

    Raises :class:`OpenFileCheckUnavailable` when neither ``lsof`` nor ``fuser`` is installed. A
    process this user cannot inspect (another account's, without privilege) is invisible to both;
    that limit is the tools', and it is why the check guards against our own sessions rather than
    claiming the host is quiet.
    """
    which = which or shutil.which
    lsof = which("lsof")
    if lsof:
        return _LsofIndex(lsof)
    fuser = which("fuser")
    if fuser:
        return _FuserCheck(fuser)
    raise OpenFileCheckUnavailable(
        "neither lsof nor fuser is installed, so live open files cannot be ruled out; "
        "install one, or pass --no-open-file-check to accept that risk explicitly"
    )


class _NoCheck:
    """The explicit opt-out: nothing is reported as held. Only ever built on the caller's request."""

    tool = "none (open-file check disabled)"

    def holders(self, path: Path) -> list[int]:  # noqa: ARG002 -- same interface as the real checks
        return []


def checker_for(enabled: bool):
    """The open-file check a command uses: the real one, or the explicit opt-out when ``enabled`` is
    False. Raises :class:`OpenFileCheckUnavailable` when enabled and no tool is installed."""
    return open_file_checker() if enabled else _NoCheck()


def _checker(check_open: bool, checker):
    return checker if checker is not None else checker_for(check_open)


# --- walking and the deny-list ---------------------------------------------------------------------


def denied(path: Path, deny: Iterable[Path | str]) -> Path | None:
    """The deny-list entry ``path`` is (or lies under), compared lexically -- never by reading it."""
    candidate = Path(os.path.abspath(path))
    for entry in deny:
        root = Path(os.path.abspath(entry))
        if candidate == root or root in candidate.parents:
            return root
    return None


def _walk_entries(root: Path) -> Iterator[Path]:
    """``root`` and everything under it, without following a symlink (a link is listed, not entered)."""
    yield root
    if root.is_symlink() or not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            yield Path(dirpath) / name


def _inventory(root: Path) -> dict:
    """Files, bytes and devices under ``root`` by ``lstat``: what a copy must reproduce."""
    files = links = size = 0
    devices: set[int] = set()
    for entry in _walk_entries(root):
        try:
            st = entry.lstat()
        except OSError:
            continue
        devices.add(st.st_dev)
        if stat.S_ISLNK(st.st_mode):
            links += 1
        elif stat.S_ISREG(st.st_mode):
            files += 1
            size += st.st_size
    return {"files": files, "symlinks": links, "bytes": size, "devices": len(devices)}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


# --- move ------------------------------------------------------------------------------------------


def _rsync_source(source: Path) -> str:
    # A trailing separator copies a directory's CONTENTS into the destination directory.
    return f"{source}{os.sep}" if source.is_dir() else str(source)


def _rsync_verify(rsync: str, source: Path, destination: Path) -> list[str]:
    """Names whose bytes or metadata differ between the trees, by a checksum dry run."""
    done = subprocess.run(
        [rsync, "-aHnc", "--out-format=%n", "--", _rsync_source(source), str(destination)],
        capture_output=True,
        text=True,
        timeout=None,
    )
    if done.returncode != 0:
        return [f"rsync verify failed ({done.returncode}): {done.stderr.strip()[:300]}"]
    # A directory line only reports a directory's timestamp; content differences are file lines.
    return [line for line in done.stdout.splitlines() if line and not line.endswith("/")]


def _make_removable(root: Path) -> None:
    """Give every DIRECTORY under ``root`` owner write, so its entries can be unlinked."""
    for entry in _walk_entries(root):
        try:
            st = entry.lstat()
        except OSError:
            continue
        if stat.S_ISDIR(st.st_mode) and not st.st_mode & stat.S_IWUSR:
            entry.chmod(stat.S_IMODE(st.st_mode) | stat.S_IWUSR | stat.S_IXUSR)


def move(
    source: Path,
    destination: Path,
    *,
    apply: bool = False,
    check_open: bool = True,
    deny: Sequence[Path | str] = (),
    protected: Callable[[Path], list[str]] | None = None,
    checker=None,
    rsync: str | None = None,
) -> dict:
    """Move ``source`` to ``destination`` and leave a relative symlink at the old name.

    The copy is verified by checksum before anything is removed, so a failure at any step leaves the
    source intact (a failed copy may leave a partial destination behind, which is reported). The open
    file check runs twice: before the copy, and again immediately before the source is removed, since
    a copy of a large tree takes long enough for a process to start using it.

    ``protected`` returns reasons a path must not move (leases, pins, tracked files), supplied by the
    caller that knows the policy. Returns a record whose ``status`` is ``planned``, ``moved``,
    ``refused`` or ``failed`` and whose ``reasons`` say why.
    """
    source, destination = Path(os.path.abspath(source)), Path(os.path.abspath(destination))
    record: dict = {"source": str(source), "destination": str(destination), "apply": apply, "reasons": []}

    def refuse(*reasons: str) -> dict:
        record["status"] = "refused"
        record["reasons"].extend(reasons)
        return record

    for path in (source, destination):
        hit = denied(path, deny)
        if hit is not None:
            return refuse(f"{path} is under the deny-list entry {hit}")
    try:
        st = source.lstat()
    except OSError as exc:
        return refuse(f"source unreadable: {exc}")
    if stat.S_ISLNK(st.st_mode):
        return refuse("source is a symlink (already moved, or not ours to follow)")
    if not (stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode)):
        return refuse("source is neither a directory nor a regular file")
    if os.path.lexists(destination):
        return refuse("destination already exists")
    if source == destination or source in destination.parents or destination in source.parents:
        return refuse("source and destination overlap")
    inventory = _inventory(source)
    record.update(inventory)
    if inventory["devices"] > 1:
        return refuse("source spans more than one filesystem (a mount point inside it)")
    if protected is not None:
        reasons = protected(source)
        if reasons:
            return refuse(*reasons)
    rsync = rsync or shutil.which("rsync")
    if rsync is None:
        return refuse("rsync is not installed")
    try:
        check = _checker(check_open, checker)
        record["open_file_check"] = check.tool
        holders = check.holders(source)
    except OpenFileCheckUnavailable as exc:
        return refuse(str(exc))
    if holders:
        return refuse(f"held open by pid(s) {', '.join(map(str, holders))}")
    if not apply:
        record["status"] = "planned"
        return record

    destination.parent.mkdir(parents=True, exist_ok=True)
    copied = subprocess.run(
        [rsync, "-aH", "--", _rsync_source(source), str(destination)], capture_output=True, text=True, timeout=None
    )
    record["status"] = "failed"
    if copied.returncode != 0:
        record["reasons"].append(f"copy failed ({copied.returncode}): {copied.stderr.strip()[:300]}; source kept")
        return record
    differing = _rsync_verify(rsync, source, destination)
    copy = _inventory(destination)
    if differing or any(copy[k] != inventory[k] for k in ("files", "symlinks", "bytes")):
        record["reasons"].append(
            f"verify failed ({len(differing)} differing, {copy} vs {inventory}); source kept, copy at destination"
        )
        record["differing"] = differing[:20]
        return record
    try:
        holders = check.holders(source)
    except OpenFileCheckUnavailable as exc:
        record["reasons"].append(f"{exc}; source kept, verified copy at destination")
        return record
    if holders:
        record["reasons"].append(
            f"held open by pid(s) {', '.join(map(str, holders))} after the copy; "
            "source kept, verified copy at destination"
        )
        return record
    if stat.S_ISDIR(st.st_mode):
        _make_removable(source)
        shutil.rmtree(source)
    else:
        source.unlink()
    target = os.path.relpath(destination, source.parent)
    source.symlink_to(target, target_is_directory=stat.S_ISDIR(st.st_mode))
    record.update(status="moved", symlink=target)
    return record


# --- linking duplicates to one another ---------------------------------------------------------------


def _writable_parent(path: Path) -> int | None:
    """Borrow owner write on ``path``'s directory; return the mode to put back (None: untouched)."""
    parent = path.parent
    mode = parent.lstat().st_mode
    if mode & stat.S_IWUSR:
        return None
    parent.chmod(stat.S_IMODE(mode) | stat.S_IWUSR)
    return stat.S_IMODE(mode)


def _replace_with_link(keep: Path, name: Path) -> None:
    """Make ``name`` another directory entry for ``keep``'s inode, atomically (stage, then rename)."""
    restore = _writable_parent(name)
    try:
        handle, staged = tempfile.mkstemp(dir=name.parent, prefix=f".{name.name}.", suffix=".link")
        os.close(handle)
        link = Path(staged)
        try:
            link.unlink()
            os.link(keep, link)
            os.replace(link, name)
        except OSError:
            link.unlink(missing_ok=True)
            raise
    finally:
        if restore is not None:
            name.parent.chmod(restore)


def link_duplicates(
    groups: Sequence[tuple[int, Sequence[Path]]],
    *,
    apply: bool = False,
    check_open: bool = True,
    deny: Sequence[Path | str] = (),
    tracked: Callable[[Path], bool] | None = None,
    checker=None,
) -> dict:
    """Hard-link every name in each content group to one inode per filesystem.

    ``groups`` are ``(size, names)`` whose names hold identical bytes (as
    ``storage_cli.dedup_candidates`` finds them). Unlike the store-based ``dedup``, no copy is
    placed anywhere: the first name on each device keeps its inode and the others become links to
    it, so this works on a filesystem the content store is not on. Each name is re-digested
    immediately before its swap, so a file rewritten since the scan is skipped, not linked.

    The kept inode is made read-only afterwards, because its mode is now shared: an in-place write
    through one name would reach every other. Directory modes are borrowed and restored exactly.
    Symlinks, tracked files (``tracked``), names under ``deny`` and names a live process holds are
    skipped and counted.
    """
    out = {"linked": 0, "released_bytes": 0, "skipped": [], "apply": apply}
    try:
        check = _checker(check_open, checker)
    except OpenFileCheckUnavailable as exc:
        out["refused"] = str(exc)
        return out
    out["open_file_check"] = check.tool
    for size, names in groups:
        by_device: dict[int, list[tuple[Path, os.stat_result]]] = {}
        for name in names:
            name = Path(name)
            try:
                st = name.lstat()
            except OSError as exc:
                out["skipped"].append((str(name), f"unreadable: {exc}"))
                continue
            if not stat.S_ISREG(st.st_mode):
                out["skipped"].append((str(name), "not a regular file"))
                continue
            if denied(name, deny) is not None:
                out["skipped"].append((str(name), "deny-list"))
                continue
            by_device.setdefault(st.st_dev, []).append((name, st))
        for members in by_device.values():
            if len(members) < 2:
                continue
            keep, keep_st = members[0]
            try:
                keep_digest = _digest(keep)
            except OSError as exc:
                out["skipped"].append((str(keep), f"unreadable: {exc}"))
                continue
            for name, st in members[1:]:
                if (st.st_dev, st.st_ino) == (keep_st.st_dev, keep_st.st_ino):
                    continue  # already one inode
                if tracked is not None and tracked(name):
                    out["skipped"].append((str(name), "tracked or uninspectable git state"))
                    continue
                try:
                    holders = check.holders(name)
                    same = _digest(name) == keep_digest
                except (OSError, OpenFileCheckUnavailable) as exc:
                    out["skipped"].append((str(name), str(exc)))
                    continue
                if holders:
                    out["skipped"].append((str(name), f"held open by pid(s) {holders}"))
                    continue
                if not same:
                    out["skipped"].append((str(name), "content changed since the scan"))
                    continue
                if apply:
                    try:
                        _replace_with_link(keep, name)
                    except OSError as exc:
                        out["skipped"].append((str(name), f"link failed: {exc}"))
                        continue
                out["linked"] += 1
                # Freed only when this name held the last link to its old inode.
                out["released_bytes"] += size if st.st_nlink == 1 else 0
            if apply:
                mode = stat.S_IMODE(keep.lstat().st_mode)
                if mode & 0o222:
                    keep.chmod(mode & ~0o222)
    return out


# --- worktree census -------------------------------------------------------------------------------


def _git(repo: Path, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    # --no-optional-locks: `git status` otherwise refreshes the index, which is a WRITE to a tree this
    # census promises only to read (and possibly another session's working tree).
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def _porcelain_worktrees(text: str) -> list[dict]:
    rows: list[dict] = []
    row: dict = {}
    for line in [*text.splitlines(), ""]:
        if not line:
            if row:
                rows.append(row)
            row = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            row["path"] = value
        elif key == "HEAD":
            row["head"] = value
        elif key == "branch":
            row["branch"] = value.removeprefix("refs/heads/")
        elif key in ("detached", "bare"):
            row[key] = True
        elif key in ("locked", "prunable"):
            row[key] = value or True
    return rows


def default_base(repo: Path) -> str | None:
    """The ref ``merged`` is judged against: the remote's HEAD, else ``origin/main``, else ``main``."""
    head = _git(repo, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD")
    if head.returncode == 0 and head.stdout.strip():
        return head.stdout.strip()
    for ref in ("origin/main", "main", "origin/master", "master"):
        if _git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode == 0:
            return ref
    return None


def _size(path: Path) -> int | None:
    du = shutil.which("du")
    if du is None:
        return None
    # -P: never follow a symlink; -x: stay on the worktree's filesystem; -b: apparent bytes.
    done = subprocess.run([du, "-sbPx", "--", str(path)], capture_output=True, text=True, timeout=3600)
    if done.returncode != 0:
        return None  # an unreadable entry makes the total a lower bound: report the size as unknown
    head = done.stdout.split(maxsplit=1)
    return int(head[0]) if head and head[0].isdigit() else None


def worktrees(
    repo: Path,
    *,
    base: str | None = None,
    deny: Sequence[Path | str] = (),
    sizes: bool = True,
    check_open: bool = True,
    checker=None,
) -> dict:
    """Classify every worktree of ``repo``. Read-only: nothing is created, locked or refreshed.

    Each row carries ``state`` (``clean``/``dirty``/``missing``/``denied``), the counts of modified
    tracked paths and untracked paths, ``merged`` into ``base`` and ``ahead`` of it, ``size`` in
    bytes, ``locked``/``prunable`` as git reports them, and ``holders`` -- the pids of live processes
    holding anything under it (``None`` when the open-file check is unavailable or disabled).
    A worktree under ``deny`` is listed by path and not read at all.
    """
    repo = Path(repo)
    listing = _git(repo, "worktree", "list", "--porcelain")
    if listing.returncode != 0:
        raise RuntimeError(f"not a git repository or worktree list failed: {listing.stderr.strip()[:300]}")
    base = base or default_base(repo)
    check = None
    open_check = "disabled"
    if check_open or checker is not None:
        try:
            check = _checker(True, checker)
            open_check = check.tool
        except OpenFileCheckUnavailable as exc:
            open_check = f"unavailable: {exc}"
    rows = []
    for row in _porcelain_worktrees(listing.stdout):
        path = Path(row["path"])
        row.setdefault("branch", None if row.get("detached") else row.get("branch"))
        if denied(path, deny) is not None:
            row["state"] = "denied"
            rows.append(row)
            continue
        if not path.is_dir():
            row["state"] = "missing"
            rows.append(row)
            continue
        status = _git(path, "status", "--porcelain=v1", "--untracked-files=normal")
        if status.returncode != 0:
            row["state"] = "unknown"
            row["error"] = status.stderr.strip()[:200]
        else:
            lines = status.stdout.splitlines()
            row["untracked"] = sum(1 for line in lines if line.startswith("??"))
            row["modified"] = len(lines) - row["untracked"]
            row["state"] = "dirty" if lines else "clean"
        head = row.get("head")
        if base and head:
            row["merged"] = _git(repo, "merge-base", "--is-ancestor", head, base).returncode == 0
            ahead = _git(repo, "rev-list", "--count", f"{base}..{head}")
            row["ahead"] = (
                int(ahead.stdout.strip()) if ahead.returncode == 0 and ahead.stdout.strip().isdigit() else None
            )
        row["size"] = _size(path) if sizes else None
        try:
            row["holders"] = check.holders(path) if check is not None else None
        except OpenFileCheckUnavailable:
            row["holders"] = None
        rows.append(row)
    return {"repo": str(repo), "base": base, "open_file_check": open_check, "worktrees": rows}
