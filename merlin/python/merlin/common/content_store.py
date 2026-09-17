"""One copy of each distinct byte-string, hard-linked wherever it is needed.

Several places in this repo freeze a tree so that the bytes some long-running work reads cannot
change under it: an agent run's declared input closure, a resumable performance suite's source
snapshot. The freeze has to be a copy of the *bytes* -- a link to the live file would not survive an
in-place edit of the source, which is the whole risk being defended against -- but it does not have
to be a copy *per consumer*, and it was. A campaign whose runs all granted the same toolchain and the
same model weights wrote those identical bytes once per run: 12.8 GB per run, 235 GiB across one
campaign, for inputs that were byte-identical every time.

This module holds the bytes once, at mode 0444 (0555 when the file is executable), under a store
keyed by content; a consumer gets a **hard link** to the store object. The immutability property is
unchanged, because the store holds its own copy: truncating the source in place mutates the source's
inode and cannot reach the store's. What changes is that the second consumer of the same content adds
a directory entry and no bytes.

Two properties make the store safe to keep under the declared-purgeable cache root:

* **A link is a claim.** The bytes live for as long as any name references them, so removing a store
  object with one remaining link cannot take data from a live snapshot. ``orphans()`` finds exactly
  those, which is what ``merlin-storage prune`` acts on.
* **A reused object is re-digested before it is handed out.** Trusting the file name would let a
  corrupted or truncated object silently become a run's input, and that run's own digest would then
  certify the wrong bytes as the frozen treatment -- a result attributed to inputs it never ran on.

``MERLIN_BUNDLE_CAS`` relocates the store (a hard link cannot cross a filesystem, so point it at the
volume holding an out-of-tree workspace); setting it **empty** disables sharing and every consumer
falls back to a plain copy, which is the escape hatch if a filesystem ever reports links it does not
honor. Every entry point degrades to a copy rather than failing, so the store is an optimization and
never a dependency.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable, Collection
from pathlib import Path

from merlin.common.paths import artifacts_dir

# ``shutil.copytree``'s ignore contract: (directory, entry names) -> names to skip.
IgnoreRule = Callable[[str, list[str]], Collection[str]]

NAMESPACE = "bundle-inputs-cas"
LOCATION_ENV = "MERLIN_BUNDLE_CAS"
_CHUNK = 8 * 1024 * 1024


def default_root() -> Path:
    """Where the store lives when the operator has not moved it."""
    return artifacts_dir() / "cache" / NAMESPACE


def store_root() -> Path | None:
    """The store to use, or ``None`` when sharing is disabled and consumers must copy."""
    override = os.environ.get(LOCATION_ENV)
    if override is not None:
        return Path(override) if override.strip() else None
    root = default_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def digest_file(path: Path) -> tuple[str, int]:
    """``(sha256, size)`` of the bytes actually read, following links."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def object_for(root: Path, source: Path) -> Path | None:
    """The store object holding ``source``'s bytes, created on first sight. ``None`` if unavailable.

    The key is the content AND whether the file is executable, because a mode is a property of the
    inode and two grants sharing one would fight over it -- a granted compiler or script has to stay
    runnable inside a sandbox, and a plain file with the same bytes must not become runnable.

    A reused object is RE-DIGESTED before it is handed out. Trusting the file name would let a
    corrupted or truncated object silently become a consumer's input, and that consumer's own seal
    would then certify the wrong bytes as the frozen treatment -- a result attributed to inputs it
    never ran on, which is the failure the freeze exists to prevent.

    Creation is create-if-absent, not replace. ~13 sessions share this host, so several campaigns
    materializing the same grant at the same moment is the normal case: an unconditional rename let
    each racing writer overwrite the previous winner's directory entry AFTER that winner had already
    linked its own inode, so a burst of N first-time runs ended on N separate inodes holding
    identical bytes -- correct, and the whole saving lost. ``os.link`` onto the final name fails with
    ``EEXIST`` instead, which is how a loser learns to use the winner's object.
    """
    sha, size = digest_file(source)
    runnable = bool(source.stat().st_mode & 0o111)
    obj = root / sha[:2] / (f"{sha}.x" if runnable else sha)
    mode = 0o555 if runnable else 0o444
    try:
        for _ in range(4):
            if obj.is_file():
                stat = obj.stat()
                if stat.st_size == size and digest_file(obj)[0] == sha:
                    return obj
                # Present but not the bytes it claims. Drop it and let this pass recreate it; a
                # concurrent writer doing the same thing is harmless, since both write the source.
                obj.chmod(stat.st_mode | 0o200)
                obj.unlink(missing_ok=True)
            obj.parent.mkdir(parents=True, exist_ok=True)
            # Stage under a name no other writer can hold, so nothing can link a half-written file.
            handle, staged = tempfile.mkstemp(dir=obj.parent, prefix=f"{obj.name}.", suffix=".pending")
            os.close(handle)
            pending = Path(staged)
            try:
                shutil.copyfile(source, pending, follow_symlinks=True)
                pending.chmod(mode)
                os.link(pending, obj)  # atomic create-if-absent
            except FileExistsError:
                continue  # another writer won; the loop re-validates ITS object
            finally:
                pending.unlink(missing_ok=True)  # the bytes live on via obj once the link is made
            return obj
        return None  # lost the race four times: copy rather than spin
    except OSError:
        return None


def place_file(source: Path, dst: Path, root: Path | None) -> bool:
    """Put ``source``'s bytes at ``dst``. Returns whether the store's inode is now shared.

    Falls back to a copy whenever the link cannot be made -- no store, a cross-filesystem
    destination (``EXDEV``), or a filesystem without hard links. The result is byte-identical
    either way; only its disk cost differs.
    """
    obj = object_for(root, source) if root is not None else None
    if obj is not None:
        try:
            os.link(obj, dst)
            return True
        except OSError:
            pass
    shutil.copy2(source, dst, follow_symlinks=True)
    return False


def place_tree(source: Path, dst: Path, root: Path | None) -> None:
    """Deep-freeze a directory, dereferencing symlinks.

    Links are followed so an absolute link into an external checkout cannot remain a live escape
    from the frozen tree; directory identity is tracked so a link pointing back up cannot make that
    an unbounded walk. A dangling link raises, and so does a non-regular file, because silently
    omitting either would make the frozen tree smaller than what was declared.
    """

    def walk(current: Path, out: Path, ancestry: tuple[tuple[int, int], ...]) -> None:
        stat = current.stat()  # follows links; a dangling one raises
        key = (stat.st_dev, stat.st_ino)
        # A cycle is a directory that contains ITSELF, which is what makes the walk unbounded. Two
        # distinct links to one shared directory are not that -- a tree may legitimately declare the
        # same corpus twice -- so the check is against this branch's ancestry, not everything seen.
        if key in ancestry:
            raise RuntimeError(f"refusing to freeze a symlink cycle at {current}")
        out.mkdir(parents=True, exist_ok=True)
        for entry in sorted(current.iterdir()):
            target = out / entry.name
            if entry.is_dir():  # follows links, as a dereferencing copy would
                walk(entry, target, ancestry + (key,))
            elif entry.is_file():
                place_file(entry, target, root)
            else:
                # A device node, a socket, a fifo -- or a link to none of the above. Freezing the
                # tree smaller than what was declared is the failure to avoid, so say so and stop.
                raise RuntimeError(
                    f"refusing to freeze {entry}: not a regular file or directory "
                    f"(a dangling symlink, or a device/socket/fifo)"
                )
        shutil.copystat(current, out)

    walk(source, dst, ())


def is_shared(path: Path) -> bool:
    """True if other names reference these bytes, so this path's mode is not ours to change.

    A store-backed entry is already read-only and is shared with every other consumer of the same
    content. chmod follows the inode, so changing it here would reach into those other consumers'
    frozen trees. Unlinking needs write permission on the DIRECTORY, not on the file, so leaving
    these alone costs a caller nothing.
    """
    try:
        return not path.is_symlink() and path.stat(follow_symlinks=False).st_nlink > 1
    except OSError:
        return False


def orphans(root: Path | None = None) -> tuple[list[Path], int]:
    """Store objects with exactly one link: nothing references them, so nothing loses them."""
    root = root if root is not None else default_root()
    found: list[Path] = []
    total = 0
    if not root.is_dir():
        return found, total
    for path in root.rglob("*"):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat(follow_symlinks=False)
        except OSError:
            continue
        if stat.st_nlink == 1:
            found.append(path)
            total += stat.st_size
    return found, total


def adopt(path: Path, root: Path | None) -> int:
    """Re-point ``path`` at the store object holding its bytes. Returns the bytes this freed.

    ``place_file`` gets the saving at the moment a tree is frozen. This gets it afterwards, for the
    trees that were written before the store existed: the bytes are already on disk under some name,
    and the copy that made them a duplicate has already been paid for. Digest the file, put it in the
    store if it is not there yet, and swap the directory entry for a link to the object -- after
    which every other name holding those bytes that gets adopted costs nothing.

    The swap is ``os.replace`` onto a link staged in the same directory, so the name never disappears
    and a reader that already has the file open keeps reading the inode it opened. Nothing is removed
    if the link cannot be made; the original entry stays exactly as it was.

    **Adoption makes the file read-only, and its mode is then shared with every other name for those
    bytes.** That is correct for a frozen tree and wrong for anything a later step rewrites in place,
    so the caller decides what is eligible -- ``merlin-storage dedup`` refuses any tree whose own
    integrity check reads the file mode, because chmod follows the inode and would reach into the
    other holders' trees. Returns 0 when nothing changed, including when ``path`` already IS the
    store's inode.
    """
    if root is None or path.is_symlink() or not path.is_file():
        return 0
    try:
        before = path.stat(follow_symlinks=False)
    except OSError:
        return 0
    obj = object_for(root, path)
    if obj is None:
        return 0
    try:
        held = obj.stat(follow_symlinks=False)
    except OSError:
        return 0
    if (held.st_dev, held.st_ino) == (before.st_dev, before.st_ino):
        return 0  # already the store's inode: this name is the saving
    try:
        handle, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".adopt")
    except OSError:
        return 0  # no write permission on the directory; leave it alone
    os.close(handle)
    link = Path(staged)
    try:
        link.unlink()
        os.link(obj, link)
        os.replace(link, path)
    except OSError:
        link.unlink(missing_ok=True)
        return 0
    # Freed only if this name was the last one holding the old inode. When it was not, the bytes go
    # when the other names are adopted too, and counting them here would report the saving twice.
    return before.st_size if before.st_nlink == 1 else 0
