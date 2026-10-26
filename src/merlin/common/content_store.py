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
* **A reused object is checked before it is handed out.** Trusting the file name would let a
  corrupted or truncated object silently become a run's input, and that run's own digest would then
  certify the wrong bytes as the frozen treatment -- a result attributed to inputs it never ran on.
  The explicit snapshot lease opt-in may reuse its digest under a freshly checked Linux read lease;
  unsupported objects and ordinary callers retain full hashing. Placement closes its final lease
  check before returning, and later snapshot checks bind the same actual inode independently.

``MERLIN_BUNDLE_CAS`` relocates the store (a hard link cannot cross a filesystem, so point it at the
volume holding an out-of-tree workspace); setting it **empty** disables sharing and every consumer
falls back to a read-only copy with the same executable classification, which is the escape hatch
if a filesystem ever reports links it does not honor. Store or hard-link unavailability degrades
to a copy, so the store is an optimization and never a dependency.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable, Collection, Iterator
from pathlib import Path

from merlin.common import read_lease_digest
from merlin.common.paths import artifacts_dir

# ``shutil.copytree``'s ignore contract: (directory, entry names) -> names to skip.
IgnoreRule = Callable[[str, list[str]], Collection[str]]

NAMESPACE = "bundle-inputs-cas"
LOCATION_ENV = "MERLIN_BUNDLE_CAS"
_CHUNK = 8 * 1024 * 1024
_READ_LEASES = contextvars.ContextVar("content_store_read_leases", default=True)


@contextlib.contextmanager
def without_read_leases() -> Iterator[None]:
    """Digest store objects in full within this scope; never take a read lease or a CPU slot for one.

    A lease's custody recheck leases a shared compute slot. A caller that already holds its slot
    while it waits on this placement -- a grade capturing its compiler source in a worker -- would
    otherwise queue behind every other such caller for each leased file: measured, 20 grading
    threads each holding a slot and waiting on workers that waited for a slot, one capsule in 15
    minutes. Full hashing of a bounded tree costs milliseconds and needs no slot.
    """
    token = _READ_LEASES.set(False)
    try:
        yield
    finally:
        _READ_LEASES.reset(token)


def default_root() -> Path:
    """Where the store lives when the operator has not moved it."""
    return artifacts_dir() / "cache" / NAMESPACE


def store_root(namespace: str | None = None) -> Path | None:
    """The store to use, or ``None`` when sharing is disabled and consumers must copy.

    ``namespace`` selects a sibling store on the same volume for a consumer whose links must not
    land on another consumer's inodes. Frozen run inputs are watched by inode: every hard link a
    different consumer adds or removes on the same bytes changes their link count, which the
    snapshot guard rightly treats as an event (link counts are part of its aliasing checks). A
    consumer that links the same bytes very often (one compiler-source capture per test case)
    therefore keeps its own objects, so the run's frozen inputs stay quiet.
    """
    override = os.environ.get(LOCATION_ENV)
    if override is not None:
        if not override.strip():
            return None
        root = Path(override)
    else:
        root = default_root()
    if namespace is not None:
        if not namespace or "/" in namespace or namespace in {".", ".."}:
            raise ValueError("content store namespace must be a plain name")
        root = root.with_name(f"{root.name}.{namespace}")
    if override is None or namespace is not None:
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
    """Validate an original store object before handing its identity to a consumer."""
    if not _READ_LEASES.get():
        return _object_for(root, source)
    with read_lease_digest.verification():
        return _object_for(root, source)


def _object_digest(path: Path) -> str:
    # Only the public store object participates: source reads and private copy
    # policy remain unchanged. Canonical aliases may reuse the same leased inode.
    if _READ_LEASES.get() and read_lease_digest.enabled():
        canonical = path.absolute()
        if canonical.resolve() == canonical and canonical.stat().st_size >= read_lease_digest.MIN_BYTES:
            selected = read_lease_digest.digest(canonical)
            if selected is not None:
                return selected
    return digest_file(path)[0]


def _object_for(root: Path, source: Path) -> Path | None:
    """The store object holding ``source``'s bytes, created on first sight. ``None`` if unavailable.

    The key is the content AND whether the file is executable, because a mode is a property of the
    inode and two grants sharing one would fight over it -- a granted compiler or script has to stay
    runnable inside a sandbox, and a plain file with the same bytes must not become runnable.

    A reused object is byte-verified before it is handed out. Trusting the file name would let a
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
            if obj.is_symlink():
                return None  # never repair modes through a foreign cache alias
            if obj.is_file():
                stat = obj.stat()
                if stat.st_size == size and _object_digest(obj) == sha:
                    # Cleanup of a read-only consumer can chmod a shared inode.
                    # Revalidate bytes first, then restore the store's typed mode
                    # before handing out another link (or trusting old aliases).
                    if stat.st_mode & 0o7777 != mode:
                        obj.chmod(mode)
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


def _observe_source(
    current: Path, canonical: Path, out: Path, observe: Callable[[Path, Path, Path], None] | None
) -> None:
    """Report link-expansion spellings without assigning any ownership policy."""
    if observe is None:
        return
    spelling = current.absolute()
    for _ in range(41):
        observe(spelling, canonical, out)
        prefix = Path(spelling.anchor)
        for index, part in enumerate(spelling.parts[1:], start=1):
            prefix /= part
            if prefix.is_symlink():
                # Preserve the link's physical ownership spelling too. Do not
                # collapse '..' before earlier links have been interpreted:
                # lexical normpath and filesystem resolution differ there.
                observe(prefix.parent.resolve(strict=True) / prefix.name, canonical, out)
                target = Path(os.readlink(prefix))
                if not target.is_absolute():
                    target = prefix.parent / target
                spelling = target.joinpath(*spelling.parts[index + 1 :]).absolute()
                break
        else:
            if spelling.resolve(strict=True) != canonical:
                raise RuntimeError(f"source ownership changed while freezing {current}")
            return
    raise RuntimeError(f"refusing to freeze a symlink cycle at {current}")


def place_file(
    source: Path,
    dst: Path,
    root: Path | None,
    *,
    observe: Callable[[Path, Path, Path], None] | None = None,
    preserve_permissions: bool = False,
) -> bool:
    """Freeze a file and finish any selected immutable-object checks before returning."""
    if not _READ_LEASES.get():
        return _place_file(source, dst, root, observe=observe, preserve_permissions=preserve_permissions)
    with read_lease_digest.verification():
        return _place_file(source, dst, root, observe=observe, preserve_permissions=preserve_permissions)


def _place_file(
    source: Path,
    dst: Path,
    root: Path | None,
    *,
    observe: Callable[[Path, Path, Path], None] | None = None,
    preserve_permissions: bool = False,
) -> bool:
    """Put ``source``'s bytes at ``dst``. Returns whether the store's inode is now shared.

    Falls back to a copy whenever the link cannot be made -- no store, a cross-filesystem
    destination (``EXDEV``), or a filesystem without hard links. The result has identical bytes
    and, by default, mode 0444 or 0555 when the source is executable; only its disk cost differs.
    The caller owns destination paths and any existing regular file it allows to be overwritten.
    A destination symlink refuses before placement; this helper grants no ownership over other
    hard links to an existing destination or aliases in its parent path.

    Permission-preserving inputs bypass the store and retain the source's read
    and execute bits, never adding group/other access to an owner-only input.
    Write and special bits are always removed. This policy is explicit: merely
    disabling the store does not change public input permissions.
    """
    if preserve_permissions and root is not None:
        raise ValueError("permission-preserving inputs cannot use the shared content store")
    if dst.is_symlink():
        raise ValueError(f"refusing to freeze destination symlink: {dst}")
    if preserve_permissions and is_shared(dst):
        raise ValueError(f"refusing to overwrite a shared permission-preserving destination: {dst}")
    if observe is not None:
        canonical = source.resolve(strict=True)
        _observe_source(source, canonical, dst, observe)
        source = canonical
    obj = object_for(root, source) if root is not None else None
    if obj is not None:
        try:
            os.link(obj, dst)
            return True
        except OSError:
            pass
    source_mode = source.stat().st_mode
    mode = source_mode & 0o555 if preserve_permissions else (0o555 if source_mode & 0o111 else 0o444)
    shutil.copy2(source, dst, follow_symlinks=True)
    dst.chmod(mode)
    return False


def place_tree(
    source: Path,
    dst: Path,
    root: Path | None,
    *,
    observe: Callable[[Path, Path, Path], None] | None = None,
    preserve_permissions: bool = False,
    ignore: IgnoreRule | None = None,
) -> None:
    """Deep-freeze a directory, dereferencing symlinks.

    Links are followed so an absolute link into an external checkout cannot remain a live escape
    from the frozen tree; directory identity is tracked so a link pointing back up cannot make that
    an unbounded walk. A dangling link raises, and so does a non-regular file, because silently
    omitting either would make the frozen tree smaller than what was declared.
    ``observe(lexical_source, canonical_source, destination)`` sees every copied
    node, including files served from an existing CAS object. Ownership belongs
    to this invocation's source, never to the first writer of identical bytes.
    A node reached through links can produce multiple observations: each lexical
    link expansion retains its ownership spelling, all with the final canonical
    source. The observer cannot change copy/cache policy. ``preserve_permissions``
    applies the independent, read/execute-only policy to files and directories.
    That opt-in tree requires new destination directories and never follows a
    pre-existing destination alias. Default directory modes remain caller-owned.
    ``ignore`` is ``shutil.copytree``'s rule: entry names it returns for a directory are not frozen.
    """
    if preserve_permissions and root is not None:
        raise ValueError("permission-preserving inputs cannot use the shared content store")

    def walk(current: Path, out: Path, ancestry: tuple[tuple[int, int], ...], canonical: Path) -> None:
        if preserve_permissions and (out.exists() or out.is_symlink()):
            raise ValueError(f"permission-preserving tree destination must be new: {out}")
        stat = canonical.stat()  # follows links; a dangling one raises
        key = (stat.st_dev, stat.st_ino)
        # A cycle is a directory that contains ITSELF, which is what makes the walk unbounded. Two
        # distinct links to one shared directory are not that -- a tree may legitimately declare the
        # same corpus twice -- so the check is against this branch's ancestry, not everything seen.
        if key in ancestry:
            raise RuntimeError(f"refusing to freeze a symlink cycle at {current}")
        _observe_source(current, canonical, out, observe)
        out.mkdir(parents=True, exist_ok=True, mode=0o700 if preserve_permissions else 0o777)
        entries = sorted(canonical.iterdir())
        skipped = set(ignore(str(canonical), [entry.name for entry in entries])) if ignore is not None else set()
        for entry in entries:
            if entry.name in skipped:
                continue
            target = out / entry.name
            if entry.is_dir():  # follows links, as a dereferencing copy would
                walk(current / entry.name, target, ancestry + (key,), entry.resolve(strict=True))
            elif entry.is_file():
                actual = entry.resolve(strict=True)
                _observe_source(current / entry.name, actual, target, observe)
                place_file(actual, target, root, preserve_permissions=preserve_permissions)
            else:
                # A device node, a socket, a fifo -- or a link to none of the above. Freezing the
                # tree smaller than what was declared is the failure to avoid, so say so and stop.
                raise RuntimeError(
                    f"refusing to freeze {entry}: not a regular file or directory "
                    f"(a dangling symlink, or a device/socket/fifo)"
                )
        shutil.copystat(canonical, out)
        if preserve_permissions:
            out.chmod(stat.st_mode & 0o555)

    walk(source, dst, (), source.resolve(strict=True))


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


def adopt(path: Path, root: Path | None, *, before_replace: Callable[[], None] | None = None) -> int:
    """Adopt a caller-owned file, with final selected store custody before returning."""
    with read_lease_digest.verification():
        return _adopt(path, root, before_replace=before_replace)


def _adopt(path: Path, root: Path | None, *, before_replace: Callable[[], None] | None = None) -> int:
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
    ``before_replace`` lets a storage owner recheck its current liveness/authority
    after any potentially long object copy and immediately before replacing a name.
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
        if before_replace is not None:
            before_replace()
        os.replace(link, path)
    except OSError:
        return 0
    finally:
        link.unlink(missing_ok=True)
    # Freed only if this name was the last one holding the old inode. When it was not, the bytes go
    # when the other names are adopted too, and counting them here would report the saving twice.
    return before.st_size if before.st_nlink == 1 else 0
