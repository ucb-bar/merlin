"""SHA-256 helpers: the one definition behind merlin's content hashes and digest-shape checks.

Measured 2026-09-14: about sixty private helpers across compare/, perf/, runtime/, frontends/, targetgen/
and plotting/ re-implemented four operations under five names (``_sha``, ``_sha256``, ``_digest``,
``_is_sha``, ``_is_sha256``), and the NAME did not say which: ``_digest`` hashed a JSON value in most
modules, a file in one, and returned a bool (a digest-SHAPE check) in two others. Where the bodies
overlapped they agreed byte for byte; the hazard was the naming, so each operation now has one name.

``is_sha256`` is strict by default -- a ``str`` of exactly 64 lowercase hex characters -- because a digest
compared as data must be spelled one way. ``allow_upper=True`` exists for the two readers that accepted
either case; nothing else should need it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# ---- digest ------------------------------------------------------------------------------------

HEX_LOWER = frozenset("0123456789abcdef")
SHA256_HEX_LEN = 64
_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Lowercase hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Lowercase hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: "str | Path") -> str:
    """Lowercase hex SHA-256 of a file's bytes, streamed. Raises ``OSError`` when it cannot be read --
    a digest that silently stood in for a missing file would certify nothing."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object, *, allow_upper: bool = False) -> bool:
    """True when ``value`` is a ``str`` of exactly 64 hex characters (lowercase unless ``allow_upper``)."""
    if not isinstance(value, str) or len(value) != SHA256_HEX_LEN:
        return False
    text = value.lower() if allow_upper else value
    return all(ch in HEX_LOWER for ch in text)


# ---- jsonio ------------------------------------------------------------------------------------
# Canonical JSON bytes and JSON file writers: one definition for every content-addressed document.
#
# ``canonical_json`` is the byte string merlin hashes: keys sorted, no insignificant whitespace, non-ASCII
# escaped, and NaN/Infinity REFUSED -- JSON cannot represent them, so ``json.dumps`` would emit a
# non-standard token that another reader rejects or re-hashes differently. Measured 2026-09-14: ten modules
# defined their own ``_canonical_sha`` and eleven a ``_canonical``/``_canonical_json``. The ASCII and UTF-8
# encoders among them produce identical bytes on every input (``json.dumps`` escapes non-ASCII by default),
# so they collapse here with no change to any persisted digest. The genuinely different contracts keep an
# explicit flag: ``ensure_ascii=False`` (raw UTF-8), ``trailing_newline=True`` (a newline-terminated file
# body) and ``allow_nan=True`` (the lax spelling a caller used before, preserved rather than silently
# tightened).
#
# Named ``jsonio`` rather than ``json`` so it can never shadow the standard library.


def canonical_json(
    value: Any, *, ensure_ascii: bool = True, allow_nan: bool = False, trailing_newline: bool = False
) -> bytes:
    """Sorted-key, separator-minimal JSON bytes of ``value`` (see the module docstring for the flags)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ensure_ascii, allow_nan=allow_nan)
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def canonical_sha256(
    value: Any, *, ensure_ascii: bool = True, allow_nan: bool = False, trailing_newline: bool = False
) -> str:
    """SHA-256 of :func:`canonical_json` with the same flags."""
    return sha256_bytes(
        canonical_json(value, ensure_ascii=ensure_ascii, allow_nan=allow_nan, trailing_newline=trailing_newline)
    )


def _object_key(key: Any) -> str:
    """The member name ``json.dumps`` writes for ``key``."""
    if isinstance(key, str):
        return key
    if key is True or key is False or key is None:
        return json.dumps(key)
    if isinstance(key, (int, float)):
        return json.dumps(key)
    raise TypeError(f"keys must be str, int, float, bool or None, not {type(key).__name__}")


def _holds_no_object(items: list | tuple) -> bool:
    if any(isinstance(item, dict) for item in items):
        return False
    first = items[0] if items else None
    return not isinstance(first, (list, tuple)) or _holds_no_object(first)


def readable_json(value: Any, *, indent: int = 2) -> str:
    """``json.dumps(value, indent=indent)`` with every array of plain values on one line.

    ``indent`` puts each array element on its own line, so an embedded tensor is mostly indentation:
    a 65-million-element weight matrix in a capsule result took 1 GB indented and 229 MB without it,
    and that result was rewritten for every grade. Objects keep their indented layout for a human
    reader; arrays that hold no object are written compactly. ``json.loads`` of the result equals
    ``json.loads(json.dumps(value))``, so a reader or a digest of the parsed value sees no change.
    """
    pieces: list[str] = []

    def write(item: Any, level: int) -> None:
        if isinstance(item, dict):
            if not item:
                pieces.append("{}")
                return
            inner = "\n" + " " * (indent * (level + 1))
            pieces.append("{")
            for index, (key, member) in enumerate(item.items()):
                pieces.append(("," if index else "") + inner + json.dumps(_object_key(key)) + ": ")
                write(member, level + 1)
            pieces.append("\n" + " " * (indent * level) + "}")
        elif isinstance(item, (list, tuple)):
            if _holds_no_object(item):
                pieces.append(json.dumps(item, separators=(",", ":")))
                return
            inner = "\n" + " " * (indent * (level + 1))
            pieces.append("[")
            for index, member in enumerate(item):
                pieces.append(("," if index else "") + inner)
                write(member, level + 1)
            pieces.append("\n" + " " * (indent * level) + "]")
        else:
            pieces.append(json.dumps(item))

    write(value, 0)
    return "".join(pieces)


def strict_json_equal(observed: object, expected: object) -> bool:
    """Compare closed JSON values without Python's bool/int or tuple/list aliases.

    Floats, unknown containers and non-string object keys are deliberately not
    authority-bearing values under this comparator.
    """
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        return (
            all(type(key) is str for key in observed)
            and observed.keys() == expected.keys()
            and all(strict_json_equal(observed[key], value) for key, value in expected.items())
        )
    if type(expected) is list:
        return len(observed) == len(expected) and all(
            strict_json_equal(item, value) for item, value in zip(observed, expected, strict=True)
        )
    return type(expected) in (str, int, bool, type(None)) and observed == expected


def write_canonical_json(path: str | Path, value: Any) -> None:
    """Write ``value`` as strict canonical JSON plus a trailing newline, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def write_pretty_json(path: str | Path, value: Any, *, mkdir: bool = False) -> None:
    """Write ``value`` indented (2), key-sorted, newline-terminated, for files a human reads."""
    path = Path(path)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---- tree_hash ---------------------------------------------------------------------------------
# Content identity for source trees, independent of research harness installation.

_SKIP = {"build", "__pycache__", ".git"}


def hash_tree(root: Path) -> dict:
    """Hash relative source paths and bytes, excluding generated subtrees.

    The exclusion is relative to the tree being identified: a source checkout may
    itself live below out/build. Legacy digests for such locations excluded every
    file and must be recomputed; historical evidence is never rewritten.

    An unchanged tree's result is reused while every entry keeps its stat identity
    (see :mod:`merlin.common.digest`); a grade hashes the same frozen trees once
    per completed test case.
    """

    root = Path(root)
    return cached(
        "hash_tree",
        str(root.absolute()),
        lambda: tree_signature(root, skip_names=frozenset(_SKIP)),
        lambda: _hash_tree(root),
    )


def _hash_tree(root: Path) -> dict:
    if not root.exists():
        return {"present": False, "sha256": None, "n_files": 0}
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or _SKIP & set(relative.parts):
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        count += 1
    return {"present": True, "sha256": digest.hexdigest(), "n_files": count}


# ---- stat_memo ---------------------------------------------------------------------------------
# Reuse a value derived from files while every file it read keeps its stat identity.
#
# A grade re-derives the same identities from the same immutable inputs once per completed test case:
# the frozen compiler-source snapshot's tree hash, the selected FIRRTL file's digest, each submission
# file's bytes and import graph. Measured on a live grade, those repeated reads held most of the grade's
# interpreter time. A value here is reused only while the stat identity of every file it depends on is
# unchanged: any change to a file's bytes or metadata moves its ctime, which an unprivileged process
# cannot set. A file changed within :data:`SETTLED_NS` is never cached (file times advance in coarse
# ticks), a tree containing a symbolic link is never cached (its target is not covered by the link's
# stat), and every value is fully recomputed after :data:`MAX_AGE_S`.

MAX_AGE_S = 300.0
SETTLED_NS = 2_000_000_000
#: Trees larger than this are hashed directly; their signature would cost about as much as the hash.
MAX_TREE_ENTRIES = 200_000
_MAX_ENTRIES = 4096
_MAX_BYTES_HELD = 256 << 20

_CACHE: dict[tuple, tuple[Any, float, Any, int]] = {}
_HELD = 0
_LOCK = threading.Lock()
_DISABLED = os.environ.get("MERLIN_STAT_MEMO", "1").strip() == "0"


def _row(path: str, info: os.stat_result) -> tuple:
    return (path, info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def file_signature(path) -> tuple | None:
    """The stat identity of one regular file; ``None`` for a link, a non-file or an unreadable path."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (_row(os.fspath(path), info),)


def tree_signature(root, *, skip_names: frozenset = frozenset()) -> tuple | None:
    """The stat identity of ``root`` and of every entry below it; ``None`` if any entry is a link,
    unreadable, or the tree is larger than :data:`MAX_TREE_ENTRIES`.

    Entries named in ``skip_names`` (and everything below them) are left out, for a value that by
    definition never reads them."""
    root = os.fspath(root)
    rows = []
    try:
        top = os.lstat(root)
        if stat.S_ISLNK(top.st_mode):
            return None
        rows.append(_row(root, top))
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(name for name in dirnames if name not in skip_names)
            for name in [*dirnames, *sorted(name for name in filenames if name not in skip_names)]:
                path = os.path.join(directory, name)
                info = os.lstat(path)
                if stat.S_ISLNK(info.st_mode):
                    return None
                rows.append(_row(path, info))
                if len(rows) > MAX_TREE_ENTRIES:
                    return None
    except OSError:
        return None
    return tuple(rows)


def _settled(signature: tuple) -> bool:
    horizon = time.time_ns() - SETTLED_NS
    return all(row[5] < horizon and row[6] < horizon for row in signature)


def _weight(value: Any) -> int:
    return len(value) if isinstance(value, (bytes, bytearray)) else 0


def cached(kind: str, key: Any, signature: Callable[[], tuple | None], compute: Callable[[], Any]) -> Any:
    """``compute()``, reused while ``signature()`` is unchanged since it was computed.

    The signature is taken before and after computing; the value is kept only when both agree and every
    row is settled, so a file written during the computation is never cached with stale content.
    Exceptions propagate and are never cached. Mutable values are returned as copies.
    """
    global _HELD
    if _DISABLED:
        return compute()
    entry_key = (kind, key)
    before = signature()
    now = time.monotonic()
    if before is not None:
        with _LOCK:
            entry = _CACHE.get(entry_key)
        if entry is not None and entry[0] == before and now - entry[1] < MAX_AGE_S:
            value = entry[2]
            return value if isinstance(value, (bytes, str, int, float, tuple, frozenset)) else copy.deepcopy(value)
    value = compute()
    if before is not None and _settled(before) and signature() == before:
        weight = _weight(value)
        with _LOCK:
            old = _CACHE.pop(entry_key, None)
            if old is not None:
                _HELD -= old[3]
            if len(_CACHE) >= _MAX_ENTRIES or _HELD + weight > _MAX_BYTES_HELD:
                _CACHE.clear()
                _HELD = 0
            stored = value if isinstance(value, (bytes, str, int, float, tuple, frozenset)) else copy.deepcopy(value)
            _CACHE[entry_key] = (before, now, stored, weight)
            _HELD += weight
    return value


def clear() -> None:
    global _HELD
    with _LOCK:
        _CACHE.clear()
        _HELD = 0


# ---- source_membership -------------------------------------------------------------------------
# Standard-library Python source membership; callers own hashes and run receipts.
#
# Generated bytecode caches are omitted. Linked/nonregular entries are refused even
# when they are not Python sources, matching the existing native source policy.


class SourceMembershipError(ValueError):
    """A source tree cannot be inventoried without following unsafe entries."""


def python_members(directory: Path, *, label: str = "Python") -> dict[str, Path]:
    """Return sorted relative Python names and their ordinary resolved file paths."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise SourceMembershipError(f"{label} source directory is absent or symlinked: {directory}")
    inputs = {}
    for member in sorted(directory.rglob("*")):
        relative = member.relative_to(directory)
        if "__pycache__" in relative.parts:
            continue
        if member.is_symlink() or not (member.is_file() or member.is_dir()):
            raise SourceMembershipError(f"{label} implementation contains symlinked or nonregular entries: {member}")
        if member.is_file() and member.suffix == ".py":
            inputs[relative.as_posix()] = member.resolve()
    return inputs
