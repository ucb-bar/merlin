"""Fresh capture issuer for a fully snapshotted static executable.

The only payload mounts are private source/runtime snapshots (read-only) and a
fresh capture directory (writable). Python, shared libraries, virtualenvs, and
model2MLIR loaders are deliberately ineligible for this issuer.
"""

from __future__ import annotations

import hashlib
import errno
import json
import os
import secrets
import shutil
import stat
import struct
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.common import strict_json

SCHEMA = "merlin.capture_execution_attestation.v1"
ISSUER = "merlin.sealed-static-capture.v1"
_FLAGS = (
    "--unshare-all",
    "--unshare-user",
    "--disable-userns",
    "--new-session",
    "--die-with-parent",
    "--clearenv",
    "--setenv",
    "HOME",
    "/no-home",
    "--setenv",
    "XDG_CACHE_HOME",
    "/no-cache",
    "--setenv",
    "PATH",
    "/runtime/bin",
    "--dir",
    "/runtime",
    "--dir",
    "/source",
    "--dir",
    "/capture-out",
    "--chdir",
    "/",
)


class SealedCaptureError(ValueError):
    """The capture cannot earn, or no longer retains, a sealed-execution claim."""


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_path(value: Path, *, exists: bool) -> Path:
    """Refuse indirect ancestors before copy or destination creation."""
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if ".." in path.parts:
        raise SealedCaptureError(f"path is not canonical: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise SealedCaptureError(f"path has a symlink component: {current}")
    if exists and not path.exists():
        raise SealedCaptureError(f"required path is absent: {path}")
    return path


def _require_link_directory(component: Path, link: Path) -> None:
    """Refuse resolution dependencies on any other link, including the target."""
    try:
        mode = component.lstat().st_mode
    except FileNotFoundError as exc:
        raise SealedCaptureError(f"dangling directory link: {link}") from exc
    except OSError as exc:
        raise SealedCaptureError(f"unresolvable directory link: {link}") from exc
    if stat.S_ISLNK(mode):
        raise SealedCaptureError(f"unresolvable directory link has a symlink component: {link}: {component}")
    if not stat.S_ISDIR(mode):
        raise SealedCaptureError(f"non-directory symlink target component: {link}: {component}")


def _directory_symlink(path: Path, root: Path) -> dict[str, str]:
    """Validate one exact relative link text against real directories in its tree."""
    target = os.readlink(path)
    if Path(target).is_absolute():
        raise SealedCaptureError(f"absolute directory link is unsupported: {path}")
    relative = path.parent.relative_to(root) / target
    lexical_target = Path(os.path.normpath(relative))
    if ".." in lexical_target.parts:
        raise SealedCaptureError(f"directory link points outside selected tree: {path}")
    current = root
    _require_link_directory(current, path)
    # Check even components cancelled by '..': the kernel would traverse those
    # directories too, so normalizing away another link could change the referent.
    for part in relative.parts:
        if part == "..":
            if current == root:
                raise SealedCaptureError(f"directory link points outside selected tree: {path}")
            current = current.parent
        else:
            current = current / part
        _require_link_directory(current, path)
    return {"kind": "symlink", "target": target}


def _dangling_symlink(path: Path) -> dict[str, str] | None:
    """Legacy dangling-file policy, including its explicit directory refusal."""
    try:
        path.stat()
    except FileNotFoundError:
        target = os.readlink(path)
        if target.endswith("/") or target.rsplit("/", 1)[-1] in {".", ".."}:
            raise SealedCaptureError(f"dangling directory link: {path}")
        return {"kind": "dangling_symlink", "target": target}
    except OSError as exc:
        raise SealedCaptureError(f"unresolvable source link: {path}") from exc
    return None


_BASE_ROOT_XATTR = "user.merlin.base_root"


def _base_link_root(path: Path, root: Path) -> tuple[Path, Path]:
    """Recover the base boundary and its guest absolute name after relocation."""
    for parent in (path.parent, *path.parent.parents):
        if not parent.is_relative_to(root):
            break
        try:
            original = Path(os.fsdecode(os.getxattr(parent, _BASE_ROOT_XATTR, follow_symlinks=False)))
        except OSError as exc:
            if exc.errno not in (errno.ENODATA, errno.ENOTSUP):
                raise
        else:
            if not original.is_absolute() or ".." in original.parts:
                raise SealedCaptureError(f"invalid staged base boundary: {parent}")
            return parent, original
    return root, root


def _base_symlink(path: Path, root: Path) -> dict[str, str] | None:
    """Resolve internal hops only; external text is never a host lookup source.

    Absolute paths and '..' are processed before lstat, including components
    later cancelled by '..'. A repeated resolution state is a cycle.
    Internal file aliases retain the historical content-copy behavior.
    """
    root, guest_base = _base_link_root(path, root)
    target = os.readlink(path)
    pending = list(Path(target).parts)
    current = path.parent
    seen = set()
    hops = 0
    mode = root.lstat().st_mode
    while pending:
        state = (current, tuple(pending))
        if state in seen or hops > 256:
            raise SealedCaptureError(f"cyclic base link: {path}")
        seen.add(state)
        part = pending.pop(0)
        if part == "/":
            # An absolute target outside root is opaque, without even lstat.
            absolute = Path("/", *pending)
            if not absolute.is_relative_to(guest_base):
                return _external_link(target)
            pending = list(absolute.relative_to(guest_base).parts)
            current = root
            continue
        current = current.parent if part == ".." else current / part
        if not current.is_relative_to(root):
            return _external_link(target)
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            if target.endswith("/") or target.rsplit("/", 1)[-1] in {".", ".."}:
                raise SealedCaptureError(f"dangling directory link: {path}")
            return {"kind": "dangling_symlink", "target": target}
        except OSError as exc:
            raise SealedCaptureError(f"unresolvable base link: {path}") from exc
        if stat.S_ISLNK(mode):
            hops += 1
            pending = list(Path(os.readlink(current)).parts) + pending
            current = current.parent
        elif pending and not stat.S_ISDIR(mode):
            raise SealedCaptureError(f"non-directory base link component: {path}")
    if stat.S_ISDIR(mode):
        return {"kind": "symlink", "target": target}
    return None


def _external_link(target: str) -> dict[str, str]:
    # Kind plus exact text only: the guest's view of the referent is a property
    # of its namespace, never of this host, so nothing about it is recorded.
    return {"kind": "external_symlink", "target": target}


_UNREADABLE_XATTR_PREFIX = "user.merlin.unreadable."
_UNREADABLE_FIELDS = ("mode", "size", "uid", "gid")


def _unreadable_xattr(name: str) -> str:
    return _UNREADABLE_XATTR_PREFIX + _digest(os.fsencode(name))


def _unreadable_file(path: Path) -> dict[str, Any] | None:
    """Metadata of a regular file this process cannot read; never its bytes."""
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        return None
    if not info.st_mode & 0o444 or not os.access(path, os.R_OK, follow_symlinks=False):
        return {"kind": "unreadable_file", "mode": stat.S_IMODE(info.st_mode),
                "size": info.st_size, "uid": info.st_uid, "gid": info.st_gid}
    return None


def _omitted_members(directory: Path) -> dict[str, dict[str, Any]]:
    """Unreadable base files deliberately ABSENT from this staged directory.

    A staged/guest copy never materializes them (a guest read fails with
    ENOENT); the parent directory instead carries one marker per member so
    the copy reproduces the selected inventory rows exactly.
    """
    try:
        attributes = os.listxattr(directory, follow_symlinks=False)
    except OSError as exc:
        if exc.errno == errno.ENOTSUP:
            return {}
        raise
    rows: dict[str, dict[str, Any]] = {}
    for attribute in sorted(attributes):
        if not attribute.startswith(_UNREADABLE_XATTR_PREFIX):
            continue
        try:
            row = json.loads(os.getxattr(directory, attribute, follow_symlinks=False))
        except ValueError as exc:
            raise SealedCaptureError(f"invalid omitted unreadable member: {directory}") from exc
        name = row.get("name") if isinstance(row, dict) else None
        if (
            not isinstance(name, str) or name in {"", ".", ".."} or "/" in name
            or attribute != _unreadable_xattr(name)
            or set(row) != {"kind", "name", *_UNREADABLE_FIELDS} or row["kind"] != "unreadable_file"
            or any(type(row[key]) is not int or row[key] < 0 for key in _UNREADABLE_FIELDS)
        ):
            raise SealedCaptureError(f"invalid omitted unreadable member: {directory}")
        if os.path.lexists(directory / name):
            raise SealedCaptureError(f"omitted unreadable member is present: {directory / name}")
        rows[name] = {key: value for key, value in row.items() if key != "name"}
    return rows


def _present_members(rows: dict[str, dict[str, Any]]) -> list[str]:
    """Inventory names that exist on disk (omitted unreadable files do not)."""
    return [name for name, row in rows.items() if row.get("kind") != "unreadable_file"]


def _record_omitted(directory: Path, name: str, row: dict[str, Any]) -> None:
    os.setxattr(directory, _unreadable_xattr(name), _json({**row, "name": name}), follow_symlinks=False)


def _tree(
    root: Path, *, allow_directory_symlinks: bool = False, allow_dangling_symlinks: bool = False
) -> dict[str, dict[str, Any]]:
    """Bind exact membership, content and modes; opt in to opaque base link text.

    The compatibility flag allow_dangling_symlinks enables base metadata:
    dangling/external links, internal directory chains and unreadable files.
    """
    if root.is_symlink() or not root.is_dir():
        raise SealedCaptureError(f"tree is absent or indirect: {root}")
    members: dict[str, dict[str, Any]] = {}

    def visit(path: Path, name: str) -> None:
        if path.is_symlink():
            if allow_dangling_symlinks:
                link = _base_symlink(path, root)
                if link is not None:
                    members[name] = link
                    return
            if not allow_directory_symlinks:
                raise SealedCaptureError(f"tree contains a symlink: {name}")
            members[name] = _directory_symlink(path, root)
            return
        before = path.stat(follow_symlinks=False)
        mode = stat.S_IMODE(before.st_mode)
        if stat.S_ISDIR(before.st_mode):
            children = sorted(path.iterdir(), key=lambda child: child.name)
            omitted = _omitted_members(path)
            if omitted and not allow_dangling_symlinks:
                raise SealedCaptureError(f"unreadable tree member: {name}")
            members[name] = {
                "kind": "directory", "mode": mode,
                "members": sorted([*(child.name for child in children), *omitted]),
            }
            for child in children:
                visit(child, child.name if name == "." else f"{name}/{child.name}")
            for child_name, row in omitted.items():
                members[child_name if name == "." else f"{name}/{child_name}"] = row
        elif stat.S_ISREG(before.st_mode):
            unreadable = _unreadable_file(path)
            if unreadable is not None:
                if not allow_dangling_symlinks:
                    raise SealedCaptureError(f"unreadable tree member: {name}")
                members[name] = unreadable
                return

            def identity(row):
                return row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_mode, row.st_uid, row.st_gid

            for attempt in range(3):
                digest = _file_digest(path)
                after = path.stat(follow_symlinks=False)
                unchanged = identity(before) == identity(after)
                if unchanged and before.st_ctime_ns == after.st_ctime_ns:
                    members[name] = {"kind": "file", "mode": mode, "bytes": before.st_size, "sha256": digest}
                    break
                # Another immutable-runtime snapshot can legitimately link or
                # unlink this inode while it is hashed, changing only ctime and
                # its link count. Never accept that unstable read: rehash and
                # require a fully stable identity, including ctime, afterward.
                # Actual content/mode/owner/mtime changes still fail immediately;
                # callers also compare the final content SHA to selected bytes.
                if not unchanged or before.st_nlink == after.st_nlink or attempt == 2:
                    raise SealedCaptureError(f"file changed during inventory: {name}")
                before = after
        else:
            raise SealedCaptureError(f"tree contains a nonregular member: {name}")

    visit(root, ".")
    return dict(sorted(members.items()))


def _static_elf(path: Path) -> None:
    """Reject scripts, dynamic interpreters and shared-library dependencies."""
    raw = path.read_bytes()
    if len(raw) < 64 or raw[:4] != b"\x7fELF" or raw[5] != 1 or raw[4] not in (1, 2):
        raise SealedCaptureError("capture command must be a little-endian static ELF executable")
    bit64 = raw[4] == 2
    header = "<HHIQQQIHHHHHH" if bit64 else "<HHIIIIIHHHHHH"
    fields = struct.unpack_from(header, raw, 16)
    e_type, phoff, phentsize, phnum = fields[0], fields[4], fields[8], fields[9]
    expected = 56 if bit64 else 32
    if e_type != 2 or phentsize < expected or phnum == 0 or phoff + phentsize * phnum > len(raw):
        raise SealedCaptureError("capture command is not a bounded static ET_EXEC ELF")
    for index in range(phnum):
        (kind,) = struct.unpack_from("<I", raw, phoff + index * phentsize)
        if kind in (2, 3):  # PT_DYNAMIC or PT_INTERP
            raise SealedCaptureError("dynamic ELF/runtime dependencies are not supported by this issuer")
    if not path.stat().st_mode & 0o111:
        raise SealedCaptureError("capture command is not executable")


def _payload_command(command: tuple[str, ...], runtime: Path) -> None:
    if not command or any(not isinstance(part, str) or not part or "\x00" in part for part in command):
        raise SealedCaptureError("capture command must be a nonempty literal argv")
    entry = PurePosixPath(command[0])
    if not entry.is_relative_to(PurePosixPath("/runtime")) or entry == PurePosixPath("/runtime"):
        raise SealedCaptureError("capture executable must be inside the sealed runtime")
    if ".." in entry.parts or entry.as_posix() != command[0]:
        raise SealedCaptureError("capture executable path is not canonical")
    relative = entry.relative_to("/runtime")
    executable = runtime / relative
    if executable.is_symlink() or not executable.is_file():
        raise SealedCaptureError("sealed capture executable is absent or indirect")
    _static_elf(executable)


def _bwrap_binary(value: Path | None) -> Path:
    path = Path(value) if value is not None else Path(shutil.which("bwrap") or "")
    if not path.is_absolute():
        raise SealedCaptureError("bubblewrap binary is unavailable or indirect")
    path = _canonical_path(path, exists=True)
    if not path.is_file():
        raise SealedCaptureError("bubblewrap binary is unavailable or indirect")
    return path


def _policy(command: tuple[str, ...]) -> str:
    return _digest(_json({"flags": _FLAGS, "command": command, "mounts": ["runtime:ro", "source:ro", "capture:rw"]}))


def _execute(bwrap: Path, runtime: Path, source: Path, output: Path, command: tuple[str, ...]) -> dict[str, Any]:
    argv = [
        str(bwrap),
        *_FLAGS,
        "--ro-bind",
        str(runtime),
        "/runtime",
        "--ro-bind",
        str(source),
        "/source",
        "--bind",
        str(output),
        "/capture-out",
        "--",
        *command,
    ]
    completed = subprocess.run(argv, env={}, cwd="/", stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace")[:1000]
        raise SealedCaptureError(f"sealed capture exited {completed.returncode}: {detail}")
    return {
        "returncode": completed.returncode,
        "stdout": {"bytes": len(completed.stdout), "sha256": _digest(completed.stdout)},
        "stderr": {"bytes": len(completed.stderr), "sha256": _digest(completed.stderr)},
    }


def issue(
    source_root: Path,
    runtime_root: Path,
    command: tuple[str, ...],
    run_dir: Path,
    *,
    bwrap_binary: Path | None = None,
) -> Path:
    """Execute only private snapshots and seal a new run's byte-bound receipt."""
    source_root = _canonical_path(source_root, exists=True)
    runtime_root = _canonical_path(runtime_root, exists=True)
    run_dir = _canonical_path(run_dir, exists=False)
    if (
        source_root == runtime_root
        or source_root.is_relative_to(runtime_root)
        or runtime_root.is_relative_to(source_root)
    ):
        raise SealedCaptureError("source and runtime must be distinct nonoverlapping trees")
    if any(
        run_dir == root or run_dir.is_relative_to(root) or root.is_relative_to(run_dir)
        for root in (source_root, runtime_root)
    ):
        raise SealedCaptureError("run directory must be outside source and runtime trees")
    before_source, before_runtime = _tree(source_root), _tree(runtime_root)
    _payload_command(command, runtime_root)
    bwrap = _bwrap_binary(bwrap_binary)
    bwrap_digest = _file_digest(bwrap)
    run_dir.mkdir(parents=False, exist_ok=False)
    snapshots = run_dir / "snapshots"
    snapshots.mkdir()
    source, runtime = snapshots / "source", snapshots / "runtime"
    shutil.copytree(source_root, source, symlinks=False)
    shutil.copytree(runtime_root, runtime, symlinks=False)
    if (_tree(source_root), _tree(runtime_root)) != (before_source, before_runtime):
        raise SealedCaptureError("source or runtime changed during private snapshot")
    if (_tree(source), _tree(runtime)) != (before_source, before_runtime):
        raise SealedCaptureError("private source or runtime snapshot differs from selected bytes")
    output = run_dir / "capture"
    output.mkdir()
    process = _execute(bwrap, runtime, source, output, command)
    if (_tree(source), _tree(runtime)) != (before_source, before_runtime):
        raise SealedCaptureError("private source or runtime changed during execution")
    if _file_digest(bwrap) != bwrap_digest:
        raise SealedCaptureError("bubblewrap binary changed during execution")
    output_members = _tree(output)
    if output_members["."]["members"] == []:
        raise SealedCaptureError("capture produced no output")
    document = {
        "schema": SCHEMA,
        "issuer": ISSUER,
        "issuer_sha256": _file_digest(Path(__file__)),
        "status": "local_sealed_static_execution",
        "fresh_execution": True,
        "source_closure_verified": False,
        "static_source_closure_observed": True,
        "scope": "static_elf_process_only",
        "nonce": secrets.token_hex(16),
        "command": list(command),
        "policy_sha256": _policy(command),
        "bwrap_sha256": bwrap_digest,
        "source": before_source,
        "runtime": before_runtime,
        "output": output_members,
        "process": process,
    }
    receipt = run_dir / "capture_execution_attestation.json"
    with receipt.open("xb") as stream:
        stream.write(_json(document) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt.chmod(0o444)
    return receipt


def replay_verify(run_dir: Path, *, bwrap_binary: Path | None = None) -> dict[str, Any]:
    """Re-run the sealed command and demand identical output; trust no status field alone."""
    run_dir = _canonical_path(run_dir, exists=True)
    receipt = run_dir / "capture_execution_attestation.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise SealedCaptureError("sealed execution receipt is absent or indirect")
    try:
        document = strict_json.loads(receipt.read_bytes())
    except (ValueError, UnicodeDecodeError) as exc:
        raise SealedCaptureError("sealed execution receipt is unreadable") from exc
    if not isinstance(document, dict) or (
        document.get("schema") != SCHEMA
        or document.get("issuer") != ISSUER
        or document.get("issuer_sha256") != _file_digest(Path(__file__))
        or document.get("status") != "local_sealed_static_execution"
        or document.get("fresh_execution") is not True
        or document.get("source_closure_verified") is not False
        or document.get("static_source_closure_observed") is not True
        or document.get("scope") != "static_elf_process_only"
        or not isinstance(document.get("nonce"), str)
        or len(document["nonce"]) != 32
    ):
        raise SealedCaptureError("receipt has no supported, current static execution issuer")
    command = document.get("command")
    if not isinstance(command, list) or document.get("policy_sha256") != _policy(tuple(command)):
        raise SealedCaptureError("sealed execution policy differs")
    source, runtime, output = (run_dir / "snapshots/source", run_dir / "snapshots/runtime", run_dir / "capture")
    if _tree(source) != document.get("source") or _tree(runtime) != document.get("runtime"):
        raise SealedCaptureError("sealed source or runtime bytes differ")
    _payload_command(tuple(command), runtime)
    expected_output = _tree(output)
    if expected_output != document.get("output") or expected_output["."]["members"] == []:
        raise SealedCaptureError("capture output bytes differ from the execution receipt")
    bwrap = _bwrap_binary(bwrap_binary)
    if _file_digest(bwrap) != document.get("bwrap_sha256"):
        raise SealedCaptureError("bubblewrap binary differs from the execution receipt")
    with tempfile.TemporaryDirectory(prefix="capture-replay-", dir=run_dir) as temporary:
        replay = Path(temporary)
        replay.chmod(expected_output["."]["mode"])
        replay_process = _execute(bwrap, runtime, source, replay, tuple(command))
        if replay_process != document.get("process"):
            raise SealedCaptureError("fresh sealed replay process logs differ from capture")
        if _tree(replay) != expected_output:
            raise SealedCaptureError("fresh sealed replay output differs from capture")
    if _tree(source) != document["source"] or _tree(runtime) != document["runtime"] or _tree(output) != expected_output:
        raise SealedCaptureError("sealed evidence changed during replay")
    return {
        "schema": SCHEMA,
        "status": "replay_verified_static",
        "source_closure_verified": False,
        "static_replay_source_closure_verified": True,
        "historical_execution_verified": False,
        "receipt_sha256": _file_digest(receipt),
        "scope": "static_elf_process_only",
    }
