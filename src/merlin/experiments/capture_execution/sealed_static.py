"""Fresh capture issuer for a fully snapshotted static executable.

The only payload mounts are private source/runtime snapshots (read-only) and a
fresh capture directory (writable). Python, shared libraries, virtualenvs, and
model2MLIR loaders are deliberately ineligible for this issuer.
"""

from __future__ import annotations

import hashlib
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


def _tree(root: Path) -> dict[str, dict[str, Any]]:
    """Bind exact directory membership, file content and execute permissions."""
    if root.is_symlink() or not root.is_dir():
        raise SealedCaptureError(f"tree is absent or indirect: {root}")
    members: dict[str, dict[str, Any]] = {}

    def visit(path: Path, name: str) -> None:
        if path.is_symlink():
            raise SealedCaptureError(f"tree contains a symlink: {name}")
        before = path.stat(follow_symlinks=False)
        mode = stat.S_IMODE(before.st_mode)
        if stat.S_ISDIR(before.st_mode):
            children = sorted(path.iterdir(), key=lambda child: child.name)
            members[name] = {"kind": "directory", "mode": mode, "members": [child.name for child in children]}
            for child in children:
                visit(child, child.name if name == "." else f"{name}/{child.name}")
        elif stat.S_ISREG(before.st_mode):

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
