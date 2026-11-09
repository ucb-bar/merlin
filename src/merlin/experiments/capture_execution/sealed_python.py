"""Fresh, isolated Python execution with a deliberately non-admissible receipt.

The caller supplies a *complete guest root* containing CPython, its standard
library, loader and shared libraries.  The process can only read that copied
root and a separately copied source tree.  This is a useful process-level
boundary, but it is not a Model2MLIR/PyTorch source-closure certificate: native
``dlopen``, framework data, checkpoints and host snapshot immutability need a
separately reviewed policy before Phase 0 may admit a capture.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.common import strict_json

from .sealed_static import _bwrap_binary, _canonical_path, _digest, _file_digest, _json, _tree

SCHEMA = "merlin.sealed_python_diagnostic.v1"
ISSUER = "merlin.sealed-python-diagnostic.v1"
_TIMEOUT_SECONDS = 120
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
    "/capture-out/cache",
    "--setenv",
    "PATH",
    "/usr/bin",
    "--chdir",
    "/",
)
_MOUNT_POINTS = ("source", "capture-out", "tmp", "dev")


class SealedPythonError(ValueError):
    """The execution does not satisfy this narrow diagnostic policy."""


def _member(value: str, *, suffix: str | None = None) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {".", ".."} for part in path.parts)
        or "\\" in value
        or "\x00" in value
        or (suffix is not None and path.suffix != suffix)
    ):
        raise SealedPythonError(f"unsafe selected member: {value!r}")
    return path


def _selection(runtime: Path, source: Path, interpreter: str, script: str, args: tuple[str, ...]) -> tuple[str, ...]:
    interpreter_path = PurePosixPath(interpreter)
    if (
        not interpreter_path.is_absolute()
        or interpreter_path.as_posix() != interpreter
        or ".." in interpreter_path.parts
        or not interpreter_path.name.startswith("python3")
        or not interpreter_path.is_relative_to(PurePosixPath("/usr/bin"))
    ):
        raise SealedPythonError("interpreter must be a canonical /usr/bin/python3 executable")
    selected_interpreter = runtime.joinpath(*interpreter_path.parts[1:])
    if selected_interpreter.is_symlink() or not selected_interpreter.is_file():
        raise SealedPythonError("selected guest Python executable is absent or indirect")
    if selected_interpreter.read_bytes()[:4] != b"\x7fELF" or not selected_interpreter.stat().st_mode & 0o111:
        raise SealedPythonError("selected guest Python executable is not an executable ELF")
    script_path = source.joinpath(*_member(script, suffix=".py").parts)
    if script_path.is_symlink() or not script_path.is_file():
        raise SealedPythonError("selected source script is absent or indirect")
    if any(not isinstance(arg, str) or "\x00" in arg for arg in args):
        raise SealedPythonError("capture arguments must be literal strings without NUL")
    for name in _MOUNT_POINTS:
        placeholder = runtime / name
        if placeholder.is_symlink() or not placeholder.is_dir() or any(placeholder.iterdir()):
            raise SealedPythonError(f"guest root needs an empty /{name} mount point")
    return (interpreter, "-I", "-S", "-B", f"/source/{script}", *args)


def _policy(command: tuple[str, ...]) -> str:
    return _digest(
        _json(
            {
                "flags": _FLAGS,
                "command": command,
                "mounts": ["guest-root:ro", "source:ro", "capture:rw", "tmp:tmpfs", "dev:private"],
                "timeout_seconds": _TIMEOUT_SECONDS,
            }
        )
    )


def _execute(bwrap: Path, runtime: Path, source: Path, output: Path, command: tuple[str, ...]) -> dict[str, Any]:
    argv = [
        str(bwrap),
        *_FLAGS,
        "--ro-bind",
        str(runtime),
        "/",
        "--ro-bind",
        str(source),
        "/source",
        "--bind",
        str(output),
        "/capture-out",
        "--tmpfs",
        "/tmp",
        "--dev",
        "/dev",
        "--",
        *command,
    ]
    try:
        result = subprocess.run(
            argv, env={}, cwd="/", stdin=subprocess.DEVNULL, capture_output=True, timeout=_TIMEOUT_SECONDS
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SealedPythonError(f"sandbox execution failed: {type(exc).__name__}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace")[:1000]
        raise SealedPythonError(f"sandboxed Python exited {result.returncode}: {detail}")
    return {
        "returncode": result.returncode,
        "stdout": {"bytes": len(result.stdout), "sha256": _digest(result.stdout)},
        "stderr": {"bytes": len(result.stderr), "sha256": _digest(result.stderr)},
    }


def issue(
    source_root: Path,
    guest_root: Path,
    script: str,
    run_dir: Path,
    *,
    interpreter: str = "/usr/bin/python3",
    args: tuple[str, ...] = (),
    bwrap_binary: Path | None = None,
) -> Path:
    """Run from private snapshots; never issue a Phase 0 admission claim."""
    source_root = _canonical_path(source_root, exists=True)
    guest_root = _canonical_path(guest_root, exists=True)
    run_dir = _canonical_path(run_dir, exists=False)
    if source_root == guest_root or source_root.is_relative_to(guest_root) or guest_root.is_relative_to(source_root):
        raise SealedPythonError("source and guest root must be distinct nonoverlapping trees")
    if any(
        run_dir == root or run_dir.is_relative_to(root) or root.is_relative_to(run_dir)
        for root in (source_root, guest_root)
    ):
        raise SealedPythonError("run directory must be outside selected input trees")
    source_before, runtime_before = _tree(source_root), _tree(guest_root)
    command = _selection(guest_root, source_root, interpreter, script, args)
    bwrap = _bwrap_binary(bwrap_binary)
    bwrap_sha = _file_digest(bwrap)
    run_dir.mkdir(parents=False, exist_ok=False)
    snapshots = run_dir / "snapshots"
    snapshots.mkdir()
    source, runtime = snapshots / "source", snapshots / "guest-root"
    shutil.copytree(source_root, source, symlinks=False)
    shutil.copytree(guest_root, runtime, symlinks=False)
    if (_tree(source_root), _tree(guest_root)) != (source_before, runtime_before):
        raise SealedPythonError("selected inputs changed during private snapshot")
    if (_tree(source), _tree(runtime)) != (source_before, runtime_before):
        raise SealedPythonError("private snapshots differ from selected input bytes")
    output = run_dir / "capture"
    output.mkdir()
    process = _execute(bwrap, runtime, source, output, command)
    if (_tree(source), _tree(runtime)) != (source_before, runtime_before) or _file_digest(bwrap) != bwrap_sha:
        raise SealedPythonError("private inputs or sandbox binary changed during execution")
    output_members = _tree(output)
    if not output_members["."]["members"]:
        raise SealedPythonError("sandboxed Python produced no capture artifacts")
    document = {
        "schema": SCHEMA,
        "issuer": ISSUER,
        "issuer_sha256": _file_digest(Path(__file__)),
        "status": "local_sandbox_execution_nonadmissible",
        "fresh_execution": True,
        "source_closure_verified": False,
        "phase0_admissible": False,
        "scope": "selected_guest_root_and_source_only; no Model2MLIR/PyTorch qualification",
        "nonce": secrets.token_hex(16),
        "script": script,
        "command": list(command),
        "policy_sha256": _policy(command),
        "bwrap_sha256": bwrap_sha,
        "source": source_before,
        "guest_root": runtime_before,
        "output": output_members,
        "process": process,
    }
    receipt = run_dir / "sealed_python_diagnostic.json"
    with receipt.open("xb") as stream:
        stream.write(_json(document) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt.chmod(0o444)
    return receipt


def replay_verify(run_dir: Path, *, bwrap_binary: Path | None = None) -> dict[str, Any]:
    """Replay the same saved inputs and output bytes, without upgrading admission."""
    run_dir = _canonical_path(run_dir, exists=True)
    receipt = run_dir / "sealed_python_diagnostic.json"
    if receipt.is_symlink() or not receipt.is_file():
        raise SealedPythonError("sealed Python diagnostic receipt is absent or indirect")
    try:
        document = strict_json.loads(receipt.read_bytes())
    except (ValueError, UnicodeDecodeError) as exc:
        raise SealedPythonError("sealed Python diagnostic receipt is unreadable") from exc
    if not isinstance(document, dict) or (
        document.get("schema") != SCHEMA
        or document.get("issuer") != ISSUER
        or document.get("issuer_sha256") != _file_digest(Path(__file__))
        or document.get("status") != "local_sandbox_execution_nonadmissible"
        or document.get("fresh_execution") is not True
        or document.get("source_closure_verified") is not False
        or document.get("phase0_admissible") is not False
        or document.get("scope") != "selected_guest_root_and_source_only; no Model2MLIR/PyTorch qualification"
        or not isinstance(document.get("nonce"), str)
        or len(document["nonce"]) != 32
    ):
        raise SealedPythonError("receipt has no supported non-admissible Python issuer")
    command, script = document.get("command"), document.get("script")
    if (
        not isinstance(command, list)
        or len(command) < 5
        or not all(isinstance(part, str) for part in command)
        or not isinstance(script, str)
        or document.get("policy_sha256") != _policy(tuple(command))
    ):
        raise SealedPythonError("sealed Python diagnostic policy differs")
    source, runtime, output = run_dir / "snapshots/source", run_dir / "snapshots/guest-root", run_dir / "capture"
    if _tree(source) != document.get("source") or _tree(runtime) != document.get("guest_root"):
        raise SealedPythonError("sealed Python source or guest-root bytes differ")
    if tuple(command) != _selection(runtime, source, command[0], script, tuple(command[5:])):
        raise SealedPythonError("sealed Python command differs from selected members")
    expected_output = _tree(output)
    if expected_output != document.get("output") or not expected_output["."]["members"]:
        raise SealedPythonError("sealed Python output bytes differ")
    bwrap = _bwrap_binary(bwrap_binary)
    if _file_digest(bwrap) != document.get("bwrap_sha256"):
        raise SealedPythonError("sandbox binary differs from sealed Python receipt")
    with tempfile.TemporaryDirectory(prefix="python-replay-", dir=run_dir) as temporary:
        replay = Path(temporary)
        replay.chmod(expected_output["."]["mode"])
        if _execute(bwrap, runtime, source, replay, tuple(command)) != document.get("process"):
            raise SealedPythonError("fresh Python replay process differs")
        if _tree(replay) != expected_output:
            raise SealedPythonError("fresh Python replay output differs")
    if (
        _tree(source) != document["source"]
        or _tree(runtime) != document["guest_root"]
        or _tree(output) != expected_output
    ):
        raise SealedPythonError("sealed Python evidence changed during replay")
    return {
        "schema": SCHEMA,
        "status": "replay_verified_nonadmissible",
        "fresh_replay": True,
        "source_closure_verified": False,
        "phase0_admissible": False,
        "receipt_sha256": _file_digest(receipt),
    }
