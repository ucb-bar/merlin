"""Fresh, isolated Python execution with a deliberately non-admissible receipt.

The caller supplies a *complete guest root* containing CPython, its standard
library, loader and shared libraries.  The process can only read that copied
root and a separately copied source tree.  This is a useful process-level
boundary, but it is not a Model2MLIR/PyTorch source-closure certificate: native
``dlopen``, framework data, checkpoints and host snapshot immutability need a
separately reviewed policy before Phase 0 may admit a capture.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.common import record_formats as strict_json
from merlin.experiments.capture_execution.sealed_static import (
    SealedCaptureError,
    _bwrap_binary,
    _canonical_path,
    _digest,
    _file_digest,
    _json,
    _tree,
)

# ---- sealed_python -----------------------------------------------------------------------------

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


# ---- sealed_failure ----------------------------------------------------------------------------
# Unverified execution diagnostics, deliberately separate from capture admission.

# Replay-only predecessor: successful guest policy and byte comparisons are unchanged.
PRE_RETENTION_ISSUER_SHA256 = "1c0b4e2bf38893f63c08f7a39e308037e7e80f01615b89d19a4e504a75dc5a20"


def _write(path: Path, raw: bytes) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(raw)


def process(failure_dir: Path, returncode: int, stdout: bytes, stderr: bytes) -> None:
    """Retain actual streams before any materialization or publication checks."""
    if returncode:
        directory = failure_dir
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    else:
        directory = failure_dir.with_name(failure_dir.name.removesuffix("-failure") + "-execution")
        directory = Path(tempfile.mkdtemp(prefix=directory.name + "-", dir=directory.parent))
    _write(directory / "stdout.bin", stdout)
    _write(directory / "stderr.bin", stderr)
    record = {
        "schema": "merlin.sealed_capture_process_diagnostic.v1",
        "status": "unverified",
        "phase0_admission": "not_granted",
        "returncode": returncode,
        "stdout_bytes": len(stdout),
        "stderr_bytes": len(stderr),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }
    _write(directory / "process.json", (json.dumps(record, sort_keys=True) + "\n").encode())


@contextmanager
def publication(run: Path, *, replay: Path | None, plan: dict, selection_sha256: str | None):
    """Delete successful replay scratch; keep failed output without minting a pending receipt.

    This record is not consumable by attest/replay_verify and supplies no recovery API.
    It only preserves observations that would otherwise disappear on a publication error.
    """
    try:
        yield
    except BaseException as exc:
        record = {
            "schema": "merlin.sealed_capture_failure_diagnostic.v1",
            "status": "unverified",
            "phase0_admission": "not_granted",
            "source_closure_verified": False,
            "recovery_supported": False,
            "selection_sha256": selection_sha256,
            "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
            "capture": "capture",
            "replay": None if replay is None else replay.relative_to(run).as_posix(),
            "error": {"type": type(exc).__name__, "detail": str(exc)[:4096]},
        }
        try:
            _write(run / "sealed_capture_unverified_failure.json", (json.dumps(record, sort_keys=True) + "\n").encode())
        except OSError as diagnostic_error:
            exc.add_note(f"unverified failure record could not be retained: {type(diagnostic_error).__name__}")
        raise
    else:
        if replay is not None:
            shutil.rmtree(replay, ignore_errors=True)


# ---- sealed_inventory --------------------------------------------------------------------------
# Exact identities from one fresh inventory; independent boundaries rescan.


def _snapshot_rows(rows: dict[str, dict[str, Any]], relative: str = ".") -> dict[str, Any]:
    """Digest an exact subtree without reading its bytes again at this checkpoint."""
    member = PurePosixPath(relative)
    if member.is_absolute() or ".." in member.parts or member.as_posix() != relative:
        raise SealedCaptureError("inventory subtree path is not canonical")
    if rows.get(relative, {}).get("kind") != "directory":
        raise SealedCaptureError("inventory subtree is absent or not a directory")
    if relative != ".":
        prefix = relative + "/"
        rows = {
            ".": rows[relative],
            **{name[len(prefix) :]: info for name, info in rows.items() if name.startswith(prefix)},
        }
    compact = [(name, *sorted(info.items())) for name, info in sorted(rows.items())]
    return {
        "members": len(rows),
        "bytes": sum(row.get("bytes", 0) for row in rows.values()),
        "sha256": _digest(_json(compact)),
    }


def _verify_staged_selection(
    plan: dict[str, Any],
    source: Path,
    runtime: Path,
    *,
    source_rows: dict[str, dict[str, Any]] | None = None,
    runtime_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reject changed selected bytes before executing any guest code."""
    from merlin.experiments.capture_execution import m2m_plan as _merlin_m2m_plan_1
    from merlin.experiments.capture_execution import m2m_sandbox as M

    def identity(path: Path) -> dict[str, Any]:
        root, rows = (runtime, runtime_rows) if path.is_relative_to(runtime) else (source, source_rows)
        return (
            _merlin_m2m_plan_1._snapshot_tree(path)
            if rows is None
            else _snapshot_rows(rows, path.relative_to(root).as_posix())
        )

    roots = {
        "venv": runtime / "opt/capture-venv",
        "base": runtime / Path(plan["base"]).relative_to("/"),
        "m2m": source / "m2m-src/m2m",
        "workload": source / "workload",
        "schemas": source / "merlin-src/merlin/_data/schemas",
    }
    selected_trees = plan["selected_trees"]
    if plan.get("frozen_origin") is not None:
        try:
            origin_commit = M.verify_frozen_receipt(
                plan["frozen_origin"],
                (source / "m2m-origin.json").read_bytes(),
                Path(plan["m2m_root"]),
                selected_trees["m2m"],
            )
        except (M.M2MOriginError, OSError) as exc:
            raise M.SealedM2MError("staged frozen M2M origin differs from selection") from exc
        if origin_commit != plan["m2m_commit"]:
            raise M.SealedM2MError("staged frozen M2M origin revision differs from selected plan")
    for name, path in roots.items():
        if identity(path) != selected_trees[name]:
            raise M.SealedM2MError(f"staged {name} bytes differ from the pre-execution selection")
    if plan.get("schema") == M.SCHEMA_V3:
        for row in plan["selected_inputs"]:
            _merlin_m2m_plan_1._verify_selected_input(row, source / "inputs" / row["guest_member"])
    if plan["schemas_root"] == str(Path(plan["merlin_root"]) / "_data/schemas"):
        if identity(source / "merlin-src/merlin") != selected_trees["merlin"]:
            raise M.SealedM2MError("staged Merlin bytes differ from the pre-execution selection")
    return selected_trees["schemas"]


def _replay_output_difference(original: Path, replay: Path) -> str:
    """Failure-only detail; never relax exact byte, membership or mode equality."""
    from merlin.experiments.capture_execution.sealed_static import _tree

    expected, actual = _tree(original), _tree(replay)
    for name in sorted(set(expected) | set(actual)):
        if name not in actual:
            return f"missing member {name!r}"
        if name not in expected:
            return f"unexpected member {name!r}"
    modes = []
    for name, row in sorted(expected.items()):
        other = actual[name]
        if row.get("kind") != other.get("kind"):
            return f"member kind differs at {name!r}"
        if row.get("sha256") != other.get("sha256") or row.get("bytes") != other.get("bytes"):
            return f"content hash or size differs at {name!r}"
        if {k: v for k, v in row.items() if k != "mode"} != {k: v for k, v in other.items() if k != "mode"}:
            return f"inventory fields differ at {name!r}"
        if row.get("mode") != other.get("mode"):
            modes.append((name, row["mode"], other["mode"]))
    if modes:
        name, old, new = modes[0]
        return f"mode-only difference at {name!r} (expected {old:#o}, actual {new:#o}); check caller umask"
    return "materialized receipt differs although file inventory matches"


def _verify_replayed_output(replay, source, output, doc, schema, plan) -> None:
    """The original complete materialization and inventory checks, with failure detail."""
    from merlin.experiments.capture_execution import m2m_plan as _merlin_m2m_plan_2
    from merlin.experiments.capture_execution import m2m_sandbox as M

    materialized = (
        _merlin_m2m_plan_2._materialized(replay, source, output)
        if schema == M.SCHEMA_V1
        else (
            _merlin_m2m_plan_2._materialized_v3(replay, source, output, plan)
            if schema == M.SCHEMA_V3
            else _merlin_m2m_plan_2._materialized_v2(replay, source, output, plan)
        )
    )
    if materialized != doc.get("materialized") or _merlin_m2m_plan_2._snapshot_tree(replay) != doc["output"]:
        detail = _replay_output_difference(output, replay)
        raise M.SealedM2MError("fresh M2M replay differs: " + detail)
