"""Explicit ownership and retention of generated state, independent of its directory name.

Writers hold a lease while using a path. Pins keep evidence after the writer exits. Storage cleanup
uses the same lock as lease/pin updates, so a writer cannot acquire a lease between the final check
and removal. Readers never create output directories, lock files, or lifecycle records.

Leases do not expire by age. Even a known-dead owner remains protected until explicitly acknowledged:
its child workers might still be running. A lease on another host or one we cannot inspect remains
protected too.
Historical output without a recorded owner remains unclassified for retention and pending cleanup.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from merlin.common.paths import out_dir

_RECORD = ".merlin-storage.json"
_LOCK = ".merlin-storage.lock"
_TERMINAL = frozenset({"completed", "failed", "abandoned"})


def _path(path: Path | str) -> str:
    return str(Path(path).resolve())


def _read(root: Path) -> dict:
    path = root / _RECORD
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": 1, "paths": {}}
    except (OSError, ValueError) as exc:
        raise OSError(f"cannot establish storage ownership from {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("paths"), dict):
        raise OSError(f"invalid storage ownership registry: {path}")
    for name, row in data["paths"].items():
        if (
            not isinstance(name, str)
            or not Path(name).is_absolute()
            or not isinstance(row, dict)
            or row.get("state") not in {"unclassified", "running", *_TERMINAL}
            or not isinstance(row.get("leases"), dict)
            or not isinstance(row.get("pins"), dict)
        ):
            raise OSError(f"invalid ownership record in {path}")
    return data


def _write(root: Path, data: dict) -> None:
    temporary = root / f"{_RECORD}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / _RECORD)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _locked(root: Path) -> Iterator[dict]:
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / _LOCK).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield _read(root)


def _process_identity(pid: int) -> dict | None:
    """Linux boot + process start distinguish PID reuse; None means the process disappeared."""
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
    except FileNotFoundError:
        if Path("/proc").is_dir():
            return None
        raise OSError("process identity is unavailable on this host") from None
    try:
        # comm is parenthesized and can contain spaces or ')'. Fields after its final ')' start
        # with state (field 3); process start time is field 22.
        fields = stat.rpartition(") ")[2].split()
        if fields[0] == "Z":
            return None
        return {"boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip(), "start": fields[19]}
    except (IndexError, ValueError) as exc:
        raise OSError("could not parse process identity") from exc


def _owner() -> dict:
    pid = os.getpid()
    try:
        identity = _process_identity(pid)
    except OSError:
        identity = None
    return {"host": socket.gethostname(), "pid": pid, "identity": identity}


def _owner_state(owner: object) -> str:
    if not isinstance(owner, dict) or owner.get("host") != socket.gethostname():
        return "unknown"
    pid = owner.get("pid")
    if type(pid) is not int or pid <= 0 or not isinstance(owner.get("identity"), dict):
        return "unknown"
    try:
        actual = _process_identity(pid)
    except OSError:
        return "unknown"
    return "dead" if actual is None or actual != owner["identity"] else "live"


def _row(data: dict, path: Path | str) -> dict:
    return data["paths"].setdefault(_path(path), {"state": "unclassified", "leases": {}, "pins": {}})


def _state(row: dict) -> str:
    if row["leases"]:
        states = [_owner_state(owner) for owner in row["leases"].values()]
        if "live" in states:
            return "running"
        if "unknown" in states:
            return "unknown"
        return "owner-dead"
    return row["state"]


def _related(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _blockers(path: Path, data: dict, require_terminal: bool) -> list[str]:
    candidate = path.resolve()
    reasons: list[str] = []
    for name in (_RECORD, _LOCK):
        control = (out_dir() / name).resolve()
        if candidate == control or candidate in control.parents:
            reasons.append(f"storage lifecycle control state: {control}")
    terminal = False
    for name, row in data["paths"].items():
        owned = Path(name)
        if not _related(candidate, owned):
            continue
        state = _state(row)
        if state in {"running", "unknown", "owner-dead"}:
            reasons.append(f"{state} lease/state: {name}")
        if row["pins"]:
            reasons.append(f"retention pin: {name}")
        if state in _TERMINAL and (owned == candidate or owned in candidate.parents):
            terminal = True
    if require_terminal and not terminal:
        reasons.append("no recorded terminal owner; historical state is unclassified")
    return reasons


def blockers(path: Path | str, *, require_terminal: bool = False) -> list[str]:
    """Explain protections on a path, its parents, or its descendants without creating state."""
    try:
        return _blockers(Path(path), _read(out_dir()), require_terminal)
    except OSError as exc:
        return [str(exc)]


@contextmanager
def removal_guard(
    path: Path | str,
    *other_paths: Path | str,
    require_terminal: bool = False,
) -> Iterator[None]:
    """Hold the lifecycle lock through final checks and a removal, rename, or replacement."""
    with _locked(out_dir()) as data:
        for candidate in (path, *other_paths):
            reasons = _blockers(Path(candidate), data, require_terminal)
            if reasons:
                raise PermissionError(f"refusing to change {candidate}: {'; '.join(reasons)}")
        yield


@dataclass(frozen=True)
class Lease:
    """A writer's ownership token; close it only after its child workers have stopped."""

    path: str
    token: str
    root: Path

    def close(self, status: str = "completed") -> None:
        if status not in _TERMINAL:
            raise ValueError(f"unknown terminal status: {status}")
        with _locked(self.root) as data:
            row = _row(data, self.path)
            if self.token not in row["leases"]:
                raise ValueError("lease is already closed or no longer owned by this token")
            del row["leases"][self.token]
            row["state"] = "running" if row["leases"] else status
            _write(self.root, data)


def acquire(path: Path | str, *, owner: str) -> Lease:
    """Register a writer before creating/using its output or starting child workers."""
    if not owner.strip():
        raise ValueError("a lease needs an owner description")
    root = out_dir().resolve()
    token = uuid.uuid4().hex
    with _locked(root) as data:
        row = _row(data, path)
        row["leases"][token] = {**_owner(), "description": owner}
        row["state"] = "running"
        _write(root, data)
    return Lease(_path(path), token, root)


@contextmanager
def lease(path: Path | str, *, owner: str) -> Iterator[Lease]:
    """Hold a lease for an operation, recording completion or failure on exit."""
    held = acquire(path, owner=owner)
    try:
        yield held
    except BaseException:
        held.close("failed")
        raise
    else:
        held.close()


def pin(path: Path | str, *, reason: str) -> str:
    """Keep cited evidence until its explicit pin token is removed."""
    if not reason.strip():
        raise ValueError("a retention pin needs a reason")
    token = uuid.uuid4().hex
    root = out_dir().resolve()
    with _locked(root) as data:
        _row(data, path)["pins"][token] = reason
        _write(root, data)
    return token


def unpin(path: Path | str, token: str) -> None:
    root = out_dir().resolve()
    with _locked(root) as data:
        del _row(data, path)["pins"][token]
        _write(root, data)


def acknowledge_abandoned(path: Path | str, *, reason: str) -> None:
    """Acknowledge crashed work only after separately verifying all child workers have stopped.

    This is an explicit operator decision, not age-based recovery. It cannot override a live or
    uninspectable lease, create ownership for historical output, or remove retention pins.
    """
    if not reason.strip():
        raise ValueError("abandonment acknowledgement needs a reason")
    root = out_dir().resolve()
    with _locked(root) as data:
        row = data["paths"].get(_path(path))
        if row is None or not row["leases"]:
            raise ValueError("no crashed ownership lease to acknowledge")
        if any(_owner_state(owner) != "dead" for owner in row["leases"].values()):
            raise PermissionError("live or uninspectable owners must close their own leases")
        row["leases"] = {}
        row["state"] = "abandoned"
        row["abandoned_reason"] = reason
        _write(root, data)


def inventory() -> dict:
    """Read-only ownership summary for storage reports, including uninspectable leases."""
    try:
        data = _read(out_dir())
    except OSError as exc:
        return {"status": "unknown", "reason": str(exc), "paths": []}
    return {
        "status": "known",
        "paths": [
            {"path": name, "state": _state(row), "pins": dict(row["pins"]), "leases": len(row["leases"])}
            for name, row in sorted(data["paths"].items())
        ],
    }
