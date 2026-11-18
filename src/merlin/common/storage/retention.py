"""What may be deleted: leases, pins and citations that keep generated output alive."""

from __future__ import annotations

import argparse
import bisect
import fnmatch
import json
import os
import socket
import sys
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from merlin.common.artifacts import storage_contract as contract
from merlin.common.paths import out_dir

# ---- storage_lifecycle -------------------------------------------------------------------------
# Explicit ownership and retention of generated state, independent of its directory name.
#
# Writers hold a lease while using a path. Pins keep evidence after the writer exits. Storage cleanup
# uses the same lock as lease/pin updates, so a writer cannot acquire a lease between the final check
# and removal. Readers never create output directories, lock files, or lifecycle records.
#
# Leases do not expire by age. Even a known-dead owner remains protected until explicitly acknowledged:
# its child workers might still be running. A lease on another host or one we cannot inspect remains
# protected too.
# Historical output without a recorded owner remains unclassified for retention and pending cleanup.

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


# ---- cited_evidence ----------------------------------------------------------------------------
# Which paths a seal, release, run record or derivation cites -- so no cleanup deletes them.
#
# On 2026-10-17 and again on 2026-10-24 a disk cleanup deleted files that sealed Phase 0 evidence
# pinned by digest: the captures behind a corpus derivation, then the catalogs inside a reviewed
# derivation itself. Both times the deletion looked safe from the filesystem ("old", "staged", "a
# copy exists elsewhere") and both times the loss surfaced only when a later preparation re-read the
# exact path and digest. The knowledge that a path is live was never missing; it was written down in
# the records that cite it, and nothing that deletes consulted them.
#
# This module reads those records and answers one question: would removing a path take bytes away
# from a record that will outlive the removal? The record roster -- which file names are citing
# records, which seal anchors protect a whole tree, which directories are not worth walking -- is
# data in ``src/merlin/resources/contracts/storage.yaml`` (``cited_evidence``), not names in code.
#
# What counts as a citation, read structurally from each record's JSON or YAML:
#
# * an absolute path string anywhere in the document (argv lists and ``definition:`` fields included);
# * a relative ``path`` / ``*_path`` value in a mapping that also carries a ``*sha256`` digest, resolved
#   against the record's own directory;
# * the member names of an ``artifacts`` mapping whose rows carry a digest (a producer receipt);
# * the whole tree under a declared seal anchor, and every path a record lists under ``protects``.
#
# What a citation forbids: removing the cited path or any directory containing it. Removing something
# INSIDE a cited directory is forbidden only when the citation pins the tree's content (it carries a
# digest) or anchors it (a seal) -- a bare mention of a cache root is not a claim on each file in it.
# A citation never blocks a removal that also removes the record making it (a run unit deleted whole
# takes its own run record with it), except a seal anchor, which is the point of sealing.

ROOTS_ENV = "MERLIN_CITED_ROOTS"
_MAX_PATH = 4096
_DEFAULT_MAX_RECORD_BYTES = 16 << 20


@dataclass(frozen=True)
class Citation:
    path: str
    record: str
    kind: str  # "path" (bare mention), "pinned" (carries a digest), "anchor" (sealed tree)
    sha256: str | None = None

    def describe(self) -> str:
        pin = f" @ {self.sha256[:12]}" if self.sha256 else ""
        return f"{self.kind} citation of {self.path}{pin} by {self.record}"


def roster() -> dict:
    """The contract's ``cited_evidence`` block: ``records``, ``anchors``, ``skip_dirs``, ``extra_roots``."""
    from merlin.common.artifacts import storage_contract

    block = storage_contract().get("cited_evidence") or {}
    if not isinstance(block, dict):
        raise ValueError("storage cited_evidence must be a mapping")
    return block


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _norm(path: str) -> str:
    return os.path.normpath(path)


def _relative_member(value: str) -> bool:
    if not value or len(value) > _MAX_PATH or "\n" in value or "://" in value or value.startswith("/"):
        return False
    parts = Path(value).parts
    return bool(parts) and ".." not in parts


def _absolute(value: str) -> bool:
    return value.startswith("/") and len(value) < _MAX_PATH and "\n" not in value and value != "/"


def _digest_of(mapping: dict, key: str) -> str | None:
    """The digest a mapping pairs with ``key``: ``x_path`` -> ``x_sha256``, ``path`` -> ``sha256`` or the only one."""
    stem = key[: -len("path")] if key.endswith("path") else None
    if stem is not None and _is_digest(mapping.get(stem + "sha256")):
        return mapping[stem + "sha256"]
    digests = [v for k, v in mapping.items() if isinstance(k, str) and k.endswith("sha256") and _is_digest(v)]
    return digests[0] if key == "path" and len(digests) == 1 else None


def _walk_document(document: object, base: Path, record: str) -> Iterator[Citation]:
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            has_digest = any(isinstance(k, str) and k.endswith("sha256") and _is_digest(v) for k, v in node.items())
            artifacts = node.get("artifacts")
            if isinstance(artifacts, dict):
                for name, row in artifacts.items():
                    if isinstance(name, str) and _relative_member(name) and isinstance(row, dict):
                        digest = row.get("sha256")
                        if _is_digest(digest):
                            yield Citation(_norm(str(base / name)), record, "pinned", digest)
            for key, value in node.items():
                if isinstance(value, str) and isinstance(key, str):
                    digest = _digest_of(node, key) if (key == "path" or key.endswith("_path")) else None
                    if _absolute(value):
                        yield Citation(_norm(value), record, "pinned" if digest else "path", digest)
                    elif has_digest and (key == "path" or key.endswith("_path")) and _relative_member(value):
                        yield Citation(_norm(str(base / value)), record, "pinned" if digest else "path", digest)
                elif isinstance(value, dict | list):
                    stack.append(value)
        elif isinstance(node, list):
            for value in node:
                if isinstance(value, str):
                    if _absolute(value):
                        yield Citation(_norm(value), record, "path")
                elif isinstance(value, dict | list):
                    stack.append(value)


def _load(path: Path) -> object | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        if path.suffix == ".json":
            return json.loads(raw)
        import yaml

        return yaml.safe_load(raw)
    except Exception:  # noqa: BLE001 -- an unreadable or malformed record cites nothing
        return None


def record_citations(path: Path, *, anchors: Iterable[dict] = ()) -> list[Citation]:
    """Every citation one record file makes, including the trees its seal anchors."""
    path = Path(os.path.abspath(path))
    document = _load(path)
    record = str(path)
    out = list(_walk_document(document, path.parent, record)) if document is not None else []
    if isinstance(document, dict) and isinstance(document.get("protects"), list):
        for value in document["protects"]:
            if isinstance(value, str) and (_absolute(value) or _relative_member(value)):
                out.append(Citation(_norm(str(path.parent / value)), record, "anchor"))
    for anchor in anchors:
        pattern = str(anchor.get("record") or "")
        if pattern and fnmatch.fnmatchcase(path.as_posix(), "*/" + pattern):
            root = path
            for _ in range(int(anchor.get("up", 1))):
                root = root.parent
            out.append(Citation(_norm(str(root)), record, "anchor"))
    return out


def _is_record(name: str, relative: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) or fnmatch.fnmatchcase(relative, "*/" + p) for p in patterns)


def declared_roots() -> list[Path]:
    """The out/ root, the storage scan roots, the contract's extra roots and ``MERLIN_CITED_ROOTS``."""
    from merlin.common.paths import out_dir

    roots = [out_dir()]
    try:
        roots += scan_roots()
    except (OSError, ValueError):
        pass
    roots += extra_roots()
    kept: list[Path] = []
    for path in sorted({Path(os.path.abspath(p)) for p in roots}, key=lambda p: len(p.parts)):
        if path.is_dir() and not any(parent in path.parents or parent == path for parent in kept):
            kept.append(path)
    return kept


class Registry:
    """Citations indexed by normalized path, with the record files they were read from."""

    def __init__(self, citations: Iterable[Citation], *, roots: Iterable[Path] = (), records: int = 0):
        self.by_path: dict[str, list[Citation]] = {}
        for citation in citations:
            self.by_path.setdefault(citation.path, []).append(citation)
            real = os.path.realpath(citation.path)
            if real != citation.path:
                self.by_path.setdefault(real, []).append(citation)
        self._sorted = sorted(self.by_path)
        self.roots = [str(r) for r in roots]
        self.records = records

    def __len__(self) -> int:
        return len(self.by_path)

    def citations(self, candidate: Path | str, *, ignore_records_under: Path | str | None = None) -> list[Citation]:
        """Citations that removing ``candidate`` would violate.

        ``ignore_records_under`` drops citations made by records inside that tree: a scratch root's
        own working receipts are not a seal, release, run record or derivation of it.
        """
        target = _norm(os.path.abspath(candidate))
        real = os.path.realpath(target)
        found: list[Citation] = []
        for probe in dict.fromkeys((target, real)):
            # The candidate itself, or anything inside it.
            found += self.by_path.get(probe, [])
            index = bisect.bisect_left(self._sorted, probe + os.sep)
            while index < len(self._sorted) and self._sorted[index].startswith(probe + os.sep):
                found += self.by_path[self._sorted[index]]
                index += 1
            # A tree containing the candidate, when the citation pins or seals that whole tree.
            for parent in Path(probe).parents:
                for citation in self.by_path.get(str(parent), []):
                    if citation.kind in {"pinned", "anchor"}:
                        found.append(citation)
        ignored = _norm(os.path.abspath(ignore_records_under)) + os.sep if ignore_records_under else None
        out: list[Citation] = []
        for citation in dict.fromkeys(found):
            record = citation.record
            if ignored and record.startswith(ignored) and citation.kind != "anchor":
                continue
            inside = record == real or record.startswith(real + os.sep) or record.startswith(target + os.sep)
            if inside and citation.kind != "anchor":
                continue  # the removal takes the citing record with it
            out.append(citation)
        return out

    def blockers(self, candidate: Path | str, *, limit: int = 3) -> list[str]:
        cited = self.citations(candidate)
        if not cited:
            return []
        shown = "; ".join(c.describe() for c in cited[:limit])
        more = f" (and {len(cited) - limit} more)" if len(cited) > limit else ""
        return [f"cited evidence: {shown}{more}"]


def build(roots: Iterable[Path] | None = None) -> Registry:
    """Walk ``roots`` for citing records and index everything they cite."""
    block = roster()
    patterns = [str(p) for p in block.get("records") or []]
    anchors = [a for a in block.get("anchors") or [] if isinstance(a, dict)]
    skip = {str(s) for s in block.get("skip_dirs") or []}
    limit = int(block.get("max_record_bytes") or _DEFAULT_MAX_RECORD_BYTES)
    roots = list(roots) if roots is not None else declared_roots()
    citations: list[Citation] = []
    records = 0
    for root in roots:
        root = Path(os.path.abspath(root))
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in skip]
            for name in files:
                path = Path(current) / name
                relative = path.relative_to(root).as_posix()
                if not _is_record(name, relative, patterns):
                    continue
                try:
                    if path.is_symlink() or path.stat().st_size > limit:
                        continue
                except OSError:
                    continue
                records += 1
                citations += record_citations(path, anchors=anchors)
    return Registry(citations, roots=roots, records=records)


_CACHE: dict[tuple[str, ...], tuple[float, Registry]] = {}


def registry(roots: Iterable[Path] | None = None, *, refresh: bool = False, max_age: float | None = None) -> Registry:
    """The registry for ``roots`` (default: the declared roots), built once per process.

    ``max_age`` (seconds) rebuilds a cached registry older than that, for long-lived callers that
    must see records written after they started.
    """
    import time

    chosen = list(roots) if roots is not None else declared_roots()
    key = tuple(str(r) for r in chosen)
    cached = _CACHE.get(key)
    if refresh or cached is None or (max_age is not None and time.monotonic() - cached[0] > max_age):
        cached = _CACHE[key] = (time.monotonic(), build(chosen))
    return cached[1]


def extra_roots() -> list[Path]:
    """Only the roots beyond out/: the contract's ``extra_roots`` and ``MERLIN_CITED_ROOTS``."""
    from merlin.common.paths import repo_root

    roots = [p for pattern in roster().get("extra_roots") or [] for p in sorted(repo_root().glob(str(pattern)))]
    roots += [Path(p) for p in os.environ.get(ROOTS_ENV, "").split(os.pathsep) if p.strip()]
    return [Path(os.path.abspath(p)) for p in roots if Path(p).is_dir()]


def clear() -> None:
    _CACHE.clear()


def citation_blockers(path: Path | str, roots: Iterable[Path] | None = None) -> list[str]:
    """Why removing ``path`` would break a citing record (empty when nothing cites it)."""
    return registry(roots).blockers(path)


def require_uncited(path: Path | str, roots: Iterable[Path] | None = None) -> None:
    reasons = citation_blockers(path, roots)
    if reasons:
        raise PermissionError(f"refusing to remove {path}: {'; '.join(reasons)}")


def add_arguments(sub) -> None:
    """The ``merlin-storage cited`` subcommand: ask before deleting anything by hand."""
    cited = sub.add_parser(
        "cited",
        help="which records cite a path; exit 1 when removing any given path would break one (read-only)",
    )
    cited.add_argument("paths", nargs="+", type=Path, help="paths you intend to delete")
    cited.add_argument("--root", action="append", type=Path, default=[], help="also scan this tree for records")
    cited.add_argument("--json", action="store_true", help="emit the citations instead of a summary")


def run(args: argparse.Namespace) -> int:
    roots = declared_roots() + [Path(os.path.abspath(r)) for r in args.root]
    found = registry(roots)
    rows = {str(p): [c.__dict__ for c in found.citations(p)] for p in args.paths}
    if args.json:
        print(json.dumps({"roots": found.roots, "records": found.records, "paths": rows}, indent=2))
    else:
        print(f"{found.records} citing records under {len(found.roots)} root(s)")
        for path, citations in rows.items():
            verdict = f"CITED ({len(citations)})" if citations else "not cited"
            print(f"  {verdict:>12}  {path}")
            for row in citations[:5]:
                print(f"      {row['kind']}: {row['record']}")
    return 1 if any(rows.values()) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="merlin-cited", description=__doc__.splitlines()[0])
    add_arguments(parser.add_subparsers(dest="command", required=True))
    sys.exit(run(parser.parse_args()))


def out_roots() -> tuple[str, ...]:
    """The top-level directories under out/ that the convention names."""
    return tuple(contract().get("out_roots") or ())


def declared_concerns() -> tuple[str, ...]:
    """Every concern allowed to own a subtree of out/artifacts/."""
    return tuple(sorted(contract().get("concerns") or {}))


def scan_roots(extra: list[Path] | None = None) -> list[Path]:
    """Every root that can hold generated state: the out/ root, plus the declared workspace roots.

    An agent run freezes its declared input closure as a SIBLING of its workspace, and a
    capsule-bench workspace lives beside its target experiment rather than under out/. So the roots
    this tool must walk are not all reachable from ``out_dir()``, and the first version of it walked
    only that -- it was blind to 40 GB of completed closures and one 8.7 GB closure abandoned
    mid-copy, which is precisely the class it exists to find.

    The extra roots come from ``src/merlin/resources/contracts/storage.yaml`` rather than from a literal here, for
    the reason every other path in that directory is declared: no library module spells a checkout
    directory, so relocating a workspace is an edit to data. A pattern matching nothing is skipped.
    """
    from merlin.common.paths import repo_root  # noqa: PLC0415

    base = repo_root()
    roots = [out_dir()]
    patterns = contract().get("scan_roots") or []
    for pattern in patterns:
        for path in sorted(base.glob(str(pattern))):
            if path.is_dir() and not path.is_symlink():
                roots.append(path)
    for path in extra or []:
        if path.is_dir():
            roots.append(path)
    # A root nested inside another would be walked twice and double-counted.
    kept: list[Path] = []
    for path in sorted({p.resolve() for p in roots}, key=lambda p: len(p.parts)):
        if not any(parent in path.parents for parent in kept):
            kept.append(path)
    return kept


def holds_tracked_files(source: Path) -> bool:
    """Whether git has anything under this directory in its index.

    A fold leaves a SYMLINK at the old name, and git does not walk one: every tracked file under a
    folded directory immediately reads as deleted from the working tree, even though the bytes are
    intact behind the link. Moving those is a git operation, not a disk one, so a directory holding
    tracked content is reported and skipped -- which happened to two concerns holding five curated
    files the first time this ran.

    No enclosing repository means there is no index to protect. An index that exists but cannot
    be inspected is UNKNOWN and protected, never treated as an empty index.
    """
    import subprocess  # noqa: PLC0415

    source = source.resolve()
    repository = next((parent for parent in (source, *source.parents) if (parent / ".git").exists()), None)
    if repository is None:
        return False
    try:
        done = subprocess.run(
            ["git", "ls-files", "-z", "--", str(source)], capture_output=True, cwd=repository, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return done.returncode != 0 or bool(done.stdout.strip(b"\0"))


def declared_retention() -> list[dict]:
    """The contract's ``retention.pinned`` rules: ``{pattern, reason, doc}`` rows."""
    rows = (contract().get("retention") or {}).get("pinned") or []
    if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("pattern"), str) for r in rows):
        raise ValueError("storage retention.pinned must be a list of {pattern, reason} mappings")
    return rows


def retention_reasons(path: Path) -> list[str]:
    """Why the declared retention keeps ``path``: it is, holds, or lies inside a pinned pattern."""
    from fnmatch import fnmatchcase

    try:
        relative = Path(path).resolve().relative_to(out_dir().resolve()).parts
    except ValueError:
        return []
    reasons = []
    for rule in declared_retention():
        pattern = tuple(rule["pattern"].split("/"))
        shared = min(len(relative), len(pattern))
        if not all(fnmatchcase(relative[i], pattern[i]) for i in range(shared)):
            continue
        if len(relative) < len(pattern) and not any(Path(path).glob("/".join(pattern[len(relative) :]))):
            continue  # a container that holds no pinned unit
        reasons.append(f"declared retention: {rule.get('reason') or rule['pattern']}")
    return reasons


def protection_reasons(path: Path, *, require_terminal: bool = False) -> list[str]:
    reasons = blockers(path, require_terminal=require_terminal)
    reasons += retention_reasons(path) + citation_blockers(path)
    if holds_tracked_files(path):
        reasons.append("tracked files or an uninspectable Git index")
    return reasons


# --- retention ------------------------------------------------------------------------------------
# A concern that is big because each unit is bloated and one that is big because nothing ever drops a
# finished campaign want opposite fixes, and ``experiments`` is what tells them apart. This acts on
# the second: perf-bench holds 475 units averaging 110 MB, which no amount of de-duplication touches.
#
# Ordering is by the timestamp token the naming convention puts in the unit's own name, never by
# mtime. An mtime is not a statement about a run: it changes when anything walks or edits the tree,
# and a purge's own deletions update the mtimes of the units it touched, so an mtime rule reports the
# units you just edited as the live ones. A unit whose name carries no timestamp is never dropped --
# it cannot be placed in the order, so it cannot be shown to be old.
_TS_LEN = 16


def unit_timestamp(name: str) -> str | None:
    """The ``YYYYMMDDTHHMMSSZ`` token in a run or product name, or None if it carries none."""
    for token in name.split("_"):
        if (
            len(token) == _TS_LEN
            and token[8] == "T"
            and token[15] == "Z"
            and token[:8].isdigit()
            and token[9:15].isdigit()
        ):
            return token
    return None
