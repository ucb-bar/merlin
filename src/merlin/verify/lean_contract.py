"""Public immutable scope of Lean checks, independent of target and candidate names.

Results describe a modeled transformation, never hardware conformance or timing.
Saved results need replay; only the host that ran a check can issue live evidence.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import stat
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "merlin.lean_transform_verification.v1"
DOMAIN_VERSION = "1"
EXTENDED_DOMAIN_VERSION = "2"
AUTHORED_DOMAIN_VERSION = "3"
DOMAINS = (
    "scalar_integer_v1",
    "ordered_loop_segmentation_v1",
    "buffer_lifetime_v1",
    "dependency_order_v1",
)
EXTENDED_DOMAINS = (*DOMAINS, "total_scalar_rewrite_v1", "ordered_loop_unrolling_v1")
AUTHORED_DOMAINS = (*EXTENDED_DOMAINS, "resource_geometry_v1", "state_effect_schedule_v1")
_FILE_CACHE: dict[tuple, str] = {}

# Whole-inventory verification memo. A completed ``LeanToolchain.verify`` is reused only while a
# stamp of every selected file and every directory of every import root is unchanged. The stamp is
# (mode, dev, ino, size, mtime_ns, ctime_ns) from lstat; ctime cannot be restored by utime. Any
# stat error, symlink, stamp mismatch, unreadable or foreign-owned record falls back to the full
# sha256 and membership verification, which is unchanged. Nothing here weakens either check.
MEMO_DIR_ENV = "MERLIN_TOOLCHAIN_MEMO_DIR"
_RACY_NS = 2_000_000_000  # a stamp this close to the check is not trusted: a later write could share its tick
_MEMO: dict[str, tuple[str, tuple[str, ...]]] = {}
_MEMO_LOCKS: dict[str, threading.Lock] = {}
_MEMO_GUARD = threading.Lock()
MEMO_STATS = {"full": 0, "memo": 0}


def file_digest(path: Path) -> str:
    stat = path.stat()
    fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_mode)
    key = (str(path.resolve()), *fingerprint)
    if key not in _FILE_CACHE:
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_mode,
        ) != fingerprint:
            raise ValueError("selected dependency changed while its bytes were observed")
        _FILE_CACHE[key] = observed
    return _FILE_CACHE[key]


def imported_product(path: Path) -> bool:
    """Companion products and native imports belong to the same frozen closure."""
    return any(
        path.name.endswith(suffix)
        for suffix in (".olean", ".olean.private", ".olean.server", ".ilean", ".ir", ".so", ".dll", ".dylib")
    )


def coverage_policy_record(*, version: str = DOMAIN_VERSION) -> dict:
    """The common policy. Candidates cannot narrow the host's selected coverage."""
    if version not in {DOMAIN_VERSION, EXTENDED_DOMAIN_VERSION, AUTHORED_DOMAIN_VERSION}:
        raise ValueError("unknown formal coverage registry version")
    domains = {"1": DOMAINS, "2": EXTENDED_DOMAINS, "3": AUTHORED_DOMAINS}[version]
    policy = {
        "schema": "merlin.lean_coverage_policy.v1",
        "mode": "strict_within_coverage",
        "registry_version": version,
        "domains": list(domains),
        "unknown_within_coverage": "reject",
        "outside_coverage": "independent_checks_required",
    }
    if version == AUTHORED_DOMAIN_VERSION:
        policy["native_reflection"] = {
            "selection": "optional_original_toolchain_permission_and_pinned_sat_solver",
            "method": "lean_native_reflection",
            "additional_tcb": ["Lean.ofReduceBool", "Lean.trustCompiler", "selected Lean native LRAT checker"],
            "kernel_only": False,
        }
    return policy


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def document_digest(value: Any) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


@dataclass(frozen=True)
class LeanToolchain:
    """Explicit executable and import roots, with a closed reviewed file inventory.

    No PATH discovery, install, network or source mutation happens during a check.
    The inventory covers the selected executable and imported library products.
    The caller is responsible for the provenance of this reviewed selection.
    """

    executable: Path
    executable_sha256: str
    library_roots: tuple[Path, ...]
    files: tuple[tuple[Path, str], ...]
    work_root: Path
    timeout_s: int = 120
    memory_mb: int = 1024
    dependency_revision: str = ""
    sat_solver: tuple[Path, str] | None = None
    allow_native_reflection: bool = False
    memo_dir: Path | None = field(default=None, compare=False)

    @classmethod
    def from_manifest(cls, path: Path, *, work_root: Path) -> LeanToolchain:
        raw = json.loads(Path(path).read_text())
        if raw.get("schema") != "merlin.lean_toolchain.v1":
            raise ValueError("unknown Lean toolchain manifest")
        root = Path(path).absolute().parent

        def selected(value):
            if not isinstance(value, str) or not value or ".." in Path(value).parts:
                raise ValueError("invalid Lean manifest path")
            p = Path(value)
            return p if p.is_absolute() else root / p

        executable = raw["executable"]
        solver = raw.get("sat_solver")
        if solver is not None and (type(solver) is not dict or set(solver) != {"path", "sha256"}):
            raise ValueError("Lean SAT manifest selection requires its exact path/SHA256")
        result = cls(
            executable=selected(executable["path"]),
            executable_sha256=executable["sha256"],
            library_roots=tuple(selected(p) for p in raw["library_roots"]),
            files=tuple((selected(row["path"]), row["sha256"]) for row in raw["files"]),
            work_root=Path(work_root).absolute(),
            timeout_s=raw.get("timeout_s", 120),
            memory_mb=raw.get("memory_mb", 1024),
            dependency_revision=raw.get("dependency_revision", ""),
            sat_solver=(selected(solver["path"]), solver["sha256"]) if solver is not None else None,
            allow_native_reflection=raw.get("allow_native_reflection", False),
        )
        result.verify()
        return result

    def verify(self, *, deadline: float | None = None) -> None:
        from .lean_sat_runtime import validate_sat_solver

        validate_sat_solver(self.sat_solver, self.files)
        if type(self.allow_native_reflection) is not bool or (self.allow_native_reflection and self.sat_solver is None):
            raise ValueError("Lean native reflection requires original boolean permission and a pinned SAT solver")
        if (
            type(self.timeout_s) is not int
            or self.timeout_s < 1
            or type(self.memory_mb) is not int
            or self.memory_mb < 1
        ):
            raise ValueError("Lean needs positive timeout and memory budgets")
        if not self.executable.is_absolute() or not self.executable.is_file():
            raise FileNotFoundError("selected Lean executable is unavailable")
        if not self.library_roots or any(not p.is_absolute() or not p.is_dir() for p in self.library_roots):
            raise FileNotFoundError("selected Lean import roots are unavailable")
        if not self.files or len({str(p) for p, _ in self.files}) != len(self.files):
            raise ValueError("Lean dependency inventory is empty or duplicated")
        self._verify_inventory(deadline)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Lean dependency verification exceeded its selected wall budget")
        if not self.work_root.is_absolute() or not self.work_root.is_dir():
            raise FileNotFoundError("selected Lean work root is unavailable")

    def _verify_inventory(self, deadline: float | None) -> None:
        """Executable and dependency bytes plus import-product membership, memoized by stamp."""
        identity = self.identity()
        if self._memo_hit(identity):
            MEMO_STATS["memo"] += 1
            return
        with self._memo_guard(identity, deadline):
            if self._memo_hit(identity):
                MEMO_STATS["memo"] += 1
                return
            started = time.time_ns()
            before = self.inventory_stamp()
            self._verify_inventory_in_full(deadline)
            if before is not None and before == self.inventory_stamp() and self.stamp_settled(before, started):
                self._memo_publish(identity, before)

    def _verify_inventory_in_full(self, deadline: float | None) -> None:
        MEMO_STATS["full"] += 1
        if file_digest(self.executable) != self.executable_sha256:
            raise ValueError("selected Lean executable bytes changed")
        for path, expected in self.files:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Lean dependency verification exceeded its selected wall budget")
            if not path.is_absolute() or not path.is_file() or file_digest(path) != expected:
                raise ValueError("selected Lean dependency bytes changed")
        selected = {str(p.resolve()) for p, _ in self.files if imported_product(p)}
        observed = {
            str(p.resolve())
            for root in self.library_roots
            for p in root.rglob("*")
            if p.is_file() and imported_product(p)
        }
        if selected != observed:
            raise ValueError("Lean transitive import product membership differs from the frozen inventory")

    # -- stamps ------------------------------------------------------------------------------

    def _tree_dirs(self) -> tuple[str, ...] | None:
        """Every directory at or below an import root; None when a link makes the stamp unsound."""
        found = []
        pending = [str(root) for root in self.library_roots]
        while pending:
            directory = pending.pop()
            found.append(directory)
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        return None
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(entry.path)
        return tuple(sorted(found))

    def inventory_stamp(self, tree_dirs: tuple[str, ...] | None = None) -> tuple[str, tuple[str, ...]] | None:
        """(digest, tree directories) of every input to the memoized result, or None if unstampable."""
        try:
            if tree_dirs is None:
                tree_dirs = self._tree_dirs()
                if tree_dirs is None:
                    return None
            members = {str(self.executable), *(str(p) for p, _ in self.files)}
            ancestors = set()
            for name in (*members, *(str(r) for r in self.library_roots)):
                parent = os.path.dirname(name)
                while parent not in ancestors:
                    ancestors.add(parent)
                    if parent == os.path.dirname(parent):
                        break
                    parent = os.path.dirname(parent)
            ancestors.difference_update(tree_dirs)
            rows = []
            for name in sorted(members | set(tree_dirs)):
                st = os.lstat(name)
                if name in members and not stat.S_ISREG(st.st_mode):
                    return None
                if name not in members and not stat.S_ISDIR(st.st_mode):
                    return None
                rows.append((name, st.st_mode, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns))
            for name in sorted(ancestors):
                st = os.lstat(name)
                if not stat.S_ISDIR(st.st_mode):
                    return None
                # Ancestors outside the roots only need to stay the same directories (no link swap).
                rows.append((name, st.st_mode, st.st_dev, st.st_ino))
            # Every directory holding a selected file under an import root, and the roots themselves,
            # must be stamped, or an added unlisted product could hide in an unstamped directory.
            roots = tuple(str(root) for root in self.library_roots)
            covered = set(tree_dirs)
            if not set(roots) <= covered:
                return None
            for name in members:
                parent = os.path.dirname(name)
                if any(parent == r or parent.startswith(r + os.sep) for r in roots) and parent not in covered:
                    return None
            digest_ = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
            return digest_, tree_dirs
        except OSError:
            return None

    def stamp_settled(self, stamp, started_ns: int) -> bool:
        """Refuse stamps whose newest timestamp is within the racy window of the check."""
        newest = 0
        try:
            for name in (str(self.executable), *(str(p) for p, _ in self.files), *stamp[1]):
                st = os.lstat(name)
                newest = max(newest, st.st_mtime_ns, st.st_ctime_ns)
        except OSError:
            return False
        return newest + _RACY_NS <= started_ns

    # -- memo store --------------------------------------------------------------------------

    def _memo_directory(self) -> Path | None:
        raw = self.memo_dir if self.memo_dir is not None else os.environ.get(MEMO_DIR_ENV)
        if not raw:
            return None
        directory = Path(raw)
        try:
            os.makedirs(directory, mode=0o700, exist_ok=True)
            st = os.lstat(directory)
        except OSError:
            return None
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o022:
            return None
        return directory

    def _memo_hit(self, identity: str) -> bool:
        record = _MEMO.get(identity)
        if record is None:
            record = self._memo_load(identity)
        if record is None:
            return False
        current = self.inventory_stamp(record[1])
        return current is not None and current[0] == record[0]

    def _memo_load(self, identity: str):
        directory = self._memo_directory()
        if directory is None:
            return None
        path = directory / f"{identity}.json"
        try:
            st = os.lstat(path)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o022:
                return None
            raw = json.loads(path.read_bytes())
            if raw.get("identity") != identity or type(raw.get("stamp")) is not str:
                return None
            dirs = raw["tree_dirs"]
            if type(dirs) is not list or any(type(d) is not str for d in dirs):
                return None
        except (OSError, ValueError, KeyError, AttributeError):
            return None
        return raw["stamp"], tuple(dirs)

    def _memo_publish(self, identity: str, stamp) -> None:
        _MEMO[identity] = stamp
        directory = self._memo_directory()
        if directory is None:
            return
        temporary = directory / f".{identity}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            payload = {"identity": identity, "stamp": stamp[0], "tree_dirs": list(stamp[1])}
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(payload, stream)
            os.replace(temporary, directory / f"{identity}.json")
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(temporary)

    @contextlib.contextmanager
    def _memo_guard(self, identity: str, deadline: float | None):
        """One verifier per identity at a time (threads, then processes); the rest then hit the memo."""
        with _MEMO_GUARD:
            lock = _MEMO_LOCKS.setdefault(identity, threading.Lock())
        self._acquire(lambda: lock.acquire(False), deadline)
        stream = None
        try:
            directory = self._memo_directory()
            if directory is not None:
                with contextlib.suppress(OSError):
                    stream = open(directory / f"{identity}.lock", "a")  # noqa: SIM115
                    self._acquire(lambda: _try_flock(stream), deadline)
            yield
        finally:
            if stream is not None:
                stream.close()  # closing releases the flock
            lock.release()

    @staticmethod
    def _acquire(attempt, deadline: float | None) -> None:
        while not attempt():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Lean dependency verification exceeded its selected wall budget")
            time.sleep(0.05)

    def identity(self) -> str:
        return document_digest(
            {
                "executable": self.executable_sha256,
                "roots": [str(p) for p in self.library_roots],
                "files": [(str(p), h) for p, h in self.files],
                "revision": self.dependency_revision,
                "timeout_s": self.timeout_s,
                "memory_mb": self.memory_mb,
                "sat_solver": (str(self.sat_solver[0]), self.sat_solver[1]) if self.sat_solver else None,
                "allow_native_reflection": self.allow_native_reflection,
            }
        )


def _try_flock(stream) -> bool:
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


@dataclass(frozen=True)
class LeanTransformReceipt:
    schema: str
    method: str
    domain: str
    source_sha256: str
    target_sha256: str
    semantics_sha256: str
    status: str
    scope: tuple[str, ...]
    assumptions: tuple[str, ...]
    reason: str | None = None
    query_sha256: str | None = None
    toolchain_sha256: str | None = None
    axioms: tuple[str, ...] = ()
    elapsed_s: float | None = None
    cache_hit: bool = False

    @property
    def verified(self) -> bool:
        return self.status == "verified" and self.method in {"lean_kernel", "lean_native_reflection"}

    def to_dict(self) -> dict:
        return asdict(self)
