"""Public immutable scope of Lean checks, independent of target and candidate names.

Results describe a modeled transformation, never hardware conformance or timing.
Saved results need replay; only the host that ran a check can issue live evidence.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "merlin.lean_transform_verification.v1"
DOMAIN_VERSION = "1"
EXTENDED_DOMAIN_VERSION = "2"
DOMAINS = (
    "scalar_integer_v1",
    "ordered_loop_segmentation_v1",
    "buffer_lifetime_v1",
    "dependency_order_v1",
)
EXTENDED_DOMAINS = (*DOMAINS, "total_scalar_rewrite_v1", "ordered_loop_unrolling_v1")
_FILE_CACHE: dict[tuple, str] = {}


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
    if version not in {DOMAIN_VERSION, EXTENDED_DOMAIN_VERSION}:
        raise ValueError("unknown formal coverage registry version")
    return {
        "schema": "merlin.lean_coverage_policy.v1",
        "mode": "strict_within_coverage",
        "registry_version": version,
        "domains": list(DOMAINS if version == DOMAIN_VERSION else EXTENDED_DOMAINS),
        "unknown_within_coverage": "reject",
        "outside_coverage": "independent_checks_required",
    }


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
        result = cls(
            executable=selected(executable["path"]),
            executable_sha256=executable["sha256"],
            library_roots=tuple(selected(p) for p in raw["library_roots"]),
            files=tuple((selected(row["path"]), row["sha256"]) for row in raw["files"]),
            work_root=Path(work_root).absolute(),
            timeout_s=raw.get("timeout_s", 120),
            memory_mb=raw.get("memory_mb", 1024),
            dependency_revision=raw.get("dependency_revision", ""),
        )
        result.verify()
        return result

    def verify(self, *, deadline: float | None = None) -> None:
        if (
            type(self.timeout_s) is not int
            or self.timeout_s < 1
            or type(self.memory_mb) is not int
            or self.memory_mb < 1
        ):
            raise ValueError("Lean needs positive timeout and memory budgets")
        if not self.executable.is_absolute() or not self.executable.is_file():
            raise FileNotFoundError("selected Lean executable is unavailable")
        if file_digest(self.executable) != self.executable_sha256:
            raise ValueError("selected Lean executable bytes changed")
        if not self.library_roots or any(not p.is_absolute() or not p.is_dir() for p in self.library_roots):
            raise FileNotFoundError("selected Lean import roots are unavailable")
        if not self.files or len({str(p) for p, _ in self.files}) != len(self.files):
            raise ValueError("Lean dependency inventory is empty or duplicated")
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
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Lean dependency verification exceeded its selected wall budget")
        if not self.work_root.is_absolute() or not self.work_root.is_dir():
            raise FileNotFoundError("selected Lean work root is unavailable")

    def identity(self) -> str:
        return document_digest(
            {
                "executable": self.executable_sha256,
                "roots": [str(p) for p in self.library_roots],
                "files": [(str(p), h) for p, h in self.files],
                "revision": self.dependency_revision,
                "timeout_s": self.timeout_s,
                "memory_mb": self.memory_mb,
            }
        )


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
        return self.status == "verified" and self.method == "lean_kernel"

    def to_dict(self) -> dict:
        return asdict(self)
