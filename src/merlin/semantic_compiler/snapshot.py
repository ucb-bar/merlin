"""Reproducible native target-selection snapshot, pending full compiler packaging.

This freezes target descriptors and builds the general `egg` bridge from its
lockfile. It is a component artifact, not a target dialect/emitter/runtime.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .allocate import Reservation, StorageBank
from .model import KernelRequest
from .rules import ConcatViewDescriptor, InstructionDescriptor, ViewDescriptor, bundle_catalog
from .search import MaterializationDecision, SearchAblations, SearchLimits, SearchResult, select_and_allocate


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_NATIVE_SOURCE_FILES = (
    "__init__.py",
    "allocate.py",
    "constant_indexing.py",
    "constant_folding.py",
    "egg_bridge.py",
    "extract.py",
    "guarded_softmax_source.py",
    "linalg_bridge.py",
    "linalg_constants.py",
    "linalg_recognition.py",
    "codegen.py",
    "model.py",
    "masks.py",
    "ordered_k_tiling.py",
    "ordered_mn_tiling.py",
    "ordered_mnk_tiling.py",
    "ordered_scan_source.py",
    "precision_codegen.py",
    "precision_constants.py",
    "precision_intake.py",
    "precision_transform.py",
    "predicate_values.py",
    "precision_values.py",
    "reference.py",
    "rules.py",
    "search.py",
    "scalar_region.py",
    "scalar_reference.py",
    "softmax_source.py",
    "snapshot.py",
    "target_binding.py",
    "tensor_assembly.py",
    "tiled_softmax.py",
    "verify.py",
)
_FRONTEND_SOURCE_FILES = (
    "bucketize_source.py",
    "compile_inputs.py",
    "linalg_bf16_add.py",
    "linalg_bf16_mul.py",
    "linalg_bf16_relu.py",
    "linalg_fp8_cast.py",
    "linalg_mlir.py",
    "linalg_ordered_fp8.py",
    "linalg_patterns.py",
    "linalg_scalar_support.py",
    "linalg_boolean_patterns.py",
    "linalg_integer_reductions.py",
    "linalg_extremum_patterns.py",
)
_SHARED_SOURCE_FILES = (
    "capture/safetensors.py",
    "compile/semantic_regions.py",
    "common/ir_lock.py",
    "runtime/fp8_formats.py",
    "semantic_compiler/buffer_allocation.py",
    "xdsl_dialects/_common.py",
    "xdsl_dialects/fp8.py",
)


def packaged_egg_bridge_path() -> Path:
    """Return the Rust bridge shipped with Merlin, in a checkout or wheel."""
    crate = Path(__file__).resolve().with_name("egg_bridge")
    required = ("Cargo.toml", "Cargo.lock", "src/main.rs")
    missing = [name for name in required if not (crate / name).is_file()]
    if missing:
        raise FileNotFoundError(f"installed Merlin native bridge source is incomplete: {missing}")
    return crate


def _compiler_sources() -> dict[str, str]:
    """Bind the installed selector, frontend, and source-preparation controller."""
    package = Path(__file__).resolve().parent
    frontend = package.parent / "frontends"
    targetgen = package.parent / "targetgen"
    merlin = package.parent
    try:
        return {
            **{name: _hash_file(package / name) for name in _NATIVE_SOURCE_FILES},
            **{f"frontends/{name}": _hash_file(frontend / name) for name in _FRONTEND_SOURCE_FILES},
            **{name: _hash_file(merlin / name) for name in _SHARED_SOURCE_FILES},
            "targetgen/cli.py": _hash_file(targetgen / "cli.py"),
        }
    except OSError as exc:
        raise ValueError("native compiler Python sources are unavailable") from exc


def _check_compiler_sources(manifest: dict[str, Any]) -> None:
    if manifest.get("compiler_sources") != _compiler_sources():
        raise ValueError("native compiler Python sources differ from snapshot manifest")


def _runtime_dependencies() -> dict[str, str]:
    """Record versions of the parser, numerical library and actual solver."""
    try:
        import numpy
        import z3

        xdsl_version = importlib.metadata.version("xdsl")
        z3_package_version = importlib.metadata.version("z3-solver")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise ValueError(f"native compiler runtime dependency is unavailable: {exc}") from exc

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "xdsl": xdsl_version,
        "numpy": numpy.__version__,
        "z3-solver": z3_package_version,
        "z3-library": z3.get_version_string(),
    }


def _check_runtime_dependencies(manifest: dict[str, Any]) -> None:
    if manifest.get("runtime_dependencies") != _runtime_dependencies():
        raise ValueError("native compiler runtime dependencies differ from snapshot manifest")


@dataclass(frozen=True)
class NativeTargetProfile:
    target_identity: str
    descriptors: tuple[InstructionDescriptor, ...]
    banks: tuple[StorageBank, ...]
    views: tuple[ViewDescriptor | ConcatViewDescriptor, ...] = ()

    def __post_init__(self) -> None:
        if not self.target_identity or not self.descriptors or not self.banks:
            raise ValueError("native target profile needs identity, instructions and storage")
        names = {bank.name for bank in self.banks}
        if len(names) != len(self.banks):
            raise ValueError("native target profile has duplicate storage banks")
        if any(
            descriptor.output_storage not in names or any(storage not in names for storage in descriptor.input_storages)
            for descriptor in self.descriptors
        ):
            raise ValueError("instruction refers to an undeclared storage bank")
        if len({descriptor.name for descriptor in self.descriptors}) != len(self.descriptors):
            raise ValueError("native target profile has duplicate instruction names")
        bundle_catalog(self.descriptors)
        if any(view.storage not in names for view in self.views):
            raise ValueError("view refers to an undeclared storage bank")
        if len({view.name for view in self.views}) != len(self.views) or {view.name for view in self.views} & {
            descriptor.name for descriptor in self.descriptors
        }:
            raise ValueError("native target profile has duplicate realization names")

    def record(self) -> dict[str, Any]:
        record = {
            "schema": "merlin.native_target_profile.v7" if self.views else "merlin.native_target_profile.v6",
            "target_identity": self.target_identity,
            "descriptors": [descriptor.record() for descriptor in self.descriptors],
            "banks": [bank.record() for bank in self.banks],
        }
        if self.views:
            record["views"] = [view.record() for view in self.views]
        return record

    def digest(self) -> str:
        return hashlib.sha256(_encoded(self.record())).hexdigest()

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> NativeTargetProfile:
        base = {"schema", "target_identity", "descriptors", "banks"}
        schema = row.get("schema")
        if (
            (schema == "merlin.native_target_profile.v6" and set(row) != base)
            or (schema == "merlin.native_target_profile.v7" and set(row) != base | {"views"})
            or schema not in {"merlin.native_target_profile.v6", "merlin.native_target_profile.v7"}
        ):
            raise ValueError("unexpected native target profile schema or fields")
        if schema == "merlin.native_target_profile.v7" and (not isinstance(row["views"], list) or not row["views"]):
            raise ValueError("v7 target profile needs declared views")
        return cls(
            row["target_identity"],
            tuple(InstructionDescriptor.from_record(item) for item in row["descriptors"]),
            tuple(StorageBank.from_record(item) for item in row["banks"]),
            tuple(ViewDescriptor.from_record(item) for item in row.get("views", ())),
        )


@dataclass(frozen=True)
class NativeSnapshot:
    root: Path
    profile: NativeTargetProfile
    bridge: Path
    manifest: dict[str, Any]

    def select(
        self,
        request: KernelRequest,
        *,
        fixed_inputs: dict[str, int] | None = None,
        reservations: tuple[Reservation, ...] = (),
        fixed_outputs: tuple[int | None, ...] | None = None,
        limits: SearchLimits = SearchLimits(),
        ablations: SearchAblations = SearchAblations(),
        accept_materialization: Callable[[SearchResult], tuple[bool, str] | MaterializationDecision] | None = None,
    ) -> SearchResult:
        if _hash_file(self.root / "profile.json") != self.manifest["profile_sha256"] or (
            _hash_file(self.bridge) != self.manifest["bridge_sha256"]
        ):
            raise ValueError("native snapshot changed after it was opened")
        _check_compiler_sources(self.manifest)
        _check_runtime_dependencies(self.manifest)
        if request.target_identity != self.profile.target_identity:
            raise ValueError("kernel target identity differs from native snapshot")
        return select_and_allocate(
            request,
            self.profile.descriptors,
            self.profile.banks,
            bridge=self.bridge,
            views=self.profile.views,
            fixed_inputs=fixed_inputs,
            reservations=reservations,
            fixed_outputs=fixed_outputs,
            limits=limits,
            ablations=ablations,
            accept_materialization=accept_materialization,
        )


def build_native_snapshot(
    profile: NativeTargetProfile,
    *,
    destination: Path,
    crate: Path,
    cargo_target_dir: Path,
    source_revision: str,
    build_timeout_s: int = 600,
) -> NativeSnapshot:
    """Build the pinned bridge and freeze one target view without ACT tooling."""
    if destination.exists():
        raise FileExistsError(f"native snapshot destination already exists: {destination}")
    if not source_revision or build_timeout_s <= 0:
        raise ValueError("native snapshot needs source revision and positive build timeout")
    manifest_path = crate / "Cargo.toml"
    lock_path = crate / "Cargo.lock"
    if not manifest_path.is_file() or not lock_path.is_file():
        raise FileNotFoundError("native e-graph crate manifest or lockfile is missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="native-target-build-", dir=destination.parent))
    cargo_target_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(cargo_target_dir)
    try:
        started = time.monotonic()
        result = subprocess.run(
            ["cargo", "build", "--locked", "--offline", "--release", "--manifest-path", str(manifest_path)],
            capture_output=True,
            check=False,
            timeout=build_timeout_s,
            env=env,
        )
        (temporary / "cargo.stdout.log").write_bytes(result.stdout)
        (temporary / "cargo.stderr.log").write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(f"native e-graph bridge build failed; diagnostics: {temporary}")
        binary = cargo_target_dir / "release/merlin-egg-bridge"
        if not binary.is_file():
            raise RuntimeError(f"native e-graph bridge build produced no binary; diagnostics: {temporary}")
        (temporary / "bin").mkdir()
        copied = temporary / "bin/merlin-egg-bridge"
        shutil.copy2(binary, copied)
        (temporary / "profile.json").write_bytes(_encoded(profile.record()) + b"\n")
        manifest = {
            "schema": "merlin.native_target_snapshot.v6",
            "status": "selection_only",
            "source_revision": source_revision,
            "target_identity": profile.target_identity,
            "profile_sha256": _hash_file(temporary / "profile.json"),
            "bridge_sha256": _hash_file(copied),
            "cargo_lock_sha256": _hash_file(lock_path),
            "compiler_sources": _compiler_sources(),
            "runtime_dependencies": _runtime_dependencies(),
            "bridge_build_seconds": round(time.monotonic() - started, 6),
        }
        (temporary / "manifest.json").write_bytes(_encoded(manifest) + b"\n")
        os.replace(temporary, destination)
    except subprocess.TimeoutExpired as exc:
        (temporary / "build-timeout.txt").write_text(str(exc) + "\n")
        raise RuntimeError(f"native e-graph bridge build timed out; diagnostics: {temporary}") from exc
    return open_native_snapshot(destination)


def open_native_snapshot(root: Path) -> NativeSnapshot:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "merlin.native_target_snapshot.v6" or manifest.get("status") != "selection_only":
        raise ValueError("native snapshot has invalid manifest")
    profile_path, bridge = root / "profile.json", root / "bin/merlin-egg-bridge"
    if _hash_file(profile_path) != manifest["profile_sha256"] or _hash_file(bridge) != manifest["bridge_sha256"]:
        raise ValueError("native snapshot content differs from manifest")
    _check_compiler_sources(manifest)
    _check_runtime_dependencies(manifest)
    profile = NativeTargetProfile.from_record(json.loads(profile_path.read_text()))
    if profile.target_identity != manifest["target_identity"]:
        raise ValueError("native snapshot target identity differs")
    return NativeSnapshot(root, profile, bridge, manifest)
