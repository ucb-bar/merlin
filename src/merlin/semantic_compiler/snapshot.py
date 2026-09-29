"""Reproducible native target-selection snapshot, pending full compiler packaging.

This freezes target descriptors and builds the general `egg` bridge from its
lockfile. It is a component artifact, not an Atlas dialect/emitter/runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .allocate import Reservation, StorageBank
from .model import KernelRequest
from .rules import InstructionDescriptor
from .search import SearchLimits, SearchResult, select_and_allocate


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class NativeTargetProfile:
    target_identity: str
    descriptors: tuple[InstructionDescriptor, ...]
    banks: tuple[StorageBank, ...]

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

    def record(self) -> dict[str, Any]:
        return {
            "schema": "merlin.native_target_profile.v4",
            "target_identity": self.target_identity,
            "descriptors": [descriptor.record() for descriptor in self.descriptors],
            "banks": [bank.record() for bank in self.banks],
        }

    def digest(self) -> str:
        return hashlib.sha256(_encoded(self.record())).hexdigest()

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> NativeTargetProfile:
        if set(row) != {"schema", "target_identity", "descriptors", "banks"} or row["schema"] != (
            "merlin.native_target_profile.v4"
        ):
            raise ValueError("unexpected native target profile schema or fields")
        return cls(
            row["target_identity"],
            tuple(InstructionDescriptor.from_record(item) for item in row["descriptors"]),
            tuple(StorageBank.from_record(item) for item in row["banks"]),
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
    ) -> SearchResult:
        if _hash_file(self.root / "profile.json") != self.manifest["profile_sha256"] or (
            _hash_file(self.bridge) != self.manifest["bridge_sha256"]
        ):
            raise ValueError("native snapshot changed after it was opened")
        if request.target_identity != self.profile.target_identity:
            raise ValueError("kernel target identity differs from native snapshot")
        return select_and_allocate(
            request,
            self.profile.descriptors,
            self.profile.banks,
            bridge=self.bridge,
            fixed_inputs=fixed_inputs,
            reservations=reservations,
            fixed_outputs=fixed_outputs,
            limits=limits,
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
            ["cargo", "build", "--locked", "--release", "--manifest-path", str(manifest_path)],
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
            "schema": "merlin.native_target_snapshot.v4",
            "status": "selection_only",
            "source_revision": source_revision,
            "target_identity": profile.target_identity,
            "profile_sha256": _hash_file(temporary / "profile.json"),
            "bridge_sha256": _hash_file(copied),
            "cargo_lock_sha256": _hash_file(lock_path),
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
    if manifest.get("schema") != "merlin.native_target_snapshot.v4" or manifest.get("status") != "selection_only":
        raise ValueError("native snapshot has invalid manifest")
    profile_path, bridge = root / "profile.json", root / "bin/merlin-egg-bridge"
    if _hash_file(profile_path) != manifest["profile_sha256"] or _hash_file(bridge) != manifest["bridge_sha256"]:
        raise ValueError("native snapshot content differs from manifest")
    profile = NativeTargetProfile.from_record(json.loads(profile_path.read_text()))
    if profile.target_identity != manifest["target_identity"]:
        raise ValueError("native snapshot target identity differs")
    return NativeSnapshot(root, profile, bridge, manifest)
