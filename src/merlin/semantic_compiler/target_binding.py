"""Explicitly selected, installed target binding for native compilation.

The binding owns target facts and physical emission. Merlin owns selection,
extraction, allocation, and checking; an entry point is never a second engine.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

from .model import KernelRequest
from .search import SearchLimits
from .snapshot import NativeSnapshot, NativeTargetProfile


class NativeTargetBinding(Protocol):
    @staticmethod
    def profile() -> NativeTargetProfile: ...

    @staticmethod
    def compile(
        snapshot: NativeSnapshot,
        request: KernelRequest,
        *,
        fixed_inputs: dict[str, int],
        fixed_outputs: tuple[int, ...],
        target_source: Path,
        destination: Path,
        limits: SearchLimits,
    ) -> dict[str, object]: ...


def verify_native_publication(
    staged: Path,
    manifest: dict[str, object],
    *,
    engine: str,
    request_digest: str,
    target_identity: str,
) -> None:
    """Check the immutable identity of files before publishing a target result."""
    binary = staged / "program.bin"
    persisted_path = staged / "manifest.json"
    if not binary.is_file() or not persisted_path.is_file():
        raise ValueError("selected target did not emit a native binary and manifest")
    persisted = json.loads(persisted_path.read_text())
    if persisted != json.loads(json.dumps(manifest)) or manifest.get("engine") != engine or (
        manifest.get("request_digest") != request_digest
        or manifest.get("target_identity") != target_identity
        or manifest.get("binary_sha256") != hashlib.sha256(binary.read_bytes()).hexdigest()
    ):
        raise ValueError("native emitted artifact identity differs from checked request, target, or binary")
    plan = staged / "execution_plan.json"
    expected_plan_digest = manifest.get("execution_plan_sha256")
    if plan.exists() != (expected_plan_digest is not None):
        raise ValueError("native execution plan and manifest identity disagree")
    if plan.exists() and (not plan.is_file() or hashlib.sha256(plan.read_bytes()).hexdigest() != expected_plan_digest):
        raise ValueError("native execution plan differs from checked manifest identity")


def load_native_target_binding(name: str) -> NativeTargetBinding:
    """Load one named installed provider; never infer a target from a request."""
    if not name or not name.isidentifier():
        raise ValueError("native target support needs one explicit entry point name")
    matches = [point for point in entry_points(group="merlin.native_target_compilers") if point.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one installed native target support named {name!r}, found {len(matches)}")
    binding = matches[0].load()
    if not callable(getattr(binding, "profile", None)) or not callable(getattr(binding, "compile", None)):
        raise TypeError("native target support lacks profile or compile interface")
    return binding
