"""Explicitly selected, installed target binding for native compilation.

The binding owns target facts and physical emission. Merlin owns selection,
extraction, allocation, and checking; an entry point is never a second engine.
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
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol

from merlin.common.storage.content_store import digest_file
from merlin.compiler.semantic.model import KernelRequest
from merlin.compiler.semantic.rules import (
    ConcatViewDescriptor,
    InstructionDescriptor,
    ViewDescriptor,
    bundle_catalog,
)
from merlin.compiler.semantic.search.allocation import Reservation, StorageBank
from merlin.compiler.semantic.search.exploration import (
    MaterializationDecision,
    SearchAblations,
    SearchLimits,
    SearchResult,
    select_and_allocate,
)

# ---- snapshot ----------------------------------------------------------------------------------
# Reproducible native target-selection snapshot, pending full compiler packaging.
#
# This freezes target descriptors and builds the general `egg` bridge from its
# lockfile. It is a component artifact, not a target dialect/emitter/runtime.


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Paths are relative to the installed merlin package, including moved owners.
_COMPILER_SOURCE_FILES = (
    "capture/safetensors.py",
    "common/digest.py",
    "common/fp8_formats.py",
    "compiler/build/host_bridge.py",
    "compiler/build/deployment.py",
    "compiler/build/handoff.py",
    "compiler/build/memory_budget.py",
    "compiler/build/memory_placement.py",
    "compiler/build/materialization.py",
    "compiler/build/native.py",
    "compiler/build/regions.py",
    "compiler/build/region_fit.py",
    "compiler/build/staging.py",
    "compiler/build/views.py",
    "compiler/transforms/dense_views.py",
    "compiler/dialects/xdsl_context.py",
    "compiler/frontend/__init__.py",
    "compiler/frontend/input_types.py",
    "compiler/frontend/inputs.py",
    "compiler/frontend/linalg_mlir.py",
    "compiler/frontend/patterns/__init__.py",
    "compiler/frontend/patterns/bucketize.py",
    "compiler/frontend/patterns/comparisons.py",
    "compiler/frontend/patterns/linalg_composite_math.py",
    "compiler/frontend/patterns/linalg_integer_reductions.py",
    "compiler/frontend/patterns/linalg_math_patterns.py",
    "compiler/frontend/patterns/linalg_native_pointwise.py",
    "compiler/frontend/patterns/linalg_patterns.py",
    "compiler/frontend/patterns/narrow_float.py",
    "compiler/frontend/signature.py",
    "compiler/frontend/staged.py",
    "compiler/ir/boundary.py",
    "compiler/ir/graph.py",
    "compiler/ir/mlir_query.py",
    "compiler/ir/xdsl_io.py",
    "compiler/lower/c/application.py",
    "compiler/lower/c/bf16_linear.py",
    "compiler/lower/c/bucketize.py",
    "compiler/lower/c/conversions.py",
    "compiler/lower/c/dispatch.py",
    "compiler/lower/c/float_arithmetic.py",
    "compiler/lower/c/float_casts.py",
    "compiler/lower/c/host_ops/__init__.py",
    "compiler/lower/c/host_ops/order.py",
    "compiler/lower/c/host_ops/relu.py",
    "compiler/lower/c/host_ops/softmax_precision.py",
    "compiler/lower/c/integer.py",
    "compiler/lower/c/layout.py",
    "compiler/lower/c/parameters.py",
    "compiler/lower/c/pointwise_host.py",
    "compiler/lower/c/softmax_mask.py",
    "compiler/lower/c/indexed_update.py",
    "compiler/lower/c/batched.py",
    "compiler/lower/c/predicates.py",
    "compiler/lower/c/runtime.py",
    "compiler/lower/c/unary_math.py",
    "compiler/lower/c/weights.py",
    "compiler/partition/dispatch_program.py",
    "compiler/partition/global_plan.py",
    "compiler/partition/global_plan_emission.py",
    "compiler/partition/memory_plan.py",
    "compiler/partition/outline.py",
    "compiler/partition/semantic_regions.py",
    "compiler/planning/cover_selection.py",
    "compiler/quantization/contraction/__init__.py",
    "compiler/quantization/contraction/codegen.py",
    "compiler/quantization/contraction/constants.py",
    "compiler/quantization/contraction/contract.py",
    "compiler/quantization/contraction/intake.py",
    "compiler/quantization/contraction/lowering.py",
    "compiler/quantization/contraction/scaled_fp8.py",
    "compiler/quantization/contraction/transform.py",
    "compiler/quantization/contraction/values.py",
    "compiler/quantization/contraction/batched.py",
    "compiler/schedule/counted_loop.py",
    "compiler/schedule/launch_schedule.py",
    "compiler/schedule/storage/arena.py",
    "compiler/semantic/__init__.py",
    "compiler/semantic/affine_views.py",
    "compiler/semantic/allocation_types.py",
    "compiler/semantic/codegen.py",
    "compiler/semantic/constant_folding.py",
    "compiler/semantic/constant_math.c",
    "compiler/semantic/constant_math_backend.py",
    "compiler/semantic/constant_math_precision.py",
    "compiler/semantic/instruction.py",
    "compiler/semantic/linalg/__init__.py",
    "compiler/semantic/linalg/arithmetic.py",
    "compiler/semantic/linalg/contractions.py",
    "compiler/semantic/linalg/guarded_softmax_source.py",
    "compiler/semantic/linalg/indexed_update_source.py",
    "compiler/semantic/linalg/linalg_bridge.py",
    "compiler/semantic/linalg/linalg_constants.py",
    "compiler/semantic/linalg/linalg_preparation.py",
    "compiler/semantic/linalg/linalg_recognition.py",
    "compiler/semantic/linalg/linalg_returns.py",
    "compiler/semantic/linalg/linalg_state.py",
    "compiler/semantic/linalg/linalg_types.py",
    "compiler/semantic/linalg/linalg_views.py",
    "compiler/semantic/linalg/ordered_scan_source.py",
    "compiler/semantic/linalg/qualified_rmsnorm_source.py",
    "compiler/semantic/linalg/registered_source.py",
    "compiler/semantic/linalg/rmsnorm_source.py",
    "compiler/semantic/linalg/scalar.py",
    "compiler/semantic/linalg/scalar_regions.py",
    "compiler/semantic/linalg/silu_source.py",
    "compiler/semantic/linalg/softmax_source.py",
    "compiler/semantic/linalg/softmax_value_source.py",
    "compiler/semantic/linalg/tensor_assembly.py",
    "compiler/semantic/model.py",
    "compiler/semantic/predicate_values.py",
    "compiler/semantic/prequantized_bias.py",
    "compiler/semantic/reference.py",
    "compiler/semantic/rules.py",
    "compiler/semantic/scalar_reference.py",
    "compiler/semantic/search/__init__.py",
    "compiler/semantic/search/allocation.py",
    "compiler/semantic/search/allocation_schedule.py",
    "compiler/semantic/search/buffers.py",
    "compiler/semantic/search/exploration.py",
    "compiler/semantic/search/extraction.py",
    "compiler/semantic/search/verification.py",
    "compiler/semantic/source_preparation.py",
    "compiler/semantic/target_binding.py",
    "compiler/semantic/tensor_constants.py",
    "compiler/semantic/tiled_rmsnorm.py",
    "compiler/semantic/tiled_softmax.py",
    "compiler/semantic/view_descriptors.py",
    "compiler/transforms/dense/ir.py",
    "compiler/transforms/input_widening.py",
    "compiler/transforms/operand_layouts.py",
    "compiler/transforms/operand_permutations.py",
    "compiler/transforms/panels.py",
    "compiler/transforms/pointwise.py",
    "compiler/transforms/reduction.py",
    "compiler/transforms/result_narrowing.py",
    "compiler/transforms/slice_sizes.py",
    "compiler/transforms/tiling.py",
    "development/targetgen/cli.py",
    "development/targetgen/execution/host/admission.py",
    "development/targetgen/execution/native/native_application.py",
    "development/targetgen/execution/native/native_application_cli.py",
    "development/targetgen/execution/native/native_codegen_cli.py",
    "target/soc.py",
    "target/system.py",
    "runtime/host_math_supplier/README.md",
    "runtime/host_math_supplier/__init__.py",
    "runtime/host_math_supplier/build.py",
    "runtime/host_math_supplier/embedded.py",
    "runtime/host_math_supplier/hardware_sqrt.c",
    "runtime/host_math_supplier/host-unary-math-policy.json",
    "runtime/host_math_supplier/host_math.h",
    "runtime/host_math_supplier/round_support.c",
    "runtime/host_math_supplier/supplier_support.h",
    "runtime/host_math_supplier/third_party/core-math/LICENSE",
    "runtime/host_math_supplier/third_party/core-math/REVISION",
    "runtime/host_math_supplier/third_party/core-math/cos.c",
    "runtime/host_math_supplier/third_party/core-math/cosf.c",
    "runtime/host_math_supplier/third_party/core-math/dint.h",
    "runtime/host_math_supplier/third_party/core-math/erff.c",
    "runtime/host_math_supplier/third_party/core-math/exp.c",
    "runtime/host_math_supplier/third_party/core-math/expf.c",
    "runtime/host_math_supplier/third_party/core-math/log.c",
    "runtime/host_math_supplier/third_party/core-math/logf.c",
    "runtime/host_math_supplier/third_party/core-math/pow/dint.h",
    "runtime/host_math_supplier/third_party/core-math/pow/pow.c",
    "runtime/host_math_supplier/third_party/core-math/pow/pow.h",
    "runtime/host_math_supplier/third_party/core-math/pow/qint.h",
    "runtime/host_math_supplier/third_party/core-math/powf.c",
    "runtime/host_math_supplier/third_party/core-math/rsqrtf.c",
    "runtime/host_math_supplier/third_party/core-math/sin.c",
    "runtime/host_math_supplier/third_party/core-math/sinf.c",
    "runtime/host_math_supplier/third_party/core-math/tanhf.c",
    "verification/references/saved_model.py",
    "compiler/frontend/patterns/linalg_batch_norm.py",
    "compiler/frontend/patterns/linalg_bf16_sigmoid.py",
    "compiler/frontend/patterns/linalg_window_max.py",
    "compiler/frontend/policy_inputs.py",
    "compiler/frontend/saved_program_signature.py",
    "compiler/lower/c/host_ops/batch_norm.py",
    "compiler/lower/c/host_ops/sigmoid.py",
    "compiler/lower/c/host_ops/gelu.py",
    "compiler/lower/c/host_ops/f32_math.py",
    "compiler/lower/c/host_ops/layer_norm.py",
    "compiler/lower/c/host_ops/affine_contraction.py",
    "compiler/lower/c/host_ops/window_max.py",
    "compiler/quantization/policy_generation.py",
    "compiler/semantic/linalg/cnn.py",
    "compiler/semantic/linalg/source_compositions.py",
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
    merlin = Path(__file__).resolve().parent.parent.parent
    try:
        return {name: _hash_file(merlin / name) for name in _COMPILER_SOURCE_FILES}
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


# ---- target_binding ----------------------------------------------------------------------------


class NativeCompilationError(RuntimeError):
    """A target binding's diagnosed native search failure, without fallback."""

    _STATUSES = {
        "unsupported_semantics",
        "unqualified_target",
        "infeasible_candidate",
        "search_timeout",
        "resource_limit",
        "tool_unavailable",
        "numerical_mismatch",
        "execution_failure",
        "compile_error",
    }

    def __init__(self, status: str, reason: str) -> None:
        if status not in self._STATUSES or not reason:
            raise ValueError("native compilation failure needs a known status and reason")
        self.status = status
        self.reason = reason
        super().__init__(f"{status}: {reason}")


class NativeTargetBinding(Protocol):
    # Providers may additionally implement plan_boundary_abi(request) for
    # explicit --abi auto requests. It returns fixed_inputs/fixed_outputs in
    # provider units and must check its storage/layout requirements.
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
    persisted_path = staged / "manifest.json"
    if staged.is_symlink() or persisted_path.is_symlink() or not persisted_path.is_file():
        raise ValueError("selected target did not emit a native manifest")
    persisted = json.loads(persisted_path.read_text())
    if (
        persisted != json.loads(json.dumps(manifest))
        or manifest.get("engine") != engine
        or (manifest.get("request_digest") != request_digest or manifest.get("target_identity") != target_identity)
    ):
        raise ValueError("native emitted artifact identity differs from checked request or target")
    stage_artifacts = manifest.get("stage_artifacts")
    if stage_artifacts is not None:
        if not isinstance(stage_artifacts, dict) or not stage_artifacts:
            raise ValueError("native stage artifacts need a nonempty path-to-digest map")
        for relative, digest in stage_artifacts.items():
            if not isinstance(relative, str) or not relative:
                raise ValueError("native stage artifact has an unsafe path")
            path = Path(relative)
            if (
                path.is_absolute()
                or not path.parts
                or path.as_posix() != relative
                or "\\" in relative
                or ".." in path.parts
                or relative == "manifest.json"
            ):
                raise ValueError("native stage artifact has an unsafe path")
            artifact = staged / path
            if (
                not isinstance(digest, str)
                or any((staged / Path(*path.parts[:depth])).is_symlink() for depth in range(1, len(path.parts) + 1))
                or not artifact.is_file()
                or not artifact.resolve().is_relative_to(staged.resolve())
                or digest_file(artifact)[0] != digest
            ):
                raise ValueError(f"native stage artifact changed or is missing: {relative}")
    if manifest.get("artifact_kind") == "host_application":
        regions = manifest.get("regions")
        segments = staged / "segments"
        if (manifest.get("segments") != [] or not isinstance(regions, list) or not regions
                or any(not isinstance(row, dict) or row.get("placement") != "host"
                       or row.get("native_programs") != [] for row in regions)
                or not isinstance(stage_artifacts, dict) or not stage_artifacts
                or not segments.is_dir() or segments.is_symlink() or any(segments.iterdir())
                or (staged / "program.bin").exists() or (staged / "execution_plan.json").exists()):
            raise ValueError("host-only application must contain authenticated host regions and no device programs")
        return
    if manifest.get("artifact_kind") == "program_set":
        if (staged / "program.bin").exists() or (staged / "execution_plan.json").exists():
            raise ValueError("program set must not masquerade as one native binary")
        rows = manifest.get("segments")
        if not isinstance(rows, list) or not rows:
            raise ValueError("native program set has no segment identities")
        seen: set[str] = set()
        branches: dict[str, set[str]] = {"segments": set()}
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("index") != index:
                raise ValueError("native program set has invalid segment order")
            relative = row.get("path")
            if (
                not isinstance(relative, str)
                or len(Path(relative).parts) < 2
                or Path(relative).parts[0] != "segments"
                or Path(relative).as_posix() != relative
                or "\\" in relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or relative in seen
                or any(relative.startswith(previous + "/") or previous.startswith(relative + "/") for previous in seen)
            ):
                raise ValueError("native program set has an unsafe or repeated segment path")
            seen.add(relative)
            parts = Path(relative).parts
            for depth in range(1, len(parts)):
                parent = Path(*parts[:depth]).as_posix()
                branches.setdefault(parent, set()).add(parts[depth])
            segment = staged / relative
            if (any((staged / Path(*parts[:depth])).is_symlink() for depth in range(1, len(parts) + 1))
                    or not segment.is_dir() or not segment.resolve().is_relative_to(staged.resolve())):
                raise ValueError("native program set segment is missing or escapes its package")
            segment_manifest = segment / "manifest.json"
            if (
                segment_manifest.is_symlink()
                or not segment_manifest.is_file()
                or row.get("manifest_sha256") != hashlib.sha256(segment_manifest.read_bytes()).hexdigest()
            ):
                raise ValueError("native program set segment manifest changed")
            if not isinstance(row.get("request_digest"), str) or not row["request_digest"]:
                raise ValueError("native program set segment has no request identity")
            detail = json.loads(segment_manifest.read_text())
            if detail.get("artifact_kind") == "program_set" or row.get("binary_sha256") != detail.get("binary_sha256"):
                raise ValueError("native program set segment binary identity changed")
            verify_native_publication(
                segment,
                detail,
                engine=engine,
                request_digest=row["request_digest"],
                target_identity=target_identity,
            )
        if not (staged / "segments").is_dir() or (staged / "segments").is_symlink():
            raise ValueError("native program set segment root is missing or linked")
        for relative, children in branches.items():
            for item in (staged / relative).iterdir():
                if item.name in children and item.is_dir() and not item.is_symlink():
                    continue
                # A multi-image region may retain source/plan metadata beside
                # its image leaves. The parent publication must authenticate
                # each such regular file; extra directories remain refused.
                member = item.relative_to(staged).as_posix()
                if (relative != "segments" and item.name not in {"manifest.json", "program.bin", "execution_plan.json"}
                        and isinstance(stage_artifacts, dict) and member in stage_artifacts
                        and not item.is_symlink() and item.is_file()):
                    continue
                raise ValueError("native program set has an unaccounted segment")
        return
    if manifest.get("artifact_kind") not in {None, "program"}:
        raise ValueError("unknown native publication kind")
    binary = staged / "program.bin"
    if (
        binary.is_symlink()
        or not binary.is_file()
        or binary.stat().st_size == 0
        or manifest.get("binary_sha256") != hashlib.sha256(binary.read_bytes()).hexdigest()
    ):
        raise ValueError("native emitted binary differs from checked manifest")
    plan = staged / "execution_plan.json"
    expected_plan_digest = manifest.get("execution_plan_sha256")
    if plan.is_symlink() or plan.exists() != (expected_plan_digest is not None):
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
