"""Generation-time captures through the admitted sealed runner.

Verified Phase 0 admits a capture only from the sealed Model2MLIR runner, with its attestation
re-checked against the bytes on disk (operator policy, 2026-10-01). Capsules written from PyTorch
during generation (operation probes, derived micro models) capture at generation time, so in verified
mode each such capture is preselected, issued in the sandbox, replayed and attested exactly like a
declared iteration capture, and the capsule records the attestation it was built from. Without a
selected sealed runtime, generation-time capture keeps its historical diagnostic path.

The selection travels to the generation process in one environment variable, set when the run is
frozen. The capture request is checked against what the sealed policy can express; anything else --
a declared loader environment, a pinned interpreter, a quantization scheme instead of a recipe, an
already-materialized model -- fails closed instead of being captured outside the seal.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from merlin.capture.artifacts import M2MUnavailable
from merlin.capture.pytorch import PytorchRefSource
from merlin.experiments.capture_execution.m2m_origin import (
    frozen_selector,
    git_origin,
    selected_source_origin,
    verify_frozen_selector,
)
from merlin.experiments.capture_execution.m2m_plan import _source_tree, _venv_home
from merlin.experiments.capture_execution.m2m_sandbox import (
    _capture_api_missing,
    _frontend_trace_api_missing,
    _static_integer_reference_api_missing,
)
from merlin.experiments.capture_execution.sealed_static import _canonical_path, _file_digest

# ---- sealed_generation -------------------------------------------------------------------------

CONFIG_ENV = "MERLIN_PHASE0_SEALED_CAPTURE"
_BUNDLE_ONLY = frozenset({"capture_receipt.json", "model.mlir"})


def configured() -> dict[str, Any] | None:
    raw = os.environ.get(CONFIG_ENV)
    if not raw:
        return None
    config = json.loads(raw)
    required = {"m2m_root", "venv", "runs_root"}
    if not isinstance(config, dict) or not required <= set(config):
        raise ValueError(f"{CONFIG_ENV} must name {sorted(required)}")
    if "execution_timeout_seconds" in config:
        capture_timeout(config["execution_timeout_seconds"])
    return config


def capture_source() -> PytorchRefSource:
    """The capture source generation writes PyTorch capsules with: sealed when one is selected."""
    config = configured()
    return SealedCaptureSource(config) if config is not None else PytorchRefSource()


def bind_source(entries: list[dict[str, Any]], *, verified: bool) -> PytorchRefSource | None:
    """Select one capture adapter before any member is written.

    A materialized capture already carries its own sealed execution receipt. All
    generation-time frontend captures in a verified corpus must instead use the
    run's one selected sealed adapter; ambient fallback is diagnostic only.
    """
    needs_capture = any(
        not entry.get("materialized_capture")
        and (
            entry.get("kind") == "model"
            or entry.get("op") == "model"
            or entry.get("source") == "pytorch"
            or entry.get("pytorch_ref")
        )
        for entry in entries
    )
    if not needs_capture:
        return None
    source = capture_source()
    if verified and not isinstance(source, SealedCaptureSource):
        raise ValueError("verified Phase 0 requires a selected sealed Model2MLIR runtime before capsule writes")
    if verified and not source.available():
        raise ValueError("verified Phase 0 selected a sealed Model2MLIR runtime that is unavailable")
    return source


class SealedCaptureSource(PytorchRefSource):
    """A :class:`PytorchRefSource` whose every capture is a fresh, attested sealed run."""

    def __init__(self, config: dict[str, Any]):
        venv = Path(config["venv"])
        super().__init__(m2m_dir=Path(config["m2m_root"]), python=venv / "bin/python")
        self.config = dict(config)
        self._local = threading.local()
        self._staging_lock = threading.Lock()
        self._slot_lock = threading.Lock()
        self.parallel_safe = True
        if config.get("tmp_root"):
            # Capture work directories (and every other temporary file of this generation process)
            # live on the run's own filesystem rather than whatever host /tmp happens to have room.
            Path(config["tmp_root"]).mkdir(parents=True, exist_ok=True)
            tempfile.tempdir = str(config["tmp_root"])

    def _staged_merlin(self, package: Path, schemas: Path) -> tuple[Path, Path]:
        """A writable copy of the running Merlin package and its schemas, staged once per process.

        A frozen run executes from a read-only source snapshot, and the sealed runner bundles the
        selected schemas into its copy of the package, which a read-only package refuses. The copy
        holds the same bytes (checked here) with owner-writable modes; the sealed plan then binds the
        staged tree exactly as it binds any selected package.
        """
        staged = Path(self.config["runs_root"]) / "_merlin"
        package_copy, schemas_copy = staged / "src/merlin", staged / "src/merlin/resources/schemas"
        with self._staging_lock:
            if not getattr(self, "_staged", False):
                for source, destination in ((package, package_copy), (schemas, schemas_copy)):
                    if destination.exists():
                        shutil.rmtree(destination)
                    shutil.copytree(source, destination, symlinks=False)
                    for path in [destination, *destination.rglob("*")]:
                        path.chmod(path.stat().st_mode | 0o200)
                    originals = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file())
                    copies = sorted(p.relative_to(destination) for p in destination.rglob("*") if p.is_file())
                    if originals != copies or any(
                        (source / name).read_bytes() != (destination / name).read_bytes() for name in originals
                    ):
                        raise M2MUnavailable("the staged Merlin package differs from the running one")
                self._staged = True
        return package_copy, schemas_copy

    @property
    def attestations(self) -> list[dict]:
        """Only captures made by this writer thread can attest its current capsule."""
        if not hasattr(self._local, "attestations"):
            self._local.attestations = []
        return self._local.attestations

    def _cache_slot(self, *args, **kwargs):  # noqa: ANN002, ANN003 -- every sealed capture is fresh
        return None

    def _launch_worker(self, cmd, *, env, **request):  # noqa: ANN001
        from merlin.common.paths import module_source_path, schemas_dir
        from merlin.development.testgen.sealed_capture.attestation import attest_sealed_m2m
        from merlin.development.testgen.sealed_capture.selection import issue, select, verify
        from merlin.experiments.capture_execution import sealed_static as native_integer

        native = self.config.get("source_mode") == native_integer.MODE
        if self.config.get("source_mode") not in (None, native_integer.MODE):
            raise M2MUnavailable("sealed capture source_mode is unsupported")
        if native and (
            not request.get("already_quantized")
            or request.get("recipe_path")
            or request.get("scheme")
            or request.get("stage_fp32")
            or request.get("agreement_tolerance")
        ):
            raise M2MUnavailable("native integer source mode requires unchanged integer source without transformations")
        if request.get("scheme") or (request.get("already_quantized") and not native):
            raise M2MUnavailable("a sealed capture selects a recipe or a float dtype, never a scheme or a prior model")
        if request.get("declared_env"):
            raise M2MUnavailable("a sealed capture admits no declared loader environment")
        if Path(request["interpreter"]) != self.python:
            raise M2MUnavailable("a sealed capture runs only the selected runtime's interpreter")
        stage_fp32 = request.get("stage_fp32", False)
        if type(stage_fp32) is not bool:
            raise M2MUnavailable("stage_fp32 must be an explicit boolean")
        if (
            stage_fp32
            and str(request["dtype"]) not in {"fp32", "f32"}
            and not (str(request["dtype"]) == "int8" and request.get("recipe_path") is not None)
        ):
            raise M2MUnavailable("FP32 staging requires a float capture or selected int8 recipe")
        root = Path(self.config["runs_root"])
        root.mkdir(parents=True, exist_ok=True)
        with self._slot_lock:
            slot = root / f"{len(list(root.iterdir())):04d}-{request['op']}-{request['dtype']}"
            slot.mkdir()
        workload = slot / "workload"
        workload.mkdir()
        shutil.copyfile(request["loader_py"], workload / "loader.py")
        recipe = None
        if request.get("recipe_path") is not None:
            recipe = slot / "quant_recipe.json"
            shutil.copyfile(request["recipe_path"], recipe)
        tolerance = request.get("agreement_tolerance")
        merlin_root, schemas_root = self._staged_merlin(Path(module_source_path("merlin")).parent, Path(schemas_dir()))
        identity = select(
            m2m_root=Path(self.config["m2m_root"]),
            frozen_origin=self.config.get("frozen_origin"),
            workload_root=workload,
            worker=merlin_root / "capture/preparation/worker.py",
            venv=Path(self.config["venv"]),
            schemas_root=schemas_root,
            run_dir=slot / "run",
            output_dir=slot / "selection",
            dtype=str(request["dtype"]),
            recipe=recipe,
            bwrap_binary=Path(self.config["bwrap"]) if self.config.get("bwrap") else None,
            # Absent: the historical fixed 120 s and unchanged selection bytes.
            execution_timeout_seconds=self.config.get("execution_timeout_seconds"),
            worker_options={
                **({"source_mode": native_integer.MODE} if native else {}),
                **({"agreement_tolerance": [float(value) for value in tolerance]} if tolerance else {}),
                **({"stage_fp32": True} if stage_fp32 else {}),
            }
            or None,
            **({"runtime_reuse": self.config["runtime_reuse"]} if "runtime_reuse" in self.config else {}),
        )
        selection = Path(identity["path"])
        expected_package = self.config.get("package")
        if expected_package is not None:
            from merlin.development.testgen.sealed_capture.selection import load

            selected_package = load(selection, expected_sha256=identity["sha256"])["plan"]["selected_trees"]["m2m"]
            if selected_package != expected_package:
                raise M2MUnavailable("the Model2MLIR package differs from the one selected when the run was frozen")
        issue(selection, expected_sha256=identity["sha256"])
        model = slot / "run/capture/model.mlir"
        replay = verify(selection, expected_sha256=identity["sha256"], model_path=model)
        attestation = attest_sealed_m2m(replay, selection_path=selection, model_path=model)
        capture = slot / "run/capture"
        if (capture / "linalg.mlir").read_bytes() != model.read_bytes():
            raise M2MUnavailable("sealed capture's program differs from the attested model bytes")
        workdir = Path(request["workdir"])
        # The worker's own result files, as an unsealed capture leaves them. The bundle's receipt and
        # model copy stay in the sealed run: the attestation, not a copy, is what binds them.
        for member in sorted(capture.iterdir()):
            if member.name in _BUNDLE_ONLY or member.is_symlink() or not member.is_file():
                continue
            shutil.copyfile(member, workdir / member.name)
        meta = json.loads((workdir / "meta.json").read_text(encoding="utf-8"))
        # A materialized bundle records its members relative to itself; resolve them in the copy.
        for key in ("frontend_trace", "framework_catalog"):
            row = meta.get(key)
            if isinstance(row, dict) and isinstance(row.get("path"), str) and not Path(row["path"]).is_absolute():
                row["path"] = str(workdir / row["path"])
        meta["capture_execution_attestation"] = attestation
        (workdir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self.attestations.append(attestation)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def verified_capture_failure(capsule: dict, *, require_attestation: bool = False) -> str | None:
    """Why the original capture seal is unavailable, or ``None``.

    The optional stricter mode always reopens the complete original attestation,
    including materialized captures. It never consumes a cached verification.
    """
    from merlin.development.testgen.sealed_capture.attestation import (
        AttestationNotVerified,
        require_verified_execution,
    )

    if capsule.get("materialized_capture") and not require_attestation:
        return None  # a declared iteration capture, attested at derivation and re-checked by evidence
    if not require_attestation and not (capsule.get("pytorch_ref") or capsule.get("source") == "pytorch"):
        return None
    attestations = capsule.get("capture_execution_attestations") or []
    raw = (capsule.get("frontend_trace") or {}).get("capture_mlir_sha256")
    if not attestations:
        return "generation-time capture carries no sealed-runner attestation"
    try:
        for attestation in attestations:
            require_verified_execution(attestation)
    except AttestationNotVerified as exc:
        return str(exc)
    attested_models = set()
    for row in attestations:
        capture = row.get("capture") or {}
        if capture.get("kind") == "session":
            # require_verified_execution above checks the entire original roster;
            # this only selects a member from that complete verified owner.
            attested_models.update(program.get("model_sha256") for program in capture.get("programs", []))
        else:
            attested_models.add(capture.get("model_sha256"))
    if raw is not None and raw not in attested_models:
        return "the capsule's captured program is not the attested capture"
    return None


# ---- m2m_runtime -------------------------------------------------------------------------------
# Explicit host Model2MLIR selection for diagnostic frozen derivations.
#
# The selected M2M source package is copied into the run. The interpreter and
# its dependency trees remain on the host, but their exact bytes and membership
# are checked before freezing and every launch/resume. This is a byte-checked
# selection on a managed host, *not* a sandbox or Phase 0 capture admission.

SCHEMA = "merlin.phase0.selected_m2m_runtime.v1"
_MAX_SOURCE_COPY_BYTES = 15_000_000_000


def read_runtime_reuse(path: Path) -> dict:
    """Read one explicit ordinary, bounded selection; the runtime verifier owns its meaning."""
    from merlin.development.testgen.generation.staging import read_mapping

    return read_mapping(path, subject="capture runtime selection")


def _workload_names(root: Path, synth_profile: Path | None) -> tuple[str, ...]:
    if synth_profile is None or not synth_profile.is_file():
        return ()
    from merlin.experiments.corpus.capsules.model_inputs import resolve_model_loader

    document = yaml.safe_load(synth_profile.read_bytes()) or {}
    if not isinstance(document, dict) or not isinstance(document.get("capsules", []), list):
        raise ValueError("selected synthesis profile has no valid capsule list")
    names = set()
    for entry in document.get("capsules", []):
        if not isinstance(entry, dict):
            raise ValueError("selected synthesis profile has a malformed capsule")
        if entry.get("materialized_capture") or entry.get("micro_model"):
            continue
        if entry.get("kind") != "model" and entry.get("op") != "model":
            continue
        from merlin.development.testgen.public_sources.boundaries import SCHEMA_V3, selected_loader

        if (entry.get("public_boundary") or {}).get("schema") == SCHEMA_V3:
            # These exact generated sources belong to the pinned synthesis
            # input tree, not the independently selected Model2MLIR package.
            selected_loader(entry, synth_profile.parent)
            continue
        loader = resolve_model_loader(entry, root)
        try:
            relative = loader.relative_to(root / "workloads")
        except ValueError as exc:
            raise ValueError(
                "live model loader is outside selected Model2MLIR workloads; materialize it first"
            ) from exc
        if len(relative.parts) != 2 or relative.name != "loader.py" or not loader.is_file():
            raise ValueError(f"selected Model2MLIR workload loader is absent or indirect: {loader}")
        names.add(relative.parts[0])
    return tuple(sorted(names))


def _check_workload_declarations(root: Path, names: tuple[str, ...], python: Path) -> None:
    for name in names:
        directory = root / "workloads" / name
        declaration = directory / "capture.toml"
        if declaration.is_symlink():
            raise ValueError(f"selected Model2MLIR capture declaration is indirect: {declaration}")
        document = tomllib.loads(declaration.read_text()) if declaration.is_file() else {}
        configured = document.get("venv")
        if configured:
            path = Path(str(configured))
            path = path if path.is_absolute() else directory / path
            if (path / "bin/python").absolute() != python:
                raise ValueError(
                    f"Model2MLIR workload {name} selects another Python; materialize it or select a separate runtime"
                )
        if document.get("upstream"):
            raise ValueError(f"Model2MLIR workload {name} selects external source; materialize its capture first")
        locations = document.get("env") or {}
        if not isinstance(locations, dict):
            raise ValueError(f"Model2MLIR workload {name} has an invalid capture environment")
        if any(isinstance(value, str) and Path(value).is_dir() for value in locations.values()):
            raise ValueError(f"Model2MLIR workload {name} selects an unbound external directory")


def observe(
    root: Path,
    python: Path,
    *,
    synth_profile: Path | None = None,
    workload_names: tuple[str, ...] | None = None,
    require_source_origin: bool = False,
) -> dict:
    """Bind the source package, venv and base Python selected by an operator."""
    root = _canonical_path(Path(root), exists=True)
    python = Path(python)
    if python.name != "python":
        raise ValueError("selected Model2MLIR Python must be named bin/python")
    python = _canonical_path(python.parent, exists=True) / python.name
    package = root / "m2m"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"selected Model2MLIR package is absent: {package}")
    names = _workload_names(root, synth_profile) if workload_names is None else workload_names
    if len(names) != len(set(names)) or any(Path(name).name != name or name in (".", "..") for name in names):
        raise ValueError("invalid selected Model2MLIR workload names")
    _check_workload_declarations(root, names, python)
    workloads = {name: _source_tree(root / "workloads" / name) for name in names}
    package_inventory = _source_tree(package, skip_python_cache=True)
    source_origin = (
        selected_source_origin(
            root,
            package_inventory,
            _source_tree(package, skip_python_cache=True, readonly_modes=True),
        )
        if require_source_origin
        else None
    )
    missing_capture_api = _capture_api_missing(root)
    missing_frontend_trace_api = _frontend_trace_api_missing(root)
    missing_static_integer_api = _static_integer_reference_api_missing(root)
    copy_bytes = package_inventory["bytes"] + sum(row["bytes"] for row in workloads.values())
    if copy_bytes > _MAX_SOURCE_COPY_BYTES:
        raise ValueError("selected Model2MLIR package and workload bytes exceed the 15 GB source-copy limit")
    return {
        "schema": SCHEMA,
        "status": "diagnostic_host_runtime",
        "root": str(root),
        "python": str(python),
        "package": package_inventory,
        **({"source_origin": source_origin} if source_origin is not None else {}),
        "same_conversion_capture_api": {
            "status": "available_for_sealed_preflight" if not missing_capture_api else "incompatible",
            "missing": list(missing_capture_api),
            "phase0_admission": "not_granted",
        },
        "frontend_trace_api": {
            "status": "available_for_sealed_preflight" if not missing_frontend_trace_api else "incompatible",
            "missing": list(missing_frontend_trace_api),
            "phase0_admission": "not_granted",
        },
        "static_integer_reference_api": {
            "status": "available_for_sealed_preflight" if not missing_static_integer_api else "incompatible",
            "missing": list(missing_static_integer_api),
            "phase0_admission": "not_granted",
        },
        "workloads": workloads,
        "source_copy_bytes": copy_bytes,
        **_runtime(python),
        "phase0_admission": "not_granted",
    }


def _runtime(python: Path) -> dict:
    venv = python.parent.parent
    if python != venv / "bin/python" or not (venv / "pyvenv.cfg").is_file():
        raise ValueError("selected Model2MLIR Python must be a venv's bin/python")
    base = _venv_home(venv).resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("selected Model2MLIR Python is absent or not executable")
    if python.resolve() != (base / "bin/python3.12").resolve():
        raise ValueError("selected Model2MLIR venv does not use its declared base Python")
    return {
        "base": str(base),
        "venv": _source_tree(venv, skip_lib64=True),
        "base_python": _source_tree(base),
        "python_sha256": _file_digest(python),
    }


def stage(selection: dict, destination: Path) -> dict:
    """Copy selected package bytes once; keep the audited host runtime explicit."""
    if (
        selection.get("schema") != SCHEMA
        or observe(
            Path(selection["root"]),
            Path(selection["python"]),
            workload_names=tuple(selection["workloads"]),
            require_source_origin="source_origin" in selection,
        )
        != selection
    ):
        raise ValueError("selected Model2MLIR runtime changed before freezing")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("frozen Model2MLIR source destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    required_space = selection["source_copy_bytes"] + max(64_000_000, selection["source_copy_bytes"] // 10)
    if required_space > shutil.disk_usage(destination.parent).free:
        raise ValueError("selected Model2MLIR source exceeds available run storage")
    destination.mkdir(parents=True)
    shutil.copytree(
        Path(selection["root"]) / "m2m",
        destination / "m2m",
        symlinks=False,
        ignore=lambda _directory, names: {name for name in names if name == "__pycache__"},
    )
    if _source_tree(destination / "m2m") != selection["package"]:
        raise ValueError("selected Model2MLIR source changed while staging")
    for name, expected in selection["workloads"].items():
        source = Path(selection["root"]) / "workloads" / name
        copied = destination / "workloads" / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, copied, symlinks=False)
        if _source_tree(copied) != expected:
            raise ValueError(f"selected Model2MLIR workload changed while staging: {name}")
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    for path in (p for p in destination.rglob("*") if p.is_dir()):
        path.chmod(path.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    if "source_origin" in selection:
        # The source remains owned by its Git repository through the copy; a
        # concurrent commit or edit is real selected-origin drift.
        if (
            git_origin(Path(selection["root"]), clean=True)["commit"] != selection["source_origin"]["commit"]
            or _source_tree(Path(selection["root"]) / "m2m", skip_python_cache=True) != selection["package"]
        ):
            raise ValueError("selected Model2MLIR origin changed while staging")
    frozen_package = _source_tree(destination / "m2m")
    if "source_origin" in selection and frozen_package != selection["source_origin"]["readonly_package"]:
        raise ValueError("selected Model2MLIR readonly copy differs from source bytes or executable modes")
    return {
        **selection,
        "frozen_root": str(destination),
        "frozen_package": frozen_package,
        "frozen_workloads": {name: _source_tree(destination / "workloads" / name) for name in selection["workloads"]},
    }


def verify(frozen: dict) -> None:
    """Reject changed copied source or live runtime before every attempt."""
    _require_frozen_selection(frozen)
    selected = {
        key: value for key, value in frozen.items() if key not in {"frozen_root", "frozen_package", "frozen_workloads"}
    }
    # The original package/workload source is not reopened: only copied source
    # is authoritative after staging. The selected host venv is still live.
    current = _runtime(Path(frozen["python"]))
    if any(current[key] != selected[key] for key in ("venv", "base_python", "python_sha256", "base")):
        raise ValueError("selected Model2MLIR host runtime changed; freeze a new run")
    verify_frozen_copy(frozen)


def _require_frozen_selection(frozen: dict) -> None:
    if (
        frozen.get("schema") != SCHEMA
        or frozen.get("status") != "diagnostic_host_runtime"
        or frozen.get("phase0_admission") != "not_granted"
    ):
        raise ValueError("unsupported frozen Model2MLIR selection")


def verify_frozen_copy(frozen: dict) -> None:
    """Verify archived source bytes without reopening the historical host runtime."""
    _require_frozen_selection(frozen)
    copied = Path(frozen["frozen_root"])
    if copied.is_symlink() or not copied.is_dir() or copied.stat().st_mode & 0o222:
        raise ValueError("frozen Model2MLIR source root is absent, indirect or writable")
    expected_roots = {"m2m"} | ({"workloads"} if frozen["workloads"] else set())
    if {member.name for member in copied.iterdir()} != expected_roots:
        raise ValueError("frozen Model2MLIR source root membership changed")
    for member in copied.rglob("*"):
        if member.is_symlink() or not (member.is_dir() or member.is_file()) or member.stat().st_mode & 0o222:
            raise ValueError("frozen Model2MLIR source contains an indirect, nonregular or writable member")
    if frozen["workloads"] and {member.name for member in (copied / "workloads").iterdir()} != set(frozen["workloads"]):
        raise ValueError("frozen Model2MLIR workload membership changed")
    if _source_tree(copied / "m2m") != frozen.get("frozen_package"):
        raise ValueError("frozen Model2MLIR source package changed")
    if "source_origin" in frozen:
        origin = frozen["source_origin"]
        if not isinstance(origin, dict) or origin.get("readonly_package") != frozen["frozen_package"]:
            raise ValueError("frozen Model2MLIR source package differs from selected readonly origin")
    if not isinstance(frozen.get("frozen_workloads"), dict) or set(frozen["frozen_workloads"]) != set(
        frozen["workloads"]
    ):
        raise ValueError("frozen Model2MLIR workload membership changed")
    for name, expected in frozen["frozen_workloads"].items():
        if _source_tree(copied / "workloads" / name) != expected:
            raise ValueError(f"frozen Model2MLIR workload changed: {name}")


def environment(frozen: dict) -> dict[str, str]:
    """Route all historical M2M aliases to one selected source and interpreter."""
    root = frozen["frozen_root"]
    python = frozen["python"]
    return {
        "MERLIN_M2M_DIR": root,
        "MERLIN_MODEL2MLIR": root,
        "MERLIN_M2M_PYTHON": python,
        "MERLIN_PHASE0_M2M_REQUIRED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def receipt(frozen: dict) -> bytes:
    """Small, inspectable run artifact identifying this diagnostic selection."""
    return (json.dumps(frozen, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sealed_capture_config(
    selected: dict,
    artifact_root: Path,
    *,
    execution_timeout_seconds: int | None = None,
    bwrap: str | Path | None = None,
    runtime_reuse: dict | None = None,
) -> dict:
    """Bind generation captures to the selected owner, using copied sources after freezing.

    ``execution_timeout_seconds`` is the operator-selected sandbox timeout every generation
    capture is selected with. Absent, the key is omitted so the configuration (and every capture
    selection made from it) keeps its historical bytes and the fixed 120 s timeout.
    """
    preflight_capture_sandbox(bwrap)
    copied = "frozen_root" in selected
    private = artifact_root / "private"
    config = {
        "m2m_root": selected["frozen_root" if copied else "root"],
        "package": selected["frozen_package" if copied else "package"],
        "venv": str(Path(selected["python"]).parent.parent),
        "runs_root": str(private / "sealed-captures"),
        "tmp_root": str(private / "tmp"),
    }
    if copied and "source_origin" in selected:
        selector = frozen_selector(private / "m2m-runtime.json")
        verify_frozen_selector(selector, Path(selected["frozen_root"]), selected["frozen_package"])
        config["frozen_origin"] = selector
    if execution_timeout_seconds is not None:
        config["execution_timeout_seconds"] = capture_timeout(execution_timeout_seconds)
    if bwrap is not None:
        # The operator-selected sandbox binary every generation capture runs under. Absent, the key
        # is omitted and the capture resolves the system one, as before.
        config["bwrap"] = capture_bwrap(bwrap)
    if runtime_reuse is not None:
        config["runtime_reuse"] = _capture_runtime_reuse(selected, runtime_reuse, artifact_root)
    return config


def _capture_runtime_reuse(selected: dict, selection: dict, destination: Path) -> dict:
    """Reopen the exact selected host runtime and existing read-only composition."""
    from merlin.experiments.capture_execution import readonly_runtime
    from merlin.experiments.capture_execution.m2m_plan import _system_libs

    runtime = _runtime(Path(selected["python"]))
    if any(selected.get(key) != value for key, value in runtime.items()):
        raise ValueError("selected Model2MLIR runtime changed before read-only reuse")
    venv = Path(selected["python"]).parent.parent
    site = venv / "lib/python3.12/site-packages"
    torch = next(site.glob("torch/_C*.so"), None)
    numpy = next(site.glob("numpy/_core/_multiarray_umath*.so"), None)
    if torch is None or numpy is None:
        raise ValueError("selected runtime lacks its Torch/NumPy ELF roots")
    plan = {
        # Match sealed_m2m.prepare_plan's declared guest mount, which may
        # differ lexically from the resolved host base already verified above.
        "base": str(_venv_home(venv)),
        "system_libs": [str(path) for path in _system_libs(Path(selected["python"]), torch, numpy)],
        "selected_trees": {"venv": runtime["venv"], "base": runtime["base_python"]},
    }
    # Resolve/preflight precedes creating the fresh run. The nearest existing
    # parent supplies only the filesystem check; this does not create a cache.
    destination = Path(destination)
    while not destination.exists():
        destination = destination.parent
    return readonly_runtime.verify(plan, selection, destination)


def capture_bwrap(value) -> str:
    """An explicit, absolute, executable sandbox binary for generation captures, or refuse."""
    import os

    path = Path(str(value)).expanduser()
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK) or path.name != "bwrap":
        raise ValueError(f"Phase 0 capture bwrap must be an absolute executable named bwrap: {value}")
    return str(path)


def preflight_capture_sandbox(value=None) -> str:
    """Check the actual capture namespace policy before source/runtime staging."""
    from merlin.experiments.capture_execution.m2m_sandbox import _probe_sandbox

    selected = capture_bwrap(value if value is not None else shutil.which("bwrap"))
    _probe_sandbox(Path(selected))
    return selected


def capture_timeout(value) -> int:
    """A generation-capture timeout within the sealed checkpoint-free capture bounds, or refuse."""
    from merlin.experiments.capture_execution import m2m_sandbox as sealed_m2m

    if value is None or not sealed_m2m._selected_timeout_valid(sealed_m2m.SCHEMA, value):
        raise ValueError(
            f"Phase 0 capture timeout must be an integer between {sealed_m2m._TIMEOUT_SECONDS} and "
            f"{sealed_m2m._MAX_SELECTED_CAPTURE_SECONDS} seconds"
        )
    return value
