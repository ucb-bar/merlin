"""Explicit host Model2MLIR selection for diagnostic frozen derivations.

The selected M2M source package is copied into the run. The interpreter and
its dependency trees remain on the host, but their exact bytes and membership
are checked before freezing and every launch/resume. This is a byte-checked
selection on a managed host, *not* a sandbox or Phase 0 capture admission.
"""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from pathlib import Path

import yaml

from merlin_experiments.capture_execution.m2m_origin import (
    frozen_selector,
    git_origin,
    selected_source_origin,
    verify_frozen_selector,
)
from merlin_experiments.capture_execution.sealed_m2m import (
    _capture_api_missing,
    _frontend_trace_api_missing,
    _source_tree,
    _static_integer_reference_api_missing,
    _venv_home,
)
from merlin_experiments.capture_execution.sealed_static import _canonical_path, _file_digest

SCHEMA = "merlin.phase0.selected_m2m_runtime.v1"
_MAX_SOURCE_COPY_BYTES = 15_000_000_000


def _workload_names(root: Path, synth_profile: Path | None) -> tuple[str, ...]:
    if synth_profile is None or not synth_profile.is_file():
        return ()
    from merlin.targetgen.capsule_source import resolve_model_loader

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
) -> dict:
    """Bind generation captures to the selected owner, using copied sources after freezing.

    ``execution_timeout_seconds`` is the operator-selected sandbox timeout every generation
    capture is selected with. Absent, the key is omitted so the configuration (and every capture
    selection made from it) keeps its historical bytes and the fixed 120 s timeout.
    """
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
    return config


def capture_bwrap(value) -> str:
    """An explicit, absolute, executable sandbox binary for generation captures, or refuse."""
    import os

    path = Path(str(value)).expanduser()
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK) or path.name != "bwrap":
        raise ValueError(f"Phase 0 capture bwrap must be an absolute executable named bwrap: {value}")
    return str(path)


def capture_timeout(value) -> int:
    """A generation-capture timeout within the sealed checkpoint-free capture bounds, or refuse."""
    from merlin_experiments.capture_execution import sealed_m2m

    if value is None or not sealed_m2m._selected_timeout_valid(sealed_m2m.SCHEMA, value):
        raise ValueError(
            f"Phase 0 capture timeout must be an integer between {sealed_m2m._TIMEOUT_SECONDS} and "
            f"{sealed_m2m._MAX_SELECTED_CAPTURE_SECONDS} seconds"
        )
    return value
