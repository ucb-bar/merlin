"""Resolve explicit model workload declarations and freeze their loader dependencies."""

from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path

from merlin.capture.artifacts import M2MUnavailable
from merlin.capture.pytorch import _m2m_dir
from merlin.common.paths import repo_root


def resolve_model_loader(entry: dict, m2m_dir: str | Path | None = None) -> Path:
    """A model capsule entry names either a workload (``model: small_llama``) or an explicit ``loader``.

    A relative ``loader`` is resolved REPO-ROOT-FIRST, then against the m2m checkout. Both roots are
    needed and the order matters: a whole-model capsule whose network is defined in THIS repo (the
    self-contained case -- the capsule's own ``capsule.pytorch.py`` is the source of record) can then be
    named by the repo-root-relative path this repo already uses for every other in-tree reference, while
    a capsule that names a model2MLIR workload keeps working unchanged. Resolving only against the m2m
    root, as this did, made the in-repo case unnameable: every whole-model capsule had to keep its
    network in an external checkout, which is exactly the dependency that leaves a capsule unbuildable
    from a clean clone.
    """
    root = Path(m2m_dir) if m2m_dir else _m2m_dir()
    if entry.get("loader"):
        p = Path(entry["loader"])
        if p.is_absolute():
            return p
        in_repo = repo_root() / p
        return in_repo if in_repo.is_file() else (root / p)
    name = entry.get("model")
    if not name:
        raise ValueError("model capsule entry needs 'model' (workload name) or 'loader' (path)")
    exact = root / "workloads" / name / "loader.py"
    if exact.is_file():
        return exact
    return _workload_named(root, name) or exact


def _workload_named(root: Path, name: str) -> Path | None:
    """The workload directory a ROSTER name denotes, when it is not spelled the same.

    A roster declares a model (``resnet50``); a workload directory carries the checkpoint revision
    (``resnet50_v1_5``). ``dse_guidance.models`` already resolves the capture direction of this by
    longest-prefix match -- without which a fully captured ResNet reads as ABSENT -- and the capsule
    direction had no resolution at all, so a roster model with a revisioned workload simply could not be
    named by a synthesized capsule.

    Prefix, and only prefix: a revision suffix extends the model name, it never replaces it. Ambiguity
    is an error rather than a pick, because two revisions are two different networks and choosing by
    sort order would silently certify the wrong one. ``None`` when nothing matches, so the caller
    reports the path it expected rather than a guess.
    """
    wroot = root / "workloads"
    if not wroot.is_dir():
        return None
    hits = sorted(d for d in wroot.iterdir() if d.is_dir() and d.name.startswith(name) and (d / "loader.py").is_file())
    if len(hits) == 1:
        return hits[0] / "loader.py"
    if len(hits) > 1:
        raise ValueError(
            f"roster model {name!r} matches {len(hits)} workload directories "
            f"({[d.name for d in hits]}); name the revision in the roster -- picking one by sort order "
            f"would certify a different network than the one declared"
        )
    return None


def resolve_model_workload(entry: dict, m2m_dir: str | Path | None = None) -> str | None:
    """The model2MLIR WORKLOAD DIRECTORY behind this entry's loader, or ``None`` when it has none.

    The roster names a model (``resnet50``) and the workload directory carries the checkpoint revision
    (``resnet50_v1_5``); the entry's own ``model`` string is therefore NOT the key under which the
    model's capture declaration is filed. Resolving the loader first and reading the directory back off
    it means one resolution serves both -- the loader and its declaration always come from the same
    workload, which a second lookup by name could not guarantee.

    ``None`` for an in-repo loader (a derived micro model writes its own): it is not a workload, so it
    declares no capture environment, and inventing one for it would be a fabricated fact.
    """
    root = (Path(m2m_dir) if m2m_dir else _m2m_dir()) / "workloads"
    loader = resolve_model_loader(entry, m2m_dir)
    try:
        loader.absolute().relative_to(root.absolute())
    except ValueError:
        return None
    try:
        rel = loader.resolve().relative_to(root.resolve())
    except (ValueError, OSError) as exc:
        raise M2MUnavailable(f"selected Model2MLIR workload loader escapes its source tree: {loader}") from exc
    return rel.parts[0] if len(rel.parts) >= 2 else None


def _model_capture_config(workload: str, m2m_dir: str | Path | None = None) -> dict:
    """Read the selected workload declaration, never an ambient Model2MLIR checkout.

    The selected source tree is byte-bound by the frozen experiment. A missing declaration
    means no settings were declared; an unreadable or malformed declaration is an error.
    """
    if not workload or Path(workload).name != workload or workload in (".", ".."):
        raise M2MUnavailable(f"invalid Model2MLIR workload name: {workload!r}")
    root = Path(m2m_dir) if m2m_dir is not None else _m2m_dir()
    directory = root / "workloads" / workload
    loader = directory / "loader.py"
    if directory.is_symlink() or not loader.is_file() or loader.is_symlink():
        raise M2MUnavailable(f"selected Model2MLIR workload loader is missing or indirect: {loader}")
    declaration = directory / "capture.toml"
    if declaration.is_symlink():
        raise M2MUnavailable(f"selected Model2MLIR capture declaration is indirect: {declaration}")
    if not declaration.exists():
        return {}
    try:
        document = tomllib.loads(declaration.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise M2MUnavailable(f"cannot read selected Model2MLIR capture declaration: {declaration}") from exc
    declared_env = document.get("env", {})
    if not isinstance(declared_env, dict) or any(not isinstance(value, str) for value in declared_env.values()):
        raise M2MUnavailable(f"selected Model2MLIR capture environment is invalid: {declaration}")
    return document


def model_capture_declaration_identity(workload: str | None, m2m_dir: str | Path | None = None) -> dict:
    """Portable byte identity of the selected workload settings (never a host path)."""
    if not workload:
        return {"status": "not_a_model2mlir_workload"}
    _model_capture_config(workload, m2m_dir)
    root = Path(m2m_dir) if m2m_dir is not None else _m2m_dir()
    relative = Path("workloads") / workload / "capture.toml"
    declaration = root / relative
    if not declaration.is_file():
        return {"status": "absent", "path": relative.as_posix()}
    raw = declaration.read_bytes()
    return {
        "status": "selected",
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def model_capture_env(workload: str | None, m2m_dir: str | Path | None = None) -> dict:
    """Read selected host locations and historical full-fidelity settings for this workload."""
    if not workload:
        return {}
    from merlin.capture import bundle as _bundle

    config = _model_capture_config(workload, m2m_dir)
    locations = {
        str(key): value
        for key, value in config.get("env", {}).items()
        if isinstance(value, str) and value and Path(value).is_dir()
    }
    return {**locations, **_bundle.full_env(workload)}


def model_capture_python(workload: str | None, m2m_dir: str | Path | None = None) -> Path | None:
    """The selected workload's pinned interpreter; reject a missing declared pin."""
    if not workload:
        return None
    config = _model_capture_config(workload, m2m_dir)
    declared = config.get("venv")
    if declared is None or declared == "":
        return None
    if not isinstance(declared, str):
        raise M2MUnavailable(f"selected Model2MLIR workload {workload!r} has an invalid venv declaration")
    root = Path(m2m_dir) if m2m_dir is not None else _m2m_dir()
    directory = root / "workloads" / workload
    value = Path(declared)
    python = (value if value.is_absolute() else directory / value) / "bin" / "python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise M2MUnavailable(f"selected Model2MLIR workload {workload!r} pins a missing Python: {python}")
    return python


def freeze_materialized_loader_profile(selection: dict, destination: Path) -> dict:
    """Publish the exact selected adjacent loader input, for every workload role."""
    import hashlib

    selected, digest = selection.get("profile_path"), selection.get("profile_sha256")
    if selected is None and digest is None:
        return {}
    if not isinstance(selected, str) or not isinstance(digest, str) or len(digest) != 64:
        raise M2MUnavailable("materialized loader profile needs its selected path and digest")
    profile = Path(selected)
    loader = Path(selection.get("loader_path") or "")
    if (
        profile.name != "profile.json"
        or profile.parent != loader.parent
        or any(path.is_symlink() for path in (profile, *profile.parents))
        or not profile.is_file()
    ):
        raise M2MUnavailable("materialized loader profile must be the exact adjacent ordinary input")
    raw = profile.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise M2MUnavailable("materialized loader profile changed during publication")
    destination = Path(destination)
    (destination / "profile.json").write_bytes(raw)
    return {"loader_profile": "profile.json", "loader_profile_sha256": digest}


def freeze_model_loader_dependencies(
    workload: str | None, destination: Path, capture_meta: dict, m2m_dir: str | Path | None = None
) -> str | None:
    """Copy loader-imported source declared by the workload into a capsule-local Python tree.

    A frozen loader that imports an upstream module is not frozen if it still relies on an absolute
    checkout path.  Workload ``capture.toml`` already declares that upstream source.  Preserve only
    Python sources (never checkpoints, datasets, build products, or VCS metadata); the validation worker
    adds their directories to ``sys.path`` without recovering a host path from sanitized loader text.
    """
    import shutil

    from merlin.common.digest import sha256_file

    if not workload:
        return None
    root_value = _model_capture_config(workload, m2m_dir).get("upstream")
    if not root_value:
        return None
    root = Path(str(root_value))
    if not root.is_dir():
        raise M2MUnavailable(f"selected Model2MLIR workload dependency root is missing: {root}")
    sources = []
    observed_hashes = {}
    for item in capture_meta.get("loader_dependency_sources") or []:
        source = Path(str((item or {}).get("path") or ""))
        try:
            source.resolve().relative_to(root.resolve(strict=True))
        except (OSError, ValueError):
            continue
        if item.get("sha256") is not None and (not source.is_file() or source.is_symlink()):
            raise M2MUnavailable(f"loader dependency changed since capture observation: {source}")
        if source.suffix == ".py" and source.is_file() and not source.is_symlink():
            observed = item.get("sha256")
            if observed is not None:
                if sha256_file(source) != observed:
                    raise M2MUnavailable(f"loader dependency changed since capture observation: {source}")
                observed_hashes[source] = observed
            sources.append(source)
    sources = sorted(set(sources))
    if not sources:
        return None
    dependency_root = destination / "capsule.loader_deps"
    if dependency_root.exists():
        shutil.rmtree(dependency_root)
    for source in sources:
        relative = source.resolve(strict=True).relative_to(root.resolve(strict=True))
        target = dependency_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if source in observed_hashes and sha256_file(target) != observed_hashes[source]:
            raise M2MUnavailable(f"loader dependency changed while staging: {source}")
    license_files = sorted(
        p for p in root.iterdir() if p.is_file() and p.name.lower().startswith(("license", "copying"))
    )
    if not license_files:
        raise M2MUnavailable(f"workload dependency source {root} declares no license; refusing to redistribute it")
    shutil.copyfile(license_files[0], dependency_root / license_files[0].name)
    return dependency_root.name
