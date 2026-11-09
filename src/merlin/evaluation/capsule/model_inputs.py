"""Capsule evaluation: model inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_MODEL_HOST_SNAPSHOT_ROOT_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"


_MODEL_HOST_SNAPSHOT_REQUIRED_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"


_MODEL_HOST_SNAPSHOT_RECORD_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD"


def _resolve_model_host_lane(target: str, dtype: str):
    """Return ``(experiment, package_path, identity)`` for targeted whole-model compilation.

    Kept as a small seam because descriptor resolution, package-content validation and datatype
    compatibility are one fail-closed decision; callers must not independently recreate part of it.
    """
    from merlin.targetgen.corpora import descriptor_path
    from merlin.targetgen.target_experiment import load_target_experiment

    snapshot_repo, snapshot_identity = _verified_model_host_lane_snapshot()
    descriptor = _model_frozen_input(descriptor_path(target), snapshot_identity)
    if not descriptor.is_file():
        raise ValueError(f"target {target!r} has no experiment descriptor at {descriptor}")
    experiment = load_target_experiment(descriptor)
    if experiment.target != target:
        raise ValueError(f"descriptor {descriptor} names target {experiment.target!r}, not {target!r}")
    # SELECT the lane by dtype, rather than validating whichever single lane the descriptor happened to
    # declare. The cross-check below already refused a mismatch, so a target needing two precisions had
    # no way to express it and simply failed for one of them; the matrix makes the selection the
    # descriptor's to state, and the check below stays as the belt-and-braces that catches a package
    # whose knobs drifted from the profile it is filed under.
    package, identity = experiment.resolve_host_lane(root=snapshot_repo, dtype=dtype)
    if snapshot_identity is not None:
        identity["run_snapshot"] = snapshot_identity

    # An explicit package bypasses compile_rvv's datatype-aware default selection. Reject a known
    # cross-datatype substitution: otherwise an int8 model with an fp32 package sets
    # ``int8_compute=False`` and the numeric outcome is about a different host program. Dtypes such as
    # fp8 deliberately have no scalar strategy; compile_rvv derives their scalar lane from captured IR,
    # so only a strategy it can decide here is gated here.
    from merlin import compile_cli as _compile_cli

    expected_strategy = _compile_cli._DTYPE_STRATEGY.get(dtype)
    if expected_strategy is not None and identity["dtype_strategy"] != expected_strategy:
        raise ValueError(
            f"descriptor host package declares dtype_strategy={identity['dtype_strategy']!r}, but "
            f"model compile_dtype={dtype!r} requires {expected_strategy!r}"
        )
    return experiment, package, identity


def _resolve_model_numeric_policy(target: str | None) -> tuple[dict | None, dict | None]:
    policy, identity, _ = resolve_numeric_policy(target)
    return policy, identity


def resolve_numeric_policy(
    target: str | None, *, descriptor: Path | None = None
) -> tuple[dict | None, dict | None, Any]:
    """Return policy, byte identity and its descriptor through the already-bound snapshot.

    Consumers deriving a binding must use the returned experiment, not reload its mutable original
    path after policy verification. This does not attest external capability facts or tools.
    """
    if target is None:
        return None, None, None
    from merlin.common.paths import repo_root
    from merlin.experiments.corpus.numeric_policy import load_declared_numeric_policy, numeric_profile_path
    from merlin.targetgen.corpora import descriptor_path
    from merlin.targetgen.target_experiment import load_target_experiment

    _, snapshot = _verified_model_host_lane_snapshot()
    source = Path(descriptor) if descriptor is not None else descriptor_path(target)
    experiment = load_target_experiment(_model_frozen_input(source, snapshot))
    if experiment.target != target:
        raise ValueError("model numerical-policy descriptor names a different target")
    original_repo = Path(snapshot["source_repo"]) if snapshot is not None else repo_root()
    declared = numeric_profile_path(experiment.numeric_profile, repo=original_repo)
    frozen = None
    if declared is not None and snapshot is not None:
        frozen = _model_frozen_input(declared, snapshot)
    policy, identity = load_declared_numeric_policy(
        experiment,
        repo=original_repo,
        frozen_profile=frozen,
        frozen_resolver=(lambda source: _model_frozen_input(source, snapshot)) if snapshot is not None else None,
    )
    return policy, identity, experiment


def _model_frozen_input(source: Path, snapshot: dict | None) -> Path:
    """Map only declared frozen bytes after rechecking the host's expected marker."""
    if snapshot is None:
        return source
    from merlin.targetgen.sandbox import bwrap as BW

    root = Path(snapshot["root"])
    marker = root / "snapshot.json"
    import hashlib

    raw = marker.read_bytes()
    if hashlib.sha256(raw).hexdigest() != snapshot["manifest_sha256"]:
        raise ValueError("model host-lane snapshot marker changed during input resolution")
    bundle = _model_snapshot_bundle(json.loads(raw))
    [frozen] = BW.snapshot_input_paths(root.parent / "workspace", bundle, [source], repo=Path(snapshot["source_repo"]))
    return frozen


def _model_snapshot_bundle(manifest: dict) -> dict:
    """Declaration membership from a marker already bound to host-owned provenance."""
    return {key: [{"path": path} for path in manifest.get(key, [])] for key in ("allowed", "host_inputs")}


def _verified_model_host_lane_snapshot() -> tuple[Path | None, dict[str, Any] | None]:
    """Return the verified run-snapshot repo root used for host-lane grading.

    A bwrap experiment sets ``*_REQUIRED`` and points ``*_ROOT`` at its host-only
    ``bundle_inputs`` directory. The aggregate was recomputed immediately before export; resolving and
    hashing the declared host package below joins the bytes mounted for the agent to the bytes passed
    into ``compile_model``. Using the live worktree instead would let a post-launch edit change only the
    grader's compiler.

    Callers outside the bwrap experiment remain compatible: with neither variable set, the descriptor
    resolves against the live repository exactly as before. A declared/required snapshot never falls
    back to that path.
    """
    raw_required = os.environ.get(_MODEL_HOST_SNAPSHOT_REQUIRED_ENV, "").strip().lower()
    if raw_required not in ("", "0", "false", "no", "off", "1", "true", "yes", "on"):
        raise ValueError(f"{_MODEL_HOST_SNAPSHOT_REQUIRED_ENV} has invalid boolean value {raw_required!r}")
    required = raw_required in ("1", "true", "yes", "on")
    raw_root = os.environ.get(_MODEL_HOST_SNAPSHOT_ROOT_ENV, "").strip()
    if not raw_root:
        if required or os.environ.get(_MODEL_HOST_SNAPSHOT_RECORD_ENV):
            raise ValueError(
                f"bwrap model grading requires {_MODEL_HOST_SNAPSHOT_ROOT_ENV}; no verified run snapshot was provided"
            )
        return None, None

    lexical_root = Path(raw_root)
    if not lexical_root.is_absolute() or lexical_root.is_symlink():
        raise ValueError(f"{_MODEL_HOST_SNAPSHOT_ROOT_ENV} must name a non-symlink absolute directory")
    try:
        root = lexical_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"model host-lane run snapshot is missing at {lexical_root}") from exc
    marker = root / "snapshot.json"
    if not root.is_dir() or marker.is_symlink() or not marker.is_file():
        raise ValueError(f"model host-lane run snapshot is incomplete or unsafe at {root}")
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"model host-lane run snapshot marker is unreadable at {marker}") from exc
    from merlin.targetgen.sandbox import bwrap as BW

    BW.require_snapshot_ownership(manifest)
    try:
        expected = json.loads(os.environ.get(_MODEL_HOST_SNAPSHOT_RECORD_ENV, ""))
        source_repo = Path(manifest["repo"])
        if not source_repo.is_absolute() or ".." in source_repo.parts:
            raise ValueError("invalid original snapshot repository")
        BW.verify_snapshot_binding(
            root.parent / "workspace", _model_snapshot_bundle(manifest), expected, repo=source_repo
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("model host-lane snapshot lacks matching host-owned provenance") from exc

    # Preserve the bound aggregate identity here. Mapping declaration/profile bytes through
    # snapshot_input_paths verifies the payload; HostLane.resolve additionally checks the selected
    # compiler package before and after compilation.
    digest = manifest.get("content_sha256")
    n_files = manifest.get("n_files")
    n_bytes = manifest.get("n_bytes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(ch not in "0123456789abcdef" for ch in digest)
        or not isinstance(n_files, int)
        or n_files < 1
        or not isinstance(n_bytes, int)
        or n_bytes < 1
    ):
        raise ValueError("model host-lane run snapshot marker has invalid aggregate identity")

    repo = root / "repo"
    if repo.is_symlink() or not repo.is_dir():
        raise ValueError(f"model host-lane run snapshot has no safe repository payload at {repo}")
    resolved_repo = repo.resolve(strict=True)
    if root not in resolved_repo.parents:
        raise ValueError(f"model host-lane run snapshot repository escapes {root}")
    return resolved_repo, {
        "root": str(root),
        "repo_root": str(resolved_repo),
        "content_sha256": digest,
        "n_files": n_files,
        "n_bytes": n_bytes,
        "version": 4,
        "manifest_sha256": expected["manifest_sha256"],
        "source_repo": manifest.get("repo"),
    }
