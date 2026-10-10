"""Validate an operator-selected package for the Chipyard readiness probe."""

from __future__ import annotations

from pathlib import Path

import yaml


def select_reference_backend(raw: str | None, *, target: str) -> tuple[Path, dict]:
    if not raw:
        raise ValueError("Chipyard readiness requires --reference-backend /absolute/package/path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError("--reference-backend must be an absolute package path")
    path = path.resolve(strict=True)
    manifest_path = path / "manifest.yaml"
    if not manifest_path.is_file():
        raise ValueError(f"selected reference backend has no manifest.yaml: {path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"selected reference backend has an invalid manifest: {path}")
    if manifest.get("target") != target:
        raise ValueError(f"selected reference backend targets {manifest.get('target')!r}, expected {target!r}")
    if manifest.get("artifact_type") != "mlir_oot_target_backend":
        raise ValueError("selected reference backend is not an MLIR OOT target backend")
    if manifest.get("integrity_exempt") is not False:
        raise ValueError("selected reference backend must declare integrity_exempt: false")
    if manifest.get("language") not in ("cpp", "python"):
        raise ValueError("selected reference backend must declare language cpp or python")
    if not isinstance(manifest.get("entrypoints"), dict) or not manifest["entrypoints"].get("tool"):
        raise ValueError("selected reference backend has no tool entrypoint")
    return path, manifest


#: The campaign's elaborated-RTL engine pin (selfcheck refuses a --sim that contradicts it).
REQUIRED_ENGINE_ENV = "MERLIN_REQUIRED_RTL_ENGINE"


def timing_probe_environment(environment: dict, *, engine: str) -> tuple[dict, str | None]:
    """The grade environment for the oracle-TIMING probe, which runs ``engine`` by design.

    A campaign may pin a different engine for certification. That pin governs the run's grades, not
    this calibration probe, so it is lifted for the probe only and returned: the caller must still
    prove the pinned engine itself reaches a real verdict. Returns ``(env, pinned_other_engine)``;
    the input mapping is never mutated.
    """
    probe = dict(environment)
    pinned = (probe.get(REQUIRED_ENGINE_ENV) or "").strip() or None
    if pinned is None or pinned == engine:
        return probe, None
    probe.pop(REQUIRED_ENGINE_ENV)
    return probe, pinned
