"""Repo-root and well-known directory resolution.

Small, dependency-light helpers so the rest of the package never hard-codes layout
assumptions. Honors ``MERLIN_REPO_ROOT`` for installed/relocated checkouts; otherwise
resolves the repo root relative to this source file.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root.

    Honors ``MERLIN_REPO_ROOT``; otherwise resolves ``<repo>`` from this file's location
    (``<repo>/merlin/python/merlin/common/paths.py`` -> ``parents[4]``).
    """
    env = os.environ.get("MERLIN_REPO_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4]


def merlin_dir() -> Path:
    """Return ``<repo>/merlin``."""
    return repo_root() / "merlin"


def schemas_dir() -> Path:
    """Return ``<repo>/merlin/schemas`` (mirrors ``common.schemas.schemas_dir``)."""
    env = os.environ.get("MERLIN_SCHEMAS_DIR")
    if env:
        return Path(env)
    return merlin_dir() / "schemas"


def targets_dir() -> Path:
    """Return ``<repo>/merlin/targets``."""
    return merlin_dir() / "targets"


# --- Generated-output roots. All generated products live under a single top-level ``out/`` with
#     three subdirs (runs/ artifacts/ build/). These helpers are the SINGLE source of truth for the
#     root names — callers must never hard-code the literal strings. Honors ``MERLIN_OUT_ROOT`` for
#     relocated/installed checkouts (mirrors ``MERLIN_REPO_ROOT``). ---
def out_dir() -> Path:
    """Return the generated-output root ``<repo>/out`` (honors ``MERLIN_OUT_ROOT``)."""
    env = os.environ.get("MERLIN_OUT_ROOT")
    if env:
        return Path(env)
    return repo_root() / "out"


def runs_dir() -> Path:
    """Return ``<repo>/out/runs`` (aet-managed experiment runs)."""
    return out_dir() / "runs"


def artifacts_dir() -> Path:
    """Return ``<repo>/out/artifacts`` (versioned products, measurements, caches, recaptures)."""
    return out_dir() / "artifacts"


def build_dir() -> Path:
    """Return ``<repo>/out/build`` (compiled trees, baseline toolchains, OOT codegen)."""
    return out_dir() / "build"


# --- External, machine-specific dependency locations (chipyard, toolchains, boards, sibling
#     repos). Repo-INTERNAL paths never go here — the repo finds itself via repo_root(). These
#     differ per machine, so they live in a gitignored ``.env`` at the repo root. Copy
#     ``.env.example`` -> ``.env`` and edit. Resolve in code with ``ext_path('chipyard')``. ---
import functools


@functools.lru_cache(maxsize=1)
def _dotenv() -> dict[str, str]:
    """Parse ``<repo>/.env`` (``KEY=VALUE`` lines, ``#`` comments) into a dict. Cached."""
    out: dict[str, str] = {}
    p = repo_root() / ".env"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def ext_path(name: str) -> Path:
    """Resolve an external, machine-specific dependency location by short key.

    Reads ``MERLIN_EXT_<NAME_UPPERCASE>`` from the process environment (wins) or from the
    gitignored ``<repo>/.env``. Raises ``KeyError`` if unset (copy ``.env.example`` -> ``.env``).
    Example: ``ext_path('chipyard')`` -> reads ``MERLIN_EXT_CHIPYARD``.
    """
    key = f"MERLIN_EXT_{name.upper()}"
    val = os.environ.get(key) or _dotenv().get(key)
    if not val:
        known = sorted(k[len("MERLIN_EXT_"):].lower() for k in _dotenv() if k.startswith("MERLIN_EXT_"))
        raise KeyError(
            f"external path {name!r} unset — set {key} in .env (copy .env.example). Known: {known}"
        )
    return Path(val)
