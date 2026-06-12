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
