"""Compiler-side kernel library home.

This directory is the **canonical home** of the kernel ISA manifest
(``manifest.json``) that the merlin compiler generates/consumes. The
simulator-side Programs live in the npu_model submodule under
``third_party/npu_model/npu_model/configs/programs/``, following
npu_model's one-file-per-kernel convention (see
``configs/programs/smolvla_silu.py`` for the canonical template).

Python helpers that used to live here (stitch, manifest loader,
attention_acc generators, fixtures) have moved into npu_model as
underscore-prefixed modules alongside the Program files.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_npu_model_importable() -> None:
    """Legacy convenience: adds third_party/npu_model to sys.path so
    ``import npu_model`` works when this shim is imported from a plain
    Python interpreter without ``uv run`` set up the environment.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "third_party" / "npu_model"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break


_ensure_npu_model_importable()

try:
    import npu_model.configs.isa_definition  # noqa: F401
except ImportError:
    pass
