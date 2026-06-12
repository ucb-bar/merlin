"""Tiling / blocking features.

Decision recorded: *is the computation tiled/blocked* and the *depth* of the iteration nest
(a structural decision about blocking levels), never the tile sizes themselves.

Families with an explicit tiling directive (Gemmini nested scratchpad loops, Triton
``program_id``/``BLOCK_SIZE`` grids, Exo ``divide_loop``/``tile_outer_loops``) are tiled when
their marker fires. For C-vector families (RVV/AVX/NEON) the tiling marker is merely
"a loop exists", so we additionally require either a >=2-deep nest or register blocking
(multiple live accumulators) to avoid calling a flat vector loop "tiled".
"""
from __future__ import annotations

import re

from merlin.kernels.markers import target_family
from merlin.kernels.types import NormalizedKernel

_FOR_RE = re.compile(r"\bfor\s*\(")
_DO_RE = re.compile(r"\bdo\s*\{")
_PYFOR_RE = re.compile(r"\bfor\s+\w+\s+in\s")
_VACC_RE = re.compile(r"\bvacc(\d+)\b")

# Families whose tiling marker is a specific directive, not just loop-presence.
_DIRECTIVE_TILING = {"gemmini", "triton", "exo_schedule"}


def extract_loops(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    text = nk.raw_text
    fam = target_family(nk.target)
    levels = len(_FOR_RE.findall(text)) + len(_DO_RE.findall(text)) + len(_PYFOR_RE.findall(text))
    register_blocked = len(set(_VACC_RE.findall(text))) >= 2
    if "tiling_blocking" not in fired:
        tiling = False
    elif fam in _DIRECTIVE_TILING:
        tiling = True
    else:
        tiling = levels >= 2 or register_blocked
    return {
        "tiling": bool(tiling),
        "tiling_levels": levels,
        "register_blocked": register_blocked,
    }
