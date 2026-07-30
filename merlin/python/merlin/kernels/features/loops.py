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

from merlin.kernels.framework_contracts import load_feature_contract
from merlin.kernels.markers import target_family
from merlin.kernels.types import NormalizedKernel

from ._tokens import count_loops, distinct_registers

# Vector-accumulator register naming convention (a generic register-blocking heuristic, applied to
# every family — not a per-target branch): >=2 distinct live accumulators means register blocking.
_ACC_REGISTER_PREFIX = "vacc"


def extract_loops(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    text = nk.raw_text
    # Whether the tiling marker is an explicit directive (vs mere loop-presence) is data, per ISA
    # family — not an `if fam in {...}` branch.
    tiling_is_directive = bool(
        load_feature_contract(target_family(nk.target)).get("loops", {}).get("tiling_is_directive")
    )
    levels = count_loops(text)
    register_blocked = distinct_registers(text, _ACC_REGISTER_PREFIX) >= 2
    if "tiling_blocking" not in fired:
        tiling = False
    elif tiling_is_directive:
        tiling = True
    else:
        tiling = levels >= 2 or register_blocked
    return {
        "tiling": bool(tiling),
        "tiling_levels": levels,
        "register_blocked": register_blocked,
    }
