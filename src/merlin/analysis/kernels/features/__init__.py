"""Feature extraction: NormalizedKernel -> typed feature dict + fired markers.

``extract_all`` runs the marker table once over the kernel text, then composes every
``extract_*`` feature function. Feature functions are pure and source-agnostic (they read the
kernel and the fired-motif dict, never source-specific globals), so the same code handles
XNNPACK-RVV, Autocomp-Gemmini, and Exo-compiled C.

All feature values are *decisions* (bool / small enum / structural count), never tuned
constants such as tile sizes or LMUL values.
"""

from __future__ import annotations

from merlin.analysis.kernels.features.ast_struct import extract_ast_struct
from merlin.analysis.kernels.features.async_ops import extract_async_ops
from merlin.analysis.kernels.features.dispatch import extract_dispatch
from merlin.analysis.kernels.features.epilogue import extract_epilogue
from merlin.analysis.kernels.features.loops import extract_loops
from merlin.analysis.kernels.features.memory import extract_memory
from merlin.analysis.kernels.features.packing import extract_packing
from merlin.analysis.kernels.features.roles import extract_roles
from merlin.analysis.kernels.features.rvv_intrinsics import extract_rvv_intrinsics
from merlin.analysis.kernels.features.shape_regime import extract_shape_regime
from merlin.analysis.kernels.features.target_features import extract_target_features
from merlin.analysis.kernels.features.vector import extract_vector
from merlin.analysis.kernels.markers import fired_markers
from merlin.analysis.kernels.types import NormalizedKernel

_EXTRACTORS = (
    extract_shape_regime,
    extract_packing,
    extract_vector,
    extract_rvv_intrinsics,
    extract_ast_struct,
    extract_memory,
    extract_loops,
    extract_epilogue,
    extract_async_ops,
    extract_target_features,
    extract_roles,
    extract_dispatch,
)


def extract_all(nk: NormalizedKernel) -> tuple[dict, dict[str, list[str]]]:
    """Return ``(features, fired_markers)`` for one kernel.

    ``features`` is the merged dict written to ``kernel_record.features``; ``fired_markers``
    maps motif -> matched substrings.
    """
    fired = fired_markers(nk.raw_text, nk.target)
    features: dict = {}
    for fn in _EXTRACTORS:
        features.update(fn(nk, fired))
    return features, fired
