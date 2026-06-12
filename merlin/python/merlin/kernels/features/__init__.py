"""Feature extraction: NormalizedKernel -> typed feature dict + fired markers.

``extract_all`` runs the marker table once over the kernel text, then composes every
``extract_*`` feature function. Feature functions are pure and source-agnostic (they read the
kernel and the fired-motif dict, never source-specific globals), so the same code handles
XNNPACK-RVV, Autocomp-Gemmini, and Exo-compiled C.

All feature values are *decisions* (bool / small enum / structural count), never tuned
constants such as tile sizes or LMUL values.
"""
from __future__ import annotations

from merlin.kernels.markers import fired_markers
from merlin.kernels.types import NormalizedKernel

from .async_ops import extract_async_ops
from .dispatch import extract_dispatch
from .epilogue import extract_epilogue
from .loops import extract_loops
from .memory import extract_memory
from .packing import extract_packing
from .roles import extract_roles
from .shape_regime import extract_shape_regime
from .target_features import extract_target_features
from .vector import extract_vector

_EXTRACTORS = (
    extract_shape_regime,
    extract_packing,
    extract_vector,
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
    maps motif -> matched substrings, used by :mod:`merlin.kernels.evidence`.
    """
    fired = fired_markers(nk.raw_text, nk.target)
    features: dict = {}
    for fn in _EXTRACTORS:
        features.update(fn(nk, fired))
    return features, fired
