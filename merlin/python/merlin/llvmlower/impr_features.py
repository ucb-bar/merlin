"""Fork-scoped, **default-off** compiler-feature registry for ``impr_<target>_vX`` forks.

The baseline RVV compiler (``pipeline.RVV_TRANSFORM_SCHEDULE`` + ``build_rvv_pipeline`` pass list)
is FROZEN — kernel-mining experiments never edit it. KNOB/FLAG/PATTERN improvements ride the
existing ``transform_schedule=``/``cflags_override=`` seams. But PASS- and HEURISTIC-class
improvements need actual compiler code, which still must not perturb the baseline. This registry
is how: each improvement is a NAMED feature with a hook that edits the pipeline pass list and/or
the transform schedule; an ``impr_`` fork's manifest lists the ``compiler_features`` it enables,
threaded through ``build_app`` -> ``lower_*`` -> ``build_rvv_pipeline(features=...)``.

Invariant: with ``features == frozenset()`` (the baseline / any non-impr build), the hooks are
never invoked, so the emitted pipeline string and schedule are **byte-identical** to today
(guarded by ``test_impr_features``). A feature only changes codegen when a fork explicitly enables
it, so it can be measured against the immutable baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ImprFeature:
    """One named, default-off compiler change.

    ``edit_pipeline`` rewrites the ordered MLIR pass list (PASS/HEURISTIC at the pipeline level).
    ``edit_schedule`` rewrites the transform-dialect schedule text (PATTERN at the schedule level
    that goes beyond what a hand-written ``schedule.mlir`` knob expresses). Either may be None.
    Both are pure functions (input -> new value); they must be deterministic.
    """
    name: str
    action_class: str  # "PASS" | "HEURISTIC" | "PATTERN"
    description: str
    edit_pipeline: Callable[[list[str]], list[str]] | None = None
    edit_schedule: Callable[[str], str] | None = None


_REGISTRY: dict[str, ImprFeature] = {}


def register(feature: ImprFeature) -> ImprFeature:
    if feature.name in _REGISTRY:
        raise ValueError(f"duplicate impr feature {feature.name!r}")
    _REGISTRY[feature.name] = feature
    return feature


def get(name: str) -> ImprFeature:
    if name not in _REGISTRY:
        raise KeyError(f"unknown impr feature {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def known() -> list[str]:
    return sorted(_REGISTRY)


def normalize(features) -> frozenset[str]:
    """Accept None / list / set / frozenset -> validated frozenset (every name must be registered)."""
    if not features:
        return frozenset()
    fs = frozenset(features)
    for n in fs:
        get(n)  # raises on unknown
    return fs


def apply_pipeline(passes: list[str], features: frozenset[str]) -> list[str]:
    """Apply each enabled feature's pipeline edit, in a stable (sorted) order. Empty -> unchanged
    list object content (identity), so the joined string is byte-identical to the baseline."""
    if not features:
        return passes
    out = list(passes)
    for name in sorted(features):
        f = get(name)
        if f.edit_pipeline is not None:
            out = f.edit_pipeline(out)
    return out


def apply_schedule(schedule_text: str, features: frozenset[str]) -> str:
    """Apply each enabled feature's schedule edit, in stable order. Empty -> unchanged text."""
    if not features:
        return schedule_text
    out = schedule_text
    for name in sorted(features):
        f = get(name)
        if f.edit_schedule is not None:
            out = f.edit_schedule(out)
    return out


# ---- registered features ------------------------------------------------------------
# Keep this list small and evidence-justified. Each entry corresponds to a typed CompilerAction
# (PASS/HEURISTIC/PATTERN) surfaced by the action catalog from a mined kernel divergence.

def _vfmacc_schedule_edit(text: str) -> str:
    """fma_broadcast_policy (mined from openblas/xnnpack GEMM): recover a fused multiply-add.

    The baseline schedule tiles the matmul K dim to 1, so the vectorizer never forms a
    ``vector.contract`` and ``lower_contraction`` cannot fuse — the emitted asm is separate
    ``vfmul.vv``+``vfadd.vv`` (empirically confirmed). This edit gives the contraction a K-vector
    (tile/vectorize K=4) so a real contraction forms and lowers (with the outerproduct strategy)
    to ``vector.fma`` -> ``llvm.fmuladd`` -> ``vfmacc``. Gated to the fork; baseline untouched.
    """
    out = text.replace("tile_sizes [4, 8, 1]", "tile_sizes [4, 8, 4]")
    out = out.replace("vector_sizes [4, 8, 1]", "vector_sizes [4, 8, 4]")
    out = out.replace("transform.apply_patterns.vector.lower_contraction\n",
                      'transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"\n')
    return out


register(ImprFeature(
    name="fused_vfmacc_contraction",
    action_class="PATTERN",
    description="ATTEMPT (mined: fma_broadcast_policy): K-vector tile + outerproduct lowering aiming "
                "to emit fused vfmacc. MEASURED NO-OP — certified+decoded (test R7) on a 64^3 matmul: "
                "vfmacc still 0 (outerproduct expands to separate vfmul; -ffp-contract=fast does not "
                "fuse the MLIR-emitted chain). Kept as the recorded experiment; the real fix is a "
                "vector.fma-forming lowering pattern (deferred PASS work-item in action_catalog).",
    edit_schedule=_vfmacc_schedule_edit,
))
