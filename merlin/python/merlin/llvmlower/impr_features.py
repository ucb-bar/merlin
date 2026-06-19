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

# The vfmacc-forming schedule (developed by MLIR-level iteration, R-cont): vectorize the func's
# contraction to a real `vector.contract`, lower it via the OUTERPRODUCT strategy to a chain of
# `vector.outerproduct{kind=add}` (acc + a⊗b — the fused MAC), then `lower_outerproduct` turns each
# into `vector.fma` -> `llvm.intr.fmuladd` -> the RISC-V backend emits `vfmacc`. The baseline's
# [4,8,1] K=1 tiling never forms a contraction (so outerproduct/K-tile/fast-math were all measured
# no-ops); vectorize_children DOES form it. Appropriate for kernel-sized contraction workloads (the
# mining testbed) — NOT a whole transformer (vectorize_children there explodes vector.extract, per
# pipeline.py). Gated to the impr fork; baseline schedule untouched.
_VFMACC_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %v = transform.structured.vectorize_children_and_apply_patterns %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %v {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def _vfmacc_schedule_edit(text: str) -> str:
    """Replace the baseline schedule with the vfmacc-forming recipe (full replacement — the
    structure differs fundamentally from the tile+vectorize baseline)."""
    return _VFMACC_SCHEDULE


register(ImprFeature(
    name="fused_vfmacc_contraction",
    action_class="PASS",
    description="mined fma_broadcast_policy: form a real vector.contract -> outerproduct(kind=add) "
                "-> vector.fma -> llvm.fmuladd -> vfmacc (vectorize_children + lower_contraction "
                "outerproduct + lower_outerproduct). Closes the separate-vfmul.vv+vfadd.vv gap. For "
                "kernel-sized contraction workloads (vectorize_children explodes on whole models).",
    edit_schedule=_vfmacc_schedule_edit,
))
