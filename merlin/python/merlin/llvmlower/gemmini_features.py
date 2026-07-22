"""Fork-scoped, **default-off** compiler-feature registry for the Gemmini OOT codegen backend.

The Gemmini analog of ``impr_features`` (the RVV feature registry). The baseline Gemmini codegen
(``runtime/backends/gemmini_codegen_mlir.emit_kernel_mlir``) is FROZEN — it emits a weight-stationary
(WS) tile program with a fixed K-accumulation pattern. A target-dialect-generation experiment must be
able to *modify* that codegen (choose the dataflow, keep the output accumulator resident across the
reduction) without perturbing the frozen baseline. This registry is how: each modification is a NAMED,
default-off feature whose hook edits a :class:`GemminiCodegenOpts` — the option bundle the codegen reads.

Invariant (mirrors ``impr_features``): with ``features == frozenset()`` the hooks are never invoked, so
the codegen options — and therefore the emitted RoCC tile program — are **byte-identical** to the frozen
baseline (guarded by ``test_gemmini_features``). A feature only changes codegen when a fork explicitly
enables it, so it can be measured against the immutable baseline.

Status: the option bundle + registry are live; threading :class:`GemminiCodegenOpts` INTO
``emit_kernel_mlir`` (so an enabled feature actually re-emits the program) is the forkable-now step, so
the two seeded features are ``forkable_now=False`` in ``action_catalog`` until that thread lands. This is
the concrete "where/how to modify the compiler" the arm-3 agent is handed via ``seam_location``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable


@dataclass(frozen=True)
class GemminiCodegenOpts:
    """The knobs the Gemmini codegen reads. Defaults reproduce the frozen baseline byte-for-byte:
    weight-stationary dataflow, and the tile accumulator round-tripped through the accumulator SRAM
    per K-tile (``accumulator_resident`` off)."""
    dataflow: str = "ws"                 # "ws" (weight-stationary, baseline) | "os" (output-stationary)
    accumulator_resident: bool = False   # keep the MxN output PE/acc-resident across the K reduction


@dataclass(frozen=True)
class GemminiFeature:
    """One named, default-off Gemmini codegen change.

    ``edit_opts`` is a pure, deterministic function ``GemminiCodegenOpts -> GemminiCodegenOpts`` (it must
    return a NEW value, never mutate) applied only when the feature is enabled. ``action_class`` places
    it on the FLAG/KNOB/HEURISTIC/PASS/CODEGEN escalation ladder (the same ladder ``action_catalog`` uses).
    """
    name: str
    action_class: str  # "FLAG" | "KNOB" | "HEURISTIC" | "PASS" | "CODEGEN"
    description: str
    edit_opts: Callable[[GemminiCodegenOpts], GemminiCodegenOpts]


_REGISTRY: dict[str, GemminiFeature] = {}


def register(feature: GemminiFeature) -> GemminiFeature:
    if feature.name in _REGISTRY:
        raise ValueError(f"duplicate gemmini feature {feature.name!r}")
    _REGISTRY[feature.name] = feature
    return feature


def get(name: str) -> GemminiFeature:
    if name not in _REGISTRY:
        raise KeyError(f"unknown gemmini feature {name!r}; registered: {sorted(_REGISTRY)}")
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


def apply_opts(opts: GemminiCodegenOpts, features: frozenset[str]) -> GemminiCodegenOpts:
    """Fold each enabled feature's ``edit_opts`` over the options, in a stable (sorted) order. Empty
    features -> the options are returned UNCHANGED, so the emitted program is byte-identical to the
    frozen baseline."""
    if not features:
        return opts
    out = opts
    for name in sorted(features):
        out = get(name).edit_opts(out)
    return out


# --- the seeded default-off features (the two Gemmini LEVER axes in cca_contract) -------------------
register(GemminiFeature(
    name="gemmini_dataflow_select",
    action_class="HEURISTIC",
    description="choose the systolic dataflow (WS vs OS) instead of the codegen's fixed weight-stationary "
                "CFG_EX; OS keeps partial sums in the PE array and can cut mvin/mvout traffic for "
                "output-reuse-heavy tilings.",
    edit_opts=lambda o: replace(o, dataflow="os")))

register(GemminiFeature(
    name="gemmini_accumulator_resident",
    action_class="PASS",
    description="keep the MxN output tile accumulator-resident across the K reduction (accumulate in the "
                "accumulator SRAM with ACC_ACCUM and read out ONCE after the K-loop) instead of a "
                "per-K-tile accumulator round-trip.",
    edit_opts=lambda o: replace(o, accumulator_resident=True)))
