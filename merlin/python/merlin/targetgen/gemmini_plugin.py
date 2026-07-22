"""Gemmini backend compiler-plugin — the backend-SPECIFIC content that registers into the
backend-AGNOSTIC core.

Design invariant (see the repo's "backend-agnostic core" rule): the core — ``kernels.cca`` (schema),
``kernels.cca_compare`` (diff), ``kernels.cca_contract`` (bijection), ``kernels.action_catalog``
(router + seam map), ``kernels.microkernel`` (resolver registry), ``rvvgen.beam``/``sweep`` (search) —
knows NOTHING about Gemmini. Everything Gemmini-specific (which SpatialFacet divergence maps to which
compiler lever, where that lever lives, what the default-off codegen features are) is registered HERE,
at load time, through the core's registration APIs (``register_route`` / ``register_seam`` /
``gemmini_features.register``).

This module is the in-tree REFERENCE plugin. The target-dialect-generation flow (arm 3/4) produces the
same shape — routes + seams + features — GENERATED / beam-searched into an OOT package; that generated
plugin is loaded the same way, with ``oot_package`` pointing the seams at the agent's own middle-end. So
adding or changing a Gemmini lever is an ad-hoc, out-of-tree edit; the core never changes.

Activate with ``gemmini_plugin.register()`` (idempotent). Pass ``oot_package`` to bind the seams to a
generated package root (else they resolve as the reference + a ``<oot_package>`` placeholder).
"""
from __future__ import annotations

from ..kernels.action_catalog import _Route, register_route, register_seam

_LOADED = False


def register(oot_package: str | None = None) -> None:
    """Register Gemmini routes + seams + features into the agnostic core. Idempotent."""
    global _LOADED
    if _LOADED:
        return

    # 1. features — the default-off codegen modifications self-register on import (backend-specific).
    from ..llvmlower import gemmini_features  # noqa: F401  (import triggers registration)

    # 1b. micro-kernel resolver — realize the agnostic MicrokernelSpec as Gemmini codegen knobs, so
    #     microkernel.resolve("gemmini", spec) works (the beam's micro-kernel granularity is expressible).
    from ..kernels import microkernel
    from . import gemmini_cca
    microkernel.register_resolver("gemmini", gemmini_cca.gemmini_microkernel_resolver)

    # 2. seams — OOT-package-relative (the agent edits its GENERATED middle-end; the in-tree emitter is
    #    cited only as the reference implementation, never an edit target on our core).
    register_seam(
        "gemmini_features",
        "<oot_package>/passes/gemmini_features.py  "
        "[reference: merlin/python/merlin/llvmlower/gemmini_features.py]",
        "register a default-off Gemmini codegen feature (edits GemminiCodegenOpts) in the OOT package's "
        "pluggable feature registry", False, backend="gemmini")
    register_seam(
        "gemmini_codegen",
        "<oot_package>/lowering/  (the generated OOT backend's RoCC tile-program lowering)  "
        "[reference: merlin/python/merlin/runtime/backends/gemmini_codegen_mlir.py]",
        "the OOT backend's tile-program emitter — thread GemminiCodegenOpts through it to make a "
        "gemmini_features route forkable", True, backend="gemmini")

    # 3. routes — a mined SpatialFacet divergence -> the concrete Gemmini codegen lever. forkable_now is
    #    False until GemminiCodegenOpts is threaded through the emitter; the route still names WHERE and
    #    HOW to make the change (the "which section of the compiler do I modify" answer).
    register_route("gemmini", _Route(
        axis="spatial.dataflow",
        when=lambda d: d.expert in ("os", "ws") and d.expert != d.ours,
        action_class="HEURISTIC",
        target_seam="gemmini_features:gemmini_dataflow_select",
        change="select the systolic dataflow (WS/OS) via GemminiCodegenOpts.dataflow instead of a fixed "
               "weight-stationary CFG_EX.",
        forkable_now=False,
        expected_effect="expected: OS keeps partial sums PE-resident, cutting mvin/mvout traffic for "
                        "output-reuse-heavy tilings (not yet measured — awaiting the opts thread)",
        intended_facet={"spatial.dataflow": "os"}))
    register_route("gemmini", _Route(
        axis="spatial.accumulator_resident",
        when=lambda d: bool(d.expert) and d.ours in (False, None),
        action_class="PASS",
        target_seam="gemmini_features:gemmini_accumulator_resident",
        change="keep the MxN output accumulator-resident across the K reduction (accumulate in the "
               "accumulator SRAM with ACC_ACCUM, read out once after the K-loop) via "
               "GemminiCodegenOpts.accumulator_resident, instead of a per-K-tile round-trip.",
        forkable_now=False,
        expected_effect="expected: removes the per-K-tile accumulator round-trip traffic (not yet "
                        "measured — awaiting the opts thread into the emitter)",
        intended_facet={"spatial.accumulator_resident": True}))

    _LOADED = True
