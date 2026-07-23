"""Generate a target's agent task prompt from a shared template + DERIVED slots.

The task prompts are the last gemmini-hardcoded surface: the deliverable layout, grading model, QA loop,
integrity rules, and status lines are identical for every target (the EXPERIMENT axis — mode/arm/condition
— selects which shared blocks compose, and is target-agnostic), while the only target-specific content is
a small set of slots derived from {the descriptor + the RTL fact bundle + the codegen endpoint}. This
module computes those slots; the template composition consumes them.

Guiding rule: for a FIXED (experiment, arm, condition), two targets' prompts differ ONLY in these slots.
Nothing here is hand-authored per target — every slot traces to a descriptor field, an mlc derivation,
or the endpoint kind.
"""
from __future__ import annotations

# How the 4th-entrypoint artifact is described to the agent, per codegen endpoint. Fork-free .insn on
# stock LLVM is the default (see memory no-forked-toolchain-bringup); never prescribes a forked toolchain.
_ENDPOINT_DESC = {
    "inline_asm_insn": ("lower your target dialect to an LLVM-dialect module of raw `.insn` (the target's "
                        "command ISA, from the discovered ISA facts) — assembled by STOCK clang/LLVM, no "
                        "forked toolchain"),
    "upstream_target": ("lower your target dialect to an upstream LLVM target (e.g. RVV / SPIR-V), "
                        "compiled by stock LLVM"),
    "external_backend": ("emit the target's device kernel source that the target's toolchain compiles "
                         "(only where a command-ISA `.insn` path is not available)"),
}


def prompt_slots(te, manifest) -> dict:
    """The complete set of DERIVED, target-specific prompt slots for one target.

    ``te`` is a :class:`TargetExperiment` (descriptor); ``manifest`` is a :class:`CapabilityManifest`.
    Returns a flat ``{slot: value}`` dict — the only content that varies across targets for a fixed
    experiment/arm/condition."""
    from .rtl.mlc_bridge import render_fact_bundle
    target = te.target
    return {
        "target": target,
        "tool_stem": f"{target}-opt",                 # not "gemmini-opt"
        "kernel_symbol": f"{target}_kernel",          # not "gemmini_kernel"
        "endpoint_kind": manifest.endpoint_kind,
        "endpoint_desc": _ENDPOINT_DESC.get(manifest.endpoint_kind, _ENDPOINT_DESC["inline_asm_insn"]),
        "isa_facts": render_fact_bundle(target),      # the provenance-tagged ISA brief (agent info)
        "corpus_families": te.corpus_siblings(),      # globbed, not a hardcoded ISA/layers/model_slices list
        "sim_tiers": dict(manifest.tier_sim),         # from the manifest, not "spike/verilator" literals
        "prior_backend_deny": list(te.prior_backends),
        "isa_headers": list(te.isa_headers),
        "hwbringup_set": te.hwbringup_set,
    }
