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


# ONE shared task template. The blocks below are identical for every target and every experiment; the
# only target-specific content is the {slot} substitutions (from prompt_slots). The experiment axis
# (mode/arm/condition) selects which optional blocks render — it is target-agnostic. Compiler-not-kernel
# + the integrity rule are stated once, universally.
_TEMPLATE = """# Task: generate a {target} MLIR out-of-tree target backend (capsule_bench — {scope_label})

You are an autonomous agent. Produce a **non-exempt out-of-tree MLIR target backend package** for the
{target} accelerator under `submission/`. Your package is graded — through its CLI entrypoints only,
never imported — by compiling workload **capsules** (interface MLIR) and matching the target's reference
behavior. This is a **compiler/backend** task: your COMPILER generates the target artifact by lowering
the interface — you never author a compute kernel.

## Scope
Make **every** public/dev capsule under the declared corpus pass. Families are discovered, not restated:
{corpus_families}
Read each capsule's `capsule.yaml` + `capsule.interface.mlir` for its op/shapes/dtypes/epilogue, and the
target-agnostic contracts (`command_buffer_abi.yaml`, `interface_grammar.md`, the command-buffer schema).
Derive everything (rounding, tiling, dtypes, im2col, padding) from the contract + the target's own docs
below — nothing is restated here. The numeric `golden.yaml` is withheld; iterate against the QA gate.
Build ONE general backend for every family — do not special-case individual capsules.

## Deliverable (write into `submission/`)
```
submission/
  manifest.yaml   # artifact_type: mlir_oot_target_backend; target: {target}; language: cpp|python;
                  # integrity_exempt: false; (cpp) a build block; the 4 command argv templates
  mlir_oot/       # your OOT sources: input dialect + {target} target dialect + passes + {tool_stem}
  REPORT.md       # what you built + honest scope/limitations + a final status line (see end)
  docs/           # public_facts_used.md (every target fact you used + its source) + iteration_notes.md
```

## The 4 CLI entrypoints (your package is invoked ONLY via these)
- `parse`: `{{tool}} --verify-diagnostics {{input_mlir}}` — parse + verify the `merlin_iface` interface MLIR
- `lower_interface_to_target`: `{{tool}} --convert-iface-to-{target} {{input_mlir}}` — emit {target}-dialect MLIR
- `emit_command_buffer`: `{{tool}} --emit-command-buffer={{output_json}} {{input_mlir}}` — schema-valid `command_buffer.json`
- `emit_target_artifact`: `{{tool}} --convert-iface-to-{target} --emit-target-artifact {{input_mlir}}` — {endpoint_desc}; the emitted module defines `{kernel_symbol}`

## Grading + your QA signal
Per capsule the runner certifies exact `golden == reference(cb) == simulate(cb) == oracle` across the
tier ladder, and checks required instruction coverage. You cannot run the oracle; after each round a QA
gate writes a redacted `qa/verdict.json` (status / failure_plane / tiers / all_pass — NO golden values).
Read it at each round start and fix by failure_plane. Iterate until `all_pass: true`.

## Hard rules (integrity)
- `integrity_exempt: false`; no `import merlin` / `merlin.runtime.reference` / `reference_outputs`.
- **Compute must be compiler-GENERATED, never an authored/library kernel.** No hand C compute kernels, no
  copying/calling the target's high-level device libraries as the answer — your passes generate the code.
- Never hardcode/embed outputs (hidden capsules run after you freeze). One general backend.
- Do not read withheld goldens, hidden capsules, prior backends, or Merlin internals.

## Target ISA facts (derived — build your lowering on these)
{isa_facts}
{seam_menu}## Final status line (end of `submission/REPORT.md`) — write exactly one of:
1. "Backend passes all required public/dev capsules and is ready for hidden grading."
2. "Backend does not yet pass all required public/dev capsules; remaining failures listed by capsule + plane."
3. "Backend is not comparable because it violates the compiler/runtime/integrity boundary."
"""


# OPTIONAL block (merlin_assisted arms only): the discoverable, machine-checkable menu of OOT
# modification points. The CCA spine (granted only to the assisted arm) is more than a set of files to
# read — two answer-free calls ENUMERATE the full, target-specific lever set, so the agent knows WHICH
# seams exist and the next-stronger lever for each, not merely that the files are present. Fully
# target-agnostic: both calls are parameterized by {target}; nothing is hardcoded to one accelerator.
_SEAM_MENU = """## Menu of OOT modification points (merlin_assisted — the machine-checkable lever set)
The granted CCA spine is not just files to read: two answer-free calls ENUMERATE the full,
target-specific set of compiler seams you may modify for `{target}`, so you build the right lever set
instead of guessing from the file tree (neither imports the oracle or the grader):
- `cca_contract.check_bijection("{target}")` — the *what-to-build* checklist: which lever axes this
  target's ISA/RTL admits vs. which the compiler already routes (`orphan_fields` = leverable axes still
  to wire; `orphan_routes` = routes with no backing lever). Build every leverable axis; add no phantom.
- `action_catalog.escalation_ladder(axis, "{target}")` — for one axis, the full
  FLAG→KNOB→HEURISTIC→PASS→CODEGEN ladder weakest→strongest, each row naming the concrete OOT-relative
  seam file to edit and whether it is forkable today (the "which section, and the next stronger lever"
  answer). The seams point at YOUR generated OOT package, not our in-tree reference.

"""


def _is_assisted_arm(arm: str) -> bool:
    """The seam menu is exposed only to arms that are actually granted the CCA spine (the assisted arms);
    raw_baseline is not, so it must not be told to reach for tools it cannot use. Target-agnostic."""
    return "merlin_assisted" in arm


def render_prompt(te, manifest, experiment: str = "full", arm: str = "raw_baseline") -> str:
    """Render a target's full task prompt = the ONE shared template + the derived slots. ``experiment``
    and ``arm`` are the target-AGNOSTIC axes (they select the scope label / optional blocks); the target
    axis is entirely the {slot} substitutions, so for a fixed (experiment, arm) two targets' prompts
    differ ONLY in those slots."""
    s = prompt_slots(te, manifest)
    scope = {"full": "FULL SUITE", "realistic": "REALISTIC", "pilot": "PILOT SUBSET"}.get(experiment, experiment)
    families = "\n".join(f"- `{p}`" for p in s["corpus_families"]) or "- (the declared capsule corpus)"
    seam_menu = _SEAM_MENU.format(target=s["target"]) if _is_assisted_arm(arm) else ""
    return _TEMPLATE.format(target=s["target"], scope_label=scope, corpus_families=families,
                            tool_stem=s["tool_stem"], kernel_symbol=s["kernel_symbol"],
                            endpoint_desc=s["endpoint_desc"], isa_facts=s["isa_facts"],
                            seam_menu=seam_menu)
