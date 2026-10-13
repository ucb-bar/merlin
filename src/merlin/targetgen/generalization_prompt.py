"""Versioned compiler task requirements shared by every authoring arm.

This text grants no tool, path, numerical permission or evaluation authority.
Keep strategy, hardware facts and protected examples out of this common block.
New wording requires a new version so archived prompt bytes remain attributable.
"""

from __future__ import annotations

GENERAL_COMPILER_CONTRACT_V1 = """## General compiler requirements (shared v1)

Build a compiler for the declared supported domain, including unseen dimensions
and compositions of supported operations. Phase 1's destination is functionally
correct end-to-end compilation of supported models through the ordinary upstream
pipeline. Public small cases are affordable checks of those operation families;
later programs may have much larger shapes, longer reductions and many interacting
operations. Do not assume later inputs are copies of the visible cases. Passing
small cases alone is not proof of full-model correctness.

Derive applicability from the current input IR, operation semantics, layouts,
numerical contract and admitted hardware capabilities. Do not select behavior
by capsule identity, model name, test membership, provenance ID or expected
output. Shape specialization is allowed when derived from the input IR with
explicit legality and resource checks; do not substitute a table of known tests.

For each substantive lowering or transform, record its semantic invariant,
preconditions, shape/resource limits and refusal behavior. Account for tails,
multiple tiles, reduction state, index/byte arithmetic and allocation lifetime.
Inputs inside the supported domain must remain supported when they exceed a
single resident allocation; implement a legal general path or report the domain
gap. Do not silently truncate work, omit outputs or claim unsupported cases pass.

Keep Phase 1 numerical execution within the declared work and storage budgets.
Use independent small cases that exercise multiple tiles and long thin shapes,
and small chains, forks/joins, shared producers, views, multiple outputs and
repeated invocation. Preserve the original numerical contract, including
rounding, accumulation order and quantization; local agreement does not establish
whole-graph agreement. Use only admitted generators, inputs and tools.

When an admitted compile-only path is available, compile independent larger
shapes through the same ordinary lowering, object generation and linking.
Inspect emitted bounds, resource use, dependencies and complete output coverage.
Static instruction count need not grow with input size: loops can execute more
work with the same code. A compile-only result is compilation evidence, never a
numerical execution pass. Report unavailable checks and unproved properties.

During Phase 2, preserve correctness and evaluate independent size, capacity and
reuse regimes using the admitted tools. Include host work, preparation, movement,
dispatch and output materialization in the declared cost scope. Predictions
outside a qualified regime remain unknown; tiny-case speedups do not establish
large-input or full-model speedups. Do not obtain protected validation inputs,
reference schedules, private answers or historical tuning results. Final
validation follows compiler freeze under the host's declared evaluation policy.

## Plan driven by the admitted tools

1. Read the supported domain, numerical rules, budgets, writable owners and tool
   inventory. Use its actual commands and availability, not an experiment-level
   label or an assumed tool installation. This common plan grants no extra access.
2. Inventory the supported input semantics and their lowering paths. Plan shape
   handling, resource legality, complete outputs and host/device composition.
   Identify unsupported or unknown cases explicitly before claiming coverage.
3. Build the ordinary compiler path. Use admitted parsers, IR inspection,
   verification, build/link and hardware-fact tools for the questions they answer.
   Verify that a changed pass actually changes the intended emitted program.
4. Check each invariant with the smallest independent case that exercises it.
   Use available functional tools for numerical and interaction evidence; use
   admitted compile-only checks for scale legality. Keep their evidence separate.
5. Read tool feedback, locate the failing stage or missing proof, and repair the
   general rule. In Phase 2 choose available cost analysis or measurement by the
   unresolved bottleneck and uncertainty; preserve correctness on each revision.
6. Record scope, regressions and unavailable checks. Run the declared promotion
   gates before freezing; leave protected final evaluation to the host. Neither
   this plan nor a successful tool invocation establishes an untested property.
"""

GENERAL_COMPILER_CONTRACT_V2 = GENERAL_COMPILER_CONTRACT_V1.replace(
    "## General compiler requirements (shared v1)", "## General compiler requirements (shared v2)"
).replace(
    "Static instruction count need not grow with input size: loops can execute more\n"
    "work with the same code. A compile-only result is compilation evidence, never a\n"
    "numerical execution pass. Report unavailable checks and unproved properties.",
    "Check public emitted-code growth before expensive simulations or full-model builds.\n"
    "Large and unseen dimensions should generally change runtime loop bounds, not\n"
    "replicate static instructions in proportion to all tensor elements. Legal\n"
    "specialization remains allowed under the original semantics and resource limits.\n"
    "Keep emitted IR, object-build time and storage within the operator's selected\n"
    "budgets; preserve all work and outputs rather than truncate to fit a limit.\n"
    "Functional numerical correctness, complete eligible accelerator placement and\n"
    "required structural/trace checks remain separate acceptance requirements.\n"
    "A small artifact, successful compile-only check or completed public stress\n"
    "ladder proves none of these requirements or full-domain generality. Report\n"
    "unavailable checks and unproved properties. Batch independent ready work on\n"
    "the verified resource pools while retaining one stable compiler snapshot.",
)


GENERAL_COMPILER_CONTRACT_V3 = GENERAL_COMPILER_CONTRACT_V2.replace(
    "## General compiler requirements (shared v2)", "## General compiler requirements (shared v3)"
) + """

## Mandatory accelerator coverage

Every source computation that can legally execute on the accelerator under the
original hardware capabilities and numerical contract must be lowered to it.
Derive that demand from the original source independently of candidate placement,
available compiler forms, model names and visible test membership. Missing
lowering support is a compiler gap, never a hardware exclusion. Large dimensions,
code-size limits and compilation or simulation timeouts cannot justify host
fallback for supported computation; implement legal loops, tiling or another
general accelerator path, or leave the case failed or incomplete.

Mixed host/device groups may retain independently justified unsupported
postprocessing and host orchestration, including address calculations, control
and transfers. Their supported tensor computation must still execute on the
accelerator. Bind every host computation to its independently checked hardware
or numerical exclusion. Bind legally eliminated source work to its equivalence
proof. Unknown eligibility or omitted source computation prevents completion.

Report demanded, requested, lowered and covered accelerator work, justified host
computation and unresolved source work separately. Coverage requires the complete
original demand in the linked program and source-bound device evidence alongside
the declared numerical checks. A claimed accelerator group or correct numerical
output alone cannot establish that the accelerator performed its computation.
"""


def append_general_compiler_contract(text: str) -> str:
    """Serve the same block once, including when an explicit task already has it."""
    if any(contract in text for contract in (
        GENERAL_COMPILER_CONTRACT_V1, GENERAL_COMPILER_CONTRACT_V2, GENERAL_COMPILER_CONTRACT_V3
    )):
        return text
    return text.rstrip() + "\n\n" + GENERAL_COMPILER_CONTRACT_V3
