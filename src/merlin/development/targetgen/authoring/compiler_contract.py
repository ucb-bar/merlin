"""Versioned compiler deliverables for backend authoring.

Historical versions retain their exact expressions; fresh generation selects v6.
These instructions grant no source access, target truth or verification authority.
"""

from __future__ import annotations

# ---- generalization_prompt ---------------------------------------------------------------------

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


GENERAL_COMPILER_CONTRACT_V3 = (
    GENERAL_COMPILER_CONTRACT_V2.replace(
        "## General compiler requirements (shared v2)", "## General compiler requirements (shared v3)"
    )
    + """

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
)


GENERAL_COMPILER_CONTRACT_V4 = (
    GENERAL_COMPILER_CONTRACT_V3.replace(
        "## General compiler requirements (shared v3)", "## General compiler requirements (shared v4)"
    )
    + """

## Backend source package

Publish a normal source package: Python implementations use src/<package>/,
pyproject.toml and a merlin.backends entrypoint; native implementations use
CMakeLists.txt with real sources under lib/ or tools/ and public headers under
include/ when needed. Keep target matching, legality, lowering and encoders in
the backend. Import shared Merlin passes instead of copying Merlin into it.

Use the one host-selected common Merlin version for every backend. Reuse its
public passes and contracts. Put backend-specific pass additions or alternatives
inside the OOT package, for example transforms/overrides/, and select them
explicitly in its pipeline. Call the base pass or verifier where applicable;
do not copy its implementation, monkeypatch Merlin, or keep a separate Merlin
checkout or fork for each backend. If a common hook is missing, update the
common Merlin implementation and contract once for all affected backends.

A compiler_library/ already supplied by an admitted experiment is the exact
base input. Keep those staged bytes read-only and unchanged for new work; the
historical staging/grant protocol is separate. Local override sources belong
to the OOT namespace and its declared source identity. A deliberate common-base
change needs one separately declared common Merlin revision, not an embedded
package copy or a backend-private fork.

Do not embed compiler_library/, generic-library/, Merlin source, build outputs,
binaries or run results in the published backend. Keep exact paper reproductions
on their original immutable refs. Host publication uses source-layout version
3.0 and enforces this organization before export; numerical and hardware
evidence remain separate.
"""
)


GENERAL_COMPILER_CONTRACT_V5 = (
    GENERAL_COMPILER_CONTRACT_V4.replace(
        "## General compiler requirements (shared v4)", "## General compiler requirements (shared v5)"
    )
    .replace("Phase 1's destination", "Backend generation's objective")
    .replace("Phase 1 numerical execution", "development-time numerical execution")
    .replace("During Phase 2", "During optimization")
    .replace("In Phase 2", "During optimization")
    + """

## Connected target compiler

Deliver a compiler backend over the required program-IR domain. Small operation
examples are programs used during development, not the definition of its input
space. A dialect definition, a library of kernel builders, a shape-to-kernel
dispatcher or emitted artifacts alone does not satisfy this deliverable.

Declare accepted source IR, operation/type/shape/control-flow conditions, named
implementation families and actual conversion pipelines through the shared
compiler-description interface. Those are implementation claims. The original
required domain, target semantics and independent acceptance policy remain
authoritative; narrowing the claim does not excuse a required missing lowering.
Retain unknown, unsupported and unimplemented as distinct outcomes.

The normal compiler entry must inspect actual input operations, types, values
and effects, run applicable transformations and lower the selected source. Its
ordinary path must handle unseen legal sizes and supported compositions without
requiring callers to instantiate a named kernel class. Specialized kernel
builders may implement a conversion strategy internally, with their original
legality checks; they cannot replace source conversion and program integration.

Expose the target's source-pattern and implementation proposals before final
lowering. Bind each proposal to actual operations, every live input/output,
numerical/layout conditions, resource/memory requirements, pipeline and artifact
ABI. Describe fusion families and their executable matching conditions in OOT.
The specification owns arithmetic and hardware meaning; a fusion catalogue does
not redefine them. Shared Merlin checks graph boundaries and selects compatible
alternatives. No backend-specific matcher or target-name branch belongs in core.

For dialect-producing backends, provide inspectable typed target IR and its
conversion to the selected executable representation. An opaque wrapper around
already encoded instructions is not a replacement for the semantic operations
needed by scheduling and verification. Pipelines must state which operations
may remain at every boundary. Final admission rejects unresolved operations;
successful structural conversion alone is not a semantic correctness proof.
Binary and command-stream backends need no LLVM detour.

Preserve quantization producers and numerical ordering through matching and
fusion, including accumulator/bias domains, rounding, saturation and activation.
Preserve branches, loops, aliases, effects, shared producers and all live results
within the declared domain. Missing implementation or unknown legality must not
silently become scalar success or disappear from source coverage.

Demonstrate integration through the common model planner, partitioner and
runtime ABI using the same source conversion as small programs. A target backend
does not need a duplicate model frontend, global planner or scalar compiler.
An admitted host/vector complement may implement unsupported source work only
under the original placement and numerical policy. Accelerator-required work
retains its independent coverage obligation.

Describe source structure, interface completeness, emitted IR/artifacts,
numerical correctness and observed hardware use as separate evidence. Source-only
development leaves execution qualification pending. Generate larger conformance
programs per run under ignored output storage; do not add permanent tests for
every helper, flag, shape or implementation detail.
"""
)


GENERAL_COMPILER_CONTRACT_V6 = (
    GENERAL_COMPILER_CONTRACT_V5.replace(
        "## General compiler requirements (shared v5)", "## General compiler requirements (shared v6)"
    )
    .replace(
        "Keep target matching, legality, lowering and encoders in\nthe backend.",
        "Keep Interface implementation legality, target lowering and encoders in\nthe backend.",
    )
    .replace(
        "Expose the target's source-pattern and implementation proposals before final\n"
        "lowering. Bind each proposal to actual operations, every live input/output,\n"
        "numerical/layout conditions, resource/memory requirements, pipeline and artifact\n"
        "ABI. Describe fusion families and their executable matching conditions in OOT.\n"
        "The specification owns arithmetic and hardware meaning; a fusion catalogue does\n"
        "not redefine them. Shared Merlin checks graph boundaries and selects compatible\n"
        "alternatives. No backend-specific matcher or target-name branch belongs in core.",
        "Declare implemented Interface operations, numerical/layout constraints, resource\n"
        "requirements, fusion capabilities and artifact ABI in OOT. Merlin discovers\n"
        "upstream source groups, proves shared fusion rules, selects placement and\n"
        "materializes the complete Merlin Interface program. The backend proposes\n"
        "implementations for that Interface program; it does not parse the upstream\n"
        "model graph or own a parallel source-fusion pipeline. Target instruction\n"
        "fusion and local physical schedules remain OOT. The specification owns\n"
        "arithmetic and hardware meaning; compiler capabilities cannot redefine it.",
    )
    + """

## Merlin Interface compiler boundary

The normal backend input is versioned Merlin Interface IR, not captured FX,
linalg source, a kernel name or an encoded instruction wrapper. Reuse the single
selected Merlin dialect definitions. An older interface grammar must remain an
explicitly named compatibility route; do not silently equate dialect versions.

Merlin owns capture/import, numerical preparation, semantic simplification,
source-region discovery and fusion, global placement, partitioning, and shared
Interface construction. Backend capabilities flow upstream as implementation
claims and constraints. Backends may reject an unsupported Interface program;
they may not change its precision, erase live results or choose a different
global partition. Shared passes and scheduling APIs are imported from Merlin.

The OOT package owns Interface-to-target conversion, target-operation semantics,
local physical strategies, command effects, encoders and target artifacts. Its
normal public compiler entry lowers actual Interface operations for the declared
domain, including unseen admitted dimensions, without callers selecting kernel
builders. Device binary and command-stream routes need no LLVM detour. Return
artifacts and ABI requirements to Merlin for build, execution and independent
verification. Preserve both upstream and Interface identities in shared receipts.

This boundary does not imply every operation, fusion family or full model is
implemented. Report required but unimplemented Interface conversions explicitly.
Keep legacy experiment routes available only by deliberate selection, and never
count a legacy route or scalar result as qualification of the normal backend.

Use the reviewed SDK30 APIs from the one selected Merlin release:
`merlin.compiler.ir.interface_contract.parse_interface` reads the canonical
input; `merlin.compiler.backend.interface` exports compiler descriptions,
implementation recipes and the shared Source/Artifact/Compilation records.
Import shared transforms and schedules from that same release. Do not import
Merlin's host source-preparation driver, selected-source planner or evaluator,
copy an SDK into the backend, or replace shared functions through import order.
The host owns SDK selection and exact grants; these instructions grant none.
"""
)


def append_general_compiler_contract(text: str) -> str:
    """Serve the same block once, including when an explicit task already has it."""
    if any(
        contract in text
        for contract in (
            GENERAL_COMPILER_CONTRACT_V1,
            GENERAL_COMPILER_CONTRACT_V2,
            GENERAL_COMPILER_CONTRACT_V3,
            GENERAL_COMPILER_CONTRACT_V4,
            GENERAL_COMPILER_CONTRACT_V5,
            GENERAL_COMPILER_CONTRACT_V6,
        )
    ):
        return text
    return text.rstrip() + "\n\n" + GENERAL_COMPILER_CONTRACT_V6
