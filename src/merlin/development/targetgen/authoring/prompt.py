"""Versioned compiler task requirements shared by every authoring arm.

This text grants no tool, path, numerical permission or evaluation authority.
Keep strategy, hardware facts and protected examples out of this common block.
New wording requires a new version so archived prompt bytes remain attributable.
"""

from __future__ import annotations

from merlin.development.targetgen.authoring.compiler_contract import (
    GENERAL_COMPILER_CONTRACT_V1 as GENERAL_COMPILER_CONTRACT_V1,
    GENERAL_COMPILER_CONTRACT_V2 as GENERAL_COMPILER_CONTRACT_V2,
    GENERAL_COMPILER_CONTRACT_V3 as GENERAL_COMPILER_CONTRACT_V3,
    GENERAL_COMPILER_CONTRACT_V4 as GENERAL_COMPILER_CONTRACT_V4,
    GENERAL_COMPILER_CONTRACT_V5 as GENERAL_COMPILER_CONTRACT_V5,
    GENERAL_COMPILER_CONTRACT_V6 as GENERAL_COMPILER_CONTRACT_V6,
    append_general_compiler_contract as append_general_compiler_contract,
)


# ---- generate_prompt ---------------------------------------------------------------------------
# Generate a target's agent task prompt from a shared template + DERIVED slots.
#
# The task prompts are the last surface hardcoded to one target: the deliverable layout, grading model, QA loop,
# integrity rules, and status lines are identical for every target (the EXPERIMENT axis — mode/arm/condition
# — selects which shared blocks compose, and is target-agnostic), while the only target-specific content is
# a small set of slots derived from {the descriptor + the RTL fact bundle + the codegen endpoint}. This
# module computes those slots; the template composition consumes them.
#
# Guiding rule: for a FIXED (experiment, arm, condition), two targets' prompts differ ONLY in these slots.
# Nothing here is hand-authored per target — every slot traces to a descriptor field, an mlc derivation,
# or the endpoint kind.


def _is_assisted_arm(arm: str) -> bool:
    """The seam menu is exposed only to arms that are actually granted the CCA spine (the assisted arms);
    raw_baseline is not, so it must not be told to reach for tools it cannot use. Target-agnostic."""
    return "merlin_assisted" in arm or "rtlchecks" in arm


def _enforced_workflow(
    arm: str,
    endpoint_kind: str,
    granted_tools,
    target: str,
    simt_mlir: bool = False,
    screen_tier: str = "",
    screen_sim: str = "",
) -> str:
    """The per-arm MANDATORY development workflow — a compulsory checklist (not an optional aid) matching
    the experiment ladder: it names ONLY the tools the arm actually grants, so raw_baseline gets just the
    build + self-check floor, EL2 the C++ generators, EL3 the xDSL/CCA + own-artifact lint, and EL4
    ALSO the RTL-facts derivation. Fully DERIVED from ``granted_tools`` (the arm's allowed-tool set from its
    bundle grant) so it stays target-agnostic and precisely per-arm; when the grant set is not threaded
    (direct/test callers) it falls back to a coarse arm-string approximation. This is what compels the agent
    to USE the tools (availability != usage) and to develop in xDSL with no regex."""
    if granted_tools is None:  # coarse fallback; real runs thread the grant
        assisted = _is_assisted_arm(arm)
        granted_tools = set()
        if assisted:
            granted_tools |= {"xdsl_dialects", "kernels/cca", "action_catalog"}
        if "rtlchecks" in arm:
            granted_tools.add("development/targetgen/rtl/")
        if arm == "cpp_merlininfra":
            granted_tools = {"targetgen/generate/mlir_scaffold"}
    g = granted_tools

    def _has(sub: str) -> bool:
        return any(sub in p for p in g)

    has_xdsl = _has("xdsl_dialects") or _has("development/targetgen/synthesize")
    has_cca = _has("kernels/cca") or _has("action_catalog")
    has_rtl = _has("development/targetgen/rtl/") or _has("rtl_facts")
    has_cpp = _has("mlir_scaffold") or _has("llvm_plan") or _has("target_repo")
    art = (
        "submission/lowered.llvm.mlir"
        if simt_mlir
        else "submission/kernel.S"
        if endpoint_kind == "external_backend"
        else "your emitted submission/*.mlir"
    )
    L = [
        "## MANDATORY development workflow (do ALL of these BEFORE the final status line — not optional)",
        "1. Your compiler backend lives under `submission/`; compute is COMPILER-GENERATED (never a hand kernel).",
        "2. Base every ISA / mesh / datapath / encoding decision on the **Target ISA facts** above + the",
        "   capability contract under `src/merlin/resources/contracts/` — never guess or hardcode; derive any fact not given.",
        # THE COMMAND IS THE INSTRUCTION. The prose above this list already said cheapest-first, and the
        # recommended invocation still named no tier — so it inherited the certifying default, and one
        # measured run spent EVERY self-check on the slowest engine: 6 min, 7 min, 75 min, 96 min of
        # turnaround across five requests, a 15x degradation of the loop the agent works in. An agent
        # following the brief literally must land on the cheap step, so the cheap step is what the
        # command says, with the certifying run named separately as the less frequent one.
        *(
            [
                "3. PER-EDIT LOOP — after each substantive build, screen the smallest affected scope on the",
                f"   CHEAP tier (`{screen_tier}` -> {screen_sim}; seconds per capsule):",
                f"   `python3 agent_selfcheck.py --submission submission --tiers {screen_tier}",
                "   --capsules <changed-capsule-or-subset>`. Run THAT after every edit. The certifying",
                "   RTL run is the SEPARATE, less frequent step — same command with no `--tiers` (it runs",
                "   every tier each capsule declares mandatory, as the grade does; MINUTES per capsule):",
                "   run it once the screen is clean, at convergence milestones, and once with",
                "   `--capsules all` before declaring done. A screen can only refute, never certify, so",
                "   a submission that never cleared the RTL run is not done. A submission you did not",
                "   self-check at all is not acceptable. Also run `python3 agent_selfcheck.py --submission",
            ]
            if screen_tier
            else [
                "3. After each substantive build, run `python3 agent_selfcheck.py --submission submission",
                "   --capsules <changed-capsule-or-subset>` for the smallest affected scope. Run `--capsules all`",
                "   at convergence milestones and once before declaring done — not after every edit. A submission",
                "   you did not self-check is not acceptable. Also run `python3 agent_selfcheck.py --submission",
            ]
        ),
        "   submission --shape-coverage`, which probes the",
        "   SAME operation at one tile and at two tiles in each of M, K and N. It costs no simulator (it runs",
        "   only your emit path), so run it often. **The capsules are a FIXED SET OF SHAPES: passing all of",
        "   them says nothing about whether you lower anything else, and you are graded on shapes you have",
        "   not seen.** Treat emission counts and `collapsed` findings as diagnostics: a loop can perform",
        "   more work without emitting more static instructions. Check actual bounds, tails, reductions",
        "   and output coverage before claiming an axis is supported. These emit-only probes establish",
        "   neither numerical correctness nor full-domain coverage.",
        "   `--candidate-diagnostics` inspects your public source, emitted operations, dtypes, host CFG,",
        "   declared placement and actual build products without running an oracle. Unsupported joins",
        "   remain UNKNOWN. `--dynamic-census` appends bounded activity from your own linked executable",
        "   after the original functional screen, using the selected runtime when available. Activity",
        "   counts are descriptive; they do not establish target cycles, physical traffic or utilization.",
        "   The full `--capsules all` self-check returns `offload_census` from the SAME graded public",
        "   operator artifacts: per program, accelerator work, declared host work, declines and silent",
        "   emission. Read its `all_accounted`, `programs` and `host_compute_vetoed` fields before",
        "   declaring done. If that field is absent or incomplete, or you need an earlier emit-only",
        "   diagnostic, run `python3 agent_selfcheck.py --submission submission --offload-census`.",
        "   **A program that passes with all of its work on the host is not what this target is for.**",
        "   Placing a region on the host is legitimate when the hardware cannot express it: DECLARE it",
        "   (`params.lane_placement`, or `declined` with a reason). `silent` rows and",
        "   `placement_undeclared` rows are findings to fix, and `routed_macs` should not fall while",
        "   your pass count rises.",
        "4. GRADEABLE-FLOOR FIRST (do this in your FIRST minutes, before deep encoder / ISA / parse work):",
        "   write `submission/manifest.yaml` declaring your entrypoints + a minimal CLI that ANSWERS all of",
        "   them (even trivially / with empty output) so `agent_selfcheck` can invoke your package and the",
        "   grader reaches the capsules. A round that ends WITHOUT a valid manifest scores 0 no matter how",
        "   much compiler you built — make the package structurally gradeable EARLY, THEN iterate on real",
        "   codegen. If you run low on time, a graded-but-imperfect package beats an ungradeable one.",
        # CADENCE, stated for every arm. The runs that converged fastest spent ~a third of their tool
        # calls in exactly this loop (compile + self-check + read verdict); the run that plateaued spent
        # 17% of its calls there and 15% of them in `sleep` poll loops instead. Both facts came out of
        # the rollouts, not the score, so state the loop rather than hope it is inferred.
        "5. CLOSE THE LOOP on every edit: change `submission/` -> build -> `agent_selfcheck.py` on the",
        "   changed capsule -> read `qa/verdict.json`. Blocking waits exist for the slow steps: "
        "`python await_verdict.py` for the next grade, `python simjob.py wait` for a simulation job.",
        "   Never spend a turn asking whether something finished — a `sleep`/`stat`/`pgrep` loop around",
        "   a file or a PID burns your budget without advancing the compiler.",
    ]
    n = 6
    if has_cpp and not has_xdsl:  # EL2
        L.append(
            f"{n}. Scaffold the package with the granted C++ OOT generators "
            "(`targetgen/generate/{mlir_scaffold,llvm_plan,target_repo}`), not ad-hoc hand files."
        )
        n += 1
    if has_xdsl:  # EL3 and EL4
        L.append(
            f"{n}. Author the backend as an **xDSL pass pipeline** (`xdsl_dialects/`, "
            "`targetgen/synthesize/`, `targetgen/generate/`) — structured IR passes, NOT ad-hoc string "
            "assembly, and with **NO regular expressions** (`import re` / regex text-matching is "
            "prohibited; parse the IR structurally). This is checked on your submission."
        )
        n += 1
        if has_cca:
            L.append(
                f"{n}. Enumerate your lever set: run `python cca_contract.py check-bijection {target}` "
                f"+ `python action_catalog.py escalation-ladder <axis> {target}` (runnable CLIs, like "
                "`isa_tools.py`) and build every leverable axis they list."
            )
            n += 1
        if endpoint_kind == "external_backend" and not simt_mlir:
            L.append(
                f"{n}. Assemble every instruction word with `python isa_tools.py asm ops.txt` — it packs "
                "the exact bits this target's own encoder uses and REFUSES to emit a wrong word. NEVER "
                "hand-pack `.word` hex: a word that is opcode-legal but decodes to a DIFFERENT op moves "
                "no data (your DMA/load silently no-ops) and scores 0 even though `lint` passes. Confirm "
                "with `disasm` that each word decodes back to the op you intended."
            )
            n += 1
        elif endpoint_kind == "inline_asm_insn":
            # The RoCC analogue of the `.word` mandate above. Stated as the recommended route, not a
            # command to rewrite an existing emitter: the agent owns its codegen. The rationale is the
            # whole point — field POSITIONS are a derived fact of the target, so a hand-written shift
            # expression is a re-derivation of something the tool already knows, and re-derivations drift.
            L.append(
                f"{n}. Produce your instruction words with `python isa_tools.py asm ops.txt` rather than "
                "hand-packed shifts: it takes every field POSITION from this target's own derived ISA "
                "model, whereas a hand-built `(a << 32) | (b << 16) | ...` re-derives those positions by "
                "hand and is where a field quietly ends up in the wrong one. How you emit stays your call."
            )
            n += 1
        if simt_mlir:
            # A SIMT-MLIR target emits an LLVM-dialect kernel LOWERING (compiled fork-free), not a self-hosted
            # word stream to assemble/lint. The mandate is that your xDSL passes BUILD the `llvm.func @<kernel>`
            # module (verified IR) that the shared llvmlower front compiles — the runner owns the harness.
            L.append(
                f"{n}. Emit `{art}` per the LLVM-dialect MLIR contract above: your xDSL pass pipeline BUILDS "
                f"the `builtin.module` with `llvm.func @{target}_kernel(...)` (pointer operands in "
                "[weight]++[lhs]++[out] order) that COMPUTES the op and `llvm.return`s — no prints, no "
                "`.insn`/`.word`, no `mu_schedule`. Verify the module parses/`verify()`s before self_check."
            )
            n += 1
        else:
            # `lint`/`disasm` answer "is this a real instruction?", which a correctly-NAMED instruction
            # carrying a wrong field ALSO answers yes to — so a submission can be numerically green and
            # trace-green and still diverge on the hardware plane. The check has to compare the DECODED
            # FIELDS against the intent the command buffer already states, field by field. The named
            # fields are the encoding-only axes generally (every target's `disasm` shows some subset);
            # no capsule, op, or expected value is named — the step reads the same whichever capsule fails.
            L.append(
                f"{n}. Before every self_check, run `python isa_tools.py lint` and `disasm` on {art} and "
                "confirm every instruction decodes (nothing UNKNOWN or ambiguous) and the kernel halts. "
                "A clean `lint` is NOT enough — a correctly-NAMED instruction carrying a WRONG FIELD "
                "decodes cleanly and still diverges on the hardware plane. So also read the `disasm` "
                "operand fields back against what your command buffer declares for that same command — "
                "DMA/DRAM address and row pitch, readout dtype / element width, accumulate and dataflow "
                "bits, config scale — and reconcile every one that disagrees."
            )
            n += 1
    if has_rtl:  # EL4 only (the CIRCT / RTL-facts level)
        L.append(
            f"{n}. EL4 (RTL-informed Merlin): DERIVE the ISA / mesh / datapath from the granted RTL-extracted facts "
            "(`targetgen/rtl/` + the RTL facts pin) — do not hand-invent them — and run the CIRCT RTL "
            "checks on your lowering. Your backend must be a compilation FROM those RTL-derived facts."
        )
        n += 1
        # NAME the APIs. This arm is scored on whether these four calls actually happened, and a prompt
        # that says only "derive from the granted RTL facts" leaves `sed -n '1,320p' rtl_facts/facts.json`
        # as the obvious reading -- which gets the facts and scores zero on all four checks. Measured on a
        # live arm-4 round: TASK.md contained none of these names, and four of the five failing
        # conformance checks were exactly these. Same failure as an earlier run that faked an ISA because
        # the prompt never named the shipped ISA files: name the surface or it is not used.
        L.append(
            f"{n}. Run these four — reading the files instead of CALLING them does not count:\n"
            '   - `from rtl import facts; facts.load_facts("' + target + '")` — the RTL fact bundle '
            "(arrays / memories / datapaths / interfaces); every ISA, mesh and datapath number comes "
            "from this dict.\n"
            "   - `from rtl_backend import target_profile, derived_levers; "
            'derived_levers(target_profile("' + target + '"))` — the compiler levers this hardware '
            "implies; build to the list it returns.\n"
            "   - `from generate import mlir_scaffold; mlir_scaffold.generate(<dialect plan>)` — "
            "scaffold the dialect/pass skeleton from the derived plan instead of hand-writing it "
            "(`llvm_plan.generate` / `target_repo.generate_skeleton` count too).\n"
            "   - read the `rtl_checks` block of `qa/verdict.json` whenever a new grade lands "
            "(wait for one with `python await_verdict.py`, never a poll loop) — that is where this "
            "arm's RTL feedback on YOUR lowering arrives; fix what it reports."
        )
        n += 1
    return "\n".join(L) + "\n\n"


SHARED_PASS_PROMPT_APIS = (
    "merlin.compiler.transforms.pipeline_state",
    "merlin.compiler.kernel_ir.ir",
    "merlin.compiler.schedule.pipeline.stages",
)


SHARED_PASS_CANONICAL_PROMPT_APIS = (
    "merlin.compiler.transforms.pipeline_state",
    "merlin.compiler.kernel_ir.ir",
    "merlin.compiler.schedule.pipeline.stages",
)


def shared_pass_prompt_record(requirement, *, granted_tools, library):
    """Check original v1 Kernel-library access; this issues no new grants."""
    from merlin.development.targetgen.authoring.library_admission import CompilerLibrarySelection
    from merlin.development.targetgen.authoring.shared_pass_contract import validate_shared_pass_requirement

    if type(library) is not CompilerLibrarySelection:
        raise ValueError("shared-pass requirements need an original selected compiler library")
    if type(granted_tools) not in (set, frozenset, tuple, list) or any(type(p) is not str for p in granted_tools):
        raise ValueError("shared-pass requirements need explicit observed tool grants")
    required = validate_shared_pass_requirement(requirement)
    selected = library.record()
    apis = (
        SHARED_PASS_PROMPT_APIS
        if library.contract.permits(SHARED_PASS_PROMPT_APIS[-1])
        else SHARED_PASS_CANONICAL_PROMPT_APIS
    )
    for module in apis:
        if not library.contract.permits(module):
            raise ValueError(f"shared-pass v1 selected library lacks required public API: {module}")
        member = next(member for member in library.contract.members if member.module == module)
        if not {module, member.path, str(library.root / member.path)}.intersection(granted_tools):
            raise ValueError(f"shared-pass v1 public API is not in the observed grants: {module}")
    if library.record() != selected:
        raise ValueError("shared-pass selected library changed during prompt construction")
    return {
        "schema": "merlin.phase1.shared_pass_prompt.v1",
        "requirement": required,
        "library": {
            "contract_sha256": selected["contract_sha256"],
            "review_id": selected["contract"]["review_id"],
            "public_modules": selected["contract"]["public_modules"],
        },
    }


def shared_pass_manifest_template(requirement):
    """Answer-free structural example; unknown rows authorize no transformation."""
    from merlin.development.targetgen.authoring.shared_pass_contract import validate_shared_pass_requirement

    required = validate_shared_pass_requirement(requirement)
    version = 1 if required["schema"] == "merlin.shared_pass_requirement.v1" else 2
    contracts = ("semantics", "effects", "capabilities") + (("resources", "completion") if version == 2 else ())
    features = sorted(set(required["foundations"]).union(*map(set, required["families"].values())))
    boundaries = {
        name: {
            "ir": "selected_target_mlir" if name == "device" else f"selected_{name}_mlir",
            "command": "emit_target_bundle" if name == "device" else f"consume_{name}",
            "contracts": {kind: "unknown" for kind in contracts},
            "features": features.copy(),
        }
        for name in required["boundaries"]
    }
    return {
        "target": "YOUR_SELECTED_TARGET",
        "target_ir_consumer": {
            "schema": "merlin.target_ir_consumer.v1",
            "command": "emit_target_bundle",
            "input": "selected_target_mlir",
            "output": "command_buffer_and_target_artifact",
        },
        "commands": {
            row["command"]: {"argv": ["{tool}", "{input_mlir}", "{output_json}"]} for row in boundaries.values()
        },
        "shared_pass_contract": {
            "schema": f"merlin.shared_pass_contract.v{version}",
            "boundaries": boundaries,
            "pipeline": required["pipeline"],
            "controls": {
                "default_policy": [],
                "choices": {name: {"enabled": [False], "max_growth_ops": 0} for name in required["families"]},
                "max_repairs": 0,
                "retain_original": True,
            },
            "families": [
                {
                    "name": name,
                    "status": "unknown",
                    "reason": "Original qualification not supplied",
                    "boundary": "computation",
                    "requirements": needs,
                }
                for name, needs in required["families"].items()
            ],
        },
    }


def _shared_pass_prompt_block_v1(record):
    """Instructions for a newly selected compiler, not evidence of consumption."""
    import json

    return (
        "\n\n## Selected shared-pass compiler requirements (v1)\n\n"
        "The original host selected the following requirement and reviewed library content revision. "
        "Use only its actually granted APIs. This v1 selection uses the public Kernel and existing "
        "PipelineSession/schedule_pipeline integration; it does not prescribe every future target's IR. "
        "The record grants no numerical, effects, hardware, resource or timing qualification.\n\n"
        "```json\n" + json.dumps(record, sort_keys=True, indent=2) + "\n```\n\n"
        "The following exact manifest shape is an answer-free template. Replace command placeholders and "
        "unknown contract identities with actual interfaces; a feature name only declares an interface. "
        "Unknown rows enable no pass. Mark a family supported only with original legality premises and "
        "actual consumption checks, then include it in default_policy in boundary order, "
        "preserving pass prerequisites, and permit "
        "[false,true]. Required-supported families must be implemented rather than left unknown.\n\n"
        "```json\n"
        + json.dumps(shared_pass_manifest_template(record["requirement"]), sort_keys=True, indent=2)
        + "\n```\n\n"
        "Expose typed computation, device scheduling and emitted-program boundaries through the "
        "declared shared_pass_contract commands. Preserve scalar semantics, reduction order, layouts, "
        "aliases/lifetimes, completion and all observable outputs. Project the original selected software "
        "and facts into capability/effect interfaces; qualification remains an independent checker "
        "obligation. Names and hashes alone are insufficient.\n\n"
        "Run legal, available shared passes automatically under deterministic bounded defaults in the "
        "normal compiler pipeline during Phase 1 iteration, at their respective IR levels. Consume the "
        "actual transformed IR in the next lowering/emission stage; producing a side report while "
        "emitting the original program does not meet the requirement. Reuse existing pass prerequisites, "
        "current five-way analysis stamps, invalidation, bounded repair and final emitted checks. "
        "Retain the legal original route on unknown legality, missing capabilities or exhausted budgets.\n\n"
        "Account for every required family as supported, unsupported or unknown with a precise reason. "
        "For each input record applied, no-match, unavailable or legal fallback; do not claim all nine "
        "families execute on unsupported hardware. Keep bounded enable/disable and policy controls, "
        "including explicit baseline and shared-default modes. Phase 1 must pass its original numerical, "
        "placement and final-artifact checks with defaults; Phase 2 tunes only legal alternatives to "
        "those defaults under identical library access. Structural declarations and this prompt are "
        "not an independent realization or execution witness.\n"
    )


def shared_pass_prompt_block(record):
    """Instructions for a newly selected compiler, not evidence of consumption."""
    if record["requirement"]["schema"] == "merlin.shared_pass_requirement.v1":
        return _shared_pass_prompt_block_v1(record)
    import json

    version = 1 if record["requirement"]["schema"] == "merlin.shared_pass_requirement.v1" else 2
    return (
        f"\n\n## Selected shared-pass compiler requirements (v{version})\n\n"
        "The original host selected the following requirement and reviewed library content revision. "
        "Use only its actually granted APIs. This selection uses the public Kernel and existing "
        "PipelineSession/schedule_pipeline integration; it does not prescribe every future target's IR. "
        "The record grants no numerical, effects, hardware, resource or timing qualification.\n\n"
        "```json\n" + json.dumps(record, sort_keys=True, indent=2) + "\n```\n\n"
        "The following exact manifest shape is an answer-free template. Replace command placeholders and "
        "unknown contract identities with actual interfaces; a feature name only declares an interface. "
        "Unknown rows enable no pass. Mark a family supported only with original legality premises and "
        "actual consumption checks, then include it in default_policy in boundary order, "
        "preserving pass prerequisites, and permit "
        "[false,true]. Required-supported families must be implemented rather than left unknown.\n\n"
        "```json\n"
        + json.dumps(shared_pass_manifest_template(record["requirement"]), sort_keys=True, indent=2)
        + "\n```\n\n"
        "Expose typed computation, device scheduling and emitted-program boundaries through the "
        "declared shared_pass_contract commands. Preserve scalar semantics, reduction order, layouts, "
        "aliases/lifetimes, completion and all observable outputs. Project the original selected software "
        "and facts into capability/effect interfaces; qualification remains an independent checker "
        "obligation. Names and hashes alone are insufficient.\n\n"
        "Run legal, available shared passes automatically under deterministic bounded defaults in the "
        "normal compiler pipeline during Phase 1 iteration, at their respective IR levels. Consume the "
        "actual transformed IR in the next lowering/emission stage; producing a side report while "
        "emitting the original program does not meet the requirement. Reuse existing pass prerequisites, "
        "current five-way analysis stamps, invalidation, bounded repair and final emitted checks. "
        "Retain the legal original route on unknown legality, missing capabilities or exhausted budgets.\n\n"
        "Account for every required family as supported, unsupported or unknown with a precise reason. "
        "For each input record applied, no-match, unavailable or legal fallback; do not claim all declared "
        "families execute on unsupported hardware. Keep bounded enable/disable and policy controls, "
        "including explicit baseline and shared-default modes. Phase 1 must pass its original numerical, "
        "placement and final-artifact checks with defaults; Phase 2 tunes only legal alternatives to "
        "those defaults under identical library access. Structural declarations and this prompt are "
        "not an independent realization or execution witness.\n"
        "Fusion requires exact scalar/reduction correspondence and physical resource, alias and completion "
        "laws. Buffering/pipelining requires explicit private-slot capacity, in-flight bounds, dependency "
        "and wait/completion semantics; capacity alone permits no overlap. Apply admitted mechanisms by "
        "default; unknown premises retain the original route and cannot satisfy required_supported. "
        "Qualification must independently check actual selected IR against emitted instruction operands "
        "and complete numerical/state outputs, with ignored-selection, wrong-operand, missing-wait and "
        "premature-buffer-reuse mutations. A package-generated report is not that checker.\n"
        "Implement target adaptation with operand, layout, effect and completion projections and target "
        "emitter callbacks. Prefer adapters over copied shared algorithms. Propose missing shared API "
        "extensions for separate review and versioning; do not modify the frozen library or its "
        "independent verification rules during this run.\n"
    )


def render_prompt(
    te,
    manifest,
    experiment: str = "full",
    arm: str = "raw_baseline",
    granted_tools=None,
    *,
    shared_pass_requirement=None,
    shared_pass_library=None,
) -> str:
    """Render a target's full task prompt = the ONE shared template + the derived slots. ``experiment``
    and ``arm`` are the target-AGNOSTIC axes (they select the scope label / optional blocks); the target
    axis is entirely the {slot} substitutions, so for a fixed (experiment, arm) two targets' prompts
    differ ONLY in those slots."""
    from merlin.development.targetgen.authoring.prompt_slots import (
        _ISA_DEVTOOLS,
        _ROCC_DEVTOOLS,
        _SEAM_MENU,
        _TEMPLATE,
        _is_simt_mlir,
        prompt_slots,
    )

    selected = (
        None
        if shared_pass_requirement is None
        else shared_pass_prompt_record(
            shared_pass_requirement, granted_tools=granted_tools, library=shared_pass_library
        )
    )
    s = prompt_slots(te, manifest)
    scope = {"full": "FULL SUITE", "realistic": "REALISTIC", "pilot": "PILOT SUBSET"}.get(experiment, experiment)
    # The full family set = the primary corpus + its discovered siblings (deduped, ordered), so no family
    # (e.g. the primary `isa/`) is silently dropped from the agent's view. Each is a `- `…`` bullet.
    fam_paths = list(dict.fromkeys([s["corpus_rel"], *s["corpus_families"]]))
    families = "\n".join(f"- `{p}`" for p in fam_paths) or "- `(the declared capsule corpus)`"
    # The sim tier ladder, one `- `…`` bullet per manifest tier (target-specific values; never literals).
    # Every bullet — including the empty-manifest fallback — starts with "- `" so it is a derived per-target
    # line (stripped when comparing the cross-target shared skeleton), never part of the shared body.
    ladder = (
        "\n".join(f"- `{tier}` → {sim}" for tier, sim in sorted(s["sim_tiers"].items()))
        or "- `(the target's declared sim tiers)`"
    )
    seam_menu = _SEAM_MENU.format(target=s["target"]) if _is_assisted_arm(arm) else ""
    # ISA dev tools: assisted arms only. A self-hosted-ISA (external_backend) target gets the .word/kernel.S
    # toolset; a RoCC/MLIR (inline_asm_insn) target gets the llvm.inline_asm variant; any other endpoint gets
    # nothing (empty slot renders nothing).
    # A SIMT-MLIR external target emits an LLVM-dialect kernel lowering compiled fork-free (no self-hosted-ISA
    # words to assemble/disassemble/lint), so the .word toolset does not apply — its grounding is the
    # LLVM-dialect kernel contract in isa_spec instead.
    _mlir = _is_simt_mlir(manifest)
    isa_dev_tools = ""
    if _is_assisted_arm(arm) and not _mlir:
        if s["endpoint_kind"] == "external_backend":
            isa_dev_tools = _ISA_DEVTOOLS
        elif s["endpoint_kind"] == "inline_asm_insn":
            isa_dev_tools = _ROCC_DEVTOOLS
    enforced_workflow = _enforced_workflow(
        arm,
        s["endpoint_kind"],
        granted_tools,
        s["target"],
        simt_mlir=_mlir,
        screen_tier=s["screen_tier"],
        screen_sim=s["screen_sim"],
    )
    body = _TEMPLATE.format(
        target=s["target"],
        scope_label=scope,
        general_compiler_contract=GENERAL_COMPILER_CONTRACT_V6,
        corpus_families=families,
        tool_stem=s["tool_stem"],
        kernel_symbol=s["kernel_symbol"],
        endpoint_desc=s["endpoint_desc"],
        emit_framing=s["emit_framing"],
        emit_symbol_note=s["emit_symbol_note"],
        grading_model=s["grading_model"],
        isa_facts=s["isa_facts"],
        sim_tier_ladder=ladder,
        seam_menu=seam_menu,
        isa_spec=s["isa_spec"],
        dram_contract=s["dram_contract"],
        termination_contract=s["termination_contract"],
        isa_dev_tools=isa_dev_tools,
        enforced_workflow=enforced_workflow,
    )
    return body if selected is None else body + shared_pass_prompt_block(selected)
