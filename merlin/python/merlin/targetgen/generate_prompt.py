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
    "inline_asm_insn": ("lower your target dialect to an **LLVM-dialect MLIR** module (a `.mlir`, NOT "
                        "textual LLVM-IR) whose command-ISA instructions are `llvm.inline_asm` ops wrapping "
                        "raw `.insn` directives — with opcode/func3/func7 from the discovered ISA facts, "
                        "assembled by STOCK clang/LLVM, no forked toolchain. EVERY operand must be an SSA "
                        "value defined earlier — an immediate as `%c = llvm.mlir.constant(<imm> : i64) : "
                        "i64`, a pointer via `llvm.ptrtoint` of an arg — then passed by name. Canonical:\n"
                        "    %c = llvm.mlir.constant(1441801 : i64) : i64\n"
                        "    %d = llvm.mlir.constant(16 : i64) : i64\n"
                        "    llvm.inline_asm has_side_effects \".insn r <op>, <f3>, <f7>, x0, $0, $1\", "
                        "\"r,r\" %c, %d : (i64, i64) -> ()\n"
                        "NEVER an inline integer literal operand like `... \"r,r\" (65540, 16)` — that is "
                        "invalid MLIR: it neither assembles NOR decodes, so CONFIG etc. read back as UNKNOWN "
                        "and the instruction class is scored missing. And do NOT emit textual LLVM-IR "
                        "(`call void asm sideeffect \"...\"`): the runner decodes `llvm.inline_asm` MLIR ops, "
                        "so a `.ll`-style body reads back as an empty instruction trace"),
    "upstream_target": ("lower your target dialect to an upstream LLVM target (e.g. RVV / SPIR-V), "
                        "compiled by stock LLVM"),
    "external_backend": ("emit a `kernel.S` of `.word`/`.insn` directives — the target's OWN encoded "
                         "instructions (self-hosted ISA), assembled to IMEM words by STOCK LLVM "
                         "(`llvm-mc`), then run on the target's cosim/RTL; no forked toolchain"),
    "command_buffer": ("emit the target's schema-valid command buffer directly — the artifact the "
                       "target's runtime consumes; no `.insn` assembly (the target has no command ISA, "
                       "e.g. a spatial tensor tile driven by one-hot op ports)"),
}


def _isa_spec_block(te) -> str:
    """Name the SHIPPED, real ISA sources (green card + ISA definition + the worked example kernel) and
    tell the agent to derive encodings from THEM — the fix for the observed failure where the agent,
    given only the discovered legal-opcode list, invents a plausible-but-wrong opcode/encoding scheme
    that assembles cleanly yet computes garbage on the RTL (scores 0). Target-agnostic: rendered only
    when the descriptor ships hardware-spec files (``isa_headers`` / ``hwbringup_set``); empty otherwise
    (a target with no shipped ISA header — e.g. a command-ISA target derived purely from facts — is
    unaffected, so the block never fabricates a source)."""
    headers = list(getattr(te, "isa_headers", []) or [])
    hw = getattr(te, "hwbringup_set", None)
    if not headers and not hw:
        return ""
    lines = [f"**Shipped {te.target} ISA — the source of truth for instruction encodings (derive, never invent):**",
             f"The real {te.target} ISA is shipped read-only in your bundle. Derive EVERY instruction's",
             "exact encoding from these files. Do NOT invent opcodes, mnemonics, instruction classes, or a",
             "bit layout: a plausible-but-invented encoding assembles cleanly yet decodes to garbage on the",
             "target and scores 0 (this is the single most common failure on a self-hosted ISA)."]
    for h in headers:
        lines.append(f"- `{h}`")
    if hw:
        lines += [f"- `{hw}/` (also mounted as `{te.target}/`) — RTL + ISA headers + README + a WORKED",
                  "  example kernel under `example_kernel/`. Translate the example's real instructions into",
                  "  your emitted encoding using the exact field layout the ISA definition specifies; the",
                  "  legal-opcode values in the ISA facts below are DECODE GATES, not the instruction",
                  "  semantics — take semantics + field packing from these files, never from the value list."]
    return "\n".join(lines) + "\n\n"


def _emit_framing(bundle: dict, endpoint: str = "inline_asm_insn") -> str:
    """A CONCRETE, derived one-liner describing what the 4th-entrypoint artifact must be — derived from
    the fact bundle + the codegen ENDPOINT (never a gemmini literal). A RoCC ``inline_asm_insn`` target
    emits a host `.insn` word stream encoding the discovered opcodes; a self-hosted-ISA ``external_backend``
    target emits a ``kernel.S`` of `.word`/`.insn` directives — the target's OWN encoded instructions —
    that STOCK LLVM (`llvm-mc`) assembles into IMEM words, NOT an MLIR module and NOT the model's bespoke
    mnemonics; a spatial ``command_buffer`` target emits a command stream over its op-port tile. A target
    with no HW dialect degrades to the generic phrasing."""
    f = bundle.get("fields", {})
    dim = f.get("mesh_dim", {})
    _mesh = (f" driving the discovered {dim['value']}x{dim['value']} systolic mesh"
             if dim.get("derived") and dim.get("value") else "")
    # SELF-HOSTED ISA (external_backend, e.g. atlas): the 4th artifact is a `kernel.S` of `.word`/`.insn`
    # directives that STOCK LLVM (llvm-mc) assembles to IMEM words — the encoding lives in the emitted
    # directives, grounded on the ISA definition + example kernel shipped in the bundle. NOT an MLIR
    # module, NOT `llvm.inline_asm`, and NOT the model's bespoke assembler mnemonics (llvm-mc can't
    # assemble `VMATMUL`-style mnemonics — only `.word`/`.insn`). Grounding the agent on the bundled ISA
    # is what stops it inventing opcodes (the AW2 hallucination finding).
    if endpoint == "external_backend":
        return ("a `kernel.S` of `.word`/`.insn` directives encoding the target's OWN instructions "
                "(compute each 32-bit encoding from the opcode/funct/field layout in the ISA definition "
                "shipped in your bundle; the bundled example kernel shows the required instruction "
                "sequence) that STOCK LLVM (`llvm-mc -triple=riscv64`) assembles into IMEM words — emit "
                "assembler text ONLY: NOT an MLIR module, NOT `llvm.inline_asm`, NOT the model's mnemonic "
                "assembler syntax (stock LLVM cannot assemble the target's custom mnemonics)" + _mesh)
    # SPATIAL tensor-tile (OPU): a command buffer over one-hot op ports driving an outer-product tile —
    # NOT a RoCC .insn stream over a systolic mesh. Frame it from the discovered tile geometry + command
    # set (gemmini has no tile_dim, so it never takes this branch and stays byte-identical).
    td = f.get("tile_dim", {})
    if td.get("derived") and isinstance(td.get("value"), dict):
        tv = td["value"]
        geo = (f"{tv.get('rows')}x{tv.get('cols')}" if tv.get("rows") and tv.get("cols")
               else str(tv.get("dim") or "")).strip()
        cats = (f.get("op_categories", {}) or {}).get("value") or []
        dts = [d.get("name") for d in ((f.get("dtypes", {}) or {}).get("value") or []) if d.get("name")]
        line = f"a command stream driving the discovered {geo} outer-product accumulator tile".rstrip()
        if cats:
            line += f" with the {{{', '.join(cats)}}} command set"
        if dts:
            line += f" over the {', '.join(dts)} datapaths"
        return line
    lo = f.get("legal_opcodes", {})
    dim = f.get("mesh_dim", {})
    parts = []
    if lo.get("derived") and lo.get("value"):
        vals = lo["value"]
        parts.append(f"a word stream encoding ONLY the {len(vals)} discovered legal command opcodes "
                     f"(funct field {min(vals)}..{max(vals)}; enumerated in the ISA facts below)")
    if dim.get("derived") and dim.get("value"):
        parts.append(f"driving the discovered {dim['value']}x{dim['value']} systolic mesh")
    if not parts:
        return "a word stream encoding only the target's discovered legal command ISA (see the ISA facts below)"
    return "; ".join(parts)


# The certification-model sentence, DERIVED from the corpus goldens — NOT hardcoded to gemmini's integer
# 3-way. The grader (capsule_golden.is_independent_float_golden) decides per capsule whether a golden is an
# INDEPENDENT float reference graded within a tolerance (a float datapath: fp8/bf16 MXU) or the exact-integer
# self-consistency check. We classify the target's declared corpus with that SAME signal so the prompt tells
# the agent the grading model the runner will actually apply — never a per-target English branch.
_GRADE_INTEGER = ("Per capsule the runner certifies exact-integer `golden == reference(cb) == simulate(cb) "
                  "== oracle` (no tolerance) across the sim tier ladder — derived from the manifest, not "
                  "restated:")
_GRADE_FLOAT = ("Per capsule the runner certifies the emitted artifact's program-oracle output against an "
                "INDEPENDENT float `golden` within the capsule's declared tolerance (its `grade_policy` "
                "atol/rtol) across the sim tier ladder; the integer `reference(cb) == simulate(cb)` "
                "self-consistency cross-checks do NOT apply to a float datapath and report `not_applicable` "
                "— derived from the corpus goldens, not restated:")


def _corpus_uses_independent_float_goldens(te) -> bool:
    """True iff any capsule under the declared corpus (+ its discovered siblings) carries an INDEPENDENT
    float golden — the exact signal the grader keys on. Derived from the corpus, never a target name; IO
    failures degrade to the integer model (the conservative default the shared skeleton has always used)."""
    import yaml

    from merlin.common.paths import repo_root
    from .capsule_golden import is_independent_float_golden
    root = repo_root()
    corpora = ([te.capsule_corpus] if te.capsule_corpus else [])
    corpora += [root / rel.rstrip("/") for rel in te.corpus_siblings()]
    for corpus in corpora:
        if not corpus or not corpus.is_dir():
            continue
        for capy in sorted(corpus.glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(capy.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if is_independent_float_golden(cap, capy.parent):
                return True
    return False


# DRAM address contract — rendered ONLY for a self-hosted-ISA (external_backend) target graded on the
# program oracle, where the emitted kernel and the oracle must agree on where each tensor lives in DRAM.
# Empty for RoCC/command_buffer targets (their operands ride the command buffer, no explicit addresses).
_DRAM_CONTRACT = """
## DRAM address map (self-hosted ISA — your kernel and the oracle must agree)
The program oracle runs your assembled kernel on the target's cosim: it PRELOADS each input tensor into
DRAM and reads the OUTPUT tensor back from DRAM. So in `command_buffer.json` you MUST declare **every
tensor your kernel touches** — inputs, weights, AND the output — each as `{{shape, dtype, role}}`, and
give each a `base` (its DRAM byte address). Your kernel must load each input from, and store the output
to, EXACTLY those `base` addresses — the oracle preloads inputs and captures the output there. The output
tensor MUST appear (its result is read from its `base`); omit it and the grade cannot see your answer. All
addresses must lie inside the DRAM region defined by your ISA memory map (the oracle relocates that region
to the model's aperture, so an address below the DRAM base cannot be indexed). If you omit a `base` the
harness assigns a canonical one inside that same DRAM region, but then your kernel must target that layout —
declaring them yourself is the reliable path. Addresses are per-capsule; size them from each tensor's
shape x dtype.
"""


# Program-termination contract (external_backend / self-hosted ISA only). Derived, never a literal
# terminator mnemonic: it NAMES the requirement (every program must reach the ISA's halt instruction) and
# points at the shipped ISA definition + example kernel for the actual encoding. This closes the observed
# failure where the agent emitted a runnable-but-non-terminating kernel: the functional oracle ran it to
# the cycle cap and every capsule failed "did not halt within N instructions" BEFORE any numeric check —
# so a correct matmul that never halts still scored 0, round after round, and the agent never saw why.
_TERMINATION_CONTRACT = """
## Program termination (REQUIRED — a non-halting kernel fails before numerics)
The functional oracle runs your assembled kernel to a fixed instruction/cycle cap and then STOPS. Your
emitted program MUST reach the target's terminating instruction — the one the ISA definition marks as
asserting the machine's halt/done signal — on every control path (the shipped example kernel ends with
it). If it never halts, the capsule fails at the functional tier (`did not halt within N instructions`)
and the numeric comparison never runs: a numerically-correct kernel that does not terminate still scores
0. Derive the terminator's exact encoding from the ISA definition (do not invent it), and emit it as the
final instruction of your kernel.
"""


def _termination_contract(te, manifest) -> str:
    """The program-termination contract for a self-hosted-ISA target, with the terminator's DERIVED encoding
    appended when it can be behaviorally derived from the target's own ISA definition (the op whose sole
    semantic effect is asserting the machine finish flag). Public ISA structure — the agent already has the
    ISA definition — so naming the exact `.word` is an aid, not a leak, and it is derived per-target (no
    literal terminator here). Falls back to the generic contract if the terminator is not derivable."""
    if manifest.endpoint_kind != "external_backend":
        return ""
    try:
        from .isa_model import isa_model_for
        m = isa_model_for(te)
        words = sorted({v for _, v in m.halt_signatures})
        if m.halt_mnemonics and words:
            names = " / ".join(m.halt_mnemonics)
            wl = ", ".join(f"`.word {w:#010x}`" for w in words)
            return (_TERMINATION_CONTRACT + f"\nFor this target the terminator is **{names}**, which the "
                    f"ISA's own encoder emits as {wl} (operands zero) — verify with the ISA dev tools' "
                    f"`disasm`/`lint`, and make it the final instruction on every path.\n")
    except Exception:  # noqa: BLE001 — terminator not derivable (no model venv / no such op) -> generic text
        pass
    return _TERMINATION_CONTRACT


def prompt_slots(te, manifest) -> dict:
    """The complete set of DERIVED, target-specific prompt slots for one target.

    ``te`` is a :class:`TargetExperiment` (descriptor); ``manifest`` is a :class:`CapabilityManifest`.
    Returns a flat ``{slot: value}`` dict — the only content that varies across targets for a fixed
    experiment/arm/condition."""
    from .rtl.mlc_bridge import render_fact_bundle_for, fact_bundle_for
    target = te.target
    bundle = fact_bundle_for(target)                  # KIND-routed; discovered ONCE; feeds brief + emit framing
    return {
        "target": target,
        "tool_stem": f"{target}-opt",                 # not "gemmini-opt"
        "kernel_symbol": f"{target}_kernel",          # not "gemmini_kernel"
        # The 4th artifact defines a kernel SYMBOL only when it is a code module (.insn / upstream / an
        # external kernel); a command_buffer endpoint's artifact IS the command buffer (no module symbol).
        "emit_symbol_note": ("" if manifest.endpoint_kind == "command_buffer"
                             else f"; the emitted module defines `{target}_kernel`"),
        "endpoint_kind": manifest.endpoint_kind,
        "endpoint_desc": _ENDPOINT_DESC.get(manifest.endpoint_kind, _ENDPOINT_DESC["inline_asm_insn"]),
        # The grading model the runner will actually apply — float-tolerance vs exact-integer — classified
        # from the corpus goldens (never a per-target branch), so the agent is not told the wrong contract.
        "grading_model": _GRADE_FLOAT if _corpus_uses_independent_float_goldens(te) else _GRADE_INTEGER,
        "emit_framing": _emit_framing(bundle, manifest.endpoint_kind),  # endpoint-aware, derived from the bundle
        "isa_facts": render_fact_bundle_for(target, bundle),  # KIND-routed provenance-tagged ISA brief (agent info)
        "corpus_rel": te.corpus_rel(),                # the primary corpus (isa/), repo-root-relative
        "corpus_families": te.corpus_siblings(),      # globbed, not a hardcoded ISA/layers/model_slices list
        "sim_tiers": dict(manifest.tier_sim),         # from the manifest, not "spike/verilator" literals
        "prior_backend_deny": list(te.prior_backends),
        "isa_headers": list(te.isa_headers),
        "hwbringup_set": te.hwbringup_set,
        "isa_spec": _isa_spec_block(te),              # names the shipped real ISA files (derive, don't invent)
        # DRAM address contract (external_backend only): declare every tensor + base so the emitted kernel
        # and the program oracle agree on operand/result addresses (the atlas 0/11 output-base bug).
        "dram_contract": _DRAM_CONTRACT if manifest.endpoint_kind == "external_backend" else "",
        # program-termination contract (external_backend only): the emitted kernel must reach the ISA's
        # halt instruction or it fails functional before numerics (the atlas non-halting-kernel wall).
        "termination_contract": _termination_contract(te, manifest),
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
below — nothing is restated here. The numeric reference golden is withheld; iterate against the QA gate.
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
- `emit_target_artifact`: `{{tool}} --convert-iface-to-{target} --emit-target-artifact {{input_mlir}}` — {endpoint_desc}: {emit_framing}{emit_symbol_note}

Declare these four commands in `manifest.yaml` exactly as the runner expects — see the OOT backend
contract (`mlir_oot_backend_contract.yaml`) and the manifest schema (`schemas/manifest.schema.json`).
{dram_contract}{termination_contract}
## Cross-round memory (each round is a FRESH session)
You have NO memory of prior rounds except what is on disk. Between rounds the harness writes
`qa/round_brief.md` — your progress log across all graded rounds (per-round pass count, failure planes,
lowest mismatch) plus your own notes and a nudge if you stopped journaling. **At the START of every round,
read `qa/round_brief.md` and `docs/iteration_notes.md` before touching code**: build on what you already
worked out, and do NOT undo a change that improved an earlier round. **After every change, append to
`docs/iteration_notes.md`** what you changed, what the verdict showed, and your next hypothesis — that file
and the brief are your only durable memory across rounds.

## Grading + your QA signal
{grading_model}
{sim_tier_ladder}
and checks the required instruction coverage per capsule (it decodes your emitted artifact into an
instruction trace). You cannot run the oracle; after each round a QA gate writes a redacted
`qa/verdict.json` per capsule — `status`, `failure_plane`, `trace_violations`, `numeric_status`,
`mismatch_count`, `tiers` (L0–L3), and `all_pass` — with NO golden/expected values. Read it at each
round start and fix by `failure_plane` + `trace_violations`. Iterate until `all_pass: true`.

Useful self-checks you CAN run locally (no oracle needed): build your tool, run the 4 entrypoints on
each `capsule.interface.mlir`, and confirm the emitted `command_buffer.json` validates against the
command-buffer schema and your lowered artifact looks right.

**Iterate FAST — smallest scope, cheapest checks first.** When you `self_check`, check ONLY the capsule you
just changed (it accepts a single capsule or subset and returns in seconds) — do NOT re-grade all capsules
on every edit; run the full set once before you declare done. The slow cycle-accurate RTL check runs only
AFTER you converge on the fast functional tier, so tight, narrow loops cost you nothing.

## Hard rules (integrity)
- `integrity_exempt: false`; no `import merlin`, no `merlin.runtime.reference` calls, no baked-in reference outputs.
- **Compute must be compiler-GENERATED, never an authored/library kernel.** No hand C compute kernels, no
  copying/calling the target's high-level device libraries as the answer — your passes generate the code.
- Never hardcode/embed outputs (hidden capsules run after you freeze). One general backend.
- Do not read withheld goldens, hidden capsules, prior backends, or Merlin internals.

## Target ISA facts (derived — build your lowering on these)
{isa_spec}{isa_facts}
{isa_dev_tools}{seam_menu}## Final status line (end of `submission/REPORT.md`) — write exactly one of:
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


# OPTIONAL block (assisted arms + external_backend only): the derived ISA dev tools staged in the sandbox
# (assembler / disassembler / linter). Oracle-free and reveal no golden — they encode the syntax the agent
# chose and inspect the agent's OWN emitted words — so they are a fair authoring aid for the assisted arms,
# never a leak. Fully target-agnostic: derived from the target's own shipped ISA definition, no accelerator
# named here.
_ISA_DEVTOOLS = """## ISA dev tools (assembler / disassembler / linter / debugger) — staged as `isa_tools.py`
You have a derived toolset for this target's self-hosted ISA (oracle-free for asm/disasm/lint: they encode
the syntax YOU choose and inspect YOUR OWN emitted words; they never reveal a golden). Use it to avoid the
two most common raw-ISA mistakes — invented encodings and a program that never halts:
- `python isa_tools.py asm ops.txt` — assemble a mnemonic listing (`MNEMONIC field=value, ...`, one per
  line) into the correct `.word` lines for your `kernel.S`, packed into the exact bits this target's own
  encoder uses. It REFUSES rather than emit a wrong word, so you never hand-pack a 32-bit instruction.
- `python isa_tools.py disasm submission/kernel.S` — decode your assembled kernel back to
  `{mnemonic, operands}`; a word that decodes to nothing is an invented/garbled encoding.
- `python isa_tools.py lint submission/kernel.S --op <op>` — flag illegal opcodes, a missing terminator
  (a kernel that never halts fails every capsule before numerics), and instruction-class coverage vs the op.
Run these locally as often as you like; they need no oracle. **Run `lint` (and `disasm`) BEFORE every
`self_check`** — they are instant and catch the encoding / halt / missing-compute-role mistakes that would
otherwise waste a functional-sim run.

When your kernel assembles and halts but the OUTPUT is wrong (e.g. all zeros), stop guessing and OBSERVE the
data path with the lite debugger:
- `python isa_tools.py debug submission/kernel.S --capsule <name> --run-to <N> --region BASE:NBYTES ...`
  runs your kernel on the functional model up to instruction `N` (omit `--run-to` to run to halt), then
  reports `pc`, the scalar registers, `halt_reason`, `instr_count`, and a hex dump of each DRAM `--region`
  you name (`--region` repeats; `BASE` may be hex `0x..` or decimal, matching the addresses your kernel
  uses). Watch a region fill — or stay zero — as your DMA/load/compute/store sequence advances: dump the
  input tile right after your DMA.LOAD to confirm data actually landed, then the scratch/accumulator region
  after the compute. `all_zero: true` right after a load means your load didn't write where you think.
  Add `--state` for a VALUE-FREE populated-map of on-chip memory (staging SRAM / register-file / accumulator
  banks) — one `populated: true/false` per bank, no values — so you can see exactly which stage's data
  landed (did VLOAD reach SRAM? did the weight push populate a register bank? did the MXU write the
  accumulator?) and pinpoint the stage that silently no-ops.
  This is the fastest way to localize which stage silently no-ops. The OUTPUT region is withheld (that is
  the answer) — debug the INPUT and scratch regions; correctness of the final output comes from `self_check`.

"""


# The RoCC / inline_asm_insn analogue of _ISA_DEVTOOLS: same staged `isa_tools.py`, but the canonical
# artifact is `llvm.inline_asm` MLIR (not a `.word` kernel), so the tools speak MLIR. Oracle-free, no golden.
_ROCC_DEVTOOLS = """## ISA dev tools (assembler / disassembler / linter) — staged as `isa_tools.py`
You have a derived RoCC toolset (oracle-free; it encodes the syntax YOU choose and inspects YOUR OWN
emitted MLIR — never a golden). Use it so you never hand-write a wrong `.insn` op:
- `python isa_tools.py asm ops.txt` — assemble a listing (one `CLASS rs1 rs2` per line, e.g.
  `CONFIG_EX 0 0`, `MVIN 0x80000000 16`, `PRELOAD 256 0`, `COMPUTE_PRELOADED 0 0`, `MVOUT 0xA0000000 16`,
  `FLUSH 0 0`, `FENCE`) into the CANONICAL `llvm.inline_asm` MLIR — each operand a
  `%c = llvm.mlir.constant(<v> : i64)` SSA value — packed with this target's derived opcode/func3/func7.
  It REFUSES rather than emit a wrong instruction (unknown class, or a CONFIG whose rs1 subtype bits don't
  match). Paste its output into your emitted `.mlir` — NEVER hand-write inline-integer-literal operands like
  `"r,r" (65540, 16)`: that is invalid MLIR that neither assembles NOR decodes (it reads back as UNKNOWN and
  the class scores missing).
- `python isa_tools.py disasm submission/<your>.mlir` — decode your emitted MLIR back to instruction
  classes; anything that comes back UNKNOWN is a non-canonical/garbled instruction.
- `python isa_tools.py lint submission/<your>.mlir` — flag UNKNOWN instructions + show the decoded class
  histogram. Run it BEFORE every `self_check` (it is instant and catches the exact encoding mistake that
  makes an otherwise-correct kernel fail the trace gate).

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
    # The full family set = the primary corpus + its discovered siblings (deduped, ordered), so no family
    # (e.g. the primary `isa/`) is silently dropped from the agent's view. Each is a `- `…`` bullet.
    fam_paths = list(dict.fromkeys([s["corpus_rel"], *s["corpus_families"]]))
    families = "\n".join(f"- `{p}`" for p in fam_paths) or "- `(the declared capsule corpus)`"
    # The sim tier ladder, one `- `…`` bullet per manifest tier (target-specific values; never literals).
    # Every bullet — including the empty-manifest fallback — starts with "- `" so it is a derived per-target
    # line (stripped when comparing the cross-target shared skeleton), never part of the shared body.
    ladder = "\n".join(f"- `{tier}` → {sim}" for tier, sim in sorted(s["sim_tiers"].items())) \
        or "- `(the target's declared sim tiers)`"
    seam_menu = _SEAM_MENU.format(target=s["target"]) if _is_assisted_arm(arm) else ""
    # ISA dev tools: assisted arms only. A self-hosted-ISA (external_backend) target gets the .word/kernel.S
    # toolset; a RoCC/MLIR (inline_asm_insn) target gets the llvm.inline_asm variant; any other endpoint gets
    # nothing (empty slot renders nothing).
    isa_dev_tools = ""
    if _is_assisted_arm(arm):
        if s["endpoint_kind"] == "external_backend":
            isa_dev_tools = _ISA_DEVTOOLS
        elif s["endpoint_kind"] == "inline_asm_insn":
            isa_dev_tools = _ROCC_DEVTOOLS
    return _TEMPLATE.format(target=s["target"], scope_label=scope, corpus_families=families,
                            tool_stem=s["tool_stem"], kernel_symbol=s["kernel_symbol"],
                            endpoint_desc=s["endpoint_desc"], emit_framing=s["emit_framing"],
                            emit_symbol_note=s["emit_symbol_note"], grading_model=s["grading_model"],
                            isa_facts=s["isa_facts"], sim_tier_ladder=ladder, seam_menu=seam_menu,
                            isa_spec=s["isa_spec"], dram_contract=s["dram_contract"],
                            termination_contract=s["termination_contract"], isa_dev_tools=isa_dev_tools)
