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


def _is_simt_mlir(manifest) -> bool:
    """True when the target's 4th artifact is an LLVM-dialect MLIR module (``lowered.llvm.mlir``) that the
    oracle compiles FORK-FREE (stock LLVM rv32 + the target's RTL-derived Muon re-encode, no vendor fork) —
    the thesis path where the agent emits a COMPILER LOWERING, not a hand kernel.

    DERIVED, no target literal: resolve the 4th-output filename EXACTLY as the runner does
    (:func:`runner_config.runner_config_from_manifest`) — the contract's explicit ``runner.fourth_output_name``
    if present, else the endpoint-kind default — and key on the ``.mlir`` suffix, so the PROMPT's emit form
    stays in lockstep with what the ORACLE actually compiles (they must never disagree)."""
    if getattr(manifest, "endpoint_kind", None) != "external_backend":
        return False
    from .runner_config import ENDPOINT_ARTIFACT
    fourth = getattr(manifest, "fourth_output_name", None) or ENDPOINT_ARTIFACT.get(manifest.endpoint_kind, "")
    return str(fourth or "").endswith(".mlir")


def _simt_mlir_emit_framing() -> str:
    """The 4th-artifact one-liner for a SIMT core graded on the MLIR thesis path: an LLVM-dialect MLIR module
    (a compiler LOWERING the agent's xDSL passes build), compiled fork-free — NOT C/C++ source, NOT a
    ``.word``/``.insn`` self-hosted stream."""
    # Name no ISA here: the re-encode is whatever the TARGET's RTL-derived encoding says, and this
    # module is shared by every SIMT target. Saying it generically is also more accurate.
    return ("an LLVM-dialect MLIR module (`builtin.module` with `llvm.func @<kernel>`) — a COMPILER LOWERING "
            "your xDSL passes produce — which the runner compiles FORK-FREE (stock LLVM rv32 + the target's "
            "own RTL-derived instruction re-encode, no vendor fork) and runs on the cosim; NOT C/C++ source, "
            "NOT `.word`/`.insn` assembler, NOT a self-hosted kernel")


def _simt_mlir_grounding(target: str) -> str:
    """The LLVM-dialect MLIR kernel contract for a SIMT core on the fork-free thesis path (replaces the .word
    self-hosted grounding). The agent emits ONLY the kernel FUNCTION as verified LLVM-dialect IR: the runner
    OWNS the harness (embeds the cb operands, calls the kernel, prints OUT/DONE) and the SIMT runtime OWNS the
    warps/barriers, so the kernel is plain scalar compute over pointer operands. Mirrors the gemmini
    LLVM-dialect contract (the SAME shared ``llvmlower``→stock-LLVM front), minus the RoCC ``.insn`` — the
    reference emitter (:func:`merlin.runtime.backends.muon_codegen_mlir.emit_kernel_mlir`) emits this shape."""
    sym = f"{target}_kernel"
    skeleton = (
        "module {\n"
        f"  llvm.func @{sym}(%W: !llvm.ptr, %L: !llvm.ptr, %O: !llvm.ptr) {{\n"
        "    // O = L @ W  (row-major; M, K, N come from the interface). Emit the loads\n"
        "    // (llvm.getelementptr + llvm.load), the multiply-accumulate (llvm.fmul / llvm.fadd), and the\n"
        "    // stores (llvm.store) — or lower scf/arith loops to this. Your xDSL passes BUILD this IR.\n"
        "    llvm.return\n"
        "  }\n"
        "}\n")
    return (
        f"**Your 4th artifact is an LLVM-dialect MLIR module (`submission/lowered.llvm.mlir`) defining "
        f"`llvm.func @{sym}` — a COMPILER LOWERING, not a hand kernel.** The runner compiles it FORK-FREE via "
        "the SHARED `llvmlower` front (MLIR → LLVM IR → STOCK clang rv32 object), then re-encodes it to the "
        "target's own ISA and runs it on the cosim. Contract:\n"
        f"1. Emit ONE `builtin.module` containing `llvm.func @{sym}(...)`. Every argument is `!llvm.ptr`, in "
        "the ABI order **[weight] ++ [lhs in command order] ++ [outputs in command order]** (the generic "
        "kernel_abi); pointees are row-major f32.\n"
        "2. The function COMPUTES the op (loads → multiply-accumulate → stores into the output pointers) and "
        "`llvm.return`s. It is plain scalar compute over the pointer operands — the SIMT warps / barriers / "
        "scheduling are the RUNTIME's (the fork-free BSP spawns warps around your kernel), so you do NOT write "
        "`mu_schedule`, barriers, or thread-id logic.\n"
        "3. Emit NO prints, NO `.insn`/`.word`, NO DRAM base map, NO halt — the runner owns the harness (it "
        "embeds the operands, calls your kernel, and prints the `OUT`/`DONE` protocol) and the BSP owns "
        "boot/halt. A kernel that writes its output pointers and returns is complete.\n"
        "4. BUILD the module with your xDSL pass pipeline (typed IR, `verify()`-checked) — NEVER by string "
        "assembly and NEVER with regex; this is checked on your submission.\n"
        "\nThe module skeleton (structure is mandatory; the reference backend emits this shape):\n"
        "```mlir\n" + skeleton + "```\n\n")


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
    lines.append(_fixed_format_encoding_block(te))
    return "\n".join(l for l in lines if l) + "\n\n"


def _fixed_format_encoding_block(te) -> str:
    """For a FIXED-FORMAT wide-word target (a SIMT core whose whole ISA is one field layout selected by an
    opcode field), render the RTL-DERIVED encoding — the exact bit layout, opcode table, and address spaces
    the compiler must emit — so the agent packs correct words instead of inventing a layout. Empty for a
    variable-format self-hosted ISA (that grounding is the shipped ISA definition above)."""
    try:
        from .isa_model import isa_model_for_target
        m = isa_model_for_target(getattr(te, "target", ""))
    except Exception:  # noqa: BLE001
        return ""
    if not m.is_fixed_format():
        return ""
    fl = sorted(m.field_layout.items(), key=lambda kv: -kv[1][0])
    fields = ", ".join(f"{n}[{hi}:{lo}]" for n, (hi, lo) in fl)
    ops = ", ".join(sorted(m.opcode_table))
    out = [f"\n**Derived {m.inst_width}-bit instruction encoding (from this target's RTL decoder — the exact",
           "layout the hardware decodes; emit words that match it):**",
           f"- fields: {fields}",
           f"- opcodes: {ops}"]
    if m.address_spaces:
        sp = ", ".join(f"{k}={v}" for k, v in sorted(m.address_spaces.items()))
        out.append(f"- address spaces (field `{m.address_space_field}`): {sp} — a memory op's space is set "
                   "in that field; a plain load defaults to the first space, so a scratchpad access MUST set it.")
    out.append("The `disasm`/`lint` tools decode your emitted words against exactly this layout — run them "
               "before every `self_check`.")
    return "\n".join(out)


def _cmdbuf_opcodes() -> list[str]:
    """The command buffer's legal `opcode` values, READ FROM THE SCHEMA that validation uses.

    Rendered into the prompt rather than restated in prose so the two cannot drift: if the enum grows, an
    agent is told about the new opcode by the same file that will accept it. Returns [] when the schema is
    unavailable or stops declaring a closed enum, so the prompt simply omits the sentence rather than
    asserting a vocabulary it could not confirm.
    """
    import json
    from ..common.paths import repo_root
    path = repo_root() / "merlin" / "contract" / "schemas" / "command_buffer.schema.json"
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
        enum = (((schema.get("properties") or {}).get("commands") or {}).get("items") or {}) \
            .get("properties", {}).get("opcode", {}).get("enum")
    except Exception:  # noqa: BLE001 — a prompt must render even if the schema moved
        return []
    return [str(v) for v in enum] if isinstance(enum, list) else []


def _emit_framing(bundle: dict, endpoint: str = "inline_asm_insn", inst_width: int = 32) -> str:
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
                f"(compute each {inst_width}-bit encoding from the opcode/funct/field layout in the ISA "
                "definition shipped in your bundle; the bundled example kernel shows the required instruction "
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
        if dts:
            line += f" over the {', '.join(dts)} datapaths"
        # TWO VOCABULARIES, and conflating them is a wasted round. `op_categories` are the unit's one-hot
        # RTL PORTS, read off `io_op_<cat>_` names -- they describe the hardware. The command buffer is a
        # target-INDEPENDENT ABI whose `opcode` enum is closed and lives in the schema; that is what the
        # agent emits, and what the arc oracle consumes. Naming only the ports (as this line used to) reads
        # as an instruction to emit `"opcode": "macc"`, which the schema rejects -- while the mapping
        # between the two was stated nowhere. Both are rendered, each labelled, and the ABI side is read
        # from the schema so it cannot drift from what validation will accept.
        if cats:
            line += (f". The unit's op ports are {{{', '.join(cats)}}} -- that is the HARDWARE's command "
                     f"set, which the backend drives; it is not what you write in the buffer")
        opcodes = _cmdbuf_opcodes()
        if opcodes:
            line += (f". The buffer's own `opcode` values are the ABI's: {{{', '.join(opcodes)}}} "
                     f"(closed enum, validated)")
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
# A corpus may carry BOTH: an int8 systolic target whose generalization capsules are authored in float
# through the PyTorch frontend grades exact-integer on some capsules and within a tolerance on others.
# Collapsing that to either single sentence tells the agent the wrong grading model for the rest of the
# corpus, so the mixed case says so explicitly and names the per-capsule signal that decides it.
_GRADE_MIXED = ("This corpus is MIXED, so the grading model is decided PER CAPSULE by that capsule's own "
                "golden, not once for the target: a capsule carrying an INDEPENDENT float `golden` is "
                "certified against the program oracle within its declared tolerance (its `grade_policy` "
                "atol/rtol) and its integer `reference(cb) == simulate(cb)` self-consistency cross-checks "
                "report `not_applicable`; every other capsule is certified exact-integer "
                "`golden == reference(cb) == simulate(cb) == oracle` with no tolerance. Both apply across "
                "the sim tier ladder — derived from the corpus goldens, not restated:")


def _corpus_golden_regimes(te) -> tuple[bool, bool]:
    """``(has_float, has_integer)`` over the declared corpus (+ discovered siblings), classified with the
    SAME per-capsule signal the grader uses. Split out from the old any()-style predicate because a corpus
    with even one float golden was reported as wholly float, which is wrong for a mixed corpus."""
    import yaml

    from merlin.common.paths import repo_root
    from .capsule_golden import is_independent_float_golden
    root = repo_root()
    corpora = ([te.capsule_corpus] if te.capsule_corpus else [])
    corpora += [root / rel.rstrip("/") for rel in te.corpus_siblings()]
    has_float = has_int = False
    for corpus in corpora:
        if not corpus or not corpus.is_dir():
            continue
        for capy in sorted(corpus.glob("*/capsule.yaml")):
            try:
                cap = yaml.safe_load(capy.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if is_independent_float_golden(cap, capy.parent):
                has_float = True
            else:
                has_int = True
    return has_float, has_int


def _grading_model(te) -> str:
    """The certification sentence for this corpus: float-only, integer-only, or explicitly mixed."""
    has_float, has_int = _corpus_golden_regimes(te)
    if has_float and has_int:
        return _GRADE_MIXED
    return _GRADE_FLOAT if has_float else _GRADE_INTEGER


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


# DRAM address contract — selected by the HARNESS'S address model (see _dram_contract). This is the
# FIXED-PRELOAD model: a self-hosted-ISA (external_backend) target graded on the program oracle, which
# preloads each tensor at a DECLARED base, so the emitted kernel must target those exact addresses.
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


# The POINTER-ARGS model — for a target whose bare-metal harness passes each operand buffer as a POINTER
# argument to the emitted kernel (the sim allocates the buffers at run time; there is no fixed preload
# address). Target-agnostic: it states the calling convention (the interface tensors become the kernel's
# pointer args) and the universal invariant (off-chip addresses are runtime-owned, so they come from the
# args, never a baked literal) — no address value, no target/opcode literal. This was the MISSING contract
# that let a backend bake DRAM address 0 (unindexable) and trap on every capsule before any numeric check.
_POINTER_ARGS_CONTRACT = """
## DRAM addressing — your kernel receives the operands as POINTER ARGUMENTS
Your emitted kernel function is the lowering of the capsule's interface function: it receives **each
interface tensor as a pointer argument**, in the interface's order (each tensor's role — input / weight /
output — is declared in the interface MLIR you lower). The bare-metal harness ALLOCATES those buffers at
run time and passes their addresses in — there is NO fixed, known-ahead-of-time DRAM address. So for every
memory-movement instruction (the ISA facts mark which classes move data between DRAM and the accelerator),
compute its DRAM address FROM the matching pointer argument: `%a = llvm.ptrtoint <that arg> : !llvm.ptr to
i64`, then use `%a` (optionally plus a constant tile/element offset) as the address operand. NEVER bake a
literal DRAM address (0, or any constant): a baked address cannot match the buffer the harness allocated,
so the kernel accesses the wrong memory and faults on every capsule. On-chip scratchpad / accumulator
addresses ARE fixed constants — only the off-chip DRAM addresses must come from the arguments. The ISA dev
tools flag a baked DRAM address, so you can catch this before the oracle runs.
"""


def _dram_contract(manifest) -> str:
    """Select the DRAM-address contract by the harness's ADDRESS MODEL, derived from the endpoint kind —
    never a per-target literal. ``external_backend`` grades on the program oracle, which PRELOADS each
    tensor at a declared base (fixed_preload) → the kernel targets those addresses. An inline-asm/RoCC
    target's bare-metal harness PASSES each operand as a pointer arg (pointer_args) → the kernel derives
    addresses from its args. A command-buffer endpoint carries operands in the buffer (no explicit
    kernel-side addresses) → no contract."""
    ek = getattr(manifest, "endpoint_kind", None)
    if ek == "external_backend":
        return _DRAM_CONTRACT           # fixed_preload
    if ek == "inline_asm_insn":
        return _POINTER_ARGS_CONTRACT   # pointer_args
    return ""


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
    _iw = 32                                           # instruction width for the emit framing — DERIVED, not literal
    try:
        from .isa_model import isa_model_for_target
        _iw = isa_model_for_target(target).inst_width or 32
    except Exception:  # noqa: BLE001 — no derivable model -> keep the 32-bit default phrasing
        pass
    # A SIMT external target graded on the MLIR thesis path (lowered.llvm.mlir) emits an LLVM-dialect MLIR
    # kernel LOWERING compiled fork-free, not a self-hosted .word/.insn stream. Its grounding is the
    # LLVM-dialect kernel contract, and the .word-ISA contracts (DRAM base map + halt-op) do not apply — the
    # runner owns the harness (embed/call/print) and the BSP owns boot/halt.
    _mlir = _is_simt_mlir(manifest)
    return {
        "target": target,
        "tool_stem": f"{target}-opt",                 # not "gemmini-opt"
        "kernel_symbol": f"{target}_kernel",          # not "gemmini_kernel"
        # The 4th artifact defines a kernel SYMBOL only when it is a code module (.insn / upstream / an
        # external kernel); a command_buffer endpoint's artifact IS the command buffer (no module symbol).
        "emit_symbol_note": ("" if manifest.endpoint_kind == "command_buffer"
                             else f"; the emitted module defines `{target}_kernel`"),
        "endpoint_kind": manifest.endpoint_kind,
        "endpoint_desc": ("emit an LLVM-dialect MLIR kernel lowering (compiled fork-free)" if _mlir
                          else _ENDPOINT_DESC.get(manifest.endpoint_kind, _ENDPOINT_DESC["inline_asm_insn"])),
        # The grading model the runner will actually apply — float-tolerance vs exact-integer — classified
        # from the corpus goldens (never a per-target branch), so the agent is not told the wrong contract.
        "grading_model": _grading_model(te),
        "emit_framing": (_simt_mlir_emit_framing() if _mlir
                         else _emit_framing(bundle, manifest.endpoint_kind, inst_width=_iw)),  # endpoint+width derived
        "isa_facts": render_fact_bundle_for(target, bundle),  # KIND-routed provenance-tagged ISA brief (agent info)
        "corpus_rel": te.corpus_rel(),                # the primary corpus (isa/), repo-root-relative
        "corpus_families": te.corpus_siblings(),      # globbed, not a hardcoded ISA/layers/model_slices list
        "sim_tiers": dict(manifest.tier_sim),         # from the manifest, not "spike/verilator" literals
        "prior_backend_deny": list(te.prior_backends),
        "isa_headers": list(te.isa_headers),
        "hwbringup_set": te.hwbringup_set,
        # SIMT-MLIR target: ground the agent in the LLVM-dialect kernel contract (a compiler lowering compiled
        # fork-free), NOT the self-hosted-ISA .word spec. Otherwise: name the shipped real ISA files.
        "isa_spec": _simt_mlir_grounding(target) if _mlir else _isa_spec_block(te),
        # DRAM address contract (self-hosted-ISA external_backend only): declare every tensor + base so the
        # emitted .word kernel and the program oracle agree on operand/result addresses (the atlas 0/11
        # output-base bug). A SIMT-MLIR kernel takes pointer operands from the runner harness — no DRAM map.
        "dram_contract": "" if _mlir else _dram_contract(manifest),
        # program-termination contract (self-hosted-ISA only): the emitted .word kernel must reach the ISA's
        # halt instruction. A SIMT-MLIR kernel just `llvm.return`s; the BSP owns boot/halt.
        "termination_contract": "" if _mlir else _termination_contract(te, manifest),
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
  docs/           # PLAN.md (first-round design plan) + public_facts_used.md (facts used + source) + iteration_notes.md
```

## The 4 CLI entrypoints (your package is invoked ONLY via these)
- `parse`: `{{tool}} --verify-diagnostics {{input_mlir}}` — parse + verify the `merlin_iface` interface MLIR
- `lower_interface_to_target`: `{{tool}} --convert-iface-to-{target} {{input_mlir}}` — emit {target}-dialect MLIR
- `emit_command_buffer`: `{{tool}} --emit-command-buffer={{output_json}} {{input_mlir}}` — schema-valid `command_buffer.json`
- `emit_target_artifact`: `{{tool}} --convert-iface-to-{target} --emit-target-artifact {{input_mlir}}` — {endpoint_desc}: {emit_framing}{emit_symbol_note}

Declare these four commands in `manifest.yaml` exactly as the runner expects — see the OOT backend
contract (`mlir_oot_backend_contract.yaml`) and the manifest schema (`schemas/manifest.schema.json`).
{dram_contract}{termination_contract}
## Plan before you build (FIRST round only)
If `qa/verdict.json` does not exist yet, this is the first round: **before writing any code, write
`docs/PLAN.md`** surveying the whole task, then build to that plan. Do NOT re-plan from scratch on later
rounds — follow and refine PLAN.md. Keep each item to a line or two:
- **Corpus**: the families/capsules you must pass and the distinct op/shape/dtype/epilogue cases in them.
- **Input ingestion**: how your `parse` entrypoint consumes the interface MLIR — parse it **structurally**
  (a real IR / grammar parser), do NOT hand-roll a lexer or text-parser; a bespoke input parser is the most
  common self-inflicted first-round failure.
- **Dialect + lowering**: the target-dialect ops you define and the interface->target rewrite passes.
- **Encoding**: how each instruction class is packed from the derived ISA facts (opcodes/fields), and how
  you check that encoding before grading.
- **Addressing + termination**: where operand addresses come from and how the program signals completion.
- **Verification loop**: the cheapest self-check per change, escalating to the full set only to converge.
It is your design contract with yourself — short and honest; update it only when your strategy changes.

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
- **Nor any other capsule-specific value.** Read every extent and every attribute from the capsule you
  were handed, never from the set you happened to see. Two things a held-out capsule caught in a
  submission that passed its public suite: a dispatch guarded on ONE operand extent, so a matmul with a
  second tile in that dimension wrote nothing; and an accumulator-scale epilogue that emitted the literal
  constant the public capsule happened to use, while the surrounding code parsed the real attribute and
  discarded it. Both passed every public capsule and failed the holdout that changed only that value.
- Do not read withheld goldens, hidden capsules, prior backends, or Merlin internals.

## Target ISA facts (derived — build your lowering on these)
{isa_spec}{isa_facts}
{isa_dev_tools}{seam_menu}{enforced_workflow}## Final status line (end of `submission/REPORT.md`) — write exactly one of:
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
instead of guessing from the file tree (neither imports the oracle or the grader). Both are runnable
CLIs exactly like `isa_tools.py` — run them from the workspace root:
- `python cca_contract.py check-bijection {target}` — the *what-to-build* checklist: which lever axes
  this target's ISA/RTL admits vs. which the compiler already routes (`orphan_fields` = leverable axes
  still to wire; `orphan_routes` = routes with no backing lever). Build every leverable axis; add no
  phantom. (API form, if you prefer: `from cca_contract import check_bijection; check_bijection("{target}")`.)
- `python action_catalog.py escalation-ladder <axis> {target}` — for one axis, the full
  FLAG→KNOB→HEURISTIC→PASS→CODEGEN ladder weakest→strongest, each row naming the concrete OOT-relative
  seam file to edit and whether it is forkable today (the "which section, and the next stronger lever"
  answer). The seams point at YOUR generated OOT package, not our in-tree reference. (API form:
  `from action_catalog import escalation_ladder; escalation_ladder("<axis>", "{target}")`.)

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

When a capsule passes the cheap tiers but fails the hardware oracle (the numeric/trace check is green yet
the RTL oracle disagrees), OBSERVE the hardware behavior of YOUR OWN command buffer with the lite debugger:
- `python isa_tools.py debug submission/command_buffer.json --capsule <name>` — answers YOUR command
  buffer on the RTL-derived arc model and reports per-op HARDWARE STATE: `per_command` (cycles +
  scratchpad-read / accumulator-write / DRAM-refill counts for each command), aggregate `metrics`
  (bytes moved, accumulator commits, evictions), and the RTL `oracle` fingerprint. The output VALUES and
  the pass/fail verdict are WITHHELD (that is the answer key). This runs your INTENDED computation, so pair
  it with `disasm`/`lint` on the emitted `.mlir` (the encoding) to catch a field the command buffer cannot
  carry (a store stride, a readout dtype, a tile DRAM offset).

"""


def _is_assisted_arm(arm: str) -> bool:
    """The seam menu is exposed only to arms that are actually granted the CCA spine (the assisted arms);
    raw_baseline is not, so it must not be told to reach for tools it cannot use. Target-agnostic."""
    return "merlin_assisted" in arm or "rtlchecks" in arm


def _enforced_workflow(arm: str, endpoint_kind: str, granted_tools, target: str, simt_mlir: bool = False) -> str:
    """The per-arm MANDATORY development workflow — a compulsory checklist (not an optional aid) matching
    the experiment ladder: it names ONLY the tools the arm actually grants, so raw_baseline gets just the
    build + self-check floor, arm-2 the C++ generators, arm-3 the xDSL/CCA + own-artifact lint, and arm-4
    ALSO the RTL-facts derivation. Fully DERIVED from ``granted_tools`` (the arm's allowed-tool set from its
    bundle grant) so it stays target-agnostic and precisely per-arm; when the grant set is not threaded
    (direct/test callers) it falls back to a coarse arm-string approximation. This is what compels the agent
    to USE the tools (availability != usage) and to develop in xDSL with no regex."""
    if granted_tools is None:                                   # coarse fallback; real runs thread the grant
        assisted = _is_assisted_arm(arm)
        granted_tools = set()
        if assisted:
            granted_tools |= {"xdsl_dialects", "kernels/cca", "action_catalog"}
        if "rtlchecks" in arm:
            granted_tools.add("targetgen/rtl/")
        if arm == "cpp_merlininfra":
            granted_tools = {"targetgen/generate/mlir_scaffold"}
    g = granted_tools
    def _has(sub: str) -> bool:
        return any(sub in p for p in g)
    has_xdsl = _has("xdsl_dialects") or _has("targetgen/synthesize")
    has_cca = _has("kernels/cca") or _has("action_catalog")
    has_rtl = _has("targetgen/rtl/") or _has("rtl_facts")
    has_cpp = _has("mlir_scaffold") or _has("llvm_plan") or _has("target_repo")
    art = ("submission/lowered.llvm.mlir" if simt_mlir
           else "submission/kernel.S" if endpoint_kind == "external_backend"
           else "your emitted submission/*.mlir")
    L = ["## MANDATORY development workflow (do ALL of these BEFORE the final status line — not optional)",
         "1. Your compiler backend lives under `submission/`; compute is COMPILER-GENERATED (never a hand kernel).",
         "2. Base every ISA / mesh / datapath / encoding decision on the **Target ISA facts** above + the",
         "   capability contract under `merlin/contract/` — never guess or hardcode; derive any fact not given.",
         "3. After EVERY build, run `python3 agent_selfcheck.py --submission submission --capsules all` and",
         "   iterate until all required capsules pass — a submission you did not self-check is not acceptable.",
         "4. GRADEABLE-FLOOR FIRST (do this in your FIRST minutes, before deep encoder / ISA / parse work):",
         "   write `submission/manifest.yaml` declaring your entrypoints + a minimal CLI that ANSWERS all of",
         "   them (even trivially / with empty output) so `agent_selfcheck` can invoke your package and the",
         "   grader reaches the capsules. A round that ends WITHOUT a valid manifest scores 0 no matter how",
         "   much compiler you built — make the package structurally gradeable EARLY, THEN iterate on real",
         "   codegen. If you run low on time, a graded-but-imperfect package beats an ungradeable one."]
    n = 5
    if has_cpp and not has_xdsl:                                # arm-2
        L.append(f"{n}. Scaffold the package with the granted C++ OOT generators "
                 "(`targetgen/generate/{mlir_scaffold,llvm_plan,target_repo}`), not ad-hoc hand files.")
        n += 1
    if has_xdsl:                                                # arm-3 and arm-4
        L.append(f"{n}. Author the backend as an **xDSL pass pipeline** (`xdsl_dialects/`, "
                 "`targetgen/synthesize/`, `targetgen/generate/`) — structured IR passes, NOT ad-hoc string "
                 "assembly, and with **NO regular expressions** (`import re` / regex text-matching is "
                 "prohibited; parse the IR structurally). This is checked on your submission.")
        n += 1
        if has_cca:
            L.append(f"{n}. Enumerate your lever set: run `python cca_contract.py check-bijection {target}` "
                     f"+ `python action_catalog.py escalation-ladder <axis> {target}` (runnable CLIs, like "
                     "`isa_tools.py`) and build every leverable axis they list.")
            n += 1
        if endpoint_kind == "external_backend" and not simt_mlir:
            L.append(f"{n}. Assemble every instruction word with `python isa_tools.py asm ops.txt` — it packs "
                     "the exact bits this target's own encoder uses and REFUSES to emit a wrong word. NEVER "
                     "hand-pack `.word` hex: a word that is opcode-legal but decodes to a DIFFERENT op moves "
                     "no data (your DMA/load silently no-ops) and scores 0 even though `lint` passes. Confirm "
                     "with `disasm` that each word decodes back to the op you intended.")
            n += 1
        if simt_mlir:
            # A SIMT-MLIR target emits an LLVM-dialect kernel LOWERING (compiled fork-free), not a self-hosted
            # word stream to assemble/lint. The mandate is that your xDSL passes BUILD the `llvm.func @<kernel>`
            # module (verified IR) that the shared llvmlower front compiles — the runner owns the harness.
            L.append(f"{n}. Emit `{art}` per the LLVM-dialect MLIR contract above: your xDSL pass pipeline BUILDS "
                     f"the `builtin.module` with `llvm.func @{target}_kernel(...)` (pointer operands in "
                     "[weight]++[lhs]++[out] order) that COMPUTES the op and `llvm.return`s — no prints, no "
                     "`.insn`/`.word`, no `mu_schedule`. Verify the module parses/`verify()`s before self_check.")
            n += 1
        else:
            L.append(f"{n}. Before every self_check, run `python isa_tools.py lint` and `disasm` on {art} and "
                     "confirm every instruction decodes (nothing UNKNOWN or ambiguous) and the kernel halts.")
            n += 1
    if has_rtl:                                                 # arm-4 only (the CIRCT / RTL-facts arm)
        L.append(f"{n}. RTL-checks arm: DERIVE the ISA / mesh / datapath from the granted RTL-extracted facts "
                 "(`targetgen/rtl/` + the RTL facts pin) — do not hand-invent them — and run the CIRCT RTL "
                 "checks on your lowering. Your backend must be a compilation FROM those RTL-derived facts.")
        n += 1
    return "\n".join(L) + "\n\n"


def render_prompt(te, manifest, experiment: str = "full", arm: str = "raw_baseline",
                  granted_tools=None) -> str:
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
    enforced_workflow = _enforced_workflow(arm, s["endpoint_kind"], granted_tools, s["target"], simt_mlir=_mlir)
    return _TEMPLATE.format(target=s["target"], scope_label=scope, corpus_families=families,
                            tool_stem=s["tool_stem"], kernel_symbol=s["kernel_symbol"],
                            endpoint_desc=s["endpoint_desc"], emit_framing=s["emit_framing"],
                            emit_symbol_note=s["emit_symbol_note"], grading_model=s["grading_model"],
                            isa_facts=s["isa_facts"], sim_tier_ladder=ladder, seam_menu=seam_menu,
                            isa_spec=s["isa_spec"], dram_contract=s["dram_contract"],
                            termination_contract=s["termination_contract"], isa_dev_tools=isa_dev_tools,
                            enforced_workflow=enforced_workflow)
