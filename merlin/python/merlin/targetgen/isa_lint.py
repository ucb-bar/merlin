"""Static, oracle-free linter for a self-hosted-ISA kernel — catches the cheap-but-fatal errors before an
oracle run is ever spent, all from the derived :class:`~merlin.targetgen.isa_model.IsaModel`.

v1 ships the two checks that are fully static and false-positive-free:
  * ``illegal_opcode`` — a word that decodes to no instruction the ISA defines (an invented/garbled
    encoding); rock-solid, since legality IS "matches a derived decode signature".
  * ``halt_present`` — the kernel must reach a terminating instruction, or it runs to the cycle cap and
    every capsule fails before numerics. Active once the target's halt op set is derived (behaviorally,
    model-side); until then it reports an honest INFO rather than a false verdict.

Deliberately NOT guessed in v1 (documented, to avoid false positives): DRAM-aperture address checking needs
light const-propagation (the address lives in a register), and config-before-compute needs a control-role
signal the datapath-derived taxonomy does not expose. Both are future extensions on the same model.

No golden, no target name, no ``re``.
"""
from __future__ import annotations

from .isa_model import IsaModel
from . import isa_disasm as D
from . import isa_taxonomy as IT

Finding = dict


def _ambiguous_findings(recs: list[Finding]) -> list[Finding]:
    """A word that matches MORE THAN ONE decode signature is an overlapping/ambiguous encoding: the
    disassembler cannot say which instruction it is, so the hardware decoder's choice cannot be trusted
    either. The derived assembler never emits one — only a hand-packed word can — so surface it as an error
    (the tool-level complement to authoring via ``isa_tools asm``). Purely decode-derived: no golden, no op
    name literal."""
    out: list[Finding] = []
    for r in recs:
        amb = r.get("ambiguous")
        if amb:
            names = ", ".join(str(a) for a in amb)
            out.append({"rule": "ambiguous_decode", "severity": "error", "index": r["index"],
                        "detail": f"word {r['word']} matches {len(amb)} instruction signatures ({names}) — "
                                  "an overlapping/ambiguous encoding the decoder cannot resolve to one op; "
                                  "assemble it with the derived encoder instead of hand-packing the word"})
    return out


def _lint_fixed(model: IsaModel, words: list[int]) -> list[Finding]:
    """Lint a FIXED-FORMAT kernel (one field layout selected by an opcode field — the mlc isa_encoding
    derivation). Two static, false-positive-free checks: an undecodable opcode, and a memory instruction
    carrying an address-space selector value the target does not define (a fabricated extension — the same
    bug class as a plain load routed into a scratchpad region). The address-space check runs only when the
    target's spaces + selector field are derived (derive-or-skip)."""
    recs = D.disassemble(model, words)
    findings: list[Finding] = []
    real = [r for r in recs if not r.get("illegal")]
    for r in recs:
        if r.get("illegal"):
            findings.append({"rule": "illegal_opcode", "severity": "error", "index": r["index"],
                             "detail": f"word {r['word']} has an opcode this ISA does not define — an "
                                       "invented or mis-encoded instruction (use the derived encoder)"})
    findings.extend(_ambiguous_findings(real))
    if words and not real:
        findings.append({"rule": "no_recognized_instructions", "severity": "error",
                         "detail": "no emitted word decodes to a defined instruction — the kernel is empty "
                                   "or entirely mis-encoded"})
    if model.address_spaces and model.address_space_field:
        valid = set(model.address_spaces.values())
        by_val = {v: k for k, v in model.address_spaces.items()}
        for r in real:
            v = r.get("operands", {}).get(model.address_space_field)
            if v is not None and v not in valid:
                spaces = ", ".join(f"{k}={n}" for k, n in sorted(model.address_spaces.items()))
                findings.append({"rule": "undefined_address_space", "severity": "warning", "index": r["index"],
                                 "detail": f"instruction '{r.get('mnemonic')}' selects address space "
                                           f"{v} in field '{model.address_space_field}', which this target "
                                           f"does not define (spaces: {spaces}) — the memory access will not "
                                           "route to a real space"})
    return findings


def lint(model: IsaModel, words: list[int], *, op: str = "matmul", output_dtype: str | None = None,
         epilogue: tuple[str, ...] = (), movement: bool = False) -> list[Finding]:
    """Lint an assembled word stream → a list of findings, each
    ``{rule, severity, detail[, index]}`` (severity ∈ error/warning/info). Empty findings = clean by these
    checks (not a full correctness proof — that is the oracle's job). An empty model yields a single INFO
    (no ISA definition to lint against).

    ``op``/``output_dtype``/``epilogue``/``movement`` describe what the capsule asks the kernel to compute;
    they drive the structural required-role check (a matmul kernel with no systolic multiply, or no memory
    op to load operands / store the result, cannot be correct). That check is purely ROLE-derived from the
    model — no target name, no class literal, no golden — and skips any role the target's ISA does not
    define (derive-or-skip, never a false positive)."""
    if model.is_fixed_format():
        return _lint_fixed(model, words)
    if model.is_empty():
        return [{"rule": "no_isa_model", "severity": "info",
                 "detail": "this target ships no ISA definition; static ISA lint is unavailable"}]

    recs = D.disassemble(model, words)
    findings: list[Finding] = []

    # 1) illegal opcode — a word matching no derived decode signature.
    for r in recs:
        if r.get("illegal"):
            findings.append({"rule": "illegal_opcode", "severity": "error", "index": r["index"],
                             "detail": f"word {r['word']} decodes to no instruction this ISA defines — "
                                       "an invented or mis-packed encoding (use the derived assembler)"})

    # 2) program termination — a kernel that never reaches a terminating instruction runs to the cycle cap
    #    and fails the functional tier before numerics. Matched by the terminator ops' DERIVED decode
    #    SIGNATURE (fixed opcode/funct bits), not by a decoded class name: a terminator and a barrier can
    #    share one coarse semantic class (e.g. both "nullary"), so a class-name match would falsely accept a
    #    fence; the signature separates them by their own opcode.
    real = [r for r in recs if not r.get("illegal")]

    # 1b) ambiguous decode — a word matching more than one signature (the disassembler cannot resolve it).
    findings.extend(_ambiguous_findings(real))

    if model.halt_signatures:
        def _is_halt(w: int) -> bool:
            return any((w & m) == v for m, v in model.halt_signatures)
        names = ", ".join(model.halt_mnemonics) or "the ISA terminator"
        if not any(_is_halt(w) for w in words):
            findings.append({"rule": "no_halt", "severity": "error",
                             "detail": f"no terminating instruction ({names}) present — the program will not "
                                       "halt and every capsule fails before numerics; emit the terminator as "
                                       "the final instruction"})
        elif words and not _is_halt(words[-1]):
            findings.append({"rule": "halt_not_last", "severity": "warning",
                             "detail": f"a terminating instruction ({names}) is present but is not the last "
                                       "instruction; ensure every control path ends at the terminator"})
    else:
        findings.append({"rule": "halt_unknown", "severity": "info",
                         "detail": "termination could not be statically verified (no terminator op is derived "
                                   "for this target); confirm the kernel reaches the ISA terminator"})

    # 3) no recognized instructions — every word is illegal (or the kernel is empty). The program does
    #    nothing the ISA can execute; the output region is never written.
    if words and not real:
        findings.append({"rule": "no_recognized_instructions", "severity": "error",
                         "detail": "no emitted word decodes to a defined instruction — the kernel is empty "
                                   "or entirely mis-encoded, so it cannot produce output"})

    # 4) required-role coverage — a kernel that omits a semantic ROLE the capsule's op needs (e.g. a matmul
    #    capsule with no systolic multiply, or no memory op to load operands / store the result) produces
    #    wrong output before numerics matter. Checked by ROLE, so it is robust to a target having several
    #    classes per role, and it skips any role the target's ISA does not define (a target that reaches the
    #    op a different way is never falsely flagged).
    present_roles = {r.get("role") for r in real if r.get("role")}
    for slot in IT.required_role_slots(op=op, output_dtype=output_dtype, epilogue=epilogue,
                                       movement=movement):
        defined = [r for r in slot if model.roles.get(r)]          # roles this target actually ships
        if not defined:
            continue                                               # target has no such role → do not require it
        if not any(r in present_roles for r in slot):
            label = defined[0]
            classes = ", ".join((model.roles.get(label) or [])[:3]) or label
            findings.append({"rule": "missing_required_role", "severity": "warning",
                             "detail": f"a '{op}' kernel needs a '{label}'-role instruction (this ISA "
                                       f"defines {classes}) but the kernel emits none — its output cannot "
                                       "be correct; add it before spending an oracle run"})

    return findings


def format_findings(findings: list[Finding]) -> str:
    """Render findings as compact agent-readable lines (most severe first)."""
    order = {"error": 0, "warning": 1, "info": 2}
    rows = sorted(findings, key=lambda f: order.get(f.get("severity"), 3))
    out = []
    for f in rows:
        at = f" @{f['index']}" if "index" in f else ""
        out.append(f"[{f.get('severity', '?').upper()}] {f.get('rule')}{at}: {f.get('detail')}")
    return "\n".join(out) if out else "clean (no static ISA-lint findings)"
