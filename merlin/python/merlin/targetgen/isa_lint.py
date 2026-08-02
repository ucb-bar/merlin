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

Finding = dict


def lint(model: IsaModel, words: list[int]) -> list[Finding]:
    """Lint an assembled word stream → a list of findings, each
    ``{rule, severity, detail[, index]}`` (severity ∈ error/warning/info). Empty findings = clean by these
    checks (not a full correctness proof — that is the oracle's job). An empty model yields a single INFO
    (no ISA definition to lint against)."""
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
    #    and fails the functional tier before numerics. Checked against the DERIVED halt op set when known.
    real = [r for r in recs if not r.get("illegal")]
    if model.halt_mnemonics:
        halt = set(model.halt_mnemonics)
        if not any(r.get("mnemonic") in halt for r in real):
            findings.append({"rule": "no_halt", "severity": "error",
                             "detail": f"no terminating instruction ({', '.join(sorted(halt))}) present — "
                                       "the program will not halt and every capsule fails before numerics; "
                                       "emit the terminator as the final instruction"})
        elif real and real[-1].get("mnemonic") not in halt:
            findings.append({"rule": "halt_not_last", "severity": "warning",
                             "detail": "a terminating instruction is present but is not the last instruction; "
                                       "ensure every control path ends at the terminator"})
    else:
        findings.append({"rule": "halt_unknown", "severity": "info",
                         "detail": "termination could not be statically verified (the halt op set is not "
                                   "derived for this target); confirm the kernel reaches the ISA terminator"})

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
