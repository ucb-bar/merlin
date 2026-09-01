"""Does an instruction perturb architectural state it does not declare as a destination?

An ISA model gives each instruction its encoding and its destination FIELDS (``vd``, ``rd``, …). Nothing
in an encoding says what the hardware actually writes, and a shipped architectural spec can disagree with
the elaborated design. When it does, the divergence is close to undetectable from artifacts: the emitted
program decodes correctly, every instruction is of the right class for its op, and the kernel still
returns a wrong answer -- so a decode census, a role census and a lint pass all report clean.

That is not hypothetical. One target's zero-fill immediate declares a ``vd`` field, and its shipped
reference implementation writes exactly that register; on the RTL-derived oracle it clears the WHOLE
vector register file. A backend that staged two operands emitted the fill twice, the second one wiped the
first operand, and the kernel returned its second input unchanged. Six rounds of agent effort went into
the arithmetic.

The measurement here is a DIFFERENTIAL on the target's own oracle, so it needs no model of the
instruction and no per-target special case: take a program that is known to produce a correct result,
inject the instruction under test at a point where operand registers are LIVE but not yet consumed, and
re-run. If the output moves, the instruction wrote something the program was still relying on. That is a
fact about the hardware, obtained from the hardware.

The oracle is injected as a callable, so the analysis is testable without one.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

# Roles that name an operand-bearing register transfer vs. an arithmetic consumer. Both come from the
# ISA model's own derived role vocabulary -- nothing here spells a mnemonic.
_LOAD_ROLES = ("memory",)
_COMPUTE_ROLE_PREFIX = "tensor_compute"


def declared_destinations(model, mnemonic: str) -> set[str]:
    """The destination FIELDS this instruction declares (the write-set it advertises)."""
    fields = set(model.fields_of(mnemonic) or {})
    return {f for f in fields if f in ("vd", "rd")}


def live_injection_index(model, words: Iterable[int]) -> int | None:
    """The index to inject at: after the LAST operand load and before the FIRST arithmetic consumer.

    At that point every operand the program loaded is live in a register and none has been consumed, so
    an instruction that clobbers register state will change the result, while one that respects its
    declared destination cannot. Returns None when the program has no such window (nothing loaded, or
    nothing computed), where the probe would prove nothing and must not report a verdict.
    """
    from . import isa_disasm

    records = isa_disasm.disassemble(model, list(words))
    first_compute = next((r["index"] for r in records
                          if (r.get("role") or "").startswith(_COMPUTE_ROLE_PREFIX)), None)
    if first_compute is None:
        return None                                  # nothing consumes a register; nothing to perturb
    # A register transfer BEFORE that consumer. The role vocabulary does not separate load from store
    # (both are `memory`), and taking the last memory op outright picks a store that runs after the
    # compute -- which is not a live window at all, and reported "no window" for every program.
    if not any((r.get("role") or "") in _LOAD_ROLES and r["index"] < first_compute for r in records):
        return None
    return first_compute


def probe_instruction(model, words: list[int], mnemonic: str, operands: dict[str, int], *,
                      run: Callable[[list[int]], Any], settle: int | None = None) -> dict[str, Any]:
    """Inject ``mnemonic`` into ``words`` at the live window and report whether the output moved.

    ``run`` takes a word list and returns the program's outputs (any comparable structure). ``settle`` is
    an optional word to place immediately after the injection (a target whose pipeline needs an explicit
    wait), so a difference cannot be blamed on not having waited long enough -- pass the LONGEST settle
    the program itself uses and the timing explanation is excluded by construction.
    """
    from . import isa_asm

    at = live_injection_index(model, words)
    if at is None:
        return {"mnemonic": mnemonic, "verdict": "no_live_window",
                "reason": "program has no load-then-compute window; a difference would not be attributable"}
    injected = list(words)
    payload = [isa_asm.assemble_line(model, mnemonic, operands)]
    if settle is not None:
        payload.append(int(settle))
    injected[at:at] = payload
    before, after = run(list(words)), run(injected)
    moved = before != after
    return {
        "mnemonic": mnemonic,
        "declared_destinations": sorted(declared_destinations(model, mnemonic)),
        "injected_at": at,
        "perturbs_live_state": bool(moved),
        "verdict": "writes_beyond_declared_destination" if moved else "respects_declared_destination",
        "baseline_outputs": before,
        "injected_outputs": after,
    }


def audit(model, words: list[int], candidates: Iterable[tuple[str, dict[str, int]]], *,
          run: Callable[[list[int]], Any], settle: int | None = None) -> list[dict[str, Any]]:
    """Probe several instructions against one known-good program; the offenders come back first."""
    out = [probe_instruction(model, words, m, ops, run=run, settle=settle) for m, ops in candidates]
    return sorted(out, key=lambda r: not r.get("perturbs_live_state"))
