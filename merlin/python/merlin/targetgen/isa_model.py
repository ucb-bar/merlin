"""ONE consolidated, DERIVED machine model for a self-hosted-ISA (``external_backend``) target — the single
seam the raw-ISA developer tools (assembler / disassembler / linter / debugger) all consume.

Today a target's ISA facts are scattered across the fact bundle, mlc discovery, the instruction taxonomy,
and the shipped green-card prose. This module assembles the subset the dev tools need into one object,
resolved PURELY from the target's descriptor + its own shipped ``isa_definition.py`` (via the model-venv
:mod:`~merlin.targetgen.isa_taxonomy`). It holds NO opcode/field table of its own and NO target-name
literal — everything is derived from the target's own encoder, so any target that ships an ISA definition
gets the tools for free, and a target that ships none yields an empty model (the tools then no-op).

The per-mnemonic ``fields`` map (operand-bit -> word-bit, derived by differential-probing the model's own
``to_bytecode``) is what lets the assembler PACK and the disassembler UNPACK operands into exactly the bits
the hardware uses — with no hand-authored, per-target field table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IsaModel:
    """A target's derived instruction model. ``by_mnemonic`` maps each op's (class) name to its derived
    entry ``{class, role, opcode, funct*, fixed_mask, fixed_value, fields}``; ``asm_mnemonics`` maps the
    assembler syntax the target's ISA spec exposes to that class name; ``roles`` groups classes by their
    derived structural role; ``dram_base`` is the target's DRAM aperture floor; ``halt_mnemonics`` is the
    behaviorally-derived terminator op set (ops whose sole semantic effect is asserting the machine's finish
    flag) and ``halt_signatures`` their decode (fixed_mask, fixed_value) signatures for detection — both empty
    for a target whose ISA defines no such op (the linter then reports an honest INFO)."""

    target: str
    by_mnemonic: dict[str, dict] = field(default_factory=dict)
    asm_mnemonics: dict[str, str] = field(default_factory=dict)
    roles: dict[str, list[str]] = field(default_factory=dict)
    dram_base: int = 0
    halt_mnemonics: tuple[str, ...] = ()
    halt_signatures: tuple[tuple[int, int], ...] = ()

    # -- lookups ---------------------------------------------------------------------------------
    def is_empty(self) -> bool:
        """True when the target ships no ISA definition (the tools then no-op instead of guessing)."""
        return not self.by_mnemonic

    def resolve(self, mnemonic: str) -> dict | None:
        """The derived entry for a mnemonic. Accepts either the op's class name (a ``by_mnemonic`` key) or
        an assembler mnemonic the target's ISA spec exposes (mapped via ``asm_mnemonics``); case-insensitive
        on the assembler alias. Returns None if the mnemonic is not defined by this ISA."""
        if mnemonic in self.by_mnemonic:
            return self.by_mnemonic[mnemonic]
        for cand in (mnemonic, mnemonic.upper(), mnemonic.lower()):
            cls = self.asm_mnemonics.get(cand)
            if cls and cls in self.by_mnemonic:
                return self.by_mnemonic[cls]
        return None

    def fields_of(self, mnemonic: str) -> dict[str, list[int | None]]:
        """The operand-bit -> word-bit map for a mnemonic's operands ({attr: [word_bit|None|-1]}). Empty if
        the op takes no operands or the mnemonic is undefined."""
        ent = self.resolve(mnemonic)
        return dict(ent.get("fields") or {}) if ent else {}

    def signatures(self) -> list[tuple[str, int, int]]:
        """(class, fixed_mask, fixed_value) for every op with a derived decode signature — the legality
        oracle: a word is ILLEGAL iff it matches none of these (``word & mask == value``)."""
        out: list[tuple[str, int, int]] = []
        for name, ent in self.by_mnemonic.items():
            m, v = ent.get("fixed_mask"), ent.get("fixed_value")
            if isinstance(m, int) and isinstance(v, int):
                out.append((ent.get("class") or name, m, v))
        return out


def isa_model_for(te_or_target: Any, *, model_ext: str | None = None, timeout: int = 120) -> IsaModel:
    """Build the derived :class:`IsaModel` for a target. Accepts a ``TargetExperiment`` or a target name.
    Resolves the ISA taxonomy (operand field maps + decode signatures + roles) from the target's own shipped
    ISA definition in the model venv, and the DRAM aperture floor from the target's memory map. Returns an
    EMPTY model (``is_empty()``) for a target that ships no ISA definition — so a caller/tool no-ops rather
    than assuming any target. No target-name literal; nothing here is specific to one accelerator."""
    from . import isa_taxonomy as IT

    # descriptor + target name, without importing a specific target
    if isinstance(te_or_target, str):
        from .target_experiment import load_target_experiment
        from merlin.common.paths import merlin_dir
        target = te_or_target
        p = merlin_dir() / "experiments" / "capsule_bench" / "targets" / target / "target_experiment.yaml"
        te = load_target_experiment(p) if p.is_file() else None
    else:
        te = te_or_target
        target = getattr(te, "target", "")

    if te is None:
        return IsaModel(target=target)

    try:
        tax = IT.derive_isa_taxonomy(te, model_ext=model_ext, timeout=timeout)
    except Exception:  # noqa: BLE001 — model venv / ISA def absent -> empty model, tools no-op
        tax = {}
    if not tax:
        return IsaModel(target=target)

    by_mnem = tax.get("by_mnemonic") or {}
    asm = tax.get("asm_mnemonics") or {}
    roles = IT._classes_by_role(tax)

    dram_base = 0
    try:
        from .dram_facts import dram_base_for
        dram_base = int(dram_base_for(target) or 0)
    except Exception:  # noqa: BLE001 — no derivable memory map -> floor 0 (linter then skips the check)
        dram_base = 0

    # the DERIVED program-terminator set (behavioral, from isa_introspect) — op names for the linter's hint
    # text and their decode signatures for detection. Empty for a target whose ISA def has no such op (the
    # linter then reports an honest INFO instead of a false 'no halt').
    halt_mnem = tuple(tax.get("halt_mnemonics") or ())
    halt_sigs = tuple((int(m), int(v)) for m, v in (tax.get("halt_signatures") or [])
                      if isinstance(m, int) and isinstance(v, int))

    return IsaModel(target=target, by_mnemonic=by_mnem, asm_mnemonics=asm, roles=roles,
                    dram_base=dram_base, halt_mnemonics=halt_mnem, halt_signatures=halt_sigs)
