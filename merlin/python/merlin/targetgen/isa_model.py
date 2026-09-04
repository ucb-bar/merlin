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

from collections.abc import Mapping
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
    # --- fixed-format encoding (a target whose EVERY instruction shares one field layout, opcode-selected,
    # e.g. a wide-word SIMT core). Derived by mlc's isa_encoding pass (field bit-ranges + opcode table) and
    # consumed by the disassembler's field-layout path. Empty for a variable-format self-hosted ISA, whose
    # per-op decode signatures live in ``by_mnemonic`` instead.
    inst_width: int = 32
    field_layout: dict[str, tuple[int, int]] = field(default_factory=dict)
    opcode_table: dict[str, int] = field(default_factory=dict)
    # address-space selection: ``address_spaces`` maps a memory-space name to the value the
    # ``address_space_field`` (an opcode-extension selector) carries for it — derived from the target's own
    # address-space qualifier macros. Empty for a single flat address space.
    address_spaces: dict[str, int] = field(default_factory=dict)
    address_space_field: str = ""
    # runtime ABI: the SIMT runtime contract a fork-free backend needs on top of the encoding — derived by
    # mlc's runtime_abi pass from the target's own artifacts (core RTL CSR table + SFU dispatch + core params,
    # the shipped BSP asm, the sim config). Keys: ``base_isa_family`` (e.g. "riscv32"), ``xlen``,
    # ``special_csrs`` {role: number}, ``sfu_ops`` {op: {opcode, funct3}}, ``apertures`` {name: address},
    # ``provenance``. Empty for a target whose runtime ABI has not been derived (the consumer fails closed).
    runtime_abi: dict = field(default_factory=dict)

    def special_csr(self, role: str) -> int:
        """The derived CSR number for a named role (e.g. ``warp_id``/``num_warps``/``mhartid``). Raises when the
        runtime ABI does not carry it — the fork-free consumer must fail closed, never guess a CSR number."""
        csrs = (self.runtime_abi or {}).get("special_csrs") or {}
        if role not in csrs:
            raise KeyError(f"runtime_abi has no special CSR {role!r} for target {self.target!r} "
                           "(derive it, do not hardcode)")
        return int(csrs[role])

    def sfu_op(self, op: str) -> dict:
        """The derived ``{opcode, funct3}`` for a SIMT-control op (e.g. ``tmc``/``wspawn``). Raises when the
        runtime ABI does not carry it (fail closed)."""
        ops = (self.runtime_abi or {}).get("sfu_ops") or {}
        if op not in ops:
            raise KeyError(f"runtime_abi has no SFU op {op!r} for target {self.target!r} "
                           "(derive it, do not hardcode)")
        return dict(ops[op])

    def aperture(self, name: str) -> int:
        """The derived address for a named memory aperture (``dram_base``/``stack_base``/``console_mmio``).
        Raises when the runtime ABI does not carry it (fail closed)."""
        aps = (self.runtime_abi or {}).get("apertures") or {}
        if name not in aps:
            raise KeyError(f"runtime_abi has no aperture {name!r} for target {self.target!r} "
                           "(derive it, do not hardcode)")
        return int(aps[name])

    def base_isa_family(self) -> str:
        """The derived base-ISA family (e.g. ``riscv32``), or ``""`` when not derived."""
        return str((self.runtime_abi or {}).get("base_isa_family") or "")

    def is_fixed_format(self) -> bool:
        """True when the target's ISA is one fixed field layout selected by an opcode field (the mlc
        isa_encoding derivation), so the disassembler extracts fields rather than matching signatures."""
        return bool(self.field_layout) and "opcode" in self.field_layout and bool(self.opcode_table)

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


def isa_model_from_encoding(target: str, fact: dict) -> IsaModel:
    """Build a fixed-format :class:`IsaModel` from an mlc ``isa_encoding`` fact (``{inst_width, fields,
    opcodes, provenance}`` — the field bit-ranges + opcode table derived from the target's RTL decoder).
    Returns an empty model when the fact carries no field layout (caller no-ops). Nothing here is
    target-specific; the fact is the sole input."""
    fields_in = fact.get("fields") or {}
    field_layout = {str(k): (int(v[0]), int(v[1])) for k, v in fields_in.items()
                    if isinstance(v, (list, tuple)) and len(v) == 2}
    opcode_table = {str(k): int(v) for k, v in (fact.get("opcodes") or {}).items()}
    if not field_layout or not opcode_table:
        return IsaModel(target=target)
    width = int(fact.get("inst_width") or 0) or (max((hi for hi, _ in field_layout.values()), default=31) + 1)
    spaces = {str(k): int(v) for k, v in (fact.get("address_spaces") or {}).items()}
    as_field = str(fact.get("address_space_field") or "")
    runtime_abi = fact.get("runtime_abi") if isinstance(fact.get("runtime_abi"), dict) else {}
    return IsaModel(target=target, inst_width=width, field_layout=field_layout, opcode_table=opcode_table,
                    address_spaces=spaces, address_space_field=(as_field if as_field in field_layout else ""),
                    runtime_abi=runtime_abi or {})


def isa_model_for_target(target: str) -> IsaModel:
    """The IsaModel for a target, preferring the mlc-derived fixed-format encoding fact (from the RTL
    decoder) when present, else the shipped-ISA-definition probe (:func:`isa_model_for`). This is the seam a
    fixed-format wide-word target (e.g. a SIMT core) enters through; a variable-format self-hosted ISA falls
    through to the probe path unchanged."""
    try:
        from .rtl import mlc_bridge
        fact = mlc_bridge.isa_encoding_for(target)
    except Exception:  # noqa: BLE001 — mlc absent / cache missing -> fall back to the probe path
        fact = None
    if fact:
        m = isa_model_from_encoding(target, fact)
        if not m.is_empty() or m.is_fixed_format():
            return m
    probed = isa_model_for(target)
    if not probed.is_empty():
        return probed
    # THIRD SOURCE: a RoCC-style accelerator has neither an mlc encoding fact nor a shipped ISA
    # definition, so both paths above return empty and the target reads as "ships no ISA definition"
    # -- which says we looked in two places, not that the ISA is unknown. Its RTL facts carry the
    # decode table `merlin.kernels.decode.rocc` already disassembles against. Consulted last so a
    # target with either richer source is unaffected, and still empty when no table exists.
    try:
        from .rtl import facts as _facts
        derived = isa_model_from_rocc_facts(target, _facts.load_facts(target) or {})
    except Exception:  # noqa: BLE001 - no facts bundle is an absence of evidence, not an error
        return probed
    return derived if not derived.is_empty() else probed

#: RISC-V base instruction-word field positions. A property of the 32-bit RISC-V encoding itself --
#: field WIDTHS, not accelerator values -- and the same layout ``merlin.kernels.decode.rocc``
#: disassembles against. Every VALUE (which opcode, which funct means what) comes from the target's
#: own RTL decode table. Nothing here is per-target.
_RISCV_FIELD_BITS = {
    "funct": list(range(25, 32)), "rs2": list(range(20, 25)), "rs1": list(range(15, 20)),
    "xd": [14], "xs1": [13], "xs2": [12], "rd": list(range(7, 12)),
    "opcode": list(range(0, 7)),
}
_ROCC_FUNCT_SHIFT = 25


def isa_model_from_rocc_facts(target: str, facts: "Mapping[str, Any]") -> IsaModel:
    """Build the ISA model a RoCC-style accelerator derives from its OWN decoder table.

    :func:`isa_model_for_target` reaches two sources: an mlc fixed-format encoding fact, and a shipped
    ISA definition in the model venv. A RoCC accelerator has neither, so it reports "ships no ISA
    definition" -- which reads as though the ISA were unknown. It is not. The target's RTL facts carry
    ``interfaces.funct_decode_table`` (``custom_opcode`` + ``legal_funct`` + ``names``), the same
    table :mod:`merlin.kernels.decode.rocc` already disassembles against. This is that third source.

    The instruction word is RISC-V's, so the field POSITIONS come from the base encoding; the opcode
    and every funct value come from this target's decoder. Verified by round-tripping all 26 gemmini
    mnemonics through ``merlin.kernels.decode.rocc.fields_of``: 26/26 decode back to the same name.

    ``xd``/``xs1``/``xs2`` are carried as operand fields and are never identity bits. In RoCC they say
    whether THIS instruction writes rd and reads rs1/rs2, so they vary between instructions of the
    same command; pinning them dropped every conformant instruction with a different operand shape.

    ``facts`` is the ``facts`` body of the target's RTL fact bundle. Returns an EMPTY model when the
    bundle carries no decode table, so a caller no-ops rather than assuming an encoding.
    """
    body = facts.get("facts") if isinstance(facts.get("facts"), Mapping) else facts
    table = next((i for i in (body.get("interfaces") or ())
                  if isinstance(i, Mapping) and i.get("name") == "funct_decode_table"), None)
    if not table:
        return IsaModel(target=target)
    opcode, names = table.get("custom_opcode"), table.get("names") or {}
    if not isinstance(opcode, int) or isinstance(opcode, bool) or not names:
        return IsaModel(target=target)

    identity_mask = (0x7F << _ROCC_FUNCT_SHIFT) | 0x7F
    operands = {k: list(v) for k, v in _RISCV_FIELD_BITS.items() if k not in ("funct", "opcode")}
    by_mnemonic: dict[str, Any] = {}
    for raw_funct, mnemonic in sorted(names.items(), key=lambda kv: int(kv[0])):
        funct = int(raw_funct)
        by_mnemonic[str(mnemonic)] = {
            "class": "RoCCCustom", "mnemonic": str(mnemonic), "opcode": opcode,
            "role": "accelerator", "funct3": None, "funct7": funct, "funct2": None,
            "fixed_mask": identity_mask,
            "fixed_value": (funct << _ROCC_FUNCT_SHIFT) | opcode,
            "fields": {k: list(v) for k, v in operands.items()},
        }
    return IsaModel(
        target=target, by_mnemonic=by_mnemonic, asm_mnemonics=tuple(sorted(by_mnemonic)),
        # A role maps to the instruction CLASSES that fill it -- `candidate_ops` selects
        # mnemonics whose entry["class"] is in this list, so mapping to mnemonics yields an
        # empty menu for every role.
        roles={"accelerator": ["RoCCCustom"]}, inst_width=32,
        field_layout={k: (max(v), min(v)) for k, v in _RISCV_FIELD_BITS.items()},
        opcode_table={m: e["funct7"] for m, e in by_mnemonic.items()},
    )
