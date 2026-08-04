"""Derived disassembler + instruction-coverage for a self-hosted-ISA target. Decodes each emitted 32-bit
word back to ``{mnemonic, role, operands}`` using the target's OWN decode signatures + operand field maps
(the inverse of :mod:`~merlin.targetgen.isa_asm`), and diffs the instruction classes a kernel actually
contains against those a capsule requires.

Two feedback failures this closes, both instant and oracle-free: an INVENTED encoding shows up as a word
that decodes to nothing the ISA defines (``illegal``), and a wrong kernel shows up as a coverage gap
("matmul capsule, zero compute instructions"). No golden, no target name, no ``re``.
"""
from __future__ import annotations

from .isa_model import IsaModel
from . import isa_taxonomy as IT


def _matches(word: int, ent: dict) -> bool:
    m, v = ent.get("fixed_mask"), ent.get("fixed_value")
    return isinstance(m, int) and isinstance(v, int) and (word & m) == v


def _decode_operand(bits: list[int | None], word: int) -> int:
    """Gather an operand value out of a word via the derived per-bit map (linear bits only; a ``-1``
    non-linear bit cannot be reversed and is skipped — the value is then best-effort)."""
    val = 0
    for i, wb in enumerate(bits):
        if isinstance(wb, int) and wb >= 0 and (word >> wb) & 1:
            val |= (1 << i)
    return val


def _sel(word: int, hi: int, lo: int) -> int:
    return (word >> lo) & ((1 << (hi - lo + 1)) - 1)


def _fixed_reverse_opcode(model: IsaModel) -> dict[int, list[str]]:
    """Map the (opcode-field-width) opcode value -> mnemonic(s). The opcode-table values may be wider than
    the opcode field (an extension selector carved off the top); match on the field-width low bits, so an
    extension variant that shares the base opcode groups under it (disambiguated later by the ext field)."""
    hi, lo = model.field_layout["opcode"]
    mask = (1 << (hi - lo + 1)) - 1
    rev: dict[int, list[str]] = {}
    for mnem, val in model.opcode_table.items():
        rev.setdefault(int(val) & mask, []).append(mnem)
    return rev


def _disassemble_fixed(model: IsaModel, words: list[int]) -> list[dict]:
    """Field-layout decode for a fixed-format ISA (every instruction shares one layout, opcode-selected).
    Extracts each declared field; ``illegal`` when the opcode value is not in the derived table."""
    wmask = (1 << model.inst_width) - 1
    nib = (model.inst_width + 3) // 4
    op_hi, op_lo = model.field_layout["opcode"]
    rev = _fixed_reverse_opcode(model)
    recs: list[dict] = []
    for i, raw in enumerate(words):
        w = int(raw) & wmask
        opv = _sel(w, op_hi, op_lo)
        mnems = rev.get(opv)
        operands = {name: _sel(w, hi, lo) for name, (hi, lo) in model.field_layout.items()
                    if name != "opcode"}
        rec: dict = {"index": i, "word": f"0x{w:0{nib}x}", "operands": operands}
        if not mnems:
            rec.update({"illegal": True, "mnemonic": None})
        else:
            rec["mnemonic"] = mnems[0]
            if len(mnems) > 1:
                rec["ambiguous"] = list(mnems)
        recs.append(rec)
    return recs


def disassemble(model: IsaModel, words: list[int]) -> list[dict]:
    """Decode a word stream → per-instruction records. Each record is
    ``{index, word, mnemonic, role, operands}`` for a recognized instruction, or
    ``{index, word, illegal: True, mnemonic: None}`` for a word that matches NO op the ISA defines (the
    fingerprint of an invented/garbled encoding). ``ambiguous`` lists all classes when a word matches more
    than one signature (an overlapping encoding worth surfacing). Empty model → every word is ``illegal``.

    Two decode strategies, chosen by the model: a FIXED-FORMAT target (one field layout selected by an
    opcode field — the mlc ``isa_encoding`` derivation) decodes by field extraction at the target's
    ``inst_width``; a variable-format self-hosted ISA decodes by matching each op's derived signature."""
    if model.is_fixed_format():
        return _disassemble_fixed(model, words)
    recs: list[dict] = []
    entries = list(model.by_mnemonic.values())
    for i, raw in enumerate(words):
        w = int(raw) & 0xFFFFFFFF
        hits = [e for e in entries if _matches(w, e)]
        if not hits:
            recs.append({"index": i, "word": f"0x{w:08x}", "illegal": True, "mnemonic": None})
            continue
        ent = hits[0]
        ops = {attr: _decode_operand(bits, w) for attr, bits in (ent.get("fields") or {}).items()}
        rec = {"index": i, "word": f"0x{w:08x}", "mnemonic": ent.get("class"),
               "role": ent.get("role"), "operands": ops}
        if len(hits) > 1:
            rec["ambiguous"] = [e.get("class") for e in hits]
        recs.append(rec)
    return recs


def present_classes(records: list[dict]) -> list[str]:
    """Ordered, deduped semantic classes present in a disassembled stream (recognized instructions only)."""
    seen, out = set(), []
    for r in records:
        c = r.get("mnemonic")
        if c and not r.get("illegal") and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def coverage(model: IsaModel, records: list[dict], *, required: list[str] | None = None,
             op: str = "matmul", output_dtype: str | None = None,
             epilogue: tuple[str, ...] = (), movement: bool = False) -> dict:
    """Diff the instruction classes a kernel CONTAINS against those it should. ``required`` may be supplied
    directly (e.g. a capsule's declared ``expected.instruction_classes``); otherwise it is derived by
    semantic ROLE from the model (``op``/``output_dtype``/``epilogue``/``movement``), never a hardcoded
    list. Returns ``{required, present, missing, n_illegal}`` — ``missing`` non-empty or ``n_illegal`` > 0
    is an actionable, oracle-free failure the agent can fix before spending an oracle run."""
    if required is None:
        required = IT.required_classes_from_roles(model.roles, op=op, output_dtype=output_dtype,
                                                  epilogue=epilogue, movement=movement)
    present = present_classes(records)
    missing = [c for c in required if c not in present]
    n_illegal = sum(1 for r in records if r.get("illegal"))
    return {"required": list(required), "present": present, "missing": missing, "n_illegal": n_illegal}
