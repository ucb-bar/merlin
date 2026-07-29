#!/usr/bin/env python3
"""Model-venv helper: DERIVE a self-hosted target's instruction taxonomy from its shipped ISA definition
(the repo's ISA doc), so the corpus/trace expectations are DISCOVERED, never hardcoded. Runs INSIDE the
target model's own venv (the ISA definition imports the model package, e.g. ``npu_model``), the same
cross-venv pattern as :mod:`npu_emit`.

Given the path to the target's ISA-definition module (each instruction is a class carrying ``opcode`` +
``funct*`` fields and a semantic base pattern, e.g. ``MXUMatMul`` / ``TensorBaseOffset`` / ``MXUWeightPush``
/ ``DMARegUnary``), it emits a JSON taxonomy:

    {"by_class": {"<semantic_pattern>": [{"mnemonic","opcode","funct3","funct7","funct2"}, ...], ...},
     "by_mnemonic": {"<MNEMONIC>": {"class","opcode",...}, ...},
     "asm_mnemonics": {"<assembler-mnemonic>": "<class-name>", ...}}   # from the model's IsaSpec, if any

``asm_mnemonics`` maps the ASSEMBLER syntax (e.g. ``VMATPUSH.W.MXU0``) to the op class
(``VMATPUSH_WEIGHT_MXU0``) so an example kernel written in mnemonics can be mapped back to semantic classes.
Merlin holds no opcode table — everything here comes from the model's own ISA definition.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("_merlin_isa_def", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                      # imports the model pkg (present in this venv)
    return mod


# Candidate operand attribute names an instruction format may read in to_bytecode(). Setting one the
# format does not use is harmless (it never reaches the packed word); this list only needs to be a
# SUPERSET of operand fields across formats. No opcode/funct here — those are the FIXED bits we derive.
_OPERAND_ATTRS = ("rd", "rs1", "rs2", "imm", "vd", "vs1", "vs2", "es1", "shamt",
                  "imm16", "imm12", "imm20", "uimm", "vt", "vm", "aq", "rl", "csr")


def _fixed_signature(cls) -> tuple[int, int] | None:
    """Derive an op's FIXED opcode/funct bit signature (mask, value) from the ISA def's OWN encoder — no
    field-position assumptions, works for any instruction format. Encode the op with all-zero operands
    (``to_bytecode`` on a bare instance) to get the fixed bits, then flip each candidate operand field and
    XOR: bits that move are operand bits, the complement is the fixed opcode/funct mask. A word decodes to
    this op iff ``word & mask == value``. Returns None if the op cannot be encoded."""
    try:
        base = int(object.__new__(cls).to_bytecode()) & 0xFFFFFFFF
    except Exception:  # noqa: BLE001 — not encodable without construction; skip
        return None
    variable = 0
    for attr in _OPERAND_ATTRS:
        inst = object.__new__(cls)
        try:
            setattr(inst, attr, 0x3F)          # a small all-ones pattern; _mask truncates to field width
            variable |= (int(inst.to_bytecode()) & 0xFFFFFFFF) ^ base
        except Exception:  # noqa: BLE001 — attr not used by this format / not settable
            continue
    mask = (~variable) & 0xFFFFFFFF
    return mask, base & mask


def _pattern_module(mod) -> str:
    """The module the semantic base patterns live in (…isa_patterns) — inferred from the ISA def's own
    imports, so no model name is hardcoded."""
    for _n, obj in vars(mod).items():
        m = getattr(obj, "__module__", "") or ""
        if m.endswith("isa_patterns"):
            return m
    return "isa_patterns"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--isa-module", required=True, help="path to the target's ISA-definition .py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mod = _load_module(a.isa_module)
    patmod = _pattern_module(mod)
    by_class: dict[str, list] = {}
    by_mnem: dict[str, dict] = {}
    for name, obj in vars(mod).items():
        if name.startswith("_") or not inspect.isclass(obj) or not hasattr(obj, "opcode"):
            continue
        op = getattr(obj, "opcode", None)
        if not isinstance(op, int):                  # skip the imported format base classes (RType/IType…)
            continue
        sem = next((b.__name__ for b in obj.__mro__ if getattr(b, "__module__", "") == patmod), None)
        if sem is None:
            continue
        entry = {"mnemonic": name, "opcode": op,
                 "funct3": getattr(obj, "funct3", None), "funct7": getattr(obj, "funct7", None),
                 "funct2": getattr(obj, "funct2", None)}
        # DECODE signature: the fixed opcode/funct bits derived from the op's own encoder (mask,value), so
        # an emitted instruction word can be classified back to its semantic class — the field-decode that
        # powers the kernel class-coverage / tiling checks. Position-free + format-agnostic.
        sigv = _fixed_signature(obj)
        if sigv is not None:
            entry["fixed_mask"], entry["fixed_value"] = sigv
        by_class.setdefault(sem, []).append(entry)
        by_mnem[name] = {"class": sem, **entry}

    # the assembler-mnemonic -> class map (the model's own IsaSpec), best-effort — lets an example kernel
    # written in assembler syntax be mapped back to semantic classes.
    asm: dict[str, str] = {}
    try:
        from npu_model.isa import IsaSpec  # type: ignore
        for mn, cls in getattr(IsaSpec, "operations", {}).items():
            asm[str(mn)] = getattr(cls, "__name__", str(cls))
    except Exception:  # noqa: BLE001 — IsaSpec is model-specific; taxonomy still valid without it
        pass

    with open(a.out, "w") as f:
        json.dump({"by_class": by_class, "by_mnemonic": by_mnem, "asm_mnemonics": asm}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
