#!/usr/bin/env python3
"""Model-venv helper: DERIVE a self-hosted target's instruction taxonomy from its shipped ISA definition
(the repo's ISA doc), so the corpus/trace expectations are DISCOVERED, never hardcoded. Runs INSIDE the
target model's own venv (the ISA definition imports the model package, e.g. ``npu_model``), the same
cross-venv pattern as :mod:`npu_emit`.

Given the path to the target's ISA-definition module (each instruction is a class carrying ``opcode`` +
``funct*`` fields and a semantic base pattern, e.g. ``MXUMatMul`` / ``TensorBaseOffset`` / ``MXUWeightPush``
/ ``DMARegUnary``), it emits a JSON taxonomy:

    {"by_class": {"<semantic_pattern>": [{"mnemonic","opcode","role","funct3","funct7","funct2"}, ...], ...},
     "by_mnemonic": {"<MNEMONIC>": {"class","role","opcode",...}, ...},
     "asm_mnemonics": {"<assembler-mnemonic>": "<class-name>", ...}}   # from the model's IsaSpec, if any

``role`` is a SEMANTIC role derived structurally from the pattern's own typed operands (memory / matmul /
weight_load / acc_readout[_scaled] / acc_seed / tensor_compute_unary / tensor_compute_binary / scalar) —
so merlin's kernel structural checks select classes by role, never by a hardcoded pattern name.

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


# Generic hardware register-file categories a systolic/tensor accelerator exposes. We recognise the ISA
# def's OWN declared register semantics (each operand type carries a ``reg_name``, e.g. "accumulator",
# "weight buffer", "matrix register", "scalar register", "exponent register") by the standard concept
# word in that name — NOT by a target-specific class name. A target whose ISA names its PE-array result
# register "accumulator" and its stationary-weight store "weight" classifies correctly; the words are
# universal systolic-array vocabulary, and merlin core never sees them (it consumes the derived role).
_REG_CONCEPTS = (("accumulat", "accumulator"), ("weight", "weight"), ("exponent", "exponent"),
                 ("matrix", "tensor"), ("tensor", "tensor"), ("scalar", "scalar"))


def _reg_concept(reg_name: str | None) -> str | None:
    rn = (reg_name or "").lower()
    return next((concept for token, concept in _REG_CONCEPTS if token in rn), None)


def _operand_kinds(sem_cls) -> list[str]:
    """Ordered operand categories of a semantic pattern, from its OWN typed operand annotations (dest
    first). A register operand contributes its ISA-declared concept (accumulator/weight/tensor/scalar/
    exponent); an immediate contributes ``imm``. Everything is read off the ISA def's types — position
    and format agnostic — so the role below falls out of the datapath the operands describe."""
    modname = getattr(sem_cls, "__module__", "") or ""
    gmod = sys.modules.get(modname)
    kinds: list[str] = []
    for _attr, ann in (getattr(sem_cls, "__annotations__", {}) or {}).items():
        t = getattr(gmod, ann, None) if isinstance(ann, str) else ann
        if t is None or not isinstance(t, type):
            continue
        concept = _reg_concept(getattr(t, "reg_name", None))
        if concept:
            kinds.append(concept)
        elif issubclass(t, int):                     # a bounded immediate (offset / literal)
            kinds.append("imm")
    return kinds


def _role_for_pattern(sem_cls) -> str:
    """Derive a semantic ROLE for an instruction pattern from the datapath its operands describe — so the
    kernel structural checks (tile-count over the compute op, field-sanity over the memory op, canonical
    load->weight->matmul->readout order) select classes by ROLE, never by a hardcoded pattern name.

    Rules, all structural: a base scalar register + immediate offset carrying a tensor operand is a
    tensor MEMORY access; an op writing the accumulator from tensor+weight sources is the systolic
    MATMUL; writing the weight store is a WEIGHT_LOAD; reading the accumulator back to a tensor register
    is an ACC_READOUT (scaled when it also consumes an exponent register — the fp8 path); tensor->tensor
    is a TENSOR_COMPUTE (the vector unary/binary epilogue). Anything else is scalar/control."""
    kinds = _operand_kinds(sem_cls)
    if not kinds:
        return "scalar"
    dest, srcs = kinds[0], kinds[1:]
    has_tensor = any(k == "tensor" for k in kinds)
    if "scalar" in kinds and "imm" in kinds and has_tensor:
        return "memory"                               # tensor base+offset load/store (DRAM address)
    if dest == "accumulator" and "weight" in srcs:
        return "matmul"                               # systolic multiply into the accumulator
    if dest == "weight":
        return "weight_load"                          # push stationary weights
    if dest == "accumulator":
        return "acc_seed"                             # push a seed/bias into the accumulator
    if "accumulator" in srcs and dest == "tensor":
        return "acc_readout_scaled" if "exponent" in kinds else "acc_readout"
    if dest == "tensor" and srcs and all(k == "tensor" for k in srcs):
        # tensor->tensor: one source == unary (a relu-style epilogue), two+ == binary
        return "tensor_compute_unary" if len(srcs) == 1 else "tensor_compute_binary"
    return "scalar"


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
        sem_cls = next((b for b in obj.__mro__ if getattr(b, "__module__", "") == patmod), None)
        if sem_cls is None:
            continue
        sem = sem_cls.__name__
        entry = {"mnemonic": name, "opcode": op, "role": _role_for_pattern(sem_cls),
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
