#!/usr/bin/env python3
"""Model-venv helper: DERIVE a self-hosted target's instruction taxonomy from its shipped ISA definition
(the repo's ISA doc), so the corpus/trace expectations are DISCOVERED, never hardcoded. Runs INSIDE the
target model's own venv (the ISA definition imports the target's own model package), the same
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


def _base_word(cls) -> int | None:
    """The op encoded with all operands zero — the fixed opcode/funct bits. None if not encodable."""
    try:
        return int(object.__new__(cls).to_bytecode()) & 0xFFFFFFFF
    except Exception:  # noqa: BLE001 — not encodable without construction; skip
        return None


def _operand_fields(cls, base: int) -> dict[str, list[int | None]]:
    """Per-operand-bit → word-bit map for every operand attribute the format actually uses, derived from
    the ISA def's OWN encoder — no field-position assumptions, works for any instruction format (contiguous,
    shifted, or permuted fields alike). For each candidate attr we PER-BIT probe: set operand bit ``i`` only
    and XOR the encoded word with the all-zero base; the single word bit that moves is where operand bit
    ``i`` lands (``None`` if that operand bit is dropped, ``-1`` if it moves more than one word bit — a
    non-linear field the encoder must refuse rather than mis-pack). Returns ``{attr: bits}`` only for attrs
    that move at least one bit (i.e. are used by this format). This is the substrate the merlin-side
    assembler/disassembler pack/unpack against — the model's encoder stays the source of truth."""
    fields: dict[str, list[int | None]] = {}
    for attr in _OPERAND_ATTRS:
        # cheap use-check first: a wide all-ones pattern; if nothing moves, the format ignores this attr.
        probe = object.__new__(cls)
        try:
            setattr(probe, attr, 0x7FFFFFFF)
            if ((int(probe.to_bytecode()) & 0xFFFFFFFF) ^ base) == 0:
                continue
        except Exception:  # noqa: BLE001 — attr not settable on this format
            continue
        bits: list[int | None] = []
        for i in range(32):
            inst = object.__new__(cls)
            try:
                setattr(inst, attr, 1 << i)
                moved = (int(inst.to_bytecode()) & 0xFFFFFFFF) ^ base
            except Exception:  # noqa: BLE001 — value out of this field's range
                bits.append(None)
                continue
            if moved == 0:
                bits.append(None)
            elif (moved & (moved - 1)) == 0:          # exactly one word bit moved -> linear placement
                bits.append(moved.bit_length() - 1)
            else:
                bits.append(-1)                        # multi-bit move -> non-linear, flag for refusal
        while bits and bits[-1] is None:               # trim trailing unused operand bits
            bits.pop()
        if any(b is not None for b in bits):
            fields[attr] = bits
    return fields


def _fixed_signature_from_fields(base: int, fields: dict[str, list[int | None]]) -> tuple[int, int]:
    """FIXED opcode/funct signature (mask, value): the variable bits are exactly the operand-field bits
    (union over the per-bit map), the complement is fixed. A word decodes to this op iff ``word & mask ==
    value``. Derived from the accurate per-bit field map (supersedes the old 0x3F probe, which under-detected
    fields wider than 6 bits)."""
    variable = 0
    for bits in fields.values():
        for b in bits:
            if isinstance(b, int) and b >= 0:
                variable |= (1 << b)
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


def _discover_asm_operations(mod) -> dict:
    """The model's assembler-mnemonic -> op-class map, discovered WITHOUT any model-name literal. The ISA
    spec is an object exposing an ``operations`` dict ({mnemonic: op class}); it is normally imported by the
    ISA-definition module, so we scan (a) the loaded module's own namespace and (b) the namespaces of the
    modules the module imported (already in ``sys.modules``, restricted to the ISA module's own top-level
    package so we never touch unrelated packages). First object with a non-empty ``operations`` dict wins."""
    def _ops_of(ns) -> dict | None:
        for obj in ns:
            ops = getattr(obj, "operations", None)
            if isinstance(ops, dict) and ops:
                return ops
        return None

    hit = _ops_of(vars(mod).values())
    if hit is not None:
        return hit
    top = (getattr(mod, "__package__", "") or "").split(".")[0]
    if top:
        for name, m in list(sys.modules.items()):
            if m is not None and (name == top or name.startswith(top + ".")):
                hit = _ops_of(vars(m).values())
                if hit is not None:
                    return hit
    return {}


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
        # DECODE + ENCODE signature, both derived from the op's OWN encoder (position-free, format-agnostic):
        #  * fixed_mask/fixed_value — the fixed opcode/funct bits, so an emitted word classifies back to its
        #    semantic class (powers class-coverage / tiling / the disassembler);
        #  * fields — per operand-bit -> word-bit map, so the merlin-side assembler can PACK an operand into
        #    the exact bits the model's encoder uses (and the disassembler can unpack it) without any
        #    hand-authored field table. Non-linear operand bits are marked -1 so the assembler refuses them
        #    rather than emit a silently-wrong word.
        base = _base_word(obj)
        if base is not None:
            fields = _operand_fields(obj, base)
            entry["fixed_mask"], entry["fixed_value"] = _fixed_signature_from_fields(base, fields)
            if fields:
                entry["fields"] = fields
        by_class.setdefault(sem, []).append(entry)
        by_mnem[name] = {"class": sem, **entry}

    # the assembler-mnemonic -> class map (the model's own ISA spec), best-effort — lets an example kernel
    # written in assembler syntax be mapped back to semantic classes. DISCOVERED from the loaded ISA module
    # / its package (an object exposing ``operations``: {mnemonic -> op class}), never a model-name import,
    # so no target is hardcoded and any model that ships such a spec is picked up.
    asm: dict[str, str] = {}
    try:
        for mn, cls in _discover_asm_operations(mod).items():
            asm[str(mn)] = getattr(cls, "__name__", str(cls))
    except Exception:  # noqa: BLE001 — no spec found; taxonomy still valid without it
        pass

    with open(a.out, "w") as f:
        json.dump({"by_class": by_class, "by_mnemonic": by_mnem, "asm_mnemonics": asm}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
