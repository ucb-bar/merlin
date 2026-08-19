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


def _operand_names(cls) -> list[str]:
    """The op's operand field names, from the ISA def's OWN typed annotations across the class MRO (the
    pattern base declares them) plus the candidate superset. Used to build an encodable zero instance without
    running the op's ``__init__`` — position/format agnostic, no field-name assumption beyond the superset."""
    names: list[str] = []
    for base in getattr(cls, "__mro__", (cls,)):
        for attr in getattr(base, "__annotations__", {}) or {}:
            if not attr.startswith("_") and attr not in names:
                names.append(attr)
    for attr in _OPERAND_ATTRS:
        if attr not in names:
            names.append(attr)
    return names


def _zeroed_instance(cls):
    """An instance of ``cls`` with every operand field that has NO class default set to 0, so the op's own
    ``to_bytecode`` can encode it (RISC-V-style ops declare operands with no default and set them only in
    ``__init__``, which ``object.__new__`` skips). Class-defaulted attributes are LEFT ALONE — that preserves
    a fixed discriminator carried as a default (e.g. an ebreak whose imm default distinguishes it from ecall);
    zeroing it would corrupt the op's identity. Returns the raw instance (may still be non-encodable, caller
    guards)."""
    inst = object.__new__(cls)
    for attr in _operand_names(cls):
        if not hasattr(inst, attr):                    # missing (no class default) -> fill with 0
            try:
                setattr(inst, attr, 0)
            except Exception:  # noqa: BLE001 — not settable on this format; leave it
                pass
    return inst


def _base_word(cls) -> int | None:
    """The op encoded with all FREE operands zero (class-default discriminators kept) — the fixed opcode/funct
    bits. None if the op cannot be encoded even so."""
    try:
        return int(_zeroed_instance(cls).to_bytecode()) & 0xFFFFFFFF
    except Exception:  # noqa: BLE001 — not encodable; skip (no signature, honestly)
        return None


def _operand_fields(cls, base: int) -> dict[str, list[int | None]]:
    """Per-operand-bit → word-bit map for every operand attribute the format actually uses, derived from
    the ISA def's OWN encoder — no field-position assumptions, works for any instruction format (contiguous,
    shifted, or permuted fields alike). For each candidate attr we PER-BIT probe: set operand bit ``i`` only
    and XOR the encoded word with the all-zero base; the single word bit that moves is where operand bit
    ``i`` lands (``None`` if that operand bit is dropped, ``-1`` if it moves more than one word bit — a
    non-linear field the encoder must refuse rather than mis-pack). Returns ``{attr: bits}`` only for attrs
    that move at least one bit (i.e. are used by this format). This is the substrate the merlin-side
    assembler/disassembler pack/unpack against — the model's encoder stays the source of truth.

    Also returns ``touched`` — the union of EVERY word bit that ANY operand bit moves, including aliased bits
    (one operand bit that lands in more than one word bit, e.g. an encoder that mirrors an immediate into a
    second slot). The per-bit ``fields`` map keeps only LINEAR placements (``-1`` for aliased, so the packing
    assembler refuses them rather than mis-encode), but the DECODE mask must treat every touched bit as
    variable — otherwise a legal instruction whose aliased bits are non-zero is falsely rejected."""
    fields: dict[str, list[int | None]] = {}
    touched = 0
    for attr in _OPERAND_ATTRS:
        # cheap use-check first: a wide all-ones pattern; if nothing moves, the format ignores this attr.
        # Probe on a ZEROED instance (all other free operands set) so the op is encodable even when it needs
        # several operands — a bare instance would raise on the unset ones and hide every real field.
        probe = _zeroed_instance(cls)
        try:
            setattr(probe, attr, 0x7FFFFFFF)
            if ((int(probe.to_bytecode()) & 0xFFFFFFFF) ^ base) == 0:
                continue
        except Exception:  # noqa: BLE001 — attr not settable on this format
            continue
        bits: list[int | None] = []
        for i in range(32):
            inst = _zeroed_instance(cls)
            try:
                setattr(inst, attr, 1 << i)
                moved = (int(inst.to_bytecode()) & 0xFFFFFFFF) ^ base
            except Exception:  # noqa: BLE001 — value out of this field's range
                bits.append(None)
                continue
            touched |= moved                           # every word bit this operand can vary (decode mask)
            if moved == 0:
                bits.append(None)
            elif (moved & (moved - 1)) == 0:          # exactly one word bit moved -> linear placement
                bits.append(moved.bit_length() - 1)
            else:
                bits.append(-1)                        # multi-bit move -> aliased, refuse in the packer
        while bits and bits[-1] is None:               # trim trailing unused operand bits
            bits.pop()
        if any(b is not None for b in bits):
            fields[attr] = bits
    return fields, touched


def _declared_operands(cls) -> set:
    """Operand attributes the format classes in ``cls``'s MRO declare in their OWN annotations."""
    out = set()
    for base in getattr(cls, "__mro__", (cls,)):
        for attr in getattr(base, "__annotations__", {}) or {}:
            if attr in _OPERAND_ATTRS:
                out.add(attr)
    return out


def _canonical_placements(entries: list[dict]) -> dict:
    """The ONE bit placement each operand attribute uses across the whole ISA, where it is unambiguous.

    Instruction formats in a family place a given operand in the same word bits (a decoder reads one field
    for it). So when every format that encodes ``rd`` linearly agrees on its bits, that placement IS the
    ISA's placement, derived — not assumed. An attribute whose formats DISAGREE yields nothing, and the
    caller then repairs nothing (fail closed rather than guess)."""
    seen: dict = {}
    for e in entries:
        for attr, bits in (e.get("fields") or {}).items():
            if any(b is None or b == -1 for b in bits):
                continue                                # only fully-linear placements are evidence
            seen.setdefault(attr, set()).add(tuple(bits))
    return {a: list(v.pop()) for a, v in seen.items() if len(v) == 1}


def _repair_dropped_operands(entries: list[dict], classes: dict) -> list[dict]:
    """Restore an operand its shipped encoder DECLARES but never packs, and report what was restored.

    A shipped encoder can carry a field-packing bug: the atlas ``IType.to_bytecode`` assigns ``rd`` and
    then immediately overwrites it with ``imm``, so ``rd`` reaches no word bit while ``imm``'s low bits
    reach two. The functional simulator hides this (it reads the decoded object, not the word); the RTL
    decoder does not, and merlin's ``--fix-itype-rd`` shim already corrects the same bug on the program
    path. Deriving the encoder from the unrepaired model yields an assembler that CANNOT write the
    instruction that sets up an address register — most of a kernel's scalar prologue — and that reports
    the immediate as unpackable, because the stolen bits look like an aliased field.

    The repair wraps the op's own encoder (place the declared operand at the placement the REST of this
    ISA uses for it, after clearing those bits) and then RE-PROBES through it, so the resulting field map
    is self-consistent: the operand appears, and the operand that was stealing its bits goes back to being
    linear. Evidence-driven and self-disabling — an operand is restored only when its format declares it,
    the probe found it packed nowhere, and every other format in the ISA agrees on where it lives. The
    moment the shipped encoder is fixed the probe finds the field and nothing is repaired. Each repair is
    recorded in ``entry['repaired']`` so a consumer can surface it rather than silently trust a patch."""
    canon = _canonical_placements(entries)
    for e in entries:
        cls = classes.get(e["mnemonic"])
        if cls is None:
            continue
        dropped = sorted(a for a in _declared_operands(cls) - set(e.get("fields") or {})
                         if canon.get(a))
        if not dropped:
            continue
        orig = cls.to_bytecode
        placements = {a: canon[a] for a in dropped}

        def _repaired(self, _orig=orig, _pl=placements):
            word = int(_orig(self)) & 0xFFFFFFFF
            for attr, bits in _pl.items():
                for b in bits:
                    word &= ~(1 << b)                  # the stolen bits belong to this operand
                val = int(getattr(self, attr, 0) or 0)
                for i, b in enumerate(bits):
                    if val >> i & 1:
                        word |= 1 << b
            return word

        cls.to_bytecode = _repaired
        try:
            base = _base_word(cls)
            if base is None:
                continue
            fields, touched = _operand_fields(cls, base)
            e["fixed_mask"], e["fixed_value"] = _fixed_signature_from_touched(base, touched)
            if fields:
                e["fields"] = fields
            e["repaired"] = dropped
        finally:
            cls.to_bytecode = orig                     # never leave the shared ISA module mutated
    return entries


def _fixed_signature_from_touched(base: int, touched: int) -> tuple[int, int]:
    """FIXED opcode/funct signature (mask, value): the variable bits are EVERY word bit any operand can move
    (``touched``, aliased bits included), the complement is fixed. A word decodes to this op iff
    ``word & mask == value``. Using the full touched set (not just linear field bits) is what keeps the decode
    from falsely rejecting a legal instruction whose encoder mirrors an operand into extra bits."""
    mask = (~touched) & 0xFFFFFFFF
    return mask, base & mask


def _fixed_signature_from_fields(base: int, fields: dict[str, list[int | None]]) -> tuple[int, int]:
    """Back-compat: derive the fixed signature from the linear-only per-bit field map (used where the aliased
    touched-set is unavailable). Prefer :func:`_fixed_signature_from_touched`."""
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


# The instruction the machine model treats as a PROGRAM TERMINATOR is derived BEHAVIORALLY, from the op's
# own semantic effect — never by a mnemonic literal ("halt"/"ecall") or an opcode. We run each op's semantic
# method against a recording stand-in for the machine state and keep the ops whose ENTIRE effect is to raise
# one boolean state flag to True (and touch nothing else): that is exactly what a terminator does (assert the
# machine's "finished" signal), and it structurally excludes a barrier/fence (whose effect is empty) and every
# compute/memory op (which reads unset operands and/or calls state read/write methods). The specific flag name
# is discovered from the model (recorded), not assumed. Universal machine-model vocabulary, no target fact.
_SEM_METHODS = ("exec", "execute", "semantic", "apply", "effect", "step", "__call__")


class _StateRec:
    """A recording stand-in for the machine state passed to an op's semantic method: it captures every
    attribute SET and every method CALL the op performs, so a terminator (whose only effect is raising a
    boolean flag) is distinguishable from a barrier (no effect) or a datapath op (calls read/write methods /
    indexes registers). Attribute reads return a callable+indexable stub so a pure-flag op does not crash;
    a datapath op that does arithmetic on an unset operand still raises (caught upstream)."""

    def __init__(self):
        object.__setattr__(self, "sets", {})
        object.__setattr__(self, "calls", [])

    def __setattr__(self, k, v):
        self.sets[k] = v

    def __getattr__(self, k):
        rec = self

        class _Stub:
            def __call__(self, *a, **kw):
                rec.calls.append(k)
                return _Stub()

            def __getitem__(self, i):
                rec.calls.append(k)
                return _Stub()

        return _Stub()


def _sem_method(cls):
    """The op's semantic-effect method (the one the interpreter runs to mutate machine state), by name from a
    small candidate set — discovered, not assumed. Returns the bound-callable name or None."""
    for name in _SEM_METHODS:
        fn = getattr(cls, name, None)
        if callable(fn) and name not in ("to_bytecode",):
            # only accept a method actually defined on an op (own or a pattern base), not object.__call__
            if name != "__call__" or "__call__" in getattr(cls, "__dict__", {}):
                return name
    return None


def _terminator_flag(cls) -> str | None:
    """Behaviorally probe one op: run its semantic method against a recording state and return the boolean
    flag it raises IFF that is its entire effect (one attribute set to True, no method calls, no other sets) —
    the derived signature of a program terminator. None otherwise. Never constructs the real (heavy) machine
    state and never names the flag in advance."""
    name = _sem_method(cls)
    if name is None:
        return None
    rec = _StateRec()
    try:
        getattr(cls, name)(object.__new__(cls), rec)
    except Exception:  # noqa: BLE001 — a datapath op reads unset operands and raises; not a terminator
        return None
    if rec.calls:
        return None                                   # touched a register/memory method -> not a pure halt
    true_bools = [k for k, v in rec.sets.items() if isinstance(v, bool) and v is True]
    if len(rec.sets) == 1 and len(true_bools) == 1:   # sole effect: raise one boolean flag
        return true_bools[0]
    return None


def _halt_ops(mod, by_mnem: dict) -> tuple[list[str], str | None]:
    """The DERIVED terminator op set: op names whose semantic effect is solely to assert the machine's finish
    flag, plus the (consensus) flag name for provenance. Empty when no op has that effect (the linter then
    reports an honest INFO rather than a false 'no halt'). Behavioral — no mnemonic/opcode literal."""
    flag_votes: dict[str, list[str]] = {}
    for name in by_mnem:
        cls = getattr(mod, name, None)
        if cls is None:
            continue
        flag = _terminator_flag(cls)
        if flag:
            flag_votes.setdefault(flag, []).append(name)
    if not flag_votes:
        return [], None
    flag = max(flag_votes, key=lambda f: len(flag_votes[f]))   # the flag the most terminator ops raise
    return sorted(flag_votes[flag]), flag


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


# Attribute names an ISA op class may use to declare its own ASSEMBLER syntax (the token an example kernel
# writes, e.g. ``vmatmul.mxu0``). ``mnemonic`` is the documented convention (``isa_patterns.py``: "mnemonic
# (str): The name of the instruction"); the others are best-effort synonyms so any spec that names its
# syntax is picked up. Structural — no target/mnemonic literal.
_ASM_MNEMONIC_ATTRS = ("mnemonic", "asm", "asm_name", "asm_string", "syntax")


def _asm_mnemonic_of(cls) -> str | None:
    """The op class's OWN declared assembler mnemonic (its ``mnemonic``/``asm``/… ClassVar), or None when it
    declares none. This lets an example kernel written in the target's real assembler syntax map back to the
    semantic class even when no container ``operations`` dict is reachable — derived from the class itself,
    fail-closed (``NotImplemented``/non-str/empty -> None)."""
    for attr in _ASM_MNEMONIC_ATTRS:
        v = getattr(cls, attr, None)
        if isinstance(v, str) and v and v is not NotImplemented:
            return v
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--isa-module", required=True, help="path to the target's ISA-definition .py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mod = _load_module(a.isa_module)
    patmod = _pattern_module(mod)
    by_class: dict[str, list] = {}
    by_mnem: dict[str, dict] = {}
    asm_from_classes: dict[str, str] = {}          # asm-syntax token -> class name, from each op's own ClassVar
    op_classes: dict = {}                          # mnemonic -> op class, for the dropped-operand repair pass
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
            fields, touched = _operand_fields(obj, base)
            entry["fixed_mask"], entry["fixed_value"] = _fixed_signature_from_touched(base, touched)
            if fields:
                entry["fields"] = fields
        by_class.setdefault(sem, []).append(entry)
        by_mnem[name] = {"class": sem, **entry}
        op_classes[name] = obj
        am = _asm_mnemonic_of(obj)                   # the op's OWN assembler syntax (e.g. vmatmul.mxu0)
        if am:
            asm_from_classes[am] = name

    # Restore any operand the shipped encoder declares but never packs (a field-packing bug in the model
    # would otherwise reach merlin as 'this instruction has no such operand'). Runs over ALL entries at
    # once because the repair's evidence is cross-format: where the rest of the ISA puts that operand.
    # ``by_mnem`` is rebuilt afterwards since its entries are copies made before this pass.
    _repair_dropped_operands([e for ents in by_class.values() for e in ents], op_classes)
    for sem, ents in by_class.items():
        for e in ents:
            by_mnem[e["mnemonic"]] = {"class": sem, **e}

    # the assembler-mnemonic -> class map (the model's own ISA spec), best-effort — lets an example kernel
    # written in assembler syntax be mapped back to semantic classes. DISCOVERED from the loaded ISA module
    # / its package (an object exposing ``operations``: {mnemonic -> op class}), never a model-name import,
    # so no target is hardcoded and any model that ships such a spec is picked up.
    # Start from each op class's own declared assembler mnemonic (always available when the spec follows the
    # ``mnemonic`` ClassVar convention), then let a container ``operations`` dict — if reachable — override /
    # extend it. This makes the map work even when the ``IsaSpec`` container object isn't in a scanned
    # namespace (the standalone-``isa_definition.py`` load), which previously left ``asm_mnemonics`` empty.
    asm: dict[str, str] = dict(asm_from_classes)
    try:
        for mn, cls in _discover_asm_operations(mod).items():
            asm[str(mn)] = getattr(cls, "__name__", str(cls))
    except Exception:  # noqa: BLE001 — no spec found; taxonomy still valid without it
        pass

    # the DERIVED program-terminator set (behavioral: ops whose sole effect is asserting the machine finish
    # flag) + their decode signatures, so the static linter can detect "the kernel reaches a terminator"
    # without a mnemonic/opcode literal. Terminators sharing a coarse semantic class (e.g. a fence and a halt
    # both "nullary") are separated here by their own fixed opcode/funct signature, not by the class name.
    halt_mnemonics, halt_flag = _halt_ops(mod, by_mnem)
    halt_signatures = []
    for hm in halt_mnemonics:
        ent = by_mnem.get(hm, {})
        m, v = ent.get("fixed_mask"), ent.get("fixed_value")
        if isinstance(m, int) and isinstance(v, int):
            halt_signatures.append([m, v])

    with open(a.out, "w") as f:
        json.dump({"by_class": by_class, "by_mnemonic": by_mnem, "asm_mnemonics": asm,
                   "halt_mnemonics": halt_mnemonics, "halt_flag": halt_flag,
                   "halt_signatures": halt_signatures}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
