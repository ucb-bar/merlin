"""Decode a Gemmini RoCC instruction trace from a package's emitted ``lowered.llvm.mlir``.

This is a **runner-owned** decode: the backend-under-test emits custom-3 ``.insn r`` RoCC
inline-asm; this module reads *what it actually emitted* and reconstructs a structured trace.
Keeping the trace a measurement of the package's output (rather than a package-provided artifact)
makes it a fair, parity-clean observation that applies identically to the baseline and the
Merlin-assisted package.

It is fail-closed: any inline-asm instruction form it does not understand is recorded as class
``UNKNOWN`` (never silently dropped), so ``trace_check`` can reject it.

The bit layouts here mirror the encoders in the package's ``GemminiToLLVM.cpp`` (and the certified
native path ``runtime/backends/gemmini_codegen_mlir.py``):

    DIM=16; pack(addr)=(DIM<<48)|(DIM<<32)|(addr & 0xFFFFFFFF)
    CONFIG subtype = rs1 & 0x3   (EX=0, LD=1, ST=2)
    CONFIG_ST rs1 = (acc_act<<2)|2 ; rs2 = (f32_bits(scale)<<32) | (out_row_stride_bytes)
    readout addr bits: C_ACC=0xA0000000 (full i32), ACC_I8=0x80000000 (scaled i8),
                       ACC_ACCUM=0x40000000 (accumulate-onto)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

# --- encoding constants: DERIVED from the SINGLE source — the readout bits + RTL-code->class map +
# config subtype from the manifest's encoding block, and the mesh DIM from the CIRCT-extracted facts
# (arrays[mesh]); not hand-copied. Byte-parity with the former literals is pinned by test_encoding_manifest.
# This retires one of the three triplicated copies (the decoder's). GARBAGE/MASK32 are universal.
GARBAGE = 0xFFFFFFFF
MASK32 = 0xFFFFFFFF


def _load_isa(target: str) -> dict:
    """Derive the RoCC ISA constants for ``target`` from its RTL facts + capability manifest. No target
    is baked in — the caller passes the target it is grading; the decoder holds no default."""
    from .target_experiment import load_capability_manifest
    from .rtl.facts import load_facts
    m = load_capability_manifest(target)
    enc, rb = m.encoding, m.encoding["readout_bits"]
    facts = load_facts(target)["facts"]
    # DIM (systolic mesh dimension) is a CIRCT-extracted FACT (arrays[mesh]), not a manifest field —
    # same source the codegen emitter reads, so the decoder's DIM cannot drift from the encoder's.
    mesh = next((a for a in facts.get("arrays", []) if a.get("name") == "mesh"), {})
    dim = mesh.get("rows")   # UNKNOWN (None) if the target declares no mesh — no baked gemmini DIM=16
    # The custom major opcode is a per-target FACT read from facts (funct_decode_table.custom_opcode),
    # NOT a baked literal: it is the RISC-V-standard encoding of the custom SLOT the target's RoCC is
    # wired to (SoC OpcodeSet), resolved by circt_introspect from the target's reviewed
    # encoding.rocc_custom_slot. It may be None (UNKNOWN) for a target that declares no slot — the
    # decoder then does not filter by major opcode (see _parse_insn). funct3 is the RoCC xd/xs1/xs2
    # register-usage field — it VARIES per instruction (e.g. a result-returning op sets xd=1), so it is
    # NOT an identity constraint; instruction identity is func7 (-> FUNCT_CLASS).
    fdt = next((i for i in facts.get("interfaces", []) if i.get("name") == "funct_decode_table"), {})
    custom_opcode = fdt.get("custom_opcode")
    return {"DIM": dim, "F1": rb["f1"], "C_ACC": rb["c_acc"], "ACC_I8": rb["acc_i8"],
            "ACC_ACCUM": rb["acc_accum"], "FULL_C_BIT": rb["full_c_bit"],
            "CUSTOM_OPCODE": custom_opcode, "FUNCT3": fdt.get("funct3"),
            "FUNCT_CLASS": dict(enc["semantic_class"]), "CONFIG_SUBTYPE": dict(enc["config_subtype"])}


# Per-target ISA constants are resolved LAZILY and cached — nothing target-specific loads at import.
# (GARBAGE / MASK32 above are universal 32-bit masks, not target facts.)
_ISA_CACHE: dict[str, dict] = {}


def isa_constants(target: str) -> dict:
    """The derived RoCC ISA constants for ``target`` (DIM, readout bits, CUSTOM_OPCODE, FUNCT_CLASS,
    CONFIG_SUBTYPE, …), cached. This is the single source the codegen/header generators check against."""
    isa = _ISA_CACHE.get(target)
    if isa is None:
        isa = _ISA_CACHE[target] = _load_isa(target)
    return isa


def funct_class_for(target: str) -> dict:
    """``func7 -> instruction-class`` map for ``target`` (best-effort cross-check consumers)."""
    return isa_constants(target)["FUNCT_CLASS"]

# --- structural IR decode (parse the IR, do not string-match text) ----------------------------
# This decoder is a fair MEASUREMENT of whatever the backend emitted, so it must SEE every legal
# SPELLING of an instruction and fail closed (UNKNOWN) on anything it cannot fully decode — never
# silently drop one. Text line-matching repeatedly broke that contract by being too narrow (numeric-
# only SSA ids; "r,r"-only constraints; and — the case this replaces — reading operands only from the
# PRETTY ``llvm.inline_asm … "asm","cons" %a, %b`` spelling, so a conformant backend that emitted the
# equally-legal GENERIC ``"llvm.inline_asm"(%a, %b) {asm_string=…}`` form had its operands go unseen
# and every CONFIG mis-classified UNKNOWN). The two spellings are the SAME MLIR op and lower
# identically, so we stop reading text and walk the PARSED xDSL IR: operands come from the op's SSA
# operand list, their values from the defining ops — spelling-agnostic by construction, per the repo's
# "parse structurally, use the xDSL/MLIR IR" rule. Text that does not parse as a module fails closed.


def _parse_module(text: str):
    """Parse ``text`` as an MLIR module with xDSL (builtin+func+llvm loaded; unregistered ops allowed so
    any unmodeled op still round-trips in generic form). Returns the ModuleOp, or None if the text is not
    a parseable module — the caller then fails closed (a visible UNKNOWN), never guessing at the input."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.dialects.llvm import LLVM
    from xdsl.parser import Parser
    ctx = Context(allow_unregistered=True)
    for d in (Builtin, Func, LLVM):
        ctx.load_dialect(d)
    try:
        return Parser(ctx, text).parse_module()
    except Exception:  # noqa: BLE001 — any parse fault is a fail-closed "undecodable", surfaced upstream
        return None


def _const_value(op) -> int | None:
    """The integer value of an ``llvm.mlir.constant`` defining op, else None."""
    v = getattr(op, "properties", {}).get("value")
    data = getattr(getattr(v, "value", None), "data", None)
    try:
        return int(data) if data is not None else None
    except (TypeError, ValueError):
        return None


def _resolve(val, _depth: int = 0) -> _Val:
    """Resolve an SSA value to a ``_Val`` by walking its defining op STRUCTURALLY: a block/function
    argument (``argbase``), an ``llvm.mlir.constant`` (``const``), a transparent ``llvm.ptrtoint`` (pass
    through to the pointer it converts), or an ``llvm.add`` of base+constant (``argbase`` with an offset)
    / two constants (folded ``const``). Anything else fails closed to ``unknown`` — never a guessed
    value. This replaces the old SSA line-table plumbing with def-use walking on the parsed IR. A depth
    bound keeps a pathological/adversarial def chain from ever hanging the grader (fail-closed to
    ``unknown`` past the bound — valid scalar-address chains are shallow)."""
    from xdsl.ir import BlockArgument
    if _depth > 64:
        return _Val("unknown")
    if isinstance(val, BlockArgument):
        return _Val("argbase", arg_index=val.index, offset=0)
    op = val.owner
    name = getattr(op, "name", "")
    if name == "llvm.mlir.constant":
        iv = _const_value(op)
        return _Val("const", value=iv) if iv is not None else _Val("unknown")
    if name == "llvm.ptrtoint" and len(op.operands) >= 1:
        return _resolve(op.operands[0], _depth + 1)
    if name == "llvm.add" and len(op.operands) == 2:
        a, b = _resolve(op.operands[0], _depth + 1), _resolve(op.operands[1], _depth + 1)
        base = a if a.kind == "argbase" else (b if b.kind == "argbase" else None)
        const = a if a.kind == "const" else (b if b.kind == "const" else None)
        if base is not None and const is not None:
            return _Val("argbase", arg_index=base.arg_index,
                        offset=(base.offset or 0) + (const.value or 0))
        if a.kind == "const" and b.kind == "const":
            return _Val("const", value=(a.value or 0) + (b.value or 0))
        return _Val("unknown")
    return _Val("unknown")


def _asm_template(op) -> str:
    """The instruction TEMPLATE an ``llvm.inline_asm`` op carries (``.insn …`` / ``fence``), read from its
    ``asm_string`` property — the same place in the parsed op regardless of surface spelling."""
    s = getattr(op, "properties", {}).get("asm_string")
    return (getattr(s, "data", "") or "").strip()


def _parse_insn(template: str, custom_opcode: int | None) -> tuple[int, bool] | None:
    """Parse an R-type ``.insn r <opcode>, <func3>, <func7>, <rd>, <rs1>, <rs2>`` template into
    (func7, rd_is_x0) iff ``<opcode>`` is ``custom_opcode`` (the target's RTL-derived custom opcode);
    else None (caller records UNKNOWN, fail-closed). The opcode is COMPARED (as an int) to the derived
    fact — not string-matched to a literal — so ``0x7b``/``0x7B``/``123`` all work and a different
    target's opcode is respected. ``func3`` (RoCC xd/xs1/xs2) is NOT constrained: it varies per
    instruction; identity is func7. Both the 2-operand (rd=x0) and 3-operand (rd=a GPR) forms are
    accepted; L2/L3 RTL is the correctness gate."""
    fields = [f.strip() for f in template.split(",")]
    if len(fields) < 4:
        return None
    head = fields[0].split()  # e.g. [".insn", "r", "0x7b"]
    if len(head) != 3 or head[0] != ".insn" or head[1] != "r":
        return None
    try:
        opcode = int(head[2], 0)   # accepts 0x-hex or decimal
        func7 = int(fields[2], 0)  # instruction identity (fields[1] is func3, deliberately ignored)
    except ValueError:
        return None
    if custom_opcode is not None and opcode != custom_opcode:
        return None
    return func7, (fields[3] == "x0")


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


@dataclass
class _Val:
    kind: str                 # "const" | "argbase" | "unknown"
    value: int | None = None  # for const
    arg_index: int | None = None
    offset: int | None = None


def _operand(v: _Val | None) -> dict:
    if v is None:
        return {"raw": None, "kind": "unknown", "arg_index": None, "offset": None}
    if v.kind == "const":
        return {"raw": v.value, "kind": "const", "arg_index": None, "offset": None}
    if v.kind == "argbase":
        return {"raw": None, "kind": "argbase", "arg_index": v.arg_index, "offset": v.offset}
    return {"raw": None, "kind": "unknown", "arg_index": None, "offset": None}


def _pack_fields(v: int) -> dict:
    return {"rows": (v >> 48) & 0xFFFF, "cols": (v >> 32) & 0xFFFF, "addr": v & MASK32}


@dataclass
class Trace:
    source: str | None
    custom_opcode: int | None = None
    funct3: int | None = None
    instructions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        hist: dict[str, int] = {}
        for ins in self.instructions:
            hist[ins["class"]] = hist.get(ins["class"], 0) + 1
        # abi codes are reported as hex STRINGS ("0x7b"/"0x3") per the instruction_trace schema — format
        # the derived ints, don't emit raw ints (a schema ContractViolation) or a baked literal.
        def _hex(v: int | None) -> str | None:
            return f"{v:#x}" if v is not None else None
        return {
            "source": self.source,
            "abi": {"custom_opcode": _hex(self.custom_opcode), "funct3": _hex(self.funct3)},
            "instructions": self.instructions,
            "summary": {"class_histogram": hist},
        }


def _decode_one(funct: int, rs1: _Val | None, rs2: _Val | None, isa: dict) -> tuple[str, dict]:
    """Return (class, decoded-fields) for one .insn given resolved operands and the target ``isa``."""
    full_c_bit, acc_accum = isa["FULL_C_BIT"], isa["ACC_ACCUM"]
    base = isa["FUNCT_CLASS"].get(funct, "UNKNOWN")
    r1 = rs1.value if (rs1 and rs1.kind == "const") else None
    r2 = rs2.value if (rs2 and rs2.kind == "const") else None
    dec: dict = {}

    if base == "CONFIG":
        sub = isa["CONFIG_SUBTYPE"].get((r1 & 0x3) if r1 is not None else -1, "CONFIG_UNKNOWN")
        if sub == "CONFIG_LD":
            # The mvin scale float is packed in rs1[63:32] (same high-word layout CONFIG_ST uses for the
            # store/acc scale below). Expose it so a degenerate load scale (e.g. 0.0, which multiplies every
            # loaded element to zero) is visible in the trace instead of only manifesting as all-zeros on the
            # hardware oracle. Identity is 1.0; absent/unresolved rs1 -> None (fail-open, never a wrong value).
            ld_scale_bits = ((r1 >> 32) & MASK32) if r1 is not None else None
            dec = {"subtype": "LD", "stride": r2,
                   "scale": _f32_from_bits(ld_scale_bits) if ld_scale_bits is not None else None,
                   "scale_bits": ld_scale_bits}
        elif sub == "CONFIG_ST":
            acc_act = ((r1 >> 2) & 0x3) if r1 is not None else None
            scale_bits = ((r2 >> 32) & MASK32) if r2 is not None else None
            dec = {
                "subtype": "ST",
                "acc_act": acc_act,
                "relu": (acc_act == 1) if acc_act is not None else None,
                "acc_scale": _f32_from_bits(scale_bits) if scale_bits is not None else None,
                "acc_scale_bits": scale_bits,
                "out_stride_bytes": (r2 & MASK32) if r2 is not None else None,
            }
        else:
            dec = {"subtype": "EX"}
        return (sub if sub != "CONFIG_UNKNOWN" else "UNKNOWN"), dec

    if base in ("MVIN",):
        dec = {"dram": _operand(rs1)}
        if r2 is not None:
            dec.update(_pack_fields(r2))
            dec["spad_addr"] = r2 & MASK32
        return base, dec

    if base == "MVOUT":
        dec = {"dram": _operand(rs1)}
        if r2 is not None:
            dec.update(_pack_fields(r2))
            acc_addr = r2 & MASK32
            dec["acc_addr"] = acc_addr
            dec["readout"] = "i32" if (acc_addr & full_c_bit) else "i8"
        return base, dec

    if base == "PRELOAD":
        dec = {}
        if r1 is not None:
            dec["weight_spad"] = r1 & MASK32
        if r2 is not None:
            c_addr = r2 & MASK32
            dec["c_addr"] = c_addr
            dec["accumulate"] = bool(c_addr & acc_accum)
            dec["readout"] = "i32" if (c_addr & full_c_bit) else "i8"
        return base, dec

    if base in ("COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"):
        dec = {}
        if r1 is not None:
            dec["a_spad"] = r1 & MASK32
        if r2 is not None:
            dec["bd"] = r2 & MASK32
            dec["garbage"] = (r2 & MASK32) == GARBAGE
        return base, dec

    if base == "FLUSH":
        return base, {}

    return ("UNKNOWN" if base == "UNKNOWN" else base), dec


def decode_module(module, *, target: str, source: str | None = None) -> dict:
    """Decode a parsed MLIR module's RoCC instruction stream into a structured trace, using ``target``'s
    RTL-derived ISA facts. Walks the IR in program order; each ``llvm.inline_asm`` op is classified by its
    ``.insn`` template (func7 -> class) with operands resolved structurally from the SSA graph. Fail-
    closed: an inline-asm whose template is neither a custom-opcode ``.insn`` nor ``fence`` is recorded
    UNKNOWN, never dropped."""
    isa = isa_constants(target)
    trace = Trace(source=source, custom_opcode=isa["CUSTOM_OPCODE"], funct3=isa["FUNCT3"])
    idx = 0
    for op in module.walk():
        if getattr(op, "name", "") != "llvm.inline_asm":
            continue
        template = _asm_template(op)
        if template == "fence":
            trace.instructions.append({"index": idx, "class": "FENCE", "funct": None, "decoded": {}})
        elif template.startswith(".insn"):
            parsed = _parse_insn(template, isa["CUSTOM_OPCODE"])
            if parsed is not None:
                funct, _rd_is_x0 = parsed
                ops = list(op.operands)  # SSA source operands, in order (rs1, rs2) — spelling-agnostic
                rs1 = _resolve(ops[0]) if len(ops) >= 1 else None
                rs2 = _resolve(ops[1]) if len(ops) >= 2 else None
                klass, dec = _decode_one(funct, rs1, rs2, isa)
                trace.instructions.append({
                    "index": idx, "class": klass, "funct": funct,
                    "rs1": _operand(rs1), "rs2": _operand(rs2), "decoded": dec,
                })
            else:
                # a .insn on a non-custom opcode: fail-closed (record, do not drop).
                trace.instructions.append({"index": idx, "class": "UNKNOWN", "funct": None,
                                           "raw": template, "decoded": {}})
        else:
            # an inline-asm we do not recognize (not .insn, not fence): fail-closed.
            trace.instructions.append({"index": idx, "class": "UNKNOWN", "funct": None,
                                       "raw": template, "decoded": {}})
        idx += 1
    return trace.to_dict()


# --- tolerant text-scan fallback --------------------------------------------------------------
# We cannot anticipate every form a backend emits — a bare fragment, an alternative op spelling
# (``"llvm.intr.inlineasm"()`` / textual ``call … asm``), a not-yet-modeled dialect. So when the input
# does not parse as an MLIR module, we do NOT black it out: we scan it line by line, detect inline-asm by
# its instruction TEMPLATE (spelling-agnostic), record each one, resolve operands best-effort from the
# surrounding scalar SSA, and mark UNKNOWN (never drop) whatever cannot be classified. Structural decode
# is preferred (correct operands); this guarantees a present stream is always SEEN, never mis-measured as
# empty. Operands are read from the WHOLE asm line's non-quoted text (both the ``(%a,%b){…}`` and trailing
# ``… %a, %b`` operand positions) — the narrowness that once read only post-quote operands is gone.
_SSA_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$-")


def _read_ssa(s: str, i: int) -> tuple[str, int]:
    j = i + 1
    while j < len(s) and s[j] in _SSA_CHARS:
        j += 1
    return s[i + 1:j], j


def _ssa_names(s: str) -> list[str]:
    """All ``%<ident>`` names in ``s``, left to right."""
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "%":
            name, i = _read_ssa(s, i)
            out.append(name)
        else:
            i += 1
    return out


def _quote_split(line: str) -> tuple[list[str], str]:
    """(strings-inside-quotes, all-text-OUTSIDE-quotes). The asm ops here carry no escaped quotes, so a
    plain split is exact. The outside text holds the operand SSA refs in EITHER position; the inside holds
    the instruction template. Reading operands from the whole outside (not only post-last-quote) is what
    makes the scan spelling-agnostic."""
    parts = line.split('"')
    return parts[1::2], "".join(parts[0::2])


def _decode_by_text_scan(text: str, *, target: str, source: str | None = None) -> dict:
    """Fallback decode for input that is not a parseable module. Same output shape as
    :func:`decode_module`; see the section comment above for the tolerance contract."""
    isa = isa_constants(target)
    ssa: dict[str, _Val] = {}
    trace = Trace(source=source, custom_opcode=isa["CUSTOM_OPCODE"], funct3=isa["FUNCT3"])
    idx = 0
    for line in text.splitlines():
        inside, outside = _quote_split(line)
        template = next((s.strip() for s in inside
                         if s.strip().startswith(".insn") or s.strip() == "fence"), "")
        if template:
            if template == "fence":
                trace.instructions.append({"index": idx, "class": "FENCE", "funct": None, "decoded": {}})
            else:
                parsed = _parse_insn(template, isa["CUSTOM_OPCODE"])
                if parsed is not None:
                    funct, _rd_is_x0 = parsed
                    # operands = %refs in the non-quoted text, excluding an ``%res =`` result name
                    operand_text = outside.split("=", 1)[1] if (
                        "=" in outside and outside.split("=", 1)[0].strip().startswith("%")) else outside
                    names = _ssa_names(operand_text)
                    rs1 = ssa.get(names[0]) if len(names) >= 1 else None
                    rs2 = ssa.get(names[1]) if len(names) >= 2 else None
                    klass, dec = _decode_one(funct, rs1, rs2, isa)
                    trace.instructions.append({
                        "index": idx, "class": klass, "funct": funct,
                        "rs1": _operand(rs1), "rs2": _operand(rs2), "decoded": dec,
                    })
                else:
                    trace.instructions.append({"index": idx, "class": "UNKNOWN", "funct": None,
                                               "raw": line.strip(), "decoded": {}})
            idx += 1
            continue
        # scalar SSA plumbing that feeds operand resolution (constant / ptrtoint-of-arg / add)
        head, sep, _ = line.partition("=")
        lhs = _read_ssa(head.strip(), 0)[0] if sep and head.strip().startswith("%") else None
        if "llvm.mlir.constant(" in line:
            num = line.partition("llvm.mlir.constant(")[2].split(":", 1)[0].strip()
            try:
                if lhs is not None:
                    ssa[lhs] = _Val("const", value=int(num, 0))   # 0-base: accept hex/dec/neg literals
            except ValueError:
                pass
        elif "llvm.ptrtoint" in line and "%arg" in line:
            digits = ""
            for ch in line.partition("%arg")[2]:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if lhs is not None and digits:
                ssa[lhs] = _Val("argbase", arg_index=int(digits), offset=0)
        elif "llvm.add" in line:
            names = _ssa_names(line.partition("llvm.add")[2])
            if lhs is not None and len(names) >= 2:
                va, vb = ssa.get(names[0]), ssa.get(names[1])
                base = next((v for v in (va, vb) if v and v.kind == "argbase"), None)
                const = next((v for v in (va, vb) if v and v.kind == "const"), None)
                if base is not None and const is not None:
                    ssa[lhs] = _Val("argbase", arg_index=base.arg_index,
                                    offset=(base.offset or 0) + (const.value or 0))
                elif va and vb and va.kind == "const" and vb.kind == "const":
                    ssa[lhs] = _Val("const", value=(va.value or 0) + (vb.value or 0))
    return trace.to_dict()


def decode_text(text: str, source: str | None = None, *, target: str) -> dict:
    """Decode LLVM-MLIR ``text`` into a structured instruction trace dict, using ``target``'s RTL-derived
    ISA facts (custom opcode + func7->class map). The target is required — the decoder holds no default and
    bakes in nothing target-specific. PARSED structurally when the text is a module (see
    :func:`decode_module`) so operands are read from the IR regardless of the ``llvm.inline_asm`` spelling;
    otherwise a tolerant line scan (:func:`_decode_by_text_scan`) still SEES every present instruction and
    marks UNKNOWN (never drops) what it cannot classify — we do not assume we know every form a backend
    emits."""
    module = _parse_module(text)
    if module is not None:
        return decode_module(module, target=target, source=source)
    return _decode_by_text_scan(text, target=target, source=source)


def decode_file(path: str | Path, *, target: str) -> dict:
    p = Path(path)
    return decode_text(p.read_text(encoding="utf-8"), source=str(p), target=target)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Decode RoCC trace from lowered.llvm.mlir")
    ap.add_argument("input")
    ap.add_argument("--target", required=True, help="target whose RTL-derived ISA to decode against")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    trace = decode_file(a.input, target=a.target)
    out = json.dumps(trace, indent=2)
    if a.out:
        Path(a.out).write_text(out)
        print(f"wrote {a.out}: {len(trace['instructions'])} instructions")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
