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
    dim = mesh.get("rows", 16)
    # The custom major opcode is an RTL-extracted FACT (funct_decode_table.custom_opcode), NOT a baked
    # literal: the decoder must match whatever opcode the RTL decoder actually reserves. funct3 is the
    # RoCC xd/xs1/xs2 register-usage field — it VARIES per instruction (e.g. a result-returning op sets
    # xd=1), so it is NOT an identity constraint; instruction identity is func7 (-> FUNCT_CLASS).
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

# --- structured line parsing (no regex) -------------------------------------------------------
# This decoder is a fair MEASUREMENT of whatever the backend emitted, so it must SEE every inline-asm
# spelling and fail closed (UNKNOWN) on anything it can't fully decode — never silently drop a line.
# Regex line-matching repeatedly broke that contract by being too narrow (numeric-only SSA ids;
# "r,r"-only constraints; the pretty ``llvm.inline_asm`` op spelling only — each silently dropped
# valid-but-different backend output). We tokenize with plain string ops instead: robust to naming,
# operand-count, whitespace, and op-spelling variation, with no hidden pattern to out-narrow the input.

# An SSA identifier body: MLIR allows numeric (%0) or named (%c0, %a.1, %w-1) ids.
_SSA_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$-")


def _is_asm(line: str) -> bool:
    """A line carries an inline-asm instruction iff one of its quoted strings is an instruction
    TEMPLATE (``.insn …`` or ``fence``). Detecting by the template — not the wrapper keyword — is
    spelling-AGNOSTIC: it fires for MLIR (``llvm.inline_asm`` / ``"llvm.intr.inlineasm"()``) AND
    textual LLVM-IR (``call … asm sideeffect ".insn …"``) AND any future wrapper. A valid emission can
    never be silently dropped for using an unrecognized wrapper (the recurring narrowness bug); the
    worst case is a fail-closed UNKNOWN."""
    inside, _ = _quoted(line)
    return any(s.strip().startswith(".insn") or s.strip() == "fence" for s in inside)


def _read_ssa(s: str, i: int) -> tuple[str, int]:
    """``s[i]`` is '%'; return (identifier, index-after-it)."""
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


def _lhs_ssa(line: str) -> str | None:
    """The result name of ``%v = ...``, else None."""
    head, sep, _ = line.partition("=")
    if not sep:
        return None
    head = head.strip()
    return _read_ssa(head, 0)[0] if head.startswith("%") else None


def _quoted(line: str) -> tuple[list[str], str]:
    """Split on double-quotes: (strings-inside-quotes, text-after-the-last-quote). The asm ops here
    carry no escaped quotes, so a plain split is exact."""
    parts = line.split('"')
    inside = parts[1::2]
    after = parts[-1] if len(parts) % 2 == 1 else ""
    return inside, after


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
            dec = {"subtype": "LD", "stride": r2}
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


def _asm_template(inside: list[str]) -> str:
    """The instruction template among an inline-asm op's quoted strings (the generic op form quotes
    the op NAME too, so it is not always the first) — the one that is ``.insn ...`` or ``fence``."""
    for s in inside:
        st = s.strip()
        if st.startswith(".insn") or st == "fence":
            return st
    return ""


def decode_text(text: str, source: str | None = None, *, target: str) -> dict:
    """Decode lowered LLVM-MLIR text into a structured instruction trace dict, using ``target``'s
    RTL-derived ISA facts (custom opcode + func7->class map). The target is required — the decoder
    holds no default and bakes in nothing target-specific."""
    isa = isa_constants(target)
    custom_opcode = isa["CUSTOM_OPCODE"]
    ssa: dict[str, _Val] = {}
    trace = Trace(source=source, custom_opcode=custom_opcode, funct3=isa["FUNCT3"])
    idx = 0

    for line in text.splitlines():
        # Inline-asm first: it carries the RoCC instructions. SEE every spelling, fail closed on
        # anything not fully decodable — never drop a present asm line.
        if _is_asm(line):
            inside, after = _quoted(line)
            template = _asm_template(inside)
            parsed = _parse_insn(template, custom_opcode) if template.startswith(".insn") else None
            if parsed is not None:
                funct, _rd_is_x0 = parsed
                ops = _ssa_names(after)  # trailing source operands, in order (rs1, rs2)
                rs1 = ssa.get(ops[0]) if len(ops) >= 1 else None
                rs2 = ssa.get(ops[1]) if len(ops) >= 2 else None
                klass, dec = _decode_one(funct, rs1, rs2, isa)
                trace.instructions.append({
                    "index": idx, "class": klass, "funct": funct,
                    "rs1": _operand(rs1), "rs2": _operand(rs2), "decoded": dec,
                })
            elif template == "fence":
                trace.instructions.append({"index": idx, "class": "FENCE", "funct": None,
                                           "decoded": {}})
            else:
                # An inline-asm we do not recognize: fail-closed (record, do not drop).
                trace.instructions.append({"index": idx, "class": "UNKNOWN", "funct": None,
                                           "raw": line.strip(), "decoded": {}})
            idx += 1
            continue

        # Scalar SSA plumbing that feeds operand resolution.
        lhs = _lhs_ssa(line)
        if "llvm.mlir.constant(" in line:
            _, _, rest = line.partition("llvm.mlir.constant(")
            num = rest.split(":", 1)[0].strip()
            try:
                if lhs is not None:
                    ssa[lhs] = _Val("const", value=int(num))
            except ValueError:
                pass
            continue
        if "llvm.ptrtoint" in line:
            _, _, rest = line.partition("%arg")
            digits = ""
            for ch in rest:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if lhs is not None and digits:
                ssa[lhs] = _Val("argbase", arg_index=int(digits), offset=0)
            continue
        if "llvm.add" in line:
            _, _, rest = line.partition("llvm.add")
            names = _ssa_names(rest)
            if lhs is not None and len(names) >= 2:
                va, vb = ssa.get(names[0]), ssa.get(names[1])
                base = next((v for v in (va, vb) if v and v.kind == "argbase"), None)
                const = next((v for v in (va, vb) if v and v.kind == "const"), None)
                if base is not None and const is not None:
                    ssa[lhs] = _Val("argbase", arg_index=base.arg_index,
                                    offset=(base.offset or 0) + (const.value or 0))
                elif va and vb and va.kind == "const" and vb.kind == "const":
                    ssa[lhs] = _Val("const", value=va.value + vb.value)
                else:
                    ssa[lhs] = _Val("unknown")
            continue

    return trace.to_dict()


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
