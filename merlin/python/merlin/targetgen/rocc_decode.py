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

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

# --- encoding constants: DERIVED from the SINGLE source — the readout bits + RTL-code->class map +
# config subtype from the manifest's encoding block, and the mesh DIM from the CIRCT-extracted facts
# (arrays[mesh]); not hand-copied. Byte-parity with the former literals is pinned by test_encoding_manifest.
# This retires one of the three triplicated copies (the decoder's). GARBAGE/MASK32 are universal.
GARBAGE = 0xFFFFFFFF
MASK32 = 0xFFFFFFFF


def _load_isa() -> dict:
    from .target_experiment import load_capability_manifest
    from .rtl.facts import load_facts
    m = load_capability_manifest("gemmini")
    enc, rb = m.encoding, m.encoding["readout_bits"]
    # DIM (systolic mesh dimension) is a CIRCT-extracted FACT (arrays[mesh]), not a manifest field —
    # same source the codegen emitter reads, so the decoder's DIM cannot drift from the encoder's.
    mesh = next((a for a in load_facts("gemmini")["facts"].get("arrays", []) if a.get("name") == "mesh"), {})
    dim = mesh.get("rows", 16)
    return {"DIM": dim, "F1": rb["f1"], "C_ACC": rb["c_acc"], "ACC_I8": rb["acc_i8"],
            "ACC_ACCUM": rb["acc_accum"], "FULL_C_BIT": rb["full_c_bit"],
            "FUNCT_CLASS": dict(enc["semantic_class"]), "CONFIG_SUBTYPE": dict(enc["config_subtype"])}


_isa = _load_isa()
DIM = _isa["DIM"]
F1 = _isa["F1"]                      # 1.0f
C_ACC = _isa["C_ACC"]               # full-i32 accumulator readout base
ACC_I8 = _isa["ACC_I8"]             # scaled-i8 readout base
ACC_ACCUM = _isa["ACC_ACCUM"]       # accumulate-onto bit
FULL_C_BIT = _isa["FULL_C_BIT"]     # set in C_ACC, clear in ACC_I8 -> distinguishes i32 vs i8 readout
_FUNCT_CLASS = _isa["FUNCT_CLASS"]  # funct -> base instruction class (CONFIG refined via rs1 & 0x3)
_CONFIG_SUBTYPE = _isa["CONFIG_SUBTYPE"]

# --- line regexes -----------------------------------------------------------------------------
# SSA identifiers may be numeric (%0) OR named (%c0, %w, %a.1): the reference emitter happens to use
# numeric names, but any conformant backend may use descriptive ones, so capture a full MLIR SSA id
# ([A-Za-z0-9_.$-]). The ssa table is keyed by the raw name string. (Was %(\d+), which silently
# decoded ONLY numeric-named operands and classified valid named-SSA traces as UNKNOWN.)
_SSA = r"%([A-Za-z0-9_.$-]+)"
_CONST_RE = re.compile(rf"{_SSA}\s*=\s*llvm\.mlir\.constant\((-?\d+)\s*:\s*i64\)")
_PTI_RE = re.compile(rf"{_SSA}\s*=\s*llvm\.ptrtoint\s+%arg(\d+)")
_ADD_RE = re.compile(rf"{_SSA}\s*=\s*llvm\.add\s+{_SSA},\s*{_SSA}")
# Two conformant custom-3 RoCC encodings are accepted (both: funct7=group1, then the two SOURCE
# operands rs1,rs2 as the trailing SSA values; the RTL oracle at L2/L3 remains the correctness gate):
#   2-operand, no result:  .insn r 0x7b, 0x3, <funct>, x0, $0, $1", "r,r[,clobbers]" %rs1, %rs2
#   3-operand, with rd:    .insn r 0x7b, 0x3, <funct>, $0, $1, $2", "=r,r,r[,clobbers]" %rs1, %rs2
# (the 3-operand form binds rd to a real GPR — $0 is the output, $1/$2 the inputs; an LHS %vN = ...
#  carries the result.) Tolerate trailing constraint tokens (e.g. ~{memory}). Anything else stays
# UNKNOWN (fail-closed). Was 2-operand-with-exactly-"r,r" only, which marked the valid 3-operand
# backend output as all-UNKNOWN.
_INSN_RE = re.compile(rf'\.insn r 0x7b, 0x3, (\d+), x0, \$0, \$1",\s*"r,r[^"]*"\s+{_SSA},\s*{_SSA}')
_INSN_RE_RD = re.compile(rf'\.insn r 0x7b, 0x3, (\d+), \$0, \$1, \$2",\s*"=?r,r,r[^"]*"\s+{_SSA},\s*{_SSA}')
_FENCE_RE = re.compile(r'llvm\.inline_asm[^"]*"fence"')
_ANY_ASM_RE = re.compile(r"llvm\.inline_asm")


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
    instructions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        hist: dict[str, int] = {}
        for ins in self.instructions:
            hist[ins["class"]] = hist.get(ins["class"], 0) + 1
        return {
            "source": self.source,
            "abi": {"custom_opcode": "0x7b", "funct3": "0x3"},
            "instructions": self.instructions,
            "summary": {"class_histogram": hist},
        }


def _decode_one(funct: int, rs1: _Val | None, rs2: _Val | None) -> tuple[str, dict]:
    """Return (class, decoded-fields) for one .insn given resolved operands."""
    base = _FUNCT_CLASS.get(funct, "UNKNOWN")
    r1 = rs1.value if (rs1 and rs1.kind == "const") else None
    r2 = rs2.value if (rs2 and rs2.kind == "const") else None
    dec: dict = {}

    if base == "CONFIG":
        sub = _CONFIG_SUBTYPE.get((r1 & 0x3) if r1 is not None else -1, "CONFIG_UNKNOWN")
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
            dec["readout"] = "i32" if (acc_addr & FULL_C_BIT) else "i8"
        return base, dec

    if base == "PRELOAD":
        dec = {}
        if r1 is not None:
            dec["weight_spad"] = r1 & MASK32
        if r2 is not None:
            c_addr = r2 & MASK32
            dec["c_addr"] = c_addr
            dec["accumulate"] = bool(c_addr & ACC_ACCUM)
            dec["readout"] = "i32" if (c_addr & FULL_C_BIT) else "i8"
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


def decode_text(text: str, source: str | None = None) -> dict:
    """Decode lowered LLVM-MLIR text into a structured instruction trace dict."""
    ssa: dict[str, _Val] = {}
    trace = Trace(source=source)
    idx = 0

    for line in text.splitlines():
        m = _CONST_RE.search(line)
        if m and "inline_asm" not in line:
            ssa[m.group(1)] = _Val("const", value=int(m.group(2)))
            continue
        m = _PTI_RE.search(line)
        if m:
            ssa[m.group(1)] = _Val("argbase", arg_index=int(m.group(2)), offset=0)
            continue
        m = _ADD_RE.search(line)
        if m and "inline_asm" not in line:
            dst, a, b = m.group(1), m.group(2), m.group(3)
            va, vb = ssa.get(a), ssa.get(b)
            base = next((v for v in (va, vb) if v and v.kind == "argbase"), None)
            const = next((v for v in (va, vb) if v and v.kind == "const"), None)
            if base is not None and const is not None:
                ssa[dst] = _Val("argbase", arg_index=base.arg_index,
                                offset=(base.offset or 0) + (const.value or 0))
            elif va and vb and va.kind == "const" and vb.kind == "const":
                ssa[dst] = _Val("const", value=va.value + vb.value)
            else:
                ssa[dst] = _Val("unknown")
            continue

        if _ANY_ASM_RE.search(line):
            mi = _INSN_RE.search(line) or _INSN_RE_RD.search(line)
            if mi:
                funct = int(mi.group(1))
                rs1 = ssa.get(mi.group(2))
                rs2 = ssa.get(mi.group(3))
                klass, dec = _decode_one(funct, rs1, rs2)
                trace.instructions.append({
                    "index": idx, "class": klass, "funct": funct,
                    "rs1": _operand(rs1), "rs2": _operand(rs2), "decoded": dec,
                })
                idx += 1
            elif _FENCE_RE.search(line):
                trace.instructions.append({"index": idx, "class": "FENCE", "funct": None,
                                           "decoded": {}})
                idx += 1
            else:
                # An inline-asm we do not recognize: fail-closed (record, do not drop).
                trace.instructions.append({"index": idx, "class": "UNKNOWN", "funct": None,
                                           "raw": line.strip(), "decoded": {}})
                idx += 1

    return trace.to_dict()


def decode_file(path: str | Path) -> dict:
    p = Path(path)
    return decode_text(p.read_text(encoding="utf-8"), source=str(p))


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Decode RoCC trace from lowered.llvm.mlir")
    ap.add_argument("input")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    trace = decode_file(a.input)
    out = json.dumps(trace, indent=2)
    if a.out:
        Path(a.out).write_text(out)
        print(f"wrote {a.out}: {len(trace['instructions'])} instructions")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
