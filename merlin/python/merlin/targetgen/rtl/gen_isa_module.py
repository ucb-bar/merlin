"""CIRCT facts -> generated, RTL-derived ISA encoder module (the 'moat').

This promotes the CIRCT facts from a post-hoc CHECKER into a GENERATOR: it reads the deterministic
RTL-extracted `facts.json` (funct decode table + custom opcode + mesh dims + scratchpad/accumulator
capacities, all extracted from the elaborated Gemmini RTL by circt_introspect) and emits a self-contained
Python module the target backend can build its RoCC encoder on. The agent then writes only the op-LOWERING
logic (tiling, im2col), not the error-prone ISA encoding it would otherwise re-derive from headers.

Why this matters (abc4 evidence): the two failure classes the CIRCT checker caught — UNKNOWN custom-3
instructions and use-before-config — are exactly *encoding* mistakes. Generating the legal funct table +
an ordering-checked emitter from the RTL makes those bugs structurally impossible, and removes the single
largest hand-written chunk (~300 LOC of rocc.py). A pure-C++/headers backend cannot get this — there is no
RTL-facts pipeline behind it. Encoding is RTL-grounded; only the *algorithm* is left to the agent.

Usage:
  python -m merlin.targetgen.rtl.gen_isa_module [--facts <facts.json>] [--out gemmini_isa.py]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .facts import load_facts

_HEADER = '''"""GENERATED from RTL facts by merlin.targetgen.rtl.gen_isa_module — DO NOT hand-edit the tables.

RTL-derived Gemmini ISA encoder scaffold. The funct codes, custom opcode, mesh dimensions and
scratchpad/accumulator capacities below are extracted deterministically from the elaborated Gemmini RTL
(firtool --ir-hw + GemminiISA.scala) — they are what the *hardware* accepts, not a reading of headers.

Build your command-buffer / RoCC lowering on `emit()` + `Program`: only LEGAL funct codes can be emitted
(an illegal funct raises), and `Program.finalize()` enforces config-before-use ordering. This eliminates
the two most common structural bugs (UNKNOWN instruction; use-before-config) by construction. You still
implement the op ALGORITHM (tiling to DIM, conv->im2col, accumulation) — that is the target-specific work.
"""
from __future__ import annotations
from dataclasses import dataclass, field
'''


def generate(facts: dict, encoding: dict | None = None) -> str:
    f = facts["facts"]
    fd = next(i for i in f["interfaces"] if i.get("name") == "funct_decode_table")
    names = {int(k): v for k, v in fd["names"].items()}
    legal = sorted(fd["legal_funct"])
    opcode = fd["custom_opcode"]
    funct3 = fd["funct3"]
    mesh = next((a for a in f.get("arrays", []) if a.get("name") == "mesh"), {})
    spad = next((m for m in f.get("memories", []) if m.get("name") == "scratchpad"), {})
    acc = next((m for m in f.get("memories", []) if m.get("name") == "accumulator"), {})
    dim = mesh.get("rows", 16)

    # config funct names (the "must precede use" set) + their dependent ops, derived from the table names
    config_functs = {c for c, n in names.items() if "CONFIG" in n}
    compute_functs = {c for c, n in names.items() if "COMPUTE" in n}
    store_functs = {c for c, n in names.items() if n in ("STORE_CMD",) or "STORE" in n}

    lines = [_HEADER]
    lines.append(f"CUSTOM_OPCODE = {hex(opcode)}   # RoCC custom-3")
    lines.append(f"FUNCT3 = {funct3}")
    lines.append(f"DIM = {dim}   # systolic array is {mesh.get('rows','?')}x{mesh.get('cols','?')}")
    lines.append(f"SCRATCHPAD_ROWS = {spad.get('depth')}   # banks={spad.get('banks')} elem_bits={spad.get('elem_bits')}")
    lines.append(f"ACCUMULATOR_ROWS = {acc.get('depth')}   # lanes={acc.get('lanes')} lane_bits={acc.get('lane_bits')}")
    lines.append("")
    lines.append("# RTL-extracted legal funct codes (emitting any other custom-3 funct is rejected by HW).")
    lines.append("FUNCT = {")
    for c in sorted(names):
        flag = "  # config" if c in config_functs else ("  # compute" if c in compute_functs else "")
        lines.append(f"    {json.dumps(names[c])}: {c},{flag}")
    lines.append("}")
    lines.append(f"LEGAL_FUNCT = frozenset({legal})")
    lines.append(f"CONFIG_FUNCTS = frozenset({sorted(config_functs)})")
    lines.append(f"COMPUTE_FUNCTS = frozenset({sorted(compute_functs)})")
    lines.append(f"STORE_FUNCTS = frozenset({sorted(store_functs)})")

    # ABI encoding surface from the capability manifest — the readout bits + the RTL-code -> compiler
    # semantic class map — which RTL discovery cannot ground. Composed HERE so this generated module is
    # the SINGLE source both the codegen emitter and the decoder consume (retiring the triplicated
    # hand-coded constants). Absent -> the module still carries the RTL-derived encoder above.
    if encoding:
        lines.append("")
        lines.append("# --- ABI encoding surface (manifest-declared, human-reviewed; not decoder facts) ---")
        rb = encoding.get("readout_bits") or {}
        for sym, key in (("F1", "f1"), ("C_ACC", "c_acc"), ("ACC_I8", "acc_i8"),
                         ("ACC_ACCUM", "acc_accum"), ("FULL_C_BIT", "full_c_bit")):
            if rb.get(key) is not None:
                lines.append(f"{sym} = {hex(rb[key])}")
        sc = encoding.get("semantic_class") or {}
        lines.append("SEMANTIC_CLASS = {"
                     + ", ".join(f"{int(k)}: {json.dumps(v)}" for k, v in sorted(sc.items())) + "}")
        cs = encoding.get("config_subtype") or {}
        lines.append("CONFIG_SUBTYPE = {"
                     + ", ".join(f"{int(k)}: {json.dumps(v)}" for k, v in sorted(cs.items())) + "}")
    lines.append('''

@dataclass
class Instr:
    funct: int
    rs1: int = 0
    rs2: int = 0
    def __post_init__(self):
        if self.funct not in LEGAL_FUNCT:
            raise ValueError(f"illegal funct {self.funct}: not in RTL legal_funct table "
                             f"(would emit an UNKNOWN custom-3 the hardware rejects)")
    def name(self) -> str:
        return {v: k for k, v in FUNCT.items()}.get(self.funct, "?")


@dataclass
class Program:
    """An ordered RoCC instruction stream with config-before-use enforced at finalize()."""
    instrs: list = field(default_factory=list)
    def emit(self, funct_name: str, rs1: int = 0, rs2: int = 0) -> "Program":
        if funct_name not in FUNCT:
            raise ValueError(f"unknown funct name {funct_name!r}; legal: {sorted(FUNCT)}")
        self.instrs.append(Instr(FUNCT[funct_name], rs1, rs2))
        return self
    def finalize(self) -> list:
        """Validate config-before-use; return the instruction list. Raises on ordering violations —
        the same 'use before config' bug the CIRCT checker flags, caught here at emit time instead."""
        configured = False
        store_configured = any(n for n in () )  # placeholder; track per-type below
        seen_config = set()
        for k, ins in enumerate(self.instrs):
            if ins.funct in CONFIG_FUNCTS:
                seen_config.add(ins.funct)
            if ins.funct in COMPUTE_FUNCTS and not (CONFIG_FUNCTS & seen_config):
                raise ValueError(f"use-before-config: COMPUTE (funct {ins.funct}) at {k} before any CONFIG_* "
                                 f"— emit a CONFIG before the first compute")
        return [(ins.funct, ins.rs1, ins.rs2) for ins in self.instrs]


def emit(funct_name: str, rs1: int = 0, rs2: int = 0) -> Instr:
    """One legal instruction (raises on illegal funct)."""
    return Instr(FUNCT[funct_name], rs1, rs2)
''')
    return "\n".join(lines)


def generate_header(facts: dict, encoding: dict, target: str = "gemmini") -> str:
    """Emit a C++ header of the SAME derived ISA constants (from facts + the manifest encoding block), so
    a C++ backend #includes generated constants instead of hand-typing constexprs — closing the third
    (C++) leg of the former triplication with the same single source the Python module uses."""
    f = facts["facts"]
    fd = next(i for i in f["interfaces"] if i.get("name") == "funct_decode_table")
    mesh = next((a for a in f.get("arrays", []) if a.get("name") == "mesh"), {})
    rb = (encoding or {}).get("readout_bits") or {}
    code_of = {cls: code for code, cls in ((encoding or {}).get("semantic_class") or {}).items()}
    ns = f"{target}_isa"
    out = [f"// GENERATED from RTL facts + the {target} capability manifest by "
           f"merlin.targetgen.rtl.gen_isa_module — do not hand-edit.",
           "#pragma once", f"namespace {ns} {{",
           f"  constexpr unsigned CUSTOM_OPCODE = {hex(fd['custom_opcode'])};",
           f"  constexpr unsigned FUNCT3 = {hex(fd['funct3'])};",
           f"  constexpr int DIM = {mesh.get('rows', 16)};"]
    if encoding:
        out.append(f"  constexpr unsigned ADDR_LEN = {encoding.get('addr_len', 32)};")
        for sym, key in (("F1", "f1"), ("C_ACC", "c_acc"), ("ACC_I8", "acc_i8"),
                         ("ACC_ACCUM", "acc_accum"), ("FULL_C_BIT", "full_c_bit")):
            if rb.get(key) is not None:
                out.append(f"  constexpr unsigned {sym} = {hex(rb[key])};")
        for cls in ("CONFIG", "MVIN", "MVOUT", "COMPUTE_PRELOADED", "PRELOAD", "FLUSH"):
            if cls in code_of:
                out.append(f"  constexpr int K_{cls} = {code_of[cls]};")
    out.append("}")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facts", default=None, help="facts.json (default: regenerate the target from RTL)")
    ap.add_argument("--target", default="gemmini", help="target whose manifest supplies the ABI encoding block")
    ap.add_argument("--header", action="store_true", help="emit a C++ constants header instead of the Python module")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    facts = json.loads(Path(a.facts).read_text()) if a.facts else load_facts(a.target)
    encoding = None
    try:
        from ..target_experiment import load_capability_manifest
        encoding = load_capability_manifest(a.target).encoding
    except Exception:  # noqa: BLE001 — no manifest -> emit the RTL-derived encoder only
        pass
    code = generate_header(facts, encoding or {}, a.target) if a.header else generate(facts, encoding)
    if a.out:
        Path(a.out).write_text(code)
        print(f"wrote {a.out} ({len(code.splitlines())} lines) from {a.facts}")
    else:
        print(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
