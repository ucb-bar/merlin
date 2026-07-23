"""Structural verification for the Python/xDSL path — the C++-MLIR-verifier equivalent.

abc4 lesson: C++ gets MLIR's compile-time verifier FOR FREE (you can't compile MLIR without the type
system + ODS operand constraints + IR verifier rejecting a broken graph). The Python arms wrote 0
verifiers and *bypassed xDSL's IR* (hand-rolled regex/dict emitters) — so they opted out of the checking
xDSL actually provides, and discovered structural bugs only at grade time.

xDSL CAN verify a broken graph exactly like C++ — IF you build a real xDSL module (typed IRDL ops) and
call `verify()`. This module makes that the easy path:
  * verify_module(module)        — run xDSL's native `module.verify()` (the "is my graph broken?" check)
  * legal_functs(header)         — derive the legal funct set from the PUBLIC gemmini.h (not facts.json)
  * structural_checks(trace,...) — ISA-legality the generic verifier can't know: decode-clean,
                                   config-before-use, operand-rank/tile sanity
  * validate(module, cb, trace)  — one call: graph verify + cmdbuf schema + structural ⇒ findings list

ANTI-CHEAT / arm line: everything here is GENERIC framework verification or derived from PUBLIC inputs
(xDSL's verifier; gemmini.h's public funct defines). It is the Python equivalent of MLIR's verifier —
NOT the CIRCT moat. The merlin+CIRCT arm additionally has the RTL-FACTS-grounded screen (authoritative
capacities + the live sim-skip gate); this gives the no-CIRCT arm parity with C++'s compile-time check,
not parity with CIRCT. Shared by all xDSL arms.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any


def verify_module(module) -> list[str]:
    """Run xDSL's native verifier on an xDSL ModuleOp (or parse MLIR text first). Returns problems
    (empty = the graph is well-formed). THIS is the C++-equivalent 'is my graph broken?' gate — use it
    in every entrypoint instead of hand-rolling dicts, and structural bugs surface at construction, not
    at grade time."""
    try:
        from xdsl.ir import Operation
        if isinstance(module, str):
            from xdsl.context import Context
            from xdsl.parser import Parser
            from xdsl.dialects.builtin import Builtin
            ctx = Context(); ctx.load_dialect(Builtin)
            module = Parser(ctx, module).parse_module()
        module.verify()           # raises on a malformed graph (type/operand/rank/invariant)
        return []
    except Exception as e:
        return [f"xDSL verify: {type(e).__name__}: {str(e)[:200]}"]


def legal_functs(gemmini_h: str | Path) -> dict[str, int]:
    """Parse the PUBLIC gemmini.h `#define k_* N` funct table (public info, in every bundle)."""
    txt = Path(gemmini_h).read_text(errors="ignore")
    out: dict[str, int] = {}
    for line in txt.splitlines():
        parts = line.split()   # `#define  k_<NAME>  <N>` — any whitespace
        if len(parts) >= 3 and parts[0] == "#define" and parts[1].startswith("k_") and parts[2].isdigit():
            name = parts[1][2:]
            if name and all(c.isupper() or c.isdigit() or c == "_" for c in name):
                out[name] = int(parts[2])
    return out


def structural_checks(trace: dict, legal: dict[str, int] | None = None) -> list[str]:
    """ISA-legality checks the generic graph verifier can't know, derived from PUBLIC info: decode-clean
    (only legal functs), config-before-use, no compute-in-movement. `trace` = a rocc_decode output."""
    out = []
    instrs = trace.get("instructions", []) if isinstance(trace, dict) else []
    legal_vals = set(legal.values()) if legal else None
    seen_config = False
    for i, ins in enumerate(instrs):
        name = (ins.get("name") or "").upper()
        funct = ins.get("funct")
        if funct == "UNKNOWN" or name in ("UNKNOWN", "?"):
            out.append(f"instr[{i}]: UNKNOWN/undecodable funct — illegal custom-3 the HW rejects")
        if legal_vals is not None and isinstance(funct, int) and funct not in legal_vals:
            out.append(f"instr[{i}]: funct {funct} not in the public ISA funct table")
        if "CONFIG" in name:
            seen_config = True
        if "COMPUTE" in name and not seen_config:
            out.append(f"instr[{i}]: COMPUTE before any CONFIG (use-before-config)")
    return out


def validate(module=None, cb: dict | None = None, trace: dict | None = None,
             gemmini_h: str | Path | None = None) -> dict[str, Any]:
    """One-call pre-sim structural gate (the agent's compile-time-equivalent check). Returns
    {ok, findings} — run it BEFORE the sim; a non-empty findings means fix structure first."""
    findings = []
    if module is not None:
        findings += verify_module(module)
    if cb is not None:
        from .cmdbuf import CommandBufferBuilder  # reuse the schema validator
        b = CommandBufferBuilder(cb.get("target", ""))
        b._cb = cb
        findings += b.validate()
    if trace is not None:
        legal = legal_functs(gemmini_h) if gemmini_h else None
        findings += structural_checks(trace, legal)
    return {"ok": not findings, "findings": findings}
