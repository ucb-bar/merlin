"""Structural verification for the Python/xDSL path — the C++-MLIR-verifier equivalent.

abc4 lesson: C++ gets MLIR's compile-time verifier FOR FREE (you can't compile MLIR without the type
system + ODS operand constraints + IR verifier rejecting a broken graph). The Python arms wrote 0
verifiers and *bypassed xDSL's IR* (hand-rolled regex/dict emitters) — so they opted out of the checking
xDSL actually provides, and discovered structural bugs only at grade time.

xDSL CAN verify a broken graph exactly like C++ — IF you build a real xDSL module (typed IRDL ops) and
call `verify()`. This module makes that the easy path:
  * verify_module(module)        — run xDSL's native `module.verify()` (the "is my graph broken?" check)
  * legal_functs(header, define_prefix=...) — derive a declared ISA-code set from a public header
  * structural_checks(trace,...) — generic decode and declared-code checks only
  * validate(module, cb, trace)  — one call: graph verify + cmdbuf schema + structural ⇒ findings list

ANTI-CHEAT / arm line: everything here is GENERIC framework verification or derived from an explicit
PUBLIC input. Protocol ordering, instruction roles, capacities and trace assertions belong to the
selected target support's RTL-check implementation. This gives the no-CIRCT arm a compile-time graph
check, not parity with RTL-grounded screening. Shared by all xDSL arms.
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
            from xdsl.dialects.builtin import Builtin
            from xdsl.parser import Parser

            ctx = Context()
            ctx.load_dialect(Builtin)
            module = Parser(ctx, module).parse_module()
        module.verify()  # raises on a malformed graph (type/operand/rank/invariant)
        return []
    except Exception as e:
        return [f"xDSL verify: {type(e).__name__}: {str(e)[:200]}"]


def legal_functs(header: str | Path, *, define_prefix: str) -> dict[str, int]:
    """Parse a target-declared public ``#define <prefix><NAME> <integer>`` table.

    The prefix is an input, never an assumed ISA convention. A header with no matching definitions
    is not evidence of an empty legal instruction set.
    """
    if not define_prefix or not define_prefix.isidentifier():
        raise ValueError("define_prefix must be a non-empty C identifier prefix")
    txt = Path(header).read_text(errors="ignore")
    out: dict[str, int] = {}
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] != "#define" or not parts[1].startswith(define_prefix):
            continue
        name = parts[1][len(define_prefix) :]
        if not name or not all(c.isupper() or c.isdigit() or c == "_" for c in name):
            continue
        try:
            out[name] = int(parts[2], 0)
        except ValueError:
            continue
    if not out:
        raise ValueError(f"no integer ISA definitions with prefix {define_prefix!r} in {header}")
    return out


def structural_checks(trace: dict, legal: dict[str, int] | None = None) -> list[str]:
    """Decode and declared-code checks, independent of instruction names or ordering protocol.

    A target's config-before-compute or other role rule must be checked by selected support, not
    inferred from substrings in instruction names.
    """
    if not isinstance(trace, dict) or not isinstance(trace.get("instructions"), list):
        return ["trace: expected an instructions list"]
    out = []
    instrs = trace["instructions"]
    legal_vals = set(legal.values()) if legal is not None else None
    for i, ins in enumerate(instrs):
        if not isinstance(ins, dict):
            out.append(f"instr[{i}]: expected an instruction record")
            continue
        raw_name = ins.get("name")
        name = raw_name.upper() if isinstance(raw_name, str) else ""
        funct = ins.get("funct")
        if funct == "UNKNOWN" or name in ("UNKNOWN", "?"):
            out.append(f"instr[{i}]: UNKNOWN/undecodable instruction")
        if legal_vals is not None and funct is not None:
            if isinstance(funct, bool) or not isinstance(funct, int):
                out.append(f"instr[{i}]: funct must be an integer code")
            elif funct not in legal_vals:
                out.append(f"instr[{i}]: funct {funct} not in the public ISA funct table")
    return out


def validate(
    module=None,
    cb: dict | None = None,
    trace: dict | None = None,
    *,
    isa_header: str | Path | None = None,
    define_prefix: str | None = None,
) -> dict[str, Any]:
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
        if (isa_header is None) != (define_prefix is None):
            raise ValueError("isa_header and define_prefix must be supplied together")
        legal = legal_functs(isa_header, define_prefix=define_prefix) if isa_header is not None else None
        findings += structural_checks(trace, legal)
    return {"ok": not findings, "findings": findings}
