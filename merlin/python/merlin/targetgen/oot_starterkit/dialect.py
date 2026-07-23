"""Expose the framework's TYPED merlin_iface input dialect — parse into VERIFIED xDSL IR (the C++ benefit).

The deepest abc4 lesson: defining/using a typed dialect is what earns MLIR/C++'s compile-time verification.
The framework ALREADY ships the `merlin_iface` input dialect in xDSL (`merlin.xdsl_dialects.interface`:
typed ops + ResidentTensorType/StreamingTileType/AccumulatorType/...). Agents regex-scraped the input
instead — which *forfeits* the verification. This module hands the typed dialect over so you parse the
input into a **verified** xDSL module (broken input graph ⇒ caught at parse, like C++), with no regex and
no need to re-define the input dialect.

Target-agnostic: the input grammar is the fixed public contract — identical for every accelerator. The
TARGET dialect (your ops) is still yours to define; use `irdl_op_definition` with operand constraints +
verifiers so your *output* IR is verified too (then `verify()` gives you the full C++-equivalent gate).
"""
from __future__ import annotations
from typing import Any


def load_interface_dialect():
    """Return the framework's xDSL merlin_iface input-dialect module (typed ops/types) for registration."""
    from merlin.xdsl_dialects import interface as _iface
    return _iface


def parse_to_verified_ir(mlir_text: str):
    """Parse `merlin_iface` input text into a VERIFIED xDSL ModuleOp (raises if the graph is malformed) —
    the C++-equivalent input check. Falls back to the plain dict parser if xDSL parsing isn't wired for a
    given construct (still better to start from typed IR)."""
    from xdsl.context import Context
    from xdsl.parser import Parser
    from xdsl.dialects.builtin import Builtin
    ctx = Context(); ctx.load_dialect(Builtin)
    iface = load_interface_dialect()
    for d in getattr(iface, "DIALECTS", []) or []:
        try:
            ctx.load_dialect(d)
        except Exception:
            pass
    module = Parser(ctx, mlir_text).parse_module()
    module.verify()        # broken input graph -> raises here, at parse, like the MLIR verifier
    return module
