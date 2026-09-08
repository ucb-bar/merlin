"""Grammar-routing parse: one xDSL Context that admits BOTH input grammars.

`merlin_iface` v0.1 and the `linalg-on-tensors` capsule form are both real MLIR, so both are
parsed by the same structural parser; the module's own attributes decide which reader runs.
"""
from __future__ import annotations

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, linalg, math, memref, scf, tensor
from xdsl.dialects.builtin import ModuleOp
from xdsl.parser import Parser

from .iface_dialect import GRAMMAR_VERSION, MERLIN_IFACE


class GrammarError(Exception):
    """The module is well-formed MLIR but not a grammar version this backend implements."""

_DIALECTS = (builtin.Builtin, func.Func, linalg.Linalg, tensor.Tensor, arith.Arith,
             math.Math, scf.Scf, memref.MemRef)


def context() -> Context:
    ctx = Context(allow_unregistered=True)
    for dialect in _DIALECTS:
        ctx.load_dialect(dialect)
    ctx.load_dialect(MERLIN_IFACE)
    return ctx


def parse_module(text: str) -> ModuleOp:
    """Parse + verify, and reject a grammar version this backend does not implement.

    The version gate belongs HERE, in `parse`: the contract says a consumer must reject a version
    it does not implement, and a module that only fails later has already been accepted.
    """
    module = Parser(context(), text).parse_module()
    module.verify()
    attr = module.attributes.get("merlin_iface.version")
    if attr is not None:
        version = getattr(attr, "data", None)
        if version != GRAMMAR_VERSION:
            raise GrammarError(
                f"merlin_iface version {version!r} is not implemented (this backend implements "
                f"{GRAMMAR_VERSION!r})")
    return module


def is_merlin_iface(module: ModuleOp) -> bool:
    return "merlin_iface.version" in module.attributes
