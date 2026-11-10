"""Prepare full graph-chain candidates without Lean/native execution."""

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.parser import Parser


def parser(text):
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    return Parser(context, text).parse_module()
