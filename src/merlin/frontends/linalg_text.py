"""Pure xDSL tensor-source parsing, independent of capture and compiler tools."""

def strip_paren_results(text: str) -> str:
    """Drop the parentheses from a multi-result ``linalg`` region terminator.

    MLIR prints a multi-result ``linalg.generic`` as (real line from
    ``out/runs/rvv/beam/matmul/.../generated/v/model.prepared.mlir``)::

        } -> (tensor<1x32xf32>, tensor<1x32xi64>)

    and xDSL's linalg custom syntax rejects the parenthesized form — still true on the
    xDSL pinned here (0.68), verified by parsing that exact text: ``Expected '->'``. So the
    parens are removed BEFORE the parse; this repair is what makes the module parseable at
    all, which is why it is textual.

    Scanned structurally: a ``}``, optional whitespace, ``->``, optional whitespace, then a
    parenthesized group holding no further parentheses. Anything else is left verbatim —
    in particular a nested-paren type list is NOT something this normalizer understands, so
    it reaches the parser unchanged and fails there loudly instead of being mangled here.
    """
    out: list[str] = []
    i = 0
    while True:
        brace = text.find("}", i)
        if brace < 0:
            out.append(text[i:])
            return "".join(out)
        k = brace + 1
        while k < len(text) and text[k].isspace():
            k += 1
        if text[k : k + 2] == "->":
            k += 2
            while k < len(text) and text[k].isspace():
                k += 1
        else:
            k = -1
        if k < 0 or k >= len(text) or text[k] != "(":
            out.append(text[i : brace + 1])  # not a `} -> (...)` terminator
            i = brace + 1
            continue
        close = k + 1
        while close < len(text) and text[close] not in "()":
            close += 1
        if close >= len(text) or text[close] != ")" or close == k + 1:
            out.append(text[i : brace + 1])  # unbalanced / nested / empty — leave alone
            i = brace + 1
            continue
        out.append(text[i:k])  # `} -> ` verbatim, whitespace included
        out.append(text[k + 1 : close])  # the result-type list, parens dropped
        i = close + 1


def make_context():
    """A permissive context for tensor sources and prepared bufferized writers."""
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.bufferization import Bufferization
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.cf import Cf
    from xdsl.dialects.func import Func
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.math import Math
    from xdsl.dialects.memref import MemRef
    from xdsl.dialects.scf import Scf
    from xdsl.dialects.tensor import Tensor

    # Teach the parser the fp8 element types (and let them satisfy arith/math's float constraints)
    # BEFORE any parse, exactly as `xdsl_dialects._common.make_context` does. Without this the two
    # context builders disagree: a capsule spelling `f8E4M3FN` -- or an ordinary `arith.truncf %x :
    # f32 to f8E4M3FN` requantize -- loads through one reader and fails `type expected` in the other.
    from ..xdsl_dialects.fp8 import register_fp8_float_constraints, register_fp8_types

    register_fp8_types()
    register_fp8_float_constraints()

    ctx = Context(allow_unregistered=True)
    for d in (Builtin, Func, Arith, Linalg, Tensor, Scf, Math, Cf, MemRef, Bufferization):
        ctx.load_dialect(d)
    return ctx


def parse_mlir_text(text: str, ctx=None):
    """Parse linalg-on-tensors MLIR text into an xDSL module.

    ``ctx`` lets a caller supply a context that loads extra dialects (e.g. the quant-aware
    context in :mod:`merlin.frontends.quant_ext`); it defaults to the permissive
    :func:`make_context`.
    """
    from xdsl.parser import Parser

    from ..common.ir_lock import IR_LOCK

    text = strip_paren_results(text)
    # THE serialization point for xDSL parsing, held here rather than at call sites because the call
    # sites are not discoverable by inspection: locking the two obvious ones in `build_app` still left
    # `c_runtime.generate` parsing twice for the @forward signature, and the resulting race surfaced as
    # a *mutation* invariant ("Can't add to a block an operation already attached to a block") on
    # perfectly valid IR, in whichever image happened to lose the race. Parsing a whole model is tens
    # of seconds against builds and simulations that take minutes, so the cost of serializing here is
    # small and the alternative is a build whose success depends on scheduling.
    with IR_LOCK:
        return Parser(ctx or make_context(), text).parse_module()
