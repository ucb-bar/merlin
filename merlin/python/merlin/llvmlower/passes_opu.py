"""Route selected int8 contractions to a matrix-unit microkernel, structurally.

This is the step that turns a certified microkernel into a compiler capability. Everything before it
produced a kernel that was correct in isolation; nothing before it made a model *use* one.

**The rewrite is structural, not textual.** The three existing board backends
(``ours_board``/``xnnpack_board``/``openblas_board``) match a contraction with a regex over one line of
MLIR and anchor the declaration they insert with a second regex. They are grandfathered in
``regex_allowlist.txt`` with the migration target written next to them ("-> xDSL / str.find"), and that
list only ever shrinks — so a fourth copy would be a regression. More to the point, a line-oriented
matcher cannot see what decides legality here: the element types, the iterator types, and the indexing
maps. This walks the IR.

**Candidates come from the existing classifier, not a new one.** :func:`kernels.shapes.observe_contractions`
already recognises both named contractions and the contraction *generics* that the int8 rewrite produces
— which is the only form that matters, because ``passes_quant_int`` lowers every contraction to a
``linalg.generic`` with ``(i8, i8, i32)`` operands and it is not renamed back to ``linalg.matmul`` until
``linalg-specialize-generic-ops`` runs later in the pipeline. It also carries ``dtypes`` positionally as
``(lhs, rhs, out)``, so legality is a comparison rather than a re-read of the module.

**What the callee promises.** ``@merlin_opu_gemm_i8_<n>(A, B, C_init) -> C`` computes
``C = A @ B`` in int32 with ``A`` M-major (``MxK``) and ``B`` K-major (``KxN``) — the layout the prepared
IR already has (maps ``(d0,d2)``/``(d2,d1)``). The unit itself needs BOTH operands K-major, so the shim
transposes ``A``; that packing is a real cost the routing decision has to price, not an implementation
detail, and it is why this pass does not pretend the layouts already match.

``bufferization.access`` on the declaration's arguments is load-bearing: without it one-shot-bufferize
defensively copies the weight operand, which for a 12-block transformer is a large amount of pointless
memcpy. The same attributes appear in the board backends for the same reason.

**The init operand must be a zero fill, and that is a correctness condition rather than a convenience.**
``linalg.matmul`` computes ``C_init + A @ B``; the microkernel OVERWRITES its output
(``c[i*n+j] = bias ? bias[j] : 0`` plus the reduction). Those two agree exactly when — and only when —
``C_init`` is zero. Every one of spectformer's 90 candidates is a ``linalg.fill`` of ``0 : i32``, so
nothing is lost by requiring it; what is gained is that a model whose contractions accumulate onto a live
init is DECLINED instead of silently computing ``A @ B`` and dropping the addend.

**Selection is a parameter, never a policy.** ``select`` decides which contractions move; with no
selector nothing is rewritten and the module is returned untouched. A pass that decided for itself would
duplicate the routing decision that :mod:`merlin.targetgen.routing` and the e-graph exist to make.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["OpuRewrite", "Routed", "SYMBOL_PREFIX", "patch_declaration_arg_attrs",
           "rewrite_contractions_to_opu", "routable_contractions", "unpatched_declarations",
           "zero_initialised"]

#: Symbol stem for the emitted callees. One numbered symbol per distinct type signature, because MLIR
#: function types are monomorphic — the same reason the board backends number theirs.
SYMBOL_PREFIX = "merlin_opu_gemm_i8"

#: The operand/accumulator element types this path computes. Compared as data against
#: ``ContractionShape.dtypes``, which is positionally ``(lhs, rhs, out)``.
INT8_DTYPES: tuple[str, str, str] = ("i8", "i8", "i32")

#: Only the rank-2 class. ``linalg.batch_matmul`` is deliberately NOT routed: a contract declaring
#: ``matmul`` gaps it, which is the correct fail-closed reading, and silently folding a batch dim into
#: the tile would compute something the contract never promised. For spectformer that leaves 16 of 106
#: contractions on the vector path, which is a coverage fact to report rather than to paper over.
ROUTABLE_OP = "linalg.matmul"


@dataclass(frozen=True)
class Routed:
    """One contraction that was moved, and the signature it was moved to."""

    symbol: str
    parallel: tuple[int, ...]
    reduction: tuple[int, ...]
    fqn: str = ""

    @property
    def m(self) -> int:
        return int(self.parallel[0])

    @property
    def n(self) -> int:
        return int(self.parallel[1])

    @property
    def k(self) -> int:
        return int(self.reduction[0])


@dataclass(frozen=True)
class OpuRewrite:
    """What the rewrite did. ``signatures`` is what the C side must define."""

    routed: tuple[Routed, ...] = ()
    #: ``{symbol: (m, n, k)}`` — one entry per DISTINCT type signature actually called.
    signatures: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    skipped: tuple[tuple[str, str], ...] = ()      # (what, why)

    @property
    def count(self) -> int:
        return len(self.routed)

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count,
                "routed": [{"symbol": r.symbol, "m": r.m, "n": r.n, "k": r.k, "fqn": r.fqn}
                           for r in self.routed],
                "signatures": {k: list(v) for k, v in self.signatures.items()},
                "skipped": [{"what": w, "why": y} for w, y in self.skipped]}


def zero_initialised(op) -> bool:
    """Whether ``op``'s accumulator init is provably an all-zero fill.

    The microkernel writes its output rather than adding to it, so routing a contraction whose init
    carries live values would DROP that addend — a wrong answer, not a slow one. Answering "no" for
    anything not recognised is the fail-closed direction: an unrecognised init might be zero, and a
    contraction wrongly left on the vector path is merely slower.

    Structural: the init operand's defining op must be ``linalg.fill`` of an ``arith.constant`` integer
    zero. A block argument (whose ``owner`` is a ``Block``, not an ``Operation``) is not a fill and is
    rejected by the same check.
    """
    from xdsl.ir import Operation

    operands = list(getattr(op, "operands", ()))
    if len(operands) < 3:
        return False
    fill = operands[2].owner
    if not isinstance(fill, Operation) or getattr(fill, "name", "") != "linalg.fill":
        return False
    fill_operands = list(fill.operands)
    if not fill_operands:
        return False
    const = fill_operands[0].owner
    if not isinstance(const, Operation) or getattr(const, "name", "") != "arith.constant":
        return False
    return _int_attr_value(getattr(const, "value", None)) == 0


def _int_attr_value(attr) -> int | None:
    """The integer an ``IntegerAttr`` carries, or None when it is not one.

    ``IntegerAttr`` nests its payload (``attr.value.data``); reaching through it explicitly rather than
    parsing ``str(attr)`` means a float or index attribute returns None instead of being misread.
    """
    inner = getattr(attr, "value", None)
    data = getattr(inner, "data", None)
    return data if isinstance(data, int) else None


def routable_contractions(module) -> list[tuple[Any, Any]]:
    """``[(op, shape)]`` for every contraction this path COULD take, with no decision made.

    Separated from the rewrite so a caller (a cost model, an e-graph, a report) can enumerate the
    candidate set without mutating anything — the same split
    :func:`routing.route_candidates` makes for the same reason.

    "Could" means LEGAL, not profitable: rank-2 int8 ``matmul`` accumulating into a zero init. Whether a
    legal contraction is worth moving is the ``select`` caller's decision.
    """
    from ..kernels.shapes import observe_contractions

    out: list[tuple[Any, Any]] = []
    for op, shape in observe_contractions(module):
        if shape.op != ROUTABLE_OP:
            continue
        if tuple(shape.dtypes) != INT8_DTYPES:
            continue
        if len(shape.parallel) != 2 or len(shape.reduction) != 1:
            continue
        if not zero_initialised(op):
            continue
        out.append((op, shape))
    return out


def _signature_key(shape) -> tuple[int, int, int]:
    return (int(shape.parallel[0]), int(shape.parallel[1]), int(shape.reduction[0]))


def rewrite_contractions_to_opu(module, *,
                               select: Callable[[Any], bool] | None = None) -> OpuRewrite:
    """Replace each selected int8 contraction with a call to the matrix-unit kernel.

    Mutates ``module`` in place and returns what it did. ``select`` receives the
    :class:`~merlin.kernels.microkernel.ContractionShape` and returns whether that contraction should
    move; ``None`` moves nothing, so the pass is inert unless a decision has been made elsewhere.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Block, Region

    if select is None:
        return OpuRewrite(skipped=(("all", "no selector supplied, so nothing is routed"),))

    candidates = routable_contractions(module)
    chosen = [(op, sh) for op, sh in candidates if select(sh)]
    skipped: list[tuple[str, str]] = []
    if not chosen:
        skipped.append(("all", f"{len(candidates)} routable contraction(s), none selected"))
        return OpuRewrite(skipped=tuple(skipped))

    # One symbol per distinct signature: MLIR function types are monomorphic, so a 256x196/K=768
    # contraction and a 196x1024/K=256 one cannot share a callee.
    symbols: dict[tuple[int, int, int], str] = {}
    routed: list[Routed] = []

    for op, shape in chosen:
        key = _signature_key(shape)
        sym = symbols.get(key)
        if sym is None:
            sym = f"{SYMBOL_PREFIX}_{len(symbols)}"
            symbols[key] = sym

        operands = list(op.operands)
        if len(operands) != 3 or len(op.results) != 1:
            # A contraction whose operand count is not (lhs, rhs, out-init) is not the shape this
            # callee promises; skip it rather than emit a call with the wrong arity.
            skipped.append((sym, f"expected 3 operands and 1 result, got {len(operands)} and "
                                 f"{len(op.results)}"))
            continue

        call = func.CallOp(sym, operands, [op.results[0].type])
        op.results[0].replace_all_uses_with(call.results[0])
        parent = op.parent
        parent.insert_op_before(call, op)
        op.detach()
        op.erase()

        fqn = ""
        prov = getattr(op, "attributes", {}).get("prov.fqn") if hasattr(op, "attributes") else None
        if isinstance(prov, StringAttr):
            fqn = prov.data
        routed.append(Routed(symbol=sym, parallel=tuple(shape.parallel),
                             reduction=tuple(shape.reduction), fqn=fqn))

    # Declarations go at the END of the module body. The board backends anchor theirs "before the first
    # func.func" with a regex over the printed text; appending to the module's own op list needs no
    # anchor at all and cannot land inside a function.
    body: Block = module.body.block
    for key, sym in symbols.items():
        lhs_t, rhs_t, out_t = _signature_types(module, key)
        # Without these one-shot-bufferize inserts a defensive copy of the weight operand. They are a
        # PROPERTY on FuncOp, not a discardable attribute, so they have to go through the constructor —
        # assigning `attributes["arg_attrs"]` parses and prints as nothing at all, which is how a
        # silently-copied weight would have shipped.
        read = DictionaryAttr({"bufferization.access": StringAttr("read")})
        write = DictionaryAttr({"bufferization.access": StringAttr("write")})
        decl = func.FuncOp(sym, ((lhs_t, rhs_t, out_t), (out_t,)), Region(),
                           visibility="private",
                           arg_attrs=ArrayAttr([read, read, write]))
        body.add_op(decl)

    return OpuRewrite(routed=tuple(routed),
                      signatures={s: k for k, s in symbols.items()},
                      skipped=tuple(skipped))


#: The access each callee argument has, positionally. Read by :func:`patch_declaration_arg_attrs`.
_ARG_ACCESS = ("read", "read", "write")


def patch_declaration_arg_attrs(text: str, rewrite: OpuRewrite) -> str:
    """Put the ``bufferization.access`` attributes back into the printed declarations.

    **This repairs a printer limitation, it is not a design choice.** xDSL stores ``arg_attrs`` on a
    ``func.FuncOp`` correctly and prints them when the function has a body — but for a bodyless
    DECLARATION it prints only the types, so the attributes silently never reach the text mlir-opt
    parses. The consequence is not cosmetic: without them one-shot-bufferize defensively copies the
    weight operand of every routed contraction.

    Done with ``str`` operations rather than a pattern, and anchored on the exact symbols this pass
    emitted rather than on a shape of MLIR — the same discipline ``llvmlower/op_profile.instrument``
    uses. ``passes_xdsl`` already repairs ``tensor.extract_slice`` after the same round-trip, so a
    post-print fixup is the established way to handle the printer here.

    A declaration that cannot be found is left alone and reported by :func:`unpatched_declarations`
    rather than silently skipped.
    """
    out = text
    for sym in rewrite.signatures:
        head = f"func.func private @{sym}("
        at = out.find(head)
        if at < 0:
            continue
        open_paren = at + len(head) - 1
        close = out.find(")", open_paren)
        if close < 0:
            continue
        inner = out[open_paren + 1:close]
        if "bufferization.access" in inner:
            continue          # already annotated; re-splitting would double every attribute
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) != len(_ARG_ACCESS):
            continue
        annotated = ", ".join(f'{p} {{bufferization.access = "{acc}"}}'
                              for p, acc in zip(parts, _ARG_ACCESS, strict=True))
        out = out[:open_paren + 1] + annotated + out[close:]
    return out


def unpatched_declarations(text: str, rewrite: OpuRewrite) -> tuple[str, ...]:
    """Symbols whose declaration in ``text`` still carries no access attributes.

    Non-empty means the weight operands of those callees will be copied by bufferization, so a caller
    that cares about it can fail rather than ship the copies.
    """
    missing = []
    for sym in rewrite.signatures:
        at = text.find(f"func.func private @{sym}(")
        if at < 0:
            missing.append(sym)
            continue
        line_end = text.find("\n", at)
        line = text[at:line_end if line_end > 0 else len(text)]
        if "bufferization.access" not in line:
            missing.append(sym)
    return tuple(missing)


def _signature_types(module, key: tuple[int, int, int]):
    """The three tensor types for one signature, built from the element types this path computes."""
    from xdsl.dialects.builtin import IntegerType, TensorType

    m, n, k = key
    i8 = IntegerType(8)
    i32 = IntegerType(32)
    return (TensorType(i8, [m, k]), TensorType(i8, [k, n]), TensorType(i32, [m, n]))
