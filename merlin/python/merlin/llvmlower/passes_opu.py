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

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["OpuRewrite", "Routed", "SIDECAR_NAME", "SYMBOL_PREFIX", "load_sidecar",
           "patch_declaration_arg_attrs", "rewrite_contractions_to_opu", "rewrite_prepared_file",
           "routable_contractions", "tile_filling_selector", "unpatched_declarations",
           "zero_initialised"]

#: Where the rewrite records what it minted, beside the prepared module.
#:
#: A file rather than a return value because the two halves run in different PROCESSES: the lowering
#: happens in a subprocess that re-imports these modules, which is the same reason
#: ``impr_features._try_lazy_register`` exists. The build step needs the signatures to generate the C
#: side, and passing them in memory silently produced an empty set there.
SIDECAR_NAME = "opu_signatures.json"

#: Symbol stem for the emitted callees. One numbered symbol per distinct type signature, because MLIR
#: function types are monomorphic — the same reason the board backends number theirs.
SYMBOL_PREFIX = "merlin_opu_gemm_i8"

#: The operand/accumulator element types this path computes. Compared as data against
#: ``ContractionShape.dtypes``, which is positionally ``(lhs, rhs, out)``.
INT8_DTYPES: tuple[str, str, str] = ("i8", "i8", "i32")

#: ``{op: number of PARALLEL dims}``. A batch dimension is a LOOP over the rank-2 contraction, never a
#: third tile axis -- the unit contracts two dims and reduces a third, and folding a batch into the tile
#: would compute something the hardware never promised. So the batched form is admitted only because the
#: callee runs the *same certified rank-2 kernel* once per slice, on operands the slices do not share.
#:
#: This was deliberately matmul-only until the cost of the gap was measured: `linalg.batch_matmul` is
#: 11.89% of spectformer's runtime (attention's QK^T and attn.V) and the vector path runs it at 0.248
#: MACs/cycle, its narrow-shape floor. Declining it was the right fail-closed default while the callee
#: could only do rank 2; it is not right once the callee can do the batch.
ROUTABLE_OPS: dict[str, int] = {"linalg.matmul": 2, "linalg.batch_matmul": 3}

#: Retained for callers that ask what the rank-2 class is called.
ROUTABLE_OP = "linalg.matmul"


@dataclass(frozen=True)
class Routed:
    """One contraction that was moved, and the signature it was moved to."""

    symbol: str
    parallel: tuple[int, ...]
    reduction: tuple[int, ...]
    fqn: str = ""

    @property
    def batch(self) -> int:
        """Leading batch extent, or 1 when the contraction is rank-2. The batch is a LOOP over slices."""
        return int(self.parallel[0]) if len(self.parallel) == 3 else 1

    @property
    def m(self) -> int:
        # Counted from the END, so a batch dim in front does not shift M and N onto the wrong extents.
        return int(self.parallel[-2])

    @property
    def n(self) -> int:
        return int(self.parallel[-1])

    @property
    def k(self) -> int:
        return int(self.reduction[0])


@dataclass(frozen=True)
class OpuRewrite:
    """What the rewrite did. ``signatures`` is what the C side must define."""

    routed: tuple[Routed, ...] = ()
    #: ``{symbol: (m, n, k)}``, or ``(b, m, n, k)`` for a batched one — one entry per DISTINCT type
    #: signature actually called. The arity is what tells the C emitter which entry shape to generate.
    signatures: dict[str, tuple[int, ...]] = field(default_factory=dict)
    skipped: tuple[tuple[str, str], ...] = ()      # (what, why)
    #: The tile edge the decision was made against, when the caller supplied it. Without it the
    #: sidecar cannot say whether a routed contraction fills a tile, so the field is recorded as
    #: UNKNOWN rather than assumed — an unrecorded rule is exactly what this class exists to prevent.
    tile_edge: int | None = None

    @property
    def count(self) -> int:
        return len(self.routed)

    def sub_tile(self) -> tuple["Routed", ...]:
        """Routed contractions that do NOT fill a tile in both parallel dimensions.

        This is the default :func:`tile_filling_selector` rule, evaluated after the fact. A custom
        selector may legitimately route these — but the artifact must SAY that it did.

        MEASURED, and why this is not bookkeeping: a whole-model Gemma image was built with a
        selector that dropped the M requirement, so 183 contractions ran at M=8 on a 64-lane unit.
        Nothing in the build, the sidecar or the measurement recorded that the default rule had been
        overridden, and the run cost 11.6 hours of FireSim before the output could be graded.
        """
        if self.tile_edge is None:
            return ()
        e = int(self.tile_edge)
        return tuple(r for r in self.routed if min(r.m, r.n) < e)

    def to_dict(self) -> dict[str, Any]:
        sub = self.sub_tile()
        return {"count": self.count,
                "routed": [{"symbol": r.symbol, "b": r.batch, "m": r.m, "n": r.n, "k": r.k,
                            "fqn": r.fqn} for r in self.routed],
                "signatures": {k: list(v) for k, v in self.signatures.items()},
                "skipped": [{"what": w, "why": y} for w, y in self.skipped],
                "routing_rule": {
                    "tile_edge": self.tile_edge if self.tile_edge is not None else "UNKNOWN",
                    "fills_default_tile_rule": (None if self.tile_edge is None
                                                else not sub),
                    "sub_tile_routed": len(sub),
                    "sub_tile_dims": sorted({d for r in sub
                                             for d in (("m",) if r.m < int(self.tile_edge or 0) else ())
                                             + (("n",) if r.n < int(self.tile_edge or 0) else ())}),
                    "sub_tile_signatures": sorted({r.symbol for r in sub}),
                }}


def zero_initialised(op) -> bool:
    """Whether ``op``'s accumulator init is provably an all-zero fill.

    The microkernel writes its output rather than adding to it, so routing a contraction whose init
    carries live values would DROP that addend — a wrong answer, not a slow one. Answering "no" for
    anything not recognised is the fail-closed direction: an unrecognised init might be zero, and a
    contraction wrongly left on the vector path is merely slower.

    Structural: the init operand's defining op must be ``linalg.fill`` of an ``arith.constant`` zero,
    integer OR float. A block argument (whose ``owner`` is a ``Block``, not an ``Operation``) is not a
    fill and is rejected by the same check.

    The float case is not cosmetic. This predicate is a CORRECTNESS condition -- it decides whether the
    addend may be dropped -- and "is the init all zeros" has nothing to do with the element type. While
    it recognised integers only, every float contraction answered "no", so a float accelerator could
    never take a single layer through this path no matter what its datapath declared. That reads as a
    device with no work rather than as a predicate that cannot see it.
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
    return _is_zero_attr(getattr(const, "value", None))


def _is_zero_attr(attr) -> bool:
    """Whether ``attr`` is a zero constant of any numeric kind. Unrecognised => False (fail closed).

    Both signed zeros count: dropping a ``-0.0`` addend changes no result, since ``x + -0.0 == x`` for
    every finite ``x`` and ``0.0 + -0.0 == 0.0``. A NaN or a non-numeric attribute is not zero and is
    refused by the comparison itself.
    """
    if _int_attr_value(attr) == 0:
        return True
    value = getattr(attr, "value", None)
    raw = getattr(value, "data", value)
    if isinstance(raw, float):
        return raw == 0.0
    return False


def _int_attr_value(attr) -> int | None:
    """The integer an ``IntegerAttr`` carries, or None when it is not one.

    ``IntegerAttr`` nests its payload (``attr.value.data``); reaching through it explicitly rather than
    parsing ``str(attr)`` means a float or index attribute returns None instead of being misread.
    """
    inner = getattr(attr, "value", None)
    data = getattr(inner, "data", None)
    return data if isinstance(data, int) else None


def routable_contractions(module, *, device: str | None = None,
                          dtypes: "tuple[str, str, str] | None" = None) -> list[tuple[Any, Any]]:
    """``[(op, shape)]`` for every contraction this path COULD take, with no decision made.

    Separated from the rewrite so a caller (a cost model, an e-graph, a report) can enumerate the
    candidate set without mutating anything — the same split
    :func:`routing.route_candidates` makes for the same reason.

    "Could" means LEGAL, not profitable: a rank-2 ``matmul`` on this datapath accumulating into a zero
    init. Whether a legal contraction is worth moving is the ``select`` caller's decision.

    **The datapath is a parameter, not a law.** ``INT8_DTYPES`` is this unit's first declared
    ``accumulate`` rule written down, and written down it belongs to nobody: a second device either
    inherits this one's precision or needs a second copy of the pass. Passing ``device=`` derives the
    triple from that device instead (:mod:`merlin.system.offload`), and ``dtypes=`` states it outright.
    Neither is the default, so every existing caller keeps this unit's behaviour exactly.
    """
    from ..kernels.shapes import observe_contractions

    want = tuple(dtypes) if dtypes else INT8_DTYPES
    if device is not None:
        from merlin.system.offload import device_dtype_triples
        derived = device_dtype_triples(device)
        if not derived:
            return []                    # fail closed: an underivable datapath routes nothing
        accepted = set(derived)
    else:
        accepted = {want}

    out: list[tuple[Any, Any]] = []
    for op, shape in observe_contractions(module):
        want_parallel = ROUTABLE_OPS.get(shape.op)
        if want_parallel is None:
            continue
        if tuple(shape.dtypes) not in accepted:
            continue
        if len(shape.parallel) != want_parallel or len(shape.reduction) != 1:
            continue
        if not zero_initialised(op):
            continue
        out.append((op, shape))
    return out


def _signature_key(shape) -> tuple[int, ...]:
    """``(M, N, K)``, or ``(B, M, N, K)`` when the contraction carries a batch dim.

    The key IS the callee's identity: MLIR function types are monomorphic, so two contractions share a
    symbol only if every extent agrees — and a batched one cannot share with a rank-2 one even at the
    same M/N/K, because the descriptors it is passed have a different arity.
    """
    return (*(int(d) for d in shape.parallel), int(shape.reduction[0]))


def rewrite_contractions_to_opu(module, *,
                               select: Callable[[Any], bool] | None = None,
                               tile_edge: int | None = None) -> OpuRewrite:
    """Replace each selected int8 contraction with a call to the matrix-unit kernel.

    Mutates ``module`` in place and returns what it did. ``select`` receives the
    :class:`~merlin.kernels.microkernel.ContractionShape` and returns whether that contraction should
    move; ``None`` moves nothing, so the pass is inert unless a decision has been made elsewhere.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Block, Region

    if select is None:
        return OpuRewrite(skipped=(("all", "no selector supplied, so nothing is routed"),),
                          tile_edge=tile_edge)

    candidates = routable_contractions(module)
    chosen = [(op, sh) for op, sh in candidates if select(sh)]
    skipped: list[tuple[str, str]] = []
    if not chosen:
        skipped.append(("all", f"{len(candidates)} routable contraction(s), none selected"))
        return OpuRewrite(skipped=tuple(skipped), tile_edge=tile_edge)

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
                      skipped=tuple(skipped), tile_edge=tile_edge)


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


def tile_filling_selector(tile_edge: int) -> Callable[[Any], bool]:
    """Select contractions whose output is at least one whole tile in both parallel dimensions.

    The unit computes a ``tile_edge x tile_edge`` outer product per accumulate whether or not the operands
    fill it, so a contraction narrower than a tile in either direction pays for lanes it does not use;
    :class:`routing.MeasuredCost` prices exactly that. This is the crude form of that decision, stated as
    a threshold — it exists so the compile path can be exercised before the e-graph makes the call, and it
    takes ``tile_edge`` as an argument because the edge is a derived hardware fact
    (:func:`kernels.opu_cert.tile_edge_for_config`) and a threshold baked here would be wrong on every
    other configuration of the unit.
    """
    edge = int(tile_edge)
    if edge < 1:
        raise ValueError(f"tile_edge={tile_edge} is not a lane count; it comes from the hardware's own "
                         "vector length and a guessed one selects the wrong contractions")

    def select(shape) -> bool:
        # The LAST TWO parallel extents are the tile's, whatever the rank: a batch dim in front is a loop
        # over slices and says nothing about how full a tile each slice makes.
        return min(int(shape.parallel[-2]), int(shape.parallel[-1])) >= edge

    return select


def rewrite_prepared_file(prepared: "str | Path", work: "str | Path", *,
                         select: Callable[[Any], bool] | None,
                         tile_edge: int | None = None) -> OpuRewrite:
    """Rewrite a prepared module ON DISK in place and record what it minted.

    This is the seam a whole-model build uses: it reads the module the preparation passes produced,
    routes the selected contractions, prints it back, repairs the declarations the printer drops, and
    writes the sidecar the build's C side reads.

    It REFUSES to write a module whose declarations lost their access attributes. That check is not
    belt-and-braces: the attributes are dropped silently by the printer, the consequence is a defensive
    copy of every routed weight, and a large amount of pointless memcpy in a shipped model is exactly the
    kind of regression nothing would attribute back to here.
    """
    from ..frontends.linalg_mlir import parse_mlir_file
    from ..xdsl_dialects._common import text as to_text

    prepared, work = Path(prepared), Path(work)
    module = parse_mlir_file(prepared)
    rewrite = rewrite_contractions_to_opu(module, select=select, tile_edge=tile_edge)
    if rewrite.count:
        text = patch_declaration_arg_attrs(to_text(module), rewrite)
        missing = unpatched_declarations(text, rewrite)
        if missing:
            raise RuntimeError(
                f"declarations {list(missing)} carry no bufferization.access attributes, so "
                "one-shot-bufferize would copy the weight operand of every contraction routed to them; "
                "refusing to write the module")
        prepared.write_text(text, encoding="utf-8")
    work.mkdir(parents=True, exist_ok=True)
    (work / SIDECAR_NAME).write_text(json.dumps(rewrite.to_dict(), indent=2), encoding="utf-8")
    return rewrite


def load_sidecar(work: "str | Path") -> dict[str, tuple[int, int, int]]:
    """``{symbol: (m, n, k)}`` as recorded beside a prepared module, or ``{}`` when nothing was routed.

    An absent sidecar means the rewrite never ran, which is the same thing as nothing routed as far as
    the build is concerned — but a MALFORMED one is an error, because it means the rewrite ran and the
    build would otherwise emit a translation unit missing the symbols the module calls.
    """
    path = Path(work) / SIDECAR_NAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    sigs = payload.get("signatures", {})
    if not isinstance(sigs, dict):
        raise ValueError(f"{path} records no usable `signatures` map")
    out: dict[str, tuple[int, int, int]] = {}
    for sym, extents in sigs.items():
        if not isinstance(extents, (list, tuple)) or len(extents) not in (3, 4):
            raise ValueError(f"{path}: signature {sym!r} is not an (m, n, k) triple or a "
                             f"(b, m, n, k) quad: {extents!r}")
        out[str(sym)] = tuple(int(e) for e in extents)
    return out


def _signature_types(module, key: tuple[int, ...]):
    """The three tensor types for one signature, built from the element types this path computes.

    A leading batch extent is carried through onto all three operands: the callee takes the whole batch
    and loops over it, so its type is rank-3 throughout rather than a rank-2 type called several times.
    """
    from xdsl.dialects.builtin import IntegerType, TensorType

    *batch, m, n, k = (int(v) for v in key)
    i8 = IntegerType(8)
    i32 = IntegerType(32)
    return (TensorType(i8, [*batch, m, k]),
            TensorType(i8, [*batch, k, n]),
            TensorType(i32, [*batch, m, n]))
