"""Structurally outline selected signed-int8 contractions into external calls.

The callee computes i8 x i8 -> i32 with row-major (M,K), (K,N) operands,
overwriting a zero-initialized output. Rank-3 calls carry a leading batch extent.
The caller owns selection, symbol naming, sidecar naming and kernel implementation;
this module neither selects hardware nor generates a target kernel. No selector
means no mutation. Nonzero or unknown initial accumulators are refused.

A contraction in any other layout -- a transposed operand, several batch, free or
reduced dims, a windowed (convolution) input -- is read as a GEMM by
:mod:`contraction_normal_form` and rewritten into the callee's layout with explicit
transposes, collapses and an im2col gather, so every kernel implementing the ABI
takes it unchanged. A selector with ``layouts = False`` keeps the historical
canonical-only legality. A selector with a ``decide(shape, layout_elements=)`` method
(:class:`matrix_route_cost.CostModelSelector`) returns a priced decision, including
whether to swap the operands' roles; that decision is recorded per contraction.

File rewriting records the exact called signatures and repairs printer-dropped
bufferization access attributes before writing the prepared module. Kernel builders
consume that same sidecar rather than reconstructing the set of required symbols.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.kernels.shapes import zero_initialised as zero_initialised

__all__ = [
    "ContractionRewrite",
    "NotRouted",
    "Routed",
    "load_sidecar",
    "rewrite_contractions",
    "rewrite_prepared_file",
    "routable_contractions",
    "tile_filling_selector",
    "zero_initialised",
]

#: Signed-int8 inputs with int32 accumulation; this is an ABI, not a hardware fact.
INT8_DTYPES: tuple[str, str, str] = ("i8", "i8", "i32")

#: Named contraction classes and their parallel ranks. A batch is not a tile axis.
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
    #: Where in the model it sits (``conv`` / ``attention`` / ``linear`` ...), from provenance; report only.
    site: str = ""
    #: ``as_is`` (C = A.B) or ``swapped`` (C^T = B^T.A^T): the extents above are the KERNEL's either way.
    orientation: str = "as_is"
    #: Data movement added to reach the kernel layout (``transpose_b``, ``im2col_a``, ...); empty = none.
    steps: tuple[str, ...] = ()
    #: The selector's priced decision, when it gave one (see :mod:`matrix_route_cost`).
    decision: dict | None = None

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
class NotRouted:
    """One contraction left on the host, and why."""

    op: str
    parallel: tuple[int, ...]
    reduction: tuple[int, ...]
    why: str
    fqn: str = ""
    site: str = ""
    decision: dict | None = None

    @property
    def macs(self) -> int:
        out = 1
        for extent in (*self.parallel, *self.reduction):
            out *= int(extent)
        return out


@dataclass(frozen=True)
class ContractionRewrite:
    """What the rewrite did. ``signatures`` is what the C side must define."""

    routed: tuple[Routed, ...] = ()
    #: ``{symbol: (m, n, k)}``, or ``(b, m, n, k)`` for a batched one — one entry per DISTINCT type
    #: signature actually called. The arity is what tells the C emitter which entry shape to generate.
    signatures: dict[str, tuple[int, ...]] = field(default_factory=dict)
    skipped: tuple[tuple[str, str], ...] = ()  # (what, why)
    #: Every observed contraction that was NOT routed, with the reason: illegal for this ABI (dtype,
    #: layout, accumulator) or declined by the selector (e.g. smaller than one tile). This is what tells a
    #: user up front how much of a model the unit will see; it is a report, not an error.
    not_routed: tuple[NotRouted, ...] = ()
    #: The tile edge the decision was made against, when the caller supplied it. Without it the
    #: sidecar cannot say whether a routed contraction fills a tile, so the field is recorded as
    #: UNKNOWN rather than assumed — an unrecorded rule is exactly what this class exists to prevent.
    tile_edge: int | None = None
    #: The selector's own account of its rule (``describe()``), when it has one: the policy name and
    #: the geometry its estimates were derived from. ``None`` for a plain ``shape -> bool`` selector.
    policy: dict | None = None

    @property
    def count(self) -> int:
        return len(self.routed)

    def sub_tile(self) -> tuple[Routed, ...]:
        """Routed contractions that do NOT fill a tile in both parallel dimensions.

        This is the default :func:`tile_filling_selector` rule, evaluated after the fact. A custom
        selector may legitimately route these — but the artifact must SAY that it did.

        """
        if self.tile_edge is None:
            return ()
        e = int(self.tile_edge)
        return tuple(r for r in self.routed if min(r.m, r.n) < e)

    def unexplained_sub_tile(self) -> tuple[Routed, ...]:
        """Sub-tile routes that carry NO recorded decision: the ones a build log must warn about.

        A priced decision (see :mod:`matrix_route_cost`) is the explanation the warning exists to force:
        it states the padding, the utilisation and the estimate that justified the route.
        """
        return tuple(r for r in self.sub_tile() if not r.decision)

    def to_dict(self) -> dict[str, Any]:
        sub = self.sub_tile()
        return {
            "count": self.count,
            "routed": [
                {
                    "symbol": r.symbol,
                    "b": r.batch,
                    "m": r.m,
                    "n": r.n,
                    "k": r.k,
                    "fqn": r.fqn,
                    "site": r.site,
                    "orientation": r.orientation,
                    "steps": list(r.steps),
                    **({"decision": r.decision} if r.decision else {}),
                }
                for r in self.routed
            ],
            "signatures": {k: list(v) for k, v in self.signatures.items()},
            "skipped": [{"what": w, "why": y} for w, y in self.skipped],
            "not_routed": [
                {
                    "op": r.op,
                    "parallel": list(r.parallel),
                    "k": list(r.reduction),
                    "why": r.why,
                    "fqn": r.fqn,
                    "site": r.site,
                    "macs": r.macs,
                    **({"decision": r.decision} if r.decision else {}),
                }
                for r in self.not_routed
            ],
            "macs": {
                "routed": sum(r.batch * r.m * r.n * r.k for r in self.routed),
                "not_routed": sum(r.macs for r in self.not_routed),
            },
            "routing_rule": {
                "tile_edge": self.tile_edge if self.tile_edge is not None else "UNKNOWN",
                "policy": self.policy if self.policy is not None else "selector without a description",
                "fills_default_tile_rule": (None if self.tile_edge is None else not sub),
                "sub_tile_routed": len(sub),
                "sub_tile_unexplained": len(self.unexplained_sub_tile()),
                "sub_tile_dims": sorted(
                    {
                        d
                        for r in sub
                        for d in (("m",) if r.m < int(self.tile_edge or 0) else ())
                        + (("n",) if r.n < int(self.tile_edge or 0) else ())
                    }
                ),
                "sub_tile_signatures": sorted({r.symbol for r in sub}),
            },
        }


def routable_contractions(
    module, *, device: str | None = None, dtypes: tuple[str, str, str] | None = None
) -> list[tuple[Any, Any]]:
    """``[(op, shape)]`` for every contraction this path COULD take, with no decision made.

    Separated from the rewrite so a caller (a cost model, an e-graph, a report) can enumerate the
    candidate set without mutating anything — the same split
    :func:`routing.route_candidates` makes for the same reason.

    "Could" means LEGAL, not profitable: a rank-2 or batched matmul accumulating into a zero
    init. Whether a legal contraction is worth moving is the ``select`` caller's decision.

    Candidate enumeration may use explicit ``dtypes`` or device-derived triples.
    This does not widen the rewrite ABI: ``rewrite_contractions`` always selects
    the signed-int8/int32 candidate set.
    """
    want = tuple(dtypes) if dtypes else INT8_DTYPES
    if device is not None:
        from merlin.system.offload import device_dtype_triples

        derived = device_dtype_triples(device)
        if not derived:
            return []  # fail closed: an underivable datapath routes nothing
        accepted = set(derived)
    else:
        accepted = {want}

    return [(e.op, e.shape) for e in _census(module, accepted) if e.why is None]


@dataclass(frozen=True)
class _Entry:
    """One contraction in the census: the kernel's view of it (when legal) or the reason it has none."""

    op: Any
    shape: Any  # the kernel view (as_is) when legal; otherwise the extents the op reads, for the report
    why: str | None
    nf: Any = None  # its contraction_normal_form.NormalForm, when it has one
    site: str = ""


def _census(module, accepted: set[tuple[str, ...]], *, layouts: bool = True) -> list[_Entry]:
    """Every contraction, in program order, with ``why=None`` (legal for this ABI) or the reason it is not.

    "Every" is the observer's named and canonical-generic contractions PLUS any multiply-accumulate
    generic it does not classify (a transposed left operand, two batch dims, a windowed input): those
    used to be absent from the report altogether, so their MACs were missing from the denominator too.
    """
    from ..kernels.shapes import observe_contractions

    observed = {id(op): shape for op, shape in observe_contractions(module)}
    out: list[_Entry] = []
    for op in module.walk():
        if id(op) in observed or _is_mac_generic(op):
            out.append(_entry(op, observed.get(id(op)), accepted, layouts))
    return out


def _is_mac_generic(op) -> bool:
    """A two-input ``linalg.generic`` with a reduction iterator and a multiply-add body."""
    from ..kernels.shapes import _iterator_types

    if getattr(op, "name", "") != "linalg.generic" or len(getattr(op, "operands", ())) != 3:
        return False
    its = _iterator_types(op) or []
    if not any("reduction" in it for it in its) or not op.regions or not op.regions[0].blocks:
        return False
    names = {o.name for o in op.regions[0].blocks[0].ops}
    return {"arith.muli", "arith.addi"} <= names or {"arith.mulf", "arith.addf"} <= names


def _entry(op, shape, accepted: set[tuple[str, ...]], layouts: bool) -> _Entry:
    from . import contraction_normal_form as cnf

    site = cnf.site_of(op)
    report_shape = shape if shape is not None else _extent_shape(op)
    dtypes = tuple(shape.dtypes) if shape is not None and shape.dtypes else tuple(report_shape.dtypes)
    if dtypes not in accepted:
        return _Entry(op, report_shape, f"dtypes {dtypes} are not one of {sorted(accepted)}", site=site)
    if not zero_initialised(op):
        return _Entry(op, report_shape, "accumulator is not provably zero-initialised", site=site)
    if dtypes != INT8_DTYPES:
        # Another datapath enumerating candidates: the normal form emits int8 only, so the
        # canonical-layout legality is the whole of it.
        why = _refusal(op, shape, accepted) if shape is not None else "not a canonical matmul or batch_matmul"
        return _Entry(op, report_shape, why, site=site)
    nf = cnf.analyze(op)
    if isinstance(nf, str):
        return _Entry(op, report_shape, nf, site=site)
    if nf.steps(cnf.AS_IS) and not layouts:
        why = (_refusal(op, shape, accepted) if shape is not None else None) or (
            f"needs layout steps {list(nf.steps(cnf.AS_IS))}, which this selector does not apply"
        )
        return _Entry(op, report_shape, why, nf, site)
    return _Entry(op, nf.shape(cnf.AS_IS), None, nf, site)


def _extent_shape(op):
    """``ContractionShape`` of an op the observer does not classify: its output extents, and the extents
    of its reduction dims where an operand projects them (so the report can count its MACs)."""
    from ..kernels.microkernel import ContractionShape
    from ..kernels.shapes import _shaped
    from .contraction_normal_form import _dim_of, _iterators, _maps

    shaped = [_shaped(v) for v in op.operands]
    out = shaped[-1][0] if shaped and shaped[-1] else [1]
    dtypes = tuple(s[1] for s in shaped) if all(shaped) else ()
    maps, its = _maps(op), _iterators(op)
    red: list[int] = []
    if maps is not None and its is not None:
        for d, it in enumerate(its):
            if it != "reduction":
                continue
            for (shape, _dt), exprs in zip([s for s in shaped if s], maps):
                hit = [axis for axis, e in enumerate(exprs) if _dim_of(e) == d]
                if hit:
                    red.append(int(shape[hit[0]]))
                    break
    return ContractionShape(
        op=getattr(op, "name", "?"), parallel=tuple(int(x) for x in out), reduction=tuple(red), dtypes=dtypes
    )


def _refusal(op, shape, accepted: set[tuple[str, ...]]) -> str | None:
    want_parallel = ROUTABLE_OPS.get(shape.op)
    if want_parallel is None:
        return f"{shape.op} is not a routable contraction class"
    if tuple(shape.dtypes) not in accepted:
        return f"dtypes {tuple(shape.dtypes)} are not one of {sorted(accepted)}"
    if len(shape.parallel) != want_parallel or len(shape.reduction) != 1:
        return f"rank: {len(shape.parallel)} parallel / {len(shape.reduction)} reduction dims"
    if not zero_initialised(op):
        return "accumulator is not provably zero-initialised"
    return _layout_refusal(op, shape)


def _layout_refusal(op, shape) -> str | None:
    """None when the operands are exactly A(batch.., M, K) . B(batch.., K, N) -> C(batch.., M, N).

    The callee is typed from the shape key in that layout, so a contraction whose B is stored (N, K) --
    a transposed weight, legal linalg -- would otherwise be handed to it as if it were (K, N): a type
    error when N != K, and a silently wrong answer when N == K. Checked from the operand TYPES and, when
    the op carries them, the indexing MAPS; unreadable maps on a generic fail closed.
    """
    from ..kernels.shapes import _dim_position, _shaped, indexing_maps

    *batch, m, n = (int(d) for d in shape.parallel)
    k = int(shape.reduction[0])
    shaped = [s for s in (_shaped(v) for v in op.operands) if s is not None]
    if len(shaped) < 2:
        return "operand types are not readable"
    if list(shaped[0][0]) != [*batch, m, k] or list(shaped[1][0]) != [*batch, k, n]:
        return f"operand layout {list(shaped[0][0])} x {list(shaped[1][0])} is not A(M,K) x B(K,N)"
    maps = indexing_maps(op)
    if maps is None:
        if shape.op in ROUTABLE_OPS and getattr(op, "name", "") == shape.op:
            return None  # a named op without explicit maps has the default (row-major) layout
        return "indexing maps are not readable"
    nb = len(batch)
    dims = list(range(nb))
    expected = ([*dims, nb, nb + 2], [*dims, nb + 2, nb + 1], [*dims, nb, nb + 1])
    got = [[_dim_position(r) for r in results] for results in maps[:3]]
    if got != [list(e) for e in expected]:
        return f"indexing maps {got} are not A(m,k) x B(k,n) -> C(m,n) (a transposed operand)"
    return None


def _signature_key(shape) -> tuple[int, ...]:
    """``(M, N, K)``, or ``(B, M, N, K)`` when the contraction carries a batch dim.

    The key IS the callee's identity: MLIR function types are monomorphic, so two contractions share a
    symbol only if every extent agrees — and a batched one cannot share with a rank-2 one even at the
    same M/N/K, because the descriptors it is passed have a different arity.
    """
    return (*(int(d) for d in shape.parallel), int(shape.reduction[0]))


def rewrite_contractions(
    module,
    *,
    select: Callable[[Any], bool] | None = None,
    tile_edge: int | None = None,
    symbol_prefix: str,
) -> ContractionRewrite:
    """Replace each selected int8 contraction with a call to the caller-provided kernel.

    Mutates ``module`` in place and returns what it did. ``select`` receives the kernel's view of each
    legal contraction as a :class:`~merlin.kernels.microkernel.ContractionShape` and returns whether it
    should move; ``None`` moves nothing, so the pass is inert unless a decision has been made elsewhere.

    A selector may also carry ``decide(shape, layout_elements=)`` returning a priced decision with an
    orientation (:class:`matrix_route_cost.RouteDecision`) -- then that decision, and not ``__call__``, is
    what routes, and it is recorded -- ``layouts = False`` to keep the canonical-only legality, and
    ``describe()`` for the sidecar's ``routing_rule.policy``.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Block, Region

    from . import contraction_normal_form as cnf

    if select is None:
        return ContractionRewrite(skipped=(("all", "no selector supplied, so nothing is routed"),), tile_edge=tile_edge)

    layouts = bool(getattr(select, "layouts", True))
    decide = getattr(select, "decide", None)
    describe = getattr(select, "describe", None)
    policy = describe() if callable(describe) else None
    census = _census(module, {INT8_DTYPES}, layouts=layouts)
    candidates = [e for e in census if e.why is None]
    chosen: list[tuple[_Entry, str, dict | None]] = []
    declined: dict[int, tuple[str, dict]] = {}
    for e in candidates:
        if callable(decide):
            offer = {cnf.AS_IS: e.nf.layout_elements(cnf.AS_IS)}
            if layouts:
                offer[cnf.SWAPPED] = e.nf.layout_elements(cnf.SWAPPED)
            d = decide(e.shape, layout_elements=offer)
            if d.route:
                chosen.append((e, d.orientation, d.to_dict()))
            else:
                declined[id(e.op)] = (d.reason, d.to_dict())
        elif select(e.shape):
            chosen.append((e, cnf.AS_IS, None))
    chosen_ids = {id(e.op) for e, _o, _d in chosen}
    not_routed = tuple(
        NotRouted(
            op=e.shape.op,
            parallel=tuple(int(d) for d in e.shape.parallel),
            reduction=tuple(int(d) for d in e.shape.reduction),
            why=e.why or declined.get(id(e.op), (None,))[0] or _declined(e.shape, tile_edge),
            fqn=_fqn(e.op),
            site=e.site,
            decision=declined.get(id(e.op), (None, None))[1],
        )
        for e in census
        if id(e.op) not in chosen_ids
    )
    skipped: list[tuple[str, str]] = []
    if not chosen:
        skipped.append(("all", f"{len(candidates)} routable contraction(s), none selected"))
        return ContractionRewrite(skipped=tuple(skipped), not_routed=not_routed, tile_edge=tile_edge, policy=policy)

    # One symbol per distinct signature: MLIR function types are monomorphic, so a 256x196/K=768
    # contraction and a 196x1024/K=256 one cannot share a callee.
    symbols: dict[tuple[int, ...], str] = {}
    routed: list[Routed] = []

    for e, orientation, decision in chosen:
        op, shape = e.op, e.nf.shape(orientation)
        key = _signature_key(shape)
        sym = symbols.get(key)
        if sym is None:
            sym = f"{symbol_prefix}_{len(symbols)}"
            symbols[key] = sym
        fqn, steps = _fqn(op), e.nf.steps(orientation)

        if steps:
            # Not in the kernel's layout: transposes / collapses / an im2col gather around the call.
            cnf.emit(e.nf, orientation, lambda a, b, c, t, _s=sym: func.CallOp(_s, [a, b, c], [t]))
        else:
            call = func.CallOp(sym, list(op.operands), [op.results[0].type])
            op.results[0].replace_all_uses_with(call.results[0])
            parent = op.parent
            parent.insert_op_before(call, op)
            op.detach()
            op.erase()

        routed.append(
            Routed(
                symbol=sym,
                parallel=tuple(shape.parallel),
                reduction=tuple(shape.reduction),
                fqn=fqn,
                site=e.site,
                orientation=orientation,
                steps=tuple(steps),
                decision=decision,
            )
        )

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
        decl = func.FuncOp(
            sym,
            ((lhs_t, rhs_t, out_t), (out_t,)),
            Region(),
            visibility="private",
            arg_attrs=ArrayAttr([read, read, write]),
        )
        body.add_op(decl)

    return ContractionRewrite(
        routed=tuple(routed),
        signatures={s: k for k, s in symbols.items()},
        skipped=tuple(skipped),
        not_routed=not_routed,
        tile_edge=tile_edge,
        policy=policy,
    )


def _fqn(op) -> str:
    from xdsl.dialects.builtin import StringAttr

    prov = getattr(op, "attributes", {}).get("prov.fqn") if hasattr(op, "attributes") else None
    return prov.data if isinstance(prov, StringAttr) else ""


#: Reasons that read ``<class>: <this op's numbers>``; a report groups them by the class alone.
GROUPED_REASONS = ("smaller than one tile", "declined by the selector", "cost model")


def _declined(shape, tile_edge: int | None) -> str:
    m, n = int(shape.parallel[-2]), int(shape.parallel[-1])
    if tile_edge is not None and min(m, n) < int(tile_edge):
        return f"smaller than one tile ({int(tile_edge)}): min(M={m}, N={n})"
    return "declined by the selector"


#: This path's positional callee access policy, passed to shared declaration repair.
_ARG_ACCESS = ("read", "read", "write")


def tile_filling_selector(tile_edge: int) -> Callable[[Any], bool]:
    """Select contractions whose output is at least one whole tile in both parallel dimensions.

    The caller supplies the threshold from its routing policy; no hardware geometry
    or default tile size is inferred here.
    """
    edge = int(tile_edge)
    if edge < 1:
        raise ValueError(
            f"tile_edge={tile_edge} is not a lane count; it comes from the hardware's own "
            "vector length and a guessed one selects the wrong contractions"
        )

    def select(shape) -> bool:
        # The LAST TWO parallel extents are the tile's, whatever the rank: a batch dim in front is a loop
        # over slices and says nothing about how full a tile each slice makes.
        return min(int(shape.parallel[-2]), int(shape.parallel[-1])) >= edge

    return select


def rewrite_prepared_file(
    prepared: str | Path,
    work: str | Path,
    *,
    select: Callable[[Any], bool] | None,
    tile_edge: int | None = None,
    symbol_prefix: str,
    sidecar_name: str,
) -> ContractionRewrite:
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
    from .declaration_access import patch_declaration_arg_attrs, unpatched_declarations

    prepared, work = Path(prepared), Path(work)
    module = parse_mlir_file(prepared)
    rewrite = rewrite_contractions(module, select=select, tile_edge=tile_edge, symbol_prefix=symbol_prefix)
    if rewrite.count:
        text = patch_declaration_arg_attrs(to_text(module), rewrite.signatures, argument_access=_ARG_ACCESS)
        missing = unpatched_declarations(text, rewrite.signatures)
        if missing:
            raise RuntimeError(
                f"declarations {list(missing)} carry no bufferization.access attributes, so "
                "one-shot-bufferize would copy the weight operand of every contraction routed to them; "
                "refusing to write the module"
            )
        prepared.write_text(text, encoding="utf-8")
    work.mkdir(parents=True, exist_ok=True)
    (work / sidecar_name).write_text(json.dumps(rewrite.to_dict(), indent=2), encoding="utf-8")
    return rewrite


def load_sidecar(work: str | Path, sidecar_name: str) -> dict[str, tuple[int, ...]]:
    """``{symbol: (m, n, k)}`` as recorded beside a prepared module, or ``{}`` when nothing was routed.

    An absent sidecar means the rewrite never ran, which is the same thing as nothing routed as far as
    the build is concerned — but a MALFORMED one is an error, because it means the rewrite ran and the
    build would otherwise emit a translation unit missing the symbols the module calls.
    """
    path = Path(work) / sidecar_name
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    sigs = payload.get("signatures", {})
    if not isinstance(sigs, dict):
        raise ValueError(f"{path} records no usable `signatures` map")
    out: dict[str, tuple[int, ...]] = {}
    for sym, extents in sigs.items():
        if not isinstance(extents, (list, tuple)) or len(extents) not in (3, 4):
            raise ValueError(
                f"{path}: signature {sym!r} is not an (m, n, k) triple or a (b, m, n, k) quad: {extents!r}"
            )
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
    return (TensorType(i8, [*batch, m, k]), TensorType(i8, [*batch, k, n]), TensorType(i32, [*batch, m, n]))
