"""Structurally outline selected signed-int8 contractions into external calls.

The callee computes i8 x i8 -> i32 with row-major (M,K), (K,N) operands,
overwriting a zero-initialized output. Rank-3 calls carry a leading batch extent.
The caller owns selection, symbol naming, sidecar naming and kernel implementation;
this module neither selects hardware nor generates a target kernel. No selector
means no mutation. Nonzero or unknown initial accumulators are refused.

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
class ContractionRewrite:
    """What the rewrite did. ``signatures`` is what the C side must define."""

    routed: tuple[Routed, ...] = ()
    #: ``{symbol: (m, n, k)}``, or ``(b, m, n, k)`` for a batched one — one entry per DISTINCT type
    #: signature actually called. The arity is what tells the C emitter which entry shape to generate.
    signatures: dict[str, tuple[int, ...]] = field(default_factory=dict)
    skipped: tuple[tuple[str, str], ...] = ()  # (what, why)
    #: The tile edge the decision was made against, when the caller supplied it. Without it the
    #: sidecar cannot say whether a routed contraction fills a tile, so the field is recorded as
    #: UNKNOWN rather than assumed — an unrecorded rule is exactly what this class exists to prevent.
    tile_edge: int | None = None

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

    def to_dict(self) -> dict[str, Any]:
        sub = self.sub_tile()
        return {
            "count": self.count,
            "routed": [
                {"symbol": r.symbol, "b": r.batch, "m": r.m, "n": r.n, "k": r.k, "fqn": r.fqn} for r in self.routed
            ],
            "signatures": {k: list(v) for k, v in self.signatures.items()},
            "skipped": [{"what": w, "why": y} for w, y in self.skipped],
            "routing_rule": {
                "tile_edge": self.tile_edge if self.tile_edge is not None else "UNKNOWN",
                "fills_default_tile_rule": (None if self.tile_edge is None else not sub),
                "sub_tile_routed": len(sub),
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
    from ..kernels.shapes import observe_contractions

    want = tuple(dtypes) if dtypes else INT8_DTYPES
    if device is not None:
        from merlin.system.offload import device_dtype_triples

        derived = device_dtype_triples(device)
        if not derived:
            return []  # fail closed: an underivable datapath routes nothing
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


def rewrite_contractions(
    module,
    *,
    select: Callable[[Any], bool] | None = None,
    tile_edge: int | None = None,
    symbol_prefix: str,
) -> ContractionRewrite:
    """Replace each selected int8 contraction with a call to the caller-provided kernel.

    Mutates ``module`` in place and returns what it did. ``select`` receives the
    :class:`~merlin.kernels.microkernel.ContractionShape` and returns whether that contraction should
    move; ``None`` moves nothing, so the pass is inert unless a decision has been made elsewhere.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Block, Region

    if select is None:
        return ContractionRewrite(skipped=(("all", "no selector supplied, so nothing is routed"),), tile_edge=tile_edge)

    candidates = routable_contractions(module)
    chosen = [(op, sh) for op, sh in candidates if select(sh)]
    skipped: list[tuple[str, str]] = []
    if not chosen:
        skipped.append(("all", f"{len(candidates)} routable contraction(s), none selected"))
        return ContractionRewrite(skipped=tuple(skipped), tile_edge=tile_edge)

    # One symbol per distinct signature: MLIR function types are monomorphic, so a 256x196/K=768
    # contraction and a 196x1024/K=256 one cannot share a callee.
    symbols: dict[tuple[int, ...], str] = {}
    routed: list[Routed] = []

    for op, shape in chosen:
        key = _signature_key(shape)
        sym = symbols.get(key)
        if sym is None:
            sym = f"{symbol_prefix}_{len(symbols)}"
            symbols[key] = sym

        operands = list(op.operands)
        if len(operands) != 3 or len(op.results) != 1:
            # A contraction whose operand count is not (lhs, rhs, out-init) is not the shape this
            # callee promises; skip it rather than emit a call with the wrong arity.
            skipped.append((sym, f"expected 3 operands and 1 result, got {len(operands)} and {len(op.results)}"))
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
        routed.append(Routed(symbol=sym, parallel=tuple(shape.parallel), reduction=tuple(shape.reduction), fqn=fqn))

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
        routed=tuple(routed), signatures={s: k for k, s in symbols.items()}, skipped=tuple(skipped), tile_edge=tile_edge
    )


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
