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

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from merlin.compiler.ir.contraction_shapes import zero_initialised

# ---- int8_contractions -------------------------------------------------------------------------

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
    from merlin.compiler.ir.contraction_shapes import observe_contractions

    want = tuple(dtypes) if dtypes else INT8_DTYPES
    if device is not None:
        from merlin.target.system import device_dtype_triples

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
    :class:`~merlin.analysis.mining.microkernel.ContractionShape` and returns whether that contraction should
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
    from merlin.compiler.dialects.xdsl_context import text as to_text
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_file

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


# ---- declaration_access ------------------------------------------------------------------------
# Repair access attributes on printer-emitted private tensor declarations.
#
# Callers provide exact symbols and positional access policy. This module knows no
# accelerator, dtype, rewrite record or target ABI. It retains the existing simple
# printed-declaration format; it is not a general MLIR parser or ABI verifier.


def patch_declaration_arg_attrs(text: str, symbols: Iterable[str], *, argument_access: Sequence[str]) -> str:
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
    for sym in symbols:
        head = f"func.func private @{sym}("
        at = out.find(head)
        if at < 0:
            continue
        open_paren = at + len(head) - 1
        close = out.find(")", open_paren)
        if close < 0:
            continue
        inner = out[open_paren + 1 : close]
        if "bufferization.access" in inner:
            continue  # already annotated; re-splitting would double every attribute
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) != len(argument_access):
            continue
        annotated = ", ".join(
            f'{p} {{bufferization.access = "{acc}"}}' for p, acc in zip(parts, argument_access, strict=True)
        )
        out = out[: open_paren + 1] + annotated + out[close:]
    return out


def unpatched_declarations(text: str, symbols: Iterable[str]) -> tuple[str, ...]:
    """Symbols whose declaration in ``text`` still carries no access attributes.

    Non-empty means the weight operands of those callees will be copied by bufferization, so a caller
    that cares about it can fail rather than ship the copies.
    """
    missing = []
    generic_access = None
    for sym in symbols:
        at = text.find(f"func.func private @{sym}(")
        if at < 0:
            if generic_access is None:
                generic_access = {}
                if '"func.func"' in text:
                    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

                    module = parse_mlir_text(text)
                    for op in module.walk():
                        if op.name == "func.func" and not op.body.blocks:
                            attrs = op.arg_attrs
                            generic_access[op.sym_name.data] = bool(attrs) and all(
                                "bufferization.access" in attr.data for attr in attrs
                            )
            if generic_access.get(sym):
                continue
            missing.append(sym)
            continue
        line_end = text.find("\n", at)
        line = text[at : line_end if line_end > 0 else len(text)]
        if "bufferization.access" not in line:
            missing.append(sym)
    return tuple(missing)


# ---- quantized_sum_route -----------------------------------------------------------------------
# Explicit current-source Q1 producer/sum grouping for ordinary logical shims.
#
# Both original entries and numerical programs travel intact. Target code owns
# its temporary and device effects; grouping supplies neither qualification nor
# an allocation. A shared residual callee retains all unrelated call instances.


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class QuantizedSumRoute:
    source_pairs: tuple[tuple[str, str], ...]

    def record(self):
        if (
            type(self.source_pairs) is not tuple
            or not self.source_pairs
            or len(self.source_pairs) > 128
            or any(
                type(p) is not tuple or len(p) != 2 or any(type(s) is not str or not s for s in p) or p[0] == p[1]
                for p in self.source_pairs
            )
            or len({p[0] for p in self.source_pairs}) != len(self.source_pairs)
        ):
            raise ValueError("quantized sum needs distinct explicit original producer/consumer source pairs")
        return dict(schema="merlin.quantized_sum_route.v1", source_pairs=[list(p) for p in self.source_pairs])


def freeze_route(device):
    selected = getattr(device, "quantized_sum", None)
    if selected is None:
        return None
    if (
        type(selected) is not QuantizedSumRoute
        or getattr(device, "granularity", None) != "group"
        or any(
            getattr(device, k, None) is not None
            for k in (
                "connected_spatial",
                "sibling_source",
                "exact_selection",
                "catalog_manifest",
                "catalog_object",
                "catalog_builder",
                "prepared_transform",
                "post_offload_transform",
                "offload_selection",
            )
        )
    ):
        raise ValueError("quantized sum requires its original typed group route without opaque source/ABI rewrites")
    return selected, selected.record()


def require_route(device, frozen):
    current = freeze_route(device)
    if (
        (current is None) != (frozen is None)
        or current is not None
        and (current[0] is not frozen[0] or current[1] != frozen[1])
    ):
        raise ValueError("quantized sum original source selection changed")
    return None if current is None else current[0]


@dataclass(frozen=True)
class QuantizedSumSelection:
    module: object
    rewrite: object
    receipt: dict


def group_quantized_sum(module, rewrite, route):
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, UnitAttr
    from xdsl.rewriter import Rewriter

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.lower.device.offload import DeviceRewrite, Routed

    if type(rewrite) is not DeviceRewrite or type(route) is not QuantizedSumRoute:
        raise ValueError("original ordinary source rewrite and quantized sum route required")
    route.record()
    module.verify()
    before = text(module, generic=True)
    selected = module.clone()
    maps = {
        name: deepcopy(getattr(rewrite, name))
        for name in ("signatures", "programs", "entries", "arg_access", "call_buffers", "expected_interfaces")
    }
    records = []
    added = []
    removed = set()
    remaining_routed = list(rewrite.routed)
    for ps, cs in route.source_pairs:
        calls = [o for o in selected.walk() if isinstance(o, func.CallOp)]
        producers = [o for o in calls if o.callee.root_reference.data == ps]
        if len(producers) != 1:
            raise ValueError("selected quantized producer must name one current source call")
        producer = producers[0]
        uses = list(producer.res[0].uses)
        if (
            len(uses) != 1
            or not isinstance(uses[0].operation, func.CallOp)
            or uses[0].operation.callee.root_reference.data != cs
        ):
            raise ValueError("selected quantized producer lost its exact original sum consumer")
        consumer = uses[0].operation
        originals = []
        for call, symbol in ((producer, ps), (consumer, cs)):
            current_calls = [o for o in calls if o.callee.root_reference.data == symbol]
            inventory = [r for r in remaining_routed if r.symbol == symbol]
            if len(inventory) != len(current_calls):
                raise ValueError("quantized routing records do not cover every current call instance")
            originals.append(inventory[current_calls.index(call)])
        for row in originals:
            remaining_routed.remove(row)
        source = plan_quantized_sum_source(selected, rewrite, producer, consumer)
        source.revalidate()
        if ps in removed or cs in removed:
            raise ValueError("quantized source pairs overlap")
        pe, ce = source.receipt["entries"]
        side = source.receipt["producer_result_operand"]
        skip = 1 - side
        entry = dict(
            schema="connected_quantized_sum_entry_v1",
            kind="model_slice",
            op="quantized_producer_sum",
            operand_dtype="int8",
            source_role=pe.get("source_role", "model_derived"),
            source_reference=pe.get("source_reference", "explicit original quantized call edge"),
            components=deepcopy([pe, ce]),
            producer_result_operand=side,
        )
        binding = _sha(json.dumps(dict(entry=entry, source=source.receipt), sort_keys=True))
        symbol = "merlin_quantized_sum_" + binding[:16]
        if any(isinstance(o, func.FuncOp) and o.sym_name.data == symbol for o in selected.body.block.ops):
            raise ValueError("quantized source group symbol collision")
        operands = [*producer.arguments[:-1], consumer.arguments[skip], consumer.arguments[-1]]
        access = (*rewrite.arg_access[ps][:-1], "read", "write")
        declaration = func.FuncOp.external(symbol, [v.type for v in operands], [consumer.res[0].type])
        declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
        declaration.properties["arg_attrs"] = ArrayAttr(
            [DictionaryAttr({"bufferization.access": StringAttr(a)}) for a in access]
        )
        selected.body.block.add_op(declaration)
        replacement = func.CallOp(symbol, operands, [consumer.res[0].type])
        Rewriter.replace_op(consumer, replacement)
        destination = producer.arguments[-1].owner
        Rewriter.erase_op(producer)
        Rewriter.erase_op(destination)
        entry["name"] = symbol
        maps["signatures"][symbol] = ()
        maps["entries"][symbol] = entry
        maps["programs"][symbol] = dict(
            schema="connected_quantized_sum_program_v1",
            components=deepcopy([rewrite.programs[ps], rewrite.programs[cs]]),
            producer_result_operand=side,
        )
        maps["arg_access"][symbol] = access
        maps["call_buffers"][symbol] = [
            *deepcopy(rewrite.call_buffers[ps][:-1]),
            deepcopy(rewrite.call_buffers[cs][skip]),
            deepcopy(rewrite.call_buffers[cs][-1]),
        ]
        added.append(
            Routed(
                symbol,
                (),
                (),
                ("i8", "i8", "i8"),
                source_node_ids=tuple(n for r in originals for n in r.source_node_ids),
            )
        )
        # Retire declarations only after their last actual current call leaves.
        remaining = {o.callee.root_reference.data for o in selected.walk() if isinstance(o, func.CallOp)}
        for old in (ps, cs):
            if old not in remaining:
                removed.add(old)
                for mapping in maps.values():
                    mapping.pop(old, None)
                decl = next(o for o in selected.body.block.ops if isinstance(o, func.FuncOp) and o.sym_name.data == old)
                Rewriter.erase_op(decl)
        records.append(
            dict(
                symbol=symbol,
                components=[ps, cs],
                source_binding_sha256=binding,
                original_source=source.receipt,
                logical_argument_types=[str(v.type) for v in operands],
                result_argument=len(operands) - 1,
                quantization="original producer Q1, original ordered sum/Q2/activation",
                external_intermediate_observers=[],
            )
        )
    updated = replace(rewrite, **maps, routed=tuple(remaining_routed) + tuple(added))
    selected.verify()
    if text(module, generic=True) != before:
        raise ValueError("quantized grouping changed original source")
    return QuantizedSumSelection(
        selected,
        updated,
        dict(
            schema="merlin.quantized_sum_selection.v1",
            source_host_sha256=_sha(before),
            selected_host_sha256=_sha(text(selected, generic=True)),
            groups=records,
            target_execution=False,
            numerical_evaluation=False,
        ),
    )


def prepare_host(device, route, prepared, rewrite, directory):
    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_file

    selected = group_quantized_sum(parse_mlir_file(prepared), rewrite, route)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "host.mlir"
    path.write_text(text(selected.module, generic=True) + "\n")
    (directory / "receipt.json").write_text(json.dumps(selected.receipt, sort_keys=True, indent=2) + "\n")
    return path, selected.rewrite


def quantized_sum_interface(entry, binder, *, source_reference):
    from merlin.compiler.lower.device.whole_program import group_interface
    from merlin.target.contract.interfaces.grammar import emit_interface_mlir, parse_interface_mlir

    if entry.get("schema") != "connected_quantized_sum_entry_v1" or entry.get("producer_result_operand") not in (0, 1):
        raise ValueError("two unchanged original quantized source entries and operand order required")
    parts = entry.get("components")
    if type(parts) is not list or len(parts) != 2:
        raise ValueError("exactly two original quantized source stages required")
    first, second = [
        deepcopy(parse_interface_mlir(group_interface(e, binder, source_reference=source_reference))) for e in parts
    ]
    if (first["target"], first["abi_version"]) != (second["target"], second["abi_version"]):
        raise ValueError("quantized source target or ABI differs")
    if [c["opcode"] for c in first["commands"]] not in (
        ["RES_PACK", "CONV2D", "EVICT"],
        ["RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT"],
    ) or [c["opcode"] for c in second["commands"]] != ["RESIDUAL_ADD"]:
        raise ValueError("quantized source must retain one original contraction and one loaded sum")
    produced = next(c for c in first["commands"] if c["opcode"] in ("CONV2D", "COMMIT"))
    residual = second["commands"][0]
    edge = produced["operands"]["dst"]
    key = ("lhs", "rhs")[entry["producer_result_operand"]]
    source = residual["operands"][key]
    a, b = first["tensors"][edge], second["tensors"][source]
    if a.get("dtype") != "i8" or any(a.get(k) != b.get(k) for k in ("shape", "dtype")):
        raise ValueError("quantized source edge shape or format differs")
    for prefix, cb in [("producer_", first), ("sum_", second)]:
        cb["tensors"] = {prefix + n: s for n, s in cb["tensors"].items()}
        for c in cb["commands"]:
            c["operands"] = {k: prefix + v for k, v in c["operands"].items()}
            if "bias" in c.get("attributes", {}):
                c["attributes"]["bias"] = prefix + c["attributes"]["bias"]
    second["tensors"].pop("sum_" + source)
    residual["operands"][key] = "producer_" + edge
    return emit_interface_mlir(
        dict(
            abi_version=first["abi_version"],
            target=first["target"],
            tensors={**first["tensors"], **second["tensors"]},
            commands=[*first["commands"], *second["commands"]],
        )
    )


# ---- quantized_sum_source ----------------------------------------------------------------------
# Bind a sole-use quantized producer to its exact following residual call.
#
# The two numerical source entries remain separate and unchanged. This matcher
# owns only the current tensor edge and observers; target implementation, private
# workspace, numerical scalar proof and actual completion remain independent.


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class QuantizedSumSource:
    module: object
    rewrite: object
    producer: object
    consumer: object
    receipt: dict
    identity: str

    def revalidate(self):
        current = plan_quantized_sum_source(self.module, self.rewrite, self.producer, self.consumer)
        if self.identity != current.identity or self.receipt != current.receipt:
            raise ValueError("quantized sum original source, stages or observers changed")


def discover_quantized_sum_sources(module, rewrite):
    """Find legal current producer/residual edges without selecting a backend.

    Results are source-bound proposals. The caller can inspect the unchanged
    numerical stages and shape, try its target adapter, and select only admitted
    alternatives. Shared producer symbols are omitted because the ordinary
    grouping route identifies a producer by its unique current call. Source
    legality supplies no target effects, storage assignment or speedup claim.
    """
    from collections import Counter

    from xdsl.dialects import func

    from merlin.compiler.lower.device.offload import DeviceRewrite

    if type(rewrite) is not DeviceRewrite:
        raise ValueError("quantized discovery requires the current ordinary source rewrite")
    module.verify()
    calls = tuple(op for op in module.walk() if isinstance(op, func.CallOp))
    counts = Counter(op.callee.root_reference.data for op in calls)
    proposals = []
    for producer in calls:
        symbol = producer.callee.root_reference.data
        if (
            counts[symbol] != 1
            or len(producer.res) != 1
            or rewrite.entries.get(symbol, {}).get("op") not in ("conv2d", "matmul")
        ):
            continue
        uses = tuple(producer.res[0].uses)
        if len(uses) != 1 or not isinstance(uses[0].operation, func.CallOp):
            continue
        consumer = uses[0].operation
        if rewrite.entries.get(consumer.callee.root_reference.data, {}).get("op") != "residual_add":
            continue
        try:
            proposal = plan_quantized_sum_source(module, rewrite, producer, consumer)
        except ValueError:
            # A nonmatching source remains on its original unfused route.
            continue
        proposals.append(proposal)
    return tuple(proposals)


def plan_quantized_sum_source(module, rewrite, producer, consumer):
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import TensorType

    from merlin.compiler.dialects.xdsl_context import text
    from merlin.compiler.lower.device.offload import DeviceRewrite

    if (
        type(rewrite) is not DeviceRewrite
        or not isinstance(producer, func.CallOp)
        or not isinstance(consumer, func.CallOp)
    ):
        raise ValueError("quantized sum requires its current original calls and ordinary source rewrite")
    module.verify()
    operations = tuple(module.walk())
    if (
        producer not in operations
        or consumer not in operations
        or producer.parent is not consumer.parent
        or len(producer.res) != 1
        or len(consumer.res) != 1
    ):
        raise ValueError("quantized sum calls must be current same-block single-result stages")
    ps, cs = (op.callee.root_reference.data for op in (producer, consumer))
    pe, ce = (deepcopy(rewrite.entries.get(symbol, {})) for symbol in (ps, cs))
    if (
        pe.get("op") not in ("conv2d", "matmul")
        or ce.get("op") != "residual_add"
        or pe.get("operand_dtype") != "int8"
        or ce.get("operand_dtype") != "int8"
        or pe.get("epilogue")
        not in (["bias_add", "acc_scale"], ["bias_add", "acc_scale", "relu"], ["acc_scale"], ["acc_scale", "relu"])
        or ce.get("epilogue") not in ([], ["relu"])
    ):
        raise ValueError("quantized sum retains an original signed-i8 producer Q1 and residual Q2")
    for symbol, entry in ((ps, pe), (cs, ce)):
        program = rewrite.programs.get(symbol, {})
        if program.get("entry") != {k: v for k, v in entry.items() if k in program.get("entry", {})} or not program.get(
            "entry"
        ):
            raise ValueError("quantized sum numerical stages differ from original source programs")
    access = rewrite.arg_access
    if (
        access.get(ps) not in (("read", "read", "write"), ("read", "read", "read", "write"))
        or access.get(cs) != ("read", "read", "write")
        or len(producer.arguments) != len(access[ps])
        or len(consumer.arguments) != 3
    ):
        raise ValueError("quantized sum requires original borrowed inputs and fresh complete destinations")
    uses = list(producer.res[0].uses)
    if (
        len(uses) != 1
        or uses[0].operation is not consumer
        or uses[0].index not in (0, 1)
        or consumer.arguments[uses[0].index] is not producer.res[0]
    ):
        raise ValueError("quantized producer intermediate has an unaccounted observer or different consumer")
    side = uses[0].index
    skip = consumer.arguments[1 - side]
    output_type = producer.res[0].type
    if (
        not isinstance(output_type, TensorType)
        or str(output_type.element_type) != "i8"
        or len(output_type.get_shape()) != 2
        or any(v <= 0 for v in output_type.get_shape())
        or skip.type != output_type
        or consumer.res[0].type != output_type
        or producer.arguments[-1].type != output_type
        or consumer.arguments[-1].type != output_type
    ):
        raise ValueError(
            "quantized sum edge, skip and original outputs must have the same complete signed-i8 matrix type"
        )
    for call in (producer, consumer):
        if not isinstance(call.arguments[-1].owner, tensor.EmptyOp) or len(list(call.arguments[-1].uses)) != 1:
            raise ValueError("quantized sum original destination must be fresh and unobserved")
    cursor = producer.next_op
    while cursor is not None and cursor is not consumer:
        if not isinstance(cursor, tensor.EmptyOp):
            raise ValueError("quantized sum cannot cross an intervening operation or effect")
        cursor = cursor.next_op
    if cursor is None:
        raise ValueError("quantized sum consumer does not follow its original producer")
    declarations = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    if any(
        symbol not in declarations
        or declarations[symbol].body.blocks
        or getattr(declarations[symbol].sym_visibility, "data", None) != "private"
        for symbol in (ps, cs)
    ):
        raise ValueError("quantized sum requires original private borrowed device declarations")
    for call, symbol in ((producer, ps), (consumer, cs)):
        buffers = rewrite.call_buffers.get(symbol, ())
        if len(buffers) != len(call.arguments) or any(
            not isinstance(value.type, TensorType)
            or buffer.get("shape") != list(value.type.get_shape())
            or buffer.get("dtype") != str(value.type.element_type)
            for buffer, value in zip(buffers, call.arguments, strict=True)
        ):
            raise ValueError("quantized sum original logical buffer bindings changed")
    record = dict(
        schema="merlin.quantized_sum_source.v1",
        source_host_sha256=hashlib.sha256(text(module, generic=True).encode()).hexdigest(),
        producer_ordinal=operations.index(producer),
        consumer_ordinal=operations.index(consumer),
        components=[ps, cs],
        entries=[pe, ce],
        producer_result_operand=side,
        shape=list(output_type.get_shape()),
        intermediate_observers=[],
        intermediate_uses=1,
        quantization="original producer Q1 then original residual sum/Q2/activation",
        source_programs=[deepcopy(rewrite.programs[ps]), deepcopy(rewrite.programs[cs])],
        effect_or_target_admission=False,
    )
    return QuantizedSumSource(module, rewrite, producer, consumer, record, _digest(record))


@dataclass(frozen=True)
class QuantizedSumCommandSource:
    """The exact ordinary connected request, independent of host SSA custody."""

    command_buffer: dict
    receipt: dict
    identity: str

    def revalidate(self):
        current = plan_quantized_sum_commands(self.command_buffer)
        if self.receipt != current.receipt or self.identity != current.identity:
            raise ValueError("original connected command source or numerical stages changed")


def plan_quantized_sum_commands(command_buffer):
    """Check the complete ordinary request edge; retain both original stages.

    This establishes source dataflow only. The host's current SSA grouping,
    selected target numerical implementation, private byte layout and actual
    completion remain independently checked. No source is inferred from output.
    """
    if (
        type(command_buffer) is not dict
        or type(command_buffer.get("commands")) is not list
        or type(command_buffer.get("tensors")) is not dict
    ):
        raise ValueError("complete original connected command program required")
    commands = command_buffer["commands"]
    opcodes = [c.get("opcode") for c in commands]
    convolution = opcodes == ["RES_PACK", "CONV2D", "EVICT", "RESIDUAL_ADD"]
    if not convolution and opcodes != ["RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT", "RESIDUAL_ADD"]:
        raise ValueError("quantized request needs one original contraction and one ordered residual")
    packed, product = commands[:2]
    commit = product if convolution else commands[2]
    evict, residual = commands[-2:]
    po, co, ro = product["operands"], commit["operands"], residual["operands"]
    handle = packed["operands"]["dst"]
    weight = packed["operands"]["src"]
    if (
        packed.get("attributes", {}) != {"layout": "packed_conv_rhs" if convolution else "packed_rhs"}
        or evict["operands"] != {"handle": handle}
        or po.get("weight" if convolution else "rhs") != handle
        or not convolution
        and co.get("src") != po.get("dst")
        or set(ro) != {"lhs", "rhs", "dst"}
    ):
        raise ValueError("connected request changed original weight residency or source edge")
    edge, output = co["dst"], ro["dst"]
    positions = [i for i, k in enumerate(("lhs", "rhs")) if ro[k] == edge]
    if len(positions) != 1 or output == edge:
        raise ValueError("one sole intermediate operand and a distinct final destination required")
    side = positions[0]
    skip = ro[("rhs", "lhs")[side]]
    tensors = command_buffer["tensors"]
    attrs = commit.get("attributes", {})
    if edge not in tensors or skip not in tensors or output not in tensors or skip == output:
        raise ValueError("connected source edge, skip and output must be explicit distinct tensors")
    shape = tensors[edge].get("shape")
    if (
        type(shape) is not list
        or len(shape) != 2
        or any(type(v) is not int or v <= 0 for v in shape)
        or any(tensors[n].get("shape") != shape or tensors[n].get("dtype") != "i8" for n in (edge, skip, output))
    ):
        raise ValueError("connected source must retain its complete same-shaped signed-i8 edge")
    epilogue = attrs.get("epilogue")
    sum_attrs = residual.get("attributes", {})
    if (
        epilogue
        not in (["acc_scale"], ["acc_scale", "relu"], ["bias_add", "acc_scale"], ["bias_add", "acc_scale", "relu"])
        or attrs.get("output_dtype") != "i8"
        or sum_attrs.get("output_dtype") != "i8"
        or sum_attrs.get("epilogue") not in ([], ["relu"])
        or type(sum_attrs.get("source_scales")) is not list
        or len(sum_attrs["source_scales"]) != 3
    ):
        raise ValueError("connected request must retain original Q1 and ordered residual Q2/activation")
    lhs = po["ifm" if convolution else "lhs"]
    inputs = {lhs, weight}
    if "bias" in attrs:
        inputs.add(attrs["bias"])
    if (
        set(tensors) != inputs | {edge, skip, output}
        or {edge, output} & inputs
        or any(n not in tensors for n in inputs)
    ):
        raise ValueError("connected source has an unaccounted external tensor or intermediate observer")
    pe = dict(deepcopy(attrs), op="conv2d" if convolution else "matmul", operand_dtype="int8")
    ce = dict(deepcopy(sum_attrs), op="residual_add", operand_dtype="int8")
    record = dict(
        schema="merlin.quantized_sum_command_source.v1",
        source_command_buffer_sha256=_digest(command_buffer),
        source_kind="original ordinary connected command request; host current-SSA binding is separate",
        entries=[pe, ce],
        producer_result_operand=side,
        shape=deepcopy(shape),
        intermediate=edge,
        skip=skip,
        output=output,
        quantization="original producer Q1 then original ordered sum/Q2/activation",
        effect_or_target_admission=False,
    )
    return QuantizedSumCommandSource(command_buffer, record, _digest(record))


def split_quantized_sum_commands(source):
    """Select the two literal original statements; no contraction is rebuilt."""
    if type(source) is not QuantizedSumCommandSource:
        raise ValueError("current checked connected command source required")
    source.revalidate()
    cb = source.command_buffer
    commands = cb["commands"]
    record = source.receipt
    producer_names = set(cb["tensors"]) - {record["skip"], record["output"]}
    consumer_names = {record["intermediate"], record["skip"], record["output"]}
    common = {k: deepcopy(cb[k]) for k in ("abi_version", "target")}
    first = dict(
        common,
        tensors={n: deepcopy(v) for n, v in cb["tensors"].items() if n in producer_names},
        commands=deepcopy(commands[:-1]),
    )
    second = dict(
        common,
        tensors={n: deepcopy(v) for n, v in cb["tensors"].items() if n in consumer_names},
        commands=deepcopy(commands[-1:]),
    )
    # The retained edge is an input only in the second original statement.
    second["tensors"][record["intermediate"]]["role"] = "input"
    source.revalidate()
    return first, second


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
