"""Move selected contractions onto a DEVICE, as calls a compiled host program can make.

This is the shape the matrix-unit path already proved works end to end: replace the contraction with
a call to a private symbol, record what was minted in a sidecar so the build step (a different
process) can generate the callee, and let the host object and the device object meet in one archive.
What that path cannot do is serve a second device: its symbol stem, its dtype legality and its
operand types are literals for one unit. Here all three come from the named device.

Why a sidecar rather than a return value: the rewrite runs inside the lowering subprocess and the
build step that generates the callee runs outside it, so an in-memory hand-off silently produced an
empty signature set.

**Nothing is moved without a decision.** ``select`` is a parameter; with none, the module is returned
untouched. A pass that decided for itself would duplicate the placement decision, and then the two
would disagree -- which is exactly the state this work exists to end.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["DeviceRewrite", "SIDECAR_NAME", "load_sidecar", "rewrite_contractions_to_device",
           "symbol_stem"]

#: Where the rewrite records what it minted, beside the prepared module. One file per device, so two
#: devices in one system do not overwrite each other's signature set.
SIDECAR_NAME = "device_signatures.json"


@dataclass(frozen=True)
class Routed:
    """One contraction that was moved, and the signature it was moved to."""

    symbol: str
    parallel: tuple[int, ...]
    reduction: tuple[int, ...]
    dtypes: tuple[str, str, str]
    fqn: str = ""


@dataclass(frozen=True)
class DeviceRewrite:
    """What the rewrite did, and -- as importantly -- what it declined and why."""

    device: str
    routed: tuple[Routed, ...] = ()
    #: symbol -> (parallel..., reduction) extents, the callee's identity.
    signatures: dict[str, tuple[int, ...]] = field(default_factory=dict)
    #: (symbol-or-"all", reason). A decline is reported, never silent.
    skipped: tuple[tuple[str, str], ...] = ()

    @property
    def moved(self) -> int:
        return len(self.routed)

    def write_sidecar(self, directory: str | Path) -> Path:
        path = Path(directory) / SIDECAR_NAME
        path.write_text(json.dumps({
            "device": self.device,
            "signatures": {s: list(k) for s, k in self.signatures.items()},
            "routed": [{"symbol": r.symbol, "parallel": list(r.parallel),
                        "reduction": list(r.reduction), "dtypes": list(r.dtypes), "fqn": r.fqn}
                       for r in self.routed],
        }, indent=1), encoding="utf-8")
        return path


def load_sidecar(directory: str | Path) -> dict:
    path = Path(directory) / SIDECAR_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def symbol_stem(device: str) -> str:
    """Symbol stem for one device's callees.

    The device NAME is data threaded in by the caller, not a literal in this file, and it has to be in
    the symbol: two devices in one system mint their own callees and a shared stem would collide at
    link time with no diagnostic beyond a duplicate-symbol error.
    """
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(device))
    return f"merlin_dev_{safe}"


def _mlir_type(token: str):
    """An xDSL type for an MLIR dtype token, or None when this path cannot build one.

    None is a real answer and the caller declines that signature. Approximating a type here would
    emit a callee whose operands are a different precision from the contraction it replaced -- which
    compiles, links, and computes the wrong numbers.
    """
    from xdsl.dialects import builtin as _b

    t = str(token)
    if t.startswith("i") and t[1:].isdigit():
        return _b.IntegerType(int(t[1:]))
    named = {"f16": getattr(_b, "Float16Type", None), "f32": getattr(_b, "Float32Type", None),
             "f64": getattr(_b, "Float64Type", None), "bf16": getattr(_b, "BFloat16Type", None)}
    ctor = named.get(t)
    return ctor() if ctor else None


def _signature_types(key: tuple[int, ...], dtypes: tuple[str, str, str]):
    """The three tensor types for one signature, from the DEVICE's own datapath dtypes.

    A leading batch extent is carried onto all three operands: the callee takes the whole batch and
    loops over it, so its type is rank-3 throughout rather than a rank-2 type called several times.
    """
    from xdsl.dialects.builtin import TensorType

    *batch, m, n, k = (int(v) for v in key)
    lhs_t, rhs_t, out_t = (_mlir_type(d) for d in dtypes)
    if not (lhs_t and rhs_t and out_t):
        return None
    return (TensorType(lhs_t, [*batch, m, k]),
            TensorType(rhs_t, [*batch, k, n]),
            TensorType(out_t, [*batch, m, n]))


def _signature_key(shape) -> tuple[int, ...]:
    """``(M, N, K)``, or ``(B, M, N, K)`` with a batch dim. The key IS the callee's identity: MLIR
    function types are monomorphic, so two contractions share a symbol only if every extent agrees."""
    return (*(int(d) for d in shape.parallel), int(shape.reduction[0]))


def rewrite_contractions_to_device(module, device: str, *,
                                   select: Callable[[Any], bool] | None = None,
                                   sidecar_dir: str | Path | None = None) -> DeviceRewrite:
    """Replace each SELECTED contraction with a call to ``device``'s kernel. Mutates ``module``.

    Legality is asked of the device (:mod:`merlin.system.offload`); profitability is ``select``'s.
    Keeping them apart is the point: a device that *could* run a contraction is a hardware fact, and
    whether it *should* is a decision that belongs to one placement pass rather than to each backend.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Block, Region

    from merlin.system.offload import device_dtype_triples, offloadable_contractions

    if select is None:
        return DeviceRewrite(device=device,
                             skipped=(("all", "no selector supplied, so nothing is routed"),))

    triples = device_dtype_triples(device)
    if not triples:
        return DeviceRewrite(device=device,
                             skipped=(("all", f"{device!r} declares no derivable datapath"),))

    candidates = offloadable_contractions(module, device)
    chosen = [(op, sh) for op, sh in candidates if select(sh)]
    skipped: list[tuple[str, str]] = []
    if not chosen:
        return DeviceRewrite(device=device,
                             skipped=(("all", f"{len(candidates)} offloadable contraction(s), "
                                              f"none selected"),))

    stem = symbol_stem(device)
    symbols: dict[tuple[int, ...], str] = {}
    sig_dtypes: dict[str, tuple[str, str, str]] = {}
    routed: list[Routed] = []

    for op, shape in chosen:
        key = _signature_key(shape)
        sym = symbols.get(key)
        if sym is None:
            sym = f"{stem}_{len(symbols)}"
            symbols[key] = sym
            sig_dtypes[sym] = tuple(shape.dtypes)          # type: ignore[assignment]

        operands = list(op.operands)
        if len(operands) != 3 or len(op.results) != 1:
            # Not the (lhs, rhs, out-init) shape this callee promises. Emitting a call with the wrong
            # arity would fail far from here, so decline it with the arity that was actually seen.
            skipped.append((sym, f"expected 3 operands and 1 result, got {len(operands)} "
                                 f"and {len(op.results)}"))
            continue

        call = func.CallOp(sym, operands, [op.results[0].type])
        op.results[0].replace_all_uses_with(call.results[0])
        op.parent.insert_op_before(call, op)
        op.detach()
        op.erase()

        prov = getattr(op, "attributes", {}).get("prov.fqn") if hasattr(op, "attributes") else None
        routed.append(Routed(symbol=sym, parallel=tuple(shape.parallel),
                             reduction=tuple(shape.reduction),
                             dtypes=tuple(shape.dtypes),                # type: ignore[arg-type]
                             fqn=prov.data if isinstance(prov, StringAttr) else ""))

    body: Block = module.body.block
    minted: dict[str, tuple[int, ...]] = {}
    for key, sym in symbols.items():
        types = _signature_types(key, sig_dtypes[sym])
        if types is None:
            skipped.append((sym, f"no MLIR type for datapath {sig_dtypes[sym]}; signature declined"))
            continue
        # `bufferization.access` is load-bearing: without it one-shot-bufferize defensively copies the
        # weight operand, which for a many-block transformer is a large amount of pointless memcpy. It
        # is a PROPERTY on FuncOp, not a discardable attribute, so it goes through the constructor.
        read = DictionaryAttr({"bufferization.access": StringAttr("read")})
        write = DictionaryAttr({"bufferization.access": StringAttr("write")})
        body.add_op(func.FuncOp(sym, ((types[0], types[1], types[2]), (types[2],)), Region(),
                                visibility="private",
                                arg_attrs=ArrayAttr([read, read, write])))
        minted[sym] = key

    out = DeviceRewrite(device=device, routed=tuple(routed), signatures=minted,
                        skipped=tuple(skipped))
    if sidecar_dir is not None:
        out.write_sidecar(sidecar_dir)
    return out
