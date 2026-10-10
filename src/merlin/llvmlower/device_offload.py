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

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "BY_CONTRACTION",
    "BY_GROUP",
    "DeviceRewrite",
    "SIDECAR_NAME",
    "emit_device_program",
    "load_sidecar",
    "lower_device_submits",
    "rewrite_contractions_to_device",
    "rewrite_groups_to_device",
    "rewrite_prepared_file",
    "symbol_stem",
]

#: Where the rewrite records what it minted, beside the prepared module. One file per device, so two
#: devices in one system do not overwrite each other's signature set.
SIDECAR_NAME = "device_signatures.json"

#: The two units this module can move, named because they are DIFFERENT PROGRAMS. A contraction is
#: the multiply-accumulate alone, so the bias, the requantize, the activation and the pooling the
#: layer carries stay on the host; a closed compute group is the layer, readout and all. A build that
#: reported only "N kernels" over a mix of the two would say nothing about which of a model's layers
#: kept their readout -- which is why the granularity is a parameter, recorded in the sidecar, rather
#: than a property of whichever caller happened to run.
BY_CONTRACTION = "contraction"
BY_GROUP = "group"

#: The per-argument ``bufferization.access`` of a contraction's callee: two operands read, the
#: destination written. A group's call derives its own per symbol (``DeviceRewrite.arg_access``).
CONTRACTION_ACCESS = ("read", "read", "write")

#: Entry fields the CORPUS calls irrelevant to sameness but the EMISSION bakes into the artifact:
#: `corpus_spec.build` writes each of these multipliers into the capsule the package lowers, so two
#: layers differing only here are two kernels even though they are one demand. See `_kernel_identity`.
_BAKED_INTO_THE_KERNEL = ("acc_scale", "lhs_scale", "rhs_scale")


@dataclass(frozen=True)
class Routed:
    """One contraction -- or one closed compute group -- that was moved, and where it went."""

    symbol: str
    parallel: tuple[int, ...]
    reduction: tuple[int, ...]
    dtypes: tuple[str, str, str]
    fqn: str = ""
    operation_id: str = ""
    #: The compute group this call is, when the route was ``BY_GROUP``; ``None`` per contraction.
    group: int | None = None
    source_operation_ordinal: int = -1
    source_region: str = ""
    tensor_types: tuple[str, str, str] = ()
    #: Exact captured root identity, when complete provenance survived preparation.
    source_region_id: str | None = None
    source_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeviceRewrite:
    """What the rewrite did, and -- as importantly -- what it declined and why."""

    device: str
    routed: tuple[Routed, ...] = ()
    #: symbol -> (parallel..., reduction) extents, the callee's identity.
    signatures: dict[str, tuple[int, ...]] = field(default_factory=dict)
    #: (symbol-or-"all", reason). A decline is reported, never silent.
    skipped: tuple[tuple[str, str], ...] = ()
    package_sha256: str | None = None
    transport: str | None = None
    abi_sha256: str | None = None
    certification_sha256: tuple[str, ...] = ()
    release_review_digest: str | None = None
    software_spec_sha256: str | None = None
    capability_contract_sha256: str | None = None
    #: symbol -> exact Phase 0 interface bytes and hash, if selected by operation ID.
    expected_interfaces: dict[str, dict[str, str]] = field(default_factory=dict)
    #: :data:`BY_CONTRACTION` or :data:`BY_GROUP` -- what unit was moved. See the constants.
    granularity: str = BY_CONTRACTION
    #: symbol -> the group's own stated program (``group_command.GroupProgram.to_dict()``), whose
    #: ``entry`` carries the epilogue, the multiplier, the convolution geometry and the committed
    #: type. Empty for a contraction route, which states no program: there is nothing but extents.
    programs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: symbol -> the corpus entry a backend is asked to emit for that symbol. The map
    #: :func:`merlin.llvmlower.device_build.build_device_objects` consumes as ``entries=``.
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The ``group_prepack.prepack`` record for the routed groups, or ``None`` when no weights
    #: manifest was readable -- which is a different fact from "no group needed a prepack".
    prepack: dict[str, Any] | None = None
    #: symbol -> per-argument ``bufferization.access``. The contraction route mints one fixed shape
    #: (read, read, write); a group's call takes whatever free values its members read, so the
    #: accesses are DERIVED per symbol and the printer repair has to be told them.
    arg_access: dict[str, tuple[str, ...]] = field(default_factory=dict)
    model_sha256: str | None = None

    @property
    def moved(self) -> int:
        return len(self.routed)

    def to_sidecar(self) -> dict[str, Any]:
        """The sidecar's content, as plain JSON-able data."""
        return {
            "device": self.device,
            "model_sha256": self.model_sha256,
            "granularity": self.granularity,
            "signatures": {s: list(k) for s, k in self.signatures.items()},
            "routed": [
                {
                    "symbol": r.symbol,
                    "parallel": list(r.parallel),
                    "reduction": list(r.reduction),
                    "dtypes": list(r.dtypes),
                    "fqn": r.fqn,
                    "operation_id": r.operation_id,
                    "group": r.group,
                    "source_operation_ordinal": r.source_operation_ordinal,
                    "source_region": r.source_region,
                    "tensor_types": list(r.tensor_types),
                    **(
                        {"source_region_id": r.source_region_id, "source_node_ids": list(r.source_node_ids)}
                        if r.source_region_id is not None and r.source_node_ids
                        else {}
                    ),
                }
                for r in self.routed
            ],
            "package_sha256": self.package_sha256,
            "transport": self.transport,
            "abi_sha256": self.abi_sha256,
            "certification_sha256": list(self.certification_sha256),
            "release_review_digest": self.release_review_digest,
            "software_spec_sha256": self.software_spec_sha256,
            "capability_contract_sha256": self.capability_contract_sha256,
            "expected_interfaces": self.expected_interfaces,
            # THE STATEMENT TRAVELS WITH THE SIGNATURES, for the reason the signatures do at all: the
            # rewrite runs inside the lowering subprocess and the build that generates the callee runs
            # outside it. A build handed only extents rebuilds a bare contraction and drops whatever
            # readout the layer carries, and nothing in the artifact would say so.
            "programs": {s: dict(p) for s, p in self.programs.items()},
            "entries": {s: dict(e) for s, e in self.entries.items()},
            "prepack": self.prepack,
            "arg_access": {s: list(a) for s, a in self.arg_access.items()},
            "skipped": [[s, why] for s, why in self.skipped],
        }

    def write_sidecar(self, directory: str | Path) -> Path:
        path = Path(directory) / SIDECAR_NAME
        path.write_text(json.dumps(self.to_sidecar(), indent=1), encoding="utf-8")
        return path


def load_sidecar(directory: str | Path) -> dict:
    path = Path(directory) / SIDECAR_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def build_arguments(sidecar: dict, *, expected_granularity: str | None = None) -> dict[str, Any]:
    """``{signatures, dtypes, entries}`` -- what a device build takes from one offload sidecar.

    One reader, shared by every build that links a device side, because the interesting argument is
    the one that is easy to forget: ``entries``. A build that passed the signatures and dropped the
    statements would synthesize a bare contraction per symbol -- the same kernel count, the same
    link, and every routed layer's bias, requantize, activation and pooling gone, with nothing in the
    artifact saying so. ``None`` there is the CONTRACTION route's honest answer (it states no
    program); an empty map would instead mean "stated programs were routed and none was supplied",
    which the build rejects.
    """
    routed = sidecar.get("routed") or []
    signatures = {sym: tuple(key) for sym, key in (sidecar.get("signatures") or {}).items()}
    if expected_granularity is not None and expected_granularity not in (BY_CONTRACTION, BY_GROUP):
        raise ValueError("selected device build granularity is unsupported")
    if signatures or routed:
        granularity = sidecar.get("granularity", BY_CONTRACTION)
        if granularity not in (BY_CONTRACTION, BY_GROUP) or (
            expected_granularity is not None and granularity != expected_granularity
        ):
            raise ValueError("device sidecar differs from selected build granularity")
    dtypes = {}
    for row in routed:
        if not isinstance(row, dict) or not isinstance(row.get("symbol"), str) or row["symbol"] not in signatures:
            raise ValueError("device build routed kernel membership differs from declared signatures")
        symbol, precision = row["symbol"], row.get("dtypes")
        if (
            not isinstance(precision, (list, tuple))
            or len(precision) != 3
            or any(not isinstance(token, str) or not token for token in precision)
            or symbol in dtypes
            and dtypes[symbol] != tuple(precision)
        ):
            raise ValueError("device build routed precision is missing or inconsistent for a shared kernel")
        dtypes[symbol] = tuple(precision)
    if set(dtypes) != set(signatures):
        raise ValueError("device build routed kernel membership differs from declared signatures")
    entries = sidecar.get("entries")
    if sidecar.get("granularity") == BY_GROUP:
        # Group signatures do not state the original readout. Losing their
        # entries must not turn them into the distinct contraction-only route.
        if (
            not isinstance(entries, dict)
            or set(entries) != set(signatures)
            or any(not isinstance(entry, dict) or not entry for entry in entries.values())
        ):
            raise ValueError("device build needs a complete stated group program for every routed symbol")
    else:
        entries = entries or None
    return {
        "signatures": signatures,
        "dtypes": dtypes,
        "entries": entries,
    }


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
    named = {
        "f16": getattr(_b, "Float16Type", None),
        "f32": getattr(_b, "Float32Type", None),
        "f64": getattr(_b, "Float64Type", None),
        "bf16": getattr(_b, "BFloat16Type", None),
    }
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
    return (TensorType(lhs_t, [*batch, m, k]), TensorType(rhs_t, [*batch, k, n]), TensorType(out_t, [*batch, m, n]))


def _signature_key(shape) -> tuple[int, ...]:
    """``(M, N, K)``, or ``(B, M, N, K)`` with a batch dim. The key IS the callee's identity: MLIR
    function types are monomorphic, so two contractions share a symbol only if every extent agrees."""
    return (*(int(d) for d in shape.parallel), int(shape.reduction[0]))


def emit_device_program(module, device: str, *, select=None, selected_ops=None) -> list:
    """Record the offload as ``runtime`` dialect ops and return the contractions it claimed.

    Merlin owns a ``runtime`` dialect that says exactly this -- ``device.get``, a command buffer, an
    append per command, ``submit`` -- and until now no real model passed through it. The reason is
    structural rather than neglect: the module's TEXT goes on to upstream mlir-opt, which does not know
    this dialect, so runtime ops cannot survive into what is printed. They therefore live between two
    stages of one xDSL pass -- emitted here, lowered by :func:`lower_device_submits` before anything is
    printed -- which is the only shape in which a real model can pass through the dialect at all.

    What that buys is not decoration. The offload decision is now expressed once, device-independently,
    and the TRANSPORT decides how it is realized; a second transport becomes a second lowering rather
    than a second pass with its own idea of what is legal.
    """
    from xdsl.dialects.builtin import DictionaryAttr, StringAttr

    from merlin.system.offload import offloadable_contractions
    from merlin.xdsl_dialects import runtime as r
    from merlin.xdsl_dialects._common import HAS_XDSL

    chosen = [
        (op, sh)
        for op, sh in offloadable_contractions(module, device)
        if (op in selected_ops if selected_ops is not None else select is None or select(sh))
    ]
    if not chosen or not HAS_XDSL:
        return chosen

    fn = next((f for f in module.walk() if f.name == "func.func"), None)
    if fn is None or not fn.regions or not fn.regions[0].blocks:
        return chosen
    block = fn.regions[0].blocks[0]

    dev = r.DeviceGetOp(
        result_types=[r.DeviceType()],
        properties={"device": StringAttr(str(device)), "backend": r.BackendAttr(r.Backend.BAREMETAL)},
    )
    cb = r.CommandBufferCreateOp(
        operands=[dev.dev], result_types=[r.CommandBufferType()], properties={"target": StringAttr(str(device))}
    )
    block.insert_op_before(dev, block.first_op)
    block.insert_op_after(cb, dev)
    prev = cb
    for _op, shape in chosen:
        ap = r.CommandBufferAppendOp(
            operands=[cb.cb],
            properties={
                "opcode": StringAttr("MATMUL"),
                "args": DictionaryAttr(
                    {
                        "m": StringAttr(str(shape.parallel[-2])),
                        "n": StringAttr(str(shape.parallel[-1])),
                        "k": StringAttr(str(shape.reduction[0])),
                    }
                ),
            },
        )
        block.insert_op_after(ap, prev)
        prev = ap
    sub = r.SubmitOp(operands=[dev.dev, cb.cb], result_types=[r.EventType()])
    block.insert_op_after(sub, prev)
    return chosen


def lower_device_submits(module, device: str, *, transport: str | None) -> int:
    """Replace the ``runtime`` ops with the realization ``transport`` calls for; return how many.

    ``host_instruction`` is realized by the call-plus-shim path the rest of this module emits, so the
    printed module is byte-identical to not having gone through the dialect at all -- which is the
    point: passing through it must cost nothing to the one transport that already worked.

    Any other transport removes the ops WITHOUT a realization and says so, rather than leaving
    dialect ops in text that upstream mlir-opt is about to parse and reject with an error naming a
    dialect nobody outside this repo has heard of.
    """
    from merlin.xdsl_dialects import runtime as r
    from merlin.xdsl_dialects._common import HAS_XDSL

    if not HAS_XDSL:
        return 0
    kinds = (r.SubmitOp, r.CommandBufferAppendOp, r.CommandBufferCreateOp, r.DeviceGetOp)
    doomed = [op for op in module.walk() if isinstance(op, kinds)]
    # The results are consumed only by each other (device -> buffer -> submit), so erasing in reverse
    # order leaves nothing dangling. Nothing outside this set ever holds one: that is what makes the
    # dialect stage removable without touching the model's own values.
    for op in reversed(doomed):
        op.detach()
        op.erase()
    return len(doomed)


def rewrite_contractions_to_device(
    module,
    device: str,
    *,
    select: Callable[[Any], bool] | None = None,
    sidecar_dir: str | Path | None = None,
    exact_selection=None,
    model_sha256: str | None = None,
    catalog_selection: Mapping[str, Sequence[str]] | None = None,
    catalog_bindings: Mapping[str, Mapping[str, object]] | None = None,
) -> DeviceRewrite:
    """Replace each SELECTED contraction with a call to ``device``'s kernel. Mutates ``module``.

    Legality is asked of the device (:mod:`merlin.system.offload`); profitability is ``select``'s.
    Keeping them apart is the point: a device that *could* run a contraction is a hardware fact, and
    whether it *should* is a decision that belongs to one placement pass rather than to each backend.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Block, Region

    from merlin.system.offload import device_dtype_triples, offloadable_contractions

    if exact_selection is not None and select is not None:
        raise ValueError("exact operation selection and a shape selector cannot both decide placement")
    if catalog_selection is not None and (select is not None or exact_selection is not None):
        raise ValueError("catalog source selection cannot be combined with another placement selector")
    if catalog_bindings is not None:
        if catalog_selection is None or set(catalog_bindings) != set(catalog_selection):
            raise ValueError("catalog bindings must cover exactly the selected source regions")
        for region, binding in catalog_bindings.items():
            if (
                binding.get("region") != region
                or not isinstance(binding.get("symbol"), str)
                or not binding["symbol"]
                or type(binding.get("source_operation_ordinal")) is not int
                or binding["source_operation_ordinal"] < 0
                or tuple(binding.get("tensor_types") or ()) != tuple(catalog_selection[region])
            ):
                raise ValueError("catalog binding has no exact source ordinal, types or implementation")
    if exact_selection is not None and not exact_selection.certified:
        raise ValueError("exact operation selection has no independent accelerator certification")
    if exact_selection is not None:
        exact_selection.check_release()
        exact_selection.check_backend_contract()
    if exact_selection is None and select is None and catalog_selection is None:
        return DeviceRewrite(device=device, skipped=(("all", "no selector supplied, so nothing is routed"),))

    ordinal_by_op = {op: ordinal for ordinal, op in enumerate(module.walk())}
    selected_ops = None
    ids_by_op = {}
    if exact_selection is not None:
        from merlin.common import mlir_query as mq
        from merlin.targetgen.application_inventory import exact_int_mm_generic_operation

        if exact_selection.target != device or model_sha256 != exact_selection.model_sha256:
            raise ValueError("exact offload selection does not match device or model bytes")
        ids_by_op = {op: f"mlir:{model_sha256}:{ordinal}" for ordinal, op in enumerate(mq.walk(module))}
        selected_by_id = exact_selection.by_operation_id
        selected_ops = {op for op, operation_id in ids_by_op.items() if operation_id in selected_by_id}
        if len(selected_ops) != len(selected_by_id) or any(
            not exact_int_mm_generic_operation(op) for op in selected_ops
        ):
            raise ValueError("selected operation IDs no longer identify exact integer contractions")

    triples = device_dtype_triples(device)
    if not triples:
        if exact_selection is not None or catalog_selection is not None:
            raise ValueError(f"{device!r} declares no derivable datapath for exact source selection")
        return DeviceRewrite(device=device, skipped=(("all", f"{device!r} declares no derivable datapath"),))

    # THROUGH THE RUNTIME DIALECT. The offload is recorded as runtime ops first and realized second,
    # so the decision is expressed once and device-independently and the TRANSPORT decides how it is
    # carried out. They cannot survive into the printed text -- upstream mlir-opt does not know this
    # dialect -- so both stages happen here, before anything is printed.
    from merlin.system.derive import link_for

    try:
        from merlin.targetgen.target_experiment import load_capability_manifest

        _endpoint = getattr(load_capability_manifest(device), "endpoint_kind", None)
    except Exception:  # noqa: BLE001
        _endpoint = None
    _transport = link_for(device, _endpoint).command_transport

    candidates = offloadable_contractions(module, device)
    if selected_ops is not None and not selected_ops.issubset({op for op, _shape in candidates}):
        raise ValueError("selected operation is not eligible on this device; host fallback must be explicit")
    if catalog_selection is not None:
        matched: dict[str, list] = {region: [] for region in catalog_selection}
        for op, _shape in candidates:
            region = getattr(op.attributes.get("prov.region_id"), "data", "")
            if region not in matched or not op.results:
                continue
            types = tuple(str(x.type) for x in op.operands[:2]) + (str(op.results[0].type),)
            if types == tuple(catalog_selection[region]):
                matched[region].append(op)
        if any(len(ops) != 1 for ops in matched.values()):
            failures = {region: len(ops) for region, ops in matched.items() if len(ops) != 1}
            raise ValueError(f"catalog operations are absent or ambiguous on this device: {failures}")
        if catalog_bindings is not None and any(
            ordinal_by_op[ops[0]] != catalog_bindings[region]["source_operation_ordinal"]
            for region, ops in matched.items()
        ):
            raise ValueError("catalog source operation ordinal differs from the current source")
        selected_ops = {ops[0] for ops in matched.values()}
    chosen = emit_device_program(module, device, select=select, selected_ops=selected_ops)
    lower_device_submits(module, device, transport=_transport)
    skipped: list[tuple[str, str]] = []
    if not chosen:
        return DeviceRewrite(
            device=device, skipped=(("all", f"{len(candidates)} offloadable contraction(s), none selected"),)
        )

    stem = symbol_stem(device)
    # A source-bound implementation is part of the callee identity, alongside
    # shape and precision. Equal shapes may select different legal schedules.
    symbols: dict[tuple[tuple[int, ...], tuple[str, str, str], str], str] = {}
    sig_dtypes: dict[str, tuple[str, str, str]] = {}
    expected_interfaces: dict[str, dict[str, str]] = {}
    routed: list[Routed] = []

    for op, shape in chosen:
        key = _signature_key(shape)
        dtypes = tuple(shape.dtypes)
        source_region = getattr(op.attributes.get("prov.region_id"), "data", "")
        implementation = catalog_bindings[source_region]["symbol"] if catalog_bindings is not None else ""
        signature = (key, dtypes, implementation)
        sym = symbols.get(signature)
        if sym is None:
            sym = f"{stem}_{len(symbols)}"
            symbols[signature] = sym
            sig_dtypes[sym] = dtypes  # type: ignore[assignment]
        operation_id = ids_by_op.get(op, "")
        if exact_selection is not None:
            selected = exact_selection.by_operation_id[operation_id]
            expected = {"sha256": selected.interface_sha256, "mlir": selected.interface_mlir}
            if sym in expected_interfaces and expected_interfaces[sym] != expected:
                raise ValueError(f"{sym} shares one extent but selected interfaces differ")
            expected_interfaces[sym] = expected

        operands = list(op.operands)
        if len(operands) != 3 or len(op.results) != 1:
            # Not the (lhs, rhs, out-init) shape this callee promises. Emitting a call with the wrong
            # arity would fail far from here, so decline it with the arity that was actually seen.
            skipped.append((sym, f"expected 3 operands and 1 result, got {len(operands)} and {len(op.results)}"))
            continue

        tensor_types = tuple(str(x.type) for x in operands[:2]) + (str(op.results[0].type),)
        call = func.CallOp(sym, operands, [op.results[0].type])
        op.results[0].replace_all_uses_with(call.results[0])
        op.parent.insert_op_before(call, op)
        op.detach()
        op.erase()

        prov = getattr(op, "attributes", {}).get("prov.fqn") if hasattr(op, "attributes") else None
        routed.append(
            Routed(
                symbol=sym,
                parallel=tuple(shape.parallel),
                reduction=tuple(shape.reduction),
                dtypes=tuple(shape.dtypes),  # type: ignore[arg-type]
                fqn=prov.data if isinstance(prov, StringAttr) else "",
                operation_id=operation_id,
                source_operation_ordinal=ordinal_by_op[op],
                source_region=source_region,
                tensor_types=tensor_types,
            )
        )

    body: Block = module.body.block
    minted: dict[str, tuple[int, ...]] = {}
    for (key, _dtypes, _implementation), sym in symbols.items():
        types = _signature_types(key, sig_dtypes[sym])
        if types is None:
            skipped.append((sym, f"no MLIR type for datapath {sig_dtypes[sym]}; signature declined"))
            continue
        # `bufferization.access` is load-bearing: without it one-shot-bufferize defensively copies the
        # weight operand, which for a many-block transformer is a large amount of pointless memcpy. It
        # is a PROPERTY on FuncOp, not a discardable attribute, so it goes through the constructor.
        read = DictionaryAttr({"bufferization.access": StringAttr("read")})
        write = DictionaryAttr({"bufferization.access": StringAttr("write")})
        body.add_op(
            func.FuncOp(
                sym,
                ((types[0], types[1], types[2]), (types[2],)),
                Region(),
                visibility="private",
                arg_attrs=ArrayAttr([read, read, write]),
            )
        )
        minted[sym] = key

    if exact_selection is not None and (len(routed) != len(selected_ops) or set(minted) != set(expected_interfaces)):
        raise ValueError("an exactly selected operation was not rewritten and declared in full")

    out = DeviceRewrite(
        device=device,
        routed=tuple(routed),
        signatures=minted,
        skipped=tuple(skipped),
        package_sha256=exact_selection.package_sha256 if exact_selection is not None else None,
        transport=exact_selection.transport if exact_selection is not None else None,
        abi_sha256=exact_selection.abi_sha256 if exact_selection is not None else None,
        certification_sha256=exact_selection.certification_sha256 if exact_selection is not None else (),
        release_review_digest=exact_selection.release_binding.review_digest if exact_selection is not None else None,
        software_spec_sha256=exact_selection.software_spec_sha256 if exact_selection is not None else None,
        capability_contract_sha256=exact_selection.capability_contract_sha256 if exact_selection is not None else None,
        expected_interfaces=expected_interfaces,
        model_sha256=model_sha256,
    )
    if sidecar_dir is not None:
        out.write_sidecar(sidecar_dir)
    return out


# --- the whole-program route: one call per CLOSED COMPUTE GROUP ---------------------------------


def _kernel_identity(entry: dict[str, Any]) -> str:
    """Two groups share one device kernel only when they ask for the SAME program.

    Built on :func:`merlin.targetgen.group_capsule_entries._identity`, which is the corpus's notion of "the
    same demand": it drops the name and the requantize multipliers, because a unit's COMMAND STREAM is
    the same for every positive multiplier. That is the right key for a capsule corpus and the wrong
    one for a kernel, because :func:`merlin.targetgen.corpus_spec.build` bakes the multiplier INTO the
    artifact the package emits. Two layers that differ only in ``acc_scale`` would then share a kernel
    carrying one of the two multipliers, and the other layer would compute a scaled version of itself
    with nothing in the build saying so. So the baked numbers are appended to the corpus identity
    rather than a second identity being invented here -- one notion of sameness, extended where the
    emission makes a dropped field load-bearing.
    """
    from merlin.targetgen.group_capsule_entries import _IDENTITY_DROPS, _identity

    # Only the NUMBERS the emission bakes come back. The rest of what the corpus drops -- the name,
    # the free-text comment, the source reference naming which layer this was -- is per-layer
    # bookkeeping, and folding it back in would make every call site its own kernel and defeat the
    # sharing this key exists for. Intersected with the corpus's own list so that a field it stops
    # dropping stops being appended here too, rather than being added twice.
    baked = {key: entry[key] for key in _BAKED_INTO_THE_KERNEL if key in entry and key in _IDENTITY_DROPS}
    return _identity(entry) + "|" + json.dumps(baked, sort_keys=True)


def _entry_extents(entry: dict[str, Any]) -> tuple[int, ...]:
    """``(M, N, K)`` when the stated entry declares all three, otherwise ``()``.

    An extent triple is what a caller that only knows contractions can read off a program. A
    convolution states its geometry instead (``ci``/``kh``/``kw``/``stride``) and an elementwise sum
    reduces over nothing, so for those the honest answer is that there is no triple -- not a
    synthesized one. Consumers are expected to read the ENTRY when there is one; the triple exists so
    a signature map stays meaningful to the contraction-granular consumers that predate this route.
    """
    return tuple(int(entry[k]) for k in ("M", "N", "K")) if all(k in entry for k in ("M", "N", "K")) else ()


def _stated_shape(entry: dict[str, Any], shape_class):
    """The extent shape a group's own statement makes, or ``None`` when it declares no triple.

    The placement decides over ``(M, K, N)``; a statement that carries all three can be keyed the
    same way as an observed contraction, and one that cannot (a convolution states its geometry) is
    ``None`` rather than a triple assembled out of the geometry -- two different numbers under one
    name is how a decision gets applied to a shape nobody decided.
    """
    rows, columns = entry.get("M"), entry.get("N")
    if rows is None or columns is None:
        return None
    reduced = entry.get("K")
    # An elementwise sum reduces over NOTHING, and `()` is how `ContractionShape` says that. A
    # selector keyed on a reduction extent then declines it, which is the right answer rather than an
    # invented one: a placement derived from contraction demands never decided about this region.
    return shape_class(
        op=str(entry.get("op") or "matmul"),
        parallel=(int(rows), int(columns)),
        reduction=() if reduced is None else (int(reduced),),
    )


def _element_token(mlir_type) -> str:
    """The MLIR element-type token of a tensor type, as the shim's byte-width table spells it."""
    element = getattr(mlir_type, "element_type", mlir_type)
    return str(element)


def _integer_element(mlir_type) -> bool:
    """Does ``mlir_type`` (a tensor type) hold integer elements?

    Asked structurally of the type rather than of its spelling: the device this route builds for
    computes in the integer datapath its entry declares, so a group whose escaping value is a float
    tensor cannot be answered by that kernel however its extents look.
    """
    from xdsl.dialects.builtin import IntegerType

    element = getattr(mlir_type, "element_type", None)
    return isinstance(element, IntegerType)


def _device_operands(group, stated) -> tuple[tuple[Any, Any] | None, str]:
    """``((activation, stored), "")`` -- the two values the STATED program reads -- or the refusal.

    NOT the group's free values. A closed group reads its scale splats, its zero points and its
    padding constants as well, and a device kernel built from the group's ENTRY takes none of them:
    the multiplier is baked into the entry as ``acc_scale`` and the zero point into its numerics. A
    call carrying them would declare a callee nothing can define -- the kernel ABI a backend contract
    declares is the three-operand one -- so the operands are derived from the statement instead.

    Both operands are taken from BEFORE the dequantize, because the dequantize is what the group
    absorbed: the device reads the stored integers. A group whose operand reaches the contraction as a
    float with no dequantize behind it is refused rather than handed to an integer datapath.
    """
    root = group.root
    if len(list(root.operands)) < 2:
        return None, "this group's root reads fewer than two operands, so the kernel ABI has nothing to take"
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    values = []
    for index in (0, 1):
        operand = list(root.operands)[index]
        _adapters, dequantize, _dtype = CG._input_chain(operand)
        value = dequantize.operands[0] if dequantize is not None else operand
        if not _integer_element(value.type):
            return None, f"operand {index} reaches this root as a float, so an integer kernel cannot read it"
        values.append(value)
    stored = stated.stored_operand
    if stored is None:
        # A SUM OF TWO ACTIVATIONS HAS NO STORED SIDE, and that is a statement about the program
        # rather than a gap: `group_command.program` returns `stored_operand=None` for it on purpose.
        # Its two operands go in the order the root reads them, which is the order the entry's
        # ``lhs_scale``/``rhs_scale`` are stated in -- swapping them would apply each scale to the
        # other tensor. A contraction of two activations is the same case: the statement read them as
        # lhs[M, K] @ rhs[K, N] in the root's own order.
        return (values[0], values[1]), ""
    return (values[1 - int(stored)], values[int(stored)]), ""


def rewrite_groups_to_device(
    module,
    device: str,
    *,
    select: Callable[[Any], bool] | None = None,
    weight_args=None,
    model: str = "",
    oracle=None,
    capture: str | Path | None = None,
    sidecar_dir: str | Path | None = None,
) -> DeviceRewrite:
    """Replace each SELECTED closed COMPUTE GROUP with ONE call to ``device``'s kernel. Mutates ``module``.

    The unit this moves is the layer, not the multiply-accumulate:
    :func:`~merlin.xdsl_dialects.lowering.compute_groups.form_groups` closes a contraction together
    with the stages the target's readout takes, and
    :func:`~merlin.xdsl_dialects.lowering.group_command.program` restates that group as the entry a
    backend package builds from -- epilogue, multiplier, convolution geometry, committed type. Routing
    the contraction alone (:func:`rewrite_contractions_to_device`) leaves every one of those on the
    host, so the model runs as a device call per matmul interleaved with host epilogues: correct, and
    nothing like the program a whole-model schedule emits.

    A group is routed, or it is DECLINED BY NAME. There is no third outcome and no fallback to the
    bare contraction: a build that silently substituted extents for a statement would drop the
    readout and report the same count either way.

    ``select`` is the placement decision, passed in and never taken here -- the contract the
    contraction route keeps, for the same reason. It is offered the group's root as the
    ``ContractionShape`` the placement decided over, because
    :func:`merlin.system.place.device_selector` keys on extents and cannot read an op handle.
    """
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Region

    from merlin.kernels.microkernel import ContractionShape
    from merlin.kernels.shapes import observe_contractions
    from merlin.system.offload import device_dtype_triples
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC

    from .group_offload import capsule_entry_for

    if select is None:
        return DeviceRewrite(
            device=device,
            granularity=BY_GROUP,
            skipped=(("all", "no selector supplied, so nothing is routed"),),
        )
    # ONE DATAPATH AUTHORITY, NOT TWO. With an ``oracle`` the caller has supplied the device's own
    # statement of what it admits, and asking `system.offload` for a second one would be asking about
    # a different device. Without one the contract is the only source, and a device that declares no
    # datapath offloads nothing -- fail closed rather than route onto an unknown precision.
    if oracle is None and not device_dtype_triples(device):
        return DeviceRewrite(
            device=device,
            granularity=BY_GROUP,
            skipped=(("all", f"{device!r} declares no derivable datapath"),),
        )

    if weight_args is None and capture is not None:
        # WHICH ARGUMENT IS STORED, from the capture's own weights manifest. Without one a first
        # layer whose two operands are both model arguments cannot be told apart, and the honest
        # outcome is that group refusing BY NAME below rather than a guess about which side is the
        # weight -- so this reads the manifest when there is one and passes None on when there is not.
        from merlin.xdsl_dialects.lowering import stream_plan as _stream_plan

        weight_args = _stream_plan.weight_args_beside(capture)

    groups = CG.form_groups(module, device, oracle=oracle)
    # THE SHAPES ARE OBSERVED, NOT FILTERED FOR LEGALITY. `system.offload.offloadable_contractions`
    # answers "could this device take this contraction as it stands" -- and a captured layer does not
    # stand as an integer contraction at all: it arrives as a dequantize into an f32 matmul, which
    # that predicate declines for every layer of a quantized model (measured on a captured ResNet-50:
    # 0 of 54). Legality here is the compute-group ORACLE's answer (the group is not on the host) and
    # the decision is `select`'s; these shapes exist only to key the decision, which
    # `system.place.device_selector` does on extents.
    shape_of = {id(op): shape for op, shape in observe_contractions(module)}

    stem = symbol_stem(device)
    symbols: dict[str, str] = {}  # kernel identity -> symbol
    minted: dict[str, tuple[int, ...]] = {}
    declared: dict[str, tuple[Any, ...]] = {}  # symbol -> the operand types its declaration carries
    returns: dict[str, Any] = {}  # symbol -> the result type its declaration carries
    programs: dict[str, dict[str, Any]] = {}
    entries: dict[str, dict[str, Any]] = {}
    access: dict[str, tuple[str, ...]] = {}
    routed: list[Routed] = []
    skipped: list[tuple[str, str]] = []
    placed: list = []  # the groups that actually became calls, for the prepack record

    for group in groups:
        name = (
            f"{''.join(c if (c.isalnum() or c == '_') else '_' for c in model)}_group{group.index}"
            if model
            else f"group{group.index}"
        )
        if group.placement == CG.HOST or group.root is None:
            continue  # the planner's own decision; `group_offload.census` is where it is counted
        try:
            stated = GC.program(group, weight_args=weight_args, name=name)
        except CG.NoCapsuleForm as error:
            skipped.append((name, f"no device form: {error}"))
            continue
        # The placement is keyed on EXTENTS, so the group is offered to `select` as the shape its root
        # makes. A root the shape observer cannot read (a reduce that the group states as a window
        # mean, say) still HAS extents -- the ones its own statement declares -- so the statement is
        # the fallback rather than a decline: declining there would leave a routable layer on the host
        # for want of a second reading of the same numbers.
        shape = shape_of.get(id(group.root)) or _stated_shape(stated.entry, ContractionShape)
        if shape is None or not select(shape):
            skipped.append((name, "the placement did not select this group's root"))
            continue

        order = {id(op): index for index, op in enumerate(group.root.parent.ops)}
        if any(id(member) not in order for member in group.members):
            skipped.append((name, "the group names an op outside its root's block; it cannot be replaced by one call"))
            continue
        members = sorted(group.members, key=lambda m: order[id(m)])
        last = members[-1]
        if not last.results:
            skipped.append((name, "the group's last member produces no value, so a call cannot stand for it"))
            continue
        # EVERY ESCAPING VALUE MUST BE THE ONE THE CALL RETURNS. Formation grows a group only along
        # values with a single consumer, so this holds by construction -- which is exactly why it is
        # checked: a later clause that admitted a second consumer would make the erase below drop a
        # value something outside still reads, and the module would fail to verify far from here.
        inside = {id(member) for member in members}
        escaping = [
            member
            for member in members[:-1]
            for result in member.results
            for use in result.uses
            if id(use.operation) not in inside
        ]
        if escaping:
            skipped.append(
                (name, f"{len(escaping)} value(s) of this group are read outside it, so one call cannot stand for it")
            )
            continue

        result_type = last.results[0].type
        # THE DEVICE COMMITS THE TYPE THE CALL RETURNS. A group that stops before its closing
        # requantize still escapes as the capture's float accumulation, and an integer kernel cannot
        # produce that -- so it is refused by name rather than routed into a call whose result type
        # disagrees with what the kernel computes. (Measured on a captured micro-model: all four of
        # its device groups end at the contraction, because the matmul's result is read twice.)
        if not _integer_element(result_type):
            skipped.append(
                (name, "this group escapes as a float value, which the integer datapath it states cannot commit")
            )
            continue
        pair, why = _device_operands(group, stated)
        if pair is None:
            skipped.append((name, why))
            continue
        if stated.batch_shape:
            batch = stated.batch_shape[0]
            rows, columns, reduced = _entry_extents(stated.entry)
            expected = ((batch, rows, reduced), (batch, reduced, columns), (batch, rows, columns))
            actual = tuple(tuple(t.get_shape()) for t in (pair[0].type, pair[1].type, result_type))
            if actual != expected:
                skipped.append((name, "the batch slice ABI cannot reinterpret operand or result views"))
                continue

        region_attr = group.root.attributes.get("prov.region_id")
        nodes_attr = group.root.attributes.get("prov.source_node_ids")
        region_id = region_attr.data if isinstance(region_attr, StringAttr) and region_attr.data else None
        node_ids = (
            tuple(sorted(value.data for value in nodes_attr.data))
            if isinstance(nodes_attr, ArrayAttr)
            and nodes_attr.data
            and all(isinstance(value, StringAttr) and value.data for value in nodes_attr.data)
            else ()
        )
        if len(node_ids) != len(set(node_ids)):
            node_ids = ()

        entry = capsule_entry_for(stated.entry, index=group.index, name=name, device=device, model=model)
        param_types = (pair[0].type, pair[1].type, result_type)
        identity = _kernel_identity(entry)
        if stated.batch_shape:
            # Different batch counts are different monomorphic host wrappers,
            # while their stated per-slice device programs remain identical.
            identity += "|batch_abi:" + json.dumps([str(t) for t in param_types])
        sym = symbols.get(identity)
        if sym is None:
            sym = f"{stem}_{len(symbols)}"
            symbols[identity] = sym
            declared[sym] = param_types
            returns[sym] = result_type
            minted[sym] = (*stated.batch_shape, *_entry_extents(entry))
            programs[sym] = stated.to_dict()
            entries[sym] = entry
            access[sym] = CONTRACTION_ACCESS
        elif (declared[sym], returns[sym]) != (param_types, result_type):
            # MLIR function types are monomorphic. Two groups that ask for the same PROGRAM but hand
            # it different operand types are two callees; sharing one symbol would emit a call whose
            # types disagree with its declaration, which fails in the parser far from here.
            skipped.append((name, f"this group states the same program as @{sym} but with a different function type"))
            continue

        # DESTINATION-PASSING, like the contraction route's call: the callee writes the committed
        # result into the buffer it is handed and returns it. The destination is minted here rather
        # than reusing the contraction's zero fill, because the fill holds ACCUMULATOR elements and
        # what the group commits is what its readout narrows them to.
        destination = tensor.EmptyOp((), result_type)
        call = func.CallOp(sym, [pair[0], pair[1], destination.results[0]], [result_type])
        last.parent.insert_op_before(destination, last)
        last.parent.insert_op_before(call, last)
        last.results[0].replace_all_uses_with(call.results[0])
        for member in reversed(members):
            member.detach()
            member.erase()
        # THE DATAPATH THE CALL ACTUALLY CARRIES, read off the three types the call was built with --
        # not off the capture's contraction shape, whose element types are the FLOAT ones the
        # dequantize produced. The shim sizes its staging buffers from this triple, so recording the
        # capture's f32 there would stage four bytes per int8 element of every routed layer.
        routed.append(
            Routed(
                symbol=sym,
                parallel=tuple(int(v) for v in shape.parallel),
                reduction=tuple(int(v) for v in shape.reduction),
                dtypes=tuple(_element_token(t) for t in param_types),  # type: ignore[arg-type]
                fqn=str(entry.get("source_reference", "")),
                group=int(group.index),
                source_region_id=region_id,
                source_node_ids=node_ids,
            )
        )
        placed.append(group)

    body = module.body.block
    for sym in minted:
        # `bufferization.access` is load-bearing: without it one-shot-bufferize defensively copies the
        # weight operand, which for a many-block transformer is a large amount of pointless memcpy. It
        # is a PROPERTY on FuncOp, not a discardable attribute, so it goes through the constructor.
        attrs = ArrayAttr([DictionaryAttr({"bufferization.access": StringAttr(a)}) for a in access[sym]])
        body.add_op(
            func.FuncOp(
                sym,
                (declared[sym], (returns[sym],)),
                Region(),
                visibility="private",
                arg_attrs=attrs,
            )
        )

    prepack, why_no_prepack = _prepack_for(placed, capture)
    if why_no_prepack:
        skipped.append(("prepack", why_no_prepack))

    out = DeviceRewrite(
        device=device,
        routed=tuple(routed),
        signatures=minted,
        skipped=tuple(skipped),
        granularity=BY_GROUP,
        programs=programs,
        entries=entries,
        prepack=prepack,
        arg_access=access,
    )
    if sidecar_dir is not None:
        out.write_sidecar(sidecar_dir)
    return out


def _prepack_for(groups, capture) -> tuple[dict[str, Any] | None, str]:
    """The routed groups' prepack record, or ``None`` and the reason there is none.

    A device kernel built from a group's stated program reads its bias ALREADY FOLDED into the
    accumulator's integer domain and its weight ALREADY LAID OUT the way the program holds it. Which
    of those the build has to do is a property of the statement, so it is recorded beside it -- a
    build that had the statement and not the prepack would stage a float bias and a capture-ordered
    weight into a kernel that expects neither, and the numbers would simply be wrong.

    ``None`` is a real answer and it is never silent: with no capture beside the module there is no
    weights manifest, and a manifest is what says which argument is stored.
    """
    if not groups:
        return None, ""
    if capture is None:
        return None, "no capture was given, so no weights manifest could be read and no bias folded"
    beside = Path(capture)
    beside = beside if beside.is_dir() else beside.parent
    manifest = next(iter(sorted(beside.glob("*.manifest.json"))), None)
    weights = next(iter(sorted(beside.glob("*.safetensors"))), None)
    if manifest is None or weights is None:
        return None, f"no weights manifest and safetensors pair beside {beside}"
    from merlin.xdsl_dialects.lowering import group_prepack as GP

    try:
        return GP.prepack(groups, manifest, weights, device_layout=True)["record"], ""
    except Exception as error:  # noqa: BLE001 -- named, never a silently absent prepack
        return None, f"prepack: {type(error).__name__}: {error}"


def rewrite_prepared_file(
    prepared: str | Path,
    work: str | Path,
    device: str,
    *,
    select: Callable[[Any], bool] | None = None,
    exact_selection=None,
    granularity: str = BY_CONTRACTION,
    weight_args=None,
    model: str = "",
    capture: str | Path | None = None,
    catalog_manifest: str | Path | None = None,
    catalog_object: str | Path | None = None,
) -> DeviceRewrite:
    """Rewrite a prepared module ON DISK in place and record what it minted.

    This is the seam a whole-model build uses: read the module the preparation passes produced, move
    the selected contractions onto ``device``, print it back, repair the declarations the printer
    drops, and write the sidecar the build's device side reads.

    ``granularity`` chooses WHAT is moved -- :data:`BY_CONTRACTION` (the multiply-accumulate alone,
    epilogue left on the host) or :data:`BY_GROUP` (the closed compute group, readout and all). It is
    a parameter and not a default because the two build different programs, and the sidecar records
    which one ran so a build cannot be read as the other.

    It REFUSES to write a module whose declarations lost their access attributes, and that is not
    belt-and-braces. xDSL stores ``arg_attrs`` correctly but prints them only for a function WITH a
    body; for a bodyless declaration it prints the types alone, so the attributes never reach the text
    that gets parsed, and one-shot-bufferize then defensively copies the weight operand of every
    routed contraction. A large amount of pointless memcpy in a shipped model is exactly the kind of
    regression nothing would attribute back to here.

    The repair itself is shared with the matrix-unit path rather than reimplemented: both need the
    same fixup for the same printer, and two copies would drift.
    """
    from ..frontends.linalg_mlir import parse_mlir_file
    from ..xdsl_dialects._common import text as to_text
    from .declaration_access import patch_declaration_arg_attrs, unpatched_declarations

    if granularity not in (BY_CONTRACTION, BY_GROUP):
        raise ValueError(f"{granularity!r} is neither {BY_CONTRACTION!r} nor {BY_GROUP!r}")
    prepared, work = Path(prepared), Path(work)
    model_sha256 = hashlib.sha256(prepared.read_bytes()).hexdigest()
    if exact_selection is not None and exact_selection.model_sha256 != model_sha256:
        raise ValueError("prepared model bytes changed after exact operation selection")
    catalog_selection = None
    catalog_bindings = None
    if (catalog_manifest is None) != (catalog_object is None):
        raise ValueError("catalog rewrite needs both manifest and object")
    if catalog_manifest is not None:
        if select is not None or exact_selection is not None:
            raise ValueError("catalog source selection cannot be combined with another placement selector")
        data = json.loads(Path(catalog_manifest).read_text(encoding="utf-8"))
        if data.get("source_sha256") != model_sha256 or not data.get("coverage_complete"):
            raise ValueError("external catalog does not cover these exact prepared model bytes")
        rows = data.get("bindings") or ()
        catalog_selection = {row["region"]: tuple(row["tensor_types"]) for row in rows}
        catalog_bindings = {row["region"]: row for row in rows}
        if (
            len(catalog_selection) != len(rows)
            or not catalog_selection
            or any(not region for region in catalog_selection)
        ):
            raise ValueError("external catalog needs unique nonempty source regions")
    module = parse_mlir_file(prepared)
    if catalog_selection is not None and granularity != BY_CONTRACTION:
        raise ValueError("source catalogs bind contractions; group routing requires its own program contract")
    if exact_selection is not None and granularity != BY_CONTRACTION:
        raise ValueError("exact operation selection binds contractions; a group route cannot honour it")
    rewrite = (
        rewrite_contractions_to_device(
            module,
            device,
            select=select,
            exact_selection=exact_selection,
            model_sha256=model_sha256,
            catalog_selection=catalog_selection,
            catalog_bindings=catalog_bindings,
        )
        if granularity == BY_CONTRACTION
        else rewrite_groups_to_device(
            module, device, select=select, weight_args=weight_args, model=model, capture=capture
        )
    )
    if rewrite.moved:
        # A contraction's callee has one fixed shape; a group's call takes the free values its members
        # read, so each symbol is patched with the accesses the rewrite DERIVED for it.
        text = to_text(module)
        by_access: dict[tuple[str, ...], list[str]] = {}
        for sym in rewrite.signatures:
            by_access.setdefault(tuple(rewrite.arg_access.get(sym) or CONTRACTION_ACCESS), []).append(sym)
        for access, symbols in by_access.items():
            text = patch_declaration_arg_attrs(text, symbols, argument_access=access)
        missing = unpatched_declarations(text, rewrite.signatures)
        if missing:
            raise RuntimeError(
                f"declarations {list(missing)} carry no bufferization.access attributes, so "
                "one-shot-bufferize would copy the weight operand of every contraction routed to "
                "them; refusing to write the module"
            )
        prepared.write_text(text, encoding="utf-8")
    work.mkdir(parents=True, exist_ok=True)
    rewrite.write_sidecar(work)
    if catalog_manifest is not None:
        sidecar = work / SIDECAR_NAME
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["catalog_manifest"] = str(Path(catalog_manifest).resolve())
        data["catalog_object"] = str(Path(catalog_object).resolve())
        sidecar.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return rewrite
