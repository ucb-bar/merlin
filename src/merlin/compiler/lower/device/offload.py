"""The record of one device rewrite: what moved, under which ABI, and what was declined.

Dependency-free data shared by the device routes and the source-composition passes that only
inspect a finished rewrite. ``merlin.compiler.lower.device.offload`` performs rewrites and re-exports
these same objects.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from merlin.compiler.lower.device.build import _flags
from merlin.compiler.lower.device.device_build import DeviceBuild, _nm
from merlin.compiler.lower.device.shim import emit_dense_translation_unit
from merlin.compiler.lower.model.toolchain import clang

# ---- device_rewrite ----------------------------------------------------------------------------

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
    #: :func:`merlin.compiler.lower.device.build.build_device_objects` consumes as ``entries=``.
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: The ``group_prepack.prepack`` record for the routed groups, or ``None`` when no weights
    #: manifest was readable -- which is a different fact from "no group needed a prepack".
    prepack: dict[str, Any] | None = None
    #: symbol -> per-argument ``bufferization.access``. The contraction route mints one fixed shape
    #: (read, read, write); a group's call takes whatever free values its members read, so the
    #: accesses are DERIVED per symbol and the printer repair has to be told them.
    arg_access: dict[str, tuple[str, ...]] = field(default_factory=dict)
    model_sha256: str | None = None
    #: symbol -> the host call's operands under the logical kernel ABI, in call order: each
    #: ``{"role", "shape", "dtype"}`` (activation, stored weight, folded bias, destinations). Empty
    #: for a call that passes the fixed (lhs, rhs, out) triple.
    call_buffers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: The laid-out weights bound as trailing ``@forward`` arguments (:mod:`merlin.compiler.lower.c.weights`),
    #: or ``None`` when they stayed constants.
    blob_weights: Any = None
    #: Original checked placement record; absent on historical selector routes.
    offload_selection: dict | None = None

    @property
    def moved(self) -> int:
        return len(self.routed)

    def to_sidecar(self) -> dict[str, Any]:
        """The sidecar's content, as plain JSON-able data."""
        record = {
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
            "call_buffers": {s: [dict(b) for b in bs] for s, bs in self.call_buffers.items()},
            "skipped": [[s, why] for s, why in self.skipped],
        }
        if self.offload_selection is not None:
            record["offload_selection"] = self.offload_selection
        return record

    def write_sidecar(self, directory: str | Path) -> Path:
        path = Path(directory) / SIDECAR_NAME
        path.write_text(json.dumps(self.to_sidecar(), indent=1), encoding="utf-8")
        if self.blob_weights is not None:
            self.blob_weights.write(directory)
        return path


# ---- device_offload ----------------------------------------------------------------------------
# Move selected contractions onto a DEVICE, as calls a compiled host program can make.
#
# This is the shape the matrix-unit path already proved works end to end: replace the contraction with
# a call to a private symbol, record what was minted in a sidecar so the build step (a different
# process) can generate the callee, and let the host object and the device object meet in one archive.
# What that path cannot do is serve a second device: its symbol stem, its dtype legality and its
# operand types are literals for one unit. Here all three come from the named device.
#
# Why a sidecar rather than a return value: the rewrite runs inside the lowering subprocess and the
# build step that generates the callee runs outside it, so an in-memory hand-off silently produced an
# empty signature set.
#
# **Nothing is moved without a decision.** ``select`` is a parameter; with none, the module is returned
# untouched. A pass that decided for itself would duplicate the placement decision, and then the two
# would disagree -- which is exactly the state this work exists to end.

_BAKED_INTO_THE_KERNEL = ("acc_scale", "lhs_scale", "rhs_scale")


def load_sidecar(directory: str | Path) -> dict:
    path = Path(directory) / SIDECAR_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def build_arguments(sidecar: dict, *, expected_granularity: str | None = None) -> dict[str, Any]:
    """``{signatures, dtypes, entries, call_buffers}`` -- what a device build takes from one offload sidecar.

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
        "call_buffers": sidecar.get("call_buffers") or None,
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


def emit_device_program(module, device: str, *, select=None, selected_ops=None, candidates=None) -> list:
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

    from merlin.compiler.dialects import runtime as r
    from merlin.compiler.dialects.xdsl_context import HAS_XDSL
    from merlin.target.system import offloadable_contractions

    chosen = [
        (op, sh)
        for op, sh in (candidates if candidates is not None else offloadable_contractions(module, device))
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
    from merlin.compiler.dialects import runtime as r
    from merlin.compiler.dialects.xdsl_context import HAS_XDSL

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
    catalog_manifest_data: Mapping[str, object] | None = None,
    offload_selection=None,
    routing=None,
) -> DeviceRewrite:
    """Replace each SELECTED contraction with a call to ``device``'s kernel. Mutates ``module``.

    Legality is asked of the device (:mod:`merlin.target.system`); profitability is ``select``'s.
    Keeping them apart is the point: a device that *could* run a contraction is a hardware fact, and
    whether it *should* is a decision that belongs to one placement pass rather than to each backend.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Block, Region

    from merlin.compiler.planning.source_coverage import catalog_source_contractions
    from merlin.target.system import device_dtype_triples, offloadable_contractions

    selection_record = None
    if offload_selection is not None:
        from merlin.compiler.lower.device.selection import DeviceOffloadSelection

        if (
            type(offload_selection) is not DeviceOffloadSelection
            or routing is None
            or routing.device != device
            or routing.select is not select
            or getattr(routing, "offload_selection", None) is not offload_selection
            or exact_selection is not None
            or catalog_selection is not None
            or catalog_bindings is not None
            or catalog_manifest_data is not None
        ):
            raise ValueError("device economics requires its exact original ordinary selector route")
        selection_record = offload_selection.record(routing=routing)
    elif routing is not None:
        raise ValueError("device economics routing has no original source selection")
    if exact_selection is not None and select is not None:
        raise ValueError("exact operation selection and a shape selector cannot both decide placement")
    if catalog_selection is not None and (select is not None or exact_selection is not None):
        raise ValueError("catalog source selection cannot be combined with another placement selector")
    if catalog_manifest_data is not None and catalog_selection is None:
        raise ValueError("catalog dtype conversion requires exact source selection")
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
    if exact_selection is None and select is None and catalog_selection is None and offload_selection is None:
        return DeviceRewrite(device=device, skipped=(("all", "no selector supplied, so nothing is routed"),))

    ordinal_by_op = {op: ordinal for ordinal, op in enumerate(module.walk())}
    selected_ops = None
    ids_by_op = {}
    if offload_selection is not None:
        selected_ops = set(offload_selection.select_for(module, routing=routing))
        if not selected_ops and getattr(routing, "target_configuration", None) is not None:
            raise ValueError("all-CPU economics cannot consume an independently selected target IR cleanup")
        if {ordinal_by_op[op] for op in selected_ops} != set(selection_record["selected_ordinals"]):
            raise ValueError("device economics changed its exact source choice before rewriting")
    if exact_selection is not None:
        from merlin.compiler.ir import mlir_query as mq
        from merlin.development.targetgen.applications.inventory import exact_int_mm_generic_operation

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
        if exact_selection is not None or catalog_selection is not None or selected_ops:
            raise ValueError(f"{device!r} declares no derivable datapath for exact source selection")
        return DeviceRewrite(
            device=device,
            offload_selection=selection_record,
            skipped=(("all", f"{device!r} declares no derivable datapath"),),
        )

    # THROUGH THE RUNTIME DIALECT. The offload is recorded as runtime ops first and realized second,
    # so the decision is expressed once and device-independently and the TRANSPORT decides how it is
    # carried out. They cannot survive into the printed text -- upstream mlir-opt does not know this
    # dialect -- so both stages happen here, before anything is printed.
    from merlin.target.system import link_for

    try:
        from merlin.target.registry.experiment_descriptor import load_capability_manifest

        _endpoint = getattr(load_capability_manifest(device), "endpoint_kind", None)
    except Exception:  # noqa: BLE001
        _endpoint = None
    _transport = link_for(device, _endpoint).command_transport

    conversion_source = None
    if catalog_manifest_data is not None:
        conversion_source = catalog_source_conversion(catalog_manifest_data, device)
    candidates = (
        catalog_source_contractions(module, device, conversion_source)
        if conversion_source is not None
        else offloadable_contractions(module, device)
    )
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
    chosen = emit_device_program(module, device, select=select, selected_ops=selected_ops, candidates=candidates)
    lower_device_submits(module, device, transport=_transport)
    if offload_selection is not None and (
        len(chosen) != len(selected_ops) or {op for op, _shape in chosen} != selected_ops
    ):
        raise ValueError("device economics did not retain its complete realized operation selection")
    skipped: list[tuple[str, str]] = []
    if not chosen:
        return DeviceRewrite(
            device=device,
            offload_selection=selection_record,
            skipped=(("all", f"{len(candidates)} offloadable contraction(s), none selected"),),
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
    if offload_selection is not None:
        from merlin.compiler.transforms.certificates import ir_digest

        after = ir_digest(module)
        offload_selection.revalidate(routing=routing)
        if (
            ir_digest(module) != after
            or {item.source_operation_ordinal for item in routed} != set(selection_record["selected_ordinals"])
            or len(routed) != len(selected_ops)
            or set(minted) != {item.symbol for item in routed}
        ):
            raise ValueError("device economics did not preserve its complete realized source membership")

    out = DeviceRewrite(
        device=device,
        offload_selection=selection_record,
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


# ---- device_catalog ----------------------------------------------------------------------------
# Bind a source-identified external device catalog to whole-model call symbols.
#
# The target package owns kernel generation and its ISA policy. This module checks the
# catalog's generic dense-pointer ABI, exact source and object hashes, shape and dtype
# coverage, then emits only the memref adapter needed by the host compiler.


def catalog_source_conversion(data: Mapping[str, object], device: str) -> tuple[str, str, str] | None:
    """Validate an explicit source-to-native dtype contract; return its source ABI.

    The target supplies the implementation and numerical proof. Core checks
    that the declaration is complete, content-bound, and points to one of the
    selected device's actual native datapaths. No target format is inferred
    from a source tensor type or from a kernel symbol.
    """
    conversion = data.get("source_conversion")
    if conversion is None:
        return None
    from merlin.target.system import device_dtype_triples

    abi = data.get("abi") or {}
    if not isinstance(abi, Mapping):
        raise ValueError("catalog source conversion needs a dense source ABI")
    declared_source = abi.get("dtypes")
    if (
        not isinstance(conversion, Mapping)
        or conversion.get("schema") != "target_owned_dtype_conversion_v1"
        or not isinstance(declared_source, list)
        or len(declared_source) != 3
        or not all(isinstance(dtype, str) and dtype for dtype in declared_source)
        or conversion.get("source_dtypes") != declared_source
    ):
        raise ValueError("catalog source conversion disagrees with its dense ABI")
    source = tuple(declared_source)
    native = set(device_dtype_triples(device))
    allowed = conversion.get("target_dtypes")
    if (
        not native
        or not isinstance(allowed, list)
        or not allowed
        or any(
            not isinstance(triple, list)
            or len(triple) != 3
            or not all(isinstance(dtype, str) and dtype for dtype in triple)
            or tuple(triple) not in native
            for triple in allowed
        )
        or len({tuple(triple) for triple in allowed}) != len(allowed)
    ):
        raise ValueError("catalog source conversion names no distinct native target datapaths")
    policy = conversion.get("policy")
    if not isinstance(policy, Mapping) or not policy:
        raise ValueError("catalog source conversion has no explicit numeric policy")
    try:
        digest = hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    except (TypeError, ValueError) as error:
        raise ValueError("catalog source conversion policy is not canonical JSON") from error
    if conversion.get("policy_sha256") != digest:
        raise ValueError("catalog source conversion policy hash differs from its declaration")
    bindings = data.get("bindings") or ()
    if not bindings or any(
        not isinstance(row, Mapping) or list(row.get("target_dtypes") or ()) not in allowed for row in bindings
    ):
        raise ValueError("catalog source conversion has an unbound target format")
    return source  # type: ignore[return-value]


def build_catalog_objects(
    device: str,
    signatures: Mapping[str, Sequence[int]],
    dtypes: Mapping[str, Sequence[str]],
    *,
    manifest_path: str | Path,
    object_path: str | Path,
    source_sha256: str,
    routed: Sequence[Mapping[str, object]],
    workdir: str | Path,
    codegen_target: str = "riscv",
    cflags: Sequence[str] | None = None,
    timeout: int = 900,
) -> DeviceBuild:
    """Return a linkable catalog and shim, or raise on any incomplete binding."""
    manifest_path, object_path = Path(manifest_path), Path(object_path)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("external catalog manifest must be a regular file")
    if not object_path.is_file() or object_path.is_symlink():
        raise ValueError("external catalog object must be a regular file")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    conversion_source = catalog_source_conversion(data, device)
    abi = data.get("abi") or {}
    if (
        abi.get("argument_order") != ["lhs", "rhs", "out"]
        or abi.get("pointee_layout") != "dense_row_major"
        or abi.get("batch_call") != "whole_batch"
        or len(abi.get("dtypes") or ()) != 3
    ):
        raise ValueError("external catalog does not declare the dense pointer ABI")
    if data.get("source_sha256") != source_sha256:
        raise ValueError("external catalog was compiled from different prepared model bytes")
    if not data.get("coverage_complete") or data.get("covered_contractions") != data.get("matched_contractions"):
        raise ValueError("external catalog is incomplete")
    actual_sha = hashlib.sha256(object_path.read_bytes()).hexdigest()
    if data.get("compilation", {}).get("object_sha256") != actual_sha:
        raise ValueError("external catalog object hash differs from its compilation receipt")

    available: dict[str, tuple[bool, int, int, int, int]] = {}
    symbols: set[str] = set()
    for row in data.get("kernels", ()):
        dims = row["dimensions"]
        batched = row["batched"]
        key = (batched, dims["batch"], dims["m"], dims["n"], dims["k"])
        symbol = row["symbol"]
        if (
            type(batched) is not bool
            or any(type(v) is not int or v <= 0 for v in key[1:])
            or not _identifier(symbol)
            or symbol in symbols
        ):
            raise ValueError("external catalog has ambiguous or invalid kernel records")
        available[symbol] = key
        symbols.add(symbol)
    if len(symbols) != data.get("unique_kernels") or not symbols:
        raise ValueError("external catalog kernel count disagrees with its records")
    nm = _nm()
    if nm is None:
        raise ValueError("external catalog needs a symbol inspector")
    symbol_table = subprocess.run(
        [nm, "--format=posix", "--extern-only", str(object_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if symbol_table.returncode:
        raise ValueError(f"could not inspect catalog object symbols: {symbol_table.stderr[-300:]}")
    defined: dict[str, int] = {}
    unresolved: list[str] = []
    for line in symbol_table.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2:
            raise ValueError("unreadable catalog object symbol record")
        if fields[1].upper() == "U":
            unresolved.append(fields[0])
        elif fields[1].upper() == "T":
            defined[fields[0]] = defined.get(fields[0], 0) + 1
    if any(defined.get(symbol) != 1 for symbol in symbols) or unresolved:
        raise ValueError("catalog object is missing declared kernels or has unresolved references")

    binding_by_region: dict[str, dict] = {}
    for row in data.get("bindings", ()):
        region, symbol = row.get("region"), row.get("symbol")
        if (
            not isinstance(region, str)
            or not region
            or region in binding_by_region
            or symbol not in symbols
            or len(row.get("tensor_types") or ()) != 3
        ):
            raise ValueError("external catalog has invalid operation bindings")
        binding_by_region[region] = row
    if len(binding_by_region) != data.get("covered_contractions"):
        raise ValueError("external catalog operation count disagrees with bindings")

    expected_shapes: dict[str, tuple[bool, int, int, int, int]] = {}
    for entry, shape in signatures.items():
        if not _identifier(entry):
            raise ValueError(f"invalid offload entry symbol {entry!r}")
        dims = tuple(shape)
        if len(dims) == 3:
            m, n, k = dims
            key = (False, 1, m, n, k)
        elif len(dims) == 4:
            b, m, n, k = dims
            key = (True, b, m, n, k)
        else:
            raise ValueError(f"unsupported offload rank for {entry}")
        if tuple(dtypes.get(entry, ())) != tuple(abi["dtypes"]):
            raise ValueError(f"catalog has no proven precision for {entry}")
        expected_shapes[entry] = key
    if not expected_shapes:
        raise ValueError("external catalog has no offloaded calls to bind")
    if not routed:
        raise ValueError("external catalog needs source operation identities for routed calls")
    routed_regions: set[str] = set()
    kernels: dict[str, str] = {}
    for row in routed:
        region, entry = row.get("source_region"), row.get("symbol")
        binding = binding_by_region.get(region)
        if (
            not isinstance(region, str)
            or region in routed_regions
            or entry not in expected_shapes
            or binding is None
            or available[binding["symbol"]] != expected_shapes[entry]
            or type(binding.get("source_operation_ordinal")) is not int
            or binding["source_operation_ordinal"] != row.get("source_operation_ordinal")
            or binding.get("tensor_types") != row.get("tensor_types")
            or tuple(row.get("dtypes") or ()) != tuple(abi["dtypes"])
        ):
            raise ValueError("routed operation does not match the source catalog binding")
        if entry in kernels and kernels[entry] != binding["symbol"]:
            raise ValueError("one routed callee refers to different source-bound implementations")
        kernels[entry] = binding["symbol"]
        routed_regions.add(region)
    if set(kernels) != set(signatures) or routed_regions != set(binding_by_region):
        raise ValueError("routed operation coverage differs from catalog entries")

    unit = emit_dense_translation_unit(device, signatures, dtypes, kernel_symbol_for=kernels.__getitem__)
    if set(unit.symbols) != set(signatures) or unit.skipped:
        raise ValueError(f"external catalog shim declined routed calls: {unit.skipped}")
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    source = work / "device_catalog_shim.c"
    source.write_text(unit.text, encoding="utf-8")
    shim = work / "device_catalog_shim.o"
    result = subprocess.run(
        [clang(), *_flags(codegen_target, cflags), "-c", str(source), "-o", str(shim)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"external catalog shim compilation failed: {result.stderr[-600:]}")
    shim_symbols = subprocess.run(
        [nm, "--format=posix", "--extern-only", str(shim)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if shim_symbols.returncode:
        raise ValueError(f"could not inspect catalog shim symbols: {shim_symbols.stderr[-300:]}")
    shim_defined: set[str] = set()
    shim_undefined: set[str] = set()
    for line in shim_symbols.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2:
            raise ValueError("unreadable catalog shim symbol record")
        (shim_undefined if fields[1].upper() == "U" else shim_defined).add(fields[0])
    if not set(kernels).issubset(shim_defined) or shim_undefined != set(kernels.values()):
        raise ValueError("catalog shim entry points or kernel references disagree with routed calls")
    receipt = {
        "schema": "external_device_catalog_binding_v1",
        "source_sha256": source_sha256,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "catalog_object_sha256": actual_sha,
        "shim_object_sha256": hashlib.sha256(shim.read_bytes()).hexdigest(),
        "routed_operations": len(routed),
        "entry_to_kernel": dict(sorted(kernels.items())),
        "compiler_flags": _flags(codegen_target, cflags),
        **(
            {
                "source_conversion_policy_sha256": data["source_conversion"]["policy_sha256"],
                "source_conversion_dtypes": list(conversion_source),
            }
            if conversion_source
            else {}
        ),
    }
    (work / "catalog_binding.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return DeviceBuild(device=device, objects=(object_path, shim), shim_object=shim, kernels=kernels)


def _identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and (value[0].isalpha() or value[0] == "_")
        and all(c.isalnum() or c == "_" for c in value)
    )


__all__ = [
    "BY_CONTRACTION",
    "BY_GROUP",
    "CONTRACTION_ACCESS",
    "DeviceRewrite",
    "Routed",
    "SIDECAR_NAME",
    "emit_device_program",
    "load_sidecar",
    "lower_device_submits",
    "rewrite_contractions_to_device",
    "rewrite_groups_to_device",
    "rewrite_prepared_file",
    "symbol_stem",
]
