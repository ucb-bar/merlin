"""Compile connected source stages through checked native compilers and explicit continuations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import prod

from merlin.compiler.lower.host.borrowed_buffer_calls import BorrowedBufferEffects
from merlin.compiler.partition.regions import ConnectedRegion, RegionStage, plan_connected_region
from merlin.verification.lean.contract import document_digest

# ---- connected ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectedCompilation:
    region: ConnectedRegion
    products: tuple
    continuation: object
    checked_continuation_sha256: str
    checked_images: tuple[tuple[bytes, ...], ...]
    report: dict

    def images(self):
        """Ordered stage launches; never concatenate instructions across entrypoints."""
        self.region.revalidate()
        if document_digest(asdict(self.continuation)) != self.checked_continuation_sha256:
            raise ValueError("connected runtime continuation changed; recompile")
        if len(self.products) != len(self.region.stages) or any(
            product.source is not self.region.source or product.operation is not stage.operation
            for product, stage in zip(self.products, self.region.stages, strict=True)
        ):
            raise ValueError("connected native products no longer cover their original stages")
        actual = tuple(product.images() for product in self.products)
        if actual != self.checked_images:
            raise ValueError("connected stage images changed; recompile")
        return actual


def _compile_connected(
    region,
    *,
    windows,
    byte_orders,
    selected_contracts,
    continuation,
    stage_options,
    dense_options=None,
    dense_contracts=None,
):
    """Compile a connected region with source-owned bindings and explicit adapters.

    ``stage_options(region, stage, bindings)`` returns ordinary
    ``compile_source_pointwise`` options (Kernel, storage adapter, encoder, limits,
    optional rewrites/pass order). It cannot replace the source, selected
    contracts or bindings. All actual effects and final bytes pass native source
    conformance. Inputs/outputs may reuse storage only outside their live ranges.

    The explicit Continuation promises sequential completion and persistence;
    qualifying it on a real runtime remains the caller's independent obligation.
    Images are returned as ordered launches, not blindly concatenated binaries.
    """
    from merlin.compiler.schedule.elementwise.pointwise_source import (
        PointwiseBindings,
        compile_source_pointwise,
    )
    from merlin.compiler.schedule.pipeline.program_limits import AddressWindow, Continuation
    from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts

    if type(region) is not ConnectedRegion or not callable(stage_options):
        raise ValueError("connected compilation requires a current typed plan and target stage selector")
    region.revalidate()
    if dense_options is None and any(stage.kind != "elementwise_map" for stage in region.stages):
        raise ValueError("counted dense composition needs a multi-stage pointer/launch ABI; source plan retained")
    if any(stage.kind == "contraction" and len(set(stage.inputs)) != len(stage.inputs) for stage in region.stages):
        raise ValueError("dense pointer arguments alias; native dense route requires disjoint external buffers")
    if (
        type(selected_contracts) is not SelectedSchedulingContracts
        or selected_contracts.storage is None
        or type(continuation) is not Continuation
        or type(windows) is not tuple
        or len(windows) != len(region.values)
        or any(type(w) is not AddressWindow for w in windows)
        or type(byte_orders) is not tuple
        or len(byte_orders) != len(windows)
        or any(order not in ("little", "big") for order in byte_orders)
    ):
        raise ValueError("connected compilation needs exact storage, byte orders and sequential runtime contract")
    selected_contracts.revalidate()
    windows = tuple(AddressWindow(w.space, w.start, w.size) for w in windows)
    bindings_sha = document_digest([asdict(w) for w in windows])
    selection_sha = document_digest(selected_contracts.record())
    continuation_sha = document_digest(asdict(continuation))
    external = {window.resource: window for window in selected_contracts.storage.external}
    physical = []
    for value, window in zip(region.values, windows, strict=True):
        expected = prod(value.type.get_shape()) * value.type.element_type.width.data // 8
        bound = external.get(window.space)
        if (
            window.size != expected
            or bound is None
            or window.start < bound.start
            or window.start + window.size > bound.start + bound.size
        ):
            raise ValueError("connected tensor binding differs from complete type or selected external storage")
        physical.append((bound.backing, window.start, window.start + window.size))
    for i, (space, start, end) in enumerate(physical):
        for j, (other, low, high) in enumerate(physical[:i]):
            live = region.lifetimes[i][0] <= region.lifetimes[j][1] and region.lifetimes[j][0] <= region.lifetimes[i][1]
            if live and space == other and start < high and low < end:
                raise ValueError("connected live tensors alias; cannot retain original SSA values")
    for boundary in range(len(region.stages) - 1):
        required = {
            window.space
            for window, (first, last) in zip(windows, region.lifetimes, strict=True)
            if first <= boundary < last
        }
        if not required <= set(continuation.preserved_spaces):
            raise ValueError("connected runtime does not preserve all live tensors across launches")
    if dense_options is not None and not callable(dense_options):
        raise ValueError("dense stage selector must be callable")
    if dense_contracts is None:
        dense_contracts = selected_contracts
    dense_selection = None
    if any(stage.kind == "contraction" for stage in region.stages):
        if type(dense_contracts) is not SelectedSchedulingContracts or dense_contracts.dense is None:
            raise ValueError("mixed region requires explicit selected dense contracts")
        dense_contracts.revalidate()
        dense_selection = document_digest(dense_contracts.record())
    products = []
    for stage in region.stages:
        bindings = PointwiseBindings(
            tuple(windows[i] for i in stage.inputs),
            windows[stage.output],
            tuple(byte_orders[i] for i in (*stage.inputs, stage.output)),
            True,
        )
        owned_bindings = asdict(bindings)
        selector = dense_options if stage.kind == "contraction" else stage_options
        options = selector(region, stage, bindings)
        if asdict(bindings) != owned_bindings:
            raise ValueError("connected tensor bindings changed during target selection")
        if type(options) is not dict or set(options) & {
            "module",
            "operation",
            "bindings",
            "selected_contracts",
            "external_buffers_disjoint",
            "output_operation",
            "output_operations",
            "readout_qualification_identity",
        }:
            raise ValueError("target stage selector cannot override source-owned bindings or contracts")
        region.revalidate()
        if (
            document_digest([asdict(w) for w in windows]) != bindings_sha
            or document_digest(selected_contracts.record()) != selection_sha
            or document_digest(asdict(continuation)) != continuation_sha
        ):
            raise ValueError("connected contracts changed during target selection")
        if stage.kind == "contraction":
            from merlin.compiler.schedule.dense.source import schedule_source_dense

            dense_contracts.revalidate()
            if document_digest(dense_contracts.record()) != dense_selection:
                raise ValueError("dense contracts changed during target selection")
            product = schedule_source_dense(
                region.source,
                stage.operation,
                selected_contracts=dense_contracts,
                external_buffers_disjoint=True,
                **options,
            )
            if product is None:
                raise ValueError("no legal dense stage realization; retain unfused source")
        else:
            product = compile_source_pointwise(
                region.source, stage.operation, bindings=bindings, selected_contracts=selected_contracts, **options
            )
        products.append(product)
        if asdict(bindings) != owned_bindings:
            raise ValueError("connected tensor bindings changed during stage compilation")
    if dense_options is not None:
        return ConnectedProgram.checked(region, tuple(products), windows, byte_orders, selected_contracts, continuation)
    images = tuple(product.images() for product in products)
    if sum(map(len, images)) > continuation.max_segments:
        raise ValueError("connected executable exceeds runtime launch budget")
    result = ConnectedCompilation(
        region,
        tuple(products),
        continuation,
        continuation_sha,
        images,
        {
            "source": region.record(),
            "bindings": [asdict(w) for w in windows],
            "runtime_continuation": asdict(continuation),
            "selected_contracts_sha256": selection_sha,
            "stage_launches": [len(parts) for parts in images],
            "actual_encoded_bytes": sum(len(b) for p in images for b in p),
            "stage_conformance": [p.conformance for p in products],
            "qualification": "native source and final-byte conformance; runtime execution remains unqualified",
            "maturity": "executable_unfused_composition_not_cross_stage_fusion",
            "residency": "selected_runtime_buffer_persistence_only; no on-chip placement inferred",
        },
    )
    result.images()
    return result


def compile_connected_pointwise(region, *, windows, byte_orders, selected_contracts, continuation, stage_options):
    """Compile source-checked finite pointwise images with retained SSA boundaries."""
    return _compile_connected(
        region,
        windows=windows,
        byte_orders=byte_orders,
        selected_contracts=selected_contracts,
        continuation=continuation,
        stage_options=stage_options,
    )


def compile_connected_region(
    region,
    *,
    windows,
    byte_orders,
    selected_contracts,
    continuation,
    stage_options,
    dense_options,
    dense_contracts=None,
):
    """Compose native dense loops and pointwise images through explicit launches.

    Target callbacks select ordinary stage compiler options. ``dense_contracts``
    defaults to the common selection, or may explicitly select a different dense
    engine; no target identity is inferred. Dense stages cannot consume adjacent
    epilogues here: every SSA stage and visible intermediate remains independent.
    Returns a ConnectedProgram whose synchronous runtime passes exact input and
    output byte views to the caller's target launcher. Source is never rewritten;
    refusal leaves the caller's original unfused compilation path available.
    """
    if not callable(dense_options):
        raise ValueError("mixed composition requires explicit dense stage selection")
    return _compile_connected(
        region,
        windows=windows,
        byte_orders=byte_orders,
        selected_contracts=selected_contracts,
        continuation=continuation,
        stage_options=stage_options,
        dense_options=dense_options,
        dense_contracts=dense_contracts,
    )


# ---- connected ---------------------------------------------------------------------------------
# Synchronous execution of native mixed-region products with source-owned views.
#
# This is generic launch/buffer composition, not target instruction lowering.
# Callbacks remain responsible for target execution and completion qualification.


@dataclass(frozen=True)
class ConnectedInvocation:
    """One original SSA stage, with arguments in source operand/result order."""

    stage: object
    product: object
    bindings: object
    images: tuple[bytes, ...] | None


@dataclass(frozen=True)
class ConnectedProgram:
    region: object
    invocations: tuple[ConnectedInvocation, ...]
    continuation: object
    selected_contracts: object
    windows: tuple
    byte_orders: tuple[str, ...]
    checked_identity: str
    report: dict

    def _identity(self):
        return document_digest(
            {
                "continuation": asdict(self.continuation),
                "selection": self.selected_contracts.record(),
                "windows": [asdict(w) for w in self.windows],
                "byte_orders": self.byte_orders,
                "invocations": [
                    {
                        "bindings": asdict(i.bindings),
                        "ordinal": i.stage.ordinal,
                        "dense": {
                            "schedule": i.product.compiled.kernel.digest(),
                            "selection": i.product.selection_identity,
                        }
                        if i.images is None
                        else None,
                        "images": None if i.images is None else [b.hex() for b in i.images],
                    }
                    for i in self.invocations
                ],
            }
        )

    @classmethod
    def checked(cls, region, products, windows, byte_orders, selected_contracts, continuation):
        from merlin.compiler.schedule.elementwise.pointwise_source import PointwiseBindings

        invocations = []
        for stage, product in zip(region.stages, products, strict=True):
            bindings = PointwiseBindings(
                tuple(windows[i] for i in stage.inputs),
                windows[stage.output],
                tuple(byte_orders[i] for i in (*stage.inputs, stage.output)),
                True,
            )
            images = None if stage.kind == "contraction" else product.images()
            invocations.append(ConnectedInvocation(stage, product, bindings, images))
        launches = sum(1 if i.images is None else len(i.images) for i in invocations)
        if launches > continuation.max_segments:
            raise ValueError("connected executable exceeds runtime launch budget")
        result = cls(
            region,
            tuple(invocations),
            continuation,
            selected_contracts,
            windows,
            byte_orders,
            "",
            {
                "source": region.record(),
                "bindings": [asdict(w) for w in windows],
                "runtime_continuation": asdict(continuation),
                "launches": launches,
                "stage_forms": [
                    "checked_compact_dense" if i.images is None else "checked_pointwise_images" for i in invocations
                ],
                "maturity": "executable_unfused_composition_not_cross_stage_fusion",
                "residency": "selected_runtime_buffer_persistence_only",
                "execution_qualification": "not_established",
                "physical_savings_claimed": False,
            },
        )
        object.__setattr__(result, "checked_identity", result._identity())
        result.revalidate()
        return result

    def revalidate(self):
        from merlin.compiler.schedule.dense.source import DenseSourceResult
        from merlin.compiler.schedule.elementwise.pointwise_source import PointwiseCompilation

        self.region.revalidate()
        self.selected_contracts.revalidate()
        if self._identity() != self.checked_identity or len(self.invocations) != len(self.region.stages):
            raise ValueError("connected runtime bindings or continuation changed; recompile")
        for stage, invocation in zip(self.region.stages, self.invocations, strict=True):
            product = invocation.product
            if (
                invocation.stage != stage
                or product.source is not self.region.source
                or product.operation is not stage.operation
            ):
                raise ValueError("connected native product no longer covers its original stage")
            if stage.kind == "contraction":
                if type(product) is not DenseSourceResult or product.operations != (stage.operation,):
                    raise ValueError("connected dense stage must preserve each separate source operation")
                product.revalidate()
            elif type(product) is not PointwiseCompilation or product.images() != invocation.images:
                raise ValueError("connected pointwise image changed; recompile")

    def ordered_invocations(self):
        """Checked native products/bindings for ordinary guest code generation.

        Dense products expose their existing emit_c/emit_llvm APIs; pointwise
        entries carry source-checked images. The caller supplies target pointer
        ABI, image loading and synchronous launch emission. Returning this plan
        does not establish a compiled guest ABI or runtime qualification.
        """
        self.revalidate()
        return self.invocations

    def execute(self, backings, launch, *, backing_origins=None):
        """Run checked stages over explicit persistent bytearrays, returning live-outs.

        ``launch(invocation, inputs, output)`` consumes the native dense product
        or checked pointwise images. Inputs are read-only byte views; output is
        the exact writable SSA result view. Return None only after completion and
        visibility. Callback/runtime qualification is external, not established
        by this Python interface. No scheduling, packing or arithmetic is supplied.

        Backings use selected physical storage identities. Distinct identities
        must have distinct bytearrays; implicit aliasing through arbitrary buffer
        exporters is intentionally unsupported. Non-overlapping windows of one
        backing remain legal. Caller owns allocation and initialization.
        ``backing_origins`` maps every physical backing to the byte address
        represented by buffer[0]. Omission explicitly selects origin zero; a
        physical address is never rebased implicitly to fit a smaller buffer.
        """
        if type(backings) is not dict or not callable(launch):
            raise ValueError("connected execution requires explicit backing buffers and synchronous launcher")
        self.revalidate()
        external = {w.resource: w for w in self.selected_contracts.storage.external}
        required = {external[w.space].backing for w in self.windows}
        if any(name not in backings or type(backings[name]) is not bytearray for name in required):
            raise ValueError("connected execution requires an explicit bytearray per physical backing")
        if len({id(backings[name]) for name in required}) != len(required):
            raise ValueError("distinct selected backing identities alias at runtime")
        origins = {name: 0 for name in required} if backing_origins is None else backing_origins
        if (
            type(origins) is not dict
            or set(origins) != required
            or any(type(origin) is not int or origin < 0 for origin in origins.values())
        ):
            raise ValueError("runtime backing origins must explicitly cover every backing in byte-address units")
        views = []
        for window in self.windows:
            name = external[window.space].backing
            buffer = backings[name]
            start = window.start - origins[name]
            if start < 0 or len(buffer) < start + window.size:
                raise ValueError("connected runtime backing is smaller than its selected tensor window")
            views.append(memoryview(buffer)[start : start + window.size])
        for invocation in self.invocations:
            self.revalidate()
            inputs = tuple(views[i].toreadonly() for i in invocation.stage.inputs)
            output = views[invocation.stage.output]
            if launch(invocation, inputs, output) is not None:
                raise ValueError("connected launcher must complete synchronously and return None")
            self.revalidate()
        return {i: views[i].toreadonly() for i in self.region.outputs}


# ---- connectedhost -----------------------------------------------------------------------------
# Emit synchronous connected host calls over explicit caller-owned byte buffers.
#
# This supplies binding/sequence code, not target instructions, packing, allocation,
# completion or numerical qualification. Stage implementations remain explicit.

_C_KEYWORDS = frozenset(
    "auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while".split()
)


@dataclass(frozen=True)
class ConnectedCStage:
    """Implementation of the generated byte-view ABI under explicit premises.

    The requested symbol has signature
    int symbol(const uint8_t *const *inputs, const size_t *input_bytes,
               uint8_t *output, size_t output_bytes).
    Return zero only after full output visibility; other status stops the chain.
    Both success and failure must satisfy effects (including completion). The
    implementation owns exact image/logical-window to runtime-pointer binding.
    Identity strings are assertions, not authenticated certificates.
    """

    source: str
    effects: BorrowedBufferEffects
    input_alignments: tuple[int, ...]
    output_alignment: int
    window_binding_identity: str

    def validate(self, input_count):
        if (
            type(self.source) is not str
            or not self.source.strip()
            or type(self.effects) is not BorrowedBufferEffects
            or type(self.input_alignments) is not tuple
            or len(self.input_alignments) != input_count
            or any(
                type(n) is not int or n <= 0 or n > (1 << 64) - 1 or n & (n - 1)
                for n in (*self.input_alignments, self.output_alignment)
            )
            or type(self.window_binding_identity) is not str
            or not self.window_binding_identity
        ):
            raise ValueError(
                "connected stage requires source, explicit alignments, borrowed effects and window binding"
            )
        self.effects.__post_init__()


@dataclass(frozen=True)
class ConnectedCArtifact:
    """Detached emitted source and historical binding receipt; no execution claim."""

    source: str
    receipt: dict


def emit_connected_host_c(program, emit_stage, *, symbol, backing_origins):
    """Emit a checked sequence; stage callback consumes each actual native product.

    Backings are in the receipt's deterministic order. Origins are explicit byte
    coordinates represented by backing.base[0]. C callers supply accessible arrays
    and truthful sizes; integer checks cannot prove an allocation's accessibility.
    The selected host must provide flat uintptr_t-addressable byte memory.
    No pointers or views may escape a stage; execution is synchronous and unfused.
    """
    if (
        type(program) is not ConnectedProgram
        or not callable(emit_stage)
        or type(symbol) is not str
        or not symbol.isascii()
        or symbol in _C_KEYWORDS
        or not symbol
        or not symbol[0].isalpha()
        or not all(c.isalnum() or c == "_" for c in symbol)
    ):
        raise ValueError("connected C emission requires a checked program, callback and ordinary C symbol")
    invocations = program.ordered_invocations()
    identity = program.checked_identity
    external = {w.resource: w for w in program.selected_contracts.storage.external}
    names = tuple(sorted({external[w.space].backing for w in program.windows}))
    if (
        type(backing_origins) is not dict
        or set(backing_origins) != set(names)
        or any(type(n) is not int or n < 0 for n in backing_origins.values())
    ):
        raise ValueError("connected C backing origins must explicitly cover every physical backing")
    origins = dict(backing_origins)
    views = []
    for w in program.windows:
        name = external[w.space].backing
        offset = w.start - origins[name]
        if offset < 0 or max(offset, w.size, offset + w.size) > (1 << 64) - 1:
            raise ValueError("connected C window does not fit its explicit origin and uint64 domain")
        views.append((names.index(name), offset, w.size))

    implementations, records = [], []
    for index, invocation in enumerate(invocations):
        stage = emit_stage(invocation, f"{symbol}_stage_{index}")
        if type(stage) is not ConnectedCStage:
            raise ValueError("connected stage emitter must return ConnectedCStage")
        stage.validate(len(invocation.stage.inputs))
        program.ordered_invocations()
        if program.checked_identity != identity:
            raise ValueError("connected program changed during emission")
        implementations.append(stage)
        records.append(
            {
                "ordinal": invocation.stage.ordinal,
                "source_sha256": sha256(stage.source.encode()).hexdigest(),
                "effects": asdict(stage.effects),
                "window_binding_identity": stage.window_binding_identity,
                "input_alignments": stage.input_alignments,
                "output_alignment": stage.output_alignment,
            }
        )

    count = len(names)
    # Status -1/SIZE_MAX is argument refusal; a stage status preserves its ordinal.
    head = f"""#include <stdint.h>
#include <stddef.h>
#include <limits.h>
_Static_assert(CHAR_BIT == 8, "connected byte ABI requires 8-bit bytes");
_Static_assert(sizeof(uintptr_t) <= sizeof(uint64_t), "unsupported host pointer width");
typedef struct {{ uint8_t *base; size_t bytes; }} {symbol}_backing;
typedef struct {{ int status; size_t failed_stage; }} {symbol}_result;
"""
    lines = [
        f"{symbol}_result {symbol}(const {symbol}_backing *b, size_t count) {{",
        f"  {symbol}_result bad = {{-1, SIZE_MAX}};",
        f"  if (!b || count != {count}) return bad;",
        "  uintptr_t meta = (uintptr_t)b;",
        f"  if (meta > UINTPTR_MAX - sizeof(*b) * {count}) return bad;",
        f"  uint8_t *base[{count}]; size_t bytes[{count}]; uintptr_t start[{count}], end[{count}];",
        f"  for (size_t i=0; i<{count}; ++i) {{",
        "    base[i]=b[i].base; bytes[i]=b[i].bytes; start[i]=(uintptr_t)base[i];",
        "    if (!base[i] || bytes[i] > UINTPTR_MAX - start[i]) return bad;",
        "    end[i]=start[i]+bytes[i];",
        f"    if (start[i] < meta + sizeof(*b) * {count} && meta < end[i]) return bad;",
        "    for (size_t j=0; j<i; ++j)",
        "      if (start[i] < end[j] && start[j] < end[i]) return bad;",
        "  }",
    ]
    for i, (backing, offset, size) in enumerate(views):
        lines.extend(
            (
                f"  if (UINT64_C({offset}) > bytes[{backing}] ||",
                f"      UINT64_C({size}) > bytes[{backing}] - UINT64_C({offset})) return bad;",
            )
        )
    # Check every stage's pointer requirements before any stage can write.
    for invocation, implementation in zip(invocations, implementations, strict=True):
        ids = (*invocation.stage.inputs, invocation.stage.output)
        for i, alignment in zip(ids, (*implementation.input_alignments, implementation.output_alignment), strict=True):
            backing, offset, _ = views[i]
            lines.append(f"  if ((start[{backing}] + UINT64_C({offset})) % UINT64_C({alignment})) return bad;")
    for index, invocation in enumerate(invocations):
        n = len(invocation.stage.inputs)
        if not n:
            raise ValueError("connected C stages require at least one input")
        pointers = [f"base[{views[i][0]}] + (size_t)UINT64_C({views[i][1]})" for i in invocation.stage.inputs]
        sizes = [f"(size_t)UINT64_C({views[i][2]})" for i in invocation.stage.inputs]
        backing, offset, size = views[invocation.stage.output]
        lines.extend(
            (
                "  {",
                f"    const uint8_t *inputs[{n}] = {{{', '.join(pointers)}}};",
                f"    const size_t sizes[{n}] = {{{', '.join(sizes)}}};",
                f"    int status = {symbol}_stage_{index}(inputs, sizes, base[{backing}] + (size_t)UINT64_C({offset}), (size_t)UINT64_C({size}));",
                f"    if (status) {{ {symbol}_result failed = {{status, {index}}}; return failed; }}",
                "  }",
            )
        )
    lines.extend((f"  {symbol}_result ok = {{0, SIZE_MAX}}; return ok;", "}"))
    source = "\n".join((head, *(s.source for s in implementations), *lines)) + "\n"
    program.ordered_invocations()
    return ConnectedCArtifact(
        source,
        {
            "schema": "merlin.connected_host_c.v1",
            "program_identity": identity,
            "source_sha256": sha256(source.encode()).hexdigest(),
            "symbol": symbol,
            "backings": [{"name": name, "origin": origins[name]} for name in names],
            "views": [
                {"value": i, "backing_index": b, "offset": o, "bytes": n, "byte_order": program.byte_orders[i]}
                for i, (b, o, n) in enumerate(views)
            ],
            "live_outputs": list(program.region.outputs),
            "stages": records,
            "completion": "caller_asserted_synchronous_borrowed_stage_abi",
            "allocation": "caller_owned",
            "packing_or_byte_order_conversion": False,
            "execution_qualification": "not_established",
            "physical_savings_claimed": False,
        },
    )
