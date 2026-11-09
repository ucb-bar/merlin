"""Compile connected source stages through checked native compilers and explicit continuations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod

from merlin.compiler.partition.connected import (
    ConnectedRegion as ConnectedRegion, RegionStage as RegionStage,
    plan_connected_region as plan_connected_region,
)
from merlin.verification.lean_contract import document_digest


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
            raise ValueError('connected runtime continuation changed; recompile')
        if len(self.products) != len(self.region.stages) or any(
            product.source is not self.region.source or product.operation is not stage.operation
            for product, stage in zip(self.products, self.region.stages, strict=True)
        ):
            raise ValueError('connected native products no longer cover their original stages')
        actual = tuple(product.images() for product in self.products)
        if actual != self.checked_images:
            raise ValueError('connected stage images changed; recompile')
        return actual


def _compile_connected(region, *, windows, byte_orders, selected_contracts,
                       continuation, stage_options, dense_options=None, dense_contracts=None):
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
    from merlin.compiler.schedule.pointwise_source import PointwiseBindings
    from merlin.compiler.schedule.pointwise_source import compile_source_pointwise
    from merlin.compiler.schedule.program_limits import AddressWindow
    from merlin.compiler.schedule.program_limits import Continuation
    from merlin.compiler.schedule.selected_contracts import SelectedSchedulingContracts

    if type(region) is not ConnectedRegion or not callable(stage_options):
        raise ValueError('connected compilation requires a current typed plan and target stage selector')
    region.revalidate()
    if dense_options is None and any(stage.kind != 'elementwise_map' for stage in region.stages):
        raise ValueError('counted dense composition needs a multi-stage pointer/launch ABI; source plan retained')
    if any(stage.kind == 'contraction' and len(set(stage.inputs)) != len(stage.inputs)
           for stage in region.stages):
        raise ValueError('dense pointer arguments alias; native dense route requires disjoint external buffers')
    if (type(selected_contracts) is not SelectedSchedulingContracts or selected_contracts.storage is None
            or type(continuation) is not Continuation or type(windows) is not tuple
            or len(windows) != len(region.values) or any(type(w) is not AddressWindow for w in windows)
            or type(byte_orders) is not tuple or len(byte_orders) != len(windows)
            or any(order not in ('little', 'big') for order in byte_orders)):
        raise ValueError('connected compilation needs exact storage, byte orders and sequential runtime contract')
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
        if (window.size != expected or bound is None or window.start < bound.start
                or window.start + window.size > bound.start + bound.size):
            raise ValueError('connected tensor binding differs from complete type or selected external storage')
        physical.append((bound.backing, window.start, window.start + window.size))
    for i, (space, start, end) in enumerate(physical):
        for j, (other, low, high) in enumerate(physical[:i]):
            live = (region.lifetimes[i][0] <= region.lifetimes[j][1]
                    and region.lifetimes[j][0] <= region.lifetimes[i][1])
            if live and space == other and start < high and low < end:
                raise ValueError('connected live tensors alias; cannot retain original SSA values')
    for boundary in range(len(region.stages) - 1):
        required = {window.space for window, (first, last) in zip(windows, region.lifetimes, strict=True)
                    if first <= boundary < last}
        if not required <= set(continuation.preserved_spaces):
            raise ValueError('connected runtime does not preserve all live tensors across launches')
    if dense_options is not None and not callable(dense_options):
        raise ValueError('dense stage selector must be callable')
    if dense_contracts is None:
        dense_contracts = selected_contracts
    dense_selection = None
    if any(stage.kind == 'contraction' for stage in region.stages):
        if type(dense_contracts) is not SelectedSchedulingContracts or dense_contracts.dense is None:
            raise ValueError('mixed region requires explicit selected dense contracts')
        dense_contracts.revalidate()
        dense_selection = document_digest(dense_contracts.record())
    products = []
    for stage in region.stages:
        bindings = PointwiseBindings(tuple(windows[i] for i in stage.inputs), windows[stage.output],
            tuple(byte_orders[i] for i in (*stage.inputs, stage.output)), True)
        owned_bindings = asdict(bindings)
        selector = dense_options if stage.kind == 'contraction' else stage_options
        options = selector(region, stage, bindings)
        if asdict(bindings) != owned_bindings:
            raise ValueError('connected tensor bindings changed during target selection')
        if type(options) is not dict or set(options) & {'module', 'operation', 'bindings', 'selected_contracts',
                'external_buffers_disjoint', 'output_operation', 'output_operations', 'readout_qualification_identity'}:
            raise ValueError('target stage selector cannot override source-owned bindings or contracts')
        region.revalidate()
        if (document_digest([asdict(w) for w in windows]) != bindings_sha
                or document_digest(selected_contracts.record()) != selection_sha
                or document_digest(asdict(continuation)) != continuation_sha):
            raise ValueError('connected contracts changed during target selection')
        if stage.kind == 'contraction':
            from merlin.compiler.schedule.dense_source import schedule_source_dense
            dense_contracts.revalidate()
            if document_digest(dense_contracts.record()) != dense_selection:
                raise ValueError('dense contracts changed during target selection')
            product = schedule_source_dense(region.source, stage.operation,
                selected_contracts=dense_contracts, external_buffers_disjoint=True, **options)
            if product is None:
                raise ValueError('no legal dense stage realization; retain unfused source')
        else:
            product = compile_source_pointwise(region.source, stage.operation,
                bindings=bindings, selected_contracts=selected_contracts, **options)
        products.append(product)
        if asdict(bindings) != owned_bindings:
            raise ValueError('connected tensor bindings changed during stage compilation')
    if dense_options is not None:
        from merlin.runtime.connected import ConnectedProgram
        return ConnectedProgram.checked(region, tuple(products), windows, byte_orders,
            selected_contracts, continuation)
    images = tuple(product.images() for product in products)
    if sum(map(len, images)) > continuation.max_segments:
        raise ValueError('connected executable exceeds runtime launch budget')
    result = ConnectedCompilation(region, tuple(products), continuation, continuation_sha, images,
        {'source': region.record(), 'bindings': [asdict(w) for w in windows],
         'runtime_continuation': asdict(continuation), 'selected_contracts_sha256': selection_sha,
         'stage_launches': [len(parts) for parts in images], 'actual_encoded_bytes': sum(len(b) for p in images for b in p),
         'stage_conformance': [p.conformance for p in products],
         'qualification': 'native source and final-byte conformance; runtime execution remains unqualified',
         'maturity': 'executable_unfused_composition_not_cross_stage_fusion',
         'residency': 'selected_runtime_buffer_persistence_only; no on-chip placement inferred'})
    result.images()
    return result


def compile_connected_pointwise(region, *, windows, byte_orders, selected_contracts,
                                continuation, stage_options):
    """Compile source-checked finite pointwise images with retained SSA boundaries."""
    return _compile_connected(region, windows=windows, byte_orders=byte_orders,
        selected_contracts=selected_contracts, continuation=continuation, stage_options=stage_options)


def compile_connected_region(region, *, windows, byte_orders, selected_contracts,
                             continuation, stage_options, dense_options, dense_contracts=None):
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
        raise ValueError('mixed composition requires explicit dense stage selection')
    return _compile_connected(region, windows=windows, byte_orders=byte_orders,
        selected_contracts=selected_contracts, continuation=continuation,
        stage_options=stage_options, dense_options=dense_options, dense_contracts=dense_contracts)
