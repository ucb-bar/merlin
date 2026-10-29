"""Typed SSA composition through existing source-checked stage compilers.

This bridge preserves tensor identities across sequential runtime launches. It
is not an arbitrary graph optimizer, a target adapter or a runtime qualifier.
Pointwise stages reuse the ordinary native compiler. Mixed stages can use the
explicit synchronous buffer runtime; no cross-stage fusion is inferred.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod

from merlin.transforms.certificates import ir_digest
from merlin.verify.lean_contract import document_digest


@dataclass(frozen=True)
class RegionStage:
    operation: object
    ordinal: int
    kind: str
    inputs: tuple[int, ...]
    output: int


@dataclass(frozen=True)
class ConnectedRegion:
    source: object
    source_sha256: str
    stages: tuple[RegionStage, ...]
    values: tuple
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    lifetimes: tuple[tuple[int, int], ...]

    def revalidate(self):
        if ir_digest(self.source) != self.source_sha256:
            raise ValueError('connected region source changed; replan')
        current = tuple(self.source.walk())
        if any(stage.ordinal >= len(current) or current[stage.ordinal] is not stage.operation
               for stage in self.stages):
            raise ValueError('connected region operation ownership changed; replan')
        rebuilt = plan_connected_region(self.source, tuple(stage.operation for stage in self.stages))
        if rebuilt != self:
            raise ValueError('connected region dataflow or lifetimes changed; replan')

    def record(self):
        return {'source_ir_sha256': self.source_sha256,
            'stages': [{'ordinal': s.ordinal, 'kind': s.kind, 'inputs': s.inputs, 'output': s.output}
                       for s in self.stages],
            'values': [{'type': str(value.type), 'lifetime': life} for value, life in
                       zip(self.values, self.lifetimes, strict=True)],
            'inputs': self.inputs, 'outputs': self.outputs,
            'maturity': 'typed_source_plan_not_physical_residency'}


def plan_connected_region(module, operations):
    """Plan one ordered same-block region, retaining every external consumer.

    Source operand identity, never equal shape, connects stages. Initializers
    are excluded only after native pointwise admission proves they are unread.
    Unknown intervening operations refuse rather than disappearing. Inputs stay
    borrowed/read-only for the full region; externally observed outputs stay live.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import IntegerType, TensorType
    from merlin.compile.fusion_discovery import _structural_family
    from merlin.compile.scheduling.pointwise_source import read_pointwise_source

    operations = tuple(operations)
    members = tuple(module.walk())
    if not 2 <= len(operations) <= 64 or len(members) > 10000 or len(set(operations)) != len(operations):
        raise ValueError('connected region requires two to sixty-four distinct bounded source operations')
    ordinals = {op: i for i, op in enumerate(members)}
    block = operations[0].parent
    if block is None or any(op not in ordinals or op.parent is not block for op in operations):
        raise ValueError('connected region requires current operations from one source block')
    body = tuple(block.ops)
    positions = tuple(body.index(op) for op in operations)
    if positions != tuple(sorted(positions)):
        raise ValueError('connected region must retain original program order')
    # Admit only source-proven pure zero fills used by selected contractions.
    # This permits dense -> pointwise -> dense regions without treating an
    # arbitrary intervening operation as an ignorable initializer.
    initializers = set()
    from merlin.compile.scheduling.dense_source import read_zero_seeded_integer_matmul
    for op in operations:
        if _structural_family(op) == 'contraction':
            dtypes = tuple(str(value.type.element_type) for value in (*op.operands[:2], op.results[0]))
            read_zero_seeded_integer_matmul(op, dtypes=dtypes)
            seed = op.operands[2].owner
            if getattr(seed, 'name', None) == 'linalg.fill' and all(
                    use.operation in operations for value in seed.results for use in value.uses):
                initializers.add(seed)
    for op in body[positions[0]:positions[-1] + 1]:
        if op not in operations and op not in initializers and not (isinstance(op, (arith.ConstantOp, tensor.EmptyOp))
                                         or op.name == 'tensor.splat'):
            raise ValueError('connected region crosses an unmodeled operation or observer')
    values, inputs, stages = [], [], []
    created, last = [], []
    for index, op in enumerate(operations):
        kind = _structural_family(op)
        if kind == 'elementwise_map':
            read_pointwise_source(op, max_elements=64 << 20, max_operations=128, max_ir_bytes=65536)
            operands = tuple(op.inputs)
        elif kind == 'contraction':
            operands = tuple(op.operands[:2])
        else:
            raise ValueError('connected region needs native pointwise or declared contraction source')
        if len(op.results) != 1:
            raise ValueError('connected stage needs one explicit tensor result')
        used = []
        for value in (*operands, op.results[0]):
            typ = value.type
            if (not isinstance(typ, TensorType) or not isinstance(typ.element_type, IntegerType)
                    or typ.element_type.width.data % 8 or any(n <= 0 for n in typ.get_shape())
                    or prod(typ.get_shape()) > 64 << 20):
                raise ValueError('connected tensor requires bounded static byte-addressable integer storage')
            if value not in values:
                if value.owner in operations and value is not op.results[0]:
                    raise ValueError('connected input is not available in source program order')
                values.append(value)
                external = value is not op.results[0]
                created.append(-1 if external else index)
                last.append(len(operations) if external else index)
                if external:
                    inputs.append(len(values) - 1)
            ident = values.index(value)
            last[ident] = max(last[ident], index)
            used.append(ident)
        stages.append(RegionStage(op, ordinals[op], kind, tuple(used[:-1]), used[-1]))
    outputs = tuple(stage.output for stage in stages
                    if any(use.operation not in operations for use in values[stage.output].uses))
    for ident in outputs:
        last[ident] = len(operations)
    connected = {0}
    for _ in stages:
        for index, stage in enumerate(stages):
            for value in stage.inputs:
                producer = created[value]
                if producer >= 0 and (producer in connected or index in connected):
                    connected.update((producer, index))
    if not outputs or len(connected) != len(stages):
        raise ValueError('connected region needs one producer-consumer component and an externally visible output')
    return ConnectedRegion(module, ir_digest(module), tuple(stages), tuple(values), tuple(inputs), outputs,
                           tuple(zip(created, last, strict=True)))


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
    from merlin.compile.scheduling.pointwise_source import PointwiseBindings, compile_source_pointwise
    from merlin.compile.scheduling.program_limits import AddressWindow, Continuation
    from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts

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
            from merlin.compile.scheduling.dense_source import schedule_source_dense
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
        from merlin.compile.connected_runtime import ConnectedProgram
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
