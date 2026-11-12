"""Read-only identities for exact immutable tensor preparation opportunities.

Canonical identities retain the actual SSA root, logical dtype/layout and exact
static coordinates. They never use addresses, captured contents, labels or shape
alone. Analysis emits no packing, moves no operation and grants no external-call
purity or physical lifetime permission. A preparation emitter must additionally
bind its format, numerical/effect, storage and complete consumer lifetime proofs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from merlin.compiler.lower.model.ordered_fma_groups import _context_snapshot, _snapshot


@dataclass(frozen=True)
class TensorReadView:
    value: object
    root: object
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    _types: tuple = field(repr=False)
    _operations: tuple = field(repr=False)
    _contexts: tuple = field(repr=False)
    _block_owner: tuple = field(repr=False)

    @property
    def identity(self):
        """In-memory typed identity; SSA roots are never serialized as pointers."""
        return (self.root, self.value.type, self.offsets, self.sizes, self.strides)


def _tensor_type(value):
    from xdsl.dialects.builtin import NoneAttr, TensorType

    typ = value.type
    if (
        not isinstance(typ, TensorType)
        or not isinstance(typ.encoding, NoneAttr)
        or any(n < 0 or n >= 1 << 63 for n in typ.get_shape())
    ):
        raise ValueError("tensor preparation requires a static unencoded tensor")
    return typ


def canonical_tensor_read_view(value) -> TensorReadView:
    """Compose only exact same-rank static extract_slice coordinates.

    Unknown tensor producers remain distinct roots. Dynamic/rank-reducing slices,
    layouts and memrefs refuse; casts/reshapes are not guessed through. Empty
    extents retain source coordinates without asserting a nonempty access.
    """
    from xdsl.dialects import tensor
    from xdsl.ir import Block

    typ = _tensor_type(value)
    rank = len(typ.get_shape())
    offsets, strides = (0,) * rank, (1,) * rank
    sizes = tuple(typ.get_shape())
    current, operations, types = value, [], []
    while isinstance(current.owner, tensor.ExtractSliceOp):
        operation = current.owner
        source_type = _tensor_type(operation.source)
        out_type = _tensor_type(current)
        shape = tuple(source_type.get_shape())
        local_offsets = tuple(operation.static_offsets.get_values())
        local_sizes = tuple(operation.static_sizes.get_values())
        local_strides = tuple(operation.static_strides.get_values())
        if (
            len(operation.operands) != 1
            or len(shape) != rank
            or tuple(out_type.get_shape()) != local_sizes
            or len(local_offsets) != rank
            or len(local_sizes) != rank
            or len(local_strides) != rank
            or source_type.element_type != out_type.element_type
            or any(
                o < 0 or n < 0 or s <= 0 or o > extent or n and o + (n - 1) * s >= extent
                for o, n, s, extent in zip(local_offsets, local_sizes, local_strides, shape, strict=True)
            )
        ):
            raise ValueError("tensor preparation has no exact static same-rank slice")
        offsets = tuple(o + q * s for o, q, s in zip(local_offsets, offsets, local_strides, strict=True))
        strides = tuple(a * b for a, b in zip(strides, local_strides, strict=True))
        if any(n >= 1 << 63 for n in (*offsets, *strides)):
            raise ValueError("tensor preparation coordinates exceed signed index range")
        operations.append(operation)
        types.extend((current, operation.source))
        current = operation.source
    _tensor_type(current)
    types.extend((value, current))
    root_owner = current.owner
    if not isinstance(root_owner, Block):
        operations.extend(root_owner.walk())
    operations = tuple(dict.fromkeys(operations))
    contexts = {}
    for operation in operations:
        parent = operation.parent_op()
        while parent is not None:
            contexts[parent] = _context_snapshot(parent)
            parent = parent.parent_op()
    if isinstance(root_owner, Block):
        region = root_owner.parent
        owner = region.parent if region is not None else None
        block_owner = (root_owner, region, owner, tuple(root_owner.args))
        parent = owner
        while parent is not None:
            contexts[parent] = _context_snapshot(parent)
            parent = parent.parent_op()
    else:
        block_owner = ()
    return TensorReadView(
        value,
        current,
        offsets,
        sizes,
        strides,
        tuple((item, item.type) for item in dict.fromkeys(types)),
        tuple(_snapshot(operation) for operation in operations),
        tuple(contexts.values()),
        block_owner,
    )


def validate_tensor_read_view(view: TensorReadView) -> None:
    """Refuse changed root/view types, source operations or ownership context."""
    if (
        any(value.type != typ for value, typ in view._types)
        or any(_snapshot(w.operation) != w for w in view._operations)
        or any(_context_snapshot(w.operation) != w for w in view._contexts)
    ):
        raise ValueError("tensor preparation source changed after analysis")
    if view._block_owner:
        block, region, owner, args = view._block_owner
        if block.parent is not region or region is not None and region.parent is not owner or tuple(block.args) != args:
            raise ValueError("tensor preparation block ownership changed after analysis")
    actual = canonical_tensor_read_view(view.value)
    if actual.identity != view.identity:
        raise ValueError("tensor preparation coordinates changed after analysis")


@dataclass(frozen=True)
class TensorPreparationRequest:
    value: object
    consumer: object
    format_sha256: str


@dataclass(frozen=True)
class TensorPreparationOpportunity:
    format_sha256: str
    views: tuple[TensorReadView, ...]
    consumers: tuple
    owner_block: object
    _consumer_witnesses: tuple = field(repr=False)


def find_tensor_preparation_opportunities(requests) -> tuple[TensorPreparationOpportunity, ...]:
    """Group exact equivalent read views in one explicit sequential source block.

    Caller supplies preparation format identities. Consumers must directly read
    their requested tensors; this proves SSA equivalence and candidate placement,
    not preservation by an opaque implementation or prepared-buffer lifetime.
    """
    from xdsl.ir import Block

    grouped = {}
    for request in requests:
        pin = request.format_sha256
        if not isinstance(pin, str) or len(pin) != 64 or any(ch not in "0123456789abcdef" for ch in pin):
            raise ValueError("tensor preparation requires an explicit format identity")
        consumer = request.consumer
        if consumer.parent is None or request.value not in consumer.operands:
            raise ValueError("tensor preparation requires an actual direct consumer")
        view = canonical_tensor_read_view(request.value)
        block = consumer.parent
        positions = {operation: index for index, operation in enumerate(block.ops)}
        for value, _ in view._types:
            owner = value.owner
            if isinstance(owner, Block):
                if owner is not block:
                    raise ValueError("tensor preparation root is outside the sequential consumer block")
            elif owner.parent is not block or positions[owner] >= positions[consumer]:
                raise ValueError("tensor preparation root/view does not dominate the consumer")
        grouped.setdefault((pin, view.identity, block), []).append((view, consumer))
    return tuple(
        TensorPreparationOpportunity(
            pin,
            tuple(view for view, _ in items),
            tuple(consumer for _, consumer in items),
            block,
            tuple(_snapshot(consumer) for _, consumer in items),
        )
        for (pin, _, block), items in grouped.items()
        if len(items) > 1
    )


def validate_tensor_preparation_opportunity(opportunity) -> None:
    for view in opportunity.views:
        validate_tensor_read_view(view)
    if any(_snapshot(w.operation) != w for w in opportunity._consumer_witnesses):
        raise ValueError("tensor preparation consumer changed after analysis")
    if any(consumer.parent is not opportunity.owner_block for consumer in opportunity.consumers):
        raise ValueError("tensor preparation consumer ownership changed after analysis")
    current = find_tensor_preparation_opportunities(
        tuple(
            TensorPreparationRequest(view.value, consumer, opportunity.format_sha256)
            for view, consumer in zip(opportunity.views, opportunity.consumers, strict=True)
        )
    )
    if len(current) != 1 or current[0].views[0].identity != opportunity.views[0].identity:
        raise ValueError("tensor preparation equivalence changed after analysis")
