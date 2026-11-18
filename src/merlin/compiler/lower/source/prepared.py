"""Read-only identities for exact immutable tensor preparation opportunities.

Canonical identities retain the actual SSA root, logical dtype/layout and exact
static coordinates. They never use addresses, captured contents, labels or shape
alone. Analysis emits no packing, moves no operation and grants no external-call
purity or physical lifetime permission. A preparation emitter must additionally
bind its format, numerical/effect, storage and complete consumer lifetime proofs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects import arith, builtin, func, scf
from xdsl.dialects.linalg.ops import GenericOp, IndexOp, LinalgStructuredOperation, ReduceOp, YieldOp
from xdsl.traits import get_effects

from merlin.compiler.dialects.xdsl_context import Float8E4M3FNType
from merlin.compiler.lower.source.ordered_fma import _context_snapshot, _snapshot

# ---- tensor_preparation_identity ---------------------------------------------------------------


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


# ---- prepared_operand_effects ------------------------------------------------------------------
# Conservative typed source immutability checks for prepared operand lifetimes.
#
# This proves properties of the original tensor program. An external physical
# producer/consumer still needs its independently validated borrowed-buffer proof;
# function names or proof hashes alone do not supply that permission.


def validate_pure_tensor_function(function, functions, *, active=(), allow_named_linalg=False):
    """Require a complete nonrecursive tensor/scalar body with no opaque calls.

    Tensor linalg operations have value semantics when every shaped operand is a
    tensor and their scalar regions are pure. Buffer linalg, unknown effects,
    address conversion, external functions, recursion and captured buffers refuse.
    """
    if not isinstance(function, func.FuncOp) or len(function.body.blocks) != 1:
        raise ValueError("prepared source requires a complete single-block tensor function")
    if type(allow_named_linalg) is not bool:
        raise ValueError("named tensor operation admission requires an explicit boolean")
    if function in active:
        raise ValueError("recursive prepared source effects are unknown")
    for value in (*function.body.block.args, *function.function_type.outputs):
        _value_type(value.type if hasattr(value, "type") else value)
    for operation in function.body.block.ops:
        _operation(operation, functions, (*active, function), allow_named_linalg=allow_named_linalg)


def _value_type(typ):
    if isinstance(typ, builtin.TensorType):
        if not isinstance(typ.encoding, builtin.NoneAttr):
            raise ValueError("prepared source tensor encoding has unknown effects")
        _value_type(typ.element_type)
    elif type(typ) is Float8E4M3FNType:
        # This exact shared scalar is an immutable value. Its encoding,
        # numerical/domain policy and physical execution remain separate claims.
        pass
    elif not isinstance(typ, (builtin.IntegerType, builtin.IndexType, builtin.AnyFloat)):
        raise ValueError("prepared source may not expose buffer or address values")


def _operation(operation, functions, active, *, allow_named_linalg=False):
    for value in (*operation.operands, *operation.results):
        _value_type(value.type)
    if isinstance(operation, func.CallOp):
        if operation.callee.nested_references:
            raise ValueError("prepared source call has unsupported nested symbol resolution")
        parent = operation.parent_op()
        while parent is not None and not isinstance(parent, func.FuncOp):
            parent = parent.parent_op()
        if parent not in functions.values():
            raise ValueError("prepared source call is outside the selected root function scope")
        callee = functions.get(operation.callee.root_reference.data)
        if callee is None:
            raise ValueError("prepared source call has unknown external effects")
        validate_pure_tensor_function(callee, functions, active=active, allow_named_linalg=allow_named_linalg)
        return
    if type(operation) in (scf.ForOp, scf.IfOp, scf.YieldOp):
        if operation.properties or any(not key.startswith("prov.") for key in operation.attributes):
            raise ValueError("prepared source control operation has unknown execution attributes")
        # These exact serial control operations have only the effects of their
        # complete nested bodies, checked below. No generic region wrapper or
        # unknown loop inherits this fact. Verification checks carried types,
        # bounds operands and terminator structure before recursive inspection.
        operation.verify()
    elif isinstance(operation, (func.ReturnOp, YieldOp)):
        return
    elif isinstance(operation, (GenericOp, ReduceOp)) or (
        allow_named_linalg and isinstance(operation, LinalgStructuredOperation)
    ):
        if "library_call" in operation.attributes or "library_call" in operation.properties:
            raise ValueError("prepared source library override has unknown external effects")
        # The only admitted effect-free structured special case: immutable
        # tensor values with pure scalar regions. No buffer outputs allowed.
        if not operation.results or any(not isinstance(v.type, builtin.TensorType) for v in operation.results):
            raise ValueError("prepared source structured operation requires tensor results")
    elif type(operation) is IndexOp:
        # This xDSL version gives linalg.index its parent constraint but no
        # effect trait. It reads only the enclosing generic's logical iteration
        # coordinate. Admit that exact typed operation and a valid axis; no
        # opaque operation or buffer obtains a purity assumption from its name.
        parent = operation.parent_op()
        if (
            not isinstance(parent, GenericOp)
            or not parent.results
            or any(not isinstance(value.type, builtin.TensorType) for value in parent.operands)
            or any(not key.startswith("prov.") for key in operation.attributes)
            or set(operation.properties) != {"dim"}
            or not 0 <= operation.dim.value.data < len(parent.iterator_types)
        ):
            raise ValueError("prepared source index requires an ordinary tensor generic and a valid axis")
        operation.verify()
    elif (
        isinstance(operation, arith.ExtSIOp)
        and not operation.properties
        and all(key.startswith("prov.") for key in operation.attributes)
    ):
        # The supported xDSL version omits Pure on this standard typed integer
        # widening operation. Its exact sign extension has no memory effects;
        # no unknown arithmetic operation inherits this specific fact.
        pass
    elif get_effects(operation) != set():
        raise ValueError("prepared source operation has unknown or nonempty effects")
    for region in operation.regions:
        for block in region.blocks:
            for argument in block.args:
                _value_type(argument.type)
            for nested in block.ops:
                _operation(nested, functions, active, allow_named_linalg=allow_named_linalg)


def validate_prepared_source_lifetime(opportunity, module):
    """Check every operation from first borrow through the last source consumer.

    This is deliberately conservative: even an unrelated unknown memory effect
    refuses. The result grants no physical cached-storage or external ABI proof.
    """
    validate_tensor_preparation_opportunity(opportunity)
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    operations = tuple(opportunity.owner_block.ops)
    positions = [operations.index(call) for call in opportunity.consumers]
    if positions != sorted(set(positions)):
        raise ValueError("prepared source consumers are not in unique source order")
    for operation in operations[positions[0] : positions[-1] + 1]:
        _operation(operation, functions, ())


# ---- prepared_operand_owner --------------------------------------------------------------------
# Explicit, invocation-local ownership for immutable prepared tensor operands.
#
# This is a proof/placement prototype, not a cache or permission to delete source
# operations. A lowering must validate the representation producer and borrowed
# consumer effects independently before using the returned storage plan. Source
# SSA equivalence alone never establishes physical buffer lifetime or purity.


def _pin(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("prepared representation requires a complete proof identity")


@dataclass(frozen=True)
class PreparedStorageSpan:
    """Bytes and alignment are derived by the selected representation producer."""

    name: str
    bytes: int
    alignment: int

    def __post_init__(self):
        if not self.name or type(self.bytes) is not int or not 0 < self.bytes < 1 << 63:
            raise ValueError("invalid prepared storage extent")
        if type(self.alignment) is not int or not 0 < self.alignment < 1 << 63 or self.alignment & (self.alignment - 1):
            raise ValueError("prepared storage alignment must be a power of two")


@dataclass(frozen=True)
class PreparedRepresentation:
    """Format includes axis order, dtype, radix, scales and reconstruction order.

    Numeric and effect evidence must cover source RNE, finite admission,
    initialization, immutable prepared/source spans, and synchronous noescape
    consumption. These pins name separately checked evidence; they are not a
    substitute for checking it in the normal compiler/provider binding.
    """

    format_sha256: str
    numeric_sha256: str
    effects_sha256: str
    spans: tuple[PreparedStorageSpan, ...]

    def __post_init__(self):
        for pin in (self.format_sha256, self.numeric_sha256, self.effects_sha256):
            _pin(pin)
        if not self.spans or len({span.name for span in self.spans}) != len(self.spans):
            raise ValueError("prepared spans must have distinct ownership names")

    def layout(self):
        offset, result = 0, []
        for span in self.spans:
            offset = (offset + span.alignment - 1) & -span.alignment
            end = offset + span.bytes
            if end >= 1 << 63:
                raise ValueError("prepared storage exceeds signed index range")
            result.append((span.name, offset, end))
            offset = end
        return tuple(result), offset, max(span.alignment for span in self.spans)


class PreparedOperandOwner:
    """Linear proof-state model for one owner and one source invocation epoch.

    No addresses or tensor values are used as identity keys. Each consumer must
    present its actual source operation and the original epoch capability.
    Invalidation is terminal; a later invocation requires a fresh owner.
    """

    def __init__(self, opportunity, representation, *, validate_effects):
        validate_tensor_preparation_opportunity(opportunity)
        if opportunity.format_sha256 != representation.format_sha256:
            raise ValueError("source and physical preparation formats disagree")
        positions = {op: i for i, op in enumerate(opportunity.owner_block.ops)}
        ordered = [positions[op] for op in opportunity.consumers]
        if ordered != sorted(set(ordered)):
            raise ValueError("prepared consumers must be unique and in source order")
        # The binding-specific validator must actually check provider evidence.
        # Refusal propagates; no bool option silently authorizes purity.
        if validate_effects(opportunity, representation) is not None:
            raise ValueError("effect validator must raise on refusal, not return a permission flag")
        self.opportunity = opportunity
        self.representation = representation
        self.layout, self.capacity, self.alignment = representation.layout()
        self._block = tuple(_snapshot(op) for op in opportunity.owner_block.ops)
        self._epoch = object()
        self._next = 0
        self._active = True
        self._storage_binding = None

    @property
    def epoch(self):
        return self._epoch

    def invalidate(self):
        self._active = False

    def consume(self, consumer, *, epoch, representation, capacity, alignment, spans, storage_base, live_read_ranges):
        if not self._active or epoch is not self._epoch:
            raise ValueError("stale prepared owner epoch")
        try:
            validate_tensor_preparation_opportunity(self.opportunity)
            if tuple(_snapshot(op) for op in self.opportunity.owner_block.ops) != self._block:
                raise ValueError("prepared owner source schedule changed")
            if representation != self.representation:
                raise ValueError("prepared representation changed")
            if consumer is not self.opportunity.consumers[self._next]:
                raise ValueError("prepared consumer order or ownership mismatch")
            if (
                type(capacity) is not int
                or capacity < self.capacity
                or type(alignment) is not int
                or alignment < self.alignment
                or alignment % self.alignment
            ):
                raise ValueError("prepared allocation is insufficient")
            if tuple(spans) != self.layout:
                raise ValueError("prepared spans overlap or differ from proved layout")
            if (
                type(storage_base) is not int
                or storage_base <= 0
                or storage_base % self.alignment
                or storage_base + capacity >= 1 << 64
            ):
                raise ValueError("prepared allocation address is invalid")
            live_read_ranges = tuple(live_read_ranges)
            if not live_read_ranges:
                raise ValueError("source allocation disjointness is unknown")
            for lo, hi in live_read_ranges:
                if type(lo) is not int or type(hi) is not int or not 0 < lo < hi < 1 << 64:
                    raise ValueError("source allocation interval is invalid")
                if storage_base < hi and lo < storage_base + capacity:
                    raise ValueError("prepared allocation aliases live source storage")
            binding = (storage_base, capacity, alignment, live_read_ranges)
            if self._storage_binding is not None and binding != self._storage_binding:
                raise ValueError("prepared storage owner changed within epoch")
            self._storage_binding = binding
        except Exception:
            self.invalidate()
            raise
        self._next += 1
        if self._next == len(self.opportunity.consumers):
            self.invalidate()
        return self.layout


# ---- prepared_operand_schedule -----------------------------------------------------------------
# Explicit preparation and compiler-issued borrow leases in a private call tree.
#
# The source plan is captured before writer replacement. Installation requires a
# separate physical producer/consumer validator; the source tensor proof alone
# never authorizes an external implementation or pointer-keyed cache.


@dataclass(frozen=True)
class PreparedBorrowPlan:
    sequences: tuple
    source_arguments: tuple
    source_operations: tuple
    owner: object
    capacity: int
    alignment: int
    representation: object
    shared_arguments: tuple
    source_contexts: tuple


def plan_prepared_borrows(module, bundles, representation):
    """Bundle equivalent immutable operand views with equal ordered consumers."""
    from xdsl.dialects import func

    if not isinstance(representation, PreparedRepresentation):
        raise ValueError("prepared schedule requires a typed representation")
    bundles = tuple(tuple(bundle) for bundle in bundles)
    if not bundles or any(not bundle for bundle in bundles):
        raise ValueError("prepared schedule requires explicit source view bundles")
    sequences, arguments, occupied, block = [], [], set(), None
    shared_arguments = []
    for bundle in bundles:
        calls = tuple(bundle[0].consumers)
        if len(calls) < 2 or any(tuple(item.consumers) != calls for item in bundle):
            raise ValueError("prepared source bundle has unequal consumers")
        for opportunity in bundle:
            if opportunity.format_sha256 != representation.format_sha256:
                raise ValueError("prepared source bundle format differs from representation")
            validate_prepared_source_lifetime(opportunity, module)
        if any(not isinstance(call, func.CallOp) or call in occupied for call in calls):
            raise ValueError("prepared source calls must be distinct direct calls")
        if block is None:
            block = calls[0].parent
        if any(call.parent is not block for call in calls):
            raise ValueError("prepared sequences require one direct caller block")
        shared = []
        for opportunity in bundle:
            indices = [
                tuple(i for i, value in enumerate(call.arguments) if value is view.value)
                for call, view in zip(opportunity.consumers, opportunity.views, strict=True)
            ]
            if not indices[0] or any(index != indices[0] for index in indices):
                raise ValueError("prepared read positions differ across consumers")
            for index in indices[0]:
                if index not in shared:
                    shared.append(index)
        shared_arguments.append(tuple(shared))
        sequences.append(calls)
        arguments.extend((call, tuple(call.arguments)) for call in calls)
        occupied.update(calls)
    owner = block.parent_op()
    if not isinstance(owner, func.FuncOp) or getattr(owner.sym_visibility, "data", None) == "private":
        raise ValueError("prepared allocation requires the public invocation owner")
    positions = {op: i for i, op in enumerate(block.ops)}
    spans = sorted((positions[calls[0]], positions[calls[-1]]) for calls in sequences)
    if any(a[1] >= b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("prepared owner epochs must not overlap")
    _, payload, alignment = representation.layout()
    capacity = payload + len(sequences)  # disjoint, address-only epoch tokens
    if capacity + alignment - 1 >= 1 << 63:
        raise ValueError("prepared pool capacity overflows")
    contexts, parent = [], owner
    while parent is not None:
        contexts.append(_context_snapshot(parent))
        parent = parent.parent_op()
    return PreparedBorrowPlan(
        tuple(sequences),
        tuple(arguments),
        tuple(_snapshot(op) for op in block.ops),
        owner,
        capacity,
        alignment,
        representation,
        tuple(shared_arguments),
        tuple(contexts),
    )


def _validate_owner_pool_insertions(plan, current, report):
    """Only accept the exact private byte pools produced by workspace pooling.

    Checking retained operations alone misses newly inserted effects. Bind every
    added operation to a declared pool and all of its actual uses; even unrelated
    pure additions refuse here instead of extending the retained source proof.
    """
    from xdsl.dialects import memref
    from xdsl.dialects.builtin import DYNAMIC_INDEX, MemRefType, i8

    from merlin.compiler.lower.host.writers import PrivateWorkspaceContract, validate_private_workspaces

    contracts = tuple(PrivateWorkspaceContract(**item) for item in report.get("private_workspaces", ()))
    validate_private_workspaces(contracts)
    originals = {w.operation for w in plan.source_operations}
    added = set(current) - originals
    calls = tuple(call for sequence in plan.sequences for call in sequence)
    arguments = dict(plan.source_arguments)
    if any(len(call.arguments) != len(arguments[call]) + len(contracts) for call in calls):
        raise ValueError("prepared pooled operand count differs from declared workspace")
    positions = {op: index for index, op in enumerate(current)}
    first = min(positions[call] for call in calls)
    allowed = set()
    for index, contract in enumerate(contracts):
        value = calls[0].arguments[len(arguments[calls[0]]) + index]
        cast = value.owner
        if not isinstance(cast, memref.CastOp) or cast not in added:
            raise ValueError("prepared workspace requires a newly inserted pool cast")
        allocation = cast.source.owner
        uses = {(call, len(arguments[call]) + index) for call in calls}
        if (
            not isinstance(allocation, memref.AllocOp)
            or allocation not in added
            or allocation.operands
            or allocation.memref.type != MemRefType(i8, [contract.bytes])
            or allocation.alignment is None
            or allocation.alignment.value.data != contract.alignment
            or cast.dest.type != MemRefType(i8, [DYNAMIC_INDEX])
            or {(use.operation, use.index) for use in allocation.memref.uses} != {(cast, 0)}
            or {(use.operation, use.index) for use in value.uses} != uses
            or positions[cast] != positions[allocation] + 1
            or positions[cast] >= first
            or allocation in allowed
            or cast in allowed
        ):
            raise ValueError("prepared pool ownership, extent or use differs from workspace contract")
        allowed.update((allocation, cast))
    if added != allowed:
        raise ValueError("prepared schedule contains an unproved owner operation")


def install_prepared_borrows(
    module, plan, writer_reports, *, prepare_symbol, borrowed_symbol, validate_physical, materialize_shared_views=False
):
    """Emit one pool, explicit prepare calls and epoch/ordinal arguments.

    Call only after ordinary writer installation and workspace pooling. The
    physical bridge must use the owner pool's final address-only epoch slots,
    initialize validity on every prepare (including refusal), borrow without
    escape, and run the retained original source on failed preparation/consume.
    """
    from xdsl.dialects import arith, bufferization, func, memref
    from xdsl.dialects.builtin import (
        DYNAMIC_INDEX,
        ArrayAttr,
        DictionaryAttr,
        FunctionType,
        MemRefType,
        StringAttr,
        SymbolRefAttr,
        UnitAttr,
        i8,
        i64,
    )
    from xdsl.ir import Block, Region

    from merlin.compiler.lower.host.writers import _call_graph, _common_owner, _containing_function

    if type(materialize_shared_views) is not bool:
        raise ValueError("prepared view materialization must be explicit boolean")
    if not isinstance(plan, PreparedBorrowPlan):
        raise ValueError("prepared installation requires a retained source plan")
    _, payload, alignment = plan.representation.layout()
    if plan.capacity != payload + len(plan.sequences) or plan.alignment != alignment:
        raise ValueError("prepared allocation differs from retained representation")
    if plan.capacity + plan.alignment - 1 >= 1 << 63:
        raise ValueError("prepared aligned allocation exceeds signed index range")
    if any(_context_snapshot(w.operation) != w for w in plan.source_contexts):
        raise ValueError("prepared source context changed after analysis")
    module.verify()
    functions, callers = _call_graph(module)
    reports = tuple(writer_reports)
    if len(reports) != 1:
        raise ValueError("prepared physical binding requires one explicit writer ABI")
    report = reports[0]
    wrapper = functions.get(report.get("wrapper_symbol", report["symbol"]))
    borrowed = functions.get(report["borrowed_symbol"])
    if (
        wrapper is None
        or borrowed is None
        or borrowed.body.blocks
        or "llvm.emit_c_interface" not in borrowed.attributes
    ):
        raise ValueError("prepared binding requires the generated borrowed writer")
    helpers, owner, roots = _common_owner(module, [wrapper], callers)
    expected = {call for calls in plan.sequences for call in calls}
    if owner is not plan.owner or roots != expected:
        raise ValueError("prepared source call coverage changed")
    source = dict(plan.source_arguments)
    for call, arguments in source.items():
        if tuple(call.arguments[: len(arguments)]) != arguments:
            raise ValueError("prepared source input identity changed")
    # Ordinary writer pooling may append private operands and insert its pool.
    # Every other original owner operation must remain exactly source-identical.
    current = tuple(owner.body.block.ops)
    old_positions = []
    for witness in plan.source_operations:
        if witness.operation not in current:
            raise ValueError("prepared source schedule lost an operation")
        old_positions.append(current.index(witness.operation))
        current_witness = _snapshot(witness.operation)
        if witness.operation not in expected and current_witness != witness:
            raise ValueError("prepared source operation changed")
        if witness.operation in expected and any(
            getattr(current_witness, name) != getattr(witness, name)
            for name in ("parent", "result_types", "result_uses", "attributes", "properties", "region_contents")
        ):
            raise ValueError("prepared source call semantics changed")
    _validate_owner_pool_insertions(plan, current, report)
    if old_positions != sorted(old_positions):
        raise ValueError("prepared source operation order changed")
    if not prepare_symbol or not borrowed_symbol or prepare_symbol == borrowed_symbol:
        raise ValueError("explicit distinct prepared implementation symbols required")
    prepare_external = prepare_symbol + "_borrowed"
    symbols = {
        value.data
        for operation in module.body.block.ops
        if isinstance(value := operation.properties.get("sym_name", operation.attributes.get("sym_name")), StringAttr)
    }
    if any(name in symbols for name in (prepare_symbol, prepare_external, borrowed_symbol)):
        raise ValueError("prepared implementation symbol collision")
    borrowed_calls = callers[borrowed.sym_name.data]
    if len(borrowed_calls) != 1 or _containing_function(borrowed_calls[0]) is not wrapper:
        raise ValueError("prepared writer has an unbound borrowed call")
    first_arguments = source[plan.sequences[0][0]]
    tensor_types = tuple(value.type for value in first_arguments)
    if any(tuple(value.type for value in args) != tensor_types for args in source.values()):
        raise ValueError("prepared source tensor ABI differs between epochs")
    if validate_physical(plan, wrapper, borrowed, tuple(borrowed_calls)) is not None:
        raise ValueError("physical validator must raise on refusal, not return permission")
    # All preflight checks precede mutation. The source proof covers precisely
    # these immutable read positions. A fresh identity-layout tensor owns each
    # shared materialization through its last explicit consumer; no external
    # address equality or captured value selects sharing.
    materializations = 0
    if materialize_shared_views:
        from xdsl.dialects import tensor
        from xdsl.dialects.linalg.ops import CopyOp

        for calls, indices in zip(plan.sequences, plan.shared_arguments, strict=True):
            for index in indices:
                value = source[calls[0]][index]
                empty = tensor.EmptyOp([], value.type)
                copy = CopyOp([value], [empty.tensor], [value.type])
                owner.body.block.insert_ops_before([empty, copy], calls[0])
                for call in calls:
                    arguments = list(source[call])
                    arguments[index] = copy.results[0]
                    source[call] = tuple(arguments)
                    operands = list(call.arguments)
                    operands[index] = copy.results[0]
                    call.operands = operands
                materializations += 1
    # All preflight checks precede mutation.
    dynamic = MemRefType(i8, [DYNAMIC_INDEX])
    hidden_types = [dynamic, i64, i64]
    positions = {op: i for i, op in enumerate(owner.body.block.ops)}
    first = min(roots, key=positions.__getitem__)
    allocation = memref.AllocOp.get(i8, shape=[plan.capacity], alignment=plan.alignment)
    cast = memref.CastOp.get(allocation.memref, dynamic)
    owner.body.block.insert_ops_before([allocation, cast], first)
    hidden = {}
    for function in helpers:
        inputs = tuple(function.function_type.inputs)
        hidden[function] = tuple(
            function.body.block.insert_arg(typ, len(inputs) + i) for i, typ in enumerate(hidden_types)
        )
        function.function_type = FunctionType.from_lists([*inputs, *hidden_types], function.function_type.outputs)
        if (attrs := function.properties.get("arg_attrs")) is not None:
            function.properties["arg_attrs"] = ArrayAttr([*attrs, *(DictionaryAttr({}) for _ in hidden_types)])
    leases = {
        call: (epoch, ordinal) for epoch, calls in enumerate(plan.sequences) for ordinal, call in enumerate(calls)
    }
    for function in helpers:
        for call in callers[function.sym_name.data]:
            parent = _containing_function(call)
            if parent is owner:
                epoch, ordinal = leases[call]
                constants = [arith.ConstantOp.from_int_and_width(value, 64) for value in (epoch, ordinal)]
                owner.body.block.insert_ops_before(constants, call)
                passed = [cast.dest, *(op.result for op in constants)]
            else:
                passed = hidden[parent]
            call.operands = [*call.arguments, *passed]
    borrowed.function_type = FunctionType.from_lists([*borrowed.function_type.inputs, *hidden_types], [])
    borrowed.properties["sym_name"] = StringAttr(borrowed_symbol)
    if (attrs := borrowed.properties.get("arg_attrs")) is not None:
        borrowed.properties["arg_attrs"] = ArrayAttr([*attrs, *(DictionaryAttr({}) for _ in hidden_types)])
    borrowed_calls[0].properties["callee"] = SymbolRefAttr(borrowed_symbol)
    borrowed_calls[0].operands = [*borrowed_calls[0].arguments, *hidden[wrapper]]
    # A private tensor-to-readonly-buffer wrapper supplies the actual ranked ABI.
    block = Block(arg_types=[*tensor_types, *hidden_types])
    buffers = []
    for value in block.args[: len(tensor_types)]:
        typ = MemRefType(value.type.element_type, value.type.get_shape())
        operation = bufferization.ToBufferOp.build(
            operands=[value], result_types=[typ], properties={"read_only": UnitAttr()}
        )
        block.add_op(operation)
        buffers.append(operation.results[0])
    block.add_ops([func.CallOp(prepare_external, [*buffers, *block.args[len(tensor_types) :]], []), func.ReturnOp()])
    preparation_attrs = ArrayAttr(
        [
            *(DictionaryAttr({"bufferization.access": StringAttr("read")}) for _ in tensor_types),
            DictionaryAttr({"bufferization.access": StringAttr("write")}),
            DictionaryAttr({}),
            DictionaryAttr({}),
        ]
    )
    prepare = func.FuncOp(
        prepare_symbol,
        ([*tensor_types, *hidden_types], []),
        Region(block),
        visibility="private",
        arg_attrs=preparation_attrs,
    )
    external = func.FuncOp(
        prepare_external,
        ([*(value.type for value in buffers), *hidden_types], []),
        Region(),
        visibility="private",
        arg_attrs=preparation_attrs,
    )
    external.attributes["llvm.emit_c_interface"] = UnitAttr()
    module.body.block.add_ops([external, prepare])
    for epoch, calls in enumerate(plan.sequences):
        constants = [arith.ConstantOp.from_int_and_width(value, 64) for value in (epoch, len(calls))]
        operation = func.CallOp(prepare_symbol, [*source[calls[0]], cast.dest, *(op.result for op in constants)], [])
        owner.body.block.insert_ops_before([*constants, operation], calls[0])
    module.verify()
    return dict(
        shared_materializations=materializations,
        schema="prepared_borrow_schedule_v1",
        epochs=len(plan.sequences),
        consumers=len(expected),
        owner_bytes=plan.capacity,
        payload_bytes=plan.capacity - len(plan.sequences),
        alignment=plan.alignment,
        allocations=1,
        epoch_tokens="address-only final pool bytes",
        public_abi_unchanged=True,
        release="normal_upstream_owner_deallocation",
        fallback="physical prepared consumer must invoke retained original source on refusal",
    )
