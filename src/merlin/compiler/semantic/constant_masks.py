"""Fold a select whose condition is a uniform compile-time mask.

The condition is evaluated from declared constant bytes with the exact predicate
primitives. A uniform value stays one element, so a large broadcast costs no
dense work. Only a select whose two branches are both scalar broadcasts is
rewritten, to the branch the mask chooses, and only when the chosen scalar is a
declared constant; every other mask is left for the normal admission. After a
fold, every unused Boolean constant in the graph is removed too; a graph with no
foldable select keeps its dead Boolean constants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from merlin.compiler.semantic.constant_folding import _element
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, SemanticNode, pointwise_identity_maps
from merlin.compiler.semantic.predicate_values import PREDICATE_OPERATIONS, predicate_element, predicate_widths

SCHEMA = "merlin.constant_masks.v1"
_INTEGER = {"i64_add": "add", "i64_sub": "sub", "i64_mul": "mul"}
_MOVES = {"identity", "reshape", "gather"}


class _NotFoldable(Exception):
    """The value is runtime, undeclared, or over budget; the select stays as it is."""


@dataclass(frozen=True)
class _Value:
    shape: tuple
    width: int
    fill: int | None
    data: bytes | None

    def element(self, flat):
        if self.fill is not None:
            return self.fill
        return int.from_bytes(self.data[flat * self.width : (flat + 1) * self.width], "little")


def _value(shape, width, data):
    if data == data[:width] * prod(shape):
        return _Value(shape, width, int.from_bytes(data[:width], "little"), None)
    return _Value(shape, width, None, bytes(data))


def _flat(indices, shape):
    position = 0
    for index, extent in zip(indices, shape, strict=True):
        position = position * extent + index
    return position


def _coordinates(shape):
    index = [0] * len(shape)
    for _ in range(prod(shape)):
        yield tuple(index)
        for axis in reversed(range(len(shape))):
            index[axis] += 1
            if index[axis] < shape[axis]:
                break
            index[axis] = 0


def _check_read(read, loop_shape, source_shape):
    """An affine read stays inside its source exactly when its extreme corners do."""
    if read.loop_rank != len(loop_shape) or len(read.coefficients) != len(source_shape):
        raise _NotFoldable
    for row, offset, extent in zip(read.coefficients, read.offsets, source_shape, strict=True):
        low = offset + sum(min(c * (n - 1), 0) for c, n in zip(row, loop_shape, strict=True))
        high = offset + sum(max(c * (n - 1), 0) for c, n in zip(row, loop_shape, strict=True))
        if low < 0 or high >= extent:
            raise _NotFoldable


class _Evaluator:
    def __init__(self, source, element_budget):
        self.nodes = {node.id: node for node in source.nodes}
        self.order = {node.id: index for index, node in enumerate(source.nodes)}
        self.bindings = {row.node_id: row for row in source.constants}
        self.values = {}
        self.remaining = element_budget

    def value(self, key):
        pending, seen = [key], set()
        while pending:
            current = pending.pop()
            if current in seen or current in self.values:
                continue
            seen.add(current)
            node = self.nodes[current]
            if node.effect != "constant" and (node.effect != "pure" or not self._eligible(node)):
                raise _NotFoldable
            pending.extend(node.inputs)
        for current in sorted(seen, key=self.order.__getitem__):
            self.values[current] = self._evaluate(self.nodes[current])
        return self.values[key]

    @staticmethod
    def _eligible(node):
        return node.op in _MOVES or node.op in PREDICATE_OPERATIONS or node.op in _INTEGER

    def _charge(self, count):
        if count > self.remaining:
            raise _NotFoldable
        self.remaining -= count

    def _evaluate(self, node):
        if not prod(node.type.shape):
            raise _NotFoldable
        if node.effect == "constant":
            binding = self.bindings.get(node.id)
            if binding is None:
                raise _NotFoldable
            data = bytes.fromhex(binding.data_hex)
            width, extra = divmod(len(data), prod(node.type.shape))
            if extra or not width:
                raise _NotFoldable
            self._charge(prod(node.type.shape))
            return _value(node.type.shape, width, data)
        operands = [self.values[key] for key in node.inputs]
        if node.op in _MOVES:
            return self._move(node, operands)
        return self._pointwise(node, operands)

    def _move(self, node, operands):
        source = self.nodes[node.inputs[0]] if len(node.inputs) == 1 else None
        if (source is None or node.attrs or source.type.dtype != node.type.dtype
                or source.type.numerical_policy != node.type.numerical_policy):
            raise _NotFoldable
        value = operands[0]
        if node.op != "gather":
            if node.index_maps or prod(source.type.shape) != prod(node.type.shape) or (
                    node.op == "identity" and source.type != node.type):
                raise _NotFoldable
            return _Value(node.type.shape, value.width, value.fill, value.data)
        rank = len(node.type.shape)
        if not rank or len(node.index_maps) != 2 or node.index_maps[1] != pointwise_identity_maps(rank, 1)[1]:
            raise _NotFoldable
        read = node.index_maps[0]
        _check_read(read, node.type.shape, source.type.shape)
        if value.fill is not None:
            return _Value(node.type.shape, value.width, value.fill, None)
        self._charge(prod(node.type.shape))
        result = bytearray()
        for coordinate in _coordinates(node.type.shape):
            result.extend(value.element(_flat(read.apply(coordinate), source.type.shape)).to_bytes(value.width, "little"))
        return _value(node.type.shape, value.width, result)

    def _pointwise(self, node, operands):
        shape, sources = node.type.shape, [self.nodes[key] for key in node.inputs]
        if not shape or len(node.index_maps) != len(sources) + 1 or node.index_maps[-1] != pointwise_identity_maps(
                len(shape), 1)[-1]:
            raise _NotFoldable
        for read, item in zip(node.index_maps, sources, strict=False):
            _check_read(read, shape, item.type.shape)
        aligned = [replace(item, type=replace(item.type, shape=shape)) for item in sources]
        try:
            if node.op in _INTEGER:
                policy = f"signed-i64-{_INTEGER[node.op]}-wrap-v1"
                if (len(aligned) != 2 or any(item.type != node.type for item in aligned)
                        or node.type.dtype != "i64" or node.type.numerical_policy != "signed-i64"
                        or dict(node.attrs) != {"numeric_semantics": policy}):
                    raise _NotFoldable
                widths, width = (8, 8), 8
                compute = lambda args: _element(node, args)
            else:
                widths, width = predicate_widths(node, aligned)
                compute = lambda args: predicate_element(node, args, widths)
        except ValueError as error:
            raise _NotFoldable from error
        if any(value.width != item for value, item in zip(operands, widths, strict=True)):
            raise _NotFoldable
        if all(value.fill is not None for value in operands):
            return _Value(shape, width, compute([value.fill for value in operands]), None)
        self._charge(prod(shape))
        result = bytearray()
        for coordinate in _coordinates(shape):
            args = [value.element(0 if value.fill is not None else _flat(read.apply(coordinate), value.shape))
                    for value, read in zip(operands, node.index_maps, strict=False)]
            result.extend(compute(args).to_bytes(width, "little"))
        return _value(shape, width, result)


def _rewrite(evaluator, node):
    """Return the select as the scalar broadcast its uniform mask chooses, or None."""
    rank = len(node.type.shape)
    if (node.effect != "pure" or node.op != "select" or len(node.inputs) != 3 or node.attrs or not rank
            or node.index_maps != pointwise_identity_maps(rank, 3)):
        return None
    condition, *branches = (evaluator.nodes[key] for key in node.inputs)
    if condition.type.dtype != "i1" or condition.type.numerical_policy != "boolean" \
            or condition.type.shape != node.type.shape or any(item.type != node.type for item in branches):
        return None
    broadcast = (IndexMap(rank, (), ()), pointwise_identity_maps(rank, 1)[1])
    if any(item.effect != "pure" or item.op != "gather" or item.attrs or len(item.inputs) != 1
           or item.index_maps != broadcast or evaluator.nodes[item.inputs[0]].type.shape for item in branches):
        return None
    try:
        fill = evaluator.value(condition.id).fill
    except _NotFoldable:
        return None
    if fill not in {0, 1}:
        return None
    chosen = branches[0] if fill else branches[1]
    scalar = evaluator.nodes[chosen.inputs[0]]
    if scalar.effect != "constant" or scalar.id not in evaluator.bindings:
        return None
    return replace(node, op="gather", inputs=chosen.inputs, attrs=(), index_maps=chosen.index_maps)


def fold_constant_masks(source, *, element_budget=65536):
    """Rewrite uniform constant selects and remove the computation only they used."""
    if type(element_budget) is not int or element_budget < 0:
        raise ValueError("constant mask folding needs a nonnegative element budget")
    before = source.digest()
    empty = dict(schema=SCHEMA, element_budget=element_budget, before_digest=before, after_digest=before,
                 rewrites=[], removed_nodes=[], removed_constants=[])
    if source.physical_extents or source.boundary_layouts or source.output_layouts:
        return source, empty
    evaluator = _Evaluator(source, element_budget)
    changed, candidates = {}, set()
    for node in source.nodes:
        if node.op != "select":
            continue
        rewritten = _rewrite(evaluator, node)
        if rewritten is None:
            continue
        changed[node.id] = rewritten
        pending = list(node.inputs)
        while pending:
            key = pending.pop()
            if key not in candidates:
                candidates.add(key)
                pending.extend(evaluator.nodes[key].inputs)
    if not changed:
        return source, empty
    # An unused Boolean constant computes nothing once its masks are gone.
    candidates.update(node.id for node in source.nodes if node.effect == "constant" and node.type.dtype == "i1")
    live, kept, removed = set(source.outputs), [], []
    for position, node in reversed(list(enumerate(source.nodes))):
        current = changed.get(node.id, node)
        # Boundary inputs and state stay even when only an unchosen branch read them.
        if node.id in candidates and node.id not in live and node.effect in {"pure", "constant"}:
            removed.append(dict(position=position, node=node.record()))
            continue
        live.update(current.inputs)
        kept.append(current)
    removed.reverse()
    gone = {row["node"]["id"] for row in removed}
    constants = [dict(position=i, node_id=c.node_id, encoding=c.encoding, data_hex=c.data_hex)
                 for i, c in enumerate(source.constants) if c.node_id in gone]
    result = replace(source, nodes=tuple(reversed(kept)),
                     constants=tuple(c for c in source.constants if c.node_id not in gone))
    return result, dict(empty, after_digest=result.digest(),
                        rewrites=[source.node(key).record() for key in changed],
                        removed_nodes=removed, removed_constants=constants)


def verify_constant_masks(source, record):
    """Rebuild the original source and require an identical replay."""
    fields = {"schema", "element_budget", "before_digest", "after_digest", "rewrites", "removed_nodes",
              "removed_constants"}
    if not isinstance(record, dict) or set(record) != fields or record["schema"] != SCHEMA \
            or record["after_digest"] != source.digest():
        raise ValueError("constant mask receipt identity or fields changed")
    try:
        originals = {row["id"]: SemanticNode.from_record(row) for row in record["rewrites"]}
        removed = {row["position"]: SemanticNode.from_record(row["node"]) for row in record["removed_nodes"]}
        bindings = {row["position"]: ConstantBinding(row["node_id"], row["encoding"], row["data_hex"])
                    for row in record["removed_constants"]}
    except (KeyError, TypeError) as error:
        raise ValueError("invalid constant mask receipt row") from error
    if len(removed) != len(record["removed_nodes"]) or len(bindings) != len(record["removed_constants"]):
        raise ValueError("constant mask receipt repeats a position")
    nodes, constants = iter(originals.get(n.id, n) for n in source.nodes), iter(source.constants)
    node_total, constant_total = len(source.nodes) + len(removed), len(source.constants) + len(bindings)
    if any(type(i) is not int or not 0 <= i < node_total for i in removed) \
            or any(type(i) is not int or not 0 <= i < constant_total for i in bindings):
        raise ValueError("constant mask receipt position exceeds the original graph")
    original = replace(
        source,
        nodes=tuple(removed[i] if i in removed else next(nodes) for i in range(node_total)),
        constants=tuple(bindings[i] if i in bindings else next(constants) for i in range(constant_total)),
    )
    if original.digest() != record["before_digest"]:
        raise ValueError("constant mask original source changed")
    replayed, expected = fold_constant_masks(original, element_budget=record["element_budget"])
    if replayed.digest() != source.digest() or expected != record:
        raise ValueError("constant mask fold differs from exact replay")
    return original
