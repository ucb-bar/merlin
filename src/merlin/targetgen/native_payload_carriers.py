"""Closed payload provenance through structured regions and CFG block arguments.

A table bit follows storage carriers. Unknown provenance also follows arithmetic,
so a later arithmetic producer cannot authenticate an unrecognized carrier.
Loop backedges use a monotone fixed point, including the zero-iteration input.
"""

from collections import defaultdict, deque

from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
from xdsl.traits import Pure

TABLE, UNKNOWN = 1, 2


def op_name(op):
    return getattr(getattr(op, "op_name", None), "data", getattr(op, "name", None))


def region_yields(op, index):
    values = []
    for region in op.regions:
        for block in region.blocks:
            term = block.last_op
            if term is not None and op_name(term) in {"scf.yield", "tensor.yield", "linalg.yield"}:
                if index >= len(term.operands):
                    return None
                values.append(term.operands[index])
    return tuple(values) or None


def block_sources(value):
    block, index = value.owner, value.index
    parent = block.parent_op()
    name = op_name(parent)
    if name == "func.func" and block is parent.body.blocks.first:
        return ()  # Runtime function inputs are not source result tables.
    if name == "linalg.reduce":
        inputs = (parent.input, parent.init)
        return (inputs[index],) if index < len(inputs) else None
    if name is not None and name.startswith("linalg."):
        inputs = (*getattr(parent, "inputs", ()), *getattr(parent, "outputs", ()))
        return (inputs[index],) if index < len(inputs) else None
    if name == "scf.for":
        if index == 0:
            return ()  # The induction variable is an address, not a payload table.
        carried = index - 1
        yielded = block.last_op.operands
        if carried < len(parent.iter_args) and carried < len(yielded):
            return (parent.iter_args[carried], yielded[carried])
        return None
    if name == "scf.while":
        if block is parent.before_region.blocks.first:
            yielded = parent.after_region.block.last_op.operands
            if index < len(parent.arguments) and index < len(yielded):
                return (parent.arguments[index], yielded[index])
        if block is parent.after_region.blocks.first:
            condition = parent.before_region.block.last_op
            if op_name(condition) == "scf.condition" and index + 1 < len(condition.operands):
                return (condition.operands[index + 1],)
        return None
    if name in {"tensor.generate", "tensor.pad"}:
        return ()  # Region arguments are index coordinates.
    # Every incoming CFG edge must have a known operand-to-argument mapping.
    region = block.parent_region()
    incoming = []
    if region is not None:
        for predecessor in region.blocks:
            term = predecessor.last_op
            if term is None or block not in term.successors:
                continue
            if op_name(term) == "cf.br":
                incoming.append(term.arguments[index])
            elif op_name(term) == "cf.cond_br":
                for successor, arguments in ((term.then_block, term.then_arguments),
                                             (term.else_block, term.else_arguments)):
                    if successor is block:
                        incoming.append(arguments[index])
            else:
                return None
    return tuple(incoming) or None


def carrier_sources(value):
    """Return payload-only edges for proved carriers; None means no carrier proof."""
    op = value.owner
    name = op_name(op)
    if name is None:
        return block_sources(value)
    if name == "scf.for":
        yielded = region_yields(op, value.index)
        if yielded is not None and value.index < len(op.iter_args):
            return (op.iter_args[value.index], *yielded)
        return None
    if name == "scf.while":
        term = op.before_region.block.last_op
        return (term.operands[value.index + 1],) if op_name(term) == "scf.condition" else None
    if name in {"scf.if", "scf.index_switch", "scf.execute_region", "tensor.generate"}:
        return region_yields(op, value.index)
    if name in {
        "tensor.cast", "tensor.reshape", "tensor.extract", "tensor.extract_slice",
        "tensor.expand_shape", "tensor.collapse_shape", "tensor.splat",
        "tensor.gather", "tensor.pack", "tensor.unpack",
        "arith.extsi", "arith.extui", "arith.trunci", "arith.bitcast",
        "arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp",
        "complex.re", "complex.im",
    }:
        return (op.operands[0],)
    if name in {"tensor.insert_slice", "tensor.insert"}:
        return tuple(op.operands[:2])
    if name == "tensor.pad":
        yielded = region_yields(op, 0)
        return (op.operands[0], *yielded) if yielded is not None else None
    if name in {"linalg.broadcast", "linalg.transpose", "linalg.copy", "linalg.fill"}:
        return tuple(op.operands[:1])
    if name == "linalg.generic" or (name.startswith("linalg.") and op.regions):
        return region_yields(op, value.index)
    if name in {"tensor.concat", "tensor.from_elements", "complex.create"}:
        return tuple(op.operands)
    if name == "arith.select":
        return tuple(op.operands[1:])
    return None


class LiteralPayloads:
    """Bitwise table/unknown provenance without recursive SSA traversal."""

    def __init__(self):
        self.memo = {}
        self.unknown_operators = set()

    def classify(self, root):
        if root in self.memo:
            return self.memo[root]
        nodes, pending = {}, [root]
        while pending:
            value = pending.pop()
            if value in nodes or value in self.memo:
                continue
            seed, edges, arithmetic = self.expand(value)
            if seed & UNKNOWN:
                name = op_name(value.owner)
                if name is None:
                    name = "block argument of " + str(op_name(value.owner.parent_op()))
                self.unknown_operators.add(name)
            nodes[value] = (seed, edges, arithmetic)
            pending.extend(edges)
        # Only the new nodes and their memoized edges take part; copying the
        # whole memo on every call made classification quadratic in program size.
        reverse = defaultdict(set)
        masks = {}
        anchored = set()
        for value, (seed, edges, _) in nodes.items():
            masks[value] = seed
            if not edges:
                anchored.add(value)
            for edge in edges:
                reverse[edge].add(value)
                if edge in self.memo:
                    masks[edge] = self.memo[edge]
                    anchored.add(edge)
        work = deque(anchored)
        while work:
            for parent in reverse[work.popleft()]:
                if parent not in anchored:
                    anchored.add(parent)
                    work.append(parent)
        for value in nodes.keys() - anchored:
            masks[value] |= UNKNOWN  # An unanchored cyclic carrier proves no origin.
        work = deque(nodes)
        queued = set(nodes)
        while work:
            value = work.popleft()
            queued.discard(value)
            seed, edges, arithmetic = nodes[value]
            mask = masks[value] | seed
            for index, edge in enumerate(edges):
                transfer = (arithmetic[index] if isinstance(arithmetic, tuple)
                            else UNKNOWN if arithmetic else TABLE | UNKNOWN)
                mask |= masks[edge] & transfer
            if mask != masks[value]:
                masks[value] = mask
                for parent in reverse[value] - queued:
                    work.append(parent)
                    queued.add(parent)
        self.memo.update((value, masks[value]) for value in nodes)
        return self.memo[root]

    @staticmethod
    def expand(value):
        op = value.owner
        name = op_name(op)
        if name == "arith.constant":
            data = op.properties.get("value", op.attributes.get("value"))
            if isinstance(data, DenseIntOrFPElementsAttr):
                width, raw = data.get_element_type().compile_time_size, data.data.data
                nonuniform = any(raw[i:i+width] != raw[:width] for i in range(width, len(raw), width))
                return (TABLE if nonuniform else 0), (), False
            return 0, (), False
        if name == "linalg.index":
            return 0, (), False
        edges = carrier_sources(value)
        if edges is not None:
            return 0, edges, False
        if name is not None and name.startswith("quant_ext."):
            from merlin.targetgen.native_quant_boundary import checked_boundary

            if checked_boundary(op):
                # Representation boundaries retain table provenance from data,
                # not from per-channel parameters; UNKNOWN still taints every operand.
                return 0, tuple(op.operands), (TABLE | UNKNOWN, UNKNOWN, UNKNOWN)
        if name in {"arith.divsi", "arith.remsi"}:
            from merlin.llvmlower.integer_division_components import literal_divisor

            divisor = literal_divisor(op)
            if divisor is not None and divisor > 0:
                # Defined scalar arithmetic: new values clear TABLE, while
                # UNKNOWN propagates from every operand as for other arithmetic.
                return 0, tuple(op.operands), True
        # Known scalar arithmetic produces new values rather than copying a
        # precomputed result. Unknown inputs remain unproved through arithmetic.
        if name is not None and name.startswith(("arith.", "math.")) and op.has_trait(Pure):
            return 0, tuple(op.operands), True
        return UNKNOWN, (), False
