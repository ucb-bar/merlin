"""Scalar dataflow patterns excluded from native primitive construction.

Follow typed tensor payloads across region boundaries; addresses and unchanged
literal compare/select clamps are not predicate implementations. No source
operator names or provenance labels participate in classification.
"""

CASTS = {
    "arith.extf",
    "arith.truncf",
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.sitofp",
    "arith.uitofp",
    "tensor.cast",
    "tensor.reshape",
    "tensor.splat",
}
ADDITIONS = {"arith.addi", "arith.addf", "arith.subi", "arith.subf"}
EXTREMA = {
    "arith.maximumf",
    "arith.minimumf",
    "arith.maxnumf",
    "arith.minnumf",
    "arith.maxsi",
    "arith.maxui",
    "arith.minsi",
    "arith.minui",
}
ROUNDING = {"math.floor", "math.ceil", "math.trunc", "math.round", "math.roundeven"}
MOVEMENT = {
    "tensor.extract",
    "tensor.extract_slice",
    "tensor.insert_slice",
    "tensor.collapse_shape",
    "tensor.expand_shape",
    "tensor.concat",
}


def same_literal(first, second):
    """Equal literal tables; a uniform table equals its scalar or splat form."""
    if first is None or second is None:
        return False
    return first == second or (len(set(first)) == 1 and set(first) == set(second))


def dataflow(root, memo, expand, combine=any):
    """Evaluate a scalar SSA DAG iteratively, including deeply split programs."""
    stack, active, dependencies = [root], set(), {}
    while stack:
        value = stack[-1]
        if value in memo:
            stack.pop()
            active.discard(value)
            continue
        if value not in dependencies:
            dependencies[value] = expand(value)
        children = dependencies[value]
        if isinstance(children, bool):
            memo[value] = children
            continue
        pending = next((v for v in children if v not in memo), None)
        if pending is None:
            memo[value] = combine(memo[v] for v in children)
            continue
        active.add(value)
        if pending in active:
            raise ValueError("cyclic payload dataflow has no native construction proof")
        stack.append(pending)
    return memo[root]


class Payloads:
    """Resolve scalar payload definitions, without expanding shared DAGs."""

    def __init__(self, module):
        self.operations = set(module.walk())
        self._resolve_memo = {}
        self._contraction_memo = {}
        self._readout_memo = {}
        self._store_memo = {}
        self._closed_memo = {}

    def resolve(self, value):
        path = []
        while value not in self._resolve_memo:
            path.append(value)
            child = self.alias(value)
            if child is None:
                break
            value = child
        root = self._resolve_memo.get(value, value)
        self._resolve_memo.update((v, root) for v in path)
        return root

    def alias(self, value):
        from merlin.llvmlower._ir import generic_input

        owner = value.owner
        name = getattr(owner, "name", None)
        if name in CASTS:
            return owner.operands[0]
        if name == "linalg.generic":
            return owner.body.block.last_op.operands[value.index]
        return generic_input(value)

    def operation(self, value):
        return self.resolve(value).owner

    def constant(self, value):
        op = self.operation(value)
        if getattr(op, "name", None) != "arith.constant":
            return None
        attr = op.properties.get("value", op.attributes.get("value"))
        if hasattr(attr, "value"):
            return attr.value.data
        from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

        if isinstance(attr, DenseIntOrFPElementsAttr):
            values = iter(attr.get_values())
            first = next(values, None)
            if first is not None and all(v == first for v in values):
                return first
        return None

    def literal_values(self, value):
        """Return every element of a closed literal, uniform or not; None otherwise.

        A literal whose elements are not materialized in the IR (for example a
        resource blob) is still closed, but its values are unknown: it yields
        one ``None`` element, which no zero test can discharge.
        """
        op = self.operation(value)
        if getattr(op, "name", None) != "arith.constant":
            return None
        attr = op.properties.get("value", op.attributes.get("value"))
        if hasattr(attr, "value") and hasattr(attr.value, "data"):
            return (attr.value.data,)
        from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

        if isinstance(attr, DenseIntOrFPElementsAttr):
            return tuple(attr.get_values())
        return (None,)

    def nonzero_literal(self, root):
        """Prove a literal addend, rather than a runtime value with literal ancestry.

        Every carrier alternative must supply a closed literal: a scalar, a
        splat, or a non-uniform dense table (a per-element bias table is as
        much an offset as a uniform one). Runtime selects containing IEEE
        masks or NaN encodings are not offsets. Opaque carriers remain subject
        to the independent fail-closed screen.
        """
        from merlin.targetgen.native_payload_carriers import carrier_sources, op_name

        pending, seen, literals = [root], set(), []
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            direct = self.literal_values(value)
            if direct is not None:
                literals.extend(direct)
                continue
            edges = carrier_sources(value)
            owner = value.owner
            name = op_name(owner)
            if name == "math.copysign":
                # A sign transfer keeps its first operand's magnitude: a
                # literal magnitude with a runtime sign is still a literal
                # offset (round-half-away bias x + copysign(0.5, x)).
                edges = (owner.operands[0],)
            if edges is None and name is not None and name.startswith(("arith.", "math.")):
                # A closed literal expression remains a possible offset, also
                # when loop backedges carry it through arithmetic. Runtime
                # operands still fail this proof at their external leaves.
                edges = tuple(owner.operands)
            if not edges:
                # A runtime leaf or an opaque producer is not a literal
                # operand merely because a sibling ancestry contains constants.
                return False
            pending.extend(edges)
        # Backedges introduce no new value by themselves. Every external leaf
        # must be a literal, and a cycle without a literal seed proves nothing.
        # An element of unknown value (None) is conservatively nonzero.
        return bool(literals) and any(value is None or value != 0 for value in literals)

    def forced_special_division(self, op):
        """A floating division or remainder whose divisor is a closed literal zero."""
        from merlin.targetgen.native_payload_carriers import op_name

        if op_name(op) not in {"arith.divf", "arith.remf"}:
            return False
        from merlin.targetgen.closed_scalar import UNKNOWN, evaluate

        values = self.literal_values(op.operands[1])
        if values is None:
            closed = evaluate(op.operands[1], {}, self._closed_memo)  # a closed expression, e.g. |0|
            values = None if closed is UNKNOWN else (closed,)
        if not values or not all(v is not None and v == 0 for v in values):
            return False
        return self.observable(op.results[0])

    def observable(self, root):
        """Can this value reach a result? A select whose closed condition picks the
        other alternative discards it (for example the unused branch of a complex
        division by a real literal). Region carriers are conservatively observable."""
        from merlin.targetgen.closed_scalar import UNKNOWN, evaluate
        from merlin.targetgen.native_payload_carriers import op_name

        pending, seen = [root], set()
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            for use in value.uses:
                user = use.operation
                name = op_name(user)
                if name == "arith.select" and use.index > 0:
                    condition = evaluate(user.operands[0], {}, self._closed_memo)
                    if condition is not UNKNOWN and use.index != (1 if condition else 2):
                        continue
                if not user.results:
                    return True  # func.return, linalg/scf yields and stores observe it.
                pending.extend(user.results)
        return False

    def product_reduction(self, product):
        """Does this product multiply a reduction's accumulator by runtime payload?

        A contraction adds products into its accumulator. Multiplying the
        accumulator itself by each element is a chain of runtime products
        (a product reduction), whatever structured operation carries it.
        """
        from merlin.targetgen.native_payload_carriers import op_name

        parent = product.parent_op()
        while parent is not None and op_name(parent) not in {"linalg.reduce", "linalg.generic"}:
            parent = parent.parent_op()
        if parent is None:
            return False
        block = parent.regions[0].block
        if op_name(parent) == "linalg.reduce":
            inputs = getattr(parent, "inputs", None)
            if inputs is None:
                inputs = parent.input
            count = len(inputs) if hasattr(inputs, "__len__") else 1
        else:
            if not any(it.data.value == "reduction" for it in parent.get_iterator_types()):
                return False
            count = len(parent.inputs)
        accumulators = set(block.args[count:])
        operands = list(product.operands)
        for index, operand in enumerate(operands):
            other = operands[1 - index]
            if operand in accumulators and self.literal_values(other) is None:
                return True
        return False

    def literal_coefficient_table(self, product):
        """Is a product inside a contraction weighted by a multi-valued closed table?

        True when the product feeds a reduction (it is inside a generic with a
        reduction iterator) and one operand resolves to a closed literal tensor
        with at least two distinct nonzero values (signs count). A uniform
        literal, a 0/c selection or scaled identity is not such a table.
        """
        parent = product.parent_op()
        while parent is not None and parent.name != "linalg.generic":
            parent = parent.parent_op()
        if parent is None or not any(it.data.value == "reduction" for it in parent.get_iterator_types()):
            return False
        for operand in product.operands:
            values = self.literal_values(operand)
            if values is None:
                continue
            nonzero = {v for v in values if v is None or v != 0}
            if None in nonzero or len(nonzero) > 1:
                return True
        return False

    def offset_before(self, root):
        """Find a literal addend through casts, arithmetic and all known carriers."""
        from merlin.targetgen.native_payload_carriers import carrier_sources, op_name

        pending, seen = [root], set()
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            op = value.owner
            name = op_name(op)
            if name in ADDITIONS and any(self.nonzero_literal(v) for v in op.operands):
                return True
            edges = carrier_sources(value)
            if edges is not None:
                pending.extend(edges)
            elif name is not None and name.startswith(("arith.", "math.")):
                pending.extend(op.operands)
        return False

    def clamp(self, value):
        op = self.operation(value)
        if getattr(op, "name", None) in EXTREMA:
            return op
        if getattr(op, "name", None) != "arith.select":
            return None
        compare = self.operation(op.operands[0])
        if getattr(compare, "name", None) not in {"arith.cmpi", "arith.cmpf"}:
            return None
        if {self.resolve(v) for v in compare.operands} == {self.resolve(v) for v in op.operands[1:]}:
            return op
        return None

    def contraction_dependent(self, value):
        return dataflow(value, self._contraction_memo, self.contraction_dependencies)

    def contraction_dependencies(self, value):
        owner = value.owner
        name = getattr(owner, "name", None)
        if name == "linalg.generic":
            if any(it.data.value == "reduction" for it in owner.get_iterator_types()):
                return True
            return (owner.body.block.last_op.operands[value.index],)
        if name in {"linalg.matmul", "linalg.batch_matmul", "linalg.reduce", "scf.for"}:
            return True
        if name is None and hasattr(owner, "parent_op"):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                return (parent.inputs[value.index],)
            return False
        return tuple(getattr(owner, "operands", ()))

    def reduction_product(self, op):
        """Prove this product contributes to a local loop-carried reduction."""

        def depends(value, target):
            def expand(v):
                return True if v is target else tuple(getattr(v.owner, "operands", ()))

            return dataflow(value, {}, expand)

        parent = op.parent_op()
        while parent is not None:
            if parent.name == "scf.for":
                block = parent.body.block
                for carried, yielded in zip(block.args[1:], block.last_op.operands, strict=True):
                    if depends(yielded, op.results[0]) and depends(yielded, carried):
                        return True
            parent = parent.parent_op()
        return False

    def readout(self, value):
        """Does this observation reach a result through storage conversions only?

        Balanced digit extraction feeds subsequent arithmetic producers. A
        rounding readout publishes the narrowed value itself. Tensor shapes
        and region wrappers do not distinguish those mathematical roles.
        """
        return dataflow(value, self._readout_memo, self.readout_dependencies)

    def readout_dependencies(self, value):
        children = []
        for use in value.uses:
            consumer = use.operation
            if consumer not in self.operations:
                continue
            if consumer.name == "func.return":
                return True
            boolean_chain = consumer.name in {"arith.ori", "arith.andi", "arith.xori"} and str(value.type) == "i1"
            if consumer.name in CASTS | MOVEMENT | {"arith.fptosi", "arith.fptoui"} or boolean_chain:
                children.extend(consumer.results)
            if consumer.name == "arith.select" and use.index == 0:
                if all(self.constant(v) is not None for v in consumer.operands[1:]):
                    children.extend(consumer.results)
            if consumer.name == "scf.yield":
                parent = consumer.parent_op()
                if use.index < len(parent.results):
                    children.append(parent.results[use.index])
            if consumer.name == "linalg.yield":
                parent = consumer.parent_op()
                if parent.name in {"linalg.generic", "linalg.reduce"} and use.index < len(parent.results):
                    children.append(parent.results[use.index])
            if consumer.name in {"linalg.generic", "linalg.reduce"}:
                inputs = consumer.inputs if consumer.name == "linalg.generic" else (consumer.input,)
                if use.index < len(inputs):
                    children.append(consumer.regions[0].block.args[use.index])
        return tuple(children)

    def terminal_store(self, value):
        """Does every use of this value reach a result through storage conversions and movement only?"""
        return dataflow(value, self._store_memo, self.store_dependencies, combine=all)

    def store_dependencies(self, value):
        children = []
        for use in value.uses:
            consumer = use.operation
            if consumer not in self.operations or consumer.name == "func.return":
                continue
            if consumer.name in CASTS | MOVEMENT:
                children.extend(consumer.results)
            elif consumer.name in {"scf.yield", "linalg.yield"} and use.index < len(consumer.parent_op().results):
                children.append(consumer.parent_op().results[use.index])
            elif consumer.name == "linalg.generic" and use.index < len(consumer.inputs):
                children.append(consumer.regions[0].block.args[use.index])
            else:
                return False
        return tuple(children)

    def patterns(self, op):
        reasons = []
        name = op.name
        # Boolean union has no admitted native arithmetic primitive. An add and
        # truth test is an identity even when hidden inside a reduction.
        if name == "arith.ori" and str(op.results[0].type) == "i1" and self.readout(op.results[0]):
            parent = op.parent_op()
            if (
                parent is not None
                and parent.name in {"linalg.generic", "linalg.reduce"}
                and any(str(v.type.element_type) == "i1" for v in parent.results)
            ):
                reasons.append("multi_op_identity")
        if name in {"arith.cmpi", "arith.cmpf"}:
            if self.readout(op.results[0]) and any(self.clamp(v) is not None for v in op.operands):
                reasons.append("saturation_comparators")
            # Widened Boolean addition followed by a truth predicate, including
            # separated host/device generics. Integer RNE's internal predicates
            # operate on integer payloads rather than Boolean source leaves.
            for operand in op.operands:
                producer = self.operation(operand)
                if getattr(producer, "name", None) in ADDITIONS and all(
                    str(self.resolve(v).type) in {"i1"}
                    or str(getattr(self.resolve(v).type, "element_type", "")) == "i1"
                    for v in producer.operands
                ):
                    reasons.append("multi_op_identity")
        # Extrema reconstructed as rectifier(x-y)+y (or the subtract variant).
        if name in ADDITIONS:
            for operand in op.operands:
                clamp = self.clamp(operand)
                if clamp is None:
                    continue
                for candidate in clamp.operands:
                    difference = self.operation(candidate)
                    if getattr(difference, "name", None) not in {"arith.subf", "arith.subi"}:
                        continue
                    if any(
                        self.resolve(v) == self.resolve(other)
                        for v in difference.operands
                        for other in op.operands
                        if other is not operand
                    ):
                        reasons.append("multi_op_identity")
        # Bias followed by a rounding/narrowing observation. The exact IEEE
        # encoder uses quotient/remainder and a conditional ties-even increment,
        # rather than an unconditional literal offset in a floating readout.
        # IEEE float-to-float RNE is a storage boundary, not an integer
        # offset-rounding implementation, only when the narrowed value is
        # stored: every use reaches a result through storage conversions and
        # movement. Its input may then be an ordinary affine expression. A
        # narrowed value that feeds further arithmetic can remove a magic bias
        # in any spelling, so it is screened like every other rounding.
        storage_rne = name == "arith.truncf" and (str(op.operands[0].type), str(op.results[0].type)) in {
            ("f64", "f32"),
            ("f64", "f16"),
            ("f64", "bf16"),
            ("f32", "f16"),
            ("f32", "bf16"),
        }
        storage_rne = storage_rne and not any(
            key in op.properties or key in op.attributes for key in ("roundingmode", "roundingMode")
        )
        storage_rne = storage_rne and self.terminal_store(op.results[0])
        if not storage_rne and name in ROUNDING | {
            "arith.fptosi",
            "arith.fptoui",
            "arith.truncf",
            "arith.shrsi",
            "arith.shrui",
            "arith.divsi",
            "arith.divui",
        }:
            floating = name in ROUNDING | {"arith.fptosi", "arith.fptoui", "arith.truncf"}
            if (floating or self.readout(op.results[0])) and self.offset_before(op.operands[0]):
                reasons.append("readout_offset_rounding")
        # Floating magic-bias rounding: (x + C) - C, C nonzero.
        # A non-uniform literal table of magic biases is screened the same way.
        if name == "arith.subf":
            addition = self.operation(op.operands[0])
            offset = self.literal_values(op.operands[1])
            if (
                offset
                and any(v is None or v != 0 for v in offset)
                and getattr(addition, "name", None) == "arith.addf"
                and any(same_literal(self.literal_values(v), offset) for v in addition.operands)
            ):
                reasons.append("readout_offset_rounding")
        return sorted(set(reasons))
