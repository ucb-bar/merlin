"""Closed, payload-independent structural folds: unit extents, empty loops and carriers.

Each fold proves its rewrite from the IR alone: unit or zero loop extents,
literal seeds, unused operands and complete typed comparison cells. No tensor
values are consulted. Unknown arithmetic, multiple runtime inputs and
non-singleton reductions refuse.
"""

import math
from dataclasses import dataclass
from math import prod

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    TensorType,
    i1,
    i32,
    i64,
)
from xdsl.dialects.linalg import ops as L
from xdsl.ir import Block
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.rewriter import Rewriter
from xdsl.traits import Pure
from xdsl.transforms.dead_code_elimination import region_dce

from merlin.llvmlower._ir import PARALLEL, generic, identity, reshape
from merlin.targetgen.closed_scalar import predicate_holds
from merlin.targetgen.native_payload_carriers import UNKNOWN, LiteralPayloads
from merlin.xdsl_dialects.lowering.group_numerics import _constant_of


def _reshape(value, typ):
    """(shape literal, tensor.reshape) of value to typ."""
    shape = list(typ.get_shape())
    sizes = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [len(shape)]), shape))
    return [sizes, tensor.ReshapeOp(value, sizes.result, typ)]


def _unit_reduce(op):
    """(source, seed, kept shape) of a one-result linalg.reduce over extent-one dimensions only, else None."""
    if op.name != "linalg.reduce" or len(op.operands) != 2 or len(op.results) != 1:
        return None
    source, seed = op.operands
    shape = source.type.get_shape()
    dimensions = tuple(op.dimensions.get_values())
    if any(n < 0 for n in shape) or len(set(dimensions)) != len(dimensions):
        return None
    if any(d < 0 or d >= len(shape) or shape[d] != 1 for d in dimensions):
        return None
    expected = tuple(n for i, n in enumerate(shape) if i not in dimensions)
    return (source, seed, expected) if op.results[0].type.get_shape() == expected else None


@dataclass(frozen=True)
class Literal:
    value: object
    typ: object

    def key(self):
        if isinstance(self.typ, IntegerType):
            return str(self.typ), int(self.value) % (1 << self.typ.width.data)
        if str(self.typ) == "index":
            return "index", int(self.value)
        return str(self.typ), self.typ.pack([self.value])


RUNTIME = object()
IGNORED = object()


def _evaluate(op, environment, cell, thresholds, element):
    values = [environment[v] for v in op.operands]
    name = op.name
    typ = op.results[0].type
    if name == "arith.select":
        if not isinstance(values[0], Literal):
            raise ValueError("unproved selector predicate")
        return values[1 if values[0].value else 2]
    if name in {"arith.andi", "arith.ori", "arith.xori"} and str(typ) == "i1":
        if not all(isinstance(v, Literal) for v in values):
            raise ValueError("runtime Boolean arithmetic")
        a, b = [bool(v.value) for v in values]
        return Literal({"arith.andi": a and b, "arith.ori": a or b, "arith.xori": a != b}[name], typ)
    if name in {"arith.index_cast", "arith.index_castui"} and isinstance(values[0], Literal):
        return Literal(int(values[0].value), typ)
    if name not in {"arith.cmpi", "arith.cmpf"}:
        raise ValueError("unproved scalar operation")
    predicate = op.predicate.value.data
    if name == "arith.cmpf":

        def position(value):
            if value is RUNTIME:
                return cell
            if math.isnan(value.value):
                return None
            return 2 * thresholds.index(value.value)

        a, b = map(position, values)
        unordered = a is None or b is None
        relation = 0 if unordered else (a > b) - (a < b)
        return Literal(predicate_holds(predicate, relation, unordered, floating=True), i1)
    signed = predicate in {2, 3, 4, 5}
    operand_type = op.operands[0].type
    width = operand_type.width.data if isinstance(operand_type, IntegerType) else None

    def interval(value):
        if value is RUNTIME:
            lo, hi = cell
        else:
            lo = hi = int(value.value)
            if width is not None:
                lo = hi = lo % (1 << width)
        if signed and width is not None and lo >= 1 << (width - 1):
            lo -= 1 << width
            hi -= 1 << width
        return lo, hi

    a, b = map(interval, values)
    if values[0] is RUNTIME and values[1] is RUNTIME:
        return Literal(predicate_holds(predicate, 0), i1)
    answers = {predicate_holds(predicate, (x > y) - (x < y)) for x in a for y in b}
    if len(answers) != 1:
        raise ValueError("comparison cell straddles a predicate boundary")
    return Literal(answers.pop(), i1)


def _prove(op, ranges):
    block = op.body.block
    element = op.inputs[0].type.element_type
    integer = isinstance(element, IntegerType)
    if not integer and str(element) not in {"f16", "bf16", "f32", "f64"}:
        return None
    initial = {block.args[0]: RUNTIME}
    for arg, value in zip(list(block.args)[1:], op.outputs, strict=True):
        literal = _constant_of(value)
        if literal is None:
            if arg.uses:
                return None
            initial[arg] = IGNORED
        else:
            initial[arg] = Literal(literal, arg.type)
    for child in block.ops:
        for flag in ("fastmath", "overflowFlags"):
            value = child.properties.get(flag)
            if value is not None and str(value).split("<")[-1] != "none>":
                return None
        if child.name == "arith.constant":
            value = child.value
            if not hasattr(value, "value"):
                return None
            initial[child.results[0]] = Literal(value.value.data, child.results[0].type)
        elif child.name == "linalg.index":
            dimension = child.dim.value.data
            if ranges[dimension] != 1:
                return None
            initial[child.results[0]] = Literal(0, child.results[0].type)
    literals = [v.value for v in initial.values() if isinstance(v, Literal)]
    if integer:
        width = element.width.data
        limit = 1 << width
        boundaries = {0, limit, limit // 2}
        for value in literals:
            if isinstance(value, (int, bool)):
                word = int(value) % limit
                boundaries.update((word, word + 1))
        ordered = sorted(boundaries)
        cells = [(a, b - 1) for a, b in zip(ordered, ordered[1:]) if a < b]
        thresholds = []
    else:
        thresholds = sorted({-math.inf, math.inf, *[v for v in literals if not math.isnan(v)]})
        cells = [*range(2 * len(thresholds) - 1), None]
    results = []
    for cell in cells:
        environment = dict(initial)
        for child in block.ops:
            if child.name == "linalg.yield":
                results.append([environment[v] for v in child.operands])
                break
            if child.results and child.results[0] in initial:
                continue
            if len(child.results) != 1 or child.regions:
                return None
            environment[child.results[0]] = _evaluate(child, environment, cell, thresholds, element)
    if not results:
        return None
    proved = []
    for result, column in zip(op.results, zip(*results, strict=True), strict=True):
        if not result.uses:
            proved.append(IGNORED)
            continue
        first = next((value for value in column if isinstance(value, Literal)), None)

        def equals_literal(value, cell):
            if isinstance(value, Literal):
                return value.key() == first.key()
            return integer and value is RUNTIME and cell[0] == cell[1] == int(first.value) % (1 << element.width.data)

        if all(value is RUNTIME for value in column):
            proved.append(RUNTIME)
        elif isinstance(first, Literal) and all(
            equals_literal(value, cell) for value, cell in zip(column, cells, strict=True)
        ):
            proved.append(first)
        else:
            return None
    return proved


def fold_singleton_selectors(module):
    """Prove closed compare/select regions over one runtime scalar to be storage selectors or fills.

    The runtime scalar is partitioned into complete typed comparison cells. Each
    predicate is constant throughout a cell; the cells cover every storage word,
    including float NaNs and both zero signs.
    """
    payloads = LiteralPayloads()
    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 1 or not op.results:
            continue
        if len(op.outputs) != len(op.results):
            continue
        if payloads.classify(op.inputs[0]) & UNKNOWN:
            continue
        reductions = [i for i, it in enumerate(op.get_iterator_types()) if it.data.value == "reduction"]
        if not reductions and not any(child.name in {"arith.cmpi", "arith.cmpf"} for child in op.body.block.ops):
            continue
        try:
            ranges = op.get_static_loop_ranges()
            if any(n < 0 for n in ranges) or any(ranges[i] != 1 for i in reductions):
                continue
            proved = _prove(op, ranges)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError, OverflowError):
            continue
        if proved is None or len(proved) != len(op.results):
            continue

        valid = True
        maps = list(op.get_indexing_maps())
        nontrivial = {i for i, n in enumerate(ranges) if n != 1}

        def project(amap, shape):
            positions = []
            if len(amap.results) != len(shape):
                return None
            for expr, extent in zip(amap.results, shape, strict=True):
                if isinstance(expr, AffineDimExpr) and ranges[expr.position] == extent:
                    if expr.position in nontrivial:
                        positions.append(expr.position)
                elif not (isinstance(expr, AffineConstantExpr) and expr.value == 0 and extent == 1):
                    return None
            return positions if len(positions) == len(set(positions)) and set(positions) == nontrivial else None

        for i, proof in enumerate(proved):
            if proof is IGNORED:
                continue
            target = project(maps[len(op.inputs) + i].data, op.results[i].type.get_shape())
            if target is None:
                valid = False
                break
            if proof is not RUNTIME:
                continue
            source = project(maps[0].data, op.inputs[0].type.get_shape())
            target = project(maps[len(op.inputs) + i].data, op.results[i].type.get_shape())
            if (
                source is None
                or source != target
                or op.results[i].type.element_type != op.inputs[0].type.element_type
                or prod(op.inputs[0].type.get_shape()) != prod(op.results[i].type.get_shape())
            ):
                valid = False
                break
        if not valid:
            continue
        operations, values = [], []
        for result, proof, seed in zip(op.results, proved, op.outputs, strict=True):
            if proof is IGNORED:
                values.append(seed)
            elif proof is RUNTIME:
                operations.extend(_reshape(op.inputs[0], result.type))
                values.append(operations[-1].result)
            else:
                typ = result.type.element_type
                attr = (
                    IntegerAttr(int(proof.value), typ) if isinstance(typ, IntegerType) else FloatAttr(proof.value, typ)
                )
                constant = arith.ConstantOp(attr)
                value = tensor.SplatOp(constant.result, [], result.type)
                operations.extend([constant, value])
                values.append(value.result)
        Rewriter.replace_op(op, operations, values)
        count += 1
    return count


def normalize_singleton_boundaries(module):
    """Retain closed singleton identities before exposing scalar IEEE boundaries.

    Splitting a tuple snapshots its initializers as inputs. Prove index fills and
    integer monoid identities first, while their closed seed relation is explicit.
    Unknown or nonidentity computations retain their original scalar operations.
    """
    changed = fold_singleton_reductions(module) if any(op.name == "linalg.reduce" for op in module.walk()) else 0
    changed += fold_singleton_selectors(module)
    changed += expose_singleton_reduction_boundaries(module)
    changed += expose_singleton_tuple_regions(module)
    return changed


def fold_singleton_carriers(module):
    """Reassemble one extracted storage word through a shape-only tensor view."""
    count = 0
    for op in list(module.walk()):
        if op.name != "tensor.from_elements" or len(op.operands) != 1:
            continue
        extracted = op.operands[0].owner
        if getattr(extracted, "name", None) != "tensor.extract":
            continue
        source = extracted.tensor
        if (
            prod(source.type.get_shape()) != 1
            or prod(op.results[0].type.get_shape()) != 1
            or source.type.element_type != op.results[0].type.element_type
        ):
            continue
        indices = list(extracted.indices)
        if any(
            getattr(v.owner, "name", None) != "arith.constant"
            or not isinstance(v.owner.value, IntegerAttr)
            or v.owner.value.value.data != 0
            for v in indices
        ):
            continue
        operations = _reshape(source, op.results[0].type)
        Rewriter.replace_op(op, operations, [operations[-1].result])
        count += 1
    return count


def fold_scalar_tensor_roundtrips(module):
    """Keep rank-zero data in tensor form across singleton packaging.

    A single-use private linalg result is extracted without indices and packed
    into one singleton element. It has no independently observable alias; the
    rewrite preserves its element bits, result type, and boundary metadata.
    """
    count = 0
    for op in list(module.walk()):
        if op.name != "tensor.from_elements" or len(op.operands) != 1:
            continue
        extracted = op.operands[0]
        extract = extracted.owner
        if getattr(extract, "name", None) != "tensor.extract" or len(extract.operands) != 1:
            continue
        source = extract.operands[0]
        producer = source.owner
        if getattr(producer, "name", None) not in {"linalg.generic", "linalg.reduce"}:
            continue
        if source.type.get_shape() != () or not source.has_one_use() or not extracted.has_one_use():
            continue
        result_type = op.results[0].type
        if prod(result_type.get_shape()) != 1 or source.type.element_type != result_type.element_type:
            continue
        operations = _reshape(source, result_type)
        operations[-1].attributes.update(op.attributes)
        Rewriter.replace_op(op, operations, [operations[-1].result])
        Rewriter.erase_op(extract)
        count += 1
    module.verify()
    return count


def expose_singleton_reduction_boundaries(module):
    """A one-iteration reduction is the same scalar body with parallel iterators.

    Keep every original floating operation and initialization word. This does not
    replace a floating monoid by a byte copy or change its NaN/zero behavior.
    """
    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.reduce" or len(op.operands) != 2 or len(op.results) != 1:
            continue
        if not all(isinstance(v.type, TensorType) for v in op.operands):
            continue
        found = _unit_reduce(op)
        if found is None:
            continue
        source, seed, expected = found
        result_type = op.results[0].type
        if seed.type != result_type or source.type.element_type != result_type.element_type:
            continue
        body = op.regions[0].block
        if len(body.args) != 2 or body.last_op.name != "linalg.yield":
            continue
        if any(
            child.regions or child.name == "builtin.unregistered" or not child.has_trait(Pure)
            for child in list(body.ops)[:-1]
        ):
            continue
        operations = []
        reshaped = reshape(source, expected, operations)
        empty = tensor.EmptyOp([], result_type)
        parallel_body = Block(arg_types=[result_type.element_type] * 3)
        mapping = dict(zip(body.args, parallel_body.args[:2], strict=True))
        for child in body.ops:
            cloned = child.clone(value_mapper=mapping)
            parallel_body.add_op(cloned)
            mapping.update(zip(child.results, cloned.results, strict=True))
        rank = len(expected)
        result = generic([reshaped, seed], empty.tensor, parallel_body, [identity(rank)] * 3, [PARALLEL] * rank)
        result.attributes.update(op.attributes)
        Rewriter.replace_op(op, [*operations, empty, result], [result.results[0]])
        changed += 1
    if changed:
        module.verify()
    return changed


def fold_singleton_reductions(module):
    """Canonicalize singleton integer monoids with a proved identity seed.

    Floating extrema remain unchanged: a singleton may quiet or canonicalize NaNs.
    """
    changed = 0
    for op in list(module.walk()):
        found = _unit_reduce(op)
        if found is None:
            continue
        source, seed, _ = found
        typ = source.type.element_type
        if not isinstance(typ, IntegerType) or typ != seed.type.element_type:
            continue
        body = op.regions[0].block
        ops = list(body.ops)
        if len(ops) != 2 or ops[1].name != "linalg.yield":
            continue
        if set(ops[0].operands) != set(body.args) or list(ops[1].operands) != list(ops[0].results):
            continue
        flags = ops[0].properties.get("overflowFlags")
        if flags is not None and str(flags) != "#arith.overflow<none>":
            continue
        width = typ.width.data
        mask = (1 << width) - 1
        identities = {
            "arith.addi": 0,
            "arith.muli": 1,
            "arith.ori": 0,
            "arith.andi": mask,
            "arith.maxsi": 1 << (width - 1),
            "arith.minsi": (1 << (width - 1)) - 1,
            "arith.maxui": 0,
            "arith.minui": mask,
        }
        literal = _constant_of(seed)
        if ops[0].name not in identities or not isinstance(literal, int):
            continue
        if literal & mask != identities[ops[0].name]:
            continue
        result_type = op.results[0].type
        if result_type.element_type != typ:
            continue
        operations = _reshape(source, result_type)
        Rewriter.replace_op(op, operations, [operations[-1].result])
        changed += 1
    module.verify()
    return changed


def expose_singleton_tuple_regions(module):
    """Expose pure tuple regions with at most one reduction contribution per result.

    Snapshot every original initializer as an explicit input. This retains coupled
    tuple arithmetic and each IEEE boundary while giving each result a closed body.
    Repeated reductions and unknown/effectful region bodies remain untouched.
    """
    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) < 2 or len(op.body.blocks) != 1:
            continue
        body = op.body.block
        if body.last_op.name != "linalg.yield" or len(body.last_op.operands) != len(op.results):
            continue
        if any(
            c.regions or c.name == "builtin.unregistered" or not (c.has_trait(Pure) or isinstance(c, L.IndexOp))
            for c in list(body.ops)[:-1]
        ):
            continue
        try:
            ranges = op.get_static_loop_ranges()
        except (AttributeError, ValueError):
            continue
        removed = {i for i, it in enumerate(op.get_iterator_types()) if it.data.value == "reduction"}
        if any(ranges[i] != 1 for i in removed) or any(n <= 0 for n in ranges):
            continue
        kept = [i for i in range(len(ranges)) if i not in removed]
        replacements = [
            AffineConstantExpr(0) if i in removed else AffineDimExpr(kept.index(i)) for i in range(len(ranges))
        ]
        maps = [
            AffineMapAttr(m.data.replace_dims_and_symbols(replacements, [], len(kept), 0))
            for m in op.get_indexing_maps()
        ]
        inputs = list(op.inputs) + list(op.outputs)
        operations, results = [], []
        for index, typ in enumerate(v.type for v in op.results):
            empty = tensor.EmptyOp([], typ)
            scalar = Block(arg_types=[v.type.element_type for v in inputs] + [typ.element_type])
            mapping = dict(zip(body.args, scalar.args[:-1], strict=True))
            for child in list(body.ops)[:-1]:
                if child.name == "linalg.index":
                    axis = child.dim.value.data
                    if not 0 <= axis < len(ranges):
                        break
                    clone = (
                        arith.ConstantOp(IntegerAttr(0, IndexType()))
                        if axis in removed
                        else L.IndexOp(kept.index(axis))
                    )
                else:
                    clone = child.clone(value_mapper=mapping)
                scalar.add_op(clone)
                mapping.update(zip(child.results, clone.results, strict=True))
            else:
                scalar.add_op(L.YieldOp(mapping[body.last_op.operands[index]]))
                result = generic(
                    inputs, empty.tensor, scalar, [*maps, maps[len(op.inputs) + index]], [PARALLEL] * len(kept)
                )
                result.attributes.update(op.attributes)
                operations.extend([empty, result])
                results.append(result.results[0])
                continue
            break
        if len(results) != len(op.results):
            continue
        Rewriter.replace_op(op, operations, results)
        changed += 1
    if changed:
        module.verify()
    return changed


def fold_static_unit_indices(module):
    """An iteration dimension of extent one has a compile-time zero coordinate."""
    from merlin.llvmlower.uniform_numeric_literals import fold_uniform_numeric_literals

    changed = 0
    for op in list(module.walk()):
        if op.name == "linalg.index":
            parent = op.parent_op()
            if getattr(parent, "name", None) != "linalg.generic":
                continue
            try:
                ranges = parent.get_static_loop_ranges()
            except (ValueError, AttributeError):
                continue
            axis = op.dim.value.data
            if not 0 <= axis < len(ranges) or ranges[axis] != 1:
                continue
            constant = arith.ConstantOp(IntegerAttr(0, IndexType()))
        elif op.name in {"arith.index_cast", "arith.index_castui"}:
            source = op.operands[0].owner
            attr = source.properties.get("value") if getattr(source, "name", None) == "arith.constant" else None
            if (
                not isinstance(attr, IntegerAttr)
                or attr.value.data != 0
                or not isinstance(op.results[0].type, IntegerType)
            ):
                continue
            constant = arith.ConstantOp(IntegerAttr(0, op.results[0].type))
        else:
            continue
        Rewriter.replace_op(op, [constant], [constant.result])
        changed += 1
    if changed:
        region_dce(module.body)
        changed += fold_uniform_numeric_literals(module)
    return changed


def fold_empty_reductions(module):
    """Shape-only canonicalization of zero-iteration additive reductions."""
    count = 0
    for op in list(module.walk()):
        if op.name not in {"linalg.generic", "linalg.reduce"} or not op.results:
            continue
        if op.name == "linalg.reduce":
            seeds = [op.init]
            empty = prod(op.input.type.get_shape()) == 0
        else:
            seeds = list(op.outputs)
            try:
                empty = any(
                    iterator.data.value == "reduction" and extent == 0
                    for iterator, extent in zip(op.get_iterator_types(), op.get_static_loop_ranges(), strict=True)
                )
            except (AttributeError, ValueError):
                continue
        if not empty or len(seeds) != len(op.results):
            continue
        if any(a.type != b.type for a, b in zip(seeds, op.results)):
            continue
        # No body iteration executes: the result is precisely the original
        # seed, regardless of its dtype. No tensor payload is inspected.
        Rewriter.replace_op(op, [], seeds)
        count += 1
    if count:
        for op in list(module.walk()):
            if op.name == "tensor.extract" and not op.indices:
                source = op.tensor.owner
                if getattr(source, "name", None) == "tensor.splat":
                    Rewriter.replace_op(op, [], [source.operands[0]])
            elif (
                op.name == "tensor.from_elements"
                and op.operands
                and all(value is op.operands[0] for value in op.operands)
            ):
                filled = tensor.SplatOp(op.operands[0], [], op.results[0].type)
                Rewriter.replace_op(op, [filled], [filled.result])
    return count


def prune_unused_tensor_inputs(module):
    """Remove tensor operands whose scalar argument is unused by a generic region."""
    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.body.blocks) != 1:
            continue
        n = len(op.inputs)
        if len(op.body.block.args) != n + len(op.outputs):
            continue
        unused = [i for i in range(n) if not op.body.block.args[i].uses]
        if not unused:
            continue
        kept = [i for i in range(n) if i not in unused]
        maps = list(op.get_indexing_maps())
        retained_maps = [maps[i].data for i in kept] + [m.data for m in maps[n:]]
        if not retained_maps or any(m.num_symbols for m in retained_maps):
            continue
        combined = AffineMap(retained_maps[0].num_dims, 0, tuple(e for m in retained_maps for e in m.results))
        if combined.inverse_permutation() is None:
            continue  # An unused scalar may still define a reduction's loop domain.
        extents = {}
        consistent = True
        for i in kept + list(range(n, len(op.operands))):
            for extent, expression in zip(op.operands[i].type.get_shape(), maps[i].data.results, strict=True):
                if isinstance(expression, AffineDimExpr):
                    if extent < 0 or extents.get(expression.position, extent) != extent:
                        consistent = False
                    extents[expression.position] = extent
        for i in unused:
            for extent, expression in zip(op.operands[i].type.get_shape(), maps[i].data.results, strict=True):
                if isinstance(expression, AffineDimExpr):
                    consistent &= extent >= 0 and extents.get(expression.position) == extent
                elif isinstance(expression, AffineConstantExpr):
                    consistent &= 0 <= expression.value < extent
                else:
                    consistent = False  # No proof that the eliminated load was in bounds.
        if not consistent:
            continue
        clone = op.clone()
        for index in reversed(unused):
            clone.body.block.erase_arg(clone.body.block.args[index])
        clone.operands = [op.inputs[i] for i in kept] + list(op.outputs)
        maps = list(op.get_indexing_maps())
        clone.properties["indexing_maps"] = ArrayAttr([maps[i] for i in kept] + maps[n:])
        clone.properties["operandSegmentSizes"] = DenseArrayBase.from_list(i32, [len(kept), len(op.outputs)])
        Rewriter.replace_op(op, clone)
        changed += 1
    module.verify()
    return changed


def expose_uniform_literal_carriers(module):
    """Expose shape-only carriers of uniform literal storage as uniform fills."""

    def scalar(value):
        owner = value.owner
        if getattr(owner, "name", None) == "tensor.splat":
            literal = owner.operands[0].owner
            if getattr(literal, "name", None) == "arith.constant":
                attr = literal.properties.get("value")
                return attr if isinstance(attr, (FloatAttr, IntegerAttr)) else None
        if getattr(owner, "name", None) == "arith.constant":
            attr = owner.properties.get("value")
            if isinstance(attr, DenseIntOrFPElementsAttr) and attr.is_splat():
                return attr.get_attrs()[0]
        return None

    changed = 0
    for op in list(module.walk()):
        if op.name not in {"tensor.reshape", "tensor.cast"} or len(op.results) != 1:
            continue
        source, result = op.operands[0], op.results[0]
        if not isinstance(source.type, TensorType) or not isinstance(result.type, TensorType):
            continue
        shapes = source.type.get_shape(), result.type.get_shape()
        if any(n < 0 for shape in shapes for n in shape) or prod(shapes[0]) != prod(shapes[1]):
            continue
        if source.type.element_type != result.type.element_type:
            continue
        attr = scalar(source)
        if attr is None:
            continue
        constant = arith.ConstantOp(attr)
        filled = tensor.SplatOp(constant.result, [], result.type)
        Rewriter.replace_op(op, [constant, filled], [filled.result])
        changed += 1
    module.verify()
    return changed
