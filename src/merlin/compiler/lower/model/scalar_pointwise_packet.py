"""Opt-in interleaving of independent pure tensor pointwise arithmetic lanes.

Each lane retains scalar source operation order. Tensor value semantics and
upstream bufferization establish alias legality; no physical no-alias is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass

BORROWED_FEATURE = "packet_borrowed_pointwise_fma_division_2"
BORROWED_MARKER = "__merlin_borrowed_pointwise_packet_2__"
BORROWED_CONTRACT = "merlin.borrowed_pointwise_effects"
FEATURE = "packet_scalar_pointwise_fma_division_2"
FOUR_FEATURE = "packet_scalar_pointwise_fma_division_4"
BROADCAST_FEATURE = "packet_scalar_pointwise_broadcast_2"
MULTIPLY_FOUR_FEATURE = "packet_scalar_pointwise_multiplication_4"
TWO_MULTIPLY_FOUR_FEATURE = "packet_scalar_pointwise_two_multiplications_4"
MARKER = "__merlin_pointwise_packet_2__"
FOUR_MARKER = "__merlin_pointwise_packet_4__"
BROADCAST_MARKER = "__merlin_pointwise_broadcast_packet_2__"
MULTIPLY_FOUR_MARKER = "__merlin_pointwise_multiplication_packet_4__"
TWO_MULTIPLY_FOUR_MARKER = "__merlin_pointwise_two_multiplications_packet_4__"


@dataclass(frozen=True)
class BorrowedPointwiseEffects:
    """Caller proof obligations for reordering independent borrowed writes.

    Input spans may alias each other. Output coordinates must be private,
    disjoint from every input and already owned for the full writer lifetime.
    These explicit permissions do not follow from a memref type or symbol.
    """

    immutable_inputs: bool
    fresh_disjoint_output: bool
    stable_rounding: bool
    nontrapping_arithmetic: bool
    exception_flags_unobserved: bool

    def validate(self):
        if any(value is not True for value in vars(self).values()):
            raise ValueError("complete borrowed pointwise effect proof required")


def bind_borrowed_pointwise_packet(operation, *, effects):
    """Attach an explicit normal-pass contract to one live memref writer.

    The selected upstream feature independently checks the complete typed
    scalar body, maps, static spans and source contexts. This binding grants
    only the caller-supplied physical and floating effects, not profitability.
    Unsupported source remains unchanged under the normal feature.
    """
    from xdsl.dialects.builtin import DictionaryAttr, MemRefType, UnitAttr
    from xdsl.dialects.linalg.ops import GenericOp

    if not isinstance(effects, BorrowedPointwiseEffects):
        raise ValueError("explicit borrowed pointwise effects required")
    effects.validate()
    if (
        not isinstance(operation, GenericOp)
        or operation.results
        or len(operation.outputs) != 1
        or any(not isinstance(value.type, MemRefType) for value in operation.operands)
    ):
        raise ValueError("one ordinary borrowed memref writer required")
    if BORROWED_CONTRACT in operation.attributes:
        raise ValueError("borrowed pointwise contract already present")
    operation.attributes[BORROWED_CONTRACT] = DictionaryAttr({name: UnitAttr() for name in vars(effects)})


def _edit_borrowed_pipeline(passes):
    hits = [i for i, p in enumerate(passes) if "one-shot-bufferize" in p]
    if len(hits) != 1 or BORROWED_MARKER in passes:
        raise ValueError("borrowed pointwise packet requires one bufferization stage")
    i = hits[0]
    return [*passes[:i], BORROWED_MARKER, *passes[i:]]


def _edit_pipeline(passes, *, lanes=2, broadcast=False, multiplication=False, two_products=False):
    hits = [i for i, p in enumerate(passes) if "one-shot-bufferize" in p]
    multiply_marker = TWO_MULTIPLY_FOUR_MARKER if two_products else MULTIPLY_FOUR_MARKER
    conflicts = (multiply_marker,) if multiplication else (MARKER, FOUR_MARKER, BROADCAST_MARKER)
    if len(hits) != 1 or any(m in passes for m in conflicts):
        raise ValueError("pointwise packet requires exactly one bufferization stage")
    i = hits[0]
    marker = (
        multiply_marker if multiplication else BROADCAST_MARKER if broadcast else {2: MARKER, 4: FOUR_MARKER}[lanes]
    )
    return [*passes[:i], marker, *passes[i:]]


def ensure_registered():
    from merlin.compiler.selection.shared import ImprFeature, known, register

    if BORROWED_FEATURE not in known():
        register(
            ImprFeature(
                name=BORROWED_FEATURE,
                action_class="PASS",
                description=(
                    "Interleave two independent scalar lanes of an explicitly contracted "
                    "immutable-input, private-disjoint-output memref FMA/division writer; "
                    "preserve source arithmetic and bounded tails."
                ),
                edit_pipeline=_edit_borrowed_pipeline,
                alternative_group="borrowed_pointwise_packet",
            )
        )
    for name, lanes in ((FEATURE, 2), (FOUR_FEATURE, 4)):
        if name not in known():
            register(
                ImprFeature(
                    name=name,
                    action_class="PASS",
                    description=f"Interleave {lanes} independent scalar lanes of pure static tensor pointwise FMA/division bodies; preserve each lane arithmetic and alias semantics.",
                    edit_pipeline=lambda passes, n=lanes: _edit_pipeline(passes, lanes=n),
                    alternative_group="scalar_pointwise_packet",
                )
            )
    if BROADCAST_FEATURE not in known():
        register(
            ImprFeature(
                name=BROADCAST_FEATURE,
                action_class="PASS",
                description="Interleave two pure tensor pointwise lanes across a proved broadcast axis, sharing identical input extracts and retaining each lane's source arithmetic and bounded tails.",
                edit_pipeline=lambda passes: _edit_pipeline(passes, broadcast=True),
                alternative_group="scalar_pointwise_packet",
            )
        )
    if MULTIPLY_FOUR_FEATURE not in known():
        register(
            ImprFeature(
                name=MULTIPLY_FOUR_FEATURE,
                action_class="PASS",
                description="Interleave four independent lanes of pure static f32 multiplication bodies, preserving scalar operations and unit-axis broadcast coordinates. No FMA, division, precision change or automatic selection.",
                edit_pipeline=lambda passes: _edit_pipeline(passes, lanes=4, multiplication=True),
                alternative_group="scalar_pointwise_multiplication_packet",
            )
        )
    if TWO_MULTIPLY_FOUR_FEATURE not in known():
        register(
            ImprFeature(
                name=TWO_MULTIPLY_FOUR_FEATURE,
                action_class="PASS",
                description="Interleave four independent lanes of pure static f32 tensor bodies with exactly two rounded multiplications, optional integer casts and additions. Preserve source order and tails; no FMA, division or precision changes.",
                edit_pipeline=lambda passes: _edit_pipeline(passes, lanes=4, multiplication=True, two_products=True),
                alternative_group="scalar_pointwise_two_multiplications_packet",
            )
        )
    return FEATURE


RUNNER_PRELUDE = r"""
def _pointwise_packet_spec(op, multiplication=False, two_products=False):
    from torch_mlir import ir as _pp_ir
    if (op.operation.name != "linalg.generic" or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
        return None
    types = list(v.type for v in op.operands)
    if any(not isinstance(t, _pp_ir.RankedTensorType) or any(d < 0 for d in t.shape) for t in types):
        return None
    if len(op.operands) < 2:
        return None
    maps = [_pp_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    positions, extents = [], [None] * dims
    for m, t in zip(maps, types):
        if len(m.results) != len(t.shape):
            return None
        pos = []
        for expr, extent in zip(m.results, t.shape):
            if (multiplication and isinstance(expr, _pp_ir.AffineConstantExpr)
                    and _pp_ir.AffineConstantExpr(expr).value == 0 and extent == 1):
                pos.append(None)
                continue
            if not isinstance(expr, _pp_ir.AffineDimExpr):
                return None
            d = _pp_ir.AffineDimExpr(expr).position
            if d in pos or (extents[d] is not None and extents[d] != extent):
                return None
            pos.append(d)
            extents[d] = extent
        positions.append(pos)
    if (None in positions[-1] or sorted(positions[-1]) != list(range(dims))
            or any(d is None or d <= 0 for d in extents)):
        return None
    body = _pointwise_packet_body_spec(op, types, multiplication, two_products)
    return (positions, extents, body) if body is not None else None


def _pointwise_packet_body_spec(op, types, multiplication=False, two_products=False):
    from torch_mlir import ir as _pp_ir
    if any("strictfp" in name or "strictfp" in str(op.attributes[name]) for name in op.attributes):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if len(block.arguments) != len(types) or not body or body[-1].operation.name != "linalg.yield" or len(body[-1].operands) != 1:
        return None
    if list(block.arguments[-1].uses):
        return None
    if two_products:
        # A separate closed arithmetic family. The existing >=3 selector keeps
        # its original eligibility and is never silently broadened.
        if (not multiplication or not isinstance(types[-1].element_type, _pp_ir.F32Type)
                or any(inner.operation.name not in {"arith.constant", "arith.sitofp",
                       "arith.uitofp", "arith.mulf", "arith.addf", "linalg.yield"}
                       for inner in body)):
            return None
    allowed = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
               "arith.negf", "arith.maximumf", "arith.minimumf", "arith.cmpf", "arith.cmpi",
               "arith.select", "arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi",
               "arith.muli", "arith.shli", "arith.shrsi", "arith.shrui", "arith.bitcast",
               "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extsi",
               "arith.extui", "arith.trunci", "arith.extf", "arith.truncf", "math.fma"}
    fmas = division = multiplies = 0
    for inner in body[:-1]:
        if any("strictfp" in name or "strictfp" in str(inner.attributes[name]) for name in inner.attributes):
            return None
        if inner.operation.name not in allowed or inner.regions or len(inner.results) != 1:
            return None
        if "fastmath" in inner.attributes and str(inner.attributes["fastmath"]) != "#arith.fastmath<none>":
            return None
        if isinstance(inner.results[0].type, _pp_ir.VectorType):
            return None
        if multiplication:
            if any("strictfp" in name or "strictfp" in str(inner.attributes[name])
                   for name in inner.attributes):
                return None
            if any(not isinstance(v.type, (_pp_ir.F32Type, _pp_ir.IntegerType, _pp_ir.IndexType))
                   for v in list(inner.operands) + list(inner.results)):
                return None
            if inner.operation.name in {"math.fma", "arith.divf", "arith.extf", "arith.truncf"}:
                return None
            if inner.operation.name == "arith.mulf" and isinstance(inner.results[0].type, _pp_ir.F32Type):
                multiplies += 1
        if inner.operation.name == "math.fma" and isinstance(inner.results[0].type, _pp_ir.F32Type):
            fmas += 1
        if inner.operation.name == "arith.divf" and isinstance(inner.results[0].type, _pp_ir.F32Type):
            division += 1
    eligible = (multiplies == 2 if two_products else multiplies >= 3) if multiplication else fmas >= 4 and division
    return body if eligible else None


def _pointwise_broadcast_axis(op, positions, extents, lanes):
    # Scalar inputs are already loop invariant and do not justify an interchange.
    # Every shared load comes from the exact same immutable tensor and coordinates.
    args = list(op.regions[0].blocks[0].arguments)
    choices = []
    for axis, extent in enumerate(extents):
        if extent < lanes:
            continue
        shared = sum(bool(pos) and axis not in pos and bool(list(arg.uses))
                     for arg, pos in zip(args[:-1], positions[:-1]))
        if shared:
            choices.append((shared, extent, -axis))
    return -max(choices)[2] if choices else None


def _packetize_pointwise(ctx, module, lanes=2, broadcast=False, multiplication=False,
                         two_products=False):
    from torch_mlir import ir as _pp_ir
    todo = []
    def walk(op):
        if any("strictfp" in name or "strictfp" in str(op.attributes[name]) for name in op.attributes):
            return
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _pointwise_packet_spec(inner, multiplication=multiplication,
                                                   two_products=two_products)
                    if spec is not None:
                        axis = _pointwise_broadcast_axis(inner, spec[0], spec[1], lanes) if broadcast else len(spec[1]) - 1
                        if axis is not None:
                            todo.append((inner, spec, axis))
                    else:
                        walk(inner.operation)
    walk(module.operation)
    with ctx:
        for old, (positions, extents, scalar_body), axis in todo:
            with old.location, _pp_ir.InsertionPoint(old):
                index = _pp_ir.IndexType.get()
                tensor_type = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _pp_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                  attributes=attributes or {}, regions=regions)
                constants = {}
                def c(n):
                    if n not in constants:
                        constants[n] = create("arith.constant", results=[index], attributes={"value": _pp_ir.IntegerAttr.get(index, n)}).results[0]
                    return constants[n]
                # Constants dominate all subsequently constructed loops.
                for n in [0, 1, lanes, *extents, *range(lanes), extents[axis] // lanes * lanes]:
                    c(n)
                def packet(current, indices, width):
                    lane_indices = [indices]
                    for lane in range(1, width):
                        coordinate = create("arith.addi", [indices[axis], c(lane)], [index]).results[0]
                        ids = list(indices)
                        ids[axis] = coordinate
                        lane_indices.append(ids)
                    mappings = []
                    shared_extracts = {}
                    args = list(old.regions[0].blocks[0].arguments)
                    for ids in lane_indices:
                        mapped = {}
                        for arg, operand, pos in zip(args[:-1], old.operands[:-1], positions[:-1]):
                            key = (operand, *[c(0) if d is None else ids[d] for d in pos])
                            if broadcast and key in shared_extracts:
                                mapped[arg] = shared_extracts[key]
                            else:
                                mapped[arg] = create("tensor.extract", [operand, *key[1:]], [arg.type]).results[0]
                                if broadcast:
                                    shared_extracts[key] = mapped[arg]
                        mappings.append(mapped)
                    # Breadth-first scalar operation order exposes independent chains.
                    for scalar_op in scalar_body[:-1]:
                        for mapped in mappings:
                            cloned = create(scalar_op.operation.name,
                                [mapped.get(v, v) for v in scalar_op.operands],
                                [r.type for r in scalar_op.results],
                                {name: scalar_op.attributes[name] for name in scalar_op.attributes})
                            mapped[scalar_op.results[0]] = cloned.results[0]
                    for ids, mapped in zip(lane_indices, mappings):
                        value = mapped.get(scalar_body[-1].operands[0], scalar_body[-1].operands[0])
                        current = create("tensor.insert", [value, current, *[ids[d] for d in positions[-1]]], [tensor_type]).results[0]
                    return current
                def nest(depth, current, indices):
                    if depth == len(extents) - 1:
                        end = extents[-1] // lanes * lanes
                        if end:
                            loop = create("scf.for", [c(0), c(end), c(lanes), current], [tensor_type], regions=1)
                            block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                            with _pp_ir.InsertionPoint(block):
                                value = packet(block.arguments[1], [*indices, block.arguments[0]], lanes)
                                create("scf.yield", [value])
                            current = loop.results[0]
                        tail = extents[-1] - end
                        if tail:
                            current = packet(current, [*indices, c(end)], tail)
                        return current
                    loop = create("scf.for", [c(0), c(extents[depth]), c(1), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = nest(depth + 1, block.arguments[1], [*indices, block.arguments[0]])
                        create("scf.yield", [value])
                    return loop.results[0]
                def broadcast_group(depth, current, coordinates, width):
                    order = [d for d in range(len(extents)) if d != axis]
                    if depth == len(order):
                        return packet(current, [coordinates[d] for d in range(len(extents))], width)
                    d = order[depth]
                    loop = create("scf.for", [c(0), c(extents[d]), c(1), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = broadcast_group(depth + 1, block.arguments[1],
                                                {**coordinates, d: block.arguments[0]}, width)
                        create("scf.yield", [value])
                    return loop.results[0]

                if broadcast:
                    current = old.operands[-1]
                    end = extents[axis] // lanes * lanes
                    loop = create("scf.for", [c(0), c(end), c(lanes), current], [tensor_type], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index, tensor_type])
                    with _pp_ir.InsertionPoint(block):
                        value = broadcast_group(0, block.arguments[1], {axis: block.arguments[0]}, lanes)
                        create("scf.yield", [value])
                    result = loop.results[0]
                    tail = extents[axis] - end
                    if tail:
                        result = broadcast_group(0, result, {axis: c(end)}, tail)
                else:
                    result = nest(0, old.operands[-1], [])
                replacement = result.owner
                for name in old.attributes:
                    if name.startswith("prov."):
                        replacement.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                text = _pp_ir.StringAttr(previous).value + "," if previous is not None else ""
                transform = ("scalar_pointwise_two_multiplications_packet_" if two_products else
                             "scalar_pointwise_multiplication_packet_" if multiplication else
                             "scalar_pointwise_broadcast_packet_" if broadcast else "scalar_pointwise_packet_")
                replacement.attributes["prov.transforms"] = _pp_ir.StringAttr.get(text + transform + str(lanes))
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)


def _borrowed_pointwise_packet_spec(op):
    from torch_mlir import ir as _pp_ir
    contract_name = "merlin.borrowed_pointwise_effects"
    permissions = {"immutable_inputs", "fresh_disjoint_output", "stable_rounding",
                   "nontrapping_arithmetic", "exception_flags_unobserved"}
    if (op.operation.name != "linalg.generic" or op.results
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or contract_name not in op.attributes or len(op.operands) < 2):
        return None
    contract = op.attributes[contract_name]
    if not isinstance(contract, _pp_ir.DictAttr):
        return None
    if ({entry.name for entry in contract} != permissions
            or any(not isinstance(contract[key], _pp_ir.UnitAttr) for key in permissions)):
        return None
    types = [value.type for value in op.operands]
    if any(not isinstance(t, _pp_ir.MemRefType) or any(d <= 0 for d in t.shape) for t in types):
        return None
    maps = [_pp_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]] != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    positions, extents = [], [None] * dims
    for mapping, typ in zip(maps, types):
        if len(mapping.results) != len(typ.shape):
            return None
        try:
            strides, offset = typ.get_strides_and_offset()
        except (ValueError, TypeError, RuntimeError):
            return None
        if any(stride <= 0 for stride in strides) or offset < 0:
            return None
        pos = []
        for expr, extent in zip(mapping.results, typ.shape):
            if not isinstance(expr, _pp_ir.AffineDimExpr):
                return None
            dim = _pp_ir.AffineDimExpr(expr).position
            if dim in pos or (extents[dim] is not None and extents[dim] != extent):
                return None
            pos.append(dim)
            extents[dim] = extent
        positions.append(pos)
    # A private span does not prove that distinct logical output coordinates
    # have distinct addresses. Require a separable positive-stride layout;
    # arbitrary affine layouts and overlapping rows conservatively refuse.
    output_strides, _ = types[-1].get_strides_and_offset()
    covered = 1
    for stride, extent in sorted(zip(output_strides, types[-1].shape)):
        if extent > 1:
            if stride < covered:
                return None
            covered += (extent - 1) * stride
    # Exactly one output argument and a one-to-one full output map. Linalg
    # operand segment sizes are authoritative; scalar init reads still refuse.
    if (len(op.attributes["operandSegmentSizes"]) != 2
            or list(op.attributes["operandSegmentSizes"])[-1] != 1
            or sorted(positions[-1]) != list(range(dims))
            or any(extent is None for extent in extents)):
        return None
    body = _pointwise_packet_body_spec(op, types)
    return (positions, extents, body) if body is not None else None


def _packetize_borrowed_pointwise(ctx, module):
    from torch_mlir import ir as _pp_ir
    todo = []
    def walk(op):
        if any("strictfp" in name or "strictfp" in str(op.attributes[name]) for name in op.attributes):
            return
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _borrowed_pointwise_packet_spec(inner)
                    if spec is None:
                        walk(inner.operation)
                    else:
                        todo.append((inner, spec))
    walk(module.operation)
    with ctx:
        for old, (positions, extents, scalar_body) in todo:
            with old.location, _pp_ir.InsertionPoint(old):
                index = _pp_ir.IndexType.get()
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _pp_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                   attributes=attributes or {}, regions=regions)
                constants = {}
                def c(value):
                    if value not in constants:
                        constants[value] = create("arith.constant", results=[index],
                            attributes={"value": _pp_ir.IntegerAttr.get(index, value)}).results[0]
                    return constants[value]
                for value in [0, 1, 2, *extents, extents[-1] // 2 * 2]:
                    c(value)
                def packet(indices, width):
                    mappings, lane_indices = [], []
                    args = list(old.regions[0].blocks[0].arguments)
                    for lane in range(width):
                        ids = list(indices)
                        if lane:
                            ids[-1] = create("arith.addi", [ids[-1], c(lane)], [index]).results[0]
                        lane_indices.append(ids)
                        mapped = {}
                        for arg, operand, pos in zip(args[:-1], old.operands[:-1], positions[:-1]):
                            mapped[arg] = create(
                                "memref.load", [operand, *[ids[d] for d in pos]], [arg.type]
                            ).results[0]
                        mappings.append(mapped)
                    for scalar_op in scalar_body[:-1]:
                        for mapped in mappings:
                            cloned = create(scalar_op.operation.name,
                                [mapped.get(v, v) for v in scalar_op.operands],
                                [r.type for r in scalar_op.results],
                                {name: scalar_op.attributes[name] for name in scalar_op.attributes})
                            mapped[scalar_op.results[0]] = cloned.results[0]
                    stores = []
                    for ids, mapped in zip(lane_indices, mappings):
                        value = mapped.get(scalar_body[-1].operands[0], scalar_body[-1].operands[0])
                        stores.append(create(
                            "memref.store", [value, old.operands[-1], *[ids[d] for d in positions[-1]]]
                        ))
                    return stores
                def nest(depth, indices):
                    if depth == len(extents) - 1:
                        end = extents[-1] // 2 * 2
                        replacements = []
                        if end:
                            loop = create("scf.for", [c(0), c(end), c(2)], regions=1)
                            block = _pp_ir.Block.create_at_start(loop.regions[0], [index])
                            with _pp_ir.InsertionPoint(block):
                                packet([*indices, block.arguments[0]], 2)
                                create("scf.yield")
                            replacements.append(loop)
                        if extents[-1] != end:
                            replacements.extend(packet([*indices, c(end)], 1))
                        return replacements
                    loop = create("scf.for", [c(0), c(extents[depth]), c(1)], regions=1)
                    block = _pp_ir.Block.create_at_start(loop.regions[0], [index])
                    with _pp_ir.InsertionPoint(block):
                        nest(depth + 1, [*indices, block.arguments[0]])
                        create("scf.yield")
                    return [loop]
                replacements = nest(0, [])
                # A buffer writer may have both a packet loop and a scalar tail
                # at the top level. Each retains the enclosing source trace;
                # nested stores are represented by their outer loop.
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                transforms = _pp_ir.StringAttr(previous).value + "," if previous is not None else ""
                for replacement in replacements:
                    for name in old.attributes:
                        if name.startswith("prov."):
                            replacement.attributes[name] = old.attributes[name]
                    replacement.attributes["prov.transforms"] = _pp_ir.StringAttr.get(
                        transforms + "borrowed_scalar_pointwise_packet_2")
                old.operation.erase()
    module.operation.verify()
    return len(todo)

_PP_MARKERS = {"__merlin_borrowed_pointwise_packet_2__": 2,
               "__merlin_pointwise_packet_2__": 2, "__merlin_pointwise_packet_4__": 4,
               "__merlin_pointwise_broadcast_packet_2__": 2,
               "__merlin_pointwise_multiplication_packet_4__": 4,
               "__merlin_pointwise_two_multiplications_packet_4__": 4}
_PP_ORIG_RUN_STAGES = _run_stages

def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    selected = [m for m in passes if m in _PP_MARKERS]
    if not selected:
        return _PP_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    multiplication_marker = "__merlin_pointwise_multiplication_packet_4__"
    two_products_marker = "__merlin_pointwise_two_multiplications_packet_4__"
    multiplication_markers = {multiplication_marker, two_products_marker}
    borrowed_marker = "__merlin_borrowed_pointwise_packet_2__"
    if (len([m for m in selected if m not in multiplication_markers and m != borrowed_marker]) > 1
            or any(selected.count(m) > 1 for m in _PP_MARKERS)):
        raise ValueError("pointwise packet schedules are alternatives")
    begin = 0
    for marker in selected:
        i = passes.index(marker, begin)
        _PP_ORIG_RUN_STAGES(ctx, module, ','.join(passes[begin:i]), 0, (), (), (),
                           pre_generalize if begin == 0 else ())
        if marker == borrowed_marker:
            print('OK borrowed_pointwise_packet', 2, _packetize_borrowed_pointwise(ctx, module))
            begin = i + 1
            continue
        print('OK scalar_pointwise_packet', _PP_MARKERS[marker], _packetize_pointwise(
            ctx, module, _PP_MARKERS[marker],
            broadcast=marker == "__merlin_pointwise_broadcast_packet_2__",
            multiplication=marker in multiplication_markers,
            two_products=marker == two_products_marker))
        begin = i + 1
    _PP_ORIG_RUN_STAGES(ctx, module, ','.join(passes[begin:]), erase, mid, late, post_openmp, ())
"""


def apply_for_test(
    mlir_text: str,
    *,
    lanes: int = 2,
    broadcast: bool = False,
    multiplication: bool = False,
    two_products: bool = False,
    borrowed: bool = False,
) -> tuple[str, int]:
    """Run the shipped structural rewrite in the upstream toolchain's Python."""
    import subprocess
    import tempfile
    from pathlib import Path

    from merlin.compiler.lower.model.toolchain import m2m_python

    if type(lanes) is not int or lanes not in (2, 4):
        raise ValueError("pointwise packet lanes must be 2 or 4")
    if type(broadcast) is not bool or (broadcast and lanes != 2):
        raise ValueError("broadcast packet requires an explicit boolean and two lanes")
    if type(multiplication) is not bool or (multiplication and (lanes != 4 or broadcast)):
        raise ValueError("multiplication packet requires an explicit boolean and four ordinary lanes")
    if type(two_products) is not bool or (two_products and not multiplication):
        raise ValueError("two-product packet requires explicit multiplication selection")
    if type(borrowed) is not bool or (borrowed and (lanes != 2 or broadcast or multiplication or two_products)):
        raise ValueError("borrowed pointwise packet requires explicit two ordinary lanes")
    invocation = (
        "_packetize_borrowed_pointwise(ctx,module)"
        if borrowed
        else (
            f"_packetize_pointwise(ctx,module,{lanes},broadcast={broadcast},"
            f"multiplication={multiplication},two_products={two_products})"
        )
    )
    with tempfile.TemporaryDirectory(prefix="merlin_pointwise_packet_") as directory:
        root = Path(directory)
        src, script = root / "input.mlir", root / "run.py"
        src.write_text(mlir_text)
        script.write_text(
            "import sys\nfrom torch_mlir import ir\n"
            + RUNNER_PRELUDE.split("_PP_MARKERS =", 1)[0]
            + "ctx=ir.Context()\nmodule=ir.Module.parse(open(sys.argv[1]).read(),ctx)\n"
            + f"print('COUNT',{invocation})\nprint('MODULE_BEGIN')\nprint(module)\n"
        )
        proc = subprocess.run([str(m2m_python()), str(script), str(src)], capture_output=True, text=True, timeout=120)
        if proc.returncode:
            raise RuntimeError(f"pointwise packet rewrite failed:\n{proc.stdout}\n{proc.stderr}")
        count = int(next(line.split()[1] for line in proc.stdout.splitlines() if line.startswith("COUNT ")))
        return proc.stdout.split("MODULE_BEGIN\n", 1)[1], count
