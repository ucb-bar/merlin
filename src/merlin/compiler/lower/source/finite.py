"""Source-derived finite domains for broadcast scaled integer producers.

This analysis supplies a conservative integer-bit scale guard. It changes no
source arithmetic, installs no route, and grants no alias or floating effect
permission. A provider must bind the runtime scan, source layout/lifetime, and
RNE mode before omitting a corresponding per-element finite check.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from fractions import Fraction

from xdsl.ir import SSAValue

from merlin.compiler.lower.quant.late_quant_rne import _identity, _Instruction, _match, _match_intrinsic
from merlin.compiler.lower.source.interval import IntervalEffectContract, _numeric_context, _scalar_attributes
from merlin.compiler.lower.source.ordered_fma import _context_snapshot, _snapshot, _supported

# ---- scaled_integer_finite ---------------------------------------------------------------------

_MAX_WORD = 0x7F7FFFFF


def _exact_positive(word: int) -> Fraction:
    if type(word) is not int or not 0 <= word <= _MAX_WORD:
        raise ValueError("finite positive binary32 word required")
    exponent, fraction = word >> 23, word & 0x7FFFFF
    significand = fraction if exponent == 0 else fraction | 0x800000
    shift = -149 if exponent == 0 else exponent - 150
    return Fraction(significand) * Fraction(2) ** shift


def _floor_word(value: Fraction) -> int:
    if value < 0:
        raise ValueError("nonnegative finite-domain bound required")
    low, high = 0, _MAX_WORD
    while low < high:
        middle = (low + high + 1) // 2
        if _exact_positive(middle) <= value:
            low = middle
        else:
            high = middle - 1
    return low


@dataclass(frozen=True)
class ScaledIntegerFiniteDomain:
    """Original signed conversion and ordered constant multiplies, then scale."""

    integer_bits: int
    constant_words: tuple[int, ...]

    def scale_limit_word(self) -> int:
        if type(self.integer_bits) is not int or not 2 <= self.integer_bits <= 64:
            raise ValueError("signed integer width in [2,64] required")
        if not self.constant_words:
            raise ValueError("at least one original constant multiply required")
        # RN conversion of the entire signed domain is enclosed by +/-2^(w-1),
        # including the rounded positive maximum. Every subsequent rounded
        # intermediate is bounded above, without reassociating source products.
        bound = Fraction(2) ** (self.integer_bits - 1)
        maximum = _exact_positive(_MAX_WORD)
        for word in self.constant_words:
            if type(word) is not int or not 0 <= word < 1 << 32:
                raise ValueError("literal binary32 word required")
            exact = bound * _exact_positive(word & 0x7FFFFFFF)
            if exact > maximum:
                raise ValueError("constant chain may overflow before the variable scale")
            floor = _floor_word(exact)
            bound = _exact_positive(floor if _exact_positive(floor) == exact else floor + 1)
        return _MAX_WORD if bound == 0 else _floor_word(maximum / bound)

    def admits_scale_word(self, word: int) -> bool:
        if type(word) is not int or not 0 <= word < 1 << 32:
            raise ValueError("literal scale binary32 word required")
        return word & 0x7FFFFFFF <= self.scale_limit_word()


@dataclass(frozen=True)
class BroadcastScaledIntegerFiniteProof:
    value: object
    operation: object
    integer_operand: int
    scale_operand: int
    scale_axes: tuple[int, ...]
    domain_extents: tuple[int, ...]
    repetition: int
    domain: ScaledIntegerFiniteDomain
    _witness: tuple = field(repr=False)
    _contexts: tuple = field(repr=False)
    _effects: IntervalEffectContract = field(repr=False)


def prove_broadcast_scaled_integer_finite(value, *, effects: IntervalEffectContract):
    """Retain a typed cast/ordered-constant-multiply/broadcast-scale witness.

    Exact static parallel indexing maps establish the smaller immutable scale
    domain. Extra uses of original values remain legal: no value is replaced or
    moved. Empty/reduction/unknown-effect/strict contexts refuse preparation.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, IntegerType, Signedness, TensorType, f32
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import BlockArgument, OpResult
    from xdsl.ir.affine import AffineDimExpr

    from merlin.compiler.ir.mlir_query import op_name

    effects.validate()
    if not isinstance(value, OpResult) or value.type != f32:
        raise ValueError("typed binary32 producer required")
    block = value.owner.parent_block()
    parent = block.parent_region().parent_op() if block is not None else None
    if not isinstance(parent, linalg.GenericOp) or not _supported(parent):
        raise ValueError("nonempty static pure parallel tensor map required")
    if any(op_name(op).startswith(("func.", "llvm.")) and not _roundeven_intrinsic(op) for op in block.ops):
        raise ValueError("unknown call or effects in preparation consumer")
    if value.owner.name != "arith.mulf":
        raise ValueError("original final scale multiply required")
    scales = [v for v in value.owner.operands if isinstance(v, BlockArgument) and v.type == f32 and v.owner is block]
    if len(scales) != 1:
        raise ValueError("one tensor block-argument scale required")
    scale = scales[0]
    current = next(v for v in value.owner.operands if v is not scale)
    chain, words, constants = [], [], []
    while isinstance(current, OpResult) and current.owner.name == "arith.mulf":
        operation = current.owner
        _scalar_attributes(operation)
        matches = [
            v
            for v in operation.operands
            if isinstance(v, OpResult)
            and isinstance(v.owner, arith.ConstantOp)
            and isinstance(v.owner.value, FloatAttr)
            and v.type == f32
        ]
        if len(matches) != 1:
            raise ValueError("ordered single-literal multiply chain required")
        constant = matches[0].owner
        _scalar_attributes(constant)
        words.append(struct.unpack("<I", struct.pack("<f", constant.value.value.data))[0])
        constants.append(constant)
        chain.append(operation)
        current = next(v for v in operation.operands if v is not matches[0])
    if not isinstance(current, OpResult) or current.owner.name != "arith.sitofp":
        raise ValueError("original signed integer conversion required")
    conversion = current.owner
    _scalar_attributes(conversion)
    integer = conversion.operands[0]
    if (
        current.type != f32
        or not isinstance(integer, BlockArgument)
        or integer.owner is not block
        or not isinstance(integer.type, IntegerType)
        or integer.type.signedness.data == Signedness.UNSIGNED
    ):
        raise ValueError("source tensor signed integer argument required")
    if integer.index >= len(parent.inputs) or scale.index >= len(parent.inputs):
        raise ValueError("output initializer is not an immutable producer input")
    maps = [a.data for a in parent.indexing_maps]
    integer_map, scale_map, output_map = maps[integer.index], maps[scale.index], maps[-1]
    if integer_map != output_map or any(not isinstance(e, AffineDimExpr) for e in scale_map.results):
        raise ValueError("complete integer domain and exact scale projection required")
    axes = tuple(e.position for e in scale_map.results)
    if len(set(axes)) != len(axes):
        raise ValueError("scale projection duplicates a domain axis")
    output_type = parent.outputs[0].type
    assert isinstance(output_type, TensorType)
    extents = [0] * output_map.num_dims
    for expr, extent in zip(output_map.results, output_type.get_shape(), strict=True):
        extents[expr.position] = extent
    repetition = math.prod(extents[i] for i in range(len(extents)) if i not in axes)
    if repetition <= 1:
        raise ValueError("scale domain has no repeated source consumers")
    _scalar_attributes(value.owner)
    domain = ScaledIntegerFiniteDomain(integer.type.width.data, tuple(reversed(words)))
    domain.scale_limit_word()
    operations = tuple(dict.fromkeys([parent, *block.ops, conversion, *chain, *constants]))
    contexts = []
    for operation in operations:
        current_op = operation
        while current_op is not None:
            _numeric_context(current_op)
            if current_op not in contexts:
                contexts.append(current_op)
            current_op = current_op.parent_op()
    return BroadcastScaledIntegerFiniteProof(
        value,
        parent,
        integer.index,
        scale.index,
        axes,
        tuple(extents),
        repetition,
        domain,
        tuple(_snapshot(op) for op in operations),
        tuple(_context_snapshot(op) for op in contexts),
        effects,
    )


def validate_broadcast_scaled_integer_finite(proof):
    if not isinstance(proof, BroadcastScaledIntegerFiniteProof):
        raise ValueError("typed source finite-domain witness required")
    if any(_snapshot(w.operation) != w for w in proof._witness) or any(
        _context_snapshot(w.operation) != w for w in proof._contexts
    ):
        raise ValueError("source finite-domain operations/context changed")
    current = prove_broadcast_scaled_integer_finite(proof.value, effects=proof._effects)
    fields = ("operation", "integer_operand", "scale_operand", "scale_axes", "domain_extents", "repetition", "domain")
    if any(getattr(current, name) != getattr(proof, name) for name in fields):
        raise ValueError("source finite-domain semantic fields changed")


def emit_finite_scale_scan(domain: ScaledIntegerFiniteDomain, *, symbol: str):
    """Portable read-only integer-bit scan; caller binds exact spans/lifetime.

    No FP instruction, rounding-mode change, or pointer cache is introduced.
    Negative/overflowing layouts refuse. Empty extent performs no input read.
    A true result is valid only with the separately bound RNE source contract.
    """
    if (
        not isinstance(domain, ScaledIntegerFiniteDomain)
        or not symbol
        or not symbol.isascii()
        or not symbol.isidentifier()
    ):
        raise ValueError("typed finite domain and explicit C identifier required")
    limit = domain.scale_limit_word()
    return f"""#include <stdint.h>
#include <stddef.h>
int {symbol}(const float *scale,size_t count,ptrdiff_t stride){{
 if(!count)return 1;
 if(!scale||stride<0||(stride&&(count-1)>(size_t)PTRDIFF_MAX/(size_t)stride))return 0;
 for(size_t i=0;i<count;i++){{uint32_t word;__builtin_memcpy(&word,scale+(ptrdiff_t)i*stride,4);
  if((word&0x7fffffffu)>0x{limit:08x}u)return 0;}}
 return 1;
}}
"""


# ---- bounded_rne_maps --------------------------------------------------------------------------
# Typed scalar proof adapter for bounded-RNE tensor map scheduling.
#
# This module recognizes arithmetic, not provenance labels. It shares the complete
# numeric graph proof with host LLVM legalization. The optional tensor scheduler
# additionally proves index maps, extents, immutable tensor dependencies and all
# tail accesses before grouping memory operations.


def _roundeven_intrinsic(operation):
    """The exact scalar intrinsic emitted by ordinary host rounding."""
    from xdsl.dialects.builtin import f32

    from merlin.compiler.ir.mlir_query import op_name

    return (
        op_name(operation) == "llvm.intr.roundeven"
        and len(operation.operands) == len(operation.results) == 1
        and operation.operands[0].type == operation.results[0].type == f32
        and not operation.regions
        and not operation.successors
        and not operation.properties
        and not set(operation.attributes) - {"op_name__"}
    )


@dataclass(frozen=True)
class ScalarRneProof:
    raw_input: SSAValue
    result: SSAValue
    integer_bits: int
    bounds: tuple[int, int]


def prove_scalar_bounded_rne(block) -> ScalarRneProof | None:
    """Accept the complete typed arithmetic ties-even graph, else refuse.

    Unknown operations, nested regions, fast FP and integer overflow promises
    refuse the entire body. Constants are interpreted from typed attributes.
    No model, symbol, source-node or transform attribute selects acceptance.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType, f32

    ops = list(block.ops)
    if not ops or ops[-1].name not in ("linalg.yield", "func.return") or len(ops[-1].operands) != 1:
        return None
    values = {v: f"%arg{i}" for i, v in enumerate(block.args)}
    reverse = {_identity(n): v for v, n in values.items()}
    definitions = {}

    def dtype(t):
        return "float" if t == f32 else str(t) if isinstance(t, IntegerType) else None

    binary = {
        "arith.addi": "add",
        "arith.andi": "and",
        "arith.ori": "or",
        "arith.subf": "fsub",
        "arith.mulf": "fmul",
    }
    for serial, op in enumerate(ops[:-1]):
        if op.regions or len(op.results) != 1 or any(x not in values for x in op.operands):
            return None
        for key in ("fastmath", "overflowFlags"):
            flag = op.properties.get(key)
            if flag is not None and (
                not isinstance(flag, (arith.FastMathFlagsAttr, arith.IntegerOverflowAttr)) or flag.data
            ):
                return None
        result = op.results[0]
        if isinstance(op, arith.ConstantOp):
            value = op.value
            if isinstance(value, (FloatAttr, IntegerAttr)):
                values[result] = repr(value.value.data)
                continue
            return None
        name = f"%v{serial}"
        operands = tuple(values[v] for v in op.operands)
        predicate = callee = output_type = ""
        if op.name in binary and len(operands) == 2:
            opcode, typ = binary[op.name], dtype(result.type)
        elif op.name == "arith.negf" and len(operands) == 1:
            opcode, typ = "fneg", dtype(result.type)
        elif op.name in ("arith.minimumf", "arith.maximumf") and len(operands) == 2:
            opcode, typ = "call", dtype(result.type)
            callee = "llvm.minimum.f32" if op.name == "arith.minimumf" else "llvm.maximum.f32"
        elif _roundeven_intrinsic(op):
            opcode, typ, callee = "call", "float", "llvm.roundeven.f32"
        elif op.name in ("arith.fptosi", "arith.sitofp") and len(operands) == 1:
            opcode = "fptosi" if op.name == "arith.fptosi" else "sitofp"
            typ, output_type = dtype(op.operands[0].type), dtype(result.type)
        elif op.name in ("arith.cmpf", "arith.cmpi") and len(operands) == 2:
            opcode = "fcmp" if op.name == "arith.cmpf" else "icmp"
            predicates = {1: "oeq", 2: "ogt", 4: "olt"} if opcode == "fcmp" else {1: "ne"}
            predicate = predicates.get(op.predicate.value.data)
            if predicate is None:
                return None
            typ = dtype(op.operands[0].type)
        elif op.name == "arith.select" and len(operands) == 3:
            opcode, typ = "select", dtype(result.type)
        else:
            return None
        if typ is None or output_type is None:
            return None
        values[result] = name
        reverse[_identity(name)] = result
        definitions[_identity(name)] = _Instruction(
            name,
            opcode,
            typ,
            operands,
            serial,
            serial + 1,
            predicate,
            callee,
            output_type,
        )
    yielded = ops[-1].operands[0]
    token = values.get(yielded, "")
    final = definitions.get(_identity(token))
    proof = (_match(final, definitions) or _match_intrinsic(final, definitions)) if final is not None else None
    if proof is None:
        return None
    raw = reverse.get(_identity(proof["raw_input"]))
    if raw is None:
        return None
    return ScalarRneProof(raw, yielded, int(proof["integer_dtype"][1:]), tuple(proof["bounds"]))


def schedule_bounded_rne_maps(module, *, lanes=4, symbol_prefix="bounded_rne_packet", output_minor_batch=1):
    """Stripmine proved pure tensor maps; retain original arithmetic in helpers.

    This explicit transform changes no numeric contract. Inputs are immutable
    tensor values and the output is carried through SCF; upstream bufferization
    remains responsible for any copy required by live aliases. Full packets and
    static tails are separate, so no load or store relies on masking. Affine
    permutation inputs can be traced through one or more tensor transposes.

    ``output_minor_batch`` explicitly groups independent output-minor coordinates
    inside a contiguous-input packet loop. Each coordinate still invokes the
    same scalar-body packet, with exact tails on both dimensions. This is a
    bounded emission candidate; it makes no register, alignment or cycle claim.
    One preserves the original schedule. A coincident source/output axis or a
    unit output-minor extent keeps the original schedule with a reported fallback.
    """
    from xdsl.dialects import arith, func, scf, tensor
    from xdsl.dialects.builtin import IndexType, NoneAttr, StringAttr, TensorType
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if type(lanes) is not int or not 1 <= lanes <= 8:
        raise ValueError("packet width must be an integer in [1,8]")
    if type(output_minor_batch) is not int or not 1 <= output_minor_batch <= 8:
        raise ValueError("output minor batch must be an integer in [1,8] (bounded emission limit)")
    if (
        not isinstance(symbol_prefix, str)
        or not symbol_prefix
        or symbol_prefix[0].isdigit()
        or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in symbol_prefix)
    ):
        raise ValueError("explicit valid symbol prefix required")
    module.verify()
    # Do not turn a dead pure tensor branch into opaque helper calls. Unknown
    # effects, terminators and cycles conservatively count as observable uses.
    from xdsl.transforms.dead_code_elimination import would_be_trivially_dead

    def has_observable_use(value):
        pending = [use.operation for use in value.uses]
        seen = set()
        while pending:
            user = pending.pop()
            if user in seen:
                continue
            seen.add(user)
            if not would_be_trivially_dead(user):
                return True
            pending.extend(use.operation for result in user.results for use in result.uses)
        return False

    plans = []
    for op in list(module.walk()):
        if not isinstance(op, linalg.GenericOp) or len(op.outputs) != 1 or len(op.results) != 1:
            continue
        if not has_observable_use(op.results[0]):
            continue
        output = op.outputs[0]
        typ = output.type
        if not isinstance(typ, TensorType) or not isinstance(typ.encoding, NoneAttr):
            continue
        shape = typ.get_shape()
        rank = len(shape)
        if not rank or any(n <= 0 for n in shape) or op.results[0].type != typ:
            continue
        if len(op.body.blocks) != 1 or op.body.block.args[-1].uses:
            continue
        if tuple(x.data for x in op.iterator_types) != (linalg.IteratorType.PARALLEL,) * rank:
            continue
        maps = tuple(x.data for x in op.indexing_maps)
        if len(maps) != len(op.inputs) + 1 or maps[-1] != AffineMap.identity(rank):
            continue
        proof = prove_scalar_bounded_rne(op.body.block)
        if proof is None:
            continue
        sources = []
        legal = True
        for value, amap in zip(op.inputs, maps[:-1]):
            if amap.num_dims != rank or amap.num_symbols or any(not isinstance(e, AffineDimExpr) for e in amap.results):
                legal = False
                break
            dims = tuple(e.position for e in amap.results)
            if len(set(dims)) != len(dims):
                legal = False
                break
            if isinstance(value.type, TensorType):
                if not isinstance(value.type.encoding, NoneAttr) or value.type.get_shape() != tuple(
                    shape[d] for d in dims
                ):
                    legal = False
                    break
                while isinstance(value.owner, linalg.TransposeOp):
                    transpose = value.owner
                    perm = tuple(transpose.permutation.get_values())
                    if sorted(perm) != list(range(len(dims))) or len(transpose.inputs) != 1:
                        legal = False
                        break
                    original = transpose.inputs[0]
                    if (
                        not isinstance(original.type, TensorType)
                        or not isinstance(original.type.encoding, NoneAttr)
                        or original.type.get_shape()
                        != tuple(value.type.get_shape()[perm.index(d)] for d in range(len(dims)))
                    ):
                        legal = False
                        break
                    dims = tuple(dims[perm.index(d)] for d in range(len(dims)))
                    value = original
                if not legal:
                    break
            elif dims:
                legal = False
                break
            sources.append((value, dims))
        if not legal or not any(dims for _, dims in sources):
            continue
        # Follow a contiguous source dimension rather than guessing a model layout.
        axis = next(dims[-1] for _, dims in sources if dims)
        plans.append((op, proof, shape, sources, axis))
    names = {op.sym_name.data for op in module.walk() if isinstance(op, func.FuncOp)}
    reserved = {
        attr.data
        for op in module.walk()
        if isinstance(
            attr := op.properties.get("sym_name", op.attributes.get("sym_name")),
            StringAttr,
        )
    }
    reports = []
    for op, proof, shape, sources, axis in plans:
        output = op.outputs[0]
        typ = output.type
        body = op.body.block
        rank = len(shape)
        minor_axis = rank - 1
        minor_width = min(output_minor_batch, shape[minor_axis]) if axis != minor_axis else 1
        batched = minor_width > 1
        widths = {lanes}
        if shape[axis] % lanes:
            widths.add(shape[axis] % lanes)
        helpers = {}
        for width in sorted(widths):
            serial = len(names)
            name = f"{symbol_prefix}_{serial}_{width}"
            while name in names or name in reserved:
                serial += 1
                name = f"{symbol_prefix}_{serial}_{width}"
            names.add(name)
            block = Block(arg_types=[arg.type for _ in range(width) for arg in body.args[:-1]])
            results = []
            for lane in range(width):
                mapping = dict(
                    zip(
                        body.args[:-1],
                        block.args[lane * len(sources) : (lane + 1) * len(sources)],
                    )
                )
                for old in list(body.ops)[:-1]:
                    block.add_op(old.clone(mapping))
                results.append(mapping[body.last_op.operands[0]])
            block.add_op(func.ReturnOp(*results))
            helper = func.FuncOp(
                name,
                ([x.type for x in block.args], [proof.result.type] * width),
                Region(block),
                visibility="private",
            )
            module.body.block.add_op(helper)
            helpers[width] = name
        constant_values = {
            0,
            1,
            lanes,
            *shape,
            *range(lanes),
            (shape[axis] // lanes) * lanes,
        }
        if batched:
            constant_values.update((minor_width, (shape[minor_axis] // minor_width) * minor_width))
            constant_values.update(range(minor_width))
        constants = {n: arith.ConstantOp.from_int_and_width(n, IndexType()) for n in constant_values}

        def c(n):
            return constants[n].result

        root = Block()
        uniform = {}
        for i, (source, dims) in enumerate(sources):
            if not dims and isinstance(source.type, TensorType):
                extract = tensor.ExtractOp(source, [], source.type.element_type)
                root.add_op(extract)
                uniform[i] = extract.result
            elif not dims:
                uniform[i] = source

        def packet(parent, current, indices, width):
            all_values = []
            all_indices = []
            for lane in range(width):
                ix = list(indices)
                if lane:
                    add = arith.AddiOp(ix[axis], c(lane))
                    parent.add_op(add)
                    ix[axis] = add.result
                all_indices.append(ix)
                for i, (source, dims) in enumerate(sources):
                    if i in uniform:
                        all_values.append(uniform[i])
                        continue
                    extract = tensor.ExtractOp(source, [ix[d] for d in dims], source.type.element_type)
                    parent.add_op(extract)
                    all_values.append(extract.result)
            call = func.CallOp(helpers[width], all_values, [proof.result.type] * width)
            parent.add_op(call)
            for value, ix in zip(call.results, all_indices):
                insert = tensor.InsertOp(value, current, ix)
                parent.add_op(insert)
                current = insert.result
            return current

        order = [d for d in range(rank) if d != axis and (not batched or d != minor_axis)]

        def grouped_packet(parent, current, indices, width, minor_count):
            for minor_offset in range(minor_count):
                ix = list(indices)
                if minor_offset:
                    add = arith.AddiOp(ix[minor_axis], c(minor_offset))
                    parent.add_op(add)
                    ix[minor_axis] = add.result
                current = packet(parent, current, ix, width)
            return current

        def axis_packets(parent, current, indices, minor_count):
            full = (shape[axis] // lanes) * lanes
            if full:
                bb = Block(arg_types=[IndexType(), typ])
                ix = list(indices)
                ix[axis] = bb.args[0]
                value = grouped_packet(bb, bb.args[1], ix, lanes, minor_count)
                bb.add_op(scf.YieldOp(value))
                loop = scf.ForOp(c(0), c(full), c(lanes), [current], Region(bb))
                parent.add_op(loop)
                current = loop.results[0]
            tail = shape[axis] - full
            if tail:
                ix = list(indices)
                ix[axis] = c(full)
                current = grouped_packet(parent, current, ix, tail, minor_count)
            return current

        def loops(parent, depth, current, indices):
            if depth < len(order):
                d = order[depth]
                bb = Block(arg_types=[IndexType(), typ])
                ix = list(indices)
                ix[d] = bb.args[0]
                value = loops(bb, depth + 1, bb.args[1], ix)
                bb.add_op(scf.YieldOp(value))
                loop = scf.ForOp(c(0), c(shape[d]), c(1), [current], Region(bb))
                parent.add_op(loop)
                return loop.results[0]
            if not batched:
                return axis_packets(parent, current, indices, 1)
            full = (shape[minor_axis] // minor_width) * minor_width
            if full:
                bb = Block(arg_types=[IndexType(), typ])
                ix = list(indices)
                ix[minor_axis] = bb.args[0]
                value = axis_packets(bb, bb.args[1], ix, minor_width)
                bb.add_op(scf.YieldOp(value))
                loop = scf.ForOp(c(0), c(full), c(minor_width), [current], Region(bb))
                parent.add_op(loop)
                current = loop.results[0]
            tail = shape[minor_axis] - full
            if tail:
                ix = list(indices)
                ix[minor_axis] = c(full)
                current = axis_packets(parent, current, ix, tail)
            return current

        value = loops(root, 0, output, [c(0)] * rank)
        detached = [*constants.values(), *[root.detach_op(x) for x in list(root.ops)]]
        parent = op.parent_block()
        parent.insert_ops_before(detached, op)
        op.results[0].replace_all_uses_with(value)
        parent.erase_op(op)
        reports.append(
            {
                "shape": list(shape),
                "packet_axis": axis,
                "lanes": lanes,
                "tail": shape[axis] % lanes,
                "integer_bits": proof.integer_bits,
                "bounds": list(proof.bounds),
                "helpers": helpers,
                "numeric_contract": "original scalar body cloned; no reassociation",
                "memory_contract": (
                    "immutable input tensors, carried output tensor; upstream bufferization owns alias copies"
                ),
            }
        )
        if output_minor_batch != 1:
            reports[-1]["output_minor_batch"] = {
                "requested": output_minor_batch,
                "selected": minor_width,
                "axis": minor_axis,
                "tail": shape[minor_axis] % minor_width if batched else 0,
                "status": "CANDIDATE" if batched else "ORIGINAL_SCHEDULE_FALLBACK",
                "fallback_reason": None if batched else "source/output axes coincide or output-minor extent is one",
                "profitability": "UNKNOWN; packet arithmetic and traffic unchanged",
                "pointer_alignment": "NO_NEW_ALIGNMENT_FACT",
            }
    module.verify()
    return reports


# ---- closed_tensor_insert ----------------------------------------------------------------------
# Typed immutable tensor publication for one current scalar observation.
#
# No memory operation is moved. A static tensor carrier and its lane coordinates
# stay in their original block; only a private integer-producing DAG may change.


@dataclass(frozen=True)
class _Coordinate:
    variable: object | None
    offset: int
    lower: int
    upper: int


def _constant_index(value):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IndexType, IntegerAttr

    operation = value.owner
    if (
        not isinstance(operation, arith.ConstantOp)
        or not isinstance(value.type, IndexType)
        or not isinstance(operation.value, IntegerAttr)
    ):
        raise ValueError("static typed index bound required")
    return operation.value.value.data, operation


def _loop_bounds(loop):
    from xdsl.dialects.builtin import IndexType, TensorType

    if (
        loop.name != "scf.for"
        or len(loop.operands) != 4
        or len(loop.results) != 1
        or not isinstance(loop.results[0].type, TensorType)
        or len(loop.regions) != 1
        or len(loop.regions[0].blocks) != 1
    ):
        raise ValueError("one immutable tensor SCF carrier required")
    block = loop.regions[0].block
    if (
        len(block.args) != 2
        or not isinstance(block.args[0].type, IndexType)
        or block.args[1].type != loop.results[0].type
        or loop.operands[3].type != loop.results[0].type
    ):
        raise ValueError("typed SCF tensor carrier does not agree")
    lower, lower_op = _constant_index(loop.operands[0])
    limit, limit_op = _constant_index(loop.operands[1])
    step, step_op = _constant_index(loop.operands[2])
    if lower < 0 or limit <= lower or step <= 0:
        raise ValueError("nonempty nonnegative static SCF interval required")
    upper = lower + (limit - lower - 1) // step * step
    return lower, upper, step, (loop, lower_op, limit_op, step_op)


def _coordinate(value):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IndexType
    from xdsl.ir import BlockArgument

    if not isinstance(value.type, IndexType):
        raise ValueError("tensor publication coordinate is not index typed")
    if isinstance(value.owner, arith.ConstantOp):
        number, operation = _constant_index(value)
        return _Coordinate(None, number, number, number), (operation,)
    if isinstance(value, BlockArgument):
        loop = value.owner.parent_op()
        if value.index != 0 or loop is None or loop.name != "scf.for":
            raise ValueError("tensor lane coordinate is not a proved SCF induction value")
        lower, upper, _, witnesses = _loop_bounds(loop)
        return _Coordinate(value, 0, lower, upper), witnesses
    operation = value.owner
    if operation.name != "arith.addi" or len(operation.operands) != 2 or len(operation.results) != 1:
        raise ValueError("tensor lane coordinate has unsupported arithmetic")
    left, left_ops = _coordinate(operation.operands[0])
    right, right_ops = _coordinate(operation.operands[1])
    if left.variable is not None and right.variable is not None:
        raise ValueError("tensor lane coordinate combines induction values")
    return _Coordinate(
        left.variable if left.variable is not None else right.variable,
        left.offset + right.offset,
        left.lower + right.lower,
        left.upper + right.upper,
    ), (*left_ops, *right_ops, operation)


def pure_tensor_control(operation):
    """Recognize only pure operations and bounded tensor-only SCF loops."""
    from xdsl.traits import Pure

    if operation.has_trait(Pure):
        return True
    # tensor.empty retains its original allocation/publication. It cannot read
    # the floating environment; unlike a general effectful producer, it is a
    # virtual static tensor construction, never removed by this rewrite.
    if operation.name == "tensor.empty" and not operation.operands and len(operation.results) == 1:
        from xdsl.dialects.builtin import TensorType

        return isinstance(operation.results[0].type, TensorType) and all(
            size > 0 for size in operation.results[0].type.get_shape()
        )
    if operation.name != "scf.for":
        return False
    try:
        _loop_bounds(operation)
    except ValueError:
        return False
    block = operation.regions[0].block
    terminator = block.last_op
    return (
        terminator is not None
        and terminator.name == "scf.yield"
        and len(terminator.operands) == 1
        and terminator.operands[0].type == operation.results[0].type
        and all(child is terminator or pure_tensor_control(child) for child in block.ops)
    )


def prove_closed_tensor_insert(anchor):
    """Authenticate a single-use immutable destination and distinct typed lanes.

    Returns operations whose current operands, attributes and uses must be part
    of the caller's witness. Unsupported loops/index expressions decline.
    """
    from xdsl.dialects.builtin import TensorType, i8
    from xdsl.ir import BlockArgument

    if (
        anchor.name != "tensor.insert"
        or len(anchor.results) != 1
        or len(anchor.operands) < 3
        or anchor.operands[0].type != i8
        or not isinstance(anchor.operands[1].type, TensorType)
        or anchor.results[0].type != anchor.operands[1].type
        or anchor.operands[1].type.element_type != i8
    ):
        raise ValueError("typed scalar i8 immutable tensor insertion required")
    block = anchor.parent
    if block is None:
        raise ValueError("tensor observation has no current owning block")
    shape = anchor.results[0].type.get_shape()
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("static positive tensor observation shape required")
    chain, current = [], anchor
    while True:
        chain.append(current)
        destination = current.operands[1]
        previous = destination.owner
        if getattr(previous, "name", None) == "tensor.insert" and previous.parent is block:
            if destination is not previous.results[0]:
                raise ValueError("tensor publication has an ambiguous result")
            current = previous
        else:
            root = destination
            break
    chain.reverse()
    current = anchor
    while True:
        uses = tuple(current.results[0].uses)
        if len(uses) != 1:
            raise ValueError("tensor publication carrier has a live escape")
        use = uses[0]
        next_operation = use.operation
        if next_operation.name == "tensor.insert" and use.index == 1 and next_operation.parent is block:
            chain.append(next_operation)
            current = next_operation
            continue
        if (
            next_operation is not block.last_op
            or next_operation.name not in ("scf.yield", "func.return")
            or len(next_operation.operands) != 1
            or use.index != 0
        ):
            raise ValueError("tensor publication has no unique original return")
        terminator = next_operation
        break
    if len(tuple(root.uses)) != 1:
        raise ValueError("tensor publication root has a live escape")
    witnesses = [*chain, terminator]
    if isinstance(root, BlockArgument):
        loop = root.owner.parent_op()
        if root.index != 1 or loop is None or root.owner is not loop.regions[0].block:
            raise ValueError("tensor publication root is not the retained SCF carrier")
        witnesses.extend(_loop_bounds(loop)[3])
    elif getattr(root.owner, "name", None) == "scf.for":
        if not pure_tensor_control(root.owner):
            raise ValueError("tensor tail carrier has unknown effects")
        witnesses.extend(root.owner.walk())
    elif getattr(root.owner, "name", None) == "tensor.empty":
        witnesses.append(root.owner)
    else:
        raise ValueError("tensor publication root ownership is unsupported")
    coordinates = []
    for ordinal, operation in enumerate(chain):
        if (
            operation.parent is not block
            or len(operation.operands) != len(shape) + 2
            or len(operation.results) != 1
            or operation.results[0].type != anchor.results[0].type
            or operation.operands[0].type != i8
        ):
            raise ValueError("tensor publication chain changes type or rank")
        uses = tuple(operation.results[0].uses)
        expected = chain[ordinal + 1] if ordinal + 1 < len(chain) else terminator
        expected_index = 1 if ordinal + 1 < len(chain) else 0
        if len(uses) != 1 or uses[0].operation is not expected or uses[0].index != expected_index:
            raise ValueError("tensor publication carrier has a live escape")
        if len(tuple(operation.operands[0].uses)) != 1:
            raise ValueError("tensor integer lane has an additional live observation")
        point = []
        for index, size in zip(operation.operands[2:], shape, strict=True):
            coordinate, index_witnesses = _coordinate(index)
            if not 0 <= coordinate.lower <= coordinate.upper < size:
                raise ValueError("tensor publication coordinate is not proved in bounds")
            point.append(coordinate)
            witnesses.extend(index_witnesses)
        coordinates.append(point)
    for left_index, left in enumerate(coordinates):
        for right in coordinates[left_index + 1 :]:
            if not any(a.variable is b.variable and a.offset != b.offset for a, b in zip(left, right, strict=True)):
                raise ValueError("tensor publication lanes are not proved disjoint")
    return tuple(dict.fromkeys(witnesses))
