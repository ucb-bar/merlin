"""Ordered native squared-sum matching and materialization.

Native source executes in the caller-selected MLIR context; no tool discovery.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.compiler.lower.quant.constant_float_clamp import _finite_f32
from merlin.compiler.lower.quant.constant_fma_packet import ConstantFmaPacket
from merlin.compiler.lower.source.dense import SourceFmaBatchContract

# ---- host_reduction ----------------------------------------------------------------------------

SQUARED_SUM_SOURCE = r"""
def _sqs_spec(op):
    from torch_mlir import ir as _sqs_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 2
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    ancestor = op.operation
    while ancestor is not None:
        if any("strictfp" in name or "strictfp" in str(ancestor.attributes[name])
               for name in ancestor.attributes):
            return None
        ancestor = ancestor.parent
    types = []
    for value in op.operands:
        if not isinstance(value.type, _sqs_ir.RankedTensorType):
            return None
        tensor = _sqs_ir.RankedTensorType(value.type)
        if (not isinstance(tensor.element_type, _sqs_ir.F32Type)
                or any(extent < 0 for extent in tensor.shape)):
            return None
        types.append(tensor)
    if op.results[0].type != op.operands[1].type:
        return None
    allowed_attributes = {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    if any(name not in allowed_attributes and not name.startswith("prov.")
           for name in op.attributes):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 2 or len(body) != 3
            or [x.operation.name for x in body]
                != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any(x.regions for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for scalar in body for name in scalar.attributes)
            or any("fastmath" in x.attributes
                   and str(x.attributes["fastmath"]) != "#arith.fastmath<none>"
                   for x in body)):
        return None
    mul, add, terminator = body
    value, seed = block.arguments
    if list(mul.operands) != [value, value]:
        return None
    if list(add.operands) == [mul.results[0], seed]:
        order = [0, 1]
    elif list(add.operands) == [seed, mul.results[0]]:
        order = [1, 0]
    else:
        return None
    if list(terminator.operands) != [add.results[0]]:
        return None
    maps = [_sqs_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 2:
        return None
    dims = maps[0].n_dims
    if (dims < 1 or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
                != ["#linalg.iterator_type<parallel>"] * (dims - 1)
                   + ["#linalg.iterator_type<reduction>"]):
        return None
    extents, positions = [None] * dims, []
    for affine, tensor in zip(maps, types):
        if len(affine.results) != len(tensor.shape):
            return None
        used = []
        for expr, extent in zip(affine.results, tensor.shape):
            if not isinstance(expr, _sqs_ir.AffineDimExpr):
                return None
            position = _sqs_ir.AffineDimExpr(expr).position
            if (position in used or position >= dims
                    or (extents[position] is not None and extents[position] != extent)):
                return None
            used.append(position)
            extents[position] = extent
        positions.append(used)
    if (sorted(positions[0]) != list(range(dims))
            or sorted(positions[1]) != list(range(dims - 1))
            or any(extent is None for extent in extents)):
        return None
    return positions, extents, order


def _scalarize_squared_sums(ctx, module):
    from torch_mlir import ir as _sqs_ir
    todo = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _sqs_spec(child)
                    if spec is not None:
                        todo.append((child, spec))
                    else:
                        walk(child.operation)
    walk(module.operation)
    with ctx:
        for old, (positions, extents, add_order) in todo:
            with old.location, _sqs_ir.InsertionPoint(old):
                index = _sqs_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sqs_ir.Operation.create(name, operands=list(operands),
                        results=list(results), attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sqs_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(extent) for extent in extents]
                source_mul, source_add = list(old.regions[0].blocks[0].operations)[:2]
                def output_loop(depth, current, coordinates):
                    if depth < len(extents) - 1:
                        loop = create("scf.for", [zero, limits[depth], one, current],
                                      [tensor], regions=1)
                        block = _sqs_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sqs_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1],
                                                  [*coordinates, block.arguments[0]])
                            create("scf.yield", [updated])
                        return loop.results[0]
                    out_indices = [coordinates[d] for d in positions[1]]
                    initial = create("tensor.extract", [current, *out_indices], [scalar]).results[0]
                    loop = create("scf.for", [zero, limits[-1], one, initial],
                                  [scalar], regions=1)
                    block = _sqs_ir.Block.create_at_start(loop.regions[0], [index, scalar])
                    with _sqs_ir.InsertionPoint(block):
                        all_indices = [*coordinates, block.arguments[0]]
                        value = create("tensor.extract", [old.operands[0],
                            *[all_indices[d] for d in positions[0]]], [scalar]).results[0]
                        product = create("arith.mulf", [value, value], [scalar],
                            {name: source_mul.attributes[name] for name in source_mul.attributes}).results[0]
                        values = [product, block.arguments[1]]
                        total = create("arith.addf", [values[d] for d in add_order], [scalar],
                            {name: source_add.attributes[name] for name in source_add.attributes}).results[0]
                        create("scf.yield", [total])
                    return create("tensor.insert", [loop.results[0], current, *out_indices],
                                  [tensor]).results[0]
                result = output_loop(0, old.operands[1], [])
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                prefix = _sqs_ir.StringAttr(previous).value + "," if previous is not None else ""
                result.owner.attributes["prov.transforms"] = _sqs_ir.StringAttr.get(
                    prefix + "scalar_squared_sum_accumulator")
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)

"""


# ---- host_arithmetic ---------------------------------------------------------------------------
# Explicit shared host CPU implementations of independent source FMA chains.
#
# The selected host ISA/ABI and source arithmetic/effect permissions are required.
# These emitters issue no accelerator instruction and select no graph transform,
# approximation, workload or schedule. Their presence proves neither the actual
# host capability nor the original source permissions; callers establish those
# independently and bind emitted bytes into ordinary compilation evidence.


def _identifier(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name[0].isdigit()
        or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in name)
    ):
        raise ValueError("explicit C identifier required")
    return name


def _profile(isa: str, abi: str) -> None:
    if isa != "rv64gc" or abi != "lp64d":
        raise ValueError("explicit RV64GC/lp64d host arithmetic capability required")


@dataclass(frozen=True)
class SourceFmaPairCapability:
    isa: str
    abi: str
    ieee_f32: bool
    gradual_underflow: bool
    stable_rounding: bool
    nontrapping_unobserved_flags: bool

    def header(self, *, namespace: str = "merlin_host_source") -> str:
        _profile(self.isa, self.abi)
        _identifier(namespace)
        if any(
            value is not True
            for value in (
                self.ieee_f32,
                self.gradual_underflow,
                self.stable_rounding,
                self.nontrapping_unobserved_flags,
            )
        ):
            raise ValueError("explicit source arithmetic and effect contract required")
        return r"""#ifndef __GUARD__
#define __GUARD__
static inline void __NAMESPACE___fma_pair(
 float al,float bl,float cl,float ah,float bh,float ch,float *low,float *high){
 float l,h;
 __asm__("fmadd.s %0,%2,%3,%4\n\tfmadd.s %1,%5,%6,%7"
         : "=&f"(l),"=&f"(h)
         : "f"(al),"f"(bl),"f"(cl),"f"(ah),"f"(bh),"f"(ch));
 *low=l;*high=h;
}
#define MERLIN_SOURCE_F32_FMA_PAIR __NAMESPACE___fma_pair
#endif
""".replace("__GUARD__", namespace.upper() + "_FMA_PAIR_H").replace("__NAMESPACE__", namespace)


@dataclass(frozen=True)
class SourceFmaBatchCapability:
    isa: str
    abi: str
    contract: SourceFmaBatchContract

    def header(self, *, namespace: str = "merlin_host_source") -> str:
        _profile(self.isa, self.abi)
        _identifier(namespace)
        if not isinstance(self.contract, SourceFmaBatchContract):
            raise ValueError("typed source FMA batch contract required")
        self.contract.validate()
        # Each output is read/write and early-clobber, so a live input from
        # another lane cannot share its register. Operand order is unchanged.
        instructions = "\\n\\t".join(f"fmadd.s %{lane},%{lane + 8},%{lane},%16" for lane in range(8))
        outputs = ",".join(f'"+&f"(product[{lane}])' for lane in range(8))
        inputs = ",".join(f'"f"(fraction[{lane}])' for lane in range(8))
        guard = namespace.upper() + "_FMA_BATCH_H"
        return (
            f"#ifndef {guard}\n"
            f"#define {guard}\n"
            f"static inline void {namespace}_fma_eight(\n"
            " const float fraction[8],float product[8],float coefficient) {\n"
            f' __asm__("{instructions}"\n'
            f"         : {outputs}\n"
            f'         : {inputs},"f"(coefficient));\n'
            "}\n"
            f"#define MERLIN_SOURCE_F32_FMA_EIGHT {namespace}_fma_eight\n"
            "#endif\n"
        )


# ---- host_rounding -----------------------------------------------------------------------------
# Exact scalar CPU roundeven support, independent of accelerator selection.
#
# The implementation preserves signed zero and existing flags, quiets signaling
# NaNs through ordinary arithmetic, and uses a fixed RNE conversion. Selecting
# this body requires the caller's actual RV64GC/lp64d host capability.


def roundeven_f32_assembly(*, isa: str, abi: str, symbol: str = "merlin_host_roundevenf") -> str:
    """Emit one exact binary32 helper for an explicitly selected host profile."""
    _profile(isa, abi)
    _identifier(symbol)
    return _ASSEMBLY.replace("__SYMBOL__", symbol)


_ASSEMBLY = r"""/* Exact binary32 round-to-nearest-even for the explicitly selected RV64GC host.
 * Binary32 magnitudes >= 2^23 are already integral. Smaller finite values
 * convert exactly back from i32 after RNE conversion; restore the zero sign.
 * Preserve existing fflags (roundeven must not raise inexact). sNaNs are
 * quieted through fadd, which retains normal invalid-exception behavior.
 */
.text
.align 2
.globl __SYMBOL__
.type __SYMBOL__,@function
__SYMBOL__:
    fmv.x.w t0, fa0
    srli t1, t0, 23
    andi t1, t1, 255
    li t2, 150
    bgeu t1, t2, .Lround_large
    frflags t2
    fcvt.w.s t1, fa0, rne
    fcvt.s.w fa0, t1, rne
    fsflags t2
    fmv.x.w t1, fa0
    srli t0, t0, 31
    andi t0, t0, 1
    slli t0, t0, 31
    or t1, t1, t0
    fmv.w.x fa0, t1
    ret
.Lround_large:
    li t2, 255
    bne t1, t2, .Lround_return
    fadd.s fa0, fa0, fa0
.Lround_return:
    ret
.size __SYMBOL__, .-__SYMBOL__
.section .note.GNU-stack,"",@progbits
"""


# ---- host_fma_packet ---------------------------------------------------------------------------
# Explicit host CPU emission of source-proven scalar FMA packets.
#
# The caller selects the CPU ISA/ABI and supplies the source/effect contract.
# No accelerator facts or graph selection are encoded here.


@dataclass(frozen=True)
class RV64GCScalarFmaCapability:
    isa: str = "rv64gc"
    abi: str = "lp64d"
    ieee_f32_fma: bool = True
    ieee_gradual_underflow: bool = True

    def emit(self, packet: ConstantFmaPacket, temporary: str) -> str:
        """Emit exact source input order with protected destructive outputs.

        Early-clobber prevents a later input from sharing an already written
        output register, including cross-lane shared SSA. Tied inputs reduce live
        registers; LLVM preserves any other uses by ordinary register allocation.
        FMA uses the original default environment; no rounding/flag policy is
        changed by this emitter. The caller separately proves source legality.
        """
        if (
            (self.isa, self.abi) != ("rv64gc", "lp64d")
            or self.ieee_f32_fma is not True
            or self.ieee_gradual_underflow is not True
        ):
            raise ValueError("explicit RV64GC IEEE scalar FMA capability required")
        if type(packet) is not ConstantFmaPacket:
            raise ValueError("typed source FMA packet required")
        count = len(packet.calls)
        if count not in (2, 4):
            raise ValueError("supported scalar FMA resource packets have 2 or 4 lanes")
        lhs = [call.operands[0] for call in packet.calls]
        rhs = [call.operands[1] for call in packet.calls]
        addend = [call.operands[2] for call in packet.calls]
        expected = {"constant_rhs": (1,), "constant_addend": (2,), "constant_lhs_addend": (0, 2)}.get(packet.mode)
        if expected is None or len(packet.constant_words) != len(expected):
            raise ValueError("unsupported source-proven constant FMA mode")
        for axis, word in zip(expected, packet.constant_words, strict=True):
            for call in packet.calls:
                decoded = _finite_f32(call.operands[axis])
                if decoded is None or decoded[0] != word:
                    raise ValueError("source FMA constant witness changed")
        if any(
            not call.operands[axis].startswith("%")
            for call in packet.calls
            for axis in range(3)
            if axis not in expected
        ):
            raise ValueError("source FMA variable witness changed")
        dtype = "{ " + ", ".join(["float"] * count) + " }"
        outputs = ["=&f"] * count
        if packet.mode == "constant_rhs":
            arguments = [
                *(f"float {value}" for value in lhs),
                f"i32 {packet.constant_words[0]}",
                *(f"float {value}" for value in addend),
            ]
            instructions = f"fmv.w.x ft0,${2 * count}" + "".join(
                f"\\0Afmadd.s ${lane},${lane + count},ft0,${lane}" for lane in range(count)
            )
            constraints = [*outputs, *["f"] * count, "r", *(str(lane) for lane in range(count)), "~{ft0}"]
        elif packet.mode == "constant_addend":
            arguments = [
                *(f"float {value}" for value in rhs),
                f"i32 {packet.constant_words[0]}",
                *(f"float {value}" for value in lhs),
            ]
            instructions = f"fmv.w.x ft0,${2 * count}" + "".join(
                f"\\0Afmadd.s ${lane},${lane},${lane + count},ft0" for lane in range(count)
            )
            constraints = [*outputs, *["f"] * count, "r", *(str(lane) for lane in range(count)), "~{ft0}"]
        else:
            arguments = [*(f"i32 {word}" for word in packet.constant_words), *(f"float {value}" for value in rhs)]
            instructions = f"fmv.w.x ft0,${count}\\0Afmv.w.x ft1,${count + 1}" + "".join(
                f"\\0Afmadd.s ${lane},ft0,${lane},ft1" for lane in range(count)
            )
            constraints = [*outputs, "r", "r", *(str(lane) for lane in range(count)), "~{ft0}", "~{ft1}"]
        lines = [
            f'{temporary} = call {dtype} asm "{instructions}", "'
            + ",".join(constraints)
            + '"('
            + ", ".join(arguments)
            + ")"
        ]
        lines.extend(
            f"  {call.result} = extractvalue {dtype} {temporary}, {lane}" for lane, call in enumerate(packet.calls)
        )
        return "\n".join(lines)
