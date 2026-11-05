"""Hoist immutable tensor broadcast divisors before physical alias decisions.

Only closed rank-two all-parallel binary32 x[i,j]/y[i,0] is accepted. The
ordinary divide, scalar order, flags and tensor value semantics are preserved;
no pointer noalias assumption is introduced. The optional unit-divisor guard
uses an integer-word finite scan before copying exact x/+1 results; every
nonunit or nonfinite row keeps its literal original divide.
"""

from __future__ import annotations

import io

FEATURE = "hoist_broadcast_divisors"
UNIT_FEATURE = "guard_unit_broadcast_divisors"
SUM_UNIT_FEATURE = "prove_finite_unit_broadcast_divisors"
POW2_FEATURE = "scale_power_of_two_broadcast_divisors"


def _strict_context(operation):
    while operation is not None:
        if any("strictfp" in key or "strictfp" in str(value)
               for key, value in (*operation.attributes.items(), *operation.properties.items())):
            return True
        operation = operation.parent_op()
    return False


def _match(operation):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import Float32Type, NoneAttr, TensorType
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

    if (operation.name != "linalg.generic" or _strict_context(operation)
            or len(operation.operands) != 3 or len(operation.results) != 1
            or len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1):
        return None
    allowed = {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    if any(key not in allowed and not key.startswith("prov.")
           for key in (*operation.attributes, *operation.properties)):
        return None
    types = tuple(value.type for value in (*operation.operands, *operation.results))
    if any(not isinstance(typ, TensorType) or not isinstance(typ.element_type, Float32Type)
           or not isinstance(typ.encoding, NoneAttr) or any(d <= 0 for d in typ.get_shape())
           for typ in types):
        return None
    shape, denominator_shape = types[0].get_shape(), types[1].get_shape()
    rank = len(shape)
    if rank == 0 or types[0] != types[2] or types[0] != types[3] or len(denominator_shape) != rank:
        return None
    maps = operation.indexing_maps.data
    if (len(maps) != 3 or maps[0].data != AffineMap.identity(rank)
            or maps[2].data != AffineMap.identity(rank)
            or len(operation.iterator_types) != rank
            or any(iterator.data != IteratorType.PARALLEL for iterator in operation.iterator_types)):
        return None
    denominator_map = maps[1].data
    if denominator_map.num_dims != rank or denominator_map.num_symbols or len(denominator_map.results) != rank:
        return None
    broadcast = False
    for axis, (dimension, extent) in enumerate(zip(denominator_map.results, denominator_shape, strict=True)):
        if dimension == AffineDimExpr(axis) and extent == shape[axis]:
            continue
        if isinstance(dimension, AffineConstantExpr) and dimension.value == 0 and extent == 1:
            broadcast |= shape[axis] > 1
            continue
        return None
    if not broadcast:
        return None
    block = operation.regions[0].block
    body = tuple(block.ops)
    if (len(block.args) != 3 or block.args[2].uses or len(body) != 2
            or not isinstance(body[0], arith.DivfOp) or body[1].name != "linalg.yield"
            or tuple(body[0].operands) != tuple(block.args[:2])
            or tuple(body[1].operands) != tuple(body[0].results)
            or body[0].regions or _strict_context(body[0])):
        return None
    for key, value in (*body[0].attributes.items(), *body[0].properties.items()):
        if key == "fastmath" and str(value) == "#arith.fastmath<none>":
            continue
        if not key.startswith("prov."):
            return None
    if rank != 2 or denominator_map.results != (AffineDimExpr(0), AffineConstantExpr(0)):
        return None
    return body[0]



def rewrite_broadcast_divisors(
    module, *, unit_guard: bool = False, sum_unit_proof: bool = False, power_two_guard: bool = False,
) -> dict:
    from xdsl.dialects import arith, scf, tensor
    from xdsl.dialects.builtin import IndexType, IntegerType, StringAttr
    from xdsl.ir import Block, Region

    from .finite_sum_witness import complete_same_row_sum

    count = certified = 0
    for operation in tuple(module.walk()):
        divide = _match(operation)
        if divide is None:
            continue
        numerator, denominator, initial = operation.operands
        sum_witness = unit_guard and sum_unit_proof and complete_same_row_sum(numerator, denominator)
        certified += int(sum_witness)
        rows, columns = numerator.type.get_shape()
        constants = [arith.ConstantOp.from_int_and_width(value, IndexType())
                     for value in (0, 1, rows, columns)]
        zero, one, row_bound, column_bound = constants
        row = Block(arg_types=[IndexType(), initial.type])
        i, current = row.args
        divisor = tensor.ExtractOp(denominator, [i, zero.result], denominator.type.element_type)
        row.add_op(divisor)

        def columns(copy=False, multiplier=None):
            column = Block(arg_types=[IndexType(), initial.type])
            j, partial = column.args
            scalar = tensor.ExtractOp(numerator, [i, j], numerator.type.element_type)
            column.add_op(scalar)
            value = scalar.result
            if not copy:
                if multiplier is None:
                    cloned = divide.clone(value_mapper={divide.operands[0]: scalar.result,
                                                       divide.operands[1]: divisor.result})
                else:
                    cloned = arith.MulfOp(scalar.result, multiplier)
                    cloned.attributes.update(divide.attributes)
                    cloned.properties.update(divide.properties)
                    cloned.location = divide.location
                column.add_op(cloned)
                value = cloned.results[0]
            inserted = tensor.InsertOp(value, partial, [i, j])
            column.add_ops([inserted, scf.YieldOp(inserted.result)])
            return scf.ForOp(zero, column_bound, one, [current], column)
        if unit_guard:
            i1, i32 = IntegerType(1), IntegerType(32)
            yes = arith.ConstantOp.from_int_and_width(1, i1)
            no = arith.ConstantOp.from_int_and_width(0, i1)
            word_one = arith.ConstantOp.from_int_and_width(0x3f800000, i32)
            exponent_mask = arith.ConstantOp.from_int_and_width(0x7f800000, i32)
            divisor_word = arith.BitcastOp(divisor.result, i32)
            is_one = arith.CmpiOp(divisor_word, word_one, "eq")
            if sum_witness:
                row.add_ops([word_one, divisor_word, is_one])
                if power_two_guard:
                    mantissa_mask = arith.ConstantOp.from_int_and_width(0x007fffff, i32)
                    zero_word = arith.ConstantOp.from_int_and_width(0, i32)
                    smallest = arith.ConstantOp.from_int_and_width(0x00800000, i32)
                    largest = arith.ConstantOp.from_int_and_width(0x7e800000, i32)
                    # A normal power of two has word E<<23. Its exact reciprocal
                    # has biased exponent 254-E; both selected values are normal.
                    reciprocal_bias = arith.ConstantOp.from_int_and_width(0x7f000000, i32)
                    mantissa = arith.AndIOp(divisor_word, mantissa_mask)
                    no_fraction = arith.CmpiOp(mantissa, zero_word, "eq")
                    above = arith.CmpiOp(divisor_word, smallest, "uge")
                    below = arith.CmpiOp(divisor_word, largest, "ule")
                    bounded = arith.AndIOp(above, below)
                    admitted_power = arith.AndIOp(no_fraction, bounded)
                    reciprocal_word = arith.SubiOp(reciprocal_bias, divisor_word)
                    reciprocal = arith.BitcastOp(reciprocal_word, numerator.type.element_type)
                    row.add_ops([mantissa_mask, zero_word, smallest, largest, reciprocal_bias,
                                 mantissa, no_fraction, above, below, bounded, admitted_power,
                                 reciprocal_word, reciprocal])
                admitted_value = is_one.result
            else:
                scan_body = Block(arg_types=[IndexType(), i1])
                j, admitted = scan_body.args
                scalar = tensor.ExtractOp(numerator, [i, j], numerator.type.element_type)
                word = arith.BitcastOp(scalar.result, i32)
                exponent = arith.AndIOp(word, exponent_mask)
                finite = arith.CmpiOp(exponent, exponent_mask, "ne")
                continued = arith.AndIOp(admitted, finite)
                scan_body.add_ops([scalar, word, exponent, finite, continued, scf.YieldOp(continued)])
                scan = scf.ForOp(zero, column_bound, one, [yes], scan_body)
                unit_case = Block([scan, scf.YieldOp(scan.res[0])])
                other_case = Block([scf.YieldOp(no)])
                admitted = scf.IfOp(is_one, [i1], Region(unit_case), Region(other_case))
                row.add_ops([yes, no, word_one, exponent_mask, divisor_word, is_one, admitted])
                admitted_value = admitted.results[0]
            copied, divided = columns(copy=True), columns()
            valid_case = Block([copied, scf.YieldOp(copied.res[0])])
            if sum_witness and power_two_guard:
                multiplied = columns(multiplier=reciprocal.result)
                scaled_case = Block([multiplied, scf.YieldOp(multiplied.res[0])])
                divided_case = Block([divided, scf.YieldOp(divided.res[0])])
                alternative = scf.IfOp(admitted_power, [initial.type], Region(scaled_case), Region(divided_case))
                fallback_case = Block([alternative, scf.YieldOp(alternative.results[0])])
            else:
                fallback_case = Block([divided, scf.YieldOp(divided.res[0])])
            choice = scf.IfOp(admitted_value, [initial.type], Region(valid_case), Region(fallback_case))
            row.add_ops([choice, scf.YieldOp(choice.results[0])])
        else:
            inner = columns()
            row.add_ops([inner, scf.YieldOp(inner.res[0])])
        outer = scf.ForOp(zero, row_bound, one, [initial], row)
        outer.location = operation.location
        outer.attributes.update({key: value for key, value in operation.attributes.items()
                                 if key.startswith("prov.")})
        previous = operation.attributes.get("prov.transforms")
        outer.attributes["prov.transforms"] = StringAttr(
            (previous.data + "," if isinstance(previous, StringAttr) else "")
            + (POW2_FEATURE if power_two_guard else SUM_UNIT_FEATURE if sum_unit_proof
               else UNIT_FEATURE if unit_guard else FEATURE))
        parent = operation.parent_block()
        parent.insert_ops_before([*constants, outer], operation)
        operation.results[0].replace_all_uses_with(outer.res[0])
        parent.erase_op(operation)
        count += 1
    module.verify()
    return ({"row_invariant_divisors_hoisted": count, **({"unit_divisor_rows_guarded": count} if unit_guard else {}),
             **({"same_tensor_row_sum_witnesses": certified} if sum_unit_proof else {}),
             **({"power_two_row_sum_witnesses": certified} if power_two_guard else {})}
            if count else {})


def rewrite_text(
    source: str, *, unit_guard: bool = False, sum_unit_proof: bool = False, power_two_guard: bool = False,
) -> tuple[str, dict]:
    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(source)
    module.verify()
    report = rewrite_broadcast_divisors(module, unit_guard=unit_guard, sum_unit_proof=sum_unit_proof,
                                        power_two_guard=power_two_guard)
    if not report:
        return source, report
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue(), report


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE,
            action_class="PASS",
            description=(
                "Hoist immutable row-broadcast binary32 divisors before bufferization; "
                "preserve each original divide and tensor alias semantics, default off."
            ),
        ))
    if UNIT_FEATURE not in known():
        register(ImprFeature(
            name=UNIT_FEATURE,
            action_class="PASS",
            description=(
                "Unswitch exact positive-one row divisors with a finite-word scan and "
                "original divide fallback; preserve fenv and tensor semantics, default off."
            ),
        ))
    if SUM_UNIT_FEATURE not in known():
        register(ImprFeature(name=SUM_UNIT_FEATURE, action_class="PASS", description=(
            "Prove finite rows from their same-tensor ordinary sum when divisor is +1; "
            "retain runtime finite scan for unknown producers and original divide fallback, default off.")))
    if POW2_FEATURE not in known():
        register(ImprFeature(name=POW2_FEATURE, action_class="PASS", description=(
            "Unswitch exact positive normal power-of-two row divisors under same-sum finite witness; "
            "exact reciprocal words, unchanged rounded multiplication identity and divide fallback, default off.")))
    return FEATURE
