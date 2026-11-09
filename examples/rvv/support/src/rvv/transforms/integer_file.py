"""Explicit canonical window preparation using the shared contraction normal form.

This changes representation only for a closed signed zero-seeded integer MAC.
The original public output order is restored by the existing normal-form owner.
Provider code still owns streaming, geometry and vector scheduling.
"""

from __future__ import annotations


def normalize(module, *, windows=True, contractions=False):
    """Normalize proved window MACs to a right-gather GEMM; retain all refusals."""
    from merlin.common import mlir_query as mq
    from merlin.llvmlower import contraction_normal_form as cnf
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        IntegerAttr,
        TensorType,
        i8,
        i32,
    )
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir.affine import AffineMap

    report = {
        "normalized": 0,
        "refusals": {},
        "orientation": "window_swapped_other_as_is",
        "public_output_layout_preserved": True,
        "quantization_changed": False,
    }

    def refuse(reason):
        report["refusals"][reason] = report["refusals"].get(reason, 0) + 1

    for op in list(mq.walk(module, "linalg.generic")):
        nf = cnf.analyze(op)
        if (
            isinstance(nf, str)
            or (nf.windowed is None and not contractions)
            or (nf.windowed is not None and not windows)
        ):
            continue
        if (nf.windowed is not None and nf.windowed != 0) or nf.batch:
            refuse("requires_unbatched_left_window")
            continue
        block = op.body.block
        operations = list(block.ops)
        expected = (
            arith.ExtSIOp,
            arith.ExtSIOp,
            arith.MuliOp,
            arith.AddiOp,
            linalg.YieldOp,
        )
        if (
            [a.type for a in block.args] != [i8, i8, i32]
            or len(operations) != 5
            or not all(
                isinstance(actual, cls) for actual, cls in zip(operations, expected)
            )
        ):
            refuse("requires_closed_signed_i8_i32_mac")
            continue
        a, b, mul, add, result = operations
        if (
            list(a.operands) != [block.args[0]]
            or list(b.operands) != [block.args[1]]
            or list(mul.operands) != [a.result, b.result]
            or list(add.operands)
            not in ([mul.result, block.args[2]], [block.args[2], mul.result])
            or list(result.operands) != [add.result]
        ):
            refuse("requires_exact_mac_wiring")
            continue
        if any(
            value.data
            for operation in (mul, add)
            for name, value in operation.properties.items()
            if name == "overflowFlags"
        ):
            refuse("nondefault_integer_overflow_flags")
            continue
        seed = op.outputs[0].owner
        if not isinstance(seed, linalg.FillOp) or len(seed.inputs) != 1:
            refuse("requires_zero_fill")
            continue
        zero = seed.inputs[0].owner
        if not isinstance(zero, arith.ConstantOp) or zero.value.value.data != 0:
            refuse("requires_zero_fill")
            continue

        created = []

        def consumer(left, right, initial, result_type, original=op, holder=created):
            body = original.body.clone()
            cloned_add = list(body.block.ops)[3]
            cloned_mul = list(body.block.ops)[2]
            # Unflagged integer addition is commutative modulo 2**32. Keep the
            # provider's strict canonical MAC spelling without changing types.
            cloned_add.operands = [cloned_mul.result, body.block.args[2]]
            generic = linalg.GenericOp(
                inputs=(left, right),
                outputs=(initial,),
                body=body,
                indexing_maps=ArrayAttr(
                    [
                        AffineMapAttr(AffineMap.from_callable(function))
                        for function in (
                            lambda m, n, k: (m, k),
                            lambda m, n, k: (k, n),
                            lambda m, n, k: (m, n),
                        )
                    ]
                ),
                iterator_types=ArrayAttr(
                    [
                        linalg.IteratorTypeAttr(kind)
                        for kind in (
                            linalg.IteratorType.PARALLEL,
                            linalg.IteratorType.PARALLEL,
                            linalg.IteratorType.REDUCTION,
                        )
                    ]
                ),
                result_types=(result_type,),
            )
            holder.append(generic)
            return generic

        cnf.emit(nf, cnf.SWAPPED if nf.windowed is not None else cnf.AS_IS, consumer)
        # The public provider's canonical source transport is gather6→flat→2D.
        # Reassociation changes only the view description, with static equal
        # element counts; it neither evaluates nor packs a second gather.
        generic = created[0]
        old = generic.inputs[1].owner
        if (
            isinstance(old, tensor.CollapseShapeOp)
            and len(old.src.type.get_shape()) == 6
        ):
            matrix_shape = generic.inputs[1].type.get_shape()

            def groups(values):
                return ArrayAttr(
                    [ArrayAttr([IntegerAttr(v, 64) for v in group]) for group in values]
                )

            flat = tensor.CollapseShapeOp(
                operands=[old.src],
                result_types=[TensorType(i8, [matrix_shape[0] * matrix_shape[1]])],
                properties={"reassociation": groups([list(range(6))])},
            )
            expanded = tensor.ExpandShapeOp(
                flat.result, (), groups([[0, 1]]), matrix_shape, generic.inputs[1].type
            )
            generic.parent.insert_op_before(flat, generic)
            generic.parent.insert_op_before(expanded, generic)
            operands = list(generic.operands)
            operands[1] = expanded.result
            generic.operands = operands
            if not old.result.uses:
                old.detach()
                old.erase()
        report["normalized"] += 1
    module.verify()
    return report


def exact_divisor(extent, bound):
    """Largest power-of-two exact tile within the explicit architectural bound."""
    if type(extent) is not int or extent < 1 or type(bound) is not int or bound < 1:
        raise ValueError("positive integer extent and bound required")
    value = 1
    while value * 2 <= bound and extent % (value * 2) == 0:
        value *= 2
    return value


def register_blocks(module, target):
    """Explicit conservative schedule, with no padded M/N tile values required.

    This is an architectural/legalization policy rather than a measured optimum.
    Prime dimensions can intentionally produce one-element vector tiles.
    """
    from merlin.common import mlir_query as mq
    from xdsl.dialects.builtin import UnitAttr

    from ..packing.im2col_stream import _signed_zero_mac
    from ..policy.panel_geometry import derive
    from .perop_blocks import shape_key, tag_for

    table, entries = {}, []
    for op in mq.walk(module, "linalg.generic"):
        if not _signed_zero_mac(op):
            continue
        maps = [attribute.data for attribute in op.indexing_maps]
        from xdsl.ir.affine import AffineMap

        wanted = [
            AffineMap.from_callable(fn)
            for fn in (
                lambda m, n, k: (m, k),
                lambda m, n, k: (k, n),
                lambda m, n, k: (m, n),
            )
        ]
        if maps != wanted:
            continue
        m, k = op.inputs[0].type.get_shape()
        n = op.inputs[1].type.get_shape()[1]
        geometry = derive(target, k=k, output_width=n)
        mr, nr = exact_divisor(m, geometry.mr), exact_divisor(n, geometry.nr)
        op.attributes[tag_for("linalg.matmul", mr, nr)] = UnitAttr()
        table[shape_key("linalg.matmul", (m, n), (k,))] = (mr, nr)
        entries.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "mr": mr,
                "nr": nr,
                "architectural_geometry": geometry.to_dict(),
                "basis": "exact_power_of_two_divisors_with_explicit_register_budget",
                "performance_optimality_claimed": False,
            }
        )
    module.verify()
    return {"table": table, "entries": entries, "unclaimed_source_retained": True}


def prepare_file(
    source, work, target, *, windows=False, contractions=False, blocks=False
):
    """Public provider preparation, default-off and with a separate output file."""
    from pathlib import Path

    from merlin.common import mlir_query as mq

    if any(type(flag) is not bool for flag in (windows, contractions, blocks)):
        raise ValueError("explicit boolean preparation options required")
    source = Path(source)
    if not any((windows, contractions, blocks)):
        return source, {
            "normalization": {"normalized": 0},
            "register_blocks": {"table": {}, "entries": []},
        }
    module = mq.parse(source.read_text())
    report = {
        "normalization": normalize(module, windows=windows, contractions=contractions)
    }
    report["register_blocks"] = (
        register_blocks(module, target) if blocks else {"table": {}, "entries": []}
    )
    output = Path(work) / "model.integer-prepared.mlir"
    output.write_text(str(module))
    return output, report


def lowering_schedule(table):
    """Use the existing provider schedule owner with native vector finalization."""
    from .perop_blocks import schedule_text

    if not table:
        raise ValueError(
            "no canonical signed contraction was assigned a register block"
        )
    text = schedule_text(table, kc=1)
    anchor = "      transform.apply_patterns.vector.fold_arith_extension\n"
    if text.count(anchor) != 1:
        raise ValueError("register schedule pattern boundary changed")
    return text.replace(
        anchor,
        anchor
        + '      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"\n'
        "      transform.apply_patterns.vector.lower_masked_transfers\n"
        "      transform.apply_patterns.vector.lower_transpose\n"
        "      transform.apply_patterns.vector.lower_shape_cast\n",
    )
