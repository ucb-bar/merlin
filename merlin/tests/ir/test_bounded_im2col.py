"""Bounded host packing preserves complete integer convolutions, not device gather support."""

from dataclasses import replace
from itertools import product

import pytest
from test_im2col_panel_pack import _module
from test_selected_dense_contracts import inputs
from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.ops import GenericOp

from merlin.common import mlir_query as mq
from merlin.llvmlower import im2col_pack as ip


def build(tmp_path, *, bits=8, width=3, **shape):
    geometry = dict(c=2, kh=2, kw=3, n=1, oh=3, ow=7, f=3, sh=2, sw=1, dh=1, dw=2)
    geometry.update(shape)
    k = geometry["c"] * geometry["kh"] * geometry["kw"]
    m = geometry["n"] * geometry["oh"] * geometry["ow"]
    selected, *_ = inputs(tmp_path, widths=(bits // 8, bits // 8, 4), shape=(geometry["f"], k, m))
    source = _module(**geometry).replace("i8", f"i{bits}")
    module = mq.parse(source)
    operation = [op for op in module.walk() if type(op) is GenericOp][-1]
    match = ip.match_bounded_im2col(operation, selected_contracts=selected)
    assert match is not None
    identity = ip.bounded_source_identity(match)
    rewritten = ip.rewrite_bounded_match(match, width)
    module.verify()
    report = ip.verify_bounded_match(rewritten, expected_source=identity)
    return module, rewritten, report, selected, source, geometry


def wrap32(value):
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def actual_outputs(rewrite, image, weight):
    """Evaluate actual index SSA, slices and gather maps, independently of verifier formulas."""
    n, oh, ow, c, kh, kw, *_ = rewrite.source.geometry
    result = [[None] * (n * oh * ow) for _ in weight]
    for arm in rewrite.arms:
        loop = arm.loop
        iv = loop.body.block.args[0]

        def evaluate(value, iteration):
            if value is iv:
                return iteration
            op = value.owner
            if type(op) is arith.ConstantOp:
                return op.value.value.data
            left, right = (evaluate(v, iteration) for v in op.operands)
            return {
                arith.AddiOp: lambda: left + right,
                arith.MuliOp: lambda: left * right,
                arith.DivUIOp: lambda: left // right,
                arith.RemUIOp: lambda: left % right,
            }[type(op)]()

        def offsets(operation, iteration):
            dynamic = iter(operation.offsets)
            return [
                evaluate(next(dynamic), iteration) if x == tensor.ExpandShapeOp.DYNAMIC_INDEX else x
                for x in operation.static_offsets.get_values()
            ]

        for iteration in range(loop.ub.owner.value.value.data):
            origin = offsets(arm.patch, iteration)
            target = offsets(arm.insert, iteration)[1]
            columns = [[] for _ in range(arm.width)]
            mapping = tuple(arm.gather.indexing_maps)[0].data
            for ci, hi, wi in product(range(c), range(kh), range(kw)):
                for col in range(arm.width):
                    coordinate = mapping.eval((ci, hi, wi, 0, 0, col), ())
                    coordinate = tuple(a + b for a, b in zip(origin, coordinate, strict=True))
                    columns[col].append(image[coordinate])
            for f, row in enumerate(weight):
                for col, values in enumerate(columns):
                    acc = 0
                    for a, b in zip(row, values, strict=True):
                        acc = wrap32(acc + a * b)
                    assert result[f][target + col] is None, "overlapping realized output slices"
                    result[f][target + col] = acc
    assert all(v is not None for row in result for v in row), "uncovered realized output"
    return result


@pytest.mark.parametrize(
    "bits,width,shape",
    [
        (8, 1, {}),
        (8, 3, {}),
        (8, 7, {}),
        (16, 4, dict(n=2, c=3, kh=3, kw=2, oh=2, ow=9, f=5, sh=1, sw=2, dh=2, dw=1)),
    ],
)
def test_actual_windows_complete_outputs_tails_and_modular_accumulation(tmp_path, bits, width, shape):
    _, rewrite, report, _, _, g = build(tmp_path, bits=bits, width=width, **shape)
    n, c, kh, kw, oh, ow = (g[k] for k in ("n", "c", "kh", "kw", "oh", "ow"))
    ih = (oh - 1) * g["sh"] + (kh - 1) * g["dh"] + 1
    iw = (ow - 1) * g["sw"] + (kw - 1) * g["dw"] + 1
    maximum = (1 << (bits - 1)) - 1
    image = {p: maximum - sum(p) % 7 for p in product(range(n), range(c), range(ih), range(iw))}
    weight = [[maximum - (f + k) % 5 for k in range(c * kh * kw)] for f in range(g["f"])]
    expected = [[] for _ in weight]
    for batch, h, w in product(range(n), range(oh), range(ow)):
        for f, row in enumerate(weight):
            acc = 0
            for k, (ci, hi, wi) in enumerate(product(range(c), range(kh), range(kw))):
                acc = wrap32(acc + row[k] * image[batch, ci, h * g["sh"] + hi * g["dh"], w * g["sw"] + wi * g["dw"]])
            expected[f].append(acc)
    assert actual_outputs(rewrite, image, weight) == expected
    assert report["resident_packing_bytes"] == c * kh * kw * (width + ow % width) * bits // 8
    assert report["requested_gather_read_bytes"] == c * kh * kw * n * oh * ow * bits // 8
    assert report["requested_gather_write_bytes"] == report["requested_gather_read_bytes"]
    assert all(tuple(arm.loop.iter_args)[1] is arm.packing_empty.results[0] for arm in rewrite.arms)
    assert all(arm.packing_empty.parent_block() is arm.loop.parent_block() for arm in rewrite.arms)


@pytest.mark.parametrize(
    "mutation",
    [
        "window",
        "coverage",
        "zero",
        "scalar",
        "yield",
        "destination",
        "index_effect",
        "membership",
        "snapshot",
    ],
)
def test_realized_program_mutations_refuse(tmp_path, mutation):
    _, rewrite, _, _, _, _ = build(tmp_path)
    arm = rewrite.arms[0]
    if mutation == "window":
        arm.patch.properties["static_strides"] = builtin.DenseArrayBase.from_list(builtin.i64, [1, 1, 2, 1])
    elif mutation == "coverage":
        rewrite = replace(rewrite, arms=(replace(arm, first_ow=1), *rewrite.arms[1:]))
    elif mutation == "zero":
        arm.output_fill.inputs[0].owner.properties["value"] = builtin.IntegerAttr(1, builtin.i32)
    elif mutation == "scalar":
        op = next(op for op in arm.contraction.body.block.ops if type(op) is arith.AddiOp)
        op.operands = (op.operands[0], op.operands[0])
    elif mutation == "yield":
        arm.loop.body.block.last_op.operands = (arm.insert.results[0], arm.loop.body.block.args[2])
    elif mutation == "destination":
        arm.gather.operands = (arm.patch.results[0], arm.packing_empty.results[0])
    elif mutation == "index_effect":
        arm.loop.body.block.first_op.attributes["unknown"] = builtin.UnitAttr()
    elif mutation == "membership":
        rewrite = replace(rewrite, operations=rewrite.operations[:-1])
    else:
        rewrite = replace(rewrite, source=replace(rewrite.source, geometry=(2, *rewrite.source.geometry[1:])))
    with pytest.raises(ValueError, match="bounded im2col realization"):
        ip.verify_bounded_match(rewrite, expected_source=rewrite.source_identity)


@pytest.mark.parametrize("mutation", ["effect", "bounds", "gather_attr", "reshape", "unsigned", "shared"])
def test_unsupported_sources_retain_full_materialization(tmp_path, mutation):
    _, _, _, selected, source, _ = build(tmp_path)
    if mutation == "effect":
        source = source.replace("    %ae =", "    func.call @opaque() : () -> ()\n    %ae =")
        source = source.replace("module {", "module {\n  func.func private @opaque()")
    elif mutation == "bounds":
        # Original input width is 11. The typed static source gather then reads beyond it.
        source = source.replace("x11xi8", "x10xi8")
    elif mutation == "unsigned":
        source = source.replace("arith.extsi", "arith.extui")
    elif mutation == "shared":
        source = source.replace(
            "    %flat =",
            "    %extra = tensor.collapse_shape %g [[0, 1, 2, 3, 4, 5]] "
            ": tensor<2x2x3x1x3x7xi8> into tensor<252xi8>\n    %flat =",
        )
    module = mq.parse(source)
    operations = [op for op in module.walk() if type(op) is GenericOp]
    if mutation == "gather_attr":
        operations[0].attributes["unknown"] = builtin.UnitAttr()
    elif mutation == "reshape":
        expand = operations[-1].inputs[1].owner
        expand.attributes["unknown"] = builtin.UnitAttr()
    assert ip.match_bounded_im2col(operations[-1], selected_contracts=selected) is None


@pytest.mark.parametrize("mutation", ["seed", "gather", "reshape"])
def test_analysis_staleness_and_bad_widths_refuse_before_rewrite(tmp_path, mutation):
    _, _, _, selected, source, _ = build(tmp_path)
    module = mq.parse(source)
    operation = [op for op in module.walk() if type(op) is GenericOp][-1]
    match = ip.match_bounded_im2col(operation, selected_contracts=selected)
    assert match is not None
    for width in (0, 8, True):
        with pytest.raises(ValueError, match="width"):
            ip.rewrite_bounded_match(match, width)
    if mutation == "seed":
        operation.outputs[0].owner.inputs[0].owner.properties["value"] = builtin.IntegerAttr(2, builtin.i32)
    elif mutation == "gather":
        match.gather.attributes["unknown"] = builtin.UnitAttr()
    else:
        match.expand.attributes["unknown"] = builtin.UnitAttr()
    with pytest.raises(ValueError, match="changed after admission"):
        ip.bounded_source_identity(match)


@pytest.mark.parametrize("binding", ["input", "weight"])
def test_same_typed_external_binding_substitution_refuses_original_context(tmp_path, binding):
    _, _, _, selected, source, _ = build(tmp_path)
    source = source.replace(", %w:", ", %other: tensor<1x2x6x11xi8>, %otherw: tensor<3x12xi8>, %w:")
    module = mq.parse(source)
    operation = [op for op in module.walk() if type(op) is GenericOp][-1]
    match = ip.match_bounded_im2col(operation, selected_contracts=selected)
    assert match is not None
    original = (match.gather.inputs[0], match.contraction.inputs[0], match.contraction.outputs[0])
    rewritten = ip.rewrite_bounded_match(match, 3)
    arguments = rewritten.source.input.owner.args
    if binding == "input":
        replacement = arguments[1]
        for arm in rewritten.arms:
            arm.patch.operands = (replacement, *arm.patch.operands[1:])
    else:
        replacement = arguments[2]
        for arm in rewritten.arms:
            arm.contraction.operands = (replacement, *arm.contraction.operands[1:])
    mutated = replace(rewritten, source=replace(rewritten.source, **{binding: replacement}))
    module.verify()
    assert mutated.source_identity == mutated.source.identity  # A canonical standalone function erases SSA binding.
    with pytest.raises(ValueError, match="external SSA bindings"):
        ip.verify_bounded_match(mutated, expected_source=rewritten.source_identity, expected_bindings=original)
