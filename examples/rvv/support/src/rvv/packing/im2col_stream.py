"""Default-off, bounded producer/consumer fusion for static window contractions.

This API emits ordinary tensor/linalg/SCF IR; it contains no chip instructions.
Callers must opt in explicitly. ``build_stream`` is shared with extension
providers: its consumer callback owns arithmetic and must preserve each output
element's reduction order. The built-in adapter admits only the closed signed
i8 x i8 -> i32 zero-initialized MAC grammar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..policy.panel_geometry import PanelTarget, derive
from . import im2col_pack as old


@dataclass(frozen=True)
class WindowSpec:
    n: int
    oh: int
    ow: int
    channels: int
    kh: int
    kw: int
    sh: int = 1
    sw: int = 1
    dh: int = 1
    dw: int = 1
    input_layout: str = "NCHW"
    reduction_order: tuple[str, ...] = ("c", "kh", "kw")

    def __post_init__(self):
        if any(v <= 0 for k, v in asdict(self).items() if isinstance(v, int)):
            raise ValueError("positive static window dimensions required")
        if self.input_layout not in ("NCHW", "NHWC"):
            raise ValueError("only NCHW/NHWC source windows supported")
        if sorted(self.reduction_order) != ["c", "kh", "kw"]:
            raise ValueError("reduction order must permute c/kh/kw")

    @property
    def k(self):
        return self.channels * self.kh * self.kw

    @property
    def m(self):
        return self.n * self.oh * self.ow


@dataclass
class StreamReport:
    streamed: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)
    refusals: dict[str, int] = field(default_factory=dict)

    def refuse(self, reason):
        self.refusals[reason] = self.refusals.get(reason, 0) + 1

    def to_dict(self):
        return asdict(self)


def _slice_props(offsets, sizes):
    from xdsl.dialects.builtin import DenseArrayBase, i64

    return {
        "static_offsets": DenseArrayBase.from_list(i64, offsets),
        "static_sizes": DenseArrayBase.from_list(i64, sizes),
        "static_strides": DenseArrayBase.from_list(i64, [1] * len(sizes)),
    }


def build_stream(
    input_tensor,
    initial_acc,
    spec: WindowSpec,
    panel_width: int,
    consumer: Callable,
    *,
    panel_layout: str = "km",
    output_axis: int = 1,
):
    """Return ``(ops, result)`` for full panels and a static row tail.

    ``consumer(panel, accumulator_tile, width) -> (ops, result)`` receives a
    contiguous [K,T] (``km``) or [T,K] (``mk``) tensor and an existing output
    slice. The original accumulator is carried, never refilled or copied in
    full. Workspaces are outside the loops, carried and completely overwritten
    by each pure gather. This API is sequential; sharing its workspace across
    parallel loop iterations is not permitted.
    """
    from xdsl.dialects.arith import AddiOp, ConstantOp, DivUIOp, MuliOp, RemUIOp
    from xdsl.dialects.builtin import AffineMapAttr, IndexType, TensorType
    from xdsl.dialects.linalg.ops import (
        GenericOp,
        IteratorType,
        IteratorTypeAttr,
        YieldOp,
    )
    from xdsl.dialects.scf import ForOp
    from xdsl.dialects.scf import YieldOp as SYield
    from xdsl.dialects.tensor import (
        CollapseShapeOp,
        EmptyOp,
        ExpandShapeOp,
        ExtractSliceOp,
        InsertSliceOp,
    )
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    if panel_width < 1 or panel_layout not in ("km", "mk") or output_axis not in (0, 1):
        raise ValueError("invalid panel width/layout/output axis")
    ishape = old._static_shape(input_tensor)
    cshape = old._static_shape(initial_acc)
    if ishape is None or len(ishape) != 4 or cshape is None or len(cshape) != 2:
        raise ValueError("static rank-four input/rank-two accumulator required")
    expected_nc = (ishape[0], ishape[1 if spec.input_layout == "NCHW" else 3])
    h, w = (
        (ishape[2], ishape[3])
        if spec.input_layout == "NCHW"
        else (ishape[1], ishape[2])
    )
    if expected_nc != (spec.n, spec.channels) or cshape[output_axis] != spec.m:
        raise ValueError("input/output shape disagrees with window")
    if (spec.oh - 1) * spec.sh + (spec.kh - 1) * spec.dh >= h or (
        spec.ow - 1
    ) * spec.sw + (spec.kw - 1) * spec.dw >= w:
        raise ValueError("window exceeds input; padding must be explicit upstream")
    d = AffineExpr.dimension
    idx = IndexType()
    elem = input_tensor.type.get_element_type()
    acc_t = initial_acc.type
    dyn = ExpandShapeOp.DYNAMIC_INDEX
    par = IteratorTypeAttr(IteratorType.PARALLEL)
    result = initial_acc
    all_ops = []
    full, tail = divmod(spec.ow, panel_width)
    for count, width, first in (
        (full, panel_width, 0),
        (bool(tail), tail, full * panel_width),
    ):
        if not count:
            continue
        count = int(count)
        axes = list(spec.reduction_order)
        axes = ["t", *axes] if panel_layout == "mk" else [*axes, "t"]
        sizes = {"c": spec.channels, "kh": spec.kh, "kw": spec.kw, "t": width}
        shape = [sizes[a] for a in axes]
        workspace_t = TensorType(elem, shape)
        empty = EmptyOp([], workspace_t)
        lb = ConstantOp.from_int_and_width(0, idx)
        ub = ConstantOp.from_int_and_width(spec.n * spec.oh * count, idx)
        step = ConstantOp.from_int_and_width(1, idx)
        body = Block(arg_types=[idx, acc_t, workspace_t])
        iv, carried, workspace = body.args
        ops = []

        def constant(value, ops=ops):
            op = ConstantOp.from_int_and_width(value, idx)
            ops.append(op)
            return op.results[0]

        def binary(cls, a, b, ops=ops):
            op = cls(a, b)
            ops.append(op)
            return op.results[0]

        row = binary(DivUIOp, iv, constant(count))
        col = binary(RemUIOp, iv, constant(count))
        batch = binary(DivUIOp, row, constant(spec.oh))
        oh = binary(RemUIOp, row, constant(spec.oh))
        ow = binary(AddiOp, binary(MuliOp, col, constant(width)), constant(first))
        off = binary(AddiOp, binary(MuliOp, row, constant(spec.ow)), ow)
        ih = binary(MuliOp, oh, constant(spec.sh))
        iw = binary(MuliOp, ow, constant(spec.sw))
        hs = (spec.kh - 1) * spec.dh + 1
        ws = (width - 1) * spec.sw + (spec.kw - 1) * spec.dw + 1
        if spec.input_layout == "NCHW":
            offsets, window_sizes = [dyn, 0, dyn, dyn], [1, spec.channels, hs, ws]
            view_shape = [spec.channels, hs, ws]
        else:
            offsets, window_sizes = [dyn, dyn, dyn, 0], [1, hs, ws, spec.channels]
            view_shape = [hs, ws, spec.channels]
        view = ExtractSliceOp.build(
            operands=[input_tensor, [batch, ih, iw], [], []],
            result_types=[TensorType(elem, view_shape)],
            properties=_slice_props(offsets, window_sizes),
        )
        pos = {axis: d(i) for i, axis in enumerate(axes)}
        at = (pos["c"], pos["kh"] * spec.dh, pos["t"] * spec.sw + pos["kw"] * spec.dw)
        if spec.input_layout == "NHWC":
            at = (at[1], at[2], at[0])
        gb = Block(arg_types=[elem, elem])
        gb.add_op(YieldOp(gb.args[0]))
        gather = GenericOp(
            inputs=[view.results[0]],
            outputs=[workspace],
            body=Region(gb),
            indexing_maps=[
                AffineMapAttr(AffineMap(4, 0, at)),
                AffineMapAttr(AffineMap(4, 0, tuple(d(i) for i in range(4)))),
            ],
            iterator_types=[par] * 4,
            result_types=[workspace_t],
        )
        panel_shape = [width, spec.k] if panel_layout == "mk" else [spec.k, width]
        groups = [[0], [1, 2, 3]] if panel_layout == "mk" else [[0, 1, 2], [3]]
        panel = CollapseShapeOp(
            operands=[gather.results[0]],
            result_types=[TensorType(elem, panel_shape)],
            properties={"reassociation": old._reassoc(groups)},
        )
        tile_shape = list(cshape)
        tile_shape[output_axis] = width
        tile_offsets = [0, 0]
        tile_offsets[output_axis] = dyn
        tile_props = _slice_props(tile_offsets, tile_shape)
        tile = ExtractSliceOp.build(
            operands=[carried, [off], [], []],
            result_types=[TensorType(initial_acc.type.get_element_type(), tile_shape)],
            properties=tile_props,
        )
        consume_ops, value = consumer(panel.results[0], tile.results[0], width)
        if value.type != tile.results[0].type:
            raise ValueError("consumer must return exactly the accumulator tile type")
        put = InsertSliceOp.build(
            operands=[value, carried, [off], [], []],
            result_types=[acc_t],
            properties=tile_props,
        )
        body.add_ops(
            [
                *ops,
                view,
                gather,
                panel,
                tile,
                *consume_ops,
                put,
                SYield(put.results[0], gather.results[0]),
            ]
        )
        loop = ForOp(
            lb.results[0],
            ub.results[0],
            step.results[0],
            [result, empty.results[0]],
            Region(body),
        )
        all_ops.extend([empty, lb, ub, step, loop])
        result = loop.results[0]
    return all_ops, result


def _signed_zero_mac(op):
    """Closed five-op integer grammar; no calls, side effects or reassociation."""
    from xdsl.dialects.arith import AddiOp, ConstantOp, ExtSIOp, MuliOp
    from xdsl.dialects.builtin import i8, i32
    from xdsl.dialects.linalg.ops import FillOp, YieldOp

    block = op.body.block
    if [i.data.value for i in op.get_iterator_types()] != [
        "parallel",
        "parallel",
        "reduction",
    ]:
        return False
    if [a.type for a in block.args] != [i8, i8, i32]:
        return False
    ops = list(block.ops)
    if len(ops) != 5 or not all(
        isinstance(o, cls)
        for o, cls in zip(ops, (ExtSIOp, ExtSIOp, MuliOp, AddiOp, YieldOp))
    ):
        return False
    x, y, mul, add, yield_ = ops
    if any(o.properties["overflowFlags"].data for o in (mul, add)):
        return False
    if list(x.operands) != [block.args[0]] or list(y.operands) != [block.args[1]]:
        return False
    if x.results[0].type != i32 or y.results[0].type != i32:
        return False
    if (
        list(mul.operands) != [x.results[0], y.results[0]]
        or list(add.operands) != [mul.results[0], block.args[2]]
        or list(yield_.operands) != [add.results[0]]
    ):
        return False
    fill = op.outputs[0].owner
    if not isinstance(fill, FillOp) or len(fill.inputs) != 1:
        return False
    zero = fill.inputs[0].owner
    return isinstance(zero, ConstantOp) and zero.value.value.data == 0


def rewrite_module(
    module, target: PanelTarget, *, enabled: bool = False
) -> StreamReport:
    """Explicit opt-in rewrite of the existing canonical NCHW right-im2col form."""
    from merlin.common import mlir_query as mq
    from xdsl.dialects.builtin import AffineMapAttr, UnitAttr
    from xdsl.dialects.linalg.ops import GenericOp, IteratorType, IteratorTypeAttr
    from xdsl.rewriter import InsertPoint, Rewriter

    report = StreamReport()
    if not enabled:
        return report
    if (target.operand_bits, target.accumulator_bits) != (8, 32):
        report.refuse("unsupported_numeric_contract")
        return report
    for op in list(mq.walk(module, "linalg.generic")):
        mt = old._match_im2col(op, report)
        if mt is None:
            continue
        if not _signed_zero_mac(op):
            report.refuse("requires_pure_signed_i8_i32_zero_mac")
            continue
        try:
            geometry = derive(target, k=mt.k, output_width=mt.ow)
            spec = WindowSpec(
                *(
                    getattr(mt, name)
                    for name in (
                        "n",
                        "oh",
                        "ow",
                        "channels",
                        "kh",
                        "kw",
                        "sh",
                        "sw",
                        "dh",
                        "dw",
                    )
                )
            )

            def consume(panel, tile, width, original=op, geometry=geometry):
                par = IteratorTypeAttr(IteratorType.PARALLEL)
                red = IteratorTypeAttr(IteratorType.REDUCTION)
                inner = GenericOp(
                    inputs=[original.inputs[0], panel],
                    outputs=[tile],
                    body=original.body.clone(),
                    indexing_maps=[AffineMapAttr(mp) for mp in old._matmul_maps()],
                    iterator_types=[par, par, red],
                    result_types=[tile.type],
                )
                inner.attributes.update(original.attributes)
                from ..transforms.perop_blocks import tag_for

                inner.attributes[
                    tag_for("linalg.matmul", geometry.mr, min(width, geometry.nr))
                ] = UnitAttr()
                return [inner], inner.results[0]

            ops, result = build_stream(
                mt.gather.inputs[0], op.outputs[0], spec, geometry.panel_width, consume
            )
        except ValueError as exc:
            report.refuse(str(exc))
            continue
        Rewriter.insert_op(ops, InsertPoint.before(op))
        op.results[0].replace_all_uses_with(result)
        for obsolete in (op, mt.expand, mt.collapse, mt.gather):
            Rewriter.erase_op(obsolete)
        # This was the old full-im2col destination. It is dead after replacing
        # the pure single-use gather and must not remain as an apparent full
        # activation allocation in the rewritten IR.
        from xdsl.dialects.tensor import EmptyOp

        old_empty = mt.gather.outputs[0].owner
        if isinstance(old_empty, EmptyOp) and not list(old_empty.results[0].uses):
            Rewriter.erase_op(old_empty)
        report.streamed += 1
        report.entries.append(
            {
                "window": asdict(spec),
                "geometry": geometry.to_dict(),
                "old_activation_bytes": mt.k * mt.m,
                "numeric_contract": "signed_i8_i8_i32_modular_zero_accumulator",
                "parallel_panels": False,
            }
        )
    return report


def rewrite_prepared_file(
    prepared: str | Path,
    target: PanelTarget,
    work: str | Path,
    *,
    enabled: bool = False,
):
    """Return original bytes/path on default-off or refusal; write private IR on success."""
    from merlin.common import mlir_query as mq

    prepared = Path(prepared)
    if not enabled:
        return prepared, StreamReport()
    module = mq.parse(prepared.read_text())
    report = rewrite_module(module, target, enabled=True)
    if not report.streamed:
        return prepared, report
    module.verify()
    output = Path(work) / "model.im2col-stream.mlir"
    output.write_text(str(module))
    return output, report


def lowering_schedule(report: StreamReport) -> str:
    """Ordinary RVV register-block schedule for this exact rewrite report.

    The caller selects this schedule explicitly along with the returned IR.
    Existing schedules and feature registries are unchanged. Inner contractions
    carry the corresponding structural block tags; no chip/shape-name matching.
    """
    from ..transforms.perop_blocks import schedule_text, shape_key

    table = {}
    for entry in report.entries:
        window, geometry = entry["window"], entry["geometry"]
        k = window["channels"] * window["kh"] * window["kw"]
        for width in (geometry["panel_width"], geometry["tail_width"]):
            if width:
                table[shape_key("linalg.matmul", (1, width), (k,))] = (
                    geometry["mr"],
                    min(width, geometry["nr"]),
                )
    if not table:
        raise ValueError("no rewritten contraction to schedule")
    text = schedule_text(table, kc=1)
    anchor = "      transform.apply_patterns.vector.fold_arith_extension\n"
    if text.count(anchor) != 1:
        raise ValueError("register schedule pattern anchor changed")
    return text.replace(
        anchor,
        anchor
        + '      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"\n      transform.apply_patterns.vector.lower_masked_transfers\n      transform.apply_patterns.vector.lower_transpose\n      transform.apply_patterns.vector.lower_shape_cast\n',
    )


def lowering_pipeline(schedule_path: str | Path) -> str:
    """Explicit stream-only pipeline preserving the consumer's block tags.

    Generic-to-named specialization drops these newly constructed generic
    tags. The stream schedule already matches those exact generics, so this
    opt-in path omits that stage, then uses the existing RVV backend pipeline.
    The default backend pipeline is not edited.
    """
    from ..backend import build_pipeline

    pipeline = build_pipeline(schedule_path)
    stage = "func.func(linalg-specialize-generic-ops),"
    if pipeline.count(stage) != 1:
        raise ValueError("RVV specialization pipeline anchor changed")
    return pipeline.replace(stage, "", 1)
