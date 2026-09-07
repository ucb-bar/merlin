"""Replace the degenerate N=1, 1x1, stride-1 im2col copy with a tensor view.

For this exact geometry the capture's column tensor
``[C, 1, 1, 1, H, W]`` has the same row-major element order as its NCHW input
``[1, C, H, W]``.  Collapsing the input directly to ``[C, H*W]`` therefore supplies the ordinary
matmul without writing and reading an im2col buffer.  Anything that does not prove that identity
from shapes and affine maps is refused.  The panel packer may still optimize every non-degenerate
im2col contraction after this pass; these view contractions retain the ordinary per-op matmul arm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FEATURE = "im2col_identity_view"


@dataclass
class ViewReport:
    viewed: int = 0
    refusals: dict[str, int] = field(default_factory=dict)

    def refuse(self, reason: str) -> None:
        self.refusals[reason] = self.refusals.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {"viewed": self.viewed, "refusals": dict(sorted(self.refusals.items()))}


def _rewrite_one(mt) -> None:
    from xdsl.dialects.builtin import StringAttr, TensorType, UnitAttr
    from xdsl.dialects.tensor import CollapseShapeOp, EmptyOp
    from xdsl.rewriter import InsertPoint, Rewriter

    from .im2col_pack import _reassoc

    source = mt.gather.inputs[0]
    elem = source.type.get_element_type()
    view_t = TensorType(elem, [mt.channels, mt.m])
    view = CollapseShapeOp(
        operands=[source], result_types=[view_t],
        properties={"reassociation": _reassoc([[0, 1], [2, 3]])})
    for key, value in mt.gather.attributes.items():
        view.attributes[key] = value
    view.attributes["prov.role"] = StringAttr("im2col_view")
    view.attributes["merlin.im2col_identity_view"] = UnitAttr()

    Rewriter.insert_op(view, InsertPoint.before(mt.gather))
    mt.expand.results[0].replace_all_uses_with(view.results[0])
    old_empty = mt.gather.outputs[0].owner
    for dead in (mt.expand, mt.collapse, mt.gather):
        Rewriter.erase_op(dead)
    if isinstance(old_empty, EmptyOp) and not list(old_empty.results[0].uses):
        Rewriter.erase_op(old_empty)


def rewrite_module(module) -> ViewReport:
    """Mutate every mechanically identical im2col chain and report all near misses."""
    from ..common import mlir_query as mq
    from .im2col_pack import PackReport, _match_im2col, _static_shape

    report = ViewReport()
    candidates = []
    match_report = PackReport()
    for op in mq.walk(module, "linalg.generic"):
        mt = _match_im2col(op, match_report)
        if mt is not None:
            candidates.append(mt)
    for mt in candidates:
        inshape = _static_shape(mt.gather.inputs[0])
        if (mt.n, mt.kh, mt.kw, mt.sh, mt.sw, mt.dh, mt.dw) != (1, 1, 1, 1, 1, 1, 1):
            report.refuse("refused_nonidentity_geometry")
            continue
        if inshape != [1, mt.channels, mt.oh, mt.ow]:
            report.refuse("refused_layout_or_extent")
            continue
        # `_match_im2col` already proves the whole gather->collapse->expand chain is sole-use and
        # that its final type is [C, H*W]. Reassert the product at this semantic boundary.
        if mt.m != mt.oh * mt.ow or mt.k != mt.channels:
            report.refuse("refused_layout_or_extent")
            continue
        _rewrite_one(mt)
        report.viewed += 1
    return report


def rewrite_prepared_file(prepared: "str | Path", work: "str | Path | None" = None):
    """Write ``model.im2col_views.mlir`` only when at least one copy became a view."""
    from ..common import mlir_query as mq

    prepared = Path(prepared)
    module = mq.parse(prepared.read_text(encoding="utf-8"))
    report = rewrite_module(module)
    if not report.viewed:
        return prepared, report
    out = (Path(work) / "model.im2col_views.mlir" if work is not None
           else prepared.with_name("model.im2col_views.mlir"))
    out.write_text(str(module), encoding="utf-8")
    return out, report


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE,
            action_class="PASS",
            description=(
                "Replace a sole-use N=1, 1x1, stride/dilation-1 im2col copy with the proven "
                "row-major [1,C,H,W] -> [C,H*W] collapse view. All other geometries and layouts "
                "are refused. Non-identity convolutions remain eligible for im2col_panel_pack; "
                "viewed contractions retain ordinary per-op matmul scheduling. Default off."),
        ))
    return FEATURE
