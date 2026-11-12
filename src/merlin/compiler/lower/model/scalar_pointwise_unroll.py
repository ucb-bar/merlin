"""Default-off partial unrolling for scalar pointwise FMA/division loops.

This is a scheduling hint, not a floating-point permission. LLVM retains the
original memory dependencies and per-element arithmetic, including existing FMA
operations. It may emit a serial remainder for odd trip counts.
"""

from __future__ import annotations

FEATURE = "unroll_scalar_pointwise_fma_division_by_2"
MARKER = "__merlin_scalar_pointwise_fma_division_unroll_2__"


def _edit_pipeline(passes: list[str]) -> list[str]:
    hits = [i for i, p in enumerate(passes) if p == "convert-scf-to-cf"]
    if len(hits) != 1 or MARKER in passes:
        raise ValueError(f"{FEATURE} requires exactly one structured loop conversion")
    i = hits[0]
    return [*passes[:i], MARKER, *passes[i:]]


def ensure_registered() -> str:
    from merlin.compiler.selection.shared import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description="Request factor-two partial unrolling of static innermost scalar f32 pointwise loops with at least four existing FMAs and a division; preserve arithmetic and alias dependencies.",
                edit_pipeline=_edit_pipeline,
            )
        )
    return FEATURE


RUNNER_PRELUDE = r"""
def _pointwise_unroll_candidate(op):
    from torch_mlir import ir as _pu_ir
    if (op.operation.name != "scf.for" or len(op.operands) != 3
            or len(op.results) or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1
            or "loop_annotation" in op.attributes
            or "llvm.loop_annotation" in op.attributes):
        return False
    bounds = []
    for operand in op.operands:
        owner = operand.owner
        owner = owner.operation if isinstance(owner, _pu_ir.OpView) else owner
        if not isinstance(owner, _pu_ir.Operation) or owner.name != "arith.constant":
            return False
        try:
            bounds.append(_pu_ir.IntegerAttr(owner.attributes["value"]).value)
        except (TypeError, ValueError):
            return False
    lower, upper, step = bounds
    if step != 1 or upper - lower < 2:
        return False
    block = op.regions[0].blocks[0]
    if len(block.arguments) != 1:
        return False
    fmas = 0
    division = False
    load = store = False
    for inner in block.operations:
        name = inner.operation.name
        if inner.regions or any(isinstance(v.type, _pu_ir.VectorType)
                                for v in [*inner.operands, *inner.results]):
            return False
        if "fastmath" in inner.attributes and str(inner.attributes["fastmath"]) not in (
                "#arith.fastmath<none>", "#llvm.fastmath<none>"):
            return False
        if name in ("math.fma", "llvm.intr.fma"):
            if len(inner.results) != 1 or not isinstance(inner.results[0].type, _pu_ir.F32Type):
                return False
            fmas += 1
        elif name == "arith.divf":
            if not isinstance(inner.results[0].type, _pu_ir.F32Type):
                return False
            division = True
        elif name == "memref.load":
            load = True
        elif name == "memref.store":
            store = True
        elif not (name.startswith("arith.") or name == "affine.apply"
                  or (name == "scf.yield" and not inner.operands)):
            return False
    # Existing FMA chains are selected by scalar operation cost, not provenance.
    return fmas >= 4 and division and load and store


def _annotate_pointwise_unroll(ctx, module):
    from torch_mlir import ir as _pu_ir
    todo = []

    def walk(op):
        if any("strictfp" in str(op.attributes[name]) for name in op.attributes):
            return
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if _pointwise_unroll_candidate(inner):
                        todo.append(inner)
                    else:
                        walk(inner.operation)

    walk(module.operation)
    with ctx:
        annotation = _pu_ir.Attribute.parse("#llvm.loop_annotation<unroll = <count = 2 : i32>>")
        for op in todo:
            op.attributes["loop_annotation"] = annotation
    module.operation.verify()
    return len(todo)


_PU_MARKER = "__merlin_scalar_pointwise_fma_division_unroll_2__"
_PU_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _PU_MARKER not in passes:
        return _PU_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if passes.count(_PU_MARKER) != 1:
        raise ValueError("scalar pointwise unroll stage occurs more than once")
    i = passes.index(_PU_MARKER)
    _PU_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), erase, mid, (), (), pre_generalize)
    print('OK scalar_pointwise_fma_division_unroll', _annotate_pointwise_unroll(ctx, module))
    _PU_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), 0, (), late, post_openmp, ())
"""
