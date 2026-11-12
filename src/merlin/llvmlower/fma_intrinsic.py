"""Opt-in exact fused multiply-add lowering before the established libm stage.

Only f32/f64 math.fma becomes llvm.intr.fma after loop formation. Exponentials and other
math operations retain their existing lowering. LLVM may select a hardware FMA
or its faithful library fallback; this does not permit separate multiply/add.
"""

from __future__ import annotations

FEATURE = "lower_fma_to_intrinsic"
MARKER = "__merlin_lower_fma_to_intrinsic__"


def _edit_pipeline(passes: list[str]) -> list[str]:
    matches = [
        i for i, p in enumerate(passes) if "convert-linalg-to-loops" in p or "convert-linalg-to-parallel-loops" in p
    ]
    if not matches:
        raise ValueError(f"{FEATURE} requires a linalg-to-loops lowering stage")
    i = max(matches) + 1
    return [*passes[:i], MARKER, *passes[i:]]


def ensure_registered() -> str:
    from merlin.compiler.selection.shared import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description="Lower exact math.fma to llvm.intr.fma after loops; preserve other math lowering.",
                edit_pipeline=_edit_pipeline,
            )
        )
    return FEATURE


RUNNER_PRELUDE = r"""
def _lower_fma_intrinsics(ctx, module):
    from torch_mlir import ir as _fi_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    if inner.operation.name == "math.fma":
                        element = inner.results[0].type
                        if isinstance(element, _fi_ir.VectorType):
                            element = element.element_type
                        if isinstance(element, (_fi_ir.F32Type, _fi_ir.F64Type)):
                            todo.append(inner)
                    walk(inner.operation)

    walk(module.operation)
    with ctx:
        for old in todo:
            with _fi_ir.InsertionPoint(old):
                new = _fi_ir.Operation.create(
                    "llvm.intr.fma", results=[old.results[0].type],
                    operands=list(old.operands), loc=old.location)
                # Retain nonsemantic attribution. Dropping optional fastmath
                # permissions selects the stricter fused operation.
                for name in old.attributes:
                    if name != "fastmath":
                        new.attributes[name] = old.attributes[name]
            old.results[0].replace_all_uses_with(new.results[0])
            old.operation.erase()
    module.operation.verify()
    return len(todo)


_FI_MARKER = "__merlin_lower_fma_to_intrinsic__"
_FI_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    if _FI_MARKER not in passes:
        return _FI_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late,
                                   post_openmp, pre_generalize)
    i = passes.index(_FI_MARKER)
    _FI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), erase, mid, (), (),
                        pre_generalize)
    print('OK fma_intrinsic', _lower_fma_intrinsics(ctx, module))
    _FI_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), 0, (), late,
                        post_openmp, ())
"""
