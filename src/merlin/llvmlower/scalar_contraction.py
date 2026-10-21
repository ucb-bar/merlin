"""Exact scalar accumulators for static tensor contractions, before bufferization.

The final reduction dimension runs in increasing order with the original separate
multiply and add. Its scalar result is an scf.for iter_arg; the destination tensor
is updated once per output element. Tensor value semantics leave input/destination
alias decisions to upstream bufferization rather than asserting physical noalias.
"""

from __future__ import annotations

from .host_tensor_rewrites import CONTRACTION_SOURCE

FEATURE = "scalar_contraction_accumulator"
MARKER = "__merlin_scalar_contraction_accumulator__"
TWO_OUTPUTS_FEATURE = "scalar_contraction_accumulator_2_outputs"
TWO_OUTPUTS_MARKER = "__merlin_scalar_contraction_accumulator_2_outputs__"
FOUR_OUTPUTS_FEATURE = "scalar_contraction_accumulator_4_outputs"
FOUR_OUTPUTS_MARKER = "__merlin_scalar_contraction_accumulator_4_outputs__"
EIGHT_OUTPUTS_FEATURE = "scalar_contraction_accumulator_8_outputs"
EIGHT_OUTPUTS_MARKER = "__merlin_scalar_contraction_accumulator_8_outputs__"
RECTANGULAR_FEATURE = "scalar_contraction_accumulator_2x4_outputs"
RECTANGULAR_MARKER = "__merlin_scalar_contraction_accumulator_2x4_outputs__"
REDUCTION_UNROLL_FEATURE = "unroll_scalar_contraction_reduction_by_2"
REDUCTION_UNROLL_MARKER = "__merlin_scalar_contraction_reduction_unroll_2__"
FOUR_REDUCTION_UNROLL_FEATURE = "unroll_scalar_contraction_reduction_by_4"
FOUR_REDUCTION_UNROLL_MARKER = "__merlin_scalar_contraction_reduction_unroll_4__"
_OUTPUTS_GROUP = "scalar_contraction_accumulator_outputs"
_UNROLL_GROUP = "scalar_contraction_reduction_unroll"
_OUTPUTS_FEATURES = frozenset(
    {FEATURE, TWO_OUTPUTS_FEATURE, FOUR_OUTPUTS_FEATURE, EIGHT_OUTPUTS_FEATURE, RECTANGULAR_FEATURE}
)


def _edit_pipeline(passes: list[str], *, outputs: int = 1, rows: int = 1) -> list[str]:
    if any(
        m in passes for m in (MARKER, TWO_OUTPUTS_MARKER, FOUR_OUTPUTS_MARKER, EIGHT_OUTPUTS_MARKER, RECTANGULAR_MARKER)
    ):
        raise ValueError("scalar contraction accumulator schedules are alternatives")
    hits = [i for i, p in enumerate(passes) if "one-shot-bufferize" in p]
    if len(hits) != 1:
        raise ValueError(f"{FEATURE} requires exactly one one-shot-bufferize stage")
    # Preserve the ordinary tensor fusion/generalization boundary. Generalizing
    # named copies before fusion can change materialization of prepared read-only
    # operands even when no contraction is selected. Scalarize only the exact
    # pure contractions remaining immediately before bufferization.
    i = hits[0]
    marker = (
        RECTANGULAR_MARKER
        if rows == 2
        else {1: MARKER, 2: TWO_OUTPUTS_MARKER, 4: FOUR_OUTPUTS_MARKER, 8: EIGHT_OUTPUTS_MARKER}[outputs]
    )
    return [*passes[:i], marker, *passes[i:]]


def _edit_reduction_unroll(passes: list[str], *, count: int = 2) -> list[str]:
    if any(m in passes for m in (REDUCTION_UNROLL_MARKER, FOUR_REDUCTION_UNROLL_MARKER)):
        raise ValueError("scalar contraction reduction unroll counts are alternatives")
    hits = [
        i
        for i, p in enumerate(passes)
        if p in (MARKER, TWO_OUTPUTS_MARKER, FOUR_OUTPUTS_MARKER, EIGHT_OUTPUTS_MARKER, RECTANGULAR_MARKER)
    ]
    if len(hits) != 1:
        raise ValueError(f"{REDUCTION_UNROLL_FEATURE} requires one scalar accumulator schedule")
    i = hits[0]
    marker = {2: REDUCTION_UNROLL_MARKER, 4: FOUR_REDUCTION_UNROLL_MARKER}[count]
    return [*passes[:i], marker, *passes[i:]]


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description=(
                    "Keep exact ordered f32 tensor contraction accumulators in scalar loop arguments before "
                    "bufferization."
                ),
                edit_pipeline=_edit_pipeline,
                alternative_group=_OUTPUTS_GROUP,
            )
        )
    if FOUR_OUTPUTS_FEATURE not in known():
        register(
            ImprFeature(
                name=FOUR_OUTPUTS_FEATURE,
                action_class="PASS",
                description=(
                    "Carry four independent exact output accumulators across increasing-K tensor contractions "
                    "whose last parallel extent is divisible by four."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, outputs=4),
                alternative_group=_OUTPUTS_GROUP,
            )
        )
    if EIGHT_OUTPUTS_FEATURE not in known():
        register(
            ImprFeature(
                name=EIGHT_OUTPUTS_FEATURE,
                action_class="PASS",
                description=(
                    "Carry eight independent exact output accumulators across increasing-K tensor "
                    "contractions whose last parallel extent is divisible by eight."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, outputs=8),
                alternative_group=_OUTPUTS_GROUP,
            )
        )
    if TWO_OUTPUTS_FEATURE not in known():
        register(
            ImprFeature(
                name=TWO_OUTPUTS_FEATURE,
                action_class="PASS",
                description=(
                    "Carry two independent exact output accumulators across increasing-K tensor contractions "
                    "whose last parallel extent is divisible by two."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, outputs=2),
                alternative_group=_OUTPUTS_GROUP,
            )
        )
    if RECTANGULAR_FEATURE not in known():
        register(
            ImprFeature(
                name=RECTANGULAR_FEATURE,
                action_class="PASS",
                description=(
                    "Carry eight exact ordered accumulators across a two-row/four-column tensor contraction "
                    "tile, sharing only typed coordinate-identical immutable input reads. Partial tiles "
                    "refuse."
                ),
                edit_pipeline=lambda passes: _edit_pipeline(passes, outputs=4, rows=2),
                alternative_group=_OUTPUTS_GROUP,
            )
        )
    if REDUCTION_UNROLL_FEATURE not in known():
        register(
            ImprFeature(
                name=REDUCTION_UNROLL_FEATURE,
                action_class="PASS",
                description=(
                    "Request partial unrolling by two for exact scalar contraction reduction loops using "
                    "upstream LLVM loop annotations."
                ),
                edit_pipeline=_edit_reduction_unroll,
                alternative_group=_UNROLL_GROUP,
                requires_exactly_one_of=_OUTPUTS_FEATURES,
            )
        )
    if FOUR_REDUCTION_UNROLL_FEATURE not in known():
        register(
            ImprFeature(
                name=FOUR_REDUCTION_UNROLL_FEATURE,
                action_class="PASS",
                description=(
                    "Request partial unrolling by four for exact scalar contraction reduction loops using "
                    "upstream LLVM loop annotations."
                ),
                edit_pipeline=lambda passes: _edit_reduction_unroll(passes, count=4),
                alternative_group=_UNROLL_GROUP,
                requires_exactly_one_of=_OUTPUTS_FEATURES,
            )
        )
    return FEATURE


RUNNER_PRELUDE = (
    CONTRACTION_SOURCE
    + r"""_SC_MARKER = "__merlin_scalar_contraction_accumulator__"
_SC2_MARKER = "__merlin_scalar_contraction_accumulator_2_outputs__"
_SC4_MARKER = "__merlin_scalar_contraction_accumulator_4_outputs__"
_SC8_MARKER = "__merlin_scalar_contraction_accumulator_8_outputs__"
_SC2X4_MARKER = "__merlin_scalar_contraction_accumulator_2x4_outputs__"
_SC_MARKERS = (_SC_MARKER, _SC2_MARKER, _SC4_MARKER, _SC8_MARKER, _SC2X4_MARKER)
_SC_UNROLL2_MARKER = "__merlin_scalar_contraction_reduction_unroll_2__"
_SC_UNROLL4_MARKER = "__merlin_scalar_contraction_reduction_unroll_4__"
_SC_UNROLL_MARKERS = {_SC_UNROLL2_MARKER: 2, _SC_UNROLL4_MARKER: 4}
_SC_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    hints = [m for m in _SC_UNROLL_MARKERS if m in passes]
    if len(hints) > 1:
        raise ValueError("scalar contraction reduction unroll counts are alternatives")
    unroll = _SC_UNROLL_MARKERS[hints[0]] if hints else 0
    passes = [p for p in passes if p not in _SC_UNROLL_MARKERS]
    selected = [m for m in _SC_MARKERS if m in passes]
    if not selected:
        return _SC_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if len(selected) != 1:
        raise ValueError("scalar contraction accumulator schedules are alternatives")
    i = passes.index(selected[0])
    head_generalizes = any('linalg-generalize-named-ops' in p for p in passes[:i])
    _SC_ORIG_RUN_STAGES(ctx, module, ','.join(passes[:i]), 0, (), (), (),
                        pre_generalize if head_generalizes else ())
    if not head_generalizes:
        _SC_ORIG_RUN_STAGES(ctx, module, 'func.func(linalg-generalize-named-ops)', 0, (), (), (),
                            pre_generalize)
    rows = 2 if selected[0] == _SC2X4_MARKER else 1
    outputs = {_SC_MARKER: 1, _SC2_MARKER: 2, _SC4_MARKER: 4, _SC8_MARKER: 8, _SC2X4_MARKER: 4}[selected[0]]
    name = 'scalar_contraction_accumulator' + (
        '_2x4_outputs' if rows != 1 else '' if outputs == 1 else '_' + str(outputs) + '_outputs')
    count = _scalarize_tensor_contractions(ctx, module, outputs, unroll, rows)
    print('OK', name, count)
    if unroll:
        print('OK unroll_scalar_contraction_reduction_by_' + str(unroll), count)
    _SC_ORIG_RUN_STAGES(ctx, module, ','.join(passes[i + 1:]), erase, mid, late, post_openmp, ())
"""
)


def apply_for_test(mlir_text: str, *, outputs: int = 1, reduction_unroll: int = 0, rows: int = 1) -> tuple[str, int]:
    """Run the actual shipped structural rewrite in the toolchain owning Python."""
    import subprocess
    import tempfile
    from pathlib import Path

    from .toolchain import m2m_python

    if type(outputs) is not int or outputs not in (1, 2, 4, 8):
        raise ValueError("scalar accumulator outputs must be 1, 2, 4 or 8")
    if type(reduction_unroll) is not int or reduction_unroll not in (0, 2, 4):
        raise ValueError("scalar accumulator reduction unroll must be 0, 2 or 4")
    if type(rows) is not int or rows not in (1, 2) or (rows == 2 and outputs != 4):
        raise ValueError("rectangular scalar accumulator schedule requires two rows and four columns")
    work = Path(tempfile.mkdtemp(prefix="merlin_scalar_contraction_"))
    src, script = work / "in.mlir", work / "run.py"
    src.write_text(mlir_text)
    script.write_text(
        "import sys\nfrom torch_mlir import ir\n"
        + RUNNER_PRELUDE.split("_SC_MARKER =", 1)[0]
        + "ctx = ir.Context()\nmodule = ir.Module.parse(open(sys.argv[1]).read(), ctx)\n"
        + f"print('COUNT', _scalarize_tensor_contractions(ctx, module, {outputs}, {reduction_unroll}, {rows}))\n"
        "print('MODULE_BEGIN')\nprint(module)\n"
    )
    proc = subprocess.run([str(m2m_python()), str(script), str(src)], capture_output=True, text=True, timeout=120)
    if proc.returncode:
        raise RuntimeError(f"scalar contraction rewrite failed:\n{proc.stdout}\n{proc.stderr}")
    count = int(next(x.split()[1] for x in proc.stdout.splitlines() if x.startswith("COUNT ")))
    return proc.stdout.split("MODULE_BEGIN\n", 1)[1], count
