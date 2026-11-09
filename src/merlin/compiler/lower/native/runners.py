"""Exact child-script templates and feature-selected native runner construction."""

from __future__ import annotations

from merlin.compiler.lower.native.stages import (
    _ADJACENT_EXP_PRELUDE,
    _ALLOCA_SCOPE_LOWER_PRELUDE,
    _AUTOMATIC_HOST_FEATURES_PRELUDE,
    _BROADCAST_FOLD_PRELUDE,
    _BROADCAST_MATH_HOIST_PRELUDE,
    _CONCAT_DPS_PRELUDE,
    _CONTIGUOUS_COPY_MID_SRC,
    _CONTIGUOUS_COPY_PRELUDE,
    _COPY_EXPAND_PRELUDE,
    _FMA_INTRINSIC_PRELUDE,
    _GUARDED_ZERO_CONTRACTION_PRELUDE,
    _INT_SOFTMAX_TABLE_PRELUDE,
    _MASKED_CONTRACTION_PRELUDE,
    _MASKED_CONTRACTION_STAGE_RUNNER,
    _MID_STAGE_SRC,
    _NAMED_BROADCAST_FOLD_PRELUDE,
    _PANEL_PARALLEL_MID_SRC,
    _PANEL_PARALLEL_PRELUDE,
    _PARALLEL_COARSEN_PRELUDE,
    _PARALLEL_COARSEN_STAGE_SRC,
    _PARALLEL_GRAIN_LATE_SRC,
    _PARALLEL_GRAIN_PRELUDE,
    _PARALLEL_TEAM_PRELUDE,
    _PARALLEL_TEAM_STAGE_SRC,
    _RHS_COLUMN_REUSE_PRELUDE,
    _ROUND_INTRINSIC_PRELUDE,
    _SCALAR_CONTRACTION_PRELUDE,
    _SCALAR_ORDERED_REDUCTION_PRELUDE,
    _SCALAR_POINTWISE_PACKET_PRELUDE,
    _SCALAR_POINTWISE_UNROLL_PRELUDE,
    _SCALAR_SQUARED_SUM_PRELUDE,
    _SELFCOPY_PRELUDE,
    _TRANSPOSE_FUSE_PRELUDE,
    _TRANSPOSE_MAPS_PRELUDE,
    _UNIFORM_FILL_COPY_MID_SRC,
    _UNIFORM_FILL_COPY_PRELUDE,
    DEALLOC_CHECK_PRELUDE,
    DEALLOC_CHECK_RUNNER,
    SLICED_OUTPUT_PRELUDE,
    SLICED_OUTPUT_RUNNER,
    SPLIT_FRESH_DEALLOCS_PRELUDE,
    SPLIT_FRESH_DEALLOCS_RUNNER,
    fusion_guard,
)

_SET_LAYOUT = (
    "len(sys.argv) > 17 and sys.argv[17] and module.operation.attributes.__setitem__("
    "'llvm.data_layout', ir.StringAttr.get(sys.argv[17], ctx)); "
)


EMIT_TRANSLATE = _SET_LAYOUT + "f.write(str(llvm.translate_module_to_llvmir(module.operation)))"


EMIT_DUMP = _SET_LAYOUT + "f.write(str(module.operation))"


_RUNNER_SRC = (
    r"""
import sys

from torch_mlir import ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects import llvm
"""
    + _SELFCOPY_PRELUDE
    + _TRANSPOSE_FUSE_PRELUDE
    + _TRANSPOSE_MAPS_PRELUDE
    + _BROADCAST_FOLD_PRELUDE
    + _NAMED_BROADCAST_FOLD_PRELUDE
    + _COPY_EXPAND_PRELUDE
    + _UNIFORM_FILL_COPY_PRELUDE
    + _CONTIGUOUS_COPY_PRELUDE
    + _CONCAT_DPS_PRELUDE
    + _PARALLEL_GRAIN_PRELUDE
    + _PARALLEL_TEAM_PRELUDE
    + _PARALLEL_COARSEN_PRELUDE
    + _PANEL_PARALLEL_PRELUDE
    + _MID_STAGE_SRC
    + _CONTIGUOUS_COPY_MID_SRC
    + _UNIFORM_FILL_COPY_MID_SRC
    + _PANEL_PARALLEL_MID_SRC
    + _PARALLEL_GRAIN_LATE_SRC
    + _PARALLEL_TEAM_STAGE_SRC
    + _PARALLEL_COARSEN_STAGE_SRC
    + _ALLOCA_SCOPE_LOWER_PRELUDE
    + _ROUND_INTRINSIC_PRELUDE
    + _INT_SOFTMAX_TABLE_PRELUDE
    + _FMA_INTRINSIC_PRELUDE
    + _SCALAR_CONTRACTION_PRELUDE
    + _GUARDED_ZERO_CONTRACTION_PRELUDE
    + _ADJACENT_EXP_PRELUDE
    + _MASKED_CONTRACTION_PRELUDE
    + _MASKED_CONTRACTION_STAGE_RUNNER
    + _SCALAR_SQUARED_SUM_PRELUDE
    + _SCALAR_ORDERED_REDUCTION_PRELUDE
    + _RHS_COLUMN_REUSE_PRELUDE
    + _SCALAR_POINTWISE_UNROLL_PRELUDE
    + _SCALAR_POINTWISE_PACKET_PRELUDE
    + _BROADCAST_MATH_HOIST_PRELUDE
    + _AUTOMATIC_HOST_FEATURES_PRELUDE
    + DEALLOC_CHECK_PRELUDE
    + DEALLOC_CHECK_RUNNER
    + SPLIT_FRESH_DEALLOCS_PRELUDE
    + SPLIT_FRESH_DEALLOCS_RUNNER
    + SLICED_OUTPUT_PRELUDE
    + SLICED_OUTPUT_RUNNER
    + r'''

def _residual_vector_ops(module):
    """Return vector-dialect op names still present at the LLVM translation edge."""
    found = []

    def visit(op):
        name = op.operation.name
        if name.startswith('vector.'):
            found.append(name)
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child)

    for top in module.body.operations:
        visit(top)
    return sorted(set(found))

src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = ir.Context()
with open(src_path) as f:
    module = ir.Module.parse(f.read(), ctx)
# __MERLIN_INSPECT_PARSED__
# int_softmax_table (default-off): restructure every integer softmax in the captured IR, exactly,
# before any other rewrite sees it. See llvmlower/int_softmax_table.py.
if _INT_SOFTMAX_TABLE:
    _ist_run_and_report(ctx, module)
# fuse_transpose_b (default-off): fold `matmul(A, transpose(B))` into a transpose-b matmul BEFORE
# the pass manager runs, so the (still-named) linalg.matmul carries the transposed-B indexing map
# and the frozen RVV schedule tiles+vectorizes it while the scalar weight transpose disappears.
if _FUSE_TRANSPOSE_B:
    print("OK fuse_transpose_b", _fuse_transpose_b(module, ctx))
# fold_weight_transpose (default-off): the general form of the fold above -- a loop-invariant weight
# transpose sinks into the indexing_maps of EVERY linalg consumer, not just a linalg.matmul's B
# operand. The quantized datapath emits its contraction as a linalg.generic, so the matmul-only fold
# fires zero times on an int8 model; this one folds all 15 of small_llama's weight transposes.
if _FOLD_WEIGHT_TRANSPOSE:
    _fwt_n, _fwt_report = _fold_weight_transposes(module, ctx)
    for _fwt_kind, _fwt_detail in _fwt_report:
        print("OK fold_weight_transpose", _fwt_kind, _fwt_detail)
    print("OK fold_weight_transpose folded", _fwt_n)
# fold_broadcast_into_generic (default-off): replace a sole-use broadcast intermediate by a
# projected indexing map on an already-all-parallel generic. Reduction consumers are refused.
if _FOLD_BROADCAST:
    _bf_n, _bf_report = _fold_broadcasts(module, ctx)
    for _bf_kind, _bf_detail in _bf_report:
        print("OK fold_broadcast_into_generic", _bf_kind, _bf_detail)
    print("OK fold_broadcast_into_generic folded", _bf_n)
# concat_dps (default-off): a `tensor.concat` operand produced by a destination-passing op is
# retargeted to write STRAIGHT INTO the concatenated buffer, so bufferization has no data movement
# left to emit for it. Must run BEFORE the pass manager: after one-shot-bufferize the destination is
# no longer an operand, only a copy. See llvmlower/concat_dps.py.
if _CONCAT_DPS:
    _cd_n, _cd_report = _concat_dps(module, ctx)
    for _cd_kind, _cd_detail in _cd_report:
        print("OK concat_dps", _cd_kind, _cd_detail)
    print("OK concat_dps rewrote", _cd_n)
_run_stages(ctx, module, pipeline, _ERASE_SELF_COPY, _MID_STAGES, _LATE_STAGES,
            _POST_OPENMP_STAGES, _PRE_GENERALIZE_STAGES)
_vector_residue = _residual_vector_ops(module)
if _vector_residue:
    raise RuntimeError(
        'vector dialect survived the LLVM lowering edge: ' + ', '.join(_vector_residue))
# __MERLIN_INSPECT_LOWERED__
with open(out_path, "w") as f:
    __MERLIN_EMIT__
print("OK")
'''
)


_RUNNER = (
    _RUNNER_SRC.replace("__MERLIN_EMIT__", EMIT_TRANSLATE)
    .replace("__MERLIN_INSPECT_PARSED__", "")
    .replace("__MERLIN_INSPECT_LOWERED__", "")
)


_RUNNER_ACT_POLY_HEAD = r"""
import sys

from torch_mlir import ir
from torch_mlir import ir as _ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects import llvm
"""


_RUNNER_ACT_POLY_TAIL = (
    _SELFCOPY_PRELUDE
    + _TRANSPOSE_FUSE_PRELUDE
    + _TRANSPOSE_MAPS_PRELUDE
    + _BROADCAST_FOLD_PRELUDE
    + _NAMED_BROADCAST_FOLD_PRELUDE
    + _COPY_EXPAND_PRELUDE
    + _UNIFORM_FILL_COPY_PRELUDE
    + _CONTIGUOUS_COPY_PRELUDE
    + _CONCAT_DPS_PRELUDE
    + _PARALLEL_GRAIN_PRELUDE
    + _PARALLEL_TEAM_PRELUDE
    + _PARALLEL_COARSEN_PRELUDE
    + _PANEL_PARALLEL_PRELUDE
    + _MID_STAGE_SRC
    + _CONTIGUOUS_COPY_MID_SRC
    + _UNIFORM_FILL_COPY_MID_SRC
    + _PANEL_PARALLEL_MID_SRC
    + _PARALLEL_GRAIN_LATE_SRC
    + _PARALLEL_TEAM_STAGE_SRC
    + _PARALLEL_COARSEN_STAGE_SRC
    + _ALLOCA_SCOPE_LOWER_PRELUDE
    + _ROUND_INTRINSIC_PRELUDE
    + _INT_SOFTMAX_TABLE_PRELUDE
    + _FMA_INTRINSIC_PRELUDE
    + _SCALAR_CONTRACTION_PRELUDE
    + _GUARDED_ZERO_CONTRACTION_PRELUDE
    + _ADJACENT_EXP_PRELUDE
    + _MASKED_CONTRACTION_PRELUDE
    + _MASKED_CONTRACTION_STAGE_RUNNER
    + _SCALAR_SQUARED_SUM_PRELUDE
    + _SCALAR_POINTWISE_UNROLL_PRELUDE
    + _SCALAR_POINTWISE_PACKET_PRELUDE
    + _BROADCAST_MATH_HOIST_PRELUDE
    + _AUTOMATIC_HOST_FEATURES_PRELUDE
    + DEALLOC_CHECK_PRELUDE
    + DEALLOC_CHECK_RUNNER
    + SPLIT_FRESH_DEALLOCS_PRELUDE
    + SPLIT_FRESH_DEALLOCS_RUNNER
    + SLICED_OUTPUT_PRELUDE
    + SLICED_OUTPUT_RUNNER
    + r"""
src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = ir.Context()
with open(src_path) as f:
    module = ir.Module.parse(f.read(), ctx)
# __MERLIN_INSPECT_PARSED__
# The post-bufferization rewrites are NOT act_poly's business, but they have to run on THIS runner
# too. This tail used to drive the PassManager directly, which meant `erase_self_copy` and
# `fuse_transpose_b` were accepted, threaded through argv, and then never executed whenever
# `vectorized_transcendental_activation` was also enabled. MEASURED on small_llama_int8
# (hand_v0_int8 package, act_poly + erase_self_copy): the 17 in-loop `@memrefCopy` call sites the
# erase removes on the plain runner were all still there. Since the whole-model proposer enables
# act_poly by default, that is every beam fork -- and the erase read as an inert lever.
# int_softmax_table (default-off): restructure every integer softmax in the captured IR, exactly,
# before any other rewrite sees it. See llvmlower/int_softmax_table.py.
if _INT_SOFTMAX_TABLE:
    _ist_run_and_report(ctx, module)
if _FUSE_TRANSPOSE_B:
    print("OK fuse_transpose_b", _fuse_transpose_b(module, ctx))
# fold_weight_transpose (default-off): the general form of the fold above -- a loop-invariant weight
# transpose sinks into the indexing_maps of EVERY linalg consumer, not just a linalg.matmul's B
# operand. The quantized datapath emits its contraction as a linalg.generic, so the matmul-only fold
# fires zero times on an int8 model; this one folds all 15 of small_llama's weight transposes.
if _FOLD_WEIGHT_TRANSPOSE:
    _fwt_n, _fwt_report = _fold_weight_transposes(module, ctx)
    for _fwt_kind, _fwt_detail in _fwt_report:
        print("OK fold_weight_transpose", _fwt_kind, _fwt_detail)
    print("OK fold_weight_transpose folded", _fwt_n)
if _FOLD_BROADCAST:
    _bf_n, _bf_report = _fold_broadcasts(module, ctx)
    for _bf_kind, _bf_detail in _bf_report:
        print("OK fold_broadcast_into_generic", _bf_kind, _bf_detail)
    print("OK fold_broadcast_into_generic folded", _bf_n)
# concat_dps (default-off): a `tensor.concat` operand produced by a destination-passing op is
# retargeted to write STRAIGHT INTO the concatenated buffer, so bufferization has no data movement
# left to emit for it. Must run BEFORE the pass manager: after one-shot-bufferize the destination is
# no longer an operand, only a copy. See llvmlower/concat_dps.py.
if _CONCAT_DPS:
    _cd_n, _cd_report = _concat_dps(module, ctx)
    for _cd_kind, _cd_detail in _cd_report:
        print("OK concat_dps", _cd_kind, _cd_detail)
    print("OK concat_dps rewrote", _cd_n)
with ctx, ir.Location.unknown():
    _n = apply_activation_polynomial(module, ctx)
_run_stages(ctx, module, pipeline, _ERASE_SELF_COPY, _MID_STAGES, _LATE_STAGES,
            _POST_OPENMP_STAGES, _PRE_GENERALIZE_STAGES)
# __MERLIN_INSPECT_LOWERED__
with open(out_path, "w") as f:
    __MERLIN_EMIT__
print("OK act_poly rewrote", _n)
"""
)


def _accum_microkernel_v3_features() -> frozenset[str]:
    """Names of the accumulator-resident micro-kernel v3 features (the ones whose pipeline splices the
    SCALARIZE_MARKER and need the two-stage A-scalarization runner). Imported lazily so importing
    pipeline never pulls accum_microkernel / impr_features at module load."""
    from merlin.llvmlower.impr_features import ACCUM_RESIDENT_V3_NAMES

    return frozenset(ACCUM_RESIDENT_V3_NAMES)


def _needs_scalarize_runner(pipeline: str, feats: "frozenset[str]") -> bool:
    """Does this lowering need the two-stage A-scalarization runner?

    GROUND TRUTH is the marker's presence in the built pass pipeline — NOT membership in the static
    ACCUM_RESIDENT_V3_NAMES grid. The register block is a continuous beam-tunable knob, so v3 tuning
    points are registered on demand for arbitrary (MR,NR,KC); a static-name check silently missed them,
    the plain runner was used, and SCALARIZE_MARKER leaked into the pipeline as if it were a pass
    ("'__merlin_scalarize_a__' does not refer to a registered pass") — every non-grid MR failed at K2.
    Keying on the marker makes this correct for any registration mechanism."""
    from merlin.llvmlower.accum_microkernel import SCALARIZE_MARKER

    if SCALARIZE_MARKER in pipeline:
        return True
    return any(f in feats for f in _accum_microkernel_v3_features())


def _activation_poly_runner(emit: str = EMIT_TRANSLATE, *, fused: bool = False) -> str:
    """The lowering runner with the transcendental->polynomial rewriter spliced in (default-off
    feature). Imported here (not at module top) so importing pipeline never pulls act_poly."""
    from merlin.llvmlower.act_poly import rewrite_source

    return _RUNNER_ACT_POLY_HEAD + rewrite_source(fused=fused) + _RUNNER_ACT_POLY_TAIL.replace("__MERLIN_EMIT__", emit)


def _select_runner(
    pipeline: str,
    feats: "frozenset[str]",
    *,
    emit: str,
    inspection_dir: str | None = None,
    keep_exact: bool = False,
    printing: tuple[bool, bool] = (True, True),
) -> str:
    """Pick the lowering-runner source for these features and bind how it emits its result.

    The runner variant is chosen by FEATURE (act_poly rewriter / accum-v3 two-stage
    scalarization / plain); ``emit`` is orthogonal and chosen by TRANSPORT (in-process
    translate vs dump-for-mlir-translate). Keeping the two independent is what lets the
    multicore path carry the feature rewrites — see the EMIT_* comment.
    """
    from merlin.llvmlower.ir_inspection import bind_inspection
    from merlin.llvmlower.source_observation_stage import FEATURE as _SOURCE_OBSERVATION_FEATURE
    from merlin.llvmlower.source_observation_stage import bind_runner as _bind_source_observation_runner

    if {"approximate_transcendental_activation", "vectorized_transcendental_activation"} & feats:
        source = _activation_poly_runner(emit, fused="fuse_activation_polynomial_fma" in feats)
    elif _needs_scalarize_runner(pipeline, feats):
        from merlin.llvmlower.accum_microkernel import run_source
        from merlin.llvmlower.bmm_tail_pad import FEATURE as _BMM_TAIL_PAD_FEATURE

        source = run_source(tag_bmm_tails=_BMM_TAIL_PAD_FEATURE in feats).replace("__MERLIN_EMIT__", emit)
    else:
        source = _RUNNER_SRC.replace("__MERLIN_EMIT__", emit)
    from merlin.llvmlower.automatic_host_features import bind_runner as bind_automatic_host_runner

    source = bind_automatic_host_runner(source, features=feats)
    from merlin.llvmlower.adjacent_exp import bind_runner as bind_exact_exp_runner

    source = bind_exact_exp_runner(source, features=feats)
    source = _bind_source_observation_runner(source, selected=_SOURCE_OBSERVATION_FEATURE in feats)
    # Every variant runs elementwise fusion under the broadcast control function (fusion_guard).
    return bind_inspection(
        fusion_guard.inject(source),
        inspection_dir,
        keep_exact=keep_exact,
        print_before=printing[0],
        print_after=printing[1],
    )
