"""Published RVV whole-model vector feature implementation.

Extracted from Merlin's validated accumulator-resident path. The feature's
schedule and pipeline edits live with the RVV provider and are byte-checked.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

WHOLEMODEL_VF_NAME = "accumulator_resident_wholemodel_vf"
WHOLEMODEL_VF_CAPS = (4, 16, 16)
WHOLEMODEL_VF_MR_MM = 1
WHOLEMODEL_VF_NR_BMM = 8
SCALARIZE_MARKER = "__merlin_scalarize_a__"


def _accumulator_resident_v3_pre_schedule(
    MR: int,
    NR: int,
    KC: int,
    NR_bmm: int | None = None,
    MR_mm: int | None = None,
    skip_mm: bool = False,
    skip_bmm: bool = False,
    pad_bmm: bool = False,
) -> str:
    """PRE-bufferize schedule for the v3 (vfmacc.vf) micro-kernel — SAME as the v1/v2 pre-schedule
    but WITHOUT ``bufferize_to_allocation``.

    The v1/v2 pre-schedule promoted the C tile to a TYPED local ``memref.alloc`` (via
    ``bufferize_to_allocation``) because v1 ran ``hoist_redundant_vector_transfers`` POST-bufferize
    and that pass needed a typed alloc to (try to) see. v3 runs ``loop-invariant-subset-hoisting`` on
    the TENSOR form (before bufferize), where the K-loop carries the C tile as a value-semantic
    ``tensor<MRxNR>`` iter_arg directly (a ``tensor.extract_slice`` of the output) — no local alloc
    needed. Dropping ``bufferize_to_allocation`` removes the per-M,N-tile C copy-in/copy-out
    (``materialize_in_destination`` -> ``memref.copy``) that otherwise dominated the timed region
    (measured: ~634K of the 692K instret @64^3 was ``memrefCopy``/``memcpy`` from that promotion,
    while the actual ``forward`` compute kernel was only ~57K — at the hand ceiling). Without it the
    accumulator write lands in-place on the output buffer, so the compute kernel stands alone.

    Otherwise identical to the v1 pre-schedule: tile [MR,NR]; tile K by 1; scoped-vectorize [MR,NR,1]
    (one B row + MR A scalars per K step -> the resident-accumulator micro-kernel shape); rebuild
    ``vector.contract``; STOP (the v3 pipeline does subset-hoist + contract-lower + A-scalarize). The
    M-tail (``MR_mm``) / N-tail (``NR_bmm``) clamps carry over unchanged for whole-model safety.

    ``skip_mm`` / ``skip_bmm`` OMIT that op class's arms entirely, leaving those contractions for
    ``convert-linalg-to-loops`` (scalar). Needed because a claim can be impossible: when every block
    legal for a class's extents is 1-lane wide (an N=1 extent forces NR=1), tiling buys no vector
    lanes AND produces a parallel-dim-free ``vector.contract`` that no lowering strategy matches. Not
    claiming the class is then strictly better than claiming it badly — the model builds and runs, and
    the caller reports the class as un-vectorized rather than failing the build.
    """
    NB = NR_bmm if NR_bmm is not None else NR
    MM = MR_mm if MR_mm is not None else MR
    if __import__("os").environ.get("MERLIN_PAD_M"):
        # M-PADDING (default OFF): pad matmul M to multiple of MR=4 BEFORE tiling so small-M (openvla M=17)
        # register-blocks CLEANLY at MR=4 (no masked transfer_write scalar fallback). Measure vs MR=1 vf.
        return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %cp = transform.structured.pad %mm pad_to_multiple_of [4] {{padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32], padding_dimensions = [0]}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %t1, %lmn:2 = transform.structured.tile_using_for %padded tile_sizes [4, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [4, {NR}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""
    # EXPERIMENT (env MERLIN_VEC_EW, default OFF -> baseline byte-identical): after matmul/bmm are tiled
    # + vectorized, the remaining linalg.generic ops are the NON-matmul ops (activations/softmax/norm/
    # layout) — which otherwise fall through convert-linalg-to-loops to SCALAR (openvla's 1100ms). Match
    # + scoped-vectorize them here. NOTE: this is the BLUNT version (matches ALL generics incl. reduction
    # softmax/norm — may break cos or explode); the measurement tells us if scoped/tagged is needed.
    # PER-RANK BOUNDED vectorize (env MERLIN_VEC_RANK, default OFF): the non-matmul parallel generics
    # (tagged merlin.vec_rN by the pre-pass, N = loop rank) are vectorized with BOUNDED vector_sizes
    # [1,..,1,8] — innermost dim by 8 lanes (VLEN256/f32), NOT the plain no-sizes vectorize that
    # explodes (vector<17x576>=9792 lanes -> 8725ms). Per rank because openvla generics are rank 2/3/4.
    _ew = (
        """
    %g2 = transform.structured.match attributes{merlin.vec_r2} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g2 vector_sizes [1, 8] : !transform.any_op
    %g3 = transform.structured.match attributes{merlin.vec_r3} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g3 vector_sizes [1, 1, 8] : !transform.any_op
    %g4 = transform.structured.match attributes{merlin.vec_r4} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g4 vector_sizes [1, 1, 1, 8] : !transform.any_op"""
        if __import__("os").environ.get("MERLIN_VEC_RANK")
        else (
            """
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g : !transform.any_op"""
            if __import__("os").environ.get("MERLIN_VEC_EW")
            else ""
        )
    )
    _mm_arm = f"""
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [{MM}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MM}, {NR}, 1] : !transform.any_op"""
    if skip_mm:
        _mm_arm = ""
    _bmm_arm = f"""
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt1, %blmn:3 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NB}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %bm2 = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt2, %blk = transform.structured.tile_using_for %bm2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt2 vector_sizes [1, {MR}, {NB}, 1] : !transform.any_op"""
    if skip_bmm:
        _bmm_arm = ""
    _pad_sequence = (
        ""
        if not pad_bmm
        else f"""
  transform.named_sequence @__transform_bmm_pad(%arg0: !transform.any_op {{transform.readonly}}) {{
    %btail = transform.structured.match attributes{{merlin.bmm_pad_tail}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bpadded, %bpad, %bcp = transform.structured.pad %btail pad_to_multiple_of [{MR}, {NB}] {{padding_dimensions = [1, 2], copy_back_op = "linalg.copy"}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }}
"""
    )
    return f"""\
module attributes {{transform.with_named_sequence}} {{
{_pad_sequence}
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{{_mm_arm}{_bmm_arm}{_ew}
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _zero_attr(t: str) -> str:
    """The MLIR zero literal for element type ``t``, derived from the type rather than tabulated.

    Integer types take ``0 : iN``; floating types take the f-literal form. A tabulated map would go
    stale the first time a new element type appears, and the failure is silent (scalar fallback).
    """
    t = str(t).strip()
    if t.startswith("i") and t[1:].isdigit():
        return f"0 : {t}"
    if t.startswith(("f", "bf")):
        return f"0.000000e+00 : {t}"
    raise ValueError(
        f"no zero literal derivable for element type {t!r}; add the rule rather than "
        "letting the schedule pad with a wrongly-typed value"
    )


_ACCUM_RESIDENT_V2_LOWER_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_accum_v2_lower(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def _accum_v2_lower_schedule_path() -> Path:
    """Stable temp path for the PRE-bufferize contract-lowering schedule (content-addressed)."""
    text = _ACCUM_RESIDENT_V2_LOWER_SCHEDULE
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    p = Path(tempfile.gettempdir()) / f"merlin_accum_v2_lower_{h}.mlir"
    if not p.is_file():
        p.write_text(text, encoding="utf-8")
    return p


def _accumulator_resident_v2_pipeline(passes: list[str]) -> list[str]:
    """Splice the PRE-bufferize subset-hoist + contract-lower right after the FIRST
    transform-interpreter (still on tensors), before one-shot-bufferize.

    The pre-schedule (edit_schedule) stops at `vector.contract`; here we (1) run
    `loop-invariant-subset-hoisting` on the tensor form so the accumulator transfer pair becomes a
    `vector<MRxNR>` scf.for iter_arg (register-resident across K), then (2) run a second
    transform-interpreter that lowers the now-hoisted contraction -> outerproduct -> vector.fma.
    Running BEFORE bufferize is the whole point: post-bufferize the carried accumulator is a memref
    iter_arg the hoist cannot lift (the prior feature's measured blocker). Only runs when this feature
    is enabled; baseline pipeline untouched."""
    out = list(passes)
    # Locate the first transform-interpreter (the pre-schedule's __transform_main) and the
    # canonicalize,cse that follow it; splice after that cse (still on tensors, pre-bufferize).
    ti = None
    for i, p in enumerate(out):
        if p.startswith("transform-interpreter{entry-point=__transform_main}"):
            ti = i
            break
    if ti is None:
        return out
    # after ti come canonicalize, cse (then linalg-generalize-named-ops, one-shot-bufferize)
    insert_at = ti + 1
    # skip the canonicalize, cse pair that the pipeline puts right after the interpreter
    while insert_at < len(out) and out[insert_at] in ("canonicalize", "cse"):
        insert_at += 1
    sched = _accum_v2_lower_schedule_path()
    inject = [
        "loop-invariant-subset-hoisting",
        f"transform-preload-library{{transform-library-paths={sched}}}",
        "transform-interpreter{entry-point=__transform_accum_v2_lower}",
        "canonicalize",
        "cse",
    ]
    if out[insert_at : insert_at + len(inject)] != inject:  # idempotent
        out = out[:insert_at] + inject + out[insert_at:]
    # Same convert-vector-to-scf{full-unroll} swap as the v1 feature: the hoisted accumulator is a
    # rank-2 `vector<MRxNR>` carried by the K loop; the default convert-vector-to-scf would lower
    # rank>=2 transfers through a stack alloca scratch (re-introducing the per-K stack round-trip).
    # full-unroll lowers them to rank-1 element ops with NO stack scratch, so the carried accumulator
    # stays in the vector register file across K. Feature-scoped; baseline untouched.
    for i, p in enumerate(out):
        if p == "convert-vector-to-scf":
            out[i] = "convert-vector-to-scf{full-unroll}"
            break
    return out


def _accumulator_resident_v3_pipeline(passes: list[str]) -> list[str]:
    """v2 pipeline (PRE-bufferize subset hoist + contract lower) PLUS the SCALARIZE_MARKER sentinel
    spliced immediately AFTER the contract-lowering transform-interpreter (still on tensors, before
    one-shot-bufferize). The two-stage runner splits the pipeline at the marker and runs the
    A-scalarization rewrite there. Only runs when a v3 feature is enabled; baseline untouched."""

    out = _accumulator_resident_v2_pipeline(passes)
    # The v2 edit spliced: loop-invariant-subset-hoisting, transform-preload-library{...v2_lower},
    # transform-interpreter{entry-point=__transform_accum_v2_lower}, canonicalize, cse. Put the
    # marker right AFTER that interpreter's canonicalize,cse (contract is now vector.fma w/ f32
    # A-extracts; bufferize has not run).
    idx = None
    for i, p in enumerate(out):
        if p.startswith(
            "transform-interpreter{entry-point=__transform_accum_v2_lower}"
        ):
            idx = i
            break
    if idx is None:
        return out
    insert_at = idx + 1
    while insert_at < len(out) and out[insert_at] in ("canonicalize", "cse"):
        insert_at += 1
    if SCALARIZE_MARKER not in out:
        out = out[:insert_at] + [SCALARIZE_MARKER] + out[insert_at:]
    return out


def wholemodel_vf_schedule() -> str:
    return _accumulator_resident_v3_pre_schedule(
        *WHOLEMODEL_VF_CAPS, NR_bmm=WHOLEMODEL_VF_NR_BMM, MR_mm=WHOLEMODEL_VF_MR_MM
    )


def wholemodel_vf_pipeline(passes: list[str]) -> list[str]:
    return _accumulator_resident_v3_pipeline(passes)


_VFMACC_TILED_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [4, 16, 16] : !transform.any_op
    %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, 4, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, 4, 16, 16] : !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_outerproduct
    } : !transform.any_op
    transform.yield
  }
}
"""


def fused_vfmacc_tiled_schedule() -> str:
    return _VFMACC_TILED_SCHEDULE


# Published accumulator-resident whole-model variant (pre-vfmacc.vf).
def _accumulator_resident_pre_schedule(
    MR: int, NR: int, KC: int, NR_bmm: int | None = None, MR_mm: int | None = None
) -> str:
    """PRE-bufferize transform schedule: tile M,N; promote C to a TYPED local alloc via
    bufferize_to_allocation; tile the reduction K by 1; scoped-vectorize [MR,NR,1]; rebuild
    vector.contract. Stops at the contract (the post-bufferize schedule does the hoist + lowering).

    K is tiled by **1** (not KC) and vectorized [MR,NR,1] on purpose: that makes each K step read a
    SINGLE B row (`vector<1xNR>` -> `vector<NR>`) + MR A scalars and do MR `vfmacc` into the resident
    accumulator — structurally identical to the expert/hand micro-kernel (MR accumulator vreg-groups
    held across K, one B row + A scalars streamed per step). Vectorizing the full [MR,NR,KC] tile
    instead materializes the whole `vector<KCxNR>` B panel (KC vector registers) and SPILLS hard (40
    vector spills measured at KC=16, faulting at M>=64). The `KC` argument now sets the outer K
    register-block trip only conceptually; the streamed inner step is always 1 (bounded vreg use).
    C copy-in uses `linalg.copy` (vectorizable / cheap) rather than the default `memref.copy` runtime
    call. The batch_matmul (attention) path gets the same treatment so the feature is whole-model-safe.

    N-TAIL (``NR_bmm``): the batch_matmul (attention) N is often SMALL (e.g. llama-style N=8). When
    the batch_matmul is vectorized at the matmul NR (16) on an N=8 dim, the partial inner write needs
    a masked `vector.transfer_write` (vector<...x16> into tensor<...x8>) and LLVM-23 rejects the
    multi-op `vector.mask` -> PipelineError -> (whole-model) silent scalar fallback. Setting a
    SEPARATE batch_matmul NR (``NR_bmm`` <= the small N, default = NR) clamps NR=min(NR,N) so the
    vectorize is FULL (no mask) and the N=8 attention batch_matmul vectorizes to vfmacc. The matmul
    path keeps its own (larger) NR. For larger N, NR_bmm just tiles N cleanly. This is the N-tail fix
    that makes the accumulator-resident feature whole-model-safe for attention.

    M-TAIL (``MR_mm``): the M-side analog of the N-tail. A whole-model decode step is dominated by
    matmuls whose LEADING dim is M=1 (one token row); the schedule's MR=4 register tile then tries to
    write a `vector<4xNR>` into a `tensor<1xNR>` C tile -> a masked `vector.transfer_write` LLVM-23
    rejects with the SAME multi-op `vector.mask` PipelineError as the N-tail (silent scalar fallback,
    no vfmacc). Setting a SEPARATE matmul MR (``MR_mm`` <= the small M, default = MR) clamps
    MR=min(MR,M) on the matmul path so the inner vectorize is FULL (no mask) and the M=1 token-decode
    matmul vectorizes to vfmacc. ``MR_mm=1`` is whole-model-safe by construction: M=1 ops fit exactly
    (no mask) and any larger-M matmul just tiles M into single-row register tiles (still a real vfmacc
    chain, bit-exact). Only the matmul M tile is affected; the batch_matmul keeps its own MR. Combined
    with ``NR_bmm`` the tiled vectorize is tail-aware on BOTH the M side (matmul M<MR) and the N side
    (batch_matmul N<NR) generally — a tile that adapts MR=min(MR,M), NR=min(NR,N), not a memorized shape.
    """
    NB = NR_bmm if NR_bmm is not None else NR
    MM = MR_mm if MR_mm is not None else MR
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [{MM}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %buf, %newmm = transform.structured.bufferize_to_allocation %t1 {{memory_space = 0 : i64, bufferize_destination_only}} : !transform.any_op
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MM}, {NR}, 1] : !transform.any_op
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt1, %blmn:3 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NB}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %bbuf, %bnew = transform.structured.bufferize_to_allocation %bt1 {{memory_space = 0 : i64, bufferize_destination_only}} : !transform.any_op
    %bm2 = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt2, %blk = transform.structured.tile_using_for %bm2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt2 vector_sizes [1, {MR}, {NB}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


_ACCUM_RESIDENT_POST_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_accum_post(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f2 = transform.structured.hoist_redundant_vector_transfers %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f2 {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.apply_patterns to %f2 {
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def _post_schedule_path() -> Path:
    """Write the POST-bufferize hoist+lower schedule to a stable temp file and return its path.

    `edit_pipeline` only sees the pass-list strings (not the per-build work dir), so the second
    transform-interpreter needs a real, stable file to preload. The content is fixed, so the path is
    content-addressed (one file, written once, reused) — deterministic and safe for parallel builds.
    """
    text = _ACCUM_RESIDENT_POST_SCHEDULE
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    p = Path(tempfile.gettempdir()) / f"merlin_accum_resident_post_{h}.mlir"
    if not p.is_file():
        p.write_text(text, encoding="utf-8")
    return p


def _accumulator_resident_pipeline(passes: list[str]) -> list[str]:
    """Splice a SECOND transform-preload-library + transform-interpreter right AFTER
    one-shot-bufferize: it hoists the typed-alloc accumulator transfer pair into the K-loop vector
    iter_arg (register-resident), then lowers contraction->outerproduct->vfmacc. The PRE-bufferize
    schedule (edit_schedule) deliberately stopped at vector.contract so the hoister sees the
    read->contract->write shape on the bufferized memref. Only runs when this feature is enabled;
    baseline pipeline untouched."""
    out = list(passes)
    # find one-shot-bufferize (it carries options, so match by prefix)
    idx = None
    for i, p in enumerate(out):
        if p.startswith("one-shot-bufferize"):
            idx = i
            break
    if idx is None:
        return out
    sched = _post_schedule_path()
    # Insert the hoist IMMEDIATELY after one-shot-bufferize — i.e. BEFORE
    # buffer-results-to-out-params and buffer-hoisting. Critical: buffer-hoisting/loop-hoisting
    # would hoist the per-tile C `memref.alloc` out to the function entry and REUSE one buffer across
    # every M,N tile; the hoister then sees a write to that shared buffer (the copy-in) in the
    # K-loop's parent region and conservatively refuses to lift the accumulator transfer across K.
    # Running the hoist BEFORE buffer-hoisting (while the typed C alloc is still local to the M,N
    # tile, exactly the shape that hoists in isolation) makes it fire -> vector iter_arg.
    insert_at = idx + 1
    inject = [
        f"transform-preload-library{{transform-library-paths={sched}}}",
        "transform-interpreter{entry-point=__transform_accum_post}",
        "canonicalize",
        "cse",
    ]
    if out[insert_at : insert_at + len(inject)] != inject:  # idempotent
        out = out[:insert_at] + inject + out[insert_at:]
    # Swap the default `convert-vector-to-scf` for the full-unroll variant. The hoisted accumulator
    # is a rank-2 `vector<MRxNR>` carried by the K loop; the DEFAULT convert-vector-to-scf lowers
    # every rank>=2 vector transfer through a STACK `memref.alloca` scratch (the carried accumulator
    # then lives on the stack, re-loaded/stored each K step, and under RVV's fixed VLEN it spills
    # hard -> the `tohost=1337` wild-store fault at M>=32). `full-unroll` instead fully unrolls those
    # rank-2 transfers into rank-1 element ops with NO stack scratch, so the accumulator stays in the
    # vector register file across K (register-resident, like the hand kernel's MR vfloat32m4_t accs).
    # Feature-scoped (only when this feature is on); baseline `convert-vector-to-scf` untouched.
    for i, p in enumerate(out):
        if p == "convert-vector-to-scf":
            out[i] = "convert-vector-to-scf{full-unroll}"
            break
    return out


def wholemodel_schedule() -> str:
    return _accumulator_resident_pre_schedule(4, 16, 16, NR_bmm=8, MR_mm=1)


def wholemodel_pipeline(passes: list[str]) -> list[str]:
    return _accumulator_resident_pipeline(passes)


# RVV typed M-pad and BMM-pad accumulator-resident variants.
def _zero_attr(t: str) -> str:
    """The MLIR zero literal for element type ``t``, derived from the type rather than tabulated.

    Integer types take ``0 : iN``; floating types take the f-literal form. A tabulated map would go
    stale the first time a new element type appears, and the failure is silent (scalar fallback).
    """
    t = str(t).strip()
    if t.startswith("i") and t[1:].isdigit():
        return f"0 : {t}"
    if t.startswith(("f", "bf")):
        return f"0.000000e+00 : {t}"
    raise ValueError(
        f"no zero literal derivable for element type {t!r}; add the rule rather than "
        "letting the schedule pad with a wrongly-typed value"
    )


def _accumulator_resident_v3_mrpad_pre_schedule(
    MR: int,
    NR: int,
    KC: int,
    NR_bmm: int | None = None,
    elem_types: tuple[str, str, str] | None = None,
) -> str:
    """PER-MATMUL MR register block with an M-PAD tail — the general fix for the M=1/M%MR!=0 case.

    ``accumulator_resident_wholemodel_vf`` clamps the matmul M tile to MR_mm=1 (no A-operand reuse:
    1 B-load + 1 A-load per FMA = 2.0 loads/useful-FMA). ``..._vf_mr4`` raises it to MR=4 (A-reuse,
    1.25 loads/FMA) but is ONLY correct where M%4==0: on M not divisible by MR the partial M tile
    needs a masked ``vector.transfer_write`` (``vector<MRxNR>`` into ``tensor<(M%MR)xNR>``) that
    LLVM-23 rejects with a multi-op ``vector.mask`` PipelineError -> (measured) NR=8/non-resident/119
    in-loop spills at M=17, and an outright PipelineError -> silent scalar fallback at M=1.

    This schedule handles the tail the way the hand shim does (``round_up_mr``): it PADS the matmul M
    dimension UP to the next multiple of MR BEFORE tiling (``transform.structured.pad`` on padding
    dimension 0, padding value 0.0). Every matmul then register-blocks CLEANLY at MR — the M tile is
    always FULL (no masked write, no PipelineError), so M=1 pads to MR, M=17 pads to 20, M=64 stays 64,
    each getting the same MR resident-accumulator vfmacc.vf register block. ``transform.structured.pad``
    threads a ``tensor.extract_slice`` back to the original M rows (the ``copy_back`` result), so the
    real output is BIT-EXACT: the padded rows are ``0-row @ B = 0`` and are discarded; only the [0:M]
    slice is written back. This is per-matmul (each op pads to ITS OWN next multiple of MR — a general
    tail rule, not a per-model constant), and it composes with the v3 subset-hoist + A-scalarization
    pipeline unchanged (it still stops at ``vector.contract``).

    The batch_matmul (attention) path is IDENTICAL to ``accumulator_resident_wholemodel_vf`` — MR=4 M
    tile + the ``NR_bmm`` N-tail clamp — so this feature only upgrades the ``linalg.matmul`` path from
    the MR=1 clamp to a padded-MR register block; the proven whole-model attention path is untouched.
    """
    NB = NR_bmm if NR_bmm is not None else NR
    # The padding value must have the ELEMENT TYPE OF THE OPERAND IT PADS. This was a single f32
    # literal reused for all three, which is correct only while every contraction is f32 -- and
    # `transform.structured.pad` does not warn, it hard-errors:
    #   'transform.structured.pad' op expects a padding value of type 'i8', got 0.000000e+00 : f32
    # A transform-interpreter error is a PipelineError, which the lowering catches as a whole-model
    # SILENT SCALAR FALLBACK -- the failure mode is "mysteriously slow", not "build failed".
    # It has never fired because the int8 datapath rewrites every linalg.matmul into a
    # linalg.generic, so no i8 matmul reaches this schedule at all (see the module docstring note on
    # named-op erasure). It fires the moment that is fixed, which is why it is fixed first.
    za, zb, zc = (_zero_attr(t) for t in (elem_types or ("f32", "f32", "f32")))
    # copy_back_op = "linalg.copy" (NOT the default bufferization.materialize_in_destination): the
    # pipeline's canonicalize/cse merges the many identical `linalg.fill 0 -> tensor<MxNxf32>` matmul
    # inits into ONE shared value, and materialize_in_destination copies each padded result back into
    # that SHARED fill buffer -> one-shot-bufferize sees N writes aliasing one buffer -> "cannot avoid
    # RaW conflict" -> whole-model PipelineError -> silent scalar fallback (this is EXACTLY what breaks
    # the pre-existing vf_mr4 on rdt2). A `linalg.copy` copy-back (paired with eliminate-empty-tensors
    # in the edit_pipeline) bufferizes each padded matmul into its own buffer, so the model lowers
    # vectorized. Verified: shared-fill reproducer + rdt2/bitvla whole-model.
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %cp = transform.structured.pad %mm pad_to_multiple_of [{MR}] {{padding_values = [{za}, {zb}, {zc}], padding_dimensions = [0], copy_back_op = "linalg.copy"}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %t1, %lmn:2 = transform.structured.tile_using_for %padded tile_sizes [{MR}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MR}, {NR}, 1] : !transform.any_op
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt1, %blmn:3 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NB}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %bm2 = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt2, %blk = transform.structured.tile_using_for %bm2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt2 vector_sizes [1, {MR}, {NB}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _accumulator_resident_v3_mrpad_pipeline(passes: list[str]) -> list[str]:
    """The v3 (.vf subset-hoist + A-scalarize) pipeline PLUS ``eliminate-empty-tensors`` before
    one-shot-bufferize — required by the M-pad copy-back.

    The M-pad schedule copies each padded matmul result back with ``linalg.copy`` (not the default
    ``materialize_in_destination``); the pair (linalg.copy copy-back + eliminate-empty-tensors) is what
    lets one-shot-bufferize give each padded matmul its OWN buffer instead of aliasing the CSE-shared
    zero-fill init across every matmul (the "cannot avoid RaW conflict" whole-model PipelineError).
    Reuses ``_packed_eliminate_empties`` (same insertion the packed feature uses). Only runs when this
    feature is enabled; baseline pipeline untouched."""
    from .grid_features import _packed_eliminate_empties

    return _packed_eliminate_empties(_accumulator_resident_v3_pipeline(passes))


def _accumulator_resident_v3_bmmpad_pipeline(passes: list[str]) -> list[str]:
    """Pad odd BMM extents before the parallel transform, then run v3 lowering.

    Padding inside an ``scf.forall`` leaves its temporary allocation inside an
    allocation scope that becomes multi-block during CFG lowering.  Running the
    dedicated pad entry before ``__transform_parallel_main`` keeps the allocation
    outside the worker region; the later main entry only tiles/vectorizes.
    """
    out = _accumulator_resident_v3_mrpad_pipeline(passes)
    anchor = next(
        i
        for i, p in enumerate(out)
        if "transform-interpreter{entry-point=__transform_main}" in p
    )
    parallel = next(
        (i for i, p in enumerate(out) if "entry-point=__transform_parallel_main" in p),
        anchor,
    )
    out.insert(parallel, "transform-interpreter{entry-point=__transform_bmm_pad}")
    return out


# Dynamic RVV register-block schedule variants.
def _accumulator_resident_v3_kblocked_pre_schedule(MR: int, NR: int, KC: int) -> str:
    """v3 pre-schedule with a REAL K-blocking tile (the ``KC`` lever, which the plain v3 recipe
    silently ignores — it tiles K by 1 and never uses KC, so every KC produced the same schedule).

    Why it matters (measured): the emitted inner loop advances B by the full row stride each K step
    (``addi a0,a0,512`` for N=128 f32), touching a NEW cache line every iteration and re-walking them
    for every (M,N) tile. Instruction count matches XNNPACK's kernel almost exactly, yet we run ~5x
    slower — i.e. we are memory-stalled, not instruction-bound. Blocking K by KC keeps a B panel
    resident across the M/N tiles so those lines are reused before eviction (classic GEMM cache
    blocking), which is the standard fix for exactly this access pattern.

    Structure: tile [MR, NR] (registers) -> tile K by KC (CACHE) -> tile K by 1 (register-resident
    accumulation) -> scoped-vectorize [MR, NR, 1]. Pure code generation; no hand ukernel."""
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [{MR}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mmk = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tkc, %lkc = transform.structured.tile_using_for %mmk tile_sizes [0, 0, {KC}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MR}, {NR}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _accumulator_resident_v3_unrolled_pre_schedule(MR: int, NR: int, KC: int) -> str:
    """v3 pre-schedule with M held as MR INDEPENDENT accumulators (the `unroll_m` axis).

    The plain v3 recipe tiles M by MR and vectorizes to a 2-D ``vector<MRxNR>``, so MR must be
    vectorization-friendly — measured on K1: MR 3/5/6/7 collapse to 193-279x off XNNPACK while MR=4
    is 5.0x. An expert micro-kernel instead keeps MR SEPARATE accumulator registers (M fully
    unrolled), which is shape-agnostic and is why XNNPACK can pick MR=7.

    Recipe: tile M by 1 (each row is its own ``vector<NR>`` accumulator) x N by NR, tile K by 1,
    scoped-vectorize the [1, NR, 1] body, then UNROLL the M loop by MR -> MR independent accumulators
    carried as separate iter_args. Pure code generation; no hand ukernel."""
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [1, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [1, {NR}, 1] : !transform.any_op
    transform.loop.unroll %lmn#0 {{ factor = {MR} }} : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


_VSCALE_LANES_PER_128B = 2


def scalable_lanes(NR: int) -> int:
    """The scalable multiplier ``k`` of ``vector<[k]xT>`` for a register block ``NR`` lanes wide at
    the RVV MINIMUM VLEN (128 bits). Dtype-independent — see :data:`_VSCALE_LANES_PER_128B`."""
    if NR % _VSCALE_LANES_PER_128B:
        raise ValueError(
            f"vl_strategy='dynamic' needs an even NR (NR counts lanes at the RVV minimum VLEN of "
            f"128 bits, and a scalable vector<[k]xT> holds 2k of them); got NR={NR}."
        )
    return NR // _VSCALE_LANES_PER_128B


def _accumulator_resident_v3_scalable_pre_schedule(MR: int, NR: int, KC: int) -> str:
    """v3 pre-schedule emitting a VL-AGNOSTIC (scalable) register block — the ``vl_strategy=dynamic``
    axis. Same recipe as :func:`_accumulator_resident_v3_pre_schedule` with the N tile made SCALABLE.

    Why this is the general fix. Our fixed-width codegen asks for ``vector<NRxf32>``, and the backend
    must size that register group for the worst VLEN the ``march`` string admits (``rv64gcv`` promises
    only the 128-bit minimum), so on a VLEN=256 part every value got DOUBLE the LMUL it needed and
    ``vl`` sat at half ``VLMAX``. Pinning ``_zvl256b`` fixes it for ONE board and miscompiles on a
    narrower one. A scalable ``vector<[k]xf32>`` has no such worst case: its LMUL is fixed by the type
    and its lane count is whatever the hardware reports, which is what the expert kernels do by
    calling ``__riscv_vsetvl_e32m4`` at run time. ``NR`` is reinterpreted as lanes at the MINIMUM
    VLEN (see :func:`scalable_lanes`), so ``NR`` still names the register-block width — it just names
    the LMUL rather than a lane count that only one board would honour.

    Two things the naive scalable schedule gets wrong, plus one PRECONDITION it inherits (all
    reproduced against LLVM/MLIR 23, see docs/design/vl_agnostic_codegen.md):

      * The N loop trip count is no longer a compile-time multiple of the tile, so ``vectorize``
        MASKS every transfer. Masked scalable transfers (a) block
        ``loop-invariant-subset-hoisting``, which is what makes the accumulator a register-resident
        ``scf.for`` iter_arg — without it the K loop round-trips C through memory, the exact defect
        v3 exists to remove — and (b) lower through a ``vector.transpose`` on
        ``vector<MRx1x[k]xf32>`` that has NO scalable lowering and survives to the LLVM edge
        (``error: Dialect 'vector' not found for custom op 'vector.transpose'``). THIS transpose,
        not the ``ub.poison`` the old note named, is the real residue of the "incomplete lowering".
        FIX: PEEL the N loop (``transform.loop.peel``) so the main loop's tile is exactly
        ``[k] * vscale`` wide, then vectorize it with ``assume_dynamic_dims_match_vec_sizes`` — the
        peel is what MAKES that assumption true. No masks, no transpose; the subset hoist fires
        exactly as in the fixed-width recipe.
      * The peeled N remainder keeps its ``linalg.matmul`` and falls through
        ``convert-linalg-to-loops`` to a scalar tail — correct for any N, zero-trip whenever the
        hardware VL divides N.
      * PRECONDITION ``MR | M`` (inherited, not introduced). ``assume_dynamic_dims_match_vec_sizes``
        asserts EVERY dynamic tile dim equals its vector size, INCLUDING M. The static ``MR`` M-tile
        has a partial last iteration when ``MR`` does not divide M, and the flag then makes the body
        write ``MR`` rows into a shorter tile (measured on the K1 at 130^3: ``malloc(): corrupted
        top size``; 128^3 and 100^3, where MR|M, are bit-exact). This is the SAME "MR must divide M"
        constraint the fixed 2-D ``vector<MRxNR>`` v3 recipe already carries (see
        expert_gap_attribution.md — small-M falls off a cliff). It is NOT fixed here by peeling M:
        ``transform.loop.peel`` FAILS on a statically-divisible loop, so an M peel would break the
        common MR|M case (M=128) to protect the MR∤M one. The M-tail is an orthogonal axis
        (pad/peel, shared with the fixed recipe); until it lands, MR∤M is fail-closed (the harness
        records not_run on the crash, never a false timing), not silently wrong.

    ``KC`` is carried for naming parity with the other v3 points; like the plain v3 recipe it does
    not tile the reduction (``k_block`` is the lever that does)."""
    k = scalable_lanes(NR)
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lm, %ln = transform.structured.tile_using_for %mm tile_sizes [{MR}, [{k}], 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %lnf = transform.cast %ln : !transform.any_op to !transform.op<"scf.for">
    %main, %tail = transform.loop.peel %lnf : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %mms = transform.structured.match ops{{["linalg.matmul"]}} in %main : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mms tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MR}, [{k}], 1] {{assume_dynamic_dims_match_vec_sizes}} : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""
