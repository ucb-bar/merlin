"""Pre-bufferization subset hoisting and existing contraction lowering choices."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from merlin.compiler.optimizations.resident import _accumulator_resident_pre_schedule
from merlin.compiler.selection.features import ImprFeature

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


def _register_accumulator_resident_v2() -> list[str]:
    """Register `accumulator_resident_v2` (default tile) + a small tuning grid. Reuses the v1
    PRE-bufferize schedule (forms the contract, stops there); the v2 pipeline does the PRE-bufferize
    subset hoist + contract lowering that makes the accumulator a genuine vector iter_arg."""
    from merlin.compiler.optimizations.registry import register

    names: list[str] = []
    grid = [(4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16)]
    for MR, NR, KC in grid:
        if (MR, NR, KC) == (4, 16, 16):
            nm = "accumulator_resident_v2"
            desc = (
                "PRE-bufferize accumulator-resident micro-kernel (the genuine residency the v1 "
                "feature could not reach): tile [MR=4,NR=16], tile K by 1, scoped-vectorize -> "
                "vector.contract, then on the TENSOR form (before bufferize) run "
                "loop-invariant-subset-hoisting so the accumulator transfer pair becomes a "
                "vector<MRxNR> scf.for iter_arg (register-resident across K, NO per-K memref "
                "roundtrip), then lower contraction -> outerproduct -> vfmacc. Verified by "
                "objdump: the K-loop carries the accumulator as an llvm.array<MR x vector<NR>> "
                "loop value with NO accumulator load/store inside the loop (unlike v1's per-K "
                "vl4re8.v/vs4r.v). Bit-exact at 32/64/128 + a non-cube on spike. Residual: emits "
                "vfmacc.vv (A read as vector<MRx1> -> lane-broadcast) not the hand kernel's "
                "vfmacc.vf, so a small constant accumulator-spill from A-broadcast pressure "
                "remains; closes the residency structure + most of the instret gap but not the "
                "full hand ceiling. Default-off, baseline byte-identical."
            )
        else:
            nm = f"accum_resident_v2_{MR}_{NR}_{KC}"
            desc = (
                f"PRE-bufferize accumulator-resident tuning point (MR={MR}, NR={NR}, KC={KC}): "
                f"tensor-level subset-hoisted vector<MRxNR> K-loop iter_arg accumulator (no "
                f"per-K memref roundtrip). Default-off tuning-grid feature."
            )
        register(
            ImprFeature(
                name=nm,
                action_class="PASS",
                description=desc,
                edit_pipeline=_accumulator_resident_v2_pipeline,
                edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC: _accumulator_resident_pre_schedule(_MR, _NR, _KC)),
                schedule_replace=True,
            )
        )
        names.append(nm)
    return names


def _accumulator_resident_v3_pipeline(passes: list[str]) -> list[str]:
    """v2 pipeline (PRE-bufferize subset hoist + contract lower) PLUS the SCALARIZE_MARKER sentinel
    spliced immediately AFTER the contract-lowering transform-interpreter (still on tensors, before
    one-shot-bufferize). The two-stage runner splits the pipeline at the marker and runs the
    A-scalarization rewrite there. Only runs when a v3 feature is enabled; baseline untouched.

    The marker and the rewrite it schedules belong to the selected vector support provider."""
    out = _accumulator_resident_v2_pipeline(passes)
    # The v2 edit spliced: loop-invariant-subset-hoisting, transform-preload-library{...v2_lower},
    # transform-interpreter{entry-point=__transform_accum_v2_lower}, canonicalize, cse. Put the
    # marker right AFTER that interpreter's canonicalize,cse (contract is now vector.fma w/ f32
    # A-extracts; bufferize has not run).
    idx = None
    for i, p in enumerate(out):
        if p.startswith("transform-interpreter{entry-point=__transform_accum_v2_lower}"):
            idx = i
            break
    if idx is None:
        return out
    insert_at = idx + 1
    while insert_at < len(out) and out[insert_at] in ("canonicalize", "cse"):
        insert_at += 1
    from merlin.compiler.backend import provided

    marker = provided.provider(purpose="the accumulator micro-kernel split").accum_microkernel_marker()
    if marker not in out:
        out = out[:insert_at] + [marker] + out[insert_at:]
    return out
