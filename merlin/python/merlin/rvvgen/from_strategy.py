"""Render a transform-dialect RVV schedule FROM knobs, and mint a versioned fork package.

This is the generator the tuning agent / beam-search use: given a parent package's knobs + a set
of overrides (a lever choice), it produces a new ``schedule.mlir`` and writes a lineage-stamped
fork (via :mod:`merlin.rvvgen.fork`). The generator is faithfulness-tested: re-rendering the
``hand_v0`` knobs reproduces its verbatim schedule (so a fork only differs by the intended knob).

Knobs consumed:
  op_match:            [{op, tile:[...], vector:[...]}]   (match+tile+vectorize per contraction op)
  lowering_patterns:   [lower_contraction, lower_masked_transfers, lower_transpose, ...]
  contraction_strategy: None | "outerproduct" | "dot" | "matmulintrinsics" | "parallelarith"
                        -> appends `lowering_strategy = "<v>"` to lower_contraction (the lever that
                        recovers fused vfmacc: outerproduct -> vector.fma -> llvm.fmuladd -> vfmacc)
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .fork import mint_run_id, write_fork
from .registry import RvvPackage, load_rvv_package

# Reproduce hand_v0's semantic SSA names so re-rendering its knobs is byte-identical.
#
# TARGET-PLUGIN NOTE: `_VARS` and `render_schedule` below are the RVV RENDERER — they emit the
# RVV transform-dialect schedule (op_match tile/vectorize + vector.* lowering patterns). They are
# intentionally RVV-specific: a different target reusing the beam engine (rvvgen.beam) supplies its
# OWN render/generator (a `render_schedule(knobs) -> str` and a `mint_fork(...) -> Path`) rather
# than these. See rvvgen/TARGET_PLUGIN.md for the full target-plugin contract.
_VARS = {
    "linalg.matmul": ("mm", "t", "l"),
    "linalg.batch_matmul": ("bm", "bt", "bl"),
}


def _vars_for(op: str, i: int) -> tuple[str, str, str]:
    return _VARS.get(op, (f"m{i}", f"t{i}", f"l{i}"))


def _ints(xs) -> str:
    return "[" + ", ".join(str(int(x)) for x in xs) + "]"


def _op_block(op: str, tile: list[int], vector: list[int], i: int) -> str:
    mv, tv, lv = _vars_for(op, i)
    nloops = sum(1 for t in tile if int(t) != 0)
    res = ", ".join(["!transform.any_op"] * (nloops + 1))
    return (
        f'    %{mv} = transform.structured.match ops{{["{op}"]}} in %arg0 : '
        f"(!transform.any_op) -> !transform.any_op\n"
        f"    %{tv}, %{lv}:{nloops} = transform.structured.tile_using_for %{mv} "
        f"tile_sizes {_ints(tile)} : (!transform.any_op) -> ({res})\n"
        f"    transform.structured.vectorize %{tv} vector_sizes {_ints(vector)} : !transform.any_op\n"
    )


def _pattern_line(name: str, contraction_strategy: str | None) -> str:
    if name == "lower_contraction" and contraction_strategy:
        return (f"      transform.apply_patterns.vector.lower_contraction "
                f'lowering_strategy = "{contraction_strategy}"')
    return f"      transform.apply_patterns.vector.{name}"


def _rvv_microkernel_resolver(spec) -> list[str]:
    """RVV's realization of the TARGET-AGNOSTIC micro-kernel space (kernels.microkernel.MicrokernelSpec).

    Registered as the ``rvv`` resolver so the same knob space works for any target (Gemmini, Saturn,
    … register their own). Realizes by CODE GENERATION only — no hand ukernel; the intrinsic driver
    stays a ceiling REFERENCE.

      MR/NR/KC  -> a v3 accumulator-resident tuning point (register-resident scf.for iter_arg
                   accumulator + A-scalarized vfmacc.vf), registered on demand for ANY shape so the
                   register block is continuously beam-tunable. Chosen over the older generators on
                   measurement: v3 @MR=4 is ~4-5x off XNNPACK vs plain-tiled 18x / packed 46x.
      pack      -> the operand-packing recipe (unit-stride panels + B pre-transpose).
      vl_strategy=dynamic -> the VL-AGNOSTIC (scalable) register block: the N tile becomes a
                   ``vector<[k]xT>``, so the emitted loop sizes to the vector length the HARDWARE
                   reports (``vsetvli`` against ``VLMAX``) rather than to a compile-time width the
                   backend must widen for the worst-case VLEN. This is what makes the ``_zvl`` march
                   pin unnecessary — and it emits through the ORDINARY MLIR scalable lowering; the
                   ``llvmlower.custom_isa`` inline-asm hatch was NOT needed. Requires the peel +
                   ``assume_dynamic_dims_match_vec_sizes`` pairing (see
                   ``impr_features.ensure_v3_scalable_microkernel``).
      unroll_m  -> hold M rows as independent accumulators. Expressible, but MEASURED structurally
                   wrong on silicon (B reuse stays 1 per K step, 2x the instructions per lane-FMA at
                   every MR) — see docs/design/expert_gap_attribution.md. PRUNED from beam proposal
                   (kernels.microkernel.PRUNED_AXES); still resolvable here for tests/pins.
      KC        -> INERT on this recipe: K is tiled by 1 regardless of KC, so tuning KC changes no
                   emitted instruction (measured flat). PRUNED from proposal; the genuine
                   reduction-blocking lever is ``k_block``.

    Every realization also carries the recipe's lowering HYGIENE (see :func:`_with_hygiene`) — the
    passes the emitted code needs to be worth measuring at all, independent of dtype.
    """
    return _with_hygiene(_recipe(spec))


def _with_hygiene(feats: list[str]) -> list[str]:
    """Append the lowering hygiene every micro-kernel realization needs to be worth measuring.

    Right now that is ``erase_self_copy``. Every recipe below tiles the output and bufferizes per
    tile, and bufferization leaves a `memref.copy %x, %x` in each tile epilogue that survives as an
    opaque rank-generic ``@memrefCopy`` runtime call (see llvmlower/selfcopy.py). It is not a schedule
    decision and not a dtype decision -- it is a property of the recipe SHAPE -- so it belongs here,
    where every dtype and every micro-kernel point inherits it at once, rather than being re-listed
    per package. Erasing is unconditionally safe (same SSA value => same base/offsets/region), and it
    is a no-op on lowerings that have no self-copy, so appending it can only help or do nothing.

    MEASURED on the live K1, kernel region, correctness-gated (min of 3):
        f32  128^3          1,710,650 -> 475,899 ins   41,195 -> 21,882 ticks   1.88x
        int8  64^3            732,447 -> 425,039 ins   14,721 -> 10,301 ticks   1.43x
        int8 128^3          4,230,288 -> 3,002,346 ins  85,760 -> 69,428 ticks  1.24x
        int8 256^3         27,423,003 -> 22,519,524 ins 567,986 -> 503,434 ticks 1.13x

    NOT appended to the whole-model ``accumulator_resident_wholemodel*`` features: those are named
    directly in ``compiler_features``, not resolved through this space, and their lowering has no
    self-copy to erase (measured on int8 128^3: 4,323,737 -> 4,325,025 ins, i.e. noise). The erase is
    a property of the v3 recipe, NOT of int8.

    ``hand_v0`` carries no ``microkernel`` knob block, so it never reaches this function and keeps its
    byte-identical control lowering.
    """
    from ..llvmlower.selfcopy import FEATURE as _SELF_COPY
    return feats if _SELF_COPY in feats else [*feats, _SELF_COPY]


def _recipe(spec) -> list[str]:
    """The micro-kernel recipe proper (no hygiene) — one feature naming this point in the space."""
    from ..kernels.microkernel import VL_DYNAMIC, UnsupportedAxis
    from ..llvmlower.impr_features import (ensure_v3_kblocked_microkernel, ensure_v3_microkernel,
                                            ensure_v3_scalable_microkernel,
                                            ensure_v3_unrolled_microkernel)
    if spec.vl_strategy == VL_DYNAMIC:
        # VL-AGNOSTIC: a scalable N register block, so the emitted loop sizes to the vector length
        # the hardware reports (vsetvli against VLMAX) instead of a compile-time width the backend
        # must widen for the worst-case VLEN. Realized through the ORDINARY MLIR scalable-vector
        # lowering -- no inline-asm escape hatch was needed. It only works with the peel +
        # assume_dynamic_dims_match_vec_sizes pairing; see ensure_v3_scalable_microkernel for what
        # the naive scalable schedule gets wrong (masked scalable transfers block the accumulator
        # hoist AND leave an unlowerable scalable vector.transpose).
        #
        # Checked FIRST, before the recipe-replacing axes below: each of those returns a schedule of
        # its own, so ordering it after them would silently emit FIXED-width code for a spec that
        # asked for dynamic VL -- the "credited a change that never happened" failure the
        # UnsupportedAxis contract exists to prevent.
        if spec.unroll_m or spec.pack or spec.k_block:
            raise UnsupportedAxis("rvv: vl_strategy='dynamic' does not compose with "
                                  "unroll_m/pack/k_block yet (each replaces the schedule); emit one "
                                  "composed recipe to combine them.")
        return [ensure_v3_scalable_microkernel(int(spec.MR), int(spec.NR), int(spec.KC))]
    if spec.k_block:
        # REAL cache blocking of the reduction (k_block is the genuine reduction-blocking lever;
        # bare KC-tuning on the default recipe is INERT and PRUNED from proposal — see
        # kernels.microkernel.PRUNED_AXES. This path stays RESOLVABLE so a package/test may pin it).
        if spec.unroll_m or spec.pack:
            raise UnsupportedAxis("rvv: k_block does not compose with unroll_m/pack yet (each "
                                  "replaces the schedule); emit one composed recipe to combine them.")
        return [ensure_v3_kblocked_microkernel(int(spec.MR), int(spec.NR), int(spec.KC))]
    if spec.unroll_m:
        # M held as MR INDEPENDENT accumulators (tile M by 1 + unroll the M loop by MR) instead of a
        # 2-D vector<MRxNR> — shape-agnostic in MR, which is what lets an expert pick MR=7. MEASURED
        # structurally wrong (MR sequential K-loops, B-reuse=1, ~2.4x slower), so `unroll_m` is PRUNED
        # from proposal (kernels.microkernel.PRUNED_AXES). Kept RESOLVABLE for tests/pins only.
        if spec.pack:
            raise UnsupportedAxis("rvv: unroll_m + pack are not composed yet (each replaces the "
                                  "schedule); emit one composed recipe before enabling both.")
        return [ensure_v3_unrolled_microkernel(int(spec.MR), int(spec.NR), int(spec.KC))]
    if spec.pack:
        from ..llvmlower.impr_features import known, register, ImprFeature, vfmacc_packed_schedule
        nm = f"vfmacc_packed_{spec.MR}_{spec.NR}_{spec.KC}"
        if nm not in known():
            register(ImprFeature(
                name=nm, action_class="PASS",
                description=f"operand-packed micro-kernel (MR={spec.MR}, NR={spec.NR}, KC={spec.KC})",
                edit_schedule=(lambda _t, _s=spec: vfmacc_packed_schedule(_s.MR, _s.NR, _s.KC)),
                schedule_replace=True))
        return [nm]
    return [ensure_v3_microkernel(int(spec.MR), int(spec.NR), int(spec.KC))]


def microkernel_features(mk: dict[str, Any], target: str = "rvv") -> list[str]:
    """Resolve a ``microkernel`` knob block to ``target``'s realization (target-agnostic dispatch)."""
    from ..kernels.microkernel import MicrokernelSpec, resolve
    return list(resolve(target, MicrokernelSpec.from_knobs(mk)))


def render_schedule(knobs: dict[str, Any]) -> str:
    """Render the transform-dialect schedule MLIR from knobs (verbatim-faithful for hand_v0).

    NOTE: a ``microkernel`` knob block does NOT render schedule text here — it resolves through the
    TARGET-AGNOSTIC space (see :func:`microkernel_features` / ``kernels.microkernel``) to this target's
    realization, whose ``edit_schedule`` replaces the schedule and whose ``edit_pipeline`` adds the
    residency/scalarize stage at build time."""
    blocks = "".join(
        _op_block(m["op"], m["tile"], m["vector"], i)
        for i, m in enumerate(knobs.get("op_match", []))
    )
    cstrat = knobs.get("contraction_strategy")
    patterns = "\n".join(_pattern_line(p, cstrat) for p in knobs.get("lowering_patterns", []))
    return (
        "module attributes {transform.with_named_sequence} {\n"
        "  transform.named_sequence @__transform_main(%arg0: !transform.any_op "
        "{transform.readonly}) {\n"
        f"{blocks}"
        "    %f = transform.structured.match ops{[\"func.func\"]} in %arg0 : "
        "(!transform.any_op) -> !transform.any_op\n"
        "    transform.apply_patterns to %f {\n"
        f"{patterns}\n"
        "    } : !transform.any_op\n"
        "    transform.yield\n"
        "  }\n"
        "}\n"
    )


def mint_fork(parent: "RvvPackage | str | Path", overrides: dict[str, Any], *,
              version: int, depth: int, timestamp: str, source_evidence: list[str],
              lever: str, target: str = "rvv", out_root: str | Path = "out/artifacts/targets",
              generated_by_agent: bool = False) -> Path:
    """Mint a versioned fork from ``parent`` with ``overrides`` applied to its knobs.

    ``lever`` in {knob, lowering_pattern, llvm_requirement}; ``source_evidence`` is the mined
    policy / kernel ids justifying the change. Returns the new package dir.
    """
    if not isinstance(parent, RvvPackage):
        parent = load_rvv_package(parent)
    knobs = copy.deepcopy(parent.knobs)
    knobs.update(overrides)
    schedule_text = render_schedule(knobs)
    run_id = mint_run_id(target, version, depth, timestamp)
    lineage = {
        "parent_run_id": parent.run_id,
        "version": version,
        "depth": depth,
        "source_evidence": source_evidence,
        "lever": lever,
        "generated_by_agent": generated_by_agent,
    }
    return write_fork(out_root, target, run_id, schedule_text=schedule_text,
                      knobs=knobs, lineage=lineage)


# Register RVV's realization of the target-agnostic micro-kernel space at import. Other targets
# (gemmini, saturn_vec, muon, …) register their own resolver the same way, so the SAME knob block
# expresses expert-kernel granularity for any compilation target.
def _register_rvv_microkernel_resolver() -> None:
    from ..kernels.microkernel import register_resolver
    register_resolver("rvv", _rvv_microkernel_resolver)


_register_rvv_microkernel_resolver()
