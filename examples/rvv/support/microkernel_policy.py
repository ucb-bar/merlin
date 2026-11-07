"""RVV Transform schedule rendering and microkernel knob realization."""

from __future__ import annotations

from typing import Any

from .blocking_cost import _rvv_best_block

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
        return f'      transform.apply_patterns.vector.lower_contraction lowering_strategy = "{contraction_strategy}"'
    return f"      transform.apply_patterns.vector.{name}"


def _rvv_microkernel_resolver(spec) -> list[str]:
    """RVV's realization of the TARGET-AGNOSTIC micro-kernel space (kernels.microkernel.MicrokernelSpec).

    Registered as the ``rvv`` resolver so the same knob space works for any target (each
    registers its own). Realizes by CODE GENERATION only — no hand ukernel; the intrinsic driver
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

    The whole-model ``accumulator_resident_wholemodel*`` features are named directly in
    ``compiler_features`` and never reach this function, so they carry the hygiene through
    ``ImprFeature.implies`` instead (``impr_features._tile_epilogue_hygiene``) -- same policy, other
    entry point.

    CORRECTION. This docstring previously said that family "has no self-copy to erase (measured on
    int8 128^3: 4,323,737 -> 4,325,025 ins, i.e. noise)". That measurement is real but was taken on
    the MR=1 member, and generalizing it to the family was wrong. Re-measured on spike at 64^3, both
    dtypes, bit-identical output on every arm: MR=1+erase is byte-identical to MR=1 (240,369 cyc int8
    / 231,313 f32, to the digit -- which is why the MR=1 reading looked like a family fact), while
    ``accumulator_resident_wholemodel_vf_mr4`` emits a per-tile ``@memrefCopy`` worth 187,520
    instructions and goes 240,369 -> 491,346 without the erase and -> 178,239 with it. The erase is a
    property of the recipe SHAPE (an output tiled and bufferized per tile, which is what MR>1 buys),
    exactly as the paragraph above says -- so the MR>1 members need it and the MR=1 member does not.

    ``hand_v0`` carries no ``microkernel`` knob block, so it never reaches this function and keeps its
    byte-identical control lowering.
    """
    _SELF_COPY = "erase_self_copy"

    return feats if _SELF_COPY in feats else [*feats, _SELF_COPY]


def _recipe(spec) -> list[str]:
    """The micro-kernel recipe proper (no hygiene) — one feature naming this point in the space."""
    from merlin.kernels.microkernel import VL_DYNAMIC, UnsupportedAxis

    from .backend import (
        ensure_v3_kblocked_microkernel,
        ensure_v3_microkernel,
        ensure_v3_scalable_microkernel,
        ensure_v3_unrolled_microkernel,
    )

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
            raise UnsupportedAxis(
                "rvv: vl_strategy='dynamic' does not compose with "
                "unroll_m/pack/k_block yet (each replaces the schedule); emit one "
                "composed recipe to combine them."
            )
        return [
            ensure_v3_scalable_microkernel(int(spec.MR), int(spec.NR), int(spec.KC))
        ]
    if spec.k_block:
        # REAL cache blocking of the reduction (k_block is the genuine reduction-blocking lever;
        # bare KC-tuning on the default recipe is INERT and PRUNED from proposal — see
        # kernels.microkernel.PRUNED_AXES. This path stays RESOLVABLE so a package/test may pin it).
        if spec.unroll_m or spec.pack:
            raise UnsupportedAxis(
                "rvv: k_block does not compose with unroll_m/pack yet (each "
                "replaces the schedule); emit one composed recipe to combine them."
            )
        return [
            ensure_v3_kblocked_microkernel(int(spec.MR), int(spec.NR), int(spec.KC))
        ]
    if spec.unroll_m:
        # M held as MR INDEPENDENT accumulators (tile M by 1 + unroll the M loop by MR) instead of a
        # 2-D vector<MRxNR> — shape-agnostic in MR, which is what lets an expert pick MR=7. MEASURED
        # structurally wrong (MR sequential K-loops, B-reuse=1, ~2.4x slower), so `unroll_m` is PRUNED
        # from proposal (kernels.microkernel.PRUNED_AXES). Kept RESOLVABLE for tests/pins only.
        if spec.pack:
            raise UnsupportedAxis(
                "rvv: unroll_m + pack are not composed yet (each replaces the "
                "schedule); emit one composed recipe before enabling both."
            )
        return [
            ensure_v3_unrolled_microkernel(int(spec.MR), int(spec.NR), int(spec.KC))
        ]
    if spec.pack:
        from merlin.llvmlower.feature_registry import ImprFeature

        from .backend import _FEATURES
        from .grid_features import vfmacc_packed_schedule

        nm = f"vfmacc_packed_{spec.MR}_{spec.NR}_{spec.KC}"
        if nm not in _FEATURES.known():
            _FEATURES.register(
                ImprFeature(
                    name=nm,
                    action_class="PASS",
                    description=f"operand-packed micro-kernel (MR={spec.MR}, NR={spec.NR}, KC={spec.KC})",
                    edit_schedule=(
                        lambda _t, _s=spec: vfmacc_packed_schedule(_s.MR, _s.NR, _s.KC)
                    ),
                    schedule_replace=True,
                )
            )
        return [nm]
    return [ensure_v3_microkernel(int(spec.MR), int(spec.NR), int(spec.KC))]


_RVV_BLOCKED_OPS: tuple[str, ...] = ("linalg.matmul", "linalg.batch_matmul")


def _rvv_microkernel_shape_policy(spec, shapes) -> list[str]:
    """RVV's SHAPE-AWARE realization: choose the register block per op class from the workload.

    Same space, same knobs — the difference is that the requested ``MR``/``NR`` are read as an upper
    BOUND rather than a pin, and each op class the schedule tiles separately (``linalg.matmul`` vs
    ``linalg.batch_matmul``) gets its own answer. That is what makes this a general capability rather
    than a per-model patch: no model name, no shape literal, and the SAME code derives
    ``accumulator_resident_wholemodel_vf``'s hand-picked clamps for openvla (matmul (1, 16) because
    gcd(M)=1) while leaving bitvla on the untouched champion block (4, 16) and moving small_llama to
    (4, 8) because gcd(N)=8 — each of those three verified by whole-model lowering.

    Falls back to the shape-BLIND recipe when the spec asks for an axis whose schedule this policy
    does not author (``vl_strategy``/``pack``/``k_block``/``unroll_m`` each REPLACE the schedule and
    have no per-op-class arm), so an unrealized axis stays an honest divergence, never a silent one.
    """
    from merlin.kernels.microkernel import VL_FIXED

    if spec.vl_strategy != VL_FIXED or spec.unroll_m or spec.pack or spec.k_block:
        return _rvv_microkernel_resolver(spec)
    from .backend import ensure_v3_perop_microkernel

    blocks: dict[str, tuple[int, int]] = {}
    for op in _RVV_BLOCKED_OPS:
        ext = [
            (s.parallel[-2], s.parallel[-1])
            for s in shapes
            if s.op == op and len(s.parallel) >= 2
        ]
        blocks[op] = _rvv_best_block(int(spec.MR), int(spec.NR), ext)
    mm, bmm = blocks["linalg.matmul"], blocks["linalg.batch_matmul"]
    if mm == bmm == (int(spec.MR), int(spec.NR)):
        return _rvv_microkernel_resolver(
            spec
        )  # nothing to adapt -> byte-identical realization
    return _with_hygiene(
        [ensure_v3_perop_microkernel(mm[0], mm[1], bmm[0], bmm[1], int(spec.KC))]
    )


def microkernel_features(
    mk: dict[str, Any], target: str = "rvv", shapes: Any = ()
) -> list[str]:
    """Resolve a ``microkernel`` knob block to ``target``'s realization (target-agnostic dispatch).

    ``shapes`` (a sequence of ``kernels.microkernel.ContractionShape``) opts into the SHAPE-AWARE
    resolution; with the default empty sequence the realization is byte-identical to before."""
    from merlin.kernels.microkernel import MicrokernelSpec, resolve_for_shapes

    return list(resolve_for_shapes(target, MicrokernelSpec.from_knobs(mk), shapes))


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
    patterns = "\n".join(
        _pattern_line(p, cstrat) for p in knobs.get("lowering_patterns", [])
    )
    return (
        "module attributes {transform.with_named_sequence} {\n"
        "  transform.named_sequence @__transform_main(%arg0: !transform.any_op "
        "{transform.readonly}) {\n"
        f"{blocks}"
        '    %f = transform.structured.match ops{["func.func"]} in %arg0 : '
        "(!transform.any_op) -> !transform.any_op\n"
        "    transform.apply_patterns to %f {\n"
        f"{patterns}\n"
        "    } : !transform.any_op\n"
        "    transform.yield\n"
        "  }\n"
        "}\n"
    )


def _register_rvv_microkernel_resolver() -> None:
    from merlin.kernels.microkernel import register_resolver, register_shape_policy

    register_resolver("rvv", _rvv_microkernel_resolver)
    register_shape_policy("rvv", _rvv_microkernel_shape_policy)
