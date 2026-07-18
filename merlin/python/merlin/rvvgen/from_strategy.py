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
      unroll_m / vl_strategy=dynamic -> NOT yet expressible; raised honestly so the beam keeps them as
                   OPEN divergences. Both are buildable in pure codegen and are the next capabilities:
                   unroll_m = hold M rows as independent accumulators (what lets an expert use MR=7;
                   our 2-D vector<MRxNR> formulation collapses at non-power-of-2 MR), and dynamic VL =
                   a vsetvli loop, which — where MLIR's scalable->RVV lowering is incomplete — can be
                   emitted through ``llvmlower.custom_isa`` (merlin.inline_asm -> llvm.inline_asm /
                   llvm.call_intrinsic), i.e. still code generation, no llvm-project fork.
    """
    from ..kernels.microkernel import VL_DYNAMIC, UnsupportedAxis
    from ..llvmlower.impr_features import ensure_v3_microkernel
    if spec.unroll_m:
        raise UnsupportedAxis(
            "rvv: unroll_m (M as independent accumulators) is not emitted yet — the v3 recipe forms a "
            "2-D vector<MRxNR>. Build it as a codegen capability; do not silently ignore the axis.")
    if spec.vl_strategy == VL_DYNAMIC:
        raise UnsupportedAxis(
            "rvv: vl_strategy='dynamic' (VL-agnostic vsetvli loop) is not emitted yet — MLIR's "
            "scalable-vector -> RVV lowering is incomplete (ub.poison survives). Realize it via "
            "llvmlower.custom_isa (merlin.inline_asm -> llvm.inline_asm) rather than ignoring it.")
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
