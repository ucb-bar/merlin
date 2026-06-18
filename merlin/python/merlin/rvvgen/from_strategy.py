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


def render_schedule(knobs: dict[str, Any]) -> str:
    """Render the transform-dialect schedule MLIR from knobs (verbatim-faithful for hand_v0)."""
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
              lever: str, target: str = "rvv", out_root: str | Path = "generated_targets",
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
