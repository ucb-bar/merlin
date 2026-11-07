"""Render a transform-dialect RVV schedule FROM knobs, and mint a versioned fork package.

This is the generator the tuning agent / beam-search use: given a parent package's knobs + a set
of overrides (a lever choice), it produces a new ``schedule.mlir`` and writes a lineage-stamped
fork (via :mod:`merlin.mining.fork`). The generator is faithfulness-tested: re-rendering the
``hand_v0`` knobs reproduces its verbatim schedule (so a fork only differs by the intended knob).

Knobs consumed:
  op_match:            [{op, tile:[...], vector:[...]}]   (match+tile+vectorize per contraction op)
  lowering_patterns:   [lower_contraction, lower_masked_transfers, lower_transpose, ...]
  contraction_strategy: None | "outerproduct" | "dot" | "matmulintrinsics" | "parallelarith"
                        -> appends `lowering_strategy = "<v>"` to lower_contraction (the lever that
                        recovers fused vfmacc: outerproduct -> vector.fma -> llvm.fmuladd -> vfmacc)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import RvvPackage


def _vector_backend():
    from ..llvmlower.backend_spi import load_vector_codegen

    return load_vector_codegen("rvv")


def _rvv_blocking_lowers(MR: int, NR: int, M: int, N: int) -> bool:
    """Compatibility entry point for the provider-owned legality policy."""
    return _vector_backend().blocking_cost_api()._rvv_blocking_lowers(MR, NR, M, N)


def _rvv_best_block(MR_cap: int, NR_cap: int, extents: list[tuple[int, int]]) -> tuple[int, int]:
    """Compatibility entry point for the provider-owned block search."""
    return _vector_backend().blocking_cost_api()._rvv_best_block(MR_cap, NR_cap, extents)


def microkernel_features(mk: dict[str, Any], target: str = "rvv", shapes: Any = ()) -> list[str]:
    """Resolve a microkernel knob through the selected target's implementation."""
    if target == "rvv":
        return _vector_backend().microkernel_features(mk, shapes=shapes)
    from ..kernels.microkernel import MicrokernelSpec, resolve_for_shapes

    return list(resolve_for_shapes(target, MicrokernelSpec.from_knobs(mk), shapes))


def render_schedule(knobs: dict[str, Any]) -> str:
    """Render a vector schedule through its provider, preserving the historical API."""
    return _vector_backend().render_schedule(knobs)


def mint_fork(
    parent: RvvPackage | str | Path,
    overrides: dict[str, Any],
    *,
    version: int,
    depth: int,
    timestamp: str,
    source_evidence: list[str],
    lever: str,
    target: str = "rvv",
    out_root: str | Path = "out/artifacts/targets",
    generated_by_agent: bool = False,
) -> Path:
    """Compatibility entrypoint for optional fork creation, not required for compilation."""
    try:
        from .fork import mint_fork as _mint_fork
    except ModuleNotFoundError as exc:
        if exc.name != f"{__package__}.fork":
            raise
        raise ModuleNotFoundError(
            "Fork creation requires the merlin-mining distribution; install packages/merlin-mining.",
            name=exc.name,
        ) from exc
    return _mint_fork(
        parent,
        overrides,
        version=version,
        depth=depth,
        timestamp=timestamp,
        source_evidence=source_evidence,
        lever=lever,
        target=target,
        out_root=out_root,
        generated_by_agent=generated_by_agent,
    )


def _register_rvv_microkernel_resolver() -> None:
    """Install the selected provider's RVV resolver for legacy direct callers."""
    _vector_backend().register_microkernel_resolver()


_register_rvv_microkernel_resolver()
