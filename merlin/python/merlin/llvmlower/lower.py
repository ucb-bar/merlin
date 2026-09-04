"""End-to-end whole-model lowering driver.

linalg-on-tensors text
  → Merlin xDSL passes (quant_ext → linalg, emit_c_interface)
  → upstream MLIR pipeline + translation (model2MLIR venv, torch-mlir wheel)
  → clang-23 codegen (x86 host .so / rv64gcv object).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .codegen import build_host_shared, compile_ll
from .passes_xdsl import preprocess_text
from .pipeline import lower_to_llvm_ir

# Registering here, and not in `impr_features` itself, keeps the epilogue-fusion stage's definition
# next to the pass list it edits while still making the NAME resolvable: this module is what every
# whole-model backend imports to lower, so by the time a feature set is normalized for a lowering the
# entry exists. Idempotent, so a second import is a no-op.
from .epilogue_fusion import ensure_registered as _register_epilogue_fusion

_register_epilogue_fusion()


@dataclass
class LowerResult:
    workdir: Path
    ll_path: Path
    stats: dict[str, Any] = field(default_factory=dict)
    host_so: Path | None = None
    riscv_obj: Path | None = None


def lower_model(mlir_text: str, workdir: str | Path,
                targets: tuple[str, ...] = ("host",), textual: bool = False,
                vectorize: bool = False, transform_schedule: str | None = None,
                hoist_static_allocs: bool = True, parallel: bool = False,
                features: "frozenset[str] | None" = None,
                parallel_harts: int | None = None) -> LowerResult:
    """Lower MLIR text end to end; emit per-target artifacts in ``workdir``.

    ``textual=True`` uses the pure-text preprocessing (no xDSL round-trip) —
    required for whole-model artifacts until xDSL prints rank-reducing
    extract_slice correctly.

    ``vectorize=True`` bakes native RVV (fixed-width vector ops) into the IR instead of
    relying on clang auto-vectorization — for the rv64gcv / Saturn-tile target.

    ``parallel_harts=N`` layers an outer OpenMP-parallel loop under that RVV schedule, so
    the object is BOTH vectorized and multicore (the multi-hart Saturn / Zephyr SMP path).
    """
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)

    if textual:
        from .passes_xdsl import preprocess_text_textual
        upstream_text, stats = preprocess_text_textual(mlir_text)
    else:
        upstream_text, stats = preprocess_text(mlir_text)
    (work / "model.upstream.mlir").write_text(upstream_text, encoding="utf-8")

    ll_text = lower_to_llvm_ir(upstream_text, workdir=work, vectorize=vectorize,
                               transform_schedule=transform_schedule,
                               hoist_static_allocs=hoist_static_allocs,
                               parallel=parallel, features=features,
                               parallel_harts=parallel_harts)
    ll_path = work / "model.ll"
    ll_path.write_text(ll_text, encoding="utf-8")

    result = LowerResult(workdir=work, ll_path=ll_path, stats=stats)
    if "host" in targets:
        result.host_so = build_host_shared(ll_path, work / "model_host.so")
    if "riscv" in targets:
        result.riscv_obj = compile_ll(ll_path, work / "model_rv64gcv.o", "riscv")
    return result


def lower_model_file(mlir_path: str | Path, workdir: str | Path,
                     targets: tuple[str, ...] = ("host",),
                     textual: bool = False, vectorize: bool = False,
                     transform_schedule: str | None = None,
                     hoist_static_allocs: bool = True,
                     parallel: bool = False,
                     features: "frozenset[str] | None" = None,
                     parallel_harts: int | None = None) -> LowerResult:
    return lower_model(Path(mlir_path).read_text(encoding="utf-8"), workdir, targets,
                       textual=textual, vectorize=vectorize,
                       transform_schedule=transform_schedule,
                       hoist_static_allocs=hoist_static_allocs,
                       parallel=parallel, features=features,
                       parallel_harts=parallel_harts)
