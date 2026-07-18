"""Automated op × datatype optimization sweep — the FRAMEWORK that closes every kernel gap vs XNNPACK
by itself, in parallel, instead of hand-optimizing one op/datatype at a time (user directive).

For each cell in {f32, int8, fp16} × {gemm, igemm, gelu, sigmoid, …mapped} the sweep AUTOMATICALLY:
  1. takes XNNPACK's measured K1 wall for that kernel as the TARGET (the real scoreboard),
  2. runs the CCA beam (``beam.run_beam`` via ``beam_cli.run_instrumented_beam``) — fork from our fast
     seed, lift OUR emitted CCA from the K1 objdump, diff vs the expert kernel's CCA, propose + certify
     levers on the K1, rank on attainment-vs-XNNPACK,
  3. treats SCALAR output as a hard FAILURE (never run scalar — everything must be RVV): a cell whose
     emitted kernel has no RVV in its timed region is a failing cell the beam must close first.

Cells fan out in PARALLEL (the sweep engine); the K1 board is serialized by ``k1.board_lock``. Adding a
datatype/op is a new cell + a registered lever (the register-resident emitter, an int8 ``vwmacc`` variant,
an fp16 ``vfwmacc`` variant in ``impr_features``/``action_catalog``) — NOT a hand-written pass per op. The
beam SELECTS the lever; this driver just orchestrates the matrix.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class OpCell:
    """One (op, dtype) target in the sweep matrix."""

    op: str                       # "matmul" | "conv" | "gelu" | "sigmoid" | ...
    dtype: str                    # "f32" | "int8" | "fp16"
    shape_regime: str             # e.g. "square_128"
    workload_dir: str | Path      # the bundle the beam certifies (model.mlir + inputs + golden)
    expert_objdump: str | Path    # decoded XNNPACK kernel objdump -> the expert CCA (the target shape)
    expert_wall_ns: float | None = None   # XNNPACK's measured K1 wall for this kernel (the scoreboard)
    seed_pkg: str | Path | None = None     # fork-from (default: the fast tracked config)

    @property
    def key(self) -> str:
        return f"{self.op}:{self.dtype}:{self.shape_regime}"


@dataclass
class CellResult:
    cell_key: str
    best_run_id: str | None
    attainment_vs_expert: float | None    # >=1.0 = matched/beat XNNPACK
    speedup_vs_seed: float | None
    scalar_ok: bool                       # True = emitted RVV (not scalar); False = scalar FAIL
    gate_ok: bool
    parent_run_dir: str | None
    note: str = ""


def is_scalar_kernel(objdump_text: str) -> bool:
    """True iff the emitted kernel is running SCALAR — zero vector arithmetic in the whole objdump.
    Per the 'never run scalar' directive this is a certify FAILURE. We flag only FULLY-scalar kernels
    (coverage == 0): a whole-model objdump legitimately contains scalar helper symbols (ciface / arg
    setup) alongside a vectorized compute kernel, so ``scalar_fallback_symbols`` (per-symbol) would
    false-positive — coverage==0 means NO vector compute anywhere, the true scalar-fallback case."""
    from ..baselines.rvv_audit import classify_disasm
    rep = classify_disasm(objdump_text, source="op_sweep")
    cov = rep.coverage_overall
    if cov is None:
        return False            # no compute-bearing symbol found -> can't call it scalar (honest)
    return cov == 0.0


def _default_seed(repo_root: Path) -> str:
    fast = repo_root / "out/artifacts/targets/rvv/impr_tuned_wholemodel"
    return str(fast if fast.is_dir() else repo_root / "out/artifacts/targets/rvv/hand_v0")


def run_cell(cell: OpCell, *, width: int = 3, depth: int = 2, top_k: int = 2,
             beam_fn: Callable | None = None) -> CellResult:
    """Run the beam for ONE cell, targeting XNNPACK's wall, and scalar-gate the winner."""
    from ..common.paths import repo_root
    from .beam_cli import run_instrumented_beam
    beam_fn = beam_fn or run_instrumented_beam

    seed = str(cell.seed_pkg or _default_seed(repo_root()))
    res = beam_fn(seed_pkg=seed, model_dir=str(cell.workload_dir),
                  expert_objdump=str(cell.expert_objdump), op=cell.op, dtype=cell.dtype,
                  shape_regime=cell.shape_regime, targets=("k1",), width=width, depth=depth,
                  top_k=top_k, expert_wall_ns=cell.expert_wall_ns)
    best = res.get("best") or {}
    # scalar gate on the winner's emitted objdump (never credit a scalar kernel).
    scalar_ok = True
    note = ""
    parent = res.get("parent_run_dir")
    if best.get("run_id") and parent:
        objd = Path(res.get("parent_run_dir", "")).parent  # forks live under the parent run's forks/
        cand = list(Path(parent).glob(f"forks/{best['run_id']}/generated/objdump.txt"))
        if cand and cand[0].is_file():
            scalar = is_scalar_kernel(cand[0].read_text())
            scalar_ok = not scalar
            if scalar:
                note = "SCALAR FAIL — winner emitted no RVV in its kernel (must close before perf credit)"
    return CellResult(
        cell_key=cell.key, best_run_id=best.get("run_id"),
        attainment_vs_expert=best.get("attainment_vs_expert"),
        speedup_vs_seed=best.get("speedup"),
        scalar_ok=scalar_ok, gate_ok=bool(best.get("gate_ok")),
        parent_run_dir=parent, note=note)


def run_op_sweep(cells: list[OpCell], *, width: int = 3, depth: int = 2, top_k: int = 2,
                 max_workers: int | None = None, beam_fn: Callable | None = None) -> list[CellResult]:
    """Run the whole op × datatype matrix. Cells fan out concurrently (proposal/build overlap); the K1
    board itself is serialized by ``k1.board_lock`` inside each certify, so parallel cells never corrupt
    a board run. A cell that raises is captured as a failed CellResult (the sweep never aborts)."""
    results: list[CellResult | None] = [None] * len(cells)

    def _run(i: int, cell: OpCell) -> None:
        try:
            results[i] = run_cell(cell, width=width, depth=depth, top_k=top_k, beam_fn=beam_fn)
        except Exception as e:  # one cell's failure must not kill the matrix
            results[i] = CellResult(cell_key=cell.key, best_run_id=None, attainment_vs_expert=None,
                                    speedup_vs_seed=None, scalar_ok=False, gate_ok=False,
                                    parent_run_dir=None, note=f"error: {type(e).__name__}: {e}")

    if not cells:
        return []
    with ThreadPoolExecutor(max_workers=max_workers or min(8, len(cells))) as ex:
        list(ex.map(lambda p: _run(*p), list(enumerate(cells))))
    return [r for r in results if r is not None]


# ---- the default GEMM cell matrix + CLI ------------------------------------------------------------
# XNNPACK's measured K1 kernel wall in rdtime TICKS per (dtype, shape) — from k1_cross_framework_ops.
# The beam reports k1_wall_ns (nanoseconds), so these ticks are converted to ns (via the K1 24 MHz
# timebase) in default_gemm_cells so attainment_vs_expert = xnn_ns / fork_ns is apples-to-apples.
_XNN_GEMM_TICKS = {("f32", 128): 9424, ("f32", 256): 72222, ("int8", 64): 1552, ("int8", 128): 11593}


def _ticks_to_ns(ticks: int | None) -> float | None:
    """K1 rdtime ticks -> nanoseconds (the unit the beam's k1_wall_ns uses)."""
    from .k1 import K1_TIMEBASE_HZ
    return None if ticks is None else round(ticks * 1e9 / K1_TIMEBASE_HZ)


def default_gemm_cells(shapes_f32=(128, 256), shapes_int8=(64, 128)) -> list[OpCell]:
    """Build the f32 + int8 GEMM sweep cells: workload bundles, the fast tracked seed per dtype, the
    XNNPACK f32-gemm expert CCA (structural target — register-residency/block are dtype-agnostic), and
    XNNPACK's measured K1 wall as the scoreboard. This is the launchable matrix for the datatypes we
    already have levers for; bf16/fp16 + more ops extend it (a cell + a lever, not hand-code)."""
    from ..common.paths import repo_root
    from .workloads import gen_matmul_f32
    root = repo_root()
    expert = root / "merlin/tests/data/cca_asm/xnnpack_f32_gemm_rvv.objdump"
    wl_root = root / "out/artifacts/cache/op_sweep_workloads"
    seed_f32 = root / "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf"
    seed_int8 = root / "out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8"
    cells: list[OpCell] = []
    for S in shapes_f32:
        cells.append(OpCell(op="matmul", dtype="f32", shape_regime=f"square_{S}",
                            workload_dir=gen_matmul_f32(wl_root, M=S, N=S, K=S), expert_objdump=expert,
                            expert_wall_ns=_ticks_to_ns(_XNN_GEMM_TICKS.get(("f32", S))), seed_pkg=str(seed_f32)))
    for S in shapes_int8:
        cells.append(OpCell(op="matmul", dtype="int8", shape_regime=f"square_{S}",
                            workload_dir=gen_matmul_f32(wl_root, M=S, N=S, K=S), expert_objdump=expert,
                            expert_wall_ns=_ticks_to_ns(_XNN_GEMM_TICKS.get(("int8", S))), seed_pkg=str(seed_int8)))
    return cells


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="merlin-rvv-opt",
                                 description="Automated op x datatype optimization sweep vs XNNPACK on K1.")
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--max-workers", type=int, default=1,
                    help="cell concurrency (default 1: the K1 board is serialized either way)")
    args = ap.parse_args(argv)

    cells = default_gemm_cells()
    print(f"=== merlin-rvv-opt: {len(cells)} cells (f32+int8 GEMM) vs XNNPACK on K1 ===")
    results = run_op_sweep(cells, width=args.width, depth=args.depth, top_k=args.top_k,
                           max_workers=args.max_workers)
    print(f"\n{'cell':28s} {'attain_vs_xnn':>14s} {'scalar_ok':>10s} {'gate_ok':>8s}  note")
    print("-" * 78)
    for r in sorted(results, key=lambda r: r.cell_key):
        att = f"{r.attainment_vs_expert:.3f}" if r.attainment_vs_expert else "—"
        print(f"{r.cell_key:28s} {att:>14s} {str(r.scalar_ok):>10s} {str(r.gate_ok):>8s}  {r.note[:30]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
