"""Calibrate the analytical cost model against the npu_model cycle-level simulator.

Minimal, report-only, analytical-first. The npu_model ships fixed-size single-pass matmul /
fused-matmul+bias programs; we run a few through the cycle-level simulator and compare the
simulated cycles to the analytical model's *single-invocation* (H=1) prediction, fed cost-model
params derived from the same ``HardwareConfig`` (``cost_model_from_npu``). This validates the
per-invocation building block; the multi-step reuse extrapolation that I2/I3 exploit stays
analytical. Agreement is *reported*, not retro-fitted, to preserve architecture-independence.

The simulator lives outside the merlin package (``tmp/dse/npu_model``) and needs torch, so the
hook is import-guarded — callers should ``skipif`` when ``available()`` is False.
"""
from __future__ import annotations

import sys
from pathlib import Path

from merlin.common import paths
from merlin.common.yaml import write_yaml
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region
from merlin.dse.cost_model import evaluate_cost
from merlin.dse.hardware_space import cost_model_from_npu
from merlin.dse.variants import contract_plans

# (program class name, M, K, N, epilogue) — fixed-size shippable npu programs.
DEFAULT_POINTS = [
    ("ParameterizedMatmul32x32x32Program", 32, 32, 32, False),
    ("ParameterizedMatmul64x32x96Program", 64, 32, 96, False),
    ("ParameterizedFusedMatmulBias32x32Program", 32, 32, 32, True),
]


def npu_root() -> Path:
    return paths.work_dir() / "tmp" / "dse" / "npu_model"


def available() -> bool:
    """True iff the npu_model simulator (and torch) can be imported."""
    try:
        _imports()
        return True
    except Exception:
        return False


def _imports():
    root = str(npu_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from npu_model.configs.hardware.default import DefaultHardwareConfig  # noqa: F401
    from npu_model.configs import programs  # noqa: F401
    from tests.helpers import run_simulation  # noqa: F401
    return DefaultHardwareConfig, programs, run_simulation


def _analytical_cycles(M: int, K: int, N: int, epilogue: bool, cm: dict) -> float:
    """Analytical single-invocation (H=1) opaque-call cost for an M×K×N matmul."""
    rpv = compute_rpv(build_region(H=1, reuse_count=1, dtype="i8", epilogue=epilogue,
                                   K=K, M=M, N=N))
    plan = contract_plans(rpv)["I0"]
    return evaluate_cost(rpv, plan, cm)["cycles"]


def calibrate(points=None, max_cycles: int = 200000,
              out_dir: str | Path | None = None) -> dict:
    """Run the calibration points and return a report (also written to calibration.yaml)."""
    DefaultHardwareConfig, programs, run_simulation = _imports()
    hw = DefaultHardwareConfig()
    cm = cost_model_from_npu(hw)

    rows = []
    for name, M, K, N, epilogue in (points or DEFAULT_POINTS):
        program_cls = getattr(programs, name, None)
        if program_cls is None:
            continue
        sim = run_simulation(program_cls(), hw, max_cycles=max_cycles, verbose=False)
        try:
            simulated = int(sim.get_stats().cycles)
        finally:
            close = getattr(sim, "close", None)
            if callable(close):
                close()
        analytical = _analytical_cycles(M, K, N, epilogue, cm)
        rows.append({
            "point": name,
            "M": M, "K": K, "N": N, "epilogue": epilogue,
            "analytical_cycles": round(analytical, 1),
            "simulated_cycles": simulated,
            "ratio": round(analytical / simulated, 3) if simulated else None,
        })

    ratios = [r["ratio"] for r in rows if r["ratio"]]
    report = {
        "cost_model": cm,
        "rows": rows,
        "calibration_factor": round(sorted(ratios)[len(ratios) // 2], 3) if ratios else None,
        "note": "report-only agreement check; analytical model is not retro-fitted.",
    }
    out = Path(out_dir) if out_dir else paths.artifacts_dir() / "dse" / "vla_action_chunk_decode"
    out.mkdir(parents=True, exist_ok=True)
    write_yaml(out / "calibration.yaml", report, header="npu_model calibration (report-only)")
    return report
