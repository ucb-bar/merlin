"""The M1 phase-transition experiment (the headline artifact).

Sweeps the synthetic VLA action-chunk decode region over horizon/reuse/dtype/epilogue and,
for each point, costs the I0–I3 contracts. The result shows the *best interface changing
category* as reuse grows (I0 opaque -> I2 resident -> I3 resident+commit) — the thesis claim.

Always writes a deterministic ``phase_transition.csv`` (headless-safe). Writes a
``phase_transition.png`` only when matplotlib is available. Also emits exploitability reports
for the residency and accumulator-commit features over the reuse sweep.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from merlin.common import paths
from merlin.common.yaml import write_yaml
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.synthesize import FEATURE_ACCUMULATOR, FEATURE_RESIDENT
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region, sweep_axes
from merlin.dse.cost_model import evaluate_cost
from merlin.dse.exploitability import compute_exploitability, row_for
from merlin.dse.hardware_space import default_cost_model
from merlin.dse.variants import contract_plans

CSV_COLUMNS = ["H", "reuse_count", "dtype", "epilogue", "contract", "cycles", "energy", "best"]


def phase_transition(axes: dict | None = None, cost_model: dict | None = None,
                     out_dir: str | Path | None = None,
                     workload: str = "vla_action_chunk_decode") -> dict:
    """Run the sweep and return ``{rows, exploitability, csv}``; write artifacts if out_dir.

    Each point sets ``reuse_count == H`` (the action loop reuses the same W every step). The
    ``reuse_count`` axis from ``sweep_axes`` is used separately for the exploitability report.
    """
    ax = axes or sweep_axes()
    cm = cost_model or default_cost_model()

    rows: list[dict] = []
    for dtype in ax["dtype"]:
        for epilogue in ax["epilogue"]:
            for H in ax["H"]:
                rpv = compute_rpv(build_region(H=H, reuse_count=H, dtype=dtype,
                                               epilogue=epilogue, K=256, M=1, N=256))
                lat = {c: evaluate_cost(rpv, p, cm)
                       for c, p in contract_plans(rpv).items()}
                best = min(lat, key=lambda c: lat[c]["cycles"])
                for contract, cost in lat.items():
                    rows.append({
                        "H": H, "reuse_count": H, "dtype": dtype,
                        "epilogue": epilogue, "contract": contract,
                        "cycles": round(cost["cycles"], 2), "energy": cost["energy"],
                        "best": contract == best,
                    })

    # Exploitability over the reuse sweep (i8, epilogue on), per feature.
    expl = {}
    for feature in (FEATURE_RESIDENT, FEATURE_ACCUMULATOR):
        erows = []
        for reuse in ax["reuse_count"]:
            rpv = compute_rpv(build_region(H=reuse, reuse_count=reuse, dtype="i8",
                                           epilogue=True, K=256, M=1, N=256))
            erows.append(row_for(rpv, feature, reuse, cm))
        expl[feature] = compute_exploitability(workload, feature, "reuse_count", erows)

    csv_text = _to_csv(rows)
    result = {"rows": rows, "exploitability": expl, "csv": csv_text}

    if out_dir is None:
        out_dir = paths.repo_root() / "artifacts" / "dse" / workload
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase_transition.csv").write_text(csv_text, encoding="utf-8")
    for feature, report in expl.items():
        write_yaml(out / f"exploitability_{feature}.yaml", report,
                   header=f"exploitability_report: {workload} / {feature}")
    _maybe_plot(rows, out / "phase_transition.png")
    return result


def _to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in CSV_COLUMNS})
    return buf.getvalue()


def _maybe_plot(rows: list[dict], path: Path) -> bool:
    """Plot latency vs H for I0–I3 (i8, epilogue on). No-op if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    sel = [r for r in rows if r["dtype"] == "i8" and r["epilogue"] is True]
    contracts = ["I0", "I1", "I2", "I3"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in contracts:
        pts = sorted([(r["H"], r["cycles"]) for r in sel if r["contract"] == c])
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=c)
    ax.set_xlabel("action horizon H (= weight reuse)")
    ax.set_ylabel("estimated cycles")
    ax.set_yscale("log")
    ax.set_title("VLA action-chunk decode: best interface vs reuse")
    ax.legend(title="contract")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
