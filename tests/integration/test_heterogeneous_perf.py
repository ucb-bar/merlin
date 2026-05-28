"""Heterogeneous schedule perf gate.

Asserts the heterogeneous run isn't slower than the all-CPU baseline by
more than 5 % wall time, and the median plan-vs-observed gap stays
under 25 % across the dispatch trace. Same `qnn_board` gating as
`test_heterogeneous_correctness.py`.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E = REPO_ROOT / "tools" / "run_heterogeneous_e2e.py"

PERF_CASES = [
    pytest.param("dronet", "dispatch", ["CPU_P", "CPU_E", "QNN_GPU"], id="dronet-dispatch-3target"),
    pytest.param("dronet", "layer", ["CPU_P", "CPU_E", "QNN_GPU"], id="dronet-layer-3target"),
]

REPETITIONS = 5  # iterations on the perf gate; keep tight for CI time.
WALL_TOLERANCE = 1.05  # heterogeneous wall ≤ baseline × 1.05
GAP_PCT_MAX = 25.0  # median |run-plan|/plan ≤ 25%


def _run_n(model: str, granularity: str, machines: list[str], baseline_trace: Path | None, n: int) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        out_dir = (
            REPO_ROOT
            / "eval"
            / "qrb5165"
            / "heterogeneous"
            / f"{model}_{granularity}_{'_'.join(sorted(machines))}_run{i}"
        )
        cmd = [
            "uv",
            "run",
            "python",
            str(E2E),
            "--model",
            model,
            "--granularity",
            granularity,
            "--machines",
            *machines,
            "--output-dir",
            str(out_dir),
            "--repetitions",
            "5",
        ]
        if baseline_trace is not None:
            cmd += ["--baseline-trace", str(baseline_trace)]
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if res.returncode != 0:
            pytest.fail(
                f"run_heterogeneous_e2e.py iter{i} rc={res.returncode}\n" f"stderr:\n{res.stderr[-1500:]}",
            )
        s = json.loads((out_dir / "summary.json").read_text())
        out.append(s)
    return out


@pytest.mark.qnn_board
@pytest.mark.parametrize("model,granularity,machines", PERF_CASES)
def test_heterogeneous_perf(model: str, granularity: str, machines: list[str]) -> None:
    base = _run_n(model, granularity, ["CPU_P", "CPU_E"], baseline_trace=None, n=REPETITIONS)
    base_walls = [s["observed_ms"] for s in base]
    base_med = statistics.median(base_walls)

    het = _run_n(model, granularity, machines, baseline_trace=Path(base[0]["trace"]), n=REPETITIONS)
    het_walls = [s["observed_ms"] for s in het]
    het_med = statistics.median(het_walls)

    assert het_med <= base_med * WALL_TOLERANCE, (
        f"heterogeneous median {het_med:.2f}ms exceeds baseline "
        f"{base_med:.2f}ms × {WALL_TOLERANCE} = "
        f"{base_med * WALL_TOLERANCE:.2f}ms",
    )

    gap_meds = [s["gap_pct_median"] for s in het if "gap_pct_median" in s]
    if gap_meds:
        gap_overall = statistics.median(gap_meds)
        assert gap_overall <= GAP_PCT_MAX, (
            f"median plan-vs-observed gap {gap_overall:.1f}% exceeds " f"{GAP_PCT_MAX}%",
        )
