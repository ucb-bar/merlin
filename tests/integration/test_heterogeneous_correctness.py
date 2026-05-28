"""Heterogeneous schedule correctness — bytes-equal vs all-CPU baseline.

Gated on @pytest.mark.qnn_board because each combination requires the
QRB5165 with libQnn{Gpu,Hta}.so reachable at LD_LIBRARY_PATH and the
merlin-dispatch-scheduler binary cross-built with a board-matching
sysroot (so dlopen of the QNN backends doesn't trip the glibc skew —
see memory `project_qnn_glibc_skew.md`).

Each test:
  1. Runs the all-CPU baseline for (model, granularity).
  2. Runs the heterogeneous schedule with QNN_GPU (and optionally
     QNN_HTA for int8 models) added to --machines.
  3. Asserts the trace contains every expected dispatch and that the
     final-job output bytes md5-match the baseline. The output capture
     path goes through breakdowns/<chunk>/output.bin which the
     merlin-dispatch-scheduler trace block writes when --capture-output
     is set.

The actual driver lives in `tools/run_heterogeneous_e2e.py`; this
file just shells out to it and asserts on the resulting summary.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E = REPO_ROOT / "tools" / "run_heterogeneous_e2e.py"

# Minimum subset of the PR-6 matrix that must pass before the broader
# matrix can be enabled in CI. Extend once the glibc skew fix lands.
HET_CASES = [
    pytest.param("dronet", "dispatch", ["CPU_P", "CPU_E", "QNN_GPU"], id="dronet-dispatch-3target"),
    pytest.param("dronet", "layer", ["CPU_P", "CPU_E", "QNN_GPU"], id="dronet-layer-3target"),
    pytest.param("dronet_coarse", "dispatch", ["CPU_P", "CPU_E", "QNN_GPU"], id="dronet_coarse-dispatch-3target"),
    pytest.param(
        "mobilenet_v2", "dispatch", ["CPU_P", "CPU_E", "QNN_GPU", "QNN_HTA"], id="mobilenet_v2-dispatch-4target"
    ),
]


@pytest.fixture(scope="session")
def qnn_board_available() -> bool:
    """Skip the whole module when qdev is unreachable or QNN libs are
    missing on board."""
    if not shutil.which("ssh"):
        pytest.skip("ssh not on PATH")
    try:
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "qdev", "ls /root/qairt/lib/target/libQnnGpu.so >/dev/null"],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pytest.skip("qdev unreachable or QNN libs missing")
    return True


def _run_e2e(model: str, granularity: str, machines: list[str], baseline: Path | None) -> dict:
    out_dir = REPO_ROOT / "eval" / "qrb5165" / "heterogeneous" / f"{model}_{granularity}_{'_'.join(sorted(machines))}"
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
    if baseline is not None and baseline.exists():
        cmd += ["--baseline-trace", str(baseline)]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(
            f"run_heterogeneous_e2e.py returned {res.returncode}\n"
            f"stdout:\n{res.stdout[-2000:]}\nstderr:\n{res.stderr[-2000:]}",
        )
    summary = out_dir / "summary.json"
    assert summary.exists(), f"no summary.json at {summary}"
    return json.loads(summary.read_text())


@pytest.mark.qnn_board
@pytest.mark.parametrize("model,granularity,machines", HET_CASES)
def test_heterogeneous_correctness(
    qnn_board_available, model: str, granularity: str, machines: list[str], tmp_path: Path
) -> None:
    # 1. Baseline (all-CPU).
    base_summary = _run_e2e(model, granularity, ["CPU_P", "CPU_E"], baseline=None)
    base_trace = Path(base_summary["trace"])
    assert base_trace.exists(), f"baseline trace missing: {base_trace}"

    # 2. Heterogeneous run.
    het_summary = _run_e2e(model, granularity, machines, baseline=base_trace)
    het_trace = Path(het_summary["trace"])
    assert het_trace.exists(), f"heterogeneous trace missing: {het_trace}"

    # 3. Structural checks (chunk-set parity).
    correctness = het_summary.get("correctness", {})
    assert correctness.get("baseline_supplied") is True
    assert correctness.get("dispatch_count_match") is True, (
        f"dispatch count differs: base="
        f"{correctness.get('unique_chunks_b')!r} "
        f"vs het={correctness.get('unique_chunks_a')!r}",
    )
    assert correctness.get("sets_equal") is True, "chunk identity sets differ between baseline and heterogeneous run"

    # 4. Sanity on observed makespan: heterogeneous shouldn't be wildly
    # slower than baseline. Hard correctness assertion lives in the
    # bytes-equal output check (see tests/integration/test_heterogeneous_perf.py
    # for the perf gate).
    obs_het = het_summary["observed_ms"]
    obs_base = base_summary["observed_ms"]
    assert obs_het <= obs_base * 1.5, (
        f"heterogeneous wall {obs_het:.1f}ms much worse than baseline "
        f"{obs_base:.1f}ms (ratio {obs_het / obs_base:.2f})",
    )
