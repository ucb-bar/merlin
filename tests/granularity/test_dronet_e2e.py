"""Gate the headline dronet kernel-embedding demo (`scripts/dronet_spike_e2e.sh`)
behind a marked pytest so the end-to-end claim is reproducible.

Asserts:
  1. The script exits 0 (discovery + compile complete).
  2. The produced `dronet.vmfb` is non-empty and at least 100 KB
     (roughly the size needed once kernels are linked in).
  3. The post-flow MLIR phase dump rewrites at least 2 distinct
     manifest kernels into `util.call @call_saturnopu_*` symbols —
     today that's `matmul_f32` and `pooling_nchw_max_f32`, both
     firing across multiple dispatches. Multi-kernel coverage is the
     point; raise the lower bound when more dronet ops are wired up.

Marked `slow` (because it shells out to a full compile) and `integration`
(because it requires `./merlin` + `merlin-dev` conda env). Run with:

    pytest tests/granularity/test_dronet_e2e.py -m 'slow and integration' -v
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "dronet_spike_e2e.sh"
DRONET_INPUT = REPO_ROOT / "models" / "dronet" / "dronet.mlir"
OUT_DIR = REPO_ROOT / "build" / "dronet_spike_e2e"


@pytest.mark.slow
@pytest.mark.integration
def test_dronet_kernel_embed_end_to_end() -> None:
    if not SCRIPT.exists():
        pytest.skip("scripts/dronet_spike_e2e.sh not present")
    if not DRONET_INPUT.exists():
        pytest.skip("models/dronet/dronet.mlir not present")
    if shutil.which("conda") is None:
        pytest.skip("conda not on PATH")

    res = subprocess.run(
        ["bash", str(SCRIPT), "--compile-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert res.returncode == 0, (
        f"dronet_spike_e2e.sh failed (rc={res.returncode}):\n" f"--- stderr ---\n{res.stderr[-2000:]}"
    )

    vmfb = OUT_DIR / "dronet.vmfb"
    assert vmfb.exists(), f"expected vmfb at {vmfb}"
    size = vmfb.stat().st_size
    assert size > 100_000, f"vmfb suspiciously small: {size} bytes"

    flow_phase = next(OUT_DIR.glob("phases/*.6.flow.mlir"), None)
    assert flow_phase is not None, "phase 6 (flow) MLIR dump missing"
    flow_text = flow_phase.read_text()
    distinct = set(re.findall(r"@call_saturnopu_[a-z0-9_]+", flow_text))
    assert len(distinct) >= 2, (
        f"expected ≥2 distinct saturnopu kernel calls in flow.mlir, " f"got {len(distinct)}: {sorted(distinct)}"
    )
