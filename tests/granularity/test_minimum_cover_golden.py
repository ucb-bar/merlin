"""Golden-file test for `tools.kernels.discover --minimum-cover --auto-fuse`
on dronet.

This pins the cover-set size + fused-signature count so any silent
algorithmic drift in the discovery / set-cover code shows up as a test
failure (instead of rotting the dev-blog and architecture doc that
quote these numbers).

If you intentionally change the discovery algorithm, update GOLDEN below
and the corresponding numbers in:
  * docs/dev_blog/2026-04-29-kernel-embedding-status-and-demo-guide.md
  * docs/architecture/kernel_matching_methodology.md

Marked `slow` + `integration`. Skipped when `./merlin` is missing.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DRONET = REPO_ROOT / "models" / "dronet" / "dronet.mlir"
MERLIN = REPO_ROOT / "merlin"

# Pinned numbers from the documented baseline. The cover set may shift by
# ±1 if the body classifier picks up a new op pattern; widen these
# assertions before chasing exact equality.
GOLDEN = {
    "cover_kernels": 9,  # "9 kernels = 100% coverage"
    "cover_dispatches": 48,  # "covers 48 dispatches"
    "cover_shape_variants": 33,
    "fused_signatures": 32,  # "Fused dispatches detected at flow phase (32 unique)"
}


@pytest.mark.slow
@pytest.mark.integration
def test_dronet_minimum_cover_golden(tmp_path: pathlib.Path) -> None:
    if not DRONET.exists():
        pytest.skip("models/dronet/dronet.mlir not present")
    if not MERLIN.exists() or shutil.which("conda") is None:
        pytest.skip("./merlin / conda not available")

    out_dir = tmp_path / "discover"
    cmd = [
        "conda",
        "run",
        "-n",
        "merlin-dev",
        "uv",
        "run",
        "python",
        "-m",
        "tools.kernels.discover",
        str(DRONET),
        "--target",
        "saturn_opu_spike",
        "--hw",
        "SPIKE",
        "--output",
        str(out_dir),
        "--minimum-cover",
        "--auto-fuse",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=600)
    assert res.returncode == 0, f"discover failed (rc={res.returncode}):\n--- stderr ---\n" f"{res.stderr[-2000:]}"
    text = res.stdout

    m_kernels = re.search(r"(\d+)\s+kernels\s*=\s*100%\s+coverage", text)
    m_dispatch = re.search(
        r"covers\s+(\d+)\s+dispatches\s+across\s+(\d+)\s+shape\s+variants",
        text,
    )
    assert m_kernels, f"missing kernels-= summary in:\n{text[-2000:]}"
    assert m_dispatch, f"missing dispatches/shape-variants summary in:\n{text[-2000:]}"
    assert int(m_kernels.group(1)) == GOLDEN["cover_kernels"]
    assert int(m_dispatch.group(1)) == GOLDEN["cover_dispatches"]
    assert int(m_dispatch.group(2)) == GOLDEN["cover_shape_variants"]

    fused = re.search(
        r"Fused dispatches detected at flow phase\s+\((\d+)\s+unique",
        text,
    )
    assert fused, "could not find fused-dispatch summary"
    assert int(fused.group(1)) == GOLDEN["fused_signatures"]
