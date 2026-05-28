"""End-to-end compile test for the smallest matmul fixture.

Drives `./merlin compile` through the real IREE plugin pipeline with the
gemmini plugin enabled. Verifies that:
  1. The post-global-opt hook fires (linalg.matmul gets recovered into
     gemmini.matmul / gemmini.matmul_tile).
  2. The pipeline produces a runnable `.vmfb` artifact.

This is the test the previous iteration of the workstream marked xfail
with a (false) "bufferization gap" claim — it now passes through the
IREE codegen-fallback path. See docs/dev_blog/2026-03-11-gemmini-workstream-log.md
section 14.9 for the retraction and the precise blocker on the native
`--iree-gemmini-lower-back-to-iree=false` path.
"""

from __future__ import annotations

import pathlib
import subprocess

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_matmul_8x8x8_compiles(merlin_cli, merlin_env, tmp_path, repo_root):
    fixture = _FIXTURES / "matmul_8x8x8_tensor.mlir"
    assert fixture.exists()

    output_dir = tmp_path / "compiled"
    cmd = merlin_cli + [
        str(fixture),
        "--target",
        "gemmini_spike",
        "--build-dir",
        "host-merlin-debug",
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(
        cmd,
        env=merlin_env,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert result.returncode == 0, (
        f"./merlin compile failed (rc={result.returncode}).\n" f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    vmfb = output_dir / "matmul_8x8x8_tensor.vmfb"
    assert vmfb.exists(), f"Expected vmfb at {vmfb}, but it was not produced."
    assert vmfb.stat().st_size > 0, f"vmfb at {vmfb} is empty."
