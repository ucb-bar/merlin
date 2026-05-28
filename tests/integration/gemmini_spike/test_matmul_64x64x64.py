"""End-to-end compile test for the 64x64x64 matmul fixture.

Same flow as test_matmul_8x8x8_compiles, but for a shape that's a clean
multiple of libgemmini.so's DIM=16 tile (no padding). See the sibling
test for the rationale on the IREE codegen-fallback path.
"""

from __future__ import annotations

import pathlib
import subprocess

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def test_matmul_64x64x64_compiles(merlin_cli, merlin_env, tmp_path, repo_root):
    fixture = _FIXTURES / "matmul_64x64x64_tensor.mlir"
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

    vmfb = output_dir / "matmul_64x64x64_tensor.vmfb"
    assert vmfb.exists(), f"Expected vmfb at {vmfb}, but it was not produced."
    assert vmfb.stat().st_size > 0, f"vmfb at {vmfb} is empty."
