"""End-to-end mxGemmini FP4 test: ./merlin sim → VCS → diff vs golden."""

from __future__ import annotations

import pathlib
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "mlp_3layer_fp4.mlir"
EXPECTED = HERE / "fixtures" / "expected_fp4.txt"


@pytest.mark.timeout(1800)
def test_mxgemmini_mlp_fp4(merlin_cli: str, repo_root: pathlib.Path) -> None:
    if not FIXTURE.exists():
        pytest.skip(
            f"Fixture not generated: {FIXTURE}. "
            "Run `python tests/integration/gemmini_mx_vcs/mlp_3layer_torch.py` first."
        )
    if not EXPECTED.exists():
        pytest.skip(f"Golden not generated: {EXPECTED}")

    cmd = [
        merlin_cli,
        "sim",
        str(FIXTURE),
        "--target",
        "gemmini_mx_vcs",
        "--hw",
        "VCS_FP4",
        "--reference",
        str(EXPECTED),
        "--simulator",
        "vcs",
        "--config",
        "RadianceGemminiOnlyConfig",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("=== stdout ===\n", result.stdout)
        print("=== stderr ===\n", result.stderr)
    assert result.returncode == 0, f"./merlin sim returned {result.returncode}; see captured output."
    assert "[merlin-sim] PASS" in result.stdout, "expected '[merlin-sim] PASS' in stdout; got:\n" + result.stdout
