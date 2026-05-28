"""Regression test: re-run `refresh_phase_dumps.sh` and assert the
expected kernel rewrites still land in the regenerated COVERAGE.txt.

This guards against silent IR drift in the embed pipeline. If a future
compiler-side change shifts where the rewrite happens, makes the matcher
miss, or splits the dispatch differently, the regenerated coverage report
won't contain the expected `util.call @call_saturnopu_*` line and the
test fails — surfacing the drift before it rots the docs that reference
these snapshots.

Marked `slow` + `integration` because it shells out to `./merlin compile`
twice (one per kernel). Skipped when `./merlin` or conda are missing.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "benchmarks" / "SaturnOPU" / "kernels" / "phase_dumps" / "refresh_phase_dumps.sh"
DUMP_ROOT = REPO_ROOT / "benchmarks" / "SaturnOPU" / "kernels" / "phase_dumps"

EXPECTED: dict[str, list[str]] = {
    "add_f32": ["util.call @call_saturnopu_add_f32"],
    "linear_f32": ["util.call @call_saturnopu_linear_f32"],
}


def _run(target: str) -> None:
    res = subprocess.run(
        ["bash", str(SCRIPT), target],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert res.returncode == 0, (
        f"refresh_phase_dumps.sh {target} failed (rc={res.returncode}):\n" f"--- stderr ---\n{res.stderr[-2000:]}"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("kernel", sorted(EXPECTED.keys()))
def test_phase_dumps_regenerate_cleanly(kernel: str) -> None:
    if not SCRIPT.exists():
        pytest.skip("refresh_phase_dumps.sh not present")
    if shutil.which("conda") is None:
        pytest.skip("conda not on PATH")

    _run("add" if kernel == "add_f32" else "linear")

    coverage = DUMP_ROOT / kernel / "COVERAGE.txt"
    assert coverage.exists(), f"{coverage} not regenerated"
    text = coverage.read_text()
    for needle in EXPECTED[kernel]:
        assert needle in text, f"{kernel}: expected '{needle}' in COVERAGE.txt; got:\n{text}"
