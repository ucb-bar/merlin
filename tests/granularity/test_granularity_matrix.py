"""Granularity test matrix.

For each fixture under tests/granularity/fixtures/{tile,layer,megakernel}/
this test asserts that compiling its `in.mlir` against the real
`benchmarks/SaturnOPU/kernels` manifest:

  1. Succeeds with `--kernels-strict-coverage` (every linalg dispatch
     matched a manifest kernel).
  2. Rewrites the linalg op into a `flow.dispatch @kb_<kernel>::<entry>`
     symbol in the post-flow MLIR phase dump.

We deliberately do NOT byte-compare against a baseline run: the kernels
target `llvm-cpu-spacemit-x60` (RISC-V), which can't execute on the host
that runs the test. The byte-equal claim belongs in the per-board pytest
under `test_rvv_kernels_on_spike.py`.

Fixtures with a `skip` file are skipped with the file's contents as
the reason — currently i8 cases (no i8 kernel authored yet) and the
megakernel fused fixtures (Phase Stretch).
"""

from __future__ import annotations

import dataclasses
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES_ROOT = REPO_ROOT / "tests" / "granularity" / "fixtures"
KERNELS_DIR = REPO_ROOT / "benchmarks" / "SaturnOPU" / "kernels"
MERLIN = REPO_ROOT / "merlin"

# Map fixture-name patterns to the manifest kernel they should match.
EXPECTED_KERNEL: dict[str, str] = {
    "matmul_small_f32": "kb_saturnopu_matmul_f32",
    "matmul_medium_f32": "kb_saturnopu_matmul_f32",
    "elementwise_small_f32": "kb_saturnopu_add_f32",
    "elementwise_medium_f32": "kb_saturnopu_add_f32",
}


@dataclasses.dataclass
class Fixture:
    granularity: str
    name: str
    path: pathlib.Path

    @property
    def in_mlir(self) -> pathlib.Path:
        return self.path / "in.mlir"

    @property
    def skip_marker(self) -> pathlib.Path:
        return self.path / "skip"


def _discover_fixtures() -> list[Fixture]:
    fixtures: list[Fixture] = []
    for granularity in ("tile", "layer", "megakernel"):
        base = FIXTURES_ROOT / granularity
        if not base.exists():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir() or not (child / "in.mlir").exists():
                continue
            fixtures.append(Fixture(granularity, child.name, child))
    return fixtures


def _have_merlin() -> bool:
    return MERLIN.exists() and shutil.which("conda") is not None


@pytest.mark.integration
@pytest.mark.parametrize(
    "fixture",
    _discover_fixtures(),
    ids=lambda f: f"{f.granularity}/{f.name}",
)
def test_kernel_embedding_rewrites(fixture: Fixture, tmp_path: pathlib.Path) -> None:
    """Compile the fixture against benchmarks/SaturnOPU/kernels and verify
    that the linalg op was rewritten into a kernel-embedded dispatch."""
    if fixture.skip_marker.exists():
        pytest.skip(f"{fixture.name}: {fixture.skip_marker.read_text().strip()}")
    if not _have_merlin():
        pytest.skip("./merlin or conda not available")
    expected = EXPECTED_KERNEL.get(fixture.name)
    if expected is None:
        pytest.skip(f"{fixture.name}: no expected kernel mapping")

    out_dir = tmp_path / "out"
    cmd = [
        str(MERLIN),
        "compile",
        str(fixture.in_mlir),
        "--target",
        "spacemit_x60",
        "--hw",
        "RVV",
        "--kernels-dir",
        str(KERNELS_DIR),
        "--kernels-strict-coverage",
        "--dump-phases",
        "--output-dir",
        str(out_dir),
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False, timeout=240)
    assert res.returncode == 0, (
        f"./merlin compile failed for {fixture.name} " f"(rc={res.returncode}):\n--- stderr ---\n{res.stderr[-2000:]}"
    )

    flow_phase = next(out_dir.glob("phases/*.6.flow.mlir"), None)
    assert flow_phase is not None, "phase 6 (flow) MLIR dump missing"
    flow_text = flow_phase.read_text()
    assert expected in flow_text, (
        f"{fixture.name}: expected {expected} in flow.mlir but rewrite "
        f"didn't land — match.mlir or named_op spec may not fit this input."
    )
