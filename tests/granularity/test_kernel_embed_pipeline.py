"""End-to-end test: ./merlin compile --kernels-dir produces a vmfb with the
custom kernel embedded.

Two cases:
- The fixture under tests/granularity/fixtures/embed_pipeline/ (synthetic
  manifest with a tiny f32 add kernel).
- The benchmarks/SaturnOPU/kernels/ manifest (the real RVV-intrinsics add
  kernel that's also exercised standalone on Spike).

Both compile a synthetic linalg.generic-add input MLIR with --kernels-dir
and assert:

  1. iree-compile succeeds and emits a vmfb.
  2. The auto-generated transform_spec.mlir lands in <out>/kernels_cache/.
  3. Phase 6 (post-flow) MLIR contains the rewrite to flow.dispatch into
     the kernel — i.e. the linalg op was matched and rewritten, not codegen'd
     in-place.
  4. Phase 10 (post-HAL) MLIR carries the linked .o path.

Marked `integration` because it shells out to ./merlin which requires the
`merlin-dev` conda env.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _have_merlin() -> bool:
    return (REPO_ROOT / "merlin").exists() and shutil.which("conda") is not None


EMBED_CASES: list[tuple[str, str, str]] = [
    (
        "fixture_add_f32",
        "tests/granularity/fixtures/embed_pipeline",
        "tests/granularity/fixtures/embed_pipeline/add_input.mlir",
    ),
    (
        "saturnopu_add_f32",
        "benchmarks/SaturnOPU/kernels",
        "tests/granularity/fixtures/embed_pipeline/add_input.mlir",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("kernels_dir_rel", "input_rel"),
    [(case[1], case[2]) for case in EMBED_CASES],
    ids=[case[0] for case in EMBED_CASES],
)
def test_kernel_embed_pipeline(tmp_path: pathlib.Path, kernels_dir_rel: str, input_rel: str) -> None:
    if not _have_merlin():
        pytest.skip("./merlin or conda not available")

    out_dir = tmp_path / "out"
    cmd = [
        str(REPO_ROOT / "merlin"),
        "compile",
        input_rel,
        "--target",
        "spacemit_x60",
        "--hw",
        "RVV",
        "--kernels-dir",
        kernels_dir_rel,
        "--dump-phases",
        "--output-dir",
        str(out_dir),
    ]
    res = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert res.returncode == 0, (
        f"./merlin compile failed (rc={res.returncode}):\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )

    # 1. vmfb emitted.
    vmfbs = list(out_dir.glob("*.vmfb"))
    assert vmfbs, f"no .vmfb produced under {out_dir}"

    # 2. auto-generated transform spec.
    spec = out_dir / "kernels_cache" / "transform_spec.mlir"
    assert spec.exists(), f"transform_spec.mlir not generated at {spec}"

    # 3. Rewrite landed: phase 6 (post-flow) should contain the kb_ dispatch.
    phase6 = next(out_dir.glob("phases/*.6.flow.mlir"), None)
    assert phase6 is not None, "phase 6 dump missing"
    flow_text = phase6.read_text()
    assert "flow.dispatch @kb_" in flow_text, (
        "linalg op was not rewritten to flow.dispatch into the embedded "
        "kernel; check the match.mlir against the input MLIR."
    )

    # 4. .o linked into hal.executable.variant.
    phase10 = next(out_dir.glob("phases/*.10.executable-targets.mlir"), None)
    assert phase10 is not None, "phase 10 dump missing"
    hal_text = phase10.read_text()
    assert "hal.executable.object" in hal_text, (
        "executable-targets phase MLIR is missing hal.executable.object — " "the precompiled .o was not linked in."
    )
