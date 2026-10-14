"""The tracked MLP bundles (``merlin/tests/data/mlp_bundles/``) still match a fresh model2MLIR capture.

They are a snapshot of another repo's output, checked in so the external-kernel tests run without a
model2MLIR checkout. A snapshot does not notice when model2MLIR's lowering moves on -- every test that
reads it keeps passing on the old IR -- so this re-captures both workloads with the configured
model2MLIR (``MERLIN_M2M_DIR`` / ``MERLIN_M2M_VENV``) through the same generator that writes them, and
requires every file to be byte-identical. Skipped when no model2MLIR checkout with these workloads is
configured; on a mismatch, regenerate with ``build_tools/scripts/gen_mlp_kernel_fixtures.py``.
"""

from __future__ import annotations

import subprocess

import pytest

from merlin.common.paths import merlin_dir, repo_root

WORKLOADS = ("mlp_basic", "mlp_basic_triton")
TRACKED = merlin_dir() / "tests" / "data" / "mlp_bundles"


def _model2mlir():
    from merlin.llvmlower import toolchain

    m2m, python = toolchain.m2m_dir(), toolchain.m2m_python()
    if not python.is_file():
        pytest.skip(f"no model2MLIR venv at {python}")
    missing = [w for w in WORKLOADS if not (m2m / "workloads" / w / "loader.py").is_file()]
    if missing:
        pytest.skip(f"model2MLIR checkout {m2m} has no workloads {missing}")
    return m2m, python


def test_tracked_bundles_match_a_fresh_capture(tmp_path):
    m2m, python = _model2mlir()
    gen = repo_root() / "build_tools" / "scripts" / "gen_mlp_kernel_fixtures.py"
    proc = subprocess.run(
        [str(python), str(gen), "--m2m", str(m2m), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"fresh capture failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"

    stale = []
    for w in WORKLOADS:
        tracked = sorted(p.name for p in (TRACKED / w).iterdir())
        fresh = sorted(p.name for p in (tmp_path / w).iterdir())
        assert tracked == fresh, f"{w}: tracked files {tracked} != generated files {fresh}"
        stale += [f"{w}/{n}" for n in fresh if (TRACKED / w / n).read_bytes() != (tmp_path / w / n).read_bytes()]
    assert not stale, (
        f"tracked MLP bundles are stale against model2MLIR at {m2m}: {stale}. Regenerate with "
        f"`$MERLIN_M2M_VENV/bin/python build_tools/scripts/gen_mlp_kernel_fixtures.py --m2m "
        f"$MERLIN_M2M_DIR --out merlin/tests/data/mlp_bundles`, then re-run the external-kernel tests."
    )
