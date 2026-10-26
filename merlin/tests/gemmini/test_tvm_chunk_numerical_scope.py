"""Legacy dynamic-row probes must not silently become a new numerical release."""

import subprocess
import sys

from merlin.common.paths import repo_root


def test_checkpoint_probe_requires_explicit_legacy_scope_before_preparation(tmp_path):
    output = tmp_path / "probe"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root() / "examples/gemmini/comparisons/tvm/prepare_firesim_chunks.py"),
            "checkpoint-probes",
            "--checkpoint",
            str(tmp_path / "missing-checkpoint"),
            "--tokens",
            str(tmp_path / "missing-tokens"),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires --legacy-diagnostic" in result.stderr
    assert not output.exists()
