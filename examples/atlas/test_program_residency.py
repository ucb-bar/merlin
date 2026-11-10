"""Compose the generic residency policy with real Atlas primitives."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent


def test_program_residency(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required")
    command = [cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
               f"-I{ROOT / 'merlin/runtime/c'}"]
    atlas = Path(os.environ.get("ATLAS_MLIR_ROOT", ROOT.parent / "atlas-mlir"))
    if not (atlas / "runtime/atlas_program_slots.c").is_file():
        pytest.skip("Atlas target primitives checkout required")
    command.extend([
        f"-I{atlas / 'runtime'}", f"-I{SOURCE}",
        str(SOURCE / "program_session_test.c"),
        str(atlas / "runtime/atlas_host.c"),
        str(atlas / "runtime/atlas_program_slots.c"),
    ])
    executable = tmp_path / "residency"
    subprocess.run([*command, "-o", str(executable)], check=True, capture_output=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert "passed" in result.stdout
