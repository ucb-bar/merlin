"""Compile the generic residency policy against a fake target."""

from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).resolve().parent


def test_program_residency(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required")
    executable = tmp_path / "residency"
    subprocess.run([
        cc, "-std=c11", "-Wall", "-Wextra", "-Werror",
        f"-I{ROOT / 'merlin/runtime/c'}",
        str(SOURCE / "program_residency_test.c"), "-o", str(executable),
    ], check=True, capture_output=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert "Generic fake-target tests passed" in result.stdout
