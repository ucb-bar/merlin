"""Bounded source-identity/arbitration tests; no simulator or fitted labels."""
from pathlib import Path
import shutil
import subprocess


def test_memory_milestones(tmp_path):
    package = Path(__file__).resolve().parents[1]
    compiler = shutil.which("c++")
    assert compiler
    output = tmp_path / "memory-milestones-test"
    subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-I", str(package / "include"),
                    str(package / "tests/test_memory_milestones.cpp"), "-o", str(output)],
                   check=True, capture_output=True, timeout=30)
    subprocess.run([str(output)], check=True, capture_output=True, timeout=5)
