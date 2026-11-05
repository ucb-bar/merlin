"""Compile one bounded evidence-window test; no simulator or timing qualification."""
from pathlib import Path
import shutil
import subprocess


def test_instruction_context_evidence(tmp_path):
    package = Path(__file__).resolve().parents[1]
    compiler = shutil.which("c++")
    assert compiler, "native timing contract checks require a C++17 compiler"
    output = tmp_path / "instruction-context-test"
    subprocess.run([compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror",
                    "-I", str(package / "include"),
                    str(package / "tests/test_instruction_context.cpp"), "-o", str(output)],
                   check=True, capture_output=True, timeout=30)
    subprocess.run([str(output)], check=True, capture_output=True, timeout=5)
