"""Bounded native checks of the explicit advisory credit consumer; no hardware."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


@pytest.mark.parametrize("source", ["test_engine.cpp", "test_reservations.cpp", "test_reserved_submission.cpp"])
def test_canonical_timing_headers_native(tmp_path, source):
    root = repo_root() / "packages/merlin-timing"
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if compiler is None:
        pytest.skip("explicit C++17 compiler unavailable")
    compiler = Path(compiler).resolve()
    inputs = [
        root / "tests" / source,
        *sorted((root / "include/merlin_timing").glob("*.hpp")),
        compiler,
        repo_root() / "merlin/tests/infra/test_timing_reserved_submission.py",
    ]

    def pin(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    before = [pin(path) for path in inputs]
    image = tmp_path / "check"
    commands = [
        [
            str(compiler),
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-O0",
            "-I",
            str(root / "include"),
            str(root / "tests" / source),
            "-o",
            str(image),
        ],
        [str(image)],
    ]
    records = []
    for index, argv in enumerate(commands):
        result = subprocess.run(argv, cwd=tmp_path, capture_output=True, timeout=60, check=False)
        stdout, stderr = tmp_path / f"{index}.stdout", tmp_path / f"{index}.stderr"
        stdout.write_bytes(result.stdout)
        stderr.write_bytes(result.stderr)
        records.append({"argv": argv, "returncode": result.returncode, "stdout": pin(stdout), "stderr": pin(stderr)})
        assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert [pin(path) for path in inputs] == before
    (tmp_path / "native.json").write_text(
        json.dumps(
            {
                "inputs": before,
                "image": pin(image),
                "commands": records,
                "scope": "synthetic advisory model arithmetic only; no hardware or timing qualification",
            },
            indent=2,
        )
        + "\n"
    )
