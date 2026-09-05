"""An input the compiler accepts but emits nothing for must be refused before it is measured.

The shipped compiler consumes one closed interface dialect. Handed anything else -- the
linalg-on-tensors a real model capture produces -- it exits 0, emits an empty command list with a
`declined` note, and says nothing on stderr. Reproduced on the sealed package.

The artefact of that failure is a SPECTACULAR speedup: a kernel that does no work finishes almost
immediately, so anyone benchmarking the compiler on their own models would measure a huge win on
exactly the inputs it cannot compile. Hence a check that runs before any measurement.
"""
from __future__ import annotations

import json
import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import check_handover_input as CHK  # noqa: E402


def _package(tmp_path, buffer: dict | None, *, returncode: int = 0):
    """A stand-in compiler that writes whatever command buffer the test asks for."""
    package = tmp_path / "submission"
    (package / "mlir_oot").mkdir(parents=True)
    tool = package / "mlir_oot" / "gemmini-opt"
    body = ["import sys, json, pathlib"]
    if buffer is not None:
        body += [
            "out = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--emit-command-buffer=')]",
            f"pathlib.Path(out[0]).write_text(json.dumps({buffer!r}))",
        ]
    body.append(f"sys.exit({returncode})")
    tool.write_text("\n".join(body), encoding="utf-8")
    (package / "manifest.yaml").write_text(
        yaml.safe_dump({"entrypoints": {"tool": "mlir_oot/gemmini-opt"}}), encoding="utf-8")
    return package


def _source(tmp_path):
    p = tmp_path / "in.interface.mlir"
    p.write_text("module {}\n", encoding="utf-8")
    return p


def test_a_declined_module_is_refused(tmp_path):
    package = _package(tmp_path, {"commands": [], "declined": {"reason": "not routed"}})
    row = CHK.inspect(package, _source(tmp_path))
    assert row["status"] == "empty_kernel"
    assert "never did" in row["reason"] or "no work" in row["reason"]


def test_an_empty_command_list_is_refused_even_without_a_declined_note(tmp_path):
    """The note is the compiler's admission; its absence must not make an empty kernel pass."""
    package = _package(tmp_path, {"commands": []})
    assert CHK.inspect(package, _source(tmp_path))["status"] == "empty_kernel"


def test_a_real_command_buffer_passes(tmp_path):
    package = _package(tmp_path, {"commands": [{"opcode": "MVIN"}, {"opcode": "COMPUTE"}]})
    row = CHK.inspect(package, _source(tmp_path))
    assert row["status"] == "ok" and row["commands"] == 2


def test_an_honest_compiler_rejection_is_distinguished_from_an_empty_kernel(tmp_path):
    """A non-zero exit is a real failure and must not be reported as the silent-empty case."""
    package = _package(tmp_path, None, returncode=1)
    assert CHK.inspect(package, _source(tmp_path))["status"] == "refused_by_compiler"


def test_exit_status_marks_a_refused_input(tmp_path):
    package = _package(tmp_path, {"commands": [], "declined": {"reason": "not routed"}})
    rc = CHK.main(["--package", str(package), "--input", str(_source(tmp_path))])
    assert rc == 2, "a refused input must be a non-zero exit, or a script will measure it anyway"
