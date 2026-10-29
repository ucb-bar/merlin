"""Source-only eligibility custody, with synthetic exporter/recognizer results."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.verify import compiler_ir_source_profile as profiles
from merlin.verify.gate_errors import GateUnavailable


@pytest.mark.parametrize("mutation", ["export", "source"])
def test_changed_export_or_source_cannot_fabricate_a_source_exclusion(tmp_path, monkeypatch, mutation):
    from merlin.verify.lean import check

    stub = tmp_path / "unused-checker"
    stub.touch()
    monkeypatch.setattr(check, "lean_exe", lambda: stub)
    validation = SimpleNamespace(
        gate=SimpleNamespace(directory=tmp_path),
        tools=SimpleNamespace(compiler_python=Path(sys.executable), sdk_bin=tmp_path),
    )
    raw = "builtin.module { func.func @forward(%x:i32) -> i32 { func.return %x : i32 } }"

    def invoke(command, **kwargs):
        if "source-profile" not in command:
            exported = Path(command[-1])
            exported.write_text(
                json.dumps(
                    {
                        "schema": "merlin.lean.ir.v1",
                        "source": {"sha256": profiles.digest(raw), "bytes": len(raw.encode())},
                    }
                )
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        exported = Path(command[-2])
        path = exported if mutation == "export" else exported.with_name("source.mlir")
        path.write_text("changed while original-source recognizer reopens its input")
        return SimpleNamespace(
            returncode=1,
            stderr="",
            stdout=json.dumps(
                {
                    "schema": "merlin.compiler-source-profile.v1",
                    "profile": "tensor",
                    "index_bits": 64,
                    "supported": False,
                    "reason": "a different source is outside the model",
                    "boundary_proof": False,
                    "policy_approval": False,
                }
            ),
        )

    monkeypatch.setattr(profiles, "run_owned", invoke)
    with pytest.raises(GateUnavailable, match="original bytes changed"):
        profiles.admit_source(raw, pass_name="cse", validation=validation)
    assert not list(tmp_path.rglob("admission.json"))


def test_unexpected_host_parser_failure_is_not_a_source_exclusion(monkeypatch):
    from host_graph_fusion_cases import source

    from merlin.verify import host_graph_fusion_projection as projection

    def unavailable(*args):
        raise RuntimeError("unexpected parser implementation failure")

    monkeypatch.setattr(projection, "parse_source", unavailable)
    with pytest.raises(RuntimeError, match="unexpected parser"):
        profiles.admit_host_source(source(), fusion_groups=((0, 1, 2, 3, 4, 5),), entry="forward")


@pytest.mark.parametrize("mutation", [None, "input", "python"])
def test_native_parse_exclusion_requires_original_input_and_tools(tmp_path, monkeypatch, mutation):
    from merlin.verify.compiler_profile import SourceOutsideProfile
    from merlin.verify.lean import check

    stub = tmp_path / "unused-checker"
    stub.touch()
    python = tmp_path / "owned-interpreter-sentinel"
    python.write_text("never executed: host protocol control")
    monkeypatch.setattr(check, "lean_exe", lambda: stub)
    validation = SimpleNamespace(
        gate=SimpleNamespace(directory=tmp_path), tools=SimpleNamespace(compiler_python=python, sdk_bin=tmp_path)
    )
    raw = "builtin.module { func.func @forward(%x:i32) -> i32 { func.return %x : i32 } }"
    calls = []

    def invoke(command, **kwargs):
        assert command[1].endswith("source_profile_export.py")  # Never launch the Lean source recognizer.
        calls.append(command)
        source, destination = map(Path, command[-2:])
        destination.write_text(
            json.dumps(
                {
                    "schema": "merlin.compiler-source-export.v1",
                    "source": {"sha256": profiles.digest(raw), "bytes": len(raw.encode())},
                    "supported": False,
                    "reason": "native source syntax outside the selected grammar",
                }
            )
        )
        if mutation == "input":
            source.write_text("different source")
        elif mutation == "python":
            python.write_text("changed tool bytes")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(profiles, "run_owned", invoke)
    expected = GateUnavailable if mutation else SourceOutsideProfile
    with pytest.raises(expected):
        profiles.admit_source(raw, pass_name="cse", validation=validation)
    assert len(calls) == 1
