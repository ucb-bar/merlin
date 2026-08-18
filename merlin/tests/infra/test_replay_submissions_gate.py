"""The pre-spend gate that replays frozen agent submissions must have teeth.

Every offline gate we ran before the 2026-08 model campaign exercised the harness against a package
that was correct, absent, empty, or deliberately cheating. None exercised one that was present,
schema-valid and semantically WRONG -- the only kind a real agent produces. The defects that cost a
mismeasured model all lived in that gap, so the gate that closes it must itself be tested: a gate that
cannot fail proves nothing.
"""
from __future__ import annotations

import importlib.util
import textwrap

import pytest

from merlin.common.paths import merlin_dir

_MOD = merlin_dir() / "experiments/capsule_bench/harness/replay_submissions.py"


def _load():
    spec = importlib.util.spec_from_file_location("replay_submissions", _MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rs():
    if not _MOD.is_file():
        pytest.skip(f"{_MOD} not present")
    return _load()


def _pkg(root, name, commands):
    d = root / name / "submission"
    (d / "mlir_oot").mkdir(parents=True)
    (d / "mlir_oot" / "tool.py").write_text("#!/usr/bin/env python3\n")
    (d / "manifest.yaml").write_text(textwrap.dedent(f"""
        schema_version: "1.0"
        artifact_type: mlir_oot_target_backend
        target: t
        language: python
        integrity_exempt: false
        authoring:
          mode: agent_generated_from_recipe
        entrypoints:
          tool: mlir_oot/tool.py
        commands:
        {commands}
    """))
    return d


_GOOD = """  parse:
            argv: ["python3", "mlir_oot/tool.py", "{input_mlir}", "{output_json}"]
          lower_interface_to_target:
            argv: ["python3", "mlir_oot/tool.py", "{input_mlir}", "{output_json}"]
          emit_command_buffer:
            argv: ["python3", "mlir_oot/tool.py", "{input_mlir}", "{output_json}"]
          lower_target_to_llvm:
            argv: ["python3", "mlir_oot/tool.py", "{input_mlir}"]"""


def _verdicts(rs, root, src):
    out = []
    for p in rs.discover(root):
        out += [r["verdict"] for r in rs.replay(p, input_mlir=src)]
    return out


def test_a_well_formed_package_is_all_ok(tmp_path, rs):
    src = tmp_path / "in.mlir"; src.write_text("module {}")
    _pkg(tmp_path, "good", _GOOD)
    assert set(_verdicts(rs, tmp_path, src)) == {"ok"}


def test_an_argv_naming_a_missing_script_is_unactionable(tmp_path, rs):
    """The Nemotron class: schema-valid manifest, file the runner cannot reach."""
    src = tmp_path / "in.mlir"; src.write_text("module {}")
    _pkg(tmp_path, "bad", _GOOD.replace('"mlir_oot/tool.py", "{input_mlir}", "{output_json}"]',
                                        '"mlir_oot/absent.py", "{input_mlir}", "{output_json}"]', 1))
    assert "unactionable" in _verdicts(rs, tmp_path, src)


def test_an_unknown_placeholder_is_actionable_not_opaque(tmp_path, rs):
    """The runner must NAME an invented placeholder rather than pass it through."""
    src = tmp_path / "in.mlir"; src.write_text("module {}")
    _pkg(tmp_path, "ph", _GOOD.replace('"{input_mlir}", "{output_json}"]',
                                       '"{input_json}", "{output_json}"]', 1))
    v = _verdicts(rs, tmp_path, src)
    assert "actionable" in v and "unactionable" not in v


def test_the_gate_exit_code_reflects_a_defect(tmp_path, rs):
    """A launch can only be gated on this if a defect is a non-zero exit."""
    src = tmp_path / "in.mlir"; src.write_text("module {}")
    _pkg(tmp_path, "bad", _GOOD.replace('"mlir_oot/tool.py", "{input_mlir}", "{output_json}"]',
                                        '"mlir_oot/absent.py", "{input_mlir}", "{output_json}"]', 1))
    assert rs.main(["--runs-root", str(tmp_path), "--input", str(src)]) == 1
    _pkg(tmp_path / "clean", "good", _GOOD)
    assert rs.main(["--runs-root", str(tmp_path / "clean"), "--input", str(src)]) == 0


def test_scratch_copies_are_not_replayed(tmp_path, rs):
    """Per-round _qa_work copies would multiply every finding; they are excluded by design."""
    src = tmp_path / "in.mlir"; src.write_text("module {}")
    _pkg(tmp_path, "good", _GOOD)
    _pkg(tmp_path / "_qa_work" / "cand_01", "good", _GOOD)
    assert len(rs.discover(tmp_path)) == 1
