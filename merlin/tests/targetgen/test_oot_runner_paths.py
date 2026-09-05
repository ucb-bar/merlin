"""Entrypoints run FROM the package root, and an unknown placeholder fails loudly.

Three defects measured on a real agent run, each of which made a harness fault read as a broken
submission:

* `run_entrypoint` never passed `cwd`, though `_resolve_argv` documents "steps run with
  cwd=pkg.directory" and `build_package` honours it. A package that declared its entrypoints
  package-relative -- what the contract describes -- failed every capsule at `parse` with "no such
  file", naming a file that was present. Twelve agent rounds went into chasing it.
* The misrooting rescue only fired for the declared tool's own basename, so it was dead for a package
  that declares a script per command, and dead entirely when `entrypoints.tool` names an interpreter.
* An unknown placeholder reached the package verbatim and surfaced as
  `FileNotFoundError: '{input_json}'` from inside the submission's own traceback.
"""
from __future__ import annotations

import textwrap

import pytest

from merlin.targetgen import oot_runner
from merlin.targetgen.oot_runner import CertFailure, _resolve_argv, load_package, run_entrypoint


def _pkg(tmp_path, argv, *, tool="mlir_oot/tool.py"):
    d = tmp_path / "submission"
    (d / "mlir_oot").mkdir(parents=True)
    (d / "mlir_oot" / "tool.py").write_text("#!/usr/bin/env python3\n")
    (d / "mlir_oot" / "parse.py").write_text(textwrap.dedent("""
        import sys, pathlib
        src = pathlib.Path(sys.argv[1]).read_text()
        pathlib.Path(sys.argv[2]).write_text(src)
        print("PARSED_OK")
    """))
    (d / "manifest.yaml").write_text(textwrap.dedent(f"""
        artifact_type: mlir_oot_target_backend
        target: t
        language: python
        integrity_exempt: false
        authoring:
          mode: agent_generated_from_recipe
        entrypoints:
          tool: {tool}
        commands:
          parse:
            argv: {argv}
          lower_interface_to_target:
            argv: ["python3", "mlir_oot/tool.py", "{{input_mlir}}"]
          emit_command_buffer:
            argv: ["python3", "mlir_oot/tool.py", "{{input_mlir}}", "{{output_json}}"]
          lower_target_to_llvm:
            argv: ["python3", "mlir_oot/tool.py", "{{input_mlir}}"]
    """))
    return load_package(str(d))


def test_a_package_relative_entrypoint_runs(tmp_path):
    """The shape the contract describes must work."""
    pkg = _pkg(tmp_path, '["python3", "mlir_oot/parse.py", "{input_mlir}", "{output_json}"]')
    src = tmp_path / "in.mlir"; src.write_text("module {}")
    out = tmp_path / "out.json"
    r = run_entrypoint(pkg, "parse", src, out, timeout=60)
    assert r.returncode == 0, r.stderr
    assert out.read_text() == "module {}"


def test_a_relative_input_path_still_resolves(tmp_path, monkeypatch):
    """Moving cwd to the package must not break a caller that passes a relative capsule path."""
    pkg = _pkg(tmp_path, '["python3", "mlir_oot/parse.py", "{input_mlir}", "{output_json}"]')
    src = tmp_path / "in.mlir"; src.write_text("module {}")
    monkeypatch.chdir(tmp_path)
    r = run_entrypoint(pkg, "parse", "in.mlir", tmp_path / "o.json", timeout=60)
    assert r.returncode == 0, r.stderr


def test_a_submission_prefixed_path_is_rescued(tmp_path):
    """The package root IS the submission dir, so `submission/x` is unambiguously double-rooted."""
    pkg = _pkg(tmp_path, '["python3", "./submission/mlir_oot/parse.py", "{input_mlir}", "{output_json}"]',
               tool="/usr/bin/python3")
    argv = _resolve_argv(pkg, "parse", tmp_path / "in.mlir", tmp_path / "o.json")
    assert "mlir_oot/parse.py" in argv, argv
    assert not any(a.startswith("./submission/") for a in argv), argv


def test_a_real_sibling_reference_is_left_alone(tmp_path):
    """The rescue must only fire for paths that do not exist as written."""
    pkg = _pkg(tmp_path, '["python3", "mlir_oot/parse.py", "{input_mlir}", "{output_json}"]')
    argv = _resolve_argv(pkg, "parse", tmp_path / "in.mlir", tmp_path / "o.json")
    assert "mlir_oot/parse.py" in argv


def test_an_unknown_placeholder_fails_closed(tmp_path):
    pkg = _pkg(tmp_path, '["python3", "mlir_oot/parse.py", "{input_json}", "{output_json}"]')
    with pytest.raises(CertFailure) as e:
        _resolve_argv(pkg, "parse", tmp_path / "in.mlir", tmp_path / "o.json")
    msg = str(e.value)
    assert "{input_json}" in msg, msg
    assert "{input_mlir}" in msg, "the message must name the placeholders that DO exist"


def test_the_known_placeholders_do_not_trip_the_guard(tmp_path):
    pkg = _pkg(tmp_path, '["python3", "{tool}", "{input_mlir}", "{output_json}"]')
    argv = _resolve_argv(pkg, "parse", tmp_path / "in.mlir", tmp_path / "o.json")
    assert not any("{" in a for a in argv), argv


def test_cli_accepts_gsim_as_a_cycle_accurate_oracle(monkeypatch):
    """GSIM's internal adapter is useless to automation if argparse rejects its public spelling."""
    observed = {}

    def fake_certify(*_args, **kwargs):
        observed.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(oot_runner, "certify", fake_certify)
    assert oot_runner.main([
        "--package", "submission", "--input", "case.interface.mlir", "--run-id", "g0",
        "--simulator", "gsim",
    ]) == 0
    assert observed["simulator"] == "gsim"
