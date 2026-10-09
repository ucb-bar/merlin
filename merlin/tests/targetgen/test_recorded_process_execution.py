"""Real native process attribution controls, without an ISA/runtime claim."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.common.paths import module_source_path
from merlin.targetgen.contract import process_execution as P
from merlin.targetgen.contract.build_service import file_digest


@pytest.fixture
def selected(tmp_path):
    executable = Path(sys.executable).resolve(strict=True)
    owner = module_source_path("merlin.targetgen.contract.process_execution")
    program = tmp_path / "original.py"
    program.write_text("import sys\nsys.stdout.write('original stdout\\n')\nsys.stderr.write('native stderr\\n')\n")
    process = P.RecordedProcessExecution(
        executable,
        ("-I", "-B", "{elf}"),
        tmp_path,
        (("PATH", "/usr/bin:/bin"), ("LC_ALL", "C")),
        tmp_path / "records",
        "stdout",
        tuple((str(path), file_digest(path)) for path in (executable, owner)),
    )
    return process, program


def test_actual_process_consumes_exact_selected_artifact_and_stream(selected):
    process, program = selected
    console = process.run_elf(program, timeout=5)
    joined = process.consumption(elf=program, console=console)
    actual = I.verify(Path(joined["record"]["path"]))
    I.require_environment(Path(joined["record"]["path"]), environment=dict(process.environment))
    assert actual["argv"] == [str(process.executable), "-I", "-B", str(program)]
    assert actual["inputs"] == [joined["elf"]]
    assert console == "original stdout\n"
    assert Path(joined["stderr"]["path"]).read_bytes() == b"native stderr\n"
    assert "unqualified" in joined["scope"]


def test_explicit_combined_stream_is_actual_merged_process_bytes(selected):
    process, program = selected
    process = replace(process, stream="combined")
    console = process.run_elf(program, timeout=5, capture_bytes=True)
    joined = process.consumption(elf=program, console=console)
    assert set(console.splitlines()) == {b"original stdout", b"native stderr"}
    assert Path(joined["stdout"]["path"]).read_bytes() == console
    assert Path(joined["stderr"]["path"]).read_bytes() == b""


def test_same_console_different_artifact_cannot_reopen(selected, tmp_path):
    process, program = selected
    console = process.run_elf(program, timeout=5)
    other = tmp_path / "different.py"
    other.write_text(program.read_text() + "# distinct bytes, same complete stream\n")
    with pytest.raises(ValueError, match="different or changed requested ELF"):
        process.consumption(elf=other, console=console)


def test_saved_actual_record_cannot_replace_an_unexecuted_owner(selected, tmp_path):
    process, program = selected
    console = process.run_elf(program, timeout=5)
    observed = process.consumption(elf=program, console=console)
    untouched = replace(process, record_root=tmp_path / "never-executed")
    untouched.record_root.mkdir(mode=0o700)
    (untouched.record_root / "saved-record.json").write_text(json.dumps(I.verify(Path(observed["record"]["path"]))))
    with pytest.raises(ValueError, match="no actual completed native execution"):
        untouched.consumption(elf=program, console=console)


def test_actual_redirected_process_refuses_its_contradictory_argv(selected, tmp_path, monkeypatch):
    process, program = selected
    other = tmp_path / "redirected.py"
    other.write_text(program.read_text() + "# same output, independently different program\n")
    native_run = I.run

    def redirect(argv, **kwargs):
        return native_run([*argv[:-1], str(other)], **kwargs)

    monkeypatch.setattr(P.I, "run", redirect)
    with pytest.raises(ValueError, match="actual tool/argv/input differs"):
        process.run_elf(program, timeout=5)
    records = tuple(process.record_root.rglob("invocation.json"))
    assert len(records) == 1
    actual = I.verify(records[0])
    assert actual["argv"][-1] == str(other)
    assert Path(actual["stdout"]["path"]).read_bytes() == b"original stdout\n"


def test_saved_console_without_engine_cannot_use_unrelated_process(selected, tmp_path, monkeypatch):
    process, program = selected
    I.run([sys.executable, "-I", "-c", "pass"], directory=tmp_path / "compiler", stage="unrelated_compile", check=True)
    monkeypatch.setattr(
        P.I, "run", lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, b"original stdout\n", b"")
    )
    with pytest.raises(ValueError, match="unique actual native invocation"):
        process.run_elf(program, timeout=5)


def test_saved_console_cannot_replace_actual_native_stdout(selected, monkeypatch):
    process, program = selected
    native_run = I.run

    def forged_console(argv, **kwargs):
        result = native_run(argv, **kwargs)
        return subprocess.CompletedProcess(result.args, result.returncode, b"forged saved output\n", result.stderr)

    monkeypatch.setattr(P.I, "run", forged_console)
    with pytest.raises(ValueError, match="returned console differs"):
        process.run_elf(program, timeout=5)


@pytest.mark.parametrize("changed", ["elf", "stdout", "stderr"])
def test_actual_input_or_output_mutation_refuses_reopen(selected, changed):
    process, program = selected
    console = process.run_elf(program, timeout=5)
    observed = process.consumption(elf=program, console=console)
    selected_file = program if changed == "elf" else Path(observed[changed]["path"])
    selected_file.write_bytes(b"mutated actual member")
    with pytest.raises(ValueError, match="changed requested ELF|input, dependency or product changed"):
        process.consumption(elf=program, console=console)


def test_actual_native_nonzero_and_timeout_remain_failed(selected):
    process, program = selected
    program.write_text("raise SystemExit(18)\n")
    with pytest.raises(subprocess.CalledProcessError) as failed:
        process.run_elf(program, timeout=5)
    assert failed.value.returncode == 18
    program.write_text("import time\ntime.sleep(10)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        process.run_elf(program, timeout=0.05)
    with pytest.raises(ValueError, match="no actual completed native execution"):
        process.consumption(elf=program, console="")


def test_closed_selection_requires_one_operand_and_owned_members(selected, tmp_path):
    process, program = selected
    for arguments in ((), ("{elf}", "{elf}"), ("--input={elf}",)):
        with pytest.raises(ValueError, match="exact explicit command"):
            replace(process, argv_template=arguments).verify()
    with pytest.raises(ValueError, match="selected tool or fixed source owner"):
        replace(process, source_pins=process.source_pins[:1]).verify()
    shared = tmp_path / "unsafe"
    shared.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="private and owned"):
        replace(process, record_root=shared).verify()
    alias = tmp_path / "alias.py"
    alias.symlink_to(program)
    with pytest.raises(ValueError, match="canonical explicit path|linked members"):
        process.run_elf(alias, timeout=5)
