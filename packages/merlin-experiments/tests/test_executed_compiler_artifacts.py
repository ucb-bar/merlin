"""Original completed-call artifact binding; synthetic runs are not device evidence."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from test_executed_program_retention import _build, _result

from merlin.common import invocation_record as I
from merlin.targetgen import executed_program as E
from merlin.targetgen.contract.readback_policy import canonical_sha256


def _oracle():
    return b"original complete outputs\n"


def fixture(tmp_path):
    work = tmp_path / "generated"
    build = _build(work, b"original ELF", b"original harness")
    cb = {"tensors": {}, "params": {"signed_stimulus": [-3, 3]}}
    build.update(status="completed", command_buffer_sha256=canonical_sha256(cb))
    console = _oracle()
    with I.observe_call(
        work, stage="execution", function=_oracle, arguments={}, inputs=(work / "package_kernel.elf",)
    ) as observed:
        observed.returned(stdout=console)
    result = _result(work, build, "1" * 64)
    result.update(
        console=console,
        execution_invocation={
            "path": str(observed.path),
            "sha256": hashlib.sha256(observed.path.read_bytes()).hexdigest(),
        },
    )
    return work, result, {"command_buffer": cb, "lowered_mlir": "module { /* exact compiler source */ }"}


def test_v2_retains_actual_source_kernel_console_and_original_invocation(tmp_path):
    work, result, arguments = fixture(tmp_path)
    original = copy.deepcopy(result)
    assert E.retain(result, sim="functional-fixture", workdir=work, compiler_artifacts=arguments) is result
    (record,) = E.collect(work)
    artifacts = record["compiler_artifacts"]
    assert record["schema"] == E.SCHEMA_V2 and artifacts["status"] == "retained"
    files = {role: work / pin["path"] for role, pin in artifacts["members"].items()}
    assert json.loads(files["command_buffer"].read_bytes()) == arguments["command_buffer"]
    assert files["lowered_mlir"].read_text() == arguments["lowered_mlir"]
    assert files["console"].read_bytes() == result["console"]
    assert files["kernel_object"].read_bytes() == b"kernel object"
    assert files["invocation"].read_bytes() == observed_bytes(result)
    _build(work, b"next tier ELF", b"next tier harness", b"next tier object")
    assert E.collect(work)[0]["compiler_artifacts"]["status"] == "retained"
    assert files["kernel_object"].read_bytes() == b"kernel object"
    assert result == original


def observed_bytes(result):
    from pathlib import Path

    return Path(result["execution_invocation"]["path"]).read_bytes()


@pytest.mark.parametrize("change", ["console", "buffer", "invocation", "kernel", "missing"])
def test_inconsistent_original_execution_has_no_compiler_artifact_authority(tmp_path, change):
    work, result, arguments = fixture(tmp_path)
    if change == "console":
        result["console"] = b"different outputs"
    elif change == "buffer":
        arguments["command_buffer"]["params"]["signed_stimulus"] = [0, 0]
    elif change == "invocation":
        result["execution_invocation"]["sha256"] = "f" * 64
    elif change == "kernel":
        (work / "kernel.o").write_bytes(b"changed object")
    else:
        del result["execution_invocation"]
    E.retain(result, sim="functional-fixture", workdir=work, compiler_artifacts=arguments)
    (record,) = E.collect(work)
    if record["status"] == "retained":
        assert record["compiler_artifacts"]["status"] == "unretained"
    else:
        assert record["status"] == "unretained"


def test_changed_retained_compiler_artifact_is_not_a_stable_observation(tmp_path):
    work, result, arguments = fixture(tmp_path)
    E.retain(result, sim="functional-fixture", workdir=work, compiler_artifacts=arguments)
    (record,) = E.collect(work)
    (work / record["compiler_artifacts"]["members"]["lowered_mlir"]["path"]).write_text("changed")
    assert E.collect(work)[0]["status"] == "copy_changed"
