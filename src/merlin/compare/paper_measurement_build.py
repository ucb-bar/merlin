"""Rebuild or admit the selected executable and retain its exact build inputs."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .paper_measurement_contract import _contract
from .paper_measurement_protocol import _bound_file, _canonical_sha, _closed, _fsync, _is_sha, _sha_bytes, _sha_file
from .paper_model_object_builder import (
    expected_recipe,
    object_build_argv,
    regenerate_model_object,
    stage_compiler_input,
)
from .paper_toolchain_authority import verify_build_tool


def _build_executable(root: Path, contract: Mapping[str, Any], destination: Path, remaining_seconds) -> dict[str, Any]:
    build = contract["build"]
    authority_path = _bound_file(root, build["toolchain_authority"], "independent toolchain authority")
    tool = _bound_file(root, build["tool"], "build tool")
    build_tool_identity = verify_build_tool(
        tool,
        authority_path=authority_path,
        authority_sha256=str(build["toolchain_authority"]["sha256"]),
        target=str(contract["target"]),
        expected_identity_sha256=str(build["build_tool_identity_sha256"]),
    )
    sources = {name: _bound_file(root, ref, f"build source {name}") for name, ref in build["sources"].items()}
    inputs = {name: _bound_file(root, ref, f"build input {name}") for name, ref in build["inputs"].items()}
    regeneration = None
    if contract["registry_id"] != "unit_test_v1":
        builder = inputs["object_builder"]
        if _sha_file(builder) != _sha_file(Path(__file__).with_name("paper_model_object_builder.py")):
            raise ValueError("retained model-object builder differs from shipped registry source")
        package = json.loads(inputs["package_receipt"].read_text(encoding="utf-8"))
        derived = destination.parent / "registry-regenerated-model-object.o"
        regeneration = regenerate_model_object(
            recipe=str(package["object_recipe"]),
            registry_id=str(contract["registry_id"]),
            target=str(contract["target"]),
            compiler_input=inputs["compiler_input"],
            tool=tool,
            output=derived,
            source_identity_sha256=str(package["compiler_or_framework_source_sha256"]),
            capture_sha256=str(contract["artifact_sha256"]),
            runtime_artifact_sha256=str(package["runtime_artifact_sha256"]),
            timeout_seconds=min(float(build["timeout_seconds"]), remaining_seconds()),
        )
        if (
            _sha_file(derived) != _sha_file(sources["model_object"])
            or regeneration["generated_source_sha256"] != package["generated_model_source_sha256"]
        ):
            raise ValueError("bound model object differs from registry-owned regeneration")
    replacements = {
        "{tool}": str(tool),
        "{output}": str(destination),
        **{f"{{source:{name}}}": str(value) for name, value in sources.items()},
        **{f"{{input:{name}}}": str(value) for name, value in inputs.items()},
    }
    argv = [replacements.get(part, part) for part in build["argv"]]
    started = time.monotonic_ns()
    if contract["registry_id"] == "executorch_v1" and contract["target"] != "unit-test":
        shutil.copy2(derived, destination)
        completed_returncode, completed_stdout, completed_stderr = 0, b"", b""
    else:
        completed = subprocess.run(
            argv,
            cwd=root,
            env=dict(build["environment"]),
            capture_output=True,
            timeout=min(float(build["timeout_seconds"]), remaining_seconds()),
            check=False,
        )
        completed_returncode = completed.returncode
        completed_stdout, completed_stderr = completed.stdout, completed.stderr
    ended = time.monotonic_ns()
    if completed_returncode != 0 or not destination.is_file() or not destination.read_bytes().startswith(b"\x7fELF"):
        raise ValueError(
            f"frozen executable build failed rc={completed_returncode}: "
            f"{completed_stderr.decode(errors='replace')[-500:]}"
        )
    digest = _sha_file(destination)
    if digest != build["expected_executable_sha256"]:
        raise ValueError("materialized executable differs from the frozen build receipt")
    result = {
        "registry_id": contract["registry_id"],
        "operation": (
            "sealed_executable_verification"
            if contract["registry_id"] == "executorch_v1" and contract["target"] != "unit-test"
            else "compiler_rebuild"
        ),
        "argv": argv,
        "command_sha256": _canonical_sha(argv),
        "exit_code": completed_returncode,
        "started_monotonic_ns": started,
        "ended_monotonic_ns": ended,
        "stdout_sha256": _sha_bytes(completed_stdout),
        "stderr_sha256": _sha_bytes(completed_stderr),
        "executable_sha256": digest,
        "tool_sha256": _sha_file(tool),
        "toolchain_authority_sha256": _sha_file(authority_path),
        "build_tool_identity_sha256": build_tool_identity,
        "source_sha256": {name: _sha_file(value) for name, value in sources.items()},
        "input_sha256": {name: _sha_file(value) for name, value in inputs.items()},
    }
    if regeneration is not None:
        result["object_regeneration"] = regeneration
    return result


def _validate_build_receipt(root: Path, contract: Mapping[str, Any], value: object) -> Mapping[str, Any]:
    fields = {
        "registry_id",
        "operation",
        "argv",
        "command_sha256",
        "exit_code",
        "started_monotonic_ns",
        "ended_monotonic_ns",
        "stdout_sha256",
        "stderr_sha256",
        "executable_sha256",
        "tool_sha256",
        "toolchain_authority_sha256",
        "build_tool_identity_sha256",
        "source_sha256",
        "input_sha256",
    }
    if contract["registry_id"] != "unit_test_v1":
        fields.add("object_regeneration")
    receipt = _closed(value, fields, "controller build receipt")
    build = contract["build"]
    authority_path = _bound_file(root, build["toolchain_authority"], "independent toolchain authority")
    tool = _bound_file(root, build["tool"], "build tool")
    verified_tool_identity = verify_build_tool(
        tool,
        authority_path=authority_path,
        authority_sha256=str(build["toolchain_authority"]["sha256"]),
        target=str(contract["target"]),
        expected_identity_sha256=str(build["build_tool_identity_sha256"]),
    )
    sources = {name: _bound_file(root, ref, f"build source {name}") for name, ref in build["sources"].items()}
    inputs = {name: _bound_file(root, ref, f"build input {name}") for name, ref in build["inputs"].items()}
    argv = receipt["argv"]
    replacements = {
        "{tool}": str(tool),
        **{f"{{source:{name}}}": str(path) for name, path in sources.items()},
        **{f"{{input:{name}}}": str(path) for name, path in inputs.items()},
    }
    expected_argv = [replacements.get(part, part) for part in build["argv"]]
    output_index = build["argv"].index("{output}")
    expected_argv[output_index] = (
        argv[output_index] if isinstance(argv, list) and len(argv) == len(expected_argv) else ""
    )
    if (
        not isinstance(argv, list)
        or not all(isinstance(part, str) for part in argv)
        or argv != expected_argv
        or receipt["registry_id"] != contract["registry_id"]
        or receipt["operation"]
        != (
            "sealed_executable_verification"
            if (contract["registry_id"] == "executorch_v1" and contract["target"] != "unit-test")
            else "compiler_rebuild"
        )
        or receipt["command_sha256"] != _canonical_sha(argv)
        or receipt["exit_code"] != 0
        or receipt["ended_monotonic_ns"] <= receipt["started_monotonic_ns"]
        or receipt["executable_sha256"] != build["expected_executable_sha256"]
        or receipt["tool_sha256"] != _sha_file(tool)
        or receipt["toolchain_authority_sha256"] != _sha_file(authority_path)
        or receipt["build_tool_identity_sha256"] != verified_tool_identity
        or receipt["source_sha256"] != {name: _sha_file(path) for name, path in sources.items()}
        or receipt["input_sha256"] != {name: _sha_file(path) for name, path in inputs.items()}
        or not all(
            _is_sha(receipt[field]) for field in ("stdout_sha256", "stderr_sha256", "executable_sha256", "tool_sha256")
        )
        or argv[0]
        != (
            "verify_executorch_sealed_session"
            if (contract["registry_id"] == "executorch_v1" and contract["target"] != "unit-test")
            else str(tool)
        )
        or any(
            argv.count(str(path))
            != (
                1
                if (
                    contract["registry_id"] != "executorch_v1"
                    or contract["target"] == "unit-test"
                    or name == "model_object"
                )
                else 0
            )
            for name, path in sources.items()
        )
        or (contract["registry_id"] == "unit_test_v1" and any(argv.count(str(path)) != 1 for path in inputs.values()))
        or (contract["registry_id"] != "unit_test_v1" and any(argv.count(str(path)) != 0 for path in inputs.values()))
    ):
        raise ValueError("controller build receipt does not reconcile to frozen build plan")
    if contract["registry_id"] != "unit_test_v1":
        regeneration = _closed(
            receipt["object_regeneration"],
            {"recipe", "compiler_input_sha256", "generated_source_sha256", "object_build_argv", "model_object_sha256"},
            "model-object regeneration receipt",
        )
        package = json.loads(inputs["package_receipt"].read_text(encoding="utf-8"))
        if (
            regeneration["recipe"] != package["object_recipe"]
            or regeneration["compiler_input_sha256"] != _sha_file(inputs["compiler_input"])
            or regeneration["generated_source_sha256"] != package["generated_model_source_sha256"]
            or regeneration["object_build_argv"] != object_build_argv(str(package["object_recipe"]))
            or regeneration["model_object_sha256"] != _sha_file(sources["model_object"])
        ):
            raise ValueError("model-object regeneration receipt does not reconcile")
    return receipt


def _retain_contract(contract_path: Path, output: Path) -> Path:
    contract, _ = _contract(contract_path)
    root, document = contract_path.parent, dict(contract)
    resources = output / "inputs"
    resources.mkdir()

    def retain(ref: object, label: str, name: str) -> dict[str, str]:
        source = _bound_file(root, ref, label)
        destination = resources / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        _fsync(destination)
        return {"path": destination.relative_to(output).as_posix(), "sha256": _sha_file(destination)}

    document["artifact"] = retain(contract["artifact"], "cell artifact", "artifact")
    document["study_spec"] = retain(contract["study_spec"], "canonical frozen study", "frozen-study.yaml")
    document["backend_template"] = retain(
        contract["backend_template"], "frozen backend template", "backend-template.yaml"
    )
    document["reference_output"] = retain(contract["reference_output"], "reference output", "reference-output")
    document["session_manifest"] = retain(contract["session_manifest"], "session manifest", "session-manifest.json")
    document["measurement_io_receipt"] = retain(
        contract["measurement_io_receipt"], "measurement I/O generation receipt", "measurement-io-receipt.json"
    )
    document["board_probe_source"] = retain(
        contract["board_probe_source"], "board probe source", "paper_k1_board_probe.c"
    )
    document["inputs"] = {
        name: retain(ref, f"cell input {name}", f"runtime/{name}") for name, ref in contract["inputs"].items()
    }
    build = dict(contract["build"])
    build["tool"] = retain(build["tool"], "build tool", "build/tool")
    build["toolchain_authority"] = retain(
        build["toolchain_authority"], "independent toolchain authority", "build/inputs/toolchain-authority.json"
    )
    build["sources"] = {
        name: retain(ref, f"build source {name}", f"build/sources/{name}{Path(str(ref['path'])).suffix}")
        for name, ref in build["sources"].items()
    }
    retained_inputs = {}
    for name, ref in build["inputs"].items():
        if name == "compiler_input" and contract["registry_id"] in {"merlin_compile_v1", "executorch_v1"}:
            source = _bound_file(root, ref, "build input compiler_input")
            destination = resources / "build/inputs/compiler_input"
            stage_compiler_input(
                source, destination, recipe=expected_recipe(str(contract["registry_id"]), str(contract["target"]))
            )
            _fsync(destination)
            retained_inputs[name] = {
                "path": destination.relative_to(output).as_posix(),
                "sha256": _sha_file(destination),
            }
        else:
            retained_inputs[name] = retain(ref, f"build input {name}", f"build/inputs/{name}")
    build["inputs"] = retained_inputs
    document["build"] = build
    if "session_abi" in contract:
        document["session_abi"] = {
            "protocol": "MRLNSES2",
            "descriptor": retained_inputs["session_descriptor"],
        }
    retained = output / "measurement_contract.yaml"
    retained.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    _fsync(retained)
    return retained


def _preflight_build_before_private_io(contract_path: Path) -> None:
    """Complete one exact build before receipt retention can copy private resources."""
    contract, _ = _contract(contract_path)
    deadline_ns = time.monotonic_ns() + contract["timeout_seconds"] * 1_000_000_000

    def remaining_seconds() -> float:
        remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining <= 0:
            raise TimeoutError("paper pre-private build exceeded its whole-cell deadline")
        return remaining

    with tempfile.TemporaryDirectory(prefix="merlin-paper-pre-private-build-") as temporary:
        _build_executable(contract_path.parent, contract, Path(temporary) / "cell_executable", remaining_seconds)
