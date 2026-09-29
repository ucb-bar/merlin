"""Installed TargetGen entrypoints for the selection-only native package."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.semantic_compiler.allocate import StorageBank
from merlin.semantic_compiler.model import ConstantBinding, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.rules import InstructionDescriptor
from merlin.semantic_compiler.snapshot import NativeTargetProfile


def _invoke(*arguments: object) -> subprocess.CompletedProcess[str]:
    script = Path(sys.executable).with_name("merlin-targetgen")
    assert script.is_file(), "install the Merlin CLI entrypoint before this integration test"
    return subprocess.run([str(script), *(str(item) for item in arguments)], text=True,
                          capture_output=True, timeout=180, check=False)


def test_installed_native_build_select_and_failure_replace_stale_result(tmp_path: Path) -> None:
    dtype = TensorType((1,), "i32", "exact-i32")
    descriptor = InstructionDescriptor(
        "identity", "identity", ("external",), "external", "i32", "exact-i32", (1,),
        input_dtypes=("i32",), input_numerical_policies=("exact-i32",),
    )
    add = InstructionDescriptor(
        "add", "add", ("external", "external"), "external", "i32", "exact-i32", (1,),
        input_dtypes=("i32", "i32"), input_numerical_policies=("exact-i32", "exact-i32"),
    )
    profile = NativeTargetProfile("synthetic-cli-1", (descriptor, add),
                                  (StorageBank("external", "dram", 3, "tile"),))
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), dtype, effect="input"),
               SemanticNode("y", "identity", ("x",), dtype)),
        outputs=("y",), output_storages=("external",), input_storages=(("x", "external"),),
        target_identity=profile.target_identity, source_identity="fresh-public-kernel",
    )
    profile_path, request_path, abi_path = (tmp_path / name for name in ("profile.json", "request.json", "abi.json"))
    profile_path.write_text(json.dumps(profile.record()))
    request_path.write_text(json.dumps(request.record()))
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [1]}))
    snapshot, output = tmp_path / "native-snapshot", tmp_path / "selection.json"
    bridge = repo_root() / "src/merlin/semantic_compiler/egg_bridge"

    built = _invoke("native-build", "--engine", "merlin_native", "--profile", profile_path,
                    "--crate", bridge, "--cargo-target-dir", tmp_path / "cargo-target",
                    "--source-revision", "public-cli-test", "--out", snapshot)
    assert built.returncode == 0, built.stderr + built.stdout
    assert json.loads(built.stdout)["status"] == "selection_only"
    assert (snapshot / "bin/merlin-egg-bridge").is_file()

    selected = _invoke("native-select", "--engine", "merlin_native", "--snapshot", snapshot,
                       "--request", request_path, "--abi", abi_path, "--out", output)
    assert selected.returncode == 0, selected.stderr + selected.stdout
    report = json.loads(output.read_text())
    assert report["status"] == "selected" and report["engine"] == "merlin_native"
    assert report["scope"] == "selection_only" and report["check_fingerprint"]
    assert report["allocation"]["addresses"] and report["selected_graph"]

    with_constant = KernelRequest(
        nodes=(SemanticNode("x", "input", (), dtype, effect="input"),
               SemanticNode("c", "constant", (), dtype, effect="constant"),
               SemanticNode("sum", "add", ("x", "c"), dtype)),
        outputs=("sum",), output_storages=("external",), input_storages=(("x", "external"),),
        target_identity=profile.target_identity,
        constants=(ConstantBinding("c", "i32-le", "feffffff"),),
    )
    request_path.write_text(json.dumps(with_constant.record()))
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [2]}))
    selected_constant = _invoke("native-select", "--engine", "merlin_native", "--snapshot", snapshot,
                                "--request", request_path, "--abi", abi_path, "--out", output)
    assert selected_constant.returncode == 0, selected_constant.stderr + selected_constant.stdout
    requirement = json.loads(output.read_text())["constant_requirements"]
    assert requirement == [{"value_id": 1, "storage": "external", "address": 1,
                            "node_id": "c", "encoding": "i32-le", "data_hex": "feffffff"}]

    broken = with_constant.record()
    broken["target_identity"] = "changed-target"
    request_path.write_text(json.dumps(broken))
    failed = _invoke("native-select", "--engine", "merlin_native", "--snapshot", snapshot,
                     "--request", request_path, "--abi", abi_path, "--out", output)
    assert failed.returncode == 2
    replacement = json.loads(output.read_text())
    assert replacement["status"] == "compile_error" and "selected_graph" not in replacement
    assert "target identity differs" in replacement["reason"]

    wrong_engine = _invoke("native-select", "--engine", "act_reference", "--snapshot", snapshot,
                           "--request", request_path, "--out", output)
    assert wrong_engine.returncode == 2 and "invalid choice" in wrong_engine.stderr
