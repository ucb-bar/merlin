"""Installed TargetGen entrypoints for the selection-only native package."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.semantic_compiler.allocate import StorageBank
from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.model import ConstantBinding, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.rules import AxisEquality, InstructionDescriptor
from merlin.semantic_compiler.snapshot import NativeTargetProfile
from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication


def _invoke(*arguments: object) -> subprocess.CompletedProcess[str]:
    script = Path(sys.executable).with_name("merlin-targetgen")
    assert script.is_file(), "install the Merlin CLI entrypoint before this integration test"
    return subprocess.run(
        [str(script), *(str(item) for item in arguments)], text=True, capture_output=True, timeout=180, check=False
    )


def test_native_publication_refuses_changed_execution_plan(tmp_path: Path) -> None:
    binary = b"\x13\x00\x00\x00"
    plan = b'{"inputs":[],"outputs":[]}\n'
    (tmp_path / "program.bin").write_bytes(binary)
    (tmp_path / "execution_plan.json").write_bytes(plan)
    manifest = {
        "engine": "merlin_native",
        "request_digest": "source-1",
        "target_identity": "target-1",
        "binary_sha256": hashlib.sha256(binary).hexdigest(),
        "execution_plan_sha256": hashlib.sha256(plan).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    kwargs = {"engine": "merlin_native", "request_digest": "source-1", "target_identity": "target-1"}
    verify_native_publication(tmp_path, manifest, **kwargs)
    (tmp_path / "execution_plan.json").write_bytes(plan + b" ")
    with pytest.raises(ValueError, match="execution plan differs"):
        verify_native_publication(tmp_path, manifest, **kwargs)
    (tmp_path / "execution_plan.json").unlink()
    with pytest.raises(ValueError, match="plan and manifest identity disagree"):
        verify_native_publication(tmp_path, manifest, **kwargs)


def test_native_publication_checks_every_program_set_segment(tmp_path: Path) -> None:
    rows = []
    for index in range(2):
        root = tmp_path / "segments" / f"{index:03d}"
        root.mkdir(parents=True)
        binary = bytes((index, 0, 0, 0))
        plan = json.dumps({"output": index}).encode()
        (root / "program.bin").write_bytes(binary)
        (root / "execution_plan.json").write_bytes(plan)
        detail = {
            "engine": "merlin_native",
            "request_digest": f"segment-{index}",
            "target_identity": "target-1",
            "binary_sha256": hashlib.sha256(binary).hexdigest(),
            "execution_plan_sha256": hashlib.sha256(plan).hexdigest(),
        }
        serialized = json.dumps(detail).encode()
        (root / "manifest.json").write_bytes(serialized)
        rows.append(
            {
                "index": index,
                "path": f"segments/{index:03d}",
                "request_digest": f"segment-{index}",
                "manifest_sha256": hashlib.sha256(serialized).hexdigest(),
                "binary_sha256": detail["binary_sha256"],
            }
        )
    manifest = {
        "artifact_kind": "program_set",
        "engine": "merlin_native",
        "request_digest": "source-1",
        "target_identity": "target-1",
        "segments": rows,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    kwargs = {"engine": "merlin_native", "request_digest": "source-1", "target_identity": "target-1"}
    verify_native_publication(tmp_path, manifest, **kwargs)
    binary_path = tmp_path / "segments/001/program.bin"
    binary_path.write_bytes(b"\x13\x00\x00\x00")
    with pytest.raises(ValueError, match="native emitted binary differs"):
        verify_native_publication(tmp_path, manifest, **kwargs)
    binary_path.write_bytes(bytes((1, 0, 0, 0)))
    (tmp_path / "segments/extra").mkdir()
    with pytest.raises(ValueError, match="unaccounted segment"):
        verify_native_publication(tmp_path, manifest, **kwargs)
    (tmp_path / "segments/extra").rmdir()
    rows[1]["path"] = "segments/../001"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe or repeated segment path"):
        verify_native_publication(tmp_path, manifest, **kwargs)


def test_native_compilation_failure_keeps_search_status() -> None:
    failure = NativeCompilationError("resource_limit", "bounded search exhausted")
    assert failure.status == "resource_limit"
    assert failure.reason == "bounded search exhausted"
    with pytest.raises(ValueError, match="known status"):
        NativeCompilationError("selected", "cannot be a failure")


def test_installed_native_build_select_and_failure_replace_stale_result(tmp_path: Path) -> None:
    dtype = TensorType((1,), "i32", "exact-i32")
    descriptor = InstructionDescriptor(
        "identity",
        "identity",
        ("external",),
        "external",
        "i32",
        "exact-i32",
        (1,),
        input_dtypes=("i32",),
        input_numerical_policies=("exact-i32",),
    )
    add = InstructionDescriptor(
        "add",
        "add",
        ("external", "external"),
        "external",
        "i32",
        "exact-i32",
        (1,),
        input_dtypes=("i32", "i32"),
        input_numerical_policies=("exact-i32", "exact-i32"),
    )
    profile = NativeTargetProfile("synthetic-cli-1", (descriptor, add), (StorageBank("external", "dram", 3, "tile"),))
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), dtype, effect="input"), SemanticNode("y", "identity", ("x",), dtype)),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity=profile.target_identity,
        source_identity="fresh-public-kernel",
    )
    profile_path, request_path, abi_path = (tmp_path / name for name in ("profile.json", "request.json", "abi.json"))
    profile_path.write_text(json.dumps(profile.record()))
    request_path.write_text(json.dumps(request.record()))
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [1]}))
    snapshot, output = tmp_path / "native-snapshot", tmp_path / "selection.json"
    built = _invoke(
        "native-build",
        "--engine",
        "merlin_native",
        "--profile",
        profile_path,
        "--cargo-target-dir",
        tmp_path / "cargo-target",
        "--source-revision",
        "public-cli-test",
        "--out",
        snapshot,
    )
    assert built.returncode == 0, built.stderr + built.stdout
    assert json.loads(built.stdout)["status"] == "selection_only"
    assert (snapshot / "bin/merlin-egg-bridge").is_file()

    selected = _invoke(
        "native-select",
        "--engine",
        "merlin_native",
        "--snapshot",
        snapshot,
        "--request",
        request_path,
        "--abi",
        abi_path,
        "--out",
        output,
    )
    assert selected.returncode == 0, selected.stderr + selected.stdout
    report = json.loads(output.read_text())
    assert report["status"] == "selected" and report["engine"] == "merlin_native"
    assert report["scope"] == "selection_only" and report["check_fingerprint"]
    assert report["allocation"]["addresses"] and report["selected_graph"]
    assert report["diagnostic_only"] is False and report["diagnostic_ablations"] == []

    limits = {
        "schema": "merlin.native_search_limits.v1",
        "iterations": 8,
        "egraph_nodes": 5000,
        "candidate_nodes": 1,
        "candidates": 256,
        "orders_per_candidate": 32,
        "solver_timeout_ms": 5000,
        "wall_timeout_s": 60,
    }
    limits_path = tmp_path / "search-limits.json"
    limits_path.write_text(json.dumps(limits))
    chain = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), dtype, effect="input"),
            SemanticNode("a", "add", ("x", "x"), dtype),
            SemanticNode("y", "add", ("a", "x"), dtype),
        ),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity=profile.target_identity,
    )
    request_path.write_text(json.dumps(chain.record()))
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [2]}))
    exhausted = _invoke(
        "native-select",
        "--engine",
        "merlin_native",
        "--snapshot",
        snapshot,
        "--request",
        request_path,
        "--abi",
        abi_path,
        "--search-limits",
        limits_path,
        "--out",
        output,
    )
    assert exhausted.returncode == 2, exhausted.stderr + exhausted.stdout
    exhausted_report = json.loads(output.read_text())
    assert exhausted_report["status"] == "resource_limit"
    assert exhausted_report["search_limits"] == limits
    assert exhausted_report["candidate_attempts"] == 0
    assert exhausted_report["selected_graph"] is None

    limits["candidate_nodes"] = 2
    limits_path.write_text(json.dumps(limits))
    recovered = _invoke(
        "native-select",
        "--engine",
        "merlin_native",
        "--snapshot",
        snapshot,
        "--request",
        request_path,
        "--abi",
        abi_path,
        "--search-limits",
        limits_path,
        "--out",
        output,
    )
    assert recovered.returncode == 0, recovered.stderr + recovered.stdout
    recovered_report = json.loads(output.read_text())
    assert recovered_report["status"] == "selected"
    assert recovered_report["search_limits"] == limits

    limits["candidate_nodes"] = True
    limits_path.write_text(json.dumps(limits))
    invalid = _invoke(
        "native-select",
        "--engine",
        "merlin_native",
        "--snapshot",
        snapshot,
        "--request",
        request_path,
        "--abi",
        abi_path,
        "--search-limits",
        limits_path,
        "--out",
        output,
    )
    assert invalid.returncode == 2
    assert json.loads(output.read_text())["status"] == "compile_error"
    assert "selected_graph" not in json.loads(output.read_text())
    request_path.write_text(json.dumps(request.record()))
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [1]}))

    diagnostic = _invoke(
        "native-select",
        "--engine",
        "merlin_native",
        "--snapshot",
        snapshot,
        "--request",
        request_path,
        "--abi",
        abi_path,
        "--ablation",
        "disable_structural_rewrites",
        "--out",
        output,
    )
    assert diagnostic.returncode == 0, diagnostic.stderr + diagnostic.stdout
    diagnostic_report = json.loads(output.read_text())
    assert diagnostic_report["status"] == "selected"
    assert diagnostic_report["qualification"] == "component_diagnostic"
    assert diagnostic_report["diagnostic_only"] is True
    assert diagnostic_report["diagnostic_ablations"] == ["disable_structural_rewrites"]

    with_constant = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), dtype, effect="input"),
            SemanticNode("c", "constant", (), dtype, effect="constant"),
            SemanticNode("sum", "add", ("x", "c"), dtype),
        ),
        outputs=("sum",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity=profile.target_identity,
        constants=(ConstantBinding("c", "i32-le", "feffffff"),),
    )
    request_path.write_text(json.dumps(with_constant.record()))
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [2]}))
    selected_constant = _invoke(
        "native-select",
        "--engine",
        "merlin_native",
        "--snapshot",
        snapshot,
        "--request",
        request_path,
        "--abi",
        abi_path,
        "--out",
        output,
    )
    assert selected_constant.returncode == 0, selected_constant.stderr + selected_constant.stdout
    requirement = json.loads(output.read_text())["constant_requirements"]
    assert requirement == [
        {
            "value_id": 1,
            "storage": "external",
            "address": 1,
            "node_id": "c",
            "encoding": "i32-le",
            "data_hex": "feffffff",
        }
    ]

    broken = with_constant.record()
    broken["target_identity"] = "changed-target"
    request_path.write_text(json.dumps(broken))
    failed = _invoke(
        "native-select",
        "--engine",
        "merlin_native",
        "--snapshot",
        snapshot,
        "--request",
        request_path,
        "--abi",
        abi_path,
        "--out",
        output,
    )
    assert failed.returncode == 2
    replacement = json.loads(output.read_text())
    assert replacement["status"] == "compile_error" and "selected_graph" not in replacement
    assert "target identity differs" in replacement["reason"]

    wrong_engine = _invoke(
        "native-select", "--engine", "act_reference", "--snapshot", snapshot, "--request", request_path, "--out", output
    )
    assert wrong_engine.returncode == 2 and "invalid choice" in wrong_engine.stderr


def test_installed_native_select_parses_linalg_with_exact_source_identity(tmp_path: Path) -> None:
    source = """module { func.func @work(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
      %c: tensor<2x2xi32>) -> tensor<2x2xi32> {
      %r = linalg.matmul ins(%a, %b : tensor<2x2xi32>, tensor<2x2xi32>)
        outs(%c : tensor<2x2xi32>) -> tensor<2x2xi32>
      func.return %r : tensor<2x2xi32>
    } }"""
    input_path = tmp_path / "kernel.mlir"
    input_path.write_bytes(source.replace("\n", "\r\n").encode())
    target_identity = "synthetic-cli-linalg-1"
    translated = translate_linalg_text(input_path.read_bytes().decode(), entry="work", target_identity=target_identity)
    descriptor = InstructionDescriptor(
        "contract",
        "matmul_accumulate",
        ("external",) * 3,
        "external",
        "i32",
        "i32-wrap-k-ascending",
        (2,),
        input_dtypes=("i32", "i32", "i32"),
        input_numerical_policies=("i32-wrap-k-ascending",) * 3,
        input_ranks=(2, 2, 2),
        index_maps=translated.request.nodes[-1].index_maps,
        shape_contract="relations",
        shape_equalities=(
            AxisEquality("in0", 0, "out", 0),
            AxisEquality("in0", 1, "in1", 0),
            AxisEquality("in1", 1, "out", 1),
            AxisEquality("in2", 0, "out", 0),
            AxisEquality("in2", 1, "out", 1),
        ),
    )
    profile = NativeTargetProfile(target_identity, (descriptor,), (StorageBank("external", "dram", 5, "tile"),))
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile.record()))
    snapshot, output = tmp_path / "snapshot", tmp_path / "selection.json"
    bridge = repo_root() / "src/merlin/semantic_compiler/egg_bridge"
    built = _invoke(
        "native-build",
        "--engine",
        "merlin_native",
        "--profile",
        profile_path,
        "--crate",
        bridge,
        "--cargo-target-dir",
        tmp_path / "cargo-target",
        "--source-revision",
        "public-linalg-cli-test",
        "--out",
        snapshot,
    )
    assert built.returncode == 0, built.stderr + built.stdout

    selected = _invoke(
        "native-select",
        "--engine",
        "merlin_native",
        "--snapshot",
        snapshot,
        "--linalg",
        input_path,
        "--linalg-entry",
        "work",
        "--out",
        output,
    )
    assert selected.returncode == 0, selected.stderr + selected.stdout
    report = json.loads(output.read_text())
    assert report["status"] == "selected" and report["candidate_digest"]
    assert report["source_kind"] == "parsed_linalg"
    assert report["source_identity"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert report["source_operations"] == ["linalg.matmul", "func.return"]

    missing_entry = _invoke(
        "native-select", "--engine", "merlin_native", "--snapshot", snapshot, "--linalg", input_path, "--out", output
    )
    assert missing_entry.returncode == 2
    assert json.loads(output.read_text())["status"] == "compile_error"
    assert "selected_graph" not in json.loads(output.read_text())

    invalid = input_path.read_bytes().replace(b"i32", b"f32")
    input_path.write_bytes(invalid)
    unsupported = _invoke(
        "native-select",
        "--engine",
        "merlin_native",
        "--snapshot",
        snapshot,
        "--linalg",
        input_path,
        "--linalg-entry",
        "work",
        "--out",
        output,
    )
    assert unsupported.returncode == 2
    assert json.loads(output.read_text())["status"] == "unsupported_semantics"
    assert "selected_graph" not in json.loads(output.read_text())
