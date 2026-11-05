"""Installed TargetGen entrypoints for the selection-only native package."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.semantic_compiler.allocate import StorageBank
from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.model import ConstantBinding, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.rules import AxisEquality, InstructionDescriptor
from merlin.semantic_compiler.snapshot import NativeTargetProfile
from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication


def test_precision_layouts_bind_the_exact_transformed_request(tmp_path: Path) -> None:
    from merlin.targetgen.cli import _bind_native_physical_layouts

    tensor = TensorType((2, 2), "i8", "selected-policy")
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), tensor, effect="input"),),
        outputs=("x",), output_storages=("external",),
        input_storages=(("x", "external"),), target_identity="synthetic-layout-target",
        lowering_policy="diagnostic", source_identity="captured-source",
    )
    row = {
        "schema": "merlin.physical_boundary_layouts.v2",
        "target_identity": request.target_identity,
        "source_identity": request.source_identity,
        "request_digest": request.digest(),
        "physical_extents": {"x": 4},
        "boundary_layouts": {"x": "row-major"},
        "output_layouts": ["row-major"],
    }
    path = tmp_path / "layouts.json"
    path.write_text(json.dumps(row))
    args = SimpleNamespace(
        physical_layouts=str(path), request=None, ordered_k_tile=None,
        ordered_m_tile=None, ordered_n_tile=None, outer_tile_index=None,
    )
    bound, stage = _bind_native_physical_layouts(args, request, precision_plan=object())
    assert bound.boundary_layout("x") == "row-major"
    assert bound.physical_extent("x") == 4
    assert json.loads(stage)["request_digest"] == request.digest()

    changed_policy = KernelRequest(
        nodes=(SemanticNode("x", "input", (), TensorType((2, 2), "i8", "changed-policy"),
                            effect="input"),),
        outputs=request.outputs, output_storages=request.output_storages,
        input_storages=request.input_storages, target_identity=request.target_identity,
        lowering_policy=request.lowering_policy, source_identity=request.source_identity,
    )
    with pytest.raises(ValueError, match="precision-transformed request"):
        _bind_native_physical_layouts(args, changed_policy, precision_plan=object())

    path.write_text(json.dumps({key: value for key, value in row.items() if key != "request_digest"}
                               | {"schema": "merlin.physical_boundary_layouts.v1"}))
    with pytest.raises(ValueError, match="exact-request v2"):
        _bind_native_physical_layouts(args, request, precision_plan=object())


def _invoke(*arguments: object) -> subprocess.CompletedProcess[str]:
    script = Path(sys.executable).with_name("merlin-targetgen")
    assert script.is_file(), "install the Merlin CLI entrypoint before this integration test"
    return subprocess.run(
        [str(script), *(str(item) for item in arguments)], text=True, capture_output=True, timeout=180, check=False
    )


def test_native_generation_and_new_kernel_work_with_act_access_blocked(tmp_path: Path) -> None:
    """Exercise a fresh profile and v3 kernel with ACT imports/processes denied."""
    (tmp_path / "sitecustomize.py").write_text(
        "import builtins, importlib.abc, os, subprocess, sys\n"
        "_denied = {'act', 'act_backend', 'taidl', 'taidl_to'}\n"
        "_original_import = builtins.__import__\n"
        "def _import(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in _denied: raise ImportError('ACT access blocked')\n"
        "    return _original_import(name, *args, **kwargs)\n"
        "builtins.__import__ = _import\n"
        "class _Finder(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in _denied: raise ImportError('ACT access blocked')\n"
        "sys.meta_path.insert(0, _Finder())\n"
        "_original_popen = subprocess.Popen\n"
        "def _popen(args, *other, **kwargs):\n"
        "    executable = args[0] if isinstance(args, (list, tuple)) else args.split()[0]\n"
        "    if os.path.basename(executable) in {'act', 'act-backend', 'act_backend'}:\n"
        "        raise OSError('ACT executable access blocked')\n"
        "    return _original_popen(args, *other, **kwargs)\n"
        "subprocess.Popen = _popen\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    script = Path(sys.executable).with_name("merlin-targetgen")

    tensor = TensorType((2, 2), "i8", "exact")
    request = KernelRequest(
        nodes=(
            SemanticNode("x", "input", (), tensor, effect="input"),
            SemanticNode("c", "constant", (), tensor, effect="constant"),
            SemanticNode("y", "add", ("x", "c"), tensor),
        ),
        outputs=("y",), output_storages=("external",),
        input_storages=(("x", "external"),), target_identity="synthetic-act-absent-v3",
        constants=(ConstantBinding("c", "i8", "01020304"),),
        physical_extents=(("x", 2), ("c", 2)),
    )
    descriptor = InstructionDescriptor(
        "add", "add", ("external", "external"), "external", "i8", "exact", (2,),
        input_dtypes=("i8", "i8"), input_numerical_policies=("exact", "exact"),
    )
    profile = NativeTargetProfile(
        request.target_identity, (descriptor,), (StorageBank("external", "memory", 5, "word"),)
    )
    (tmp_path / "profile.json").write_text(json.dumps(profile.record()))
    (tmp_path / "request.json").write_text(json.dumps(request.record()))
    (tmp_path / "abi.json").write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [4]}))

    def invoke(*arguments: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(script), *(str(item) for item in arguments)], env=env,
            text=True, capture_output=True, timeout=180, check=False,
        )

    snapshot = tmp_path / "snapshot"
    built = invoke(
        "native-build", "--engine", "merlin_native", "--profile", tmp_path / "profile.json",
        "--cargo-target-dir", tmp_path / "cargo", "--source-revision", "public-act-absent-v3",
        "--out", snapshot,
    )
    assert built.returncode == 0, built.stderr + built.stdout
    assert json.loads(built.stdout)["status"] == "selection_only"
    result_path = tmp_path / "selection.json"
    selected = invoke(
        "native-select", "--engine", "merlin_native", "--snapshot", snapshot,
        "--request", tmp_path / "request.json", "--abi", tmp_path / "abi.json", "--out", result_path,
    )
    assert selected.returncode == 0, selected.stderr + selected.stdout
    result = json.loads(result_path.read_text())
    assert result["status"] == "selected" and result["engine"] == "merlin_native"
    assert [(value["source_node"], value["extent"]) for value in result["selected_graph"]["values"]] == [
        ("x", 2), ("c", 2), ("y", 1)
    ]


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


def test_native_publication_checks_selected_stage_artifacts(tmp_path: Path) -> None:
    binary = b"\x13\x00\x00\x00"
    selected = b'{"order":[0],"addresses":{"0":1}}\n'
    (tmp_path / "program.bin").write_bytes(binary)
    (tmp_path / "native-selection.json").write_bytes(selected)
    manifest = {
        "engine": "merlin_native",
        "request_digest": "source-1",
        "target_identity": "target-1",
        "binary_sha256": hashlib.sha256(binary).hexdigest(),
        "stage_artifacts": {"native-selection.json": hashlib.sha256(selected).hexdigest()},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    kwargs = {"engine": "merlin_native", "request_digest": "source-1", "target_identity": "target-1"}
    verify_native_publication(tmp_path, manifest, **kwargs)

    (tmp_path / "native-selection.json").write_bytes(selected + b" ")
    with pytest.raises(ValueError, match="native stage artifact changed"):
        verify_native_publication(tmp_path, manifest, **kwargs)
    (tmp_path / "native-selection.json").write_bytes(selected)

    manifest["stage_artifacts"] = {"../outside.json": hashlib.sha256(selected).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe path"):
        verify_native_publication(tmp_path, manifest, **kwargs)

    manifest["stage_artifacts"] = {"native-selection.json": hashlib.sha256(selected).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "native-selection.json").unlink()
    (tmp_path / "native-selection.json").symlink_to(tmp_path / "program.bin")
    with pytest.raises(ValueError, match="native stage artifact changed"):
        verify_native_publication(tmp_path, manifest, **kwargs)


def test_native_verify_reopens_source_snapshot_and_selected_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from merlin.semantic_compiler import snapshot as snapshot_module
    from merlin.targetgen import cli

    dtype = TensorType((1,), "i32", "exact-i32")
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), dtype, effect="input"),),
        outputs=("x",), output_storages=("external",),
        input_storages=(("x", "external"),), target_identity="synthetic-verify-1",
        lowering_policy="diagnostic",
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request.record()))
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_text('{"identity":"snapshot-1"}\n')

    class _Profile:
        target_identity = "synthetic-verify-1"

    class _Snapshot:
        root = snapshot_root
        profile = _Profile()

    monkeypatch.setattr(snapshot_module, "open_native_snapshot", lambda _root: _Snapshot())
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    binary = b"\x13\x00\x00\x00"
    selected = {
        "engine": "merlin_native", "request_digest": request.digest(),
        "target_identity": request.target_identity, "check_fingerprint": "checked-1",
    }
    selected_bytes = (json.dumps(selected, sort_keys=True) + "\n").encode()
    (artifact / "program.bin").write_bytes(binary)
    (artifact / "native-selection.json").write_bytes(selected_bytes)
    manifest = {
        "engine": "merlin_native", "compile_mode": "diagnostic",
        "request_digest": request.digest(), "target_identity": request.target_identity,
        "native_snapshot_sha256": hashlib.sha256((snapshot_root / "manifest.json").read_bytes()).hexdigest(),
        "native_check_fingerprint": "checked-1",
        "binary_sha256": hashlib.sha256(binary).hexdigest(),
        "stage_artifacts": {"native-selection.json": hashlib.sha256(selected_bytes).hexdigest()},
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    status_path = tmp_path / "verify-status.json"
    command = [
        "native-verify", "--engine", "merlin_native", "--snapshot", str(snapshot_root),
        "--request", str(request_path), "--mode", "diagnostic", "--artifact", str(artifact),
        "--out", str(status_path),
    ]
    assert cli.main(command) == 0
    assert json.loads(status_path.read_text())["status"] == "verified_artifacts"

    (artifact / "native-selection.json").write_bytes(selected_bytes + b" ")
    assert cli.main(command) == 2
    assert "stage artifact changed" in json.loads(status_path.read_text())["reason"]
    (artifact / "native-selection.json").write_bytes(selected_bytes)

    (snapshot_root / "manifest.json").write_text('{"identity":"snapshot-2"}\n')
    assert cli.main(command) == 2
    assert "snapshot identity differs" in json.loads(status_path.read_text())["reason"]
    (snapshot_root / "manifest.json").write_text('{"identity":"snapshot-1"}\n')

    changed = KernelRequest(
        nodes=request.nodes, outputs=request.outputs, output_storages=request.output_storages,
        input_storages=request.input_storages, target_identity=request.target_identity,
        lowering_policy="diagnostic", source_identity="different-source",
    )
    request_path.write_text(json.dumps(changed.record()))
    assert cli.main(command) == 2
    assert "identity differs" in json.loads(status_path.read_text())["reason"]


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
    with pytest.raises(ValueError, match="known status"):
        NativeCompilationError("tool_unavailable", "")


def test_native_compile_reports_the_target_binding_failure_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnosed handoff failure keeps its earliest boundary instead of becoming compile_error."""
    from merlin.semantic_compiler import snapshot as snapshot_module
    from merlin.semantic_compiler import target_binding
    from merlin.targetgen import cli

    dtype = TensorType((1,), "i32", "exact-i32")
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), dtype, effect="input"), SemanticNode("y", "identity", ("x",), dtype)),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        target_identity="synthetic-handoff-1",
    )
    identity = InstructionDescriptor(
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
    profile = NativeTargetProfile("synthetic-handoff-1", (identity,), (StorageBank("external", "dram", 2, "tile"),))

    class _Snapshot:
        def __init__(self) -> None:
            self.profile = profile

    class _Binding:
        def profile(self) -> NativeTargetProfile:
            return profile

        def compile(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            raise NativeCompilationError("tool_unavailable", "generated target parser absent")

    monkeypatch.setattr(snapshot_module, "open_native_snapshot", lambda _root: _Snapshot())
    monkeypatch.setattr(target_binding, "load_native_target_binding", lambda _name: _Binding())
    request_path, abi_path = tmp_path / "request.json", tmp_path / "abi.json"
    request_path.write_text(json.dumps(request.record()))
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [1]}))
    status_path, output = tmp_path / "status.json", tmp_path / "compiled"
    returned = cli.main(
        [
            "native-compile",
            "--engine",
            "merlin_native",
            "--support",
            "synthetic-support",
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--request",
            str(request_path),
            "--abi",
            str(abi_path),
            "--target-source",
            str(tmp_path),
            "--mode",
            request.lowering_policy,
            "--out",
            str(output),
            "--status-file",
            str(status_path),
        ]
    )
    assert returned == 2 and not output.exists()
    status = json.loads(status_path.read_text())
    assert status["status"] == "tool_unavailable"
    assert "generated target parser absent" in status["reason"]


def test_native_select_abi_reservations_are_optional_and_checked(tmp_path: Path) -> None:
    from merlin.semantic_compiler.allocate import Reservation
    from merlin.targetgen.cli import _load_native_select_abi

    assert _load_native_select_abi(None) == ({}, None, ())
    abi_path = tmp_path / "abi.json"
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [2]}))
    assert _load_native_select_abi(str(abi_path)) == ({"x": 0}, (2,), ())
    row = {"storage": "external", "start": 1, "extent": 1}
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": None, "reservations": [row]}))
    assert _load_native_select_abi(str(abi_path)) == ({"x": 0}, None, (Reservation("external", 1, 1),))
    refused = (
        ({"fixed_inputs": {"x": 0}, "reservations": [row]}, "fixed_inputs and fixed_outputs"),
        ({"fixed_inputs": {}, "fixed_outputs": None, "samples": []}, "fixed_inputs and fixed_outputs"),
        ({"fixed_inputs": {}, "fixed_outputs": None, "reservations": row}, "storage/start/extent"),
        ({"fixed_inputs": {}, "fixed_outputs": None, "reservations": [{**row, "end": 2}]}, "storage/start/extent"),
        ({"fixed_inputs": {}, "fixed_outputs": None, "reservations": [{**row, "extent": 0}]}, "positive extent"),
    )
    for abi, message in refused:
        abi_path.write_text(json.dumps(abi))
        with pytest.raises(ValueError, match=message):
            _load_native_select_abi(str(abi_path))


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
    assert report["exploration_attempts"] >= 1
    assert report["diagnostic_only"] is False and report["diagnostic_ablations"] == []
    assert report["reservations"] == []

    reservation = {"storage": "external", "start": 1, "extent": 1}
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [2], "reservations": [reservation]}))
    reserved = _invoke(
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
    assert reserved.returncode == 0, reserved.stderr + reserved.stdout
    reserved_report = json.loads(output.read_text())
    assert reserved_report["status"] == "selected"
    assert reserved_report["reservations"] == [reservation]
    assert reserved_report["check_fingerprint"] != report["check_fingerprint"]
    occupied = range(reservation["start"], reservation["start"] + reservation["extent"])
    assert not set(reserved_report["allocation"]["addresses"].values()) & set(occupied)

    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [1], "reservations": [reservation]}))
    blocked = _invoke(
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
    assert blocked.returncode == 2, blocked.stderr + blocked.stdout
    blocked_report = json.loads(output.read_text())
    assert blocked_report["status"] != "selected" and blocked_report["allocation"] is None
    assert blocked_report["reservations"] == [reservation]
    abi_path.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [1]}))

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
        input_layouts=("row-major-tile",) * 3,
        output_layout="row-major-tile",
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

    layouts_path = tmp_path / "physical-layouts.json"
    layouts = {
        "schema": "merlin.physical_boundary_layouts.v1",
        "source_identity": report["source_identity"],
        "target_identity": target_identity,
        "physical_extents": {"arg0": 1, "arg1": 1, "arg2": 1},
        "boundary_layouts": {"arg0": "row-major-tile", "arg1": "row-major-tile", "arg2": "row-major-tile"},
        "output_layouts": ["row-major-tile"],
    }
    layouts_path.write_text(json.dumps(layouts))
    qualified = _invoke(
        "native-select", "--engine", "merlin_native", "--snapshot", snapshot,
        "--linalg", input_path, "--linalg-entry", "work",
        "--physical-layouts", layouts_path, "--out", output,
    )
    assert qualified.returncode == 0, qualified.stderr + qualified.stdout
    qualified_report = json.loads(output.read_text())
    assert qualified_report["status"] == "selected"
    assert qualified_report["request_digest"] != report["request_digest"]

    exact_layouts = {**layouts, "schema": "merlin.physical_boundary_layouts.v2",
                     "request_digest": report["request_digest"]}
    layouts_path.write_text(json.dumps(exact_layouts))
    exact = _invoke(
        "native-select", "--engine", "merlin_native", "--snapshot", snapshot,
        "--linalg", input_path, "--linalg-entry", "work",
        "--physical-layouts", layouts_path, "--out", output,
    )
    assert exact.returncode == 0, exact.stderr + exact.stdout
    assert json.loads(output.read_text())["request_digest"] == qualified_report["request_digest"]

    layouts_path.write_text(json.dumps({**exact_layouts, "request_digest": "stale-request"}))
    stale_request = _invoke(
        "native-select", "--engine", "merlin_native", "--snapshot", snapshot,
        "--linalg", input_path, "--linalg-entry", "work",
        "--physical-layouts", layouts_path, "--out", output,
    )
    assert stale_request.returncode == 2
    assert "precision-transformed request" in json.loads(output.read_text())["reason"]

    layouts_path.write_text(json.dumps({**layouts, "source_identity": "stale-source"}))
    stale = _invoke(
        "native-select", "--engine", "merlin_native", "--snapshot", snapshot,
        "--linalg", input_path, "--linalg-entry", "work",
        "--physical-layouts", layouts_path, "--out", output,
    )
    assert stale.returncode == 2
    assert json.loads(output.read_text())["status"] == "compile_error"

    layouts_path.write_text(json.dumps({**layouts, "boundary_layouts": {}}))
    empty = _invoke(
        "native-select", "--engine", "merlin_native", "--snapshot", snapshot,
        "--linalg", input_path, "--linalg-entry", "work",
        "--physical-layouts", layouts_path, "--out", output,
    )
    assert empty.returncode == 2
    assert json.loads(output.read_text())["status"] == "compile_error"

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
