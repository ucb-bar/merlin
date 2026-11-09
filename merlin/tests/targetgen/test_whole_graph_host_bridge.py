"""A selected whole graph keeps its source signature and ordered results."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import pickle
from pathlib import Path

import pytest
from xdsl.dialects import func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.targetgen import cli
from merlin.targetgen.contract.model_host_bridge import diagnostic_whole_graph_host_bridge

SOURCE = """builtin.module {
  func.func @forward(%left: tensor<2x3xf32>, %right: tensor<3x2xf32>)
      -> (tensor<3x2xf32>, tensor<2x3xf32>) {
    func.return %right, %left : tensor<3x2xf32>, tensor<2x3xf32>
  }
}
"""


def test_region_bridge_preserves_actual_source_ids_and_retained_bodies():
    from io import StringIO

    from xdsl.dialects.builtin import StringAttr
    from xdsl.printer import Printer

    from merlin.compiler.build.hostbridge import diagnostic_host_bridge
    from merlin.compiler.ir.graph import _program_graph
    from merlin.tests.ir.test_outline import CHAIN

    module = parse_mlir_text(CHAIN)
    for index, op in enumerate(op for op in module.walk() if op.name == "linalg.matmul"):
        op.attributes["prov.region_id"] = StringAttr(f"region{index}")
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    source = stream.getvalue() + "\n"
    graph = _program_graph(parse_mlir_text(source), hashlib.sha256(source.encode()).hexdigest())
    regions = [
        {"region_id": row["provenance"]["prov.region_id"], "operation_id": row["operation_id"],
         "status": "emitted" if index == 0 else "unsupported_semantics",
         "static_inputs_status": "packed_or_runtime" if index == 0 else "not_emitted",
         "request_digest": "selected-request" if index == 0 else None}
        for index, row in enumerate(row for row in graph["operations"] if row["mlir_operation"] == "linalg.matmul")
    ]
    record, text = diagnostic_host_bridge(source, "forward", regions)
    bridge = parse_mlir_text(text)
    bridge.verify()
    functions = {op.sym_name.data: op for op in bridge.body.block.ops if isinstance(op, func.FuncOp)}
    assert len(record["external_symbols"]) == 1
    assert not functions[record["external_symbols"][0]].body.blocks
    assert record["required_native_unmet"] == ["region1"]
    assert [row["selected_route"] for row in record["bindings"]] == ["direct_native", "retained_host"]
    assert not record["linked"] and not record["executed"] and not record["strict_native_eligible"]
    altered = copy.deepcopy(regions)
    altered[0]["operation_id"] = "unrelated-source"
    with pytest.raises(ValueError, match="operation IDs differ"):
        diagnostic_host_bridge(source, "forward", altered)


def _selected() -> dict:
    request = "1" * 64
    target = "synthetic-native-tensor-target"
    execution = {"output_addresses": [2048, 4096]}
    return {
        "program": SOURCE,
        "entry": "forward",
        "device_symbol": "selected_graph",
        "compilation": {
            "engine": "merlin_native",
            "artifact_kind": "program",
            "qualification_scope": "diagnostic_only",
            "request_digest": request,
            "target_identity": target,
            "binary_sha256": "2" * 64,
            "native_check_fingerprint": "3" * 64,
            "execution_plan_sha256": hashlib.sha256(
                (json.dumps(execution, sort_keys=True) + "\n").encode()
            ).hexdigest(),
        },
        "selection": {
            "engine": "merlin_native",
            "request_digest": request,
            "target_identity": target,
            "check_fingerprint": "3" * 64,
            "source_outputs": ["right", "left"],
            "graph_outputs": [2, 1],
        },
        "precision_plan": {
            "request_digest": request,
            "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            "boundary_conversions": [
                {"source": "input0"}, {"source": "input1"},
                {"source": "right"}, {"source": "left"},
            ],
        },
        "execution_plan": execution,
    }


def test_whole_graph_bridge_preserves_ordered_c_interface() -> None:
    from merlin.compiler.build import hostbridge

    assert importlib.import_module("merlin.targetgen.contract.model_host_bridge") is hostbridge
    for name in ("diagnostic_host_bridge", "diagnostic_whole_graph_host_bridge"):
        assert pickle.loads(f"cmerlin.targetgen.contract.model_host_bridge\n{name}\n.".encode()) is getattr(
            hostbridge, name
        )
    assert any(path.as_posix().endswith("compiler/ir/graph.py") for path in hostbridge.implementation_paths())
    record, text = diagnostic_whole_graph_host_bridge(**_selected())
    module = parse_mlir_text(text)
    module.verify()
    functions = {op.sym_name.data: op for op in module.body.block.ops
                 if isinstance(op, func.FuncOp)}
    assert set(functions) == {"forward", "selected_graph"}
    assert not functions["selected_graph"].body.blocks
    assert "llvm.emit_c_interface" in functions["selected_graph"].attributes
    assert "llvm.emit_c_interface" in functions["forward"].attributes
    block = functions["forward"].body.block
    call, returned = tuple(block.ops)
    assert isinstance(call, func.CallOp)
    assert isinstance(returned, func.ReturnOp)
    assert list(call.operands) == list(block.args)
    assert list(returned.operands) == list(call.results)
    assert record["input_types"] == ["tensor<2x3xf32>", "tensor<3x2xf32>"]
    assert record["result_types"] == ["tensor<3x2xf32>", "tensor<2x3xf32>"]
    assert record["source_outputs"] == ["right", "left"]
    assert record["status"] == "typed_declarations_only"
    assert not record["linked"] and not record["executed"]


@pytest.mark.parametrize("change", [
    lambda data: data.update(program=data["program"] + "\n"),
    lambda data: data["compilation"].update(engine="act_reference"),
    lambda data: data["compilation"].update(binary_sha256=None),
    lambda data: data["compilation"].update(native_check_fingerprint=None),
    lambda data: data["selection"].update(source_outputs=["left", "right"]),
    lambda data: data["selection"].update(graph_outputs=[1]),
    lambda data: data["execution_plan"].update(output_addresses=[2048, 2048]),
    lambda data: data["execution_plan"].update(output_addresses=[2048]),
    lambda data: data["precision_plan"].update(source_sha256="0" * 64),
])
def test_whole_graph_bridge_rejects_changed_selection(change) -> None:
    data = copy.deepcopy(_selected())
    change(data)
    with pytest.raises(ValueError, match="whole-graph selection differs"):
        diagnostic_whole_graph_host_bridge(**data)


def test_whole_graph_bridge_rejects_changed_source_signature() -> None:
    data = _selected()
    changed = SOURCE.replace("%right, %left :", "%left, %right :")
    # The changed source is invalid for its declared ordered result types.
    with pytest.raises(Exception):
        parse_mlir_text(changed).verify()
    data["entry"] = "missing"
    with pytest.raises(ValueError, match="one defined entry"):
        diagnostic_whole_graph_host_bridge(**data)


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_cli_bridge_checks_native_artifact_bytes_before_writing(tmp_path: Path, newline) -> None:
    selected = _selected()
    raw_source = SOURCE.replace("\n", newline).encode("utf-8")
    selected["precision_plan"]["source_sha256"] = hashlib.sha256(raw_source).hexdigest()
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    source = tmp_path / "source.mlir"
    source.write_bytes(raw_source)
    members = {
        "native-selection.json": json.dumps(selected["selection"], sort_keys=True).encode(),
        "precision-plan.json": json.dumps(selected["precision_plan"], sort_keys=True).encode(),
        "execution_plan.json": (json.dumps(selected["execution_plan"], sort_keys=True) + "\n").encode(),
        "program.bin": b"\x01\x02\x03\x04",
    }
    for name, content in members.items():
        (compiled / name).write_bytes(content)
    compilation = selected["compilation"]
    compilation["binary_sha256"] = hashlib.sha256(members["program.bin"]).hexdigest()
    compilation["stage_artifacts"] = {
        name: hashlib.sha256(members[name]).hexdigest()
        for name in ("native-selection.json", "precision-plan.json")
    }
    (compiled / "manifest.json").write_text(json.dumps(compilation, sort_keys=True) + "\n")

    def invoke(destination: Path) -> int:
        return cli.main([
            "native-bridge-graph", "--linalg", str(source), "--entry", "forward",
            "--compiled", str(compiled), "--device-symbol", "selected_graph",
            "--out", str(destination),
        ])

    good = tmp_path / "good"
    assert invoke(good) == 0
    assert parse_mlir_text((good / "host-bridge.mlir").read_text()).verify() is None
    published = json.loads((good / "manifest.json").read_text())
    assert published["status"] == "typed_declarations_only"
    assert published["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    (compiled / "program.bin").write_bytes(b"\x00\x00\x00\x00")
    bad = tmp_path / "bad"
    assert invoke(bad) == 2
    assert not (bad / "host-bridge.mlir").exists()
    assert json.loads((bad / "manifest.json").read_text())["status"] == "compile_error"

    # An output inside the compiled artifact is refused without writing into it.
    before = {path.name: path.read_bytes() for path in compiled.iterdir()}
    inside = compiled / "bridge"
    assert invoke(inside) == 2
    assert not inside.exists()
    assert {path.name: path.read_bytes() for path in compiled.iterdir()} == before


def _write_compiled(root: Path, selected: dict, layouts: dict | None = None) -> None:
    root.mkdir()
    members = {
        "native-selection.json": json.dumps(selected["selection"], sort_keys=True).encode(),
        "precision-plan.json": json.dumps(selected["precision_plan"], sort_keys=True).encode(),
        "execution_plan.json": (json.dumps(selected["execution_plan"], sort_keys=True) + "\n").encode(),
        "program.bin": b"\x01\x02\x03\x04",
    }
    if layouts is not None:
        members["physical-boundary-layouts.json"] = json.dumps(layouts, sort_keys=True).encode()
    for name, content in members.items():
        (root / name).write_bytes(content)
    compilation = selected["compilation"]
    compilation["binary_sha256"] = hashlib.sha256(members["program.bin"]).hexdigest()
    compilation["stage_artifacts"] = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in members.items() if name not in {"execution_plan.json", "program.bin"}
    }
    (root / "manifest.json").write_text(json.dumps(compilation, sort_keys=True) + "\n")


def test_cli_bridge_accepts_precision_plan_bound_before_physical_layouts(tmp_path: Path) -> None:
    # Binding v2 physical layouts changes the compiled request digest, while the
    # precision plan keeps the digest of the request it transformed. The
    # tracked layout document carries that pre-layout digest.
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    pre_layout = "1" * 64
    bound = "4" * 64
    layouts = {
        "schema": "merlin.physical_boundary_layouts.v2",
        "target_identity": "synthetic-native-tensor-target",
        "source_identity": "captured-source",
        "request_digest": pre_layout,
        "physical_extents": {"input0": 6, "input1": 6},
        "boundary_layouts": {"input0": "row-major", "input1": "row-major"},
        "output_layouts": ["row-major", "row-major"],
    }

    def invoke(compiled: Path, destination: Path) -> int:
        return cli.main([
            "native-bridge-graph", "--linalg", str(source), "--entry", "forward",
            "--compiled", str(compiled), "--device-symbol", "selected_graph",
            "--out", str(destination),
        ])

    def selected_after_layouts() -> dict:
        selected = _selected()
        selected["compilation"]["request_digest"] = bound
        selected["selection"]["request_digest"] = bound
        return selected

    accepted = tmp_path / "accepted"
    _write_compiled(accepted, selected_after_layouts(), layouts)
    assert invoke(accepted, tmp_path / "accepted-bridge") == 0
    record = json.loads((tmp_path / "accepted-bridge" / "manifest.json").read_text())
    assert record["status"] == "typed_declarations_only" and record["request_digest"] == bound

    untracked = tmp_path / "untracked"
    _write_compiled(untracked, selected_after_layouts())
    assert invoke(untracked, tmp_path / "untracked-bridge") == 2

    stale = tmp_path / "stale"
    _write_compiled(stale, selected_after_layouts(), {**layouts, "request_digest": "5" * 64})
    assert invoke(stale, tmp_path / "stale-bridge") == 2

    legacy = tmp_path / "legacy"
    _write_compiled(legacy, selected_after_layouts(), {
        key: value for key, value in layouts.items() if key != "request_digest"
    } | {"schema": "merlin.physical_boundary_layouts.v1"})
    assert invoke(legacy, tmp_path / "legacy-bridge") == 2


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


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_explicit_graph_policy_compiles_tracks_replays_and_refuses_unselected_precision(tmp_path, monkeypatch, newline):
    from types import SimpleNamespace

    from merlin.semantic_compiler import snapshot as snapshot_module
    from merlin.semantic_compiler import target_binding

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_text('{"identity":"selected-snapshot"}\n')
    source = tmp_path / "source.mlir"
    raw_program = SOURCE.replace("\n", newline)
    source.write_bytes(raw_program.encode("utf-8"))
    policy = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "intent": "explicit-diagnostic-fixture"}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    dtype = TensorType((1,), "i32", "exact-i32")
    request = KernelRequest(nodes=(SemanticNode("x", "input", (), dtype, effect="input"),),
                            outputs=("x",), input_storages=(("x", "external"),), output_storages=("external",),
                            target_identity="synthetic-graph-fixture", lowering_policy="diagnostic",
                            source_identity=policy["source_sha256"])
    profile = SimpleNamespace(target_identity=request.target_identity, digest=lambda: "selected-profile")
    snapshot = SimpleNamespace(profile=profile, root=snapshot_root)
    preparations = []

    class Binding:
        @staticmethod
        def profile():
            return profile

        @staticmethod
        def prepare_precision_graph(program, *, entry, policy, **kwargs):
            assert program == raw_program and entry == "forward"
            if policy != json.loads(policy_path.read_text()) or policy["source_sha256"] != request.source_identity:
                raise ValueError("source/policy differs")
            preparations.append(kwargs)
            record = {"request_digest": request.digest(), "source_sha256": policy["source_sha256"], "policy": policy}
            return SimpleNamespace(request=request, source_operations=("func.return",), record=lambda: record)

        @staticmethod
        def compile(selected, prepared, *, destination, **kwargs):
            assert selected is snapshot and prepared is request
            destination.mkdir()
            binary = b"\x13\x00\x00\x00"
            (destination / "program.bin").write_bytes(binary)
            manifest = {"engine": "merlin_native", "request_digest": request.digest(),
                        "target_identity": request.target_identity, "compile_mode": "diagnostic",
                        "binary_sha256": hashlib.sha256(binary).hexdigest(),
                        "native_snapshot_sha256": hashlib.sha256(
                            (snapshot_root / "manifest.json").read_bytes()).hexdigest(),
                        "constant_inputs": [], "stage_artifacts": {}}
            (destination / "manifest.json").write_text(json.dumps(manifest))
            return manifest

    monkeypatch.setattr(snapshot_module, "open_native_snapshot", lambda _: snapshot)
    monkeypatch.setattr(target_binding, "load_native_target_binding", lambda _: Binding)
    abi = tmp_path / "abi.json"
    abi.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [1]}))
    artifact, status = tmp_path / "compiled", tmp_path / "verify.json"
    base = ["--engine", "merlin_native", "--support", "synthetic", "--snapshot", str(snapshot_root),
            "--linalg", str(source), "--linalg-entry", "forward", "--diagnostic-graph-policy", str(policy_path),
            "--mode", "diagnostic"]
    assert cli.main(["native-compile", *base, "--abi", str(abi), "--target-source", str(tmp_path),
                     "--out", str(artifact)]) == 0
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["stage_artifacts"]["precision-plan.json"] == hashlib.sha256(
        (artifact / "precision-plan.json").read_bytes()).hexdigest()
    assert json.loads((artifact / "precision-plan.json").read_text())["source_sha256"] == hashlib.sha256(
        source.read_bytes()).hexdigest()
    command = ["native-verify", *base, "--artifact", str(artifact), "--out", str(status)]
    assert cli.main(command) == 0 and len(preparations) == 2
    assert json.loads(status.read_text())["status"] == "verified_artifacts"
    def bounded_selection(*args, **kwargs):
        raise target_binding.NativeCompilationError("resource_limit", "fixture selection budget")
    snapshot.select = bounded_selection
    selection = tmp_path / "selection.json"
    assert cli.main(["native-select", *base, "--out", str(selection)]) == 2
    assert len(preparations) == 3  # The select flag actually ran the same preparation.
    assert json.loads(selection.read_text())["status"] == "resource_limit"
    (artifact / "precision-plan.json").write_bytes(b"{}\n")
    assert cli.main(command) == 2
    assert "stage artifact changed" in json.loads(status.read_text())["reason"]
    before = {p.name: p.read_bytes() for p in artifact.iterdir()}
    assert cli.main(command[:-1] + [str(artifact / "status.json")]) == 2
    assert before == {p.name: p.read_bytes() for p in artifact.iterdir()}
    non_diagnostic = ["strict-native" if item == "diagnostic" else item for item in base]
    rejected = tmp_path / "rejected"
    assert cli.main(["native-compile", *non_diagnostic, "--abi", str(abi), "--target-source", str(tmp_path),
                     "--out", str(rejected)]) == 2
    assert not rejected.exists()
    assert cli.main(["native-compile", *base, "--application", "--abi", str(abi),
                     "--target-source", str(tmp_path), "--out", str(rejected)]) == 2
    assert not rejected.exists()
    status.write_text('{"status":"verified_artifacts"}')
    assert cli.main([*command, "--application"]) == 2
    assert json.loads(status.read_text())["status"] == "compile_error"


def test_staged_graph_policy_keeps_exact_policy_entry_and_original_source_binding(tmp_path, monkeypatch):
    from argparse import Namespace
    from types import SimpleNamespace

    from merlin.compiler.frontend import inputs
    from merlin.compiler.frontend.native import load_selected_request

    dtype = TensorType((1,), "i32", "exact-i32")
    request = KernelRequest(nodes=(SemanticNode("x", "input", (), dtype, effect="input"),),
                            outputs=("x",), input_storages=(("x", "external"),), output_storages=("external",),
                            target_identity="synthetic-staged-graph", lowering_policy="diagnostic")
    policy = {"source_sha256": "a" * 64}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    staged = SimpleNamespace(program=SOURCE, entry="forward", precision_policy=policy,
                             original_source_sha256="a" * 64)
    monkeypatch.setattr(inputs, "load_compile_inputs", lambda _: staged)
    visits = []
    def prepare(source, *, entry, policy, original_source_sha256):
        visits.append((source, entry, policy, original_source_sha256))
        return SimpleNamespace(request=request, source_operations=("func.return",), record=lambda: {})
    args = Namespace(diagnostic_graph_policy=str(policy_path), compile_inputs="selected-staged",
                     linalg=None, linalg_entry=None)
    binding = SimpleNamespace(prepare_precision_graph=prepare)
    loaded, kind, _, _ = load_selected_request(args, target_identity=request.target_identity,
                                              mode="diagnostic", binding=binding)
    assert loaded is request and kind == "source_bound_diagnostic_staged_graph"
    assert visits == [(SOURCE, "forward", policy, "a" * 64)]
    staged.precision_policy = {**policy, "different": True}
    with pytest.raises(ValueError, match="match its compiler-input policy"):
        load_selected_request(args, target_identity=request.target_identity, mode="diagnostic", binding=binding)
    assert len(visits) == 1
    staged.precision_policy = policy
    with pytest.raises(ValueError, match="diagnostic mode"):
        load_selected_request(args, target_identity=request.target_identity, mode="strict-native", binding=binding)
    assert len(visits) == 1
    def old_hook(source, *, entry, policy):
        pytest.fail("a staged premise must never be dropped to retry an older hook")
    with pytest.raises(TypeError, match="original_source_sha256"):
        load_selected_request(args, target_identity=request.target_identity, mode="diagnostic",
                              binding=SimpleNamespace(prepare_precision_graph=old_hook))
