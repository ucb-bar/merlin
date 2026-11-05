"""A selected whole graph keeps its source signature and ordered results."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from xdsl.dialects import func

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen import cli
from merlin.targetgen.contract.model_host_bridge import diagnostic_whole_graph_host_bridge

SOURCE = """builtin.module {
  func.func @forward(%left: tensor<2x3xf32>, %right: tensor<3x2xf32>)
      -> (tensor<3x2xf32>, tensor<2x3xf32>) {
    func.return %right, %left : tensor<3x2xf32>, tensor<2x3xf32>
  }
}
"""


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


def test_cli_bridge_checks_native_artifact_bytes_before_writing(tmp_path: Path) -> None:
    selected = _selected()
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
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
    assert json.loads((good / "manifest.json").read_text())["status"] == "typed_declarations_only"
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
